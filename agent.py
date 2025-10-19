from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Dict, Any, List, Tuple

from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema.output_parser import StrOutputParser
import json
import re
import logging

from prompts import agent_prompt
from tools import duckduckgo_tool, weather_tool, WeatherForecast_tool, trip_budget_tool
from rag import rag_tool, get_llm
from config import config

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
#  TOOLS + LLM
# -------------------------------------------------------------------
tools = [rag_tool, weather_tool, WeatherForecast_tool, duckduckgo_tool, trip_budget_tool]
llm = get_llm()
output_parser = StrOutputParser()

# -------------------------------------------------------------------
#  MEMORY MANAGEMENT
# -------------------------------------------------------------------
session_histories: Dict[str, ChatMessageHistory] = defaultdict(ChatMessageHistory)

def get_session_history(session_id: str) -> ChatMessageHistory:
    """Retrieve or create chat history for a given session."""
    return session_histories[session_id]

def get_memory(session_id: str) -> ConversationBufferMemory:
    """Return ConversationBufferMemory tied to that ChatMessageHistory."""
    return ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=get_session_history(session_id),
        input_key="input",
        output_key="output",
        return_messages=True,
    )

# -------------------------------------------------------------------
#  AGENT CREATION (ReAct)
# -------------------------------------------------------------------
react_agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=agent_prompt,
)

# -------------------------------------------------------------------
#  EXECUTOR FACTORY
# -------------------------------------------------------------------
def make_agent_executor(session_id: str) -> AgentExecutor:
    """Build an AgentExecutor with active session memory."""
    memory = get_memory(session_id)
    return AgentExecutor(
        agent=react_agent,
        tools=tools,
        memory=memory,
        handle_parsing_errors=True,
        verbose=True,  # Enable verbose for debugging
        return_intermediate_steps=True,
        max_iterations=config.get("agent", {}).get("max_iterations", 8),
    )

# -------------------------------------------------------------------
#  CACHED EXECUTOR WRAPPER
# -------------------------------------------------------------------
@lru_cache(maxsize=8)
def get_agent_executor(session_id: str = "default") -> RunnableWithMessageHistory:
    """Cache a memory-enabled AgentExecutor per session."""
    base_exec = make_agent_executor(session_id)
    return RunnableWithMessageHistory(
        base_exec,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

# # -------------------------------------------------------------------
# #  SOURCE CLEANUP
# # -------------------------------------------------------------------
def extract_sources_from_steps(intermediate_steps: List[Tuple[Any, str]]) -> str:
    """
    Collect and de-duplicate sources cleanly.
    Looks through tool observations for lines that likely contain sources.
    """
    needles = ("Source:", "Sources:", "DuckDuckGoSearch", "http", "https", "backpacker")
    scrub = ("### Sources Used:", "**Sources:**", "Sources:", "Source -", "•")
    found = []

    for _, observation in intermediate_steps:
        if not observation:
            continue
        for raw in observation.splitlines():
            line = raw.strip()
            if any(n in line for n in needles):
                for s in scrub:
                    line = line.replace(s, "")
                line = line.strip("-•: ").strip()
                if line:
                    found.append(line)

    # stable order while de-duplicating
    seen = set()
    deduped = []
    for line in found:
        if line not in seen:
            seen.add(line)
            deduped.append(line)
    return "\n".join(deduped)

# -------------------------------------------------------------------
#  HELPER: EXTRACT PLACE AND DAYS FOR TRIP BUDGET PLANNER
# -------------------------------------------------------------------
def extract_place_and_days(query: str) -> str:
    """Extract place and days from query and return JSON string for TripBudgetPlanner."""
    try:
        # Detect number of days
        days_match = re.search(r"(\d+)\s*(?:day|days)", query.lower())
        days = int(days_match.group(1)) if days_match else 3

        # Extract location (remove filler words & digits)
        cleaned = re.sub(r"\b(forecast|for|in|of|the|next|day|days|weather|trip|plan|budget)\b", "", query, flags=re.I)
        cleaned = re.sub(r"\d+", "", cleaned).strip().strip(",.")
        place = cleaned or "Tasmania"

        result = json.dumps({"place": place, "days": days})
        logger.debug(f"Extracted for TripBudgetPlanner: {result}")
        return result
    except Exception as e:
        result = json.dumps({"place": "Tasmania", "days": 3})
        logger.error(f"Fallback for TripBudgetPlanner: {result}, error: {e}")
        return result

# -------------------------------------------------------------------
#  MAIN PROCESS FUNCTION (Prompt-driven routing)
# -------------------------------------------------------------------
def process_query(query: str, session_id: str = "default") -> Dict[str, str]:
    """
    Routes query through the ReAct agent using prompt-defined rules.
    Handles all query types via the agent's reasoning.
    """
    try:
        executor = get_agent_executor(session_id)
        # Preprocess query for TripBudgetPlanner to guide the agent
        input_query = query
        if any(keyword in query.lower() for keyword in ["trip", "budget", "plan", "cost", "itinerary", "expense"]):
            trip_budget_input = extract_place_and_days(query)
            # Append guidance to the query to ensure TripBudgetPlanner is used with correct input
            input_query = f"{query}\n\nFor trip planning, use TripBudgetPlanner with input: {trip_budget_input}"
        
        input_dict = {"input": input_query}
        logger.debug(f"Input to executor: {input_dict}")
        result: Dict[str, Any] = executor.invoke(
            input_dict,
            config={"configurable": {"session_id": session_id}},
        )
        logger.debug(f"Executor result: {result}")

        answer = result.get("output", "") or "No answer generated."
        sources = extract_sources_from_steps(result.get("intermediate_steps", []))

        # Fallback to DuckDuckGo if the agent didn't produce something meaningful
        if not answer or len(answer.strip()) < 10 or "No answer" in answer:
            try:
                web_results = duckduckgo_tool.run(query)
                logger.debug(f"Fallback to DuckDuckGo: {web_results}")
                return {"answer": web_results, "sources": "DuckDuckGo Search (fallback)"}
            except Exception as e:
                logger.error(f"DuckDuckGo fallback failed: {e}")
                return {"answer": answer.strip(), "sources": sources.strip() or "No sources"}

        return {"answer": answer.strip(), "sources": sources.strip() or "—"}

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Unexpected error in process_query: {error_msg}")
        # Handle agent stop/timeout conditions gracefully
        if ("iteration limit" in error_msg.lower()) or ("time limit" in error_msg.lower()):
            try:
                web_results = duckduckgo_tool.run(query)
                logger.debug(f"Timeout fallback to DuckDuckGo: {web_results}")
                return {"answer": web_results, "sources": "DuckDuckGo Search (timeout fallback)"}
            except Exception as e:
                logger.error(f"DuckDuckGo timeout fallback failed: {e}")
                return {"answer": "The agent timed out and the fallback also failed.", "sources": "—"}

        # Any other unexpected errors
        return {"answer": f"Unexpected error: {error_msg}", "sources": "No sources due to error."}