"""
Agent Module (LangChain v0.3+ compatible)
-----------------------------------------
Handles reasoning and routing between:
- RAGTool (document-based answers)
- WeatherTool (live data)
- DuckDuckGoSearch (web lookup)

✔ Fixed: LLM now *sees* conversation history
✔ Persistent memory per session
✔ RunnableWithMessageHistory (v0.3+ style)
✔ Clean structured outputs
"""

from collections import defaultdict
from functools import lru_cache
from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema.output_parser import StrOutputParser

from prompts import agent_prompt
from tools import duckduckgo_tool, weather_tool, trip_budget_tool
from rag import rag_tool, get_llm
from config import config

# -------------------------------------------------------------------
#  TOOLS + LLM
# -------------------------------------------------------------------
tools = [rag_tool, weather_tool, duckduckgo_tool]
llm = get_llm()
output_parser = StrOutputParser()

# -------------------------------------------------------------------
#  MEMORY MANAGEMENT
# -------------------------------------------------------------------
session_histories = defaultdict(ChatMessageHistory)

def get_session_history(session_id: str):
    """Retrieve or create chat history for each session."""
    return session_histories[session_id]

def get_memory(session_id: str):
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
agent = create_react_agent(
    tools=tools,
    llm=llm,
    prompt=agent_prompt.partial(tools="\n".join([t.name for t in tools])),
)

# -------------------------------------------------------------------
#  EXECUTOR FACTORY
# -------------------------------------------------------------------
def make_agent_executor(session_id: str):
    """Build an AgentExecutor with active session memory."""
    memory = get_memory(session_id)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,  
        handle_parsing_errors=True,
        verbose=False,
        return_intermediate_steps=True,
        max_iterations=config["agent"]["max_iterations"],
    )

# -------------------------------------------------------------------
#  CACHED EXECUTOR WRAPPER
# -------------------------------------------------------------------
@lru_cache(maxsize=8)
def get_agent_executor(session_id="default"):
    """Cache a memory-enabled AgentExecutor per session."""
    base_exec = make_agent_executor(session_id)
    return RunnableWithMessageHistory(
        base_exec,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

# -------------------------------------------------------------------
#  SOURCE CLEANUP
# -------------------------------------------------------------------
def extract_sources_from_steps(intermediate_steps):
    """Collect and de-duplicate sources cleanly without repeating 'Sources:' headers."""
    all_sources = set()
    for _, observation in intermediate_steps:
        # Split the text lines that came from tools (rag_tool, duckduckgo_tool, etc.)
        for line in observation.split("\n"):
            # Keep only lines that *look like sources or URLs*
            if any(x in line for x in ["Source:", "DuckDuckGoSearch", "backpacker2.pdf", "http"]):
                clean_line = (
                    line.replace("### Sources Used:", "")
                        .replace("**Sources:**", "")
                        .replace("Sources:", "")
                        .replace("Source -", "")
                        .replace("•", "")
                        .strip()
                )
                if clean_line:
                    all_sources.add(clean_line)

    # Return a simple, deduplicated list — no internal “Sources:” header
    return "\n".join(sorted(all_sources))



# -------------------------------------------------------------------
#  MAIN PROCESS FUNCTION (Enhanced with weather + fallback handling)
# -------------------------------------------------------------------
def process_query(query: str, session_id: str = "default"):
    """
    Routes query through the ReAct agent and returns structured data.
    Handles weather detection, timeouts, iteration limits, and fallback routing gracefully.
    """
    query_lower = query.lower()

    # ---  1. Early Weather Detection (runs before agent) ---
    weather_keywords = [
        "weather", "temperature", "rain", "forecast", "wind",
        "climate", "humidity", "snow", "storm", "tomorrow", "next week"
    ]
    if any(word in query_lower for word in weather_keywords):
        print(" Detected weather query — routing directly to WeatherTool.")
        try:
            # Use existing weather tool (already imported at top)
            weather_result = weather_tool.run(query)
            return {
                "answer": weather_result,
                "sources": "WeatherTool"
            }
        except Exception as e:
            return {
                "answer": f" Weather lookup failed: {e}",
                "sources": "Weather tool error"
            }

        # ---  2. Budget / Trip Planner Detection ---
    budget_keywords = [
        "budget", "cost", "plan my trip", "trip plan", "itinerary", "estimate", "calculate", "expenses"
    ]
    if any(word in query_lower for word in budget_keywords):
        print(" Detected budget/trip planner query — routing directly to TripBudgetPlanner Tool.")
        try:
            budget_result = trip_budget_tool.run(query)
            return {
                "answer": budget_result,
                "sources": "TripBudgetPlanner Tool"
            }
        except Exception as e:
            return {
                "answer": f" Budget planning failed: {e}",
                "sources": "TripBudgetPlanner Tool error"
            }

    
    # ---  2. Normal RAG Agent Processing ---
    try:
        executor = get_agent_executor(session_id)
        result = executor.invoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}},
        )

        answer = result.get("output", "No answer generated.")
        sources = extract_sources_from_steps(result.get("intermediate_steps", []))

        # --- Clean up the source text ---
        if sources:
            lines = [line.strip() for line in sources.splitlines() if line.strip()]
            cleaned = [l for l in lines if not l.lower().startswith("sources")]
            sources = "\n".join(dict.fromkeys(cleaned))

        # --- 3. Fallback to DuckDuckGo if no relevant RAG answer ---
        if not answer or "No answer" in answer or len(answer.strip()) < 10:
            print(" No relevant match in RAG — using DuckDuckGo fallback...")
            web_results = duckduckgo_tool.duckduckgo_with_sources(query)
            return {
                "answer": web_results,
                "sources": "DuckDuckGo Search "
            }

        # --- 4. Normal Success ---
        return {"answer": answer.strip(), "sources": sources.strip()}

    except Exception as e:
        error_msg = str(e)

        # --- 5. Handle agent stop / timeout conditions gracefully ---
        if "iteration limit" in error_msg.lower() or "time limit" in error_msg.lower():
            print(" Agent reached time or iteration limit — switching to DuckDuckGo fallback.")
            web_results = duckduckgo_tool.duckduckgo_with_sources(query)
            return {
                "answer": web_results,
                "sources": "DuckDuckGo Search  (timeout fallback)"
            }

        # --- 6. Handle any other unexpected errors ---
        return {
            "answer": f" Unexpected error: {error_msg}",
            "sources": "No sources due to error."
        }


