# app.py
"""
Gradio Frontend for Tasmania Backpacker Bot
-------------------------------------------
- Connects to LangChain ReAct Agent via process_query()
- Keeps chat memory via session_id (per-session continuity)
- Displays only the latest answer per user message
- Compatible with LangChain v0.3+ and Gradio v4+
"""

import gradio as gr
from agent import process_query
from config import config

# -------------------------------------------------------------------
# CHAT LOGIC
# -------------------------------------------------------------------
def on_submit(message, history):
    """Handle user input and maintain conversation continuity."""
    session_id = "tas_session"  # static session ID; can be randomized later

    # Get response from agent (returns dict with 'answer' and 'sources')
    result = process_query(message, session_id=session_id)
    answer = result.get("answer", "No answer generated.")
    sources = result.get("sources", "")

    # Combine nicely formatted reply (but no repetition)
    full_response = f"**Answer:** {answer}"
    if sources:
        full_response += f"\n\n**Sources:** {sources}"

    return full_response


# -------------------------------------------------------------------
# OPTIONAL RESET BUTTON LOGIC
# -------------------------------------------------------------------
def clear_chat():
    """Resets the conversation memory for the demo session."""
    from agent import session_histories, session_memories
    session_id = "tas_session"
    if session_id in session_histories:
        del session_histories[session_id]
    if session_id in session_memories:
        del session_memories[session_id]
    return "🧹 Chat memory cleared!"


# -------------------------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------------------------
custom_css = """
.custom-submit {
    background: linear-gradient(45deg, #4CAF50, #45a049);
    color: white;
    border: none;
    border-radius: 25px;
    padding: 10px 20px;
    font-weight: bold;
    transition: all 0.3s ease;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
.custom-submit:hover {
    background: linear-gradient(45deg, #45a049, #4CAF50);
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
}
"""

# -------------------------------------------------------------------
# GRADIO INTERFACE
# -------------------------------------------------------------------
with gr.Blocks(css=custom_css, title=config["ui"]["title"]) as demo:
    gr.Markdown(
        f"""
        <h1 style='text-align:center; font-size:2.5rem; color:#2e7d32;'>🏕️ {config['ui']['title']}</h1>
        <p style='text-align:center; font-size:1.1rem;'>{config['ui']['description']}</p>
        <p style='text-align:center;'>Ask about travel tips, weather, or backpacking spots!</p>
        """
    )

    chatbot = gr.ChatInterface(
        fn=on_submit,
        type="messages",
        examples=config["ui"]["examples"],
        title=config["ui"]["title"],
        description=config["ui"]["description"],
        submit_btn="Submit",
        theme="default",
    )

    # Optional button to clear session memory
    gr.Button("Reset Chat 🧹").click(clear_chat, None, None)

# -------------------------------------------------------------------
# APP LAUNCH
# -------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(share=config["ui"]["share"], debug=config["ui"]["debug"])
