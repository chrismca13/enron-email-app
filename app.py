import streamlit as st
from langchain.agents import AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage
import io
import sys

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=False,
    callbacks=[stdout_callback]
)

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "logs" not in st.session_state:
    st.session_state.logs = ""

# --- UI Layout ---
st.title("📬 Enron Email Agent")
st.markdown("Ask a question about Jeffrey Skilling's emails and Enron's stock performance.")

question = st.text_area("Ask a question:", height=100, key="input_question")
submit = st.button("Submit")

st.markdown("### 💡 Example questions")
st.markdown("- Which baseball games has Jeffrey Skilling gone to and when?")
st.markdown("- How did Enron's stock perform around the time of those events?")
st.markdown("- What family events are mentioned in the emails?")

# --- Answer Tabs ---
response_tab, thought_tab, history_tab = st.tabs(["💬 Response", "🧠 Thought Process", "🕒 Chat History"])

if submit and question:
    # Capture stdout (for thought process)
    buffer = io.StringIO()
    sys_stdout = sys.stdout
    sys.stdout = buffer

    # Run the agent
    response = agent_executor.invoke({"input": question})

    # Restore stdout and get logs
    sys.stdout = sys_stdout
    logs = buffer.getvalue()
    st.session_state.logs = logs

    # Save messages
    st.session_state.chat_history.append(("User", question))
    st.session_state.chat_history.append(("Assistant", response["output"]))

# Show LLM response
with response_tab:
    if st.session_state.chat_history:
        st.markdown("#### Assistant Response")
        st.markdown(st.session_state.chat_history[-1][1])

# Show captured thought process
with thought_tab:
    st.markdown("#### Agent Reasoning & Tool Usage")
    st.text(st.session_state.logs or "No logs captured yet.")

# Show chat history
with history_tab:
    st.markdown("#### Full Conversation History")
    for role, msg in st.session_state.chat_history:
        st.markdown(f"**{role}:** {msg}")
