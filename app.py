import streamlit as st
from langchain.agents import AgentExecutor
from langchain_core.messages import HumanMessage, AIMessage
import io
import sys
import pandas as pd
from langchain_aws.chat_models import ChatBedrockConverse
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate



model = ChatBedrockConverse(
    model="us.meta.llama3-1-70b-instruct-v1:0"
)


# Function to combine docs into a single string with citation info
def combine_docs(docs):
    combined_text = []
    for doc in docs:
        subject = doc.metadata.get("subject", "No Subject")
        date = doc.metadata.get("date", "No Date")
        combined_text.append(f"[Subject: {subject}, Date: {date}]\n{doc.page_content}")
    return "\n\n".join(combined_text)

# Read in stock data
def enron_stock_data():
    enron_stock = pd.read_excel('../assets/enron_stock_prices.xlsx')

    # A bunch of (probably unnecessarry data manipulation to remove space and clean the df
    enron_stock = enron_stock.iloc[3:]
    
    enron_stock = enron_stock.reset_index(drop = True)
    enron_stock = enron_stock.rename(columns=enron_stock.iloc[0])# enron_stock.drop(enron_stock.index[0])
    
    enron_stock = enron_stock.drop(enron_stock.index[0])\
                             .reset_index(drop = True)
    
    return enron_stock

# Email search tool
@tool
def search_emails(question: str) -> str:
    """Search Jeffrey Skilling's emails to answer factual questions."""
    return email_chain.invoke(question).content

# stock summary tool
    
@tool
def stock_over_time(start_date: str, end_date: str) -> str:
    """
    Look up Enron's stock percentage change between two dates (inclusive).

    Use this tool when you have specific event dates (e.g., from emails), and need to determine how the stock changed in that time range.

    Dates must be in format 'YYYY-MM-DD'. Return the percent change and a short summary.

    Input format:
    - start_date (str): Format 'YYYY-MM-DD'
    - end_date (str): Format 'YYYY-MM-DD'

    Returns:
    * Stock price on {start_date}: $100
    * Stock price on {end_date}: $110
    * % change over time: 10%
    """
    import pandas as pd

    enron_stock = enron_stock_data()

    # Convert date columns to datetime if not already
    enron_stock['Date'] = pd.to_datetime(enron_stock['Date'])

    # Parse input dates
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    # Filter rows within date range (inclusive)
    df_range = enron_stock[(enron_stock['Date'] >= start_dt) & (enron_stock['Date'] <= end_dt)]

    if df_range.empty:
        return f"No stock data available between {start_date} and {end_date}."

    # Get closing prices on start and end dates (nearest available dates)
    start_close = df_range.iloc[0]['Close']
    end_close = df_range.iloc[-1]['Close']

    # Calculate change
    abs_change = end_close - start_close
    pct_change = (abs_change / start_close) * 100

    # Format result
    if abs_change >= 0:
        result = (f"Between {start_date} and {end_date}, Enron stock increased by "
                  f"{pct_change:.2f}% from ${start_close:.2f} to ${end_close:.2f}.")
    else:
        result = (f"Between {start_date} and {end_date}, Enron stock decreased by "
                  f"{abs(pct_change):.2f}% from ${start_close:.2f} to ${end_close:.2f}.")

    return result


embedding = HuggingFaceEmbeddings()

chroma_db = Chroma(
    collection_name="vector_store",
    embedding_function=embedding,
    ersist_directory="./chroma_db"
)

retriever = chroma_db.as_retriever()

from langchain_core.runnables import RunnableLambda

contextualize = RunnableLambda(lambda question: retriever.get_relevant_documents(question))
format_context = RunnableLambda(combine_docs)

from langchain_core.prompts import MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an assistant helping summarize changes in Enron's stock price 
    around the dates regarding messages from Jeffrey Skilling's emails.
    Always call `stock_over_time` after extracting relevant dates from emails.

You can use two tools:
1. `search_emails(question)` — to retrieve emails relevant to the user's question.
2. `stock_over_time(start_date, end_date)` — to look up Enron's stock performance during a specific time window.

When answering questions:
- First use `search_emails` to retrieve relevant emails.
- Carefully extract any relevant **event dates** from the relevant emails.
- Then call `stock_over_time` using those dates to report how Enron's stock performed during that period.

Always include dates and citations in your final answer.
If the information isn’t available, say so clearly.

Example question:
Which baseball games has Jeffrey Skilling gone to and when? How did the stock price change around that time?

Example answer:
Jeffrey Skilling was invited to an Astros game (vs Phillies) at the Enron Box at Enron Field on Monday, May 7th.
(Email Source: Christie Patrick@ECT, 05/04/2001 10:00 AM)

The enron stock price went from:
* Stock price on 05/04/2001: $59
* Stock price on 05/11/2001: $56.9
* % change over time: -3.6%

"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Question: {input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_aws.function_calling import ToolsOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# Tool list
tools = [search_emails, stock_over_time]

# Wrap LLM with tool-calling capability
llm_with_tools = model.bind_tools(tools)

# # Create agent
agent = create_tool_calling_agent(
    llm=llm_with_tools,
    tools=tools,
    prompt=prompt
)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory
)

# agent_executor = AgentExecutor(
#     agent=agent,
#     tools=tools,
#     memory=memory,
#     verbose=False,
#     callbacks=[stdout_callback]
# )

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
