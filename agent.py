from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Import the tools we built in the previous step
from tools import financial_tools

# Load environment variables (your API key)
load_dotenv()

# -------------------------------------------------------------------
# 1. Define the Graph State
# This acts as the "Memory" of the agent. It keeps a running list of all 
# messages, tool calls, and tool outputs as the agent loops.
# -------------------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]

# -------------------------------------------------------------------
# 2. Initialize the LLM
# We use gpt-4o-mini because it is exceptionally good at tool routing 
# and very cost-effective. We bind our custom tools to it.
# -------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite-preview')
llm_with_tools = llm.bind_tools(financial_tools)

# -------------------------------------------------------------------
# 3. Define the Nodes (The "Doers")
# -------------------------------------------------------------------
def chatbot_node(state: State):
    """The main brain. It reads the state and decides the next action."""
    print("   [Agent] Thinking...")
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# LangGraph provides a prebuilt ToolNode that automatically executes 
# the Python functions we defined in tools.py
tool_node = ToolNode(tools=financial_tools)

# -------------------------------------------------------------------
# 4. Build and Compile the Graph (The "Workflow")
# -------------------------------------------------------------------
graph_builder = StateGraph(State)

# Add our nodes to the graph
graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_node("tools", tool_node)

# Draw the edges (how the nodes connect)
graph_builder.add_edge(START, "chatbot")

# The conditional edge is the magic:
# If the LLM decides it needs a tool, route to "tools".
# If the LLM decides it has enough info to answer, route to END.
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# After a tool finishes running, always go back to the chatbot 
# so it can evaluate the result.
graph_builder.add_edge("tools", "chatbot")

# Compile the graph into an executable application
agent_app = graph_builder.compile()

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    print("\n--- Testing the Financial Agent ---")
    
    # A complex question that requires ALL THREE tools to answer correctly
    test_question = "Look up Apple's 2022 total net sales from the database, then get today's live AAPL stock price, and calculate the percentage difference between the two numbers (just for fun)."
    
    print(f"\nUser: {test_question}\n")
    
    # Run the graph
    events = agent_app.stream(
        {"messages": [HumanMessage(content=test_question)]},
        stream_mode="values"
    )
    
    for event in events:
        if "messages" in event:
            event["messages"][-1].pretty_print()