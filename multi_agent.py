from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

# -------------------------------------------------------------------
# 1. Define the Multi-Agent State
# -------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    current_agent: str
    is_valid: bool  # The Critic will use this to approve or reject the work

# -------------------------------------------------------------------
# 2. Define the Agent Nodes (Skeleton Structure)
# -------------------------------------------------------------------
def planner_node(state: AgentState):
    print("\n📝 [Planner Agent] Breaking down the user query into a step-by-step financial plan...")
    return {"current_agent": "planner"}

def doc_agent_node(state: AgentState):
    print("📄 [Doc Agent] Searching Qdrant for 10-K financial tables and risks...")
    return {"current_agent": "doc_agent"}

def market_agent_node(state: AgentState):
    print("📈 [Market Agent] Fetching live stock price and calculating ratios...")
    return {"current_agent": "market_agent"}

def critic_agent_node(state: AgentState):
    print("🧐 [Critic Agent] Reviewing the combined data for consistency and hallucinations...")
    # For this skeleton test, we will hardcode this to True. 
    # Later, Gemini will make this decision!
    return {"current_agent": "critic", "is_valid": True}

# -------------------------------------------------------------------
# 3. Build the Graph Workflow
# -------------------------------------------------------------------
builder = StateGraph(AgentState)

# Add our specialized agents to the graph
builder.add_node("planner", planner_node)
builder.add_node("doc_agent", doc_agent_node)
builder.add_node("market_agent", market_agent_node)
builder.add_node("critic", critic_agent_node)

# Draw the standard directed flow
builder.add_edge(START, "planner")
builder.add_edge("planner", "doc_agent")
builder.add_edge("doc_agent", "market_agent")
builder.add_edge("market_agent", "critic")

# -------------------------------------------------------------------
# 4. The Self-Correcting Loop (The Critic's Router)
# -------------------------------------------------------------------
def critic_router(state: AgentState):
    """If the Critic finds an error, it sends the flow back to the Planner."""
    if state.get("is_valid"):
        print("✅ [System] Critic approved the analysis. Generating final report.")
        return END
    else:
        print("❌ [System] Critic rejected the analysis! Sending back to Planner for rework.")
        return "planner"

# The edge leaving the Critic is conditional based on the router function
builder.add_conditional_edges("critic", critic_router)

# Compile the final application
financial_ai_system = builder.compile()

if __name__ == "__main__":
    print("\n--- Booting Multi-Agent System (Skeleton Test) ---")
    
    # We pass a test query and set our initial validation to False
    initial_state = {
        "messages": [HumanMessage(content="Analyze Apple's Q4 revenue and today's stock price.")],
        "is_valid": False
    }
    
    # Run the graph
    events = financial_ai_system.stream(initial_state, stream_mode="values")
    
    # This loop just triggers the nodes so we can see the print statements
    for event in events:
        pass