import os
import warnings
from typing import Annotated, TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from tools import (
    search_financial_documents, 
    search_web_for_company_risks,
    get_live_stock_price, 
    calculate_percentage_change,
    get_company_financials
)

# Hide annoying database warnings to keep the terminal clean
warnings.filterwarnings("ignore")
load_dotenv()

# --- 1. Initialize LLMs & Bind Tools ---
base_llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0)

doc_tools = [search_financial_documents, search_web_for_company_risks]
doc_llm = base_llm.bind_tools(doc_tools)

market_tools = [get_live_stock_price, get_company_financials, calculate_percentage_change]
market_llm = base_llm.bind_tools(market_tools)

all_tools = doc_tools + market_tools

# --- 2. State & Pydantic Schemas ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    is_valid: bool
    feedback: str
    sender: str 

class CriticResponse(BaseModel):
    is_valid: bool = Field(description="True if analysis is accurate, False if it needs rework.")
    feedback: str = Field(description="Feedback on what is missing or wrong.")

# --- 3. The Agent Nodes (With STATE AWARENESS!) ---
def planner_node(state: AgentState):
    print("\n📝 [Planner] Organizing strategy...")
    sys_msg = SystemMessage(content="You are the Head Planner. Write a brief 2-step plan: 1. Doc Agent searches database. 2. Market Agent fetches live prices/financials.")
    if not state.get("is_valid") and state.get("feedback"):
        sys_msg.content += f"\nCRITIC FEEDBACK TO FIX: {state['feedback']}"
    response = base_llm.invoke([sys_msg] + state["messages"])
    return {"messages": [response], "sender": "planner"}

def doc_agent_node(state: AgentState):
    print("\n📄 [Doc Agent] Processing documents...")
    # Check if the last message in memory is a Tool result
    if state["messages"] and isinstance(state["messages"][-1], ToolMessage):
        # We got the data! Force the LLM to summarize and STOP calling tools.
        sys_msg = SystemMessage(content="You now have the database search results. Summarize the risk factors strictly in 2 sentences. DO NOT use any tools.")
        # Notice we use base_llm so it physically cannot use tools here
        response = base_llm.invoke([sys_msg] + state["messages"]) 
    else:
        sys_msg = SystemMessage(content="You are the Doc Agent. If the user asks about Apple, use the search_financial_documents tool. If they ask about ANY OTHER company, use the search_web_for_company_risks tool. Use your tool exactly ONCE.")
        response = doc_llm.invoke([sys_msg] + state["messages"])
        
    return {"messages": [response], "sender": "doc_agent"}

def market_agent_node(state: AgentState):
    print("\n📈 [Market Agent] Processing market data...")
    if state["messages"] and isinstance(state["messages"][-1], ToolMessage):
        sys_msg = SystemMessage(content="You now have the live market data. Summarize the price and financials strictly in 2 sentences. DO NOT use any tools.")
        response = base_llm.invoke([sys_msg] + state["messages"])
    else:
        sys_msg = SystemMessage(content="You are the Market Agent. You MUST use the get_live_stock_price tool exactly ONCE. Do not output text, just call the tool.")
        response = market_llm.invoke([sys_msg] + state["messages"])
        
    return {"messages": [response], "sender": "market_agent"}

def critic_agent_node(state: AgentState):
    print("\n🧐 [Critic] Auditing the data...")
    sys_msg = SystemMessage(content="You are the Lead Auditor. Review the conversation. Did they gather data? Output is_valid=True if yes, False if hallucinated.")
    structured_critic = base_llm.with_structured_output(CriticResponse)
    evaluation = structured_critic.invoke([sys_msg] + state["messages"])
    
    if evaluation.is_valid: print("   ✅ APPROVED: Data looks solid. Sending to Reporter.")
    else: print(f"   ❌ REJECTED: {evaluation.feedback}")
    return {"is_valid": evaluation.is_valid, "feedback": evaluation.feedback, "sender": "critic"}

def reporter_node(state: AgentState):
    print("\n✍️ [Reporter] Formatting final professional report...")
    sys_msg = SystemMessage(content="You are the Lead Financial Analyst. Write a final, structured Markdown report based on the data gathered. Use headings: ### Financial Summary, ### Market Insight, ### Risk Factors. DO NOT USE TOOLS.")
    response = base_llm.invoke([sys_msg] + state["messages"])
    return {"messages": [response], "sender": "reporter"}

# --- 4. Build the Graph ---
builder = StateGraph(AgentState)

builder.add_node("planner", planner_node)
builder.add_node("doc_agent", doc_agent_node)
builder.add_node("market_agent", market_agent_node)
builder.add_node("critic", critic_agent_node)
builder.add_node("reporter", reporter_node)
builder.add_node("tools", ToolNode(all_tools))

builder.add_edge(START, "planner")
builder.add_edge("planner", "doc_agent")

builder.add_conditional_edges("doc_agent", tools_condition, {"tools": "tools", "__end__": "market_agent"})
builder.add_conditional_edges("market_agent", tools_condition, {"tools": "tools", "__end__": "critic"})

def route_tool_output(state: AgentState):
    return state["sender"]

builder.add_conditional_edges("tools", route_tool_output)

def critic_router(state: AgentState):
    if state.get("is_valid"): return "reporter"
    else: return "planner"

builder.add_conditional_edges("critic", critic_router)
builder.add_edge("reporter", END)

financial_ai_system = builder.compile()

# --- 5. Run it! ---
if __name__ == "__main__":
    print("\n--- Booting Autonomous Financial Analyst ---")
    test_question = "Find Apple's risk factors from the database, then fetch their live stock price (AAPL) and recent financials."
    print(f"\nUser: {test_question}\n")
    
    initial_state = {"messages": [HumanMessage(content=test_question)], "is_valid": True, "feedback": "", "sender": "user"}
    
    # Run the graph cleanly
    final_state = financial_ai_system.invoke(initial_state)
    
    print("\n" + "="*60)
    print("FINAL PROFESSIONAL REPORT")
    print("="*60 + "\n")
    
    # Directly print the content of the very last message (the Reporter's output)
    if final_state and "messages" in final_state:
         print(final_state["messages"][-1].content)