import os

import yfinance as yf
from langchain_core.tools import tool
from qdrant_client import QdrantClient
from langchain_community.tools import DuckDuckGoSearchRun
from pydantic import BaseModel, Field

# 1. The Database Retriever
@tool
def search_financial_documents(query: str) -> str:
    """Searches the local Qdrant database for historical Apple financial data and risks."""
    print(f"   [Tool] Searching Database for: '{query}'")
    client = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    client.set_model("BAAI/bge-small-en-v1.5")
    client.set_sparse_model("Qdrant/bm25")
    
    results = client.query(collection_name="apple_financials", query_text=query, limit=3)
    context = "\n\n---\n\n".join([res.document for res in results])
    return context if context else "No relevant documents found."

# -------------------------------------------------------------------
# TOOL 1.5: The Dynamic Web Researcher
# -------------------------------------------------------------------
@tool
def search_web_for_company_risks(company_name: str) -> str:
    """
    Searches the live internet for a company's latest risk factors, 
    business operations, and recent news. Use this for ANY company.
    """
    print(f"   [Tool] Browsing the web for: {company_name} risk factors...")
    try:
        search = DuckDuckGoSearchRun()
        # We craft a highly specific search query for the AI
        query = f"{company_name} latest annual report risk factors and business challenges"
        results = search.invoke(query)
        return results
    except Exception as e:
        return f"Could not find web data for {company_name}."

# 2. Real-Time Price Fetcher
@tool
def get_live_stock_price(ticker: str) -> str:
    """Fetches the true, up-to-the-second live stock price for ANY ticker symbol."""
    print(f"   [Tool] Fetching live price for: {ticker}")
    try:
        stock = yf.Ticker(ticker)
        live_price = stock.fast_info['last_price'] 
        return f"The real-time live price of {ticker} is ${live_price:.2f}"
    except Exception:
        return f"Could not fetch live price for {ticker}."

# 3. Dynamic Financial Statements
@tool
def get_company_financials(ticker: str) -> str:
    """Fetches recent annual revenue and gross profit for ANY publicly traded company."""
    print(f"   [Tool] Fetching financials for: {ticker}")
    try:
        stock = yf.Ticker(ticker)
        financials = stock.financials
        recent_date = financials.columns[0]
        total_revenue = financials.loc["Total Revenue", recent_date]
        gross_profit = financials.loc["Gross Profit", recent_date]
        return f"Financials for {ticker} ({recent_date.date()}):\nRevenue: ${total_revenue:,.2f}\nGross Profit: ${gross_profit:,.2f}"
    except Exception:
        return f"Could not fetch financial statements for {ticker}."

# 4. The Calculator
@tool
def calculate_percentage_change(old_value: float, new_value: float) -> str:
    """Calculates the exact percentage change between two numbers."""
    print(f"   [Tool] Calculating math: {old_value} -> {new_value}")
    try:
        change = ((new_value - old_value) / old_value) * 100
        direction = "increase" if change >= 0 else "decrease"
        return f"That is a {abs(change):.2f}% {direction}."
    except ZeroDivisionError:
        return "Cannot divide by zero."

# Export the tools
financial_tools = [
    search_financial_documents, 
    search_web_for_company_risks, # <-- ADDED HERE
    get_live_stock_price, 
    get_company_financials, 
    calculate_percentage_change
]
