import os
import math
import yfinance as yf
from langchain_core.tools import tool
from langchain_qdrant import QdrantVectorStore
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from pydantic import BaseModel, Field

# 1. The Database Retriever
@tool
def search_financial_documents(query: str) -> str:
    """Searches the Qdrant vector database for SEC filings and financial documents."""
    print(f"   [Tool] Searching documents for: {query}")
    try:
        # 1. Initialize the exact same embedding model used during ingestion
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        
        # 2. Connect to the LangChain-managed Qdrant collection
        vector_store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name="apple_financials",
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
        )
        
        # 3. Perform the search
        results = vector_store.similarity_search(query, k=3)
        
        if not results:
            return "No relevant financial documents found."
            
        # Format the chunks into a single string for the LLM
        return "\n\n".join([doc.page_content for doc in results])
    except Exception as e:
        return f"Failed to search documents: {e}"

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
    
# -------------------------------------------------------------------
# TOOL 6: Compound Annual Growth Rate (CAGR)
# -------------------------------------------------------------------
@tool
def calculate_cagr(start_value: float, end_value: float, years: int) -> str:
    """
    Calculates the Compound Annual Growth Rate (CAGR).
    Use this when the user asks about revenue growth, profit growth, 
    or annualized returns over a specific number of years.
    """
    print(f"   [Tool] Calculating CAGR over {years} years...")
    try:
        if start_value <= 0 or years <= 0:
            return "Error: Start value and years must be greater than zero."
        
        cagr = (end_value / start_value) ** (1 / years) - 1
        return f"The CAGR is {cagr * 100:.2f}%"
    except Exception as e:
        return f"Failed to calculate CAGR: {e}"

# -------------------------------------------------------------------
# TOOL 7: Discounted Cash Flow (DCF) Valuation
# -------------------------------------------------------------------
@tool
def calculate_basic_dcf(
    current_free_cash_flow: float, 
    growth_rate: float, 
    discount_rate: float, 
    terminal_multiple: float, 
    years_to_project: int = 5
) -> str:
    """
    Performs a basic Discounted Cash Flow (DCF) valuation to find the intrinsic value of a business.
    Inputs required (as decimals, e.g., 0.10 for 10%):
    - current_free_cash_flow: The most recent year's FCF.
    - growth_rate: Projected annual growth rate of FCF.
    - discount_rate: The required rate of return (WACC).
    - terminal_multiple: The exit multiple to calculate terminal value.
    """
    print(f"   [Tool] Executing DCF Valuation Engine...")
    try:
        total_present_value = 0
        projected_fcf = current_free_cash_flow
        
        # 1. Calculate Present Value of projected cash flows
        for year in range(1, years_to_project + 1):
            projected_fcf *= (1 + growth_rate)
            discount_factor = (1 + discount_rate) ** year
            present_value = projected_fcf / discount_factor
            total_present_value += present_value
            
        # 2. Calculate Terminal Value (Exit Multiple Method)
        terminal_value = projected_fcf * terminal_multiple
        pv_terminal_value = terminal_value / ((1 + discount_rate) ** years_to_project)
        
        # 3. Sum for Enterprise Value
        intrinsic_enterprise_value = total_present_value + pv_terminal_value
        
        return f"Estimated Intrinsic Enterprise Value: ${intrinsic_enterprise_value:,.2f} (Based on {years_to_project}yr projection)"
    except Exception as e:
        return f"Failed to calculate DCF: {e}"

# Export the tools
financial_tools = [
    search_financial_documents, 
    search_web_for_company_risks, # <-- ADDED HERE
    get_live_stock_price, 
    get_company_financials, 
    calculate_percentage_change,
    calculate_cagr,        # <-- ADDED
    calculate_basic_dcf
]
