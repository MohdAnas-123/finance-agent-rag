import yfinance as yf
from langchain_core.tools import tool
from qdrant_client import QdrantClient

# -------------------------------------------------------------------
# TOOL 1: The Historical Database Retriever
# -------------------------------------------------------------------
@tool
def search_financial_documents(query: str) -> str:
    """
    Searches the Apple 10-K financial report for historical data, 
    company operations, risk factors, and past revenue.
    Use this when the user asks about past performance or corporate structure.
    """
    print(f"   [Tool Execution] Searching Qdrant Database for: '{query}'")
    client = QdrantClient(url="http://localhost:6333")
    
    # Using Qdrant's native query method which automatically uses the FastEmbed models
    # we configured in Phase 1 to perform hybrid search.
    results = client.query(
        collection_name="apple_financials",
        query_text=query,
        limit=3
    )
    
    # FastEmbed stores the original text in the 'document' field
    context = "\n\n---\n\n".join([res.document for res in results])
    return context if context else "No relevant financial documents found."

# -------------------------------------------------------------------
# TOOL 2: The Live Market Fetcher
# -------------------------------------------------------------------
@tool
def get_live_stock_price(ticker: str) -> str:
    """
    Fetches the current, live stock price for a given ticker symbol (e.g., AAPL, NVDA).
    Use this strictly when the user asks for 'current', 'today', or 'live' stock prices.
    """
    print(f"   [Tool Execution] Fetching live market data for: {ticker}")
    try:
        stock = yf.Ticker(ticker)
        # Get the most recent day's closing price
        price = stock.history(period="1d")['Close'].iloc[-1]
        return f"The current live price of {ticker} is ${price:.2f}"
    except Exception as e:
        return f"Could not fetch price for {ticker}. Check if the ticker symbol is correct."

# -------------------------------------------------------------------
# TOOL 3: The Deterministic Calculator
# -------------------------------------------------------------------
@tool
def calculate_percentage_change(old_value: float, new_value: float) -> str:
    """
    Calculates the exact percentage change (like YoY growth) between two numbers.
    Always use this tool instead of calculating math in your head.
    """
    print(f"   [Tool Execution] Calculating math: {old_value} -> {new_value}")
    try:
        change = ((new_value - old_value) / old_value) * 100
        direction = "increase" if change >= 0 else "decrease"
        return f"That is a {abs(change):.2f}% {direction}."
    except ZeroDivisionError:
        return "Cannot divide by zero."

# Combine them into an array so we can bind them to our Agent later
financial_tools = [search_financial_documents, get_live_stock_price, calculate_percentage_change]

if __name__ == "__main__":
    # Quick sanity check to make sure our tools work locally
    print("Testing Market Tool:", get_live_stock_price.invoke({"ticker": "AAPL"}))
    print("Testing Math Tool:", calculate_percentage_change.invoke({"old_value": 394328, "new_value": 383285}))