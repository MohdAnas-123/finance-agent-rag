import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from langchain_core.messages import HumanMessage
from multi_agent import financial_ai_system

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="AI Financial Analyst",
    page_icon="📈",
    layout="wide" # Upgraded to wide mode for dashboards!
)

# --- 2. Header Section ---
st.title("📈 Autonomous AI Financial Analyst")
st.markdown("""
Welcome to the multi-agent financial research system. 
Enter a financial question and the target ticker symbol. The AI agents will audit the data and generate a report, while the dashboard renders live market trends.
""")
st.divider()

# --- 3. User Input (Split into columns) ---
input_col1, input_col2 = st.columns([3, 1])

with input_col1:
    query = st.text_input(
        "What would you like to analyze?", 
        placeholder="e.g. Analyze the latest risks and market performance for Nvidia."
    )
with input_col2:
    ticker = st.text_input("Ticker Symbol", placeholder="e.g. NVDA").upper()

# --- 4. Execution & UI Streaming ---
if st.button("Run Comprehensive Analysis", type="primary"):
    if not query or not ticker:
        st.warning("Please enter both a query and a ticker symbol to begin.")
    else:
        with st.spinner("Agents are gathering data and auditing results. This may take a few seconds..."):
            
            # --- A. Fetch Deterministic Chart Data ---
            try:
                stock_data = yf.Ticker(ticker)
                hist = stock_data.history(period="6mo") 
                
                # ADD THIS CHECK:
                if hist.empty:
                    st.error(f"Could not find market data. Please ensure '{ticker}' is a valid stock ticker symbol (e.g., AAPL, NVDA).")
                    fig = None
                else:
                    # Create an interactive Candlestick chart
                    fig = go.Figure(data=[go.Candlestick(x=hist.index,
                                    open=hist['Open'],
                                    high=hist['High'],
                                    low=hist['Low'],
                                    close=hist['Close'])])
                    
                    fig.update_layout(
                        title=f"{ticker} 6-Month Stock Trend", 
                        template="plotly_dark", 
                        margin=dict(l=20, r=20, t=40, b=20),
                        xaxis_rangeslider_visible=False
                    )
                    
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2]
                    price_change = current_price - prev_price
                    pct_change = (price_change / prev_price) * 100
                    
            except Exception as e:
                fig = None
                st.error(f"Could not load chart data for {ticker}. Error: {e}")

            # --- B. Run the LangGraph Brain ---
            try:
                # Inject the ticker into the query so agents know which stock to analyze
                enriched_query = f"{query} (Ticker: {ticker})" if ticker else query
                initial_state = {
                    "messages": [HumanMessage(content=enriched_query)], 
                    "is_valid": True, 
                    "feedback": "", 
                    "sender": "user",
                    "retry_count": 0
                }
                
                # ADD THE RECURSION LIMIT HERE (e.g., max 50 agent steps)
                final_state = financial_ai_system.invoke(
                    initial_state, 
                    config={"recursion_limit": 50} 
                )
                
                report_success = True
            except Exception as e:
                report_success = False
                st.error(f"An error occurred during AI analysis: {e}")

        # --- 5. Render the Dashboard ---
        if report_success:
            st.success("Analysis Complete!")
            st.divider()
            
            # Create a 2-column layout for the results
            report_col, chart_col = st.columns([1.5, 1])
            
            # Left Column: The AI Report
            with report_col:
                st.subheader("📝 AI Analyst Report")
                # Extract report text robustly from the final state
                report_text = None
                # Scan messages backward to find the last one with real text content
                for msg in reversed(final_state["messages"]):
                    content = msg.content if hasattr(msg, 'content') else None
                    if content is None:
                        continue
                    if isinstance(content, str) and content.strip():
                        report_text = content
                        break
                    elif isinstance(content, list):
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict) and "text" in block and block["text"].strip():
                                text_parts.append(block["text"])
                            elif isinstance(block, str) and block.strip():
                                text_parts.append(block)
                        if text_parts:
                            report_text = "\n".join(text_parts)
                            break
                
                if report_text:
                    st.markdown(report_text)
                else:
                    st.warning("No report content was generated. Please try again.")
                    
            # Right Column: The Data Visualizations
            with chart_col:
                st.subheader("📊 Market Data")
                if fig is not None:
                    # Display the large metric number
                    st.metric(
                        label=f"{ticker} Current Price", 
                        value=f"${current_price:.2f}", 
                        delta=f"{price_change:.2f} ({pct_change:.2f}%)"
                    )
                    # Display the interactive chart
                    st.plotly_chart(fig, use_container_width=True)