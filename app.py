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

# --- 1.5 Session State & Sidebar ---
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "current_view" not in st.session_state:
    st.session_state.current_view = None

with st.sidebar:
    if st.button("➕ New Analysis", use_container_width=True, type="primary"):
        st.session_state.current_view = None
        st.rerun()
    st.divider()
    
    st.header("📋 Recent Analyses")
    if not st.session_state.analysis_history:
        st.info("No run history yet.")
    else:
        for i, record in enumerate(reversed(st.session_state.analysis_history)):
            label = f"{record['ticker']}: {record['query'][:30]}..."
            if st.button(label, key=f"hist_{len(st.session_state.analysis_history)-1-i}"):
                st.session_state.current_view = record

# --- 2. Header Section ---
st.title("📈 Autonomous AI Financial Analyst")
st.markdown("""
Welcome to the multi-agent financial research system. 
Enter a financial question and the target ticker symbol. The AI agents will audit the data and generate a report, while the dashboard renders live market trends.
""")
st.divider()

# --- 3. User Input ---
TEMPLATES = {
    "Custom query...": "",
    "Risk & Market Analysis": "Analyze the company's latest risks and market performance.",
    "Intrinsic Value (DCF)": "What is the intrinsic value of the company assuming a 5% growth rate, a 10% discount rate, and a 15x terminal multiple?",
    "Revenue Growth (CAGR)": "Fetch the company's latest financials and calculate the revenue growth rate over the last 3 years.",
    "Comprehensive Audit": "Give me a comprehensive risk and valuation report for the company."
}

if "query_input" not in st.session_state:
    st.session_state.query_input = ""

def update_query_from_template():
    choice = st.session_state.template_choice
    st.session_state.query_input = TEMPLATES[choice]

st.selectbox(
    "💡 Preset Templates", 
    list(TEMPLATES.keys()), 
    key="template_choice",
    on_change=update_query_from_template
)

input_col1, input_col2 = st.columns([3, 1])

with input_col1:
    query = st.text_input(
        "What would you like to analyze?", 
        key="query_input",
        placeholder="e.g. Analyze the latest risks and market performance for Nvidia."
    )
with input_col2:
    ticker = st.text_input("Ticker Symbol", placeholder="e.g. NVDA").upper()

# --- 4. Execution & UI Streaming ---
if st.button("Run Comprehensive Analysis", type="primary"):
    if not query or not ticker:
        st.warning("Please enter both a query and a ticker symbol to begin.")
    else:
        progress_bar = st.progress(5, text="🚀 Initializing Analysis Engine...")
        with st.container():
            
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
                
                # Stream the agent execution to update progress bar
                accumulated_messages = initial_state["messages"].copy()
                for event in financial_ai_system.stream(initial_state, config={"recursion_limit": 50}):
                    for node_name, output in event.items():
                        if "messages" in output:
                            accumulated_messages.extend(output["messages"])
                            
                        # Update progress bar based on node_name
                        if node_name == "planner":
                            progress_bar.progress(20, text="📄 Doc Agent reviewing documents...")
                        elif node_name == "doc_agent":
                            progress_bar.progress(40, text="📈 Market Agent analyzing quantitative data...")
                        elif node_name == "market_agent":
                            progress_bar.progress(60, text="🧐 Critic auditing results...")
                        elif node_name == "tools":
                            progress_bar.progress(75, text="🛠️ Fetching real-world tools data...")
                        elif node_name == "critic":
                            progress_bar.progress(90, text="✍️ Reporter formatting final report...")
                        elif node_name == "reporter":
                            progress_bar.progress(100, text="✅ Analysis complete!")

                final_state = {"messages": accumulated_messages}
                
                # Extract report text robustly from the final state
                report_text = None
                for msg in reversed(final_state["messages"]):
                    content = msg.content if hasattr(msg, 'content') else None
                    if content is None: continue
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
                            
                # Save to history
                record = {
                    "query": query,
                    "ticker": ticker,
                    "report_text": report_text,
                    "fig": fig,
                    "current_price": current_price if 'current_price' in locals() else None,
                    "price_change": price_change if 'price_change' in locals() else None,
                    "pct_change": pct_change if 'pct_change' in locals() else None
                }
                st.session_state.analysis_history.append(record)
                if len(st.session_state.analysis_history) > 10:
                    st.session_state.analysis_history.pop(0)
                st.session_state.current_view = record

            except Exception as e:
                st.error(f"An error occurred during AI analysis: {e}")

# --- 5. Render the Dashboard ---
if st.session_state.current_view:
    view = st.session_state.current_view
    st.success("Analysis Ready!")
    st.divider()
    
    # Create a 2-column layout for the results
    report_col, chart_col = st.columns([1.5, 1])
    
    # Left Column: The AI Report
    with report_col:
        st.subheader("📝 AI Analyst Report")
        if view["report_text"]:
            st.markdown(view["report_text"])
        else:
            st.warning("No report content was generated. Please try again.")
            
    # Right Column: The Data Visualizations
    with chart_col:
        st.subheader("📊 Market Data")
        if view["fig"] is not None:
            # Display the large metric number
            st.metric(
                label=f"{view['ticker']} Current Price", 
                value=f"${view['current_price']:.2f}", 
                delta=f"{view['price_change']:.2f} ({view['pct_change']:.2f}%)"
            )
            # Display the interactive chart
            st.plotly_chart(view["fig"], use_container_width=True)

# --- 6. Multi-Ticker Comparison ---
st.divider()
st.header("📊 Multi-Ticker Market Comparison")
st.markdown("Compare actual market data for up to 3 tickers side-by-side.")

tickers_input = st.text_input("Enter up to 3 comma-separated tickers", placeholder="e.g. AAPL, MSFT, GOOG")

if st.button("Compare Market Performance", type="secondary"):
    tickers_list = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if not tickers_list:
        st.warning("Please enter at least one ticker.")
    elif len(tickers_list) > 3:
        st.warning("Please enter a maximum of 3 tickers.")
    else:
        with st.status("Fetching market data...", expanded=True) as status:
            data_frames = {}
            for t in tickers_list:
                try:
                    hist = yf.Ticker(t).history(period="6mo")
                    if not hist.empty:
                        data_frames[t] = hist
                except:
                    pass
            
            if not data_frames:
                status.update(label="Failed to fetch data", state="error")
                st.error("Could not retrieve market data for the provided tickers.")
            else:
                status.update(label="Data fetched successfully!", state="complete")
                
                # Plotly Normalized Line Chart
                fig_comp = go.Figure()
                for t, hist in data_frames.items():
                    if len(hist) > 0:
                        first_price = hist['Close'].iloc[0]
                        normalized = (hist['Close'] / first_price) * 100
                        fig_comp.add_trace(go.Scatter(x=hist.index, y=normalized, mode='lines', name=t))
                        
                fig_comp.update_layout(
                    title="6-Month Normalized Performance (Base=100)",
                    template="plotly_dark",
                    hovermode="x unified",
                    yaxis_title="Normalized Price (%)",
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_comp, use_container_width=True)
                
                # Columns for metrics
                cols = st.columns(max(len(data_frames), 3))
                for i, (t, hist) in enumerate(data_frames.items()):
                    with cols[i]:
                        current_price = hist['Close'].iloc[-1]
                        prev_price = hist['Close'].iloc[-2]
                        latest_pct = ((current_price - prev_price) / prev_price) * 100
                        
                        st.subheader(f"{t}")
                        st.metric(
                            "Current Price", 
                            f"${current_price:.2f}", 
                            f"{latest_pct:.2f}% (1d)"
                        )