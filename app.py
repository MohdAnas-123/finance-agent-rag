import streamlit as st
from langchain_core.messages import HumanMessage
from multi_agent import financial_ai_system # Importing your genius brain!

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="AI Financial Analyst",
    page_icon="📈",
    layout="centered"
)

# --- 2. Header Section ---
st.title("📈 Autonomous AI Financial Analyst")
st.markdown("""
Welcome to the multi-agent financial research system. 
Enter a company ticker or a financial question, and my AI agents will:
1. Search the database and live web for risk factors.
2. Fetch up-to-the-second live stock prices and financials.
3. Audit the data and generate a professional report.
""")
st.divider()

# --- 3. User Input ---
query = st.text_input(
    "What would you like to analyze?", 
    placeholder="e.g., Analyze the latest risks and market performance for Tesla (TSLA)"
)

# --- 4. Execution & UI Streaming ---
if st.button("Run Analysis", type="primary"):
    if not query:
        st.warning("Please enter a company or ticker symbol to begin.")
    else:
        # A nice loading spinner while the agents do their work
        with st.spinner("Agents are gathering and auditing data. This may take a few seconds..."):
            try:
                # Set up the initial state for the graph
                initial_state = {
                    "messages": [HumanMessage(content=query)], 
                    "is_valid": True, 
                    "feedback": "", 
                    "sender": "user"
                }
                
                # Run the LangGraph system
                final_state = financial_ai_system.invoke(initial_state)
                
                # Extract and display the final report
                if final_state and "messages" in final_state:
                    final_content = final_state["messages"][-1].content
                    
                    st.success("Analysis Complete!")
                    st.divider()
                    
                    # Safely unpack Gemini's text format and render as Markdown
                    if isinstance(final_content, list):
                        st.markdown(final_content[0].get("text", ""))
                    else:
                        st.markdown(final_content)
                        
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")