# 📈 Autonomous AI Financial Analyst

**Live Demo:** [Streamlit Web App](https://finance-agent-rag-8kicnxjsteh28k5gnnfvgo.streamlit.app/)

An enterprise-grade, agentic AI system that autonomously analyzes company earnings reports, retrieves live market data, and generates structured financial insights. 

Unlike standard LLM chatbots, this system utilizes a deterministic **LangGraph state machine** to break down complex financial queries, route tasks to specialized AI agents, and enforce self-correction via an automated Critic Agent before outputting a final markdown report.

---

## 🧠 System Architecture

This project implements a multi-agent collaborative network to eliminate hallucinations and ensure deterministic data retrieval:

* **Planner Agent:** Breaks down natural language user queries into a step-by-step execution strategy.
* **Doc Agent:** Handles unstructured text. Executes Hybrid Vector Search for static 10-K filings, or utilizes dynamic web search for live corporate risk extraction.
* **Market Agent:** Interfaces with live REST APIs to fetch down-to-the-second stock tick prices and recent corporate financial statements (Revenue, Gross Profit).
* **Critic Agent (Auditor):** Evaluates the gathered data against strict validation rules. If hallucination or missing data is detected, it rejects the state and forces the system to re-execute the tool calls.
* **Reporter Agent:** Compiles validated data into a structured format consisting of Financial Summaries, Market Insights, and Risk Factors.

---

## 🛠️ Tech Stack

* **Orchestration:** LangChain, LangGraph
* **LLM:** Google Gemini 1.5 Flash (via `langchain-google-genai`)
* **Vector Database:** Qdrant Cloud with FastEmbed (`BAAI/bge-small-en-v1.5` & `BM25`)
* **Live Data Tools:** `yfinance` (Global Markets), `duckduckgo-search` (Live Web)
* **Frontend UI:** Streamlit
* **Document Processing:** `pypdf`, Semantic Text Chunking

---

## 🚀 Quick Start (Local Deployment)

Want to run the multi-agent system locally? Follow these steps:

**1. Clone the repository**

git clone [https://github.com/MohdAnas-123/finance-agent-rag.git](https://github.com/MohdAnas-123/finance-agent-rag.git)
cd finance-agent-rag

**2. Install dependencies**

pip install -r requirements.txt

**3. Set up Environment Variables**

Create a .env file in the root directory and add your API keys:

GOOGLE_API_KEY=your_gemini_api_key
QDRANT_URL=your_qdrant_cloud_cluster_url
QDRANT_API_KEY=your_qdrant_api_key

**4. Run the Streamlit UI**

streamlit run app.py

📈 System Limitations & Future Scalability
During development, I identified a critical bottleneck regarding standard Retrieval-Augmented Generation (RAG) when dealing with global stock markets:

Current State: The system uses a Proof-of-Concept static Qdrant collection for deep RAG (Apple 10-K), while utilizing live web-search for other global tickers to prevent vector bloat on the local machine.

Future Work (Dynamic Ephemeral RAG): To support full, deep-document ingestion for any global stock without causing storage crashes, future iterations will implement "Just-In-Time" RAG. This pipeline will dynamically download SEC EDGAR filings, embed them into temporary Qdrant namespaces, and delete the collection immediately post-execution to optimize memory limits.
