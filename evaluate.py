import os
from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain_core.messages import HumanMessage

# Import your actual LangGraph brain
from multi_agent import financial_ai_system 

load_dotenv()
client = Client()

# --- 1. Define the Test Dataset (Ground Truth) ---
# We are creating a dataset of companies and known facts to test against.
eval_dataset = [
    {
        "inputs": {"query": "What was Apple's total revenue in Q4 2022?"},
        "outputs": {"expected_fact": "394.3"} # In billions
    },
    {
        "inputs": {"query": "What is the ticker symbol for Tesla?"},
        "outputs": {"expected_fact": "TSLA"}
    }
]

# --- 2. Create the Dataset in LangSmith ---
dataset_name = "Finance_Agent_Accuracy_Test"
if not client.has_dataset(dataset_name=dataset_name):
    dataset = client.create_dataset(
        dataset_name=dataset_name, 
        description="Automated tests for the Financial AI Agent."
    )
    for example in eval_dataset:
        client.create_example(
            inputs=example["inputs"], 
            outputs=example["outputs"], 
            dataset_id=dataset.id
        )

# --- 3. Define the Target (Crash-Proof Wrapper) ---
def predict_agent_response(inputs: dict) -> dict:
    """Wrapper function to pass the dataset query to your LangGraph system."""
    initial_state = {
        "messages": [HumanMessage(content=inputs["query"])], 
        "is_valid": True, 
        "feedback": "", 
        "sender": "user"
    }
    
    try:
        # Try to run the agent
        final_state = financial_ai_system.invoke(initial_state)
        
        final_content = final_state["messages"][-1].content
        if isinstance(final_content, list):
            final_text = final_content[0].get("text", "")
        else:
            final_text = final_content
            
        return {"actual_output": final_text}
        
    except Exception as e:
        # If the agent crashes, return the exact error message!
        return {"actual_output": f"AGENT CRASHED: {str(e)}"}

# --- 4. Define the Evaluator (Safe Grader) ---
def check_factual_accuracy(run, example) -> dict:
    """Checks if the exact expected fact appears in the AI's final report."""
    expected = example.outputs["expected_fact"]
    
    # Safe check: Did the agent return anything at all?
    if not run.outputs or "actual_output" not in run.outputs:
        return {"key": "factual_accuracy", "score": 0}
        
    actual = run.outputs["actual_output"]
    
    # Did it crash? Score 0. Otherwise, check for the expected fact.
    if "AGENT CRASHED" in actual:
        score = 0
    else:
        score = 1 if expected in actual else 0
        
    return {"key": "factual_accuracy", "score": score}

# --- 5. Run the Evaluation ---
if __name__ == "__main__":
    print("🧪 Starting Automated Agent Evaluation...")
    
    experiment_results = evaluate(
        predict_agent_response,
        data=dataset_name,
        evaluators=[check_factual_accuracy],
        experiment_prefix="ci-test-run",
    )
    
    print("\n✅ Evaluation complete! View the detailed tracing report in your LangSmith dashboard.")