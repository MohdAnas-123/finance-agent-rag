import os
import urllib.request
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore

# Load your Qdrant Cloud URLs and API keys from the .env file
load_dotenv()

# We use a real Apple Q4 Financial Statement for our data source
APPLE_REPORT_URL = "https://s2.q4cdn.com/470004039/files/doc_financials/2022/q4/FY22_Q4_Consolidated_Financial_Statements.pdf"
PDF_PATH = "data/apple_financials.pdf"

def download_financial_data():
    """Downloads the financial PDF if it doesn't already exist."""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(PDF_PATH):
        print("Downloading Apple Financial Report...")
        req = urllib.request.Request(APPLE_REPORT_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(PDF_PATH, 'wb') as out_file:
            out_file.write(response.read())
        print("✅ Download complete.")
    else:
        print("✅ PDF already exists in data folder.")

def parse_and_chunk_report(file_path):
    """Loads the PDF and chunks it semantically based on financial document structure."""
    print(f"Parsing document: {file_path}...")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    full_text = "\n".join([page.page_content for page in pages])

    financial_separators = ["\nITEM ", "\nPART ", "\nNote ", "\n\n", "\n", ". ", " "]
    text_splitter = RecursiveCharacterTextSplitter(
        separators=financial_separators,
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )
    chunks = text_splitter.create_documents([full_text])
    print(f"✅ Successfully split the report into {len(chunks)} semantic chunks.")
    return chunks

def push_to_qdrant_cloud(chunks):
    """Embeds the chunks and uploads them to your live Qdrant Cloud cluster."""
    print("🧠 Generating embeddings and uploading to Qdrant Cloud... (This may take a minute)")
    
    # We use the same BAAI fastembed model you specified in your architecture
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    # This creates the collection and uploads the vectors securely to the cloud
    QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        collection_name="apple_financials",
        force_recreate=True # Overwrites the database if it already exists
    )
    print("🚀 Successfully uploaded all vectors to Qdrant Cloud!")

if __name__ == "__main__":
    print("--- STARTING CLOUD INGESTION PIPELINE ---")
    download_financial_data()
    document_chunks = parse_and_chunk_report(PDF_PATH)
    
    # The crucial missing step!
    push_to_qdrant_cloud(document_chunks)
    
    print("--- INGESTION COMPLETE. YOU CAN NOW RUN YOUR APP! ---")