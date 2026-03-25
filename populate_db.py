from qdrant_client import QdrantClient
from ingest import parse_and_chunk_report, PDF_PATH

# 1. Connect to our local Dockerized Qdrant instance
client = QdrantClient(url="http://localhost:6333")

# 2. Configure the Hybrid Embedding Models (Running locally via FastEmbed)
# Dense model for meaning
client.set_model("BAAI/bge-small-en-v1.5")
# Sparse model for exact keyword matching
client.set_sparse_model("Qdrant/bm25")

def ingest_to_qdrant(file_path: str, collection_name: str = "apple_financials"):
    print(f"Fetching chunks for {file_path}...")
    
    # Get the chunks from our ingest script
    chunks = parse_and_chunk_report(file_path)
    if not chunks:
        print("❌ No chunks found. Exiting.")
        return

    # Extract text and create metadata so the LLM knows exactly where info came from
    documents = [chunk.page_content for chunk in chunks]
    metadata = [{"source": file_path, "chunk_index": i} for i in range(len(chunks))]

    print(f"Generating hybrid embeddings and uploading {len(documents)} chunks to Qdrant...")
    print("This might take a minute on the first run as it downloads the embedding models.")
    
    # client.add() automatically creates the collection, calculates both dense & sparse vectors, and uploads them
    client.add(
        collection_name=collection_name,
        documents=documents,
        metadata=metadata,
        parallel=4 # Uses multiple CPU cores to speed things up
    )
    
    print(f"✅ Successfully ingested data into collection: '{collection_name}'")

if __name__ == "__main__":
    ingest_to_qdrant(PDF_PATH)