from qdrant_client import QdrantClient

def test_qdrant():
    try:
        client = QdrantClient(url="http://localhost:6333")
        collections = client.get_collections()
        print("✅ Successfully connected to Qdrant!")
        print(f"Collections available: {collections.collections}")
    except Exception as e:
        print("❌ Failed to connect. Check if Docker is running.")
        print(e)

if __name__ == "__main__":
    test_qdrant()