import os
import urllib.request
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# We use a real Apple Q4 Financial Statement for our data source
APPLE_REPORT_URL = "https://s2.q4cdn.com/470004039/files/doc_financials/2022/q4/FY22_Q4_Consolidated_Financial_Statements.pdf"
PDF_PATH = "data/apple_financials.pdf"

def download_financial_data():
    """Downloads the financial PDF if it doesn't already exist."""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(PDF_PATH):
        print("Downloading Apple Financial Report...")
        # SEC and Investor sites require a User-Agent header to not block the request
        req = urllib.request.Request(APPLE_REPORT_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(PDF_PATH, 'wb') as out_file:
            out_file.write(response.read())
        print("✅ Download complete.")
    else:
        print("✅ PDF already exists in data folder.")

def parse_and_chunk_report(file_path):
    """Loads the PDF and chunks it semantically based on financial document structure."""
    print(f"Parsing document: {file_path}...")
    
    # Load the PDF
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    # Combine all pages into a single string for structural chunking
    full_text = "\n".join([page.page_content for page in pages])

    # We define custom separators specific to corporate filings.
    # The splitter will prioritize keeping entire "Notes" or "Items" together before splitting by paragraph.
    financial_separators = [
        "\nITEM ", 
        "\nPART ", 
        "\nNote ", 
        "\n\n", 
        "\n", 
        ". ", 
        " "
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        separators=financial_separators,
        chunk_size=1000,   # Larger chunk size to capture full financial context
        chunk_overlap=150, # Overlap prevents cutting a sentence in half
        length_function=len,
    )

    chunks = text_splitter.create_documents([full_text])
    
    print(f"✅ Successfully split the report into {len(chunks)} semantic chunks.")
    return chunks

if __name__ == "__main__":
    download_financial_data()
    document_chunks = parse_and_chunk_report(PDF_PATH)
    
    # Print the first two chunks to verify they maintained their financial context
    print("\n--- CHUNK PREVIEW ---")
    for i in range(2):
        print(f"\n[Chunk {i + 1}] (Length: {len(document_chunks[i].page_content)} chars)")
        print(document_chunks[i].page_content[:300] + "...\n")
        print("-" * 40)