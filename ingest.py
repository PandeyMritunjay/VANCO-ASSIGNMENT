"""
Ingest NCERT Physics PDF into ChromaDB for RAG.
Run this script to populate the vector store with document chunks.
"""

import os
from dotenv import load_dotenv

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables from script directory
env_path = os.path.join(SCRIPT_DIR, '.env')
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    # Try loading from parent directory
    parent_env = os.path.join(os.path.dirname(SCRIPT_DIR), '.env')
    if os.path.exists(parent_env):
        load_dotenv(parent_env)

from chunking import Chunk, SectionAwareChunker
from vector_store import create_vector_store
from keyword_search import create_keyword_search

# Set Hugging Face cache to local directory
HF_CACHE_DIR = os.path.join(SCRIPT_DIR, "hf_cache")
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = HF_CACHE_DIR
os.environ['HF_HUB_CACHE'] = HF_CACHE_DIR

def main():
    print("Starting ingestion of NCERT Physics PDF...")
    
    # PDF URL and path
    pdf_url = os.getenv("PDF_URL", "https://www.drishtiias.com/images/pdf/NCERT-Class-12-Physics-Part-1.pdf")
    pdf_path = os.getenv("PDF_PATH", "./data/NCERT-Class-12-Physics-Part-1.pdf")
    
    # Use absolute path for PDF
    if not os.path.isabs(pdf_path):
        pdf_path = os.path.join(SCRIPT_DIR, pdf_path)
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    
    # Download PDF if not exists
    if not os.path.exists(pdf_path):
        print(f"Downloading PDF from {pdf_url}...")
        import requests
        response = requests.get(pdf_url)
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        print(f"PDF downloaded to {pdf_path}")
    
    # Chunk the PDF
    print("Chunking PDF...")
    chunker = SectionAwareChunker(
        chunk_size=500,
        chunk_overlap=50
    )
    
    # Extract text from PDF
    import pypdf
    pages = []
    with open(pdf_path, 'rb') as f:
        pdf_reader = pypdf.PdfReader(f)
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            pages.append({
                'text': text,
                'page_number': page_num + 1
            })
    
    chunks = chunker.chunk_document(pages)
    print(f"Created {len(chunks)} chunks")
    
    # Create vector store
    chroma_path = os.path.join(SCRIPT_DIR, "chroma_db")
    print(f"Creating vector store at {chroma_path}...")
    vector_store = create_vector_store(
        persist_directory=chroma_path,
        collection_name="ncert_physics"
    )
    
    # Add chunks to vector store
    print("Adding chunks to vector store...")
    vector_store.add_chunks(chunks)
    print(f"Added {len(chunks)} chunks to vector store")
    
    # Verify
    stats = vector_store.get_stats()
    print(f"Vector store stats: {stats}")
    
    # Index for keyword search
    print("Indexing for keyword search...")
    keyword_search = create_keyword_search()
    documents = [chunk.text for chunk in chunks]
    metadatas = [
        {
            'page_number': chunk.page_number,
            'chapter': chunk.chapter,
            'section': chunk.section,
            'chunk_id': chunk.chunk_id,
            'word_count': chunk.metadata.get('word_count', 0),
            'type': chunk.metadata.get('type', 'text')
        }
        for chunk in chunks
    ]
    keyword_search.index_documents(documents, metadatas)
    kw_stats = keyword_search.get_stats()
    print(f"Keyword search stats: {kw_stats}")
    
    print("Ingestion complete!")

if __name__ == "__main__":
    main()
