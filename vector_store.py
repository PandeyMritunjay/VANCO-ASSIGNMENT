"""
Vector database operations using ChromaDB for semantic retrieval.
"""

import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

from chunking import Chunk


class VectorStore:
    """
    ChromaDB-based vector store for semantic retrieval.
    Uses sentence-transformers for embeddings.
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "ncert_physics",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = self.embedding_model.encode(texts)
        return embeddings.tolist()
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of Chunk objects to add
        """
        if not chunks:
            return
        
        # Prepare data
        ids = [f"chunk_{chunk.chunk_id}" for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts)
        
        metadatas = []
        for chunk in chunks:
            metadata = {
                'page_number': chunk.page_number,
                'chapter': chunk.chapter,
                'section': chunk.section,
                'chunk_id': chunk.chunk_id,
                'word_count': chunk.metadata.get('word_count', 0),
                'type': chunk.metadata.get('type', 'text')
            }
            metadatas.append(metadata)
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of results with text, metadata, and similarity scores
        """
        query_embedding = self.embed_text(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata
        )
        
        # Format results
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
                })
        
        return formatted_results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific chunk by its ID."""
        results = self.collection.get(ids=[chunk_id])
        
        if results['ids']:
            return {
                'id': results['ids'][0],
                'text': results['documents'][0],
                'metadata': results['metadatas'][0]
            }
        return None
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        count = self.collection.count()
        return {
            'total_chunks': count,
            'collection_name': self.collection_name,
            'embedding_model': self.embedding_model_name,
            'persist_directory': self.persist_directory
        }


def create_vector_store(
    persist_directory: str = "./chroma_db",
    collection_name: str = "ncert_physics",
    embedding_model: str = "all-MiniLM-L6-v2"
) -> VectorStore:
    """Factory function to create a vector store."""
    return VectorStore(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_model=embedding_model
    )
