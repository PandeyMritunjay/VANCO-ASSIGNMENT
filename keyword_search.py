"""
Keyword search using BM25 for exact term matching.
"""

from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
import re
from collections import Counter


class KeywordSearch:
    """
    BM25-based keyword search for exact term matching.
    Complements semantic search with precise term retrieval.
    """
    
    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.tokenized_docs = []
        self.doc_metadata = []
    
    def index_documents(
        self,
        documents: List[str],
        metadata: List[Dict[str, Any]]
    ) -> None:
        """
        Index documents for BM25 search.
        
        Args:
            documents: List of document texts
            metadata: List of metadata dictionaries corresponding to documents
        """
        self.documents = documents
        self.doc_metadata = metadata
        
        # Tokenize documents
        self.tokenized_docs = [
            self._tokenize(doc) for doc in documents
        ]
        
        # Initialize BM25
        self.bm25 = BM25Okapi(self.tokenized_docs)
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        Simple tokenization - can be enhanced with better preprocessing.
        """
        # Convert to lowercase
        text = text.lower()
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # Split into tokens
        tokens = text.split()
        return tokens
    
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for documents using BM25.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of results with text, metadata, and BM25 scores
        """
        if self.bm25 is None:
            return []
        
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        # Format results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return results with positive scores
                results.append({
                    'text': self.documents[idx],
                    'metadata': self.doc_metadata[idx],
                    'score': float(scores[idx]),
                    'rank': len(results) + 1
                })
        
        return results
    
    def search_with_filters(
        self,
        query: str,
        top_k: int = 5,
        chapter_filter: str = None,
        page_filter: int = None
    ) -> List[Dict[str, Any]]:
        """
        Search with optional filters.
        
        Args:
            query: Search query
            top_k: Number of results
            chapter_filter: Filter by chapter name
            page_filter: Filter by page number
            
        Returns:
            Filtered search results
        """
        all_results = self.search(query, top_k=top_k * 2)  # Get more to filter
        
        filtered_results = []
        for result in all_results:
            metadata = result['metadata']
            
            # Apply filters
            if chapter_filter and metadata.get('chapter') != chapter_filter:
                continue
            if page_filter and metadata.get('page_number') != page_filter:
                continue
            
            filtered_results.append(result)
            
            if len(filtered_results) >= top_k:
                break
        
        return filtered_results
    
    def get_document_by_index(self, index: int) -> Dict[str, Any]:
        """Get a document by its index."""
        if 0 <= index < len(self.documents):
            return {
                'text': self.documents[index],
                'metadata': self.doc_metadata[index]
            }
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the keyword search index."""
        return {
            'total_documents': len(self.documents),
            'indexed': self.bm25 is not None
        }


def create_keyword_search() -> KeywordSearch:
    """Factory function to create a keyword search instance."""
    return KeywordSearch()
