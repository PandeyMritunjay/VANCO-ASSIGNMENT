"""
Hybrid retrieval combining vector search, graph search, and keyword search.
Implements reranking and fusion of results from multiple sources.
"""

from typing import List, Dict, Any, Optional
from vector_store import VectorStore
from graph_store import GraphStore
from keyword_search import KeywordSearch


class HybridRetriever:
    """
    Hybrid retrieval system that combines:
    - Vector database (semantic search)
    - Graph database (knowledge graph traversal)
    - Keyword search (BM25)
    
    Uses reciprocal rank fusion (RRF) for result ranking.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        graph_store: GraphStore,
        keyword_search: KeywordSearch,
        vector_weight: float = 0.5,
        keyword_weight: float = 0.3,
        graph_weight: float = 0.2
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.keyword_search = keyword_search
        
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.graph_weight = graph_weight
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        use_vector: bool = True,
        use_keyword: bool = True,
        use_graph: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve relevant documents using hybrid search.
        
        Args:
            query: User query
            top_k: Number of results to return
            use_vector: Whether to use vector search
            use_keyword: Whether to use keyword search
            use_graph: Whether to use graph search
            
        Returns:
            Dictionary with combined results and retrieval metadata
        """
        all_results = {
            'vector_results': [],
            'keyword_results': [],
            'graph_results': [],
            'combined_results': [],
            'retrieval_metadata': {}
        }
        
        # Vector search
        if use_vector:
            vector_results = self.vector_store.search(query, n_results=top_k)
            all_results['vector_results'] = vector_results
            all_results['retrieval_metadata']['vector_count'] = len(vector_results)
        
        # Keyword search
        if use_keyword:
            keyword_results = self.keyword_search.search(query, top_k=top_k)
            all_results['keyword_results'] = keyword_results
            all_results['retrieval_metadata']['keyword_count'] = len(keyword_results)
        
        # Graph search (only if graph_store is available)
        if use_graph and self.graph_store is not None:
            graph_results = self._graph_search(query, top_k=top_k)
            all_results['graph_results'] = graph_results
            all_results['retrieval_metadata']['graph_count'] = len(graph_results)
        else:
            all_results['retrieval_metadata']['graph_count'] = 0
        
        # Combine results using reciprocal rank fusion
        combined = self._reciprocal_rank_fusion(
            all_results['vector_results'],
            all_results['keyword_results'],
            all_results['graph_results'],
            top_k=top_k
        )
        
        all_results['combined_results'] = combined
        all_results['retrieval_metadata']['total_combined'] = len(combined)
        
        return all_results
    
    def _graph_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the knowledge graph.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of graph-based results
        """
        results = []
        
        # Search for concepts
        concepts = self.graph_store.search_concepts(query, limit=top_k)
        
        for concept in concepts:
            # Get related concepts
            related = self.graph_store.get_related_concepts(
                concept['name'],
                limit=3
            )
            
            # Get formulas
            formulas = self.graph_store.get_formulas_for_concept(concept['name'])
            
            # Get definitions
            definitions = self.graph_store.get_definitions_for_concept(concept['name'])
            
            results.append({
                'type': 'concept',
                'name': concept['name'],
                'description': concept['description'],
                'page_number': concept['page_number'],
                'chapter': concept['chapter'],
                'related_concepts': related,
                'formulas': formulas,
                'definitions': definitions,
                'source': 'graph'
            })
        
        return results
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
        top_k: int = 10,
        k: int = 60  # RRF constant
    ) -> List[Dict[str, Any]]:
        """
        Combine results using reciprocal rank fusion.
        
        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            graph_results: Results from graph search
            top_k: Number of final results
            k: RRF constant (higher = more uniform ranking)
            
        Returns:
            Combined and ranked results
        """
        # Score dictionary: key = document identifier, value = RRF score
        scores = {}
        result_info = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results):
            doc_id = result.get('id', f"vector_{rank}")
            scores[doc_id] = scores.get(doc_id, 0) + (self.vector_weight / (k + rank + 1))
            result_info[doc_id] = {
                'text': result['text'],
                'metadata': result['metadata'],
                'sources': ['vector'],
                'vector_rank': rank + 1,
                'vector_score': result.get('similarity', 0)
            }
        
        # Process keyword results
        for rank, result in enumerate(keyword_results):
            # Create a simple ID based on text
            doc_id = f"keyword_{hash(result['text']) % 10000}"
            scores[doc_id] = scores.get(doc_id, 0) + (self.keyword_weight / (k + rank + 1))
            
            if doc_id in result_info:
                result_info[doc_id]['sources'].append('keyword')
                result_info[doc_id]['keyword_rank'] = rank + 1
                result_info[doc_id]['keyword_score'] = result.get('score', 0)
            else:
                result_info[doc_id] = {
                    'text': result['text'],
                    'metadata': result['metadata'],
                    'sources': ['keyword'],
                    'keyword_rank': rank + 1,
                    'keyword_score': result.get('score', 0)
                }
        
        # Process graph results
        for rank, result in enumerate(graph_results):
            doc_id = f"graph_{hash(result['name']) % 10000}"
            scores[doc_id] = scores.get(doc_id, 0) + (self.graph_weight / (k + rank + 1))
            
            # Convert graph result to text format
            text = f"Concept: {result['name']}\nDescription: {result['description']}"
            if result['formulas']:
                text += f"\nFormulas: " + ", ".join([f['expression'] for f in result['formulas']])
            
            metadata = {
                'page_number': result['page_number'],
                'chapter': result['chapter'],
                'type': 'concept'
            }
            
            if doc_id in result_info:
                result_info[doc_id]['sources'].append('graph')
                result_info[doc_id]['graph_rank'] = rank + 1
            else:
                result_info[doc_id] = {
                    'text': text,
                    'metadata': metadata,
                    'sources': ['graph'],
                    'graph_rank': rank + 1
                }
        
        # Sort by RRF score
        sorted_results = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Format final results
        final_results = []
        for doc_id, score in sorted_results[:top_k]:
            info = result_info[doc_id]
            info['rrf_score'] = score
            final_results.append(info)
        
        return final_results
    
    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 5,
        context_window: int = 2
    ) -> Dict[str, Any]:
        """
        Retrieve results with surrounding context.
        
        Args:
            query: User query
            top_k: Number of results
            context_window: Number of surrounding chunks to include
            
        Returns:
            Results with expanded context
        """
        results = self.retrieve(query, top_k=top_k)
        
        # Expand context for each result
        for result in results['combined_results']:
            page_number = result['metadata'].get('page_number')
            if page_number:
                # Get additional chunks from the same page
                page_results = self.vector_store.search(
                    query,
                    n_results=context_window * 2,
                    filter_metadata={'page_number': page_number}
                )
                result['context_chunks'] = page_results
        
        return results


def create_hybrid_retriever(
    vector_store: VectorStore,
    graph_store: GraphStore,
    keyword_search: KeywordSearch,
    vector_weight: float = 0.5,
    keyword_weight: float = 0.3,
    graph_weight: float = 0.2
) -> HybridRetriever:
    """Factory function to create a hybrid retriever."""
    return HybridRetriever(
        vector_store=vector_store,
        graph_store=graph_store,
        keyword_search=keyword_search,
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
        graph_weight=graph_weight
    )
