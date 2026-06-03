"""
Answer generation with citations using OpenAI.
Ensures answers are grounded in retrieved evidence.
"""

import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class AnswerGenerator:
    """
    Generates answers grounded in retrieved evidence.
    Includes citations and refuses to answer when evidence is insufficient.
    """
    
    def __init__(self, model: str = None):
        self.client = OpenAI(
            base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            api_key=os.getenv("NVIDIA_API_KEY")
        )
        self.model = model or os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")
    
    def generate_answer(
        self,
        query: str,
        retrieved_results: List[Dict[str, Any]],
        max_tokens: int = 500,
        conversation_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate an answer grounded in retrieved evidence.
        
        Args:
            query: User query
            retrieved_results: Retrieved documents with metadata
            max_tokens: Maximum tokens for answer
            conversation_context: Previous conversation history for follow-up questions
            
        Returns:
            Dictionary with answer, citations, and metadata
        """
        # Check if we have sufficient evidence
        if not retrieved_results:
            return {
                'answer': "I cannot answer this question based on the provided NCERT Physics document. The information is not available in the source material.",
                'citations': [],
                'has_evidence': False,
                'evidence_count': 0
            }
        
        # Prepare context from retrieved results
        context = self._prepare_context(retrieved_results)
        
        # Generate answer
        answer = self._generate_with_context(query, context, max_tokens, conversation_context)
        
        # Extract citations
        citations = self._extract_citations(retrieved_results)
        
        return {
            'answer': answer,
            'citations': citations,
            'has_evidence': True,
            'evidence_count': len(retrieved_results),
            'sources_used': self._count_sources(retrieved_results)
        }
    
    def _prepare_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Prepare context from retrieved results.
        
        Args:
            results: Retrieved documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, result in enumerate(results, 1):
            text = result['text']
            metadata = result['metadata']
            
            # Add source information
            page = metadata.get('page_number', 'Unknown')
            chapter = metadata.get('chapter', 'Unknown')
            section = metadata.get('section', 'Unknown')
            
            context_part = f"""
[Source {i}]
Page: {page}
Chapter: {chapter}
Section: {section}
Text: {text}
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _generate_with_context(
        self,
        query: str,
        context: str,
        max_tokens: int,
        conversation_context: Optional[str] = None
    ) -> str:
        """
        Generate answer using OpenAI with context.
        
        Args:
            query: User query
            context: Retrieved context
            max_tokens: Maximum tokens
            conversation_context: Previous conversation history for follow-up questions
            
        Returns:
            Generated answer
        """
        system_prompt = """You are a helpful assistant that answers questions based ONLY on the provided NCERT Physics document context.

Rules:
1. Answer the question using ONLY the information in the provided context.
2. If the answer is not in the context, say "This information is not available in the provided document."
3. Be concise and direct.
4. Include page numbers in your answer when referencing information.
5. Do not use outside knowledge or make assumptions beyond the context.
6. For formulas, write them clearly using standard notation.
7. For comparisons, present the information in a structured way.
8. If the user asks a follow-up question (like "tell more about it", "explain further", etc.), refer to the previous conversation context to understand what they're asking about."""

        # Build user prompt with optional conversation context
        if conversation_context:
            user_prompt = f"""Previous conversation:
{conversation_context}

Context from NCERT Physics document:
{context}

Current question: {query}

Answer:"""
        else:
            user_prompt = f"""Context from NCERT Physics document:
{context}

Question: {query}

Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.2,
                top_p=0.7,
                stream=False
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            error_detail = str(e)
            if "404" in error_detail:
                return f"Error: NVIDIA API model '{self.model}' not found. Please check if the model name is correct."
            elif "401" in error_detail or "403" in error_detail:
                return f"Error: NVIDIA API authentication failed. Please check your API key."
            else:
                return f"Error generating answer: {error_detail}"
    
    def _extract_citations(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract citation information from results.
        
        Args:
            results: Retrieved documents
            
        Returns:
            List of citation dictionaries
        """
        citations = []
        seen_pages = set()
        
        for result in results:
            metadata = result['metadata']
            page = metadata.get('page_number')
            chapter = metadata.get('chapter')
            section = metadata.get('section')
            
            # Avoid duplicate citations from the same page
            if page not in seen_pages:
                citations.append({
                    'page_number': page,
                    'chapter': chapter,
                    'section': section,
                    'sources': result.get('sources', [])
                })
                seen_pages.add(page)
        
        return citations
    
    def _count_sources(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count how many results came from each source type."""
        source_counts = {'vector': 0, 'keyword': 0, 'graph': 0}
        
        for result in results:
            for source in result.get('sources', []):
                if source in source_counts:
                    source_counts[source] += 1
        
        return source_counts
    
    def generate_with_explanation(
        self,
        query: str,
        retrieved_results: List[Dict[str, Any]],
        max_tokens: int = 500
    ) -> Dict[str, Any]:
        """
        Generate answer with retrieval explanation.
        
        Args:
            query: User query
            retrieved_results: Retrieved documents
            max_tokens: Maximum tokens
            
        Returns:
            Answer with retrieval path explanation
        """
        # Generate standard answer
        answer_data = self.generate_answer(query, retrieved_results, max_tokens)
        
        # Add retrieval explanation
        answer_data['retrieval_explanation'] = self._explain_retrieval(retrieved_results)
        
        return answer_data
    
    def _explain_retrieval(self, results: List[Dict[str, Any]]) -> str:
        """Generate explanation of the retrieval process."""
        if not results:
            return "No relevant information was found in the document."
        
        explanation_parts = []
        
        # Count sources
        source_counts = self._count_sources(results)
        explanation_parts.append(f"Retrieved {len(results)} relevant chunks using hybrid search:")
        
        if source_counts['vector'] > 0:
            explanation_parts.append(f"- {source_counts['vector']} from semantic/vector search")
        if source_counts['keyword'] > 0:
            explanation_parts.append(f"- {source_counts['keyword']} from keyword/BM25 search")
        if source_counts['graph'] > 0:
            explanation_parts.append(f"- {source_counts['graph']} from knowledge graph")
        
        # Add page range
        pages = [r['metadata'].get('page_number') for r in results if r['metadata'].get('page_number')]
        if pages:
            explanation_parts.append(f"\nInformation found across pages: {min(pages)}-{max(pages)}")
        
        return "\n".join(explanation_parts)


def create_answer_generator(model: str = None) -> AnswerGenerator:
    """Factory function to create an answer generator."""
    return AnswerGenerator(model=model)
