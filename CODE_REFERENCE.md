# Code Reference - All Source Files

This document contains the complete source code for all files in the RAG application.

---

## chunking.py

```python
"""
Section-aware chunking strategy for NCERT Physics PDF.
Preserves document structure, page numbers, and context.
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    page_number: int
    chapter: str
    section: str
    chunk_id: int
    metadata: Dict[str, Any]


class SectionAwareChunker:
    """
    Implements section-aware chunking that:
    - Respects chapter and section boundaries
    - Preserves page numbers
    - Maintains context windows
    - Handles formulas and tables
    """
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Patterns for detecting structure
        self.chapter_pattern = re.compile(
            r'^(CHAPTER|Chapter)\s+\d+[:\.\s]+(.+)$',
            re.MULTILINE
        )
        self.section_pattern = re.compile(
            r'^(\d+\.\d+)\s+(.+)$',
            re.MULTILINE
        )
        self.formula_pattern = re.compile(
            r'\$[^$]+\$|\\\[.*?\\\]|\\\(.*?\\\)',
            re.DOTALL
        )
    
    def extract_structure(self, text: str) -> Dict[str, Any]:
        """Extract chapter and section information from text."""
        chapters = []
        sections = []
        
        lines = text.split('\n')
        current_chapter = "Unknown"
        
        for line in lines:
            chapter_match = self.chapter_pattern.match(line.strip())
            if chapter_match:
                current_chapter = chapter_match.group(2).strip()
                chapters.append(current_chapter)
            
            section_match = self.section_pattern.match(line.strip())
            if section_match:
                sections.append({
                    'number': section_match.group(1),
                    'title': section_match.group(2).strip(),
                    'chapter': current_chapter
                })
        
        return {
            'chapters': chapters,
            'sections': sections
        }
    
    def chunk_text(
        self,
        text: str,
        page_number: int,
        chapter: str = "Unknown",
        section: str = "Unknown"
    ) -> List[Chunk]:
        """
        Chunk text while preserving structure and context.
        
        Args:
            text: The text to chunk
            page_number: Source page number
            chapter: Current chapter name
            section: Current section name
            
        Returns:
            List of Chunk objects
        """
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Split into paragraphs
        paragraphs = self._split_into_paragraphs(text)
        
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for para in paragraphs:
            # If paragraph is too large, split it
            if len(para) > self.chunk_size * 2:
                para_chunks = self._split_large_paragraph(para)
                for para_chunk in para_chunks:
                    if len(current_chunk) + len(para_chunk) > self.chunk_size:
                        if len(current_chunk) >= self.min_chunk_size:
                            chunks.append(Chunk(
                                text=current_chunk.strip(),
                                page_number=page_number,
                                chapter=chapter,
                                section=section,
                                chunk_id=chunk_id,
                                metadata={
                                    'type': 'text',
                                    'word_count': len(current_chunk.split())
                                }
                            ))
                            chunk_id += 1
                        current_chunk = para_chunk
                    else:
                        current_chunk += " " + para_chunk
            else:
                if len(current_chunk) + len(para) > self.chunk_size:
                    if len(current_chunk) >= self.min_chunk_size:
                        chunks.append(Chunk(
                            text=current_chunk.strip(),
                            page_number=page_number,
                            chapter=chapter,
                            section=section,
                            chunk_id=chunk_id,
                            metadata={
                                'type': 'text',
                                'word_count': len(current_chunk.split())
                            }
                        ))
                        chunk_id += 1
                    current_chunk = para
                else:
                    current_chunk += " " + para
        
        # Add remaining chunk
        if len(current_chunk) >= self.min_chunk_size:
            chunks.append(Chunk(
                text=current_chunk.strip(),
                page_number=page_number,
                chapter=chapter,
                section=section,
                chunk_id=chunk_id,
                metadata={
                    'type': 'text',
                    'word_count': len(current_chunk.split())
                }
            ))
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers from text
        text = re.sub(r'\d+\s*$', '', text)
        return text.strip()
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_large_paragraph(self, text: str) -> List[str]:
        """Split a large paragraph into smaller chunks."""
        chunks = []
        sentences = re.split(r'[.!?]+\s+', text)
        
        current = ""
        for sent in sentences:
            if len(current) + len(sent) > self.chunk_size:
                if current:
                    chunks.append(current)
                current = sent
            else:
                current += " " + sent if current else sent
        
        if current:
            chunks.append(current)
        
        return chunks
    
    def chunk_document(
        self,
        pages: List[Dict[str, Any]]
    ) -> List[Chunk]:
        """
        Chunk an entire document with page-level structure.
        
        Args:
            pages: List of page dictionaries with 'text' and 'page_number'
            
        Returns:
            List of all chunks from the document
        """
        all_chunks = []
        current_chapter = "Unknown"
        current_section = "Unknown"
        global_chunk_id = 0  # Global counter for unique chunk IDs
        
        for page in pages:
            text = page['text']
            page_number = page['page_number']
            
            # Extract structure from this page
            structure = self.extract_structure(text)
            
            # Update current chapter/section if found
            if structure['chapters']:
                current_chapter = structure['chapters'][-1]
            if structure['sections']:
                current_section = f"{structure['sections'][-1]['number']} {structure['sections'][-1]['title']}"
            
            # Chunk the page
            chunks = self.chunk_text(
                text=text,
                page_number=page_number,
                chapter=current_chapter,
                section=current_section
            )
            
            # Update chunk IDs to be globally unique
            for chunk in chunks:
                chunk.chunk_id = global_chunk_id
                global_chunk_id += 1
            
            all_chunks.extend(chunks)
        
        return all_chunks


def create_chunker(
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    min_chunk_size: int = 100
) -> SectionAwareChunker:
    """Factory function to create a chunker with specified parameters."""
    return SectionAwareChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size
    )
```

---

## generator.py

```python
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
```

---

## graph_store.py

```python
"""
Graph database operations using Neo4j for knowledge graph.
Stores concepts, formulas, definitions, and relationships.
"""

from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
import re


class GraphStore:
    """
    Neo4j-based knowledge graph for NCERT Physics.
    Captures relationships between concepts, formulas, and definitions.
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = ""
    ):
        self.uri = uri
        self.username = username
        self.password = password
        
        self.driver = GraphDatabase.driver(
            uri,
            auth=(username, password)
        )
        
        # Initialize the graph schema
        self._initialize_schema()
    
    def close(self):
        """Close the database connection."""
        self.driver.close()
    
    def _initialize_schema(self):
        """Create indexes and constraints for efficient querying."""
        with self.driver.session() as session:
            # Create uniqueness constraints
            session.run("""
                CREATE CONSTRAINT concept_name_unique IF NOT EXISTS
                FOR (c:Concept) REQUIRE c.name IS UNIQUE
            """)
            session.run("""
                CREATE CONSTRAINT formula_id_unique IF NOT EXISTS
                FOR (f:Formula) REQUIRE f.id IS UNIQUE
            """)
            session.run("""
                CREATE CONSTRAINT definition_id_unique IF NOT EXISTS
                FOR (d:Definition) REQUIRE d.id IS UNIQUE
            """)
            
            # Create indexes
            session.run("""
                CREATE INDEX chapter_name_index IF NOT EXISTS
                FOR (ch:Chapter) ON (ch.name)
            """)
            session.run("""
                CREATE INDEX section_title_index IF NOT EXISTS
                FOR (s:Section) ON (s.title)
            """)
    
    def add_chapter(self, name: str, page_number: int) -> str:
        """Add a chapter node."""
        with self.driver.session() as session:
            result = session.run("""
                MERGE (ch:Chapter {name: $name})
                SET ch.page_number = $page_number
                RETURN ch.name as name
            """, name=name, page_number=page_number)
            return result.single()["name"]
    
    def add_section(
        self,
        number: str,
        title: str,
        chapter_name: str,
        page_number: int
    ) -> str:
        """Add a section node and link to chapter."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (ch:Chapter {name: $chapter_name})
                MERGE (s:Section {number: $number, title: $title})
                SET s.page_number = $page_number
                MERGE (ch)-[:HAS_SECTION]->(s)
                RETURN s.number as number, s.title as title
            """, number=number, title=title, chapter_name=chapter_name, page_number=page_number)
            return f"{result.single()['number']} {result.single()['title']}"
    
    def add_concept(
        self,
        name: str,
        description: str,
        chapter_name: str,
        section_number: str,
        page_number: int
    ) -> str:
        """Add a concept node and link to section."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (s:Section {number: $section_number})
                MERGE (c:Concept {name: $name})
                SET c.description = $description,
                    c.page_number = $page_number,
                    c.chapter = $chapter_name
                MERGE (s)-[:CONTAINS_CONCEPT]->(c)
                RETURN c.name as name
            """, name=name, description=description, section_number=section_number,
                page_number=page_number, chapter_name=chapter_name)
            return result.single()["name"]
    
    def add_formula(
        self,
        formula_id: str,
        expression: str,
        description: str,
        concept_name: str,
        page_number: int
    ) -> str:
        """Add a formula node and link to concept."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Concept {name: $concept_name})
                MERGE (f:Formula {id: $formula_id})
                SET f.expression = $expression,
                    f.description = $description,
                    f.page_number = $page_number
                MERGE (c)-[:HAS_FORMULA]->(f)
                RETURN f.id as id
            """, formula_id=formula_id, expression=expression, description=description,
                concept_name=concept_name, page_number=page_number)
            return result.single()["id"]
    
    def add_definition(
        self,
        definition_id: str,
        term: str,
        definition: str,
        concept_name: str,
        page_number: int
    ) -> str:
        """Add a definition node and link to concept."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Concept {name: $concept_name})
                MERGE (d:Definition {id: $definition_id})
                SET d.term = $term,
                    d.definition = $definition,
                    d.page_number = $page_number
                MERGE (c)-[:HAS_DEFINITION]->(d)
                RETURN d.id as id
            """, definition_id=definition_id, term=term, definition=definition,
                concept_name=concept_name, page_number=page_number)
            return result.single()["id"]
    
    def add_relationship(
        self,
        concept1: str,
        relationship_type: str,
        concept2: str
    ) -> None:
        """Add a relationship between two concepts."""
        with self.driver.session() as session:
            session.run("""
                MATCH (c1:Concept {name: $concept1})
                MATCH (c2:Concept {name: $concept2})
                MERGE (c1)-[r:RELATED {type: $relationship_type}]->(c2)
            """, concept1=concept1, concept2=concept2, relationship_type=relationship_type)
    
    def search_concepts(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for concepts by name or description.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching concepts
        """
        with self.driver.session() as session:
            results = session.run("""
                MATCH (c:Concept)
                WHERE c.name CONTAINS $query OR c.description CONTAINS $query
                RETURN c.name as name, c.description as description, 
                       c.page_number as page_number, c.chapter as chapter
                LIMIT $limit
            """, query=query, limit=limit)
            
            concepts = []
            for record in results:
                concepts.append({
                    'name': record["name"],
                    'description': record["description"],
                    'page_number': record["page_number"],
                    'chapter': record["chapter"]
                })
            return concepts
    
    def get_related_concepts(self, concept_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get concepts related to a given concept.
        
        Args:
            concept_name: Name of the concept
            limit: Maximum number of results
            
        Returns:
            List of related concepts
        """
        with self.driver.session() as session:
            results = session.run("""
                MATCH (c:Concept {name: $concept_name})-[r:RELATED]->(related:Concept)
                RETURN related.name as name, related.description as description,
                       r.type as relationship_type, related.page_number as page_number
                LIMIT $limit
            """, concept_name=concept_name, limit=limit)
            
            concepts = []
            for record in results:
                concepts.append({
                    'name': record["name"],
                    'description': record["description"],
                    'relationship_type': record["relationship_type"],
                    'page_number': record["page_number"]
                })
            return concepts
    
    def get_formulas_for_concept(self, concept_name: str) -> List[Dict[str, Any]]:
        """Get all formulas related to a concept."""
        with self.driver.session() as session:
            results = session.run("""
                MATCH (c:Concept {name: $concept_name})-[:HAS_FORMULA]->(f:Formula)
                RETURN f.expression as expression, f.description as description,
                       f.page_number as page_number
            """, concept_name=concept_name)
            
            formulas = []
            for record in results:
                formulas.append({
                    'expression': record["expression"],
                    'description': record["description"],
                    'page_number': record["page_number"]
                })
            return formulas
    
    def get_definitions_for_concept(self, concept_name: str) -> List[Dict[str, Any]]:
        """Get all definitions related to a concept."""
        with self.driver.session() as session:
            results = session.run("""
                MATCH (c:Concept {name: $concept_name})-[:HAS_DEFINITION]->(d:Definition)
                RETURN d.term as term, d.definition as definition,
                       d.page_number as page_number
            """, concept_name=concept_name)
            
            definitions = []
            for record in results:
                definitions.append({
                    'term': record["term"],
                    'definition': record["definition"],
                    'page_number': record["page_number"]
                })
            return definitions
    
    def extract_and_add_entities_from_text(
        self,
        text: str,
        chapter: str,
        section: str,
        page_number: int
    ) -> None:
        """
        Extract entities from text and add to graph.
        This is a simple extraction - can be enhanced with NLP.
        """
        # Simple pattern-based extraction
        # Extract formulas (text between $ or LaTeX patterns)
        formula_pattern = re.compile(r'\$([^$]+)\$|\\\[([^\\]+)\\\]')
        formulas = formula_pattern.findall(text)
        
        # Extract definitions (patterns like "X is defined as" or "X means")
        definition_pattern = re.compile(
            r'(\w+(?:\s+\w+)*)\s+(?:is defined as|means|refers to)\s+([^.,]+)',
            re.IGNORECASE
        )
        definitions = definition_pattern.findall(text)
        
        # Extract key terms (capitalized words that might be concepts)
        concept_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b')
        concepts = concept_pattern.findall(text)
        
        # Add to graph (simplified - in production, use NER)
        section_number = section.split()[0] if section else "0"
        
        for i, formula in enumerate(formulas):
            formula_text = formula[0] if formula[0] else formula[1]
            self.add_formula(
                formula_id=f"formula_{page_number}_{i}",
                expression=formula_text,
                description="Formula",
                concept_name=chapter,
                page_number=page_number
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the graph."""
        with self.driver.session() as session:
            stats = {}
            
            # Count nodes by type
            result = session.run("MATCH (c:Chapter) RETURN count(c) as count")
            stats['chapters'] = result.single()["count"]
            
            result = session.run("MATCH (s:Section) RETURN count(s) as count")
            stats['sections'] = result.single()["count"]
            
            result = session.run("MATCH (c:Concept) RETURN count(c) as count")
            stats['concepts'] = result.single()["count"]
            
            result = session.run("MATCH (f:Formula) RETURN count(f) as count")
            stats['formulas'] = result.single()["count"]
            
            result = session.run("MATCH (d:Definition) RETURN count(d) as count")
            stats['definitions'] = result.single()["count"]
            
            # Count relationships
            result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            stats['relationships'] = result.single()["count"]
            
            return stats


def create_graph_store(
    uri: str = "bolt://localhost:7687",
    username: str = "neo4j",
    password: str = ""
) -> GraphStore:
    """Factory function to create a graph store."""
    return GraphStore(uri=uri, username=username, password=password)
```

---

## ingest.py

```python
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
```

---

## keyword_search.py

```python
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
```

---

## retrieval.py

```python
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
```

---

## streamlit_app.py

```python
import streamlit as st
import os
import importlib
import sys
from dotenv import load_dotenv

# Force reload modules to pick up changes
modules_to_reload = ['generator', 'retrieval', 'vector_store', 'keyword_search', 'graph_store']
for module in modules_to_reload:
    if module in sys.modules:
        importlib.reload(sys.modules[module])

from vector_store import create_vector_store
from keyword_search import create_keyword_search
from retrieval import create_hybrid_retriever
from generator import create_answer_generator
from graph_store import create_graph_store

# Get script directory for relative paths (must be before load_dotenv)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load environment variables from script directory
env_path = os.path.join(SCRIPT_DIR, '.env')
load_dotenv(env_path)

# Page configuration
st.set_page_config(
    page_title="NCERT Physics RAG",
    page_icon="📚",
    layout="centered"
)

# Initialize session state
if 'vector_store' not in st.session_state:
    with st.spinner("Loading vector store..."):
        chroma_path = os.path.join(SCRIPT_DIR, os.getenv("CHROMA_PERSIST_DIRECTORY", "chroma_db").lstrip("./"))
        st.session_state.vector_store = create_vector_store(
            persist_directory=chroma_path
        )
        # Debug: check if chroma_db loaded correctly
        stats = st.session_state.vector_store.get_stats()
        st.sidebar.write(f"Vector Store: {stats['total_chunks']} chunks")
        st.sidebar.write(f"Path: {chroma_path}")

if 'keyword_search' not in st.session_state:
    with st.spinner("Loading keyword search..."):
        st.session_state.keyword_search = create_keyword_search()
        # Index documents from vector store for keyword search
        if 'vector_store' in st.session_state:
            vector_store = st.session_state.vector_store
            collection = vector_store.collection
            count = collection.count()
            st.sidebar.write(f"ChromaDB count: {count}")
            if count > 0:
                # Get all documents from chroma_db
                all_data = collection.get()
                documents = all_data['documents']
                metadatas = all_data['metadatas']
                st.sidebar.write(f"Documents loaded: {len(documents)}")
                # Index in keyword search
                st.session_state.keyword_search.index_documents(documents, metadatas)
                kw_stats = st.session_state.keyword_search.get_stats()
                st.sidebar.write(f"Keyword Search: {kw_stats['total_documents']} docs, indexed: {kw_stats['indexed']}")

if 'graph_store' not in st.session_state:
    try:
        with st.spinner("Connecting to Neo4j..."):
            st.session_state.graph_store = create_graph_store(
                uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                username=os.getenv("NEO4J_USERNAME", "neo4j"),
                password=os.getenv("NEO4J_PASSWORD", "")
            )
    except Exception as e:
        st.session_state.graph_store = None
        st.warning(f"Neo4j not available: {e}. System will run without graph search.")

if 'retriever' not in st.session_state:
    with st.spinner("Initializing hybrid retriever..."):
        st.session_state.retriever = create_hybrid_retriever(
            vector_store=st.session_state.vector_store,
            graph_store=st.session_state.graph_store,
            keyword_search=st.session_state.keyword_search
        )

if 'generator' not in st.session_state:
    with st.spinner("Initializing answer generator..."):
        st.session_state.generator = create_answer_generator()

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Default settings
top_k = 5
use_graph = True
show_retrieval = False

# Header
st.title("NCERT Physics RAG")
st.caption("Ask questions about Class 12 Physics (Part 1)")

# Clear chat button
if st.button("Clear Chat", type="secondary"):
    st.session_state.messages = []
    st.rerun()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.write(message["content"])
        else:
            # Display assistant message with answer and citations
            st.markdown(message["content"]["answer"])
            
            if message["content"]["citations"]:
                with st.expander("Citations"):
                    for citation in message["content"]["citations"]:
                        st.markdown(f"**Page {citation['page_number']}** - {citation['chapter']}")
                        st.caption(f"*{citation['section']}*")

# Chat input
if prompt := st.chat_input("Ask a question about NCERT Physics..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving information and generating answer..."):
            # Build conversation context for retrieval
            conversation_context = ""
            if len(st.session_state.messages) > 1:
                # Get last few exchanges (excluding current message)
                recent_messages = st.session_state.messages[-6:-1]  # Last 3 exchanges
                for msg in recent_messages:
                    if msg["role"] == "user":
                        conversation_context += f"User: {msg['content']}\n"
                    elif msg["role"] == "assistant":
                        conversation_context += f"Assistant: {msg['content']['answer']}\n"
            
            # Build enhanced query with context
            if conversation_context:
                enhanced_query = f"Conversation context:\n{conversation_context}\n\nCurrent question: {prompt}"
            else:
                enhanced_query = prompt
            
            # Retrieve
            results = st.session_state.retriever.retrieve(
                query=enhanced_query,
                top_k=top_k,
                use_vector=True,
                use_keyword=True,
                use_graph=use_graph
            )
            # Debug: log retrieval results
            st.sidebar.write(f"Retrieval results: {len(results.get('combined_results', []))} items")
            if results.get('combined_results'):
                st.sidebar.write(f"First result text preview: {results['combined_results'][0].get('text', '')[:100]}...")
            
            # Generate answer with conversation context
            try:
                response = st.session_state.generator.generate_answer(
                    query=prompt,
                    retrieved_results=results['combined_results'],
                    max_tokens=500,
                    conversation_context=conversation_context if conversation_context else None
                )
            except TypeError:
                # Fallback for old generator without conversation_context support
                response = st.session_state.generator.generate_answer(
                    query=prompt,
                    retrieved_results=results['combined_results'],
                    max_tokens=500
                )
            
            # Display answer
            st.markdown(response['answer'])
            
            # Display citations
            if response['citations']:
                with st.expander("Citations"):
                    for citation in response['citations']:
                        st.markdown(f"**Page {citation['page_number']}** - {citation['chapter']}")
                        st.caption(f"*{citation['section']}*")
            
            # Add assistant response to chat history
            assistant_message = {
                "role": "assistant",
                "content": {
                    "answer": response['answer'],
                    "citations": response['citations'],
                    "retrieval_info": None
                }
            }
            st.session_state.messages.append(assistant_message)
```

---

## vector_store.py

```python
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

# Set Hugging Face cache to local directory
HF_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache")
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = HF_CACHE_DIR
os.environ['HF_HUB_CACHE'] = HF_CACHE_DIR


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
        
        # Initialize embedding model with local cache and error handling
        cache_dir = os.path.join(os.path.dirname(persist_directory), "model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        try:
            self.embedding_model = SentenceTransformer(
                embedding_model,
                cache_folder=cache_dir
            )
        except Exception as e:
            # If download fails, try again with a different approach
            print(f"Error loading model: {e}. Retrying...")
            import time
            time.sleep(2)
            self.embedding_model = SentenceTransformer(
                embedding_model,
                cache_folder=cache_dir
            )
        
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
```

---

## download_model.py

```python
"""
Script to pre-download the SentenceTransformer model to avoid download issues during app startup.
Run this once before running the Streamlit app.
"""

import os
from sentence_transformers import SentenceTransformer

# Set cache directory
HF_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache")
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = HF_CACHE_DIR
os.environ['HF_HUB_CACHE'] = HF_CACHE_DIR

print("Downloading SentenceTransformer model...")
print(f"Cache directory: {HF_CACHE_DIR}")

try:
    model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=HF_CACHE_DIR)
    print("Model downloaded successfully!")
    print(f"Model saved to: {HF_CACHE_DIR}")
except Exception as e:
    print(f"Error downloading model: {e}")
    print("Please check your internet connection and try again.")
```

---

## Configuration Files

### requirements.txt

```
# Core dependencies
langchain
langchain-community
langchain-openai
openai

# PDF processing
pypdf
pdfplumber

# Vector database
chromadb
sentence-transformers
torchvision

# Graph database
neo4j

# Keyword search
rank-bm25

# Web framework
fastapi
uvicorn
pydantic
python-multipart

# Frontend
jinja2
aiofiles
streamlit

# Utilities
python-dotenv
tiktoken
```

### runtime.txt

```
python3.11
```

### .env.example

```
# NVIDIA API Configuration
NVIDIA_API_KEY=your_nvidia_api_key_here
NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
NVIDIA_MODEL=meta/llama-3.1-8b-instruct

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY=./chroma_db

# Neo4j Configuration (Optional)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password

# PDF Configuration
PDF_URL=https://www.drishtiias.com/images/pdf/NCERT-Class-12-Physics-Part-1.pdf
PDF_PATH=./data/NCERT-Class-12-Physics-Part-1.pdf
```

---

**Total Files: 9 Python files + 3 configuration files**
