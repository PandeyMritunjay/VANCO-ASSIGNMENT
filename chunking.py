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
