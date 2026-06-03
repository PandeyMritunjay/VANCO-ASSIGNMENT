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
