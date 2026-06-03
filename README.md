# RAG Application - Architecture & Limitations

 LInk: https://vanco-assignment-29.streamlit.app/
## Architecture Diagram
 
The system uses a hybrid retrieval approach combining multiple search methods:
 
```
┌─────────────────────────────────────────────────────────────────┐
│                         User Query                              │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Query Processing                             │
│  - Conversation context building (for follow-up questions)      │
│  - Query vectorization for semantic search                      │
│  - Query tokenization for keyword search                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
┌─────────────────┐ ┌──────────────┐ ┌──────────────┐
│  Vector Search  │ │Keyword Search│ │ Graph Search │
│  (ChromaDB)     │ │   (BM25)     │ │   (Neo4j)    │
├─────────────────┤ ├──────────────┤ ├──────────────┤
│ - Semantic      │ │ - Exact      │ │ - Related    │
│   similarity    │ │   phrase     │ │   concepts   │
│ - 384-dim       │ │   matches    │ │ - Entities   │
│   embeddings    │ │ - TF-IDF     │ │ - Relations  │
│ - Cosine        │ │   scoring    │ │ - Traversal  │
│   similarity    │ │              │ │              │
└────────┬────────┘ └──────┬───────┘ └──────┬───────┘
         │                 │                │
         └─────────────────┼────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Reciprocal Rank Fusion (RRF)                       │
│  - Merges ranked lists from all retrieval methods               │
│  - Score: 1/(k + rank) where k=60                               │
│  - Robust to score scale differences                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Context Formatting                           │
│  - Combines top-k results with metadata                         │
│  - Includes page numbers, chapters, sections                    │
│  - Formats for LLM prompt                                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  LLM Response Generation                        │
│  - Model: NVIDIA NIM (Llama 3.1 8B Instruct)                    │
│  - System prompt enforces grounding in context                  │
│  - Generates answer with citations                              │
│  - Refuses to answer if context insufficient                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Final Response                             │
│  - Grounded answer with page references                         │
│  - Citations with chapter and section info                      │
│  - Evidence count and source tracking                           │
└─────────────────────────────────────────────────────────────────┘
```
 
**Components:**
- **Vector Database (ChromaDB):** Stores document chunks as 384-dimensional embeddings using sentence-transformers (all-MiniLM-L6-v2). Performs semantic search using cosine similarity.
- **Graph Database (Neo4j):** Optional knowledge graph with nodes for chapters, sections, formulas, and concepts. Edges represent relationships like contains, defines, relates_to, and uses.
- **Keyword Search (BM25):** Custom BM25 implementation for exact phrase matching using TF-IDF scoring with document length normalization.
- **Semantic Search:** Vector-based search using embeddings to find semantically similar content.
- **LLM Response Generation:** NVIDIA NIM API (Llama 3.1 8B Instruct) with system prompt enforcing grounding in retrieved context.
 
**Architecture Highlights:**
- Hybrid retrieval: Vector + Keyword + Graph for better recall and precision
- Reciprocal Rank Fusion (RRF) for robust result merging
- Grounded answer generation with citations from chunk metadata
- Graceful degradation when Neo4j is unavailable (falls back to vector + keyword)
- Conversation context support for follow-up questions
 
---
 
## Limitations and Improvement Plan
 
### Current Limitations

- **Chunking:** Fixed size may break logical units, split formulas, simple overlap strategy
- **Formula Handling:** Plain text storage, no LaTeX rendering, no formula-specific indexing
- **Graph Quality:** Rule-based extraction, limited relationship types, no entity disambiguation
- **Hallucination Risks:** LLM may generate unsupported content, no answer verification
- **Latency:** Sequential operations, no caching, 1-3 second total response time
- **Cost:** API costs per token, no budget controls or optimization
- **Scalability:** Single-machine deployment, no load balancing or horizontal scaling
- **Evaluation:** No automated metrics, RAG evaluation framework, or A/B testing
 
### Improvement Roadmap

- **Phase 1:** Evaluation framework, formula handling, caching, cost tracking
- **Phase 2:** Adaptive chunking, improved graph quality, query expansion
- **Phase 3:** Re-ranking, multi-hop reasoning, multi-document support
- **Phase 4:** Cloud deployment, monitoring, authentication, optimization
- **Phase 5:** Cross-document reasoning, advanced chains, multi-file support
 

