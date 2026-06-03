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
