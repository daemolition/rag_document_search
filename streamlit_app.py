import streamlit as st
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).parent))

from src.config.config import Config
from src.document_ingestion.document_processor import DocumentProcessor
from src.vectorestore.vectorestore import VectoreStore
from src.graph_builder.rag_graph import GraphBuilder
from src.llm.ollama import LLM

# Page configuration
st.set_page_config(
    page_title="RAG Search",
    layout="centered"
)

# CSS
st.markdown("""
    <style>
      .stButton > button {
          width: 100%;
          background-color: #4CAF50;
          color: white;
          font-weight: bold;
      }
    </style>            
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session sate variables"""
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "history" not in st.session_state:
        st.session_state.history = []
        

@st.cache_resource
def initialize_rag():
    """Initialize the RAG system (cached)"""
    
    try:
        ollama_llm = LLM(model=Config.LLM_MODEL)
        model = ollama_llm.get_llm
        
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        vectore_store = VectoreStore()
        
        # Use default URLs
        urls = Config.DEFAULT_URLS
        
        # Process documents
        documents = doc_processor.process_urls(urls)
        
        # Create vetore store
        vectore_store.create_vectorstore(documents=documents)
        
        # Initialize the graph
        builder = GraphBuilder(
            retriever=vectore_store.get_retriever(),
            llm=model
        )
        builder.build()
        
        return builder, len(documents)      
    
    except Exception as e:
        print(f"Error: {e}")
        return None, 0
        

def main():
    """Main application"""
    init_session_state()
    
    # Title
    st.title("Rag Document Search")
    st.markdown("Ask questions about the loaded documents")
    
    # Initialize system
    if not st.session_state.initialized:
        with st.spinner("Loading system..."):
            rag_system, num_chunks = initialize_rag()
            if rag_system:
                st.session_state.rag_system = rag_system
                st.session_state.initialized = True
                st.success(f"System ready! ({num_chunks} document chunks loaded)")
    
    st.markdown("---")
    
    # Search interface
    with st.form("search_form"):
        question = st.text_input("Enter your question", placeholder="What would you like to know?")
        submit = st.form_submit_button("Search")
        
    # Process search
    if submit and question:
        if st.session_state.rag_system:
            with st.spinner("Searching..."):
                start_time = time.time()
                
                # Get the answer
                result = st.session_state.rag_system.run(question)
                
                elapsed_time = time.time() - start_time
                # Add to history
                st.session_state.history.append({
                    'question':question,
                    'answer': result['answer'],
                    'time': elapsed_time
                })
        
                # Display the answer
                st.markdown("Answer")
                st.success(result['answer'])
                
                # Show retrieved docs
                with st.expander("Source documents"):
                    for i, doc in enumerate(result['retrieved_documents'], 1):
                        st.text_area(
                            f"Document {i}",
                            doc.page_content[:300] + "...",
                            height=100,
                            disabled=True
                        )
                        
                st.caption(f"Response Time; {elapsed_time:.2f} seconds")
                
    if st.session_state.history:
        st.markdown("---")
        st.markdown("Recent Searches")
        
        for item in reversed(st.session_state.history[-3:]):
            with st.container():
                st.markdown(f"Q: {item['question']}")
                st.markdown(f"A: {item['answer'][:200]}...")
                st.caption(f"Time: {item['time']:.2f}s")
                st.markdown("")
                
                
if __name__ == "__main__":
    main()