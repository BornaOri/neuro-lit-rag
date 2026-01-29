"""
NeuroLitRAG - Streamlit App
Secure deployment for Streamlit Cloud
"""

import streamlit as st
import os
from dotenv import load_dotenv

# ============================================
# SECURE API KEY HANDLING
# ============================================
# Priority: 1) Streamlit Cloud secrets 2) Local .env file
# NEVER hardcode your API key!

load_dotenv()

try:
    # Streamlit Cloud deployment
    if 'COHERE_API_KEY' in st.secrets:
        os.environ['COHERE_API_KEY'] = st.secrets['COHERE_API_KEY']
except Exception:
    pass  # Running locally with .env file

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="NeuroLitRAG",
    page_icon="🧠",
    layout="wide"
)

# ============================================
# STYLING
# ============================================
st.markdown("""
<style>
    .main-header { 
        font-size: 2.5rem; 
        font-weight: bold; 
        background: linear-gradient(90deg, #3B82F6, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .citation-box {
        background-color: #1E293B;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 3px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)


def check_api_key():
    """Verify API key is configured."""
    key = os.getenv("COHERE_API_KEY")
    if not key or key == "paste_your_api_key_here":
        st.error("⚠️ Cohere API key not configured!")
        st.markdown("""
        **For local development:**
        1. Create a `.env` file
        2. Add: `COHERE_API_KEY=your_key_here`
        
        **For Streamlit Cloud:**
        1. Go to App Settings → Secrets
        2. Add: `COHERE_API_KEY = "your_key_here"`
        
        Get a free key: https://dashboard.cohere.com/api-keys
        """)
        st.stop()


@st.cache_resource
def load_rag():
    """Initialize and cache RAG system."""
    from src.pipeline import NeuroLitRAG
    rag = NeuroLitRAG()
    rag.load_demo_data()
    return rag


def main():
    # Header
    st.markdown('<p class="main-header">🧠 NeuroLitRAG</p>', unsafe_allow_html=True)
    st.markdown("*AI-Powered Neuroscience Literature Search & Synthesis*")
    st.markdown("Built with **Cohere Embed + Rerank + Command**")
    st.divider()
    
    # Check API key
    check_api_key()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        use_reranking = st.checkbox(
            "Use Cohere Rerank", 
            value=True,
            help="Improves relevance of results"
        )
        
        st.divider()
        
        st.header("🔬 How It Works")
        st.markdown("""
        1. 🔍 **Embed** your query
        2. 📚 **Search** vector database
        3. 🎯 **Rerank** for relevance
        4. ✍️ **Generate** cited answer
        """)
        
        st.divider()
        
        st.header("📊 Demo Data")
        st.caption("7 neuroscience papers on memory, plasticity, and neurodegeneration")
        
        st.divider()
        
        st.markdown("**Built for Cohere** 🚀")
        st.caption("[GitHub Repo](https://github.com)")
    
    # Load RAG
    with st.spinner("🔄 Loading NeuroLitRAG..."):
        rag = load_rag()
    
    # Main interface
    st.subheader("🔍 Ask a Neuroscience Question")
    
    # Example questions
    examples = [
        "What is the role of the hippocampus in memory?",
        "How does dopamine affect reward processing?",
        "What role do microglia play in Alzheimer's disease?",
        "What are the mechanisms of synaptic plasticity?",
    ]
    
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state.query = ex
    
    # Query input
    query = st.text_input(
        "Your question:",
        value=st.session_state.get("query", ""),
        placeholder="e.g., How does sleep affect memory consolidation?"
    )
    
    # Search button
    if st.button("🔍 Search", type="primary", use_container_width=True) and query:
        with st.spinner("🔄 Searching and generating answer..."):
            result = rag.query(query, use_reranking=use_reranking)
        
        # Display answer
        st.subheader("📝 Answer")
        st.markdown(result["answer"])
        
        # Display citations
        st.subheader(f"📚 Sources ({result['sources_used']} cited)")
        
        for c in result["citations"]:
            with st.container():
                st.markdown(f"**[{c['number']}]** {c['authors']} ({c['year']})")
                st.markdown(f"*{c['title']}*")
                st.caption(c['journal'])
                st.divider()
        
        # Rerank scores
        if result.get("rerank_scores"):
            st.subheader("🎯 Relevance Scores")
            cols = st.columns(len(result["rerank_scores"]))
            for i, score in enumerate(result["rerank_scores"]):
                with cols[i]:
                    st.metric(f"Source {i+1}", f"{score:.1%}")


if __name__ == "__main__":
    main()
