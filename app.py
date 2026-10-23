"""HR Policy Assistant - Streamlit Web Application

College GenAI Mid-Term Project: Project 1 — HR Policy Assistant
Provides an interactive QA interface over the 79-page Enso Group HR Policy Manual.
Uses persistent ChromaDB retrieval and Google Gemini for strictly grounded answers.
"""

import os
import sys
import time
from pathlib import Path

# Explicitly ensure .env is loaded
env_file = Path(__file__).resolve().parent / ".env"
if env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_file, override=True)
    except ImportError:
        pass
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))

import streamlit as st

# Import core RAG pipeline functions
from rag_pipeline import (
    get_vector_store,
    ask_question,
    DEFAULT_PDF_PATH,
    EXPECTED_PAGE_COUNT,
    CHAT_MODEL,
    EMBEDDING_MODEL
)

# Page configuration
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📋",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom styling for a clean, professional, minimal academic demo
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.2rem;
    }
    .badge-info {
        display: inline-block;
        background-color: #f1f5f9;
        color: #475569;
        padding: 0.3rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 500;
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    .answer-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #3b82f6;
        border-radius: 6px;
        padding: 1.2rem;
        margin-top: 1.2rem;
        margin-bottom: 1.2rem;
    }
    .not-found-card {
        background-color: #fffbeb;
        border: 1px solid #fef3c7;
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        padding: 1.2rem;
        margin-top: 1.2rem;
        margin-bottom: 1.2rem;
    }
    .source-tag {
        display: inline-block;
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .sample-btn {
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# 5 Evaluation benchmark questions
SAMPLE_QUESTIONS = [
    "How many days of Privilege Leave (PL) does a permanent employee get per year?",
    "What is the probation period for employees up to Senior Manager?",
    "What is the company's health-insurance premium amount?",
    "What is the current GST rate in India?",
    "How many days of maternity leave are given?"
]

# Initialize cached vector store
@st.cache_resource(show_spinner="Loading HR Manual Vector Index...")
def load_vector_store():
    """Load persistent ChromaDB vector store once per session."""
    return get_vector_store()

# Initialize session states
if "query_input" not in st.session_state:
    st.session_state.query_input = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# Sidebar
with st.sidebar:
    st.subheader("Document Overview")
    st.markdown(f"**Manual:** `Enso Group HR Manual`")
    st.markdown(f"**Pages:** `{EXPECTED_PAGE_COUNT} pages`")
    st.markdown(f"**Path:** `{DEFAULT_PDF_PATH}`")
    
    st.markdown("---")
    st.subheader("System Architecture")
    st.markdown(f"• **LLM:** `{CHAT_MODEL}`")
    st.markdown(f"• **Embeddings:** `{EMBEDDING_MODEL}`")
    st.markdown("• **Vector Store:** `ChromaDB (Local)`")
    st.markdown("• **Grounding:** Strict Anti-Hallucination")
    
    st.markdown("---")
    st.subheader("Grounding Rules")
    st.caption(
        "Answers are derived solely from the text of the PDF. "
        "Questions not explicitly answered in the manual return **Not found**."
    )
    
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.query_input = ""
        st.session_state.last_result = None
        st.session_state.pending_query = None
        st.rerun()

# Header
st.markdown('<div class="main-title">HR Policy Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Ask questions about the Enso Group HR Policy Manual</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="badge-info">📄 Source: Enso Group HR Policy Manual (79 Pages) &nbsp;•&nbsp; Strict Evidence Grounding</div>',
    unsafe_allow_html=True
)

# Sample Questions Section
st.markdown("##### Try Evaluation Questions")
cols = st.columns(1)
for idx, q_text in enumerate(SAMPLE_QUESTIONS, start=1):
    if st.button(f"Q{idx}: {q_text}", key=f"sample_{idx}", use_container_width=True):
        st.session_state.pending_query = q_text
        st.session_state.query_input = q_text

st.markdown("---")

# Question Input Section
query_from_input = st.text_input(
    "Enter your policy question:",
    value=st.session_state.query_input,
    placeholder="e.g., How many days of Privilege Leave does an employee get?",
    key="text_box"
)

col_ask, col_space = st.columns([1, 4])
with col_ask:
    ask_button = st.button("Ask Assistant", type="primary", use_container_width=True)

# Determine if a query should be executed
active_query = None
if ask_button and query_from_input.strip():
    active_query = query_from_input.strip()
elif st.session_state.pending_query:
    active_query = st.session_state.pending_query
    st.session_state.pending_query = None

# Execute query if triggered
if active_query:
    with st.spinner("Searching manual and generating grounded response..."):
        try:
            vs = load_vector_store()
            start_t = time.time()
            result = ask_question(active_query, vector_store=vs)
            result["elapsed"] = round(time.time() - start_t, 2)
            st.session_state.last_result = result
            st.session_state.query_input = active_query
        except Exception as e:
            st.session_state.last_result = {
                "question": active_query,
                "answer": f"System Error: {e}",
                "sources": [],
                "excerpts": [],
                "status": "error",
                "error": str(e),
                "elapsed": 0.0
            }

# Display Results Section
res = st.session_state.last_result
if res:
    question = res.get("question", "")
    answer = res.get("answer", "")
    sources = res.get("sources", [])
    excerpts = res.get("excerpts", [])
    status = res.get("status", "unknown")
    elapsed = res.get("elapsed", 0.0)

    st.markdown(f"**Question:** *{question}*")

    if status == "error":
        st.error(f"⚠️ {answer}")
        st.caption("Please verify that your `GEMINI_API_KEY` is correctly set in `.env` and that the network is accessible.")

    elif status == "not_found" or answer == "Not found":
        st.markdown(f"""
        <div class="not-found-card">
            <h4 style="margin-top:0; color: #b45309;">Result: Not found</h4>
            <p style="margin-bottom:0; color: #78350f;">
                The HR Policy Manual does not contain sufficient evidence to answer this question.
                In adherence to anti-hallucination protocols, external knowledge or unverified guesses are not permitted.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"Search completed in {elapsed}s • 0 citations returned")

    else:
        # Successful answer display
        st.markdown(f"""
        <div class="answer-card">
            <h4 style="margin-top:0; color: #1e3a8a;">Answer</h4>
            <div style="font-size: 1.02rem; line-height: 1.6; color: #1e293b;">
                {answer}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Source page tags
        if sources:
            st.markdown("**Referenced Policy Pages:**")
            tag_html = "".join([f'<span class="source-tag">📄 Page {p}</span>' for p in sources])
            st.markdown(tag_html, unsafe_allow_html=True)

        # Supporting excerpts accordion
        if excerpts:
            with st.expander(f"🔍 Supporting Excerpts from HR Manual ({len(excerpts)} chunks retrieved)"):
                for idx, exc in enumerate(excerpts, start=1):
                    src_page = sources[idx - 1] if idx - 1 < len(sources) else "Manual"
                    st.markdown(f"**Excerpt {idx} (from Page {src_page}):**")
                    st.markdown(f"> *{exc.strip()}*")
                    if idx < len(excerpts):
                        st.markdown("---")

        st.caption(f"Retrieved and generated in {elapsed}s from Enso Group HR Policy Manual")
