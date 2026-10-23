"""HR Policy Assistant - Core RAG Pipeline

Built with PyMuPDF, LangChain, ChromaDB, and Google Gemini API.
Answers employee questions strictly from the Enso Group HR Policy Manual.
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Safe dotenv loader with manual fallback if dotenv is missing
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.exists():
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma

# Constants
DEFAULT_PDF_PATH = Path("data/hr_manual.pdf")
DEFAULT_CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "enso_hr_policy_manual"
EMBEDDING_MODEL = "models/gemini-embedding-001"
CHAT_MODEL = "gemini-3.5-flash"
EXPECTED_PAGE_COUNT = 79
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# Strict System Prompt
SYSTEM_PROMPT = """You are an authoritative HR Assistant for Enso Group. Your task is to answer employee questions strictly and solely based on the provided HR Policy Manual excerpts.

CRITICAL RULES:
1. The HR Manual excerpts provided below are your ONLY source of truth.
2. NEVER use outside knowledge, general assumptions, or external facts (such as general country laws, tax rates, or general industry standards).
3. If the answer is not explicitly stated in the provided excerpts, or if the excerpts do not contain sufficient evidence to answer the question, output EXACTLY AND ONLY:
Not found
4. If the answer is found in the excerpts:
   - Provide a direct, concise, and accurate answer based strictly on the excerpts.
   - Mention the relevant policy section and page number when available.
5. Do NOT invent, guess, or extrapolate. If uncertain or only partially mentioned without the specific answer requested, return:
Not found
"""

def extract_text_from_content(content: Any) -> str:
    """Extract plain text safely from LLM response content (supports str, list of dicts, etc.)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        return "".join(parts).strip()
    return str(content).strip()


def get_api_key() -> str:
    """Retrieve GEMINI_API_KEY safely without logging secret values."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key or key.strip() in ("", "your_gemini_api_key_here"):
        raise ValueError("GEMINI_API_KEY is not configured. Please add your Gemini API key to .env")
    return key.strip()


def calculate_pdf_hash(pdf_path: Path) -> str:
    """Calculate SHA-256 hash of the PDF to track document modifications."""
    hasher = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_and_load_pdf(pdf_path: Path = DEFAULT_PDF_PATH) -> Tuple[List[Document], Dict[str, Any]]:
    """Validate existence, 79-page count, and extract text page by page with PyMuPDF.
    
    Returns (list_of_documents, statistics_dict).
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found at '{pdf_path}'. Please ensure hr_manual.pdf is in data/")
    
    doc = pymupdf.open(str(pdf_path))
    page_count = len(doc)
    if page_count != EXPECTED_PAGE_COUNT:
        raise ValueError(
            f"PDF validation error: expected {EXPECTED_PAGE_COUNT} pages, but found {page_count}."
        )
    
    documents = []
    sparse_pages = []
    
    for idx, page in enumerate(doc):
        page_num = idx + 1
        text = page.get_text() or ""
        text_clean = text.strip()
        
        # Track divider / sparse pages (<50 characters)
        if len(text_clean) < 50:
            sparse_pages.append(page_num)
            
        doc_obj = Document(
            page_content=text,
            metadata={"page": page_num, "source": str(pdf_path)}
        )
        documents.append(doc_obj)
        
    stats = {
        "total_pages": page_count,
        "sparse_pages": sparse_pages,
        "total_characters": sum(len(d.page_content) for d in documents)
    }
    return documents, stats


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Document]:
    """Split page documents into overlapping chunks while preserving original page metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    all_chunks = []
    for doc in documents:
        page_chunks = splitter.split_documents([doc])
        for chunk in page_chunks:
            # Ensure metadata preservation
            chunk.metadata["page"] = doc.metadata["page"]
            chunk.metadata["source"] = doc.metadata["source"]
            all_chunks.append(chunk)
    return all_chunks


def get_vector_store(
    pdf_path: Path = DEFAULT_PDF_PATH,
    persist_dir: Path = DEFAULT_CHROMA_DIR,
    force_reindex: bool = False
) -> Chroma:
    """Load existing ChromaDB index or build a new persistent index with document fingerprinting."""
    api_key = get_api_key()
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key
    )
    
    fingerprint_file = persist_dir / "fingerprint.json"
    current_hash = calculate_pdf_hash(pdf_path)
    
    # Check if index exists and is up to date
    if not force_reindex and persist_dir.exists() and fingerprint_file.exists():
        try:
            with open(fingerprint_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if (
                meta.get("pdf_hash") == current_hash
                and meta.get("chunk_size") == CHUNK_SIZE
                and meta.get("embedding_model") == EMBEDDING_MODEL
            ):
                return Chroma(
                    collection_name=COLLECTION_NAME,
                    persist_directory=str(persist_dir),
                    embedding_function=embeddings
                )
        except Exception:
            pass  # Fall through to re-indexing if fingerprint cannot be read
            
    # Index needs to be built
    documents, stats = validate_and_load_pdf(pdf_path)
    chunks = chunk_documents(documents)
    
    # Remove stale index directory if exists
    if persist_dir.exists():
        import shutil
        shutil.rmtree(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(persist_dir),
        embedding_function=embeddings
    )
    
    # Batch indexing with rate-limit handling
    batch_size = 40
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        max_retries = 3
        for attempt in range(max_retries):
            try:
                vector_store.add_documents(batch)
                break
            except Exception as e:
                err_text = str(e)
                if ("429" in err_text or "RESOURCE_EXHAUSTED" in err_text) and attempt < max_retries - 1:
                    time.sleep(35)
                else:
                    raise e
                    
    # Save fingerprint
    fingerprint_data = {
        "pdf_hash": current_hash,
        "page_count": stats["total_pages"],
        "total_chunks": len(chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
        "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(fingerprint_file, "w", encoding="utf-8") as f:
        json.dump(fingerprint_data, f, indent=2)
        
    return vector_store


def ask_question(
    question: str,
    vector_store: Optional[Chroma] = None,
    top_k: int = 4
) -> Dict[str, Any]:
    """Retrieve relevant chunks and generate a grounded answer using Gemini.
    
    Returns:
        dict: {
            "question": str,
            "answer": str,
            "sources": List[int],
            "excerpts": List[str],
            "status": "success" | "not_found" | "error",
            "error": Optional[str]
        }
    """
    clean_q = question.strip()
    if not clean_q:
        return {
            "question": question,
            "answer": "Not found",
            "sources": [],
            "excerpts": [],
            "status": "not_found",
            "error": None
        }

    try:
        api_key = get_api_key()
    except Exception as e:
        return {
            "question": question,
            "answer": "Error: GEMINI_API_KEY is missing",
            "sources": [],
            "excerpts": [],
            "status": "error",
            "error": str(e)
        }

    # Ensure vector store is available
    try:
        if vector_store is None:
            vector_store = get_vector_store()
    except Exception as e:
        return {
            "question": question,
            "answer": "Error: Vector store initialization failed",
            "sources": [],
            "excerpts": [],
            "status": "error",
            "error": str(e)
        }

    # Retrieve candidate chunks
    try:
        retrieved_docs = vector_store.similarity_search(clean_q, k=top_k)
    except Exception as e:
        return {
            "question": question,
            "answer": "Error: Document retrieval failed",
            "sources": [],
            "excerpts": [],
            "status": "error",
            "error": str(e)
        }

    if not retrieved_docs:
        return {
            "question": question,
            "answer": "Not found",
            "sources": [],
            "excerpts": [],
            "status": "not_found",
            "error": None
        }

    # Format context for Gemini
    context_blocks = []
    seen_pages = []
    excerpts = []
    
    for doc in retrieved_docs:
        page_num = doc.metadata.get("page", "Unknown")
        if page_num not in seen_pages:
            seen_pages.append(page_num)
        snippet = doc.page_content.strip()
        context_blocks.append(f"--- [Excerpt from Page {page_num}] ---\n{snippet}")
        excerpts.append(snippet[:350] + ("..." if len(snippet) > 350 else ""))

    combined_context = "\n\n".join(context_blocks)

    # Prompt generation with Gemini
    try:
        llm = ChatGoogleGenerativeAI(
            model=CHAT_MODEL,
            google_api_key=api_key,
            temperature=0.0
        )
        
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Context Excerpts from HR Manual:\n{combined_context}\n\n"
            f"Question: {clean_q}\n\n"
            f"Answer:"
        )
        
        response = llm.invoke(prompt)
        raw_answer = extract_text_from_content(response.content)
    except Exception as e:
        return {
            "question": question,
            "answer": "Error: Gemini generation failed",
            "sources": [],
            "excerpts": [],
            "status": "error",
            "error": str(e)
        }

    # Check for "Not found"
    norm_answer = raw_answer.strip()
    if norm_answer.lower() == "not found" or norm_answer.lower().startswith("not found"):
        return {
            "question": question,
            "answer": "Not found",
            "sources": [],
            "excerpts": [],
            "status": "not_found",
            "error": None
        }

    # Extract source page numbers from answer or fallback to retrieved pages
    cited_pages = []
    for p in seen_pages:
        if isinstance(p, int) and (f"page {p}" in norm_answer.lower() or f"page: {p}" in norm_answer.lower()):
            cited_pages.append(p)
    if not cited_pages:
        cited_pages = [p for p in seen_pages if isinstance(p, int)]

    return {
        "question": question,
        "answer": norm_answer,
        "sources": cited_pages,
        "excerpts": excerpts,
        "status": "success",
        "error": None
    }
