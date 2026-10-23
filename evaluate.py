"""Evaluation Script for HR Policy Assistant

Runs the 5 evaluation questions against the Enso Group HR Policy Manual using the RAG pipeline.
Prints question, answer, source pages, supporting excerpts, and verification indications.
"""

import sys
import time
from pathlib import Path
from rag_pipeline import (
    get_vector_store,
    ask_question,
    validate_and_load_pdf,
    DEFAULT_PDF_PATH
)

EVALUATION_QUESTIONS = [
    "How many days of Privilege Leave (PL) does a permanent employee get per year?",
    "What is the probation period for employees up to Senior Manager?",
    "What is the company's health-insurance premium amount?",
    "What is the current GST rate in India?",
    "How many days of maternity leave are given?"
]

def run_evaluation():
    print("=" * 75)
    print(" HR POLICY ASSISTANT - EVALUATION TEST ")
    print("=" * 75)
    
    # Step 1: Pre-flight document check
    print("\n[1/3] Validating HR Manual PDF...")
    try:
        docs, stats = validate_and_load_pdf(DEFAULT_PDF_PATH)
        print(f"  ✓ PDF validated: {stats['total_pages']} pages loaded.")
        if stats['sparse_pages']:
            print(f"  ℹ Divider/Sparse pages identified: {stats['sparse_pages']}")
    except Exception as e:
        print(f"  ✗ PDF Validation Error: {e}")
        sys.exit(1)

    # Step 2: Initialize vector store (loads existing ChromaDB or indexes if needed)
    print("\n[2/3] Loading ChromaDB Vector Index...")
    t0 = time.time()
    try:
        vector_store = get_vector_store()
        print(f"  ✓ ChromaDB index ready in {round(time.time() - t0, 2)}s.")
    except Exception as e:
        print(f"  ✗ Vector Store Error: {e}")
        sys.exit(1)

    # Step 3: Run evaluation questions
    print("\n[3/3] Running Evaluation Questions...\n")
    results = []
    
    for idx, question in enumerate(EVALUATION_QUESTIONS, start=1):
        print("-" * 75)
        print(f"Question {idx}: {question}")
        print("-" * 75)
        
        q_start = time.time()
        res = ask_question(question, vector_store=vector_store)
        elapsed = round(time.time() - q_start, 2)
        
        answer = res.get("answer", "Not found")
        sources = res.get("sources", [])
        excerpts = res.get("excerpts", [])
        status = res.get("status", "unknown")
        error = res.get("error")
        
        if error:
            print(f"System Error: {error}")
            needs_verification = "YES (System Error Occurred)"
        elif answer == "Not found":
            print("Answer: Not found")
            print("Source Pages: None (Not present in HR Manual)")
            needs_verification = "NO (Document explicitly does not contain this information)"
        else:
            print(f"Answer: {answer}")
            print(f"Source Page(s): {sources}")
            if excerpts:
                print(f"Supporting Excerpt:\n  \"{excerpts[0].replace(chr(10), ' ')[:250]}...\"")
            needs_verification = "RECOMMENDED (Verify against cited page)"

        print(f"Verification Indication: {needs_verification}")
        print(f"Response Time: {elapsed}s\n")
        
        results.append({
            "num": idx,
            "question": question,
            "answer": answer,
            "sources": sources,
            "status": status,
            "needs_verification": needs_verification,
            "elapsed": elapsed
        })
        
        # Brief pause between queries to stay well within API rate limits
        time.sleep(2)

    # Summary
    print("=" * 75)
    print(" EVALUATION SUMMARY ")
    print("=" * 75)
    for r in results:
        src_str = f"Page {r['sources']}" if r['sources'] else "N/A"
        print(f"Q{r['num']}: [{r['status'].upper()}] - {src_str} - {r['answer'][:60]}")
    print("=" * 75)

if __name__ == "__main__":
    run_evaluation()
