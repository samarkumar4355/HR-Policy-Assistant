# HR Policy Assistant (GenAI Mid-Term Project)

A beginner-friendly Retrieval-Augmented Generation (RAG) web application that answers employee questions strictly based on the **Enso Group HR Policy Manual** (79 pages).

---

## 1. System Architecture & Workflow

```
[hr_manual.pdf (79 pages)]
         │
         ▼
[PyMuPDF Page-by-Page Extraction] ── (Preserves 1-indexed page metadata: 1-79)
         │
         ▼
[Recursive Character Chunking] ───── (1,500 chars, 200 char overlap)
         │
         ▼
[Gemini Embeddings] ──────────────── (models/gemini-embedding-001)
         │
         ▼
[Persistent ChromaDB Store] ──────── (Fingerprinted cache in chroma_db/)
         │
         ├─── Query: Employee Question (via Streamlit UI or evaluate.py)
         ▼
[Similarity Search & Retrieval] ──── (Top-k relevant chunks)
         │
         ▼
[Strict Grounding Prompt] ────────── (Zero outside knowledge / anti-hallucination)
         │
         ▼
[Gemini Answer Generation] ───────── (gemini-3.5-flash)
         │
         ├── Found ──────► Answer + Page Badges + Supporting Excerpts
         └── Missing ────► Exactly "Not found"
```

---

## 2. Evaluation Questions & Verified Answers

All 5 benchmark evaluation questions were tested directly against the source PDF (`data/hr_manual.pdf`):

| # | Question | Verified Answer | Source Page & Section | Verification Status |
|---|---|---|---|---|
| **1** | How many days of Privilege Leave (PL) does a permanent employee get per year? | **12 days** per calendar year (1st Jan - 31st Dec) | **Page 60**, Annexure III, `[G] Leave Policy - Privilege Leave (PL)` | Verified against PDF |
| **2** | What is the probation period for employees up to Senior Manager? | **3 months** from the date of joining | **Page 42**, Annexure I, `[B] Probation Policy` | Verified against PDF |
| **3** | What is the company's health-insurance premium amount? | **`Not found`** (The manual mentions travel insurance and medical reimbursement, but does not state any health insurance premium amount) | None (Not in manual) | Verified absent from PDF |
| **4** | What is the current GST rate in India? | **`Not found`** (External tax information is intentionally excluded from the company HR manual; model refuses to hallucinate) | None (Not in manual) | Verified absent from PDF |
| **5** | How many days of maternity leave are given? | **12 weeks (90 days)** with pay (not more than 6 weeks / 45 days preceding expected delivery) + **4 weeks** for miscarriage | **Page 62**, Annexure III, `[G] Leave Policy - Maternity Leave` | Verified against PDF |

---

## 3. Project Structure

```
hr_maual_project/
├── app.py                 # Streamlit web user interface
├── rag_pipeline.py        # PDF extraction, ChromaDB indexer, and Gemini RAG pipeline
├── evaluate.py            # Automated CLI test suite for the 5 evaluation questions
├── requirements.txt       # Dependencies (PyMuPDF, LangChain, ChromaDB, Gemini)
├── .env                   # Local secrets (GEMINI_API_KEY)
├── .env.example           # Example configuration template
├── .gitignore             # Excludes secrets, chroma_db, and virtual environments
├── README.md              # Project documentation and evaluation report
├── chroma_db/             # Persistent vector database and fingerprint cache
├── data/
│   └── hr_manual.pdf      # Source 79-page HR policy document
└── screenshots/           # Demonstration screenshots for submission
    ├── ui_main_screen.png
    ├── ui_q1_privilege_leave.png
    └── ui_q4_not_found.png
```

---

## 4. Setup & Running Instructions

### Step 1: Environment & API Key
Ensure you are using your Anaconda Python environment where packages are installed:
```bash
conda activate base
```

Configure your `.env` file with your Gemini API key:
```ini
GEMINI_API_KEY=your_actual_api_key_here
```

### Step 2: Install Dependencies (if not already installed)
```bash
pip install -r requirements.txt
```

### Step 3: Run the Automated Evaluation Suite
```bash
/opt/anaconda3/bin/python3 evaluate.py
```
*(Or `python3 evaluate.py` with conda activated)*

### Step 4: Launch the Streamlit Web Application
```bash
/opt/anaconda3/bin/streamlit run app.py
```
The application will open in your browser at `http://localhost:8501`.

---

## 5. Demonstration Screenshots

Captured live in the browser and ready for submission:
- `screenshots/ui_main_screen.png` — Main application interface with evaluation question buttons.
- `screenshots/ui_q1_privilege_leave.png` — Successful answer with cited Page 60 badge and supporting excerpts.
- `screenshots/ui_q4_not_found.png` — Grounded anti-hallucination response displaying `Result: Not found`.




![alt text](<Screenshot 2026-10-23 at 4.30.05 PM.png>) ![alt text](<Screenshot 2026-10-23 at 4.29.50 PM.png>) ![alt text](<Screenshot 2026-10-23 at 4.29.33 PM.png>)

![alt text](<Screenshot 2026-10-23 at 4.29.22 PM.png>) ![alt text](<Screenshot 2026-10-23 at 4.28.18 PM.png>) ![alt text](<Screenshot 2026-10-23 at 4.27.29 PM.png>)