---
title: VaultMind
emoji: 🧠
colorFrom: violet
colorTo: indigo
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
---

# VaultMind 🧠📚

**An End-to-End Hybrid RAG System with Citation Grounding, Multi-Source Comparison & Section-Aware Document Processing.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Gradio UI](https://img.shields.io/badge/UI-Gradio_6.x-orange.svg)](https://gradio.app/)
[![ChromaDB](https://img.shields.io/badge/Vector_Store-ChromaDB-green.svg)](https://trychroma.com/)
[![Groq & Ollama](https://img.shields.io/badge/LLM-Groq_%7C_Ollama-purple.svg)](https://groq.com/)

---

## ⚡ Quickstart Guide (Clone → Set Up → Run in < 20 Minutes)

Whether you are reviewing the code or testing the retrieval capabilities on your own documents, you can get VaultMind running locally in minutes.

### 1. Clone & Create Virtual Environment

```powershell
# Clone the repository
git clone https://github.com/YourUsername/VaultMind.git
cd VaultMind

# Create and activate virtual environment (Windows PowerShell)
python -m venv .venv
.venv\Scripts\activate

# (On Linux / macOS)
# python3 -m venv .venv && source .venv/bin/activate
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure API Keys (`.env`)

Create a `.env` file in the project root directory. By default, VaultMind uses **Groq's free tier** (`llama-3.1-8b-instant`) for blazing-fast, structured generation. You can also run completely local/offline using **Ollama** (`qwen2.5:7b`).

```env
# Default Backend: Groq (Free Tier — 14,400 req/day)
VAULTMIND_LLM_BACKEND=groq
GROQ_API_KEY=gsk_your_groq_api_key_here

# Optional Alternative Backends:
# VAULTMIND_LLM_BACKEND=ollama
# VAULTMIND_LLM_BACKEND=openai
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
```

### 4. Add Your Source PDFs

Place your target PDF document(s) inside the `data/raw/` directory:

```powershell
mkdir -p data/raw
# Place your textbook or technical PDFs into data/raw/
```

### 5. Launch VaultMind!

You can run VaultMind either via the **Interactive Gradio Web UI** or the **CLI Pipeline**:

#### Option A: Interactive Web UI (Recommended)
```powershell
python app.py
```
* Open your browser to **`http://localhost:7860`** (or the auto-assigned local port).
* **Screen 1 (Upload):** View detected PDFs or upload multiple new files, then click **Process & Chat**.
* **Screen 2 (Chat):** Select active source PDFs dynamically, ask complex technical queries, and view grounded answers with expandable page-level citations (`Page 107 — 12. ACID Properties in DBMS`).

#### Option B: End-to-End Interactive CLI Pipeline
```powershell
python src/pipeline.py
```
* Interactive numbered menu lets you select specific PDFs (or `a` for all) and input queries directly.
* View granular breakdown of every stage: Ingestion → Section Chunking → BGE Embeddings → ChromaDB Storage → Hybrid BM25 + Dense RRF Retrieval → Cross-Encoder Reranking → Guardrail Verification → LLM Answer.
* **Per-Source Comparison Prompt:** When multiple sources are retrieved, opt in to generate side-by-side answers for each PDF alongside an automated LLM agreement analysis!

---

## 🏗️ System Architecture & Workflow

```text
                  Raw Technical PDFs (data/raw/)
                              │
                    ┌─────────▼─────────┐
                    │ Document Ingestion│  PyMuPDF (fitz) + pdfplumber
                    │ & Kerning Fallback│  Table preservation & spacing recovery
                    └─────────┬─────────┘
                              │
                     Structured Pages (JSON)
                              │
                    ┌─────────▼─────────┐
                    │ Section Builder & │  Hierarchical split alignment with
                    │ Semantic Chunking │  page/section/heading metadata
                    └─────────┬─────────┘
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
     ┌───────────────────┐         ┌───────────────────┐
     │ BM25 In-Memory    │         │ ChromaDB Vector   │
     │ Lexical Index     │         │ Store (BGE-v1.5)  │
     └─────────┬─────────┘         └─────────┬─────────┘
               │                             │
               └──────────────┬──────────────┘
                              ▼
                Reciprocal Rank Fusion (RRF)
            (Per-source fair candidate pooling)
                              │
                              ▼
               Cross-Encoder Reranking Engine
               (ms-marco-MiniLM-L-6-v2 top-k)
                              │
                              ▼
                  Safety Guardrails Layer
                (Out-of-scope & quality check)
                              │
                              ▼
               Citation-Grounded LLM Generation
               (JSON Schema forced exact quotes)
```

---

## 🌟 Key Engineering Features

### 1. Robust Section-Aware Ingestion & Spacing Recovery (`ingest.py`)
* **Multi-Engine Parsing:** Combines **PyMuPDF (`fitz`)** for rapid block-level heading detection and font-size heuristics with **pdfplumber** for table bounding-box isolation and caption assignment.
* **Kerning & Spacing Fallback:** Academic PDFs often encode character glyph advances without explicit space characters (resulting in garbled text like `Whatisatransaction?`). VaultMind applies a 3-signal heuristic (`_has_spacing_issue`) to detect zero-advance concatenation and automatically re-extracts affected bounding boxes using `pdfplumber` character `x-position` word inference.
* **Table Deduplication:** Text blocks overlapping table coordinates (`TABLE_OVERLAP_THRESHOLD`) are consumed directly into markdown table structures, preserving vertical reading order without duplicate paragraphs.

### 2. Hierarchical Metadata Preservation (`chunk.py`)
Every chunk maintains exact trace metadata:
```json
{
  "id": "chunk_01049",
  "text": "A transaction is a single logical unit of work that accesses...",
  "metadata": {
    "start_page": 107,
    "end_page": 107,
    "source_doc": "Database Management Systems.pdf",
    "section_title": "12. ACID Properties in DBMS",
    "heading_level": 1,
    "chunk_index": 1049
  }
}
```

### 3. Hybrid RRF Retrieval with Per-Source Fair Pooling (`retrieve.py`)
* **Lexical + Dense Fusion:** Combines exact keyword recall (`rank_bm25`) with semantic dense similarity (`ChromaDB` via `BAAI/bge-small-en-v1.5`), normalized through **Reciprocal Rank Fusion (RRF)**: $\text{Score} = \frac{1}{60 + r_{\text{BM25}}} + \frac{1}{60 + r_{\text{Dense}}}$.
* **Per-Source Candidate Allocation (`retrieve_per_source`):** Prevents large 600-page textbooks from crowding out concise 15-page reference manuals by running hybrid RRF across each active document independently, pooling top candidates before global reranking.
* **Cross-Encoder Reranking:** Passes the fused pool through `cross-encoder/ms-marco-MiniLM-L-6-v2` for precise query-to-passage semantic scoring (`rerank_score`).

### 4. Safety Guardrails & Hallucination Prevention (`guardrails.py`)
* Evaluates the top reranked chunks before generation.
* Rejects out-of-scope queries (`"How do I bake a cake?"`) or poorly retrieved candidate sets (`rerank_score < threshold`), returning a clean refusal state rather than fabricating technical claims.

### 5. Citation-Grounded Structured Generation (`generate.py`)
* Enforces strict JSON Schema parsing on generation responses.
* Ensures every claim is tied directly to retrieved chunks, displaying exact page ranges and source PDF names inside the Gradio UI and CLI outputs.

### 6. Multi-Source Comparison & Agreement Analysis (`compare.py`)
* When 2+ sources are retrieved for a query, VaultMind can isolate chunks by document and generate **standalone per-source answers side-by-side**.
* Automatically triggers an **LLM Agreement Meta-Analysis** to synthesize what all documents agree on, unique details provided by each source, and any technical contradictions.

---

## 💻 CLI & Advanced Usage

### Running `pipeline.py` with CLI Arguments

You can automate or script VaultMind without interactive prompts using command-line flags:

```powershell
# Run query directly on specific PDFs
python src/pipeline.py --pdfs "data/raw/DBMS.pdf,data/raw/SQL_Reference.pdf" --query "What is BCNF?"

# Skip document processing & ingestion (re-use existing ChromaDB index)
python src/pipeline.py --retrieve-only --query "What is a foreign key?" --top-k 5 --rerank-pool 20

# Force a specific LLM backend explicitly
python src/pipeline.py --retrieve-only --query "Explain ACID properties" --backend groq
```

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--pdfs` | Comma-separated paths to PDFs for ingestion. | Scans `data/raw/` interactively |
| `--query` | Technical question to ask. | Interactive terminal prompt |
| `--top-k` | Number of reranked chunks passed to the LLM. | `5` |
| `--rerank-pool`| Number of RRF candidates passed to the cross-encoder. | `20` |
| `--backend` | LLM provider (`groq`, `openai`, `anthropic`, `ollama`). | Reads `VAULTMIND_LLM_BACKEND` |
| `--retrieve-only` | Skip stages 1–4 (ingestion/embedding/store) and run fast search.| `False` |

### Utility & Diagnostic Scripts

* **`python scripts/diagnose_chunk.py`**: Audits the current `chunks.json` index, prints chunk counts per source, and tests text search patterns to verify indexing integrity.
* **`python scripts/test_heuristic.py`**: Runs unit verification across the 3-signal spacing anomaly detector (`_has_spacing_issue`) to guarantee zero-advance kerning detection.

---

## 📂 Project Structure

```text
vaultmind/
├── app.py                     # Gradio 6.x Two-Screen Web Application
├── requirements.txt           # Project dependencies
├── README.md                  # Documentation (This file)
├── .env                       # Environment variables & API keys
│
├── data/
│   ├── raw/                   # Source PDF textbooks & manuals
│   └── processed/             # Extracted JSON pages, chunks & NumPy embeddings
│       ├── pages.json
│       ├── chunks.json
│       └── embeddings.npy
│
├── src/
│   ├── ingest.py              # PyMuPDF + pdfplumber parsing & kerning recovery
│   ├── chunk.py               # Hierarchical section & semantic text splitter
│   ├── embed.py               # BAAI/bge-small-en-v1.5 embedding generator
│   ├── store.py               # ChromaDB persistent storage & incremental upserts
│   ├── retrieve.py            # Hybrid BM25+Dense RRF & Cross-Encoder reranker
│   ├── generate.py            # Structured JSON schema generation & citations
│   ├── guardrails.py          # Out-of-scope refusal & quality check layer
│   ├── compare.py             # Per-source side-by-side comparison & meta-analysis
│   └── pipeline.py            # End-to-end orchestration CLI & menu workflow
│
├── scripts/
│   ├── diagnose_chunk.py      # Corpus verification & index inspection utility
│   └── test_heuristic.py      # Unit tests for spacing anomaly heuristics
│
├── docs/
│   ├── adr/
│   │   ├── ADR-001-corpus-selection.md
│   │   ├── ADR-002-hybrid-retrieval-over-dense-only-retrieval.md
│   │   └── ADR-003-generation-model-selection.md
│   └── design_doc.md
│
└── vector_store/              # Persistent local ChromaDB database files
```

---

## 🏛️ Architecture Decision Records (ADRs)

Engineering trade-offs and foundational decisions are documented in [`docs/adr/`](file:///c:/Users/Aditya%20Gogoi/Documents/My%20Stuff/Internship/VaultMind/docs/adr):

* **[ADR-001: Corpus Selection](file:///c:/Users/Aditya%20Gogoi/Documents/My%20Stuff/Internship/VaultMind/docs/adr/ADR-001-corpus-selection.md)** — Why focusing on a dense academic/technical domain (like Database Management Systems) allows rigorous evaluation of citation grounding and structural parsing.
* **[ADR-002: Hybrid Retrieval over Dense-Only](file:///c:/Users/Aditya%20Gogoi/Documents/My%20Stuff/Internship/VaultMind/docs/adr/ADR-002-hybrid-retrieval-over-dense-only-retrieval.md)** — Why dense vector embeddings alone fail on exact keyword match, acronyms, and SQL syntax, making BM25 + dense RRF mandatory for technical QA.
* **[ADR-003: Generation Model Selection](file:///c:/Users/Aditya%20Gogoi/Documents/My%20Stuff/Internship/VaultMind/docs/adr/ADR-003-generation-model-selection.md)** — Why **Groq (`llama-3.1-8b-instant`)** and **Ollama (`qwen2.5:7b`)** were chosen over paid commercial models to ensure zero-latency, free-tier sustainability, and structured JSON reliability for student/research evaluation runs.

---

## 👤 Author & License

Developed by **Aditya Gogoi** as part of the **LPU Summer Internship 2026** (Segment: *Foundations of Applied Machine Learning* — Project I2: *Document Q&A / RAG over a Focused Corpus*).