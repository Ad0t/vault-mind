# VaultMind 🧠📚

> **An End-to-End Hybrid RAG System with Citation Grounding, Multi-Source Comparison & Section-Aware Document Processing.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit UI](https://img.shields.io/badge/UI-Streamlit_1.x-red.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/Vector_Store-ChromaDB-green.svg)](https://trychroma.com/)
[![Groq & Ollama](https://img.shields.io/badge/LLM-Groq_%7C_Ollama-purple.svg)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 2. 🎬 Demo

### Live Web Application
Try the live interactive web application hosted on Streamlit Cloud:
👉 **[Launch VaultMind Live App](https://vault-mind-ysj2lx5petec4frsyax4up.streamlit.app)** 

---

## 3. 🧩 Problem Statement

Traditional Retrieval-Augmented Generation (RAG) systems fail when deployed on dense academic textbooks and technical reference manuals. First, academic PDFs are notorious for **character kerning anomalies**—encoding character glyph advances without explicit whitespace (`WhatisatransactioninDBMS?`), which breaks standard chunkers and destroys keyword search. Second, **dense vector embeddings alone (`cosine similarity`) frequently fail on technical domains**, missing exact keyword queries, SQL syntax (`SELECT ... JOIN`), and specific acronyms (`BCNF`, `2PL`, `ACID`). Finally, when querying across multiple documents (e.g., a 600-page textbook versus a 15-page SQL cheat sheet), standard retrieval pools allow the larger document to crowd out concise, high-relevance chunks from reference manuals.

**VaultMind solves these challenges through a specialized, section-aware engineering pipeline.** It applies a 3-signal heuristic (`_has_spacing_issue`) during ingestion to detect zero-advance character concatenation and automatically reconstructs word boundaries using `pdfplumber` character bounding-box inference. It pairs an in-memory **BM25 Lexical Index** with a **ChromaDB Dense Vector Store (`BAAI/bge-small-en-v1.5`)**, fusing them via **Reciprocal Rank Fusion (RRF)** with **Per-Source Fair Pooling**. By running RRF across each document independently before passing candidates to a `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker, VaultMind guarantees balanced multi-source recall. Every generated claim is strictly bound to verifiable chunks using enforced JSON Schema parsing, displaying exact page numbers, section titles, and expandable source previews right inside the UI.

---

## 4. 🏗️ Architecture Diagram

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
                              │
                              ▼
          Multi-Source Side-by-Side Comparison Table
           & Automated LLM Agreement Meta-Analysis
```

---

## 5. 🛠️ Tech Stack

| Component | Technology / Library | Rationale & Role |
| :--- | :--- | :--- |
| **Interactive UI** | **Streamlit (`streamlit>=1.41.0`)** | Responsive two-screen interface with permanent fixed sidebar, dynamic active-source selection, expandable citations, and side-by-side comparison tables. |
| **PDF Ingestion & OCR** | **PyMuPDF (`fitz`) + `pdfplumber`** | Dual-engine extraction: `fitz` for rapid block heading hierarchies and `pdfplumber` for table bounding-box preservation and kerning anomaly reconstruction. |
| **Section Chunking** | **Custom Hierarchical Chunker (`src/chunk.py`)** | Splits text along semantic heading boundaries (`# Section Header`) while attaching exact page numbers, heading levels, and document provenance to every chunk. |
| **Dense Embeddings** | **`BAAI/bge-small-en-v1.5` (`sentence-transformers`)** | State-of-the-art compact dense representation (384 dimensions) optimized for retrieval benchmarks. |
| **Vector Database** | **ChromaDB (`chromadb`)** | Embedded persistent vector storage (`vector_store/`) supporting deterministic MD5 chunk IDs and zero-copy local search. |
| **Lexical Index** | **`rank-bm25`** | Exact term matching index running in-memory to catch SQL keywords, table attributes, and technical acronyms missed by dense similarity. |
| **Retrieval Fusion** | **Reciprocal Rank Fusion (RRF)** | Fuses lexical and dense ranks ($r_{\text{BM25}}$ and $r_{\text{Dense}}$) across each document independently (`retrieve_per_source`) to prevent document starvation. |
| **Cross-Encoder Reranker**| **`cross-encoder/ms-marco-MiniLM-L-6-v2`** | Passes pooled candidate pairs (`Query + Passage`) through a deep cross-attention transformer to score semantic relevance (`rerank_score`). |
| **Safety Guardrails** | **Custom Heuristic & Score Thresholds (`guardrails.py`)** | Rejects out-of-domain queries (`"How do I bake a cake?"`) or low-confidence retrieval pools before calling the generation API. |
| **Structured LLM Engine** | **Groq (`llama-3.1-8b-instant`) / Ollama (`qwen2.5:7b`)** | Free-tier cloud generation (`Groq` at 800+ tokens/sec) and offline local execution (`Ollama`), enforcing strict JSON Schema output for exact `[n]` citations. |

---

## 6. ⚡ Quickstart Guide

### Prerequisites
* **Python:** Version `3.10` or higher (`python --version`).
* **Git:** For cloning the repository.
* **API Key (Optional):** A free **Groq API Key** (`gsk_...`) for cloud inference, or a local **Ollama** installation (`qwen2.5:7b`) for 100% offline execution.

### Install
1. **Clone the repository and enter the directory:**
   ```powershell
   git clone https://github.com/Ad0t/VaultMind.git
   cd VaultMind
   ```

2. **Create and activate an isolated virtual environment:**
   ```powershell
   # Windows PowerShell
   python -m venv .venv
   .venv\Scripts\activate

   # macOS / Linux
   # python3 -m venv .venv && source .venv/bin/activate
   ```

3. **Install project dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the project root containing your preferred backend configuration:
   ```env
   # Default Backend: Groq (Free Tier — 14,400 requests/day)
   VAULTMIND_LLM_BACKEND=groq
   GROQ_API_KEY=gsk_your_groq_api_key_here

   # Optional Alternatives:
   # VAULTMIND_LLM_BACKEND=ollama
   # VAULTMIND_LLM_BACKEND=openai
   # OPENAI_API_KEY=sk-...
   ```

### Run
Launch the application via either the **Streamlit Web UI** or the **End-to-End CLI Pipeline**:

#### Option A: Streamlit Web UI (Recommended)
```powershell
streamlit run app.py
```
* Open your browser to **`http://localhost:8501`**.
* Upload your PDFs in the sidebar, click **⚡ Process & Start Chat**, and ask complex technical questions with instant inline citations!

#### Option B: Interactive CLI Pipeline
```powershell
python src/pipeline.py
```
* Or run automated batch queries directly from your terminal:
  ```powershell
  python src/pipeline.py --query "What are the four ACID properties in a DBMS?" --top-k 5
  ```

### Test
Run our utility and verification scripts to confirm system integrity:

```powershell
# 1. Audit corpus indexing, verify chunk counts per document, and test search patterns
python scripts/diagnose_chunk.py

# 2. Run unit verification across the 3-signal spacing anomaly detector (kerning recovery)
python scripts/test_heuristic.py
```

---

## 7. 📚 Data Sources

VaultMind is designed and evaluated on a focused **Database Management Systems (DBMS)** domain corpus located inside `data/raw/`:
* **Academic Textbooks:** Comprehensive university textbooks (e.g., *Database Management Systems* by Ramakrishnan & Gehrke / Korth).
* **Technical Reference Manuals:** SQL grammar sheets, index architecture specifications, and query optimization manuals.
* **Lecture Notes:** Condensed university slide decks covering normalization (`1NF` → `BCNF`), concurrency control (`Two-Phase Locking`), and crash recovery (`ARIES`).

**Why this corpus?** Technical textbooks present rigorous stress-tests for RAG systems. They contain nested hierarchies, complex multi-column tables, mathematical formulas, and heavy acronym density. By curating a domain-specific corpus, VaultMind allows precise, repeatable verification of citation grounding and section-aware boundary extraction.

---

## 8. 🏛️ Architecture Decision Records (ADRs)

Engineering trade-offs and foundational decisions are documented in detail inside [`docs/adr/`](file:///c:/Users/Aditya%20Gogoi/Documents/My%20Stuff/Internship/VaultMind/docs/adr):

* **[ADR-001: Corpus Selection](file:///c:/Users/Aditya%20Gogoi/Documents/My%20Stuff/Internship/VaultMind/docs/adr/ADR-001-corpus-selection.md)** — Explains why targeting a dense, technical domain (DBMS) rather than generic Wikipedia articles forces rigorous solutions for table preservation, acronym resolution, and section-level metadata tracking.
* **[ADR-002: Hybrid Retrieval over Dense-Only Retrieval](file:///c:/Users/Aditya%20Gogoi/Documents/My%20Stuff/Internship/VaultMind/docs/adr/ADR-002-hybrid-retrieval-over-dense-only-retrieval.md)** — Documents why dense embeddings (`bge-small`) alone fail on exact SQL keywords and numerical constraint lookups, making in-memory BM25 lexical recall + Reciprocal Rank Fusion mandatory.
* **[ADR-003: Generation Model Selection](file:///c:/Users/Aditya%20Gogoi/Documents/My%20Stuff/Internship/VaultMind/docs/adr/ADR-003-generation-model-selection.md)** — Details the decision to prioritize **Groq (`llama-3.1-8b-instant`)** and **Ollama (`qwen2.5:7b`)** over expensive commercial APIs to guarantee zero-latency execution, sustainability, and strict JSON Schema compliance without recurring cloud costs.

---

## 9. 🚀 Mini-Extension: Multi-Source Comparison & Agreement Meta-Analysis

### What is it?
When a user asks a question whose retrieved chunks span two or more distinct documents (for example, retrieving excerpts from both a general *DBMS Textbook* and a specific *Oracle SQL Reference*), VaultMind unlocks a specialized **Compare Sources** capability:
1. **Side-by-Side Comparison Table:** Isolates retrieved chunks by their parent document and generates independent, standalone answers for each source directly side-by-side.
2. **Automated LLM Agreement Meta-Analysis:** Triggers a synthesis pass (`_analyze_agreement`) that evaluates all per-source answers to produce:
   * **Consensus:** Core principles and definitions all sources agree upon.
   * **Unique Contributions:** Specific examples, syntax, or depth unique to individual documents.
   * **Contradictions & Nuances:** Technical discrepancies across different textbook editions or vendor implementations.

### Why did we build this?
In real-world engineering and academic research, no single document holds the absolute truth. Standard RAG systems mash chunks from multiple files into one monolithic prompt, often causing the LLM to hallucinate compromises or ignore subtle discrepancies across sources. Our mini-extension treats each document as an independent authority, giving users clear comparative visibility into how different references tackle the exact same technical problem.

---

## 10. ⚠️ Known Limitations

1. **Scanned / Image-Only PDFs:** VaultMind currently ingests documents with embedded text layers via `PyMuPDF` and `pdfplumber`. PDFs consisting purely of scanned image bitmaps require pre-processing with OCR (`Tesseract` or `EasyOCR`) before ingestion.
2. **Cross-Page Table Splits:** If a complex, multi-page data table splits across a hard page boundary without repeating table headers, the chunker may divide rows across two separate section chunks.
3. **Extreme Multi-Hop Reasoning:** When a technical query requires synthesizing facts across 4+ disconnected chapters (e.g., tracing an exact disk block write all the way from SQL parsing to hardware RAID parity), fixed `top-k` candidate pools can miss intermediate conceptual bridges.

---

## 11. 🔮 What I'd Do in 3rd Year (Research & Engineering Roadmap)

To transition VaultMind into an enterprise-grade research platform during **3rd Year B.Tech**, we have outlined five major architectural milestones:

1. **GraphRAG & Knowledge Graph Integration:** Extracting concepts into a property graph (`Neo4j` / `NetworkX`) to enable graph-constrained multi-hop reasoning across chapter boundaries.
2. **Agentic SQL Query & Sandbox Execution:** Equipping the LLM with tool-calling capabilities (`run_sql_query`) against an embedded SQLite/Postgres sandbox to verify generated SQL code execution plans live.
3. **Multi-Modal Vision LLM Indexing:** Using Vision-Language models (`ColPali` / `LLaVA`) to directly index visual flowcharts, B+ Tree split diagrams, and ER schemas.
4. **Comprehensive Benchmark Suite (`Ragas` & `DeepEval`):** Formalizing evaluation rigor against a 100+ golden Q&A dataset to correlate automated LLM-as-a-judge scores with human expert evaluations.
5. **Domain-Specific Cross-Encoder Fine-Tuning:** Fine-tuning compact reranking models specifically on contrastive DBMS and SQL query-passage pairs.

👉 **Read the complete, detailed 3rd Year Technical Roadmap here:** **[`docs/roadmap/3rd_year_roadmap.md`](file:///c:/Users/Aditya%20Gogoi/Documents/My%20Stuff/Internship/VaultMind/docs/roadmap/3rd_year_roadmap.md)**

---

## 12. 📄 License & Acknowledgements

### Author & Project Context
Developed by **Aditya Gogoi** as part of the **LPU Summer Internship 2026** under the segment **Foundations of Applied Machine Learning** (Project I2: *Document Q&A / RAG over a Focused Corpus*).

### License
This project is open-source and released under the **[MIT License](https://opensource.org/licenses/MIT)**. You are free to use, modify, and distribute this software for educational, research, and commercial purposes.

### Acknowledgements & Open-Source Credits
VaultMind stands on the shoulders of incredible open-source tools and models:
* **[Streamlit](https://streamlit.io/)** for the responsive web application framework.
* **[ChromaDB](https://trychroma.com/)** for embedded, zero-config vector storage.
* **[Hugging Face & Sentence-Transformers](https://huggingface.co/)** for `bge-small-en-v1.5` dense embeddings and `ms-marco-MiniLM-L-6-v2` cross-encoder rerankers.
* **[PyMuPDF (`fitz`) & pdfplumber](https://pymupdf.readthedocs.io/)** for precision PDF block extraction and table preservation.
* **[Groq & Ollama](https://groq.com/)** for ultra-fast cloud inference and offline local LLM execution.