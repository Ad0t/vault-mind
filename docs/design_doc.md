# VaultMind — Initial Design Document

## Problem Statement

**I2 — Document Q&A: RAG over a Focused Corpus**

---

## Project Goal

VaultMind aims to build a Retrieval-Augmented Generation (RAG) system that enables users to query a collection of documents using natural language and receive grounded responses with page-level citations.

The system will ensure that answers are generated only from the document corpus and will refuse to answer questions that fall outside the scope of the available documents.

---

## Target Users

* Students studying academic subjects.
* Learners searching through textbooks and notes.
* Researchers working with a small collection of research papers.

---

## Corpus

The first version of VaultMind will use a single-domain corpus.

**Selected Corpus:** *To be finalized*

**Expected Size:** 100–300 pages

**Reason for Selection:**

* Single-domain content improves retrieval quality.
* Moderate corpus size keeps the project manageable.
* Documents contain headings and page structure that support citation generation.

---

## Functional Requirements

### Core Features

1. Ingest PDF documents.
2. Extract text while preserving page numbers.
3. Chunk extracted text into smaller semantic units.
4. Generate embeddings for all chunks.
5. Store embeddings in ChromaDB.
6. Retrieve relevant chunks using hybrid retrieval.
7. Generate answers using an LLM.
8. Provide page-level citations.
9. Refuse out-of-scope questions.

### Mini-Extension

Compare two documents and generate a structured comparison with citations.

---

## Proposed Architecture

```text
PDF Documents
       ↓
PyMuPDF Extraction
       ↓
Chunking Layer
       ↓
Embedding Generation
       ↓
ChromaDB Storage
       ↓
Hybrid Retrieval
(BM25 + Dense + RRF)
       ↓
Guardrail Layer
       ↓
LLM Generation
       ↓
Answer + Citations
```

---

## Tech Stack

| Component            | Choice                       | Reason                              |
| -------------------- | ---------------------------- | ----------------------------------- |
| Programming Language | Python                       | Rich AI ecosystem                   |
| PDF Parsing          | PyMuPDF                      | Fast and provides page metadata     |
| Chunking             | LangChain Recursive Splitter | Simple and effective                |
| Embeddings           | BGE-small-en-v1.5            | Open-source and CPU-friendly        |
| Vector Database      | ChromaDB                     | Lightweight and easy to deploy      |
| Sparse Retrieval     | rank-bm25                    | Efficient keyword search            |
| Retrieval Fusion     | Reciprocal Rank Fusion       | Combines dense and sparse retrieval |
| LLM                  | Claude Haiku / GPT-4o-mini   | Low latency and cost                |
| UI                   | Streamlit                    | Rapid application development       |
| Deployment           | Streamlit Community Cloud    | Free and easy deployment            |

---

## Success Criteria

The project will be considered successful if:

* Users can ask questions and receive accurate answers.
* Every answer contains valid citations.
* The system correctly refuses out-of-scope questions.
* The system supports comparison between two documents.

---

## Risks

* Poor chunking may reduce retrieval quality.
* Citation hallucination by the LLM.
* Free-tier deployment limitations.
* Variability in PDF formatting.

---

## Timeline

### Week 1

Document ingestion and data layer.

### Week 2

Retrieval pipeline and answer generation.

### Week 3

Guardrails, evaluation, and mini-extension.

### Week 4

Deployment, testing, and documentation.
