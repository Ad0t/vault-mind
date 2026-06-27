# VaultMind

**VaultMind** is a Retrieval-Augmented Generation (RAG) system that enables users to ask questions about a collection of documents and receive accurate answers with page-level citations.

## Problem Statement Code

**I2 — Document Q&A: RAG over a Focused Corpus**

## Segment

**Foundations of Applied Machine Learning**

## Author

**Aditya Gogoi**

---

## Project Overview

VaultMind allows users to interact with documents conversationally. Users can upload or select documents and ask natural language questions. The system retrieves relevant information from the document corpus and generates grounded answers along with citations.

The primary objective is to reduce hallucinations by ensuring that answers are generated only from the retrieved document context.

---

## Features

* PDF document ingestion
* Page-level text extraction
* Intelligent document chunking
* Hybrid retrieval (BM25 + Dense Retrieval)
* Citation-aware answer generation
* Out-of-scope query detection
* Confidence-based refusal mechanism
* Multi-document comparison (mini-extension)

---

## Corpus

The initial version of VaultMind uses a domain-specific corpus consisting of:

* **Corpus:** *To be finalized*
* **Domain:** *Single-domain academic material*
* **Size:** Approximately 100–300 pages

---

## Architecture

```text
PDF Documents
      ↓
PDF Parsing (PyMuPDF)
      ↓
Chunking
      ↓
Embeddings
      ↓
ChromaDB
      ↓
Hybrid Retrieval
(BM25 + Dense + RRF)
      ↓
LLM
      ↓
Answer + Citations
```

## Tech Stack

| Component        | Technology                 |
| ---------------- | -------------------------- |
| Language         | Python                     |
| UI               | Streamlit                  |
| PDF Parsing      | PyMuPDF                    |
| Embeddings       | BGE-small-en-v1.5          |
| Vector Database  | ChromaDB                   |
| Sparse Retrieval | rank-bm25                  |
| Framework        | LangChain                  |
| LLM              | Claude Haiku / GPT-4o-mini |
| Deployment       | Streamlit Community Cloud  |

---

## Project Structure

```text
vaultmind/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── ingest.py
│   ├── chunk.py
│   ├── embed.py
│   ├── retrieve.py
│   ├── generate.py
│   └── guardrails.py
│
├── docs/
│   ├── design_doc.md
│   └── adr/
│
├── eval/
├── tests/
└── vector_store/
```

## What I Learned This Week

* Learned the overall architecture of Retrieval-Augmented Generation (RAG) systems.
* Understood the importance of preserving metadata such as page numbers for citations.
* Explored vector databases and how embeddings are used for semantic search.
* Learned how hybrid retrieval combines keyword-based and semantic search techniques.
* Understood why guardrails are necessary to reduce hallucinations in LLM applications.

---

## Future Enhancements

* Support for additional document formats.
* Advanced semantic chunking.
* Re-ranking retrieved chunks.
* Cross-document comparison and summarization.

---

## License

This project is developed as part of the LPU Summer Internship 2026.

