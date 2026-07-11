# VaultMind

**VaultMind** is a Retrieval-Augmented Generation (RAG) system that enables users to ask questions about technical documents and receive grounded answers with page-level citations. The project focuses on building a production-style RAG pipeline from scratch while understanding every stage of document processing, retrieval, and generation.

## Problem Statement Code

**I2 — Document Q&A: RAG over a Focused Corpus**

## Segment

**Foundations of Applied Machine Learning**

## Author

**Aditya Gogoi**

---

# Project Overview

VaultMind converts PDF documents into a searchable knowledge base by extracting structured content, generating semantic embeddings, storing them in a vector database, and retrieving relevant document chunks during question answering.

The project emphasizes:

- Structured document parsing
- Metadata preservation
- Explainable retrieval
- Citation-grounded answers
- Reduced hallucinations

The initial version uses a single-domain academic corpus (DBMS textbook) to simplify evaluation and improve retrieval quality.

---

# Current Features

## Document Processing

- PDF text extraction using **PyMuPDF**
- Automatic heading detection
- Section-aware parsing
- Table extraction using **pdfplumber**
- Structured document representation

## Chunking

- Hierarchical chunking based on document sections
- Recursive semantic chunk splitting
- Configurable chunk size and overlap
- Metadata preservation:
  - Page number
  - Section title
  - Heading level
  - Source document
  - Chunk index

## Embeddings

- Semantic embeddings using **BAAI/bge-small-en-v1.5**
- Batch embedding generation
- NumPy embedding storage

## Vector Database

- ChromaDB vector store
- Persistent local database
- Metadata-aware storage

## Retrieval *(In Progress)*

- Dense vector retrieval
- Hybrid retrieval design (BM25 + Dense Retrieval)
- Metadata filtering (planned)
- Reciprocal Rank Fusion (planned)

## Answer Generation *(Planned)*

- Citation-aware responses
- Confidence-based refusal
- Out-of-scope detection

---

# Corpus

VaultMind v1 uses a single academic textbook.

| Property | Value |
|----------|-------|
| Domain | Database Management Systems |
| Corpus | Single textbook |
| Size | ~150–300 pages |
| Structure | Chapters, sections, headings, tables |

---

# Architecture

```text
                  PDF
                   │
        ┌──────────▼──────────┐
        │   Document Parsing  │
        │ PyMuPDF + pdfplumber│
        └──────────┬──────────┘
                   │
          Structured Pages
                   │
        Heading Detection
                   │
             Section Builder
                   │
     Recursive Chunking
                   │
     Metadata Preservation
                   │
      BGE Embeddings
                   │
      ChromaDB Storage
                   │
      Dense Retrieval
                   │
   Hybrid Retrieval (Planned)
                   │
      LLM Generation
                   │
      Answer + Citations
```

---

# Tech Stack

| Component | Technology |
|------------|------------|
| Language | Python |
| PDF Parsing | PyMuPDF |
| Table Extraction | pdfplumber |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Embedding Framework | SentenceTransformers |
| Vector Database | ChromaDB |
| Retrieval | Dense Retrieval |
| Hybrid Retrieval | BM25 + Dense (Planned) |
| Metadata | JSON |
| LLM | GPT-4o-mini / Claude Haiku *(Planned)* |
| UI | Streamlit *(Planned)* |

---

# Project Structure

```text
vaultmind/
│
├── app.py                    # Streamlit application (planned)
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── pages.json
│       ├── chunks.json
│       └── embeddings.npy
│
├── src/
│   ├── ingest.py
│   ├── chunk.py
│   ├── embed.py
│   ├── store.py
│   ├── retrieve.py
│   ├── generate.py
│   └── guardrails.py
│
├── docs/
│   ├── adr/
│   ├── ADR-001.md
│   ├── ADR-002.md
│   └── design_doc.md
│
├── vector_store/
├── tests/
└── eval/
```

---

# Current Pipeline

```
PDF
 ↓
Extract Text
 ↓
Detect Headings
 ↓
Extract Tables
 ↓
Build Sections
 ↓
Semantic Chunking
 ↓
Generate Embeddings
 ↓
Store in ChromaDB
 ↓
Retrieve Relevant Chunks
 ↓
Generate Answer
```

---

# What I Learned

## Week 1

- Learned the complete architecture of Retrieval-Augmented Generation systems.
- Understood how document parsing impacts retrieval quality.
- Explored semantic embeddings and vector databases.
- Learned how metadata enables citation-aware generation.
- Understood the importance of preserving document hierarchy.

## Week 2

- Implemented heading-aware document parsing.
- Learned why chunk boundaries should align with document structure.
- Added table extraction using pdfplumber.
- Understood why tables should remain intact during chunking.
- Explored BGE embeddings and embedding generation.
- Built a ChromaDB vector store.
- Learned how Approximate Nearest Neighbor (ANN) search enables efficient retrieval.
- Studied dense retrieval and why hybrid retrieval improves recall.
- Explored metadata filtering for improving retrieval precision.
- Learned the role of Architecture Decision Records (ADRs) in documenting engineering decisions.

---

# What Surprised Me

Semantic search alone is not always sufficient for technical documents. Exact keyword searches, equation numbers, acronyms, and named entities can still be missed by dense embeddings, which is why combining BM25 with dense retrieval is considered a production best practice. I also learned that preserving document structure and metadata has a significant impact on retrieval quality and citation accuracy.

---

# Future Enhancements

- Hybrid retrieval (BM25 + Dense)
- Reciprocal Rank Fusion (RRF)
- Metadata filtering
- Cross-encoder re-ranking
- Scanned PDF OCR support
- Image extraction and indexing
- Multi-document retrieval
- Conversation memory
- Evaluation pipeline
- Streamlit deployment

---

# Architecture Decisions

- **ADR-001:** Corpus Selection for VaultMind
- **ADR-002:** Why Hybrid Retrieval over Dense-Only Retrieval

---

# License

This project is developed as part of the **LPU Summer Internship 2026**.