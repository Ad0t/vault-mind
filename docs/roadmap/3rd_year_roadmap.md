# VaultMind — 3rd Year Technical & Research Roadmap 🚀🧠

This document outlines the advanced architectural extensions, research methodologies, and engineering milestones planned for **VaultMind** during the **3rd Year of B.Tech (Computer Science / Applied Machine Learning)**.

---

## 1. GraphRAG & Knowledge Graph Integration
While our current **Hybrid RRF (BM25 + Dense BGE)** pipeline excels at targeted section retrieval and keyword matching, multi-hop reasoning across chapters (e.g., connecting a high-level *ACID definition* in Chapter 12 to a low-level *Write-Ahead Log recovery protocol* in Chapter 18) remains challenging within a fixed `top-k` window.

### Planned Implementation:
* **Entity & Relation Extraction:** Use LLM-powered extraction to build a **property graph** of DBMS concepts (`Transaction` *enforces* `Isolation`, `BCNF` *is a stricter form of* `3NF`, `Index` *improves* `Range Query`).
* **Graph Vector Storage:** Store entities and relationship triples in a graph-aware database (e.g., **Neo4j** or **NetworkX + ChromaDB hybrid**).
* **Graph-Constrained Traversal:** When a query requires cross-chapter synthesis, traverse adjacent graph nodes to expand the context dynamically before passing chunks to the cross-encoder.

---

## 2. Agentic SQL Query & Sandbox Execution Engine
For a Database Management Systems (DBMS) assistant, answering questions from text is only half the battle. Users frequently ask about SQL queries, schema normalizations, and transaction locks.

### Planned Implementation:
* **Embedded SQLite/PostgreSQL Sandbox:** Spin up isolated, in-memory SQL databases seeded with sample educational schemas (`Bank`, `University`, `E-Commerce`).
* **Tool-Calling Agent (`LangGraph` / `Function Calling`):** Equip the LLM with executable tools (`run_sql_query`, `explain_query_plan`, `test_concurrency`).
* **Live Query Verification:** When a user asks *"Write a SQL query to find students with GPA above 3.8 and show the execution plan"*, VaultMind will generate the query, execute it inside the sandbox, verify zero syntax errors, and return both the output and the `EXPLAIN QUERY PLAN` metrics alongside textual textbook citations.

---

## 3. Multi-Modal Vision LLM for Complex Tables & Architecture Diagrams
Current PDF ingestion (`PyMuPDF` + `pdfplumber`) extracts tabular text and block headers, but struggles with visual flowcharts (e.g., B+ Tree node splits, ER Diagrams, or Serializability Dependency Graphs).

### Planned Implementation:
* **Visual Bounding Box Extraction:** Render pages containing complex diagrams or mathematical tables as high-resolution PNG images during ingestion.
* **Vision-Language Indexing (ColPali / GPT-4o-mini / LLaVA):** Embed visual patches directly or use a Vision LLM to generate dense structural summaries (`"Diagram depicting a 3-level B+ Tree where root node [15, 30] splits into leaf nodes..."`) and index them into ChromaDB.

---

## 4. Comprehensive Evaluation & Benchmark Suite (`LLM-as-a-Judge` & Human Scoring)
To transition VaultMind from an engineering prototype to a publication-ready system, we will formalize our evaluation rigor.

### Planned Implementation:
* **Curated Golden Dataset:** Construct a rigorous evaluation benchmark comprising 100+ multi-difficulty DBMS questions paired with exact ground-truth answer strings and required page/chunk references.
* **Automated Metrics (`Ragas` & `DeepEval`):**
  * **Faithfulness / Hallucination Rate:** Measuring whether every claim in the generated output can be logically entailed by the retrieved excerpts.
  * **Answer Relevance & Context Precision:** Quantifying how well the hybrid RRF engine filters out noise.
  * **Citation Accuracy Score:** Verifying that inline `[n]` citations point to the exact page where the fact is stated.
* **Human-in-the-Loop vs. LLM Judge Correlation:** Compare automated `Llama-3.3-70B` / `GPT-4o` evaluation scores against domain-expert human grading to validate metric reliability.

---

## 5. Domain-Specific Cross-Encoder Fine-Tuning
Our current reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is trained on general web search queries (MS-MARCO).

### Planned Implementation:
* **Contrastive Pair Generation:** Mine our DBMS textbook chunks to create positive/hard-negative technical pairs (`Query: "What is strict schedule?"`, `Positive: "A schedule where no transaction reads/writes until..."`, `Hard Negative: "A serializable schedule where transactions execute sequentially..."`).
* **Model Fine-Tuning:** Fine-tune a compact cross-encoder (`bge-reranker-small` or `MiniLM`) explicitly on DBMS and SQL semantics, achieving superior ranking precision with sub-50ms CPU inference time.
