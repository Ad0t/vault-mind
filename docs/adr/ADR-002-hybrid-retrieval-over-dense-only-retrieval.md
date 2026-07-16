# ADR-002: Hybrid Retrieval over Dense-Only Retrieval


## Context

VaultMind is a Retrieval-Augmented Generation (RAG) system that retrieves relevant document chunks before generating grounded answers with citations.

The retrieval system should:

- Retrieve semantically relevant information even when users phrase questions differently from the source text.
- Handle exact keyword searches for technical terms, equation numbers, acronyms, named entities, and identifiers.
- Maintain high recall across different query styles.
- Produce reliable candidates for answer generation.

Several retrieval approaches were considered:

1. Dense vector retrieval using embeddings only.
2. Sparse lexical retrieval (BM25) only.
3. Hybrid retrieval combining dense embeddings with BM25.

## Decision

Use a hybrid retrieval approach that combines dense vector search with BM25 lexical retrieval.

Dense retrieval captures semantic similarity between the query and document chunks, while BM25 retrieves chunks containing exact keyword matches. The results from both methods are combined using a score fusion strategy before selecting the final documents for answer generation.

## Rationale

- Dense embeddings capture semantic meaning beyond exact wording.
- BM25 performs well for exact technical terminology, equation numbers, acronyms, and named entities.
- Hybrid retrieval covers the failure modes of both approaches by combining semantic and lexical search.
- Improves retrieval quality for technical and educational documents.
- Provides stronger retrieval candidates for downstream RAG generation.

## Consequences

### Positive

- Improved retrieval accuracy for both semantic and keyword-based queries.
- Better handling of technical vocabulary and exact identifiers.
- More robust performance across different user query styles.
- Higher recall than using dense-only or BM25-only retrieval.

### Negative

- Increased implementation complexity.
- Requires maintaining both a vector index and a BM25 index.
- Slightly higher retrieval latency due to combining two retrieval methods.

## Alternatives Considered

### Dense-Only Retrieval

Rejected because semantic embeddings may miss exact keyword matches, equation numbers, acronyms, or named entities that are important in technical documents.

### BM25-Only Retrieval

Rejected because lexical search cannot capture semantic similarity, paraphrases, or conceptually related queries.

### Cross-Encoder Re-ranking Only

Rejected as a standalone solution because it depends on an initial candidate set. Hybrid retrieval produces stronger candidates before any re-ranking stage.

## Future Work

Future versions may incorporate metadata filtering, reciprocal rank fusion (RRF), or cross-encoder re-ranking to further improve retrieval quality while preserving citation accuracy.