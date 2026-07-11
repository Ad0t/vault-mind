import argparse
import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from chunk import CHUNKS_PATH, load_chunks

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_PATH = PROJECT_ROOT / "vector_store"
COLLECTION_NAME = "dbms"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Common English stopwords to exclude from BM25 tokenization. Stopwords add
# noise to lexical matching because they appear in nearly every chunk, so
# removing them keeps BM25 scores focused on meaningful query terms.
BM25_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "and", "or", "but", "if", "then", "else", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "to", "from", "in", "on", "off", "over", "under", "again", "further",
    "do", "does", "did", "doing", "how", "when", "where", "why",
    "can", "will", "just", "should", "now", "it", "its", "as", "so",
}


def tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25 matching.

    Extracts word tokens with a regex (so punctuation like the trailing
    "?" in "normalization?" doesn't get glued onto the word), lowercases
    them, and drops common stopwords that would otherwise dilute lexical
    matching with noise (e.g. "is", "what", "the").
    """
    words = re.findall(r"\b\w+\b", text.lower())
    return [word for word in words if word not in BM25_STOPWORDS]


# -------------------------
# Dense Retrieval
# -------------------------
def dense_search(
    query: str,
    top_k: int = 20,
    chroma_path: str | Path = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
    metadata_filter: dict | None = None,
):
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(collection_name)

    model = SentenceTransformer(MODEL_NAME)

    query_embedding = model.encode(query).tolist()

    # Metadata filtering (optional):
    # Chroma's `where` parameter restricts the search to documents whose
    # metadata matches the given filter, e.g. {"source_doc": "some.pdf"}.
    # Passing `where=None` (the default when metadata_filter is None)
    # disables filtering entirely, so existing calls without a filter
    # behave exactly as before.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=metadata_filter if metadata_filter else None,
    )

    ranking = []

    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    for doc_id, doc, meta in zip(ids, docs, metas):
        ranking.append(
            {
                "id": doc_id,
                "text": doc,
                "metadata": meta,
            }
        )

    return ranking


# -------------------------
# BM25 Retrieval
# -------------------------
def bm25_search(
    query: str,
    chunks: list[dict],
    top_k: int = 20,
):
    corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(corpus)

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    ranking = []

    for idx in ranked_indices:
        chunk = chunks[idx]

        ranking.append(
            {
                "id": chunk["chunk_id"],
                "text": chunk["text"],
                "metadata": {
                    "start_page": chunk["start_page"],
                    "end_page": chunk["end_page"],
                    "section_title": chunk["section_title"],
                    "source_doc": chunk["source_doc"],
                    "heading_level": chunk["heading_level"],
                    "chunk_index": chunk["chunk_index"],
                },
            }
        )

    return ranking


# -------------------------
# Reciprocal Rank Fusion
# -------------------------
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    documents = {}

    for ranking in rankings:

        for rank, item in enumerate(ranking):

            doc_id = item["id"]

            documents[doc_id] = item

            scores.setdefault(doc_id, 0)

            scores[doc_id] += 1 / (k + rank + 1)

    fused = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    results = []

    for doc_id, score in fused:
        doc = documents[doc_id]
        doc["rrf_score"] = score
        results.append(doc)

    return results


# -------------------------
# Hybrid Retrieval
# -------------------------
def retrieve(
    query: str,
    top_k: int = 5,
    metadata_filter: dict | None = None,
):

    chunks = load_chunks(CHUNKS_PATH)

    # metadata_filter is only forwarded to dense_search(). BM25 operates
    # over the local in-memory chunk corpus (not Chroma), so metadata
    # filtering has no equivalent there and is intentionally left unchanged.
    dense_results = dense_search(
        query,
        top_k=20,
        metadata_filter=metadata_filter,
    )

    bm25_results = bm25_search(
        query,
        chunks,
        top_k=20,
    )

    fused = reciprocal_rank_fusion(
        [
            dense_results,
            bm25_results,
        ]
    )

    return fused[:top_k]


# -------------------------
# Pretty Printing
# -------------------------
def print_results(results, query):

    print("\n" + "=" * 80)
    print("QUERY")
    print("=" * 80)
    print(query)

    print("\nTOP RESULTS\n")

    for i, result in enumerate(results, start=1):

        meta = result["metadata"]

        print("=" * 80)
        print(f"Rank          : {i}")
        print(f"Chunk ID      : {result['id']}")
        start_page = meta["start_page"]
        end_page = meta["end_page"]
        page_label = (
            str(start_page)
            if start_page == end_page
            else f"{start_page}-{end_page}"
        )
        print(f"Page          : {page_label}")
        print(f"Section       : {meta['section_title']}")
        print(f"RRF Score     : {result['rrf_score']:.5f}")
        print("-" * 80)
        print(result["text"][:500])
        print()


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Hybrid BM25 + Dense Retriever"
    )

    parser.add_argument(
        "--query",
        type=str,
        default="What is normalization?",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    results = retrieve(
        args.query,
        args.top_k,
    )

    print_results(
        results,
        args.query,
    )

    # -------------------------
    # Metadata filtering examples
    # -------------------------
    # metadata_filter restricts dense retrieval to chunks whose Chroma
    # metadata matches the given key/value pair(s). BM25 results are
    # unaffected since BM25 runs over the full local chunk corpus.

    # Filter by source document:
    # results = retrieve(
    #     query="Explain normalization",
    #     top_k=5,
    #     metadata_filter={
    #         "source_doc": "Database Management Systems.pdf"
    #     },
    # )

    # Filter by start page:
    # results = retrieve(
    #     query="Explain transactions",
    #     top_k=5,
    #     metadata_filter={
    #         "start_page": 42
    #     },
    # )

    # Filter by section title:
    # results = retrieve(
    #     query="Explain BCNF",
    #     top_k=5,
    #     metadata_filter={
    #         "section_title": "Normalization"
    #     },
    # )

    # Filter by heading level:
    # results = retrieve(
    #     query="Explain indexing",
    #     top_k=5,
    #     metadata_filter={
    #         "heading_level": 2
    #     },
    # )