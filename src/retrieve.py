import argparse
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from chunk import CHUNKS_PATH, load_chunks

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_PATH = PROJECT_ROOT / "vector_store"
COLLECTION_NAME = "dbms"
MODEL_NAME = "BAAI/bge-small-en-v1.5"


# -------------------------
# Dense Retrieval
# -------------------------
def dense_search(
    query: str,
    top_k: int = 20,
    chroma_path: str | Path = CHROMA_PATH,
    collection_name: str = COLLECTION_NAME,
):
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(collection_name)

    model = SentenceTransformer(MODEL_NAME)

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
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
        chunk["text"].lower().split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(corpus)

    query_tokens = query.lower().split()

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
                    "page_number": chunk["page_number"],
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
):

    chunks = load_chunks(CHUNKS_PATH)

    dense_results = dense_search(
        query,
        top_k=20,
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
        print(f"Page          : {meta['page_number']}")
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