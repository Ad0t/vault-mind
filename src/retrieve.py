import argparse
import re
import sys
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from chunk import CHUNKS_PATH, load_chunks

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHROMA_PATH = PROJECT_ROOT / "vector_store"
COLLECTION_NAME = "dbms"
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Cross-encoder used for the optional reranking stage. A cross-encoder reads
# the query and the passage *together* in a single forward pass, so it can
# model their interaction directly -- unlike bi-encoders (BGE) which embed
# them independently. This makes cross-encoders much more accurate at judging
# relevance, but also slower: they run N inference calls (one per candidate)
# rather than a single query embedding, which is why they are applied only on
# the small RRF candidate pool rather than the full corpus.
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Common English stopwords to exclude from BM25 tokenization. Stopwords add
# noise to lexical matching because they appear in nearly every chunk, so
# removing them keeps BM25 scores focused on meaningful query terms.
#
# NOTE: Wh-question words (what, which, who, how, when, where, why) are
# intentionally kept OUT of this list. Removing them collapses queries like
# "What is normalization?" down to a single token ("normalization"), which
# causes BM25 to rank purely on term frequency and lets short, dense chunks
# beat the actual definitional chunk. Keeping wh-words means short queries
# still carry enough signal to distinguish definitional vs. feature-listing
# content.
BM25_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those",
    "and", "or", "but", "if", "then", "else", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during",
    "to", "from", "in", "on", "off", "over", "under", "again", "further",
    "do", "does", "did", "doing",
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

    # BAAI/bge-small-en-v1.5 is an asymmetric retrieval model: documents are
    # indexed as-is (done correctly in embed.py), but *queries* must be
    # prefixed with this instruction so they land in the same region of the
    # embedding space as the passage vectors. Without the prefix the query
    # vector drifts into a different subspace and cosine similarity becomes
    # unreliable, causing relevant definitional chunks to be outranked by
    # loosely-related ones.
    BGE_QUERY_INSTRUCTION = (
        "Represent this sentence for searching relevant passages: "
    )
    query_embedding = model.encode(BGE_QUERY_INSTRUCTION + query).tolist()

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
    # Index section_title + text so that a chunk whose heading is
    # "9.Introduction to Database Normalization" scores higher for
    # "normalization" queries than a chunk whose body merely mentions it
    # once in passing. Without this, BM25 sees only the body text and
    # systematically under-ranks definitional/intro chunks whose key
    # concept is explicit in their heading but diluted in their body.
    corpus = [
        tokenize(chunk["section_title"] + " " + chunk["text"])
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
# Cross-Encoder Reranker
# -------------------------
def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    model_name: str = RERANKER_MODEL_NAME,
) -> list[dict]:
    """
    Re-rank RRF candidates using a cross-encoder model.

    A cross-encoder reads (query, passage) pairs *together* and outputs a
    relevance logit, which is far more accurate than the cosine similarity
    between independently-encoded vectors used by dense retrieval. The
    trade-off is speed: one forward pass per candidate, so this is applied
    on the small RRF pool (e.g. top-20) rather than the full corpus.

    The rerank_score stored on each result is the raw cross-encoder logit.
    Higher is always more relevant (no normalisation needed for ranking).
    """
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name)

    pairs = [(query, candidate["text"]) for candidate in candidates]
    scores = model.predict(pairs)

    scored = [
        {**candidate, "rerank_score": float(score)}
        for candidate, score in zip(candidates, scores)
    ]

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)

    return scored[:top_k]


# -------------------------
# Hybrid Retrieval
# -------------------------
def retrieve(
    query: str,
    top_k: int = 5,
    metadata_filter: dict | None = None,
    rerank_pool_size: int | None = None,
    source_filter: list[str] | None = None,
) -> list[dict]:
    """
    Run hybrid BM25 + dense retrieval fused with RRF.

    rerank_pool_size: when provided, the function returns this many RRF
                      candidates instead of top_k. Use this when the caller
                      will pass the results through rerank() afterwards --
                      the reranker needs a wider candidate pool to choose from
                      so it can surface the correct definitional chunk even if
                      RRF ranked it outside the default top-k.

    source_filter: optional list of source_doc filenames (e.g. ["DBMS.pdf"]).
                   When set, both the BM25 corpus and the ChromaDB dense
                   search are restricted to chunks from those sources only.
                   Supports selecting/deselecting PDFs in the UI without
                   re-embedding anything.
    """

    chunks = load_chunks(CHUNKS_PATH)

    # Apply source filter to BM25 corpus (operates on local chunks list)
    if source_filter:
        chunks = [c for c in chunks if c.get("source_doc") in source_filter]

        # Build the ChromaDB where clause for dense search
        # Use $in for multiple sources; equality for a single source
        # (ChromaDB supports both but $in is consistent either way)
        if metadata_filter is None:
            metadata_filter = (
                {"source_doc": {"$in": list(source_filter)}}
                if len(source_filter) > 1
                else {"source_doc": source_filter[0]}
            )

    # metadata_filter is only forwarded to dense_search(). BM25 operates
    # over the local in-memory chunk corpus (not Chroma), so metadata
    # filtering has no equivalent there and is intentionally left unchanged.
    dense_results = dense_search(
        query,
        top_k=20,
        metadata_filter=metadata_filter,
    )

    # Widen the BM25 candidate pool so that chunks ranked 21-30 by
    # BM25 still enter the RRF fusion (they may rank highly in dense).
    bm25_results = bm25_search(
        query,
        chunks,
        top_k=30,
    )

    fused = reciprocal_rank_fusion(
        [
            dense_results,
            bm25_results,
        ]
    )

    # Return a wider pool when the caller will rerank, otherwise cut at top_k.
    n = rerank_pool_size if rerank_pool_size is not None else top_k
    return fused[:n]



def retrieve_per_source(
    query: str,
    top_k: int = 5,
    candidates_per_source: int = 5,
    rerank_pool_size_per_source: int = 20,
    active_sources: list[str] | None = None,
) -> list[dict]:
    """
    Run BM25 + dense + RRF retrieval independently for each source PDF,
    then merge all candidate pools into one list for global reranking.

    Why this matters
    ----------------
    When multiple PDFs are loaded, a large document can monopolise every
    top-k slot because it contributes proportionally more chunks to both
    the BM25 corpus and the ChromaDB index.  Per-source retrieval fixes
    this by giving every PDF an equal share of the candidate pool before
    the cross-encoder reranker decides the final ordering.

    Parameters
    ----------
    query                       : the user's question
    top_k                       : desired final result count (passed to
                                  rerank() by the caller — not applied here)
    candidates_per_source       : candidates collected from each PDF
    rerank_pool_size_per_source : RRF pool size for each per-source search
    active_sources              : restrict to these source_doc names only;
                                  None means use all sources in chunks.json
    """
    chunks  = load_chunks(CHUNKS_PATH)
    sources = sorted({c.get("source_doc", "") for c in chunks} - {""})

    # Honour the active-source filter from the UI
    if active_sources:
        sources = [s for s in sources if s in active_sources]

    if not sources:
        return []

    # Single source — plain retrieve is equivalent and cheaper
    if len(sources) == 1:
        return retrieve(
            query,
            top_k=top_k,
            rerank_pool_size=rerank_pool_size_per_source,
            source_filter=sources,
        )

    merged:   list[dict] = []
    seen_ids: set[str]   = set()

    for source in sources:
        per_source = retrieve(
            query,
            top_k=candidates_per_source,
            rerank_pool_size=rerank_pool_size_per_source,
            source_filter=[source],
        )
        for chunk in per_source:
            if chunk["id"] not in seen_ids:
                seen_ids.add(chunk["id"])
                merged.append(chunk)

    return merged


# -------------------------
# Pretty Printing
# -------------------------
def print_results(results, query, preview_chars: int | None = None):
    """
    Pretty-print retrieved results.

    preview_chars: maximum characters of chunk text to display.
                   None (the default) prints the full chunk text.
                   Pass an integer (e.g. 500) for a shorter preview;
                   a trailing '...' is added so it's clear the text
                   continues beyond what is shown.
    """

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
        print(f"Source        : {meta['source_doc']}")
        print(f"RRF Score     : {result['rrf_score']:.5f}")
        if "rerank_score" in result:
            print(f"Rerank Score  : {result['rerank_score']:.4f}")
        print("-" * 80)
        # Plain print() is safe here because __main__ reconfigures stdout
        # to UTF-8 before calling this function.
        text = result["text"]
        if preview_chars is not None and len(text) > preview_chars:
            print(text[:preview_chars] + "...")
        else:
            print(text)
        print()


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":

    # Reconfigure stdout to UTF-8 so Unicode characters from the PDF
    # (e.g. bullet points encoded as private-use codepoints) don't crash
    # the terminal on Windows where the default codec is cp1252.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

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

    parser.add_argument(
        "--preview-chars",
        type=int,
        default=None,
        help=(
            "Truncate displayed chunk text to this many characters. "
            "Omit (or set to 0) to print the full chunk text (default)."
        ),
    )

    parser.add_argument(
        "--no-rerank",
        dest="rerank",
        action="store_false",
        default=True,
        help=(
            "Disable the cross-encoder reranking stage and return raw RRF "
            "results. Faster but less accurate -- useful for quick iteration "
            "or when the reranker model is not available."
        ),
    )

    parser.add_argument(
        "--rerank-pool",
        type=int,
        default=20,
        help=(
            "Number of RRF candidates to pass to the cross-encoder reranker. "
            "Larger values give the reranker more to choose from but are "
            "slower. Only used when --rerank is set. Default: 20."
        ),
    )

    args = parser.parse_args()

    if args.rerank:
        # Fetch a wider RRF pool so the reranker has enough candidates to
        # surface the correct chunk even if RRF ranked it outside top_k.
        candidates = retrieve(
            args.query,
            top_k=args.top_k,
            rerank_pool_size=args.rerank_pool,
        )
        results = rerank(args.query, candidates, top_k=args.top_k)
    else:
        results = retrieve(args.query, args.top_k)

    print_results(
        results,
        args.query,
        preview_chars=args.preview_chars or None,
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