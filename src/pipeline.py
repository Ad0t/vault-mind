"""
pipeline.py — End-to-end VaultMind pipeline.

Runs all stages in sequence:
  1. Ingest   — PDF → structured pages
  2. Chunk    — pages → sections → overlapping chunks
  3. Embed    — chunks → BGE embeddings
  4. Store    — embeddings + metadata → ChromaDB
  5. Retrieve — hybrid BM25 + dense search, fused with RRF
  6. Rerank   — cross-encoder reranking of the RRF candidate pool
  7. Guard    — out-of-scope and sanity checks
  8. Generate — LLM answer with page-level citations

Run the full pipeline:
    python src/pipeline.py

Run with a custom query:
    python src/pipeline.py --query "What is BCNF?"

Skip ingestion/embedding/storage (re-use existing processed data):
    python src/pipeline.py --retrieve-only --query "What is indexing?"
"""

import argparse
import sys

from ingest import extract_pages, save_pages
from chunk import build_sections, chunk_sections, save_chunks
from embed import generate_embeddings, save_embeddings
from store import store_chunks
from retrieve import retrieve, rerank, print_results
from guardrails import run_guardrails
from generate import generate


def run_pipeline(
    query: str = "What is normalization?",
    top_k: int = 5,
    rerank_pool: int = 20,
    backend: str | None = None,
    retrieve_only: bool = False,
) -> None:
    """
    Execute the full VaultMind pipeline.

    Parameters
    ----------
    query         : question to answer
    top_k         : number of final chunks passed to the LLM
    rerank_pool   : RRF candidate pool size for the cross-encoder reranker
    backend       : LLM backend ('openai' or 'anthropic'); None reads env var
    retrieve_only : if True, skip ingestion / embedding / storage and go
                    straight to retrieval (useful when processed data already
                    exists on disk)
    """

    if not retrieve_only:
        # ------------------------------------------------------------------
        # Stage 1: Ingest
        # ------------------------------------------------------------------
        print("Stage 1: Ingesting PDF...")
        pages = extract_pages()
        save_pages(pages)
        print(f"  Extracted {len(pages)} pages.")

        # ------------------------------------------------------------------
        # Stage 2: Chunk
        # ------------------------------------------------------------------
        print("\nStage 2: Chunking...")
        sections = build_sections(pages)
        chunks = chunk_sections(sections)
        save_chunks(chunks)
        print(f"  Created {len(chunks)} chunks from {len(sections)} sections.")

        # ------------------------------------------------------------------
        # Stage 3: Embed
        # ------------------------------------------------------------------
        print("\nStage 3: Generating embeddings...")
        embeddings = generate_embeddings(chunks)
        save_embeddings(embeddings)
        print(f"  Generated {len(embeddings)} embeddings (dim={len(embeddings[0])}).")

        # ------------------------------------------------------------------
        # Stage 4: Store
        # ------------------------------------------------------------------
        print("\nStage 4: Storing in ChromaDB...")
        collection = store_chunks(chunks, embeddings)
        print(f"  Stored {collection.count()} chunks.")

    # ----------------------------------------------------------------------
    # Stage 5: Retrieve (hybrid BM25 + dense + RRF)
    # ----------------------------------------------------------------------
    print(f"\nStage 5: Retrieving candidates for: {query!r}")
    candidates = retrieve(
        query,
        top_k=top_k,
        rerank_pool_size=rerank_pool,
    )
    print(f"  Retrieved {len(candidates)} RRF candidates.")

    # ----------------------------------------------------------------------
    # Stage 6: Rerank (cross-encoder)
    # ----------------------------------------------------------------------
    print("\nStage 6: Reranking with cross-encoder...")
    reranked = rerank(query, candidates, top_k=top_k)
    print(f"  Reranked to top {len(reranked)} chunks.")

    print("\n--- Retrieved context ---")
    print_results(reranked, query, preview_chars=200)

    # ----------------------------------------------------------------------
    # Stage 7: Guardrails
    # ----------------------------------------------------------------------
    print("\nStage 7: Running guardrails...")
    guard = run_guardrails(query, reranked)

    if not guard.allowed:
        print("\n[GUARDRAIL TRIGGERED]")
        print(guard.reason)
        return

    print("  Guardrails passed.")

    # ----------------------------------------------------------------------
    # Stage 8: Generate
    # ----------------------------------------------------------------------
    print(f"\nStage 8: Generating answer ({backend or 'openai'} backend)...")
    try:
        answer = generate(query, reranked, backend=backend)
        print("\n" + "=" * 70)
        print("ANSWER")
        print("=" * 70)
        print(answer)
    except (ImportError, EnvironmentError) as exc:
        print(f"\n[GENERATION SKIPPED] {exc}")
        print(
            "\nTo enable generation:\n"
            "  1. pip install openai          (or: pip install anthropic)\n"
            "  2. Add OPENAI_API_KEY=sk-...   (or ANTHROPIC_API_KEY=...) to .env"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="VaultMind end-to-end pipeline")

    parser.add_argument(
        "--query",
        type=str,
        default="What is normalization?",
        help="Question to answer.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to pass to the LLM (default: 5).",
    )
    parser.add_argument(
        "--rerank-pool",
        type=int,
        default=20,
        help="RRF pool size for the cross-encoder reranker (default: 20).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["openai", "anthropic"],
        help="LLM backend. Defaults to VAULTMIND_LLM_BACKEND env var or 'openai'.",
    )
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        default=False,
        help=(
            "Skip ingestion, embedding, and storage stages. Use this when "
            "processed data already exists on disk."
        ),
    )

    args = parser.parse_args()

    run_pipeline(
        query=args.query,
        top_k=args.top_k,
        rerank_pool=args.rerank_pool,
        backend=args.backend,
        retrieve_only=args.retrieve_only,
    )