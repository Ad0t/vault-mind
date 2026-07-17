"""
pipeline.py — End-to-end VaultMind pipeline.

Stages:
  1. Ingest   — PDF(s) → structured pages
  2. Chunk    — pages  → sections → overlapping chunks
  3. Embed    — chunks → BGE embeddings
  4. Store    — embeddings + metadata → ChromaDB
  5. Retrieve — hybrid BM25 + dense search, fused with RRF
  6. Rerank   — cross-encoder reranking of the RRF candidate pool
  7. Guard    — out-of-scope and sanity checks
  8. Generate — LLM answer with page-level citations

Usage
-----
Full pipeline (scans data/raw/ for PDFs, prompts for question):
    python src/pipeline.py

Skip ingestion/embedding/storage (re-use existing processed data):
    python src/pipeline.py --retrieve-only

Pass everything non-interactively:
    python src/pipeline.py --pdfs "data/raw/A.pdf,data/raw/B.pdf" --query "What is BCNF?"
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR      = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------------------------------
# PDF discovery & interactive selection
# ---------------------------------------------------------------------------

def scan_pdfs() -> list[Path]:
    """Return all PDFs found in data/raw/, sorted by name."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(RAW_DIR.glob("*.pdf"))


def select_pdfs(pdfs: list[Path]) -> list[Path]:
    """
    Interactively ask the user which PDFs to process.

    • 0 PDFs found  → exit with a helpful message
    • 1 PDF found   → auto-selected (no prompt needed)
    • 2+ PDFs found → numbered menu; user types indices or 'a' for all
    """
    if not pdfs:
        print(
            f"\n  No PDF files found in {RAW_DIR}\n"
            "  Place at least one PDF in that directory and try again.\n"
        )
        sys.exit(1)

    if len(pdfs) == 1:
        print(f"\n  Found 1 PDF — auto-selected: {pdfs[0].name}")
        return pdfs

    print(f"\n  Found {len(pdfs)} PDFs in {RAW_DIR.relative_to(PROJECT_ROOT)}:")
    for i, pdf in enumerate(pdfs, 1):
        size_mb = pdf.stat().st_size / 1_048_576
        print(f"    [{i}] {pdf.name}  ({size_mb:.1f} MB)")

    print("\n  Enter numbers to select (e.g. 1  or  1,3) — or 'a' for all:")

    while True:
        raw = input("  > ").strip().lower()

        if raw in ("a", "all", ""):
            return pdfs

        try:
            indices  = [int(x.strip()) for x in raw.split(",") if x.strip()]
            selected = [pdfs[i - 1] for i in indices if 1 <= i <= len(pdfs)]
            if selected:
                print(
                    f"\n  Selected: {', '.join(p.name for p in selected)}"
                )
                return selected
        except (ValueError, IndexError):
            pass

        print(f"  Invalid — enter numbers between 1 and {len(pdfs)}, or 'a'.")


# ---------------------------------------------------------------------------
# Question prompt
# ---------------------------------------------------------------------------

def prompt_query() -> str:
    """Interactively read a non-empty question from the user."""
    print()
    while True:
        query = input("  Enter your question: ").strip()
        if query:
            return query
        print("  Question cannot be empty — please try again.")


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def run_pipeline(
    query: str,
    pdf_paths: list[Path] | None = None,
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
    pdf_paths     : list of PDF paths to ingest (ignored when retrieve_only=True)
    top_k         : number of final chunks passed to the LLM
    rerank_pool   : RRF candidate pool size for the cross-encoder reranker
    backend       : LLM backend ('groq', 'openai', 'anthropic'); None reads env var
    retrieve_only : if True, skip stages 1-4 and go straight to retrieval
    """
    from ingest import extract_pages, save_pages
    from chunk import build_sections, chunk_sections, save_chunks
    from embed import generate_embeddings, save_embeddings
    from store import store_chunks
    from retrieve import retrieve_per_source, rerank, print_results
    from guardrails import run_guardrails
    from generate import generate

    if not retrieve_only:
        if not pdf_paths:
            print("  No PDFs provided for ingestion.")
            sys.exit(1)

        # ------------------------------------------------------------------
        # Stage 1: Ingest — iterate over every selected PDF
        # ------------------------------------------------------------------
        print("\nStage 1: Ingesting PDF(s)...")
        all_pages: list[dict] = []
        for pdf_path in pdf_paths:
            print(f"  Reading {pdf_path.name}...")
            pages = extract_pages(pdf_path)
            all_pages.extend(pages)
            print(f"    → {len(pages)} pages")

        save_pages(all_pages)
        print(f"  Total: {len(all_pages)} pages across {len(pdf_paths)} file(s).")

        # ------------------------------------------------------------------
        # Stage 2: Chunk
        # ------------------------------------------------------------------
        print("\nStage 2: Chunking...")
        sections = build_sections(all_pages)
        chunks   = chunk_sections(sections)
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
    # Stage 5: Retrieve — per-source BM25 + dense + RRF, then merge
    # ----------------------------------------------------------------------
    print(f"\nStage 5: Retrieving candidates for: {query!r}")
    candidates = retrieve_per_source(
        query,
        top_k=top_k,
        candidates_per_source=max(top_k, rerank_pool // 4),
        rerank_pool_size_per_source=rerank_pool,
    )

    # Show how many candidates each source contributed
    from collections import Counter
    source_counts = Counter(
        c["metadata"]["source_doc"] for c in candidates
    )
    for src, cnt in sorted(source_counts.items()):
        print(f"  {src}: {cnt} candidate(s)")
    print(f"  Total pool: {len(candidates)} candidates across {len(source_counts)} source(s).")

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
    backend_label = backend or "groq"
    print(f"\nStage 8: Generating answer ({backend_label} backend)...")
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
            "  1. pip install groq\n"
            "  2. Add GROQ_API_KEY=gsk_... to .env\n"
            "  (see .env for all supported backends)"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="VaultMind end-to-end pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pdfs",
        type=str,
        default=None,
        help=(
            "Comma-separated PDF paths to ingest. "
            "Omit to scan data/raw/ and pick interactively."
        ),
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Question to answer. Omit to be prompted interactively.",
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
        choices=["groq", "openai", "anthropic"],
        help="LLM backend (default: reads VAULTMIND_LLM_BACKEND env var or 'groq').",
    )
    parser.add_argument(
        "--retrieve-only",
        action="store_true",
        default=False,
        help=(
            "Skip ingestion, embedding, and storage stages. "
            "Useful when processed data already exists on disk."
        ),
    )

    args = parser.parse_args()

    # ── PDF selection ────────────────────────────────────────────────────────
    pdf_paths: list[Path] | None = None

    if not args.retrieve_only:
        if args.pdfs:
            # Passed explicitly via --pdfs "a.pdf,b.pdf"
            pdf_paths = [Path(p.strip()) for p in args.pdfs.split(",") if p.strip()]
            missing   = [p for p in pdf_paths if not p.exists()]
            if missing:
                print(f"Error: file(s) not found: {', '.join(str(p) for p in missing)}")
                sys.exit(1)
        else:
            # Interactive: scan data/raw/ and let the user pick
            all_pdfs  = scan_pdfs()
            pdf_paths = select_pdfs(all_pdfs)

    # ── Question ─────────────────────────────────────────────────────────────
    query = args.query or prompt_query()

    # ── Run ──────────────────────────────────────────────────────────────────
    run_pipeline(
        query=query,
        pdf_paths=pdf_paths,
        top_k=args.top_k,
        rerank_pool=args.rerank_pool,
        backend=args.backend,
        retrieve_only=args.retrieve_only,
    )