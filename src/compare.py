"""
compare.py — Per-source answer comparison.

Given a query and the global reranked pool, generates a separate answer
for each source PDF using only its own chunks, then prints them side by
side. Optionally runs an LLM meta-pass to highlight agreements and
differences across sources.

Called from pipeline.py after the main answer when the user opts in.
"""

import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── Visual helpers ────────────────────────────────────────────────────────────

_W  = 80          # total line width
_DW = "═" * _W   # double rule  (section headers)
_SW = "─" * _W   # single rule  (chunk separators)


def _header(text: str) -> None:
    print("\n" + _DW)
    print(text)
    print(_DW)


def _sub(text: str) -> None:
    print("\n" + "━" * _W)
    print(text)
    print("━" * _W)


# ── Core comparison logic ─────────────────────────────────────────────────────

def compare_sources(
    query: str,
    reranked: list[dict],
    backend: str | None = None,
    top_k_per_source: int = 3,
    analyze_agreement: bool = True,
) -> None:
    """
    For each source PDF in `reranked`, generate a standalone answer using
    only that PDF's chunks, then optionally run an agreement analysis.

    Parameters
    ----------
    query              : the original user question
    reranked           : the global reranked result list from rerank()
    backend            : LLM backend ('groq', 'openai', 'anthropic'); None → env var
    top_k_per_source   : max chunks per source passed to the LLM
    analyze_agreement  : if True and 2+ sources produced answers, ask the LLM
                         to compare them and note agreements / differences
    """
    from guardrails import run_guardrails
    from generate import generate_structured

    # ── Group reranked chunks by source, preserving rerank order ─────────────
    by_source: dict[str, list[dict]] = defaultdict(list)
    for chunk in reranked:
        src = chunk.get("metadata", {}).get("source_doc", "unknown")
        by_source[src].append(chunk)

    sources = list(by_source.keys())

    if len(sources) < 2:
        print(f"\n  Only one source in results ({sources[0] if sources else '—'}).")
        print("  Nothing to compare — process more PDFs to enable comparison.")
        return

    _header(f"COMPARISON  —  {len(sources)} sources  —  Query: {query!r}")

    source_answers: dict[str, str] = {}   # source → answer text (for meta-analysis)

    for i, source in enumerate(sources, 1):
        chunks = by_source[source][:top_k_per_source]

        _sub(f"Source {i} of {len(sources)}: {source}  ({len(chunks)} chunk(s) used)")

        # Show the top chunks for this source before generating
        for j, chunk in enumerate(chunks, 1):
            meta  = chunk.get("metadata", {})
            page  = _page_label(meta)
            score = chunk.get("rerank_score")
            score_str = f"  rerank={score:.4f}" if score is not None else ""
            print(f"\n  [{j}] Page {page}  ·  {meta.get('section_title','')[:60]}{score_str}")
            preview = chunk["text"][:300]
            if len(chunk["text"]) > 300:
                preview += "…"
            print(f"      {preview.replace(chr(10), chr(10) + '      ')}")

        print()

        # Guardrails on this source's chunks only
        guard = run_guardrails(query, chunks)
        if not guard.allowed:
            print(f"  ⊘  Guardrail triggered — no relevant content in this source.")
            print(f"     {guard.reason}")
            continue

        # Generate answer from this source alone
        try:
            result = generate_structured(query, chunks, backend=backend)
            if result.get("error"):
                print(f"  ⚠  Generation error: {result['error']}")
                continue

            answer = result.get("answer", "")
            print(_SW)
            print("ANSWER FROM THIS SOURCE:")
            print(_SW)
            print(answer)

            # Append source citations
            kept = result.get("context_chunks", [])
            if kept:
                print("\n  Sources used:")
                for k, c in enumerate(kept, 1):
                    meta  = c.get("metadata", {})
                    page  = _page_label(meta)
                    title = meta.get("section_title", "")
                    print(f"    [{k}] Page {page} — {title}")

            source_answers[source] = answer

        except (ImportError, EnvironmentError) as exc:
            print(f"  ⚠  Generation skipped: {exc}")

    # ── Agreement analysis ────────────────────────────────────────────────────
    if analyze_agreement and len(source_answers) >= 2:
        _agreement_analysis(query, source_answers, backend)


def _page_label(meta: dict) -> str:
    s = meta.get("start_page", "?")
    e = meta.get("end_page",   "?")
    return str(s) if s == e else f"{s}–{e}"


def _agreement_analysis(
    query: str,
    source_answers: dict[str, str],
    backend: str | None = None,
) -> None:
    """
    Ask the LLM to compare the per-source answers and note what they
    agree on, what's unique to each, and any contradictions.
    """
    import os
    from generate import _BACKENDS

    if backend is None:
        backend = os.environ.get("VAULTMIND_LLM_BACKEND", "groq").lower()

    if backend not in _BACKENDS:
        return

    _header("AGREEMENT ANALYSIS  (LLM-generated)")

    # Build the meta-prompt
    numbered = "\n\n".join(
        f"Answer from {src}:\n{ans}"
        for src, ans in source_answers.items()
    )

    system = (
        "You are a neutral academic analyst. "
        "You are given multiple answers to the same question, each sourced from "
        "a different document. Your job is to briefly compare them:\n"
        "1. What key points do all sources agree on?\n"
        "2. What does each source add that others don't?\n"
        "3. Are there any direct contradictions?\n"
        "Keep your analysis concise (3–6 bullet points per section)."
    )

    user = (
        f"Question: {query}\n\n"
        f"{numbered}\n\n"
        "Compare the answers above following the three-part structure."
    )

    try:
        # Directly call the backend with the meta-prompt (no context building needed)
        fn = _BACKENDS[backend]
        # _BACKENDS functions take (query, context) — we repurpose them:
        # pass the combined answers as "context" and the analysis instruction as "query"
        analysis = fn(
            query="Analyze and compare the answers above.",
            context=f"Question asked: {query}\n\n{numbered}",
        )
        print(analysis)
    except Exception as exc:
        print(f"  ⚠  Agreement analysis failed: {exc}")


# ── Standalone smoke test ─────────────────────────────────────────────────────

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    # Quick sanity check — no LLM call, just the formatting path
    fake_chunks = [
        {
            "id": f"chunk_{src}_{i}",
            "text": f"Sample text about transactions from {src}, chunk {i}.",
            "metadata": {
                "start_page": i * 10,
                "end_page": i * 10,
                "source_doc": src,
                "section_title": f"Section {i}",
                "heading_level": 1,
                "chunk_index": i,
            },
            "rrf_score": 0.05 - i * 0.01,
            "rerank_score": 9.0 - i,
        }
        for src in ("A.pdf", "B.pdf")
        for i in range(1, 3)
    ]

    print("compare.py — smoke test (no LLM)")
    by_source: dict = defaultdict(list)
    for c in fake_chunks:
        by_source[c["metadata"]["source_doc"]].append(c)

    _header("COMPARISON TEST")
    for src, chunks in by_source.items():
        _sub(f"Source: {src}")
        for j, ch in enumerate(chunks, 1):
            meta = ch["metadata"]
            print(f"  [{j}] Page {meta['start_page']}  ·  {meta['section_title']}")
            print(f"      {ch['text']}")
    print("\nSmoke test passed — formatting looks good.")
