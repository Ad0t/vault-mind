"""
generate.py — LLM answer generation with citation-aware context building.

Pipeline position:
    retrieve() → rerank() → build_context() → call_llm() → format_citations()

Supported backends:
    • groq      (default) — Llama-3.1-8b-instant (free) — set GROQ_API_KEY
    • openai              — GPT-4o-mini            — set OPENAI_API_KEY
    • anthropic           — Claude Haiku           — set ANTHROPIC_API_KEY

Select backend via the VAULTMIND_LLM_BACKEND environment variable, or pass
`backend` directly to generate().

Usage (standalone):
    python src/generate.py --query "What is normalization?"
    python src/generate.py --query "What is BCNF?" --backend groq
    python src/generate.py --query "What is indexing?" --backend openai
"""

import argparse
import os
import re
import sys
from pathlib import Path

# Load .env file if present (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; set env vars directly if not installed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieve import retrieve, rerank  # noqa: E402  (path set above)

# ---------------------------------------------------------------------------
# Unicode cleanup
# ---------------------------------------------------------------------------

# PDF files encoded with Symbol / Wingdings fonts map standard bullet and
# arrow glyphs into the Unicode Private Use Area (U+F000–U+F8FF). These
# characters are invisible or show as boxes in most terminals and LLM APIs.
# We replace them with their closest visible equivalents so the LLM receives
# clean, readable text.
_PRIVATE_USE_MAP: dict[str, str] = {
    "\uf0b7": "\u2022",  # bullet  •
    "\uf076": "\u2022",  # bullet variant  •
    "\uf0d8": "\u25b6",  # filled right-arrow  ▶
    "\uf0fc": "\u2713",  # check mark  ✓
    "\uf0de": "\u25b2",  # up-arrow  ▲
    "\uf0fe": "\u25a0",  # filled square  ■
}
# Build a single compiled regex that matches any private-use codepoint so we
# can replace unknown ones too (fall back to the generic replacement char).
_PUA_RE = re.compile(r"[\uf000-\uf8ff]")


def _replace_pua(match: re.Match) -> str:
    char = match.group()
    return _PRIVATE_USE_MAP.get(char, "\ufffd")  # \ufffd = replacement char


def clean_text(text: str) -> str:
    """
    Replace PDF Private-Use-Area artifacts with readable Unicode equivalents.
    Also collapses excessive whitespace introduced by PDF line-wrapping.
    """
    text = _PUA_RE.sub(_replace_pua, text)
    # Collapse runs of blank lines down to one (PDF often leaves triple-newlines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------

# Chunks shorter than this after cleaning are considered parsing noise and
# excluded from the LLM context (but kept in citations if they slip through).
_MIN_CHUNK_CHARS = 50


def build_context(
    chunks: list[dict],
    min_chars: int = _MIN_CHUNK_CHARS,
) -> tuple[str, list[dict]]:
    """
    Prepare retrieved chunks for the LLM prompt.

    Steps applied (in order):
    1. Clean private-use Unicode in each chunk's text.
    2. Filter out chunks whose cleaned text is shorter than `min_chars`
       (these are usually table fragments or stray lines from the PDF parser).
    3. Deduplicate by exact text match — the same small table can appear in
       multiple sections of the textbook and burn context window tokens twice.

    Returns
    -------
    context_str   : numbered, labelled excerpts ready to paste into the prompt
    kept_chunks   : the filtered/deduped chunk list (same order as context_str,
                    used by format_citations to build the source list)
    """
    seen: set[str] = set()
    kept: list[dict] = []

    for chunk in chunks:
        cleaned = clean_text(chunk["text"])

        if len(cleaned) < min_chars:
            continue  # noise — too short to be useful

        if cleaned in seen:
            continue  # exact duplicate — skip

        seen.add(cleaned)
        kept.append({**chunk, "text": cleaned})

    # Build numbered excerpts with page + section labels so the LLM can cite
    # them correctly without hallucinating source information.
    parts: list[str] = []
    for i, chunk in enumerate(kept, start=1):
        meta = chunk["metadata"]
        start = meta["start_page"]
        end   = meta["end_page"]
        page_label = str(start) if start == end else f"{start}–{end}"
        parts.append(
            f"[{i}] (Page {page_label} | {meta['section_title']})\n{chunk['text']}"
        )

    return "\n\n".join(parts), kept


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are VaultMind, an expert assistant for Database Management Systems (DBMS).
You answer questions using ONLY the numbered document excerpts provided by the user.

Rules you must follow:
1. Ground every claim in the provided excerpts. Do not use outside knowledge.
2. Cite your sources inline using [n] notation (e.g., "Normalization is… [1].").
3. If the excerpts do not contain enough information, say so clearly and do not guess.
4. Lead definitional answers with the definition before elaborating.
5. Keep answers concise and accurate. Avoid restating the question.\
"""

_USER_TEMPLATE = """\
Excerpts from the DBMS textbook:

{context}

Question: {query}

Answer using only the excerpts above. Cite sources with [n] notation.\
"""


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

def _call_openai(
    query: str,
    context: str,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Call the OpenAI Chat Completions API.

    Requires: pip install openai
              OPENAI_API_KEY environment variable
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "OpenAI backend requires the openai package.\n"
            "Install it with:  pip install openai"
        ) from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file or shell environment."
        )

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _USER_TEMPLATE.format(context=context, query=query)},
        ],
        temperature=0.1,   # low temperature for factual, grounded answers
        max_tokens=800,
    )
    return response.choices[0].message.content.strip()


def _call_anthropic(
    query: str,
    context: str,
    model: str = "claude-haiku-20240307",
) -> str:
    """
    Call the Anthropic Messages API.

    Requires: pip install anthropic
              ANTHROPIC_API_KEY environment variable
    """
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "Anthropic backend requires the anthropic package.\n"
            "Install it with:  pip install anthropic"
        ) from exc

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file or shell environment."
        )

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=800,
        system=_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": _USER_TEMPLATE.format(context=context, query=query)},
        ],
        temperature=0.1,
    )
    return response.content[0].text.strip()


def _call_groq(
    query: str,
    context: str,
    model: str = "llama-3.1-8b-instant",
) -> str:
    """
    Call the Groq Chat Completions API.

    Groq provides OpenAI-compatible endpoints with free-tier access to
    several open-weight models. Recommended options:
        llama-3.1-8b-instant    — fastest, great for RAG (default)
        llama-3.3-70b-versatile — smarter, slightly slower, still free
        gemma2-9b-it            — Google's model, solid alternative

    Requires: pip install groq
              GROQ_API_KEY environment variable
              Get a free key at: https://console.groq.com
    """
    try:
        from groq import Groq
    except ImportError as exc:
        raise ImportError(
            "Groq backend requires the groq package.\n"
            "Install it with:  pip install groq"
        ) from exc

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to your .env file.\n"
            "Get a free key at: https://console.groq.com"
        )

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _USER_TEMPLATE.format(context=context, query=query)},
        ],
        temperature=0.1,   # low temperature for factual, grounded answers
        max_tokens=800,
    )
    return response.choices[0].message.content.strip()


_BACKENDS: dict[str, callable] = {
    "groq":      _call_groq,
    "openai":    _call_openai,
    "anthropic": _call_anthropic,
}


# ---------------------------------------------------------------------------
# Citation formatter
# ---------------------------------------------------------------------------

def format_citations(kept_chunks: list[dict]) -> str:
    """
    Build the source list that is appended after the LLM's answer.

    Example output:
        --- Sources ---
        [1] Page 52 — 9.Introduction to Database Normalization
            Database Management Systems.pdf
        [2] Page 53 — The features of database normalization are as follows:
            Database Management Systems.pdf
    """
    if not kept_chunks:
        return ""

    lines = ["\n--- Sources ---"]
    for i, chunk in enumerate(kept_chunks, start=1):
        meta = chunk["metadata"]
        start = meta["start_page"]
        end   = meta["end_page"]
        page_label = str(start) if start == end else f"{start}–{end}"
        lines.append(
            f"[{i}] Page {page_label} — {meta['section_title']}\n"
            f"    {meta['source_doc']}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate(
    query: str,
    chunks: list[dict],
    backend: str | None = None,
    min_chars: int = _MIN_CHUNK_CHARS,
) -> str:
    """
    Generate a citation-grounded answer from retrieved (and reranked) chunks.

    Parameters
    ----------
    query     : the user's question
    chunks    : reranked chunks returned by rerank() — each must have keys
                'text', 'metadata' (with start_page / end_page / section_title
                / source_doc), and ideally 'rerank_score'.
    backend   : 'openai' or 'anthropic'. None reads VAULTMIND_LLM_BACKEND
                env var, defaulting to 'openai'.
    min_chars : chunks shorter than this (after cleaning) are excluded from
                the LLM context to avoid feeding noise.

    Returns
    -------
    A string containing the LLM's answer followed by a formatted source list.
    """
    if backend is None:
        backend = os.environ.get("VAULTMIND_LLM_BACKEND", "groq").lower()

    if backend not in _BACKENDS:
        raise ValueError(
            f"Unknown backend {backend!r}. Choose from: {list(_BACKENDS)}"
        )

    context, kept_chunks = build_context(chunks, min_chars=min_chars)

    if not kept_chunks:
        return (
            "No usable context was found after filtering. "
            "The retrieved chunks may be too short or duplicated."
        )

    answer   = _BACKENDS[backend](query, context)
    citations = format_citations(kept_chunks)
    return answer + "\n" + citations


def generate_structured(
    query: str,
    chunks: list[dict],
    backend: str | None = None,
    min_chars: int = _MIN_CHUNK_CHARS,
) -> dict:
    """
    Like generate() but returns a structured dict instead of a formatted string.

    Used by the Streamlit UI so it can render citations as interactive
    expandable blocks rather than a flat text append.

    Returns
    -------
    {
        "answer"         : str | None   — LLM response text (None on error)
        "context_chunks" : list[dict]   — cleaned/deduped chunks used in prompt
        "error"          : str | None   — human-readable error message, or None
    }
    """
    if backend is None:
        backend = os.environ.get("VAULTMIND_LLM_BACKEND", "groq").lower()

    if backend not in _BACKENDS:
        return {
            "answer": None,
            "context_chunks": [],
            "error": f"Unknown backend {backend!r}. Choose from: {list(_BACKENDS)}",
        }

    context, kept_chunks = build_context(chunks, min_chars=min_chars)

    if not kept_chunks:
        return {
            "answer": None,
            "context_chunks": [],
            "error": "No usable context found after filtering short/duplicate chunks.",
        }

    try:
        answer = _BACKENDS[backend](query, context)
        return {
            "answer": answer,
            "context_chunks": kept_chunks,
            "error": None,
        }
    except (ImportError, EnvironmentError, Exception) as exc:
        return {
            "answer": None,
            "context_chunks": kept_chunks,
            "error": str(exc),
        }



# ---------------------------------------------------------------------------
# CLI (standalone usage)
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="VaultMind — RAG generation layer",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What is normalization?",
        help="Question to answer from the DBMS corpus.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["groq", "openai", "anthropic"],
        help=(
            "LLM backend to use. Defaults to VAULTMIND_LLM_BACKEND env var "
            "or 'groq' if not set."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve and pass to the LLM.",
    )
    parser.add_argument(
        "--rerank-pool",
        type=int,
        default=20,
        help="RRF candidate pool size fed into the cross-encoder reranker.",
    )

    args = parser.parse_args()

    print(f"Retrieving context for: {args.query!r}\n")

    candidates = retrieve(
        args.query,
        top_k=args.top_k,
        rerank_pool_size=args.rerank_pool,
    )
    reranked = rerank(args.query, candidates, top_k=args.top_k)

    print("=" * 70)
    answer = generate(args.query, reranked, backend=args.backend)
    print(answer)
