"""
guardrails.py — Input validation and relevance-based out-of-scope detection.

Called by the generation layer BEFORE the LLM API is invoked so that:
  • Empty / nonsense queries are rejected cheaply (no model inference needed).
  • Queries outside the DBMS corpus are caught using the cross-encoder
    relevance score, preventing hallucinated "answers" on topics the textbook
    doesn't cover.

Usage (standalone smoke-test):
    python src/guardrails.py
"""

import re
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Thresholds & constants
# ---------------------------------------------------------------------------

# Cross-encoder (ms-marco-MiniLM-L-6-v2) outputs raw logits (no fixed range).
# Empirically observed score ranges against this DBMS corpus:
#
#   ≥  5.0   Very strong match  (e.g. direct definitional query)
#   0 – 5.0  Moderate match     (related topic, partial answer)
#  -2 – 0    Weak match         (tangentially related)
#   < -2.0   No real match      (off-topic or out-of-scope)
#
# We use -2.0 as the rejection threshold: conservative enough to avoid
# blocking legitimate but oddly phrased DBMS questions, but low enough to
# catch clearly unrelated queries like "What is photosynthesis?".
RELEVANCE_THRESHOLD: float = -2.0

# A query must contain at least this many alphabetic characters to be
# considered a real question (blocks pure symbol / numeric inputs).
MIN_ALPHA_CHARS: int = 3

# Maximum query length accepted (very long inputs are likely prompt injection
# attempts or copy-paste errors).
MAX_QUERY_CHARS: int = 1000


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class GuardrailResult(NamedTuple):
    """
    allowed : True  → proceed to generation
              False → return `reason` to the user instead of calling the LLM
    reason  : human-readable explanation when allowed is False; empty string
              when allowed is True.
    """
    allowed: bool
    reason: str


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_query_sanity(query: str) -> GuardrailResult:
    """
    Lightweight pre-checks that do not require any model inference.

    Rejects:
    • Empty or whitespace-only strings.
    • Queries with fewer than MIN_ALPHA_CHARS alphabetic characters
      (e.g. "123", "???", "@#$").
    • Queries longer than MAX_QUERY_CHARS (likely prompt injection or
      accidental paste).
    """
    stripped = query.strip()

    if not stripped:
        return GuardrailResult(
            allowed=False,
            reason="Please enter a question.",
        )

    alpha_count = sum(1 for ch in stripped if ch.isalpha())
    if alpha_count < MIN_ALPHA_CHARS:
        return GuardrailResult(
            allowed=False,
            reason=(
                "Your input doesn't look like a question. "
                "Please ask something about Database Management Systems."
            ),
        )

    if len(stripped) > MAX_QUERY_CHARS:
        return GuardrailResult(
            allowed=False,
            reason=(
                f"Your query is too long ({len(stripped)} characters). "
                f"Please keep it under {MAX_QUERY_CHARS} characters."
            ),
        )

    return GuardrailResult(allowed=True, reason="")


def check_relevance(
    results: list[dict],
    threshold: float = RELEVANCE_THRESHOLD,
) -> GuardrailResult:
    """
    Reject queries whose best-matching chunk scores below `threshold`.

    The cross-encoder rerank_score is a direct measure of how well the
    retrieved passage answers the query. A very low score means the
    retriever found nothing truly relevant — generating from that context
    would produce hallucinated or confabulated answers.

    If results have no rerank_score (reranker was not used), this check
    is bypassed — we can't assess relevance without that signal.
    """
    if not results:
        return GuardrailResult(
            allowed=False,
            reason=(
                "No document chunks were retrieved for your query. "
                "Please try rephrasing your question."
            ),
        )

    top = results[0]

    # Reranker was skipped — can't assess relevance, allow through.
    if "rerank_score" not in top:
        return GuardrailResult(allowed=True, reason="")

    score = top["rerank_score"]

    if score < threshold:
        return GuardrailResult(
            allowed=False,
            reason=(
                "I couldn't find relevant information about this topic in the "
                "DBMS textbook. VaultMind is designed to answer questions about "
                "Database Management Systems.\n\n"
                "Try rephrasing your question, or check that it's related to "
                "topics like SQL, normalization, transactions, indexing, ER "
                "models, or relational algebra.\n\n"
                f"(Relevance score: {score:.2f} — threshold: {threshold:.2f})"
            ),
        )

    return GuardrailResult(allowed=True, reason="")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_guardrails(
    query: str,
    results: list[dict],
    relevance_threshold: float = RELEVANCE_THRESHOLD,
) -> GuardrailResult:
    """
    Run all guardrail checks in priority order and return the first failure,
    or a passing GuardrailResult if all checks pass.

    Order:
    1. Query sanity   — no model needed, fails fast on bad input.
    2. Relevance      — uses the cross-encoder score on retrieved results.

    Parameters
    ----------
    query               : the raw user query string
    results             : reranked chunks returned by rerank()
    relevance_threshold : cross-encoder score below which the query is
                          considered out-of-scope (default RELEVANCE_THRESHOLD)
    """
    # 1. Sanity check
    sanity = check_query_sanity(query)
    if not sanity.allowed:
        return sanity

    # 2. Relevance check
    relevance = check_relevance(results, threshold=relevance_threshold)
    if not relevance.allowed:
        return relevance

    return GuardrailResult(allowed=True, reason="")


# ---------------------------------------------------------------------------
# Smoke test (run directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Simulate reranked results with various scores for testing
    def _make_result(score: float) -> dict:
        return {
            "id": "chunk_00001",
            "text": "Database normalization is the process of organizing attributes...",
            "metadata": {
                "start_page": 1, "end_page": 1,
                "section_title": "Normalization", "source_doc": "DBMS.pdf",
                "heading_level": 1, "chunk_index": 0,
            },
            "rrf_score": 0.03,
            "rerank_score": score,
        }

    test_cases = [
        # (query, results, expected_allowed)
        ("",                        [],                    False),  # empty query
        ("???",                     [],                    False),  # no alpha chars
        ("A" * 1001,                [],                    False),  # too long
        ("What is photosynthesis?", [_make_result(-5.0)],  False),  # out-of-scope
        ("What is normalization?",  [_make_result(5.3)],   True),   # valid
        ("What is a primary key?",  [_make_result(0.5)],   True),   # moderate score
    ]

    print("Guardrail smoke test\n" + "=" * 50)
    all_pass = True
    for query, results, expected in test_cases:
        result = run_guardrails(query, results)
        status = "✓" if result.allowed == expected else "✗ FAIL"
        if result.allowed != expected:
            all_pass = False
        label = repr(query[:40]) if len(query) <= 40 else repr(query[:37] + "...")
        print(f"{status}  allowed={result.allowed}  query={label}")
        if not result.allowed:
            print(f"      reason: {result.reason[:100]}")

    print("=" * 50)
    print("All tests passed." if all_pass else "SOME TESTS FAILED.")
