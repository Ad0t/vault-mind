"""
run_evaluation.py — Automated Evaluation Runner for VaultMind

This script executes the 20+ benchmark Q&A pairs defined in `data/eval_dataset.json`
through the VaultMind hybrid RAG pipeline (BM25 + Dense RRF -> Rerank -> Guardrails -> LLM).

It outputs two evaluation artifacts:
  1. `data/eval_results.csv`  — CSV scorecard with columns for human/LLM rating (Correctness, Citation Precision, Completeness).
  2. `data/eval_report.md`    — Markdown report with exact rubrics and side-by-side comparison tables.

Usage:
    python scripts/run_evaluation.py
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

# Add src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from retrieve import retrieve_per_source, rerank
from guardrails import run_guardrails
from generate import generate_structured, format_citations


DATA_DIR      = PROJECT_ROOT / "data"
EVAL_DATASET  = DATA_DIR / "eval_dataset.json"
OUTPUT_CSV    = DATA_DIR / "eval_results.csv"
OUTPUT_MD     = DATA_DIR / "eval_report.md"


def load_dataset() -> list[dict]:
    if not EVAL_DATASET.exists():
        print(f"Error: {EVAL_DATASET} not found.")
        sys.exit(1)
    with open(EVAL_DATASET, "r", encoding="utf-8") as f:
        return json.load(f)


def run_eval():
    print("=" * 70)
    print("🧠 VaultMind Automated Evaluation Suite (20+ Benchmark Questions)")
    print("=" * 70)

    dataset = load_dataset()
    print(f"Loaded {len(dataset)} evaluation questions across {len(set(q['category'] for q in dataset))} categories.\n")

    results = []
    top_k = 5
    rerank_pool = 20

    for i, item in enumerate(dataset, start=1):
        qid      = item["id"]
        category = item["category"]
        diff     = item["difficulty"]
        query    = item["query"]
        gt       = item["ground_truth_summary"]

        print(f"[{i:02d}/{len(dataset):02d}] ({category} | {diff}) Query: {query}")

        start_time = time.time()

        # 1. Retrieve
        candidates = retrieve_per_source(
            query,
            top_k=top_k,
            candidates_per_source=max(top_k, rerank_pool // 4),
            rerank_pool_size_per_source=rerank_pool,
        )

        # 2. Rerank
        reranked = rerank(query, candidates, top_k=top_k)

        # 3. Guardrails
        guard = run_guardrails(query, reranked)

        if not guard.allowed:
            gen_answer = f"[GUARDRAIL REFUSAL] {guard.reason}"
            citations_str = "None (Refused)"
        else:
            # 4. Generate
            gen_res = generate_structured(query, reranked)
            if gen_res.get("error"):
                gen_answer = f"[GENERATION ERROR] {gen_res['error']}"
            else:
                gen_answer = gen_res.get("answer", "No answer returned.")
            
            # Format citations cleanly
            citations_str = format_citations(gen_res.get("context_chunks", []))

        latency = round(time.time() - start_time, 2)
        print(f"       -> Latency: {latency}s | Chunks retrieved: {len(reranked)}\n")

        results.append({
            "id": qid,
            "category": category,
            "difficulty": diff,
            "query": query,
            "ground_truth_summary": gt,
            "generated_answer": gen_answer.strip(),
            "retrieved_citations": citations_str.strip(),
            "latency_sec": latency,
            "correctness_score": "",        # To be filled by Human / LLM judge (1-5)
            "citation_precision_score": "", # To be filled by Human / LLM judge (1-5)
            "completeness_score": "",       # To be filled by Human / LLM judge (1-5)
            "evaluator_notes": ""
        })

    # ── Write CSV Scorecard ──────────────────────────────────────────────────
    print(f"\nWriting scorecard to: {OUTPUT_CSV}")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    # ── Write Markdown Report Template ───────────────────────────────────────
    print(f"Writing markdown scorecard to: {OUTPUT_MD}")
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("# VaultMind — Human & LLM Evaluation Report (20+ Q&A Benchmark)\n\n")
        f.write("## 📐 Evaluation Rubric (3 Axes | Score 1 to 5)\n\n")
        f.write("### 1. Correctness (Factuality & Entailment)\n")
        f.write("- **5 (Perfect):** Every statement is completely accurate and logically entailed by the ground truth and retrieved chunks.\n")
        f.write("- **3 (Partial):** Core concept is correct, but contains minor inaccuracies or slight misunderstandings.\n")
        f.write("- **1 (Hallucinated / Incorrect):** Directly contradicts established DBMS principles or makes unsupported claims.\n\n")
        f.write("### 2. Citation Precision (Grounding & Provenance)\n")
        f.write("- **5 (Exact):** Inline `[n]` citations point precisely to the exact page and section header where the claim is stated.\n")
        f.write("- **3 (Approximate):** Citations point to the right document/chapter, but specific page numbers are broad or slightly off.\n")
        f.write("- **1 (Unverified / Missing):** No citations provided, or citations point to completely irrelevant sections.\n\n")
        f.write("### 3. Completeness (Depth & Structure)\n")
        f.write("- **5 (Comprehensive & Structured):** Leads with a clear definition, uses bullet points/numbered lists for key properties, and covers all critical facets.\n")
        f.write("- **3 (Adequate):** Answers the basic question but misses key nuances or presents as a flat paragraph.\n")
        f.write("- **1 (Incomplete):** Omits critical details or cuts off abruptly.\n\n")
        f.write("*(Note: For out-of-scope queries `q21-q22`, a clean `[GUARDRAIL REFUSAL]` score is **5/5/5** across all axes for correct boundary enforcement)*\n\n")
        f.write("---\n\n## 📊 Evaluation Scorecard Table\n\n")
        f.write("| ID | Category | Query | Correctness (1-5) | Citation Precision (1-5) | Completeness (1-5) | Notes |\n")
        f.write("| :--- | :--- | :--- | :---: | :---: | :---: | :--- |\n")
        for r in results:
            clean_q = r["query"].replace("\n", " ")
            f.write(f"| `{r['id']}` | **{r['category']}** | {clean_q} | | | | |\n")

        f.write("\n---\n\n## 📝 Detailed Q&A Comparison Breakdown\n\n")
        for r in results:
            f.write(f"### `{r['id']}` — {r['query']}\n")
            f.write(f"- **Category:** {r['category']} | **Difficulty:** {r['difficulty']} | **Latency:** {r['latency_sec']}s\n\n")
            f.write(f"#### 🎯 Ground Truth Summary\n> {r['ground_truth_summary']}\n\n")
            f.write(f"#### 🤖 VaultMind Generated Answer\n```markdown\n{r['generated_answer']}\n```\n")
            if r["retrieved_citations"] and r["retrieved_citations"] != "None (Refused)":
                f.write(f"\n#### 📚 Retrieved Citations\n```text\n{r['retrieved_citations']}\n```\n")
            f.write("\n#### ✍️ Scores (`1-5`):\n")
            f.write("- **Correctness:** ______ / 5\n")
            f.write("- **Citation Precision:** ______ / 5\n")
            f.write("- **Completeness:** ______ / 5\n")
            f.write("- **Evaluator Notes:** _________________________________________\n\n---\n\n")

    print("\n✅ Evaluation run complete! Open `data/eval_results.csv` or `data/eval_report.md` to start grading.")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    run_eval()
