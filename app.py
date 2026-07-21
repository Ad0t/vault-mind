"""
app.py — VaultMind Gradio UI

Two-screen flow:
  Screen 1 (Upload)  — drop one or more PDFs → "Process & Chat" button
  Screen 2 (Chat)    — chatbot with a source selector (check/uncheck per PDF);
                        a "Compare Sources" button appears automatically when
                        the last query retrieved relevant content from 2+ PDFs.
                        Clicking it generates a side-by-side HTML table with
                        per-PDF answers, pages, sections, and confidence scores.

Run with:
    python app.py
"""

import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# ── Path & env setup ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import gradio as gr

# ── Constants ─────────────────────────────────────────────────────────────────
RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CHUNKS_PATH   = PROCESSED_DIR / "chunks.json"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Custom CSS ────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; }

/* ── Header ── */
.vm-header { text-align:center; padding:1.6rem 0 0.8rem 0; }
.vm-title {
    font-size:2.2rem; font-weight:700;
    background:linear-gradient(135deg,#6366f1 0%,#8b5cf6 55%,#a78bfa 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; margin:0; line-height:1.2;
}
.vm-subtitle { color:#94a3b8; font-size:0.9rem; margin:0.3rem 0 0 0; }

/* ── Upload screen ── */
.upload-card {
    max-width:560px; margin:2rem auto;
    background:#1e293b; border:1px solid #334155;
    border-radius:16px; padding:2rem;
}

/* ── Confidence badges ── */
.badge {
    display:inline-block; padding:0.15rem 0.6rem; border-radius:999px;
    font-size:0.72rem; font-weight:600; letter-spacing:0.04em; margin:0.4rem 0;
}
.badge-high    { background:#14532d; color:#4ade80; }
.badge-medium  { background:#1e3a5f; color:#60a5fa; }
.badge-low     { background:#44321a; color:#fbbf24; }
.badge-refusal { background:#450a0a; color:#f87171; }

/* ── Citation <details> ── */
details {
    background:#1e293b; border:1px solid #334155; border-radius:8px;
    padding:0.5rem 0.8rem; margin:0.4rem 0; font-size:0.88rem;
}
details summary {
    cursor:pointer; font-weight:600; color:#a5b4fc;
    padding:0.2rem 0; list-style:none; user-select:none;
}
details summary::-webkit-details-marker { display:none; }
details summary::before { content:'▶ '; font-size:0.65rem; color:#6366f1; }
details[open] summary::before { content:'▼ '; }
.citation-body {
    border-left:3px solid #6366f1; padding:0.6rem 0.8rem;
    margin:0.5rem 0 0.3rem 0; color:#cbd5e1;
    white-space:pre-wrap; font-size:0.85rem; line-height:1.6;
}
.citation-footer { color:#64748b; font-size:0.75rem; margin-top:0.3rem; }

/* ── Chatbot ── */
#chatbot { min-height:460px; }

/* ── Primary buttons ── */
#process-btn, #send-btn {
    background:linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    color:white !important; border:none !important;
    border-radius:8px !important; font-weight:600 !important;
}

/* ── Compare button ── */
#compare-btn {
    background:linear-gradient(135deg,#0f172a,#1e293b) !important;
    border:1px solid #6366f1 !important;
    color:#a5b4fc !important; border-radius:8px !important;
    font-weight:600 !important; transition:all 0.2s ease !important;
    white-space:nowrap !important;
}
#compare-btn:hover {
    background:linear-gradient(135deg,#1e293b,#334155) !important;
    border-color:#818cf8 !important; color:#c4b5fd !important;
}

/* ── Comparison table ── */
.compare-wrap { margin:0.6rem 0 1rem 0; }
.compare-table {
    width:100%; border-collapse:collapse; font-size:0.84rem;
    border:1px solid #334155; border-radius:8px; overflow:hidden;
}
.compare-table th {
    background:#1e293b; color:#a5b4fc; font-weight:600;
    padding:0.65rem 0.9rem; text-align:left;
    border-bottom:2px solid #6366f1;
}
.compare-table td {
    padding:0.65rem 0.9rem; border-bottom:1px solid #334155;
    vertical-align:top; color:#cbd5e1; line-height:1.6;
}
.compare-table td:first-child {
    font-weight:600; color:#94a3b8; white-space:nowrap;
    background:rgba(30,41,59,0.6);
}
.compare-table tbody tr:last-child td { border-bottom:none; }
.compare-table tbody tr:hover td { background:rgba(99,102,241,0.06); }
.compare-table tbody tr:hover td:first-child { background:rgba(30,41,59,0.9); }
"""


# ── Pipeline helpers ───────────────────────────────────────────────────────────

def _load_all_chunks() -> list[dict]:
    if CHUNKS_PATH.exists():
        with CHUNKS_PATH.open(encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_chunks(chunks: list[dict]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def _known_sources() -> list[str]:
    """Return unique source_doc names from the current chunks.json."""
    return sorted({c["source_doc"] for c in _load_all_chunks()})


def process_pdfs(file_objs) -> tuple[str, list[str]]:
    """
    Ingest, chunk, embed, and upsert one or more uploaded PDF files.

    Returns (status_message, list_of_just_processed_source_names).
    Only the PDFs processed in this call are returned — previously
    processed PDFs sitting in chunks.json are intentionally excluded
    so the UI only shows what the user uploaded in this session.
    """
    from ingest import extract_pages
    from chunk import build_sections, chunk_sections
    from embed import generate_embeddings
    from store import upsert_chunks

    if not file_objs:
        return "⚠️ No files selected.", _known_sources()

    processed = []
    errors    = []

    for file_obj in file_objs:
        src_path = Path(file_obj.name)
        dest     = RAW_DIR / src_path.name
        shutil.copy(src_path, dest)

        try:
            pages      = extract_pages(dest)
            sections   = build_sections(pages)
            new_chunks = chunk_sections(sections)

            existing = _load_all_chunks()
            source   = dest.name
            existing = [c for c in existing if c.get("source_doc") != source]
            merged   = existing + new_chunks
            _save_chunks(merged)

            embeddings = generate_embeddings(new_chunks)
            # Remove stale ChromaDB entries for this source before upserting
            # so re-uploaded PDFs never leave orphaned old chunks behind.
            from store import delete_source
            delete_source(source)
            upsert_chunks(new_chunks, embeddings)
            processed.append(dest.name)

        except Exception as exc:
            errors.append(f"{src_path.name}: {exc}")

    all_sources = _known_sources()

    if errors:
        msg = (
            f"✅ Processed: {', '.join(processed)}\n"
            f"❌ Errors: {'; '.join(errors)}"
        )
    else:
        msg = f"✅ Ready — {len(processed)} PDF(s) processed: {', '.join(processed)}"

    # Return only the PDFs processed this call, not all historical sources.
    return msg, processed


# ── Chat formatting helpers ────────────────────────────────────────────────────

def _page_label(meta: dict) -> str:
    s, e = meta["start_page"], meta["end_page"]
    return str(s) if s == e else f"{s}–{e}"


def _confidence_badge(score: float | None) -> str:
    if score is None:
        return ""
    if score >= 4.0:
        return '<span class="badge badge-high">● High confidence</span>'
    if score >= 0.5:
        return '<span class="badge badge-medium">● Moderate confidence</span>'
    return '<span class="badge badge-low">● Low confidence</span>'


def _citations_html(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    n      = len(chunks)
    header = (
        f'<p style="color:#94a3b8;font-size:0.8rem;font-weight:600;'
        f'text-transform:uppercase;letter-spacing:0.05em;margin:0.8rem 0 0.4rem 0;">'
        f'📚 Sources — {n} excerpt{"s" if n != 1 else ""}</p>'
    )
    blocks = []
    for i, chunk in enumerate(chunks, 1):
        meta      = chunk["metadata"]
        page      = _page_label(meta)
        title     = meta["section_title"]
        score     = chunk.get("rerank_score")
        score_str = f" &nbsp;·&nbsp; relevance {score:.2f}" if score is not None else ""
        safe_text = (
            chunk["text"]
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        blocks.append(
            f'<details><summary>'
            f'[{i}] &nbsp;'
            f'<span style="color:#818cf8;font-weight:700">📄 {meta["source_doc"]}</span>'
            f' &nbsp;·&nbsp; Page {page}'
            f' &nbsp;·&nbsp; {title[:45]}{score_str}'
            f'</summary>'
            f'<div class="citation-body">{safe_text}</div>'
            f'<div class="citation-footer">'
            f'{meta["source_doc"]} &nbsp;·&nbsp; Page {page} &nbsp;·&nbsp; {title}'
            f'</div>'
            f'</details>'
        )
    return header + "\n".join(blocks)


def _format_response(result: dict) -> str:
    if result.get("refusal"):
        return (
            '<span class="badge badge-refusal">⊘ Out of scope</span>\n\n'
            + result["refusal"]
        )
    if result.get("error"):
        return f"⚠️ **Error:** {result['error']}"

    answer = result.get("answer", "")
    badge  = _confidence_badge(result.get("top_score"))
    cites  = _citations_html(result.get("context_chunks", []))
    return f"{answer}\n\n{badge}\n\n{cites}"


def _safe(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# ── Compare helpers ────────────────────────────────────────────────────────────

def _compare_per_source(
    query: str,
    reranked: list[dict],
    top_k_per_source: int = 3,
    backend=None,
) -> dict[str, dict]:
    """
    For each source present in the reranked pool, generate a standalone
    answer using only that source's chunks.

    Returns:
        { source_doc: {
            "blocked": bool,
            "reason":  str          (only when blocked),
            "answer":  str,
            "chunks":  list[dict],
            "score":   float | None,
          }
        }
    """
    from guardrails import run_guardrails
    from generate import generate_structured

    by_source: dict[str, list[dict]] = defaultdict(list)
    for chunk in reranked:
        src = chunk.get("metadata", {}).get("source_doc", "unknown")
        by_source[src].append(chunk)

    results: dict[str, dict] = {}

    for source, chunks in by_source.items():
        top_chunks = chunks[:top_k_per_source]

        guard = run_guardrails(query, top_chunks)
        if not guard.allowed:
            results[source] = {
                "blocked": True,
                "reason":  guard.reason,
                "chunks":  top_chunks,
                "score":   top_chunks[0].get("rerank_score") if top_chunks else None,
            }
            continue

        try:
            r = generate_structured(query, top_chunks, backend=backend)
            results[source] = {
                "blocked": False,
                "answer":  r.get("answer", ""),
                "chunks":  r.get("context_chunks", top_chunks),
                "score":   top_chunks[0].get("rerank_score") if top_chunks else None,
                "error":   r.get("error"),
            }
        except Exception as exc:
            results[source] = {
                "blocked": True,
                "reason":  str(exc),
                "chunks":  top_chunks,
                "score":   top_chunks[0].get("rerank_score") if top_chunks else None,
            }

    return results


def _compare_table_html(query: str, per_source: dict[str, dict]) -> str:
    """Build the HTML side-by-side comparison table."""
    sources = list(per_source.keys())
    if not sources:
        return "<p style='color:#94a3b8'>No sources to compare.</p>"

    # ── Column headers ────────────────────────────────────────────────────────
    src_headers = "".join(
        f'<th style="min-width:200px">📄 {_safe(s)}</th>'
        for s in sources
    )

    # ── Cell builders ─────────────────────────────────────────────────────────
    def _na_cell(msg="—"):
        return f'<td style="color:#64748b">{msg}</td>'

    def answer_cell(src: str) -> str:
        r = per_source[src]
        if r.get("blocked"):
            reason = _safe(r.get("reason", "No relevant content in this source"))
            return f'<td><em style="color:#94a3b8">⊘ {reason}</em></td>'
        ans = _safe(r.get("answer") or "N/A")
        if len(ans) > 520:
            ans = ans[:520] + "…"
        return f'<td>{ans}</td>'

    def pages_cell(src: str) -> str:
        r = per_source[src]
        if r.get("blocked"):
            return _na_cell()
        pages = sorted({c["metadata"]["start_page"] for c in r.get("chunks", [])})
        return f'<td>{", ".join(str(p) for p in pages)}</td>'

    def sections_cell(src: str) -> str:
        r = per_source[src]
        if r.get("blocked"):
            return _na_cell()
        seen: list[str] = []
        for c in r.get("chunks", []):
            t = c["metadata"]["section_title"]
            if t not in seen:
                seen.append(t)
        return '<td>' + "<br>".join(_safe(s) for s in seen[:3]) + '</td>'

    def confidence_cell(src: str) -> str:
        r = per_source[src]
        if r.get("blocked"):
            return _na_cell()
        score = r.get("score")
        if score is None:
            return _na_cell()
        if score >= 4.0:
            label, col = "High",     "#4ade80"
        elif score >= 0.5:
            label, col = "Moderate", "#60a5fa"
        else:
            label, col = "Low",      "#fbbf24"
        return (
            f'<td>'
            f'<span style="color:{col};font-weight:600">{label}</span>'
            f'<span style="color:#64748b;font-size:0.8em">&nbsp;({score:.2f})</span>'
            f'</td>'
        )

    row_defs = [
        ("✏️ Answer",     answer_cell),
        ("📄 Pages Used", pages_cell),
        ("📑 Sections",   sections_cell),
        ("🎯 Confidence", confidence_cell),
    ]

    tbody = "".join(
        f'<tr><td>{label}</td>{"".join(cell_fn(s) for s in sources)}</tr>'
        for label, cell_fn in row_defs
    )

    q_short = _safe(query[:60] + ("…" if len(query) > 60 else ""))

    return (
        f'<div class="compare-wrap">'
        f'<p style="color:#94a3b8;font-size:0.8rem;font-weight:600;'
        f'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.7rem">'
        f'📊 Source Comparison &nbsp;·&nbsp; "{q_short}"</p>'
        f'<div style="overflow-x:auto">'
        f'<table class="compare-table">'
        f'<thead><tr><th style="min-width:130px">Aspect</th>{src_headers}</tr></thead>'
        f'<tbody>{tbody}</tbody>'
        f'</table>'
        f'</div>'
        f'</div>'
    )


# ── Core query execution ───────────────────────────────────────────────────────

def run_query(
    query: str,
    active_sources: list[str],
    top_k: int,
    rerank_pool: int,
) -> dict:
    from retrieve import retrieve_per_source, rerank
    from guardrails import run_guardrails
    from generate import generate_structured

    candidates = retrieve_per_source(
        query,
        top_k=top_k,
        candidates_per_source=max(top_k, rerank_pool // 4),
        rerank_pool_size_per_source=rerank_pool,
        active_sources=active_sources if active_sources else None,
    )
    reranked = rerank(query, candidates, top_k=top_k)
    guard    = run_guardrails(query, reranked)

    if not guard.allowed:
        return {
            "answer":        None,
            "context_chunks": [],
            "error":         None,
            "refusal":       guard.reason,
            "top_score":     reranked[0].get("rerank_score") if reranked else None,
            "_all_reranked": reranked,   # kept for compare feature
        }

    result = generate_structured(query, reranked)
    result["refusal"]       = None
    result["top_score"]     = reranked[0].get("rerank_score") if reranked else None
    result["_all_reranked"] = reranked   # kept for compare feature
    return result


# ── Event handlers ─────────────────────────────────────────────────────────────

def on_process(file_objs):
    """Called when user clicks 'Process & Start Chat'."""
    status_msg, just_processed = process_pdfs(file_objs)

    if not just_processed:
        # Processing failed completely — stay on upload screen
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value=status_msg),
            gr.update(choices=[], value=[]),
            [],
        )

    return (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(value=status_msg),
        # Only show files uploaded in this session, not historical chunks.json entries
        gr.update(choices=just_processed, value=just_processed),
        [],
    )


def on_add_more(file_objs, current_sources):
    """Called when user adds more PDFs from the chat screen."""
    if not file_objs:
        return gr.update(), []

    _, just_processed = process_pdfs(file_objs)
    # Merge newly processed PDFs with whatever is already in the session selector
    merged = sorted(set(current_sources or []) | set(just_processed))
    return gr.update(choices=merged, value=merged), []


def chat_fn(
    message: str,
    history: list,
    active_sources: list,
    top_k: int,
    rerank_pool: int,
):
    """
    Handle a chat message.

    Returns:
        history, cleared_query_box, compare_row_update, last_result_state
    """
    if not message.strip():
        return history, "", gr.update(visible=False), None

    history = list(history)
    history.append({"role": "user", "content": message})

    if not active_sources:
        history.append({
            "role": "assistant",
            "content": "⚠️ No sources selected. Please check at least one PDF in the Sources panel.",
        })
        return history, "", gr.update(visible=False), None

    result = run_query(message, active_sources, int(top_k), int(rerank_pool))

    # Decide whether to show the compare button.
    # We need 2+ distinct sources in the full reranked pool (not just context_chunks).
    all_reranked    = result.get("_all_reranked", [])
    sources_present = list({
        c.get("metadata", {}).get("source_doc")
        for c in all_reranked
        if c.get("metadata", {}).get("source_doc")
    })
    show_compare = len(sources_present) >= 2
    new_state    = {"query": message, "reranked": all_reranked} if show_compare else None

    response = _format_response(result)
    history.append({"role": "assistant", "content": response})

    return history, "", gr.update(visible=show_compare), new_state


def compare_fn(last_state: dict | None, history: list):
    """
    Generator: show loading state → run per-source generation → render table.

    Outputs: chatbot, compare_row
    """
    if not last_state:
        yield history, gr.update()
        return

    query    = last_state.get("query", "")
    reranked = last_state.get("reranked", [])

    if not reranked or not query:
        yield history, gr.update()
        return

    history = list(history)
    history.append({
        "role": "assistant",
        "content": "🔍 Comparing sources — generating a separate answer for each PDF…",
    })
    yield history, gr.update(visible=False)   # hide button while working

    per_source = _compare_per_source(query, reranked)
    table_html = _compare_table_html(query, per_source)
    history[-1] = {"role": "assistant", "content": table_html}
    yield history, gr.update(visible=True)    # restore button when done


def back_to_upload():
    return (
        gr.update(visible=True),   # upload_group
        gr.update(visible=False),  # chat_group
        [],                        # chatbot
        gr.update(visible=False),  # compare_row
        None,                      # last_result_state
    )


def clear_chat():
    return [], "", gr.update(visible=False), None


# ── UI ─────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="VaultMind") as demo:

    # ── Shared state ──────────────────────────────────────────────────────────
    # Stores the last query + all reranked chunks for the compare feature.
    last_result_state = gr.State(None)

    # ── Shared header ─────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="vm-header">
        <p class="vm-title">🧠 VaultMind</p>
        <p class="vm-subtitle">
            Retrieval-Augmented Q&amp;A · upload your PDFs and chat with them
        </p>
    </div>
    """)

    # ══════════════════════════════════════════════════════════════════════════
    # Screen 1 — Upload
    # ══════════════════════════════════════════════════════════════════════════
    with gr.Group(visible=True) as upload_group:

        with gr.Column(elem_classes=["upload-card"]):
            gr.Markdown("### 📄 Upload your source PDFs")
            gr.Markdown(
                "Select one or more PDF files. They will be ingested, chunked, "
                "and embedded — then you can chat with them."
            )
            file_input = gr.File(
                label="",
                file_count="multiple",
                file_types=[".pdf"],
                height=160,
            )
            process_btn = gr.Button(
                "Process & Start Chat →",
                elem_id="process-btn",
                variant="primary",
                size="lg",
            )
            status_box = gr.Textbox(
                label="",
                interactive=False,
                visible=True,
                lines=2,
                placeholder="Processing status will appear here…",
            )

    # ══════════════════════════════════════════════════════════════════════════
    # Screen 2 — Chat
    # ══════════════════════════════════════════════════════════════════════════
    with gr.Group(visible=False) as chat_group:

        with gr.Row():

            # ── Sources sidebar ───────────────────────────────────────────────
            with gr.Column(scale=1, min_width=220):

                gr.Markdown("### 📚 Sources")
                source_selector = gr.CheckboxGroup(
                    label="",
                    choices=[],
                    value=[],
                    info="Check PDFs to include · uncheck to exclude from retrieval.",
                )

                add_more_input = gr.File(
                    label="Add more PDFs",
                    file_count="multiple",
                    file_types=[".pdf"],
                    height=100,
                )
                add_more_btn = gr.Button("+ Add & Process", variant="secondary", size="sm")

                gr.Markdown("---")
                gr.Markdown("### ⚙️ Settings")
                top_k = gr.Slider(
                    label="Chunks shown to LLM",
                    minimum=1, maximum=10, value=5, step=1,
                )
                rerank_pool = gr.Slider(
                    label="Rerank pool size",
                    minimum=5, maximum=40, value=20, step=5,
                )
                gr.Markdown("---")
                back_btn  = gr.Button("← Upload new PDFs", variant="secondary", size="sm")
                clear_btn = gr.Button("🗑️ Clear chat",     variant="secondary", size="sm")

            # ── Chat panel ────────────────────────────────────────────────────
            with gr.Column(scale=3):

                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="",
                    height=500,
                    layout="bubble",
                    buttons=["copy"],
                    render_markdown=True,
                    allow_tags=True,
                    avatar_images=(
                        None,
                        "https://api.dicebear.com/7.x/bottts/svg?seed=VaultMind",
                    ),
                )

                with gr.Row():
                    query_box = gr.Textbox(
                        placeholder="Ask a question about your documents…",
                        label="",
                        lines=1,
                        scale=5,
                        show_label=False,
                        submit_btn=False,
                    )
                    send_btn = gr.Button(
                        "Send ➤",
                        elem_id="send-btn",
                        scale=1,
                        variant="primary",
                    )

                # ── Compare row — hidden until 2+ sources are in results ──────
                with gr.Row(visible=False) as compare_row:
                    compare_btn = gr.Button(
                        "🔍 Compare Sources",
                        elem_id="compare-btn",
                        variant="secondary",
                        size="sm",
                        scale=0,
                    )
                    gr.HTML(
                        '<span style="color:#64748b;font-size:0.82rem;'
                        'align-self:center;padding-left:0.6rem;">'
                        'Results found in multiple PDFs — click to compare side-by-side'
                        '</span>'
                    )

    # ── Event wiring ──────────────────────────────────────────────────────────

    process_btn.click(
        on_process,
        inputs=[file_input],
        outputs=[upload_group, chat_group, status_box, source_selector, chatbot],
    )

    add_more_btn.click(
        on_add_more,
        inputs=[add_more_input, source_selector],
        outputs=[source_selector, chatbot],
    )

    # Chat — outputs now include compare_row visibility and the result state
    chat_inputs  = [query_box, chatbot, source_selector, top_k, rerank_pool]
    chat_outputs = [chatbot, query_box, compare_row, last_result_state]
    query_box.submit(chat_fn, inputs=chat_inputs, outputs=chat_outputs)
    send_btn.click(chat_fn,  inputs=chat_inputs, outputs=chat_outputs)

    # Compare button — generator hides button while working, restores after
    compare_btn.click(
        compare_fn,
        inputs=[last_result_state, chatbot],
        outputs=[chatbot, compare_row],
    )

    # Clear chat — also hides compare button and clears state
    clear_btn.click(
        clear_chat,
        outputs=[chatbot, query_box, compare_row, last_result_state],
    )

    # Back to upload — hides compare button and clears state
    back_btn.click(
        back_to_upload,
        outputs=[upload_group, chat_group, chatbot, compare_row, last_result_state],
    )


# ── Launch ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    is_spaces = os.getenv("SPACE_ID") is not None
    demo.launch(
        server_name="0.0.0.0" if is_spaces else None,
        server_port=7860 if is_spaces else None,
        share=False,
        inbrowser=not is_spaces,
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="indigo",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=CSS,
    )

