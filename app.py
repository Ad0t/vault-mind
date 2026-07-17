"""
app.py — VaultMind Gradio UI

Two-screen flow:
  Screen 1 (Upload)  — drop one or more PDFs → "Process & Chat" button
  Screen 2 (Chat)    — chatbot with a source selector to toggle active PDFs

Run with:
    python app.py
"""

import json
import os
import shutil
import sys
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
    max-width: 560px;
    margin: 2rem auto;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 2rem;
}
.upload-hint { color:#94a3b8; font-size:0.85rem; text-align:center; margin-top:0.5rem; }

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

/* ── Buttons ── */
#process-btn, #send-btn {
    background:linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    color:white !important; border:none !important;
    border-radius:8px !important; font-weight:600 !important;
}
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

    Returns (status_message, list_of_all_source_names).
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
            # ── Ingest ──────────────────────────────────────────────────────
            pages = extract_pages(dest)

            # ── Chunk ───────────────────────────────────────────────────────
            sections   = build_sections(pages)
            new_chunks = chunk_sections(sections)

            # ── Merge with existing chunks.json (upsert by source) ──────────
            existing = _load_all_chunks()
            source   = dest.name
            # Remove old chunks for this source (handles re-upload)
            existing = [c for c in existing if c.get("source_doc") != source]
            merged   = existing + new_chunks
            _save_chunks(merged)

            # ── Embed + upsert into ChromaDB ─────────────────────────────
            import numpy as np
            embeddings = generate_embeddings(new_chunks)
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

    return msg, all_sources


# ── Chat helpers ───────────────────────────────────────────────────────────────

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


def run_query(query: str, active_sources: list[str], top_k: int, rerank_pool: int) -> dict:
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
            "answer": None, "context_chunks": [], "error": None,
            "refusal": guard.reason,
            "top_score": reranked[0].get("rerank_score") if reranked else None,
        }

    result = generate_structured(query, reranked)
    result["refusal"]   = None
    result["top_score"] = reranked[0].get("rerank_score") if reranked else None
    return result


# ── Event handlers ─────────────────────────────────────────────────────────────

def on_process(file_objs):
    """Called when user clicks 'Process & Start Chat'."""
    status_msg, all_sources = process_pdfs(file_objs)

    if not all_sources:
        # Processing failed completely — stay on upload screen
        return (
            gr.update(visible=True),   # upload_group
            gr.update(visible=False),  # chat_group
            gr.update(value=status_msg),
            gr.update(choices=[], value=[]),
            [],
        )

    return (
        gr.update(visible=False),                    # upload_group
        gr.update(visible=True),                     # chat_group
        gr.update(value=status_msg),
        gr.update(choices=all_sources, value=all_sources),
        [],                                          # clear chat history
    )


def on_add_more(file_objs, current_sources):
    """Called when user adds more PDFs from the chat screen."""
    if not file_objs:
        return gr.update(), []

    _, all_sources = process_pdfs(file_objs)
    return gr.update(choices=all_sources, value=all_sources), []


def chat_fn(message: str, history: list, active_sources: list, top_k: int, rerank_pool: int):
    if not message.strip():
        return history, ""

    history = list(history)
    history.append({"role": "user", "content": message})

    if not active_sources:
        history.append({
            "role": "assistant",
            "content": "⚠️ No sources selected. Please check at least one PDF in the Sources panel.",
        })
        return history, ""

    result   = run_query(message, active_sources, int(top_k), int(rerank_pool))
    response = _format_response(result)
    history.append({"role": "assistant", "content": response})
    return history, ""


def back_to_upload():
    return gr.update(visible=True), gr.update(visible=False), []


# ── UI ─────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="VaultMind") as demo:

    # ── Shared header ──────────────────────────────────────────────────────────
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
                    info="Uncheck a PDF to exclude it from retrieval.",
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

    # ── Event wiring ──────────────────────────────────────────────────────────

    # Process uploaded PDFs → switch to chat screen
    process_btn.click(
        on_process,
        inputs=[file_input],
        outputs=[upload_group, chat_group, status_box, source_selector, chatbot],
    )

    # Add more PDFs from chat screen
    add_more_btn.click(
        on_add_more,
        inputs=[add_more_input, source_selector],
        outputs=[source_selector, chatbot],
    )

    # Chat submit
    chat_inputs  = [query_box, chatbot, source_selector, top_k, rerank_pool]
    chat_outputs = [chatbot, query_box]
    query_box.submit(chat_fn, inputs=chat_inputs, outputs=chat_outputs)
    send_btn.click(chat_fn,  inputs=chat_inputs, outputs=chat_outputs)

    # Clear chat
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, query_box])

    # Back to upload
    back_btn.click(
        back_to_upload,
        outputs=[upload_group, chat_group, chatbot],
    )


# ── Launch ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=None,   # auto-assign a free port (avoids conflicts on restart)
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="indigo",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ),
        css=CSS,
    )
