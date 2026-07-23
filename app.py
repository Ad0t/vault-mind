"""
app.py — VaultMind Streamlit UI

Sidebar : Upload PDFs → Process → Source checkboxes → Settings
Main    : Chat interface with expandable citations, Compare Sources table
"""

import json
import os
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

import streamlit as st
import streamlit.components.v1 as _st_components

# ── Constants ─────────────────────────────────────────────────────────────────
RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
CHUNKS_PATH   = PROCESSED_DIR / "chunks.json"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Page config (must be the FIRST Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="VaultMind — Document Q&A",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif !important; }
#MainMenu, footer, header   { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%);
    border-right: 1px solid #2d2d4e;
}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span { color: #e2e8f0 !important; }

/* ── Sidebar: hide only the CLOSE button so users can't collapse it ── */
/* collapsedControl (the reopen arrow) is intentionally kept visible   */
/* in case the browser has a cached collapsed state on first load.     */
[data-testid="stSidebarCollapseButton"] { display: none !important; }

/* ── Main header ── */
.vm-header { text-align: center; padding: 1.5rem 0 0.5rem 0; }
.vm-header h1 {
    font-size: 2.2rem;
    background: linear-gradient(135deg, #6366f1, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.vm-header p { color: #94a3b8; margin: 0.3rem 0 0 0; font-size: 1rem; }

/* ── Compare table ── */
.compare-wrap { overflow-x: auto; margin-top: 0.5rem; }
.compare-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
.compare-table th {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: #fff !important;
    padding: 10px 14px;
    text-align: left;
    white-space: nowrap;
}
.compare-table td {
    padding: 10px 14px;
    border: 1px solid #2d2d4e;
    vertical-align: top;
    color: #cbd5e1;
}
.compare-table tr:nth-child(even) td { background: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

# ── Force sidebar open — clears cached collapsed state and forces DOM open ──
_st_components.html("""
<script>
(function() {
    var parent = window.parent;
    var pdoc   = parent.document;
    var pstore = parent.localStorage;

    // Strategy 1: clear any Streamlit localStorage keys that store sidebar state.
    // Only do this once per browser session to avoid a reload loop.
    if (!parent.sessionStorage.getItem('vm_sb_fixed')) {
        parent.sessionStorage.setItem('vm_sb_fixed', '1');
        try {
            Object.keys(pstore).forEach(function(k) {
                if (k.toLowerCase().indexOf('sidebar') !== -1 ||
                    k.toLowerCase().indexOf('collapse') !== -1) {
                    pstore.removeItem(k);
                }
            });
        } catch(e) {}
    }

    // Strategy 2: directly force the sidebar element visible via inline style.
    function forceVisible() {
        var sb = pdoc.querySelector('[data-testid="stSidebar"]');
        if (sb) {
            sb.style.setProperty('display',    'block',   'important');
            sb.style.setProperty('visibility', 'visible', 'important');
            sb.style.setProperty('transform',  'none',    'important');
        }
        // Strategy 3: also try clicking the expand toggle as a fallback.
        var btn = pdoc.querySelector('[data-testid="collapsedControl"]');
        if (btn) btn.click();
    }

    // Run immediately, then again after Streamlit finishes rendering.
    forceVisible();
    setTimeout(forceVisible, 300);
    setTimeout(forceVisible, 800);
})();
</script>
""", height=0)

# ── Session-state defaults ─────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "messages":          [],   # list[{"role", "content", "context_chunks"}]
    "processed_sources": [],   # filenames processed in this session
    "active_sources":    [],   # currently checked checkboxes
    "last_reranked":     None, # full reranked pool from last query
    "last_query":        "",   # query string from last successful run
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Helpers: chunk persistence ─────────────────────────────────────────────────

def _load_all_chunks() -> list[dict]:
    if CHUNKS_PATH.exists():
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_chunks(chunks: list[dict]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)


# ── Core business logic ────────────────────────────────────────────────────────

def process_pdfs(uploaded_files) -> tuple[str, list[str]]:
    """
    Ingest, chunk, embed and upsert Streamlit UploadedFile objects.

    Returns (status_message, list_of_just_processed_source_names).
    Only files processed in THIS call are returned — previously ingested
    PDFs in chunks.json are intentionally excluded so the UI source list
    stays scoped to what the user uploaded in this session.
    """
    from ingest import extract_pages
    from chunker import build_sections, chunk_sections
    from embed import generate_embeddings
    from store import upsert_chunks, delete_source

    processed: list[str] = []
    errors:    list[str] = []

    for uf in uploaded_files:
        dest = RAW_DIR / uf.name
        dest.write_bytes(uf.getbuffer())          # save upload to disk

        try:
            pages      = extract_pages(dest)
            sections   = build_sections(pages)
            new_chunks = chunk_sections(sections)

            # Merge with existing chunks, replacing any old version of this PDF
            existing = _load_all_chunks()
            source   = dest.name
            existing = [c for c in existing if c.get("source_doc") != source]
            _save_chunks(existing + new_chunks)

            embeddings = generate_embeddings(new_chunks)
            delete_source(source)                 # remove stale ChromaDB entries
            upsert_chunks(new_chunks, embeddings)
            processed.append(source)

        except Exception as exc:
            errors.append(f"{uf.name}: {exc}")

    if errors:
        msg = (
            f"✅ Processed: {', '.join(processed)}\n"
            f"❌ Errors: {'; '.join(errors)}"
        )
    else:
        msg = f"✅ Ready — {len(processed)} PDF(s) processed: {', '.join(processed)}"

    return msg, processed


def run_query(
    query: str,
    active_sources: list[str],
    top_k: int = 5,
    rerank_pool: int = 20,
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
            "answer":         None,
            "context_chunks": [],
            "error":          None,
            "refusal":        guard.reason,
            "top_score":      reranked[0].get("rerank_score") if reranked else None,
            "_all_reranked":  reranked,
        }

    result = generate_structured(query, reranked)
    result["refusal"]       = None
    result["top_score"]     = reranked[0].get("rerank_score") if reranked else None
    result["_all_reranked"] = reranked
    return result


# ── Compare helpers ────────────────────────────────────────────────────────────

def _compare_per_source(
    query: str,
    reranked: list[dict],
    top_k_per_source: int = 3,
) -> dict[str, dict]:
    from guardrails import run_guardrails
    from generate import generate_structured

    by_source: dict[str, list] = defaultdict(list)
    for chunk in reranked:
        src = chunk.get("metadata", {}).get("source_doc", "unknown")
        by_source[src].append(chunk)

    results: dict[str, dict] = {}
    for source, chunks in by_source.items():
        top   = chunks[:top_k_per_source]
        guard = run_guardrails(query, top)
        if not guard.allowed:
            results[source] = {
                "blocked": True,
                "reason":  guard.reason,
                "chunks":  top,
                "score":   top[0].get("rerank_score") if top else None,
            }
            continue
        try:
            r = generate_structured(query, top)
            results[source] = {
                "blocked": False,
                "answer":  r.get("answer", ""),
                "chunks":  r.get("context_chunks", top),
                "score":   top[0].get("rerank_score") if top else None,
                "error":   r.get("error"),
            }
        except Exception as exc:
            results[source] = {
                "blocked": True,
                "reason":  str(exc),
                "chunks":  top,
                "score":   top[0].get("rerank_score") if top else None,
            }
    return results


def _safe(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _page_label(meta: dict) -> str:
    s, e = meta["start_page"], meta["end_page"]
    return str(s) if s == e else f"{s}–{e}"


def _compare_table_html(query: str, per_source: dict[str, dict]) -> str:
    """Build the HTML side-by-side comparison table."""
    sources = list(per_source.keys())
    if not sources:
        return "<p style='color:#94a3b8'>No sources to compare.</p>"

    src_headers = "".join(
        f'<th style="min-width:200px">📄 {_safe(s)}</th>' for s in sources
    )

    def _na(msg="—"):
        return f'<td style="color:#64748b">{msg}</td>'

    def answer_cell(src: str) -> str:
        r = per_source[src]
        if r.get("blocked"):
            return (
                f'<td><em style="color:#94a3b8">⊘ '
                f'{_safe(r.get("reason", "No relevant content"))}</em></td>'
            )
        ans = _safe(r.get("answer") or "N/A")
        return f'<td>{ans[:520] + "…" if len(ans) > 520 else ans}</td>'

    def pages_cell(src: str) -> str:
        r = per_source[src]
        if r.get("blocked"):
            return _na()
        pages = sorted({c["metadata"]["start_page"] for c in r.get("chunks", [])})
        return f'<td>{", ".join(str(p) for p in pages)}</td>'

    def sections_cell(src: str) -> str:
        r = per_source[src]
        if r.get("blocked"):
            return _na()
        seen: list[str] = []
        for c in r.get("chunks", []):
            t = c["metadata"]["section_title"]
            if t not in seen:
                seen.append(t)
        return "<td>" + "<br>".join(_safe(s) for s in seen[:3]) + "</td>"

    def confidence_cell(src: str) -> str:
        r = per_source[src]
        if r.get("blocked"):
            return _na()
        score = r.get("score")
        if score is None:
            return _na()
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
        f'<tr><td><strong>{label}</strong></td>'
        f'{"".join(fn(s) for s in sources)}</tr>'
        for label, fn in row_defs
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
        f'</table></div></div>'
    )


# ── Render a single assistant message ──────────────────────────────────────────

def _render_message(msg: dict) -> None:
    """Render one chat history entry (user or assistant)."""
    avatar = "🧑" if msg["role"] == "user" else "🧠"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"], unsafe_allow_html=True)

        chunks = msg.get("context_chunks", [])
        if chunks and msg["role"] == "assistant":
            with st.expander(f"📚 {len(chunks)} source excerpt(s)"):
                for i, chunk in enumerate(chunks, 1):
                    meta      = chunk["metadata"]
                    page      = _page_label(meta)
                    score     = chunk.get("rerank_score")
                    score_str = f" · relevance {score:.2f}" if score is not None else ""
                    st.markdown(
                        f"**[{i}] 📄 {meta['source_doc']}**"
                        f" · Page {page}"
                        f" · *{meta['section_title'][:50]}{score_str}*"
                    )
                    preview = chunk["text"]
                    st.caption(preview[:300] + ("…" if len(preview) > 300 else ""))
                    if i < len(chunks):
                        st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 VaultMind")
    st.caption("Hybrid RAG · Multi-source · Citation-grounded")
    st.divider()

    st.markdown("**📄 Upload PDFs**")
    uploaded = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        if st.button("⚡ Process & Start Chat", type="primary", use_container_width=True):
            with st.spinner("Ingesting · Chunking · Embedding…"):
                msg, just_processed = process_pdfs(uploaded)
            if just_processed:
                merged = sorted(
                    set(st.session_state.processed_sources) | set(just_processed)
                )
                st.session_state.processed_sources = merged
                st.session_state.active_sources    = merged
                st.success(msg)
            else:
                st.error(msg)

    # Source checkboxes — only shown after processing
    if st.session_state.processed_sources:
        st.divider()
        st.markdown("**Active Sources**")
        st.caption("Uncheck to exclude from retrieval")
        active: list[str] = []
        for src in st.session_state.processed_sources:
            if st.checkbox(src, value=src in st.session_state.active_sources, key=f"chk_{src}"):
                active.append(src)
        st.session_state.active_sources = active

        st.divider()
        st.markdown("**⚙️ Settings**")
        top_k       = st.slider("Chunks shown to LLM",  1,  10,  5, 1)
        rerank_pool = st.slider("Rerank pool size",      5,  40, 20, 5)
    else:
        top_k, rerank_pool = 5, 20

    if st.session_state.messages:
        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages      = []
            st.session_state.last_reranked = None
            st.session_state.last_query    = ""
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Main area
# ══════════════════════════════════════════════════════════════════════════════

if not st.session_state.processed_sources:
    # ── Welcome screen ─────────────────────────────────────────────────────────
    st.markdown("""
    <div class="vm-header">
        <h1>🧠 VaultMind</h1>
        <p>Retrieval-Augmented Q&A · upload your PDFs and chat with them</p>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.info("👈 Upload your PDFs in the sidebar to get started", icon="📄")
        st.markdown("""
**How it works:**
1. Upload one or more PDFs in the sidebar
2. Click **⚡ Process & Start Chat**
3. Ask any question — VaultMind searches your documents using hybrid BM25 + dense retrieval
4. When answers come from **multiple PDFs**, a **🔍 Compare Sources** button appears
        """)

else:
    # ── Chat screen ────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="vm-header" style="padding:0.5rem 0;">
        <h1 style="font-size:1.8rem;">🧠 VaultMind</h1>
    </div>
    """, unsafe_allow_html=True)

    # Replay chat history
    for msg in st.session_state.messages:
        _render_message(msg)

    # Compare button — visible only when last result spanned 2+ sources
    if st.session_state.last_reranked:
        sources_in_result = {
            c.get("metadata", {}).get("source_doc")
            for c in st.session_state.last_reranked
            if c.get("metadata", {}).get("source_doc")
        }
        if len(sources_in_result) >= 2:
            hint_col, btn_col = st.columns([5, 1])
            with hint_col:
                st.caption(
                    "💡 Results found in multiple PDFs — compare them side-by-side →"
                )
            with btn_col:
                compare_clicked = st.button("🔍 Compare Sources", type="secondary")

            if compare_clicked:
                with st.spinner("Generating per-source answers…"):
                    per_source = _compare_per_source(
                        st.session_state.last_query,
                        st.session_state.last_reranked,
                    )
                table_html = _compare_table_html(
                    st.session_state.last_query, per_source
                )
                st.session_state.messages.append({
                    "role":          "assistant",
                    "content":       table_html,
                    "context_chunks": [],
                })
                st.rerun()

    # ── Chat input ─────────────────────────────────────────────────────────────
    prompt = st.chat_input(
        "Ask a question about your documents…",
        disabled=not st.session_state.active_sources,
    )

    if prompt:
        if not st.session_state.active_sources:
            st.warning("Please select at least one source in the sidebar.")
        else:
            st.session_state.messages.append({
                "role": "user", "content": prompt, "context_chunks": [],
            })

            with st.chat_message("user", avatar="🧑"):
                st.markdown(prompt)

            with st.chat_message("assistant", avatar="🧠"):
                with st.spinner("Searching & generating…"):
                    result = run_query(
                        prompt,
                        st.session_state.active_sources,
                        top_k=top_k,
                        rerank_pool=rerank_pool,
                    )

                # ── Refusal ────────────────────────────────────────────────────
                if result.get("refusal"):
                    content = f"⊘ **Out of scope:** {result['refusal']}"
                    st.warning(content)
                    st.session_state.messages.append({
                        "role": "assistant", "content": content, "context_chunks": [],
                    })
                    st.session_state.last_reranked = result.get("_all_reranked")
                    st.session_state.last_query    = prompt

                # ── Backend error ──────────────────────────────────────────────
                elif result.get("error"):
                    content = f"⚠️ **Error:** {result['error']}"
                    st.error(content)
                    st.session_state.messages.append({
                        "role": "assistant", "content": content, "context_chunks": [],
                    })

                # ── Success ────────────────────────────────────────────────────
                else:
                    answer   = result.get("answer", "No answer generated.")
                    chunks   = result.get("context_chunks", [])
                    reranked = result.get("_all_reranked", [])

                    st.markdown(answer)

                    if chunks:
                        with st.expander(f"📚 {len(chunks)} source excerpt(s)"):
                            for i, chunk in enumerate(chunks, 1):
                                meta      = chunk["metadata"]
                                page      = _page_label(meta)
                                score     = chunk.get("rerank_score")
                                score_str = (
                                    f" · relevance {score:.2f}" if score is not None else ""
                                )
                                st.markdown(
                                    f"**[{i}] 📄 {meta['source_doc']}**"
                                    f" · Page {page}"
                                    f" · *{meta['section_title'][:50]}{score_str}*"
                                )
                                preview = chunk["text"]
                                st.caption(
                                    preview[:300] + ("…" if len(preview) > 300 else "")
                                )
                                if i < len(chunks):
                                    st.divider()

                    st.session_state.messages.append({
                        "role":          "assistant",
                        "content":       answer,
                        "context_chunks": chunks,
                    })
                    st.session_state.last_reranked = reranked
                    st.session_state.last_query    = prompt

            st.rerun()
