"""Streamlit chat UI. Talks to the FastAPI backend over HTTP."""

import html
import itertools
import json
import os
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
import streamlit as st

# 127.0.0.1 rather than localhost: on Windows, "localhost" tries IPv6 first and
# adds ~2 s to every request because uvicorn only listens on IPv4 by default.
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

# Single-process hosts (e.g. Streamlit Community Cloud) can't run uvicorn separately,
# so with EMBEDDED_API=true the UI starts the FastAPI backend in a background thread.
EMBEDDED_API = os.getenv("EMBEDDED_API", "").lower() in {"1", "true", "yes"}

SUGGESTED_QUESTIONS = [
    "Summarize the key ideas of these documents",
    "What problem does each document address?",
    "What are the main results or conclusions?",
    "Explain the core method in simple terms",
]

st.set_page_config(page_title="DocuChat", page_icon="📄", layout="wide")
st.markdown(
    f"<style>{(Path(__file__).parent / 'styles.css').read_text(encoding='utf-8')}</style>",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []  # each: {"role", "content", "sources"}


@st.cache_resource(show_spinner="Starting the backend and loading the embedding model…")
def start_embedded_api() -> None:
    import uvicorn

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.main import app

    url = urlparse(API_URL)
    server = uvicorn.Server(uvicorn.Config(app, host=url.hostname, port=url.port or 8000))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("The embedded API failed to start; check the logs.")
        time.sleep(0.5)


if EMBEDDED_API:
    start_embedded_api()


def api(method: str, path: str, timeout: float = 30, **kwargs):
    response = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    response.raise_for_status()
    return response.json() if response.content else None


def html_block(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚 {len(sources)} sources"):
        cards = []
        for s in sources:
            snippet = html.escape(" ".join(s["snippet"].split()))
            ellipsis = "…" if len(s["snippet"]) >= 300 else ""
            cards.append(
                f'<div class="dc-src"><div class="dc-src-head">'
                f'<span class="dc-src-num">{s["index"]}</span>'
                f'<span class="dc-src-file">{html.escape(s["filename"])}</span>'
                f'<span class="dc-src-page">p. {s["page"]}</span></div>'
                f'<div class="dc-src-snippet">{snippet}{ellipsis}</div></div>'
            )
        html_block("".join(cards))


def stream_answer(question: str, history: list[dict], doc_ids: list[str], sources: list[dict]):
    """Yield answer tokens from the NDJSON stream; collects sources as a side effect."""
    payload = {"question": question, "history": history, "doc_ids": doc_ids}
    with requests.post(f"{API_URL}/chat/stream", json=payload, stream=True, timeout=300) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            event = json.loads(line)
            if event["type"] == "sources":
                sources.extend(event["data"])
            elif event["type"] == "token":
                yield event["data"]
            elif event["type"] == "error":
                yield f"\n\n⚠️ Error: {event['data']}"


def set_all_selected(doc_ids: list[str], value: bool) -> None:
    for doc_id in doc_ids:
        st.session_state[f"select_{doc_id}"] = value


def ask(question: str) -> None:
    st.session_state.pending_question = question


# ---------------- Sidebar: brand, upload, documents ----------------
with st.sidebar:
    html_block(
        '<div class="dc-brand"><div class="dc-logo">📄</div><div>'
        '<div class="dc-brand-name">DocuChat</div>'
        '<div class="dc-brand-sub">Multi-PDF RAG assistant</div></div></div>'
    )

    try:
        health = api("GET", "/health", timeout=5)
    except requests.RequestException:
        html_block('<div class="dc-status down"><span class="dc-dot"></span>API offline</div>')
        st.error(f"Cannot reach the API at {API_URL}. Start it with `uvicorn app.main:app`.")
        st.button("Retry", icon=":material/refresh:")
        st.stop()
    html_block('<div class="dc-status"><span class="dc-dot"></span>API connected</div>')

    st.divider()
    html_block('<div class="dc-side-label">Upload</div>')
    with st.form("upload", clear_on_submit=True):
        uploads = st.file_uploader(
            "Choose PDF files", type="pdf", accept_multiple_files=True, label_visibility="collapsed"
        )
        submitted = st.form_submit_button("Upload & index", type="primary", icon=":material/upload:")
    if submitted and uploads:
        files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in uploads]
        with st.spinner(f"Indexing {len(files)} file(s)…"):
            try:
                results = api("POST", "/documents", files=files, timeout=900)
            except requests.RequestException as exc:
                st.error(f"Upload failed: {exc}")
                results = []
        for r in results:
            if r["status"] == "indexed":
                st.toast(f"Indexed **{r['filename']}** ({r['document']['chunks']} chunks)", icon="✅")
            elif r["status"] == "duplicate":
                st.toast(f"**{r['filename']}** is already indexed", icon="ℹ️")
            else:
                st.error(f"{r['filename']}: {r['error']}")

    try:
        documents = api("GET", "/documents")
    except requests.RequestException as exc:
        st.error(f"Could not load documents from the API: {exc}")
        st.button("Retry", icon=":material/refresh:")
        st.stop()

    all_ids = [d["doc_id"] for d in documents]
    for doc_id in all_ids:
        st.session_state.setdefault(f"select_{doc_id}", True)

    html_block(f'<div class="dc-side-label">Documents · {len(documents)}</div>')
    if documents:
        left, right = st.columns(2)
        left.button("Select all", on_click=set_all_selected, args=(all_ids, True), type="tertiary")
        right.button("Select none", on_click=set_all_selected, args=(all_ids, False), type="tertiary")

    selected_ids = []
    for doc in documents:
        with st.container(key=f"doc_{doc['doc_id']}"):
            name_col, delete_col = st.columns([6, 1], vertical_alignment="center")
            if name_col.checkbox(doc["filename"], key=f"select_{doc['doc_id']}"):
                selected_ids.append(doc["doc_id"])
            if delete_col.button(
                "", icon=":material/delete:", key=f"delete_{doc['doc_id']}", type="tertiary", help="Delete"
            ):
                try:
                    api("DELETE", f"/documents/{doc['doc_id']}")
                except requests.RequestException as exc:
                    st.error(f"Delete failed: {exc}")
                else:
                    st.session_state.pop(f"select_{doc['doc_id']}", None)
                    st.rerun()
            name_col.caption(f"{doc['pages']} pages · {doc['chunks']} chunks")

    st.divider()
    if st.button("New chat", icon=":material/add_comment:", disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.rerun()
    html_block('<div class="dc-footer">FastAPI · LangChain · ChromaDB · Streamlit</div>')


# ---------------- Main: hero, stats, chat ----------------
html_block(
    '<div class="dc-eyebrow">⚡ Retrieval-Augmented Generation</div>'
    '<div class="dc-title">Chat with your <span class="dc-grad">PDFs</span></div>'
    '<p class="dc-sub">Ask questions across multiple documents and get answers with page-level citations.</p>'
    '<div class="dc-chips">'
    f'<span class="dc-chip">🧠 LLM · <b>{html.escape(health["llm_model"])}</b></span>'
    f'<span class="dc-chip">🔎 Embeddings · <b>{html.escape(health["embedding_model"].split("/")[-1])}</b></span>'
    f'<span class="dc-chip">☁️ Provider · <b>{html.escape(health["llm_provider"])}</b></span>'
    "</div>"
)

if not documents:
    html_block(
        '<div class="dc-steps">'
        '<div class="dc-step"><div class="dc-step-num">1</div><div class="dc-step-title">Upload PDFs</div>'
        '<div class="dc-step-text">Add one or more PDFs from the sidebar. They are chunked, embedded and stored in ChromaDB.</div></div>'
        '<div class="dc-step"><div class="dc-step-num">2</div><div class="dc-step-title">Pick documents</div>'
        '<div class="dc-step-text">Tick the documents you want to search. Untick any to leave them out.</div></div>'
        '<div class="dc-step"><div class="dc-step-num">3</div><div class="dc-step-title">Ask anything</div>'
        '<div class="dc-step-text">Get grounded answers with citations to the exact file and page.</div></div>'
        "</div>"
    )
    st.stop()

selected_docs = [d for d in documents if d["doc_id"] in selected_ids]
stats = [
    ("Documents", len(documents)),
    ("Selected", len(selected_docs)),
    ("Pages", sum(d["pages"] for d in selected_docs)),
    ("Chunks", sum(d["chunks"] for d in selected_docs)),
]
html_block(
    '<div class="dc-stats">'
    + "".join(
        f'<div class="dc-stat"><div class="dc-stat-label">{label}</div><div class="dc-stat-value">{value:,}</div></div>'
        for label, value in stats
    )
    + "</div>"
)

prompt = st.chat_input("Ask a question about your documents…")
question = prompt or st.session_state.pop("pending_question", None)

if not st.session_state.messages and not question:
    html_block('<div class="dc-section-title">Try asking</div>')
    with st.container(key="suggestions"):
        cols = st.columns(2)
        for i, suggestion in enumerate(SUGGESTED_QUESTIONS):
            cols[i % 2].button(suggestion, key=f"suggest_{i}", on_click=ask, args=(suggestion,))

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "✨"):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))

if question:
    if not selected_ids:
        st.warning("Select at least one document in the sidebar.")
        st.stop()

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="✨"):
        sources: list[dict] = []
        try:
            tokens = stream_answer(question, history, selected_ids, sources)
            # Retrieval happens before the first token arrives, so show progress until then.
            with st.spinner("Searching your documents…"):
                first = next(tokens, "")
            answer = st.write_stream(itertools.chain([first], tokens))
        except requests.RequestException as exc:
            answer = f"⚠️ Request failed: {exc}"
            st.error(answer)
        render_sources(sources)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
