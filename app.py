"""
Streamlit UI — Hybrid RAG with Ollama + ChromaDB + BM25

Run with:
    streamlit run app.py
"""

import streamlit as st

from rag_system import (
    HybridStore,
    index_text,
    ask,
    EMBED_MODEL,
    CHAT_MODEL,
    FINAL_TOP_K,
    DB_PATH,
)
from loaders import load_document

st.set_page_config(
    page_title="Hybrid RAG — Ollama",
    page_icon="🔎",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "store" not in st.session_state:
    # BM25 is rebuilt from ChromaDB automatically inside HybridStore.__init__
    with st.spinner("Loading index and rebuilding BM25…"):
        st.session_state.store = HybridStore()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of (question, answer, chunks)


# ---------------------------------------------------------------------------
# Shared indexing helper
# ---------------------------------------------------------------------------
def _index_and_report(text: str, label: str) -> int:
    n = index_text(st.session_state.store, text, label)
    return n


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📁 Knowledge base")
    st.caption(
        f"Embed: `{EMBED_MODEL}`  \n"
        f"Chat: `{CHAT_MODEL}`  \n"
        f"Vector DB: ChromaDB @ `{DB_PATH}`  \n"
        f"Keyword: BM25 (in-memory)  \n"
        f"Fusion: Reciprocal Rank Fusion"
    )

    # --- File upload ---
    st.subheader("Upload files")
    uploaded_files = st.file_uploader(
        "PDF, DOCX, TXT or MD",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files and st.button("➕ Index files", use_container_width=True):
        with st.spinner("Extracting text, embedding, indexing…"):
            total = 0
            errors = []
            for f in uploaded_files:
                try:
                    text, label = load_document(f.read(), filename=f.name)
                    total += _index_and_report(text, label)
                except Exception as e:
                    errors.append(f"{f.name}: {e}")
            if total:
                st.success(f"Added {total} chunks from {len(uploaded_files)} file(s).")
            for err in errors:
                st.error(err)

    st.divider()

    # --- URL ingestion ---
    st.subheader("Add a URL")
    url_input = st.text_input(
        "Paste a URL",
        placeholder="https://example.com/article",
        label_visibility="collapsed",
    )
    if url_input and st.button("➕ Index URL", use_container_width=True):
        with st.spinner(f"Fetching and indexing {url_input}…"):
            try:
                text, label = load_document(url_input, url=url_input)
                n = _index_and_report(text, label)
                st.success(f"Added {n} chunks from `{label}`.")
            except Exception as e:
                st.error(f"Failed: {e}")

    st.divider()

    # --- Knowledge base stats ---
    store = st.session_state.store
    n_chunks = store.count()
    sources  = store.list_sources()

    st.metric("Chunks indexed", n_chunks)
    if sources:
        st.caption("Sources:")
        for s in sources:
            # Show a file-type icon based on the extension / URL
            if s.endswith(".pdf"):
                icon = "📄"
            elif s.endswith(".docx"):
                icon = "📝"
            elif s.startswith("http") or "." in s.split("/")[-1]:
                icon = "🌐"
            else:
                icon = "📃"
            st.write(f"{icon} {s}")

    st.divider()
    if st.button("🗑️ Clear knowledge base", use_container_width=True):
        st.session_state.store.clear()
        st.session_state.chat_history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main panel — chat
# ---------------------------------------------------------------------------
st.title("🔎 Hybrid RAG Chat")
st.caption(
    "Answers grounded in your documents via **BM25 + vector search** — everything runs locally."
)

if n_chunks == 0:
    st.info("Upload a file or add a URL in the sidebar to build your knowledge base.")

# Render history
for q, ans, chunks in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(ans)
        if chunks:
            with st.expander(f"Retrieved chunks ({len(chunks)})"):
                for i, c in enumerate(chunks, 1):
                    st.markdown(f"**{i}. Source:** `{c['source']}`")
                    st.caption(c["text"][:400] + ("…" if len(c["text"]) > 400 else ""))
                    st.divider()

# New question
question = st.chat_input("Ask a question about your documents…")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        if n_chunks == 0:
            ans    = "No documents indexed yet — add some files or a URL in the sidebar."
            chunks = []
            st.write(ans)
        else:
            with st.spinner("Hybrid retrieval (BM25 + vectors) → LLM…"):
                try:
                    ans, chunks = ask(st.session_state.store, question)
                except Exception as e:
                    ans    = f"Something went wrong — is Ollama running?\n\n`{e}`"
                    chunks = []

            st.write(ans)

            if chunks:
                used_sources = sorted({c["source"] for c in chunks})
                with st.expander(f"Retrieved chunks ({len(chunks)}) · sources: {', '.join(used_sources)}"):
                    for i, c in enumerate(chunks, 1):
                        st.markdown(f"**{i}. Source:** `{c['source']}`")
                        st.caption(c["text"][:400] + ("…" if len(c["text"]) > 400 else ""))
                        if i < len(chunks):
                            st.divider()

    st.session_state.chat_history.append((question, ans, chunks))