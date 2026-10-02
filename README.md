# advanced-rag-pipeleine
advanced rag pipeline

# Hybrid RAG System — Ollama + ChromaDB + BM25

A fully local Retrieval-Augmented Generation system with **hybrid search** — vector similarity and BM25 keyword search running in parallel, results merged with Reciprocal Rank Fusion. Supports PDF, DOCX, TXT, MD, and live URL ingestion. No API keys. No cloud. Everything runs on your machine.

---

## Why Hybrid?

Pure vector search misses exact-match queries — product codes, error messages, names, version numbers. BM25 misses semantic queries where the wording differs from the document. Running both and fusing their rankings covers each other's blind spots.

```
Query
  ├── ChromaDB vector search  (semantic similarity)  ─┐
  └── BM25 keyword search     (exact term matching)  ─┴── RRF fusion → LLM → Answer
```

---

## Features

- **Hybrid retrieval** — BM25 + ChromaDB dense vectors merged with Reciprocal Rank Fusion
- **Cross-encoder reranking** — rescores the top 20 fused candidates for query relevance
- **Multi-format ingestion** — PDF, DOCX, TXT, MD, and any public URL
- **Fully local** — Ollama runs both the embedding model and the LLM on your machine
- **Persistent index** — ChromaDB stores vectors to disk; BM25 rebuilds from it on startup
- **Transparent answers** — every answer shows the source chunks the model read
- **Streamlit UI** — file uploader, URL input, chat interface, chunk expander

---

## Project Structure

```
.
├── app.py              # Streamlit UI — file upload, URL input, chat
├── rag_system.py       # Core engine — HybridStore, RRF, chunking, embedding, LLM
├── loaders.py          # Text extractors — PDF, DOCX, TXT/MD, URL
├── requirements.txt    # Python dependencies
├── chroma_db/          # Auto-created — ChromaDB persistent vector store
└── documents/          # Optional — drop files here for the CLI mode
```

---

## Prerequisites

**Ollama** must be installed and running before launching the app.

- Download and install: https://ollama.com
- Verify it's running: `ollama list`

**Python 3.11+** is required.

---

## Quick Start

### 1. Pull the models

```bash
ollama pull nomic-embed-text   # embedding model
ollama pull llama3.2           # chat model
```

### 2. Install dependencies

```bash
uv venv
source .venv/bin/activate
uv sync
```

### 3. Launch the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.
The cross-encoder model is downloaded from Hugging Face and cached locally the first time you ask a question; subsequent searches run locally.

### 4. Add documents

In the sidebar:
- **Upload files** — drag in any `.pdf`, `.docx`, `.txt`, or `.md` file and click **Index files**
- **Add a URL** — paste any public URL (Wikipedia, blog post, documentation page) and click **Index URL**

### 5. Ask questions

Type in the chat box. Each answer shows the retrieved source chunks in an expandable panel beneath it.

---

## CLI Mode

Skip the UI and index a local folder from the terminal:

```bash
mkdir documents
cp your_files/*.pdf documents/
python3 rag_system.py
```

First run indexes everything in `documents/` and saves to `chroma_db/`. Subsequent runs load the existing index and start the interactive prompt immediately.

---

## How It Works

### Ingestion pipeline

```
File / URL
    │
    ▼
loaders.py          → extract plain text (PyMuPDF for PDF,
                                          python-docx for DOCX,
                                          trafilatura for URLs)
    │
    ▼
chunk_text()        → split into 500-char overlapping windows (50-char overlap)
    │
    ▼
embed_text()        → Ollama nomic-embed-text → embedding vector
    │
    ▼
HybridStore.add()   → write to ChromaDB (disk) + update BM25 (memory)
```

### Query pipeline

```
Question
    │
    ├── embed_text()          → query vector
    │       │
    │       ▼
    │   ChromaDB.query()      → top-N chunks by cosine similarity
    │
    ├── _tokenize(question)
    │       │
    │       ▼
    │   BM25.get_scores()     → top-N chunks by keyword relevance
    │                           (zero-score chunks discarded)
    │
    ▼
  _rrf(vector_results, bm25_results)
    │
    │   score(chunk) = Σ  1 / (60 + rank)   ← summed over each list
    │
    ▼
  cross-encoder scores (question, chunk) pairs
    │
    ▼
  top-3 reranked chunks  →  grounded prompt  →  ollama.chat  →  Answer
```

### Reciprocal Rank Fusion

RRF combines two ranked lists without needing to normalize their scores (BM25 and cosine similarity are on completely different scales). A chunk appearing in both lists gets score contributions from each, naturally boosting results both retrievers agree on. No weights to tune.

---

## Configuration

All tunable constants are at the top of `rag_system.py`:

| Constant | Default | Description |
|---|---|---|
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `CHAT_MODEL` | `llama3.2` | Ollama chat model |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K` | `5` | Candidates fetched from each retriever |
| `RERANK_CANDIDATE_K` | `20` | Fused candidates scored by the cross-encoder |
| `FINAL_TOP_K` | `3` | Reranked chunks passed to the LLM |
| `RRF_K` | `60` | RRF smoothing constant |
| `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Hugging Face cross-encoder model |
| `DB_PATH` | `chroma_db` | ChromaDB persistence directory |

---

## Dependencies

| Package | Purpose |
|---|---|
| `ollama` | Local LLM and embedding inference |
| `chromadb` | Persistent vector database (HNSW, cosine) |
| `rank-bm25` | BM25Okapi keyword search index |
| `sentence-transformers` | Cross-encoder candidate reranking |
| `pymupdf` | PDF text extraction (no poppler needed) |
| `python-docx` | DOCX paragraph extraction |
| `trafilatura` | URL main-content scraping |
| `requests` | HTTP fallback for URL loader |
| `streamlit` | Web UI |

---

## Resetting the Knowledge Base

**From the UI:** click **🗑️ Clear knowledge base** in the sidebar.

**From the terminal:**

```bash
rm -rf chroma_db/
```

BM25 resets automatically on the next app start since it rebuilds from ChromaDB.

---

## Going Further

Ideas for extending this system:

- **Streaming responses** — `ollama.chat(stream=True)` + `st.write_stream()` for token-by-token output
- **Semantic chunking** — split on topic boundary shifts (cosine distance between consecutive sentence embeddings) instead of fixed character counts
- **Conversation memory** — pass prior turns as message history so the model can handle follow-up questions
- **RAGAS evaluation** — automatically score faithfulness and relevance after each answer

See `ARCHITECTURE.md` for a full technical deep-dive.