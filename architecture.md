# Architecture: Hybrid RAG System

**Stack:** Python · Ollama · ChromaDB · BM25 · Streamlit  
**Files:** `loaders.py` · `rag_system.py` · `app.py` · `requirements.txt`

---

## 1. System Overview

This is a fully local Retrieval-Augmented Generation (RAG) system with hybrid retrieval. "Hybrid" means two independent search strategies run in parallel for every query — dense vector search via ChromaDB and sparse keyword search via BM25 — and their results are merged using Reciprocal Rank Fusion (RRF) before being passed to the language model.

No data leaves the machine. No API keys required. All inference runs through a local Ollama server.

---

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         Streamlit UI  (app.py)                            │
│                                                                            │
│  Sidebar                              Main panel                          │
│  ─────────────────────────────        ─────────────────────────────────   │
│  File uploader (.pdf .docx .txt .md)  Chat history (question + answer)    │
│  URL text input                       Chat input box                       │
│  Chunk count metric                   Retrieved chunks expander            │
│  Source list with type icons          (source + 400-char text preview)     │
│  Clear knowledge base button                                               │
└──────────┬────────────────────────────────────────┬────────────────────────┘
           │ bytes / URL string                     │ question string
           ▼                                        ▼
┌──────────────────────┐              ┌─────────────────────────────────────┐
│   loaders.py         │              │   rag_system.py — ask()             │
│                      │              │                                      │
│   load_pdf()         │              │   1. embed_text(question)           │
│   load_docx()        │              │   2. store.search(q, embedding)     │
│   load_txt()         │              │      ├─ _vector_search()  ──────┐   │
│   load_url()         │              │      ├─ _bm25_search()    ──────┤   │
│   load_document()    │              │      └─ _rrf()  ◄──────────────┘   │
│   (dispatcher)       │              │   3. Build grounded prompt          │
└──────────┬───────────┘              │   4. ollama.chat(llama3.2)         │
           │ plain text               └─────────────────┬───────────────────┘
           ▼                                            │
┌──────────────────────────────────────┐               │
│   rag_system.py — index_text()       │               │
│                                      │               │
│   chunk_text()  →  embed_text()      │               │
│         │                │           │               │
│         ▼                ▼           │               │
│   HybridStore.add()                  │               │
│         │                            │               │
│    ┌────┴──────────────────────┐     │               │
│    │                           │     │               │
│    ▼                           ▼     │               │
│  ChromaDB              BM25Okapi     │               │
│  (persisted)           (in-memory)   │               │
└──────────────────────────────────────┘               │
           │                                           │
           ▼                                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         Ollama Local Server                                │
│                                                                            │
│   nomic-embed-text  (embedding — called at index time and query time)     │
│   llama3.2          (chat — called once per user question)                │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. File Responsibilities

### `loaders.py` — Text Extraction Layer

Single responsibility: given a file (as raw bytes) or a URL (as a string), return plain text. Nothing else.

| Function | Input | Library | Notes |
|---|---|---|---|
| `load_txt(data)` | `bytes` | stdlib | UTF-8 decode, ignore errors |
| `load_pdf(data)` | `bytes` | PyMuPDF (`fitz`) | No poppler dependency; iterates pages |
| `load_docx(data)` | `bytes` | `python-docx` | Extracts non-empty paragraphs only |
| `load_url(url)` | `str` | `trafilatura` + `requests` | Primary: trafilatura fetch+extract; fallback: raw GET then extract |
| `load_document(source, filename, url)` | `bytes` or `str` | — | Dispatcher: routes by extension or `url` flag |

`load_document()` returns `(text, label)` where `label` is a human-readable source name (filename or `domain/path` for URLs). The label is stored as metadata in ChromaDB and shown in the UI.

**Dependency direction:** `loaders.py` knows nothing about chunking, embedding, or the store. It is only imported by `app.py` and `rag_system.py`'s CLI entry point.

---

### `rag_system.py` — RAG Engine

The core processing layer. Owns chunking, embedding, the dual-index store, RRF fusion, and LLM prompting. No UI imports.

#### Configuration constants

| Constant | Default | Purpose |
|---|---|---|
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `CHAT_MODEL` | `llama3.2` | Ollama chat model |
| `CHUNK_SIZE` | 500 chars | Target size of each text chunk |
| `CHUNK_OVERLAP` | 50 chars | Overlap between consecutive chunks |
| `TOP_K` | 5 | Candidates fetched from each retriever before fusion |
| `FINAL_TOP_K` | 3 | Chunks passed to the LLM after RRF |
| `RRF_K` | 60 | RRF smoothing constant |
| `DB_PATH` | `chroma_db` | Directory for ChromaDB persistence |

#### `chunk_text(text)`

Fixed-size overlapping character windows. Overlap prevents sentences from being split entirely into separate chunks — the tail of one chunk becomes the head of the next. Empty chunks are filtered out.

#### `embed_text(text)`

One-line wrapper around `ollama.embeddings(model=EMBED_MODEL, prompt=text)`. Returns a `list[float]`. Called once per chunk at index time and once per query at search time. Kept separate from `HybridStore` so it can be mocked in tests.

#### `_tokenize(text)`

BM25-specific: lowercase, strip all punctuation with a regex, split on whitespace. Returns `list[str]`. Designed for speed and simplicity — BM25 does not benefit from subword tokenization.

#### `HybridStore`

The central data structure. Maintains two indexes in sync.

**Dense index (ChromaDB):**
- `chromadb.PersistentClient` writes to `chroma_db/` on disk
- Collection configured with `hnsw:space: cosine` — search uses cosine similarity, HNSW graph indexing
- Survives app restarts automatically; no manual save/load needed
- Each record stores: UUID id, embedding vector, document text, `{"source": label}` metadata

**Sparse index (BM25Okapi):**
- Lives entirely in memory as a `rank_bm25.BM25Okapi` instance
- Rebuilt from ChromaDB on every `__init__` call via `_rebuild_bm25()`
- `_rebuild_bm25()` calls `collection.get(include=["documents", "metadatas"])` to fetch all stored texts, tokenizes them, and constructs a fresh `BM25Okapi`
- This makes ChromaDB the single source of truth; BM25 is always a derived view

**`add(text, source, embedding)`:**
Writes to both indexes atomically (within the same call):
1. `collection.add()` — persisted to ChromaDB immediately
2. Appends to `_corpus` list and reconstructs `BM25Okapi` — the rebuild cost is O(n) but acceptable for typical knowledge base sizes

**`search(query, query_embedding, top_k)`:**
1. Calls `_vector_search(query_embedding, top_k=candidate_k)` — queries ChromaDB, returns top results ranked by cosine similarity
2. Calls `_bm25_search(query, top_k=candidate_k)` — scores all corpus chunks with BM25, filters zero-score results (no shared vocabulary), returns top results ranked by BM25 score
3. Calls `_rrf(vector_results, bm25_results)` — merges both ranked lists and returns `FINAL_TOP_K` best chunks

`candidate_k` is `max(top_k * 2, TOP_K)` — we deliberately over-fetch from each retriever so RRF has more material to work with before pruning to the final count.

---

## 4. Reciprocal Rank Fusion — Deep Dive

RRF is the fusion algorithm that combines the vector and BM25 ranked lists into a single ranking without requiring score normalization.

### Formula

```
RRF_score(d) = Σ  1 / (k + rank(d, list_i))
               i
```

- `d` is a document chunk
- `rank(d, list_i)` is the 0-indexed rank of `d` in result list `i` (0 = first result)
- `k` is a smoothing constant (60 in our implementation, from the original paper)
- The sum is over all result lists where `d` appears

### Why it works

The `1 / (k + rank)` scoring means:
- Rank 0 (best) contributes `1/61 ≈ 0.0164`
- Rank 4 contributes `1/65 ≈ 0.0154`
- Rank 99 contributes `1/159 ≈ 0.0063`

The `k=60` in the denominator compresses the contribution difference between high ranks — rank 0 is not dramatically more valuable than rank 5. This makes the algorithm robust to one retriever having a slightly wrong ordering.

A chunk appearing at rank 0 in the vector list AND rank 0 in the BM25 list scores `2 × 0.0164 = 0.0328`. A chunk appearing at rank 0 in only one list scores `0.0164`. The double-appearance boost is the mechanism that rewards chunks both retrievers agree on.

**Key property: no score normalization needed.** BM25 scores are in the range 0–20+. Cosine similarities are 0–1. RRF never looks at these raw scores — it only uses rank positions. This is why the algorithm generalizes across completely different retrieval methods with zero tuning.

### Implementation

```
_rrf(vector_results, bm25_results, k=60, final_top_k=3):
    scores = {}
    for rank, result in enumerate(vector_results):
        scores[result.text].rrf_score += 1 / (k + rank + 1)

    for rank, result in enumerate(bm25_results):
        scores[result.text].rrf_score += 1 / (k + rank + 1)

    return top final_top_k by descending rrf_score
```

Deduplication is handled by using the chunk text as the dict key — if the same chunk appears in both result lists, its scores accumulate into the same entry.

---

## 5. Ingestion Pipeline (Index Time)

```
User uploads file / pastes URL
          │
          ▼
  load_document()          ← loaders.py
  returns (text, label)
          │
          ▼
  index_text(store, text, label)   ← rag_system.py
          │
          ▼
  chunk_text(text)
  returns list[str] of ~500-char chunks
          │
          ▼  (for each chunk)
  embed_text(chunk)
  → Ollama nomic-embed-text
  → returns list[float]  (embedding vector)
          │
          ▼
  store.add(chunk, label, embedding)
          ├── ChromaDB collection.add()   (persisted to disk)
          └── BM25Okapi rebuilt in-memory
```

**Startup behaviour:** When `HybridStore.__init__` runs, if ChromaDB already has data (from a previous session), `_rebuild_bm25()` re-constructs the BM25 index from those stored documents. The user experiences this as a one-time spinner on app load. After rebuild, both indexes are in sync.

---

## 6. Query Pipeline (Search Time)

```
User types a question
          │
          ▼
  embed_text(question)
  → Ollama nomic-embed-text
  → query_embedding: list[float]
          │
          ├──────────────────────────────────────┐
          ▼                                      ▼
  _vector_search(query_embedding)       _bm25_search(question)
  → ChromaDB cosine HNSW query          → BM25Okapi.get_scores()
  → top-N chunks ranked by similarity   → top-N chunks ranked by keyword score
  → [{"text":…, "source":…}, …]         → [{"text":…, "source":…}, …]
  (zero-score BM25 results discarded)
          │                                      │
          └─────────────────┬────────────────────┘
                            ▼
                    _rrf(vec_results, bm25_results)
                    → merged list, top FINAL_TOP_K chunks
                            │
                            ▼
                  Build grounded prompt:
                  "Answer using ONLY this context…
                  [Source: X] chunk text
                  ---
                  [Source: Y] chunk text
                  Question: {question}"
                            │
                            ▼
                  ollama.chat(llama3.2, prompt)
                            │
                            ▼
                  returns (answer_text, top_chunks)
                            │
                  ┌─────────┴──────────────┐
                  ▼                        ▼
            st.write(answer)    expander: chunk previews + sources
```

---

## 7. Why Each Technology Was Chosen

| Choice | Reason |
|---|---|
| **ChromaDB** over FAISS | Embedded database (no server process), automatic persistence, simple Python API. FAISS is faster at scale but requires manual index serialization. |
| **BM25Okapi** (rank-bm25) | Lightweight, pure Python, no infrastructure. Rebuilt from ChromaDB on startup — ChromaDB is the persistence layer for both indexes. |
| **RRF** over weighted sum | No normalization needed (BM25 and cosine scores are incompatible scales). No weights to tune. Proven to match or beat tuned ensembles on standard benchmarks. |
| **nomic-embed-text** | State-of-the-art retrieval embedding model available via Ollama. Significantly outperforms general-purpose models on document retrieval tasks. |
| **llama3.2** | Capable general-purpose model available via Ollama. Follows the "answer only from context" instruction reliably. |
| **trafilatura** over BeautifulSoup | Purpose-built for main-content extraction from arbitrary web pages. Handles boilerplate removal (navbars, ads, footers) automatically. |
| **PyMuPDF** over pdfminer/pypdf | No poppler system dependency, fast, handles complex PDF layouts well, pure pip install. |
| **Streamlit** | Rapid UI prototyping with built-in file uploader, chat primitives, session state, and spinner components. |

---

## 8. Limitations and Next Steps

| Limitation | Impact | Remedy |
|---|---|---|
| BM25 rebuilt from ChromaDB on every startup | Startup delay proportional to corpus size | Pickle the `BM25Okapi` object to disk; compare its stored chunk count against ChromaDB to detect staleness |
| Fixed-size character chunking | A chunk may cut mid-sentence or mid-concept | Semantic chunking: split where consecutive sentence embeddings diverge in cosine space |
| No reranking after RRF | RRF rank may still include marginally relevant chunks | Cross-encoder reranker (e.g. `cross-encoder/ms-marco-MiniLM`) as a second-stage filter on the top-10 |
| Single-turn answers | LLM has no memory of previous turns in the session | Append `chat_history` as assistant/user turns in the Ollama messages list |
| URL scraping limited to static HTML | JS-heavy SPAs return empty content | Playwright headless browser as an optional backend for `load_url()` |
| No streaming responses | UI freezes while LLM generates | `ollama.chat(stream=True)` + `st.write_stream()` |
| No evaluation | No automated way to detect retrieval regressions | RAGAS faithfulness + relevance scoring after each answer |

---

## 9. Dependency Map

```
app.py
 ├── rag_system.py
 │    ├── ollama          (embedding + chat)
 │    ├── chromadb        (vector persistence)
 │    └── rank_bm25       (keyword search)
 └── loaders.py
      ├── fitz / PyMuPDF  (PDF extraction)
      ├── python-docx     (DOCX extraction)
      ├── trafilatura     (URL content extraction)
      └── requests        (HTTP fallback)
```

`loaders.py` and `rag_system.py` have no dependency on each other. `app.py` imports both. This means the RAG engine can be used from a CLI, a REST API, or any other interface with no changes.

---

## 10. Technology Stack Summary

| Layer | Technology | Version |
|---|---|---|
| UI | Streamlit | ≥ 1.38 |
| LLM + Embeddings | Ollama (`llama3.2`, `nomic-embed-text`) | ≥ 0.3 |
| Vector store | ChromaDB (embedded, HNSW, cosine) | ≥ 0.5 |
| Keyword search | rank-bm25 (`BM25Okapi`) | ≥ 0.2.2 |
| Rank fusion | Reciprocal Rank Fusion (custom, in `rag_system.py`) | — |
| PDF extraction | PyMuPDF (`fitz`) | ≥ 1.24 |
| DOCX extraction | python-docx | ≥ 1.1 |
| URL extraction | trafilatura + requests | ≥ 1.12 |