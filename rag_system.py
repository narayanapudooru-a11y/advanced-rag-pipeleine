"""
Local Hybrid RAG System — Ollama + ChromaDB + BM25
----------------------------------------------------
Retrieval pipeline:
  1. Chunk incoming text into overlapping segments
  2. Embed each chunk via Ollama (nomic-embed-text)
  3. Store embeddings in ChromaDB (persisted to disk)
  4. On startup, rebuild an in-memory BM25 index from all ChromaDB chunks
  5. At query time, run BOTH ChromaDB vector search AND BM25 keyword search
    6. Merge their ranked results with Reciprocal Rank Fusion (RRF)
    7. Re-rank the fused candidates with a cross-encoder
    8. Pass the top-k chunks to Ollama as grounded context

Why hybrid?
  - Vector search finds semantically similar chunks even with different words
  - BM25 finds exact keyword / proper-noun matches that vectors sometimes miss
  - RRF combines both rank lists without needing to tune score weights

Requirements:
  pip install ollama chromadb rank-bm25 --break-system-packages
  ollama pull nomic-embed-text
  ollama pull llama3.2
"""

import re
import uuid
import ollama
import chromadb
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBED_MODEL      = "nomic-embed-text:latest"
CHAT_MODEL       = "qwen3.5:2b"
CHUNK_SIZE       = 500     # characters per chunk
CHUNK_OVERLAP    = 50      # overlap between consecutive chunks
TOP_K            = 5       # candidates from each retriever before fusion
RERANK_CANDIDATE_K = 20    # fused candidates scored by the cross-encoder
FINAL_TOP_K      = 3       # chunks sent to the LLM after cross-encoder reranking
RRF_K            = 60      # RRF constant (higher = smoother rank blending)
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DB_PATH          = "chroma_db"
COLLECTION_NAME  = "rag_documents"


# ---------------------------------------------------------------------------
# 1. Chunking
# ---------------------------------------------------------------------------
def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping fixed-size character chunks."""
    text = text.strip().replace("\n\n", "\n")
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# 2. Embedding
# ---------------------------------------------------------------------------
def embed_text(text: str) -> list[float]:
    """Return an embedding vector for *text* via the local Ollama server."""
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


# ---------------------------------------------------------------------------
# 3. BM25 helpers
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace — fast BM25 tokens."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


# ---------------------------------------------------------------------------
# 4. HybridStore — ChromaDB (dense) + BM25 (sparse) + RRF fusion
# ---------------------------------------------------------------------------
class HybridStore:
    """
    Dual-index store:
      - ChromaDB PersistentClient for dense vector search (cosine, HNSW)
      - BM25Okapi in-memory index for sparse keyword search
      - Reciprocal Rank Fusion to merge the two result lists

    On __init__, the BM25 index is (re)built from whatever is already in
    ChromaDB, so it survives app restarts transparently.
    """

    def __init__(
        self,
        db_path: str = DB_PATH,
        collection_name: str = COLLECTION_NAME,
        reranker=None,
    ):
        self._reranker = reranker

        # --- Dense store (persisted) ---
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._collection_name = collection_name

        # --- Sparse index (in-memory, rebuilt from ChromaDB on startup) ---
        # _corpus holds the raw texts in the same order as _corpus_ids
        self._corpus: list[str] = []
        self._corpus_ids: list[str] = []
        self._corpus_sources: list[str] = []
        self._bm25: BM25Okapi | None = None
        self._rebuild_bm25()

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _rebuild_bm25(self) -> None:
        """Fetch all documents from ChromaDB and rebuild the BM25 index."""
        if self.collection.count() == 0:
            self._corpus = []
            self._corpus_ids = []
            self._corpus_sources = []
            self._bm25 = None
            return

        result = self.collection.get(include=["documents", "metadatas"])
        self._corpus = result["documents"]
        self._corpus_ids = result["ids"]
        self._corpus_sources = [m.get("source", "unknown") for m in result["metadatas"]]
        tokenized = [_tokenize(doc) for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized)

    def _bm25_search(self, query: str, top_k: int) -> list[dict]:
        """Return top_k BM25 results as {"text", "source"} dicts."""
        if self._bm25 is None or not self._corpus:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        # argsort descending, take top_k
        import numpy as np
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {"text": self._corpus[i], "source": self._corpus_sources[i]}
            for i in ranked
            if scores[i] > 0  # skip zero-score docs (completely irrelevant)
        ]

    def _vector_search(self, query_embedding: list[float], top_k: int) -> list[dict]:
        """Return top_k ChromaDB results as {"text", "source"} dicts."""
        if self.collection.count() == 0:
            return []
        n = min(top_k, self.collection.count())
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        return [
            {"text": doc, "source": meta.get("source", "unknown")}
            for doc, meta in zip(docs, metas)
        ]

    @staticmethod
    def _rrf(
        vector_results: list[dict],
        bm25_results: list[dict],
        k: int = RRF_K,
        final_top_k: int = FINAL_TOP_K,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion.

        score(d) = Σ  1 / (k + rank(d))   (summed over each result list)

        Documents are identified by their text content (de-duplicated).
        Returns the top final_top_k results sorted by descending RRF score.
        """
        scores: dict[str, dict] = {}
        for rank, result in enumerate(vector_results):
            key = result["text"]
            if key not in scores:
                scores[key] = {"rrf_score": 0.0, "data": result}
            scores[key]["rrf_score"] += 1.0 / (k + rank + 1)

        for rank, result in enumerate(bm25_results):
            key = result["text"]
            if key not in scores:
                scores[key] = {"rrf_score": 0.0, "data": result}
            scores[key]["rrf_score"] += 1.0 / (k + rank + 1)

        ranked = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
        return [entry["data"] for entry in ranked[:final_top_k]]

    def _rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """Score fused candidates for query relevance and return the best top_k."""
        if not candidates:
            return []
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(CROSS_ENCODER_MODEL)

        pairs = [(query, candidate["text"]) for candidate in candidates]
        scores = self._reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda item: item[0], reverse=True)
        return [candidate for _, candidate in ranked[:top_k]]

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def add(self, text: str, source: str, embedding: list[float]) -> None:
        """Add a single chunk to both ChromaDB and the in-memory BM25 index."""
        doc_id = str(uuid.uuid4())

        # Dense store
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{"source": source}],
        )

        # Sparse store — append and rebuild (efficient enough for incremental adds)
        self._corpus.append(text)
        self._corpus_ids.append(doc_id)
        self._corpus_sources.append(source)
        tokenized = [_tokenize(doc) for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, query_embedding: list[float], top_k: int = FINAL_TOP_K) -> list[dict]:
        """
        Hybrid search: vector + BM25, fused with RRF, then cross-encoder reranked.

        Args:
            query:           raw query string (for BM25)
            query_embedding: pre-computed embedding vector (for ChromaDB)
            top_k:           number of final results to return

        Returns:
            list of {"text": ..., "source": ...} dicts, best-first.
        """
        candidate_k = max(top_k * 2, TOP_K, RERANK_CANDIDATE_K)
        vec_results  = self._vector_search(query_embedding, top_k=candidate_k)
        bm25_results = self._bm25_search(query, top_k=candidate_k)
        candidates = self._rrf(
            vec_results, bm25_results, k=RRF_K, final_top_k=candidate_k
        )
        return self._rerank(query, candidates, top_k=top_k)

    def count(self) -> int:
        return self.collection.count()

    def list_sources(self) -> list[str]:
        if self.collection.count() == 0:
            return []
        all_meta = self.collection.get(include=["metadatas"])["metadatas"]
        return sorted({m.get("source", "unknown") for m in all_meta})

    def clear(self) -> None:
        """Wipe ChromaDB collection and reset the BM25 index."""
        self.client.delete_collection(self._collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._corpus = []
        self._corpus_ids = []
        self._corpus_sources = []
        self._bm25 = None


# ---------------------------------------------------------------------------
# 5. Embedding a document into the store
# ---------------------------------------------------------------------------
def index_text(store: HybridStore, text: str, source: str) -> int:
    """Chunk *text*, embed each chunk, and add to *store*. Returns chunk count."""
    chunks = chunk_text(text)
    for chunk in chunks:
        embedding = embed_text(chunk)
        store.add(text=chunk, source=source, embedding=embedding)
    return len(chunks)


# ---------------------------------------------------------------------------
# 6. Answering a question
# ---------------------------------------------------------------------------
def ask(store: HybridStore, question: str) -> tuple[str, list[dict]]:
    """
    Hybrid-retrieve context and generate a grounded answer.

    Returns:
        (answer_text, top_chunks)  so the caller can show the sources.
    """
    query_embedding = embed_text(question)
    top_chunks = store.search(question, query_embedding, top_k=FINAL_TOP_K)

    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}" for c in top_chunks
    )

    prompt = (
        "You are a helpful assistant. Answer the question using ONLY the context "
        "below. If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    response = ollama.chat(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"], top_chunks


# ---------------------------------------------------------------------------
# 7. CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import glob, os
    from loaders import load_document

    DOCS_FOLDER = "documents"

    store = HybridStore()

    if store.count() == 0:
        paths = glob.glob(os.path.join(DOCS_FOLDER, "*.txt")) + \
                glob.glob(os.path.join(DOCS_FOLDER, "*.md"))  + \
                glob.glob(os.path.join(DOCS_FOLDER, "*.pdf")) + \
                glob.glob(os.path.join(DOCS_FOLDER, "*.docx"))
        for path in paths:
            with open(path, "rb") as f:
                raw = f.read()
            text, label = load_document(raw, filename=os.path.basename(path))
            n = index_text(store, text, label)
            print(f"  {label} -> {n} chunks")
        print(f"Indexed {store.count()} chunks total.")
    else:
        print(f"Loaded existing index: {store.count()} chunks, BM25 rebuilt.")

    print("\nHybrid RAG ready (BM25 + vectors). Type 'exit' to quit.\n")
    while True:
        question = input("Ask: ").strip()
        if question.lower() in ("exit", "quit", ""):
            break
        answer, chunks = ask(store, question)
        print(f"\nAnswer: {answer}")
        print(f"Sources: {', '.join({c['source'] for c in chunks})}\n")