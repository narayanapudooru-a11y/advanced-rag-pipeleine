"""
loaders.py — extract plain text from any supported source type.

Supported:
  - .txt / .md   raw text files (file path or in-memory bytes)
  - .pdf         via PyMuPDF (fitz) — no poppler dependency
  - .docx        via python-docx
  - URL          via trafilatura (static pages, blogs, docs, Wikipedia)

All loaders return a plain string.  The caller is responsible for chunking.
"""

import os
import io
import requests
import fitz                        # PyMuPDF
import docx as python_docx         # python-docx
import trafilatura


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------

def load_txt(data: bytes) -> str:
    """Decode raw bytes as UTF-8 text."""
    return data.decode("utf-8", errors="ignore")


def load_pdf(data: bytes) -> str:
    """Extract text from a PDF given its raw bytes."""
    text_parts = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def load_docx(data: bytes) -> str:
    """Extract paragraph text from a DOCX given its raw bytes."""
    doc = python_docx.Document(io.BytesIO(data))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def load_url(url: str, timeout: int = 15) -> str:
    """
    Scrape a URL and extract its main content using trafilatura.
    Falls back to a raw requests fetch if trafilatura returns nothing.
    Raises ValueError if no content could be extracted.
    """
    # trafilatura handles fetch + extraction in one call
    text = trafilatura.fetch_url(url)
    if text:
        extracted = trafilatura.extract(text, include_tables=True, include_comments=False)
        if extracted and extracted.strip():
            return extracted

    # Fallback: raw GET + strip HTML tags manually
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        extracted = trafilatura.extract(resp.text, include_tables=True, include_comments=False)
        if extracted and extracted.strip():
            return extracted
    except Exception as e:
        raise ValueError(f"Could not fetch URL '{url}': {e}")

    raise ValueError(f"No readable content found at '{url}'.")


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def load_document(source: str | bytes, filename: str = "", url: str = "") -> tuple[str, str]:
    """
    Dispatch to the right loader based on the source type.

    Args:
        source:   raw bytes (for uploaded files) OR a URL string.
        filename: original filename, used to detect file type from extension.
        url:      if provided, treat source as a URL string to scrape.

    Returns:
        (text, label) where label is a short human-readable source name.
    """
    if url:
        text = load_url(url)
        # Use the domain + path as the label
        from urllib.parse import urlparse
        parsed = urlparse(url)
        label = parsed.netloc + (parsed.path.rstrip("/") or "")
        return text, label

    if isinstance(source, bytes):
        ext = os.path.splitext(filename.lower())[1]
        if ext == ".pdf":
            return load_pdf(source), filename
        elif ext == ".docx":
            return load_docx(source), filename
        else:
            # .txt, .md, or anything else — treat as plain text
            return load_txt(source), filename

    raise TypeError(f"Unsupported source type: {type(source)}")