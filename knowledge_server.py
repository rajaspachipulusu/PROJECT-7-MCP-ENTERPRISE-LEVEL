"""
Knowledge Base Server (MCP)
-----------------------------
Simulates a company's internal knowledge base (policies, guides, wikis)
exposed as a RAG-backed MCP tool.

This ties directly into RAG concepts already known:
  read docs -> chunk -> embed (real semantic embeddings via Ollama's
  nomic-embed-text) -> store -> at query time, embed the question ->
  cosine similarity search -> return best-matching chunk(s) -> the
  calling LLM then uses that content to generate its final answer.

This is the same pipeline shape used in the Project 1 (Multi-Document RAG
Chatbot) repo -- embed -> store -> similarity search -> retrieve, with the
same page-level source citation. Two document formats are supported:

  .txt files  -- indexed as a single chunk per file (short, already-
                 focused documents; no splitting needed).
  .pdf files  -- indexed per-page, then split further into ~800-character
                 paragraph-respecting chunks if a page is long, using
                 pypdf for extraction (identical library choice to the
                 Multi-Document RAG Chatbot project). Every chunk carries
                 its source file and page number for citation.

Indexing model: built once at server startup (same as before), PLUS an
explicit `reindex_knowledge_base` tool that rebuilds the index on demand.
This mirrors how the ticketing server treats state changes as an explicit,
auditable action rather than something that happens silently in the
background -- a manual reindex call flows through the same tool-call /
audit-log path as any other tool, and `last_indexed_at` tells callers how
fresh the index currently is.
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import ollama
from pypdf import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

from mcp.server.fastmcp import FastMCP
from prompt_injection_firewall import sanitize_chunk

# embed_model and log_path come from servers.yaml, passed down as env vars
# by enterprise_client.py when it spawns this server as a subprocess (this
# server has no direct access to servers.yaml -- only the client reads it).
# Defaults here let the server still run standalone / outside the client.
EMBED_MODEL = os.environ.get("KNOWLEDGE_EMBED_MODEL", "nomic-embed-text")
LOG_PATH = os.environ.get("KNOWLEDGE_LOG_PATH")  # e.g. "logs/knowledge_server.log"

logger = logging.getLogger("knowledge_server")
logger.setLevel(logging.INFO)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(logging.Formatter("[KNOWLEDGE_SERVER] %(message)s"))
logger.addHandler(_console_handler)

if LOG_PATH:
    _log_file = Path(LOG_PATH)
    _log_file.parent.mkdir(parents=True, exist_ok=True)
    _file_handler = logging.FileHandler(_log_file, encoding="utf-8")
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s [KNOWLEDGE_SERVER] %(message)s")
    )
    logger.addHandler(_file_handler)

logger.propagate = False

DOCS_DIR = Path(__file__).parent / "data" / "company_docs"
MAX_CHUNK_CHARS = 800  # target chunk size for long PDF pages

mcp = FastMCP(name="knowledge-server")

# Build the "index" once at startup -- mirrors how a real vector DB
# is populated once via an ingestion pipeline, then queried repeatedly.
# Can also be rebuilt on demand via the reindex_knowledge_base tool below.
#
# Each indexed unit is a "chunk": one .txt file = one chunk; one PDF page
# (or a paragraph-sized piece of a long page) = one chunk. _chunk_meta
# holds parallel metadata (source file, page, chunk index) for citation.
_chunk_names: list[str] = []
_chunk_texts: list[str] = []
_chunk_meta: list[dict] = []
_doc_matrix: np.ndarray | None = None
_last_indexed_at: str | None = None


def _embed(texts: list[str]) -> np.ndarray:
    """
    Embed one or more texts using Ollama's nomic-embed-text model.
    Returns an (n_texts, dim) numpy array.
    """
    response = ollama.embed(model=EMBED_MODEL, input=texts)
    return np.array(response["embeddings"], dtype=np.float32)


def _split_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """
    Split text into chunks up to ~max_chars, breaking on paragraph
    boundaries (blank lines) where possible so a chunk doesn't cut a
    sentence in half. Paragraphs longer than max_chars on their own are
    kept intact rather than force-split mid-sentence -- for policy-style
    text this is rare and a slightly oversized chunk is preferable to a
    broken one.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= max_chars or not current:
            current = candidate
        else:
            chunks.append(current)
            current = para

    if current:
        chunks.append(current)

    return chunks or [text.strip()]


def _load_pdf_chunks(path: Path) -> list[tuple[str, str, dict]]:
    """
    Extract text from a PDF page by page (pypdf, same as the
    Multi-Document RAG Chatbot project), then split each page into
    chunks. Returns a list of (chunk_name, chunk_text, metadata) tuples.
    """
    results = []
    reader = PdfReader(str(path))

    for page_num, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if not page_text:
            continue

        page_chunks = _split_into_chunks(page_text)
        for chunk_idx, chunk_text in enumerate(page_chunks):
            suffix = f" (chunk {chunk_idx + 1})" if len(page_chunks) > 1 else ""
            chunk_name = f"{path.stem} — page {page_num}{suffix}"
            results.append((
                chunk_name,
                chunk_text,
                {"source_file": path.name, "page": page_num, "chunk_index": chunk_idx},
            ))

    return results


def _load_documents() -> list[tuple[str, str, dict]]:
    """
    Walk DOCS_DIR and load every .txt and .pdf file into (name, text,
    metadata) chunk tuples. .txt files are indexed whole (one chunk per
    file); .pdf files are indexed per-page via _load_pdf_chunks.
    """
    chunks: list[tuple[str, str, dict]] = []
    if not DOCS_DIR.exists():
        return chunks

    for path in sorted(DOCS_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            chunks.append((path.stem, text, {"source_file": path.name, "page": None, "chunk_index": 0}))

    for path in sorted(DOCS_DIR.glob("*.pdf")):
        try:
            chunks.extend(_load_pdf_chunks(path))
        except Exception as exc:
            logger.error(f"Failed to read PDF '{path.name}': {exc}")

    return chunks


def _build_index() -> dict:
    """
    Rebuild the in-memory embedding index from every .txt and .pdf file
    in DOCS_DIR. Returns a small status dict so both startup logging and
    the reindex_knowledge_base tool can report the same information.
    """
    global _doc_matrix, _chunk_names, _chunk_texts, _chunk_meta, _last_indexed_at

    loaded = _load_documents()
    _chunk_names = [name for name, _, _ in loaded]
    _chunk_texts = [text for _, text, _ in loaded]
    _chunk_meta = [meta for _, _, meta in loaded]

    if not _chunk_texts:
        _doc_matrix = None
        _last_indexed_at = None
        logger.warning("No documents found to index.")
        return {"status": "empty", "documents_indexed": 0, "chunks_indexed": 0}

    try:
        _doc_matrix = _embed(_chunk_texts)
    except Exception as exc:
        # Don't silently fall back -- an embedding failure should be loud,
        # since a stale or missing index would otherwise fail silently on
        # every subsequent search call.
        logger.error(f"Failed to build embedding index: {exc}")
        _doc_matrix = None
        _last_indexed_at = None
        return {"status": "error", "error": str(exc)}

    _last_indexed_at = datetime.now(timezone.utc).isoformat()
    source_files = sorted({meta["source_file"] for meta in _chunk_meta})
    logger.info(
        f"Indexed {len(_chunk_texts)} chunks from {len(source_files)} document(s) "
        f"via {EMBED_MODEL}: {source_files}"
    )
    return {
        "status": "ok",
        "documents_indexed": len(source_files),
        "chunks_indexed": len(_chunk_texts),
        "documents": source_files,
        "last_indexed_at": _last_indexed_at,
    }


@mcp.tool()
def search_knowledge_base(question: str, top_k: int = 3) -> list[dict]:
    """
    Search the company knowledge base (policies, guides) for content
    relevant to the given question, using semantic (embedding-based)
    similarity rather than keyword matching. Returns the top_k most
    relevant chunks with their source file, page number (for PDFs),
    similarity score, and full text, so the calling model can generate
    a grounded, citable answer from them.
    """
    logger.info(f"search_knowledge_base called: question={question!r}")

    if _doc_matrix is None:
        return [{"error": "Knowledge base index is empty. Try reindex_knowledge_base."}]

    try:
        query_vec = _embed([question])
    except Exception as exc:
        logger.error(f"Failed to embed query: {exc}")
        return [{"error": f"Failed to embed query: {exc}"}]

    scores = cosine_similarity(query_vec, _doc_matrix)[0]

    ranked = sorted(
        zip(_chunk_names, _chunk_texts, _chunk_meta, scores),
        key=lambda x: x[3],
        reverse=True,
    )
    top = ranked[:top_k]

    results = []
    for name, text, meta, score in top:
        clean_text, scan_result = sanitize_chunk(text)
        if scan_result["flagged"]:
            logger.warning(
                f"FIREWALL: flagged content in {meta['source_file']} "
                f"(page {meta['page']}), patterns={scan_result['matched_patterns']}"
            )
        results.append({
            "chunk": name,
            "source_file": meta["source_file"],
            "page": meta["page"],
            "similarity_score": round(float(score), 4),
            "content": clean_text,
            "firewall_flagged": scan_result["flagged"],
        })

    return results


@mcp.tool()
def reindex_knowledge_base() -> dict:
    """
    Rebuild the knowledge base embedding index on demand -- call this
    after adding, editing, or removing documents in data/company_docs/
    so subsequent searches reflect the current content. Returns the
    number of documents indexed and a last_indexed_at timestamp.
    """
    logger.info("reindex_knowledge_base called")
    result = _build_index()
    return result


@mcp.resource("knowledge://docs/list")
def list_documents() -> str:
    """Read-only resource listing all source documents currently in the knowledge base."""
    if not _chunk_meta:
        return "No documents indexed."
    source_files = sorted({meta["source_file"] for meta in _chunk_meta})
    return "\n".join(source_files)


@mcp.resource("knowledge://index/status")
def index_status() -> str:
    """Read-only resource reporting index freshness -- when it was last built."""
    if _last_indexed_at is None:
        return "Index status: empty (not yet built or last build failed)."
    source_files = sorted({meta["source_file"] for meta in _chunk_meta})
    return (
        f"Index status: {len(_chunk_meta)} chunks from {len(source_files)} document(s), "
        f"last_indexed_at={_last_indexed_at}"
    )


_build_index()

if __name__ == "__main__":
    logger.info("Starting Knowledge Base MCP server (stdio transport)...")
    mcp.run()