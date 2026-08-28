# """
# Knowledge Base Server (MCP)
# -----------------------------
# Simulates a company's internal knowledge base (policies, guides, wikis)
# exposed as a RAG-backed MCP tool.

# This ties directly into RAG concepts already known:
#   chunk-free here (docs are already short/focused) -> embed (TF-IDF vectors
#   in this lightweight demo) -> store -> at query time, embed the question
#   -> cosine similarity search -> return best-matching doc content -> the
#   calling LLM then uses that content to generate its final answer.

# NOTE: In production you'd swap TfidfVectorizer for a real embedding model
# (e.g. Nomic, OpenAI, etc.) and a real vector DB (Chroma, etc.) -- the
# *pipeline shape* (embed -> store -> similarity search -> retrieve) is
# identical either way. TF-IDF is used here to keep this demo dependency-light.
# """

# import logging
# from pathlib import Path

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# from mcp.server.fastmcp import FastMCP

# logger = logging.getLogger("knowledge_server")
# logger.setLevel(logging.INFO)
# _handler = logging.StreamHandler()
# _handler.setFormatter(logging.Formatter("[KNOWLEDGE_SERVER] %(message)s"))
# logger.addHandler(_handler)
# logger.propagate = False

# DOCS_DIR = Path(__file__).parent / "data" / "company_docs"

# mcp = FastMCP(name="knowledge-server")

# # Build the "index" once at startup -- mirrors how a real vector DB
# # is populated once via an ingestion pipeline, then queried repeatedly.
# _doc_names: list[str] = []
# _doc_texts: list[str] = []
# _vectorizer: TfidfVectorizer | None = None
# _doc_matrix = None


# def _build_index() -> None:
#     global _vectorizer, _doc_matrix, _doc_names, _doc_texts
#     _doc_names = []
#     _doc_texts = []
#     if DOCS_DIR.exists():
#         for path in sorted(DOCS_DIR.glob("*.txt")):
#             _doc_names.append(path.stem)
#             _doc_texts.append(path.read_text(encoding="utf-8"))

#     if not _doc_texts:
#         logger.warning("No documents found to index.")
#         return

#     _vectorizer = TfidfVectorizer(stop_words="english")
#     _doc_matrix = _vectorizer.fit_transform(_doc_texts)
#     logger.info(f"Indexed {len(_doc_texts)} documents: {_doc_names}")


# @mcp.tool()
# def search_knowledge_base(question: str, top_k: int = 2) -> list[dict]:
#     """
#     Search the company knowledge base (policies, guides) for content
#     relevant to the given question. Returns the top_k most relevant
#     documents with their similarity score and full text, so the
#     calling model can generate a grounded answer from them.
#     """
#     logger.info(f"search_knowledge_base called: question={question!r}")
#     if _vectorizer is None or _doc_matrix is None:
#         return [{"error": "Knowledge base index is empty."}]

#     query_vec = _vectorizer.transform([question])
#     scores = cosine_similarity(query_vec, _doc_matrix)[0]

#     ranked = sorted(zip(_doc_names, _doc_texts, scores), key=lambda x: x[2], reverse=True)
#     top = ranked[:top_k]

#     return [
#         {"document": name, "similarity_score": round(float(score), 4), "content": text}
#         for name, text, score in top
#     ]


# @mcp.resource("knowledge://docs/list")
# def list_documents() -> str:
#     """Read-only resource listing all documents currently in the knowledge base."""
#     if not _doc_names:
#         return "No documents indexed."
#     return "\n".join(_doc_names)


# _build_index()

# if __name__ == "__main__":
#     logger.info("Starting Knowledge Base MCP server (stdio transport)...")
#     mcp.run()



"""
Knowledge Base Server (MCP)
-----------------------------
Simulates a company's internal knowledge base (policies, guides, wikis)
exposed as a RAG-backed MCP tool.

This ties directly into RAG concepts already known:
  chunk-free here (docs are already short/focused) -> embed (real semantic
  embeddings via Ollama's nomic-embed-text) -> store -> at query time,
  embed the question -> cosine similarity search -> return best-matching
  doc content -> the calling LLM then uses that content to generate its
  final answer.

This is the same pipeline shape used in the Project 1 (Multi-Document RAG
Chatbot) repo -- embed -> store -> similarity search -> retrieve -- just
applied here to short policy docs instead of chunked PDF pages.

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
from sklearn.metrics.pairwise import cosine_similarity

from mcp.server.fastmcp import FastMCP

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

mcp = FastMCP(name="knowledge-server")

# Build the "index" once at startup -- mirrors how a real vector DB
# is populated once via an ingestion pipeline, then queried repeatedly.
# Can also be rebuilt on demand via the reindex_knowledge_base tool below.
_doc_names: list[str] = []
_doc_texts: list[str] = []
_doc_matrix: np.ndarray | None = None
_last_indexed_at: str | None = None


def _embed(texts: list[str]) -> np.ndarray:
    """
    Embed one or more texts using Ollama's nomic-embed-text model.
    Returns an (n_texts, dim) numpy array.
    """
    response = ollama.embed(model=EMBED_MODEL, input=texts)
    return np.array(response["embeddings"], dtype=np.float32)


def _build_index() -> dict:
    """
    Rebuild the in-memory embedding index from every .txt file in
    DOCS_DIR. Returns a small status dict so both startup logging and
    the reindex_knowledge_base tool can report the same information.
    """
    global _doc_matrix, _doc_names, _doc_texts, _last_indexed_at

    _doc_names = []
    _doc_texts = []

    if DOCS_DIR.exists():
        for path in sorted(DOCS_DIR.glob("*.txt")):
            _doc_names.append(path.stem)
            _doc_texts.append(path.read_text(encoding="utf-8"))

    if not _doc_texts:
        _doc_matrix = None
        _last_indexed_at = None
        logger.warning("No documents found to index.")
        return {"status": "empty", "documents_indexed": 0}

    try:
        _doc_matrix = _embed(_doc_texts)
    except Exception as exc:
        # Don't silently fall back -- an embedding failure should be loud,
        # since a stale or missing index would otherwise fail silently on
        # every subsequent search call.
        logger.error(f"Failed to build embedding index: {exc}")
        _doc_matrix = None
        _last_indexed_at = None
        return {"status": "error", "error": str(exc)}

    _last_indexed_at = datetime.now(timezone.utc).isoformat()
    logger.info(
        f"Indexed {len(_doc_texts)} documents via {EMBED_MODEL}: {_doc_names}"
    )
    return {
        "status": "ok",
        "documents_indexed": len(_doc_texts),
        "documents": list(_doc_names),
        "last_indexed_at": _last_indexed_at,
    }


@mcp.tool()
def search_knowledge_base(question: str, top_k: int = 2) -> list[dict]:
    """
    Search the company knowledge base (policies, guides) for content
    relevant to the given question, using semantic (embedding-based)
    similarity rather than keyword matching. Returns the top_k most
    relevant documents with their similarity score and full text, so the
    calling model can generate a grounded answer from them.
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

    ranked = sorted(zip(_doc_names, _doc_texts, scores), key=lambda x: x[2], reverse=True)
    top = ranked[:top_k]

    return [
        {"document": name, "similarity_score": round(float(score), 4), "content": text}
        for name, text, score in top
    ]


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
    """Read-only resource listing all documents currently in the knowledge base."""
    if not _doc_names:
        return "No documents indexed."
    return "\n".join(_doc_names)


@mcp.resource("knowledge://index/status")
def index_status() -> str:
    """Read-only resource reporting index freshness -- when it was last built."""
    if _last_indexed_at is None:
        return "Index status: empty (not yet built or last build failed)."
    return f"Index status: {len(_doc_names)} documents, last_indexed_at={_last_indexed_at}"


_build_index()

if __name__ == "__main__":
    logger.info("Starting Knowledge Base MCP server (stdio transport)...")
    mcp.run()