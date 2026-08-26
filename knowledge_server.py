"""
Knowledge Base Server (MCP)
-----------------------------
Simulates a company's internal knowledge base (policies, guides, wikis)
exposed as a RAG-backed MCP tool.

This ties directly into RAG concepts already known:
  chunk-free here (docs are already short/focused) -> embed (TF-IDF vectors
  in this lightweight demo) -> store -> at query time, embed the question
  -> cosine similarity search -> return best-matching doc content -> the
  calling LLM then uses that content to generate its final answer.

NOTE: In production you'd swap TfidfVectorizer for a real embedding model
(e.g. Nomic, OpenAI, etc.) and a real vector DB (Chroma, etc.) -- the
*pipeline shape* (embed -> store -> similarity search -> retrieve) is
identical either way. TF-IDF is used here to keep this demo dependency-light.
"""

import logging
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("knowledge_server")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("[KNOWLEDGE_SERVER] %(message)s"))
logger.addHandler(_handler)
logger.propagate = False

DOCS_DIR = Path(__file__).parent / "data" / "company_docs"

mcp = FastMCP(name="knowledge-server")

# Build the "index" once at startup -- mirrors how a real vector DB
# is populated once via an ingestion pipeline, then queried repeatedly.
_doc_names: list[str] = []
_doc_texts: list[str] = []
_vectorizer: TfidfVectorizer | None = None
_doc_matrix = None


def _build_index() -> None:
    global _vectorizer, _doc_matrix, _doc_names, _doc_texts
    _doc_names = []
    _doc_texts = []
    if DOCS_DIR.exists():
        for path in sorted(DOCS_DIR.glob("*.txt")):
            _doc_names.append(path.stem)
            _doc_texts.append(path.read_text(encoding="utf-8"))

    if not _doc_texts:
        logger.warning("No documents found to index.")
        return

    _vectorizer = TfidfVectorizer(stop_words="english")
    _doc_matrix = _vectorizer.fit_transform(_doc_texts)
    logger.info(f"Indexed {len(_doc_texts)} documents: {_doc_names}")


@mcp.tool()
def search_knowledge_base(question: str, top_k: int = 2) -> list[dict]:
    """
    Search the company knowledge base (policies, guides) for content
    relevant to the given question. Returns the top_k most relevant
    documents with their similarity score and full text, so the
    calling model can generate a grounded answer from them.
    """
    logger.info(f"search_knowledge_base called: question={question!r}")
    if _vectorizer is None or _doc_matrix is None:
        return [{"error": "Knowledge base index is empty."}]

    query_vec = _vectorizer.transform([question])
    scores = cosine_similarity(query_vec, _doc_matrix)[0]

    ranked = sorted(zip(_doc_names, _doc_texts, scores), key=lambda x: x[2], reverse=True)
    top = ranked[:top_k]

    return [
        {"document": name, "similarity_score": round(float(score), 4), "content": text}
        for name, text, score in top
    ]


@mcp.resource("knowledge://docs/list")
def list_documents() -> str:
    """Read-only resource listing all documents currently in the knowledge base."""
    if not _doc_names:
        return "No documents indexed."
    return "\n".join(_doc_names)


_build_index()

if __name__ == "__main__":
    logger.info("Starting Knowledge Base MCP server (stdio transport)...")
    mcp.run()
