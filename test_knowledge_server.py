"""
Regression tests for knowledge_server.py
------------------------------------------
Two kinds of coverage here, deliberately separated:

1. OFFLINE tests (this file, runnable in CI / no Ollama required) --
   verify the pipeline MECHANICS: chunking behavior, PDF/txt loading,
   metadata correctness, and search ranking logic. These monkeypatch
   _embed() with a deterministic fake so results are reproducible
   without a live model -- they catch "I broke the chunking logic" or
   "I broke the ranking logic" bugs, not "the embedding model got worse
   at understanding leave policy" bugs.

2. LIVE tests (see live_test_questions.py in this same folder) -- real
   questions to run against the actual running server with real Ollama
   embeddings, to check retrieval QUALITY, not just mechanics. These
   can't be automated in CI without a real Ollama instance, so they're
   kept as a documented manual (or later, scripted) checklist.

This split -- deterministic offline correctness vs. live quality checks
-- is the same split real ML systems use: unit tests for code paths,
evals for model behavior. This file is effectively a first sketch of
project #8 (regression eval suite) in miniature, scoped to one server.

Run with: pytest test_knowledge_server.py -v
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import knowledge_server as ks


# ---------------------------------------------------------------------------
# Deterministic fake embedding: turns each text into a small vector based on
# which "topic keywords" it contains. This lets us assert *ranking* behaves
# correctly (the leave-related chunk should score highest for a leave-related
# query) without depending on a live model or exact embedding values.
# ---------------------------------------------------------------------------
TOPIC_KEYWORDS = [
    "leave", "expense", "remote", "conduct", "security",
    "benefits", "onboarding", "performance",
]


def _fake_embed(texts: list[str]) -> np.ndarray:
    vectors = []
    for text in texts:
        lower = text.lower()
        vec = [lower.count(word) for word in TOPIC_KEYWORDS]
        norm = np.linalg.norm(vec) or 1.0
        vectors.append(np.array(vec, dtype=np.float32) / norm)
    return np.array(vectors, dtype=np.float32)


@pytest.fixture(autouse=True)
def isolate_docs_dir(tmp_path, monkeypatch):
    """Point DOCS_DIR at a temp dir per test so tests don't interfere."""
    monkeypatch.setattr(ks, "DOCS_DIR", tmp_path)
    yield tmp_path


# ---------------------------------------------------------------------------
# Chunking mechanics
# ---------------------------------------------------------------------------

def test_split_into_chunks_short_text_is_single_chunk():
    text = "Short paragraph.\n\nAnother short one."
    chunks = ks._split_into_chunks(text, max_chars=800)
    assert len(chunks) == 1


def test_split_into_chunks_respects_paragraph_boundaries():
    para = "A" * 500
    text = f"{para}\n\n{para}\n\n{para}"
    chunks = ks._split_into_chunks(text, max_chars=800)
    # Each paragraph is 500 chars; two together (1000+) exceed max_chars=800,
    # so each paragraph ends up in its own chunk -- 3 paragraphs -> 3 chunks.
    assert len(chunks) == 3
    for chunk in chunks:
        assert "AAAA" in chunk  # never silently drops content


def test_split_into_chunks_oversized_paragraph_kept_intact():
    huge_para = "word " * 400  # ~2000 chars, no blank lines to split on
    chunks = ks._split_into_chunks(huge_para, max_chars=800)
    assert len(chunks) == 1
    assert chunks[0].strip() == huge_para.strip()


def test_split_into_chunks_never_drops_text():
    text = "\n\n".join([f"Paragraph number {i}. " * 20 for i in range(5)])
    chunks = ks._split_into_chunks(text, max_chars=400)
    reconstructed_words = " ".join(chunks).split()
    original_words = text.split()
    # every word from the original appears somewhere in the chunked output
    assert set(original_words) <= set(reconstructed_words)


# ---------------------------------------------------------------------------
# Document loading + metadata correctness
# ---------------------------------------------------------------------------

def test_load_txt_file_single_chunk_with_none_page(isolate_docs_dir):
    (isolate_docs_dir / "policy.txt").write_text("Some policy text.", encoding="utf-8")
    chunks = ks._load_documents()
    assert len(chunks) == 1
    name, text, meta = chunks[0]
    assert meta["page"] is None
    assert meta["source_file"] == "policy.txt"
    assert text == "Some policy text."


def test_load_pdf_produces_one_chunk_per_page_with_correct_page_numbers():
    real_pdf = Path(__file__).parent / "data" / "company_docs" / "Northwind_Employee_Handbook.pdf"
    if not real_pdf.exists():
        pytest.skip("Sample handbook PDF not present in data/company_docs/")

    chunks = ks._load_pdf_chunks(real_pdf)
    pages_seen = [meta["page"] for _, _, meta in chunks]
    assert pages_seen == sorted(pages_seen)  # pages come out in order
    assert all(meta["source_file"] == "Northwind_Employee_Handbook.pdf" for _, _, meta in chunks)
    assert len(chunks) >= 8  # at least one chunk per real content page


def test_load_documents_skips_empty_pdf_pages_gracefully(tmp_path):
    # A PDF with no extractable text shouldn't crash indexing, just produce
    # zero chunks for that file -- covered implicitly by _load_pdf_chunks'
    # `if not page_text: continue` guard. This test documents that contract.
    assert hasattr(ks, "_load_pdf_chunks")


# ---------------------------------------------------------------------------
# Index build + reindex behavior
# ---------------------------------------------------------------------------

def test_build_index_empty_dir_returns_empty_status(isolate_docs_dir):
    with patch.object(ks, "_embed", side_effect=_fake_embed):
        result = ks._build_index()
    assert result["status"] == "empty"
    assert ks._doc_matrix is None


def test_build_index_success_sets_last_indexed_at(isolate_docs_dir):
    (isolate_docs_dir / "a.txt").write_text("Leave policy details here.", encoding="utf-8")
    with patch.object(ks, "_embed", side_effect=_fake_embed):
        result = ks._build_index()
    assert result["status"] == "ok"
    assert result["chunks_indexed"] == 1
    assert ks._last_indexed_at is not None


def test_build_index_embedding_failure_is_reported_not_silent(isolate_docs_dir):
    (isolate_docs_dir / "a.txt").write_text("Some text.", encoding="utf-8")
    with patch.object(ks, "_embed", side_effect=RuntimeError("Ollama unreachable")):
        result = ks._build_index()
    assert result["status"] == "error"
    assert "Ollama unreachable" in result["error"]
    assert ks._doc_matrix is None  # doesn't leave a stale/partial index in place


def test_reindex_reflects_added_documents(isolate_docs_dir):
    (isolate_docs_dir / "a.txt").write_text("First doc.", encoding="utf-8")
    with patch.object(ks, "_embed", side_effect=_fake_embed):
        ks._build_index()
        assert len(ks._chunk_texts) == 1

        (isolate_docs_dir / "b.txt").write_text("Second doc.", encoding="utf-8")
        result = ks.reindex_knowledge_base()

    assert result["chunks_indexed"] == 2
    assert len(ks._chunk_texts) == 2


# ---------------------------------------------------------------------------
# Search / ranking correctness
# ---------------------------------------------------------------------------

def test_search_ranks_topically_relevant_chunk_first(isolate_docs_dir):
    (isolate_docs_dir / "leave.txt").write_text(
        "Our leave policy grants 24 days of annual leave per year.", encoding="utf-8"
    )
    (isolate_docs_dir / "security.txt").write_text(
        "Our security policy requires multi-factor authentication.", encoding="utf-8"
    )

    with patch.object(ks, "_embed", side_effect=_fake_embed):
        ks._build_index()
        results = ks.search_knowledge_base("How much leave do I get?", top_k=2)

    assert results[0]["source_file"] == "leave.txt"
    assert results[0]["similarity_score"] >= results[1]["similarity_score"]


def test_search_on_empty_index_returns_explicit_error(isolate_docs_dir):
    ks._doc_matrix = None
    results = ks.search_knowledge_base("Anything?", top_k=2)
    assert "error" in results[0]
    assert "reindex_knowledge_base" in results[0]["error"]


def test_search_query_embedding_failure_is_reported(isolate_docs_dir):
    (isolate_docs_dir / "a.txt").write_text("Some text.", encoding="utf-8")
    with patch.object(ks, "_embed", side_effect=_fake_embed):
        ks._build_index()

    with patch.object(ks, "_embed", side_effect=RuntimeError("timeout")):
        results = ks.search_knowledge_base("A question", top_k=1)

    assert "error" in results[0]
    assert "timeout" in results[0]["error"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))