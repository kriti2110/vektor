from __future__ import annotations

from vektor.index.bm25 import BM25Index


def test_empty_index_returns_nothing():
    idx = BM25Index()
    assert idx.size == 0
    assert idx.search("anything", k=5) == []


def test_exact_term_match_scores_first():
    idx = BM25Index()
    idx.add_batch(
        [
            "the quick brown fox jumps over the lazy dog",
            "python is a programming language",
            "neural networks process data through layers",
        ],
        ["d0", "d1", "d2"],
    )
    results = idx.search("python programming", k=3)
    assert results[0].doc_id == "d1"


def test_empty_query_returns_empty():
    idx = BM25Index()
    idx.add_batch(["hello world"], ["d0"])
    assert idx.search("", k=5) == []
    assert idx.search("   ", k=5) == []


def test_tokenization_is_case_insensitive():
    idx = BM25Index()
    idx.add_batch(["Hello World"], ["d0"])
    results = idx.search("hello", k=1)
    assert len(results) == 1
    assert results[0].doc_id == "d0"
