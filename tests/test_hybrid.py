"""RRF fusion tests. Skipped until Kriti implements vektor/retrieval/hybrid.py."""

from __future__ import annotations

import pytest

from vektor.index.base import SearchResult
from vektor.retrieval.hybrid import rrf_fuse


def test_rrf_full_overlap_preserves_top():
    dense = [SearchResult("a", 0.9), SearchResult("b", 0.7)]
    sparse = [SearchResult("a", 5.0), SearchResult("b", 3.0)]
    fused = rrf_fuse([dense, sparse])
    assert fused[0].doc_id == "a"
    assert fused[1].doc_id == "b"


def test_rrf_disjoint_lists_merge():
    dense = [SearchResult("a", 0.9)]
    sparse = [SearchResult("b", 5.0)]
    fused = rrf_fuse([dense, sparse])
    assert {r.doc_id for r in fused} == {"a", "b"}


def test_rrf_empty_lists():
    fused = rrf_fuse([[], []])
    assert fused == []


def test_rrf_top_k_truncation():
    dense = [SearchResult(f"d{i}", 1 - i * 0.01) for i in range(20)]
    sparse = [SearchResult(f"d{i}", 100 - i) for i in range(20)]
    fused = rrf_fuse([dense, sparse], top_k=5)
    assert len(fused) == 5


def test_rrf_doc_in_one_list_only():
    """A doc appearing only in dense should still rank reasonably."""
    dense = [SearchResult("a", 0.9), SearchResult("only_dense", 0.85)]
    sparse = [SearchResult("a", 5.0), SearchResult("only_sparse", 4.0)]
    fused = rrf_fuse([dense, sparse])
    ids = [r.doc_id for r in fused]
    # 'a' appears in both → should rank highest
    assert ids[0] == "a"
    # both unique-to-one-list docs should appear
    assert "only_dense" in ids and "only_sparse" in ids
