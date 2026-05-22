from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from vektor.index.flat import FlatIndex


def test_empty_index_returns_no_results():
    idx = FlatIndex(dim=8)
    assert idx.size == 0
    q = np.ones(8, dtype=np.float32) / np.sqrt(8)
    assert idx.search(q, k=5) == []


def test_add_and_search_returns_self_first(random_vectors):
    vecs = random_vectors(50, 16)
    ids = [f"doc{i}" for i in range(50)]
    idx = FlatIndex(dim=16)
    idx.add_batch(vecs, ids)

    # query with one of the indexed vectors — should return itself rank-1
    for i in [0, 7, 42]:
        results = idx.search(vecs[i], k=5)
        assert results[0].doc_id == f"doc{i}"
        assert results[0].score > 0.99  # near-1 for self-cosine


def test_top_k_ordering(random_vectors):
    vecs = random_vectors(20, 4)
    ids = [f"d{i}" for i in range(20)]
    idx = FlatIndex(dim=4)
    idx.add_batch(vecs, ids)

    q = vecs[0]
    results = idx.search(q, k=10)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_save_and_load_roundtrip(random_vectors):
    vecs = random_vectors(30, 8)
    ids = [f"x{i}" for i in range(30)]
    idx = FlatIndex(dim=8)
    idx.add_batch(vecs, ids)

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "idx.pkl"
        idx.save(p)
        idx2 = FlatIndex(dim=8)
        idx2.load(p)

    assert idx2.size == idx.size
    q = vecs[3]
    r1 = idx.search(q, k=5)
    r2 = idx2.search(q, k=5)
    assert [x.doc_id for x in r1] == [x.doc_id for x in r2]


def test_shape_mismatch_raises():
    idx = FlatIndex(dim=8)
    import pytest

    with pytest.raises(ValueError):
        idx.add_batch(np.zeros((3, 4), dtype=np.float32), ["a", "b", "c"])
    with pytest.raises(ValueError):
        idx.add_batch(np.zeros((3, 8), dtype=np.float32), ["a", "b"])
