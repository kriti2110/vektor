"""HNSW correctness + recall tests.

These are intentionally STRICT — they're the gate Kriti's implementation
must pass. They will all fail until vektor/index/hnsw.py is implemented.

Run with: pytest tests/test_hnsw.py -v
"""

from __future__ import annotations

import pytest

from vektor.index.flat import FlatIndex
from vektor.index.hnsw import HNSWIndex


pytestmark = pytest.mark.skip(reason="HNSW is a stub — un-skip after implementing vektor/index/hnsw.py")


def test_hnsw_basic_self_recall(random_vectors):
    """Inserting N vectors and querying with one of them should return it at rank 1."""
    vecs = random_vectors(100, 32)
    ids = [f"d{i}" for i in range(100)]
    idx = HNSWIndex(dim=32, M=8, ef_construction=50, ef_search=50, seed=0)
    idx.add_batch(vecs, ids)

    for i in [0, 17, 99]:
        results = idx.search(vecs[i], k=1)
        assert results[0].doc_id == f"d{i}"


def test_hnsw_recall_at_10_vs_flat(random_vectors):
    """HNSW recall@10 should be within 2% of flat-search ground truth."""
    n, dim = 10_000, 64
    vecs = random_vectors(n, dim)
    ids = [f"d{i}" for i in range(n)]

    flat = FlatIndex(dim=dim)
    flat.add_batch(vecs, ids)

    hnsw = HNSWIndex(dim=dim, M=16, ef_construction=200, ef_search=100, seed=0)
    hnsw.add_batch(vecs, ids)

    queries = random_vectors(100, dim, seed=42)

    recalls = []
    for q in queries:
        gt = {r.doc_id for r in flat.search(q, k=10)}
        approx = {r.doc_id for r in hnsw.search(q, k=10)}
        recalls.append(len(gt & approx) / len(gt))

    mean_recall = sum(recalls) / len(recalls)
    assert mean_recall >= 0.98, f"recall@10 = {mean_recall:.4f}, want >= 0.98"


def test_hnsw_save_load_roundtrip(random_vectors, tmp_path):
    vecs = random_vectors(200, 16)
    ids = [f"x{i}" for i in range(200)]
    idx = HNSWIndex(dim=16, M=8, ef_construction=50, ef_search=50, seed=0)
    idx.add_batch(vecs, ids)

    path = tmp_path / "hnsw.pkl"
    idx.save(path)
    idx2 = HNSWIndex(dim=16)
    idx2.load(path)
    assert idx2.size == idx.size

    q = vecs[3]
    r1 = [r.doc_id for r in idx.search(q, k=5)]
    r2 = [r.doc_id for r in idx2.search(q, k=5)]
    assert r1 == r2
