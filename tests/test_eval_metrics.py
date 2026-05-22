from __future__ import annotations

from vektor.eval.metrics import (
    latency_percentiles,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)


def test_recall_at_k_basic():
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, k=3) == 1.0
    assert recall_at_k(["a", "b", "c"], {"a", "c"}, k=1) == 0.5
    assert recall_at_k(["x"], {"a"}, k=1) == 0.0


def test_recall_at_k_no_relevant():
    assert recall_at_k(["a", "b"], set(), k=5) == 0.0


def test_mrr_returns_inverse_first_relevant_rank():
    assert mean_reciprocal_rank(["a", "b", "c"], {"b"}, k=10) == 0.5
    assert mean_reciprocal_rank(["a", "b", "c"], {"a"}, k=10) == 1.0
    assert mean_reciprocal_rank(["a", "b", "c"], {"d"}, k=10) == 0.0


def test_mrr_respects_k_cutoff():
    assert mean_reciprocal_rank(["a", "b", "c"], {"c"}, k=2) == 0.0
    assert mean_reciprocal_rank(["a", "b", "c"], {"c"}, k=3) == 1 / 3


def test_ndcg_perfect_ranking_is_1():
    score = ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3)
    assert score == 1.0


def test_ndcg_no_relevant_is_0():
    assert ndcg_at_k(["x", "y"], set(), k=2) == 0.0


def test_latency_percentiles_keys():
    p = latency_percentiles([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    for k in ("p50", "p90", "p95", "p99", "mean", "n"):
        assert k in p
    assert p["n"] == 10
    assert p["p50"] == 5.5


def test_latency_percentiles_empty():
    p = latency_percentiles([])
    assert p["n"] == 0
    assert p["p99"] == 0.0
