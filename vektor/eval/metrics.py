from __future__ import annotations

import numpy as np


def recall_at_k(predicted: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant docs appearing in the top-k predictions."""
    if not relevant:
        return 0.0
    top_k = set(predicted[:k])
    return len(top_k & relevant) / len(relevant)


def mean_reciprocal_rank(predicted: list[str], relevant: set[str], k: int = 10) -> float:
    """1 / rank of the first relevant doc in predictions[:k]. 0 if none found."""
    for i, doc_id in enumerate(predicted[:k], start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(predicted: list[str], relevant: set[str], k: int) -> float:
    """Binary NDCG@k (relevance is 0/1)."""
    gains = np.array([1.0 if doc_id in relevant else 0.0 for doc_id in predicted[:k]])
    if gains.sum() == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float((gains * discounts).sum())
    ideal_gains = np.ones(min(len(relevant), k))
    ideal_discounts = 1.0 / np.log2(np.arange(2, len(ideal_gains) + 2))
    idcg = float((ideal_gains * ideal_discounts).sum())
    return dcg / idcg


def latency_percentiles(latencies_ms: list[float]) -> dict[str, float]:
    """p50/p90/p95/p99 percentiles from a latency sample (in ms)."""
    if not latencies_ms:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "n": 0}
    a = np.array(latencies_ms)
    return {
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
        "mean": float(a.mean()),
        "n": int(a.size),
    }
