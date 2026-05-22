"""Locust load test for the VEKTOR API.

Run with:
    locust -f scripts/loadtest.py --host http://localhost:8000 \\
           --users 1000 --spawn-rate 50 --run-time 2m --headless

Sweep concurrency and write results to docs/benchmarks.md.
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task


_QUERIES = [
    "transformer attention mechanism",
    "hierarchical navigable small world graph",
    "BM25 scoring function",
    "cross encoder reranking architecture",
    "approximate nearest neighbor search",
    "vector quantization product",
    "FAISS index types",
    "inverted file index k-means",
    "semantic search wikipedia",
    "dense passage retrieval",
]


class VektorUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(10)
    def search_hybrid(self):
        self.client.post(
            "/search",
            json={
                "query": random.choice(_QUERIES),
                "k": 10,
                "strategy": "hybrid",
                "rerank": False,
            },
            name="POST /search hybrid no-rerank",
        )

    @task(3)
    def search_with_rerank(self):
        self.client.post(
            "/search",
            json={
                "query": random.choice(_QUERIES),
                "k": 10,
                "strategy": "hybrid",
                "rerank": True,
            },
            name="POST /search hybrid +rerank",
        )

    @task(1)
    def healthz(self):
        self.client.get("/healthz", name="GET /healthz")
