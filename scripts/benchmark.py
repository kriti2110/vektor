"""Benchmark an index against a query set.

Computes recall@10 / recall@100 (vs a flat baseline if available), MRR@10,
and p50/p95/p99 query latency.

Example:
    python scripts/benchmark.py \\
        --index index_store/wiki_hnsw.hnsw \\
        --backend hnsw \\
        --baseline index_store/wiki_hnsw.flat \\
        --queries data/eval_queries.jsonl
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import click

from vektor.config import settings
from vektor.eval.metrics import latency_percentiles, mean_reciprocal_rank, recall_at_k
from vektor.index.flat import FlatIndex
from vektor.index.hnsw import HNSWIndex
from vektor.ingestion.embedder import Embedder


def load_index(path: Path, backend: str):
    if backend == "flat":
        idx = FlatIndex(dim=settings.embed_dim)
    elif backend == "hnsw":
        idx = HNSWIndex(dim=settings.embed_dim)
    else:
        raise ValueError(backend)
    idx.load(path)
    return idx


@click.command()
@click.option("--index", required=True, type=click.Path(exists=True))
@click.option("--backend", required=True, type=click.Choice(["flat", "hnsw"]))
@click.option("--baseline", default=None, type=click.Path(exists=True), help="Optional flat-index baseline for recall.")
@click.option("--queries", required=True, type=click.Path(exists=True), help="JSONL: {query: str, relevant: [doc_id]}")
@click.option("--k", default=10, type=int)
@click.option("--warmup", default=10, type=int)
def main(index: str, backend: str, baseline: str | None, queries: str, k: int, warmup: int) -> None:
    embedder = Embedder(
        model_name=settings.embed_model,
        dim=settings.embed_dim,
        cache_path=settings.embed_cache_path,
    )
    idx = load_index(Path(index), backend)
    base = load_index(Path(baseline), "flat") if baseline else None

    query_set = []
    with Path(queries).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                query_set.append(json.loads(line))

    click.echo(f"loaded {len(query_set)} queries, index size {idx.size}")

    # warmup
    for q in query_set[:warmup]:
        try:
            v = embedder.encode_one(q["query"])
            idx.search(v, k=k)
        except NotImplementedError:
            click.echo(f"{backend} search is a stub — implement it first.", err=True)
            raise SystemExit(1)

    latencies_ms = []
    recall_scores = []
    mrr_scores = []

    for q in query_set:
        v = embedder.encode_one(q["query"])
        t0 = time.perf_counter()
        results = idx.search(v, k=k)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        predicted = [r.doc_id for r in results]
        relevant: set[str] = set(q.get("relevant", []))

        if base is not None and not relevant:
            # Use baseline top-k as pseudo-ground-truth when no qrels provided
            relevant = {r.doc_id for r in base.search(v, k=k)}

        recall_scores.append(recall_at_k(predicted, relevant, k))
        mrr_scores.append(mean_reciprocal_rank(predicted, relevant, k))

    pct = latency_percentiles(latencies_ms)
    click.echo("\n=== Benchmark Results ===")
    click.echo(f"backend:           {backend}")
    click.echo(f"index size:        {idx.size}")
    click.echo(f"queries:           {len(query_set)}")
    click.echo(f"recall@{k}:         {sum(recall_scores) / len(recall_scores):.4f}")
    click.echo(f"MRR@{k}:            {sum(mrr_scores) / len(mrr_scores):.4f}")
    click.echo(f"latency p50/p95/p99 ms: {pct['p50']:.2f} / {pct['p95']:.2f} / {pct['p99']:.2f}")


if __name__ == "__main__":
    main()
