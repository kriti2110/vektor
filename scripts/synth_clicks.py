"""Generate synthetic click logs for offline reranker training.

Useful before the API is collecting real traffic — Kriti can develop the
training loop against this data, then swap to the real FeedbackStore later.

Model: assume the "true" relevance ranking is a permutation of the top results,
and simulate a position-biased click model — click probability decays with rank,
modulated by relevance.
"""

from __future__ import annotations

import random
import uuid

import click

from vektor.config import settings
from vektor.rerank.feedback import FeedbackEvent, FeedbackEventType, FeedbackStore


@click.command()
@click.option("--n-queries", default=500, type=int)
@click.option("--results-per-query", default=10, type=int)
@click.option("--seed", default=42, type=int)
def main(n_queries: int, results_per_query: int, seed: int) -> None:
    rng = random.Random(seed)
    store = FeedbackStore(settings.feedback_db_path)

    sample_queries = [
        "what is hnsw",
        "how does bm25 work",
        "explain reciprocal rank fusion",
        "transformer attention",
        "vector database vs search engine",
    ]

    for _ in range(n_queries):
        qid = str(uuid.uuid4())
        q_text = rng.choice(sample_queries) + " " + str(rng.randint(0, 999))
        # rank "true" doc ids; the rank-1 doc is the genuine best
        doc_ids = [f"doc_{rng.randint(0, 999999)}" for _ in range(results_per_query)]
        true_quality = [results_per_query - i for i in range(results_per_query)]
        rng.shuffle(true_quality)

        for rank, (did, quality) in enumerate(zip(doc_ids, true_quality)):
            # position-biased click model
            position_prob = 1.0 / (rank + 1)
            relevance_prob = quality / results_per_query
            click_prob = position_prob * relevance_prob
            if rng.random() < click_prob:
                store.record(
                    FeedbackEvent(
                        query_id=qid,
                        query_text=q_text,
                        doc_id=did,
                        rank=rank,
                        event_type=FeedbackEventType.CLICK,
                        dwell_ms=rng.randint(2000, 60000),
                    )
                )

    store.close()
    click.echo(f"wrote synthetic clicks for {n_queries} queries → {settings.feedback_db_path}")


if __name__ == "__main__":
    main()
