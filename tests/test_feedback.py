from __future__ import annotations

import tempfile
from pathlib import Path

from vektor.rerank.feedback import FeedbackEvent, FeedbackEventType, FeedbackStore


def test_record_and_query():
    with tempfile.TemporaryDirectory() as d:
        store = FeedbackStore(Path(d) / "fb.sqlite")
        store.record(
            FeedbackEvent(
                query_id="q1",
                query_text="hello",
                doc_id="doc_a",
                rank=0,
                event_type=FeedbackEventType.CLICK,
                dwell_ms=5000,
            )
        )
        store.record(
            FeedbackEvent(
                query_id="q1",
                query_text="hello",
                doc_id="doc_b",
                rank=1,
                event_type=FeedbackEventType.SKIP,
            )
        )
        triples = list(store.iter_training_triples())
        store.close()

    # one click at rank 0, no docs above it, so skip-above yields nothing
    assert triples == []


def test_skip_above_yields_negative():
    with tempfile.TemporaryDirectory() as d:
        store = FeedbackStore(Path(d) / "fb.sqlite")
        # doc_a shown at rank 0 but NOT clicked
        store.record(
            FeedbackEvent("q1", "test", "doc_a", 0, FeedbackEventType.SKIP)
        )
        # doc_b clicked at rank 1 → doc_a is a negative for doc_b
        store.record(
            FeedbackEvent("q1", "test", "doc_b", 1, FeedbackEventType.CLICK)
        )
        triples = list(store.iter_training_triples())
        store.close()

    assert ("test", "doc_b", "doc_a") in triples


def test_hybrid_rrf_smoke():
    from vektor.index.base import SearchResult
    from vektor.retrieval.hybrid import rrf_fuse

    fused = rrf_fuse([[SearchResult("a", 0.9)], [SearchResult("a", 0.5)]])
    assert fused[0].doc_id == "a"
