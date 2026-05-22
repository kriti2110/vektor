from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

# Request-level
request_duration = Histogram(
    "vektor_request_duration_seconds",
    "End-to-end request latency",
    labelnames=("route", "status"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# Retrieval-level
retrieval_duration = Histogram(
    "vektor_retrieval_duration_seconds",
    "Retrieval backend latency",
    labelnames=("backend",),
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)

rerank_duration = Histogram(
    "vektor_rerank_duration_seconds",
    "Reranking latency",
    labelnames=("model",),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

# Cache
cache_hits = Counter(
    "vektor_cache_hits_total",
    "Cache hit count",
    labelnames=("layer",),
)
cache_misses = Counter(
    "vektor_cache_misses_total",
    "Cache miss count",
    labelnames=("layer",),
)

# Index
index_size = Gauge(
    "vektor_index_size",
    "Number of vectors in the index",
    labelnames=("backend",),
)

# Feedback
feedback_events = Counter(
    "vektor_feedback_events_total",
    "Feedback events received",
    labelnames=("event_type",),
)


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST
