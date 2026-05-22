"""Process-wide handles for shared resources (index, embedder, reranker, cache).

In a multi-pod K8s deploy each replica holds its own copies. For sharded
indexes you'd instead route queries to the pod that owns the shard — that's
a layer added in `infra/k8s/` (TBD).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vektor.api.cache import QueryCache
from vektor.ingestion.embedder import Embedder
from vektor.rerank.feedback import FeedbackStore


@dataclass
class AppState:
    embedder: Embedder | None = None
    dense_index: Any | None = None  # BaseIndex
    sparse_index: Any | None = None  # BM25Index
    reranker: Any | None = None
    cache: QueryCache | None = None
    feedback: FeedbackStore | None = None
    doc_text: dict[str, str] | None = None  # in-mem doc text lookup for rerank


state = AppState()
