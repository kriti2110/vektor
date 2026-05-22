# VEKTOR Architecture

## Design goals

1. **Correctness over cleverness.** Every layer has a brute-force baseline you can A/B against. If the "fast" version disagrees with the baseline, something is wrong.
2. **Measure everything.** Recall, MRR, p50/p95/p99, cache hit rate, index freshness — all exported as Prometheus metrics.
3. **Modular index backends.** `BaseIndex` is the interface; flat, HNSW, IVF, BM25 are interchangeable implementations.
4. **Async I/O end-to-end.** FastAPI + httpx + asyncio Redis. No sync calls in the request path.

## Request lifecycle

A `POST /search` request flows through these stages:

```
1. Receive → validate request body (pydantic)
2. Cache lookup → Redis GET on hash(query, k, strategy). If hit, return.
3. Query understanding:
   - normalize (lowercase, unicode NFC, strip)
   - spell-correct (optional)
   - classify intent (navigational / informational / transactional)
   - expand (generate 2-3 paraphrases via small model)
4. Parallel retrieval:
   - Dense: encode query → HNSW search (top-100)
   - Sparse: BM25 search (top-100)
5. Fusion: RRF over the two result lists → top-50
6. Reranking: cross-encoder scores (query, doc) for the top-50, sort → top-k
7. Cache write → Redis SETEX with TTL
8. Return + emit metrics + log for feedback collection
```

## Index abstraction

```python
class BaseIndex(ABC):
    @abstractmethod
    def add(self, vector: np.ndarray, doc_id: str) -> None: ...

    @abstractmethod
    def add_batch(self, vectors: np.ndarray, doc_ids: list[str]) -> None: ...

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> list[tuple[str, float]]: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...

    @abstractmethod
    def load(self, path: Path) -> None: ...

    @property
    @abstractmethod
    def size(self) -> int: ...
```

All concrete indexes (flat, HNSW, IVF) implement this. The API and benchmark scripts are agnostic to which one is loaded.

## Embedding pipeline

- **Chunker.** Splits documents at semantic boundaries — sentences first, then merges sentences greedily into ≤512-token chunks while keeping coherence. Avoids fixed-size splits that bisect ideas.
- **Embedder.** Wraps a sentence-transformers model (default: `all-MiniLM-L6-v2`, 384-dim). Batches at 64, caches by `sha256(text)` in SQLite. New docs incur new embeddings; re-ingesting the same chunk is a cache hit.
- **Loaders.** `pdf.py` (pypdf), `html.py` (BeautifulSoup, strips boilerplate), `jsonl.py` (one doc per line).

## Caching layer

- **L1 — Embedding cache (SQLite).** Persistent across runs. Keyed by chunk hash.
- **L2 — Query cache (Redis).** Per-process, 1-hour TTL. Keyed by `sha256(query|k|strategy)`.
- **L3 — Cross-encoder score cache (Redis).** Keyed by `sha256(query|doc_id)`. Useful when the same (q, d) pair is rescored across A/B variants.

## Observability

Metrics exported on `/metrics`:

| Metric | Type | Labels |
|--------|------|--------|
| `vektor_request_duration_seconds` | Histogram | `route`, `status` |
| `vektor_retrieval_duration_seconds` | Histogram | `backend` (hnsw/bm25) |
| `vektor_rerank_duration_seconds` | Histogram | `model` |
| `vektor_cache_hits_total` | Counter | `layer` (query/embed/rerank) |
| `vektor_index_size` | Gauge | `backend` |
| `vektor_feedback_events_total` | Counter | `event_type` (click/skip/dwell) |

## Deployment topology

**Local (docker-compose):** api, redis, prometheus, grafana — single-node.

**Cloud (planned):** GKE/EKS, one pod per shard, Redis Cluster sidecar, Prometheus Operator, Grafana for dashboards. K8s manifests live in `infra/k8s/` (TBD).

## Sharding strategy (planned)

For 1M+ docs:
- Shard by `hash(doc_id) % N_SHARDS`. Each shard owns its own HNSW + BM25.
- Query fan-out to all shards in parallel, merge results before rerank.
- Stateless API — any pod can answer any query, since indexes are loaded into pod memory at startup.

## What's intentionally simple

- **No distributed index.** Single-node HNSW indexes up to ~10M vectors fit in memory at MiniLM dimensions. Beyond that, shard.
- **No persistent doc store.** Doc text lives in a flat parquet file at ingest time. Production would use Postgres / S3.
- **No multi-tenant auth.** This is an infra demo, not a SaaS.
