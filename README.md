---
title: VEKTOR
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: From-scratch semantic search — HNSW + hybrid + reranker
---

# VEKTOR

**Semantic search infrastructure at scale.** Custom embedding pipeline, from-scratch HNSW index, hybrid dense+sparse retrieval, and a self-improving cross-encoder reranker — designed to serve sub-50ms p99 search over millions of documents.

> **Status:** v0.1 — end-to-end pipeline working. HNSW, RRF, reranker training all implemented. 36 tests passing.

---

## What this is

A production-grade semantic search engine built from primitives — not a wrapper around an existing vector DB. Every layer is implemented in this repo:

| Layer | What it does | Code |
|-------|--------------|------|
| **Ingestion** | Semantic-boundary chunker, batched embedder w/ SQLite cache, PDF/HTML/JSONL loaders | `vektor/ingestion/` |
| **Index** | HNSW (from scratch, pure numpy) + flat baseline + BM25 sparse | `vektor/index/` |
| **Retrieval** | Hybrid dense+sparse fusion via Reciprocal Rank Fusion | `vektor/retrieval/` |
| **Reranking** | Cross-encoder inference + fine-tuning loop on click feedback | `vektor/rerank/` |
| **Serving** | Async FastAPI, Redis cache, Prometheus metrics, Docker compose | `vektor/api/` |
| **Evaluation** | Recall@k, MRR, NDCG, p50/p95/p99 latency, Locust load test | `vektor/eval/`, `scripts/loadtest.py` |

## Quick start

```bash
git clone https://github.com/kriti2110/vektor.git
cd vektor
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 1. Pull a Wikipedia subset (10k articles, a few minutes)
python scripts/download_wikipedia.py --n 10000 --out data/wiki.jsonl

# 2. Build the index (HNSW + BM25, with chunk text store for reranking)
python scripts/build_index.py \
  --source data/wiki.jsonl \
  --backend hnsw \
  --out index_store/wiki

# 3. Start the API with the index pre-loaded
VEKTOR_DENSE_INDEX_PATH=index_store/wiki.hnsw \
VEKTOR_SPARSE_INDEX_PATH=index_store/wiki.bm25 \
VEKTOR_DOC_STORE_PATH=index_store/wiki.docs.jsonl \
VEKTOR_ENABLE_RERANKER=true \
vektor serve

# 4. Query it
curl -X POST http://localhost:8000/search \
  -H 'content-type: application/json' \
  -d '{"query": "transformer attention mechanism", "k": 5, "strategy": "hybrid", "rerank": true}'
```

For local infra (Redis, Prometheus, Grafana) alongside the API:

```bash
docker compose up -d
```

## Architecture

```
                 ┌─────────────────────────────────────────────┐
                 │                Client                       │
                 └────────────────────┬────────────────────────┘
                                      │ HTTP
                 ┌────────────────────▼────────────────────────┐
                 │              FastAPI (async)                │
                 │  ┌────────────┐  ┌──────────────────────┐   │
                 │  │  /search   │  │  Prometheus /metrics │   │
                 │  └─────┬──────┘  └──────────────────────┘   │
                 └────────┼────────────────────────────────────┘
                          │
            ┌─────────────▼──────────────┐    ┌──────────────┐
            │     Query Pipeline         │◄──►│  Redis cache │
            │  • normalize / expand      │    └──────────────┘
            │  • intent classify         │
            └─────────────┬──────────────┘
                          │
            ┌─────────────▼──────────────┐
            │    Hybrid Retrieval        │
            │  ┌────────┐   ┌─────────┐  │
            │  │ HNSW   │   │  BM25   │  │
            │  │ (dense)│   │ (sparse)│  │
            │  └───┬────┘   └────┬────┘  │
            │      └──── RRF ────┘       │
            └─────────────┬──────────────┘
                          │ top-50
            ┌─────────────▼──────────────┐
            │   Cross-Encoder Reranker   │
            │  (fine-tuned on click log) │
            └─────────────┬──────────────┘
                          │ top-k
                  ┌───────▼───────┐
                  │    Results    │
                  └───────────────┘
```

See [`docs/architecture.md`](./docs/architecture.md) for the full deep-dive.

## Implementation notes

- **HNSW** (`vektor/index/hnsw.py`): pure-numpy + `heapq`, no `hnswlib`/`faiss`. Implements level assignment via geometric distribution, greedy beam search per layer, the Malkov-paper neighbor selection heuristic, and bidirectional link pruning. Verified at recall@10 ≥ 0.95 vs flat-index ground truth on 10k 64-dim vectors. See [`docs/hnsw-notes.md`](./docs/hnsw-notes.md) for the algorithm walkthrough.
- **RRF** (`vektor/retrieval/hybrid.py`): Cormack et al. 2009. `score(d) = Σ 1/(k + rank_i(d))` across retrievers, `k=60`.
- **Reranker training** (`vektor/rerank/train.py`): pulls (q, pos, neg) triples from the feedback store via skip-above heuristic, fine-tunes a CrossEncoder with BCE loss, reports MRR before/after. Synthetic click generator at `scripts/synth_clicks.py` for offline development.

## Benchmarks

Run with `scripts/benchmark.py`; results live in [`docs/benchmarks.md`](./docs/benchmarks.md). Target SLOs:

| Metric | Target |
|--------|--------|
| Index size | 1M+ documents |
| p99 query latency | < 50 ms |
| HNSW recall@10 vs flat | ≥ 0.95 |
| Concurrent users (Locust) | 1000+ |
| Reranker MRR uplift | +8% |

## Project layout

```
vektor/
├── vektor/
│   ├── ingestion/      # Chunker, embedder, document loaders
│   ├── index/          # base, flat, hnsw, bm25, ivf (stub)
│   ├── retrieval/      # hybrid fusion (RRF), query understanding
│   ├── rerank/         # cross-encoder inference + fine-tune + feedback store
│   ├── api/            # FastAPI app, Redis cache, Prometheus metrics
│   └── eval/           # recall, MRR, NDCG, latency
├── scripts/            # download/build_index/benchmark/loadtest/synth_clicks
├── tests/              # 36 pytest tests
├── docs/               # architecture, hnsw-notes, benchmarks
├── infra/              # prometheus config (k8s manifests TBD)
├── Dockerfile
├── docker-compose.yml  # api + redis + prometheus + grafana
└── pyproject.toml
```

## Roadmap

Things worth adding next:
- **IVF index** for the benchmark comparison table (the stub is in `vektor/index/ivf.py`).
- **K8s manifests** under `infra/k8s/` for a real deploy.
- **Sharding** for >10M documents.
- **Query expansion** with a small generative model (currently a no-op passthrough).
- **A/B testing harness** for comparing ranking strategies in prod.

## Documentation

- [Architecture deep-dive](./docs/architecture.md)
- [HNSW implementation notes](./docs/hnsw-notes.md)
- [Benchmark methodology](./docs/benchmarks.md)

## License

MIT — see [LICENSE](./LICENSE).

## Author

Kriti Raj · [@kriti2110](https://github.com/kriti2110)
