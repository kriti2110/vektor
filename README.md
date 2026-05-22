# VEKTOR

**Semantic search infrastructure at scale.** Custom embedding pipeline, from-scratch HNSW index, hybrid dense+sparse retrieval, and a self-improving reranker — designed to serve sub-50ms p99 search over millions of documents.

> **Status:** Active development. Foundation scaffold complete; HNSW, reranker fine-tuning, and RRF fusion are in progress. See [`TODO_YOU_BUILD.md`](./TODO_YOU_BUILD.md) for the build roadmap.

---

## What this is

VEKTOR is a production-grade semantic search engine built from primitives — not a wrapper around an existing vector DB. The goal is to demonstrate end-to-end ownership of every layer of a modern retrieval stack:

| Layer | What it does | Implementation |
|-------|--------------|----------------|
| **Ingestion** | Chunk docs at semantic boundaries, embed in batches, cache incrementally | `vektor/ingestion/` |
| **Index** | HNSW (from scratch) + IVF + flat baseline + BM25 sparse | `vektor/index/` |
| **Retrieval** | Hybrid dense+sparse fusion via Reciprocal Rank Fusion | `vektor/retrieval/` |
| **Reranking** | Cross-encoder reranker, fine-tuned on click feedback | `vektor/rerank/` |
| **Serving** | Async FastAPI, Redis cache, Prometheus metrics, Docker/K8s | `vektor/api/` |
| **Evaluation** | Recall@k, MRR, p50/p95/p99 latency, A/B test harness | `vektor/eval/` |

## Target benchmarks

| Metric | Target | Current |
|--------|--------|---------|
| Index size | 1M+ Wikipedia documents | — |
| p99 query latency | < 50ms | — |
| Recall@10 (HNSW vs flat) | within 2% | — |
| Concurrent users (load test) | 1000+ | — |
| Reranker MRR uplift vs retrieval-only | +8% | — |

Benchmarks are reproducible via `scripts/benchmark.py` and `scripts/loadtest.py`.

## Quick start

```bash
# Install
git clone https://github.com/kriti2110/vektor.git
cd vektor
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run the API + Redis + Prometheus stack
docker compose up -d

# Or run the API standalone
uvicorn vektor.api.main:app --reload

# Ingest a small dataset
python scripts/build_index.py --source data/sample.jsonl --index hnsw

# Query
curl -X POST localhost:8000/search \
  -H 'content-type: application/json' \
  -d '{"query": "transformers in machine learning", "k": 10}'
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
                          │ top-100
            ┌─────────────▼──────────────┐
            │   Cross-Encoder Reranker   │
            │  (fine-tuned on click log) │
            └─────────────┬──────────────┘
                          │ top-10
                  ┌───────▼───────┐
                  │    Results    │
                  └───────────────┘
```

Full architecture notes in [`docs/architecture.md`](./docs/architecture.md).

## Project layout

```
vektor/
├── vektor/
│   ├── ingestion/      # Chunker, embedder, document loaders
│   ├── index/          # Flat, HNSW, IVF, BM25
│   ├── retrieval/      # Hybrid fusion, query understanding
│   ├── rerank/         # Cross-encoder reranking + feedback loop
│   ├── api/            # FastAPI app, caching, metrics
│   └── eval/           # Recall, MRR, latency evaluators
├── scripts/            # Index builders, benchmarks, load tests
├── tests/              # pytest unit + integration tests
├── docs/               # Architecture, HNSW notes, benchmark writeups
├── infra/              # Prometheus config, k8s manifests (TBD)
├── Dockerfile
├── docker-compose.yml  # API + Redis + Prometheus + Grafana
└── pyproject.toml
```

## Documentation

- [Architecture deep-dive](./docs/architecture.md)
- [HNSW implementation notes](./docs/hnsw-notes.md)
- [Benchmark methodology](./docs/benchmarks.md)
- [Build roadmap](./TODO_YOU_BUILD.md)

## License

MIT — see [LICENSE](./LICENSE).

## Author

Kriti Raj · [@kriti2110](https://github.com/kriti2110)
