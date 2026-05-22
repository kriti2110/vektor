# Benchmarks

## Methodology

All benchmarks are reproducible from a fresh checkout:

```bash
pip install -e ".[dev]"
python scripts/download_wikipedia.py --n 100000  # or 1000000 for full run
python scripts/build_index.py --backend hnsw --m 16 --ef 200
python scripts/benchmark.py --backend hnsw --queries data/eval_queries.jsonl
```

## Datasets

| Dataset | Docs | Avg chunk size | Source |
|---------|------|----------------|--------|
| wiki-10k | 10,000 | ~250 tokens | Wikipedia abstracts subset |
| wiki-100k | 100,000 | ~250 tokens | Wikipedia full-article subset |
| wiki-1M | 1,000,000 | ~400 tokens | Wikipedia dump (full articles, chunked) |

Eval query sets:
- `eval/wiki_natural_queries.jsonl` — 1000 natural-language queries with relevance judgments (qrels from BEIR's NQ subset).
- `eval/synth_paraphrase.jsonl` — 500 synthetically-paraphrased queries to test semantic robustness.

## Metrics

| Metric | Definition |
|--------|------------|
| **Recall@k** | Fraction of relevant docs (per qrels) in the top-k results |
| **MRR@10** | Mean Reciprocal Rank of first relevant result, capped at rank 10 |
| **p50/p95/p99 latency** | Single-query latency from API receive to response send (excludes network) |
| **QPS** | Concurrent throughput at saturation under Locust load |

## Results (to be filled in as implementation completes)

### Recall vs flat baseline (wiki-100k, 1000 queries)

| Backend | Recall@10 | Recall@100 | Build time | Index size |
|---------|-----------|------------|------------|------------|
| flat | 1.000 (baseline) | 1.000 | — | — |
| HNSW (M=16, ef=200) | TBD | TBD | TBD | TBD |
| IVF (nlist=1024) | TBD | TBD | TBD | TBD |
| BM25 | TBD | TBD | — | — |

### Latency (wiki-1M, single thread, post-warmup)

| Backend | p50 | p95 | p99 |
|---------|-----|-----|-----|
| flat | — (slow, will not benchmark at 1M) | — | — |
| HNSW (ef_search=50) | TBD | TBD | TBD |
| HNSW (ef_search=200) | TBD | TBD | TBD |
| Hybrid (HNSW + BM25 + RRF) | TBD | TBD | TBD |
| Hybrid + cross-encoder rerank | TBD | TBD | TBD |

### Hybrid uplift over single-backend

| Strategy | MRR@10 | Δ vs HNSW only |
|----------|--------|----------------|
| HNSW only | TBD | — |
| BM25 only | TBD | TBD |
| HNSW + BM25 (RRF) | TBD | TBD |
| HNSW + BM25 + reranker | TBD | TBD |

### Reranker fine-tune uplift

| Reranker | MRR@10 |
|----------|--------|
| No reranker | TBD |
| Base cross-encoder | TBD |
| Fine-tuned on synthetic clicks | TBD |
| Fine-tuned on real clicks (1 week of traffic) | TBD |

### Load test (Locust, 4 workers, wiki-100k)

| Concurrent users | QPS | p99 latency |
|------------------|-----|-------------|
| 100 | TBD | TBD |
| 500 | TBD | TBD |
| 1000 | TBD | TBD |

## Hardware

Benchmarks documented above are run on:
- Apple M2 Pro, 16GB RAM, macOS 14
- Single process, no GPU (CPU-only inference)

Production-scale numbers should be re-run on:
- Linux, x86_64, 32GB+ RAM
- Optional GPU for embedder/reranker (10x speedup typical)
