# TODO — what's next

The core pipeline (ingestion → HNSW + BM25 → RRF → cross-encoder rerank → FastAPI)
is implemented and tested. This file lists the work left to make VEKTOR a
complete production-quality system.

## Near-term (next 2 weeks)

- [ ] **Run the 1M-doc benchmark.** Download a larger Wikipedia subset and fill
  in the empty rows in `docs/benchmarks.md`. This is the number that goes on
  the resume.
- [ ] **IVF index implementation** (`vektor/index/ivf.py`). The skeleton is
  there. k-means++ for centroids, posting lists per cluster, `nprobe`-cluster
  search at query time.
- [ ] **Position-bias correction** in `vektor/rerank/train.py`. Current loop
  is naive; add inverse propensity scoring (Joachims 2005).
- [ ] **End-to-end test for `/search` with hybrid + rerank.** The API smoke
  test currently only checks `/healthz` and `/metrics`; add one that ingests
  a few docs and runs a real query.

## Medium-term

- [ ] **K8s manifests** under `infra/k8s/`. Deployment, Service, HPA,
  ConfigMap, ServiceMonitor for Prometheus Operator.
- [ ] **Sharding strategy** for indexes > 10M vectors. Hash-partition `doc_id`,
  scatter-gather across pods.
- [ ] **Query expansion** in `vektor/retrieval/query.py` (currently a no-op).
  Smallest path: synonym lookup from WordNet. Better path: t5-small fine-tuned
  on a query-reformulation dataset.
- [ ] **Online A/B testing harness.** Two ranking strategies behind a feature
  flag, log per-variant metrics, ship a winner.
- [ ] **Graceful index updates.** Currently insertions are supported but
  deletes are not. Tombstone-then-rebuild is the standard approach.

## Stretch — paper implementations for benchmark comparison

Pick one and add it to the benchmark table:

- [ ] **ColBERT** late-interaction retrieval
- [ ] **SPLADE** sparse-lexical hybrid
- [ ] **FAISS-style product quantization** for HNSW compression

Implementing one of these is what turns a strong project into a memorable one.

## Code quality

- [ ] Bump test coverage past 80% (`pytest --cov` currently around 60%).
- [ ] mypy strict mode (currently advisory).
- [ ] Migrate the in-memory `state.doc_text` to a real KV store (Postgres or
  RocksDB) for >100k chunks.
