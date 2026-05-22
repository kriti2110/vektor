# TODO_YOU_BUILD — components Kriti implements personally

> **Why this file exists:** The interview value of VEKTOR is "I built and understand this." If any of the components below are not implemented in your own hand, the project loses its meaning. The infrastructure around these (FastAPI, Docker, BM25 wrapper, ingestion, eval harness) is fair game to scaffold from templates — but the three pieces below are the ones interviewers will drill into. Build them yourself, then read the matching `docs/` notes to deepen understanding.

---

## 1. HNSW index from scratch — `vektor/index/hnsw.py`

**What:** Implement the Hierarchical Navigable Small World algorithm (Malkov & Yashunin, 2016) end-to-end. No `hnswlib`, no `faiss`. Just numpy.

**Why this matters in an interview:**
- HNSW underpins production search at Google, Microsoft, Meta, Pinecone, Weaviate.
- Common follow-ups: *"How does level assignment work? Why a geometric distribution? What's `ef_construction` vs `ef_search`? Why M neighbors? What happens to recall if you set M too low?"*
- If you've actually written it, all of these become natural conversation. If not, you stumble on follow-up #2.

**Acceptance criteria:**
- `HNSWIndex` class implementing the `BaseIndex` interface in `vektor/index/base.py`.
- Implements: `add(vec, doc_id)`, `search(query, k, ef)`, `save(path)`, `load(path)`.
- Level assignment via geometric distribution (`-ln(uniform) * mL`).
- Greedy search from entry point down to layer 0.
- Neighbor selection heuristic (the M-paper one, not naive top-M).
- Tests in `tests/test_hnsw.py` pass, including:
  - Recall@10 within 2% of flat search on a 10k-doc benchmark.
  - Query latency < 5ms on 100k docs.

**Study before coding:**
1. Read [`docs/hnsw-notes.md`](./docs/hnsw-notes.md) (annotated walkthrough)
2. Read the original paper: https://arxiv.org/abs/1603.09320
3. Skim Pinecone's HNSW explainer for intuition: https://www.pinecone.io/learn/series/faiss/hnsw/

**Suggested order:**
1. Greedy layer-0 search on a fixed graph (no insert yet)
2. Add insert with level=0 only (degrades to NSW)
3. Add multi-layer with level sampling
4. Add the M-neighbor selection heuristic
5. Benchmark vs flat in `scripts/benchmark.py`

---

## 2. Hybrid fusion (RRF) — `vektor/retrieval/hybrid.py`

**What:** Reciprocal Rank Fusion to merge dense (HNSW) and sparse (BM25) result lists.

**Why this matters:**
- Pure dense retrieval misses exact-keyword queries; pure sparse misses semantic paraphrases. Hybrid is what production search actually does.
- Common follow-ups: *"Why RRF instead of weighted score sum? What's `k`? What happens when one retriever returns garbage?"*

**Acceptance criteria:**
- `rrf_fuse(result_lists, k=60)` function — input is a list of ranked `[doc_id, ...]` lists, output is a single fused ranking with scores.
- Handles missing docs (doc in dense but not sparse) correctly.
- Unit tests in `tests/test_hybrid.py` covering: empty inputs, full overlap, no overlap, mismatched lengths.

**Study before coding:**
- Read the RRF paper (Cormack et al., 2009): https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
- The formula is tiny (one paragraph). Understand WHY it works — the inverse-rank smoothing is the whole insight.

**Don't:** write a weighted-sum-of-normalized-scores hybrid. RRF is empirically better and the literature backs it. If you reach for normalization, you've missed the point.

---

## 3. Reranker fine-tuning loop — `vektor/rerank/train.py`

**What:** Fine-tune a small cross-encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) on click-through data collected from the API's `/feedback` endpoint.

**Why this matters:**
- "Self-improving" is what separates this project from every other RAG demo on GitHub.
- Common follow-ups: *"What's your loss function? How do you handle position bias in click data? What's your eval split? How do you prevent the model from collapsing to 'always rerank the same doc to top'?"*

**Acceptance criteria:**
- `train.py` that:
  - Reads click logs from `feedback_store.sqlite`.
  - Constructs (query, positive_doc, negative_doc) triples.
  - Trains with margin ranking loss or InfoNCE.
  - Evaluates on a held-out test set, reports MRR@10 before/after.
- Achieves +8% MRR uplift over the base reranker on a benchmark.

**Position bias mitigation — read these first:**
- Joachims et al. on click-bias correction: https://www.cs.cornell.edu/people/tj/publications/joachims_etal_05a.pdf
- Inverse Propensity Scoring overview.

**Suggested order:**
1. Generate synthetic click logs first (script provided in `scripts/synth_clicks.py`) so you can develop offline.
2. Get training loop running on synthetic data, hit any MRR improvement.
3. Switch to real click logs when API is live.
4. Add position-bias correction.

---

## What's already built for you (and why that's okay)

These components are scaffolded because they're either (a) plumbing that's identical across projects, or (b) library wrappers where re-implementing adds zero interview value:

- `vektor/ingestion/` — chunker, embedder, loaders. Wraps sentence-transformers + sensible defaults.
- `vektor/index/flat.py` — brute-force baseline. You need this to benchmark your HNSW against.
- `vektor/index/bm25.py` — wraps `rank-bm25`. Re-implementing BM25 isn't interview-impressive.
- `vektor/api/` — FastAPI routes, Redis cache, Prometheus middleware.
- `vektor/eval/` — recall/MRR/latency calculators.
- `Dockerfile`, `docker-compose.yml`, CI — infrastructure boilerplate.

**Rule of thumb:** If you can explain in one sentence what the file does and why, you don't need to have written it line-by-line. If an interviewer would ask "walk me through the algorithm" — you'd better have written it yourself.
