from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from vektor.api import metrics
from vektor.api.cache import cache_key
from vektor.api.state import state
from vektor.retrieval.query import classify_intent, normalize_query


router = APIRouter()


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1024)
    k: int = Field(default=10, ge=1, le=100)
    strategy: str = Field(default="hybrid", pattern="^(dense|sparse|hybrid)$")
    rerank: bool = True


class SearchHit(BaseModel):
    doc_id: str
    score: float
    rank: int


class SearchResponse(BaseModel):
    query_id: str
    query: str
    intent: str
    hits: list[SearchHit]
    timings_ms: dict[str, float]
    cache_hit: bool


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    if state.dense_index is None and state.sparse_index is None:
        raise HTTPException(503, "no index loaded — run scripts/build_index.py first")

    t0 = time.perf_counter()
    q_norm = normalize_query(req.query)
    intent = classify_intent(req.query)

    # cache lookup
    key = cache_key("search", q_norm, req.k, req.strategy, req.rerank)
    if state.cache is not None:
        cached = await state.cache.get(key)
        if cached is not None:
            metrics.cache_hits.labels(layer="query").inc()
            cached["cache_hit"] = True
            return SearchResponse(**cached)
        metrics.cache_misses.labels(layer="query").inc()

    timings: dict[str, float] = {}
    dense_hits = []
    sparse_hits = []

    # retrieval
    if req.strategy in ("dense", "hybrid") and state.dense_index is not None:
        if state.embedder is None:
            raise HTTPException(503, "embedder not initialized")
        td = time.perf_counter()
        q_vec = state.embedder.encode_one(req.query)
        retrieve_k = max(req.k * 5, 50) if req.rerank else req.k
        dense_hits = state.dense_index.search(q_vec, k=retrieve_k)
        timings["dense_retrieval_ms"] = (time.perf_counter() - td) * 1000
        metrics.retrieval_duration.labels(backend="dense").observe(timings["dense_retrieval_ms"] / 1000)

    if req.strategy in ("sparse", "hybrid") and state.sparse_index is not None:
        ts = time.perf_counter()
        retrieve_k = max(req.k * 5, 50) if req.rerank else req.k
        sparse_hits = state.sparse_index.search(req.query, k=retrieve_k)
        timings["sparse_retrieval_ms"] = (time.perf_counter() - ts) * 1000
        metrics.retrieval_duration.labels(backend="sparse").observe(timings["sparse_retrieval_ms"] / 1000)

    # fusion
    if req.strategy == "hybrid" and dense_hits and sparse_hits:
        try:
            from vektor.retrieval.hybrid import rrf_fuse

            fused = rrf_fuse([dense_hits, sparse_hits], top_k=max(req.k * 5, 50) if req.rerank else req.k)
        except NotImplementedError:
            # RRF not yet implemented — fall back to dense for now
            fused = dense_hits
    else:
        fused = dense_hits or sparse_hits

    # rerank
    if req.rerank and state.reranker is not None and state.doc_text is not None and fused:
        tr = time.perf_counter()
        fused = state.reranker.rerank(
            req.query,
            fused,
            doc_text_lookup=lambda did: state.doc_text.get(did, ""),
            top_k=req.k,
        )
        timings["rerank_ms"] = (time.perf_counter() - tr) * 1000
        metrics.rerank_duration.labels(model=state.reranker.model_name).observe(
            timings["rerank_ms"] / 1000
        )
    else:
        fused = fused[: req.k]

    hits = [SearchHit(doc_id=r.doc_id, score=r.score, rank=i) for i, r in enumerate(fused)]
    timings["total_ms"] = (time.perf_counter() - t0) * 1000

    resp = SearchResponse(
        query_id=str(uuid.uuid4()),
        query=req.query,
        intent=intent.value,
        hits=hits,
        timings_ms=timings,
        cache_hit=False,
    )

    if state.cache is not None:
        await state.cache.set(key, resp.model_dump())

    metrics.request_duration.labels(route="/search", status="200").observe(
        timings["total_ms"] / 1000
    )
    return resp
