from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Response

from vektor import __version__
from vektor.api import metrics
from vektor.api.cache import QueryCache
from vektor.api.routes import feedback as feedback_routes
from vektor.api.routes import ingest as ingest_routes
from vektor.api.routes import search as search_routes
from vektor.api.state import state
from vektor.config import settings
from vektor.index.bm25 import BM25Index
from vektor.index.flat import FlatIndex
from vektor.index.hnsw import HNSWIndex
from vektor.ingestion.embedder import Embedder
from vektor.rerank.feedback import FeedbackStore


logger = logging.getLogger("vektor")


def _load_dense_index(path: Path):
    """Auto-detect backend from filename suffix."""
    suffix = path.suffix.lstrip(".")
    if suffix == "hnsw":
        idx = HNSWIndex(dim=settings.embed_dim)
    elif suffix == "flat":
        idx = FlatIndex(dim=settings.embed_dim)
    else:
        raise ValueError(f"unknown dense-index suffix: {suffix} (expected .hnsw or .flat)")
    idx.load(path)
    return idx


def _load_doc_store(path: Path) -> dict[str, str]:
    """Load a chunk_id → text mapping from JSONL ({chunk_id, text} per line)."""
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            out[obj["chunk_id"]] = obj["text"]
    return out


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=settings.log_level)
    logger.info("vektor %s starting up", __version__)

    state.embedder = Embedder(
        model_name=settings.embed_model,
        dim=settings.embed_dim,
        batch_size=settings.embed_batch_size,
        cache_path=settings.embed_cache_path,
    )
    state.feedback = FeedbackStore(settings.feedback_db_path)
    state.cache = QueryCache(settings.redis_url, ttl_seconds=settings.query_cache_ttl)
    await state.cache.connect()
    if state.cache.available:
        logger.info("redis cache connected at %s", settings.redis_url)
    else:
        logger.warning("redis unavailable at %s — running without cache", settings.redis_url)

    # Optional: load pre-built index from disk
    if settings.dense_index_path is not None and Path(settings.dense_index_path).exists():
        logger.info("loading dense index from %s", settings.dense_index_path)
        state.dense_index = _load_dense_index(Path(settings.dense_index_path))
        metrics.index_size.labels(backend="dense").set(state.dense_index.size)

    if settings.sparse_index_path is not None and Path(settings.sparse_index_path).exists():
        logger.info("loading sparse index from %s", settings.sparse_index_path)
        sparse = BM25Index()
        sparse.load(Path(settings.sparse_index_path))
        state.sparse_index = sparse
        metrics.index_size.labels(backend="sparse").set(state.sparse_index.size)

    # Doc text lookup (for rerank). Use disk-based JSONL if present, else in-memory dict.
    state.doc_text = {}
    if settings.doc_store_path is not None and Path(settings.doc_store_path).exists():
        logger.info("loading doc store from %s", settings.doc_store_path)
        state.doc_text = _load_doc_store(Path(settings.doc_store_path))

    # Optional: load reranker (slow, ~600MB; off by default)
    if settings.enable_reranker:
        from vektor.rerank.cross_encoder import CrossEncoderReranker

        logger.info("loading cross-encoder reranker (%s)", settings.rerank_model)
        state.reranker = CrossEncoderReranker(model_name=settings.rerank_model)

    yield

    logger.info("vektor shutting down")
    if state.cache is not None:
        await state.cache.close()
    if state.feedback is not None:
        state.feedback.close()


app = FastAPI(
    title="VEKTOR",
    version=__version__,
    description="Semantic search infrastructure — HNSW + hybrid retrieval + self-improving reranker",
    lifespan=lifespan,
)

app.include_router(search_routes.router, tags=["search"])
app.include_router(ingest_routes.router, tags=["ingest"])
app.include_router(feedback_routes.router, tags=["feedback"])


@app.get("/healthz")
async def healthz() -> dict:
    return {
        "status": "ok",
        "version": __version__,
        "dense_index_size": state.dense_index.size if state.dense_index is not None else 0,
        "sparse_index_size": state.sparse_index.size if state.sparse_index is not None else 0,
        "reranker_loaded": state.reranker is not None,
        "cache_available": state.cache.available if state.cache is not None else False,
    }


@app.get("/metrics")
async def get_metrics() -> Response:
    body, content_type = metrics.render_metrics()
    return Response(content=body, media_type=content_type)
