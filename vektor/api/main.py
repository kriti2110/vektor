from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response

from vektor import __version__
from vektor.api import metrics
from vektor.api.cache import QueryCache
from vektor.api.routes import feedback as feedback_routes
from vektor.api.routes import ingest as ingest_routes
from vektor.api.routes import search as search_routes
from vektor.api.state import state
from vektor.config import settings
from vektor.ingestion.embedder import Embedder
from vektor.rerank.feedback import FeedbackStore


logger = logging.getLogger("vektor")


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

    state.doc_text = {}

    # Index loading is opt-in via env / script — kept out of startup to keep
    # boot fast. Call scripts/build_index.py then restart, or POST /ingest.

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
        "index_size": state.dense_index.size if state.dense_index is not None else 0,
        "cache_available": state.cache.available if state.cache is not None else False,
    }


@app.get("/metrics")
async def get_metrics() -> Response:
    body, content_type = metrics.render_metrics()
    return Response(content=body, media_type=content_type)
