from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from vektor.api import metrics
from vektor.api.state import state
from vektor.ingestion.chunker import SemanticChunker


router = APIRouter()


class IngestRequest(BaseModel):
    doc_id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1)
    metadata: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    doc_id: str
    chunks_ingested: int


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest) -> IngestResponse:
    if state.embedder is None or state.dense_index is None:
        raise HTTPException(503, "index/embedder not initialized")

    chunker = SemanticChunker()
    chunks = chunker.chunk(req.doc_id, req.text, req.metadata)
    if not chunks:
        return IngestResponse(doc_id=req.doc_id, chunks_ingested=0)

    texts = [c.text for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]
    vectors = state.embedder.encode(texts)
    state.dense_index.add_batch(vectors, chunk_ids)

    if state.sparse_index is not None:
        state.sparse_index.add_batch(texts, chunk_ids)

    if state.doc_text is not None:
        for c in chunks:
            state.doc_text[c.chunk_id] = c.text

    metrics.index_size.labels(backend="dense").set(state.dense_index.size)
    return IngestResponse(doc_id=req.doc_id, chunks_ingested=len(chunks))
