from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from vektor.api import metrics
from vektor.api.state import state
from vektor.rerank.feedback import FeedbackEvent, FeedbackEventType


router = APIRouter()


class FeedbackRequest(BaseModel):
    query_id: str
    query_text: str
    doc_id: str
    rank: int = Field(ge=0)
    event_type: FeedbackEventType
    dwell_ms: int | None = Field(default=None, ge=0)


@router.post("/feedback")
async def feedback(req: FeedbackRequest) -> dict:
    if state.feedback is None:
        raise HTTPException(503, "feedback store not initialized")

    event = FeedbackEvent(
        query_id=req.query_id,
        query_text=req.query_text,
        doc_id=req.doc_id,
        rank=req.rank,
        event_type=req.event_type,
        dwell_ms=req.dwell_ms,
    )
    state.feedback.record(event)
    metrics.feedback_events.labels(event_type=req.event_type.value).inc()
    return {"status": "ok"}
