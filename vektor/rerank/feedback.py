from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class FeedbackEventType(str, Enum):
    CLICK = "click"
    DWELL = "dwell"  # time spent on result before returning
    SKIP = "skip"  # result shown but user picked a lower-ranked one
    EXIT = "exit"  # user clicked then bounced quickly


@dataclass
class FeedbackEvent:
    query_id: str
    query_text: str
    doc_id: str
    rank: int
    event_type: FeedbackEventType
    dwell_ms: int | None = None
    timestamp: float | None = None


class FeedbackStore:
    """SQLite-backed click-feedback store. Feeds the reranker training loop."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT NOT NULL,
                query_text TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                rank INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                dwell_ms INTEGER,
                timestamp REAL NOT NULL
            )
            """
        )
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_query_id ON feedback(query_id)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON feedback(timestamp)")
        self._conn.commit()

    def record(self, event: FeedbackEvent) -> None:
        ts = event.timestamp if event.timestamp is not None else time.time()
        self._conn.execute(
            "INSERT INTO feedback (query_id, query_text, doc_id, rank, event_type, dwell_ms, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                event.query_id,
                event.query_text,
                event.doc_id,
                event.rank,
                event.event_type.value,
                event.dwell_ms,
                ts,
            ),
        )
        self._conn.commit()

    def iter_training_triples(self, min_clicks: int = 1):
        """Yield (query_text, clicked_doc_id, skipped_doc_id) for ranking loss.

        Standard heuristic: for each query, treat clicked docs as positives
        and docs shown above the click that were NOT clicked as negatives
        (skip-above strategy from Joachims et al.).
        """
        cur = self._conn.execute(
            "SELECT query_id, query_text, doc_id, rank, event_type FROM feedback "
            "ORDER BY query_id, rank"
        )
        by_query: dict[str, list[tuple[str, str, int, str]]] = {}
        for query_id, query_text, doc_id, rank, event_type in cur:
            by_query.setdefault(query_id, []).append((query_text, doc_id, rank, event_type))

        for events in by_query.values():
            clicked = [(qt, did, r) for qt, did, r, et in events if et == FeedbackEventType.CLICK.value]
            if len(clicked) < min_clicks:
                continue
            for qt, pos_did, pos_rank in clicked:
                for _, did, r, et in events:
                    if et != FeedbackEventType.CLICK.value and r < pos_rank:
                        yield qt, pos_did, did

    def close(self) -> None:
        self._conn.close()
