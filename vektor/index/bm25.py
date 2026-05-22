from __future__ import annotations

import pickle
import re
from pathlib import Path

from vektor.index.base import SearchResult


_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text)]


class BM25Index:
    """Sparse BM25 index over tokenized text. Wraps `rank-bm25`.

    Not a `BaseIndex` (it consumes text, not vectors), but exposes the same
    `search(query, k) -> list[SearchResult]` shape so retrieval code can
    treat it uniformly.
    """

    def __init__(self) -> None:
        self._bm25 = None
        self._doc_ids: list[str] = []
        self._tokenized: list[list[str]] = []

    @property
    def size(self) -> int:
        return len(self._doc_ids)

    def add(self, text: str, doc_id: str) -> None:
        self.add_batch([text], [doc_id])

    def add_batch(self, texts: list[str], doc_ids: list[str]) -> None:
        if len(texts) != len(doc_ids):
            raise ValueError("len(texts) must equal len(doc_ids)")
        for text, did in zip(texts, doc_ids, strict=True):
            self._doc_ids.append(did)
            self._tokenized.append(_tokenize(text))
        # rank-bm25 builds the model on the full corpus, so rebuild on each add_batch
        # this is fine for batch ingestion; for streaming, use a different sparse index
        self._rebuild()

    def search(self, query: str, k: int) -> list[SearchResult]:
        if self._bm25 is None or not self._doc_ids:
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        k = min(k, len(scores))
        import numpy as np

        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [SearchResult(doc_id=self._doc_ids[i], score=float(scores[i])) for i in top_idx]

    def _rebuild(self) -> None:
        from rank_bm25 import BM25Okapi

        self._bm25 = BM25Okapi(self._tokenized)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"doc_ids": self._doc_ids, "tokenized": self._tokenized}, f)

    def load(self, path: Path) -> None:
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        self._doc_ids = list(data["doc_ids"])
        self._tokenized = list(data["tokenized"])
        self._rebuild()
