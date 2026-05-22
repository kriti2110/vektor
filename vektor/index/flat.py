from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from vektor.index.base import BaseIndex, SearchResult


class FlatIndex(BaseIndex):
    """Brute-force exact search. The recall ground truth that HNSW/IVF benchmark against.

    O(N * d) per query. Useful up to ~100k vectors with MiniLM dims (~38M floats),
    after which it gets too slow for interactive use.
    """

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._vectors: np.ndarray | None = None  # (N, dim), L2-normalized
        self._doc_ids: list[str] = []
        self._id_to_idx: dict[str, int] = {}

    @property
    def size(self) -> int:
        return 0 if self._vectors is None else self._vectors.shape[0]

    @property
    def dim(self) -> int:
        return self._dim

    def add(self, vector: np.ndarray, doc_id: str) -> None:
        self.add_batch(vector.reshape(1, -1), [doc_id])

    def add_batch(self, vectors: np.ndarray, doc_ids: list[str]) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise ValueError(f"expected shape (n, {self._dim}), got {vectors.shape}")
        if len(doc_ids) != vectors.shape[0]:
            raise ValueError("len(doc_ids) must equal vectors.shape[0]")

        vectors = vectors.astype(np.float32, copy=False)

        if self._vectors is None:
            self._vectors = vectors.copy()
        else:
            self._vectors = np.concatenate([self._vectors, vectors], axis=0)

        start = len(self._doc_ids)
        for offset, did in enumerate(doc_ids):
            self._id_to_idx[did] = start + offset
        self._doc_ids.extend(doc_ids)

    def search(self, query: np.ndarray, k: int) -> list[SearchResult]:
        if self._vectors is None or self._vectors.shape[0] == 0:
            return []

        q = query.astype(np.float32, copy=False).reshape(-1)
        # dot product == cosine since both sides are L2-normalized
        scores = self._vectors @ q

        k = min(k, scores.shape[0])
        # argpartition for O(N) top-k, then sort just the top-k
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        return [
            SearchResult(doc_id=self._doc_ids[i], score=float(scores[i]))
            for i in top_idx
        ]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "dim": self._dim,
                    "vectors": self._vectors,
                    "doc_ids": self._doc_ids,
                },
                f,
            )

    def load(self, path: Path) -> None:
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        self._dim = data["dim"]
        self._vectors = data["vectors"]
        self._doc_ids = list(data["doc_ids"])
        self._id_to_idx = {did: i for i, did in enumerate(self._doc_ids)}
