"""IVF (Inverted File) index — k-means clusters + per-cluster posting lists.

Included as a stub for benchmark comparison. You can implement this AFTER
HNSW is working — it's an easier algorithm (k-means + probe nearest clusters)
and useful for the benchmark table in docs/benchmarks.md.

For now, raises NotImplementedError. Not on the critical path.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from vektor.index.base import BaseIndex, SearchResult


class IVFIndex(BaseIndex):
    """k-means + inverted file. O(nlist * d) to train, O((nlist + nprobe * N/nlist) * d) per query."""

    def __init__(self, dim: int, nlist: int = 1024, nprobe: int = 8) -> None:
        self._dim = dim
        self.nlist = nlist
        self.nprobe = nprobe
        self._centroids: np.ndarray | None = None  # (nlist, dim)
        self._postings: list[list[int]] = []
        self._vectors: np.ndarray | None = None
        self._doc_ids: list[str] = []
        self._trained = False

    @property
    def size(self) -> int:
        return 0 if self._vectors is None else self._vectors.shape[0]

    @property
    def dim(self) -> int:
        return self._dim

    def train(self, training_vectors: np.ndarray) -> None:
        # TODO(later): k-means++ on training_vectors to get nlist centroids.
        raise NotImplementedError("IVF train — implement after HNSW")

    def add(self, vector: np.ndarray, doc_id: str) -> None:
        raise NotImplementedError("IVF add — implement after HNSW")

    def add_batch(self, vectors: np.ndarray, doc_ids: list[str]) -> None:
        raise NotImplementedError("IVF add_batch — implement after HNSW")

    def search(self, query: np.ndarray, k: int) -> list[SearchResult]:
        raise NotImplementedError("IVF search — implement after HNSW")

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "dim": self._dim,
                    "nlist": self.nlist,
                    "nprobe": self.nprobe,
                    "centroids": self._centroids,
                    "postings": self._postings,
                    "vectors": self._vectors,
                    "doc_ids": self._doc_ids,
                    "trained": self._trained,
                },
                f,
            )

    def load(self, path: Path) -> None:
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        self._dim = data["dim"]
        self.nlist = data["nlist"]
        self.nprobe = data["nprobe"]
        self._centroids = data["centroids"]
        self._postings = data["postings"]
        self._vectors = data["vectors"]
        self._doc_ids = list(data["doc_ids"])
        self._trained = data["trained"]
