from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    score: float


class BaseIndex(ABC):
    """Interface every dense index implements.

    Vectors are expected to be L2-normalized so that dot-product == cosine
    similarity. Score convention: HIGHER IS BETTER. Implementations that
    naturally produce distances must convert.
    """

    @abstractmethod
    def add(self, vector: np.ndarray, doc_id: str) -> None:
        """Add a single vector. vector shape == (dim,)."""

    @abstractmethod
    def add_batch(self, vectors: np.ndarray, doc_ids: list[str]) -> None:
        """Add a batch of vectors. vectors shape == (n, dim)."""

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> list[SearchResult]:
        """Return top-k results, highest score first."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist index to disk."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load index from disk. Replaces in-memory state."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of indexed vectors."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Vector dimensionality."""
