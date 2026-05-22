"""HNSW index — Hierarchical Navigable Small World.

╔══════════════════════════════════════════════════════════════════════════╗
║                                                                          ║
║   ⚠️  STUB. Kriti — this is yours to implement. See:                     ║
║                                                                          ║
║       docs/hnsw-notes.md      ← read first, then come back               ║
║       TODO_YOU_BUILD.md       ← acceptance criteria                      ║
║       tests/test_hnsw.py      ← what your implementation must pass       ║
║                                                                          ║
║   Do NOT call hnswlib or faiss. The point is to write this from scratch  ║
║   so you can defend it in an interview.                                  ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝

This file contains the skeleton: dataclasses, constructor, method signatures,
save/load wiring, and inline algorithm sketches in comments. The actual
algorithm bodies raise NotImplementedError — that's the part you write.

Suggested order:
  1. Implement `search_layer` (the greedy beam search at a single layer).
  2. Implement `search` using `search_layer` repeatedly down through layers.
  3. Implement `add` (with level assignment + neighbor selection heuristic).
  4. Implement `select_neighbors_heuristic`.
  5. Run `pytest tests/test_hnsw.py` and iterate until green.
"""

from __future__ import annotations

import math
import pickle
import random
from dataclasses import dataclass, field
from heapq import heappop, heappush
from pathlib import Path

import numpy as np

from vektor.index.base import BaseIndex, SearchResult


@dataclass
class _Node:
    doc_id: str
    vector: np.ndarray
    # neighbors[layer] = list[int_node_id]
    neighbors: dict[int, list[int]] = field(default_factory=dict)


class HNSWIndex(BaseIndex):
    """Hierarchical Navigable Small World (Malkov & Yashunin, 2016).

    Parameters
    ----------
    dim : int
        Vector dimensionality.
    M : int
        Max neighbors per node per layer (layer 0 gets 2*M). Typical: 16.
    ef_construction : int
        Beam width during insertion's neighbor search. Higher = better graph
        quality at build time. Typical: 200.
    ef_search : int
        Beam width during query-time search at layer 0. Higher = better recall,
        slower. Typical: 50-200.
    seed : int | None
        RNG seed for level assignment, for reproducibility.
    """

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 100,
        seed: int | None = None,
    ) -> None:
        self._dim = dim
        self.M = M
        self.M_max0 = 2 * M  # layer 0 gets double
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.mL = 1.0 / math.log(M) if M > 1 else 1.0  # level normalization

        self._rng = random.Random(seed)
        self._nodes: list[_Node] = []
        self._id_to_idx: dict[str, int] = {}
        self._entry_point: int | None = None
        self._max_level: int = -1

    # ------------------------------------------------------------------
    # BaseIndex API
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._nodes)

    @property
    def dim(self) -> int:
        return self._dim

    def add(self, vector: np.ndarray, doc_id: str) -> None:
        # TODO(Kriti):
        # 1. Validate shape, normalize if not already, allocate _Node.
        # 2. Sample level via _assign_level().
        # 3. If this is the first node, set entry_point and max_level, return.
        # 4. Greedy descent from max_level down to level+1 with ef=1.
        # 5. From min(level, max_level) down to 0:
        #      - search_layer at this layer with ef=ef_construction
        #      - pick M neighbors via select_neighbors_heuristic
        #      - link bidirectionally; prune over-connected neighbors
        # 6. If level > max_level: update entry_point and max_level.
        raise NotImplementedError("Kriti — implement insert. See docs/hnsw-notes.md §Insert")

    def add_batch(self, vectors: np.ndarray, doc_ids: list[str]) -> None:
        # default impl: just loop. you can optimize later (e.g., parallel inserts
        # are tricky because of the graph mutations).
        for vec, did in zip(vectors, doc_ids, strict=True):
            self.add(vec, did)

    def search(self, query: np.ndarray, k: int) -> list[SearchResult]:
        # TODO(Kriti):
        # 1. Handle empty index.
        # 2. ep = entry_point
        # 3. For layer in max_level..1:  ep = search_layer(query, ep, ef=1, layer)[0]
        # 4. results = search_layer(query, ep, ef=max(ef_search, k), layer=0)
        # 5. Return top-k as SearchResult(doc_id, score=similarity).
        raise NotImplementedError("Kriti — implement search. See docs/hnsw-notes.md §Full search")

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "dim": self._dim,
                    "M": self.M,
                    "ef_construction": self.ef_construction,
                    "ef_search": self.ef_search,
                    "mL": self.mL,
                    "nodes": [
                        {"doc_id": n.doc_id, "vector": n.vector, "neighbors": n.neighbors}
                        for n in self._nodes
                    ],
                    "entry_point": self._entry_point,
                    "max_level": self._max_level,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    def load(self, path: Path) -> None:
        with Path(path).open("rb") as f:
            data = pickle.load(f)
        self._dim = data["dim"]
        self.M = data["M"]
        self.M_max0 = 2 * self.M
        self.ef_construction = data["ef_construction"]
        self.ef_search = data["ef_search"]
        self.mL = data["mL"]
        self._nodes = [
            _Node(doc_id=n["doc_id"], vector=n["vector"], neighbors=n["neighbors"])
            for n in data["nodes"]
        ]
        self._id_to_idx = {n.doc_id: i for i, n in enumerate(self._nodes)}
        self._entry_point = data["entry_point"]
        self._max_level = data["max_level"]

    # ------------------------------------------------------------------
    # Helpers — these are scaffolds for your implementation
    # ------------------------------------------------------------------

    def _assign_level(self) -> int:
        # Geometric distribution via inverse transform sampling.
        # P(level == l) = exp(-l/mL) * (1 - exp(-1/mL))
        return int(-math.log(self._rng.random()) * self.mL)

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        # cosine distance = 1 - cosine_similarity. Both vectors are L2-normalized,
        # so cosine similarity == dot product. Lower distance = more similar.
        return 1.0 - float(np.dot(a, b))

    def _search_layer(
        self,
        query: np.ndarray,
        entry_idx: int,
        ef: int,
        layer: int,
    ) -> list[tuple[float, int]]:
        """Greedy beam search at a single layer.

        Returns a list of (distance, node_idx) sorted ascending by distance,
        truncated to `ef` entries.

        TODO(Kriti) — this is the core primitive everything else builds on.
        See docs/hnsw-notes.md §Search at one layer for the pseudocode.
        """
        # Hint: you'll use heapq. Python's heapq is a min-heap; for the
        # "current results" max-heap, store negated distances.
        # Hint: maintain a visited set per call (not global).
        raise NotImplementedError("Kriti — implement search_layer.")

    def _select_neighbors_heuristic(
        self,
        query: np.ndarray,
        candidates: list[tuple[float, int]],
        M: int,
    ) -> list[int]:
        """Pick M neighbors that are both close to `query` AND diverse.

        candidates: list of (distance_to_query, node_idx), already sorted
        ascending. Returns list of node_idx.

        TODO(Kriti) — see docs/hnsw-notes.md §The neighbor selection heuristic.
        """
        raise NotImplementedError("Kriti — implement select_neighbors_heuristic.")
