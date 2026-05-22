"""HNSW — Hierarchical Navigable Small World (Malkov & Yashunin, 2016).

Pure-numpy + heapq implementation. No hnswlib, no faiss.

Algorithm summary:
  - Multi-layer graph. Upper layers are sparse (long jumps), layer 0 dense.
  - Each node assigned a max level via geometric distribution: level = floor(-ln(U) * mL).
  - Insert: greedy descend from entry_point above new_level; from min(new_level, max_level)
    down to 0, search_layer with ef_construction, pick M neighbors via the diversity
    heuristic, link bidirectionally and prune over-connected neighbors.
  - Search: greedy descend from entry_point to layer 1 with ef=1; at layer 0 run a
    proper beam search with ef=ef_search; return top-k.

Vectors are L2-normalized externally so dot(a,b) == cos(a,b). Distance is 1 - cos_sim;
score returned to callers is cos_sim (1.0 = identical).
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
    # neighbors[layer] = list[int_node_idx]
    neighbors: dict[int, list[int]] = field(default_factory=dict)


class HNSWIndex(BaseIndex):
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
        self.M_max0 = 2 * M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.mL = 1.0 / math.log(M) if M > 1 else 1.0

        self._rng = random.Random(seed)
        self._nodes: list[_Node] = []
        self._id_to_idx: dict[str, int] = {}
        self._entry_point: int | None = None
        self._max_level: int = -1

    @property
    def size(self) -> int:
        return len(self._nodes)

    @property
    def dim(self) -> int:
        return self._dim

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, vector: np.ndarray, doc_id: str) -> None:
        vec = np.asarray(vector, dtype=np.float32).reshape(self._dim)
        if doc_id in self._id_to_idx:
            return  # idempotent — duplicate inserts are no-ops

        new_idx = len(self._nodes)
        new_level = self._assign_level()
        new_node = _Node(doc_id=doc_id, vector=vec, neighbors={})
        self._nodes.append(new_node)
        self._id_to_idx[doc_id] = new_idx

        # First node bootstraps the graph
        if self._entry_point is None:
            for layer in range(new_level + 1):
                new_node.neighbors[layer] = []
            self._entry_point = new_idx
            self._max_level = new_level
            return

        ep = self._entry_point
        current_max = self._max_level

        # Phase 1: greedy descent from current_max down to new_level + 1 (no inserts)
        for layer in range(current_max, new_level, -1):
            results = self._search_layer(vec, ep, ef=1, layer=layer)
            if results:
                ep = results[0][1]

        # Phase 2: insert at every layer from min(new_level, current_max) down to 0
        for layer in range(min(new_level, current_max), -1, -1):
            M_layer = self.M_max0 if layer == 0 else self.M
            candidates = self._search_layer(vec, ep, ef=self.ef_construction, layer=layer)
            neighbor_idxs = self._select_neighbors_heuristic(vec, candidates, M_layer)

            new_node.neighbors[layer] = list(neighbor_idxs)

            # bidirectional: add new_idx to each neighbor, prune if over budget
            for n_idx in neighbor_idxs:
                n_node = self._nodes[n_idx]
                n_neighbors = n_node.neighbors.setdefault(layer, [])
                if new_idx not in n_neighbors:
                    n_neighbors.append(new_idx)
                if len(n_neighbors) > M_layer:
                    # re-prune via heuristic from n's perspective
                    n_vec = n_node.vector
                    cands_with_dist = [
                        (self._distance(n_vec, self._nodes[nn].vector), nn)
                        for nn in n_neighbors
                    ]
                    cands_with_dist.sort(key=lambda x: x[0])
                    n_node.neighbors[layer] = self._select_neighbors_heuristic(
                        n_vec, cands_with_dist, M_layer
                    )

            if candidates:
                ep = candidates[0][1]

        # Phase 3: if new node sits above current_max, it becomes the new entry
        if new_level > current_max:
            for layer in range(current_max + 1, new_level + 1):
                new_node.neighbors[layer] = []
            self._max_level = new_level
            self._entry_point = new_idx

    def add_batch(self, vectors: np.ndarray, doc_ids: list[str]) -> None:
        if len(vectors) != len(doc_ids):
            raise ValueError("len(vectors) must equal len(doc_ids)")
        for vec, did in zip(vectors, doc_ids, strict=True):
            self.add(vec, did)

    def search(self, query: np.ndarray, k: int) -> list[SearchResult]:
        if self._entry_point is None or not self._nodes:
            return []

        q = np.asarray(query, dtype=np.float32).reshape(self._dim)
        ep = self._entry_point

        # greedy descent from max_level down to layer 1
        for layer in range(self._max_level, 0, -1):
            results = self._search_layer(q, ep, ef=1, layer=layer)
            if results:
                ep = results[0][1]

        # beam search at layer 0
        ef = max(self.ef_search, k)
        layer0 = self._search_layer(q, ep, ef=ef, layer=0)

        # convert distance → similarity score (higher = better)
        return [
            SearchResult(doc_id=self._nodes[idx].doc_id, score=1.0 - dist)
            for dist, idx in layer0[:k]
        ]

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
    # Internals
    # ------------------------------------------------------------------

    def _assign_level(self) -> int:
        # inverse-transform sample from geometric: P(level == l) decays exponentially
        u = self._rng.random()
        if u <= 0.0:
            u = 1e-12
        return int(-math.log(u) * self.mL)

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        # cosine distance for L2-normalized vectors: 1 - dot
        return 1.0 - float(np.dot(a, b))

    def _search_layer(
        self,
        query: np.ndarray,
        entry_idx: int,
        ef: int,
        layer: int,
    ) -> list[tuple[float, int]]:
        """Greedy beam search at a single layer.

        Returns [(distance, node_idx), ...] sorted ascending by distance,
        length <= ef.
        """
        visited: set[int] = {entry_idx}
        entry_dist = self._distance(query, self._nodes[entry_idx].vector)

        # candidates: min-heap by distance; closest unexplored explored first
        candidates: list[tuple[float, int]] = [(entry_dist, entry_idx)]
        # results: max-heap (negated) by distance; tracks current top-ef
        results: list[tuple[float, int]] = [(-entry_dist, entry_idx)]

        while candidates:
            c_dist, c_idx = heappop(candidates)
            # furthest currently in results
            f_dist = -results[0][0]
            if c_dist > f_dist:
                # no remaining candidate can beat our top-ef
                break

            for n_idx in self._nodes[c_idx].neighbors.get(layer, []):
                if n_idx in visited:
                    continue
                visited.add(n_idx)
                n_dist = self._distance(query, self._nodes[n_idx].vector)
                f_dist = -results[0][0]
                if n_dist < f_dist or len(results) < ef:
                    heappush(candidates, (n_dist, n_idx))
                    heappush(results, (-n_dist, n_idx))
                    if len(results) > ef:
                        heappop(results)

        return sorted((-neg_d, idx) for neg_d, idx in results)

    def _select_neighbors_heuristic(
        self,
        query: np.ndarray,
        candidates: list[tuple[float, int]],
        M: int,
    ) -> list[int]:
        """Pick up to M neighbors that are close to `query` AND diverse from each other.

        candidates: [(distance_to_query, node_idx)], ascending by distance.

        Heuristic (Malkov §4 simplified): accept c only if c is closer to query
        than to any already-selected neighbor. This prevents the graph from
        collapsing edges into one cluster direction.
        """
        selected: list[int] = []
        for cand_dist, cand_idx in candidates:
            if len(selected) >= M:
                break
            cand_vec = self._nodes[cand_idx].vector
            keep = True
            for sel_idx in selected:
                sel_vec = self._nodes[sel_idx].vector
                if self._distance(cand_vec, sel_vec) < cand_dist:
                    keep = False
                    break
            if keep:
                selected.append(cand_idx)
        return selected
