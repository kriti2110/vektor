from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class _EmbeddingCache:
    """SQLite-backed key→vector cache. Cheap, persistent, no external deps."""

    def __init__(self, path: Path, dim: int) -> None:
        self.path = Path(path)
        self.dim = dim
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "  key TEXT PRIMARY KEY,"
            "  vec BLOB NOT NULL,"
            "  dim INTEGER NOT NULL"
            ")"
        )
        self._conn.commit()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        if not keys:
            return {}
        placeholders = ",".join("?" * len(keys))
        cur = self._conn.execute(
            f"SELECT key, vec, dim FROM embeddings WHERE key IN ({placeholders})", keys
        )
        out: dict[str, np.ndarray] = {}
        for key, blob, dim in cur:
            if dim != self.dim:
                continue  # stale entry from a different model
            out[key] = np.frombuffer(blob, dtype=np.float32).reshape(dim)
        return out

    def put_many(self, items: Iterable[tuple[str, np.ndarray]]) -> None:
        rows = [(k, v.astype(np.float32).tobytes(), self.dim) for k, v in items]
        self._conn.executemany(
            "INSERT OR REPLACE INTO embeddings (key, vec, dim) VALUES (?, ?, ?)", rows
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class Embedder:
    """Batched sentence-transformer embedder with persistent cache.

    First call lazily loads the model. Subsequent calls reuse it. All texts
    are L2-normalized after encoding so dot-product == cosine.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dim: int = 384,
        batch_size: int = 64,
        cache_path: Path | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.dim = dim
        self.batch_size = batch_size
        self.device = device
        self._model = None
        self.cache = _EmbeddingCache(cache_path, dim) if cache_path else None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def encode(self, texts: list[str], use_cache: bool = True) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        keys = [_hash_text(t) for t in texts]

        # cache lookup
        missing_idx: list[int] = []
        if use_cache and self.cache is not None:
            cached = self.cache.get_many(list(set(keys)))
            for i, k in enumerate(keys):
                if k in cached:
                    out[i] = cached[k]
                else:
                    missing_idx.append(i)
        else:
            missing_idx = list(range(len(texts)))

        # encode misses
        if missing_idx:
            model = self._load_model()
            to_encode = [texts[i] for i in missing_idx]
            encoded = model.encode(
                to_encode,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype(np.float32)
            for j, i in enumerate(missing_idx):
                out[i] = encoded[j]

            if use_cache and self.cache is not None:
                self.cache.put_many(
                    [(keys[i], out[i]) for i in missing_idx]
                )

        return out

    def encode_one(self, text: str, use_cache: bool = True) -> np.ndarray:
        return self.encode([text], use_cache=use_cache)[0]
