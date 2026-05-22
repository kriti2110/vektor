from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def random_vectors():
    """L2-normalized random vectors of shape (n, dim)."""

    def _make(n: int, dim: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n, dim)).astype(np.float32)
        x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x

    return _make
