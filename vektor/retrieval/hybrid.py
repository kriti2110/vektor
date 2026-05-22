"""Hybrid retrieval fusion.

╔══════════════════════════════════════════════════════════════════════════╗
║   ⚠️  STUB. Kriti — implement Reciprocal Rank Fusion (RRF) here.         ║
║                                                                          ║
║   Spec: TODO_YOU_BUILD.md §2                                             ║
║   Paper: Cormack et al. 2009                                             ║
║   Tests: tests/test_hybrid.py                                            ║
║                                                                          ║
║   Do NOT use weighted score sum. RRF is the right tool for this job;     ║
║   the literature is clear and interviewers know it.                      ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

from vektor.index.base import SearchResult


def rrf_fuse(
    result_lists: list[list[SearchResult]],
    k: int = 60,
    top_k: int | None = None,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion.

    For each doc, score = sum over retrievers of 1 / (k + rank_in_that_retriever).
    Docs absent from a retriever contribute 0 from that retriever.

    Parameters
    ----------
    result_lists : list[list[SearchResult]]
        One list per retriever, each ordered best-first.
    k : int
        Smoothing constant. Cormack et al. recommend 60.
    top_k : int | None
        If set, truncate the fused output. Default returns all unique docs.

    Returns
    -------
    list[SearchResult]
        Fused ranking, best-first. score is the RRF score (not normalized).

    TODO(Kriti) — implement. See TODO_YOU_BUILD.md §2.
    """
    raise NotImplementedError("Kriti — implement RRF. See TODO_YOU_BUILD.md §2")
