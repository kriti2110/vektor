"""Reciprocal Rank Fusion — merge ranked result lists from multiple retrievers.

Cormack et al. 2009: for each doc, score = Σ 1 / (k + rank_in_retriever_i).
k=60 is the paper's recommendation; it smooths the contribution of low-ranked
docs and prevents any single retriever from dominating.
"""

from __future__ import annotations

from vektor.index.base import SearchResult


def rrf_fuse(
    result_lists: list[list[SearchResult]],
    k: int = 60,
    top_k: int | None = None,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion.

    Parameters
    ----------
    result_lists : list[list[SearchResult]]
        One list per retriever, each ordered best-first.
    k : int
        Smoothing constant. 60 is standard.
    top_k : int | None
        Truncate fused output. None returns all unique docs.

    Returns
    -------
    list[SearchResult]
        Fused ranking, best-first. score is the RRF score (unnormalized).
    """
    if not result_lists:
        return []

    scores: dict[str, float] = {}
    for results in result_lists:
        for rank, hit in enumerate(results):
            scores[hit.doc_id] = scores.get(hit.doc_id, 0.0) + 1.0 / (k + rank + 1)

    fused = [SearchResult(doc_id=did, score=score) for did, score in scores.items()]
    fused.sort(key=lambda r: r.score, reverse=True)

    if top_k is not None:
        fused = fused[:top_k]
    return fused
