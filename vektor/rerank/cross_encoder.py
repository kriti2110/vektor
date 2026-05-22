from __future__ import annotations

from vektor.index.base import SearchResult


class CrossEncoderReranker:
    """Wraps a HuggingFace cross-encoder for query-document relevance scoring.

    Inference path is built and tested. The fine-tuning loop lives in
    `vektor/rerank/train.py` — that part is Kriti's to write.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, max_length=self.max_length, device=self.device)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        doc_text_lookup,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Score and re-sort candidates by (query, doc_text) relevance.

        Parameters
        ----------
        query : str
        candidates : list[SearchResult]
            From the retrieval stage. Order is overwritten.
        doc_text_lookup : Callable[[str], str]
            Function to fetch full text for a doc_id (DB, in-memory dict, etc.)
        top_k : int | None
            Truncate output. None returns all rescored.
        """
        if not candidates:
            return []

        model = self._load()
        pairs = [(query, doc_text_lookup(c.doc_id)) for c in candidates]
        scores = model.predict(pairs, show_progress_bar=False)

        rescored = [
            SearchResult(doc_id=c.doc_id, score=float(s))
            for c, s in zip(candidates, scores, strict=True)
        ]
        rescored.sort(key=lambda r: r.score, reverse=True)

        if top_k is not None:
            rescored = rescored[:top_k]
        return rescored
