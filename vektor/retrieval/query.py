"""Query understanding: normalization, intent classification, expansion."""

from __future__ import annotations

import re
import unicodedata
from enum import Enum


class QueryIntent(str, Enum):
    NAVIGATIONAL = "navigational"  # user wants a specific page ("github wiki page")
    INFORMATIONAL = "informational"  # user wants to learn ("what is hnsw")
    TRANSACTIONAL = "transactional"  # user wants to do ("download wikipedia dump")


_WS = re.compile(r"\s+")


def normalize_query(q: str) -> str:
    """Unicode NFC + collapse whitespace + casefold."""
    q = unicodedata.normalize("NFC", q).strip()
    q = _WS.sub(" ", q)
    return q.casefold()


_NAVIGATIONAL_KEYWORDS = {"site:", "page", "homepage", "official", "url"}
_TRANSACTIONAL_KEYWORDS = {"download", "buy", "install", "get", "purchase", "subscribe"}
_INFORMATIONAL_PREFIXES = (
    "what",
    "why",
    "how",
    "when",
    "who",
    "where",
    "explain",
    "define",
    "compare",
)


def classify_intent(q: str) -> QueryIntent:
    """Heuristic intent classifier. Good enough for routing; not a research-grade model.

    TODO(later): replace with a small fine-tuned classifier if intent routing
    proves valuable in A/B tests.
    """
    qn = normalize_query(q)
    if any(k in qn for k in _NAVIGATIONAL_KEYWORDS):
        return QueryIntent.NAVIGATIONAL
    if any(qn.startswith(p) for p in _INFORMATIONAL_PREFIXES):
        return QueryIntent.INFORMATIONAL
    if any(k in qn.split() for k in _TRANSACTIONAL_KEYWORDS):
        return QueryIntent.TRANSACTIONAL
    return QueryIntent.INFORMATIONAL  # default — most search is informational


def expand_query(q: str, n_variants: int = 3) -> list[str]:
    """Generate paraphrase variants of a query.

    Placeholder: returns [q] only. A real implementation would use a small
    generative model (e.g., t5-small fine-tuned on query reformulations) or
    a synonym dictionary. Kept as a no-op stub so the pipeline runs end-to-end.
    """
    # TODO(later): integrate query expansion. Not on critical path for v0.1.
    return [q]
