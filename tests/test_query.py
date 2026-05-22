from __future__ import annotations

from vektor.retrieval.query import QueryIntent, classify_intent, normalize_query


def test_normalize_collapses_whitespace_and_casefolds():
    assert normalize_query("  Hello   World  ") == "hello world"


def test_normalize_unicode():
    # composed vs decomposed e-acute should normalize to the same form
    assert normalize_query("café") == normalize_query("café")


def test_classify_informational():
    assert classify_intent("what is hnsw") == QueryIntent.INFORMATIONAL
    assert classify_intent("how does bm25 work") == QueryIntent.INFORMATIONAL


def test_classify_navigational():
    assert classify_intent("github official page") == QueryIntent.NAVIGATIONAL


def test_classify_transactional():
    assert classify_intent("download wikipedia dump") == QueryIntent.TRANSACTIONAL


def test_default_is_informational():
    assert classify_intent("transformers attention mechanism") == QueryIntent.INFORMATIONAL
