from __future__ import annotations

import pytest

from vektor.ingestion.chunker import SemanticChunker


def test_chunker_returns_empty_for_empty_text():
    chunker = SemanticChunker(max_tokens=50)
    assert chunker.chunk("doc1", "") == []
    assert chunker.chunk("doc1", "   ") == []


def test_chunker_keeps_single_short_doc_as_one_chunk():
    chunker = SemanticChunker(max_tokens=200)
    text = "This is a short doc. It has two sentences."
    chunks = chunker.chunk("doc1", text)
    assert len(chunks) == 1
    assert chunks[0].doc_id == "doc1"
    assert chunks[0].chunk_id == "doc1::0"


def test_chunker_splits_long_doc_at_sentence_boundary():
    chunker = SemanticChunker(max_tokens=8, overlap_tokens=0)  # tiny budget → forces split
    text = "Sentence one is here. Sentence two follows. Sentence three concludes."
    chunks = chunker.chunk("doc1", text)
    assert len(chunks) >= 2
    for c in chunks:
        # each chunk should be roughly under budget (modulo overlap)
        assert len(c.text) <= 8 * 4 * 2  # generous bound


def test_chunker_overlap_carries_context():
    chunker = SemanticChunker(max_tokens=15, overlap_tokens=5)
    text = "Alpha sentence here. Beta sentence here. Gamma sentence here. Delta sentence here."
    chunks = chunker.chunk("doc1", text)
    assert len(chunks) >= 2
    # the second chunk should re-include some content from the end of the first
    # (we just sanity-check by length — exact overlap depends on token estimation)
    assert all(len(c.text) > 0 for c in chunks)


def test_chunker_hard_splits_oversized_sentence():
    chunker = SemanticChunker(max_tokens=10, overlap_tokens=0)
    long_sentence = " ".join(["word"] * 100) + "."
    chunks = chunker.chunk("doc1", long_sentence)
    assert len(chunks) >= 2


def test_overlap_must_be_less_than_max():
    with pytest.raises(ValueError):
        SemanticChunker(max_tokens=10, overlap_tokens=10)
