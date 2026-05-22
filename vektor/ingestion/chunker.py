from __future__ import annotations

import re
from dataclasses import dataclass, field


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_WHITESPACE = re.compile(r"\s+")


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def _approx_token_count(text: str) -> int:
    # rough heuristic: ~1 token per 4 chars for english. cheap enough to call in a loop;
    # swap for a real tokenizer (tiktoken / model tokenizer) when accuracy matters.
    return max(1, len(text) // 4)


def _split_sentences(text: str) -> list[str]:
    text = _WHITESPACE.sub(" ", text).strip()
    if not text:
        return []
    parts = _SENTENCE_END.split(text)
    return [p.strip() for p in parts if p.strip()]


class SemanticChunker:
    """Greedy sentence-merging chunker.

    Splits text into sentences, then merges sentences left-to-right until
    adding the next would exceed max_tokens. Maintains an `overlap_tokens`
    suffix from the previous chunk for context continuity.
    """

    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 32) -> None:
        if overlap_tokens >= max_tokens:
            raise ValueError("overlap_tokens must be < max_tokens")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, doc_id: str, text: str, metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        sentences = _split_sentences(text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        current: list[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = _approx_token_count(sent)

            if sent_tokens > self.max_tokens:
                # sentence alone exceeds budget — flush current, hard-split sentence
                if current:
                    chunks.append(self._make_chunk(doc_id, len(chunks), " ".join(current), metadata))
                    current, current_tokens = self._overlap_carryover(current)
                for sub in self._hard_split(sent):
                    chunks.append(self._make_chunk(doc_id, len(chunks), sub, metadata))
                continue

            if current_tokens + sent_tokens > self.max_tokens:
                chunks.append(self._make_chunk(doc_id, len(chunks), " ".join(current), metadata))
                current, current_tokens = self._overlap_carryover(current)

            current.append(sent)
            current_tokens += sent_tokens

        if current:
            chunks.append(self._make_chunk(doc_id, len(chunks), " ".join(current), metadata))

        return chunks

    def _make_chunk(self, doc_id: str, idx: int, text: str, metadata: dict) -> Chunk:
        return Chunk(
            doc_id=doc_id,
            chunk_id=f"{doc_id}::{idx}",
            text=text,
            metadata=dict(metadata),
        )

    def _overlap_carryover(self, current: list[str]) -> tuple[list[str], int]:
        if self.overlap_tokens <= 0 or not current:
            return [], 0
        # walk from the end, accumulate sentences until we hit overlap_tokens
        carry: list[str] = []
        carry_tokens = 0
        for sent in reversed(current):
            t = _approx_token_count(sent)
            if carry_tokens + t > self.overlap_tokens and carry:
                break
            carry.insert(0, sent)
            carry_tokens += t
        return carry, carry_tokens

    def _hard_split(self, sentence: str) -> list[str]:
        # last resort: split on word boundaries at max_tokens worth of chars
        budget_chars = self.max_tokens * 4
        words = sentence.split()
        out: list[str] = []
        buf: list[str] = []
        buf_chars = 0
        for w in words:
            if buf_chars + len(w) + 1 > budget_chars and buf:
                out.append(" ".join(buf))
                buf, buf_chars = [], 0
            buf.append(w)
            buf_chars += len(w) + 1
        if buf:
            out.append(" ".join(buf))
        return out
