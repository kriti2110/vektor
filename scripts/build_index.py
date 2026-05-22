"""Build a dense (+ optional sparse) index from a JSONL document source.

Example:
    python scripts/build_index.py \\
        --source data/wikipedia.jsonl \\
        --backend hnsw \\
        --out index_store/wiki \\
        --max-docs 10000

Outputs:
    index_store/wiki.hnsw  (or .flat)
    index_store/wiki.bm25  (when --build-sparse)
    index_store/wiki.docs.jsonl  (chunk_id → text, for rerank lookup)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import click
from tqdm import tqdm

from vektor.config import settings
from vektor.index.bm25 import BM25Index
from vektor.index.flat import FlatIndex
from vektor.index.hnsw import HNSWIndex
from vektor.ingestion.chunker import SemanticChunker
from vektor.ingestion.embedder import Embedder
from vektor.ingestion.loaders.jsonl import load_jsonl


def build_index(backend: str, dim: int):
    if backend == "flat":
        return FlatIndex(dim=dim)
    if backend == "hnsw":
        return HNSWIndex(
            dim=dim,
            M=settings.hnsw_m,
            ef_construction=settings.hnsw_ef_construction,
            ef_search=settings.hnsw_ef_search,
        )
    raise ValueError(f"unknown backend: {backend}")


@click.command()
@click.option("--source", required=True, type=click.Path(exists=True), help="JSONL file.")
@click.option("--backend", default="hnsw", type=click.Choice(["flat", "hnsw"]))
@click.option("--out", default="index_store/index", type=click.Path())
@click.option("--max-docs", default=None, type=int, help="Cap docs for dev runs.")
@click.option("--build-sparse/--no-build-sparse", default=True)
@click.option("--id-field", default="id")
@click.option("--text-field", default="text")
def main(
    source: str,
    backend: str,
    out: str,
    max_docs: int | None,
    build_sparse: bool,
    id_field: str,
    text_field: str,
) -> None:
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    embedder = Embedder(
        model_name=settings.embed_model,
        dim=settings.embed_dim,
        batch_size=settings.embed_batch_size,
        cache_path=settings.embed_cache_path,
    )
    chunker = SemanticChunker(
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    dense = build_index(backend, settings.embed_dim)
    sparse = BM25Index() if build_sparse else None

    docs_out_path = Path(str(out_path) + ".docs.jsonl")
    docs_out = docs_out_path.open("w", encoding="utf-8")

    t0 = time.time()
    docs_processed = 0
    chunks_buffer_texts: list[str] = []
    chunks_buffer_ids: list[str] = []
    BUFFER = 256  # batched embed across documents to amortize model overhead

    def flush_buffer():
        if not chunks_buffer_texts:
            return
        vectors = embedder.encode(chunks_buffer_texts)
        dense.add_batch(vectors, chunks_buffer_ids)
        if sparse is not None:
            sparse.add_batch(chunks_buffer_texts, chunks_buffer_ids)
        chunks_buffer_texts.clear()
        chunks_buffer_ids.clear()

    docs = load_jsonl(source, id_field=id_field, text_field=text_field)
    pbar = tqdm(docs, desc="ingesting", unit="doc")

    for i, (doc_id, text, _meta) in enumerate(pbar):
        if max_docs is not None and i >= max_docs:
            break
        chunks = chunker.chunk(doc_id, text)
        for c in chunks:
            chunks_buffer_texts.append(c.text)
            chunks_buffer_ids.append(c.chunk_id)
            docs_out.write(
                json.dumps({"chunk_id": c.chunk_id, "doc_id": doc_id, "text": c.text}) + "\n"
            )
            if len(chunks_buffer_texts) >= BUFFER:
                flush_buffer()
        docs_processed += 1

    flush_buffer()
    docs_out.close()

    dt = time.time() - t0
    click.echo(
        f"\nindexed {dense.size} chunks from {docs_processed} docs in {dt:.1f}s "
        f"({dense.size / max(dt, 0.001):.1f} chunk/s)"
    )

    dense_out = Path(str(out_path) + f".{backend}")
    dense.save(dense_out)
    click.echo(f"saved dense → {dense_out}")
    if sparse is not None:
        sparse_out = Path(str(out_path) + ".bm25")
        sparse.save(sparse_out)
        click.echo(f"saved sparse → {sparse_out}")
    click.echo(f"saved docs → {docs_out_path}")


if __name__ == "__main__":
    main()
