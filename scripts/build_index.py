"""Build a dense (+ optional sparse) index from a JSONL document source.

Example:
    python scripts/build_index.py \\
        --source data/wikipedia.jsonl \\
        --backend hnsw \\
        --out index_store/wiki_hnsw \\
        --max-docs 100000 \\
        --build-sparse
"""

from __future__ import annotations

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

    t0 = time.time()
    total_chunks = 0

    docs = load_jsonl(source, id_field=id_field, text_field=text_field)
    pbar = tqdm(docs, desc="ingesting", unit="doc")

    for i, (doc_id, text, _meta) in enumerate(pbar):
        if max_docs is not None and i >= max_docs:
            break
        chunks = chunker.chunk(doc_id, text)
        if not chunks:
            continue
        texts = [c.text for c in chunks]
        ids = [c.chunk_id for c in chunks]
        try:
            vectors = embedder.encode(texts)
            dense.add_batch(vectors, ids)
        except NotImplementedError:
            # HNSW.add not yet implemented — Kriti hasn't filled it in
            click.echo(
                "\nHNSW.add is a stub — implement vektor/index/hnsw.py "
                "or run with --backend flat to test the pipeline.",
                err=True,
            )
            raise SystemExit(1)
        if sparse is not None:
            sparse.add_batch(texts, ids)
        total_chunks += len(chunks)

    dt = time.time() - t0
    click.echo(f"\nindexed {dense.size} chunks in {dt:.1f}s ({dense.size / max(dt, 0.001):.1f} chunk/s)")

    dense.save(Path(str(out_path) + f".{backend}"))
    if sparse is not None:
        sparse.save(Path(str(out_path) + ".bm25"))
    click.echo(f"saved → {out_path}.{backend}" + (f" and {out_path}.bm25" if sparse else ""))


if __name__ == "__main__":
    main()
