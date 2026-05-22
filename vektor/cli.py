from __future__ import annotations

import click

from vektor import __version__


@click.group()
@click.version_option(__version__)
def main() -> None:
    """VEKTOR command-line interface."""


@main.command("serve")
@click.option("--host", default=None)
@click.option("--port", default=None, type=int)
@click.option("--reload", is_flag=True, default=False)
def serve(host: str | None, port: int | None, reload: bool) -> None:
    """Run the API server."""
    import uvicorn

    from vektor.config import settings

    uvicorn.run(
        "vektor.api.main:app",
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=reload,
    )


@main.command("ingest")
@click.argument("source", type=click.Path(exists=True))
@click.option("--backend", default="hnsw", type=click.Choice(["flat", "hnsw"]))
def ingest_cmd(source: str, backend: str) -> None:
    """Ingest documents from a JSONL file into the index. (Stub — wire up to scripts/build_index.py)"""
    click.echo(f"ingesting {source} into {backend} index — use scripts/build_index.py for now")


if __name__ == "__main__":
    main()
