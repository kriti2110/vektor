"""Download a subset of Wikipedia for indexing.

Uses HuggingFace datasets (`wikipedia` config "20220301.en"). For small dev runs
pass --n small; for the 1M-doc benchmark, pass --n 1000000 (you'll need ~10GB
free and an hour or two on first run).
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from tqdm import tqdm


@click.command()
@click.option("--n", default=10000, type=int, help="Number of articles to fetch.")
@click.option(
    "--out",
    default="data/wikipedia.jsonl",
    type=click.Path(),
    help="Output JSONL path.",
)
@click.option(
    "--config",
    default="20220301.en",
    help="HuggingFace wikipedia config name.",
)
def main(n: int, out: str, config: str) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "datasets package not installed. Run: pip install 'datasets>=2.18' apache_beam mwparserfromhell"
        )

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    click.echo(f"streaming wikipedia.{config} → {out_path} ({n} articles)")
    ds = load_dataset("wikipedia", config, split="train", streaming=True)

    with out_path.open("w", encoding="utf-8") as f:
        for i, row in enumerate(tqdm(ds, total=n)):
            if i >= n:
                break
            f.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "title": row["title"],
                        "text": row["text"],
                        "url": row.get("url", ""),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    click.echo(f"done — wrote {n} articles to {out_path}")


if __name__ == "__main__":
    main()
