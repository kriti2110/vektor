from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator


def load_jsonl(
    path: Path | str,
    id_field: str = "id",
    text_field: str = "text",
) -> Iterator[tuple[str, str, dict]]:
    """Yield (doc_id, text, metadata) tuples from a JSONL file.

    Each line is expected to be a JSON object with at least `id_field` and
    `text_field`. Remaining fields are returned as metadata.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no} invalid JSON: {e}") from e
            if id_field not in obj or text_field not in obj:
                raise KeyError(
                    f"{path}:{line_no} missing required fields {id_field!r}/{text_field!r}"
                )
            doc_id = str(obj[id_field])
            text = str(obj[text_field])
            metadata = {k: v for k, v in obj.items() if k not in (id_field, text_field)}
            yield doc_id, text, metadata
