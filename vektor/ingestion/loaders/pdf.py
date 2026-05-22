from __future__ import annotations

from pathlib import Path


def load_pdf(path: Path | str) -> tuple[str, str]:
    """Return (doc_id, text) extracted from a PDF file."""
    from pypdf import PdfReader

    path = Path(path)
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return path.stem, "\n\n".join(pages)
