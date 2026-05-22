from __future__ import annotations

from pathlib import Path


_BOILERPLATE_TAGS = {"script", "style", "nav", "footer", "header", "aside", "noscript"}


def load_html(path_or_html: Path | str, *, is_html_string: bool = False) -> tuple[str, str]:
    """Return (doc_id, text) extracted from an HTML file or raw HTML string.

    Strips script/style/nav and similar boilerplate. Doc_id is the file stem when
    loading from disk, or a sha1 prefix when given a raw string.
    """
    from bs4 import BeautifulSoup

    if is_html_string:
        import hashlib

        html = str(path_or_html)
        doc_id = "html-" + hashlib.sha1(html.encode("utf-8")).hexdigest()[:12]
    else:
        path = Path(path_or_html)
        html = path.read_text(encoding="utf-8", errors="ignore")
        doc_id = path.stem

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(list(_BOILERPLATE_TAGS)):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    return doc_id, text
