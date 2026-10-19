"""Turn uploaded files and pasted text into Reader documents.

Each source returns a :class:`src.reader.document.Document`. URL, GitHub and API
sources arrive in later phases as more modules here.
"""

from __future__ import annotations

import codecs
from pathlib import PurePosixPath

from src.reader.document import Document, finalize
from src.reader.sources.epub import parse_epub
from src.reader.sources.html import parse_html
from src.reader.sources.text import parse_text

SUPPORTED_EXTENSIONS = frozenset({".epub", ".txt", ".md", ".markdown", ".html", ".htm", ".xhtml"})


def _stem(filename: str) -> str:
    return PurePosixPath(filename.replace("\\", "/")).stem.replace("_", " ").strip() or "Untitled"


def _decode(data: bytes) -> str:
    # UTF-16 only with a byte-order mark: without one, almost any even-length
    # Windows-1252 text "decodes" as UTF-16 into CJK characters.
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16")
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def parse_upload(filename: str, data: bytes, *, title: str | None = None, max_uncompressed: int) -> Document:
    """Parse an uploaded file by its extension."""
    suffix = PurePosixPath(filename.lower().replace("\\", "/")).suffix
    if suffix not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type {suffix or '(none)'}; use one of {allowed}")
    if suffix == ".epub":
        document = parse_epub(data, max_uncompressed=max_uncompressed)
    elif suffix in {".html", ".htm", ".xhtml"}:
        document = parse_html(_decode(data))
    else:
        document = parse_text(_decode(data), markdown=suffix in {".md", ".markdown"})
    if title:
        document.title = title
    return finalize(document, _stem(filename))


def parse_pasted(text: str, *, title: str | None = None) -> Document:
    """Parse pasted text; Markdown headings and lists are understood."""
    document = parse_text(text, markdown=True)
    if title:
        document.title = title
    return finalize(document, "Pasted text")
