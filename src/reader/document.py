"""In-memory shape of an imported document before it is stored."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from src.live_tts.segmenter import split_sentences

HEADING = "heading"
PARAGRAPH = "paragraph"

# Longer paragraphs are split at sentences so one block never becomes a huge first part.
MAX_BLOCK_CHARS = 1000

_SPEAKABLE = re.compile(r"\w", re.UNICODE)


@dataclass(slots=True)
class Block:
    text: str
    kind: str = PARAGRAPH


@dataclass(slots=True)
class Chapter:
    title: str
    blocks: list[Block] = field(default_factory=list)


@dataclass(slots=True)
class Document:
    title: str
    source_type: str
    chapters: list[Chapter]
    author: str | None = None

    @property
    def char_count(self) -> int:
        return sum(len(block.text) for chapter in self.chapters for block in chapter.blocks)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_speakable(text: str) -> bool:
    """True when the text has at least one letter or digit to speak."""
    return bool(_SPEAKABLE.search(text))


def _split_long(text: str) -> list[str]:
    if len(text) <= MAX_BLOCK_CHARS:
        return [text]
    target = len(text) / math.ceil(len(text) / MAX_BLOCK_CHARS)
    pieces: list[str] = []
    start = 0
    for _sentence_start, end in split_sentences(text, MAX_BLOCK_CHARS):
        if end - start >= target:
            pieces.append(text[start:end].strip())
            start = end
    if text[start:].strip():
        pieces.append(text[start:].strip())
    return pieces


def finalize(document: Document, fallback_title: str) -> Document:
    """Drop empty blocks and chapters, split very long paragraphs and fill in a title."""
    chapters: list[Chapter] = []
    for chapter in document.chapters:
        blocks = [
            Block(piece, block.kind)
            for block in chapter.blocks
            if is_speakable(block.text)
            for piece in _split_long(normalize_space(block.text))
        ]
        if blocks:
            chapters.append(Chapter(normalize_space(chapter.title) or blocks[0].text[:80], blocks))
    if not chapters:
        raise ValueError("No readable text found")
    title = normalize_space(document.title or "")
    if not title:
        first = chapters[0].blocks[0]
        title = first.text[:80] if first.kind == HEADING else fallback_title
    return Document(
        title=title[:200] or "Untitled",
        source_type=document.source_type,
        chapters=chapters,
        author=normalize_space(document.author or "") or None,
    )
