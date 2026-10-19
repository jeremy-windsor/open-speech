"""In-memory shape of an imported document before it is stored."""

from __future__ import annotations

import math
import re
import unicodedata
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


def spoken_opening(text: str) -> str:
    """Give a paragraph's capitalised first words normal case for speech.

    Books often set a chapter's first word or two in capitals ("MR. BENNET was among").
    Kokoro reads short common words like that fine but spells out longer ones it doesn't
    know, such as names. Only an opening run followed by lowercase text changes, and
    words of one or two letters or without vowels (US, BBC) are left as acronyms.
    """
    body = text.lstrip()
    words = body.split(" ")
    run = 0
    while run < len(words) and words[run].isupper():
        run += 1
    if not run or run == len(words) or not any(c.islower() for c in words[run]):
        return text
    return text[:len(text) - len(body)] + " ".join([_title_word(word) for word in words[:run]] + words[run:])


def _title_word(word: str) -> str:
    letters = [c for c in word if c.isalpha()]
    if len(letters) < 3 or not any(_is_vowel(c) for c in letters):
        return word
    first = next(i for i, c in enumerate(word) if c.isalpha())
    return word[:first + 1] + word[first + 1:].lower()


def _is_vowel(letter: str) -> bool:
    return unicodedata.normalize("NFD", letter)[0].lower() in "aeiouy"


def _split_long(text: str) -> list[str]:
    if len(text) <= MAX_BLOCK_CHARS:
        return [text]
    target = len(text) / math.ceil(len(text) / MAX_BLOCK_CHARS)
    pieces: list[str] = []
    start = previous_end = 0
    for _sentence_start, end in split_sentences(text, MAX_BLOCK_CHARS):
        # Cut before a sentence that would overflow, otherwise once a piece reaches its even share.
        if end - start > MAX_BLOCK_CHARS and previous_end > start:
            pieces.append(text[start:previous_end].strip())
            start = previous_end
        if end - start >= target:
            pieces.append(text[start:end].strip())
            start = end
        previous_end = end
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
