"""Plain text and Markdown sources."""

from __future__ import annotations

import re

from src.live_tts.segmenter import clean_markdown
from src.reader.document import HEADING, Block, Chapter, Document

CODE_BLOCK_CUE = "Code block skipped."

_FENCE = re.compile(r"^\s*(```|~~~)")
_ATX_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_SETEXT_H1 = re.compile(r"^\s*=+\s*$")
_RULE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+")
_PLAIN_CHAPTER = re.compile(
    r"^\s*(?:chapter|book|part|prologue|epilogue|preface|introduction)\b(.{0,60})$",
    re.IGNORECASE,
)
_CHAPTER_NUMBER = re.compile(
    r"\s*(?:\d+|(?-i:[IVXLCDM]+)|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"the\s+(?:first|second|third|fourth|fifth|last))\b",
    re.IGNORECASE,
)
_GUTENBERG_START = re.compile(r"^\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*$", re.IGNORECASE | re.MULTILINE)
_GUTENBERG_END = re.compile(r"^\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*$", re.IGNORECASE | re.MULTILINE)
_GUTENBERG_FIELD = re.compile(r"^(Title|Author):\s*(.+)$", re.IGNORECASE | re.MULTILINE)


def _is_chapter_line(line: str) -> bool:
    """"CHAPTER IV", "Part One" or "Prologue", but not "Part of me wanted to leave."."""
    match = _PLAIN_CHAPTER.match(line)
    if not match:
        return False
    rest = match.group(1).strip()
    return not rest or bool(_CHAPTER_NUMBER.match(rest)) or not rest.endswith((".", "!", "?", ",", ";", ":"))


def _strip_gutenberg(text: str) -> tuple[str, str | None, str | None]:
    start = _GUTENBERG_START.search(text)
    end = _GUTENBERG_END.search(text)
    if not start or not end or end.start() <= start.end():
        return text, None, None
    fields = {key.lower(): value.strip() for key, value in _GUTENBERG_FIELD.findall(text[: start.start()])}
    return text[start.end() : end.start()], fields.get("title"), fields.get("author")


def _paragraphs(lines: list[str], markdown: bool) -> list[list[str]]:
    """Group lines into paragraphs; fenced code becomes a single cue paragraph."""
    paragraphs: list[list[str]] = []
    current: list[str] = []
    in_fence = False

    def flush() -> None:
        if current:
            paragraphs.append(current.copy())
            current.clear()

    for line in lines:
        if markdown and _FENCE.match(line):
            if not in_fence:
                flush()
                paragraphs.append([CODE_BLOCK_CUE])
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not line.strip():
            flush()
        elif markdown and _ATX_HEADING.match(line):
            flush()
            paragraphs.append([line])
        else:
            current.append(line)
    flush()
    return paragraphs


def parse_text(text: str, *, markdown: bool) -> Document:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text, title, author = _strip_gutenberg(text)
    chapters = [Chapter(title="")]

    def start_chapter(heading: str) -> None:
        if chapters[-1].blocks:
            chapters.append(Chapter(title=heading))
        else:
            chapters[-1].title = heading
        chapters[-1].blocks.append(Block(heading, HEADING))

    for lines in _paragraphs(text.split("\n"), markdown):
        if lines == [CODE_BLOCK_CUE]:
            chapters[-1].blocks.append(Block(CODE_BLOCK_CUE))
            continue
        if markdown:
            heading = _ATX_HEADING.match(lines[0]) if len(lines) == 1 else None
            if heading:
                level, heading_text = len(heading.group(1)), clean_markdown(heading.group(2))
                if level <= 2:
                    start_chapter(heading_text)
                else:
                    chapters[-1].blocks.append(Block(heading_text, HEADING))
                if not title and level == 1:
                    title = heading_text
                continue
            if len(lines) == 2 and _SETEXT_H1.match(lines[1]):
                heading_text = clean_markdown(lines[0])
                start_chapter(heading_text)
                title = title or heading_text
                continue
            if len(lines) == 1 and _RULE.match(lines[0]):
                continue
            if all(_LIST_ITEM.match(line) for line in lines):
                chapters[-1].blocks.extend(Block(clean_markdown(line)) for line in lines)
                continue
            chapters[-1].blocks.append(Block(clean_markdown(" ".join(lines))))
            continue
        joined = " ".join(line.strip() for line in lines)
        if len(lines) == 1 and _is_chapter_line(joined):
            start_chapter(joined)
        else:
            chapters[-1].blocks.append(Block(joined))

    return Document(
        title=title or "",
        author=author,
        source_type="markdown" if markdown else "text",
        chapters=chapters,
    )
