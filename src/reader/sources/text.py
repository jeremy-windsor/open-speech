"""Plain text and Markdown sources."""

from __future__ import annotations

import re

from src.live_tts.segmenter import clean_markdown
from src.reader.document import HEADING, Block, Chapter, Document

CODE_BLOCK_CUE = "Code block skipped."

# An opening code fence; a backtick fence's info string has no backticks ("```a```" is inline code).
_FENCE = re.compile(r"^\s*(`{3,}(?=[^`]*$)|~{3,})")
# An ATX heading's opening "#" run; _atx_heading takes its text apart with string operations, as
# a regex that also matched the text backtracked quadratically on a line ending in many spaces.
_ATX_OPENING = re.compile(r"\s{0,3}(#{1,6})(?=\s)")
_SETEXT_UNDERLINE = re.compile(r"^\s*(=+|-+)\s*$")
_QUOTE_MARKER = re.compile(r"^(?:\s*>)+\s?")
_RULE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")
_LIST_ITEM = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+")
# CommonMark 5.3: only a bullet, or an ordered item numbered 1, with text may interrupt a paragraph.
_INTERRUPTING_ITEM = re.compile(r"^\s*(?:[-+*]|1[.)])\s+\S")
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


def _atx_heading(line: str) -> tuple[int, str] | None:
    """(level, text) of an ATX heading line, or None.

    A closing "#" run must follow a space: "# Learning C#" keeps its "#".
    """
    opening = _ATX_OPENING.match(line)
    if not opening or len(line) - opening.end() < 2:
        return None   # the space after the "#" run needs a character after it
    text = line[opening.end() :].strip()
    unclosed = text.rstrip("#")
    if unclosed != text and unclosed[-1:].isspace():
        text = unclosed.rstrip()
    return len(opening.group(1)), text


def _starts_list(line: str) -> bool:
    """A list item that can interrupt a paragraph ("- - -" is a rule, not an item)."""
    return bool(_INTERRUPTING_ITEM.match(line)) and not _RULE.match(line)


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
    fence = ""   # the opening fence while inside fenced code

    def flush() -> None:
        if current:
            paragraphs.append(current.copy())
            current.clear()

    for line in lines:
        if fence:
            # CommonMark 4.5: only the same character, at least as many times, closes the fence.
            closing = line.strip()
            if len(closing) >= len(fence) and closing == fence[0] * len(closing):
                fence = ""
            continue
        opening = _FENCE.match(line) if markdown else None
        if opening:
            flush()
            paragraphs.append([CODE_BLOCK_CUE])
            fence = opening.group(1)
            continue
        if not line.strip():
            flush()
        elif markdown and _atx_heading(line):
            flush()
            paragraphs.append([line])
        elif markdown and (_RULE.match(line) or (current and _SETEXT_UNDERLINE.match(line))):
            # A rule ends the paragraph or list above it (CommonMark 4.1), and an underline the heading
            # it makes (4.3). An underline stays as its paragraph's last line: parse_text tells which it is.
            if not (current and _SETEXT_UNDERLINE.match(line)):
                flush()
            current.append(line)
            flush()
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
            heading = _atx_heading(lines[0]) if len(lines) == 1 else None
            if heading:
                level, heading_text = heading[0], clean_markdown(heading[1])
                if level <= 2:
                    start_chapter(heading_text)
                else:
                    chapters[-1].blocks.append(Block(heading_text, HEADING))
                if not title and level == 1:
                    title = heading_text
                continue
            # Only paragraph text can be underlined: a list item, quote or rule above "---" is no heading.
            # The underlined text may span several lines (CommonMark 4.3).
            setext = (
                _SETEXT_UNDERLINE.match(lines[-1])
                if len(lines) >= 2
                and not _LIST_ITEM.match(lines[0])
                and not _RULE.match(lines[0])
                and not any(_QUOTE_MARKER.match(line) or _starts_list(line) for line in lines[:-1])
                else None
            )
            if setext:
                heading_text = clean_markdown(" ".join(lines[:-1]))
                start_chapter(heading_text)
                if setext.group(1)[0] == "=":
                    title = title or heading_text
                continue
            if _RULE.match(lines[-1]):
                # A rule is not read; nor is "---" under a list item or quote, which it cannot underline.
                lines = lines[:-1]
                if not lines:
                    continue
            # Every line of a block quote carries its own ">" (or none, when it is a lazy continuation).
            lines = [_QUOTE_MARKER.sub("", line) for line in lines]
            # A list may start on the line right after a paragraph's text.
            start = 0 if _LIST_ITEM.match(lines[0]) else next(
                (index for index, line in enumerate(lines) if index and _starts_list(line)), len(lines)
            )
            if start:
                chapters[-1].blocks.append(Block(clean_markdown(" ".join(lines[:start]))))
            if start < len(lines):
                # A line without a marker continues the item above it (joined once: adding each
                # line to the item's text would copy it again for every line).
                items: list[list[str]] = []
                for line in lines[start:]:
                    if _LIST_ITEM.match(line):
                        items.append([line])
                    else:
                        items[-1].append(line.strip())
                chapters[-1].blocks.extend(Block(clean_markdown(" ".join(item))) for item in items)
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
