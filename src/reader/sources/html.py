"""HTML and XHTML sources (also used for each epub chapter)."""

from __future__ import annotations

from html.parser import HTMLParser

from src.reader.document import HEADING, PARAGRAPH, Block, Chapter, Document, normalize_space
from src.reader.sources.text import CODE_BLOCK_CUE

_BLOCK_TAGS = frozenset({
    "address", "article", "aside", "blockquote", "caption", "dd", "div", "dt", "figcaption",
    "footer", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "ol", "p",
    "pre", "section", "table", "tr", "ul",
})
_SKIP_TAGS = frozenset({"head", "math", "nav", "noscript", "script", "style", "sup", "svg", "template"})
_VOID_TAGS = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"})


class _BlockParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[Block, int]] = []
        self.title = ""
        self._buffer: list[str] = []
        self._buffer_has_text = False
        self._cell_separator = False
        self._heading_level = 0
        self._skip_depth = 0
        self._in_title = False
        self._pre_depth = 0
        self._pre_has_code = False

    def _flush(self) -> None:
        text = normalize_space("".join(self._buffer))
        self._buffer.clear()
        self._buffer_has_text = False
        self._cell_separator = False
        if text:
            kind = HEADING if self._heading_level else PARAGRAPH
            self.blocks.append((Block(text, kind), self._heading_level))
        self._heading_level = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "title":
            self._in_title = True
            return
        if tag in _SKIP_TAGS:
            if tag not in _VOID_TAGS:
                self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "br":
            self._buffer.append(" ")
        elif tag in ("td", "th"):
            # Table cells are read as a list; the comma waits for the next non-empty cell.
            self._cell_separator = self._buffer_has_text
        elif tag == "code" and self._pre_depth:
            self._pre_has_code = True
        if tag in _BLOCK_TAGS:
            self._flush()
            if len(tag) == 2 and tag[0] == "h" and tag[1].isdigit():
                self._heading_level = int(tag[1])
            if tag == "pre":
                self._pre_depth += 1
                self._pre_has_code = False

    def handle_startendtag(self, tag: str, attrs) -> None:
        # A self-closing <svg/> or <script/> has no end tag to leave the skip again.
        if tag in _SKIP_TAGS:
            return
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
            return
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if tag == "pre" and self._pre_depth:
            self._pre_depth -= 1
            if self._pre_has_code:
                self._buffer[:] = [CODE_BLOCK_CUE]
            self._pre_has_code = False
        if tag in _BLOCK_TAGS:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skip_depth:
            if self._cell_separator and not data.isspace():
                while self._buffer and self._buffer[-1].isspace():
                    self._buffer.pop()
                self._buffer.append(", ")
                self._cell_separator = False
            self._buffer.append(data)
            self._buffer_has_text = self._buffer_has_text or not data.isspace()

    def close(self) -> None:
        super().close()
        self._flush()


def html_to_blocks(markup: str) -> tuple[str, list[tuple[Block, int]]]:
    """Return the page title and its text blocks with heading levels (0 for paragraphs)."""
    parser = _BlockParser()
    parser.feed(markup)
    parser.close()
    return normalize_space(parser.title), parser.blocks


def parse_html(markup: str) -> Document:
    title, blocks = html_to_blocks(markup)
    chapters = [Chapter(title="")]
    for block, level in blocks:
        if block.kind == HEADING and level <= 2:
            if chapters[-1].blocks:
                chapters.append(Chapter(title=block.text))
            else:
                chapters[-1].title = block.text
            title = title or block.text
        chapters[-1].blocks.append(block)
    return Document(title=title, source_type="html", chapters=chapters)
