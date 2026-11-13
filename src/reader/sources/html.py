"""HTML and XHTML sources (also used for each epub chapter)."""

from __future__ import annotations

import codecs
import contextlib
import re
from html.parser import HTMLParser
from typing import NamedTuple

from src.reader.document import HEADING, PARAGRAPH, Block, Chapter, Document, normalize_space
from src.reader.sources.text import CODE_BLOCK_CUE

_BLOCK_TAGS = frozenset({
    "address", "article", "aside", "blockquote", "caption", "dd", "div", "dt", "figcaption",
    "footer", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "ol", "p",
    "pre", "section", "table", "tr", "ul",
})
_SKIP_TAGS = frozenset({"math", "nav", "noscript", "script", "style", "sup", "svg", "template"})
_VOID_TAGS = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr"})
# What a <head> may hold. Its end tag is optional, so any other start tag ends the head,
# except inside a <noscript> there (tracking snippets put an <img> in one).
_HEAD_TAGS = frozenset({"base", "basefont", "bgsound", "link", "meta", "noframes", "noscript", "script", "style", "template", "title"})
_DECLARED_CHARSET = re.compile(
    rb"""<\?xml[^>]*?encoding\s*=\s*["']?([\w.:-]+)|<meta[^>]*?charset\s*=\s*["']?([\w.:-]+)""", re.IGNORECASE
)
_XML_DECLARATION = re.compile(rb"""\s*<\?xml[^>]*?encoding\s*=\s*["']?([\w.:-]+)""", re.IGNORECASE)
# The WHATWG Encoding Standard's labels, by the Python codec that decodes them as browsers do:
# gb2312 is GBK (read by the gb18030 decoder), Shift_JIS is Windows-31J, EUC-KR is windows-949,
# Latin-1 and ASCII are Windows-1252, and a UTF-16 label readable as ASCII means UTF-8. Browsers
# ignore any other label, so Python codecs such as utf-7, utf-32 or base64 are never used.
_ENCODING_LABELS = {
    label: codec
    for codec, labels in (
        ("utf-8", "unicode-1-1-utf-8 unicode11utf8 unicode20utf8 utf-8 utf8 x-unicode20utf8"
                  " csunicode iso-10646-ucs-2 ucs-2 unicode unicodefeff unicodefffe utf-16 utf-16be utf-16le"),
        ("cp866", "866 cp866 csibm866 ibm866"),
        ("iso8859-2", "csisolatin2 iso-8859-2 iso-ir-101 iso8859-2 iso88592 iso_8859-2 iso_8859-2:1987 l2 latin2"),
        ("iso8859-3", "csisolatin3 iso-8859-3 iso-ir-109 iso8859-3 iso88593 iso_8859-3 iso_8859-3:1988 l3 latin3"),
        ("iso8859-4", "csisolatin4 iso-8859-4 iso-ir-110 iso8859-4 iso88594 iso_8859-4 iso_8859-4:1988 l4 latin4"),
        ("iso8859-5", "csisolatincyrillic cyrillic iso-8859-5 iso-ir-144 iso8859-5 iso88595 iso_8859-5 iso_8859-5:1988"),
        ("iso8859-6", "arabic asmo-708 csiso88596e csiso88596i csisolatinarabic ecma-114 iso-8859-6 iso-8859-6-e"
                      " iso-8859-6-i iso-ir-127 iso8859-6 iso88596 iso_8859-6 iso_8859-6:1987"),
        ("iso8859-7", "csisolatingreek ecma-118 elot_928 greek greek8 iso-8859-7 iso-ir-126 iso8859-7 iso88597"
                      " iso_8859-7 iso_8859-7:1987 sun_eu_greek"),
        ("iso8859-8", "csiso88598e csisolatinhebrew hebrew iso-8859-8 iso-8859-8-e iso-ir-138 iso8859-8 iso88598"
                      " iso_8859-8 iso_8859-8:1988 visual csiso88598i iso-8859-8-i logical"),
        ("iso8859-10", "csisolatin6 iso-8859-10 iso-ir-157 iso8859-10 iso885910 l6 latin6"),
        ("iso8859-13", "iso-8859-13 iso8859-13 iso885913"),
        ("iso8859-14", "iso-8859-14 iso8859-14 iso885914"),
        ("iso8859-15", "csisolatin9 iso-8859-15 iso8859-15 iso885915 iso_8859-15 l9"),
        ("iso8859-16", "iso-8859-16"),
        ("koi8-r", "cskoi8r koi koi8 koi8-r koi8_r"),
        ("koi8-u", "koi8-ru koi8-u"),
        ("mac-roman", "csmacintosh mac macintosh x-mac-roman"),
        ("cp874", "dos-874 iso-8859-11 iso8859-11 iso885911 tis-620 windows-874"),
        ("cp1250", "cp1250 windows-1250 x-cp1250"),
        ("cp1251", "cp1251 windows-1251 x-cp1251"),
        ("cp1252", "ansi_x3.4-1968 ascii cp1252 cp819 csisolatin1 ibm819 iso-8859-1 iso-ir-100 iso8859-1 iso88591"
                   " iso_8859-1 iso_8859-1:1987 l1 latin1 us-ascii windows-1252 x-cp1252"),
        ("cp1253", "cp1253 windows-1253 x-cp1253"),
        ("cp1254", "cp1254 csisolatin5 iso-8859-9 iso-ir-148 iso8859-9 iso88599 iso_8859-9 iso_8859-9:1989 l5 latin5"
                   " windows-1254 x-cp1254"),
        ("cp1255", "cp1255 windows-1255 x-cp1255"),
        ("cp1256", "cp1256 windows-1256 x-cp1256"),
        ("cp1257", "cp1257 windows-1257 x-cp1257"),
        ("cp1258", "cp1258 windows-1258 x-cp1258"),
        ("mac-cyrillic", "x-mac-cyrillic x-mac-ukrainian"),
        ("gb18030", "chinese csgb2312 csiso58gb231280 gb2312 gb_2312 gb_2312-80 gbk iso-ir-58 x-gbk gb18030"),
        ("big5hkscs", "big5 big5-hkscs cn-big5 csbig5 x-x-big5"),
        ("euc_jp", "cseucpkdfmtjapanese euc-jp x-euc-jp"),
        ("iso2022_jp", "csiso2022jp iso-2022-jp"),
        ("cp932", "csshiftjis ms932 ms_kanji shift-jis shift_jis sjis windows-31j x-sjis"),
        ("cp949", "cseuckr csksc56011987 euc-kr iso-ir-149 korean ks_c_5601-1987 ks_c_5601-1989 ksc5601 ksc_5601"
                  " windows-949"),
    )
    for label in labels.split()
}
# The error handler decode_declared uses, and the codecs above that read a character from a pair of bytes.
_WEB_DECODE_ERRORS = "reader-web-decode"
_DOUBLE_BYTE_CODECS = frozenset({"big5hkscs", "cp932", "cp949", "euc_jp", "gb18030"})


class _BlockParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[Block, int]] = []
        self.anchors: dict[str, int] = {}   # element id (or <a name>) -> index of the block it starts in
        self.heading_captions: dict[int, str] = {}   # heading block index -> caption text left out of it
        self.title = ""
        self._buffer: list[str] = []
        self._buffer_has_text = False
        self._cell_separator = False
        self._heading_level = 0
        self._skip_depth = 0
        self._in_head = False
        self._in_title = False
        self._title_seen = False
        self._svg_depth = 0
        self._pre_depth = 0
        self._pre_has_code = False
        self._caption_tag = ""
        self._caption_depth = 0
        self._caption_text: list[str] = []

    def _flush(self) -> None:
        text = normalize_space("".join(self._buffer))
        caption = normalize_space("".join(self._caption_text))
        self._buffer.clear()
        self._caption_text.clear()
        self._buffer_has_text = False
        self._cell_separator = False
        if self._heading_level and caption:
            if text:
                self.heading_captions[len(self.blocks)] = caption
            else:
                text = caption   # a heading that is only a caption keeps it
        if text:
            kind = HEADING if self._heading_level else PARAGRAPH
            self.blocks.append((Block(text, kind), self._heading_level))
        self._heading_level = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "head":
            self._in_head = True
            return
        if self._in_head and not self._skip_depth and tag not in _HEAD_TAGS:
            self._in_head = False
        if tag == "title":
            # Only the page's own <title>; an inline SVG icon's <title> is not the page name.
            self._in_title = not self._title_seen and not self._svg_depth
            self._title_seen = self._title_seen or self._in_title
            return
        if tag == "svg":
            self._svg_depth += 1
        if tag in _SKIP_TAGS:
            if tag not in _VOID_TAGS:
                self._skip_depth += 1
            return
        if self._skip_depth or self._in_head:
            return
        if self._caption_tag:
            if tag == self._caption_tag:
                self._caption_depth += 1
            return
        if tag not in _VOID_TAGS and _is_caption(attrs):
            # Illustration captions (Project Gutenberg books have one per picture) are not read aloud,
            # and one inside a heading is kept out of the chapter title.
            if tag in _BLOCK_TAGS:
                self._flush()
            self._caption_tag, self._caption_depth = tag, 1
            return
        if tag == "br":
            self._buffer.append(" ")
        elif tag == "img":
            alt = normalize_space(dict(attrs).get("alt") or "")
            if _is_drop_cap(alt):
                self.handle_data(alt)   # a picture of a chapter's first letter carries that letter as its alt text
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
        for name, value in attrs:
            if value and (name == "id" or (name == "name" and tag == "a")):
                self.anchors.setdefault(value, len(self.blocks))

    def handle_startendtag(self, tag: str, attrs) -> None:
        # A self-closing <svg/>, <script/> or <title/> (or caption) has no end tag to leave it again.
        if tag in _SKIP_TAGS or tag == "title" or self._caption_tag or _is_caption(attrs):
            return
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
            return
        if tag == "head":
            self._in_head = False
            return
        if tag == "svg":
            self._svg_depth = max(0, self._svg_depth - 1)
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if self._skip_depth:
            return
        if self._caption_tag:
            if tag == self._caption_tag:
                self._caption_depth -= 1
                if not self._caption_depth:
                    self._caption_tag = ""
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
        elif self._caption_tag:
            if self._heading_level and not self._skip_depth:
                self._caption_text.append(data)
        elif not self._skip_depth and not self._in_head:
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


def _is_drop_cap(alt: str) -> bool:
    """One letter, maybe after an opening quote, like the alt of a decorated first letter."""
    return 0 < len(alt) <= 2 and alt[-1].isalpha() and (len(alt) == 1 or not alt[0].isalnum())


def _is_caption(attrs) -> bool:
    return any(name == "class" and "caption" in (value or "").split() for name, value in attrs)


def declared_encoding(data: bytes, *, xml: bool = False) -> str | None:
    """The encoding a page names with its byte-order mark, XML declaration or <meta> charset.

    An XML document (an EPUB chapter) names it only with its BOM or XML declaration: XML parsers
    ignore <meta>, so one left over from a converted HTML page means nothing.
    """
    if data.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return "utf-16"
    if xml:
        declaration = _XML_DECLARATION.match(data)
        labels = [declaration.group(1)] if declaration else []
    else:
        labels = (match.group(1) or match.group(2) for match in _DECLARED_CHARSET.finditer(data[:1024]))
    for label in labels:
        # As browsers do, a label that names no web encoding is passed over.
        codec = _ENCODING_LABELS.get(label.decode("ascii").lower())
        if codec:
            return codec
    return None


def decode_declared(data: bytes, codec: str) -> str:
    """Decode a page in the encoding it declared, as a browser does: that encoding holds for the
    whole page, and a byte sequence it cannot read is one U+FFFD."""
    return data.decode(codec, errors=_WEB_DECODE_ERRORS)


def _web_decode_error(error: UnicodeError) -> tuple[str, int]:
    """What the WHATWG decoder reads where the Python codec standing in for it gives up.

    That is Windows' euro sign in GBK (0x80) and Big5 (0xA3E1), and NEC's and IBM's additions to
    JIS X 0208 in EUC-JP (circled digits, Roman numerals); anything else is one U+FFFD. A double-byte
    decoder takes the byte after a lead byte with it unless that byte is ASCII, so the characters
    after a bad pair are not read from the middle of one.
    """
    if not isinstance(error, UnicodeDecodeError):
        raise error
    data, start = error.object, error.start
    codec = codecs.lookup(error.encoding).name
    if codec not in _DOUBLE_BYTE_CODECS:
        return "\ufffd", error.end
    lead, *rest = data[start:start + 4]
    if codec == "gb18030" and lead == 0x80:
        return "\u20ac", start + 1
    if not rest or not 0x81 <= lead <= 0xFE:
        return "\ufffd", start + 1
    if codec == "gb18030" and len(rest) == 3 and 0x30 <= rest[0] <= 0x39 and rest[1] >= 0x81 and 0x30 <= rest[2] <= 0x39:
        return "\ufffd", start + 4   # a four-byte sequence that names no character
    if rest[0] < 0x80:
        return "\ufffd", start + 1   # the ASCII byte is read again on its own
    if codec == "euc_jp" and lead == 0x8F and len(rest) > 1 and 0xA1 <= rest[0] <= 0xFE and 0xA1 <= rest[1] <= 0xFE:
        return "\ufffd", start + 3   # a three-byte JIS X 0212 sequence
    if codec == "big5hkscs" and (lead, rest[0]) == (0xA3, 0xE1):
        return "\u20ac", start + 2
    if codec == "euc_jp" and lead >= 0xA1 and 0xA1 <= rest[0] <= 0xFE:
        # The same JIS X 0208 row and cell in Shift_JIS, which Windows-31J reads with those additions.
        first, second = divmod((lead - 0xA1) * 94 + rest[0] - 0xA1, 188)
        shift_jis = bytes((first + (0x81 if first < 0x1F else 0xC1), second + (0x40 if second < 0x3F else 0x41)))
        with contextlib.suppress(UnicodeDecodeError):
            return shift_jis.decode("cp932"), start + 2
    return "\ufffd", start + 2


codecs.register_error(_WEB_DECODE_ERRORS, _web_decode_error)


class ParsedPage(NamedTuple):
    title: str
    blocks: list[tuple[Block, int]]   # text blocks with heading levels (0 for paragraphs)
    anchors: dict[str, int]           # element id -> index of the block it points at
    heading_captions: dict[int, str]  # heading block index -> caption text left out of the heading


def parse_page(markup: str) -> ParsedPage:
    parser = _BlockParser()
    parser.feed(markup)
    parser.close()
    return ParsedPage(normalize_space(parser.title), parser.blocks, parser.anchors, parser.heading_captions)


def html_to_blocks(markup: str) -> tuple[str, list[tuple[Block, int]]]:
    """Return the page title and its text blocks with heading levels (0 for paragraphs)."""
    page = parse_page(markup)
    return page.title, page.blocks


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
