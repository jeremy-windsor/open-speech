"""Streaming text cleanup and speech-sized segmentation."""

from __future__ import annotations

import html
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass


_CLOSING_QUOTE = (
    r"(?:[\"')\]\u2019\u201d\u00bb]|"
    r"&(?:quot|rdquo|rsquo|#34|#8217|#8221|#x22|#x2019|#x201d);)*"
)
_SENTENCE_BOUNDARY = re.compile(rf"[.!?]{_CLOSING_QUOTE}(?=\s)", re.IGNORECASE)
_SENTENCE_AT_END = re.compile(rf"[.!?]{_CLOSING_QUOTE}$", re.IGNORECASE)
_CLAUSE_BOUNDARY = re.compile(rf"[;:]{_CLOSING_QUOTE}(?=\s)", re.IGNORECASE)
_COMMA_BOUNDARY = re.compile(r",(?=\s)")
_ENTITY_BEFORE_SEMICOLON = re.compile(
    r"&(?:#\d{1,7}|#[xX][0-9a-fA-F]{1,6}|[A-Za-z][A-Za-z0-9]{1,31})$"
)
_NATURAL_CAP_BOUNDARY = re.compile(
    rf"[,;:\u2014]{_CLOSING_QUOTE}(?=\s)",
    re.IGNORECASE,
)
# Japanese and Chinese end a sentence at the ideographic full stop and the full-width "!" and "?" (U+3002, U+FF01,
# U+FF1F), Hindi at the danda and double danda (U+0964, U+0965), all Sentence_Terminal in UAX #29; Japanese and Chinese
# put no space after them. The ideographic and full-width commas, ";" and ":" (U+3001, U+FF0C, U+FF1B, U+FF1A) end a
# clause. Such a mark, with the closing brackets and quotes after it, ends once a character that can't continue it
# follows: a streamed full stop waits for a corner bracket (U+300D) that may come next. The ASCII rules above stay as
# they are, so text without these marks is cut exactly as before.
_WIDE_SENTENCE_ENDS = "\u3002\uff01\uff1f\u0964\u0965"
_WIDE_CLAUSE_MARKS = "\uff1b\uff1a"
_WIDE_COMMAS = "\u3001\uff0c"
_WIDE_CLOSERS = "\"')]\u2019\u201d\u00bb\u300d\u300f\u301f\uff09\uff3d\u3011\u3015\u3009\u300b\uff63"
_WIDE_CLOSING = (
    rf"(?:[{re.escape(_WIDE_CLOSERS)}]|"
    r"&(?:quot|rdquo|rsquo|#34|#8217|#8221|#x22|#x2019|#x201d);)*"
)
_WIDE_FOLLOWED = rf"(?![{re.escape(_WIDE_SENTENCE_ENDS + _WIDE_CLOSERS)}.!?&])(?=[\s\S])"
_WIDE_SENTENCE_RUN = rf"[{_WIDE_SENTENCE_ENDS}]+{_WIDE_CLOSING}"
_WIDE_SENTENCE_BOUNDARY = re.compile(_WIDE_SENTENCE_RUN + _WIDE_FOLLOWED)
_WIDE_SENTENCE_AT_END = re.compile(_WIDE_SENTENCE_RUN + "$")
_WIDE_CLAUSE_BOUNDARY = re.compile(rf"[{_WIDE_CLAUSE_MARKS}]{_WIDE_CLOSING}{_WIDE_FOLLOWED}")
_WIDE_COMMA_BOUNDARY = re.compile(rf"[{_WIDE_COMMAS}]{_WIDE_CLOSING}{_WIDE_FOLLOWED}")
_WIDE_CAP_BOUNDARY = re.compile(rf"[{_WIDE_CLAUSE_MARKS}{_WIDE_COMMAS}]{_WIDE_CLOSING}{_WIDE_FOLLOWED}")
# "Each word" in text with no spaces: up to the next mark that no space follows.
_WIDE_WORD_BOUNDARY = re.compile(
    rf"(?:[{_WIDE_SENTENCE_ENDS}]+|[{_WIDE_CLAUSE_MARKS}{_WIDE_COMMAS}]){_WIDE_CLOSING}"
    rf"(?![{re.escape(_WIDE_SENTENCE_ENDS + _WIDE_CLOSERS)}.!?&\s])(?=[\s\S])"
)
_WIDE_MARKS = _WIDE_SENTENCE_ENDS + _WIDE_CLAUSE_MARKS + _WIDE_COMMAS
# A responsive cut at a full-width comma waits for about four words' worth of text, as one at ", " waits for four words.
_MIN_WIDE_CLAUSE_CHARS = 8
_BLANK_LINE = re.compile(r"\n[ \t]*\n")
_STRUCTURAL_LINE = re.compile(
    r"^[ \t]*(?:(?:#{1,6}|>|[-+*]|\d+[.)])[ \t]+\S.*|"
    r"(?:\*\*|__)(?=\S).*(?<=\S)(?:\*\*|__))[ \t]*$"
)
# Live Reader links: a label may wrap a line and hold one level of brackets ("[[1]](url)"), so
# each "[" scans only to the next unpaired one; a URL may hold one level of parentheses
# (Wikipedia's "Foo_(bar)").
_LINK_LABEL = r"(?:[^\[\]\n]|\n(?![ \t]*\n))"
_LINK_URL = r"(?:[^()\n]|\([^()\n]*\))*"
_COMPLETE_LINK = re.compile(rf"!?\[((?:{_LINK_LABEL}|\[[^\[\]\n]*\])*)\]\({_LINK_URL}\)")
# What is left of a link whose "[" went out in an earlier segment.
_LINK_TAIL = re.compile(r"\]\((?:[^()\s]|\([^()\s]*\))*\)")
_MAX_LINK_URL_CHARS = 2048
# Stands for a link label in the boundary search: not a space, letter, digit or punctuation
# mark, so no boundary rule reads it ("[1](url)." must not look like the initial "x.").
_LABEL_MASK = "\u2060"
# clean_markdown's inline spans. Each opener pairs with the first closer after it, as a lazy
# regex pairs them, but _NextMatch looks for the closer: a regex such as
# "\*(?=\S)(.+?)(?<=\S)\*" reads the rest of the line again from every opener that has no
# closer, which is quadratic and holds the GIL for the whole of one long Reader paragraph.
_LINK_OPENER = re.compile(r"!?\[")
_LABEL_END = re.compile(r"\]")
_URL_END = re.compile(r"\)")
_LINE_END = re.compile(r"\n")
# Only what CommonMark reads as raw HTML or an autolink: "x < 5 and y > 3" is prose. A comment,
# processing instruction or declaration ends at the first "-->", "?>" or ">" (_strip_html).
_HTML_OPENER = re.compile(r"<")
_COMMENT_END = re.compile(r"-->")
_INSTRUCTION_END = re.compile(r"\?>")
_DECLARATION_START = re.compile(r"<![A-Za-z]")
_DECLARATION_END = re.compile(r">")
_HTML_TAG = re.compile(
    r"</?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>"
    r"|<[A-Za-z][A-Za-z0-9+.-]{1,31}:[^\s<>]*>|<[\w.+-]+@[\w.-]+>"
)
# CommonMark 4.5: a code fence starts a line; mid-line backticks or tildes are prose.
_FENCE_RUN = re.compile(r"(?m)^[ \t]*(`{3,}|~{3,})")
# An opening line still arriving waits this long for a backtick that would make it inline code.
_MAX_FENCE_INFO_CHARS = 256
_LINE_MARKER = re.compile(r"(?m)^[ \t]*(?:#{1,6}|>|[-+*]|\d+[.)])[ \t]+")
# "**", "__" or "~~" before a non-space, closed by the same pair after one (_strip_emphasis).
_PAIRED_OPENER = re.compile(r"(\*\*|__|~~)(?=\S)")
_PAIRED_CLOSERS = {marker: re.compile(rf"(?<=\S){re.escape(marker)}") for marker in ("**", "__", "~~")}
# A "*" or "_" that starts a word, closed by one that ends a word.
_SINGLE_OPENERS = {marker: re.compile(rf"(?<!\w){re.escape(marker)}(?=\S)") for marker in "*_"}
_SINGLE_CLOSERS = {marker: re.compile(rf"(?<=\S){re.escape(marker)}(?!\w)") for marker in "*_"}
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_CODE_PLACEHOLDER = re.compile("\ufff0(\\d+)\ufff1")
_MARKER_RUN = re.compile(r"(?:[*_`]|~{2,})+")
_COMPLETED_WORD = re.compile(r"(?<!\S)\S+\s+")
# A sound tag such as "[laugh]" or "[clear throat]": Chatterbox Turbo reads each as one token, so a cut by a cap
# never falls inside one.
_SOUND_TAG_MAX_CHARS = 32
_SOUND_TAG = re.compile(rf"\[[^\[\]\n]{{1,{_SOUND_TAG_MAX_CHARS - 2}}}\]")
_REGIONAL_INDICATOR = ("\U0001f1e6", "\U0001f1ff")

_NUMBERED_LINE = re.compile(r"\s*\d+")
_WORD_AT_END = re.compile(r"([A-Za-z]+)$")
_NONTERMINAL_ABBREVIATIONS = frozenset(
    {"dr", "e.g", "etc", "i.e", "jr", "mr", "mrs", "ms", "prof", "sr", "st", "vs"}
)


@dataclass(frozen=True, slots=True)
class TextSegment:
    """A speakable text fragment and its approximate source range."""

    text: str
    source_start: int
    source_end: int


def clean_markdown(text: str, *, fragment: bool = False, at_line_start: bool = True) -> str:
    """Turn common inline Markdown into prose suitable for speech.

    ``fragment`` marks one piece of a longer stream, where a pair of emphasis or
    code markers, or a link, may have been split between pieces: a marker left
    at the edge of a word is then dropped, so ``**very`` is spoken as ``very``,
    and so is the ``](url)`` left of a link.

    ``at_line_start`` false marks text that begins mid-line in its source. A
    heading, quote or list marker exists only at a line start (CommonMark), so
    its first line keeps a leading ``2024.``, ``3)`` or ``>`` as prose.
    """
    inline_code: list[str] = []

    def stash_inline_code(match: re.Match) -> str:
        inline_code.append(match.group(1))
        return f"\ufff0{len(inline_code) - 1}\ufff1"

    def restore_inline_code(match: re.Match) -> str:
        index = int(match.group(1))
        return inline_code[index] if index < len(inline_code) else match.group(0)

    text = _INLINE_CODE.sub(stash_inline_code, text)
    text = _strip_links(text)
    if fragment:
        text = _LINK_TAIL.sub("", text)
    text = _strip_html(text)
    # A match at 0 is on the first line: every later line start follows a "\n".
    text = _LINE_MARKER.sub(lambda marker: "" if at_line_start or marker.start() else marker.group(0), text)
    text = _strip_emphasis(text)
    text = _strip_single_markers(text, "*")
    text = _strip_single_markers(text, "_")
    if fragment:
        text = re.sub(r"\S+", _drop_split_markers, text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    if inline_code:
        # In one pass: replacing each placeholder in turn would copy the whole text once per span.
        text = _CODE_PLACEHOLDER.sub(restore_inline_code, text)
    return text


class _NextMatch:
    """Where a pattern next matches at or after a position, for positions that never go back.

    A search runs again only once the position passes the match it found last, so the searches
    of one pass over a text read it about once in all.
    """

    __slots__ = ("_pattern", "_start", "_text")

    def __init__(self, pattern: re.Pattern[str], text: str) -> None:
        self._pattern = pattern
        self._text = text
        self._start: int | None = None   # where the last search found a match, -1 for none

    def at_or_after(self, position: int) -> int:
        """Where the first match at or after position starts, or -1."""
        if self._start is None or 0 <= self._start < position:
            match = self._pattern.search(self._text, position)
            self._start = match.start() if match else -1
        return self._start


def _replace_spans(
    text: str, opener: re.Pattern[str], close: Callable[[re.Match[str]], tuple[int, str] | None]
) -> str:
    """Replace spans left to right, as re.sub does.

    close(opener match) gives (where the span ends, its replacement), or None when that opener
    has no closer: the next opener is then looked for one character on.
    """
    parts: list[str] = []
    copied = position = 0
    while (start := opener.search(text, position)) is not None:
        span = close(start)
        if span is None:
            position = start.start() + 1
            continue
        end, replacement = span
        parts += (text[copied : start.start()], replacement)
        copied = position = end
    parts.append(text[copied:])
    return "".join(parts)


def _strip_links(text: str) -> str:
    """A link becomes its label: the label ends at the first "]", the URL at the first ")"."""
    if "[" not in text:
        return text
    label_ends, url_ends = _NextMatch(_LABEL_END, text), _NextMatch(_URL_END, text)

    def close(start: re.Match[str]) -> tuple[int, str] | None:
        label = start.end()
        label_end = label_ends.at_or_after(label)
        if label_end <= label or not text.startswith("(", label_end + 1):
            return None   # no "]", an empty label, or no URL after it
        url_end = url_ends.at_or_after(label_end + 2)
        return None if url_end < 0 else (url_end + 1, text[label:label_end])

    return _replace_spans(text, _LINK_OPENER, close)


def _strip_html(text: str) -> str:
    """Each HTML comment, processing instruction, declaration, tag or autolink becomes a space."""
    if "<" not in text:
        return text
    # How far past the "<" the closer may start at the earliest, its search and its length.
    comment = (4, _NextMatch(_COMMENT_END, text), 3)
    instruction = (2, _NextMatch(_INSTRUCTION_END, text), 2)
    declaration = (3, _NextMatch(_DECLARATION_END, text), 1)

    def close(start: re.Match[str]) -> tuple[int, str] | None:
        at = start.start()
        if text.startswith("<!--", at):
            skip, closers, closer_length = comment
        elif text.startswith("<?", at):
            skip, closers, closer_length = instruction
        elif _DECLARATION_START.match(text, at):
            skip, closers, closer_length = declaration
        else:
            tag = _HTML_TAG.match(text, at)
            return (tag.end(), " ") if tag else None
        end = closers.at_or_after(at + skip)
        return None if end < 0 else (end + closer_length, " ")

    return _replace_spans(text, _HTML_OPENER, close)


def _strip_emphasis(text: str) -> str:
    """"**strong**", "__strong__" and "~~struck~~" keep their text, which stays on one line."""
    if not any(marker in text for marker in _PAIRED_CLOSERS):
        return text
    closers = {marker: _NextMatch(pattern, text) for marker, pattern in _PAIRED_CLOSERS.items()}
    line_ends = _NextMatch(_LINE_END, text)

    def close(start: re.Match[str]) -> tuple[int, str] | None:
        body = start.end()
        end = closers[start.group(1)].at_or_after(body + 1)
        if end < 0 or 0 <= line_ends.at_or_after(body) < end:
            return None
        return end + 2, text[body:end]

    return _replace_spans(text, _PAIRED_OPENER, close)


def _strip_single_markers(text: str, marker: str) -> str:
    """"*soft*" or "_soft_" keeps its text; "2 * 3" and snake_case keep their markers."""
    if marker not in text:
        return text
    closers, line_ends = _NextMatch(_SINGLE_CLOSERS[marker], text), _NextMatch(_LINE_END, text)

    def close(start: re.Match[str]) -> tuple[int, str] | None:
        body = start.end()
        end = closers.at_or_after(body + 1)
        if end < 0 or 0 <= line_ends.at_or_after(body) < end:
            return None
        return end + 1, text[body:end]

    return _replace_spans(text, _SINGLE_OPENERS[marker], close)


def _drop_split_markers(word: re.Match) -> str:
    """Drop marker runs at a word's edges; "2 * 3" and snake_case keep theirs."""
    token = word.group(0)
    if not any(char.isalnum() for char in token):
        return token

    def drop(run: re.Match) -> str:
        before = token[run.start() - 1] if run.start() else ""
        after = token[run.end()] if run.end() < len(token) else ""
        return run.group(0) if before.isalnum() and after.isalnum() else ""

    return _MARKER_RUN.sub(drop, token)


def _outside_sound_tag(text: str, boundary: int) -> int:
    """``boundary``, or the start of the sound tag it would cut, when that tag doesn't start the text."""
    window = max(0, boundary - _SOUND_TAG_MAX_CHARS)
    for tag in _SOUND_TAG.finditer(text, window, boundary + _SOUND_TAG_MAX_CHARS):
        if tag.start() < boundary < tag.end():
            return tag.start() if tag.start() > 0 else boundary
    return boundary


def _last_wide_mark_end(text: str) -> int:
    """Just past the last full-width sentence or clause mark (or danda) in text and its closing brackets, else 0."""
    end = max(text.rfind(mark) for mark in _WIDE_MARKS) + 1
    if end:
        while end < len(text) and text[end] in _WIDE_CLOSERS:
            end += 1
    return end


def _is_regional_indicator(char: str) -> bool:
    return _REGIONAL_INDICATOR[0] <= char <= _REGIONAL_INDICATOR[1]


def _inside_code_span(text: str, position: int, ticks_before: int = 0) -> bool:
    """Whether position is inside inline code; ticks_before counts the line's backticks before text."""
    line_start = text.rfind("\n", 0, position) + 1
    ticks = text.count("`", line_start, position) + (0 if line_start else ticks_before)
    return ticks % 2 == 1


class LiveTextSegmenter:
    """Accept arbitrary text deltas and emit bounded, ordered TTS fragments.

    ``natural`` prefers full sentences and paragraphs. ``responsive`` also
    releases short clauses at commas, colons, and semicolons for lower-latency
    token streams. ``instant_word`` emits each completed word. Fenced code is
    discarded while a short cue is inserted once at the opening fence. A
    finished Markdown link is replaced by its label as soon as it arrives, so
    no segment can carry its URL. Both caps bound every cut.
    """

    VALID_MODES = frozenset({"natural", "responsive", "instant_word"})
    CODE_BLOCK_CUE = "Code block skipped. "

    def __init__(
        self,
        *,
        mode: str = "natural",
        max_chars: int = 200,
        max_words: int = 20,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported latency mode: {mode}")
        self.mode = mode
        self.max_chars = max(20, int(max_chars))
        self.max_words = max(1, int(max_words))
        # Every boundary lies within the caps, or (natural) ends a sentence whose speakable
        # text fits them, so only this much of the buffer is searched; a large delta stays linear.
        self._window = self.max_chars * 4 + 256 if mode == "natural" else self.max_chars + 64
        # A link that is still arriving: "[label", "[label]" or "[label](url" at the end of
        # the buffer. A longer label or URL is taken for prose, so a stray "[" cannot stall.
        self._link_in_progress = re.compile(
            rf"!?\[{_LINK_LABEL}{{0,{self.max_chars}}}"
            rf"(?:\](?:\({_LINK_URL}(?:\([^()\n]*)?)?)?\Z"
        )
        self._link_tail = self.max_chars + _MAX_LINK_URL_CHARS + 4
        self.reset()

    @property
    def accepted_chars(self) -> int:
        return self._source_cursor

    @property
    def in_code_fence(self) -> bool:
        return self._in_code_fence

    def reset(self) -> None:
        self._chars: list[str] = []
        # _chars with link-label characters as "x": soft boundaries keep each label whole.
        self._masked: list[str] = []
        self._positions: list[tuple[int, int]] = []
        self._raw_pending = ""
        self._raw_pending_start = 0
        self._source_cursor = 0
        self._in_code_fence = False
        self._fence = "```"
        # Only spaces or tabs since the last line break (or the start): a fence may begin here.
        self._line_start = True
        # The same for the next segment, in _chars: only there may it begin with a list or quote marker.
        self._segment_line_start = True
        # The next segment starts a line or a sentence, so a number there is a list number, as in the
        # harness ("Add the flour. 2. Mix well."). An idle release or a cap cut can end mid-sentence.
        self._segment_starts_line_or_sentence = True
        # Backticks on the current line in text already emitted, which decide inline code.
        self._emitted_line_ticks = 0
        # Where an unfinished link starts in _chars, or len(_chars) when none is arriving. Its
        # label may wrap a line, but only a link on the last line holds boundaries back.
        self._reduce_from = 0
        self._link_from = 0

    def append(self, text: str) -> list[TextSegment]:
        if not text:
            return []
        if not self._raw_pending:
            self._raw_pending_start = self._source_cursor
        self._raw_pending += text
        self._source_cursor += len(text)
        appended_from = len(self._chars)
        self._process_raw(force=False)
        self._reduce_links(appended_from)
        return self._pop_ready()

    def flush(self, *, idle: bool = False) -> list[TextSegment]:
        """Flush buffered prose.

        An idle flush preserves a trailing unfinished word or link. An explicit
        flush emits all remaining prose. Content inside an unclosed code fence
        stays discarded in both cases. Neither starts a new line: text appended
        afterwards continues the source where it stopped, so it starts a line
        only after a line break, as after ``reset()``.
        """
        appended_from = len(self._chars)
        self._process_raw(force=not idle)
        self._reduce_links(appended_from)
        if not idle:
            # An unfinished link is prose now. The line state stays as the text left it: a commit in
            # mid-line doesn't end the line, and Speak > Live types on after one.
            self._reduce_from = self._link_from = len(self._chars)
            segments = self._pop_ready(force=True)
            self._emitted_line_ticks = 0
            return segments
        return self._pop_ready(idle=True)

    def continue_from(self, text: str) -> None:
        """Make the next appended text continue ``text``: the source just before it, from its line's start.

        ``text`` was read already and is not spoken, and it counts in no offset. Only its line state is kept:
        whether the next text starts a line (``text`` is empty, or ends a line, or has only spaces or tabs after
        its last line break), whether it starts a sentence, and the backticks on its line. A fence or link it
        leaves open isn't carried over. Only before the first ``append()`` after ``reset()``; raises ValueError
        later (as for a client resending from mid-line after ``response.cancel``).
        """
        if self._source_cursor or self._chars or self._raw_pending:
            raise ValueError("The text before the next one can only be given before the first text")
        if not text:
            return
        tail = text[text.rfind("\n") + 1 :]
        line_start = not tail.strip(" \t")
        self._line_start = self._segment_line_start = line_start
        self._emitted_line_ticks = tail.count("`") % 2
        # Only the end decides a sentence end; a window keeps a long line linear.
        end = tail[-self._window :].rstrip()
        ends_sentence = bool(end) and self._sentence_at_end(end, len(end) == len(tail.rstrip()))
        self._segment_starts_line_or_sentence = line_start or ends_sentence

    def _consume_raw(self, length: int) -> tuple[str, int]:
        text = self._raw_pending[:length]
        start = self._raw_pending_start
        self._raw_pending = self._raw_pending[length:]
        self._raw_pending_start += length
        if text:
            line_break = text.rfind("\n")
            indented = not text[line_break + 1 :].strip(" \t")
            self._line_start = indented and (line_break >= 0 or self._line_start)
        return text, start

    def _append_plain(self, text: str, source_start: int) -> None:
        self._chars.extend(text)
        self._masked.extend(text)
        self._positions.extend(
            (source_start + index, source_start + index + 1) for index in range(len(text))
        )

    def _append_code_cue(self, source_start: int, source_end: int) -> None:
        self._chars.extend(self.CODE_BLOCK_CUE)
        self._masked.extend(self.CODE_BLOCK_CUE)
        self._positions.extend((source_start, source_end) for _ in self.CODE_BLOCK_CUE)

    def _process_raw(self, *, force: bool) -> None:
        while self._raw_pending:
            raw = self._raw_pending
            if self._in_code_fence:
                close = self._closing_fence(raw)
                if close is not None and (close.end(1) < len(raw) or force):
                    # The rest of a closing line is prose again, as before ("``` Done.").
                    self._consume_raw(close.end(1))
                    self._in_code_fence = False
                    continue
                # Discard the code but keep a closing fence that may still be arriving.
                keep_from = len(raw)
                if not force:
                    keep_from = close.start(1) if close else self._partial_fence_start(raw, self._fence[0])
                if keep_from:
                    self._consume_raw(keep_from)
                if force:
                    self._in_code_fence = False
                return

            opening, hold_from = self._opening_fence(raw, force=force)
            if opening is not None:
                if opening.start(1):
                    plain, plain_start = self._consume_raw(opening.start(1))
                    self._append_plain(plain, plain_start)
                self._fence, opening_start = self._consume_raw(len(opening.group(1)))
                self._append_code_cue(opening_start, opening_start + len(self._fence))
                self._in_code_fence = True
                continue

            if hold_from is None:
                hold_from = len(raw) if force else self._partial_fence_start(raw, "`~")
            if hold_from:
                plain, plain_start = self._consume_raw(hold_from)
                self._append_plain(plain, plain_start)
            return
        if force:
            self._in_code_fence = False

    def _opening_fence(self, raw: str, *, force: bool) -> tuple[re.Match | None, int | None]:
        """Return the first opening fence in raw, or where to hold text that may become one."""
        for match in _FENCE_RUN.finditer(raw):
            if not match.start() and not self._line_start:
                continue
            run_end = match.end(1)
            if match.group(1)[0] == "`":
                line_end = raw.find("\n", run_end)
                info = raw[run_end:] if line_end < 0 else raw[run_end:line_end]
                if "`" in info:
                    continue  # "```code```" is inline code, as in the Reader's parser
                if line_end < 0 and not force and len(info) < _MAX_FENCE_INFO_CHARS:
                    return None, match.start(1)
            elif run_end == len(raw) and not force:
                return None, match.start(1)  # the run may grow, and its length picks the closing fence
            return match, None
        return None, None

    def _closing_fence(self, raw: str) -> re.Match | None:
        """CommonMark 4.5: a closing fence starts a line, using the opening character at least as often."""
        pattern = rf"(?m)^[ \t]*({re.escape(self._fence[0])}{{{len(self._fence)},}})"
        for match in re.finditer(pattern, raw):
            if match.start() or self._line_start:
                return match
        return None

    def _partial_fence_start(self, raw: str, fence_chars: str) -> int:
        """Where a trailing run of fence characters at a line start begins, else len(raw)."""
        line_start = raw.rfind("\n") + 1
        if not line_start and not self._line_start:
            return len(raw)
        run = raw[line_start:].lstrip(" \t")
        if run and run[0] in fence_chars and run == run[0] * len(run):
            return len(raw) - len(run)
        return len(raw)

    def _reduce_links(self, appended_from: int) -> None:
        """Replace each finished link with its label, so no cut can expose a URL."""
        if len(self._chars) <= appended_from:
            return
        start = min(self._reduce_from, appended_from)
        while True:
            text = "".join(self._chars[start:])
            ticks_before = self._ticks_before(start)
            links = [
                match
                for match in _COMPLETE_LINK.finditer(text)
                if not _inside_code_span(text, match.start(), ticks_before)
            ]
            if not links:
                break
            self._replace_links(start, links)
            # An image inside a label can finish the link around it, which may start earlier.
            start = max(0, start - self._link_tail)
        tail_start = max(0, len(self._chars) - self._link_tail)
        in_progress = self._link_in_progress.search("".join(self._chars[tail_start:]))
        self._reduce_from = tail_start + in_progress.start() if in_progress else len(self._chars)
        # A "[" with a line break after it is more likely stray (":-[") than a wrapped label,
        # so later lines are not held back, as an unfinished word would not hold them.
        holds = in_progress is not None and "\n" not in in_progress.group(0)
        self._link_from = self._reduce_from if holds else len(self._chars)

    def _ticks_before(self, index: int) -> int:
        """Backticks on the line of _chars[index] before it, including text already emitted."""
        before = "".join(self._chars[:index])
        line_break = before.rfind("\n")
        if line_break >= 0:
            return before.count("`", line_break + 1)
        return self._emitted_line_ticks + before.count("`")

    def _replace_links(self, start: int, links: list[re.Match]) -> None:
        """Rebuild _chars[start:] with each link (offsets relative to start) cut to its label."""
        chars, masked, positions = self._chars[start:], self._masked[start:], self._positions[start:]
        new_chars: list[str] = []
        new_masked: list[str] = []
        new_positions: list[tuple[int, int]] = []
        cursor = 0
        for link in links:
            new_chars += chars[cursor : link.start()]
            new_masked += masked[cursor : link.start()]
            new_positions += positions[cursor : link.start()]
            label_positions = positions[link.start(1) : link.end(1)]
            if label_positions:
                # The label stands for the whole link in the source range.
                label_positions[0] = (positions[link.start()][0], label_positions[0][1])
                label_positions[-1] = (label_positions[-1][0], positions[link.end() - 1][1])
            new_chars += chars[link.start(1) : link.end(1)]
            new_masked += _LABEL_MASK * len(label_positions)
            new_positions += label_positions
            cursor = link.end()
        self._chars[start:] = new_chars + chars[cursor:]
        self._masked[start:] = new_masked + masked[cursor:]
        self._positions[start:] = new_positions + positions[cursor:]

    def _soft_text(self, length: int) -> str:
        """The first length characters with link labels and an unfinished link hidden, keeping offsets."""
        soft_text = "".join(self._masked[:length])
        if self._link_from < len(soft_text):
            soft_text = soft_text[: self._link_from] + _LABEL_MASK * (len(soft_text) - self._link_from)
        return soft_text

    def _pop_ready(self, *, force: bool = False, idle: bool = False) -> list[TextSegment]:
        # A cut reads at most _window characters from the front and the buffer's length only against max_chars,
        # so a large buffer is cut from a front part of a few windows that the rest tops up as it is used: the
        # cuts are the same, and each costs its window, not the whole buffer, so a large delta stays linear
        # (deleting every segment from the whole buffer's front would cost its square).
        tail = self._set_tail_aside()
        segments: list[TextSegment] = []
        try:
            while True:
                if tail is not None and len(self._chars) < self._window:
                    tail = self._take_from_tail(tail, self._window * 3)
                if not self._chars:
                    break
                boundary = self._next_boundary()
                if boundary is None and tail is not None and (force or idle):
                    tail = self._take_from_tail(tail)  # the fallbacks below read the whole buffer
                if boundary is None and force:
                    boundary = len(self._chars)
                elif boundary is None and idle:
                    boundary = self._idle_boundary()
                if not boundary:
                    break
                source_has_content = bool("".join(self._chars[:boundary]).strip())
                segment = self._consume_segment(boundary)
                if segment.text or source_has_content:
                    segments.append(segment)
        finally:
            if tail is not None:
                self._take_from_tail(tail)
        return segments

    def _set_tail_aside(self) -> tuple[list[str], list[str], list[tuple[int, int]], int] | None:
        """Move all but the first few windows of a large buffer aside (to _take_from_tail); None for a small one."""
        front = self._window * 4
        if len(self._chars) <= front * 2:
            return None
        tail = (self._chars[front:], self._masked[front:], self._positions[front:], 0)
        del self._chars[front:]
        del self._masked[front:]
        del self._positions[front:]
        return tail

    def _take_from_tail(
        self, tail: tuple[list[str], list[str], list[tuple[int, int]], int], count: int | None = None,
    ) -> tuple[list[str], list[str], list[tuple[int, int]], int] | None:
        """Append the next ``count`` characters set aside (all of them by default); None once none are left."""
        chars, masked, positions, start = tail
        end = len(chars) if count is None else min(len(chars), start + count)
        self._chars.extend(chars[start:end])
        self._masked.extend(masked[start:end])
        self._positions.extend(positions[start:end])
        return None if end >= len(chars) else (chars, masked, positions, end)

    def _next_boundary(self) -> int | None:
        total = len(self._chars)
        text = "".join(self._chars[: self._window])
        # Soft boundaries keep a link label whole and never fall inside an unfinished link.
        soft_text = self._soft_text(len(text))
        candidates: list[int] = []
        if self.mode == "instant_word":
            match = re.match(r"\s*\S+\s+", soft_text)
            if match:
                candidates.append(match.end())
            wide = _WIDE_WORD_BOUNDARY.search(soft_text)
            if wide:
                candidates.append(wide.end())
        else:
            newline = self._natural_newline_boundary(soft_text)
            if newline is not None:
                candidates.append(newline)
            # A sentence ending after that line break would not be chosen, so it is not looked for.
            sentence = self._sentence_boundary(soft_text, newline, self._segment_starts_line_or_sentence)
            if sentence is not None:
                candidates.append(sentence)
            if self.mode == "natural":
                natural_boundary = min(candidates) if candidates else None
                if natural_boundary is not None and self._within_hard_limits(text[:natural_boundary]):
                    return natural_boundary
                # A sentence over the caps is cut at the hard boundary instead.
                candidates = []
            else:
                clause = self._clause_boundary(soft_text)
                if clause is not None:
                    candidates.append(clause)
                for comma in _COMMA_BOUNDARY.finditer(soft_text):
                    if len(re.findall(r"\S+", text[: comma.end()])) >= 4:
                        candidates.append(comma.end())
                        break
                for comma in _WIDE_COMMA_BOUNDARY.finditer(soft_text):
                    if len("".join(text[: comma.end()].split())) >= _MIN_WIDE_CLAUSE_CHARS:
                        candidates.append(comma.end())
                        break
        # Also bounds a soft boundary past the caps, such as one very long word or label.
        hard = self._hard_boundary(text, total)
        if hard is not None:
            candidates.append(hard)
        return min(candidates) if candidates else None

    def _within_hard_limits(self, text: str) -> bool:
        speakable = clean_markdown(text, fragment=True, at_line_start=self._segment_line_start)
        return len(speakable) <= self.max_chars and len(speakable.split()) <= self.max_words

    def _natural_newline_boundary(self, text: str) -> int | None:
        if self.mode == "responsive":
            newline = text.find("\n")
            return newline + 1 if newline >= 0 else None

        # The first paragraph break, or the first line break beside a heading, list item or
        # quote. Only lines before the paragraph break are read: this runs once per segment.
        blank_line = _BLANK_LINE.search(text)
        limit = blank_line.end() if blank_line else len(text)
        newline = text.find("\n")
        # A first line that starts mid-sentence is prose, however it begins ("1) mix").
        line_is_structural = (
            newline >= 0 and self._segment_starts_line_or_sentence and self._is_structural_line(text[:newline])
        )
        while 0 <= newline < limit:
            next_newline = text.find("\n", newline + 1)
            next_line = text[newline + 1 :] if next_newline < 0 else text[newline + 1 : next_newline]
            next_is_structural = self._is_structural_line(next_line)
            if line_is_structural or next_is_structural:
                return newline + 1
            newline, line_is_structural = next_newline, next_is_structural
        return blank_line.end() if blank_line else None

    @staticmethod
    def _is_structural_line(line: str) -> bool:
        return bool(line.strip() and _STRUCTURAL_LINE.fullmatch(line))

    def _hard_boundary(self, text: str, total: int) -> int | None:
        # Both caps bound the cut: a run of short words must not carry it past max_chars.
        completed_words = list(_COMPLETED_WORD.finditer(text, 0, self.max_chars))
        if len(completed_words) >= self.max_words:
            limit = completed_words[self.max_words - 1].end()
        elif total >= self.max_chars:
            limit = self.max_chars
        else:
            return None
        boundary = self._cap_clause_boundary(text, limit) if self.mode == "natural" else None
        if boundary is None:
            boundary = text.rfind(" ", 0, limit + 1)
            if boundary < limit // 2:
                boundary = limit
            else:
                boundary += 1
            # Japanese and Chinese have no spaces: a later full-width mark in the second half.
            wide = _last_wide_mark_end(text[:limit])
            if wide > limit // 2 and (boundary == limit or wide > boundary):
                boundary = wide
            boundary = self._safe_grapheme_boundary(text, boundary)
        boundary = _outside_sound_tag(text, boundary)
        if boundary > self._link_from:
            # Cut before a link that is still arriving; at the buffer start it waits for the rest.
            return self._link_from or None
        return boundary

    def _idle_boundary(self) -> int | None:
        text = self._soft_text(self._link_from)
        if not text:
            return None
        if self._link_from < len(self._chars):
            # An unfinished link waits for the rest of it, with the word it starts in (":-[").
            return self._last_word_end(text)
        starts = self._segment_starts_line_or_sentence
        if (
            text[-1].isspace()
            or self._sentence_boundary(text, None, starts) is not None
            or self._sentence_at_end(text, starts)
        ):
            return len(text)
        return self._last_word_end(text)

    @staticmethod
    def _last_word_end(text: str) -> int | None:
        """Just past the last space, or the last full-width mark in text with no spaces; None for neither."""
        last_space = max(text.rfind(" "), text.rfind("\n"), text.rfind("\t"))
        return max(last_space + 1, _last_wide_mark_end(text)) or None

    @staticmethod
    def _sentence_boundary(
        text: str, endpos: int | None = None, starts_line_or_sentence: bool = True
    ) -> int | None:
        """The end of the first sentence that ends before endpos (the end of text by default).

        ``starts_line_or_sentence`` false marks text that begins mid-sentence and mid-line.
        """
        end = len(text) if endpos is None else endpos
        wide = _WIDE_SENTENCE_BOUNDARY.search(text, 0, end)
        for match in _SENTENCE_BOUNDARY.finditer(text, 0, end):
            punctuation_at = match.start()
            if wide is not None and punctuation_at > wide.start():
                break
            if text[punctuation_at] == "." and LiveTextSegmenter._period_is_nonterminal(
                text, punctuation_at, starts_line_or_sentence
            ):
                continue
            return match.end()
        return wide.end() if wide is not None else None

    @staticmethod
    def _clause_boundary(text: str) -> int | None:
        wide = _WIDE_CLAUSE_BOUNDARY.search(text)
        for match in _CLAUSE_BOUNDARY.finditer(text):
            if wide is not None and match.start() > wide.start():
                break
            if text[match.start()] == ";" and _ENTITY_BEFORE_SEMICOLON.search(
                text[: match.start()]
            ):
                continue
            return match.end()
        return wide.end() if wide is not None else None

    @staticmethod
    def _sentence_at_end(text: str, starts_line_or_sentence: bool = True) -> bool:
        if _WIDE_SENTENCE_AT_END.search(text):
            return True
        match = _SENTENCE_AT_END.search(text)
        if match is None:
            return False
        punctuation_at = match.start()
        return not (
            text[punctuation_at] == "."
            and LiveTextSegmenter._period_is_nonterminal(text, punctuation_at, starts_line_or_sentence)
        )

    @staticmethod
    def _cap_clause_boundary(text: str, limit: int) -> int | None:
        minimum = limit // 2
        matches = [
            match
            for match in _NATURAL_CAP_BOUNDARY.finditer(text, minimum, limit)
            if not (
                text[match.start()] == ";"
                and _ENTITY_BEFORE_SEMICOLON.search(text[: match.start()])
            )
        ]
        ends = [match.end() for match in matches[-1:]]
        ends += [match.end() for match in _WIDE_CAP_BOUNDARY.finditer(text, minimum, limit)][-1:]
        return max(ends) if ends else None

    @staticmethod
    def _period_is_nonterminal(text: str, punctuation_at: int, starts_line_or_sentence: bool = True) -> bool:
        # Read in place, not on a copy of everything before: a list of "1." lines calls this per line.
        line_start = text.rfind("\n", 0, punctuation_at) + 1
        # A number is a list number where it starts its line or its sentence ("Add the flour. 2. Mix well.").
        if (line_start or starts_line_or_sentence) and _NUMBERED_LINE.fullmatch(text, line_start, punctuation_at):
            return True
        # A longer run of letters is neither an initial nor an abbreviation either way.
        word_match = _WORD_AT_END.search(text, max(0, punctuation_at - 66), punctuation_at)
        if not word_match:
            return False
        word = word_match.group(1).lower()
        return len(word) == 1 or word in _NONTERMINAL_ABBREVIATIONS

    @staticmethod
    def _safe_grapheme_boundary(text: str, boundary: int) -> int:
        """Avoid splitting combining-mark, emoji-ZWJ, emoji-tag and flag clusters."""
        while boundary < len(text):
            char = text[boundary]
            is_extend = (
                bool(unicodedata.combining(char))
                or unicodedata.category(char) in {"Mc", "Me", "Mn"}
                or char == "\ufe0f"
                or "\U0001f3fb" <= char <= "\U0001f3ff"
                or "\U000e0020" <= char <= "\U000e007f"
            )
            if is_extend:
                boundary += 1
                continue
            if char == "\u200d" and boundary + 1 < len(text):
                boundary += 2
                continue
            if _is_regional_indicator(char):
                # UAX #29 GB12/GB13: regional indicators pair up from the start of their run.
                run = 0
                while boundary - run > 0 and _is_regional_indicator(text[boundary - run - 1]):
                    run += 1
                if run % 2:
                    boundary += 1
                    continue
            break
        return boundary

    def _consume_segment(self, boundary: int) -> TextSegment:
        raw_text = "".join(self._chars[:boundary])
        # As the boundary search read it: a link label cannot end a sentence as an initial ("[a](url).").
        spoken_end = self._soft_text(boundary).rstrip()
        positions = self._positions[:boundary]
        del self._chars[:boundary]
        del self._masked[:boundary]
        del self._positions[:boundary]
        self._reduce_from = max(0, self._reduce_from - boundary)
        self._link_from = max(0, self._link_from - boundary)
        line_break = raw_text.rfind("\n")
        ticks = raw_text.count("`", line_break + 1)
        self._emitted_line_ticks = (ticks + (0 if line_break >= 0 else self._emitted_line_ticks)) % 2
        at_line_start = self._segment_line_start
        indented = not raw_text[line_break + 1 :].strip(" \t")
        self._segment_line_start = indented and (line_break >= 0 or at_line_start)
        # Only spaces keep the start as it was; an idle release or a cap cut may end mid-sentence.
        if spoken_end:
            ends_sentence = self._sentence_at_end(spoken_end, self._segment_starts_line_or_sentence)
        else:
            ends_sentence = self._segment_starts_line_or_sentence
        self._segment_starts_line_or_sentence = self._segment_line_start or ends_sentence
        source_start = min((position[0] for position in positions), default=0)
        source_end = max((position[1] for position in positions), default=source_start)
        return TextSegment(
            text=clean_markdown(raw_text, fragment=True, at_line_start=at_line_start),
            source_start=source_start,
            source_end=source_end,
        )


def split_sentences(text: str, max_chars: int = 500) -> list[tuple[int, int]]:
    """Return ``(start, end)`` offsets of speakable sentences in already-clean prose.

    Uses the same sentence and abbreviation rules as the Live Reader. A sentence
    longer than ``max_chars`` is split at a clause, then a word, boundary.
    """
    spans: list[tuple[int, int]] = []
    position = 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            break
        # Every rule below only looks inside max_chars, so a window keeps one huge
        # paragraph linear instead of re-copying the remaining text each sentence.
        rest = text[position:position + max_chars + 64]
        boundary = LiveTextSegmenter._sentence_boundary(rest) or len(rest)
        if boundary > max_chars:
            last_space = rest.rfind(" ", 0, max_chars)
            boundary = (
                LiveTextSegmenter._cap_clause_boundary(rest, max_chars)
                or (last_space + 1 if last_space > 0 else max_chars)
            )
            # Never inside a sound tag such as [clear throat].
            boundary = _outside_sound_tag(rest, boundary)
        boundary = LiveTextSegmenter._safe_grapheme_boundary(rest, boundary)
        end = position + boundary
        trimmed = end
        while trimmed > position and text[trimmed - 1].isspace():
            trimmed -= 1
        spans.append((position, trimmed))
        position = end
    return spans
