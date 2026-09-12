"""Streaming text cleanup and speech-sized segmentation."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


_STRONG_BOUNDARY = re.compile(r"[.!?;:](?:[\"')\]]*)?(?=\s|$)")
_COMMA_BOUNDARY = re.compile(r",(?=\s|$)")
_INLINE_LINK = re.compile(r"!?\[([^\]]+)\]\([^)]*\)")
_HTML_TAG = re.compile(r"<[^>]+>")
_LINE_MARKER = re.compile(r"(?m)^[ \t]*(?:#{1,6}|>|[-+*]|\d+[.)])[ \t]+")
_PAIRED_MARKER = re.compile(r"(\*\*|__|~~)(?=\S)(.+?)(?<=\S)\1")
_SINGLE_ASTERISK = re.compile(r"(?<!\w)\*(?=\S)(.+?)(?<=\S)\*(?!\w)")
_SINGLE_UNDERSCORE = re.compile(r"(?<!\w)_(?=\S)(.+?)(?<=\S)_(?!\w)")
_INLINE_CODE = re.compile(r"`([^`\n]+)`")

_NONTERMINAL_ABBREVIATIONS = frozenset(
    {"dr", "e.g", "etc", "i.e", "jr", "mr", "mrs", "ms", "prof", "sr", "st", "vs"}
)


@dataclass(frozen=True, slots=True)
class TextSegment:
    """A speakable text fragment and its approximate source range."""

    text: str
    source_start: int
    source_end: int


def clean_markdown(text: str) -> str:
    """Turn common inline Markdown into prose suitable for speech."""
    inline_code: list[str] = []

    def stash_inline_code(match: re.Match) -> str:
        inline_code.append(match.group(1))
        return f"\ufff0{len(inline_code) - 1}\ufff1"

    text = _INLINE_CODE.sub(stash_inline_code, text)
    text = _INLINE_LINK.sub(r"\1", text)
    text = _HTML_TAG.sub(" ", text)
    text = _LINE_MARKER.sub("", text)
    text = _PAIRED_MARKER.sub(r"\2", text)
    text = _SINGLE_ASTERISK.sub(r"\1", text)
    text = _SINGLE_UNDERSCORE.sub(r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    for index, code in enumerate(inline_code):
        text = text.replace(f"\ufff0{index}\ufff1", code)
    return text


class LiveTextSegmenter:
    """Accept arbitrary text deltas and emit bounded, ordered TTS fragments.

    ``natural`` waits for punctuation, a short-clause comma, an idle flush, or
    the hard size cap. ``instant_word`` emits each completed word. Fenced code
    is discarded while a short cue is inserted once at the opening fence.
    """

    VALID_MODES = frozenset({"natural", "instant_word"})
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
        self.reset()

    @property
    def accepted_chars(self) -> int:
        return self._source_cursor

    @property
    def in_code_fence(self) -> bool:
        return self._in_code_fence

    def reset(self) -> None:
        self._chars: list[str] = []
        self._positions: list[tuple[int, int]] = []
        self._raw_pending = ""
        self._raw_pending_start = 0
        self._source_cursor = 0
        self._in_code_fence = False

    def append(self, text: str) -> list[TextSegment]:
        if not text:
            return []
        if not self._raw_pending:
            self._raw_pending_start = self._source_cursor
        self._raw_pending += text
        self._source_cursor += len(text)
        self._process_raw(force=False)
        return self._pop_ready()

    def flush(self, *, idle: bool = False) -> list[TextSegment]:
        """Flush buffered prose.

        An idle flush preserves a trailing unfinished word. An explicit flush
        emits all remaining prose. Content inside an unclosed code fence stays
        discarded in both cases.
        """
        self._process_raw(force=not idle)
        return self._pop_ready(force=not idle, idle=idle)

    def _consume_raw(self, length: int) -> tuple[str, int]:
        text = self._raw_pending[:length]
        start = self._raw_pending_start
        self._raw_pending = self._raw_pending[length:]
        self._raw_pending_start += length
        return text, start

    def _append_plain(self, text: str, source_start: int) -> None:
        self._chars.extend(text)
        self._positions.extend(
            (source_start + index, source_start + index + 1) for index in range(len(text))
        )

    def _append_code_cue(self, source_start: int, source_end: int) -> None:
        self._chars.extend(self.CODE_BLOCK_CUE)
        self._positions.extend((source_start, source_end) for _ in self.CODE_BLOCK_CUE)

    def _process_raw(self, *, force: bool) -> None:
        while self._raw_pending:
            if self._in_code_fence:
                close_at = self._raw_pending.find("```")
                if close_at >= 0:
                    self._consume_raw(close_at + 3)
                    self._in_code_fence = False
                    continue
                keep = 0 if force else min(2, len(self._raw_pending))
                discard = len(self._raw_pending) - keep
                if discard:
                    self._consume_raw(discard)
                if force:
                    self._in_code_fence = False
                return

            fence_at = self._raw_pending.find("```")
            if fence_at >= 0:
                if fence_at:
                    plain, plain_start = self._consume_raw(fence_at)
                    self._append_plain(plain, plain_start)
                _opening, opening_start = self._consume_raw(3)
                self._append_code_cue(opening_start, opening_start + 3)
                self._in_code_fence = True
                continue

            trailing_ticks = 0
            if not force:
                if self._raw_pending.endswith("``"):
                    trailing_ticks = 2
                elif self._raw_pending.endswith("`"):
                    trailing_ticks = 1
            safe_length = len(self._raw_pending) - trailing_ticks
            if safe_length:
                plain, plain_start = self._consume_raw(safe_length)
                self._append_plain(plain, plain_start)
            return

    def _pop_ready(self, *, force: bool = False, idle: bool = False) -> list[TextSegment]:
        segments: list[TextSegment] = []
        while self._chars:
            boundary = self._next_boundary()
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
        return segments

    def _next_boundary(self) -> int | None:
        text = "".join(self._chars)
        if self.mode == "instant_word":
            match = re.match(r"\s*\S+\s+", text)
            return match.end() if match else self._hard_boundary(text)

        candidates: list[int] = []
        strong = self._strong_boundary(text)
        if strong is not None:
            candidates.append(strong)
        newline = text.find("\n")
        if newline >= 0:
            candidates.append(newline + 1)
        for comma in _COMMA_BOUNDARY.finditer(text):
            if len(re.findall(r"\S+", text[: comma.end()])) >= 4:
                candidates.append(comma.end())
                break
        hard = self._hard_boundary(text)
        if hard is not None:
            candidates.append(hard)
        return min(candidates) if candidates else None

    def _hard_boundary(self, text: str) -> int | None:
        completed_words = list(re.finditer(r"\S+\s+", text))
        if len(completed_words) >= self.max_words:
            return completed_words[self.max_words - 1].end()
        if len(text) < self.max_chars:
            return None
        boundary = text.rfind(" ", 0, self.max_chars + 1)
        if boundary < self.max_chars // 2:
            boundary = self.max_chars
        else:
            boundary += 1
        return self._safe_grapheme_boundary(text, boundary)

    def _idle_boundary(self) -> int | None:
        text = "".join(self._chars)
        if not text:
            return None
        if text[-1].isspace() or self._strong_boundary(text) is not None:
            return len(text)
        last_space = max(text.rfind(" "), text.rfind("\n"), text.rfind("\t"))
        return last_space + 1 if last_space >= 0 else None

    @staticmethod
    def _strong_boundary(text: str) -> int | None:
        for match in _STRONG_BOUNDARY.finditer(text):
            punctuation_at = match.start()
            if text[punctuation_at] == "." and LiveTextSegmenter._period_is_nonterminal(
                text, punctuation_at
            ):
                continue
            return match.end()
        return None

    @staticmethod
    def _period_is_nonterminal(text: str, punctuation_at: int) -> bool:
        before = text[:punctuation_at]
        line = before.rsplit("\n", 1)[-1]
        if re.fullmatch(r"\s*\d+", line):
            return True
        word_match = re.search(r"([A-Za-z]+)$", before)
        if not word_match:
            return False
        word = word_match.group(1).lower()
        return len(word) == 1 or word in _NONTERMINAL_ABBREVIATIONS

    @staticmethod
    def _safe_grapheme_boundary(text: str, boundary: int) -> int:
        """Avoid splitting common combining-mark and emoji-ZWJ clusters."""
        while boundary < len(text):
            char = text[boundary]
            is_extend = (
                bool(unicodedata.combining(char))
                or unicodedata.category(char) in {"Mc", "Me", "Mn"}
                or char == "\ufe0f"
                or "\U0001f3fb" <= char <= "\U0001f3ff"
            )
            if is_extend:
                boundary += 1
                continue
            if char == "\u200d" and boundary + 1 < len(text):
                boundary += 2
                continue
            break
        return boundary

    def _consume_segment(self, boundary: int) -> TextSegment:
        raw_text = "".join(self._chars[:boundary])
        positions = self._positions[:boundary]
        del self._chars[:boundary]
        del self._positions[:boundary]
        source_start = min((position[0] for position in positions), default=0)
        source_end = max((position[1] for position in positions), default=source_start)
        return TextSegment(
            text=clean_markdown(raw_text),
            source_start=source_start,
            source_end=source_end,
        )
