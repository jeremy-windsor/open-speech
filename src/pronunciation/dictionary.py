from __future__ import annotations

import hashlib
import html
import json
import logging
import re
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def _json_entries(text: str, path: str) -> dict:
    """A JSON dictionary's entries, each value the text written, as in YAML (keys are always strings in JSON).

    Typed, 0.50 was spoken "0.5", 1e3 "1000.0", true "True" and null "None". A null value has no text to speak, so
    its entry is left out with a warning: "" is how an entry removes a word, and a null should not silently do that
    to every text the server speaks.
    """
    data = json.loads(text, parse_float=str, parse_int=str, parse_constant=str) or {}
    entries = {}
    for key, value in data.items():
        if value is None:
            logger.warning(
                'Pronunciation dictionary %s: %r is null, so it is left out; write "" to remove the word', path, key
            )
            continue
        entries[key] = "true" if value is True else "false" if value is False else value
    return entries


def _joined_pairs(text: str) -> str:
    """``text`` with each UTF-16 surrogate pair made the one character it stands for.

    PyYAML reads each backslash-u escape as one code point, so the pair json.dumps writes for an emoji became two lone
    surrogates; JSON reads the pair as one character (RFC 8259 section 7), and so does this. A surrogate without its
    other half is kept as it is.
    """
    return text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "surrogatepass")


class PronunciationDictionary:
    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self._entries: dict[str, str] = {}
        if path:
            self.load(path)

    def load(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            self._entries = {}
            return
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() in {".yaml", ".yml"}:
            # Keys and values are text as written, as in JSON. YAML 1.1 typing read NO as False, 12:30 as 750
            # and an empty value as None. BaseLoader resolves no implicit types and builds only strings, lists
            # and dicts, so no tag constructs an object either.
            data = yaml.load(text, Loader=yaml.BaseLoader) or {}
            if isinstance(data, dict):
                data = {_joined_pairs(str(k)): _joined_pairs(str(v)) for k, v in data.items()}
        else:
            data = _json_entries(text, path)
        self._entries = {str(k): str(v) for k, v in (data or {}).items()}

    @property
    def entries(self) -> dict[str, str]:
        return dict(self._entries)

    def _in_order(self) -> list[tuple[str, str]]:
        # Longest first; entries of one length keep the file's order (sorted() is stable).
        return sorted(self._entries.items(), key=lambda kv: len(kv[0]), reverse=True)

    def apply(self, text: str) -> str:
        out = text
        for src, dst in self._in_order():
            # Both sides are literal text: a value is spoken, never read as an re.sub template.
            out = out.replace(src, dst)
        return out

    def fingerprint(self) -> str:
        """A key for what apply() does: "" without entries, else a hash of them in the order they apply.

        The Reader puts it in its part ids, so parts cached under another dictionary are spoken again.
        """
        if not self._entries:
            return ""
        return hashlib.sha256(json.dumps(self._in_order()).encode("utf-8")).hexdigest()


# One pause is capped at ten seconds, as Amazon Polly does. Uncapped, a few bytes of
# markup would expand into unbounded text for the backend.
MAX_SSML_BREAK_MS = 10_000
# SSML 1.1 section 3.2.3 leaves each strength's pause to the processor; these are Azure's.
# A break with neither time nor strength is a medium one.
_SSML_BREAK_STRENGTH_MS = {
    "none": 0, "x-weak": 250, "weak": 500, "medium": 750, "strong": 1000, "x-strong": 1250,
}
# Markup that sits inside a sentence. Any other tag (break, p, s, speak, voice, ...) also
# separates the words on either side of it, so it is replaced with a space.
_SSML_INLINE_ELEMENTS = frozenset({"emphasis", "mark", "phoneme", "prosody", "say-as", "sub"})

# parse_ssml runs on request text, so every pattern here is linear-time (no two
# quantifiers can share a character). A tag cannot contain "<", so a failed scan from
# one "<" ends at the next.
# Comments and CDATA sections may contain "<" (XML 1.0 sections 2.5 and 2.7), so they go
# first. Each match runs to its terminator or the end of the text, so it always consumes.
_SSML_COMMENT_OR_CDATA = re.compile(
    r"<!--.*?(?:-->|\Z)|<!\[CDATA\[(.*?)(?:\]\]>|\Z)", re.DOTALL
)
_SSML_TAG = re.compile(r"<([^<>]*)>")
_SSML_TAG_NAME = re.compile(r"/?\s*([\w:.-]*)")
_SSML_ATTRIBUTE = re.compile(r"(?<![\w:.-])([\w:.-]+)\s*=\s*(?:\"([^\"]*)\"|'([^']*)')")
_SSML_TIME = re.compile(r"(\d+(?:\.\d+)?|\.\d+)(ms|s)")


def _ssml_pause(ms: float) -> str:
    if ms <= 0:
        return " "
    return " " + ("." * max(1, int(min(ms, MAX_SSML_BREAK_MS)) // 250)) + " "


def _ssml_break_ms(tag_body: str) -> float:
    # XML attribute order is not significant (XML 1.0 section 3.1).
    attributes = {
        match.group(1).lower(): match.group(2) if match.group(2) is not None else match.group(3)
        for match in _SSML_ATTRIBUTE.finditer(tag_body)
    }
    time_match = _SSML_TIME.fullmatch(attributes.get("time", "").strip())
    if time_match:
        return float(time_match.group(1)) * (1000 if time_match.group(2) == "s" else 1)
    strength = attributes.get("strength", "medium").strip().lower()
    return _SSML_BREAK_STRENGTH_MS.get(strength, _SSML_BREAK_STRENGTH_MS["medium"])


def _ssml_tag_text(match: re.Match) -> str:
    body = match.group(1)
    name = _SSML_TAG_NAME.match(body).group(1).rsplit(":", 1)[-1].lower()
    if not name:  # a comment, processing instruction or declaration
        return ""
    if name == "break" and not body.startswith("/"):
        # <break .../> and <break ...></break> are the same empty element.
        return _ssml_pause(_ssml_break_ms(body))
    return "" if name in _SSML_INLINE_ELEMENTS else " "


def _ssml_comment_or_cdata_text(match: re.Match) -> str:
    cdata = match.group(1)
    if cdata is None:  # a comment, markup inside it included, is not spoken
        return ""
    # CDATA is literal text; escape it so tag removal and the unescape below leave it as is.
    return html.escape(cdata, quote=False)


def parse_ssml(input_text: str) -> str:
    # basic subset handling without strict XML parser fragility
    text = _SSML_COMMENT_OR_CDATA.sub(_ssml_comment_or_cdata_text, input_text)
    text = _SSML_TAG.sub(_ssml_tag_text, text)
    # SSML is XML, so a literal "&" or "<" arrives escaped; decode only after tags are gone.
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()
