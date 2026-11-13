"""AI presets: named sets of everything you'd tune about how Open Speech asks the AI.

Three layers, with the same field names everywhere: the ``.env`` defaults (``OS_ASSISTANT_*``) and the code's
defaults, then a preset, then one request's overrides (a realtime ``session.update``, a book job's body). A field
left empty (``None``, or an empty text) falls back to the layer below. ``effective()`` merges and validates them.
Settings' Test is the one exception: it sends the preset as it is on screen, so there an empty field is emptied
(``effective(..., unsaved=True)``).

Two presets are built in: ``voice`` (the default for ``/v1/realtime``) and ``book`` (the default for book jobs).
You can edit both and add your own, up to ``MAX_PRESETS`` in all. They are kept in ``assistant-presets.json`` next
to ``OS_STUDIO_DB_PATH``. The file is written only when a preset is saved, atomically, and holds only what was
changed: a built-in you haven't changed isn't in it. A missing or unreadable file means only the built-ins.

Each preset also describes itself as an action for "send anything to the AI" (``POST /api/assistant/ask``): ``label``
(the action's name, such as "Summarize"), ``inputs`` (the input kinds it suits: ``text``, ``transcript``, ``document``)
and ``output`` (its default output: ``text``, ``speech`` or ``document``). They describe the preset to the web app and
never reach the AI, so they aren't in ``EffectiveSettings`` and no request can override them. A save that leaves one out
keeps its saved value; ``null`` or an empty value is the default again (``ACTION_DEFAULTS`` for a built-in; the name,
every input and ``text`` for yours).

The connection settings (the URL, the API key, the extra body, the TLS setting, the token field's name and the
template switch) are never part of a preset: no preset can change them. They come from ``.env`` and Settings ›
AI assistant (``src/assistant/connection.py``), whose model is also the default ``model`` here.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import re
import tempfile
import threading
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

from src import config as config_module
from src.assistant.errors import ERROR_CODES, ERROR_SENTENCES, sentence_for
from src.assistant.prompts import PLACEHOLDERS, placeholders_in
from src.config import (
    ASSISTANT_LIMITS,
    ASSISTANT_PROMPT_MAX_CHARS,
    ASSISTANT_REASONING_EFFORTS,
    ASSISTANT_THINKING_MODES,
)

logger = logging.getLogger(__name__)

FILE_NAME = "assistant-presets.json"
BUILTIN_IDS = ("voice", "book")
BUILTIN_NAMES = {"voice": "Voice", "book": "Book"}
MAX_PRESETS = 50
NAME_MAX_CHARS = 80
ID_MAX_CHARS = 40
MODEL_MAX_CHARS = 200
FILLERS_MAX_COUNT = 10
FILLER_MAX_CHARS = 80
ERROR_SENTENCE_MAX_CHARS = 300

VOICE_SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Your replies are read aloud. Answer in short, plain spoken sentences. "
    "Don't use Markdown, lists, tables, code, links or emoji."
)
BOOK_SYSTEM_PROMPT = (
    "You rework book text for listening, following the user's instructions. Reply only with plain prose "
    "paragraphs, because your reply is read aloud. Don't use Markdown, lists or headings, and don't comment on "
    "what you did."
)
DEFAULT_USER_TEMPLATE = "{text}"
DEFAULT_FILLERS = ("Hmm.", "OK, just a second.", "Let me think about that.")
DEFAULT_FILLER_AFTER_MS = 1200
DEFAULT_FILLER_EVERY_MS = 4000
DEFAULT_FILLER_MAX = 3

# The fields a preset or an override may set, in the order they're shown. ``thinking_budget`` is sent as
# ``reasoning_budget_tokens``, the llama.cpp-style name; 0 sends none.
TUNABLE_FIELDS = (
    "model", "system_prompt", "user_template", "thinking", "reasoning_effort", "thinking_budget", "show_thinking",
    "max_tokens", "temperature", "top_p", "timeout_s", "history_turns", "fillers", "filler_after_ms",
    "filler_every_ms", "filler_max", "speak_errors", "error_sentences", "chunk_chars",
)
# What a preset says of itself as an action: shown, never sent to the AI.
ACTION_FIELDS = ("label", "inputs", "output")
INPUT_KINDS = ("text", "transcript", "document")
OUTPUT_KINDS = ("text", "speech", "document")
ACTION_DEFAULTS = MappingProxyType({
    "voice": MappingProxyType({"label": "Answer", "inputs": ("text", "transcript"), "output": "speech"}),
    "book": MappingProxyType({"label": "Rewrite for listening", "inputs": ("document",), "output": "document"}),
})
LABEL_MAX_CHARS = NAME_MAX_CHARS
EDITABLE_FIELDS = ("name", *ACTION_FIELDS, *TUNABLE_FIELDS)
# A preset's own: a request's overrides can't set them.
OWN_FIELDS = ("name", *ACTION_FIELDS)
# What a preset's API answer adds; ignored when a body sends them back.
READ_ONLY_FIELDS = ("id", "builtin", "customized", "inherited")
PURPOSES = ("voice", "book")
_ID = re.compile(rf"[a-z0-9](?:[a-z0-9-]{{0,{ID_MAX_CHARS - 2}}}[a-z0-9])?")


class PresetError(ValueError):
    """A field is invalid (``field`` names it: a preset field, ``preset``, ``overrides`` or ``body``)."""

    def __init__(self, field_name: str, message: str) -> None:
        super().__init__(message)
        self.field = field_name
        self.message = message


class PresetNotFoundError(LookupError):
    """No preset has this id."""


class PresetConflictError(Exception):
    """``code`` is ``preset_exists`` (the id is taken) or ``too_many_presets`` (``MAX_PRESETS`` reached)."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


@dataclass(frozen=True)
class EffectiveSettings:
    """One preset with its three layers merged: what a voice turn or a book job uses. A snapshot: later
    changes to the preset or ``.env`` don't change it. It holds no server-only setting."""

    preset_id: str = "voice"
    name: str = "Voice"
    model: str = ""
    system_prompt: str = ""
    user_template: str = DEFAULT_USER_TEMPLATE
    thinking: str = "default"
    reasoning_effort: str = ""
    thinking_budget: int = 0
    show_thinking: bool = True
    max_tokens: int = 0
    temperature: float | None = None
    top_p: float | None = None
    timeout_s: float = 120.0
    history_turns: int = 20
    fillers: tuple[str, ...] = ()
    filler_after_ms: int = DEFAULT_FILLER_AFTER_MS
    filler_every_ms: int = DEFAULT_FILLER_EVERY_MS
    filler_max: int = DEFAULT_FILLER_MAX
    speak_errors: bool = True
    error_sentences: Mapping[str, str] = field(default_factory=dict)
    chunk_chars: int = 12000

    def error_sentence(self, code: str) -> str:
        """The sentence to speak for an AssistantError's code: the preset's own, else the default."""
        return sentence_for(code, dict(self.error_sentences))

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON values, for an API answer or a job record."""
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        values["fillers"] = list(self.fillers)
        values["error_sentences"] = dict(self.error_sentences)
        return values


# ── Validation ───────────────────────────────────────────────────────────────


def has_lone_surrogate(text: str) -> bool:
    """Whether ``text`` holds a UTF-16 surrogate code point: half of a pair, such as half an emoji (the JSON escape
    ``"\\ud83d"``, which a JavaScript ``slice`` can leave). UTF-8 can't encode one (RFC 3629, section 3), so such text
    can be neither stored nor sent to the AI server."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return True
    return False


def lone_surrogate_message(where: str) -> str:
    return (f"{where} has a lone surrogate (half of a UTF-16 pair, such as half an emoji), which can't be stored or "
            "sent to the AI; send whole characters")


def _text(name: str, value: Any, max_chars: int) -> str:
    if not isinstance(value, str):
        raise PresetError(name, f"{name} must be text")
    if len(value) > max_chars:
        raise PresetError(name, f"{name} must be at most {max_chars} characters")
    if has_lone_surrogate(value):
        raise PresetError(name, lone_surrogate_message(name))
    return value


def _number(name: str, value: Any, *, integer: bool) -> int | float:
    low, high = ASSISTANT_LIMITS[name]
    kind = "a whole number" if integer else "a number"
    # Whole numbers in full (1000000, never 1e+06: the thinking budget's bound).
    shown = (f"{int(low)}", f"{int(high)}") if integer else (f"{low:g}", f"{high:g}")
    problem = PresetError(name, f"{name} must be {kind} from {shown[0]} to {shown[1]}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise problem
    if isinstance(value, float) and (not math.isfinite(value) or (integer and not value.is_integer())):
        raise problem
    if not low <= value <= high:
        raise problem
    return int(value) if integer else float(value)


def _choice(name: str, value: Any, allowed: tuple[str, ...]) -> str | None:
    if value == "":
        return None
    if not isinstance(value, str) or value not in allowed:
        shown = ", ".join(repr(choice) for choice in allowed if choice)
        raise PresetError(name, f"{name} must be one of {shown}")
    return value


def allowed_placeholders(preset_id: str | None) -> tuple[str, ...]:
    """The built-in voice preset takes only {text}; the others also {title}, {author} and {chapter}."""
    return ("text",) if preset_id == "voice" else PLACEHOLDERS


def _user_template(value: Any, preset_id: str | None) -> str | None:
    template = _text("user_template", value, ASSISTANT_PROMPT_MAX_CHARS)
    if not template.strip():
        return None
    names = placeholders_in(template)
    if "text" not in names:
        raise PresetError("user_template", "user_template must contain {text}")
    allowed = allowed_placeholders(preset_id)
    unknown = [name for name in names if name not in allowed]
    if unknown:
        shown = ", ".join("{" + name + "}" for name in allowed)
        raise PresetError("user_template", f"user_template has the unknown placeholder {{{unknown[0]}}}; it may use {shown}")
    return template


def _fillers(value: Any) -> list[str]:
    if not isinstance(value, list) or len(value) > FILLERS_MAX_COUNT:
        raise PresetError("fillers", f"fillers must be a list of up to {FILLERS_MAX_COUNT} phrases")
    phrases = []
    for index, phrase in enumerate(value):
        if not isinstance(phrase, str) or not 1 <= len(phrase.strip()) <= FILLER_MAX_CHARS:
            raise PresetError("fillers", f"fillers[{index}] must be a phrase of 1 to {FILLER_MAX_CHARS} characters")
        if has_lone_surrogate(phrase):
            raise PresetError("fillers", lone_surrogate_message(f"fillers[{index}]"))
        phrases.append(phrase.strip())
    return phrases


def _error_sentences(value: Any) -> dict[str, str] | None:
    if not isinstance(value, Mapping):
        raise PresetError("error_sentences", "error_sentences must map an error code to its sentence")
    sentences = {}
    for code, sentence in value.items():
        if code not in ERROR_CODES:
            raise PresetError("error_sentences", f"error_sentences has the unknown error code {code!r}; the codes are {', '.join(ERROR_CODES)}")
        if not isinstance(sentence, str) or len(sentence.strip()) > ERROR_SENTENCE_MAX_CHARS:
            raise PresetError("error_sentences", f"error_sentences.{code} must be text of up to {ERROR_SENTENCE_MAX_CHARS} characters")
        if has_lone_surrogate(sentence):
            raise PresetError("error_sentences", lone_surrogate_message(f"error_sentences.{code}"))
        if sentence.strip():  # An empty sentence keeps the default.
            sentences[code] = sentence.strip()
    return sentences or None


def _bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise PresetError(name, f"{name} must be true or false")
    return value


def _inputs(value: Any) -> list[str] | None:
    """The input kinds, once each, in INPUT_KINDS' order; an empty list is the default."""
    shown = ", ".join(repr(kind) for kind in INPUT_KINDS)
    if not isinstance(value, list):
        raise PresetError("inputs", f"inputs must be a list of the input kinds {shown}")
    for index, kind in enumerate(value):
        if not isinstance(kind, str) or kind not in INPUT_KINDS:
            raise PresetError("inputs", f"inputs[{index}] must be one of {shown}")
    return [kind for kind in INPUT_KINDS if kind in value] or None


def _validate_field(name: str, value: Any, preset_id: str | None) -> Any:
    """The field's normalised value, or None when it is empty (the layer below applies)."""
    if name == "name":
        text = _text(name, value, NAME_MAX_CHARS).strip()
        if not text:
            raise PresetError(name, f"name must be 1 to {NAME_MAX_CHARS} characters")
        return text
    if name == "label":
        return _text(name, value, LABEL_MAX_CHARS).strip() or None
    if name == "inputs":
        return _inputs(value)
    if name == "output":
        return _choice(name, value, OUTPUT_KINDS)
    if name == "model":
        return _text(name, value, MODEL_MAX_CHARS).strip() or None
    if name == "system_prompt":
        text = _text(name, value, ASSISTANT_PROMPT_MAX_CHARS)
        return text if text.strip() else None
    if name == "user_template":
        return _user_template(value, preset_id)
    if name == "thinking":
        return _choice(name, value, ASSISTANT_THINKING_MODES)
    if name == "reasoning_effort":
        return _choice(name, value, ASSISTANT_REASONING_EFFORTS)
    if name in ("show_thinking", "speak_errors"):
        return _bool(name, value)
    if name in ("temperature", "top_p", "timeout_s"):
        return _number(name, value, integer=False)
    if name in ("thinking_budget", "max_tokens", "history_turns", "filler_after_ms", "filler_every_ms", "filler_max",
                "chunk_chars"):
        return _number(name, value, integer=True)
    if name == "fillers":
        return _fillers(value)
    if name == "error_sentences":
        return _error_sentences(value)
    raise PresetError(name, f"{name} is not a preset field")


def validate_fields(data: Any, *, preset_id: str | None, overrides: bool = False) -> dict[str, Any]:
    """The set fields of a preset body (or, with ``overrides``, of one request's overrides), normalised.

    A body may send back the read-only fields of an API answer (``id``, ``builtin``, ``customized``,
    ``inherited``); they are ignored. Overrides take only the tunable fields: the name and the action fields are a
    preset's own.
    """
    if not isinstance(data, Mapping):
        where = "overrides" if overrides else "body"
        raise PresetError(where, f"{where} must be a JSON object")
    valid: dict[str, Any] = {}
    for name, value in data.items():
        if not overrides and name in READ_ONLY_FIELDS:
            continue
        if overrides and name in OWN_FIELDS:
            raise PresetError(name, f"{name} is a preset's own; a request can't override it")
        if name not in EDITABLE_FIELDS:
            raise PresetError(str(name), f"{name} is not a preset field")
        if value is None:
            continue
        normalised = _validate_field(name, value, preset_id)
        if normalised is not None:
            valid[name] = normalised
    return valid


def slug(name: str) -> str:
    """A preset's id from its name: lower-case letters and digits joined by hyphens."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", "-", ascii_name).strip("-")
    return text[:ID_MAX_CHARS].strip("-")


# ── The three layers ─────────────────────────────────────────────────────────


def _defaults(purpose: str, job: bool = False) -> dict[str, Any]:
    """The lowest layers: OS_ASSISTANT_* and the code's defaults, for a voice turn or a book job. The default
    model is the connection's (OS_ASSISTANT_MODEL, or the one saved in Settings). A job of any purpose falls back
    to the job timeout."""
    from src.assistant import connection  # it imports this module

    s = config_module.settings
    if purpose == "book":
        system_prompt = BOOK_SYSTEM_PROMPT
        timeout_s = s.os_assistant_job_timeout_s
    else:
        system_prompt = s.os_assistant_system_prompt if s.os_assistant_system_prompt.strip() else VOICE_SYSTEM_PROMPT
        timeout_s = s.os_assistant_job_timeout_s if job else s.os_assistant_timeout_s
    return {
        "model": connection.effective(s).model,
        "system_prompt": system_prompt,
        "user_template": DEFAULT_USER_TEMPLATE,
        "thinking": s.os_assistant_thinking,
        "reasoning_effort": s.os_assistant_reasoning_effort,
        "thinking_budget": s.os_assistant_thinking_budget,
        "show_thinking": s.os_assistant_show_thinking,
        "max_tokens": s.os_assistant_max_tokens,
        "temperature": s.os_assistant_temperature,
        "top_p": s.os_assistant_top_p,
        "timeout_s": float(timeout_s),
        "history_turns": s.os_assistant_history_turns,
        "fillers": list(DEFAULT_FILLERS),
        "filler_after_ms": DEFAULT_FILLER_AFTER_MS,
        "filler_every_ms": DEFAULT_FILLER_EVERY_MS,
        "filler_max": DEFAULT_FILLER_MAX,
        "speak_errors": s.os_assistant_speak_errors,
        "error_sentences": dict(ERROR_SENTENCES),
        "chunk_chars": s.os_assistant_chunk_chars,
    }


def _resolve(
    preset_id: str, stored: Mapping[str, Any], overrides: Mapping[str, Any], purpose: str | None, job: bool = False,
) -> EffectiveSettings:
    purpose = purpose or ("book" if preset_id == "book" else "voice")
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {', '.join(PURPOSES)}")
    merged = _defaults(purpose, job)
    for layer in (stored, overrides):
        for name, value in layer.items():
            if name in OWN_FIELDS:
                continue
            # Error sentences merge per code: a code a layer leaves out keeps the one below.
            merged[name] = {**merged[name], **value} if name == "error_sentences" else value
    merged["fillers"] = tuple(merged["fillers"])
    merged["error_sentences"] = MappingProxyType(dict(merged["error_sentences"]))
    name = stored.get("name") or BUILTIN_NAMES.get(preset_id, preset_id)
    return EffectiveSettings(preset_id=preset_id, name=name, **merged)


def _action_defaults(preset_id: str, name: str) -> dict[str, Any]:
    """A preset's label, inputs and output when it sets none: the built-in's, or its name, every input and text."""
    builtin = ACTION_DEFAULTS.get(preset_id)
    if builtin is not None:
        return {"label": builtin["label"], "inputs": list(builtin["inputs"]), "output": builtin["output"]}
    return {"label": name, "inputs": list(INPUT_KINDS), "output": "text"}


def _without_default_actions(preset_id: str, stored: dict[str, Any]) -> dict[str, Any]:
    """``stored`` without an action field equal to its default, so it stays a default (as a built-in's own name is
    never stored)."""
    name = stored.get("name") or BUILTIN_NAMES.get(preset_id, preset_id)
    defaults = _action_defaults(preset_id, name)
    return {key: value for key, value in stored.items() if key not in ACTION_FIELDS or value != defaults[key]}


def _view(preset_id: str, stored: Mapping[str, Any]) -> dict[str, Any]:
    values = _resolve(preset_id, stored, {}, None).to_dict()
    del values["preset_id"]
    name = values.pop("name")
    action = _action_defaults(preset_id, name)
    action.update({key: list(stored[key]) if key == "inputs" else stored[key] for key in ACTION_FIELDS if key in stored})
    return {
        "id": preset_id,
        "name": name,
        **action,
        "builtin": preset_id in BUILTIN_IDS,
        "customized": bool(stored) or preset_id not in BUILTIN_IDS,
        "inherited": [name for name in TUNABLE_FIELDS if name not in stored],
        **values,
    }


# ── The file store ───────────────────────────────────────────────────────────


def presets_path() -> Path:
    return Path(config_module.settings.os_studio_db_path).parent / FILE_NAME


class _Store:
    """The preset file, read again only when it changes on disk, and written under one lock."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self._key: tuple[str, int, int] | None = None
        self._presets: dict[str, dict[str, Any]] = {}

    def read(self, path: Path) -> dict[str, dict[str, Any]]:
        """id -> the preset's stored fields, built-ins first; a copy the caller may change."""
        try:
            stat = path.stat()
        except FileNotFoundError:
            return {}
        except OSError as exc:
            logger.warning("Can't read the AI preset file %s (%s); using the built-in presets", path, exc)
            return {}
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        if key != self._key:
            self._presets = self._parse(path)
            self._key = key
        return copy.deepcopy(self._presets)

    def write(self, path: Path, presets: Mapping[str, Mapping[str, Any]]) -> None:
        ordered = [pid for pid in BUILTIN_IDS if pid in presets] + [pid for pid in presets if pid not in BUILTIN_IDS]
        payload = json.dumps(
            {"version": 1, "presets": [{"id": pid, **presets[pid]} for pid in ordered]}, ensure_ascii=False, indent=2,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".assistant-presets-", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        stat = path.stat()
        self._key = (str(path), stat.st_mtime_ns, stat.st_size)
        self._presets = {pid: dict(presets[pid]) for pid in ordered}

    @staticmethod
    def _parse(path: Path) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            logger.warning("Ignoring the unreadable AI preset file %s (%s); using the built-in presets", path, exc)
            return {}
        entries = data.get("presets") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            logger.warning("Ignoring the AI preset file %s: it has no list of presets; using the built-in presets", path)
            return {}
        presets: dict[str, dict[str, Any]] = {}
        for index, entry in enumerate(entries):
            preset_id = entry.get("id") if isinstance(entry, dict) else None
            if not isinstance(preset_id, str) or not _ID.fullmatch(preset_id) or preset_id in presets:
                logger.warning("Skipping preset %d in %s: it has no valid, unique id", index, path)
                continue
            try:
                stored = validate_fields({k: v for k, v in entry.items() if k != "id"}, preset_id=preset_id)
            except PresetError as exc:
                logger.warning("Skipping AI preset %r in %s: %s", preset_id, path, exc.message)
                continue
            if preset_id not in BUILTIN_IDS:
                customs = sum(1 for pid in presets if pid not in BUILTIN_IDS)
                if "name" not in stored or customs >= MAX_PRESETS - len(BUILTIN_IDS):
                    logger.warning("Skipping AI preset %r in %s: it has no name, or the limit is reached", preset_id, path)
                    continue
            presets[preset_id] = stored
        return presets


_store = _Store()


def _stored_presets() -> dict[str, dict[str, Any]]:
    with _store.lock:
        return _store.read(presets_path())


def list_presets() -> list[dict[str, Any]]:
    """Every preset with its effective values, built-ins first, plus ``builtin``, ``customized`` and
    ``inherited`` (the fields that come from ``.env`` or the code)."""
    stored = _stored_presets()
    customs = [pid for pid in stored if pid not in BUILTIN_IDS]
    return [_view(pid, stored.get(pid, {})) for pid in (*BUILTIN_IDS, *customs)]


def get_preset(preset_id: str) -> dict[str, Any] | None:
    stored = _stored_presets()
    if preset_id not in BUILTIN_IDS and preset_id not in stored:
        return None
    return _view(preset_id, stored.get(preset_id, {}))


def create_preset(body: Any) -> dict[str, Any]:
    """Save a new preset; its id is a slug of its name. Raises PresetError or PresetConflictError."""
    stored_fields = validate_fields(body, preset_id=None)
    if "name" not in stored_fields:
        raise PresetError("name", "name is required")
    preset_id = slug(stored_fields["name"])
    if not preset_id:
        raise PresetError("name", "name needs a letter or a digit, for the preset's id")
    path = presets_path()
    with _store.lock:
        presets = _store.read(path)
        if preset_id in BUILTIN_IDS or preset_id in presets:
            raise PresetConflictError(f"A preset with the id {preset_id!r} already exists", "preset_exists")
        if len(BUILTIN_IDS) + sum(1 for pid in presets if pid not in BUILTIN_IDS) >= MAX_PRESETS:
            raise PresetConflictError(f"There are already {MAX_PRESETS} presets", "too_many_presets")
        stored_fields = _without_default_actions(preset_id, stored_fields)
        presets[preset_id] = stored_fields
        _store.write(path, presets)
        return _view(preset_id, stored_fields)


def replace_preset(preset_id: str, body: Any) -> dict[str, Any]:
    """Replace a preset's editable fields: a field left out or empty comes from ``.env`` or the code. The action
    fields (``label``, ``inputs``, ``output``) are the exception: one left out keeps its saved value, because
    Settings saves only the name and the tunable fields; ``null`` or an empty value is its default again."""
    path = presets_path()
    with _store.lock:
        presets = _store.read(path)
        if preset_id not in BUILTIN_IDS and preset_id not in presets:
            raise PresetNotFoundError(preset_id)
        stored_fields = validate_fields(body, preset_id=preset_id)
        saved = presets.get(preset_id, {})
        sent = body if isinstance(body, Mapping) else {}
        stored_fields.update({key: saved[key] for key in ACTION_FIELDS if key in saved and key not in sent})
        if preset_id in BUILTIN_IDS:
            if stored_fields.get("name") == BUILTIN_NAMES[preset_id]:
                del stored_fields["name"]
        elif "name" not in stored_fields:
            raise PresetError("name", "name is required")
        stored_fields = _without_default_actions(preset_id, stored_fields)
        if stored_fields:
            presets[preset_id] = stored_fields
        else:
            presets.pop(preset_id, None)
        _store.write(path, presets)
        return _view(preset_id, stored_fields)


def delete_preset(preset_id: str) -> dict[str, Any] | None:
    """Delete a preset you made (returns None), or restore a built-in's defaults (returns it)."""
    path = presets_path()
    with _store.lock:
        presets = _store.read(path)
        if preset_id in BUILTIN_IDS:
            if preset_id in presets:
                del presets[preset_id]
                _store.write(path, presets)
            return _view(preset_id, {})
        if preset_id not in presets:
            raise PresetNotFoundError(preset_id)
        del presets[preset_id]
        _store.write(path, presets)
        return None


def effective(
    preset_id: str | None = None, overrides: Any = None, *, purpose: str | None = None, unsaved: bool = False,
    job: bool = False,
) -> EffectiveSettings:
    """The preset's settings with ``.env``, the preset and ``overrides`` merged and validated.

    ``preset_id`` None is ``voice``. ``purpose`` (``voice`` or ``book``) picks the fallbacks of a field the
    preset leaves empty, such as ``timeout_s`` (``OS_ASSISTANT_TIMEOUT_S`` or ``OS_ASSISTANT_JOB_TIMEOUT_S``) and
    the system prompt; None means ``book`` for the book preset and ``voice`` for the others. With ``job`` (a long
    input of ``POST /api/assistant/ask``, which runs as a job) the purpose keeps its system prompt but ``timeout_s``
    falls back to ``OS_ASSISTANT_JOB_TIMEOUT_S``. Raises PresetError with ``field`` ``preset`` for an unknown preset,
    or the override's field name.

    With ``unsaved`` (Settings' Test: the preset as it is on screen), ``overrides`` are the preset's own fields,
    edited but not saved. Each field sent replaces the saved one, as a save would: one sent empty (``None``, or a
    value its check empties, such as ``""``) falls to ``.env`` or the code. A field left out keeps the saved value.
    Without it, an empty override is skipped and the preset's value stays.
    """
    preset_id = "voice" if preset_id is None else preset_id
    if not isinstance(preset_id, str):
        raise PresetError("preset", "preset must be a preset id")
    stored = _stored_presets()
    if preset_id not in BUILTIN_IDS and preset_id not in stored:
        raise PresetError("preset", f"Unknown AI preset {preset_id!r}")
    sent = {} if overrides is None else overrides
    valid_overrides = validate_fields(sent, preset_id=preset_id, overrides=True)
    saved = stored.get(preset_id, {})
    if unsaved:
        saved = {name: value for name, value in saved.items() if name not in sent} | valid_overrides
        valid_overrides = {}
    return _resolve(preset_id, saved, valid_overrides, purpose, job)
