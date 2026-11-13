"""The AI server connection: where Open Speech sends its requests, and how.

The fields are ``url``, ``api_key``, ``model`` (the presets' default model), ``verify_tls``, ``template_kwargs``,
``max_tokens_field``, ``extra_body`` and ``stream`` (whether voice turns stream their replies). Each starts from
``.env`` (``OS_ASSISTANT_*``), and Settings › AI assistant can save its own value through ``/api/assistant/connection``.
A saved field overrides its ``.env`` value; a field that isn't saved keeps following ``.env``. ``effective()`` merges
the two layers, and it is what the shared client, ``Settings.assistant_enabled`` and the presets' default model use, so
a URL saved here turns the assistant on with nothing in ``.env``.

The saved fields are kept in ``assistant-connection.json``, next to ``assistant-presets.json``. The file is written
only on save, atomically and under a lock. A missing file means ``.env`` only, and so does an unreadable one, with a
warning; a field in it that isn't valid is skipped, with a warning.

``OS_ASSISTANT_CONNECTION_LOCKED=true`` makes the connection ``.env``-only again: the file is ignored, and saving,
forgetting or checking another connection raise ``ConnectionLockedError``.

The API key is write-only: no view, ``repr`` or log holds it. In a save, leaving it out keeps the stored key, ``""``
clears it and a value sets it. ``null`` for any field forgets its saved value, so it follows ``.env`` again.

The key stays with its server. It is sent only to the origin (scheme, host and port) of the URL it was set with, which
is stored beside it as ``api_key_origin``, and the ``.env`` key only to ``OS_ASSISTANT_URL``'s origin. A save that
leaves the key out keeps it for a URL of the same origin (a change of path) or for no URL, and clears it for another
origin. A check of another origin gets no stored key.

The key also stays with its certificate check. Whether the check was on when it was set is stored beside it as
``api_key_tls_checked``; the ``.env`` key counts as set with ``OS_ASSISTANT_VERIFY_TLS``. A key set with the check on
isn't sent to an https URL once the check is off. A save that turns the check off without giving the key clears it, and
a check with ``verify_tls: false`` gets no stored key. Turning the check back on keeps a key, and an http URL, which has
no certificate, isn't affected.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

from src import config as config_module
from src.assistant.presets import MODEL_MAX_CHARS, has_lone_surrogate, lone_surrogate_message
from src.config import (
    ASSISTANT_API_KEY_MAX_CHARS,
    ASSISTANT_MAX_TOKENS_FIELDS,
    check_assistant_api_key,
    check_assistant_url,
    parse_assistant_extra_body,
)

logger = logging.getLogger(__name__)

FILE_NAME = "assistant-connection.json"
FIELDS = ("url", "api_key", "model", "verify_tls", "template_kwargs", "max_tokens_field", "extra_body", "stream")
ENV_NAMES = {
    "url": "OS_ASSISTANT_URL",
    "api_key": "OS_ASSISTANT_API_KEY",
    "model": "OS_ASSISTANT_MODEL",
    "verify_tls": "OS_ASSISTANT_VERIFY_TLS",
    "template_kwargs": "OS_ASSISTANT_TEMPLATE_KWARGS",
    "max_tokens_field": "OS_ASSISTANT_MAX_TOKENS_FIELD",
    "extra_body": "OS_ASSISTANT_EXTRA_BODY",
    "stream": "OS_ASSISTANT_STREAM",
}
DEFAULTS: Mapping[str, Any] = MappingProxyType({
    "url": "", "api_key": "", "model": "", "verify_tls": True, "template_kwargs": False,
    "max_tokens_field": "max_tokens", "extra_body": MappingProxyType({}), "stream": True,
})
# Where a field's value comes from.
SOURCES = ("env", "saved", "default")
# What a view adds; ignored when a body sends them back.
READ_ONLY_FIELDS = ("enabled", "locked", "api_key_set", "api_key_tls_checked", "sources")
API_KEY_MAX_CHARS = ASSISTANT_API_KEY_MAX_CHARS
EXTRA_BODY_MAX_CHARS = 16000
# Stored beside a saved key: the origin it was set for, and whether the certificate check was on then. Not fields: no
# body sets them. A view shows only the second, read-only, so the Settings form can tell whether turning the check off
# clears the key.
KEY_ORIGIN = "api_key_origin"
KEY_TLS_CHECKED = "api_key_tls_checked"
_KEY_NAMES = ("api_key", KEY_ORIGIN, KEY_TLS_CHECKED)
_STORED_NAMES = tuple(n for name in FIELDS for n in (_KEY_NAMES if name == "api_key" else (name,)))
_DEFAULT_PORTS = {"http": 80, "https": 443}


def url_origin(url: str) -> str:
    """The URL's origin, ``scheme://host:port`` with the default port written out, or "" for no URL."""
    if not url:
        return ""
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    try:
        port = parts.port or _DEFAULT_PORTS.get(scheme, "")
    except ValueError:
        port = ""
    return f"{scheme}://[{host}]:{port}" if ":" in host else f"{scheme}://{host}:{port}"


class ConnectionFieldError(ValueError):
    """A field is invalid: ``field`` names it (a connection field, or ``body``)."""

    def __init__(self, field_name: str, message: str) -> None:
        super().__init__(message)
        self.field = field_name
        self.message = message


class ConnectionLockedError(PermissionError):
    """OS_ASSISTANT_CONNECTION_LOCKED is true: the connection is set only in ``.env``."""

    def __init__(self) -> None:
        super().__init__("The AI server connection is locked: it is set only in .env (OS_ASSISTANT_CONNECTION_LOCKED)")
        self.message = str(self)


@dataclass(frozen=True)
class Connection:
    """The effective connection: a snapshot of ``.env`` with the saved fields over it."""

    url: str = ""
    api_key: str = field(default="", repr=False)
    model: str = ""
    verify_tls: bool = True
    template_kwargs: bool = False
    max_tokens_field: str = "max_tokens"
    extra_body: Mapping[str, Any] = field(default_factory=dict)
    stream: bool = True
    sources: Mapping[str, str] = field(default_factory=dict)
    locked: bool = False
    # Whether the key in use was set with the certificate check on; None when no key is in use.
    api_key_tls_checked: bool | None = None

    @property
    def enabled(self) -> bool:
        """The assistant is on when the connection has a URL."""
        return bool(self.url)

    def view(self) -> dict[str, Any]:
        """The API answer: each field's value (the key only as ``api_key_set``, and ``api_key_tls_checked``: whether
        it was set with the certificate check on, null with no key), its source, and the lock."""
        return {
            "enabled": self.enabled,
            "locked": self.locked,
            "url": self.url,
            "model": self.model,
            "api_key_set": bool(self.api_key),
            "api_key_tls_checked": bool(self.api_key_tls_checked) if self.api_key else None,
            "verify_tls": self.verify_tls,
            "template_kwargs": self.template_kwargs,
            "max_tokens_field": self.max_tokens_field,
            "extra_body": copy.deepcopy(dict(self.extra_body)),
            "stream": self.stream,
            "sources": {name: self.sources.get(name, "default") for name in FIELDS},
        }


# ── Validation ───────────────────────────────────────────────────────────────


def _bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ConnectionFieldError(name, f"{name} must be true or false")
    return value


def _validate_field(name: str, value: Any) -> Any:
    """The field's normalised value. Raises ConnectionFieldError naming the field."""
    if name == "url":
        if not isinstance(value, str):
            raise ConnectionFieldError(name, "url must be text: the AI server's base URL, such as http://ai-server.lan:8080/v1")
        try:
            return check_assistant_url(value, "url")
        except ValueError as exc:
            raise ConnectionFieldError(name, str(exc)) from None
    if name == "api_key":
        if not isinstance(value, str):
            raise ConnectionFieldError(name, "api_key must be text")
        try:
            return check_assistant_api_key(value, "api_key")  # the rule OS_ASSISTANT_API_KEY has too
        except ValueError as exc:
            raise ConnectionFieldError(name, str(exc)) from None
    if name == "model":
        if not isinstance(value, str) or len(value.strip()) > MODEL_MAX_CHARS:
            raise ConnectionFieldError(name, f"model must be text of up to {MODEL_MAX_CHARS} characters")
        if has_lone_surrogate(value):
            raise ConnectionFieldError(name, lone_surrogate_message(name))
        return value.strip()
    if name in ("verify_tls", "template_kwargs", "stream"):
        return _bool(name, value)
    if name == "max_tokens_field":
        if value not in ASSISTANT_MAX_TOKENS_FIELDS:
            raise ConnectionFieldError(name, f"max_tokens_field must be one of {', '.join(ASSISTANT_MAX_TOKENS_FIELDS)}")
        return value
    if name == "extra_body":
        try:
            body = parse_assistant_extra_body(value, "extra_body")
        except ValueError as exc:
            raise ConnectionFieldError(name, str(exc)) from None
        if len(json.dumps(body)) > EXTRA_BODY_MAX_CHARS:
            raise ConnectionFieldError(name, f"extra_body must be at most {EXTRA_BODY_MAX_CHARS} characters of JSON")
        return body
    raise ConnectionFieldError(name, f"{name} is not a connection field; the fields are {', '.join(FIELDS)}")


def validate_fields(body: Any) -> dict[str, Any]:
    """A save's or a check's fields, normalised: name -> value, or None to forget the saved value.

    The read-only fields of a view (``enabled``, ``locked``, ``api_key_set``, ``api_key_tls_checked``, ``sources``)
    are ignored, so an answer can be sent back.
    """
    if not isinstance(body, Mapping):
        raise ConnectionFieldError("body", "body must be a JSON object")
    changes: dict[str, Any] = {}
    for name, value in body.items():
        if name in READ_ONLY_FIELDS:
            continue
        if name not in FIELDS:
            raise ConnectionFieldError(str(name), f"{name} is not a connection field; the fields are {', '.join(FIELDS)}")
        changes[name] = None if value is None else _validate_field(name, value)
    return changes


# ── The two layers ───────────────────────────────────────────────────────────


def _settings(settings):
    return config_module.settings if settings is None else settings


def _env(s) -> dict[str, Any]:
    """The .env layer. Startup validated it; a value set some other way that isn't valid counts as unset."""
    try:
        extra_body = s.assistant_extra_body
    except ValueError:
        logger.warning("Ignoring OS_ASSISTANT_EXTRA_BODY: it isn't a JSON object")
        extra_body = {}
    try:
        api_key = check_assistant_api_key(s.os_assistant_api_key or "")  # stripped, as a saved key is
    except ValueError:
        # Only the setting is named, never the key.
        logger.warning("Ignoring OS_ASSISTANT_API_KEY: it isn't printable ASCII of up to %d characters",
                       ASSISTANT_API_KEY_MAX_CHARS)
        api_key = ""
    return {
        "url": (s.os_assistant_url or "").strip(),
        "api_key": api_key,
        "model": s.os_assistant_model or "",
        "verify_tls": bool(s.os_assistant_verify_tls),
        "template_kwargs": bool(s.os_assistant_template_kwargs),
        "max_tokens_field": s.os_assistant_max_tokens_field,
        "extra_body": extra_body,
        "stream": bool(s.os_assistant_stream),
    }


def _env_url(s) -> str:
    return (s.os_assistant_url or "").strip()


def _env_verify_tls(s) -> bool:
    return bool(s.os_assistant_verify_tls)


def _unchecked_https(origin: str, verify_tls: bool) -> bool:
    """An https origin whose certificate isn't checked: whoever answers there gets what is sent."""
    return origin.startswith("https://") and not verify_tls


def _resolve(s, saved: Mapping[str, Any], locked: bool) -> Connection:
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for name, value in _env(s).items():
        if not locked and name in saved:
            values[name], sources[name] = saved[name], "saved"
        else:
            values[name] = value
            sources[name] = "default" if value == DEFAULTS[name] else "env"
    # The key goes only to the origin it was set for: a saved key to the origin saved beside it (a file from
    # before origins were kept: the URL it is used with), the .env key to OS_ASSISTANT_URL's origin.
    origin = url_origin(values["url"])
    key_origin = saved.get(KEY_ORIGIN, origin) if sources["api_key"] == "saved" else url_origin(_env_url(s))
    # And not to an https URL whose certificate check is off when it was set with the check on:
    # a saved key as saved beside it (a file from before: the check in effect now), the .env key as
    # OS_ASSISTANT_VERIFY_TLS has it.
    if sources["api_key"] == "saved":
        key_tls_checked = saved.get(KEY_TLS_CHECKED, values["verify_tls"])
    else:
        key_tls_checked = _env_verify_tls(s)
    downgraded = key_tls_checked and _unchecked_https(origin, values["verify_tls"])
    if values["api_key"] and (key_origin != origin or downgraded):
        values["api_key"], sources["api_key"] = "", "default"
    values["extra_body"] = MappingProxyType(copy.deepcopy(dict(values["extra_body"])))
    tls_checked = bool(key_tls_checked) if values["api_key"] else None
    return Connection(**values, sources=MappingProxyType(sources), locked=locked, api_key_tls_checked=tls_checked)


def connection_path(settings=None) -> Path:
    """``assistant-connection.json``, next to ``OS_STUDIO_DB_PATH`` (and ``assistant-presets.json``)."""
    return Path(_settings(settings).os_studio_db_path).parent / FILE_NAME


def is_locked(settings=None) -> bool:
    return bool(_settings(settings).os_assistant_connection_locked)


# ── The file store ───────────────────────────────────────────────────────────


class _Store:
    """The saved fields, read again only when the file changes on disk, and written under one lock."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self._key: tuple[str, int, int] | None = None
        self._saved: dict[str, Any] = {}

    def read(self, path: Path) -> dict[str, Any]:
        """The saved fields, a copy the caller may change; {} when there is no file or it can't be read."""
        with self.lock:
            try:
                stat = path.stat()
            except FileNotFoundError:
                return {}
            except OSError as exc:
                logger.warning("Can't read the AI connection file %s (%s); using .env", path, exc)
                return {}
            key = (str(path), stat.st_mtime_ns, stat.st_size)
            if key != self._key:
                self._saved = self._parse(path)
                self._key = key
            return copy.deepcopy(self._saved)

    def write(self, path: Path, saved: Mapping[str, Any]) -> None:
        ordered = {name: saved[name] for name in _STORED_NAMES if name in saved}
        payload = json.dumps({"version": 1, "connection": ordered}, ensure_ascii=False, indent=2)
        # Copied before the file is replaced: a copy that fails leaves the file and this store as they were, so
        # the server never answers from other fields than its file holds.
        copied = copy.deepcopy(ordered)
        with self.lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            # mkstemp makes the file readable by its owner only, and os.replace keeps that: it holds the key.
            descriptor, temporary = tempfile.mkstemp(prefix=".assistant-connection-", suffix=".tmp", dir=path.parent)
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
            self._saved = copied

    def remove(self, path: Path) -> None:
        with self.lock:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            self._key = None
            self._saved = {}

    @staticmethod
    def _parse(path: Path) -> dict[str, Any]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError, RecursionError) as exc:  # RecursionError: nested too deep
            logger.warning("Ignoring the unreadable AI connection file %s (%s); using .env", path, type(exc).__name__)
            return {}
        stored = data.get("connection") if isinstance(data, dict) else None
        if not isinstance(stored, dict):
            logger.warning("Ignoring the AI connection file %s: it has no connection object; using .env", path)
            return {}
        saved: dict[str, Any] = {}
        for name, value in stored.items():
            if name == KEY_ORIGIN:
                if isinstance(value, str) and url_origin(value) == value:
                    saved[name] = value
                else:
                    logger.warning("Skipping %s in the AI connection file %s: it isn't an origin", name, path)
                continue
            if name == KEY_TLS_CHECKED:
                if isinstance(value, bool):
                    saved[name] = value
                else:
                    logger.warning("Skipping %s in the AI connection file %s: it isn't true or false", name, path)
                continue
            if name not in FIELDS or value is None:
                logger.warning("Skipping %r in the AI connection file %s: it isn't a connection field", name, path)
                continue
            try:
                saved[name] = _validate_field(name, value)
            except ConnectionFieldError:
                # The message may quote the value (a URL with a password): only the field is named.
                logger.warning("Skipping %s in the AI connection file %s: its value isn't valid; using .env", name, path)
        return saved


_store = _Store()


# ── The API ──────────────────────────────────────────────────────────────────


def effective(settings=None) -> Connection:
    """The connection in use: ``.env`` with the saved fields over it, or ``.env`` alone when locked."""
    s = _settings(settings)
    locked = is_locked(s)
    saved = {} if locked else _store.read(connection_path(s))
    return _resolve(s, saved, locked)


def effective_url(settings=None) -> str:
    """The effective server URL ("" when the assistant is off). ``Settings.assistant_enabled`` reads it."""
    return effective(settings).url


def candidate(body: Any, settings=None) -> Connection:
    """The effective connection with ``body``'s fields applied by the save rules, without saving them: what
    ``/api/assistant/connection/check`` tries. While locked, only an empty body (the ``.env`` connection) is
    allowed. Raises ConnectionFieldError or ConnectionLockedError."""
    s = _settings(settings)
    changes = validate_fields(body)
    if is_locked(s):
        if changes:
            raise ConnectionLockedError()
        return effective(s)
    saved = _store.read(connection_path(s))
    _apply(s, saved, changes)
    return _resolve(s, saved, False)


def save(body: Any, settings=None) -> Connection:
    """Save the given fields over the stored ones (the rules are in the module's docstring) and return the new
    effective connection. Raises ConnectionFieldError (nothing is saved) or ConnectionLockedError."""
    s = _settings(settings)
    if is_locked(s):
        raise ConnectionLockedError()
    changes = validate_fields(body)
    path = connection_path(s)
    with _store.lock:
        saved = _store.read(path)
        _apply(s, saved, changes)
        _store.write(path, saved)
    return effective(s)


def forget(settings=None) -> Connection:
    """Delete the saved file, so every field follows ``.env`` again. Raises ConnectionLockedError."""
    s = _settings(settings)
    if is_locked(s):
        raise ConnectionLockedError()
    _store.remove(connection_path(s))
    return effective(s)


def _apply(s, saved: dict[str, Any], changes: Mapping[str, Any]) -> None:
    """A save's changes over the saved fields. A key in ``changes`` is saved for the origin of the URL in use after
    them, and with the certificate check in use after them. A stored key they leave out is kept for that origin, or
    for no URL, and cleared for another one; it is also cleared when it was set with the check on and they leave an
    https URL with the check off."""
    before = url_origin(saved.get("url", _env_url(s)))
    if saved.get("api_key"):
        # A file from before these were kept: the URL it was used with, and the check then in effect.
        saved.setdefault(KEY_ORIGIN, before)
        saved.setdefault(KEY_TLS_CHECKED, saved.get("verify_tls", _env_verify_tls(s)))
    for name, value in changes.items():
        if value is None:
            saved.pop(name, None)
        else:
            saved[name] = value
    after = url_origin(saved.get("url", _env_url(s)))
    after_verify_tls = saved.get("verify_tls", _env_verify_tls(s))
    if "api_key" in changes:
        saved.pop(KEY_ORIGIN, None)
        saved.pop(KEY_TLS_CHECKED, None)
        if saved.get("api_key"):
            saved[KEY_ORIGIN] = after
            saved[KEY_TLS_CHECKED] = after_verify_tls
    elif saved.get("api_key") and (
        (after and saved[KEY_ORIGIN] != after)
        or (saved[KEY_TLS_CHECKED] and _unchecked_https(after, after_verify_tls))
    ):
        for name in _KEY_NAMES:
            saved.pop(name, None)
