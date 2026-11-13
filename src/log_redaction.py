"""Mask the credentials a request's query string can carry in uvicorn's own log lines.

uvicorn prints each request's path with its query string: the access line on ``uvicorn.access`` and the WebSocket
"[accepted]" and "403" lines on ``uvicorn.error``. A query string can carry the deprecated ``?api_key=``, a WebSocket
``?ticket=`` (one-time, but valid for up to a minute until it is redeemed), and a Reader media link's ``exp`` and
``sig`` (the link is its own credential). Their values are logged as ``***``; every other parameter and the path stay.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from urllib.parse import unquote_plus

MASK = "***"
SECRET_PARAMETERS = frozenset({"api_key", "ticket", "exp", "sig"})
MASKED_LOGGERS = ("uvicorn.access", "uvicorn.error")

# A query parameter's name, after "?" or "&". Its value runs to the next "&" or whitespace (a "?" may be in a value).
_PARAMETER = re.compile(r"[?&]([^?&=\s]+)=")
_VALUE_END = re.compile(r"[&\s]")


def _is_secret(name: str) -> bool:
    return unquote_plus(name).lower() in SECRET_PARAMETERS


def mask_query_secrets(text: str) -> str:
    """`text` with the value of each api_key, ticket, exp and sig query parameter replaced by ***.

    Names are compared decoded and in any case. An empty value stays empty. Returns `text` itself when nothing changed.
    """
    if text.find("=") < 0:
        return text
    pieces: list[str] = []
    kept_from = scan_from = 0
    value_end = -1  # the end of the value last searched; a "?name=" inside that value ends there too
    while (parameter := _PARAMETER.search(text, scan_from)) is not None:
        value_start = parameter.end()
        if value_end < value_start:  # search each stretch once, or a query of nested "?b=" costs n^2 steps
            found = _VALUE_END.search(text, value_start)
            value_end = found.start() if found else len(text)
        if value_end > value_start and _is_secret(parameter.group(1)):
            pieces += (text[kept_from:value_start], MASK)
            kept_from = scan_from = value_end
        else:
            scan_from = value_start  # a link inside this value can carry a secret of its own
    if not pieces:
        return text
    pieces.append(text[kept_from:])
    return "".join(pieces)


def _masked(value):
    if not isinstance(value, str):
        return value
    text = str.__str__(value)  # a plain str, whatever a subclass overrides
    masked = mask_query_secrets(text)
    return value if masked is text else masked


class QuerySecretFilter(logging.Filter):
    """Masks query-string secrets in a record's str arguments, or in its message when it has none.

    The arguments keep their number and order: uvicorn's AccessFormatter unpacks exactly five. A message with
    arguments is a format string and stays as it is. The filter never raises and never drops a record.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            args = record.args
            if not args:
                if isinstance(record.msg, str):
                    record.msg = _masked(record.msg)
            elif isinstance(args, tuple):
                masked = tuple(_masked(arg) for arg in args)
                if any(new is not old for new, old in zip(masked, args)):
                    record.args = masked
            elif isinstance(args, Mapping):  # logger.info("%(path)s", {"path": ...})
                masked = {key: _masked(arg) for key, arg in args.items()}
                if any(masked[key] is not arg for key, arg in args.items()):
                    record.args = masked
        except Exception:  # noqa: BLE001 - a log line is never lost to the masking
            pass
        return True


query_secret_filter = QuerySecretFilter()  # one instance, so installing it again adds nothing


def install_query_secret_filter() -> None:
    """Mask query-string secrets on uvicorn's loggers.

    A logger's filters outlive the dictConfig that uvicorn.Config applies, which replaces handlers only.
    """
    for name in MASKED_LOGGERS:
        logging.getLogger(name).addFilter(query_secret_filter)
