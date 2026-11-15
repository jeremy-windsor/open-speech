"""The AI assistant: Open Speech sends text to a local OpenAI-compatible AI server (such as a llama.cpp server on your network), waits for the
whole reply, and speaks or saves it.

One client per saved assistant. The shared client is the active assistant's (``.env``'s with none saved):
``lifespan()`` in ``src/main.py`` builds it when the assistant is on and calls ``set_client``; at shutdown it calls
``close_client()``, which runs the ``on_close`` callbacks and then closes every client. The realtime route and the
Reader's job runner call ``get_client()`` (None when the assistant is off) and register their own shutdown with
``on_close``. ``get_client(assistant_id)`` is that assistant's client, built on first use (the shared one for the
active assistant, or for one that is gone).

The connections (the server's URL, key, model and the rest) start from ``.env`` and are saved as assistants
(``src/assistant/connection.py``). A change calls ``rebuild_client()``: a changed or deleted assistant's client is
retired. A request it already started finishes on it, a later request made through it goes to that assistant's current
client (the active one's once it is deleted), and it closes once its requests end, or after their timeout. Making
another assistant active keeps the old one's client for the requests that use it, so a job finishes on the assistant
it started with.

Public names: ``get_client``, ``set_client``, ``on_close``, ``close_client``, ``build_client``, ``rebuild_client``,
``AssistantClient``, ``AssistantReply``, ``AssistantError``, ``ERROR_SENTENCES``, ``Conversation``,
``EffectiveSettings`` and ``speakable_segments``; ``presets.effective``, ``prompts.render``,
``prompts.messages_for`` and ``connection.effective``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from src.assistant import connection
from src.assistant.client import AssistantClient, AssistantReply
from src.assistant.conversation import Conversation
from src.assistant.errors import ERROR_CODES, ERROR_SENTENCES, AssistantError
from src.assistant.presets import EffectiveSettings
from src.assistant.speech import speakable_segments

__all__ = [
    "ERROR_CODES",
    "ERROR_SENTENCES",
    "AssistantClient",
    "AssistantError",
    "AssistantReply",
    "Conversation",
    "EffectiveSettings",
    "build_client",
    "close_client",
    "connection",
    "get_client",
    "on_close",
    "rebuild_client",
    "set_client",
    "speakable_segments",
]

logger = logging.getLogger(__name__)

_client: AssistantClient | None = None
# The assistant the shared client is for: "" for .env alone, None when not known (a client given to set_client).
_client_id: str | None = None
# The clients of the other saved assistants: id -> client.
_others: dict[str, AssistantClient] = {}
_close_callbacks: list[Callable[[], Awaitable[None]]] = []
# Replaced clients still finishing a request: closer task -> client.
_retiring: dict[asyncio.Task, AssistantClient] = {}


def _tag(client, assistant_id: str):
    try:
        client.assistant_id = assistant_id
    except AttributeError:
        pass
    return client


def set_client(client: AssistantClient | None, *, assistant_id: str | None = None) -> None:
    """Set the shared client; ``assistant_id`` is the assistant it was built for ("" for ``.env`` alone)."""
    global _client, _client_id
    _client, _client_id = client, assistant_id
    if client is not None and assistant_id is not None:
        _tag(client, assistant_id)


def get_client(assistant_id: str | None = None) -> AssistantClient | None:
    """The shared client: the current one, after any rebuild. None when the assistant is off (no server URL in
    ``.env`` or saved in Settings) or not started.

    With ``assistant_id``, that saved assistant's client, built on first use; the shared one when it is the active
    assistant or gone, and None when it has no URL."""
    if not assistant_id:
        return _client
    key = connection.resolve_id(assistant_id)
    if key is None or key == connection.active_id():
        return _client
    client = _others.get(key)
    if client is None:
        current = connection.effective(assistant_id=key)
        if not current.url:
            return None
        client = _others[key] = _tag(build_client(current), key)
    return client


def on_close(callback: Callable[[], Awaitable[None]]) -> Callable[[], Awaitable[None]]:
    """Run ``callback`` (an async function with no arguments) at shutdown, before the client closes. A rebuild
    doesn't run it."""
    _close_callbacks.append(callback)
    return callback


def build_client(conn: connection.Connection) -> AssistantClient:
    """A new client for a connection with a URL. ``rebuild_client`` and the connection check build theirs here
    (tests replace it to give them an ``httpx.MockTransport``)."""
    return AssistantClient.from_connection(conn)


async def rebuild_client(settings=None, assistant_id: str | None = None) -> AssistantClient | None:
    """Bring the clients up to date after a change: to the active assistant (without ``assistant_id``: a save
    through ``/api/assistant/connection``, a delete, another one made active) or to the assistant ``assistant_id``
    (changed or deleted). Returns the shared client.

    A changed or deleted assistant's client is retired: a request in flight on it finishes, and it closes once its
    requests end, or after their timeout. When another assistant becomes active, the shared client becomes that
    one's (None when it has no URL), and the one it replaces stays its assistant's client. The ``on_close``
    callbacks don't run."""
    global _client, _client_id
    active = connection.active_id(settings) or ""
    existing = set(connection.assistant_ids(settings))
    for key in [key for key in _others if key not in existing or key == assistant_id]:
        await _retire(_others.pop(key))
    if assistant_id not in (None, active) and _client_id == active:
        return _client
    old, old_id = _client, _client_id
    if old_id == active:
        new = None
    else:
        new = _others.pop(active, None) if active else None
    if new is None:
        current = connection.effective(settings)
        new = _tag(build_client(current), active) if current.url else None
    _client, _client_id = new, active
    if old is not None and old is not new:
        if old_id and old_id != active and old_id in existing and old_id != assistant_id:
            _others[old_id] = old
        else:
            await _retire(old)
    return new


async def _retire(old) -> None:
    retire = getattr(old, "retire", None)
    if retire is not None:
        retire()
    if getattr(old, "busy", False) and hasattr(old, "aclose_when_idle"):
        task = asyncio.get_running_loop().create_task(old.aclose_when_idle(), name="assistant-client-close")
        _retiring[task] = old
        task.add_done_callback(lambda done: _retiring.pop(done, None))
        return
    try:
        await old.aclose()
    except Exception:
        logger.warning("Closing the replaced AI assistant client failed", exc_info=True)


async def close_client() -> None:
    """Run the ``on_close`` callbacks in order (one that fails is logged), then close the shared client and any
    other assistants' clients and any replaced client still finishing a request."""
    global _client, _client_id
    callbacks = list(_close_callbacks)
    _close_callbacks.clear()
    for callback in callbacks:
        try:
            await callback()
        except Exception:
            logger.warning("An AI assistant shutdown callback failed", exc_info=True)
    client, _client, _client_id = _client, None, None
    if client is not None:
        await client.aclose()
    others = list(_others.values())
    _others.clear()
    for other in others:
        try:
            await other.aclose()
        except Exception:
            logger.warning("Closing an AI assistant client failed", exc_info=True)
    loop = asyncio.get_running_loop()
    retiring = list(_retiring.items())
    _retiring.clear()
    closers = [task for task, _old in retiring if task.get_loop() is loop]
    for task in closers:
        task.cancel()
    if closers:
        await asyncio.gather(*closers, return_exceptions=True)
    for _task, old in retiring:
        try:
            await old.aclose()  # also when its closer was cancelled before it started
        except Exception:
            logger.warning("Closing a replaced AI assistant client failed", exc_info=True)
