"""The AI assistant: Open Speech sends text to a local OpenAI-compatible AI server (such as a llama.cpp server on your network), waits for the
whole reply, and speaks or saves it.

One client is shared. ``lifespan()`` in ``src/main.py`` builds it when the assistant is on and calls
``set_client``; at shutdown it calls ``close_client()``, which runs the ``on_close`` callbacks and then closes the
client. The realtime route and the Reader's job runner call ``get_client()`` (None when the assistant is off) and
register their own shutdown with ``on_close``.

The connection (the server's URL, key, model and the rest) starts from ``.env`` and can be saved through
``/api/assistant/connection`` (``src/assistant/connection.py``). A save calls ``rebuild_client()``: ``get_client()``
then returns a new client, and the old one is retired. A request it already started finishes on it, a later request made
through it goes to the new client, and it closes once its requests end, or after their timeout.

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
_close_callbacks: list[Callable[[], Awaitable[None]]] = []
# Replaced clients still finishing a request: closer task -> client.
_retiring: dict[asyncio.Task, AssistantClient] = {}


def set_client(client: AssistantClient | None) -> None:
    global _client
    _client = client


def get_client() -> AssistantClient | None:
    """The shared client: the current one, after any rebuild. None when the assistant is off (no server URL in
    ``.env`` or saved in Settings) or not started."""
    return _client


def on_close(callback: Callable[[], Awaitable[None]]) -> Callable[[], Awaitable[None]]:
    """Run ``callback`` (an async function with no arguments) at shutdown, before the client closes. A rebuild
    doesn't run it."""
    _close_callbacks.append(callback)
    return callback


def build_client(conn: connection.Connection) -> AssistantClient:
    """A new client for a connection with a URL. ``rebuild_client`` and the connection check build theirs here
    (tests replace it to give them an ``httpx.MockTransport``)."""
    return AssistantClient.from_connection(conn)


async def rebuild_client(settings=None) -> AssistantClient | None:
    """Replace the shared client with one for the effective connection (None when it has no URL), and retire the
    old one: a request in flight on it finishes, and it closes once its requests end, or after their timeout.
    The ``on_close`` callbacks don't run. Returns the new client."""
    global _client
    current = connection.effective(settings)
    new = build_client(current) if current.url else None
    old, _client = _client, new
    if old is not None and old is not new:
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
    replaced client still finishing a request."""
    global _client
    callbacks = list(_close_callbacks)
    _close_callbacks.clear()
    for callback in callbacks:
        try:
            await callback()
        except Exception:
            logger.warning("An AI assistant shutdown callback failed", exc_info=True)
    client, _client = _client, None
    if client is not None:
        await client.aclose()
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
