"""The HTTP client for the AI server: ``POST {url}/chat/completions`` with ``stream: false`` (``reply()``), or with
``stream: true`` for a voice turn (``stream_reply()``, read as server-sent events as they arrive).

Every field the request carries beyond the model, the messages and ``stream`` is sent only when it is set, because
an OpenAI-style endpoint may refuse a field it doesn't know. The thinking is split off the answer wherever the
server puts it (``reasoning_content``, ``reasoning`` or ``<think>`` tags), so it's never spoken.

The connection guards hold for every URL, from ``.env`` or saved in Settings: http or https with a host, no
credentials in it, no redirects followed, a body read in a stream and refused past ``MAX_REPLY_BYTES``, and the
server's answer never echoed: an error names only its HTTP status, or only the kind of failure when the answer isn't
valid HTTP or can't be decoded (never the bytes refused), and ``list_models`` returns only model ids.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

import httpx

from src.assistant.errors import (
    ASSISTANT_BAD_RESPONSE,
    ASSISTANT_EMPTY_REPLY,
    ASSISTANT_HTTP_ERROR,
    ASSISTANT_REPLY_CUT_OFF,
    ASSISTANT_TIMEOUT,
    ASSISTANT_UNREACHABLE,
    AssistantError,
)
from src.assistant.presets import MODEL_MAX_CHARS
from src.config import check_assistant_url

logger = logging.getLogger(__name__)

# The largest answer read, in a stream; a book piece's answer is a small fraction of this.
MAX_REPLY_BYTES = 8 * 1024 * 1024
# GET {url}/models for a connection check: its time limit, and the most model ids returned.
CHECK_TIMEOUT_S = 10.0
MAX_MODELS = 500
# A replaced client closes once its requests in flight end, or this long after the last one's deadline.
CLOSE_GRACE_S = 1.0
# What a request that couldn't reach the server raises: any httpx.HTTPError, and httpx.InvalidURL, which isn't one,
# raised while the request is built for a URL httpx can't parse.
_UNREACHABLE_ERRORS = (httpx.HTTPError, httpx.InvalidURL)
# <think>…</think> at the start of content; an unclosed one (cut off while thinking) runs to the end.
_THINK_BLOCK = re.compile(r"\s*<think>(.*?)(?:</think>|\Z)", re.DOTALL)
_OPEN_THINK = "<think>"
_CLOSE_THINK = "</think>"
# The token counts a streamed reply's usage may carry; only whole numbers are kept.
_USAGE_FIELDS = ("prompt_tokens", "completion_tokens")


@dataclass(frozen=True)
class AssistantReply:
    """The AI's answer. ``text`` never holds the thinking; ``thinking`` is ``None`` when the server sent none."""

    text: str
    thinking: str | None
    finish_reason: str | None


@dataclass(frozen=True)
class StreamedReply(AssistantReply):
    """What ``stream_reply()`` read: the reply as ``reply()`` would give it, whether it came as server-sent events
    (``streamed``; False for one plain JSON answer), and the server's token counts when it sent them (``usage``:
    ``prompt_tokens`` and ``completion_tokens``, whole numbers only)."""

    streamed: bool = True
    usage: Mapping[str, int] = dataclass_field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class ModelList:
    """What ``GET {url}/models`` answered: its HTTP ``status``, and the model ids when the body was an
    OpenAI-style list (``listed``). Nothing else of the body is kept."""

    status: int
    models: list[str] = dataclass_field(default_factory=list)
    listed: bool = False


class _TooLarge(Exception):
    pass


def display_host(url: str) -> str:
    """The server's host (and port) from a URL: no scheme, user, password, path or query."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    try:
        port = parts.port
    except ValueError:
        port = None
    return f"{host}:{port}" if port else host


class AssistantClient:
    """One ``httpx.AsyncClient`` for the AI server, built from an effective connection (``.env`` with a saved
    assistant's fields over it; ``src/assistant/connection.py``).

    When its assistant changes or is deleted (``src.assistant.rebuild_client``), it is *retired*: a request already
    in flight finishes on it, a new one goes to its assistant's current client (``get_client(assistant_id)``; once
    its assistant is deleted it fails as ``assistant_unreachable``), and it closes once its requests end
    (``aclose_when_idle``).
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str = "",
        extra_body: dict[str, Any] | None = None,
        verify_tls: bool = True,
        max_tokens_field: str = "max_tokens",
        template_kwargs: bool = False,
        stream: bool = True,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        checked = check_assistant_url(base_url, "The AI server's URL")  # ValueError: the guards refuse it
        if not checked:
            raise ValueError("The AI server's URL is empty")
        self._base = checked
        self._url = checked + "/chat/completions"
        self._host = display_host(checked)
        self._extra_body = dict(extra_body or {})
        self._max_tokens_field = max_tokens_field
        self._template_kwargs = template_kwargs
        # Stream replies (OS_ASSISTANT_STREAM), and whether the server refused stream_options.include_usage with a
        # 400, so a streamed request doesn't ask for it again.
        self._stream = bool(stream)
        self._usage_refused = False
        self._secrets = _secret_forms(api_key)
        # Loop deadlines of the requests in flight, and a future the closer waits on until they've all ended.
        self._deadlines: dict[object, float] = {}
        self._idle: asyncio.Future | None = None
        self._retired = False
        # The saved assistant this client is for ("" for .env alone).
        self.assistant_id = ""
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        # No redirects: requests go only to the connection's own URL.
        self._http = httpx.AsyncClient(verify=verify_tls, transport=transport, headers=headers, follow_redirects=False)

    @classmethod
    def from_connection(cls, connection, *, transport: httpx.AsyncBaseTransport | None = None) -> AssistantClient:
        """A client for a ``src.assistant.connection.Connection`` (which must have a URL)."""
        return cls(
            connection.url,
            api_key=connection.api_key,
            extra_body=dict(connection.extra_body),
            verify_tls=connection.verify_tls,
            max_tokens_field=connection.max_tokens_field,
            template_kwargs=connection.template_kwargs,
            stream=connection.stream,
            transport=transport,
        )

    @classmethod
    def from_settings(cls, settings, *, transport: httpx.AsyncBaseTransport | None = None) -> AssistantClient:
        """A client for the effective connection: ``.env`` (these settings) with the saved connection over it,
        unless OS_ASSISTANT_CONNECTION_LOCKED. ``lifespan()`` builds the shared client with it."""
        from src.assistant import connection

        return cls.from_connection(connection.effective(settings), transport=transport)

    @property
    def busy(self) -> bool:
        """Whether a request is in flight."""
        return bool(self._deadlines)

    @property
    def stream(self) -> bool:
        """Whether a voice turn streams its reply (Stream replies, OS_ASSISTANT_STREAM).
        A retired client answers for its assistant's current client, which its requests go to; False with none."""
        if self._retired:
            from src import assistant  # the package imports this module

            current = assistant.get_client(self.assistant_id or None)
            if current is not self:
                return getattr(current, "stream", False) is True
        return self._stream

    @property
    def retired(self) -> bool:
        """Whether a change replaced this client: new requests go to its assistant's current client."""
        return self._retired

    def retire(self) -> None:
        self._retired = True

    async def aclose_when_idle(self, grace_s: float | None = None) -> None:
        """Close once the requests in flight have ended, or ``grace_s`` (``CLOSE_GRACE_S``) after the latest of
        their deadlines, whichever comes first. Cancelling it closes the client at once."""
        grace = CLOSE_GRACE_S if grace_s is None else grace_s
        try:
            if self._deadlines:
                loop = asyncio.get_running_loop()
                self._idle = loop.create_future()
                try:
                    async with asyncio.timeout_at(max(self._deadlines.values()) + grace):
                        await self._idle
                except TimeoutError:
                    logger.warning("Closing a replaced AI client with %d request(s) still in flight", len(self._deadlines))
        finally:
            await self.aclose()

    @contextlib.contextmanager
    def _in_flight(self, timeout: float):
        token = object()
        self._deadlines[token] = asyncio.get_running_loop().time() + timeout
        try:
            yield
        finally:
            del self._deadlines[token]
            idle = self._idle
            if not self._deadlines and idle is not None and not idle.done():
                idle.set_result(None)

    def __repr__(self) -> str:
        return f"<AssistantClient {self._host}>"

    @property
    def host(self) -> str:
        """The server's host and port, without the path, query or any credentials."""
        return self._host

    @property
    def is_closed(self) -> bool:
        return self._http.is_closed

    def build_body(self, messages: list[dict[str, Any]], settings) -> dict[str, Any]:
        """The request body: only the fields that are set, then OS_ASSISTANT_EXTRA_BODY merged over them.

        ``chat_template_kwargs`` is merged key by key instead: the extra body's template variables are kept, and
        the preset's thinking controls win for the keys they set. With OS_ASSISTANT_TEMPLATE_KWARGS the effort is
        also sent as the ``reasoning_effort`` template variable, for servers whose template reads it there. The
        thinking budget goes as ``reasoning_budget_tokens`` (the llama.cpp-style name). The preset's ``reasoning_effort`` and
        budget, when set, win over the extra body's top-level keys of those names; when not, the extra body's go.
        """
        body: dict[str, Any] = {}
        if settings.model:
            body["model"] = settings.model
        body["messages"] = list(messages)
        body["stream"] = False
        template_variables: dict[str, Any] = {}
        if settings.thinking in ("on", "off"):
            template_variables["enable_thinking"] = settings.thinking == "on"
        if settings.reasoning_effort:
            body["reasoning_effort"] = settings.reasoning_effort
            if self._template_kwargs:
                template_variables["reasoning_effort"] = settings.reasoning_effort
        if settings.max_tokens and settings.max_tokens > 0:
            body[self._max_tokens_field] = settings.max_tokens
        if settings.temperature is not None:
            body["temperature"] = settings.temperature
        if settings.top_p is not None:
            body["top_p"] = settings.top_p
        extra = dict(self._extra_body)
        extra_variables = extra.pop("chat_template_kwargs", None)
        body.update(extra)
        if settings.reasoning_effort:
            # Set over the extra body: the preset's effort wins, as its other thinking controls do.
            body["reasoning_effort"] = settings.reasoning_effort
        if settings.thinking_budget and settings.thinking_budget > 0:
            # The llama.cpp-style name, set over the extra body: the preset's budget wins.
            body["reasoning_budget_tokens"] = settings.thinking_budget
        if isinstance(extra_variables, dict):
            template_variables = {**extra_variables, **template_variables}
        elif extra_variables is not None and not template_variables:
            template_variables = extra_variables  # not an object: passed through as the extra body gave it
        if template_variables:
            body["chat_template_kwargs"] = template_variables
        return body

    async def reply(self, messages: list[dict[str, Any]], settings) -> AssistantReply:
        """Send the messages and wait for the whole answer, within ``settings.timeout_s``.

        Raises AssistantError with one of the codes in ``src.assistant.errors``. A cancel (the caller's task
        cancelled) aborts the request and propagates as CancelledError. On a client a change replaced
        (``retired``), the request goes to its assistant's current client instead, so a realtime session opened
        before the save keeps working; with the connection turned off it is ``assistant_unreachable``.
        """
        if self._retired:
            from src import assistant  # the package imports this module

            current = assistant.get_client(self.assistant_id or None)
            if current is None:
                raise AssistantError("The AI assistant was turned off in its settings", ASSISTANT_UNREACHABLE)
            if current is not self:
                return await current.reply(messages, settings)
        timeout = float(settings.timeout_s)
        body = self.build_body(messages, settings)
        with self._in_flight(timeout):
            try:
                async with asyncio.timeout(timeout):
                    status, raw = await self._send("POST", self._url, timeout, json=body)
            except (TimeoutError, httpx.TimeoutException):
                raise AssistantError(f"The AI server sent no whole reply within {timeout:g} s", ASSISTANT_TIMEOUT) from None
            except _TooLarge:
                raise AssistantError(f"The AI server's reply is larger than {MAX_REPLY_BYTES} bytes", ASSISTANT_BAD_RESPONSE) from None
            except httpx.DecodingError as exc:
                raise AssistantError(f"The AI server's reply couldn't be decoded ({type(exc).__name__})", ASSISTANT_BAD_RESPONSE) from None
            except _UNREACHABLE_ERRORS as exc:
                detail = self._redact(_failure_detail(exc))
                raise AssistantError(f"Couldn't reach the AI server ({detail})", ASSISTANT_UNREACHABLE) from None
        if status != 200:
            # Only the status: the body of whatever answered is never echoed (the connection guards).
            raise AssistantError(f"The AI server answered HTTP {status}", ASSISTANT_HTTP_ERROR)
        return self._parse(raw)

    async def stream_reply(
        self, messages: list[dict[str, Any]], settings, on_piece: Callable[[str, str], None],
    ) -> StreamedReply:
        """Send the messages with ``stream: true`` and read the reply as it arrives, the whole of it within
        ``settings.timeout_s`` as for ``reply()``. A voice turn uses it.

        ``on_piece(kind, text)`` is called on the event loop for each piece as it is read, and must not block:
        ``"thinking"`` for ``reasoning_content`` (or ``reasoning``) and for a ``<think>…</think>`` block at the start
        of ``content``, ``"answer"`` for the rest of ``content``, from its first character that isn't white space. A
        ``</think>`` with no ``<think>`` before it (a chat template that opened the thinking itself) can't be told from
        the answer while it streams: such a server needs its reasoning parser on, or Stream replies off.

        The body is ``build_body()``'s with ``stream: true`` and ``stream_options: {"include_usage": true}``, for the
        token counts. A server that refuses that with HTTP 400 is asked once more without it, and if that works, this
        client doesn't ask for it again. A server that answers with one plain JSON reply is read as ``reply()`` reads
        it, then handed on as one thinking piece and one answer piece (``streamed`` False).

        Returns what ``reply()`` would (the answer stripped; ``finish_reason`` ``"length"`` when the token limit cut
        it off mid-answer). Raises AssistantError with ``reply()``'s codes, also after some pieces were handed on: a
        dropped connection, an error event in the stream (``assistant_http_error``; its text is never echoed), a piece
        that isn't a JSON object, or a stream that ends with neither a ``finish_reason`` nor ``[DONE]``
        (``assistant_bad_response``). A cancel closes the stream, so the server can stop generating, and propagates as
        CancelledError. A retired client streams through its assistant's current client, as ``reply()`` does.
        """
        if self._retired:
            from src import assistant  # the package imports this module

            current = assistant.get_client(self.assistant_id or None)
            if current is None:
                raise AssistantError("The AI assistant was turned off in its settings", ASSISTANT_UNREACHABLE)
            if current is not self:
                return await current.stream_reply(messages, settings, on_piece)
        timeout = float(settings.timeout_s)
        body = {**self.build_body(messages, settings), "stream": True}
        with self._in_flight(timeout):
            try:
                async with asyncio.timeout(timeout):
                    return await self._post_streamed(body, timeout, on_piece)
            except (TimeoutError, httpx.TimeoutException):
                raise AssistantError(f"The AI server sent no whole reply within {timeout:g} s", ASSISTANT_TIMEOUT) from None
            except _TooLarge:
                raise AssistantError(f"The AI server's reply is larger than {MAX_REPLY_BYTES} bytes", ASSISTANT_BAD_RESPONSE) from None
            except httpx.DecodingError as exc:
                raise AssistantError(f"The AI server's reply couldn't be decoded ({type(exc).__name__})", ASSISTANT_BAD_RESPONSE) from None
            except _UNREACHABLE_ERRORS as exc:
                detail = self._redact(_failure_detail(exc))
                raise AssistantError(f"Couldn't reach the AI server ({detail})", ASSISTANT_UNREACHABLE) from None

    async def _post_streamed(
        self, body: dict[str, Any], timeout: float, on_piece: Callable[[str, str], None],
    ) -> StreamedReply:
        """POST the streamed request, once more without include_usage after a 400 to it, and read the answer. Only a
        200's body is read."""
        ask_usage = not self._usage_refused
        retried = False
        while True:
            sent = {**body, "stream_options": {"include_usage": True}} if ask_usage else body
            async with self._http.stream("POST", self._url, timeout=timeout, json=sent) as response:
                status = response.status_code
                if status == 400 and ask_usage:
                    logger.info("The AI server answered HTTP 400 to stream_options.include_usage; asking without it")
                    ask_usage, retried = False, True
                    continue
                if status != 200:
                    # Only the status: the body of whatever answered is never echoed (the connection guards).
                    raise AssistantError(f"The AI server answered HTTP {status}", ASSISTANT_HTTP_ERROR)
                if retried:
                    self._usage_refused = True
                return await self._read_streamed(response, on_piece)

    async def _read_streamed(self, response: httpx.Response, on_piece: Callable[[str, str], None]) -> StreamedReply:
        """Read a 200's body as server-sent events, line by line, refused past MAX_REPLY_BYTES; or, when it starts
        as JSON does, whole, as one plain reply. Stops at ``data: [DONE]``."""
        parser = _StreamParser(on_piece)
        size = 0
        pending = b""
        whole: list[bytes] | None = None
        sniffed = False
        async for chunk in response.aiter_bytes():
            size += len(chunk)
            if size > MAX_REPLY_BYTES:
                raise _TooLarge()
            if whole is not None:
                whole.append(chunk)
                continue
            pending += chunk
            if not sniffed:
                head = pending.lstrip()
                if not head:
                    continue
                sniffed = True
                if head[:1] in (b"{", b"["):
                    whole, pending = [pending], b""
                    continue
            *lines, pending = pending.split(b"\n")
            for line in lines:
                parser.line(line)
                if parser.done:
                    break
            if parser.done:
                break
        if whole is not None:
            return self._whole_reply(b"".join(whole), on_piece)
        if pending and not parser.done:
            parser.line(pending)
        return parser.finish()

    def _whole_reply(self, raw: bytes, on_piece: Callable[[str, str], None]) -> StreamedReply:
        """A plain JSON answer to a streamed request, read as ``reply()`` reads it and handed on in one piece each."""
        reply = self._parse(raw)
        if reply.thinking:
            on_piece("thinking", reply.thinking)
        on_piece("answer", reply.text)
        return StreamedReply(
            text=reply.text, thinking=reply.thinking, finish_reason=reply.finish_reason, streamed=False,
            usage=MappingProxyType(_usage_of(json.loads(raw))),  # _parse read it already
        )

    async def list_models(self, timeout: float | None = None) -> ModelList:
        """``GET {url}/models``, which generates nothing: the connection check. Answers the status and, from an
        OpenAI-style list (``{"data": [{"id": ...}, ...]}``), only the model ids (at most ``MAX_MODELS``, each up
        to a preset model's length). Raises AssistantError ``assistant_unreachable`` or ``assistant_timeout``."""
        timeout = CHECK_TIMEOUT_S if timeout is None else float(timeout)
        with self._in_flight(timeout):
            try:
                async with asyncio.timeout(timeout):
                    status, raw = await self._send("GET", self._base + "/models", timeout)
            except (TimeoutError, httpx.TimeoutException):
                raise AssistantError(f"The AI server didn't answer within {timeout:g} s", ASSISTANT_TIMEOUT) from None
            except (_TooLarge, httpx.DecodingError):
                return ModelList(status=200)
            except _UNREACHABLE_ERRORS as exc:
                detail = self._redact(_failure_detail(exc))
                raise AssistantError(f"Couldn't reach the AI server ({detail})", ASSISTANT_UNREACHABLE) from None
        ids = _model_ids(raw) if status == 200 else None
        return ModelList(status=status, models=ids or [], listed=ids is not None)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _send(self, method: str, url: str, timeout: float, **kwargs) -> tuple[int, bytes]:
        """(status, body). Only a 200's body is read, in a stream, and refused past MAX_REPLY_BYTES."""
        async with self._http.stream(method, url, timeout=timeout, **kwargs) as response:
            if response.status_code != 200:
                return response.status_code, b""
            chunks: list[bytes] = []
            size = 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > MAX_REPLY_BYTES:
                    raise _TooLarge()
                chunks.append(chunk)
            return response.status_code, b"".join(chunks)

    def _parse(self, raw: bytes) -> AssistantReply:
        try:
            data = json.loads(raw)
        except ValueError:
            raise AssistantError("The AI server's reply is not JSON", ASSISTANT_BAD_RESPONSE) from None
        except RecursionError:  # nested deeper than the json module reads
            message = "The AI server's reply is JSON nested too deep to read"
            raise AssistantError(message, ASSISTANT_BAD_RESPONSE) from None
        choices = data.get("choices") if isinstance(data, dict) else None
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
        message = choice.get("message") if choice else None
        if not isinstance(message, dict):
            raise AssistantError("The AI server's reply has no choices[0].message", ASSISTANT_BAD_RESPONSE)
        content = message.get("content")
        if content is None:
            content = ""
        elif isinstance(content, list):
            # Content parts: join the text ones.
            content = "".join(
                part["text"] for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
                and part.get("type", "text") in ("text", "output_text")
            )
        elif not isinstance(content, str):
            raise AssistantError("The AI server's reply content is not text", ASSISTANT_BAD_RESPONSE)
        thoughts: list[str] = []
        for field in ("reasoning_content", "reasoning"):
            value = message.get(field)
            if isinstance(value, str) and value.strip() and value.strip() not in thoughts:
                thoughts.append(value.strip())
        text, tagged = _split_think_tags(content)
        if tagged and tagged.strip():
            thoughts.append(tagged.strip())
        finish_reason = choice.get("finish_reason")
        finish_reason = finish_reason if isinstance(finish_reason, str) else None
        text = text.strip()
        if not text:
            if finish_reason == "length":
                raise AssistantError(
                    "The token limit stopped the AI before its answer started: raise the token limit or lower the "
                    "thinking",
                    ASSISTANT_REPLY_CUT_OFF,
                )
            raise AssistantError("The AI's reply has no answer once the thinking is removed", ASSISTANT_EMPTY_REPLY)
        return AssistantReply(text=text, thinking="\n\n".join(thoughts) or None, finish_reason=finish_reason)

    def _redact(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "[redacted]")
        return text


def _secret_forms(secret: str) -> tuple[str, ...]:
    """The forms of a secret that error text can hold, longest first: as given, escaped as ``repr()`` escapes it
    (h11 quotes a header value it refuses with ``{!r}``), its UTF-8 bytes escaped, and stripped. A key
    ending in a line break is quoted escaped, so a plain match would miss it."""
    if not secret:
        return ()
    forms = {secret, repr(secret)[1:-1], repr(secret.encode("utf-8", "surrogatepass"))[2:-1], secret.strip()}
    return tuple(sorted((form for form in forms if form), key=lambda form: (-len(form), form)))


def _failure_detail(exc: Exception) -> str:
    """What an error says of a request that failed: the exception's type, then its text for a transport error
    (connect, read, TLS, DNS), which comes from this machine. A protocol error is named by its type with a fixed
    sentence, never its text: a LocalProtocolError is raised before anything is sent, and h11's text quotes the
    header values, the key among them; a RemoteProtocolError is raised while the answer is read, and h11's text
    quotes the line of the answer it refused."""
    if isinstance(exc, httpx.ProtocolError):
        if isinstance(exc, httpx.LocalProtocolError):
            return f"{type(exc).__name__}: the request couldn't be built; check the API key"
        return (f"{type(exc).__name__}: the server's answer isn't valid HTTP, or it closed the connection "
                "before answering")
    return f"{type(exc).__name__}: {exc}"


def _model_ids(raw: bytes) -> list[str] | None:
    """The model ids of an OpenAI-style list, or None when the body isn't one. Nothing else is taken from it."""
    try:
        data = json.loads(raw)
    except (ValueError, RecursionError):  # RecursionError: nested too deep to read
        return None
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return None
    ids: list[str] = []
    for item in items:
        model_id = item.get("id") if isinstance(item, dict) else None
        if (isinstance(model_id, str) and model_id.strip() and len(model_id) <= MODEL_MAX_CHARS
                and model_id.isprintable() and model_id not in ids):
            ids.append(model_id)
            if len(ids) >= MAX_MODELS:
                break
    return ids


def _split_think_tags(content: str) -> tuple[str, str | None]:
    """(the answer, the thinking) for content that starts with <think>…</think>, or whose template opened it."""
    match = _THINK_BLOCK.match(content)
    if match:
        return content[match.end():], match.group(1)
    close = content.find(_CLOSE_THINK)
    if close != -1:
        # The chat template opened <think> in the prompt, so the reply starts inside the thinking.
        return content[close + len(_CLOSE_THINK):], content[:close]
    return content, None


# ── A streamed reply ─────────────────────────────────────────────────────────


def _usage_of(data: Any) -> dict[str, int]:
    """The token counts of a reply or a stream's chunk (``usage``), whole numbers only; {} when it has none."""
    usage = data.get("usage") if isinstance(data, dict) else None
    if not isinstance(usage, dict):
        return {}
    return {name: usage[name] for name in _USAGE_FIELDS if type(usage.get(name)) is int and usage[name] >= 0}


def _text_of_parts(parts: list) -> str:
    """Content given as a list of parts: the text ones joined, as ``_parse`` joins them."""
    return "".join(
        part["text"] for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str) and part.get("type", "text") in ("text", "output_text")
    )


def _partial_suffix(text: str, tag: str) -> int:
    """How many of ``text``'s last characters may be the start of ``tag``, still arriving."""
    for size in range(min(len(text), len(tag) - 1), 0, -1):
        if tag.startswith(text[-size:]):
            return size
    return 0


class _StreamParser:
    """An OpenAI-style chat-completions stream, read line by line: server-sent events whose ``data:`` lines are JSON
    chunks (``choices[0].delta`` with ``reasoning_content``, ``reasoning`` or ``content``, and ``finish_reason``; a
    last chunk may carry only ``usage``), ending with ``data: [DONE]``. Each piece of the thinking and the answer goes
    to ``on_piece`` as it is read; ``finish()`` gives the reply. Comments and other fields (``event:``, ``id:``,
    ``retry:``) are skipped. A data line that is a whole chunk is taken at once, as most servers send one per
    event; one that isn't waits for the next lines of its event, up to the blank line that ends it."""

    def __init__(self, on_piece: Callable[[str, str], None]) -> None:
        self._on_piece = on_piece
        self.done = False
        self._data: list[str] = []
        self._finish_reason: str | None = None
        self._usage: dict[str, int] = {}
        self._reasoning: list[str] = []
        self._tagged: list[str] = []
        self._answer: list[str] = []
        # Where content stands: "start" (only white space so far, or what may become "<think>"), "think" (inside a
        # leading <think> block), "lead" (after it, before the answer's first character) or "answer".
        self._state = "start"
        self._held = ""

    def line(self, raw: bytes) -> None:
        try:
            text = raw.rstrip(b"\r").decode("utf-8")
        except UnicodeDecodeError:
            raise AssistantError("The AI server's stream isn't UTF-8 text", ASSISTANT_BAD_RESPONSE) from None
        if not text:
            self._dispatch()
            return
        name, _, value = text.partition(":")
        if name != "data":
            return  # a comment (a keep-alive, say), or a field that says nothing about the reply
        value = value.removeprefix(" ")  # one space after the colon isn't part of the value
        if not self._data:
            if value == "[DONE]":
                self.done = True
                return
            try:
                chunk = json.loads(value)
            except (ValueError, RecursionError):
                pass  # the start of a chunk over several lines, or not JSON: decided at the blank line
            else:
                self._chunk(chunk)
                return
        self._data.append(value)

    def finish(self) -> StreamedReply:
        """The reply, once the stream has ended. Raises AssistantError as ``_parse`` does for an empty answer, and
        ``assistant_bad_response`` for a stream that ended with neither a finish_reason nor [DONE]."""
        self._dispatch()  # an event with no blank line after it
        if self._state == "start" and self._held.strip():
            self._emit_answer(self._held.lstrip())  # it only looked like the start of "<think>"
        elif self._state == "think" and self._held:
            self._emit_tagged(self._held)  # cut off inside the thinking
        self._held = ""
        if not self.done and self._finish_reason is None:
            raise AssistantError("The AI server's stream ended before its reply did", ASSISTANT_BAD_RESPONSE)
        text = "".join(self._answer).strip()
        if not text:
            if self._finish_reason == "length":
                raise AssistantError(
                    "The token limit stopped the AI before its answer started: raise the token limit or lower the "
                    "thinking",
                    ASSISTANT_REPLY_CUT_OFF,
                )
            raise AssistantError("The AI's reply has no answer once the thinking is removed", ASSISTANT_EMPTY_REPLY)
        thoughts: list[str] = []
        for part in ("".join(self._reasoning).strip(), "".join(self._tagged).strip()):
            if part and part not in thoughts:
                thoughts.append(part)
        return StreamedReply(
            text=text, thinking="\n\n".join(thoughts) or None, finish_reason=self._finish_reason, streamed=True,
            usage=MappingProxyType(dict(self._usage)),
        )

    def _dispatch(self) -> None:
        if not self._data:
            return
        payload, self._data = "\n".join(self._data), []
        if payload == "[DONE]":
            self.done = True
            return
        try:
            chunk = json.loads(payload)
        except (ValueError, RecursionError):
            raise AssistantError("A piece of the AI server's stream is not JSON", ASSISTANT_BAD_RESPONSE) from None
        self._chunk(chunk)

    def _chunk(self, chunk: Any) -> None:
        if not isinstance(chunk, dict):
            raise AssistantError("A piece of the AI server's stream is not a JSON object", ASSISTANT_BAD_RESPONSE)
        if chunk.get("error") is not None:
            # Only that it happened: the server's text is never echoed (the connection guards).
            raise AssistantError("The AI server reported an error in the middle of its reply", ASSISTANT_HTTP_ERROR)
        self._usage = _usage_of(chunk) or self._usage
        choices = chunk.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
        if choice is None:
            return
        delta = choice.get("delta")
        if isinstance(delta, dict):
            thinking = delta.get("reasoning_content")
            if not (isinstance(thinking, str) and thinking):
                thinking = delta.get("reasoning")
            if isinstance(thinking, str) and thinking:
                self._reasoning.append(thinking)
                self._on_piece("thinking", thinking)
            content = delta.get("content")
            if isinstance(content, list):
                content = _text_of_parts(content)
            elif content is not None and not isinstance(content, str):
                raise AssistantError("The AI server's reply content is not text", ASSISTANT_BAD_RESPONSE)
            if content:
                self._content(content)
        finish_reason = choice.get("finish_reason")
        if isinstance(finish_reason, str) and finish_reason:
            self._finish_reason = finish_reason

    def _content(self, text: str) -> None:
        """A piece of ``content``: a leading <think> block is thinking, even cut across pieces; the rest is answer."""
        if self._state == "answer":
            self._emit_answer(text)
            return
        self._held += text
        if self._state == "start":
            lead = self._held.lstrip()
            if not lead or (len(lead) < len(_OPEN_THINK) and _OPEN_THINK.startswith(lead)):
                return  # only white space so far, or what may still become "<think>"
            if not lead.startswith(_OPEN_THINK):
                self._state, self._held = "answer", ""
                self._emit_answer(lead)
                return
            self._state, self._held = "think", lead[len(_OPEN_THINK):]
        if self._state == "think":
            close = self._held.find(_CLOSE_THINK)
            if close == -1:
                keep = _partial_suffix(self._held, _CLOSE_THINK)  # a "</th" may be the start of the closing tag
                self._emit_tagged(self._held[:len(self._held) - keep])
                self._held = self._held[len(self._held) - keep:]
                return
            self._emit_tagged(self._held[:close])
            self._state, self._held = "lead", self._held[close + len(_CLOSE_THINK):]
        lead = self._held.lstrip()  # "lead": the white space before the answer isn't part of it
        if lead:
            self._state, self._held = "answer", ""
            self._emit_answer(lead)

    def _emit_answer(self, text: str) -> None:
        if text:
            self._answer.append(text)
            self._on_piece("answer", text)

    def _emit_tagged(self, text: str) -> None:
        if text:
            self._tagged.append(text)
            self._on_piece("thinking", text)
