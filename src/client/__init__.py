from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import ssl
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import httpx

# The close codes with which the server refuses a stream: it accepts the socket and closes it at once, so the code and
# reason reach the client. 4001: a bad or missing API key or ticket; 1008: the Origin is not allowed; 4004: the feature
# is turned off. A reconnect cannot fix any of them.
REFUSAL_CLOSE_CODES = frozenset({1008, 4001, 4004})


class StreamRefusedError(RuntimeError):
    """The server refused the stream: it accepted the WebSocket and closed it with ``code`` and ``reason``."""

    def __init__(self, code: int, reason: str):
        super().__init__(f"The server refused the stream: close code {code}" + (f", {reason}" if reason else ""))
        self.code = code
        self.reason = reason


def _refusal(*errors: Exception | None) -> StreamRefusedError | None:
    """A StreamRefusedError for the first ConnectionClosed that carries a refusal close code, else None."""
    for error in errors:
        received = getattr(error, "rcvd", None)
        if received is not None and received.code in REFUSAL_CLOSE_CODES:
            return StreamRefusedError(received.code, received.reason)
    return None


def _first_event(raw: str | bytes) -> tuple[dict[str, Any] | None, RuntimeError | None]:
    """The server's first frame as an event, and the error to raise when it is not session.begin.

    Only session.begin starts a stream. An `error` first (a model or VAD that failed to load) means the stream did not
    start, and a reconnect would only load the failing model again.
    """
    event = None if isinstance(raw, bytes) else json.loads(raw)
    if not isinstance(event, dict):
        return None, RuntimeError("The server did not start the stream")
    if event.get("type") == "session.begin":
        return event, None
    message = event.get("message")
    return event, RuntimeError("The server did not start the stream" + (f": {message}" if message else ""))


class OpenSpeechClient:
    def __init__(
        self,
        base_url: str = "https://localhost:8100",
        api_key: str | None = None,
        ssl_verify: bool | str | ssl.SSLContext = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.ssl_verify = ssl_verify

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _ws_url(self, path: str) -> str:
        if self.base_url.startswith("https://"):
            return "wss://" + self.base_url[len("https://"):] + path
        if self.base_url.startswith("http://"):
            return "ws://" + self.base_url[len("http://"):] + path
        return self.base_url + path

    def _ws_tls(self, url: str) -> dict[str, Any]:
        """websockets connect() TLS options that honour ssl_verify as the HTTP methods do."""
        if not url.startswith("wss://") or self.ssl_verify is True:
            return {}  # ws:// takes no TLS options; wss:// verifies by default
        if isinstance(self.ssl_verify, ssl.SSLContext):
            return {"ssl": self.ssl_verify}
        if isinstance(self.ssl_verify, str):
            return {"ssl": ssl.create_default_context(cafile=self.ssl_verify)}  # a CA bundle path
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        return {"ssl": context}

    def transcribe(self, audio: bytes, model: str = "") -> dict:
        with httpx.Client(verify=self.ssl_verify, headers=self._headers(), timeout=60) as c:
            files = {"file": ("audio.wav", audio, "audio/wav")}
            data = {"model": model} if model else {}
            r = c.post(f"{self.base_url}/v1/audio/transcriptions", files=files, data=data)
            r.raise_for_status()
            return r.json()

    def speak(self, text: str, voice: str = "alloy", speed: float = 1.0, model: str = "kokoro", response_format: str = "mp3") -> bytes:
        with httpx.Client(verify=self.ssl_verify, headers=self._headers(), timeout=60) as c:
            r = c.post(f"{self.base_url}/v1/audio/speech", json={
                "model": model,
                "input": text,
                "voice": voice,
                "speed": speed,
                "response_format": response_format,
            })
            r.raise_for_status()
            return r.content

    async def async_transcribe(self, audio: bytes, model: str = "") -> dict:
        async with httpx.AsyncClient(verify=self.ssl_verify, headers=self._headers(), timeout=60) as c:
            files = {"file": ("audio.wav", audio, "audio/wav")}
            data = {"model": model} if model else {}
            r = await c.post(f"{self.base_url}/v1/audio/transcriptions", files=files, data=data)
            r.raise_for_status()
            return r.json()

    async def async_speak(self, text: str, voice: str = "alloy", speed: float = 1.0, model: str = "kokoro", response_format: str = "mp3") -> bytes:
        async with httpx.AsyncClient(verify=self.ssl_verify, headers=self._headers(), timeout=60) as c:
            r = await c.post(f"{self.base_url}/v1/audio/speech", json={
                "model": model,
                "input": text,
                "voice": voice,
                "speed": speed,
                "response_format": response_format,
            })
            r.raise_for_status()
            return r.content

    def stream_transcribe(
        self,
        audio_stream: Iterator[bytes],
        model: str = "",
        sample_rate: int = 16000,
        vad: bool = True,
        reconnect_attempts: int = 2,
    ) -> Iterator[dict[str, Any]]:
        """Sync streaming transcription over /v1/audio/stream.

        Uses a background sender thread and yields server events as they arrive. A failed
        first connection raises its error; after a drop the client reconnects up to
        reconnect_attempts times, refused reconnects included. A server that refuses the
        stream (close code 4001, 1008 or 4004) raises StreamRefusedError with the code and
        reason, without reconnecting. audio_stream is read only after the server's first
        event (session.begin), so a refused stream has read nothing from it and the caller
        can retry with the same source. A first event other than session.begin (an error: the
        model or VAD failed to load) means the stream did not start: it is yielded, then
        RuntimeError is raised, without reconnecting and with nothing read from audio_stream.
        An exception from audio_stream ends the stream and is raised here.

        The stream ends normally only when the server answers the final stop with
        session.end. If the connection closes after the stop instead, or the source ends
        while the connection is down, RuntimeError is raised: the server may not have
        transcribed the last utterance, and a new connection cannot recover it.
        """
        try:
            from websockets.sync.client import connect
            from websockets.exceptions import ConnectionClosed, InvalidHandshake
        except Exception as e:  # pragma: no cover - dependency missing
            raise RuntimeError("websockets package is required for stream_transcribe") from e

        ws_url = (
            f"{self._ws_url('/v1/audio/stream')}?model={model}&sample_rate={sample_rate}&vad={'true' if vad else 'false'}"
        )
        if model == "":
            ws_url = ws_url.replace("model=&", "")

        headers = self._headers() or None
        tls = self._ws_tls(ws_url)
        source = iter(audio_stream)
        attempts = 0
        pending_chunk: bytes | None = None
        source_done = False
        failure: list[Exception] = []
        last_error: Exception | None = None
        sender: threading.Thread | None = None
        opened = False

        while True:
            try:
                connection = connect(ws_url, additional_headers=headers, **tls)
            except (OSError, InvalidHandshake) as exc:
                if not opened:
                    raise  # the first connection: nothing has been read from the source
                last_error = exc
            else:
                opened = True
                stop_evt = threading.Event()
                send_error: list[Exception] = []
                stopping = stop_sent = ended = False
                last_error = None

                with connection as ws:
                    def _sender() -> None:
                        nonlocal pending_chunk, source_done, stopping, stop_sent
                        try:
                            if pending_chunk is not None:
                                ws.send(pending_chunk)
                                pending_chunk = None
                            while True:
                                try:
                                    chunk = next(source)
                                except StopIteration:
                                    source_done = True
                                    break
                                except Exception as exc:
                                    failure.append(exc)  # the caller's audio source failed
                                    break
                                pending_chunk = chunk
                                if stop_evt.is_set():
                                    return
                                ws.send(chunk)
                                pending_chunk = None
                            if source_done and not stop_evt.is_set():
                                stopping = True
                                ws.send(json.dumps({"type": "stop"}))
                                stop_sent = True
                        except ConnectionClosed as exc:
                            send_error.append(exc)  # the receive loop sees the drop too
                        except Exception as exc:
                            failure.append(exc)
                        if failure:
                            with contextlib.suppress(Exception):
                                ws.close()  # ends the receive loop, which raises the failure

                    try:
                        # The server's first frame (session.begin, an error, or a refusal's close) comes before the
                        # source is read, so a refused stream reads nothing.
                        raw = ws.recv()
                    except ConnectionClosed as exc:
                        last_error = exc
                    else:
                        first, not_started = _first_event(raw)
                        if not_started is not None:
                            if first is not None:
                                yield first  # the reason: an error event
                            raise not_started
                        sender = threading.Thread(target=_sender, daemon=True)
                        sender.start()
                        try:
                            while True:
                                if not isinstance(raw, bytes):
                                    event = json.loads(raw)
                                    yield event
                                    if event.get("type") == "session.end":
                                        ended = True
                                        break
                                raw = ws.recv()
                        except ConnectionClosed as exc:
                            last_error = exc
                        finally:
                            stop_evt.set()
                            sender.join(timeout=1.0)

                if stopping:
                    sender.join()
                if failure:
                    raise failure[0]
                refused = _refusal(last_error, *send_error)
                if refused:
                    raise refused from last_error
                if stop_sent and ended:
                    return
                if stopping:
                    # No session.end answered the stop, so the server may never have flushed the last utterance.
                    raise RuntimeError(
                        "stream_transcribe disconnected before the server confirmed the end of the stream"
                    ) from (send_error[0] if send_error else last_error)

            attempts += 1
            if attempts > reconnect_attempts:
                raise RuntimeError("stream_transcribe disconnected and reconnection limit reached") from last_error
            time.sleep(min(0.2 * attempts, 1.0))
            # The old sender may still be waiting on the source; it hands the chunk it reads
            # back through pending_chunk, and two threads must never iterate the source.
            if sender is not None:
                sender.join()
            if failure:
                raise failure[0]
            if source_done:
                # The source ended while the connection was down: every chunk went out on the dropped one.
                raise RuntimeError(
                    "stream_transcribe disconnected before the server confirmed the end of the stream"
                ) from last_error

    async def async_stream_transcribe(
        self,
        audio_stream: AsyncIterator[bytes] | Iterator[bytes],
        model: str = "",
        sample_rate: int = 16000,
        vad: bool = True,
        reconnect_attempts: int = 2,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async streaming transcription over /v1/audio/stream.

        Follows stream_transcribe's failure and reconnect rules; a refused stream has read
        nothing from audio_stream. A sync audio_stream is read in a worker thread, so a source
        that blocks between chunks does not stall the loop.
        """
        try:
            import websockets
            from websockets.exceptions import ConnectionClosed, InvalidHandshake
        except Exception as e:  # pragma: no cover
            raise RuntimeError("websockets package is required for async_stream_transcribe") from e

        ws_url = (
            f"{self._ws_url('/v1/audio/stream')}?model={model}&sample_rate={sample_rate}&vad={'true' if vad else 'false'}"
        )
        if model == "":
            ws_url = ws_url.replace("model=&", "")

        tls = self._ws_tls(ws_url)
        attempts = 0
        pending_chunk: bytes | None = None
        source_done = False
        failure: list[Exception] = []
        last_error: Exception | None = None
        chunks: asyncio.Queue[bytes | Exception | None] = asyncio.Queue(maxsize=1)
        pump_task: asyncio.Task[None] | None = None
        opened = False

        async def _pump() -> None:
            # One reader for every connection: cancelling a connection's sender must
            # never land inside the caller's async generator and finalize it.
            try:
                if hasattr(audio_stream, "__aiter__"):
                    async for chunk in audio_stream:  # type: ignore[union-attr]
                        await chunks.put(chunk)
                else:
                    iterator = iter(audio_stream)  # type: ignore[arg-type]
                    end = object()
                    # A live sync source blocks in next() between buffers: keep that off the event loop.
                    while (chunk := await asyncio.to_thread(next, iterator, end)) is not end:
                        await chunks.put(chunk)
            except Exception as exc:
                await chunks.put(exc)  # the sender reports it
                return
            await chunks.put(None)

        try:
            while True:
                headers = list(self._headers().items()) if self._headers() else None
                stopping = stop_sent = ended = False
                send_error: Exception | None = None
                async with contextlib.AsyncExitStack() as stack:
                    try:
                        ws = await stack.enter_async_context(
                            websockets.connect(ws_url, additional_headers=headers, **tls)
                        )
                    except (OSError, InvalidHandshake) as exc:
                        if not opened:
                            raise  # the first connection: the source has not been read
                        last_error = exc
                    else:
                        opened = True
                        connection_over = False
                        last_error = None
                        frames = aiter(ws)
                        try:
                            # The server's first frame (session.begin, an error, or a refusal's close) comes before
                            # the source is read, so a refused stream reads nothing.
                            first = await anext(frames)
                        except StopAsyncIteration:  # closed normally before any frame
                            first = None
                        except ConnectionClosed as exc:
                            first, last_error = None, exc
                        if first is not None:
                            event, not_started = _first_event(first)
                            if not_started is not None:
                                if event is not None:
                                    yield event  # the reason: an error event
                                raise not_started
                        if first is not None and pump_task is None:
                            pump_task = asyncio.create_task(_pump())

                        async def _sender() -> None:
                            nonlocal pending_chunk, source_done, stopping, stop_sent, send_error
                            try:
                                if pending_chunk is not None:
                                    await ws.send(pending_chunk)
                                    pending_chunk = None
                                while not source_done:
                                    chunk = await chunks.get()
                                    if chunk is None:
                                        source_done = True
                                    elif isinstance(chunk, Exception):
                                        failure.append(chunk)  # the caller's audio source failed
                                        break
                                    else:
                                        pending_chunk = chunk
                                        await ws.send(chunk)
                                        pending_chunk = None
                                if source_done and not connection_over:
                                    stopping = True
                                    await ws.send(json.dumps({"type": "stop"}))
                                    stop_sent = True
                            except ConnectionClosed as exc:
                                send_error = exc  # the receive loop sees the drop too
                            except Exception as exc:
                                failure.append(exc)
                            if failure:
                                with contextlib.suppress(Exception):
                                    await ws.close()  # ends the receive loop, which raises the failure

                        sender_task = asyncio.create_task(_sender()) if first is not None else None
                        try:
                            raw = first
                            while raw is not None:  # None: the connection closed normally
                                if not isinstance(raw, bytes):
                                    event = json.loads(raw)
                                    yield event
                                    if event.get("type") == "session.end":
                                        ended = True
                                        break
                                raw = await anext(frames, None)
                        except ConnectionClosed as exc:
                            last_error = exc
                        finally:
                            connection_over = True
                            if sender_task is not None and not sender_task.done():
                                with contextlib.suppress(BaseException):
                                    await asyncio.wait_for(sender_task, timeout=0.5)
                            if sender_task is not None and not sender_task.done():
                                sender_task.cancel()
                                with contextlib.suppress(BaseException):
                                    await sender_task

                if failure:
                    raise failure[0]
                refused = _refusal(last_error, send_error)
                if refused:
                    raise refused from last_error
                if stop_sent and ended:
                    return
                if stopping or source_done:
                    # No session.end answered a stop, so the server may never have flushed the last
                    # utterance; with the source finished, a new connection has nothing to resend.
                    raise RuntimeError(
                        "async_stream_transcribe disconnected before the server confirmed the end of the stream"
                    ) from (send_error or last_error)

                attempts += 1
                if attempts > reconnect_attempts:
                    raise RuntimeError("async_stream_transcribe disconnected and reconnection limit reached") from last_error
                await asyncio.sleep(min(0.2 * attempts, 1.0))
        finally:
            if pump_task is not None:
                pump_task.cancel()
                with contextlib.suppress(BaseException):
                    await pump_task

    def realtime_session(self, model: str = "") -> "RealtimeSession":
        return RealtimeSession(self, model=model)

    async def async_realtime_session(self, model: str = "") -> "AsyncRealtimeSession":
        sess = AsyncRealtimeSession(self, model=model)
        await sess.connect()
        return sess


class RealtimeSession:
    def __init__(self, client: OpenSpeechClient, model: str = ""):
        self.client = client
        self.model = model
        self._ws = None
        self._receiver: threading.Thread | None = None
        self._running = False
        self._transcript_callbacks: list[Callable[[dict[str, Any]], None]] = []
        self._audio_callbacks: list[Callable[[dict[str, Any]], None]] = []
        self._vad_callbacks: list[Callable[[dict[str, Any]], None]] = []
        self._connect()

    def _connect(self) -> None:
        from websockets.sync.client import connect

        suffix = f"/v1/realtime?model={self.model}" if self.model else "/v1/realtime"
        headers = self.client._headers() or None
        url = self.client._ws_url(suffix)
        self._ws = connect(url, subprotocols=["realtime"], additional_headers=headers, **self.client._ws_tls(url))
        self._running = True
        self._receiver = threading.Thread(target=self._recv_loop, daemon=True)
        self._receiver.start()

    def _recv_loop(self) -> None:
        while self._running and self._ws is not None:
            try:
                raw = self._ws.recv()
                if isinstance(raw, bytes):
                    continue
                evt = json.loads(raw)
                et = evt.get("type", "")
                if "transcription" in et or et == "conversation.item.created":
                    for cb in self._transcript_callbacks:
                        cb(evt)
                elif et.startswith("response.audio"):
                    for cb in self._audio_callbacks:
                        cb(evt)
                elif "speech_" in et:
                    for cb in self._vad_callbacks:
                        cb(evt)
            except Exception:
                break

    def _send(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("Realtime session is closed")
        self._ws.send(json.dumps(payload))

    def send_audio(self, chunk: bytes) -> None:
        self._send({"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode("ascii")})

    def commit(self) -> None:
        self._send({"type": "input_audio_buffer.commit"})

    def create_response(self, text: str, voice: str = "alloy") -> None:
        self._send({
            "type": "response.create",
            "response": {"instructions": text, "voice": voice, "modalities": ["audio", "text"]},
        })

    def on_transcript(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self._transcript_callbacks.append(callback)

    def on_audio(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self._audio_callbacks.append(callback)

    def on_vad(self, callback: Callable[[dict[str, Any]], None]) -> None:
        self._vad_callbacks.append(callback)

    def close(self) -> None:
        self._running = False
        if self._ws is not None:
            try:
                self._ws.close()
            finally:
                self._ws = None
        if self._receiver is not None:
            self._receiver.join(timeout=1.0)


class AsyncRealtimeSession:
    def __init__(self, client: OpenSpeechClient, model: str = ""):
        self.client = client
        self.model = model
        self._ws = None
        self._receiver_task: asyncio.Task[Any] | None = None
        self._transcript_callbacks: list[Callable[[dict[str, Any]], Any]] = []
        self._audio_callbacks: list[Callable[[dict[str, Any]], Any]] = []
        self._vad_callbacks: list[Callable[[dict[str, Any]], Any]] = []

    async def connect(self) -> None:
        import websockets

        suffix = f"/v1/realtime?model={self.model}" if self.model else "/v1/realtime"
        headers = list(self.client._headers().items()) if self.client._headers() else None
        url = self.client._ws_url(suffix)
        self._ws = await websockets.connect(
            url, subprotocols=["realtime"], additional_headers=headers, **self.client._ws_tls(url)
        )
        self._receiver_task = asyncio.create_task(self._recv_loop())

    async def _dispatch(self, callbacks: list[Callable[[dict[str, Any]], Any]], event: dict[str, Any]) -> None:
        for cb in callbacks:
            ret = cb(event)
            if asyncio.iscoroutine(ret):
                await ret

    async def _recv_loop(self) -> None:
        assert self._ws is not None
        async for raw in self._ws:
            if isinstance(raw, bytes):
                continue
            evt = json.loads(raw)
            et = evt.get("type", "")
            if "transcription" in et or et == "conversation.item.created":
                await self._dispatch(self._transcript_callbacks, evt)
            elif et.startswith("response.audio"):
                await self._dispatch(self._audio_callbacks, evt)
            elif "speech_" in et:
                await self._dispatch(self._vad_callbacks, evt)

    async def _send(self, payload: dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("Realtime session is closed")
        await self._ws.send(json.dumps(payload))

    async def send_audio(self, chunk: bytes) -> None:
        await self._send({"type": "input_audio_buffer.append", "audio": base64.b64encode(chunk).decode("ascii")})

    async def commit(self) -> None:
        await self._send({"type": "input_audio_buffer.commit"})

    async def create_response(self, text: str, voice: str = "alloy") -> None:
        await self._send({
            "type": "response.create",
            "response": {"instructions": text, "voice": voice, "modalities": ["audio", "text"]},
        })

    def on_transcript(self, callback: Callable[[dict[str, Any]], Any]) -> None:
        self._transcript_callbacks.append(callback)

    def on_audio(self, callback: Callable[[dict[str, Any]], Any]) -> None:
        self._audio_callbacks.append(callback)

    def on_vad(self, callback: Callable[[dict[str, Any]], Any]) -> None:
        self._vad_callbacks.append(callback)

    async def close(self) -> None:
        if self._receiver_task:
            self._receiver_task.cancel()
            with contextlib.suppress(BaseException):
                await self._receiver_task
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
