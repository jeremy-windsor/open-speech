from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any

import httpx


class OpenSpeechClient:
    def __init__(self, base_url: str = "http://localhost:8100", api_key: str | None = None, ssl_verify: bool = True):
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

        Uses a background sender thread and yields server events as they arrive.
        """
        try:
            from websockets.sync.client import connect
            from websockets.exceptions import ConnectionClosed
        except Exception as e:  # pragma: no cover - dependency missing
            raise RuntimeError("websockets package is required for stream_transcribe") from e

        ws_url = (
            f"{self._ws_url('/v1/audio/stream')}?model={model}&sample_rate={sample_rate}&vad={'true' if vad else 'false'}"
        )
        if model == "":
            ws_url = ws_url.replace("model=&", "")

        headers = self._headers() or None
        attempts = 0
        pending_chunk: bytes | None = None

        while attempts <= reconnect_attempts:
            stop_evt = threading.Event()
            sender_error: list[Exception] = []
            exhausted = False

            with connect(ws_url, additional_headers=headers) as ws:
                def _sender() -> None:
                    nonlocal pending_chunk, exhausted
                    try:
                        if pending_chunk is not None:
                            ws.send(pending_chunk)
                            pending_chunk = None
                        for chunk in audio_stream:
                            if stop_evt.is_set():
                                return
                            pending_chunk = chunk
                            ws.send(chunk)
                            pending_chunk = None
                        exhausted = True
                        ws.send(json.dumps({"type": "stop"}))
                    except Exception as exc:  # pragma: no cover - network timing
                        sender_error.append(exc)

                t = threading.Thread(target=_sender, daemon=True)
                t.start()

                try:
                    while True:
                        raw = ws.recv()
                        if isinstance(raw, bytes):
                            continue
                        event = json.loads(raw)
                        yield event
                        if event.get("type") == "session.end":
                            stop_evt.set()
                            break
                except ConnectionClosed:
                    stop_evt.set()
                finally:
                    t.join(timeout=1.0)

            if exhausted and pending_chunk is None:
                return
            if sender_error and exhausted:
                raise sender_error[0]

            attempts += 1
            if attempts > reconnect_attempts:
                raise RuntimeError("stream_transcribe disconnected and reconnection limit reached")
            time.sleep(min(0.2 * attempts, 1.0))

    async def async_stream_transcribe(
        self,
        audio_stream: AsyncIterator[bytes] | Iterator[bytes],
        model: str = "",
        sample_rate: int = 16000,
        vad: bool = True,
        reconnect_attempts: int = 2,
    ) -> AsyncIterator[dict[str, Any]]:
        """Async streaming transcription over /v1/audio/stream."""
        try:
            import websockets
            from websockets.exceptions import ConnectionClosed
        except Exception as e:  # pragma: no cover
            raise RuntimeError("websockets package is required for async_stream_transcribe") from e

        ws_url = (
            f"{self._ws_url('/v1/audio/stream')}?model={model}&sample_rate={sample_rate}&vad={'true' if vad else 'false'}"
        )
        if model == "":
            ws_url = ws_url.replace("model=&", "")

        attempts = 0
        pending_chunk: bytes | None = None
        source_done = False

        while attempts <= reconnect_attempts:
            headers = list(self._headers().items()) if self._headers() else None
            async with websockets.connect(ws_url, additional_headers=headers) as ws:
                send_task_exc: Exception | None = None

                async def _sender() -> None:
                    nonlocal pending_chunk, source_done, send_task_exc
                    try:
                        if pending_chunk is not None:
                            await ws.send(pending_chunk)
                            pending_chunk = None

                        if hasattr(audio_stream, "__aiter__"):
                            async for chunk in audio_stream:  # type: ignore[union-attr]
                                pending_chunk = chunk
                                await ws.send(chunk)
                                pending_chunk = None
                        else:
                            for chunk in audio_stream:  # type: ignore[not-an-iterable]
                                pending_chunk = chunk
                                await ws.send(chunk)
                                pending_chunk = None

                        source_done = True
                        await ws.send(json.dumps({"type": "stop"}))
                    except Exception as exc:  # pragma: no cover
                        send_task_exc = exc

                sender_task = asyncio.create_task(_sender())
                try:
                    async for raw in ws:
                        if isinstance(raw, bytes):
                            continue
                        event = json.loads(raw)
                        yield event
                        if event.get("type") == "session.end":
                            break
                except ConnectionClosed:
                    pass
                finally:
                    if not sender_task.done():
                        with contextlib.suppress(BaseException):
                            await asyncio.wait_for(sender_task, timeout=0.5)
                    if not sender_task.done():
                        sender_task.cancel()
                        with contextlib.suppress(BaseException):
                            await sender_task

                if source_done and pending_chunk is None:
                    return
                if send_task_exc and source_done:
                    raise send_task_exc

            attempts += 1
            if attempts > reconnect_attempts:
                raise RuntimeError("async_stream_transcribe disconnected and reconnection limit reached")
            await asyncio.sleep(min(0.2 * attempts, 1.0))

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
        self._ws = connect(self.client._ws_url(suffix), subprotocols=["realtime"], additional_headers=headers)
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
        self._ws = await websockets.connect(
            self.client._ws_url(suffix), subprotocols=["realtime"], additional_headers=headers
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
