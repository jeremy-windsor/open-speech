"""Wyoming Protocol TCP server for Open Speech.

Runs alongside the FastAPI HTTP server on a separate port (default 10400).
Implements both STT (ASR) and TTS capabilities so Home Assistant can use
Open Speech as a drop-in voice provider.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import suppress
from typing import TYPE_CHECKING

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.asr import Transcribe, Transcript
from wyoming.error import Error
from wyoming.event import Event, async_write_event
from wyoming.info import (
    AsrModel,
    AsrProgram,
    Attribution,
    Describe,
    Info,
    TtsProgram,
    TtsVoice,
)
from wyoming.server import AsyncEventHandler, AsyncTcpServer
from wyoming.tts import Synthesize

from src.backends.parakeet import LanguageUnsupportedError, check_language
from src.config import settings
from src.model_registry import stt_model_languages
from src.services.stt import whisper_language_code
from src.utils.redact import redact_paths
from src.wyoming.stt_handler import audio_format_error, handle_transcribe
from src.wyoming.tts_handler import handle_synthesize

if TYPE_CHECKING:
    from src.router import BackendRouter
    from src.tts.router import TTSRouter

logger = logging.getLogger(__name__)

# Module-level refs set by start_wyoming_server
_stt_router: BackendRouter | None = None
_tts_router: TTSRouter | None = None


# Room in an event's data section for metadata other than Synthesize text.
_METADATA_BYTES = 1024 * 1024
# The longest JSON escape of one character: "\ud83d\ude00".
_MAX_JSON_BYTES_PER_CHAR = 12

# How long a connection closed after an error keeps discarding what the client sends.
_CLOSE_LINGER_SECONDS = 2.0


def max_audio_bytes() -> int:
    """Return how much audio one transcription may buffer (OS_MAX_UPLOAD_MB).

    The port has no authentication, so it is bounded like every HTTP and WebSocket
    audio input. No single event payload may be larger either.
    """
    return settings.os_max_upload_mb * 1024 * 1024


def max_data_bytes() -> int:
    """Return how large an event's JSON data section may be.

    Audio travels as payload, so the data section only needs room for metadata and the
    longest Synthesize text TTS_MAX_INPUT_LENGTH admits, however it is escaped. Parsed
    JSON takes many times its size in memory and parsing runs on the event loop, so the
    audio limit would be far too generous.
    """
    return _METADATA_BYTES + settings.tts_max_input_length * _MAX_JSON_BYTES_PER_CHAR


class EventTooLargeError(Exception):
    """An event header declared more data or payload than the server accepts."""


class ConnectionLimiter:
    """Admits at most OS_WYOMING_MAX_CONNECTIONS connections at once (0 = no limit).

    The port has no authentication, so without a cap any client on the network could hold
    sockets and handler tasks until the server runs out. A cap that refused every new
    connection would let one client lock Home Assistant out with that many idle sockets,
    so at the cap a new connection replaces the one that has waited longest for its next
    event: Home Assistant's connections are short and busy. A new connection is refused
    only while every open one is mid-request.
    """

    def __init__(self) -> None:
        self._open: set[OpenSpeechEventHandler] = set()
        # Connections waiting for their next event, longest-waiting first, with when they began.
        self._waiting: dict[OpenSpeechEventHandler, float] = {}
        self._warned: set[str] = set()  # once per burst: until a connection is admitted under the cap

    def admit(self, handler: OpenSpeechEventHandler) -> bool:
        limit = settings.os_wyoming_max_connections
        if 0 < limit <= len(self._open):
            if not self._waiting:
                self._warn_once(
                    "refuse", "Refusing Wyoming connections: all %d are busy (OS_WYOMING_MAX_CONNECTIONS=%d)",
                    len(self._open), limit,
                )
                return False
            victim, waiting_since = next(iter(self._waiting.items()))
            self._warn_once(
                "evict",
                "%d Wyoming connections are open (OS_WYOMING_MAX_CONNECTIONS=%d): each new one "
                "closes the one that has waited longest for its next event",
                len(self._open), limit,
            )
            logger.debug("Closing a Wyoming connection idle for %.1f s to admit a new one",
                         time.monotonic() - waiting_since)
            self.release(victim)
            victim.evict()
        else:
            self._warned.clear()
        self._open.add(handler)
        return True

    def waiting(self, handler: OpenSpeechEventHandler) -> None:
        if handler in self._open:
            self._waiting[handler] = time.monotonic()

    def busy(self, handler: OpenSpeechEventHandler) -> None:
        self._waiting.pop(handler, None)

    def release(self, handler: OpenSpeechEventHandler) -> None:
        self._open.discard(handler)
        self._waiting.pop(handler, None)

    def _warn_once(self, kind: str, message: str, *args) -> None:
        if kind not in self._warned:
            self._warned.add(kind)
            logger.warning(message, *args)


async def read_event(reader: asyncio.StreamReader) -> Event | None:
    """Read one event as wyoming.event.async_read_event does, refusing oversized ones.

    async_read_event buffers whatever data and payload length a header declares, so a
    single header could make the server hold gigabytes. Returns None at the end of the
    stream, when the connection breaks (any socket error), or for a malformed event; each
    ends the connection.
    """
    try:
        line = await reader.readline()  # bounded by the reader's limit (64 KiB)
        if not line:
            return None
        header = json.loads(line)
        lengths = (header.get("data_length") or 0, header.get("payload_length") or 0)
        if not all(isinstance(length, int) and length >= 0 for length in lengths):
            return None
        data_length, payload_length = lengths
        if payload_length > max_audio_bytes():
            raise EventTooLargeError(
                f"Event payload of {payload_length} bytes exceeds the "
                f"{settings.os_max_upload_mb} MB limit (OS_MAX_UPLOAD_MB)"
            )
        if data_length > max_data_bytes():
            raise EventTooLargeError(
                f"Event data of {data_length} bytes exceeds the {max_data_bytes()}-byte limit"
            )
        data = dict(header.get("data") or {})
        if data_length:
            data.update(json.loads(await reader.readexactly(data_length)))
        payload = await reader.readexactly(payload_length) if payload_length else None
        return Event(type=header["type"], data=data, payload=payload)
    except (
        ValueError, TypeError, KeyError, AttributeError,
        RecursionError,  # deeply nested JSON
        asyncio.IncompleteReadError,
        OSError,  # ConnectionError, and the rest, e.g. ETIMEDOUT or EHOSTUNREACH from the socket
    ):
        return None


class OpenSpeechEventHandler(AsyncEventHandler):
    """Handles a single Wyoming TCP connection.

    A request that fails is answered with an error event, and the connection is then
    closed: Home Assistant ignores error events and waits for a transcript or
    audio-stop until the connection ends.
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        stt_router: BackendRouter,
        tts_router: TTSRouter,
        cli_info: Info,
        limiter: ConnectionLimiter | None = None,
    ) -> None:
        super().__init__(reader, writer)
        self.stt_router = stt_router
        self.tts_router = tts_router
        self.cli_info = cli_info
        self._limiter = limiter
        self._audio_chunks: list[bytes] = []
        self._audio_bytes = 0
        self._audio_rate: int = 16000
        self._audio_width: int = 2
        self._audio_channels: int = 1
        self._transcribe_model: str | None = None
        self._transcribe_language: str | None = None
        self._wait: asyncio.Timeout | None = None  # the deadline of the wait in _next_event
        self._evicted = False

    async def run(self) -> None:
        """Receive events until stopped or handle_event returns False.

        AsyncEventHandler.run, except that a connection over the limiter's cap is closed at
        once, read_event caps each event's size, a client whose next event takes longer
        than OS_WYOMING_IDLE_TIMEOUT_S to arrive is disconnected, and a connection ended
        by an error lingers so the client can read the error event.
        """
        if self._limiter is not None and not self._limiter.admit(self):
            self.writer.close()
            return
        self._is_running = True
        try:
            while self._is_running:
                try:
                    event = await self._next_event()
                except EventTooLargeError as exc:
                    logger.warning("Wyoming client sent an oversized event: %s", exc)
                    await self._reply_error(str(exc), "event_too_large")
                    await self._linger()
                    break
                except TimeoutError:
                    if not self._evicted:  # the limiter logged the eviction
                        logger.info(
                            "Closing a Wyoming connection: no complete event within %s s "
                            "(OS_WYOMING_IDLE_TIMEOUT_S)",
                            settings.os_wyoming_idle_timeout_s,
                        )
                    break
                if event is None or self._evicted:
                    break
                if not await self.handle_event(event):
                    await self._linger()
                    break
        finally:
            await self.disconnect()
            if self._limiter is not None:
                self._limiter.release(self)

    async def _next_event(self) -> Event | None:
        """Wait for the client's next event, at most OS_WYOMING_IDLE_TIMEOUT_S (0 = no limit).

        The deadline covers the whole event, so a client dribbling one in byte by byte is
        cut off too. Only this wait counts: transcription and synthesis run while the
        client waits on the server. Raises TimeoutError when the event did not arrive in
        time, or when the limiter evicted this connection to admit a new one.
        """
        idle_timeout = settings.os_wyoming_idle_timeout_s
        async with asyncio.timeout(idle_timeout if idle_timeout > 0 else None) as self._wait:
            if self._limiter is not None:
                self._limiter.waiting(self)
            try:
                return await read_event(self.reader)
            finally:
                if self._limiter is not None:
                    self._limiter.busy(self)

    def evict(self) -> None:
        """Close this connection, which is waiting in _next_event, to make room for a new one."""
        self._evicted = True
        if not self._wait.expired():  # else its own deadline is closing it already
            self._wait.reschedule(asyncio.get_running_loop().time())  # expires the wait at once

    async def write_event(self, event: Event) -> None:
        """Send an event; a client that stops reading for OS_WYOMING_IDLE_TIMEOUT_S is dropped.

        Otherwise a client that sends a request and never reads the reply would hold the
        handler in drain(), and its connection slot, once the socket buffers fill. Raises
        ConnectionResetError then, as for a client that disconnected.
        """
        idle_timeout = settings.os_wyoming_idle_timeout_s
        try:
            async with asyncio.timeout(idle_timeout if idle_timeout > 0 else None):
                await async_write_event(event, self.writer)
        except TimeoutError:
            self.writer.transport.abort()  # later writes fail at once instead of waiting again
            raise ConnectionResetError(
                f"Wyoming client stopped reading for {idle_timeout} s (OS_WYOMING_IDLE_TIMEOUT_S)"
            ) from None

    async def _linger(self) -> None:
        """Half-close, then discard what the client still sends, for a bounded time.

        Closing a socket with unread input resets the connection, and the reset makes
        the client drop the error event it has not read yet: a client still streaming
        audio would never see why it was cut off. The half-close tells the client the
        reply is complete; a client that then closes its side ends the wait early.
        """
        with suppress(OSError, TimeoutError):
            if self.writer.can_write_eof():
                self.writer.write_eof()
            async with asyncio.timeout(_CLOSE_LINGER_SECONDS):
                while await self.reader.read(64 * 1024):
                    pass

    async def disconnect(self) -> None:
        # wyoming's run() returns without closing the connection, which would leave the
        # socket open and a client waiting for a reply that never comes.
        self.writer.close()

    async def handle_event(self, event: Event) -> bool:
        """Process a Wyoming event. Return True to keep connection open."""
        try:
            return await self._handle_event(event)
        except ConnectionError:
            logger.info("Wyoming client disconnected during %s", event.type)
            return False
        except Exception as exc:
            logger.exception("Wyoming %s event failed", event.type)
            return await self._reply_error(f"{event.type} failed: {redact_paths(str(exc))}", type(exc).__name__)

    async def _reply_error(self, text: str, code: str | None = None) -> bool:
        """Send an error event; returns False, which ends the connection."""
        with suppress(OSError):  # the client may already be gone
            await self.write_event(Error(text=text, code=code).event())
        return False

    def _reset_audio(self) -> None:
        self._audio_chunks = []
        self._audio_bytes = 0

    async def _handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.cli_info.event())
            return True

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            self._reset_audio()
            language = getattr(transcribe, "language", None)
            if language is not None and not isinstance(language, str):
                return await self._reply_error("Transcription language must be a string", "invalid_language")
            try:
                # As HTTP does: faster-whisper would check the code only after decoding.
                language = whisper_language_code(language)
            except ValueError as exc:
                return await self._reply_error(str(exc), "invalid_language")
            self._transcribe_model = getattr(transcribe, "name", None)
            try:
                # Parakeet transcribes English only; its Info lists English alone.
                check_language(self._transcribe_model or settings.stt_model, language)
            except LanguageUnsupportedError as exc:
                return await self._reply_error(str(exc), exc.code)
            self._transcribe_language = language
            return True

        if AudioStart.is_type(event.type):
            start = AudioStart.from_event(event)
            format_error = audio_format_error(start.rate, start.width, start.channels)
            if format_error:
                self._reset_audio()
                return await self._reply_error(format_error, "unsupported_audio")
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            format_error = audio_format_error(chunk.rate, chunk.width, chunk.channels)
            if format_error:
                self._reset_audio()
                return await self._reply_error(format_error, "unsupported_audio")
            self._audio_bytes += len(chunk.audio)
            if self._audio_bytes > max_audio_bytes():
                self._reset_audio()
                return await self._reply_error(
                    f"Audio exceeds the {settings.os_max_upload_mb} MB limit (OS_MAX_UPLOAD_MB)",
                    "audio_too_large",
                )
            self._audio_rate = chunk.rate
            self._audio_width = chunk.width
            self._audio_channels = chunk.channels
            self._audio_chunks.append(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            audio_chunks = self._audio_chunks
            self._reset_audio()
            try:
                text = await handle_transcribe(
                    audio_chunks=audio_chunks,
                    rate=self._audio_rate,
                    width=self._audio_width,
                    channels=self._audio_channels,
                    stt_router=self.stt_router,
                    model=self._transcribe_model,
                    language=self._transcribe_language,
                )
            except Exception as exc:
                logger.exception("Wyoming STT transcription failed")
                return await self._reply_error(f"STT failed: {redact_paths(str(exc))}", type(exc).__name__)
            # Answer even when no audio arrived; the client waits for a transcript.
            await self.write_event(Transcript(text=text).event())
            return True

        if Synthesize.is_type(event.type):
            synthesize = Synthesize.from_event(event)
            voice_name = None
            if synthesize.voice and synthesize.voice.name:
                voice_name = synthesize.voice.name

            # False after an error event, which ends the connection as above.
            return await handle_synthesize(
                text=synthesize.text,
                voice=voice_name,
                tts_router=self.tts_router,
                write_event=self.write_event,
            )

        logger.debug("Unhandled Wyoming event type: %s", event.type)
        return True


def build_info(stt_router: BackendRouter, tts_router: TTSRouter) -> Info:
    """Build Wyoming Info describing our STT + TTS capabilities."""

    attribution = Attribution(name="Open Speech", url="https://github.com/jeremy-windsor/open-speech")

    # ASR (STT) programs
    stt_models = []
    loaded = stt_router.loaded_models()
    loaded_ids = {m.model for m in loaded}
    for m in loaded:
        stt_models.append(AsrModel(
            name=m.model,
            attribution=attribution,
            installed=True,
            description=f"STT model ({m.backend})",
            version=None,
            languages=stt_model_languages(m.model),
        ))
    if settings.stt_model not in loaded_ids:
        stt_models.append(AsrModel(
            name=settings.stt_model,
            attribution=attribution,
            installed=True,
            description="Default STT model",
            version=None,
            languages=stt_model_languages(settings.stt_model),
        ))

    asr_programs = [
        AsrProgram(
            name="open-speech",
            attribution=attribution,
            installed=True,
            description="Open Speech STT",
            version=None,
            models=stt_models,
        )
    ]

    # TTS programs
    tts_voices = []
    if settings.tts_enabled:
        try:
            voices = tts_router.list_voices(settings.tts_model)
        except Exception:
            logger.warning(
                "Could not list voices for default TTS model %s",
                settings.tts_model,
            )
            voices = []
        for v in voices:
            langs = [v.language] if v.language else ["en"]
            tts_voices.append(TtsVoice(
                name=v.id,
                attribution=attribution,
                installed=True,
                description=v.name,
                version=None,
                languages=langs,
            ))
        if not tts_voices:
            tts_voices.append(TtsVoice(
                name=settings.tts_voice,
                attribution=attribution,
                installed=True,
                description="Default voice",
                version=None,
                languages=["en"],
            ))

    tts_programs = [
        TtsProgram(
            name="open-speech",
            attribution=attribution,
            installed=True,
            description="Open Speech TTS",
            version=None,
            voices=tts_voices,
        )
    ] if settings.tts_enabled else []

    return Info(asr=asr_programs, tts=tts_programs)


async def start_wyoming_server(
    host: str,
    port: int,
    stt_router: BackendRouter,
    tts_router: TTSRouter,
) -> asyncio.Task:
    """Start the Wyoming TCP server as an asyncio task. Returns the task."""
    info = await asyncio.to_thread(build_info, stt_router, tts_router)

    server = AsyncTcpServer(host, port)
    handler_factory = make_handler_factory(stt_router, tts_router, info)
    task = asyncio.create_task(_run_server(server, handler_factory), name="wyoming-server")
    logger.info("Wyoming protocol server started on %s:%d", host, port)
    return task


def make_handler_factory(stt_router: BackendRouter, tts_router: TTSRouter, info: Info):
    """Return the factory wyoming's server calls for each connection; they share one cap."""
    limiter = ConnectionLimiter()

    def handler_factory(reader, writer):
        return OpenSpeechEventHandler(reader, writer, stt_router, tts_router, info, limiter=limiter)

    return handler_factory


async def _run_server(server: AsyncTcpServer, handler_factory) -> None:
    """Serve until cancelled, then close the listener and every open connection.

    Not AsyncTcpServer.run: cancelling its serve_forever waits until every client has
    disconnected, which a connected Home Assistant may not do for minutes.
    """
    try:
        await server.start(handler_factory)
        await asyncio.get_running_loop().create_future()  # until cancelled
    except asyncio.CancelledError:
        logger.info("Wyoming server shutting down")
        await server.stop()
    except Exception:
        logger.exception("Wyoming server error")
