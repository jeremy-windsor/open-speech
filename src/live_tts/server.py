"""Bidirectional incremental text-to-speech WebSocket server."""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import contextlib
import json
import logging
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from src.audio.postprocessing import StreamingEdgeTrimmer
from src.live_tts.segmenter import LiveTextSegmenter, TextSegment, split_sentences
from src.tts.external import ExternalProviderError
from src.tts.pipeline import float32_to_int16

logger = logging.getLogger(__name__)

MAX_AUDIO_SECONDS_PER_SEGMENT = 180

_live_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="live-tts",
)


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


class LiveTTSAdmission:
    """Process-local admission control that also exposes teardown state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[str, str] = {}

    def acquire(self, limit: int) -> tuple[str | None, str | None]:
        with self._lock:
            if len(self._states) >= max(1, limit):
                reason = "draining" if "draining" in self._states.values() else "active"
                return None, reason
            token = _id("live")
            self._states[token] = "active"
            return token, None

    def mark_draining(self, token: str) -> None:
        with self._lock:
            if token in self._states:
                self._states[token] = "draining"

    def release(self, token: str) -> None:
        with self._lock:
            self._states.pop(token, None)

    def active(self) -> bool:
        with self._lock:
            return bool(self._states)

    def reset(self) -> None:
        """Clear admission state after isolated tests."""
        with self._lock:
            self._states.clear()


_admission = LiveTTSAdmission()


def live_session_active() -> bool:
    """Return whether a Live Reader socket currently holds or is releasing its slot."""
    return _admission.active()


@dataclass(slots=True)
class PlaybackTracker:
    durations: deque[tuple[int, float]] = field(default_factory=deque)
    acknowledged_sequence: int = -1
    output_done: bool = False

    @property
    def unacknowledged_seconds(self) -> float:
        return sum(duration for _sequence, duration in self.durations)


def _queue_from_worker(
    loop: asyncio.AbstractEventLoop,
    output_queue: asyncio.Queue,
    item: tuple[Any, ...],
) -> None:
    try:
        loop.call_soon_threadsafe(output_queue.put_nowait, item)
    except RuntimeError:
        # The owning event loop can disappear during process shutdown.
        return


def _synthesize_worker(
    *,
    loop: asyncio.AbstractEventLoop,
    output_queue: asyncio.Queue,
    cancel_event: threading.Event,
    tts_router,
    text: str | list[str],
    model: str,
    voice: str,
    speed: float,
    language: str | None,
    sample_rate: int,
    frame_ms: int,
    instructions: str | None = None,
    reference_audio: bytes | None = None,
    clone_transcript: str | None = None,
    trim_silence: bool = False,
) -> None:
    """Own each router generator for its entire lifetime on this one thread.

    ``text`` may be a list of pieces of one response, spoken in order as one
    audio stream with continuing sequence numbers.
    """
    pieces = [text] if isinstance(text, str) else list(text)
    chunks = None
    sequence = 0
    frame_samples = max(1, int(sample_rate * frame_ms / 1000))
    max_frames = max(400, int(MAX_AUDIO_SECONDS_PER_SEGMENT * 1000 / frame_ms))

    def emit_samples(samples: np.ndarray) -> None:
        nonlocal sequence
        for offset in range(0, len(samples), frame_samples):
            if cancel_event.is_set():
                break
            if sequence >= max_frames:
                raise RuntimeError(
                    f"Live TTS backend exceeded the {MAX_AUDIO_SECONDS_PER_SEGMENT}-second "
                    "segment guard"
                )
            frame = samples[offset : offset + frame_samples]
            pcm = float32_to_int16(frame).astype("<i2", copy=False).tobytes()
            _queue_from_worker(
                loop,
                output_queue,
                ("audio", sequence, pcm, len(frame) / sample_rate),
            )
            sequence += 1

    def close_chunks() -> None:
        nonlocal chunks
        close = getattr(chunks, "close", None)
        chunks = None
        if callable(close):
            try:
                close()
            except Exception:
                logger.exception("Failed to close live TTS generator")

    try:
        # Do not move this generator onto the event-loop thread. TTSRouter holds
        # an RLock while it is being iterated, and that same thread must close it.
        backend_options: dict[str, Any] = {}
        if instructions:
            backend_options["instructions"] = instructions
        if reference_audio is not None:
            backend_options["reference_audio"] = reference_audio
        if clone_transcript:
            backend_options["clone_transcript"] = clone_transcript
        trimmer = StreamingEdgeTrimmer(sample_rate) if trim_silence else None
        for piece in pieces:
            if cancel_event.is_set():
                break
            chunks = tts_router.synthesize(
                text=piece,
                model=model,
                voice=voice,
                speed=speed,
                lang_code=language,
                **backend_options,
            )
            chunks_iter = iter(chunks)
            while not cancel_event.is_set():
                try:
                    chunk = next(chunks_iter)
                except StopIteration:
                    break
                samples = np.asarray(chunk, dtype=np.float32).reshape(-1)
                if not samples.size:
                    continue
                samples = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
                if trimmer is not None:
                    samples = trimmer.push(samples)
                if not samples.size:
                    continue
                emit_samples(samples)
            close_chunks()
        if trimmer is not None and not cancel_event.is_set():
            emit_samples(trimmer.finish())
        _queue_from_worker(
            loop,
            output_queue,
            ("done", sequence, threading.get_ident()),
        )
    except Exception as exc:
        logger.exception("Live TTS synthesis failed")
        # A provider's typed error carries its client-facing message, as /v1/audio/speech returns it (for example
        # a worker's language refusal); any other exception's text stays private.
        message = (exc.message if isinstance(exc, ExternalProviderError) else "") or "Live TTS synthesis failed"
        _queue_from_worker(loop, output_queue, ("error", message))
    finally:
        if chunks is not None:
            close_chunks()


class LiveTTSSession:
    """One incremental reader connection."""

    def __init__(
        self,
        websocket: WebSocket,
        *,
        tts_router,
        pronunciation_dict,
        settings,
        voice_library=None,
    ) -> None:
        self.ws = websocket
        self.tts_router = tts_router
        self.pronunciation_dict = pronunciation_dict
        self.settings = settings
        self.voice_library = voice_library
        self.id = _id("sess")
        self.model = settings.tts_model
        self.voice = settings.tts_voice
        self.speed = settings.tts_speed
        # False while self.speed is TTS_SPEED, the session's default, rather than a speed the client sent.
        self._speed_from_client = False
        self.language: str | None = None
        self.instructions: str | None = None
        self.voice_library_ref: str | None = None
        self.reference_audio: bytes | None = None
        self.clone_transcript: str | None = None
        self.latency_mode = "natural"
        # Read from the router by resolve_model_limits, off the event loop: for an isolated
        # worker each is a manifest fetch.
        self.sample_rate = 24000
        self.max_input_chars: int | None = None

        self._send_lock = asyncio.Lock()
        self._input_queue: asyncio.Queue[tuple[int, str, str]] = asyncio.Queue()
        self._segment_queue: asyncio.Queue[tuple[int, TextSegment | None] | None] = asyncio.Queue(
            maxsize=max(1, settings.tts_live_max_pending_segments)
        )
        self._segmenter = self._new_segmenter()
        self._segmenter_task: asyncio.Task | None = None
        self._synthesis_task: asyncio.Task | None = None
        self._idle_task: asyncio.Task | None = None
        self._status_task: asyncio.Task | None = None
        self._worker_future: asyncio.Future | None = None
        self._worker_cancel = threading.Event()
        self._flow_changed = asyncio.Event()
        self._playback: dict[str, PlaybackTracker] = {}
        self._generation = 0
        self._accepted_chars = 0
        self._buffered_input_chars = 0
        self._queued_commits = 0
        self._active_response_id: str | None = None
        self._paused_at: float | None = None
        self._closing = False
        self._disconnected = False
        self._last_status_payload: tuple[int, int, float] | None = None
        self._last_status_sent_at = 0.0

    def _new_segmenter(self) -> LiveTextSegmenter:
        if self.latency_mode == "natural":
            max_chars = getattr(
                self.settings,
                "tts_live_sentence_max_chars",
                self.settings.tts_live_max_segment_chars,
            )
            max_words = getattr(
                self.settings,
                "tts_live_sentence_max_words",
                self.settings.tts_live_max_segment_words,
            )
        else:
            max_chars = self.settings.tts_live_max_segment_chars
            max_words = self.settings.tts_live_max_segment_words
        if self.max_input_chars:
            # A segment the model refuses would never be spoken (Chatterbox takes 350 characters).
            max_chars = min(max_chars, self.max_input_chars)
        return LiveTextSegmenter(
            mode=self.latency_mode,
            max_chars=max_chars,
            max_words=max_words,
        )

    async def resolve_model_limits(self) -> None:
        """Read the model's rate and input limit off the event loop, as session.update does."""
        self.sample_rate = await asyncio.to_thread(self._sample_rate_for, self.model)
        self.max_input_chars = await asyncio.to_thread(self._max_input_chars_for, self.model)
        self._segmenter = self._new_segmenter()
        if not self._speed_from_client and self.speed != 1.0 and not await asyncio.to_thread(
            self._has_speed_control, self.model
        ):
            # TTS_SPEED applies only where the model has speed control, in the start state too. A session that sends
            # no session.update has omitted speed, so a fixed-speed TTS_MODEL speaks at 1.0 rather than at TTS_SPEED.
            self.speed = 1.0

    def _has_speed_control(self, model: str) -> bool:
        """Whether the model can change speed; True when the router can't say, which leaves the speed as it is."""
        if not callable(getattr(self.tts_router, "get_backend", None)):
            return True
        from src.services.tts import tts_capabilities

        try:
            capabilities = tts_capabilities(tts_router=self.tts_router, model_id=model)
        except Exception:
            return True   # an unknown model: synthesis names the problem, as before
        return bool(capabilities.get("speed_control", False))

    def _sample_rate_for(self, model: str) -> int:
        try:
            return int(self.tts_router.sample_rate_for(model) or 24000)
        except Exception:
            return 24000

    def _max_input_chars_for(self, model: str) -> int | None:
        get_limit = getattr(self.tts_router, "max_input_chars_for", None)
        try:
            limit = get_limit(model) if callable(get_limit) else None
        except Exception:
            return None
        return limit if type(limit) is int and limit > 0 else None

    def _fit_input_limit(self, text: str) -> list[str]:
        """Split text the model would refuse, as the Reader renderer does.

        The segmenter caps segments at max_input_chars, but the pronunciation
        dictionary runs afterwards and can lengthen one.
        """
        limit = self.max_input_chars
        if not limit or len(text) <= limit:
            return [text]
        pieces = [text[start:end] for start, end in split_sentences(text, limit)]
        # split_sentences keeps a grapheme cluster whole, which can pass the limit by a character.
        return [piece[offset : offset + limit] for piece in pieces for offset in range(0, len(piece), limit)]

    async def _send(self, event: dict[str, Any]) -> bool:
        if self._disconnected:
            return False
        try:
            async with self._send_lock:
                await self.ws.send_json(event)
            return True
        except Exception:
            self._disconnected = True
            self._worker_cancel.set()
            self._flow_changed.set()
            return False

    def _session_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": "open_speech.live_tts_session",
            "model": self.model,
            "voice": self.voice,
            "speed": self.speed,
            "language": self.language,
            "instructions": self.instructions,
            "voice_library_ref": self.voice_library_ref,
            "latency_mode": self.latency_mode,
            "audio": {
                "format": "pcm16",
                "encoding": "signed little-endian",
                "sample_rate": self.sample_rate,
                "channels": 1,
            },
            "limits": {
                "max_buffer_chars": self.settings.tts_live_max_buffer_chars,
                "max_pending_segments": self.settings.tts_live_max_pending_segments,
                "max_unacked_seconds": self.settings.tts_live_max_unacked_seconds,
                "max_pause_seconds": self.settings.tts_live_max_pause_s,
            },
        }

    async def run(self) -> None:
        await self.resolve_model_limits()  # session.created reports the model's own rate
        self._segmenter_task = asyncio.create_task(
            self._segment_input_loop(), name=f"live-segmenter-{self.id}"
        )
        self._synthesis_task = asyncio.create_task(
            self._synthesis_loop(), name=f"live-synthesis-{self.id}"
        )
        await self._send({"type": "session.created", "session": self._session_payload()})

        while not self._closing and not self._disconnected:
            try:
                message = await asyncio.wait_for(
                    self.ws.receive(),
                    timeout=self.settings.tts_live_idle_timeout_s,
                )
            except asyncio.TimeoutError:
                await self._send_error("Session idle timeout", "idle_timeout")
                await self.ws.close(code=4008, reason="Session idle timeout")
                break
            except WebSocketDisconnect:
                self._disconnected = True
                break
            except RuntimeError:
                # A concurrent send can discover the closed socket first and
                # move Starlette out of its connected state. In that case the
                # pending receive raises RuntimeError instead of WebSocketDisconnect.
                if self._disconnected:
                    break
                raise
            if message["type"] == "websocket.disconnect":
                self._disconnected = True
                break

            raw = message.get("text")
            if raw is None:
                await self._send_error("Events must be JSON text frames, not binary", "invalid_event")
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await self._send_error("Invalid JSON", "invalid_json")
                continue
            if not isinstance(data, dict) or not isinstance(data.get("type"), str):
                await self._send_error(
                    "Event must be a JSON object with a 'type' field",
                    "invalid_event",
                )
                continue
            if await self._pause_expired():
                break
            await self._handle_event(data)

    @property
    def worker_future(self) -> asyncio.Future | None:
        return self._worker_future

    async def shutdown(self) -> bool:
        """Stop session tasks within a bounded wait.

        Returns true only when the backend worker is no longer running. A false
        result keeps admission in the draining state until its future finishes.
        """
        self._closing = True
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.settings.tts_live_shutdown_timeout_s
        self._worker_cancel.set()
        self._flow_changed.set()
        if self._idle_task:
            self._idle_task.cancel()
        if self._status_task:
            self._status_task.cancel()
        await asyncio.gather(
            *(task for task in (self._idle_task, self._status_task) if task is not None),
            return_exceptions=True,
        )
        if self._segmenter_task:
            self._segmenter_task.cancel()
            await asyncio.gather(self._segmenter_task, return_exceptions=True)
        self._drain_queue(self._segment_queue)
        try:
            self._segment_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass
        if self._synthesis_task:
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._synthesis_task),
                    timeout=max(0, deadline - loop.time()),
                )
            except asyncio.TimeoutError:
                logger.error("Live TTS synthesis task exceeded the shutdown deadline")
                self._synthesis_task.cancel()
                await asyncio.gather(self._synthesis_task, return_exceptions=True)
        worker_future = self._worker_future
        if worker_future and not worker_future.done():
            remaining = max(0, deadline - loop.time())
            if not remaining:
                logger.error("Live TTS backend worker is still draining after shutdown")
                return False
            try:
                await asyncio.wait_for(
                    asyncio.shield(worker_future),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                logger.error("Live TTS backend worker is still draining after shutdown")
                return False
        return True

    async def _handle_event(self, data: dict[str, Any]) -> None:
        event_type = data["type"]
        handlers: dict[str, Callable[[dict[str, Any]], Any]] = {
            "session.update": self._handle_session_update,
            "input_text.append": self._handle_input_append,
            "input_text.commit": self._handle_input_commit,
            "input_text.context": self._handle_input_context,
            "response.cancel": self._handle_cancel,
            "playback.ack": self._handle_playback_ack,
            "playback.pause": self._handle_playback_pause,
            "playback.resume": self._handle_playback_resume,
            "session.keepalive": self._handle_keepalive,
        }
        handler = handlers.get(event_type)
        if handler is None:
            await self._send_error(f"Unknown event type: {event_type}", "unknown_event")
            return
        try:
            await handler(data)
        except (TypeError, ValueError, ExternalProviderError) as exc:
            await self._send_error(str(exc), "invalid_event")

    async def _handle_session_update(self, data: dict[str, Any]) -> None:
        self._discard_stale_segments()
        if self._accepted_chars or self._active_response_id or not self._segment_queue.empty():
            await self._send_error(
                "Session voice/model settings can only change before text is accepted or after cancel",
                "session_busy",
            )
            return
        update = data.get("session")
        if not isinstance(update, dict):
            raise ValueError("session.update requires a session object")
        model = update.get("model", self.model)
        voice = update.get("voice", self.voice)
        language = update.get("language", self.language)
        instructions = update.get("instructions", self.instructions)
        voice_library_ref = update.get("voice_library_ref", self.voice_library_ref)
        latency_mode = update.get("latency_mode", self.latency_mode)
        speed_sent = "speed" in update
        speed_is_default = not speed_sent and not self._speed_from_client
        try:
            speed = float(update.get("speed", self.settings.tts_speed if speed_is_default else self.speed))
        except (TypeError, ValueError) as exc:
            raise ValueError("speed must be a number") from exc
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(voice, str) or not voice.strip():
            raise ValueError("voice must be a non-empty string")
        if language is not None and not isinstance(language, str):
            raise ValueError("language must be a string or null")
        if instructions is not None and not isinstance(instructions, str):
            raise ValueError("instructions must be a string or null")
        if voice_library_ref is not None and not isinstance(voice_library_ref, str):
            raise TypeError("voice_library_ref must be a string or null")
        if isinstance(voice_library_ref, str):
            voice_library_ref = voice_library_ref.strip() or None
        if latency_mode not in LiveTextSegmenter.VALID_MODES:
            raise ValueError("latency_mode must be natural, responsive, or instant_word")
        if not 0.25 <= speed <= 4.0:
            raise ValueError("speed must be between 0.25 and 4.0")

        from src.voice_identities import resolve_named_voice

        try:
            voice, voice_library_ref = await asyncio.to_thread(
                resolve_named_voice, model.strip(), voice.strip(), voice_library_ref
            )
        except KeyError as exc:
            raise ValueError("Named voice is unavailable") from exc

        reference_audio: bytes | None = None
        clone_transcript: str | None = None
        if voice_library_ref:
            if self.voice_library is None:
                raise ValueError("Voice library is unavailable")
            try:
                reference_audio, metadata = await asyncio.to_thread(
                    self.voice_library.get, voice_library_ref
                )
            except (KeyError, OSError, ValueError) as exc:
                raise ValueError(f"Voice reference '{voice_library_ref}' is unavailable") from exc
            clone_transcript = metadata.get("transcript")

        if callable(getattr(self.tts_router, "get_backend", None)):
            from src.services.tts import tts_capabilities, validate_tts_feature_support

            # TTS_SPEED applies only where the model has speed control; a fixed-speed model (pocket-tts, chatterbox)
            # speaks at 1.0. A speed the client sent is still checked below.
            if speed_is_default and speed != 1.0:
                try:
                    capabilities = await asyncio.to_thread(
                        tts_capabilities, tts_router=self.tts_router, model_id=model.strip()
                    )
                except ValueError:
                    capabilities = {"speed_control": True}  # an unknown model: the check below names it
                if not capabilities.get("speed_control", False):
                    speed = 1.0

            feature_error = await asyncio.to_thread(
                validate_tts_feature_support,
                tts_router=self.tts_router,
                model_id=model.strip(),
                instructions=instructions.strip() if instructions else None,
                reference_audio=reference_audio,
                clone_transcript=clone_transcript,
                speed=speed,
                voice=voice.strip(),
                live_reader=True,
                # Otherwise every segment would fail at synthesis instead.
                language=language.strip() if language else None,
            )
            if feature_error:
                raise ValueError(feature_error)

        self.model = model.strip()
        self.voice = voice.strip()
        self.speed = speed
        self._speed_from_client = self._speed_from_client or speed_sent
        self.language = language.strip() if language else None
        self.instructions = instructions.strip() if instructions else None
        self.voice_library_ref = voice_library_ref
        self.reference_audio = reference_audio
        self.clone_transcript = clone_transcript
        self.latency_mode = latency_mode
        self.sample_rate = await asyncio.to_thread(self._sample_rate_for, self.model)
        self.max_input_chars = await asyncio.to_thread(self._max_input_chars_for, self.model)
        self._segmenter = self._new_segmenter()
        await self._send({"type": "session.updated", "session": self._session_payload()})

    async def _handle_input_append(self, data: dict[str, Any]) -> None:
        text = data.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("input_text.append requires non-empty text")
        encoded_length = len(text.encode("utf-8"))
        if encoded_length > self.settings.tts_live_max_buffer_chars * 4:
            raise ValueError("One input_text.append event is too large")
        if self._buffered_input_chars + len(text) > self.settings.tts_live_max_buffer_chars:
            await self._send_error(
                "Live TTS input buffer is full; wait for session.status before retrying",
                "buffer_overflow",
            )
            await self._send_status(force=True)
            return
        self._buffered_input_chars += len(text)
        self._accepted_chars += len(text)
        self._input_queue.put_nowait((self._generation, "append", text))
        self._schedule_idle_flush()
        await self._send(
            {
                "type": "input_text.accepted",
                "accepted_chars": self._accepted_chars,
                "buffered_chars": self._buffered_input_chars,
                "generation": self._generation,
            }
        )

    async def _handle_input_context(self, data: dict[str, Any]) -> None:
        """``input_text.context``: the source text just before where the next append starts, from its line's
        start. It was read already, so none of it is spoken or counted in ``accepted_chars`` or the source
        offsets; the next text is cleaned as its continuation, so a client resending from mid-line after
        ``response.cancel`` keeps a leading "12." or ">". Only before a generation's first ``input_text.append`` (after
        ``session.updated`` or ``response.cancelled``); an empty text means a line starts there, as it does by default.
        There is no reply unless it is refused."""
        text = data.get("text")
        if not isinstance(text, str):
            raise ValueError("input_text.context requires text (a string; empty means a line starts here)")
        if len(text.encode("utf-8")) > self.settings.tts_live_max_buffer_chars * 4:
            raise ValueError("One input_text.context event is too large")
        if self._accepted_chars:
            raise ValueError(
                "input_text.context must come before the first input_text.append, "
                "after session.updated or response.cancelled"
            )
        # No text of this generation is queued, so the segmenter is fresh and no queued event reads it first.
        self._segmenter.continue_from(text)

    async def _handle_input_commit(self, _data: dict[str, Any]) -> None:
        self._cancel_idle_flush()
        # Flow control holds commits in the input queue as it holds appended text, so they
        # share its bound (TTS_LIVE_MAX_BUFFER_CHARS), one unit each.
        if self._queued_commits >= self.settings.tts_live_max_buffer_chars:
            await self._send_error(
                "Live TTS input buffer is full; wait for session.status before retrying",
                "buffer_overflow",
            )
            await self._send_status(force=True)
            return
        self._queued_commits += 1
        self._input_queue.put_nowait((self._generation, "commit", ""))
        await self._send(
            {
                "type": "input_text.committed",
                "accepted_chars": self._accepted_chars,
                "generation": self._generation,
            }
        )

    async def _handle_cancel(self, _data: dict[str, Any]) -> None:
        cancelled_generation = self._generation
        cancelled_response_id = self._active_response_id
        # The cancelled response may still drain on the worker, but it no longer
        # holds the session: its output is dropped and settings may change now.
        self._active_response_id = None
        self._generation += 1
        self._worker_cancel.set()
        self._flow_changed.set()
        self._cancel_idle_flush()
        self._drain_queue(self._input_queue)
        self._drain_queue(self._segment_queue)
        self._buffered_input_chars = 0
        self._queued_commits = 0
        self._accepted_chars = 0
        self._segmenter = self._new_segmenter()
        self._playback.clear()
        await self._send(
            {
                "type": "response.cancelled",
                "generation": cancelled_generation,
                "accepted_chars": 0,
                "response_id": cancelled_response_id,
            }
        )

    async def _handle_playback_ack(self, data: dict[str, Any]) -> None:
        response_id = data.get("response_id")
        sequence = data.get("sequence")
        if not isinstance(response_id, str) or not isinstance(sequence, int):
            raise ValueError("playback.ack requires response_id and integer sequence")
        tracker = self._playback.get(response_id)
        if tracker is None or not tracker.durations or sequence <= tracker.acknowledged_sequence:
            return
        sequence = min(sequence, tracker.durations[-1][0])
        tracker.acknowledged_sequence = sequence
        while tracker.durations and tracker.durations[0][0] <= sequence:
            tracker.durations.popleft()
        if tracker.output_done and not tracker.durations:
            self._playback.pop(response_id, None)
        self._flow_changed.set()

    async def _handle_playback_pause(self, _data: dict[str, Any]) -> None:
        if self._paused_at is None:
            self._paused_at = time.monotonic()
        self._flow_changed.set()
        await self._send({"type": "playback.paused"})

    async def _handle_playback_resume(self, _data: dict[str, Any]) -> None:
        self._paused_at = None
        self._flow_changed.set()
        await self._send({"type": "playback.resumed"})

    async def _handle_keepalive(self, _data: dict[str, Any]) -> None:
        await self._send({"type": "session.keepalive.ack"})

    async def _pause_expired(self) -> bool:
        if self._paused_at is None:
            return False
        if time.monotonic() - self._paused_at <= self.settings.tts_live_max_pause_s:
            return False
        await self._send_error("Maximum continuous pause exceeded", "pause_timeout")
        await self.ws.close(code=4008, reason="Maximum continuous pause exceeded")
        self._closing = True
        return True

    def _schedule_idle_flush(self) -> None:
        self._cancel_idle_flush()

        async def handle_idle() -> None:
            try:
                idle_ms = (
                    getattr(
                        self.settings,
                        "tts_live_sentence_idle_ms",
                        self.settings.tts_live_segment_idle_ms,
                    )
                    if self.latency_mode == "natural"
                    else self.settings.tts_live_segment_idle_ms
                )
                await asyncio.sleep(idle_ms / 1000)
                self._input_queue.put_nowait((self._generation, "idle", ""))
            except asyncio.CancelledError:
                return

        self._idle_task = asyncio.create_task(handle_idle(), name=f"live-idle-{self.id}")

    def _cancel_idle_flush(self) -> None:
        if self._idle_task:
            self._idle_task.cancel()
            self._idle_task = None

    async def _segment_input_loop(self) -> None:
        try:
            while True:
                item = await self._input_queue.get()
                generation, action, text = item
                if action == "commit":
                    self._queued_commits = max(0, self._queued_commits - 1)
                if generation != self._generation:
                    continue
                if action == "append":
                    self._buffered_input_chars = max(0, self._buffered_input_chars - len(text))
                    segments = self._segmenter.append(text)
                elif action == "commit":
                    segments = self._segmenter.flush()
                else:
                    segments = self._segmenter.flush(idle=True)
                await self._send_status()
                queued: list[tuple[int, TextSegment | None]] = [(generation, segment) for segment in segments]
                if action == "commit":
                    queued.append((generation, None))
                for queued_item in queued:
                    # A cancel drains the queue, which wakes a waiting put; the rest of
                    # the cancelled text must not follow it in.
                    if generation != self._generation:
                        break
                    await self._segment_queue.put(queued_item)
        except asyncio.CancelledError:
            return

    async def _synthesis_loop(self) -> None:
        while True:
            item = await self._segment_queue.get()
            if item is None:
                return
            generation, segment = item
            if generation != self._generation or self._closing:
                continue
            if not await self._wait_for_flow():
                return
            if generation != self._generation or self._closing:
                continue
            if segment is None:
                await self._send(
                    {
                        "type": "input_text.done",
                        "accepted_chars": self._accepted_chars,
                        "generation": generation,
                    }
                )
                continue
            # What is spoken is what the dictionary leaves: a segment whose every word it removes has nothing to
            # say, and the GPU workers and Pocket-TTS refuse an empty text.
            spoken_pieces = (
                [piece for piece in self._fit_input_limit(self.pronunciation_dict.apply(segment.text)) if piece.strip()]
                if segment.text
                else []
            )
            if not spoken_pieces:
                await self._send(
                    {
                        "type": "input_text.skipped",
                        "source_start": segment.source_start,
                        "source_end": segment.source_end,
                        "generation": generation,
                    }
                )
                continue
            await self._synthesize_segment(generation, segment, spoken_pieces)
            await self._send_status()

    async def _synthesize_segment(self, generation: int, segment: TextSegment, spoken_pieces: list[str]) -> None:
        response_id = _id("resp")
        self._active_response_id = response_id
        self._worker_cancel = threading.Event()
        tracker = PlaybackTracker()
        self._playback[response_id] = tracker
        await self._send(
            {
                "type": "response.created",
                "response": {
                    "id": response_id,
                    "status": "in_progress",
                    "generation": generation,
                    "source_start": segment.source_start,
                    "source_end": segment.source_end,
                    "text": segment.text,
                },
            }
        )

        loop = asyncio.get_running_loop()
        output_queue: asyncio.Queue[tuple[Any, ...]] = asyncio.Queue()
        self._worker_future = loop.run_in_executor(
            _live_executor,
            lambda: _synthesize_worker(
                loop=loop,
                output_queue=output_queue,
                cancel_event=self._worker_cancel,
                tts_router=self.tts_router,
                text=spoken_pieces,
                model=self.model,
                voice=self.voice,
                speed=self.speed,
                language=self.language,
                instructions=self.instructions,
                reference_audio=self.reference_audio,
                clone_transcript=self.clone_transcript,
                sample_rate=self.sample_rate,
                frame_ms=self.settings.tts_live_audio_frame_ms,
                trim_silence=bool(getattr(self.settings, "tts_trim_silence", True)),
            ),
        )

        status = "completed"
        first_delta = True
        while True:
            worker_item = await output_queue.get()
            kind = worker_item[0]
            if kind == "error":
                status = "failed"
                await self._send_error(
                    worker_item[1],
                    "tts_error",
                    response_id=response_id,
                )
                break
            if kind == "done":
                break
            _kind, sequence, pcm, duration = worker_item
            if generation != self._generation or self._worker_cancel.is_set():
                status = "cancelled"
                continue
            if not await self._wait_for_flow():
                status = "cancelled"
                self._worker_cancel.set()
                break
            if generation != self._generation or self._closing:
                status = "cancelled"
                self._worker_cancel.set()
                continue
            tracker.durations.append((sequence, duration))
            event: dict[str, Any] = {
                "type": "response.output_audio.delta",
                "response_id": response_id,
                "generation": generation,
                "sequence": sequence,
                "sample_rate": self.sample_rate,
                "delta": base64.b64encode(pcm).decode("ascii"),
            }
            if first_delta:
                event.update(
                    {
                        "source_start": segment.source_start,
                        "source_end": segment.source_end,
                        "text": segment.text,
                    }
                )
                first_delta = False
            await self._send(event)

        if self._worker_future:
            await asyncio.shield(self._worker_future)
        tracker.output_done = True
        if not tracker.durations:
            self._playback.pop(response_id, None)
        if generation == self._generation and not self._disconnected:
            if status != "failed":
                await self._send(
                    {
                        "type": "response.output_audio.done",
                        "response_id": response_id,
                        "generation": generation,
                    }
                )
            await self._send(
                {
                    "type": "response.done",
                    "response": {
                        "id": response_id,
                        "status": status,
                        "generation": generation,
                        "source_start": segment.source_start,
                        "source_end": segment.source_end,
                    },
                }
            )
        self._active_response_id = None
        self._worker_future = None

    async def _wait_for_flow(self) -> bool:
        limit = float(self.settings.tts_live_max_unacked_seconds)
        while not self._closing:
            paused_at = self._paused_at
            unacknowledged = self._unacknowledged_seconds()
            if paused_at is None and unacknowledged < limit:
                return True

            if paused_at is not None:
                elapsed = time.monotonic() - paused_at
                timeout = self.settings.tts_live_max_pause_s - elapsed
                error_code = "pause_timeout"
                error_message = "Maximum continuous pause exceeded"
            else:
                timeout = self.settings.tts_live_playback_stall_timeout_s
                error_code = "playback_stalled"
                error_message = "Playback acknowledgements stopped"
            if timeout <= 0:
                await self._close_for_flow_error(error_message, error_code)
                return False

            self._flow_changed.clear()
            if paused_at != self._paused_at:
                continue
            if paused_at is None and self._unacknowledged_seconds() < limit:
                continue
            try:
                await asyncio.wait_for(self._flow_changed.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                await self._close_for_flow_error(error_message, error_code)
                return False
        return False

    async def _close_for_flow_error(self, message: str, code: str) -> None:
        await self._send_error(message, code)
        self._closing = True
        self._worker_cancel.set()
        self._flow_changed.set()
        if not self._disconnected:
            await self.ws.close(code=4008, reason=message)

    def _unacknowledged_seconds(self) -> float:
        return sum(tracker.unacknowledged_seconds for tracker in self._playback.values())

    async def _send_status(self, *, force: bool = False) -> None:
        payload = (
            self._buffered_input_chars,
            self._segment_queue.qsize(),
            round(self._unacknowledged_seconds(), 3),
        )
        if not force and payload == self._last_status_payload:
            return
        remaining = 1.0 - (time.monotonic() - self._last_status_sent_at)
        if not force and remaining > 0:
            if self._status_task is None or self._status_task.done():
                self._status_task = asyncio.create_task(
                    self._send_delayed_status(remaining),
                    name=f"live-status-{self.id}",
                )
            return
        if await self._send(
            {
                "type": "session.status",
                "generation": self._generation,
                "buffered_chars": payload[0],
                "pending_segments": payload[1],
                "unacked_seconds": payload[2],
            }
        ):
            self._last_status_payload = payload
            self._last_status_sent_at = time.monotonic()

    async def _send_delayed_status(self, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
            self._status_task = None
            await self._send_status()
        except asyncio.CancelledError:
            return

    async def _send_error(
        self,
        message: str,
        code: str,
        *,
        response_id: str | None = None,
    ) -> None:
        error: dict[str, Any] = {"message": message, "code": code}
        if response_id:
            error["response_id"] = response_id
        await self._send({"type": "error", "error": error})

    @staticmethod
    def _drain_queue(queue: asyncio.Queue) -> None:
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    def _discard_stale_segments(self) -> None:
        """Drop queued segments of a cancelled generation; the synthesis loop skips them anyway.

        The segmenter task's put that a cancel woke still lands one of them.
        """
        kept = []
        while True:
            try:
                item = self._segment_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is None or item[0] == self._generation:
                kept.append(item)
        for item in kept:
            self._segment_queue.put_nowait(item)


async def live_tts_endpoint(
    websocket: WebSocket,
    *,
    tts_router,
    pronunciation_dict,
    settings,
    voice_library=None,
) -> None:
    """Serve one ``/v1/audio/speech/stream`` connection."""
    await websocket.accept()
    token, busy_reason = _admission.acquire(settings.tts_live_max_connections)
    if token is None:
        await websocket.send_json(
            {
                "type": "error",
                "error": {
                    "code": "session_busy",
                    "message": "Live Reader is already in use",
                    "reason": busy_reason,
                },
            }
        )
        await websocket.close(code=1013, reason="Live Reader busy")
        return

    session: LiveTTSSession | None = None
    try:
        session = LiveTTSSession(
            websocket,
            tts_router=tts_router,
            pronunciation_dict=pronunciation_dict,
            settings=settings,
            voice_library=voice_library,
        )
        await session.run()
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Live TTS session failed")
        if websocket.client_state == WebSocketState.CONNECTED:
            # Without a close frame the client only sees the connection drop (1006).
            with contextlib.suppress(Exception):
                await websocket.close(code=1011, reason="Live TTS session failed")
    finally:
        _admission.mark_draining(token)
        worker_finished = True
        if session is not None:
            worker_finished = await session.shutdown()
        if worker_finished:
            _admission.release(token)
        elif session is not None and session.worker_future is not None:
            # This callback needs the application loop, but failure to run it is
            # relevant only while that same loop and process are shutting down.
            session.worker_future.add_done_callback(lambda _future: _admission.release(token))
        else:
            _admission.release(token)


def reset_live_tts_state_for_tests() -> None:
    """Reset process-local admission state between isolated app tests."""
    _admission.reset()
