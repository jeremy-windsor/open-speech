"""OpenAI Realtime API WebSocket handler.

Implements the /v1/realtime endpoint for OpenAI Realtime API compatibility.
Handles STT + TTS only — no LLM/conversation logic.

Reference: https://platform.openai.com/docs/api-reference/realtime
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import json
import logging
import time
import uuid
from typing import Any

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from src.config import settings
from src.realtime import events
from src.realtime.audio_buffer import InputAudioBuffer, decode_audio_to_pcm16, encode_pcm16_to_format
from src.realtime.session import SessionConfig
from src.router import router as stt_router
from src.tts.router import TTSRouter
from src.tts.pipeline import encode_audio
from src.vad.silero import SileroVAD, get_vad_model

logger = logging.getLogger(__name__)

# Thread pool for blocking STT/TTS operations
_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="realtime"
)


class RealtimeSession:
    """Manages a single Realtime API WebSocket session."""

    def __init__(self, websocket: WebSocket, tts_router: TTSRouter, model: str = ""):
        self.ws = websocket
        self.tts_router = tts_router
        self.config = SessionConfig(model=model or settings.stt_model)
        self.audio_buffer: InputAudioBuffer | None = None
        self._last_item_id: str | None = None
        self._cancelled_responses: set[str] = set()
        self._current_response_id: str | None = None
        self._last_commit_at = time.monotonic()

    async def initialize(self) -> None:
        """Initialize session: load VAD if needed, send session.created."""
        vad = None
        if self.config.vad_enabled:
            try:
                vad_model = await get_vad_model()
                vad = SileroVAD(vad_model.session, threshold=self.config.turn_detection.threshold)
            except Exception:
                logger.warning("Failed to load VAD model, disabling server VAD")

        self.audio_buffer = InputAudioBuffer(
            vad=vad,
            threshold=self.config.turn_detection.threshold if self.config.turn_detection else 0.5,
            silence_duration_ms=(
                self.config.turn_detection.silence_duration_ms
                if self.config.turn_detection else 500
            ),
            max_buffer_bytes=settings.os_realtime_max_buffer_mb * 1024 * 1024,
        )

        await self._send(events.session_created(self.config.to_dict()))

    async def _send(self, event: dict[str, Any]) -> None:
        """Send a JSON event to the client."""
        try:
            await self.ws.send_json(event)
        except Exception:
            pass  # Connection may be closed

    async def handle_event(self, data: dict[str, Any]) -> None:
        """Route an incoming client event."""
        event_type = data.get("type", "")
        handler = _CLIENT_HANDLERS.get(event_type)
        if handler is None:
            await self._send(events.error(
                f"Unknown event type: {event_type}",
                code="unknown_event",
                event_id=data.get("event_id"),
            ))
            return
        try:
            await handler(self, data)
        except Exception as e:
            logger.exception("Error handling event %s", event_type)
            await self._send(events.error(
                str(e),
                code="internal_error",
                event_id=data.get("event_id"),
            ))

    # ── Client event handlers ──────────────────────────────────────────────

    async def _handle_session_update(self, data: dict[str, Any]) -> None:
        self.config.update_from(data)

        # Rebuild VAD if turn_detection changed
        vad = None
        if self.config.vad_enabled:
            try:
                vad_model = await get_vad_model()
                vad = SileroVAD(vad_model.session, threshold=self.config.turn_detection.threshold)
            except Exception:
                pass

        self.audio_buffer = InputAudioBuffer(
            vad=vad,
            threshold=self.config.turn_detection.threshold if self.config.turn_detection else 0.5,
            silence_duration_ms=(
                self.config.turn_detection.silence_duration_ms
                if self.config.turn_detection else 500
            ),
            max_buffer_bytes=settings.os_realtime_max_buffer_mb * 1024 * 1024,
        )

        await self._send(events.session_updated(self.config.to_dict()))

    async def _handle_input_audio_buffer_append(self, data: dict[str, Any]) -> None:
        if (time.monotonic() - self._last_commit_at) > settings.os_realtime_idle_timeout_s:
            await self._send(events.error("Session idle timeout waiting for commit", code="idle_timeout"))
            await self.ws.close(code=4008, reason="Session idle timeout")
            return

        audio_b64 = data.get("audio", "")
        if not audio_b64:
            return

        try:
            raw = base64.b64decode(audio_b64)
        except Exception:
            await self._send(events.error("Invalid base64 audio data", code="invalid_audio"))
            return

        # Convert to PCM16 16kHz for internal processing
        try:
            pcm16 = decode_audio_to_pcm16(raw, self.config.input_audio_format, target_rate=16000)
        except Exception as e:
            await self._send(events.error(str(e), code="invalid_audio"))
            return

        try:
            vad_events = self.audio_buffer.append(pcm16)
        except BufferError as e:
            if self.audio_buffer:
                self.audio_buffer.clear()
            await self._send(events.error(str(e), code="buffer_overflow"))
            return

        for evt in vad_events:
            if evt["type"] == "speech_started":
                item_id = events._item_id()
                await self._send(events.input_audio_buffer_speech_started(
                    evt["audio_start_ms"], item_id
                ))
            elif evt["type"] == "speech_stopped":
                item_id = events._item_id()
                await self._send(events.input_audio_buffer_speech_stopped(
                    evt["audio_end_ms"], item_id
                ))
                # Auto-commit on speech end in VAD mode
                await self._commit_and_transcribe()

    async def _handle_input_audio_buffer_commit(self, data: dict[str, Any]) -> None:
        await self._commit_and_transcribe()

    async def _handle_input_audio_buffer_clear(self, data: dict[str, Any]) -> None:
        if self.audio_buffer:
            self.audio_buffer.clear()
        await self._send(events.input_audio_buffer_cleared())

    async def _handle_response_create(self, data: dict[str, Any]) -> None:
        response_data = data.get("response", {})
        modalities = response_data.get("modalities", ["audio", "text"])

        if modalities == ["text"]:
            await self._send(events.error(
                "Open Speech does not support text-only responses. We handle audio I/O only.",
                code="unsupported_modality",
            ))
            return

        # Extract text to synthesize
        instructions = response_data.get("instructions", "")
        # Also check for input items with text content
        input_items = response_data.get("input", [])
        text_to_speak = instructions
        if not text_to_speak and input_items:
            for item in input_items:
                content = item.get("content", [])
                for c in content:
                    if c.get("type") == "input_text" and c.get("text"):
                        text_to_speak = c["text"]
                        break
                if text_to_speak:
                    break

        if not text_to_speak:
            await self._send(events.error(
                "No text provided for TTS. Include 'instructions' or input text content.",
                code="missing_input",
            ))
            return

        resp_id = events._response_id()
        self._current_response_id = resp_id
        item_id = events._item_id()

        response_obj = {
            "id": resp_id,
            "object": "realtime.response",
            "status": "in_progress",
            "output": [],
        }
        await self._send(events.response_created(response_obj))

        # Run TTS in executor
        loop = asyncio.get_running_loop()
        voice = self.config.voice
        output_format = self.config.output_audio_format
        tts_model = response_data.get("model") or self.config.model or settings.tts_model

        try:
            def _synthesize():
                chunks = self.tts_router.synthesize(
                    text=text_to_speak,
                    model=tts_model,
                    voice=voice,
                    speed=1.0,
                )
                # Collect raw float32 24kHz chunks
                all_audio = []
                for chunk in chunks:
                    if isinstance(chunk, np.ndarray):
                        all_audio.append(chunk)
                    else:
                        all_audio.append(np.array(chunk, dtype=np.float32))
                if not all_audio:
                    return b""
                combined = np.concatenate(all_audio)
                # Convert to PCM16
                pcm16 = (combined * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
                # Encode to output format
                return encode_pcm16_to_format(pcm16, 24000, output_format)

            audio_data = await loop.run_in_executor(_executor, _synthesize)
        except Exception as e:
            logger.exception("TTS synthesis failed in realtime session")
            await self._send(events.error(str(e), code="tts_error"))
            response_obj["status"] = "failed"
            await self._send(events.response_done(response_obj))
            self._current_response_id = None
            return

        if resp_id in self._cancelled_responses:
            self._cancelled_responses.discard(resp_id)
            self._current_response_id = None
            return

        # Stream audio as delta events (chunk into ~4KB pieces for base64)
        CHUNK_SIZE = 3000  # ~4KB base64
        for i in range(0, len(audio_data), CHUNK_SIZE):
            if resp_id in self._cancelled_responses:
                break
            chunk = audio_data[i:i + CHUNK_SIZE]
            delta = base64.b64encode(chunk).decode("ascii")
            await self._send(events.response_audio_delta(
                resp_id, item_id, 0, 0, delta
            ))

        self._cancelled_responses.discard(resp_id)
        await self._send(events.response_audio_done(resp_id, item_id, 0, 0))

        response_obj["status"] = "completed"
        response_obj["output"] = [{
            "id": item_id,
            "object": "realtime.item",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "audio", "transcript": text_to_speak}],
        }]
        await self._send(events.response_done(response_obj))
        self._current_response_id = None

    async def _handle_response_cancel(self, data: dict[str, Any]) -> None:
        if self._current_response_id:
            self._cancelled_responses.add(self._current_response_id)

    # ── Internal helpers ───────────────────────────────────────────────────

    async def _commit_and_transcribe(self) -> None:
        """Commit audio buffer and run STT."""
        if self.audio_buffer is None:
            return

        audio_data = self.audio_buffer.commit()
        self._last_commit_at = time.monotonic()
        if not audio_data or len(audio_data) < 1600:  # less than 50ms at 16kHz
            return

        item_id = events._item_id()
        self._last_item_id = item_id

        await self._send(events.input_audio_buffer_committed(item_id, None))

        # Create conversation item
        item = {
            "id": item_id,
            "object": "realtime.item",
            "type": "message",
            "role": "user",
            "content": [{"type": "input_audio", "transcript": None}],
        }
        await self._send(events.conversation_item_created(item))

        # Transcribe in executor
        loop = asyncio.get_running_loop()
        model = self.config.model or settings.stt_model

        try:
            def _transcribe():
                # Convert PCM16 16kHz to WAV for the STT backend
                import io
                import wave
                wav_buf = io.BytesIO()
                with wave.open(wav_buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(audio_data)
                return stt_router.transcribe(
                    audio=wav_buf.getvalue(),
                    model=model,
                    language=None,
                    response_format="json",
                    temperature=0.0,
                )

            result = await loop.run_in_executor(_executor, _transcribe)
        except Exception as e:
            logger.exception("Transcription failed in realtime session")
            await self._send(events.error(str(e), code="transcription_error"))
            return

        transcript = result.get("text", "") if isinstance(result, dict) else str(result)

        await self._send(events.conversation_item_input_audio_transcription_completed(
            item_id, 0, transcript
        ))


# Handler dispatch table
_CLIENT_HANDLERS: dict[str, Any] = {
    "session.update": RealtimeSession._handle_session_update,
    "input_audio_buffer.append": RealtimeSession._handle_input_audio_buffer_append,
    "input_audio_buffer.commit": RealtimeSession._handle_input_audio_buffer_commit,
    "input_audio_buffer.clear": RealtimeSession._handle_input_audio_buffer_clear,
    "response.create": RealtimeSession._handle_response_create,
    "response.cancel": RealtimeSession._handle_response_cancel,
}


async def realtime_endpoint(
    websocket: WebSocket,
    tts_router: TTSRouter,
    model: str = "",
) -> None:
    """Main WebSocket handler for /v1/realtime."""
    await websocket.accept(subprotocol="realtime")

    session = RealtimeSession(websocket, tts_router, model=model)
    await session.initialize()

    try:
        while True:
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(), timeout=settings.os_realtime_idle_timeout_s
                )
            except asyncio.TimeoutError:
                await session._send(events.error("Session idle timeout", code="idle_timeout"))
                await websocket.close(code=4008, reason="Session idle timeout")
                break
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await session._send(events.error(
                    "Invalid JSON", code="invalid_json"
                ))
                continue

            if not isinstance(data, dict) or "type" not in data:
                await session._send(events.error(
                    "Event must be a JSON object with a 'type' field",
                    code="invalid_event",
                ))
                continue

            await session.handle_event(data)

    except WebSocketDisconnect:
        logger.debug("Realtime client disconnected (session %s)", session.config.id)
    except Exception:
        logger.exception("Realtime session error")
    finally:
        pass
