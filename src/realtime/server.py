"""OpenAI Realtime API WebSocket handler.

Implements the /v1/realtime endpoint for OpenAI Realtime API compatibility: speech to text for each turn, and
spoken replies. A ``response.create`` that carries text speaks that text word for word. With the AI assistant on
(``OS_ASSISTANT_URL``, ``src.assistant``), each turn's transcript joins the conversation; a turn that server VAD
ends (with ``create_response``), and a ``response.create`` with no text, goes to the AI (the client's own
``input_audio_buffer.commit`` asks nothing, as in OpenAI's API): short waiting phrases are spoken while it works,
then its reply. Its errors are spoken too, and its thinking is only ever sent as text. A typed question (``conversation.item.create`` with a user ``input_text``
message) joins the conversation as a transcript does, and the client's ``response.create`` answers it the same way.
The conversation is kept per connection, so a reconnect starts fresh.

With Stream replies on (``OS_ASSISTANT_STREAM``, the default), the AI is asked with ``stream: true``: its answer is cut
into sentences as they complete, by the Live Reader's segmenter, and each is spoken at once through one encoder, its
text sent just before its audio. Each AI voice turn logs one INFO line of its timings, numbers only.

Reference: https://platform.openai.com/docs/api-reference/realtime
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import concurrent.futures
import copy
import json
import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from src.assistant import (
    AssistantClient,
    AssistantError,
    AssistantReply,
    Conversation,
    EffectiveSettings,
    on_close,
    speakable_segments,
)
from src.assistant import prompts as assistant_prompts
from src.assistant.errors import ASSISTANT_BAD_RESPONSE, ASSISTANT_REPLY_CUT_OFF, ASSISTANT_UNREACHABLE
from src.assistant.presets import PresetError
from src.audio.levels import InputLevels
from src.config import ASSISTANT_PROMPT_MAX_CHARS, settings
from src.live_tts.segmenter import LiveTextSegmenter, TextSegment
from src.openai_compat import is_openai_realtime_model, resolve_stt_model, resolve_tts_model
from src.realtime import events
from src.realtime.audio_buffer import (
    InputAudioBuffer,
    InputAudioStream,
    OutputAudioStream,
    decode_audio_to_pcm16,
)
from src.backends.parakeet import check_language
from src.realtime.backchannel import is_backchannel
from src.realtime.session import (
    FORMAT_SAMPLE_RATES,
    SessionConfig,
    has_lone_surrogate,
    parse_audio_format,
    parse_voice,
)
from src.router import router as stt_router
from src.services.stt import whisper_language_code
from src.services.tts import validate_tts_input_limit
from src.tts.pipeline import float32_to_int16
from src.tts.router import TTSRouter
from src.utils.redact import redact_paths
from src.vad.silero import SileroVAD, get_vad_model
from src.voice_identities import VOICE_PREFIX, resolve_named_voice

logger = logging.getLogger(__name__)

# Thread pool for blocking STT/TTS operations
_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="realtime"
)
# Input audio decoding and Silero, kept apart so a reply being synthesized
# cannot delay turn detection
_input_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="realtime-input"
)

_active_realtime_sessions: set[int] = set()
_active_realtime_lock = asyncio.Lock()

MIN_COMMIT_BYTES = 1600  # 50 ms of 16 kHz PCM16
AUDIO_DELTA_BYTES = 3000  # ~4 KB of base64 per response.audio.delta
# Waiting phrases kept per session: model, voice, speed, format and phrase each make a new one.
FILLER_CACHE_MAX = 32
# The longest item id a client may give in conversation.item.create, as in OpenAI's API.
ITEM_ID_MAX_CHARS = 32
# A typed question's text: the limit on text typed for the AI elsewhere (Settings › AI assistant › Test).
TYPED_TEXT_MAX_CHARS = ASSISTANT_PROMPT_MAX_CHARS


class _InvalidAudio(Exception):
    """An append's audio could not be decoded; the message is the error for the client."""


class _NoServer:
    """The client of a saved AI assistant with no server URL: its turns fail as unreachable."""

    stream = False

    async def reply(self, messages, settings):
        raise AssistantError("The AI assistant has no server URL in its settings", ASSISTANT_UNREACHABLE)


def _typed_user_message(data: dict[str, Any]) -> tuple[str, list[str]]:
    """The item id and the text parts of a ``conversation.item.create`` that adds a user's typed message.

    Raises ValueError for any other item: the conversation holds only the user's words and the AI's answers, added
    at its end. Text or an id with a lone surrogate is refused too, because it can't be sent to the AI or echoed
    back.
    """
    if data.get("previous_item_id") is not None:
        raise ValueError("previous_item_id isn't supported: a typed message joins the end of the conversation")
    item = data.get("item")
    if not isinstance(item, dict):
        raise ValueError("item must be an object")
    if item.get("type") != "message" or item.get("role") != "user":
        raise ValueError("Only a user's message can be added: item.type must be 'message' and item.role 'user'")
    item_id = item.get("id")
    if item_id is None:
        item_id = events._item_id()
    elif not isinstance(item_id, str) or not 1 <= len(item_id) <= ITEM_ID_MAX_CHARS:
        raise ValueError(f"item.id must be a string of 1 to {ITEM_ID_MAX_CHARS} characters")
    elif has_lone_surrogate(item_id):
        raise ValueError("item.id has a lone surrogate (half of a UTF-16 pair), which can't be sent back")
    content = item.get("content")
    if not isinstance(content, list) or not content:
        raise ValueError("item.content must be a nonempty list of input_text parts")
    parts: list[str] = []
    for index, part in enumerate(content):
        if not isinstance(part, dict) or part.get("type") != "input_text" or not isinstance(part.get("text"), str):
            raise ValueError(f"item.content[{index}] must be an input_text part with its text; only typed text can be added")
        if has_lone_surrogate(part["text"]):
            raise ValueError(
                f"item.content[{index}].text has a lone surrogate (half of a UTF-16 pair, such as half an emoji), "
                "which can't be sent to the AI; send whole characters"
            )
        parts.append(part["text"])
    return item_id, parts


def _vad_unavailable_error(event_id: str | None = None) -> dict[str, Any]:
    return events.error(
        "Server VAD could not be loaded, so turn detection is off; commit audio manually",
        error_type="server_error",
        code="vad_unavailable",
        event_id=event_id,
    )


def _with_whisper_language(transcription: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the transcription settings with their language as a Whisper code.

    faster-whisper checks the code only after decoding a turn, so an unknown one would
    fail every transcription; it raises ValueError here instead.
    """
    if not transcription or "language" not in transcription:
        return transcription
    language = transcription["language"]
    if language is not None and not isinstance(language, str):
        raise ValueError("Transcription language must be a string")
    code = whisper_language_code(language)
    if code == language:
        return transcription
    return {**transcription, "language": code}


def _stt_model_for(config: SessionConfig) -> str:
    """The STT model a session's turns are transcribed with: a GA or OpenAI realtime session's transcription
    model, else the session's own model (legacy clients pick STT with ?model=)."""
    if config.ga or is_openai_realtime_model(config.model):
        return resolve_stt_model((config.input_audio_transcription or {}).get("model"), settings)
    return resolve_stt_model(config.model, settings)


def _with_supported_language(config: SessionConfig) -> str | None:
    """Drop a transcription language the session's STT model can't transcribe; return why, or None.

    Parakeet transcribes English only and would fail every turn with another language. As /v1/audio/stream
    does, the session says so once and carries on in English.
    """
    transcription = config.input_audio_transcription
    if not transcription or not transcription.get("language"):
        return None
    try:
        check_language(_stt_model_for(config), transcription["language"])
    except ValueError as exc:
        config.input_audio_transcription = {**transcription, "language": None}
        return f"{exc} The session continues in English."
    return None


@dataclass(frozen=True)
class _Speech:
    """How one response speaks: the TTS model, the provider's voice, the speed and the output format."""

    model: str
    voice: str
    speed: float
    output_format: str


class _Playback:
    """Where one response's audio stands against the client's playback."""

    def __init__(self, resp_id: str, item_id: str, output_format: str, lead: float | None, now: float):
        self.resp_id = resp_id
        self.item_id = item_id
        self.lead = lead
        self.bytes_per_s = FORMAT_SAMPLE_RATES[output_format] * (2 if output_format == "pcm16" else 1)
        self.ends = now
        # When the reply's first audio went out (not a waiting phrase's), for the turn's timing line.
        self.first_audio_at: float | None = None


def _result_or_none(future: asyncio.Future) -> Any:
    """A finished future's result, or None when it was cancelled or failed."""
    if future.cancelled() or future.exception() is not None:
        return None
    return future.result()


def _log_phrase_failure(future: asyncio.Future) -> None:
    if not future.cancelled() and future.exception() is not None:
        logger.warning("A waiting phrase couldn't be synthesized: %s", future.exception())


async def _wait_event(event: asyncio.Event, timeout: float) -> bool:
    """Whether ``event`` is set within ``timeout`` seconds."""
    if event.is_set():
        return True
    try:
        async with asyncio.timeout(timeout):
            await event.wait()
    except TimeoutError:
        return False
    return True


@dataclass
class _TurnTiming:
    """What one AI voice turn's log line reports: the end-of-turn silence server VAD waited for and the transcription's
    time (None for a typed question; the silence also for a client's commit), and the loop times of the AI request, its
    first piece, its first answer piece and its end. Numbers only."""

    silence_ms: int | None = None
    stt_ms: int | None = None
    asked_at: float | None = None
    first_token_at: float | None = None
    first_answer_at: float | None = None
    done_at: float | None = None
    streamed: bool = False
    reply_chars: int = 0
    tokens_in: int | None = None
    tokens_out: int | None = None

    def line(self, first_audio_at: float | None) -> str:
        """``key=value`` fields; ``-`` for what wasn't measured or reached. Times count from the request."""

        def since(at: float | None) -> str:
            if at is None or self.asked_at is None:
                return "-"
            return str(max(0, round((at - self.asked_at) * 1000)))

        def number(value: int | None) -> str:
            return "-" if value is None else str(int(value))

        return (
            f"silence_ms={number(self.silence_ms)} stt_ms={number(self.stt_ms)} "
            f"ai_first_token_ms={since(self.first_token_at)} ai_first_answer_ms={since(self.first_answer_at)} "
            f"ai_done_ms={since(self.done_at)} first_audio_ms={since(first_audio_at)} "
            f"reply_chars={int(self.reply_chars)} tokens_in={number(self.tokens_in)} "
            f"tokens_out={number(self.tokens_out)} streamed={int(self.streamed)}"
        )


# A longer answer piece is cut into sentences in a worker thread: on the loop, 4,096 characters take a few ms.
_CUT_ON_LOOP_CHARS = 4096


class _StreamedTurn:
    """One streamed AI reply: the answer as it arrives, cut into sentences by the Live Reader's segmenter; the sentences
    handed to the voice, each with the part of the answer's text it covers (its transcript delta); and the tasks that
    read the stream, speak it and say the waiting phrases.

    The voice gets a sentence as soon as it is complete. When the stream ends well the rest of the answer follows;
    after an error or a cut-off (``finish_reason`` ``"length"`` in a streamed answer) nothing more is handed over,
    not even the sentence it was in the middle of. An answer that came whole, as one plain JSON reply, is handed over
    in full even when the token limit cut it, as Stream replies off speaks it.

    A piece longer than ``_CUT_ON_LOOP_CHARS``, such as a whole JSON answer, is cut into sentences in a worker
    thread, not in ``on_piece`` on the event loop, which must not block; pieces that arrive meanwhile follow it in
    order, and ``end()`` waits for them."""

    def __init__(self, timing: _TurnTiming) -> None:
        self.loop = asyncio.get_running_loop()
        self.timing = timing
        self.segmenter = LiveTextSegmenter(
            mode="natural", max_chars=settings.tts_live_sentence_max_chars,
            max_words=settings.tts_live_sentence_max_words,
        )
        self.answer = ""
        self.thinking: list[str] = []
        self.handed = 0  # sentences (or a last bit of text) handed to the voice
        self.handed_upto = 0  # how much of the answer they cover
        # (speakable text, the answer's text it covers); None once the stream has ended
        self.sentences: asyncio.Queue[tuple[str, str] | None] = asyncio.Queue()
        self.first_sentence = asyncio.Event()
        self.failure: AssistantError | None = None
        self.reader: asyncio.Task | None = None
        self.speaker: asyncio.Task | None = None
        self.fillers: asyncio.Task | None = None
        self.cutter: asyncio.Task | None = None  # cuts _uncut into sentences in a worker thread
        self._uncut: list[str] = []  # answer pieces waiting for the cutter, in order
        self.recorded = False

    def on_piece(self, kind: str, text: str) -> None:
        """The client's callback: a piece of the thinking or of the answer, as it is read."""
        now = self.loop.time()
        if self.timing.first_token_at is None:
            self.timing.first_token_at = now
        if kind != "answer":
            self.thinking.append(text)
            return
        if self.timing.first_answer_at is None:
            self.timing.first_answer_at = now
        self.answer += text
        cutting = self.cutter is not None and not self.cutter.done()
        if not cutting and len(text) <= _CUT_ON_LOOP_CHARS:
            for segment in self.segmenter.append(text):
                self._hand_over(segment)
            return
        self._uncut.append(text)
        if not cutting:
            self.cutter = self.loop.create_task(self._cut(), name="realtime-assistant-cutter")

    async def _cut(self) -> None:
        """Cut the waiting pieces into sentences in a worker thread, in order, and hand them over on the loop."""
        while self._uncut:
            text = "".join(self._uncut)
            self._uncut.clear()
            for segment in await asyncio.to_thread(self.segmenter.append, text):
                self._hand_over(segment)

    async def end(self, reply=None, failure: AssistantError | None = None) -> None:
        """The stream ended, with the reply (a ``StreamedReply``) or with its failure. The answer read is cut into
        sentences first."""
        self.timing.done_at = self.loop.time()
        if self.cutter is not None:
            try:
                await self.cutter
            except Exception:
                logger.exception("Cutting the AI's streamed reply into sentences failed")
                failure = failure or AssistantError("The AI's reply couldn't be read", ASSISTANT_BAD_RESPONSE)
        if reply is not None:
            self.timing.streamed = bool(getattr(reply, "streamed", False))
            usage = getattr(reply, "usage", None) or {}
            self.timing.tokens_in = usage.get("prompt_tokens")
            self.timing.tokens_out = usage.get("completion_tokens")
            if not self.timing.streamed:  # one whole answer: its first token came with its end
                self.timing.first_token_at = self.timing.first_answer_at = self.timing.done_at
            # A whole answer arrived whole: the limit cut it, but all of it is spoken, as reply() gives it.
            if failure is None and reply.finish_reason == "length" and self.timing.streamed:
                failure = AssistantError(
                    "The token limit cut the AI off in the middle of its answer: raise the token limit or lower the "
                    "thinking",
                    ASSISTANT_REPLY_CUT_OFF,
                )
        if failure is None:
            for segment in self.segmenter.flush():
                self._hand_over(segment)
            if self.handed_upto < len(self.answer):  # text that says nothing, such as an unclosed code block
                self._queue("", self.answer[self.handed_upto:])
                self.handed_upto = len(self.answer)
        self.failure = failure
        self.sentences.put_nowait(None)

    def thinking_text(self) -> str | None:
        return "".join(self.thinking).strip() or None

    def kept(self) -> str:
        """The answer as far as it was handed to the voice: all of it once the stream ended well."""
        return self.answer[:self.handed_upto].strip()

    def _hand_over(self, segment: TextSegment) -> None:
        end = max(self.handed_upto, min(segment.source_end, len(self.answer)))
        self._queue(segment.text, self.answer[self.handed_upto:end])
        self.handed_upto = end

    def _queue(self, text: str, covers: str) -> None:
        self.handed += 1
        self.sentences.put_nowait((text, covers))
        self.first_sentence.set()


class RealtimeSession:
    """Manages a single Realtime API WebSocket session."""

    def __init__(
        self,
        websocket: WebSocket,
        tts_router: TTSRouter,
        model: str = "",
        *,
        ga: bool = False,
        pronounce_fn: Callable[[str], str] | None = None,
        assistant: AssistantClient | None = None,
    ):
        self.ws = websocket
        self.tts_router = tts_router
        # The AI assistant (src.assistant), or None when it's off. Replies need TTS as well.
        self.assistant = assistant
        # TTS_PRONUNCIATION_DICT: replies speak its replacements, transcripts keep the text
        self.pronounce_fn = pronounce_fn
        self.config = SessionConfig(
            model=model or (settings.tts_model if ga else settings.stt_model), ga=ga,
            assistant_on=assistant is not None and settings.tts_enabled,
        )
        # The user's words and the AI's answers, in memory: a reconnect starts fresh.
        self.conversation = Conversation()
        self._assistant_snapshot: EffectiveSettings | None = None
        # Waiting phrases' audio, per (speech, phrase); made ahead of time after session.update.
        self._filler_cache: dict[tuple[_Speech, str], asyncio.Future[bytes]] = {}
        self._filler_warm_task: asyncio.Task[None] | None = None
        # The phrase a reply is waiting for: it is for that reply's settings, so it is never dropped.
        self._filler_awaited: asyncio.Future[bytes] | None = None
        # Each phrase's job in the realtime pool, while it is being made.
        self._phrase_jobs: dict[asyncio.Future[bytes], concurrent.futures.Future[bytes]] = {}
        # Phrases the session left while their synthesis ran: it can't be stopped, so the next phrase waits for
        # it to end, and one session holds one realtime worker.
        self._phrases_left_running: set[concurrent.futures.Future[bytes]] = set()
        self._last_filler: str | None = None
        self.audio_buffer: InputAudioBuffer | None = None
        # Resampler state for the client's audio, kept across appends
        self._input_audio: InputAudioStream | None = None
        self._last_item_id: str | None = None
        self._cancelled_responses: set[str] = set()
        self._current_response_id: str | None = None
        self._response_task: asyncio.Task[None] | None = None
        self._response_cancel_event: threading.Event | None = None
        self._response_cancellable = False
        self._send_lock = asyncio.Lock()
        self._last_commit_at = time.monotonic()
        self._reply_ended_at = float("-inf")
        # The user item the current VAD turn will be committed as
        self._turn_item_id: str | None = None
        # The last turn's end-of-turn silence and transcription time, in ms, for its reply's timing line; None for
        # what it didn't have
        self._turn_measures: tuple[int | None, int | None] = (None, None)
        self._cancel_reason = "client_cancelled"
        # Barge-in: while the user talks over a reply, audio is held here
        # until we know whether they are interrupting or just saying "mm-hmm".
        self._playback_gate = asyncio.Event()
        self._playback_gate.set()
        self._barge_in_pending = False
        self._barge_in_timer: asyncio.Task[None] | None = None

    async def initialize(self) -> None:
        """Initialize session: load VAD if needed, send session.created."""
        vad_loaded = await self._apply_turn_detection()
        await self._send(events.session_created(self.config.to_dict()))
        if not vad_loaded:
            await self._send(_vad_unavailable_error())

    async def _apply_turn_detection(self) -> bool:
        """Create the input buffer, or retune it, for the current turn detection.

        Retuning keeps buffered audio and the session's audio clock. Returns
        False when server VAD was asked for but could not be loaded.
        """
        td = self.config.turn_detection
        vad = self.audio_buffer.vad if self.audio_buffer is not None else None
        vad_loaded = True
        if td is None:
            vad = None
        elif vad is None:
            try:
                vad_model = await get_vad_model()
                vad = SileroVAD(vad_model.session, threshold=td.threshold)
            except Exception:
                logger.warning("Failed to load VAD model, disabling server VAD")
                # Report the session in effect: nothing will commit turns for the client.
                self.config.turn_detection = td = None
                vad_loaded = False

        turn_settings = {
            "vad": vad,
            "threshold": td.threshold if td else 0.5,
            "silence_duration_ms": td.silence_duration_ms if td else 500,
            "prefix_padding_ms": td.prefix_padding_ms if td else 300,
        }
        if self.audio_buffer is None:
            self.audio_buffer = InputAudioBuffer(
                **turn_settings,
                max_buffer_bytes=settings.os_realtime_max_buffer_mb * 1024 * 1024,
            )
        else:
            self.audio_buffer.configure(**turn_settings)
        return vad_loaded

    def _is_tts_model(self, model: str) -> bool:
        """Whether the TTS router can speak with ``model`` (may ask an external worker)."""
        try:
            self.tts_router.get_backend(model)
        except Exception:
            return False
        return True

    def seconds_since_reply(self) -> float:
        """How long since the last reply finished; 0 while one is still streaming."""
        if self._response_task is not None and not self._response_task.done():
            return 0.0
        return time.monotonic() - self._reply_ended_at

    async def _send(self, event: dict[str, Any]) -> None:
        """Send a JSON event to the client."""
        if self.config.ga:
            event = dict(event)
            event["type"] = {
                "response.audio.delta": "response.output_audio.delta",
                "response.audio.done": "response.output_audio.done",
                "response.audio_transcript.delta": "response.output_audio_transcript.delta",
                "response.audio_transcript.done": "response.output_audio_transcript.done",
                "conversation.item.created": "conversation.item.added",
            }.get(event["type"], event["type"])
        try:
            async with self._send_lock:
                await self.ws.send_json(event)
        except Exception:
            pass  # Connection may be closed

    async def close(self) -> None:
        """Cancel and collect any in-flight response task, and the waiting phrases being made."""
        self._end_barge_in()
        warm, self._filler_warm_task = self._filler_warm_task, None
        if warm is not None:
            warm.cancel()
            await asyncio.gather(warm, return_exceptions=True)
        for future in self._filler_cache.values():
            self._cancel_phrase(future)
        self._filler_cache.clear()
        task = self._response_task
        if task is None:
            return
        should_cancel = self._response_cancellable
        self._response_cancellable = False
        self._current_response_id = None
        if should_cancel and self._response_cancel_event is not None:
            self._response_cancel_event.set()
        if should_cancel and not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        if self._response_task is task:
            self._response_task = None
            self._response_cancel_event = None

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
        except (TypeError, ValueError) as e:
            await self._send(events.error(str(e), code="invalid_request", event_id=data.get("event_id")))
        except Exception as e:
            logger.exception("Error handling event %s", event_type)
            await self._send(events.error(
                str(e),
                code="internal_error",
                event_id=data.get("event_id"),
            ))

    # ── Client event handlers ──────────────────────────────────────────────

    async def _handle_session_update(self, data: dict[str, Any]) -> None:
        previous_turn_detection = self.config.turn_detection_dict
        was_ga = self.config.ga
        had_vad = self.config.vad_enabled
        previous_voice = (self.config.voice, _tts_model_for(self.config))
        # A refused update leaves the session as it was.
        config = copy.deepcopy(self.config)
        config.update_from(data)
        config.input_audio_transcription = _with_whisper_language(config.input_audio_transcription)
        requested = data.get("session", data)
        if (
            config.ga and not was_ga and not requested.get("model")
            and not is_openai_realtime_model(config.model)
            and not await asyncio.to_thread(self._is_tts_model, config.model)
        ):
            # A legacy session's model is its STT model unless the client chose a
            # voice model in ?model=; a GA session's is the one it speaks with.
            config.model = settings.tts_model
        if config.voice.startswith(VOICE_PREFIX) and (config.voice, _tts_model_for(config)) != previous_voice:
            # As in the Live Reader, a named voice that doesn't resolve refuses the update.
            await self._provider_voice(_tts_model_for(config), config.voice)
        language_error = _with_supported_language(config)
        self.config = config

        # Buffered audio is already decoded to 16 kHz PCM, so only turn
        # detection settings affect the buffer.
        vad_loaded = True
        if self.audio_buffer is None or previous_turn_detection != self.config.turn_detection_dict:
            vad_loaded = await self._apply_turn_detection()
        if had_vad and not self.config.vad_enabled:
            # Under server VAD the client owed no commit; in manual mode it owes one from now.
            self._last_commit_at = time.monotonic()

        await self._send(events.session_updated(self.config.to_dict()))
        if language_error:
            await self._send(events.error(language_error, code="language_unsupported", event_id=data.get("event_id")))
        if not vad_loaded:
            await self._send(_vad_unavailable_error(data.get("event_id")))
        self._warm_fillers()

    async def _handle_input_audio_buffer_append(self, data: dict[str, Any]) -> None:
        # Every error below names the append.
        event_id = data.get("event_id")
        # With server VAD the server commits turns, so a client streaming
        # quiet audio is not idle; the receive loop catches silent clients.
        if (
            not self.config.vad_enabled
            and (time.monotonic() - self._last_commit_at) > settings.os_realtime_idle_timeout_s
        ):
            await self._send(events.error(
                "Session idle timeout waiting for commit", code="idle_timeout", event_id=event_id,
            ))
            await self.ws.close(code=4008, reason="Session idle timeout")
            return

        audio_b64 = data.get("audio", "")
        if not isinstance(audio_b64, str):
            await self._send(events.error("Invalid base64 audio data", code="invalid_audio", event_id=event_id))
            return
        if not audio_b64:
            return

        # A 16 MiB append takes seconds to decode and run through Silero; on the event
        # loop that stalled every other request and stream. Events are handled one at a
        # time, so nothing else touches the session's audio meanwhile.
        try:
            pcm16 = await asyncio.get_running_loop().run_in_executor(
                _input_executor, self._decode_input_audio, audio_b64,
            )
        except _InvalidAudio as e:
            await self._send(events.error(str(e), code="invalid_audio", event_id=event_id))
            return

        await self._buffer_input_audio(pcm16, event_id=event_id)

    def _decode_input_audio(self, audio_b64: str) -> bytes:
        """Decode an append's base64 audio to PCM16 at 16 kHz (blocking).

        Raises _InvalidAudio with the message for the client.
        """
        try:
            raw = base64.b64decode(audio_b64, validate=True)
        except (binascii.Error, ValueError):
            raise _InvalidAudio("Invalid base64 audio data") from None

        # Convert to PCM16 16kHz for internal processing
        try:
            held_back = b""
            if self._input_audio is None or self._input_audio.format != self.config.input_audio_format:
                # The old format's last ~1 ms still comes before the new audio
                held_back = self._input_audio.flush() if self._input_audio is not None else b""
                self._input_audio = InputAudioStream(self.config.input_audio_format)
            return held_back + decode_audio_to_pcm16(raw, self._input_audio, target_rate=16000)
        except Exception as e:
            raise _InvalidAudio(str(e)) from e

    async def _buffer_input_audio(self, pcm16: bytes, *, event_id: str | None = None) -> None:
        """Add decoded 16 kHz audio to the input buffer and act on the VAD events it raises. ``event_id`` is the
        client event that brought the audio (an append, or a commit's held-back audio), named by an overflow."""
        try:
            # Silero scores every 512-sample window, so off the event loop too
            vad_events = await asyncio.get_running_loop().run_in_executor(
                _input_executor, self.audio_buffer.append, pcm16,
            )
        except BufferError as e:
            if self.audio_buffer:
                self.audio_buffer.clear()
            await self._send(events.error(str(e), code="buffer_overflow", event_id=event_id))
            return

        committed_from_frame = 0
        for evt in vad_events:
            if evt["type"] == "speech_started":
                self._turn_item_id = events._item_id()
                await self._send(events.input_audio_buffer_speech_started(
                    evt["audio_start_ms"], self._turn_item_id
                ))
                await self._on_user_speech_started()
            elif evt["type"] == "speech_stopped":
                item_id = self._turn_item_id or events._item_id()
                self._turn_item_id = None
                await self._send(events.input_audio_buffer_speech_stopped(
                    evt["audio_end_ms"], item_id
                ))
                # Auto-commit on speech end in VAD mode. The turn's offsets
                # are into the buffer as it was before this frame's commits.
                turn = (
                    evt["_buffer_start_bytes"] - committed_from_frame,
                    evt["_buffer_end_bytes"] - committed_from_frame,
                )
                committed_from_frame = evt["_buffer_end_bytes"]
                # Speech over a reply ended before the deadline: short enough
                # to wait for the transcript instead of cancelling blind.
                self._cancel_barge_in_timer()
                await self._commit_and_transcribe(turn=turn, item_id=item_id, auto_reply=True)

    async def _handle_input_audio_buffer_commit(self, data: dict[str, Any]) -> None:
        if self.audio_buffer is not None and self._input_audio is not None:
            # The commit ends the client's audio here, so the ~1 ms the
            # resampler still holds back belongs to this turn.
            held_back = self._input_audio.flush()
            if held_back:
                await self._buffer_input_audio(held_back, event_id=data.get("event_id"))
        if self.audio_buffer is None or self.audio_buffer.buffered_bytes < MIN_COMMIT_BYTES:
            await self._send(events.error(
                "Input audio buffer is empty or shorter than 50 ms",
                code="input_audio_buffer_commit_empty",
                event_id=data.get("event_id"),
            ))
            return
        # As in OpenAI's API, the client's own commit creates no response, even under server VAD with
        # create_response: its response.create asks for one.
        await self._commit_and_transcribe()

    async def _handle_conversation_item_truncate(self, data: dict[str, Any]) -> None:
        # Clients send this after they cut playback short. The AI's history
        # keeps the whole answer for now (it isn't cut to what was heard), so
        # acknowledging it is all there is to do.
        item_id = data.get("item_id")
        audio_end_ms = data.get("audio_end_ms")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("item_id must be a nonempty string")
        if not isinstance(audio_end_ms, int) or isinstance(audio_end_ms, bool) or audio_end_ms < 0:
            raise ValueError("audio_end_ms must be a non-negative integer")
        await self._send({
            "event_id": events._event_id(),
            "type": "conversation.item.truncated",
            "item_id": item_id,
            "content_index": data.get("content_index", 0),
            "audio_end_ms": audio_end_ms,
        })

    async def _handle_conversation_item_create(self, data: dict[str, Any]) -> None:
        """A typed question: a user message of ``input_text`` parts. It joins the conversation as a spoken turn's
        transcript does (``_after_user_turn``) and, as in OpenAI's API, starts no reply of its own: the client's
        ``response.create`` asks the AI, with a spoken turn's waiting phrases, spoken errors, cancel and barge-in.
        While a reply runs it is refused, so the history never puts it before the answer it would interrupt."""
        event_id = data.get("event_id")
        item_id, parts = _typed_user_message(data)
        text = "\n".join(parts)
        if len(text) > TYPED_TEXT_MAX_CHARS:
            await self._send(events.error(
                f"Message text too long. Max: {TYPED_TEXT_MAX_CHARS} characters",
                code="string_above_max_length", event_id=event_id,
            ))
            return
        if not text.strip():
            raise ValueError("The message has no text")
        if self.assistant is None:
            await self._send(events.error(
                "The AI assistant is off, so nothing would answer a typed message",
                code="assistant_disabled", event_id=event_id,
            ))
            return
        if not self.config.assistant_on:  # the assistant's replies need TTS
            await self._send(events.error("TTS is disabled", code="tts_disabled", event_id=event_id))
            return
        if self.config.type == "transcription":
            raise ValueError("A transcription session doesn't answer typed messages")
        if self._response_task is not None and not self._response_task.done():
            await self._send(events.error(
                "A response is in progress; cancel it with response.cancel before adding a message",
                code="response_in_progress", event_id=event_id,
            ))
            return
        self._turn_measures = (None, None)  # typed: no silence waited for, nothing transcribed
        if not await self._after_user_turn(text, auto_reply=False):
            raise ValueError("The message couldn't join the conversation")
        item = {
            "id": item_id, "object": "realtime.item", "type": "message", "role": "user", "status": "completed",
            "content": [{"type": "input_text", "text": part} for part in parts],
        }
        await self._send(events.conversation_item_created(item))  # conversation.item.added in GA
        if self.config.ga:
            await self._send({
                "event_id": events._event_id(), "type": "conversation.item.done",
                "previous_item_id": None, "item": item,
            })

    async def _handle_input_audio_buffer_clear(self, data: dict[str, Any]) -> None:
        # What the resampler holds back is cleared too, so none of it reaches the next turn.
        held_back = self._input_audio.flush() if self._input_audio is not None else b""
        if self.audio_buffer:
            self.audio_buffer.clear(discarded_bytes=len(held_back))
        self._last_commit_at = time.monotonic()  # nothing is left waiting for a commit
        await self._send(events.input_audio_buffer_cleared())

    async def _handle_response_start(self, data: dict[str, Any]) -> None:
        if self._response_task is not None and not self._response_task.done():
            await self._send(events.error(
                "A response is already in progress",
                code="response_in_progress",
                event_id=data.get("event_id"),
            ))
            return
        cancel_event = threading.Event()
        self._response_cancel_event = cancel_event
        self._response_cancellable = True
        self._cancel_reason = "client_cancelled"
        task = asyncio.create_task(
            self._run_response(data, cancel_event),
            name=f"realtime-response-{self.config.id}",
        )
        self._response_task = task
        task.add_done_callback(self._consume_response_task_result)

    @staticmethod
    def _consume_response_task_result(task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except BaseException:
            logger.exception("Unhandled realtime response task failure")

    async def _run_response(self, data: dict[str, Any], cancel_event: threading.Event) -> None:
        try:
            await self._handle_response_create(data, cancel_event=cancel_event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Error handling event response.create")
            await self._send(events.error(
                redact_paths(str(exc)),
                code="internal_error",
                event_id=data.get("event_id"),
            ))
        finally:
            self._reply_ended_at = time.monotonic()
            task = asyncio.current_task()
            if self._response_task is task:
                self._end_barge_in()
                self._response_task = None
                self._response_cancellable = False
                self._current_response_id = None
            if self._response_cancel_event is cancel_event:
                self._response_cancel_event = None

    async def _finish_cancelled(self, response_obj: dict[str, Any]) -> None:
        response_obj["status"] = "cancelled"
        response_obj["status_details"] = {"type": "cancelled", "reason": self._cancel_reason}
        self._mark_response_terminal(response_obj["id"])
        await self._send(events.response_done(response_obj))

    def _mark_response_terminal(self, response_id: str) -> None:
        self._response_cancellable = False
        if self._current_response_id == response_id:
            self._current_response_id = None

    async def _handle_response_create(
        self,
        data: dict[str, Any],
        *,
        cancel_event: threading.Event | None = None,
    ) -> None:
        if cancel_event is None:
            cancel_event = threading.Event()
        if not settings.tts_enabled:  # transcription keeps working; replies need TTS
            await self._send(events.error(
                "TTS is disabled", code="tts_disabled", event_id=data.get("event_id"),
            ))
            return
        response_data = data.get("response", {})
        modalities = response_data.get("output_modalities", response_data.get("modalities", ["audio", "text"]))

        if modalities == ["text"]:
            await self._send(events.error(
                "Open Speech does not support text-only responses. We handle audio I/O only.",
                code="unsupported_modality",
                event_id=data.get("event_id"),
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

        # Text is spoken word for word; a response with no text goes to the AI.
        ask_assistant = not text_to_speak and self.assistant is not None
        if not text_to_speak and not ask_assistant:
            await self._send(events.error(
                "No text provided for TTS. Include 'instructions' or input text content.",
                code="missing_input",
                event_id=data.get("event_id"),
            ))
            return
        if isinstance(text_to_speak, str) and has_lone_surrogate(text_to_speak):
            # Its transcript and response.done couldn't be sent back.
            await self._send(events.error(
                "The response's text has a lone surrogate (half of a UTF-16 pair, such as half an emoji), which "
                "can't be sent back; send whole characters",
                code="invalid_request", event_id=data.get("event_id"),
            ))
            return
        if text_to_speak and len(text_to_speak) > settings.tts_max_input_length:
            await self._send(events.error(
                f"Input too long. Max: {settings.tts_max_input_length} characters",
                code="string_above_max_length",
                event_id=data.get("event_id"),
            ))
            return
        ai = self._assistant_settings() if ask_assistant else None
        # An AI voice turn logs its timings once it ends.
        timing = self._take_turn_timing() if ai is not None else None
        turn: _StreamedTurn | None = None
        playback: _Playback | None = None

        resp_id = events._response_id()
        self._current_response_id = resp_id
        item_id = events._item_id()

        response_obj = {
            "id": resp_id,
            "object": "realtime.response",
            "status": "in_progress",
            "output": [],
        }
        try:
            await self._send(events.response_created(response_obj))

            loop = asyncio.get_running_loop()
            voice = parse_voice(response_data.get("voice", self.config.voice))
            output_format = self.config.output_audio_format
            output_audio = response_data.get("audio", {}).get("output", {})
            if "voice" in output_audio:
                voice = parse_voice(output_audio["voice"])
            if "format" in output_audio:
                output_format = parse_audio_format(output_audio["format"])
            # A GA session's model is its voice model; a legacy session's is its STT model.
            tts_model = resolve_tts_model(
                response_data.get("model") or (self.config.model if self.config.ga else None), settings,
            )
            speech = _Speech(tts_model, await self._provider_voice(tts_model, voice), self.config.speed, output_format)
            if self.config.ga:
                item = {
                    "id": item_id, "type": "message", "role": "assistant",
                    "status": "in_progress", "content": [],
                }
                await self._send({
                    "event_id": events._event_id(), "type": "response.output_item.added",
                    "response_id": resp_id, "output_index": 0, "item": item,
                })
                await self._send({
                    "event_id": events._event_id(), "type": "response.content_part.added",
                    "response_id": resp_id, "item_id": item_id,
                    "output_index": 0, "content_index": 0,
                    "part": {"type": "output_audio", "transcript": ""},
                })
            common = {
                "response_id": resp_id, "item_id": item_id,
                "output_index": 0, "content_index": 0,
            }
            # Fast TTS finishes a long reply in a fraction of a second. If we
            # sent it all at once the reply would be "done" before the user
            # could talk over it, so when barge-in is on we stay at most
            # `lead` seconds ahead of the client's playback.
            playback = _Playback(resp_id, item_id, output_format, self._audio_lead_s(), loop.time())

            if ai is None:
                transcript = text_to_speak
                synthesis = self._start_synthesis(self._word_for_word(text_to_speak, tts_model), speech, cancel_event)
            else:
                if self._streams(self._client_for(ai)):
                    turn = _StreamedTurn(timing)
                    outcome, synthesis = await self._stream_assistant(ai, playback, speech, cancel_event, turn)
                else:
                    outcome, synthesis = await self._ask_assistant(ai, playback, speech, cancel_event, timing)
                if isinstance(outcome, AssistantError):
                    await self._finish_failed_by_assistant(
                        outcome, synthesis, playback, response_obj, cancel_event, event_id=data.get("event_id"),
                    )
                    return
                thinking = turn.thinking_text() if turn is not None else outcome.thinking
                if thinking and ai.show_thinking:
                    await self._send(events.assistant_thinking(resp_id, item_id, thinking))
                if turn is None:
                    transcript = outcome.text
                    timing.reply_chars = len(transcript)
                    # The reply's text before its audio, so a client can show it while it's read. A streamed
                    # reply's text goes out a sentence at a time instead, each just before its audio (_play).
                    await self._send({
                        "event_id": events._event_id(), "type": "response.audio_transcript.delta",
                        **common, "delta": transcript,
                    })

            if not await self._play(synthesis, playback, cancel_event):
                await self._finish_cancelled(response_obj)
                return

            await self._send(events.response_audio_done(resp_id, item_id, 0, 0))

            # A streamed reply that an error or the token limit cut off after some of it was spoken keeps what was
            # said and is reported as incomplete, with the code, never as a finished answer.
            failure = turn.failure if turn is not None else None
            if turn is not None:
                transcript = turn.kept()
                self._keep_streamed_answer(turn, item_id)
            if timing is not None:
                timing.reply_chars = len(transcript)
            status = "completed" if failure is None else "incomplete"
            response_obj["status"] = status
            if failure is not None:
                response_obj["status_details"] = {
                    "type": "incomplete", "error": {"type": "server_error", "code": failure.code},
                }
                logger.warning(
                    "The AI's streamed reply in realtime session %s stopped after %d characters were spoken (%s)",
                    self.config.id, len(transcript), failure.code,
                )
            response_obj["output"] = [{
                "id": item_id,
                "object": "realtime.item",
                "type": "message",
                "role": "assistant",
                "status": status,
                "content": [{"type": "output_audio" if self.config.ga else "audio", "transcript": transcript}],
            }]
            if self.config.ga and ai is None:
                # Word for word, the transcript follows the audio as it always has.
                await self._send({
                    "event_id": events._event_id(), "type": "response.audio_transcript.delta",
                    **common, "delta": transcript,
                })
            if self.config.ga or ai is not None:
                await self._send({
                    "event_id": events._event_id(), "type": "response.audio_transcript.done",
                    **common, "transcript": transcript,
                })
            if self.config.ga:
                await self._send({
                    "event_id": events._event_id(), "type": "response.content_part.done",
                    **common, "part": response_obj["output"][0]["content"][0],
                })
                await self._send({
                    "event_id": events._event_id(), "type": "response.output_item.done",
                    "response_id": resp_id, "output_index": 0, "item": response_obj["output"][0],
                })
            self._mark_response_terminal(resp_id)
            await self._send(events.response_done(response_obj))
        except asyncio.CancelledError:
            cancel_event.set()
            await self._finish_cancelled(response_obj)
            raise
        except Exception as e:
            logger.exception("TTS synthesis failed in realtime session")
            response_obj["status"] = "failed"
            self._mark_response_terminal(resp_id)
            # A reply the client's response.create asked for names it; one server VAD started has no event_id.
            await self._send(events.error(redact_paths(str(e)), code="tts_error", event_id=data.get("event_id")))
            await self._send(events.response_done(response_obj))
        finally:
            if turn is not None:
                # Cancelled or failed while it streamed: the stream is closed, and what was spoken is kept.
                cancel_event.set()
                await self._close_turn(turn)
                self._keep_streamed_answer(turn, item_id)
                if timing is not None and not timing.reply_chars:
                    timing.reply_chars = len(turn.kept())
            if timing is not None:
                first_audio_at = playback.first_audio_at if playback is not None else None
                logger.info("Realtime turn of session %s: %s", self.config.id, timing.line(first_audio_at))
            self._cancelled_responses.discard(resp_id)
            if self._current_response_id == resp_id:
                self._current_response_id = None

    async def _finish_failed_by_assistant(
        self,
        failure: AssistantError,
        synthesis: tuple[asyncio.Queue[bytes | None], asyncio.Future[None]] | None,
        playback: _Playback,
        response_obj: dict[str, Any],
        cancel_event: threading.Event,
        *,
        event_id: str | None = None,
    ) -> None:
        """The AI gave no answer: say so (error, then its sentence when speak_errors is on), then fail the
        response with the code. The conversation is left as it was. ``event_id`` is the client's response.create
        that asked, if one did."""
        await self._send(events.error(failure.message, error_type="server_error", code=failure.code, event_id=event_id))
        if synthesis is not None:
            try:
                played = await self._play(synthesis, playback, cancel_event)
            except Exception:
                logger.warning("The error sentence for %s couldn't be spoken", failure.code, exc_info=True)
            else:
                if not played:
                    await self._finish_cancelled(response_obj)
                    return
                await self._send(events.response_audio_done(playback.resp_id, playback.item_id, 0, 0))
        response_obj["status"] = "failed"
        response_obj["status_details"] = {"type": "failed", "error": {"type": "server_error", "code": failure.code}}
        self._mark_response_terminal(response_obj["id"])
        await self._send(events.response_done(response_obj))

    # ── Speaking ───────────────────────────────────────────────────────────

    def _start_synthesis(
        self, pieces: Callable[[], list[str]], speech: _Speech, cancel_event: threading.Event,
    ) -> tuple[asyncio.Queue[bytes | None], asyncio.Future[None]]:
        """Synthesize ``pieces()`` (run in the worker) one after another, with one encoder so their
        chunks join without seams. Each encoded chunk is handed over as soon as it exists, so the
        first audio goes out while the rest is still being generated; None ends the queue."""
        loop = asyncio.get_running_loop()
        chunks_ready: asyncio.Queue[bytes | None] = asyncio.Queue()

        def _hand_over(encoded: bytes) -> None:
            if encoded:
                loop.call_soon_threadsafe(chunks_ready.put_nowait, encoded)

        def _synthesize() -> None:
            try:
                sample_rate = self.tts_router.sample_rate_for(speech.model)
                encoder = OutputAudioStream(sample_rate, speech.output_format)
                for text in pieces():
                    if cancel_event.is_set():
                        break
                    self._synthesize_piece(encoder, text, speech, cancel_event, _hand_over)
                if not cancel_event.is_set():
                    _hand_over(encoder.flush())
            finally:
                loop.call_soon_threadsafe(chunks_ready.put_nowait, None)

        synthesis = loop.run_in_executor(_executor, _synthesize)
        # A cancelled response never awaits it; its error is reported only when it is awaited.
        synthesis.add_done_callback(lambda future: future.cancelled() or future.exception())
        return chunks_ready, synthesis

    def _synthesize_piece(
        self, encoder: OutputAudioStream, text: str, speech: _Speech, cancel_event: threading.Event,
        hand_over: Callable[[bytes], None],
    ) -> None:
        """Synthesize one piece of text into ``encoder`` (blocking), handing over each encoded chunk."""
        chunks = self.tts_router.synthesize(
            text=text,
            model=speech.model,
            voice=speech.voice,
            speed=speech.speed,
        )
        try:
            for chunk in chunks:
                if cancel_event.is_set():
                    break
                samples = np.asarray(chunk, dtype=np.float32)
                if samples.size == 0:
                    continue
                hand_over(encoder.encode(float32_to_int16(samples).tobytes()))
        finally:
            close_chunks = getattr(chunks, "close", None)
            if callable(close_chunks):
                close_chunks()

    def _word_for_word(self, text: str, model: str) -> Callable[[], list[str]]:
        def pieces() -> list[str]:
            spoken = self.pronounce_fn(text) if self.pronounce_fn is not None else text
            # As for /v1/audio/speech: what the dictionary makes of the text answers
            # only to the model's own limit, and must still have something to say.
            if not spoken.strip():
                raise ValueError("Input text is empty")
            input_error = validate_tts_input_limit(tts_router=self.tts_router, model_id=model, text=spoken)
            if input_error:
                raise ValueError(input_error)
            return [spoken]

        return pieces

    def _input_limit(self, model: str) -> int | None:
        get_limit = getattr(self.tts_router, "max_input_chars_for", None)
        if not callable(get_limit):
            return None
        try:
            limit = get_limit(model)
        except Exception:  # an external worker can fail any way; the limit is only a refinement
            return None
        return limit if type(limit) is int and limit > 0 else None

    def _speakable(self, text: str, model: str) -> Callable[[], list[str]]:
        """An AI reply is never refused for its length: it is cleaned for speech and spoken in segments,
        each within the model's own limit, even after the pronunciation dictionary."""

        def pieces() -> list[str]:
            limit = self._input_limit(model)
            spoken_pieces: list[str] = []
            for segment in speakable_segments(text, limit):
                spoken_pieces.extend(self._spoken_pieces(segment, limit))
            return spoken_pieces

        return pieces

    def _spoken_pieces(self, segment: str, limit: int | None) -> list[str]:
        """One segment of a reply as it is spoken: through the pronunciation dictionary, and cut again when that
        made it longer than the model's limit; [] when nothing is left to say."""
        spoken = self.pronounce_fn(segment) if self.pronounce_fn is not None else segment
        if not spoken.strip():
            return []
        if limit is not None and len(spoken) > limit:
            return speakable_segments(spoken, limit)
        return [spoken]

    async def _play(
        self,
        synthesis: tuple[asyncio.Queue[bytes | str | None], asyncio.Future[None] | asyncio.Task[None]],
        playback: _Playback,
        cancel_event: threading.Event,
    ) -> bool:
        """Send a synthesis's audio as it comes; False when the response was cancelled. A text item (a streamed
        reply's sentence) goes out as a transcript delta just before that sentence's audio."""
        chunks_ready, worker = synthesis
        try:
            while (item := await chunks_ready.get()) is not None:
                if isinstance(item, str):
                    if not await self._send_transcript_delta(playback, item):
                        return False
                    continue
                if playback.first_audio_at is None:
                    playback.first_audio_at = asyncio.get_running_loop().time()
                if not await self._send_audio(playback, item):
                    return False
        finally:
            cancel_event.set()  # stops the worker if we left the loop early
        await worker
        return playback.resp_id not in self._cancelled_responses

    async def _send_transcript_delta(self, playback: _Playback, text: str) -> bool:
        """A streamed reply's sentence as text, held while the user talks over the reply, as its audio is. False
        when the response was cancelled."""
        await self._playback_gate.wait()
        if playback.resp_id in self._cancelled_responses:
            return False
        await self._send({
            "event_id": events._event_id(), "type": "response.audio_transcript.delta",
            "response_id": playback.resp_id, "item_id": playback.item_id, "output_index": 0, "content_index": 0,
            "delta": text,
        })
        return True

    async def _send_audio(self, playback: _Playback, audio_data: bytes) -> bool:
        """Send audio of the response, held while the user talks over it and paced to playback when
        barge-in is on. False when the response was cancelled."""
        loop = asyncio.get_running_loop()
        for i in range(0, len(audio_data), AUDIO_DELTA_BYTES):
            while True:
                await self._playback_gate.wait()
                now = loop.time()
                playback.ends = max(playback.ends, now)
                ahead = playback.ends - now
                if playback.lead is None or ahead <= playback.lead:
                    break
                await asyncio.sleep(ahead - playback.lead)
            if playback.resp_id in self._cancelled_responses:
                return False
            piece = audio_data[i:i + AUDIO_DELTA_BYTES]
            delta = base64.b64encode(piece).decode("ascii")
            await self._send(events.response_audio_delta(playback.resp_id, playback.item_id, 0, 0, delta))
            playback.ends += len(piece) / playback.bytes_per_s
        return True

    async def _provider_voice(self, model: str, voice: str) -> str:
        """The voice to synthesize with: a named voice (``voice:<id>``) resolved for ``model``."""
        if not voice.startswith(VOICE_PREFIX):
            return voice
        try:
            provider_voice, reference_audio_id = await asyncio.to_thread(resolve_named_voice, model, voice)
        except KeyError as exc:
            raise ValueError("Named voice is unavailable") from exc
        if reference_audio_id:
            raise ValueError("Named voices with a reference recording (cloned voices) can't speak in realtime yet")
        return provider_voice

    # ── The AI assistant ───────────────────────────────────────────────────

    def _assistant_settings(self) -> EffectiveSettings:
        """The session's AI settings now. If its preset has become unusable since session.update (deleted,
        say), the last good settings are kept."""
        try:
            self._assistant_snapshot = self.config.assistant_settings()
        except PresetError as exc:
            if self._assistant_snapshot is None:
                raise
            logger.warning(
                "AI preset %r can't be used now (%s); the realtime session keeps its last settings",
                self.config.assistant_preset, exc.message,
            )
        return self._assistant_snapshot

    async def _after_user_turn(self, transcript: str | None, *, auto_reply: bool = True) -> bool:
        """The user's words go into the conversation, rendered with the preset's user template; with server
        VAD and create_response, a reply to a turn that server VAD ended starts. The client's own commit and a
        typed message (``auto_reply`` false: input_audio_buffer.commit, conversation.item.create) wait for the
        client's response.create, as in OpenAI's API. Returns whether the words joined the conversation."""
        if self.assistant is None or not self.config.assistant_on or self.config.type == "transcription":
            return False
        text = transcript.strip() if isinstance(transcript, str) else ""
        if not text:
            return False
        if self._response_task is not None and not self._response_task.done():
            # Speech over a reply that _resolve_barge_in let go on ("mm-hmm"): not a question.
            return False
        ai = self._assistant_settings()
        self.conversation.add_user(assistant_prompts.render(ai.user_template, text=text))
        turn_detection = self.config.turn_detection
        if not auto_reply or turn_detection is None or not turn_detection.create_response:
            return True  # the client asks for the reply with response.create
        await self._handle_response_start({"type": "response.create", "response": {}})
        return True

    async def _ask_assistant(
        self, ai: EffectiveSettings, playback: _Playback, speech: _Speech, cancel_event: threading.Event,
        timing: _TurnTiming | None = None,
    ) -> tuple[AssistantReply | AssistantError, tuple[asyncio.Queue[bytes | None], asyncio.Future[None]] | None]:
        """Ask the AI for its whole reply while the waiting phrases play (Stream replies off). Returns its reply
        (added to the conversation) or its error, and the synthesis of what to say now (None for an error that
        isn't spoken), once any phrase in progress has finished. A cancel stops the request and the phrases."""
        messages = assistant_prompts.messages_for(ai, self.conversation.for_request(ai.history_turns))
        arrived = asyncio.Event()
        loop = asyncio.get_running_loop()
        if timing is not None:
            timing.asked_at = loop.time()
        request = asyncio.create_task(self._client_for(ai).reply(messages, ai), name=f"realtime-assistant-{self.config.id}")
        fillers = asyncio.create_task(
            self._say_fillers(ai, playback, speech, arrived), name=f"realtime-fillers-{self.config.id}",
        )
        synthesis = None
        try:
            try:
                outcome: AssistantReply | AssistantError = await request
            except AssistantError as exc:
                outcome = exc
            arrived.set()  # no phrase starts from now on
            if timing is not None:
                timing.done_at = loop.time()
                if isinstance(outcome, AssistantReply):  # one whole answer: its first token came with its end
                    timing.first_token_at = timing.first_answer_at = timing.done_at
            if isinstance(outcome, AssistantReply):
                self.conversation.add_assistant(outcome.text, playback.item_id)
                synthesis = self._start_synthesis(self._speakable(outcome.text, speech.model), speech, cancel_event)
            elif ai.speak_errors:
                sentence = ai.error_sentence(outcome.code)
                synthesis = self._start_synthesis(self._speakable(sentence, speech.model), speech, cancel_event)
            await fillers  # a phrase that has started is finished first
        except BaseException:
            arrived.set()
            request.cancel()
            fillers.cancel()
            await asyncio.gather(request, fillers, return_exceptions=True)
            raise
        return outcome, synthesis

    # ── A streamed reply ───────────────────────────────────────────────────

    def _client_for(self, ai: EffectiveSettings):
        """The client for the turn's saved assistant (the preset's, the session's, or the active one); the
        session's own with none saved."""
        if ai.assistant:
            from src import assistant as assistants

            return assistants.get_client(ai.assistant) or _NoServer()
        return self.assistant

    @staticmethod
    def _streams(client) -> bool:
        """Whether a voice turn asks for a streamed reply: Stream replies (OS_ASSISTANT_STREAM) on the client."""
        return getattr(client, "stream", False) is True and callable(getattr(client, "stream_reply", None))

    async def _stream_assistant(
        self, ai: EffectiveSettings, playback: _Playback, speech: _Speech, cancel_event: threading.Event,
        turn: _StreamedTurn,
    ) -> tuple[AssistantError | None, tuple[asyncio.Queue[bytes | str | None], asyncio.Task[None] | asyncio.Future[None]] | None]:
        """Ask the AI for a streamed reply while the waiting phrases play; for them, "the reply has arrived" is the
        first sentence of the answer being ready. Returns once it is, and any phrase in progress has finished: None
        and the synthesis that speaks the sentences as they come. When the request ended before any of the answer
        could be spoken, returns its error and the synthesis of its sentence (None when errors aren't spoken), as
        ``_ask_assistant`` does. A cancel stops the phrases and closes the stream; the caller closes it in any case
        once the response ends (``_close_turn``)."""
        messages = assistant_prompts.messages_for(ai, self.conversation.for_request(ai.history_turns))
        turn.timing.asked_at = turn.loop.time()
        arrived = asyncio.Event()
        turn.reader = asyncio.create_task(
            self._read_reply(self._client_for(ai), messages, ai, turn), name=f"realtime-assistant-{self.config.id}",
        )
        turn.fillers = asyncio.create_task(
            self._say_fillers(ai, playback, speech, arrived), name=f"realtime-fillers-{self.config.id}",
        )
        first = asyncio.ensure_future(turn.first_sentence.wait())
        try:
            await asyncio.wait({turn.reader, first}, return_when=asyncio.FIRST_COMPLETED)
            arrived.set()  # no phrase starts from now on
            if turn.handed == 0 and turn.failure is not None:
                synthesis = None
                if ai.speak_errors:
                    sentence = ai.error_sentence(turn.failure.code)
                    synthesis = self._start_synthesis(self._speakable(sentence, speech.model), speech, cancel_event)
                await turn.fillers  # a phrase that has started is finished first
                return turn.failure, synthesis
            chunks_ready: asyncio.Queue[bytes | str | None] = asyncio.Queue()
            turn.speaker = asyncio.create_task(
                self._speak_sentences(turn, speech, cancel_event, chunks_ready),
                name=f"realtime-speaker-{self.config.id}",
            )
            await turn.fillers  # a phrase that has started is finished first; the first sentence is made meanwhile
            return None, (chunks_ready, turn.speaker)
        except BaseException:
            arrived.set()
            await self._close_turn(turn)
            raise
        finally:
            first.cancel()

    async def _read_reply(
        self, assistant: AssistantClient, messages: list[dict[str, Any]], ai: EffectiveSettings, turn: _StreamedTurn,
    ) -> None:
        """Read the streamed reply into ``turn``, which hands each sentence to the voice as it completes. A cancel
        (the response stopped) closes the stream."""
        try:
            reply = await assistant.stream_reply(messages, ai, turn.on_piece)
        except AssistantError as exc:
            await turn.end(failure=exc)
        except Exception:  # not the AI's error: the turn still ends, and the log says why
            logger.exception("Reading the AI's streamed reply failed in realtime session %s", self.config.id)
            await turn.end(failure=AssistantError("The AI's reply couldn't be read", ASSISTANT_BAD_RESPONSE))
        else:
            await turn.end(reply)

    async def _speak_sentences(
        self, turn: _StreamedTurn, speech: _Speech, cancel_event: threading.Event,
        chunks_ready: asyncio.Queue[bytes | str | None],
    ) -> None:
        """Speak the turn's sentences one after another as they are handed over, with one encoder so they join
        without seams. Each sentence's text goes into the queue just before its audio; None ends the queue."""
        loop = asyncio.get_running_loop()
        voice: dict[str, Any] = {}  # the encoder and the model's limit, made by the first sentence's worker

        def hand_over(encoded: bytes) -> None:
            if encoded:
                loop.call_soon_threadsafe(chunks_ready.put_nowait, encoded)

        def speak(text: str) -> None:
            if "encoder" not in voice:
                voice["encoder"] = OutputAudioStream(self.tts_router.sample_rate_for(speech.model), speech.output_format)
                voice["limit"] = self._input_limit(speech.model)
            for piece in self._spoken_pieces(text, voice["limit"]):
                if cancel_event.is_set():
                    return
                self._synthesize_piece(voice["encoder"], piece, speech, cancel_event, hand_over)

        def flush() -> None:
            hand_over(voice["encoder"].flush())

        try:
            while (sentence := await turn.sentences.get()) is not None:
                text, covers = sentence
                if cancel_event.is_set():
                    break
                if covers:
                    chunks_ready.put_nowait(covers)
                if text.strip():
                    # One at a time, in the realtime pool: a worker never waits on the AI for the next sentence.
                    await loop.run_in_executor(_executor, speak, text.strip())
            if not cancel_event.is_set() and "encoder" in voice:
                await loop.run_in_executor(_executor, flush)
        finally:
            chunks_ready.put_nowait(None)

    async def _close_turn(self, turn: _StreamedTurn) -> None:
        """Stop a streamed reply's tasks: the reader (which closes the stream, so the AI server can stop), the
        sentence cutter, the speaker and the phrases; and collect them."""
        tasks = [task for task in (turn.reader, turn.cutter, turn.speaker, turn.fillers) if task is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _keep_streamed_answer(self, turn: _StreamedTurn, item_id: str) -> None:
        """The history keeps what was handed to the voice, once: all of the answer when its stream ended well,
        the part spoken when it was interrupted or cut off, nothing when nothing was spoken."""
        if turn.recorded:
            return
        turn.recorded = True
        kept = turn.kept()
        if kept:
            self.conversation.add_assistant(kept, item_id)

    def _take_turn_timing(self) -> _TurnTiming:
        """The timing of the turn this reply answers, which is used once."""
        silence_ms, stt_ms = self._turn_measures
        self._turn_measures = (None, None)
        return _TurnTiming(silence_ms=silence_ms, stt_ms=stt_ms)

    async def _say_fillers(
        self, ai: EffectiveSettings, playback: _Playback, speech: _Speech, arrived: asyncio.Event,
    ) -> None:
        """The waiting phrases: the first after filler_after_ms, then the next filler_every_ms after the last one
        has played, at most filler_max, until the reply arrives."""
        phrases = list(ai.fillers)
        if not phrases or ai.filler_max <= 0:
            return
        loop = asyncio.get_running_loop()
        delay = ai.filler_after_ms / 1000
        for _ in range(ai.filler_max):
            if await _wait_event(arrived, delay):
                return
            delay = ai.filler_every_ms / 1000
            phrase = self._next_filler(phrases)
            audio = self._filler_audio(speech, phrase)
            if not audio.done():
                arrival = asyncio.ensure_future(arrived.wait())
                self._filler_awaited = audio
                try:
                    await asyncio.wait({audio, arrival}, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    self._filler_awaited = None
                    arrival.cancel()
            if arrived.is_set():
                return  # no phrase starts once the reply has arrived
            encoded = _result_or_none(audio)
            if not encoded:
                continue
            self._last_filler = phrase
            await self._send(events.assistant_filler(playback.resp_id, playback.item_id, phrase))
            if not await self._send_audio(playback, encoded):
                return
            # The gap starts when the client has played the phrase, not when it was sent: up to
            # OS_REALTIME_AUDIO_LEAD_MS of it is still to play when paced, all of it when not. So only one phrase
            # is ever ahead of the listener, and the reply follows it.
            delay += max(0.0, playback.ends - loop.time())

    def _next_filler(self, phrases: list[str]) -> str:
        """A phrase at random, never the one said last in this session (unless it is the only one)."""
        choices = [phrase for phrase in phrases if phrase != self._last_filler] or phrases
        return random.choice(choices)

    def _filler_audio(self, speech: _Speech, phrase: str) -> asyncio.Future[bytes]:
        """The phrase's encoded audio, made once per session for each model, voice, speed and format."""
        key = (speech, phrase)
        future = self._filler_cache.get(key)
        if future is not None and not (future.done() and _result_or_none(future) is None):
            return future
        self._filler_cache.pop(key, None)
        # As run_in_executor, keeping the pool's job to tell whether a cancel stopped it.
        job = _executor.submit(self._phrase_audio, speech, phrase)
        future = asyncio.wrap_future(job)
        self._phrase_jobs[future] = job
        future.add_done_callback(self._forget_phrase_job)
        future.add_done_callback(_log_phrase_failure)
        self._filler_cache[key] = future
        while len(self._filler_cache) > FILLER_CACHE_MAX:
            # The oldest goes, and is cancelled if it is still to be made.
            oldest = next(old for old, cached in self._filler_cache.items() if cached is not self._filler_awaited)
            self._cancel_phrase(self._filler_cache.pop(oldest))
        return future

    def _forget_phrase_job(self, future: asyncio.Future[bytes]) -> None:
        self._phrase_jobs.pop(future, None)

    def _cancel_phrase(self, future: asyncio.Future[bytes]) -> None:
        """Cancel a phrase: one not started never runs; one already running ends unused, and is remembered so the
        session's next phrase waits for it."""
        job = self._phrase_jobs.pop(future, None)
        future.cancel()
        if job is not None and not job.cancel() and not job.done():
            self._phrases_left_running.add(job)

    async def _after_phrases_left_running(self) -> None:
        """Wait until every phrase the session left while it ran has ended."""
        while running := {job for job in self._phrases_left_running if not job.done()}:
            self._phrases_left_running = running
            ended = [asyncio.wrap_future(job) for job in running]
            try:
                await asyncio.wait(ended)
            finally:
                for waiter in ended:
                    waiter.cancel()
        self._phrases_left_running = set()

    def _phrase_audio(self, speech: _Speech, phrase: str) -> bytes:
        """Synthesize one waiting phrase to the output format (blocking)."""
        encoder = OutputAudioStream(self.tts_router.sample_rate_for(speech.model), speech.output_format)
        spoken = self.pronounce_fn(phrase) if self.pronounce_fn is not None else phrase
        if not spoken.strip():
            return b""
        audio = bytearray()
        chunks = self.tts_router.synthesize(text=spoken, model=speech.model, voice=speech.voice, speed=speech.speed)
        try:
            for chunk in chunks:
                samples = np.asarray(chunk, dtype=np.float32)
                if samples.size:
                    audio += encoder.encode(float32_to_int16(samples).tobytes())
        finally:
            close_chunks = getattr(chunks, "close", None)
            if callable(close_chunks):
                close_chunks()
        audio += encoder.flush()
        return bytes(audio)

    def _warm_fillers(self) -> None:
        """Make the session's waiting phrases in the background, so the first one plays at once. The phrases
        still to be made for settings the session has left are cancelled."""
        if self._filler_warm_task is not None:
            self._filler_warm_task.cancel()
            self._filler_warm_task = None
        if self.assistant is None or not self.config.assistant_on or self.config.type == "transcription":
            self._drop_phrases_being_made(keep=set())
            return
        ai = self._assistant_settings()
        if not ai.fillers or ai.filler_max <= 0:
            self._drop_phrases_being_made(keep=set())
            return
        self._filler_warm_task = asyncio.create_task(
            self._warm_filler_cache(ai.fillers), name=f"realtime-fillers-warm-{self.config.id}",
        )

    def _drop_phrases_being_made(self, *, keep: set[tuple[_Speech, str]]) -> None:
        """Cancel each phrase still being made that isn't in ``keep``, except the one a reply waits for. One not
        started yet never runs; one already running can't be stopped, and ends unused."""
        for key, future in list(self._filler_cache.items()):
            if key not in keep and not future.done() and future is not self._filler_awaited:
                del self._filler_cache[key]
                self._cancel_phrase(future)

    async def _warm_filler_cache(self, phrases: tuple[str, ...]) -> None:
        model = _tts_model_for(self.config)
        try:
            voice = await self._provider_voice(model, self.config.voice)
        except Exception:
            logger.debug("Waiting phrases not made ahead: the session's voice doesn't resolve", exc_info=True)
            self._drop_phrases_being_made(keep=set())
            return
        speech = _Speech(model, voice, self.config.speed, self.config.output_audio_format)
        self._drop_phrases_being_made(keep={(speech, phrase) for phrase in phrases})
        for phrase in phrases:
            # One at a time: the workers also transcribe each turn. A wait doesn't cancel the phrase.
            await self._after_phrases_left_running()
            await asyncio.wait({self._filler_audio(speech, phrase)})

    async def _handle_response_cancel(self, data: dict[str, Any]) -> None:
        if not self._response_cancellable:
            await self._send(events.error(
                "There is no response in progress to cancel",
                code="response_cancel_not_active",
                event_id=data.get("event_id"),
            ))
            return
        await self._cancel_response("client_cancelled")

    async def _cancel_response(self, reason: str) -> None:
        if not self._response_cancellable:
            return
        self._cancel_reason = reason
        self._response_cancellable = False
        if self._current_response_id:
            self._cancelled_responses.add(self._current_response_id)
            self._current_response_id = None
        if self._response_cancel_event is not None:
            self._response_cancel_event.set()
        task = self._response_task
        if task is not None and not task.done():
            cancel_event = self._response_cancel_event
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            if self._response_task is task:
                self._response_task = None
            if self._response_cancel_event is cancel_event:
                self._response_cancel_event = None

    # ── Barge-in ───────────────────────────────────────────────────────────

    def _audio_lead_s(self) -> float | None:
        """How far ahead of playback to send reply audio; None means no limit."""
        td = self.config.turn_detection
        lead_ms = settings.os_realtime_audio_lead_ms
        if td is None or not td.interrupt_response or lead_ms <= 0:
            return None
        return lead_ms / 1000

    async def _on_user_speech_started(self) -> None:
        """The user started talking. If we are mid-reply, hold our audio."""
        td = self.config.turn_detection
        if not self._response_cancellable or td is None or not td.interrupt_response:
            return
        if self._barge_in_pending:
            return
        if self.config.input_audio_transcription is None:
            # Without a transcript we can't tell "mm-hmm" from a real interruption.
            await self._cancel_response("turn_detected")
            return
        self._barge_in_pending = True
        self._playback_gate.clear()
        self._barge_in_timer = asyncio.create_task(self._barge_in_deadline())

    async def _barge_in_deadline(self) -> None:
        """Long speech over a reply is an interruption, transcript or not.

        speech_stopped arrives silence_duration_ms after the user goes quiet
        and cancels this timer, so waiting that much longer than the limit
        means only speech that really lasted the limit gets here.
        """
        silence_ms = self.config.turn_detection.silence_duration_ms if self.config.turn_detection else 0
        await asyncio.sleep((settings.os_realtime_barge_in_max_ms + silence_ms) / 1000)
        if self._barge_in_pending:
            self._barge_in_timer = None
            await self._interrupt()

    async def _resolve_barge_in(self, transcript: str | None) -> None:
        """Decide once the overlapping speech has been transcribed."""
        if not self._barge_in_pending:
            return
        if transcript is not None and transcript.strip() and not is_backchannel(transcript):
            await self._interrupt()
        else:
            self._end_barge_in()  # an acknowledgement or noise: keep talking

    async def _interrupt(self) -> None:
        self._end_barge_in()
        await self._cancel_response("turn_detected")

    def _end_barge_in(self) -> None:
        self._barge_in_pending = False
        self._playback_gate.set()
        self._cancel_barge_in_timer()

    def _cancel_barge_in_timer(self) -> None:
        timer, self._barge_in_timer = self._barge_in_timer, None
        if timer is not None and timer is not asyncio.current_task() and not timer.done():
            timer.cancel()

    # ── Internal helpers ───────────────────────────────────────────────────

    async def _commit_and_transcribe(
        self, turn: tuple[int, int] | None = None, item_id: str | None = None, *, auto_reply: bool = False,
    ) -> None:
        """Commit a VAD turn's (start, end) buffer range, or the whole buffer, and run STT. ``auto_reply``
        (server VAD's end of turn) lets create_response start the AI's reply; a client's commit doesn't."""
        if self.audio_buffer is None:
            return

        if turn is None:
            audio_data = self.audio_buffer.commit()
        else:
            audio_data = self.audio_buffer.commit_through(turn[1], start=turn[0])
        self._last_commit_at = time.monotonic()
        if len(audio_data) < MIN_COMMIT_BYTES:
            await self._resolve_barge_in(None)
            return

        item_id = item_id or events._item_id()
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
        if self.config.ga:
            item["status"] = "completed"
        await self._send(events.conversation_item_created(item))
        if self.config.ga:
            await self._send({
                "event_id": events._event_id(), "type": "conversation.item.done",
                "previous_item_id": None, "item": item,
            })

        transcription = self.config.input_audio_transcription
        if transcription is None:  # the client turned transcription off
            await self._resolve_barge_in(None)
            return
        model = _stt_model_for(self.config)

        # Transcribe in executor
        loop = asyncio.get_running_loop()
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
                    language=transcription.get("language") or None,
                    response_format="json",
                    temperature=0.0,
                    prompt=transcription.get("prompt") or None,
                    temperature_fallback=False,  # one decode per turn keeps replies prompt
                )

            started = loop.time()
            result = await loop.run_in_executor(_executor, _transcribe)
            stt_ms = round((loop.time() - started) * 1000)
        except Exception as e:
            logger.exception("Transcription failed in realtime session")
            message = redact_paths(str(e))  # the log keeps the paths; the client gets file names
            await self._send(events.conversation_item_input_audio_transcription_failed(
                item_id, 0, message
            ))
            await self._send(events.error(message, code="transcription_error"))
            await self._resolve_barge_in(None)
            return

        transcript = result.get("text", "") if isinstance(result, dict) else str(result)

        await self._send(events.conversation_item_input_audio_transcription_completed(
            item_id, 0, transcript
        ))
        await self._resolve_barge_in(transcript)
        # For the reply's timing line: the silence server VAD waited for (a turn it ended), and the decode.
        td = self.config.turn_detection
        self._turn_measures = (td.silence_duration_ms if auto_reply and td is not None else None, stt_ms)
        await self._after_user_turn(transcript, auto_reply=auto_reply)


def _tts_model_for(config: SessionConfig) -> str:
    """The TTS model a session speaks with: a GA session's model, else TTS_MODEL."""
    return resolve_tts_model(config.model if config.ga else None, settings)


# Sessions that may be waiting on the AI; they stop at shutdown, before the shared client closes.
_assistant_sessions: set[RealtimeSession] = set()
_shutdown_registered_for: AssistantClient | None = None


async def _close_assistant_sessions() -> None:
    """At shutdown (src.assistant.on_close), before the shared AI client closes: stop every reply that may
    still be waiting on it, and ask it nothing more."""
    global _shutdown_registered_for
    _shutdown_registered_for = None
    for session in list(_assistant_sessions):
        session.assistant = None
        await session.close()


def _register_assistant_shutdown(assistant: AssistantClient) -> None:
    """Register _close_assistant_sessions once for each shared client."""
    global _shutdown_registered_for
    if _shutdown_registered_for is not assistant:
        on_close(_close_assistant_sessions)
        _shutdown_registered_for = assistant


# Handler dispatch table
_CLIENT_HANDLERS: dict[str, Any] = {
    "session.update": RealtimeSession._handle_session_update,
    "input_audio_buffer.append": RealtimeSession._handle_input_audio_buffer_append,
    "input_audio_buffer.commit": RealtimeSession._handle_input_audio_buffer_commit,
    "input_audio_buffer.clear": RealtimeSession._handle_input_audio_buffer_clear,
    "response.create": RealtimeSession._handle_response_start,
    "response.cancel": RealtimeSession._handle_response_cancel,
    "conversation.item.truncate": RealtimeSession._handle_conversation_item_truncate,
    "conversation.item.create": RealtimeSession._handle_conversation_item_create,
}


async def realtime_endpoint(
    websocket: WebSocket,
    tts_router: TTSRouter,
    model: str = "",
    pronounce_fn: Callable[[str], str] | None = None,
    assistant: AssistantClient | None = None,
) -> None:
    """Main WebSocket handler for /v1/realtime."""
    protocols = websocket.scope.get("subprotocols", [])
    selected_protocol = "realtime" if "realtime" in protocols else None
    await websocket.accept(subprotocol=selected_protocol)

    session_key = id(websocket)
    # The refusal's close handshake runs outside the lock: a client that doesn't answer it would hold every other
    # session's start and end for up to about 20 s.
    async with _active_realtime_lock:
        full = len(_active_realtime_sessions) >= settings.os_stream_max_connections
        if not full:
            _active_realtime_sessions.add(session_key)
    if full:
        await websocket.close(code=1013, reason="Too many concurrent realtime sessions")
        return

    ga = selected_protocol is None and "realtime=v1" not in websocket.headers.get("openai-beta", "")
    session = RealtimeSession(
        websocket, tts_router, model=model, ga=ga, pronounce_fn=pronounce_fn, assistant=assistant,
    )
    if assistant is not None:
        _register_assistant_shutdown(assistant)
        _assistant_sessions.add(session)

    try:
        await session.initialize()
        idle_timeout = settings.os_realtime_idle_timeout_s
        wait = idle_timeout
        while True:
            try:
                message = await asyncio.wait_for(websocket.receive(), timeout=wait)
            except asyncio.TimeoutError:
                # A reply paced to playback keeps a listening client busy, so
                # the idle window only starts once the reply has finished.
                wait = idle_timeout - session.seconds_since_reply()
                if wait > 0:
                    continue
                await session._send(events.error("Session idle timeout", code="idle_timeout"))
                await websocket.close(code=4008, reason="Session idle timeout")
                break
            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000), message.get("reason"))
            wait = idle_timeout
            raw = message.get("text")
            if raw is None:
                await session._send(events.error(
                    "Binary frames are not supported; send JSON events as text frames",
                    code="invalid_event",
                ))
                continue
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
                    event_id=data.get("event_id") if isinstance(data, dict) else None,
                ))
                continue

            await session.handle_event(data)

    except WebSocketDisconnect:
        logger.debug("Realtime client disconnected (session %s)", session.config.id)
    except Exception:
        # A reply still streaming when the client leaves marks the socket
        # disconnected, and the next receive raises RuntimeError.
        if websocket.application_state == WebSocketState.DISCONNECTED:
            logger.debug("Realtime client disconnected (session %s)", session.config.id)
        else:
            logger.exception("Realtime session error")
    finally:
        _assistant_sessions.discard(session)
        await session.close()
        async with _active_realtime_lock:
            _active_realtime_sessions.discard(session_key)
        # The input audio in numbers, never its content or the text, as /v1/audio/stream logs it
        levels = session.audio_buffer.levels if session.audio_buffer is not None else InputLevels()
        logger.info("Realtime session %s ended (%s)", session.config.id, levels.summary())
