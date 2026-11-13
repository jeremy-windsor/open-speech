"""Send anything to the AI: ``POST /api/assistant/ask``.

One operation: an input goes to the AI through a preset, and its reply comes back shown, spoken or saved.

- **Inputs** (``input.kind``): ``text`` (``{text}``); ``transcript`` (``{text}``, or ``{history_id}``: a history
  entry's text, read on the server); ``document`` (``{document_id}``, with ``chapters: [first, last]``,
  ``blocks: [first, last]``, a selection's ``block_start`` and ``block_end``, or one sentence's ``block`` and
  ``sentence``, or none of them for all of it). A stored thing is read on the server, so the browser never ships a
  whole book.
- **The message**: the listener's own ``question``, if any, then for a document the line naming it and the chapter
  (``From "<title>", chapter "<chapter>":``; none when the preset's template uses ``{title}`` or ``{chapter}``), then
  the rendered template, each followed by a blank line; a job sends them before every piece. A sentence is counted
  as the Reader counts its block's sentences, and is always answered at once.
- **Short or long.** An input that fits in one piece of the preset's ``chunk_chars`` (a document's part cut as
  book jobs cut a chapter) is answered at once. A longer one becomes a job on the book jobs' runner
  (``src/reader/assistant_jobs.py``, ``get_runner()``): a document's part as it is, and a long text or transcript
  kept as a Reader document first, so the job can be resumed. That document is for listening (pasting the text
  into the Reader gives the same); the job keeps the text itself and sends it as it was sent, cut at blank lines and
  never parsed, so the length changes how a text is sent, not what the AI gets. Jobs share the one slot and the
  Reader's job routes.
- **Outputs** (``output``): the reply's text is always in the answer. ``speech`` speaks it through the TTS router
  in the model, voice, format and speed asked, cleaned and split as the Live Reader speaks, and sends the audio
  back in base64; ``document`` saves it as a Reader document, as book jobs save theirs. A job's answers are always
  saved, as its result document; a job is never spoken here (play its result in the Reader).

The AI is reached only through ``presets.effective()``, ``prompts.messages_for()`` and the shared client's
``reply()``, with its error codes, and nothing of the AI server's answer is echoed in an error. With speech asked,
an AI error is also spoken, in the preset's sentence for its code, unless the preset's ``speak_errors`` is off.
``stream`` is accepted only as ``false`` for now: a streamed reply will take ``true`` with the same body.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from src import assistant
from src.assistant import presets, prompts
from src.assistant.errors import ASSISTANT_EMPTY_REPLY, ASSISTANT_TIMEOUT, AssistantError
from src.assistant.speech import speakable_segments
from src.cache.tts_cache import PCM_SAMPLE_RATE
from src.reader import assistant_jobs
from src.services import tts as tts_service
from src.tts.external import ExternalProviderError
from src.tts.pipeline import encode_audio, get_content_type
from src.tts.router import NoTTSBackendsError
from src.voice_identities import resolve_named_voice

logger = logging.getLogger(__name__)

FIELDS = ("preset", "input", "question", "output", "overrides", "stream")
INPUT_KINDS = presets.INPUT_KINDS
INPUT_FIELDS = {
    "text": ("kind", "text"),
    "transcript": ("kind", "text", "history_id"),
    "document": ("kind", "document_id", "chapters", "blocks", "block_start", "block_end", "block", "sentence"),
}
QUESTION_MAX_CHARS = 2000  # characters (code points), as len() counts them
OUTPUT_FIELDS = ("text", "speech", "document")
SPEECH_FIELDS = ("model", "voice", "format", "speed")
DOCUMENT_FIELDS = ("title",)
ID_MAX_CHARS = 64
NAME_MAX_CHARS = 200  # a model's or a voice's name, as the Reader's render takes them
SPEED_RANGE = (0.25, 4.0)
DEFAULT_FORMAT = "mp3"
TITLE_MAX_CHARS = assistant_jobs.TITLE_MAX_CHARS
TRANSCRIPT_TITLE = "Transcript"
SNIPPET_CHARS = 60
MB = 1024 * 1024


class AskError(Exception):
    """A request refused before the AI is asked: its HTTP status, error code and message, and the field."""

    def __init__(self, status: int, code: str, message: str, param: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.param = param

    def content(self) -> dict[str, Any]:
        error: dict[str, Any] = {"message": self.message, "code": self.code}
        if self.param is not None:
            error["param"] = self.param
        return {"error": error}


def _invalid(param: str, message: str) -> AskError:
    return AskError(422, "validation_error", message, param)


@dataclass(frozen=True)
class Speech:
    """``output.speech`` with the server's defaults filled in."""

    model: str
    voice: str
    format: str
    speed: float


@dataclass(frozen=True)
class AskRequest:
    """A checked body. ``range_form`` says how a document's part was named (``chapters``, ``blocks``, ``selection``
    or ``sentence``), for an error about it."""

    kind: str
    preset: str | None = None
    overrides: Any = None
    question: str | None = None
    text: str | None = None
    history_id: str | None = None
    document_id: str | None = None
    chapters: tuple[int, int] | None = None
    blocks: tuple[int, int] | None = None
    block: int | None = None
    sentence: int | None = None
    range_form: str | None = None
    speech: Speech | None = None
    document: bool = False
    title: str | None = None

    @property
    def preset_id(self) -> str:
        """The preset asked for, else ``book`` for a document and ``voice`` for the rest."""
        if self.preset is not None:
            return self.preset
        return "book" if self.kind == "document" else "voice"


# ── The body ─────────────────────────────────────────────────────────────────


def parse(body: Any, settings) -> AskRequest:
    """Check the body's shape and bounds (nothing is read yet). Raises AskError (422, with ``param``)."""
    if not isinstance(body, Mapping):
        raise _invalid("body", "the body must be a JSON object: {preset?, input, question?, output?, overrides?, stream?}")
    for name in body:
        if name not in FIELDS:
            raise _invalid(str(name), f"{name} is not an ask field; send {', '.join(FIELDS)}")
    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise _invalid("stream", "stream must be true or false")
    if stream:
        raise _invalid("stream", "stream: true isn't available yet; send false, or leave it out, for the whole reply")
    preset = body.get("preset")
    if preset is not None and not isinstance(preset, str):
        raise _invalid("preset", "preset must be an AI preset id")
    speech, document, title = _output(body.get("output"), settings)
    return AskRequest(
        **_input(body.get("input"), settings), preset=preset, overrides=body.get("overrides"),
        question=_question(body.get("question")), speech=speech, document=document, title=title,
    )


def _question(value: Any) -> str | None:
    """The listener's own question: the input's text rules (not blank, whole characters), up to QUESTION_MAX_CHARS
    characters; sent as typed. None when left out or null."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise _invalid("question", "question must be the text of your question, or left out for none")
    if presets.has_lone_surrogate(value):
        raise _invalid("question", presets.lone_surrogate_message("question"))
    if len(value) > QUESTION_MAX_CHARS:
        raise _invalid("question", f"question must be at most {QUESTION_MAX_CHARS:,} characters")
    return value


def _input(value: Any, settings) -> dict[str, Any]:
    kinds = ", ".join(INPUT_KINDS)
    if not isinstance(value, Mapping):
        raise _invalid("input", f"input must be an object: {{kind, ...}}, with kind one of {kinds}")
    kind = value.get("kind")
    if not isinstance(kind, str) or kind not in INPUT_KINDS:
        raise _invalid("input.kind", f"input.kind must be one of {kinds}")
    allowed = INPUT_FIELDS[kind]
    for name in value:
        if name not in allowed:
            raise _invalid(f"input.{name}", f"input.{name} is not a field of a {kind} input; send {', '.join(allowed)}")
    if kind == "document":
        return _document_input(value)
    if kind == "transcript" and value.get("history_id") is not None:
        if value.get("text") is not None:
            raise _invalid("input.history_id", "send a transcript's text or its history_id, not both")
        return {"kind": kind, "history_id": _identifier("input.history_id", value["history_id"])}
    return {"kind": kind, "text": check_text(value.get("text"), settings)}


def check_text(value: Any, settings, param: str = "input.text") -> str:
    """Text to send: not blank, whole characters, and no more than the Reader takes, since a long text is kept as a
    Reader document (READER_MAX_DOCUMENT_CHARS and READER_MAX_UPLOAD_MB)."""
    if not isinstance(value, str) or not value.strip():
        raise _invalid(param, f"{param} must be the text to send")
    if presets.has_lone_surrogate(value):
        raise _invalid(param, presets.lone_surrogate_message(param))
    limit = settings.reader_max_document_chars
    if len(value) > limit or len(value.encode("utf-8")) > settings.reader_max_upload_mb * MB:
        raise _invalid(param, f"{param} must be at most {limit:,} characters and {settings.reader_max_upload_mb} MB, "
                              "as much as the Reader takes")
    return value


def _identifier(param: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > ID_MAX_CHARS or not value.isprintable():
        raise _invalid(param, f"{param} must be an id of up to {ID_MAX_CHARS} characters")
    return value


def _index(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _ordered_pair(value: Any) -> bool:
    return isinstance(value, list) and len(value) == 2 and all(_index(item) for item in value) and value[0] <= value[1]


def _document_input(value: Mapping[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {"kind": "document", "document_id": _identifier("input.document_id", value.get("document_id"))}
    chapters, blocks = value.get("chapters"), value.get("blocks")
    start, end = value.get("block_start"), value.get("block_end")
    block, sentence = value.get("block"), value.get("sentence")
    if block is not None or sentence is not None:  # one sentence
        if any(part is not None for part in (chapters, blocks, start, end)):
            raise _invalid("input.block", "send one of a sentence's block and sentence, a selection's block_start and "
                                          "block_end, blocks, or chapters")
        if not _index(block):
            raise _invalid("input.block", "input.block must be the sentence's block index, a whole number from 0")
        if not _index(sentence):
            raise _invalid("input.sentence", "input.sentence must be the sentence's index in its block, a whole number "
                                             "from 0, as the Reader counts the block's sentences")
        return {**fields, "block": block, "sentence": sentence, "range_form": "sentence"}
    if start is not None or end is not None:
        if chapters is not None or blocks is not None:
            raise _invalid("input.block_start", "send one of a selection's block_start and block_end, blocks, or chapters")
        if not _index(start):
            raise _invalid("input.block_start", "input.block_start must be a block index, a whole number from 0")
        if not _index(end):
            raise _invalid("input.block_end", "input.block_end must be a block index, a whole number from 0")
        if end < start:
            raise _invalid("input.block_end", "input.block_end must not come before block_start")
        return {**fields, "blocks": (start, end), "range_form": "selection"}
    if chapters is not None and blocks is not None:
        raise _invalid("input.blocks", "send chapters or blocks, not both")
    if chapters is not None:
        if not _ordered_pair(chapters):
            raise _invalid("input.chapters", "input.chapters must be [first, last]: chapter indexes from 0, first <= last")
        return {**fields, "chapters": (chapters[0], chapters[1]), "range_form": "chapters"}
    if blocks is not None:
        if not _ordered_pair(blocks):
            raise _invalid("input.blocks", "input.blocks must be [first, last]: block indexes from 0, first <= last")
        return {**fields, "blocks": (blocks[0], blocks[1]), "range_form": "blocks"}
    return fields


def _output(value: Any, settings) -> tuple[Speech | None, bool, str | None]:
    if value is None:
        return None, False, None
    if not isinstance(value, Mapping):
        raise _invalid("output", "output must be an object: {text?, speech?, document?}")
    for name in value:
        if name not in OUTPUT_FIELDS:
            raise _invalid(f"output.{name}", f"output.{name} is not an output; ask for {', '.join(OUTPUT_FIELDS)}")
    text = value.get("text", True)
    if not isinstance(text, bool):
        raise _invalid("output.text", "output.text must be true or false")
    speech = None if value.get("speech") is None else _speech(value["speech"], settings)
    document, title = value.get("document"), None
    if document is not None:
        if not isinstance(document, Mapping):
            raise _invalid("output.document", "output.document must be an object: {title?}")
        for name in document:
            if name not in DOCUMENT_FIELDS:
                raise _invalid(f"output.document.{name}", f"output.document.{name} is not a field; send title")
        title = _title(document.get("title"))
    if not text and speech is None and document is None:
        raise _invalid("output", "ask for at least one output: text, speech or document")
    return speech, document is not None, title


def _speech(value: Any, settings) -> Speech:
    if not isinstance(value, Mapping):
        raise _invalid("output.speech", "output.speech must be an object: {model?, voice?, format?, speed?}")
    for name in value:
        if name not in SPEECH_FIELDS:
            raise _invalid(f"output.speech.{name}", f"output.speech.{name} is not a field; send {', '.join(SPEECH_FIELDS)}")
    fmt = value.get("format")
    fmt = DEFAULT_FORMAT if fmt is None else fmt
    formats = sorted(tts_service.VALID_TTS_RESPONSE_FORMATS)
    if not isinstance(fmt, str) or fmt not in formats:
        raise _invalid("output.speech.format", f"output.speech.format must be one of {', '.join(formats)}")
    speed = value.get("speed")
    speed = 1.0 if speed is None else speed
    low, high = SPEED_RANGE
    if (isinstance(speed, bool) or not isinstance(speed, (int, float)) or not math.isfinite(speed)
            or not low <= speed <= high):
        raise _invalid("output.speech.speed", f"output.speech.speed must be a number from {low:g} to {high:g}")
    return Speech(
        model=_name("output.speech.model", value.get("model"), settings.tts_model),
        voice=_name("output.speech.voice", value.get("voice"), settings.tts_voice),
        format=fmt,
        speed=float(speed),
    )


def _name(param: str, value: Any, default: str) -> str:
    if value is None:
        return default
    if (not isinstance(value, str) or not value.strip() or len(value) > NAME_MAX_CHARS
            or presets.has_lone_surrogate(value)):
        raise _invalid(param, f"{param} must be a name of 1 to {NAME_MAX_CHARS} characters")
    return value.strip()


def _title(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > TITLE_MAX_CHARS or presets.has_lone_surrogate(value):
        raise _invalid("output.document.title", f"output.document.title must be text of up to {TITLE_MAX_CHARS} characters")
    return value.strip() or None


# ── Speech ───────────────────────────────────────────────────────────────────


class _SpeechFailed(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


# Silence between two spoken segments, as the Reader's SENTENCE_GAP_S: each segment is trimmed to its voice, so
# without it the sentences run together.
SEGMENT_GAP_S = 0.12


class Speaker:
    """Speaks text through the TTS router as ``/v1/realtime`` speaks a reply: cleaned and split by the Live Reader's
    segmenter, the pronunciation dictionary applied to each segment, then encoded once in the format asked. The
    segments are joined with a short pause, as the Reader joins its sentences."""

    def __init__(self, tts_router, *, settings, voice_library=None, pronunciation_dict=None) -> None:
        self.tts_router = tts_router
        self.settings = settings
        self.voice_library = voice_library
        self.pronunciation_dict = pronunciation_dict

    def check(self, speech: Speech) -> None:
        """Refuse, before the AI is asked, speech this server can't give: TTS off, an unknown model, a voice the
        model doesn't have, or a speed it can't change. Raises AskError."""
        if not getattr(self.settings, "tts_enabled", True):
            raise _invalid("output.speech", "TTS is off on this server (TTS_ENABLED=false), so the reply can't be spoken")
        try:
            voice, reference = resolve_named_voice(speech.model, speech.voice, None)
            problem = tts_service.validate_tts_feature_support(
                tts_router=self.tts_router, model_id=speech.model, speed=speech.speed,
                voice=None if reference else voice,
            )
        except KeyError as exc:  # a named voice that doesn't exist
            problem = str(exc.args[0]) if exc.args else "That named voice doesn't exist"
        except ValueError as exc:
            problem = str(exc)
        except ExternalProviderError as exc:
            raise AskError(exc.status_code, exc.code, exc.message, "output.speech") from None
        except NoTTSBackendsError as exc:
            raise AskError(503, "provider_missing", str(exc), "output.speech") from None
        if problem:
            raise _invalid("output.speech", problem)

    async def speak(self, text: str, speech: Speech) -> dict[str, Any]:
        """``{"audio": {format, content_type, base64, bytes}}``, or ``{"audio_error": {code, message}}`` when it
        can't be spoken (the reply is still answered)."""
        try:
            return {"audio": await self._speak(text, speech)}
        except _SpeechFailed as exc:
            return {"audio_error": {"code": exc.code, "message": exc.message}}

    async def _speak(self, text: str, speech: Speech) -> dict[str, Any]:
        try:
            get_limit = getattr(self.tts_router, "max_input_chars_for", None)
            # On a thread: for an isolated worker it reads the worker's manifest over HTTP once the cache is stale,
            # as it is after a slow AI reply.
            limit = await asyncio.to_thread(get_limit, speech.model) if callable(get_limit) else None
            limit = limit if type(limit) is int and limit > 0 else None
            most = self.settings.tts_max_input_length
            # On a thread, and only until the limit is passed: a long reply would hold the event loop for seconds
            # just to be refused.
            segments = await asyncio.to_thread(speakable_segments, text, limit, stop_after_chars=most)
            if not segments:
                raise _SpeechFailed("speech_empty", "The reply has nothing to speak")
            if sum(len(segment) for segment in segments) > most:
                raise _SpeechFailed(
                    "speech_too_long",
                    f"The reply has more than {most:,} characters to speak, the most TTS_MAX_INPUT_LENGTH speaks at "
                    f"once: save it as a document and play it in the Reader",
                )
            samples = await tts_service.run_on_provider(
                self.tts_router, speech.model, self._synthesize, segments, speech, limit,
            )
            data = await asyncio.to_thread(encode_audio, iter([samples]), fmt=speech.format, sample_rate=PCM_SAMPLE_RATE)
        except _SpeechFailed:
            raise
        except tts_service.TTSDisabledError:
            raise _SpeechFailed("tts_disabled", "TTS is off on this server (TTS_ENABLED=false)") from None
        except ExternalProviderError as exc:
            raise _SpeechFailed(exc.code or "provider_error", exc.message) from None
        except NoTTSBackendsError as exc:
            raise _SpeechFailed("provider_missing", str(exc)) from None
        except (KeyError, ValueError) as exc:  # this server's own check of the model, voice or input
            raise _SpeechFailed("speech_invalid", str(exc.args[0]) if exc.args else type(exc).__name__) from None
        except Exception as exc:
            logger.exception("Speaking an AI reply failed")
            raise _SpeechFailed(
                "speech_failed", f"The reply couldn't be spoken ({type(exc).__name__}); the server log has the details",
            ) from None
        return {
            "format": speech.format,
            "content_type": get_content_type(speech.format),
            "base64": base64.b64encode(data).decode("ascii"),
            "bytes": len(data),
        }

    def _synthesize(self, segments: list[str], speech: Speech, limit: int | None = None) -> np.ndarray:
        """On the provider's thread: each segment at 24 kHz, with SEGMENT_GAP_S of silence between two.

        A segment the dictionary made longer than the model's ``limit`` is cut again, as ``/v1/realtime`` cuts it.
        Its pieces get the same pause: each is trimmed to its voice like a sentence, and a cut falls at a clause or a
        word, where joining them bare would run the words together.
        """
        pronounce = getattr(self.pronunciation_dict, "apply", None)
        gap = np.zeros(int(SEGMENT_GAP_S * PCM_SAMPLE_RATE), dtype=np.float32)
        parts = []
        for segment in segments:
            spoken = pronounce(segment) if callable(pronounce) else segment
            for piece in speakable_segments(spoken, limit) if limit and len(spoken) > limit else [spoken]:
                if not piece.strip():
                    continue
                if parts:
                    parts.append(gap)  # each piece is trimmed to its voice (TTS_TRIM_SILENCE)
                parts.append(tts_service.synthesize_array(
                    text=piece, model=speech.model, voice=speech.voice, speed=speech.speed,
                    sample_rate=PCM_SAMPLE_RATE, tts_router=self.tts_router, settings=self.settings,
                    voice_library=self.voice_library,
                ))
        return np.concatenate(parts).astype(np.float32, copy=False) if parts else np.zeros(0, dtype=np.float32)


# ── The input, read on the server ────────────────────────────────────────────


@dataclass(frozen=True)
class _Source:
    """What is sent: a text, or a document's part as pieces; ``long`` when it takes more than one request."""

    kind: str
    chars: int
    long: bool
    text: str | None = None
    plan: Any = None
    pieces: tuple = ()
    chapters: tuple[int, int] | None = None
    blocks: tuple[int, int] | None = None
    param: str = "input.text"


def _read_source(request: AskRequest, chosen, runner, settings) -> _Source:
    """The input as it is sent (a thread: it reads studio.db). Raises AskError (404 or 422)."""
    if request.kind == "document":
        return _document_source(request, chosen, runner)
    if request.history_id is not None:
        text, param = _history_text(request.history_id, settings), "input.history_id"
    else:
        text, param = request.text, "input.text"
    return _Source(kind=request.kind, chars=len(text), long=len(text) > chosen.chunk_chars, text=text, param=param)


def _history_text(history_id: str, settings) -> str:
    from src import storage

    with storage.snapshot() as db:
        row = db.execute("SELECT full_text, text_preview FROM history_entries WHERE id = ?", (history_id,)).fetchone()
    if row is None:
        raise AskError(404, "history_not_found", "History entry not found", "input.history_id")
    text = row["full_text"] or row["text_preview"] or ""
    if not text.strip():
        raise _invalid("input.history_id", "That history entry has no text to send")
    return check_text(text, settings, "input.history_id")


def _range_param(request: AskRequest, count: int) -> str:
    """The field that named the document's part: for a selection, the end that is outside the document."""
    if request.range_form == "selection":
        return "input.block_start" if request.blocks[0] >= count else "input.block_end"
    return f"input.{request.range_form}" if request.range_form else "input.document_id"


def _document_source(request: AskRequest, chosen, runner) -> _Source:
    store = runner.store
    document = store.get_document(request.document_id, include_blocks=False)
    if document is None:
        raise AskError(404, "document_not_found", "Reader document not found", "input.document_id")
    if request.sentence is not None:
        return _sentence_source(request, chosen, store, document)
    chapters = blocks = None
    count = assistant_jobs.block_count(document)
    try:
        if request.chapters is not None:
            chapters = assistant_jobs.chapter_range(list(request.chapters), len(document["chapters"]))
        if request.blocks is not None:
            blocks = assistant_jobs.block_range(list(request.blocks), count)
    except (assistant_jobs.ChaptersError, assistant_jobs.BlocksError) as exc:
        raise _invalid(_range_param(request, count), str(exc)) from None
    plan = assistant_jobs.plan_document(store, request.document_id, chosen.to_dict(), chapters=chapters,
                                        blocks=blocks, document=document, question=request.question, named=True)
    if plan is None:
        raise AskError(404, "document_not_found", "Reader document not found", "input.document_id")
    pieces = tuple((chapter, piece) for chapter in plan.chapters for piece in chapter.pieces)
    if not pieces:
        raise _invalid(_range_param(request, count), "That part of the document has no text to send, only headings")
    return _Source(
        kind="document", chars=sum(len(piece.text) for _chapter, piece in pieces), long=len(pieces) > 1, plan=plan,
        pieces=pieces, chapters=chapters, blocks=blocks, param="input.document_id",
    )


def _sentence_source(request: AskRequest, chosen, store, document) -> _Source:
    """One sentence of a block: short, so it is always answered at once."""
    try:
        plan = assistant_jobs.plan_sentence(store, request.document_id, chosen.to_dict(), request.block,
                                            request.sentence, document=document, question=request.question)
    except assistant_jobs.BlocksError as exc:
        raise _invalid("input.block", str(exc)) from None
    except assistant_jobs.SentenceError as exc:
        raise _invalid("input.sentence", str(exc)) from None
    if plan is None:
        raise AskError(404, "document_not_found", "Reader document not found", "input.document_id")
    chapter = plan.chapters[0]
    piece = chapter.pieces[0]
    return _Source(kind="document", chars=len(piece.text), long=False, plan=plan, pieces=((chapter, piece),),
                   param="input.document_id")


def _keep_text(store, text: str, kind: str, param: str) -> str:
    """A long text or transcript kept as a Reader document (as pasting it into the Reader does), for its job to
    point at and to read in the Reader; the job sends the text itself, not this document's blocks.
    The same text again is the same document, so a job started again sends only the pieces it hadn't finished."""
    from src.reader.sources import parse_pasted

    try:
        document = parse_pasted(text, title=TRANSCRIPT_TITLE if kind == "transcript" else None)
    except ValueError:
        raise _invalid(param, "The text has nothing the Reader can read") from None
    document_id, _created = store.save_document(document)
    return document_id


def _snippet(text: str) -> str:
    words = " ".join(text.split())
    return words if len(words) <= SNIPPET_CHARS else words[:SNIPPET_CHARS - 1].rstrip() + "…"


def _save_reply(store, request: AskRequest, source: _Source, chosen, text: str) -> dict[str, Any]:
    """The reply saved as a Reader document, as a job's result is made."""
    if source.kind == "document":
        chapter, _piece = source.pieces[0]
        name = f"{source.plan.title}: {request.title or chosen.name}"[:TITLE_MAX_CHARS]
        chapters = [(chapter.title, chapter.headings, [text])]
        author = source.plan.author
    else:
        name = (request.title or f"{chosen.name}: {_snippet(source.text)}")[:TITLE_MAX_CHARS]
        chapters = [(name, (), [text])]
        author = None
    try:
        document = assistant_jobs.answers_document(name, author, chapters)
    except ValueError:
        return {"document_error": {"code": ASSISTANT_EMPTY_REPLY, "message": "The reply has no text to read"}}
    document_id, _created = store.save_document(document)
    return {"document_id": document_id}


def _elapsed_ms(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))


# ── The operation ────────────────────────────────────────────────────────────


async def handle(payload: Any, *, settings, speaker: Speaker | None) -> tuple[int, dict[str, Any]]:
    """Answer one ``POST /api/assistant/ask`` body: ``(status, content)``. Raises AskError for a request refused
    before the AI is asked; an AI error is an answer (502, or 504 for a timeout)."""
    request = parse(payload, settings)
    try:
        chosen = await asyncio.to_thread(presets.effective, request.preset_id, request.overrides)
    except presets.PresetError as exc:
        raise _invalid(exc.field, exc.message) from None
    runner = assistant_jobs.get_runner() if settings.reader_enabled else None
    if runner is None and request.kind == "document":
        raise _invalid("input.kind", "The Reader is off (READER_ENABLED=false), so there's no document to send")
    if runner is None and request.document:
        raise _invalid("output.document", "The Reader is off (READER_ENABLED=false), so the reply can't be saved")
    if request.speech is not None:
        if speaker is None:
            raise AskError(503, "speech_unavailable", "This server's assistant routes were built without a TTS router")
        await asyncio.to_thread(speaker.check, request.speech)
    client = assistant.get_client()
    if client is None:
        raise AskError(503, "assistant_not_started", "The AI assistant client hasn't started")
    source = await asyncio.to_thread(_read_source, request, chosen, runner, settings)
    summary: dict[str, Any] = {"kind": request.kind, "chars": source.chars}
    if request.question is not None:  # counted, never shown
        summary["question_chars"] = len(request.question)
    if request.sentence is not None:
        summary.update(block=request.block, sentence=request.sentence)
    if source.long:
        if runner is None:
            raise _invalid(source.param, f"This is more than one piece of {chosen.chunk_chars:,} characters, so it "
                                         "goes as a job, and jobs need the Reader (READER_ENABLED=false)")
        job = await _start_job(request, source, runner)
        return 202, {"job": job, "preset": chosen.preset_id, "input": summary}

    if source.kind == "document":
        chapter, piece = source.pieces[0]
        messages = prompts.messages_for(chosen, [], piece.text, question=request.question, source=chapter.source,
                                        title=source.plan.title, author=source.plan.author or "", chapter=chapter.title)
    else:
        messages = prompts.messages_for(chosen, [], source.text, question=request.question)
    started = time.perf_counter()
    try:
        reply = await client.reply(messages, chosen)
    except AssistantError as exc:
        content: dict[str, Any] = {"error": {"message": exc.message, "code": exc.code}, "elapsed_ms": _elapsed_ms(started)}
        if request.speech is not None and chosen.speak_errors:
            content["sentence"] = chosen.error_sentence(exc.code)
            content.update(await speaker.speak(content["sentence"], request.speech))
        return (504 if exc.code == ASSISTANT_TIMEOUT else 502), content
    content = {
        "reply": reply.text,
        "thinking": reply.thinking if chosen.show_thinking else None,
        "finish_reason": reply.finish_reason,
        "elapsed_ms": _elapsed_ms(started),
        "preset": chosen.preset_id,
        "input": summary,
    }
    if request.speech is not None:
        content.update(await speaker.speak(reply.text, request.speech))
    if request.document:
        content.update(await asyncio.to_thread(_save_reply, runner.store, request, source, chosen, reply.text))
    return 200, content


async def _start_job(request: AskRequest, source: _Source, runner) -> dict[str, Any]:
    """Queue a long input on the book jobs' runner, with the job timeout and the preset's own prompt."""
    try:
        settings = await asyncio.to_thread(presets.effective, request.preset_id, request.overrides, job=True)
    except presets.PresetError as exc:
        raise _invalid(exc.field, exc.message) from None
    if source.kind == "document":
        document_id = request.document_id
    else:
        document_id = await asyncio.to_thread(_keep_text, runner.store, source.text, request.kind, source.param)
    try:
        job = await runner.start(document_id, settings, chapters=None if source.chapters is None else list(source.chapters),
                                 blocks=None if source.blocks is None else list(source.blocks), title=request.title,
                                 text=None if source.kind == "document" else source.text, question=request.question,
                                 named=source.kind == "document")
    except (assistant_jobs.ChaptersError, assistant_jobs.BlocksError) as exc:
        raise _invalid(source.param, str(exc)) from None
    if job is None:  # deleted meanwhile
        raise AskError(404, "document_not_found", "Reader document not found", "input.document_id")
    return job
