"""One background worker that renders Reader documents into short audio files.

There is a single job at a time: the document, voice and window of parts the
listener needs next. A new request replaces the job, which is how skips cancel
work that is no longer needed. Parts are rendered sentence by sentence at 1.0x
(the browser applies playback speed), so a skip or a Live Reader session can
interrupt between sentences. One exception: the first part of a job, which its
listener waits on before any audio, is finished and saved when a request for
another document, model or voice replaces the job, so two listeners waiting at
once each get theirs instead of taking the job from each other forever.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from src.live_tts.segmenter import split_sentences
from src.reader.document import HEADING, is_speakable, spoken_opening
from src.reader.media import MediaSigner
from src.reader.store import ReaderStore
from src.voice_identities import VOICE_PREFIX, resolve_named_voice
from src.voice_library import VoiceNotFoundError

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
SENTENCE_GAP_S = 0.12
PARAGRAPH_GAP_S = 0.45
HEADING_GAP_S = 0.8
BUSY_POLL_S = 0.25
OUTLINE_CACHE_SIZE = 16
# Part ids include this, so a change to what gets spoken renders cached parts again.
RENDER_VERSION = 2
_HEX = frozenset("0123456789abcdef")
# The typed refusal of a text in a language the model doesn't speak (ExternalProviderError.code, from a worker).
LANGUAGE_UNSUPPORTED = "language_unsupported"


@dataclass(frozen=True, slots=True)
class PartPlan:
    id: str
    first_block: int
    last_block: int


@dataclass(slots=True)
class _Job:
    document_id: str
    model: str
    voice: str
    parts: list[PartPlan]


def part_id(
    document_id: str, first_block: int, last_block: int, model: str, voice: str, voice_key: str = "", speech_key: str = ""
) -> str:
    """``voice_key`` is what a named voice speaks with (named_voice_key); a provider voice has none.
    ``speech_key`` is what else shapes the audio: the pronunciation dictionary's fingerprint, which
    turns the text into what is spoken, and post_processing_key; without one, a part keeps the id it had."""
    key = f"{document_id}\n{first_block}\n{last_block}\n{model}\n{voice}\n{RENDER_VERSION}"
    if voice_key:
        key += f"\n{voice_key}"
    if speech_key:
        key += f"\nspeech {speech_key}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def post_processing_key(trim: bool, normalize: bool) -> str:
    """The part of ``speech_key`` for TTS_TRIM_SILENCE and TTS_NORMALIZE_OUTPUT, which the synthesis
    applies to each sentence: "" when both are on (the defaults), so those parts keep their ids;
    otherwise the ones that are off, so parts cached with other processing render again."""
    off = [name for name, on in (("trim", trim), ("normalize", normalize)) if not on]
    return f"\npost-processing off: {','.join(off)}" if off else ""


class ReaderVoiceError(Exception):
    """The voice cannot be spoken with the model; ``status_code`` is the answer /v1/audio/speech gives."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def error_message(exc: BaseException) -> str:
    """What the player shows after "Couldn't play:". str() of a KeyError carries Python's quotes."""
    if isinstance(exc, VoiceNotFoundError) and exc.args:
        return f"Voice library entry '{exc.args[0]}' not found"
    if isinstance(exc, KeyError) and exc.args and isinstance(exc.args[0], str):
        return exc.args[0]
    return str(exc) or exc.__class__.__name__


def named_voice_key(model: str, voice: str, voice_library) -> str:
    """What a named voice speaks with for model: its provider voice, recording and transcript.

    Parts are keyed on it, so they render again once the named voice speaks otherwise (a new
    version, a replaced recording). A provider voice gives "", and keeps the part ids it had.
    Raises ReaderVoiceError where synthesis would fail: a named voice or recording that is gone,
    or no version for the model.
    """
    if not voice.startswith(VOICE_PREFIX):
        return ""
    try:
        provider_voice, reference = resolve_named_voice(model, voice)
        recording = voice_library.metadata(reference) if reference else {}
    except KeyError as exc:
        raise ReaderVoiceError(404, error_message(exc)) from exc
    except ValueError as exc:
        raise ReaderVoiceError(400, str(exc)) from exc
    return json.dumps([provider_voice, reference, recording.get("sha256"), recording.get("transcript")])


class ReaderRenderer:
    def __init__(
        self,
        *,
        store: ReaderStore,
        synthesize_fn: Callable[..., np.ndarray],
        encode_fn: Callable[[np.ndarray, int], bytes],
        pronounce_fn: Callable[[str], str],
        busy_fn: Callable[[], bool],
        signer: MediaSigner,
        cache_dir: str,
        lookahead_parts: int = 3,
        part_max_chars: int = 450,
        cache_max_bytes: int = 2000 * 1024 * 1024,
        media_url_ttl_s: int = 21600,
        max_input_chars_fn: Callable[[str], int | None] | None = None,
        voice_key_fn: Callable[[str, str], str] | None = None,
        speech_key_fn: Callable[[], str] | None = None,
    ) -> None:
        self.store = store
        self.synthesize_fn = synthesize_fn
        # (model, voice) -> what that voice speaks with, part of each part id (named_voice_key).
        self.voice_key_fn = voice_key_fn
        # () -> what pronounce_fn and synthesize_fn's post-processing do (PronunciationDictionary.fingerprint
        # plus post_processing_key), part of each part id.
        self.speech_key_fn = speech_key_fn
        self.max_input_chars_fn = max_input_chars_fn
        self.encode_fn = encode_fn
        self.pronounce_fn = pronounce_fn
        self.busy_fn = busy_fn
        self.signer = signer
        self.cache_dir = Path(cache_dir)
        self.lookahead_parts = max(1, lookahead_parts)
        self.part_max_chars = max(50, part_max_chars)
        self.cache_max_bytes = cache_max_bytes
        self.media_url_ttl_s = media_url_ttl_s
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._job: _Job | None = None
        self._rendering: str | None = None
        # The job of the part being rendered; forget_document clears it, so a replaced job's
        # first part, which _wanted lets the worker finish, still stops once its document is deleted.
        self._rendering_job: _Job | None = None
        self._errors: dict[str, tuple[str, str]] = {}  # part id -> (document id, message)
        self._outlines: dict[str, list[tuple[int, int, int, str]]] = {}
        self._forgets = 0   # counts forget_document calls, so an outline read before one is not cached after it
        self._thread: threading.Thread | None = None

    # ── Planning ────────────────────────────────────────────────────────────

    def plan(self, document_id: str, from_block: int, model: str, voice: str) -> list[PartPlan] | None:
        """Plan the next parts from ``from_block``, reusing parts already rendered or queued."""
        outline = self._outline(document_id)
        if outline is None:
            return None
        if not 0 <= from_block < len(outline):
            raise ValueError(f"from_block must be between 0 and {len(outline) - 1}")
        voice_key = self.voice_key_fn(model, voice) if self.voice_key_fn else ""
        speech_key = self.speech_key_fn() if self.speech_key_fn else ""

        def current_id(first: int, last: int) -> str:
            return part_id(document_id, first, last, model, voice, voice_key, speech_key)

        ready = self.store.ready_parts(document_id, model, voice)
        for first, row in list(ready.items()):
            if row["id"] != current_id(first, row["last_block"]):
                # Rendered by an older version of the renderer, before the named voice spoke
                # otherwise, or with another pronunciation dictionary or post-processing: speak it again.
                self.store.forget_part(ready.pop(first)["id"])
                _remove(Path(row["path"]))
        with self._cond:
            job = self._job
            queued = (
                {
                    part.first_block: part
                    for part in job.parts
                    if part.id == current_id(part.first_block, part.last_block)
                }
                if job and (job.document_id, job.model, job.voice) == (document_id, model, voice)
                else {}
            )
        parts: list[PartPlan] = []
        cursor = from_block
        while len(parts) < self.lookahead_parts and cursor < len(outline):
            if cursor in ready and not Path(ready[cursor]["path"]).is_file():
                # The file was removed behind our back (cache folder emptied): render it again.
                self.store.forget_part(ready.pop(cursor)["id"])
            if cursor in ready:
                row = ready[cursor]
                part = PartPlan(row["id"], row["first_block"], row["last_block"])
            elif cursor in queued:
                part = queued[cursor]
            else:
                chapter = outline[cursor][1]

                def joins(index: int) -> bool:
                    return index < len(outline) and outline[index][1] == chapter and index not in ready

                last = cursor
                chars = outline[cursor][2]
                # A heading never stands alone: a second of audio leaves a locked
                # iPhone waiting on the next part, and Safari may stop the page there.
                while outline[last][3] == HEADING and joins(last + 1):
                    last += 1
                    chars += outline[last][2]
                # The first part after a start or skip stays short, so audio starts quickly.
                if parts:
                    while joins(last + 1) and chars + outline[last + 1][2] <= self.part_max_chars:
                        last += 1
                        chars += outline[last][2]
                part = PartPlan(current_id(cursor, last), cursor, last)
            parts.append(part)
            cursor = part.last_block + 1
        return parts

    def _describe(self, parts: list[PartPlan]) -> list[dict]:
        described = []
        for part in parts:
            row = self.store.get_part(part.id)
            with self._cond:
                error = self._errors.get(part.id, (None, None))[1]
                rendering = self._rendering == part.id
            if row:
                status = "ready"
            elif error:
                status = "error"
            elif rendering:
                status = "rendering"
            else:
                status = "pending"
            described.append({
                "id": part.id,
                "first_block": part.first_block,
                "last_block": part.last_block,
                "status": status,
                "duration_ms": row["duration_ms"] if row else None,
                "url": self.signer.url(part.id, self.media_url_ttl_s) if row else None,
                "error": error if status == "error" else None,
                "sentences": json.loads(row["sentences_json"]) if row else [],
            })
        return described

    def request(self, document_id: str, from_block: int, model: str, voice: str) -> list[dict] | None:
        """Point the render job at ``from_block`` and return the planned parts."""
        parts = self.plan(document_id, from_block, model, voice)
        if parts is None:
            return None
        with self._cond:
            # The new job's parts are tried again, and errors are kept for the current job only:
            # one per failed window or voice would pile up for as long as the server runs.
            self._errors.clear()
            self._job = _Job(document_id, model, voice, parts)
            self._ensure_thread()
            self._cond.notify_all()
        return self._describe(parts)

    def status(self, document_id: str, from_block: int, model: str, voice: str) -> list[dict] | None:
        parts = self.plan(document_id, from_block, model, voice)
        return None if parts is None else self._describe(parts)

    def _outline(self, document_id: str) -> list[tuple[int, int, int, str]] | None:
        # Documents never change after import, and the player polls every half second.
        with self._cond:
            outline = self._outlines.get(document_id)
            forgets = self._forgets
        if outline is None:
            outline = self.store.outline(document_id)
            if outline is not None:
                with self._cond:
                    if forgets != self._forgets:
                        return outline   # the document may have been deleted since it was read
                    if len(self._outlines) >= OUTLINE_CACHE_SIZE:
                        self._outlines.pop(next(iter(self._outlines)))
                    self._outlines[document_id] = outline
        return outline

    def forget_document(self, document_id: str) -> None:
        with self._cond:
            if self._job and self._job.document_id == document_id:
                self._job = None
            if self._rendering_job and self._rendering_job.document_id == document_id:
                self._rendering_job = None
            self._outlines.pop(document_id, None)
            self._forgets += 1
            self._errors = {part: error for part, error in self._errors.items() if error[0] != document_id}

    # ── Worker ──────────────────────────────────────────────────────────────

    def _ensure_thread(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="reader-renderer", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        with self._cond:
            self._job = None
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)

    def _next_part(self) -> tuple[_Job, PartPlan] | None:
        job = self._job
        if job is None:
            return None
        for part in job.parts:
            if part.id not in self._errors and self.store.get_part(part.id) is None:
                return job, part
        return None

    def _sweep_cache(self) -> None:
        """Reconcile the cache folder with the parts table after a restart or crash.

        Only files the renderer itself writes (``<document id>/<part id>.mp3`` and
        their ``.tmp`` siblings) are touched, in case the cache folder was pointed
        at a folder that holds other data.
        """
        known = self.store.part_paths()
        for part, path in known.items():
            if not Path(path).is_file():
                self.store.forget_part(part)
        keep = {str(Path(path)) for path in known.values()}
        if not self.cache_dir.is_dir():
            return
        for folder in self.cache_dir.iterdir():
            if not folder.is_dir() or folder.is_symlink() or not _is_hex(folder.name, 32):
                continue
            for path in folder.iterdir():
                if path.is_file() and _is_part_file(path.name) and str(path) not in keep:
                    _remove(path)
            if not any(folder.iterdir()):
                folder.rmdir()

    def _run(self) -> None:
        try:
            self._sweep_cache()
        except Exception as exc:  # noqa: BLE001 - the sweep is best effort; a locked database must not end the worker
            logger.warning("Reader cache sweep failed: %s", exc)
        while not self._stop.is_set():
            try:
                with self._cond:
                    work = self._next_part()
                    while work is None and not self._stop.is_set():
                        self._cond.wait(timeout=30)
                        work = self._next_part()
                    if work is None:
                        return
                    job, part = work
                    self._rendering = part.id
                    self._rendering_job = job
            except Exception as exc:  # noqa: BLE001 - a locked or unreadable database must not end the worker
                logger.warning("Reader worker could not read its queue: %s", exc)
                self._stop.wait(BUSY_POLL_S * 4)
                continue
            try:
                self._render(job, part)
            except Exception as exc:  # noqa: BLE001 - surface every render failure to the player
                message = error_message(exc)
                logger.warning("Reader part %s failed: %s", part.id, message)
                with self._cond:
                    if self._job is not None and part in self._job.parts:   # nobody asks after another job's
                        self._errors[part.id] = (job.document_id, message)
            finally:
                with self._cond:
                    self._rendering = None
                    self._rendering_job = None

    def _wanted(self, job: _Job, part: PartPlan) -> bool:
        """Whether the worker goes on with ``part`` of ``job`` at a sentence boundary.

        A part the current job plans is wanted. A skip (a request for the same document, model and
        voice that no longer plans it) drops it, and so does any request when it is not its job's
        first part. The first part is finished when a request for another document, model or voice
        replaced the job: two listeners waiting at once for first parts longer than the player's
        re-request wait would otherwise take the job from each other and neither part would finish.
        """
        with self._cond:
            if self._stop.is_set() or self._rendering_job is not job:
                return False   # stopping, or the document was deleted (forget_document)
            current = self._job
            if current is not None and any(p.id == part.id for p in current.parts):
                return True
            return part.id == job.parts[0].id and (
                current is None
                or (current.document_id, current.model, current.voice) != (job.document_id, job.model, job.voice)
            )

    def _render(self, job: _Job, part: PartPlan) -> bool:
        blocks = self.store.get_blocks(job.document_id, part.first_block, part.last_block)
        if not blocks:
            self._drop_job(job.document_id)
            raise ValueError("Document no longer exists")
        # Sentences are split at import for every model; some models take less (Chatterbox 350 characters).
        limit = self.max_input_chars_fn(job.model) if self.max_input_chars_fn else None
        pieces: list[np.ndarray] = []
        sentences: list[dict] = []
        samples = 0
        spoke = False
        refusal: Exception | None = None
        for block in blocks:
            spans = block["sentences"] or [(0, len(block["text"]))]
            for index, (start, end) in enumerate(spans):
                while self.busy_fn() and self._wanted(job, part):
                    self._stop.wait(BUSY_POLL_S)
                if not self._wanted(job, part):
                    return False
                text = block["text"][start:end]
                sentences.append({"block": block["index"], "index": index, "start_ms": round(samples * 1000 / SAMPLE_RATE)})
                # What is spoken is what the dictionary leaves: a sentence it removes or leaves as punctuation has
                # nothing to say and is skipped like "* * *", keeping its mark and gap. The GPU workers refuse an empty
                # text.
                spoken = self.pronounce_fn(spoken_opening(text) if index == 0 and block["kind"] != HEADING else text)
                if is_speakable(spoken):
                    chunks = split_sentences(spoken, limit) if limit and len(spoken) > limit else [(0, len(spoken))]
                    sentence_audio: list[np.ndarray] = []
                    try:
                        for chunk_start, chunk_end in chunks:
                            chunk = spoken[chunk_start:chunk_end]
                            if not is_speakable(chunk):
                                continue   # a piece cut for the model's limit can be a rule or a run of punctuation
                            if sentence_audio:
                                # A cut falls at a clause or a word, and each piece is trimmed to its sound: joined
                                # bare, the words either side of it would run together.
                                sentence_audio.append(np.zeros(int(SENTENCE_GAP_S * SAMPLE_RATE), dtype=np.float32))
                            sentence_audio.append(np.asarray(
                                self.synthesize_fn(
                                    text=chunk,
                                    model=job.model,
                                    voice=job.voice,
                                    speed=1.0,
                                    sample_rate=SAMPLE_RATE,
                                ),
                                dtype=np.float32,
                            ))
                    except Exception as exc:
                        # A sentence the voice refuses for its language (language_unsupported) is left out whole, its
                        # mark kept with no audio, and the part goes on. Any other error fails the part.
                        if getattr(exc, "code", None) != LANGUAGE_UNSUPPORTED:
                            raise
                        refusal = exc
                    else:
                        spoke = spoke or bool(sentence_audio)
                        pieces.extend(sentence_audio)
                        samples += sum(len(audio) for audio in sentence_audio)
                last = index == len(spans) - 1
                gap_s = SENTENCE_GAP_S
                if last:
                    gap_s = HEADING_GAP_S if block["kind"] == "heading" else PARAGRAPH_GAP_S
                gap = np.zeros(int(gap_s * SAMPLE_RATE), dtype=np.float32)
                pieces.append(gap)
                samples += len(gap)

        if refusal is not None and not spoke:
            # Every sentence with something to say was refused: a silent part would let a book in a language the
            # voice can't speak play on as silence, so the part fails with the reason instead.
            raise refusal
        audio = np.concatenate(pieces) if pieces else np.zeros(int(PARAGRAPH_GAP_S * SAMPLE_RATE), dtype=np.float32)
        data = self.encode_fn(audio, SAMPLE_RATE)
        if not self._wanted(job, part):
            return False
        folder = self.cache_dir / job.document_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{part.id}.mp3"
        temporary = folder / f".{part.id}.tmp"
        temporary.write_bytes(data)
        os.replace(temporary, path)
        saved = self.store.save_part(
            part_id=part.id,
            document_id=job.document_id,
            first_block=part.first_block,
            last_block=part.last_block,
            model=job.model,
            voice=job.voice,
            path=str(path),
            size=len(data),
            duration_ms=round(len(audio) * 1000 / SAMPLE_RATE),
            sentences=sentences,
        )
        if not saved:
            _remove(path)
            self._drop_job(job.document_id)
            return False
        with self._cond:
            keep = {p.id for p in self._job.parts} if self._job else set()
        for stale in self.store.prune_parts(self.cache_max_bytes, keep):
            _remove(Path(stale))
        return True

    def _drop_job(self, document_id: str) -> None:
        """Stop working on a document that was deleted while it was rendering."""
        with self._cond:
            if self._job and self._job.document_id == document_id:
                self._job = None


def _is_hex(name: str, length: int) -> bool:
    return len(name) == length and set(name) <= _HEX


def _is_part_file(name: str) -> bool:
    if name.endswith(".mp3"):
        return _is_hex(name[:-4], 24)
    return name.startswith(".") and name.endswith(".tmp") and _is_hex(name[1:-4], 24)


def _remove(path: Path) -> None:
    # A Windows bind mount can refuse to delete a file that is still being streamed.
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not remove Reader audio %s: %s", path, exc)
