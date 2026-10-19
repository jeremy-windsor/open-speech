"""One background worker that renders Reader documents into short audio files.

There is a single job at a time: the document, voice and window of parts the
listener needs next. A new request replaces the job, which is how skips cancel
work that is no longer needed. Parts are rendered sentence by sentence at 1.0x
(the browser applies playback speed), so a skip or a Live Reader session can
interrupt between sentences.
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

from src.reader.document import HEADING, is_speakable, spoken_opening
from src.reader.media import MediaSigner
from src.reader.store import ReaderStore

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


def part_id(document_id: str, first_block: int, last_block: int, model: str, voice: str) -> str:
    key = f"{document_id}\n{first_block}\n{last_block}\n{model}\n{voice}\n{RENDER_VERSION}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


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
    ) -> None:
        self.store = store
        self.synthesize_fn = synthesize_fn
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
        self._errors: dict[str, tuple[str, str]] = {}  # part id -> (document id, message)
        self._outlines: dict[str, list[tuple[int, int, int, str]]] = {}
        self._thread: threading.Thread | None = None

    # ── Planning ────────────────────────────────────────────────────────────

    def plan(self, document_id: str, from_block: int, model: str, voice: str) -> list[PartPlan] | None:
        """Plan the next parts from ``from_block``, reusing parts already rendered or queued."""
        outline = self._outline(document_id)
        if outline is None:
            return None
        if not 0 <= from_block < len(outline):
            raise ValueError(f"from_block must be between 0 and {len(outline) - 1}")
        ready = self.store.ready_parts(document_id, model, voice)
        for first, row in list(ready.items()):
            if row["id"] != part_id(document_id, first, row["last_block"], model, voice):
                # Rendered by an older version of the renderer: speak it again.
                self.store.forget_part(ready.pop(first)["id"])
                _remove(Path(row["path"]))
        with self._cond:
            job = self._job
            queued = (
                {part.first_block: part for part in job.parts}
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
                part = PartPlan(part_id(document_id, cursor, last, model, voice), cursor, last)
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
            for part in parts:
                self._errors.pop(part.id, None)
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
        if outline is None:
            outline = self.store.outline(document_id)
            if outline is not None:
                with self._cond:
                    if len(self._outlines) >= OUTLINE_CACHE_SIZE:
                        self._outlines.pop(next(iter(self._outlines)))
                    self._outlines[document_id] = outline
        return outline

    def forget_document(self, document_id: str) -> None:
        with self._cond:
            if self._job and self._job.document_id == document_id:
                self._job = None
            self._outlines.pop(document_id, None)
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
        except OSError as exc:
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
            except Exception as exc:  # noqa: BLE001 - a locked or unreadable database must not end the worker
                logger.warning("Reader worker could not read its queue: %s", exc)
                self._stop.wait(BUSY_POLL_S * 4)
                continue
            try:
                self._render(job, part)
            except Exception as exc:  # noqa: BLE001 - surface every render failure to the player
                logger.warning("Reader part %s failed: %s", part.id, exc)
                with self._cond:
                    self._errors[part.id] = (job.document_id, str(exc) or exc.__class__.__name__)
            finally:
                with self._cond:
                    self._rendering = None

    def _wanted(self, part: PartPlan) -> bool:
        with self._cond:
            job = self._job
            return not self._stop.is_set() and job is not None and any(p.id == part.id for p in job.parts)

    def _render(self, job: _Job, part: PartPlan) -> bool:
        blocks = self.store.get_blocks(job.document_id, part.first_block, part.last_block)
        if not blocks:
            self._drop_job(job.document_id)
            raise ValueError("Document no longer exists")
        pieces: list[np.ndarray] = []
        sentences: list[dict] = []
        samples = 0
        for block in blocks:
            spans = block["sentences"] or [(0, len(block["text"]))]
            for index, (start, end) in enumerate(spans):
                while self.busy_fn() and self._wanted(part):
                    self._stop.wait(BUSY_POLL_S)
                if not self._wanted(part):
                    return False
                text = block["text"][start:end]
                sentences.append({"block": block["index"], "index": index, "start_ms": round(samples * 1000 / SAMPLE_RATE)})
                if is_speakable(text):
                    spoken = spoken_opening(text) if index == 0 and block["kind"] != HEADING else text
                    audio = np.asarray(
                        self.synthesize_fn(
                            text=self.pronounce_fn(spoken),
                            model=job.model,
                            voice=job.voice,
                            speed=1.0,
                            sample_rate=SAMPLE_RATE,
                        ),
                        dtype=np.float32,
                    )
                    pieces.append(audio)
                    samples += len(audio)
                last = index == len(spans) - 1
                gap_s = SENTENCE_GAP_S
                if last:
                    gap_s = HEADING_GAP_S if block["kind"] == "heading" else PARAGRAPH_GAP_S
                gap = np.zeros(int(gap_s * SAMPLE_RATE), dtype=np.float32)
                pieces.append(gap)
                samples += len(gap)

        audio = np.concatenate(pieces) if pieces else np.zeros(int(PARAGRAPH_GAP_S * SAMPLE_RATE), dtype=np.float32)
        data = self.encode_fn(audio, SAMPLE_RATE)
        if not self._wanted(part):
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
