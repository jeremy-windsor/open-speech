"""Async batch worker for processing multi-file transcription jobs."""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

from src.batch.store import BatchJobStore
from src.config import settings
from src.formatters import format_transcription
from src.services.stt import prepare_audio, suffix_from_filename

logger = logging.getLogger("open-speech.batch")
INTERRUPTED_JOB_ERROR = "Server restarted before batch processing completed"


def recover_interrupted_jobs(store: BatchJobStore, batch_size: int = 200) -> int:
    """Fail queued/running jobs left by a previous run and delete their spooled audio.

    Runs at startup, before any job is submitted, so every spooled upload is an orphan.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    recovered = 0
    for status in ("queued", "running"):
        while jobs := store.list_jobs(limit=batch_size, status=status):
            recovered_in_batch = 0
            for job in jobs:
                logger.warning(
                    "Recovering interrupted batch job %s (was %s)",
                    job.job_id,
                    status,
                )
                if store.update(
                    job.job_id,
                    status="failed",
                    finished_at=time.time(),
                    error=INTERRUPTED_JOB_ERROR,
                ):
                    recovered += 1
                    recovered_in_batch += 1
            if recovered_in_batch == 0:
                logger.error(
                    "Stopping interrupted-job recovery because no %s jobs could be updated",
                    status,
                )
                break
    shutil.rmtree(store.audio_root, ignore_errors=True)
    return recovered


async def _run_in_thread(func, /, *args, **kwargs):
    """Run func in the default executor; if cancelled, wait for the thread to finish.

    A running thread cannot be stopped. Keeping the job's worker slot (and the batch
    inference gate) until it is idle keeps OS_BATCH_WORKERS and OS_BATCH_MAX_PENDING
    counting work that is still running, so a queued job cannot start alongside it.
    """
    future = asyncio.get_running_loop().run_in_executor(None, functools.partial(func, *args, **kwargs))
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        while not future.done():
            try:
                await asyncio.wait({future})
            except asyncio.CancelledError:
                pass  # cancelled again (shutdown after DELETE); the thread still runs
        raise


class BatchWorker:
    """Processes batch transcription jobs asynchronously."""

    def __init__(
        self,
        store: BatchJobStore,
        router: Any,
        max_concurrent: int | None = None,
    ) -> None:
        self._store = store
        self._router = router
        if max_concurrent is None:
            max_concurrent = int(os.environ.get("OS_BATCH_WORKERS", "2"))
        self._semaphore = asyncio.Semaphore(max_concurrent)
        # A batch file holds the backend's inference lock for its whole run. One batch
        # inference at a time leaves live STT (streams, realtime, Wyoming) at most one
        # file to wait for, whatever OS_BATCH_WORKERS allows for decoding.
        self._inference_gate = asyncio.Lock()
        self._tasks: dict[str, asyncio.Task] = {}

    async def submit(
        self,
        job_id: str,
        audio_files: list[tuple[str, bytes | Path]],
        options: dict[str, Any],
    ) -> None:
        """Queue a job for background processing. Returns immediately.

        Each file is (filename, audio): the bytes, or the path of an upload spooled
        under the store's audio_dir(job_id), which is deleted once the job ends.
        """
        task = asyncio.create_task(self._run_with_semaphore(job_id, audio_files, options))
        self._tasks[job_id] = task
        task.add_done_callback(lambda t: self._job_ended(job_id))

    def _job_ended(self, job_id: str) -> None:
        self._tasks.pop(job_id, None)
        # A done callback also covers a job cancelled before its coroutine started.
        shutil.rmtree(self._store.audio_dir(job_id), ignore_errors=True)

    async def _run_with_semaphore(
        self,
        job_id: str,
        audio_files: list[tuple[str, bytes | Path]],
        options: dict[str, Any],
    ) -> None:
        async with self._semaphore:
            await self._process_job(job_id, audio_files, options)

    async def _process_job(
        self,
        job_id: str,
        audio_files: list[tuple[str, bytes | Path]],
        options: dict[str, Any],
    ) -> None:
        """Process each file sequentially, store results, update status."""
        # Each write re-serializes every result so far, so it runs off the loop.
        update = functools.partial(asyncio.to_thread, self._store.update, job_id)
        try:
            await update(status="running", started_at=time.time())
            model = options.get("model", "")
            results: list[dict[str, Any]] = []

            for filename, audio in audio_files:
                result = await self._transcribe_file(filename, audio, model, options)
                results.append(result)
                # Update results incrementally
                await update(results=results)

            await update(
                status="done",
                finished_at=time.time(),
                results=results,
            )
            logger.info("Batch job %s completed: %d files", job_id, len(results))

            # History logging removed — batch is API, not UI.
            # Only web UI requests with X-History header get logged.

        except asyncio.CancelledError:
            # asyncio.CancelledError is BaseException — catch separately so the job
            # is marked failed in the DB rather than left as a zombie "running" forever.
            logger.info("Batch job %s cancelled", job_id)
            await update(
                status="failed",
                finished_at=time.time(),
                error="Canceled",
            )
            raise  # re-raise so asyncio knows the task is cancelled

        except Exception as e:
            logger.exception("Batch job %s failed", job_id)
            await update(
                status="failed",
                finished_at=time.time(),
                error=str(e),
            )

    @staticmethod
    def _prepare_file(audio: bytes | Path, suffix: str) -> bytes:
        if isinstance(audio, Path):
            audio = audio.read_bytes()
        return prepare_audio(audio, suffix=suffix, settings=settings)

    async def _transcribe_file(
        self,
        filename: str,
        audio: bytes | Path,
        model: str,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Transcribe a single file and return a result dict."""
        start_ms = time.monotonic()
        try:
            kwargs: dict[str, Any] = {}
            if options.get("language"):
                kwargs["language"] = options["language"]
            if options.get("temperature") is not None:
                kwargs["temperature"] = options["temperature"]
            # Only the filename survives submission; unknown types get the same .ogg
            # hint as an upload with an unrecognised content type.
            suffix = suffix_from_filename(filename) or ".ogg"

            prepared = await _run_in_thread(self._prepare_file, audio, suffix)
            async with self._inference_gate:
                # Every result reports language, duration and segments, so ask for
                # verbose_json and render text/srt/vtt here, like the single-file path.
                result = await _run_in_thread(
                    self._router.transcribe,
                    audio=prepared,
                    model=model,
                    response_format="verbose_json",
                    **kwargs,
                )
            text = result.get("text", "")
            response_format = options.get("response_format")
            if response_format in ("text", "srt", "vtt"):
                text, _ = format_transcription(result, response_format)

            elapsed_ms = int((time.monotonic() - start_ms) * 1000)
            return {
                "filename": filename,
                "status": "done",
                "text": text,
                "language": result.get("language", ""),
                "duration": result.get("duration", 0),
                "model": model,
                "segments": result.get("segments", []),
                "processing_time_ms": elapsed_ms,
            }

        except Exception as e:
            elapsed_ms = int((time.monotonic() - start_ms) * 1000)
            logger.warning("Batch file %s failed: %s", filename, e)
            return {
                "filename": filename,
                "status": "failed",
                "error": str(e),
                "processing_time_ms": elapsed_ms,
            }



    async def cancel(self, job_id: str) -> bool:
        """Cancel a queued or running job.

        A file already being decoded or transcribed cannot be interrupted: the job stops
        after it, and keeps its worker slot until then.
        """
        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            return True
        return False
