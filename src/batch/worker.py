"""Async batch worker for processing multi-file transcription jobs."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any

from src.batch.store import BatchJobStore

logger = logging.getLogger("open-speech.batch")


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
        self._tasks: dict[str, asyncio.Task] = {}

    async def submit(
        self,
        job_id: str,
        audio_files: list[tuple[str, bytes]],
        options: dict[str, Any],
    ) -> None:
        """Queue a job for background processing. Returns immediately."""
        task = asyncio.create_task(self._run_with_semaphore(job_id, audio_files, options))
        self._tasks[job_id] = task
        task.add_done_callback(lambda t: self._tasks.pop(job_id, None))

    async def _run_with_semaphore(
        self,
        job_id: str,
        audio_files: list[tuple[str, bytes]],
        options: dict[str, Any],
    ) -> None:
        async with self._semaphore:
            await self._process_job(job_id, audio_files, options)

    async def _process_job(
        self,
        job_id: str,
        audio_files: list[tuple[str, bytes]],
        options: dict[str, Any],
    ) -> None:
        """Process each file sequentially, store results, update status."""
        try:
            self._store.update(job_id, status="running", started_at=time.time())
            model = options.get("model", "")
            results: list[dict[str, Any]] = []

            for filename, audio_bytes in audio_files:
                result = await self._transcribe_file(filename, audio_bytes, model, options)
                results.append(result)
                # Update results incrementally
                self._store.update(job_id, results=results)

            self._store.update(
                job_id,
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
            self._store.update(
                job_id,
                status="failed",
                finished_at=time.time(),
                error="Cancelled",
            )
            raise  # re-raise so asyncio knows the task is cancelled

        except Exception as e:
            logger.exception("Batch job %s failed", job_id)
            self._store.update(
                job_id,
                status="failed",
                finished_at=time.time(),
                error=str(e),
            )

    async def _transcribe_file(
        self,
        filename: str,
        audio_bytes: bytes,
        model: str,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        """Transcribe a single file and return a result dict."""
        start_ms = time.monotonic()
        try:
            loop = asyncio.get_running_loop()
            kwargs: dict[str, Any] = {}
            if options.get("language"):
                kwargs["language"] = options["language"]
            if options.get("response_format"):
                kwargs["response_format"] = options["response_format"]
            if options.get("temperature") is not None:
                kwargs["temperature"] = options["temperature"]

            result = await loop.run_in_executor(
                None,
                lambda: self._router.transcribe(
                    audio=audio_bytes,
                    model=model,
                    **kwargs,
                ),
            )

            elapsed_ms = int((time.monotonic() - start_ms) * 1000)
            return {
                "filename": filename,
                "status": "done",
                "text": result.get("text", ""),
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
        """Cancel a running job if possible."""
        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            return True
        return False
