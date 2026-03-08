"""Batch transcription service helpers."""

from __future__ import annotations

import time
from uuid import uuid4

from fastapi import HTTPException
from fastapi.responses import JSONResponse, Response

from src.batch.store import BatchJob


async def submit_batch_transcription(*, request, model: str, language: str | None, response_format: str, temperature: float, settings, batch_worker, batch_store):
    """Submit a batch transcription job and return the queue response."""
    if batch_worker is None:
        raise HTTPException(status_code=503, detail="Batch worker not initialized")

    pending_count = len(batch_worker._tasks)
    if pending_count >= settings.os_batch_max_pending:
        raise HTTPException(
            status_code=429,
            detail=f"Server busy: {pending_count} pending jobs. Try again later.",
        )

    form = await request.form()
    try:
        files = form.getlist("file")
        if not files:
            raise HTTPException(status_code=422, detail="No files provided")
        if not model:
            raise HTTPException(status_code=422, detail="Model is required")
        if len(files) > 20:
            raise HTTPException(status_code=422, detail="Maximum 20 files per batch")

        max_bytes = settings.os_max_upload_mb * 1024 * 1024
        max_total_bytes = settings.os_batch_max_total_mb * 1024 * 1024
        audio_files: list[tuple[str, bytes]] = []
        filenames: list[str] = []
        total_bytes = 0
        for file in files:
            data = await file.read()
            if len(data) == 0:
                raise HTTPException(status_code=400, detail=f"File {file.filename!r} is empty")
            if len(data) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"File {file.filename!r} exceeds {settings.os_max_upload_mb}MB per-file limit",
                )
            total_bytes += len(data)
            if total_bytes > max_total_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Batch total size exceeds {settings.os_batch_max_total_mb}MB aggregate limit",
                )
            audio_files.append((file.filename or "unknown", data))
            filenames.append(file.filename or "unknown")
    finally:
        await form.close()

    job = BatchJob(
        job_id=str(uuid4()),
        status="queued",
        created_at=time.time(),
        model=model,
        files=filenames,
        options={
            "model": model,
            "language": language,
            "response_format": response_format,
            "temperature": temperature,
        },
    )
    batch_store.create(job)
    await batch_worker.submit(job.job_id, audio_files, job.options)

    return JSONResponse(
        {
            "job_id": job.job_id,
            "status": "queued",
            "file_count": len(filenames),
            "created_at": job.created_at,
        }
    )


def list_jobs(*, batch_store, limit: int = 50, status: str | None = None):
    """List batch jobs."""
    if limit > 200:
        limit = 200
    jobs = batch_store.list_jobs(limit=limit, status=status)
    return JSONResponse({"jobs": [job.to_summary() for job in jobs], "total": len(jobs)})


def get_job_detail(*, batch_store, job_id: str):
    """Return detail for a single batch job."""
    job = batch_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse(job.to_detail())


def get_job_result(*, batch_store, job_id: str):
    """Return just the results for a completed batch job."""
    job = batch_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "done":
        return JSONResponse(
            {"error": "Job not complete", "status": job.status, "retry_after": 5},
            status_code=409,
        )
    return JSONResponse({"results": job.results})


async def delete_job(*, batch_store, batch_worker, job_id: str):
    """Delete or cancel a batch job."""
    job = batch_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if batch_worker and job.status in ("queued", "running"):
        await batch_worker.cancel(job.job_id)
    batch_store.delete(job_id)
    return Response(status_code=204)
