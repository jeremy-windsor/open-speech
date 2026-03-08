"""Batch transcription routes."""

from __future__ import annotations

from typing import Annotated, Callable

from fastapi import APIRouter, Form, Request

from src.services import batch as batch_service


def create_router(*, get_settings: Callable, get_batch_worker: Callable, get_batch_store: Callable) -> APIRouter:
    router = APIRouter()
    default_stt_model = get_settings().stt_model

    @router.post("/v1/audio/transcriptions/batch")
    async def batch_transcribe(
        request: Request,
        model: Annotated[str, Form()] = default_stt_model,
        language: Annotated[str | None, Form()] = None,
        response_format: Annotated[str, Form()] = "json",
        temperature: Annotated[float, Form()] = 0.0,
    ):
        settings = get_settings()
        return await batch_service.submit_batch_transcription(
            request=request,
            model=model or settings.stt_model,
            language=language,
            response_format=response_format,
            temperature=temperature,
            settings=settings,
            batch_worker=get_batch_worker(),
            batch_store=get_batch_store(),
        )

    @router.get("/v1/audio/jobs")
    async def list_batch_jobs(limit: int = 50, status: str | None = None):
        return batch_service.list_jobs(batch_store=get_batch_store(), limit=limit, status=status)

    @router.get("/v1/audio/jobs/{job_id}")
    async def get_batch_job(job_id: str):
        return batch_service.get_job_detail(batch_store=get_batch_store(), job_id=job_id)

    @router.get("/v1/audio/jobs/{job_id}/result")
    async def get_batch_job_result(job_id: str):
        return batch_service.get_job_result(batch_store=get_batch_store(), job_id=job_id)

    @router.delete("/v1/audio/jobs/{job_id}", status_code=204)
    async def delete_batch_job(job_id: str):
        return await batch_service.delete_job(
            batch_store=get_batch_store(),
            batch_worker=get_batch_worker(),
            job_id=job_id,
        )

    return router
