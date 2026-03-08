"""Speech-to-text HTTP routes."""

from __future__ import annotations

from typing import Annotated, Callable

from fastapi import APIRouter, File, Form, Request, UploadFile

from src.services import stt as stt_service


def create_router(*, get_settings: Callable, get_backend_router: Callable, get_history_manager: Callable, get_diarizer_cls: Callable, get_attach_speakers_fn: Callable) -> APIRouter:
    router = APIRouter()
    default_stt_model = get_settings().stt_model

    @router.post("/v1/audio/transcriptions")
    async def transcribe(
        raw_request: Request,
        file: Annotated[UploadFile, File()],
        model: Annotated[str, Form()] = default_stt_model,
        language: Annotated[str | None, Form()] = None,
        prompt: Annotated[str | None, Form()] = None,
        response_format: Annotated[str, Form()] = "json",
        temperature: Annotated[float, Form()] = 0.0,
        diarize: bool = False,
    ):
        settings = get_settings()
        return await stt_service.transcribe_request(
            file=file,
            model=model or settings.stt_model,
            language=language,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            diarize=diarize,
            raw_request=raw_request,
            settings=settings,
            backend_router=get_backend_router(),
            history_manager=get_history_manager(),
            diarizer_cls=get_diarizer_cls(),
            attach_speakers_fn=get_attach_speakers_fn(),
        )

    @router.post("/v1/audio/translations")
    async def translate(
        file: Annotated[UploadFile, File()],
        model: Annotated[str, Form()] = default_stt_model,
        prompt: Annotated[str | None, Form()] = None,
        response_format: Annotated[str, Form()] = "json",
        temperature: Annotated[float, Form()] = 0.0,
    ):
        settings = get_settings()
        return await stt_service.translate_request(
            file=file,
            model=model or settings.stt_model,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
            settings=settings,
            backend_router=get_backend_router(),
        )

    return router
