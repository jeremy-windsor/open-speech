"""Text-to-speech routes."""

from __future__ import annotations

from typing import Annotated, Callable

from fastapi import APIRouter, File, Form, Request, UploadFile

from src.services import tts as tts_service
from src.tts.models import ModelLoadRequest, ModelUnloadRequest, TTSSpeechRequest


def create_router(*, get_settings: Callable, get_tts_router: Callable, get_tts_cache: Callable, get_pronunciation_dict: Callable, get_history_manager: Callable, get_voice_library: Callable) -> APIRouter:
    router = APIRouter()

    @router.get("/api/tts/capabilities")
    async def get_tts_capabilities(model: str | None = None):
        return tts_service.get_tts_capabilities_response(
            settings=get_settings(),
            tts_router=get_tts_router(),
            model=model,
        )

    @router.post("/v1/audio/speech")
    async def synthesize_speech(
        request: TTSSpeechRequest,
        raw_request: Request,
        stream: bool = False,
        cache: bool = True,
    ):
        return await tts_service.synthesize_speech_response(
            request=request,
            raw_request=raw_request,
            stream=stream,
            cache=cache,
            settings=get_settings(),
            tts_router=get_tts_router(),
            tts_cache=get_tts_cache(),
            pronunciation_dict=get_pronunciation_dict(),
            history_manager=get_history_manager(),
        )

    @router.post("/v1/audio/models/load")
    async def load_tts_model(request: ModelLoadRequest | None = None):
        settings = get_settings()
        model_id = request.model if request else settings.tts_model
        return tts_service.load_tts_model(settings=settings, tts_router=get_tts_router(), model_id=model_id)

    @router.post("/v1/audio/models/unload")
    async def unload_tts_model(request: ModelUnloadRequest | None = None):
        settings = get_settings()
        model_id = request.model if request else settings.tts_model
        return tts_service.unload_tts_model(settings=settings, tts_router=get_tts_router(), model_id=model_id)

    @router.get("/v1/audio/models")
    async def list_tts_models():
        return tts_service.list_tts_models(settings=get_settings(), tts_router=get_tts_router())

    @router.get("/v1/audio/voices")
    async def list_voices(model: str | None = None):
        return tts_service.list_voices(settings=get_settings(), tts_router=get_tts_router(), model=model)

    @router.post("/v1/audio/speech/clone")
    async def clone_speech(
        input: Annotated[str, Form()],
        model: Annotated[str, Form()] = "kokoro",
        reference_audio: Annotated[UploadFile, File()] = None,
        voice_library_ref: Annotated[str | None, Form()] = None,
        voice: Annotated[str, Form()] = "Ryan",
        speed: Annotated[float, Form()] = 1.0,
        response_format: Annotated[str, Form()] = "mp3",
        transcript: Annotated[str | None, Form()] = None,
        language: Annotated[str | None, Form()] = None,
    ):
        return await tts_service.clone_speech_response(
            input_text=input,
            model=model,
            reference_audio=reference_audio,
            voice_library_ref=voice_library_ref,
            voice=voice,
            speed=speed,
            response_format=response_format,
            transcript=transcript,
            language=language,
            settings=get_settings(),
            tts_router=get_tts_router(),
            voice_library=get_voice_library(),
        )

    return router
