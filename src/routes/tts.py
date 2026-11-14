"""Text-to-speech routes."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import functools
from typing import Annotated, Callable

from fastapi import APIRouter, File, Form, Request, UploadFile, WebSocket
from fastapi.responses import JSONResponse

from src.middleware import (
    WS_CLOSE_FEATURE_OFF,
    WS_CLOSE_KEY,
    WS_CLOSE_ORIGIN,
    WS_KEY_REFUSED,
    WS_ORIGIN_REFUSED,
    refuse_websocket,
    verify_ws_api_key,
    verify_ws_origin,
)
from src.openai_compat import OPENAI_TTS_MODELS, apply_speech_defaults
from src.services import tts as tts_service
from src.tts.models import ModelLoadRequest, ModelUnloadRequest, TTSSpeechRequest

# /v1/audio/models/load and /unload wait for the TTS operation lock and then for the provider's synthesis lock, so
# behind a busy provider they queue here, on one thread (the operation lock lets one through at a time anyway),
# not in asyncio's default executor, whose threads every other request shares.
_tts_model_operations = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-model-operation")


async def _run_model_operation(fn, /, **kwargs):
    call = functools.partial(contextvars.copy_context().run, fn, **kwargs)
    return await asyncio.get_running_loop().run_in_executor(_tts_model_operations, call)


def create_router(
    *,
    get_settings: Callable,
    get_tts_router: Callable,
    get_tts_cache: Callable,
    get_pronunciation_dict: Callable,
    get_history_manager: Callable,
    get_voice_library: Callable,
    get_model_manager: Callable | None = None,
) -> APIRouter:
    router = APIRouter()

    def operation_lock():
        # ModelManager's lock of every TTS model operation: these cannot overlap a TTS load
        # through /api/models/{id}/load, nor a TTL or LRU eviction.
        return get_model_manager().operation_lock("tts") if get_model_manager else None

    @router.get("/api/tts/capabilities")
    async def get_tts_capabilities(model: str | None = None):
        return await asyncio.to_thread(
            tts_service.get_tts_capabilities_response,
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
        settings = get_settings()
        return await tts_service.synthesize_speech_response(
            request=apply_speech_defaults(request, settings),
            raw_request=raw_request,
            stream=stream,
            cache=cache,
            settings=settings,
            tts_router=get_tts_router(),
            tts_cache=get_tts_cache(),
            pronunciation_dict=get_pronunciation_dict(),
            history_manager=get_history_manager(),
            voice_library=get_voice_library(),
            openai_alias=request.model in OPENAI_TTS_MODELS,
            speed_from_settings="speed" not in request.model_fields_set,
        )

    @router.get("/v1/audio/speech/stream")
    async def live_speech_requires_websocket():
        return JSONResponse(
            status_code=426,
            content={
                "error": {
                    "code": "websocket_upgrade_required",
                    "message": "Connect to this endpoint with WebSocket (ws:// or wss://)",
                }
            },
            headers={"Upgrade": "websocket"},
        )

    @router.websocket("/v1/audio/speech/stream")
    async def live_speech(websocket: WebSocket):
        # A refused socket is accepted and then closed, so a browser gets the code and reason, not 1006. The
        # middleware has already refused a foreign Origin the same way.
        settings = get_settings()
        if not settings.tts_enabled or not settings.tts_live_enabled:
            await refuse_websocket(websocket, WS_CLOSE_FEATURE_OFF, "Live TTS is disabled")
            return
        if not verify_ws_origin(websocket):
            await refuse_websocket(websocket, WS_CLOSE_ORIGIN, WS_ORIGIN_REFUSED)
            return
        if not verify_ws_api_key(websocket):
            await refuse_websocket(websocket, WS_CLOSE_KEY, WS_KEY_REFUSED)
            return

        from src.live_tts.server import live_tts_endpoint

        await live_tts_endpoint(
            websocket,
            tts_router=get_tts_router(),
            pronunciation_dict=get_pronunciation_dict(),
            settings=settings,
            voice_library=get_voice_library(),
        )

    @router.post("/v1/audio/models/load")
    async def load_tts_model(request: ModelLoadRequest | None = None):
        settings = get_settings()
        model_id = request.model if request and request.model else settings.tts_model
        manager = get_model_manager() if get_model_manager else None
        return await _run_model_operation(
            tts_service.load_tts_model,
            settings=settings,
            tts_router=get_tts_router(),
            model_id=model_id,
            operation_lock=manager.operation_lock("tts") if manager else None,
            make_room=getattr(manager, "make_room_for", None),
        )

    @router.post("/v1/audio/models/unload")
    async def unload_tts_model(request: ModelUnloadRequest | None = None):
        settings = get_settings()
        model_id = request.model if request and request.model else settings.tts_model
        return await _run_model_operation(
            tts_service.unload_tts_model,
            settings=settings,
            tts_router=get_tts_router(),
            model_id=model_id,
            operation_lock=operation_lock(),
        )

    @router.get("/v1/audio/models")
    async def list_tts_models(loaded: bool = False):
        # Every usable TTS model, each with its load state; ?loaded=true lists only the loaded ones.
        return await asyncio.to_thread(
            tts_service.list_tts_models,
            settings=get_settings(),
            tts_router=get_tts_router(),
            model_manager=get_model_manager() if get_model_manager else None,
            loaded_only=loaded,
        )

    @router.get("/v1/audio/voices")
    async def list_voices(model: str | None = None):
        return await asyncio.to_thread(
            tts_service.list_voices,
            settings=get_settings(),
            tts_router=get_tts_router(),
            model=model,
        )

    @router.post("/v1/audio/speech/clone")
    async def clone_speech(
        input: Annotated[str, Form()],
        model: Annotated[str | None, Form()] = None,
        reference_audio: Annotated[UploadFile, File()] = None,
        voice_library_ref: Annotated[str | None, Form()] = None,
        voice: Annotated[str, Form()] = "Ryan",
        speed: Annotated[float, Form()] = 1.0,
        response_format: Annotated[str | None, Form()] = None,
        transcript: Annotated[str | None, Form()] = None,
        language: Annotated[str | None, Form()] = None,
    ):
        settings = get_settings()
        # As on /v1/audio/speech, a left-out model or format comes from the TTS_* settings.
        return await tts_service.clone_speech_response(
            input_text=input,
            model=model or settings.tts_model,
            reference_audio=reference_audio,
            voice_library_ref=voice_library_ref,
            voice=voice,
            speed=speed,
            response_format=response_format or settings.tts_default_format,
            transcript=transcript,
            language=language,
            settings=settings,
            tts_router=get_tts_router(),
            voice_library=get_voice_library(),
            pronunciation_dict=get_pronunciation_dict(),
        )

    return router
