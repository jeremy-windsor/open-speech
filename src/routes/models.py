"""Model inventory and lifecycle routes."""

from __future__ import annotations

from typing import Callable

from fastapi import APIRouter

from src.services import models as model_service
from src.services import tts as tts_service


def create_router(*, get_settings: Callable, get_backend_router: Callable, get_tts_router: Callable, get_model_manager: Callable, get_progress_service: Callable) -> APIRouter:
    router = APIRouter()

    @router.get("/v1/models")
    async def list_models():
        return model_service.list_openai_models(
            settings=get_settings(),
            backend_router=get_backend_router(),
            tts_router=get_tts_router(),
        )

    @router.get("/v1/models/{model:path}")
    async def get_model(model: str):
        return model_service.get_model_object(model=model)

    @router.get("/api/ps")
    async def list_loaded_models():
        return model_service.list_loaded_stt_models(backend_router=get_backend_router())

    @router.post("/api/ps/{model:path}")
    async def load_model_legacy(model: str):
        return model_service.load_legacy_model(model=model, backend_router=get_backend_router())

    @router.delete("/api/ps/{model:path}")
    async def unload_model_legacy(model: str):
        return model_service.unload_legacy_model(model=model, backend_router=get_backend_router())

    @router.get("/api/models")
    async def list_all_models():
        return model_service.list_all_models(
            model_manager=get_model_manager(),
            tts_capabilities_for=lambda model_id: tts_service.tts_capabilities(tts_router=get_tts_router(), model_id=model_id),
        )

    @router.get("/api/models/{model_id:path}/status")
    async def get_model_status(model_id: str):
        return await get_progress_service().get_status(model_id=model_id, model_manager=get_model_manager())

    @router.get("/api/models/{model_id:path}/progress")
    async def get_model_progress(model_id: str):
        return await get_progress_service().get_progress(model_id=model_id, model_manager=get_model_manager())

    @router.post("/api/models/{model_id:path}/load")
    async def load_model_unified(model_id: str):
        return await get_progress_service().load(model_id=model_id, model_manager=get_model_manager())

    @router.post("/api/models/{model_id:path}/download")
    async def download_model_unified(model_id: str):
        return await get_progress_service().download(model_id=model_id, model_manager=get_model_manager())

    @router.post("/api/models/{model_id:path}/prefetch")
    async def prefetch_model_unified(model_id: str):
        return await get_progress_service().download(model_id=model_id, model_manager=get_model_manager())

    @router.delete("/api/models/{model_id:path}")
    async def unload_model_unified(model_id: str):
        return await get_progress_service().unload(model_id=model_id, model_manager=get_model_manager())

    @router.delete("/api/models/{model_id:path}/artifacts")
    async def delete_model_artifacts(model_id: str):
        return await get_progress_service().delete_artifacts(model_id=model_id, model_manager=get_model_manager())

    @router.post("/api/pull/{model:path}")
    async def pull_model(model: str):
        return model_service.pull_legacy_model(model=model, backend_router=get_backend_router())

    return router
