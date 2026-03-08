"""Model lifecycle and listing service helpers."""

from __future__ import annotations

import asyncio
import logging

from fastapi import HTTPException

from src.model_manager import ModelLifecycleError, ModelState
from src.models import HealthResponse, LoadedModelsResponse, ModelListResponse, ModelObject, PullResponse

logger = logging.getLogger("open-speech")


class ModelProgressService:
    """Track in-flight model download/load progress."""

    def __init__(self) -> None:
        self.download_progress: dict[str, dict] = {}
        self.download_progress_lock = asyncio.Lock()
        self.model_operation_lock = asyncio.Lock()

    async def get_status(self, *, model_id: str, model_manager):
        info = model_manager.status(model_id)
        result = info.to_dict()
        async with self.download_progress_lock:
            progress = self.download_progress.get(model_id)
        if progress:
            progress_status = progress.get("status", "")
            if progress_status == "queued":
                result["state"] = "queued"
            elif progress_status == "downloading":
                result["state"] = "downloading"
            elif progress_status == "loading":
                result["state"] = "loading"
            elif progress_status in ("downloaded", "ready") and result.get("state") != "loaded":
                result["state"] = "downloaded"
            result["progress"] = progress.get("progress", 0)
        return result

    async def get_progress(self, *, model_id: str, model_manager):
        async with self.download_progress_lock:
            if model_id in self.download_progress:
                return self.download_progress[model_id]
        info = model_manager.status(model_id)
        if info.state == ModelState.LOADED:
            return {"status": "ready", "progress": 1.0}
        if info.state == ModelState.DOWNLOADED:
            return {"status": "downloaded", "progress": 1.0}
        return {"status": "idle", "progress": 0.0}

    async def load(self, *, model_id: str, model_manager):
        async with self.download_progress_lock:
            self.download_progress[model_id] = {"status": "queued", "progress": 0.0}
        async with self.model_operation_lock:
            async with self.download_progress_lock:
                self.download_progress[model_id] = {"status": "loading", "progress": 0.5}
            try:
                info = model_manager.load(model_id)
                async with self.download_progress_lock:
                    self.download_progress[model_id] = {"status": "ready", "progress": 1.0}
            except ModelLifecycleError as exc:
                async with self.download_progress_lock:
                    self.download_progress.pop(model_id, None)
                raise HTTPException(status_code=400, detail={"message": exc.message, "code": exc.code})
            except Exception as exc:
                async with self.download_progress_lock:
                    self.download_progress.pop(model_id, None)
                logger.exception("Failed to load model %s", model_id)
                raise HTTPException(
                    status_code=500,
                    detail={"message": str(exc), "code": "load_failed", "model": model_id},
                )
        return info.to_dict()

    async def download(self, *, model_id: str, model_manager):
        async with self.download_progress_lock:
            self.download_progress[model_id] = {"status": "queued", "progress": 0.0}
        async with self.model_operation_lock:
            async with self.download_progress_lock:
                self.download_progress[model_id] = {"status": "downloading", "progress": 0.1}
            try:
                info = model_manager.download(model_id)
                async with self.download_progress_lock:
                    self.download_progress[model_id] = {"status": "downloaded", "progress": 1.0}
                return info.to_dict()
            except ModelLifecycleError as exc:
                async with self.download_progress_lock:
                    self.download_progress.pop(model_id, None)
                raise HTTPException(status_code=400, detail={"message": exc.message, "code": exc.code})
            except Exception as exc:
                async with self.download_progress_lock:
                    self.download_progress.pop(model_id, None)
                logger.exception("Failed to download model %s", model_id)
                raise HTTPException(
                    status_code=500,
                    detail={"message": str(exc), "code": "download_failed", "model": model_id},
                )

    async def unload(self, *, model_id: str, model_manager):
        info = model_manager.status(model_id)
        if info.state != ModelState.LOADED:
            raise HTTPException(
                status_code=404,
                detail={"message": f"Model {model_id} is not loaded", "code": "not_loaded", "model": model_id},
            )
        async with self.model_operation_lock:
            model_manager.unload(model_id)
        return {"status": "unloaded", "model": model_id}

    async def delete_artifacts(self, *, model_id: str, model_manager):
        async with self.model_operation_lock:
            return model_manager.delete_artifacts(model_id)


progress_service = ModelProgressService()


def list_openai_models(*, settings, backend_router, tts_router):
    """Return OpenAI-style model listing."""
    loaded = backend_router.loaded_models()
    models = [ModelObject(id=model.model, owned_by=f"open-speech/{model.backend}") for model in loaded]
    loaded_ids = {model.model for model in loaded}
    if settings.stt_model not in loaded_ids:
        models.append(ModelObject(id=settings.stt_model))
    if settings.tts_enabled:
        tts_loaded = tts_router.loaded_models()
        tts_loaded_ids = {model.model for model in tts_loaded}
        for model in tts_loaded:
            models.append(ModelObject(id=model.model, owned_by=f"open-speech/{model.backend}"))
        if settings.tts_model not in tts_loaded_ids:
            models.append(ModelObject(id=settings.tts_model, owned_by="open-speech/tts"))
    return ModelListResponse(data=models)


def get_model_object(*, model: str):
    """Return OpenAI-style model detail."""
    return ModelObject(id=model)


def list_loaded_stt_models(*, backend_router):
    """Return legacy loaded STT models response."""
    return LoadedModelsResponse(models=backend_router.loaded_models())


def list_all_models(*, model_manager, tts_capabilities_for):
    """Return unified model inventory with TTS capabilities."""
    models = [model.to_dict() for model in model_manager.list_all()]
    for model in models:
        if model.get("type") == "tts":
            try:
                model["capabilities"] = tts_capabilities_for(model["id"])
            except Exception:
                model["capabilities"] = {}
    return {"models": models}


def health_response(*, version: str, backend_router):
    """Return health response."""
    loaded = backend_router.loaded_models()
    return HealthResponse(version=version, models_loaded=len(loaded))


def load_legacy_model(*, model: str, backend_router):
    """Load a legacy STT model."""
    for loaded in backend_router.loaded_models():
        if loaded.model != model:
            try:
                backend_router.unload_model(loaded.model)
                logger.info("Auto-unloaded STT model %s to load %s", loaded.model, model)
            except Exception as exc:
                logger.warning("Failed to auto-unload STT model %s: %s", loaded.model, exc)

    try:
        backend_router.load_model(model)
    except Exception as exc:
        logger.exception("Failed to load model %s", model)
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "loaded", "model": model}


def unload_legacy_model(*, model: str, backend_router):
    """Unload a legacy STT model."""
    if not backend_router.is_model_loaded(model):
        raise HTTPException(status_code=404, detail=f"Model {model} is not loaded")
    backend_router.unload_model(model)
    return {"status": "unloaded", "model": model}


def pull_legacy_model(*, model: str, backend_router):
    """Download a model using the legacy pull endpoint."""
    try:
        backend_router.load_model(model)
        backend_router.unload_model(model)
    except Exception as exc:
        logger.exception("Failed to pull model %s", model)
        raise HTTPException(status_code=500, detail=str(exc))
    return PullResponse(status="downloaded", model=model)
