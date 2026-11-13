"""Model lifecycle and listing service helpers."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
from contextlib import nullcontext

from fastapi import HTTPException

from src.backends.faster_whisper import NetworkModelPathError, refuse_network_model_path
from src.config import settings
from src.model_manager import ModelLifecycleError, ModelState
from src.model_registry import get_known_model
from src.models import HealthResponse, LoadedModelsResponse, ModelListResponse, ModelObject, PullResponse
from src.router import await_fetch
from src.tts.external import ExternalProviderError

logger = logging.getLogger("open-speech")

_IN_FLIGHT = ("queued", "downloading", "loading")


def _model_http_error(exc: ModelLifecycleError | ExternalProviderError) -> HTTPException:
    status_code = exc.status_code or (404 if exc.code == "unknown_model" else 400)
    return HTTPException(status_code=status_code, detail={"message": exc.message, "code": exc.code})


def _downloaded_hint(progress: dict | None, state: str) -> bool:
    """Whether a finished download or load still vouches for weights the live state cannot see."""
    # Isolated workers keep weights in their own volume, out of the artifact scan's sight. The
    # hint never outranks a live loaded, downloaded or offline state.
    return (
        bool(progress)
        and progress.get("status") in ("downloaded", "ready")
        and state in (ModelState.AVAILABLE, ModelState.PROVIDER_INSTALLED)
    )


class ModelProgressService:
    """Track in-flight model download/load progress."""

    def __init__(self) -> None:
        self.download_progress: dict[str, dict] = {}
        self.download_progress_lock = asyncio.Lock()
        # One queue per model type, as ModelManager keeps one lock per type: an STT unload
        # waiting for a running decode must not hold up a TTS load ("queued" in the UI).
        self.model_operation_locks = {"stt": asyncio.Lock(), "tts": asyncio.Lock()}
        # Models whose weights a finished download or load fetched. A failed operation
        # deletes no weights, so only an artifact deletion forgets them.
        self._fetched: set[str] = set()

    def operation_lock(self, model_type: str) -> asyncio.Lock:
        """The route-level queue of the load, unload and artifact deletion of one model type."""
        return self.model_operation_locks[model_type]

    @staticmethod
    async def _model_type(model_id: str, model_manager) -> str:
        """The model's type; an unknown model is the usual 404."""
        try:
            # Off the event loop: the type can depend on what an isolated worker reports.
            return await asyncio.to_thread(model_manager._resolve_type, model_id)
        except ModelLifecycleError as exc:
            raise _model_http_error(exc) from exc

    async def _type_lock(self, model_id: str, model_manager) -> asyncio.Lock:
        """The lock of the model's type; an unknown model is the usual 404."""
        return self.operation_lock(await self._model_type(model_id, model_manager))

    async def _fetch_stt_first(self, model_id: str, model_manager) -> None:
        """Fetch an STT model's files before its load takes the STT queue.

        So no other STT load, unload or artifact deletion waits for the download, which reads "downloading".
        """
        start_stt_fetch = getattr(model_manager, "start_stt_fetch", None)
        if not callable(start_stt_fetch):
            return
        try:
            future = await asyncio.to_thread(start_stt_fetch, model_id)
        except ModelLifecycleError as exc:
            raise _model_http_error(exc) from exc
        if not isinstance(future, concurrent.futures.Future):
            return  # loaded already, or a backend that fetches only as it loads
        async with self.download_progress_lock:
            self.download_progress[model_id] = {"status": "downloading", "progress": 0.1}
        try:
            await await_fetch(future)
        except Exception as exc:
            async with self.download_progress_lock:
                self._settle_failed(model_id)
            raise _model_http_error(model_manager.stt_fetch_failed(model_id, exc)) from exc

    @staticmethod
    async def _refuse_disabled_tts(model_id: str, model_manager, action: str) -> None:
        """TTS_ENABLED=false: a TTS model's status and progress are the load's 404 tts_disabled, as /api/models
        lists no TTS model. STT models, and the unknown_model 404 of the read itself, are left as they are."""
        if settings.tts_enabled:
            return
        try:
            # Off the event loop, as _type_lock: the type can depend on what an isolated worker reports.
            model_type = await asyncio.to_thread(model_manager._resolve_type, model_id)
        except ModelLifecycleError:
            return
        try:
            model_manager._require_enabled(model_id, model_type, action)
        except ModelLifecycleError as exc:
            raise _model_http_error(exc) from exc

    def _settle_failed(self, model_id: str) -> None:
        """Replace a failed operation's progress with what an earlier success vouches for."""
        if model_id in self._fetched:
            self.download_progress[model_id] = {"status": "downloaded", "progress": 1.0}
        else:
            self.download_progress.pop(model_id, None)

    async def get_status(self, *, model_id: str, model_manager):
        await self._refuse_disabled_tts(model_id, model_manager, "status")
        try:
            info = await asyncio.to_thread(model_manager.status, model_id)
        except ModelLifecycleError as exc:
            raise HTTPException(status_code=404, detail={"message": exc.message, "code": exc.code}) from exc
        result = info.to_dict()
        async with self.download_progress_lock:
            progress = self.download_progress.get(model_id)
        if progress:
            progress_status = progress.get("status", "")
            if progress_status in _IN_FLIGHT:
                # A reload queued behind another operation must not hide a model that is serving.
                if result.get("state") != "loaded":
                    result["state"] = progress_status
            elif _downloaded_hint(progress, result.get("state")):
                result["state"] = "downloaded"
            result["progress"] = progress.get("progress", 0)
        return result

    async def get_progress(self, *, model_id: str, model_manager):
        await self._refuse_disabled_tts(model_id, model_manager, "progress")
        async with self.download_progress_lock:
            progress = self.download_progress.get(model_id)
        if progress and progress.get("status") in _IN_FLIGHT:
            return progress
        try:
            info = await asyncio.to_thread(model_manager.status, model_id)
        except ModelLifecycleError as exc:
            raise HTTPException(status_code=404, detail={"message": exc.message, "code": exc.code}) from exc
        if info.state == ModelState.LOADED:
            return {"status": "ready", "progress": 1.0}
        if info.state == ModelState.DOWNLOADED or _downloaded_hint(progress, info.state):
            return {"status": "downloaded", "progress": 1.0}
        return {"status": "idle", "progress": 0.0}

    async def load(self, *, model_id: str, model_manager):
        model_type = await self._model_type(model_id, model_manager)
        if model_type == "stt":
            await self._fetch_stt_first(model_id, model_manager)
        operation_lock = self.operation_lock(model_type)
        async with self.download_progress_lock:
            self.download_progress[model_id] = {"status": "queued", "progress": 0.0}
        async with operation_lock:
            async with self.download_progress_lock:
                self.download_progress[model_id] = {"status": "loading", "progress": 0.5}
            try:
                info = await asyncio.to_thread(model_manager.load, model_id)
                async with self.download_progress_lock:
                    self.download_progress[model_id] = {"status": "ready", "progress": 1.0}
                    self._fetched.add(model_id)
            except ModelLifecycleError as exc:
                async with self.download_progress_lock:
                    self._settle_failed(model_id)
                raise _model_http_error(exc) from exc
            except Exception as exc:
                async with self.download_progress_lock:
                    self._settle_failed(model_id)
                logger.exception("Failed to load model %s", model_id)
                raise HTTPException(
                    status_code=500,
                    detail={"message": str(exc), "code": "load_failed", "model": model_id},
                )
        return info.to_dict()

    async def download(self, *, model_id: str, model_manager):
        # No route lock: ModelManager fetches from the hub outside every lock, and takes its
        # type's lock for a fetch that needs a load, so a long download holds up no load or unload.
        async with self.download_progress_lock:
            self.download_progress[model_id] = {"status": "downloading", "progress": 0.1}
        try:
            info = await asyncio.to_thread(model_manager.download, model_id)
            async with self.download_progress_lock:
                self.download_progress[model_id] = {"status": "downloaded", "progress": 1.0}
                self._fetched.add(model_id)
            return info.to_dict()
        except (ModelLifecycleError, ExternalProviderError) as exc:
            # The manager wraps load errors, not those of the unload and reads around the load.
            async with self.download_progress_lock:
                self._settle_failed(model_id)
            raise _model_http_error(exc) from exc
        except Exception as exc:
            async with self.download_progress_lock:
                self._settle_failed(model_id)
            logger.exception("Failed to download model %s", model_id)
            raise HTTPException(
                status_code=500,
                detail={"message": str(exc), "code": "download_failed", "model": model_id},
            )

    async def unload(self, *, model_id: str, model_manager):
        try:
            info = await asyncio.to_thread(model_manager.status, model_id)
        except ModelLifecycleError as exc:
            raise HTTPException(status_code=404, detail={"message": exc.message, "code": exc.code}) from exc
        if info.state != ModelState.LOADED:
            raise HTTPException(
                status_code=404,
                detail={"message": f"Model {model_id} is not loaded", "code": "not_loaded", "model": model_id},
            )
        async with await self._type_lock(model_id, model_manager):
            try:
                await asyncio.to_thread(model_manager.unload, model_id)
            except (ModelLifecycleError, ExternalProviderError) as exc:
                raise _model_http_error(exc) from exc
            except Exception as exc:
                logger.exception("Failed to unload model %s", model_id)
                raise HTTPException(
                    status_code=500,
                    detail={"message": str(exc), "code": "unload_failed", "model": model_id},
                )
        return {"status": "unloaded", "model": model_id}

    async def delete_artifacts(self, *, model_id: str, model_manager):
        async with await self._type_lock(model_id, model_manager):
            try:
                result = await asyncio.to_thread(model_manager.delete_artifacts, model_id)
            except ModelLifecycleError as exc:
                raise HTTPException(status_code=404, detail={"message": exc.message, "code": exc.code}) from exc
            async with self.download_progress_lock:
                self.download_progress.pop(model_id, None)
                self._fetched.discard(model_id)
            return result


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


def _unusable_tts_model(known: dict, *, settings, tts_router) -> bool:
    """Whether a registry TTS model is one no request can use, so its detail must 404.

    A TTS model needs TTS enabled and its provider installed. An isolated worker's model exists
    only once the worker is configured (/api/models hides it otherwise), and not when the
    worker's manifest leaves it out. An offline worker cannot say, so its models stay visible,
    as /api/models lists them (unavailable).
    """
    if known.get("type") != "tts":
        return False
    backend = getattr(tts_router, "_backends", {}).get(known["provider"])
    if not settings.tts_enabled or backend is None:
        return True
    advertises_model = getattr(backend, "advertises_model", None)
    return (
        bool(known.get("optional_provider"))
        and callable(advertises_model)
        and advertises_model(known["id"]) is False
    )


def get_model_object(*, model: str, settings, backend_router, tts_router):
    """Return OpenAI-style model detail."""
    known = get_known_model(model)
    if known and not _unusable_tts_model(known, settings=settings, tts_router=tts_router):
        return ModelObject(id=model, owned_by=f"open-speech/{known['provider']}")
    for advertised in list_openai_models(
        settings=settings, backend_router=backend_router, tts_router=tts_router,
    ).data:
        if advertised.id == model:
            return advertised
    raise HTTPException(status_code=404, detail=f"Model {model} not found")


def list_loaded_stt_models(*, backend_router):
    """Return legacy loaded STT models response."""
    return LoadedModelsResponse(models=backend_router.loaded_models())


def list_all_models(
    *, model_manager, tts_capabilities_for, default_stt_model: str, default_tts_model: str, tts_enabled: bool = True,
):
    """Return unified model inventory with TTS capabilities.

    With TTS disabled no TTS model can be loaded or used, so none is listed, as in /v1/models.
    """
    models = [model.to_dict() for model in model_manager.list_all()]
    if not tts_enabled:
        models = [model for model in models if model.get("type") != "tts"]
    for model in models:
        if model.get("type") == "tts":
            try:
                model["capabilities"] = tts_capabilities_for(model["id"])
            except Exception:
                model["capabilities"] = {}
    return {"models": models, "default_stt_model": default_stt_model, "default_tts_model": default_tts_model}


def health_response(*, version: str, backend_router):
    """Return health response."""
    loaded = backend_router.loaded_models()
    return HealthResponse(version=version, models_loaded=len(loaded), revision=os.getenv("OS_BUILD_REVISION"))


# The legacy routes take ModelManager.operation_lock("stt"), the lock of every other STT model
# operation, so a load cannot overlap another and leave two STT models loaded.


def _refuse_network_model(model: str) -> None:
    """400 invalid_model for a network (UNC) or device path, before anything is unloaded or stats it."""
    try:
        refuse_network_model_path(model)
    except NetworkModelPathError as exc:
        raise HTTPException(status_code=400, detail={"message": str(exc), "code": exc.code}) from exc


def load_legacy_model(*, model: str, backend_router, operation_lock=None):
    """Load a legacy STT model in place of the other loaded STT models."""
    _refuse_network_model(model)
    with operation_lock or nullcontext():
        for loaded in backend_router.loaded_models():
            if loaded.model != model:
                try:
                    backend_router.unload_model(loaded.model)
                    logger.info("Auto-unloaded STT model %s to load %s", loaded.model, model)
                except Exception as exc:
                    logger.warning("Failed to auto-unload STT model %s: %s", loaded.model, exc)

        try:
            backend_router.load_model(model)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Failed to load model %s", model)
            raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "loaded", "model": model}


def unload_legacy_model(*, model: str, backend_router, operation_lock=None):
    """Unload a legacy STT model."""
    with operation_lock or nullcontext():
        if not backend_router.is_model_loaded(model):
            raise HTTPException(status_code=404, detail=f"Model {model} is not loaded")
        backend_router.unload_model(model)
    return {"status": "unloaded", "model": model}


def pull_legacy_model(*, model: str, backend_router, operation_lock=None):
    """Download a model using the legacy pull endpoint, leaving what is loaded as it was."""
    _refuse_network_model(model)
    try:
        download_model = getattr(backend_router, "download_model", None)
        if not (callable(download_model) and download_model(model)):
            # The backend can fetch only by loading: unload again unless the model was serving.
            with operation_lock or nullcontext():
                was_loaded = backend_router.is_model_loaded(model)
                backend_router.load_model(model)
                if not was_loaded:
                    backend_router.unload_model(model)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to pull model %s", model)
        raise HTTPException(status_code=500, detail=str(exc))
    return PullResponse(status="downloaded", model=model)
