"""Open Speech application factory and runtime entrypoint."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.batch.store import BatchJobStore
from src.batch.worker import BatchWorker
from src.cache.tts_cache import TTSCache
from src.composer import MultiTrackComposer
from src.config import settings
from src.conversation import ConversationManager
from src.history import HistoryManager
from src.lifecycle import ModelLifecycleManager
from src.middleware import SecurityMiddleware
from src.model_manager import ModelManager
from src.profiles import ProfileManager
from src.pronunciation.dictionary import PronunciationDictionary
from src.diarization.pyannote_diarizer import PyannoteDiarizer, attach_text_to_speakers
from src.router import router as backend_router
from src.routes import batch as batch_routes
from src.routes import health as health_routes
from src.routes import models as model_routes
from src.routes import realtime as realtime_routes
from src.routes import streaming as streaming_routes
from src.routes import studio as studio_routes
from src.routes import stt as stt_routes
from src.routes import tts as tts_routes
from src.routes import web as web_routes
from src.services import models as model_service
from src.services import stt as stt_service
from src.services import tts as tts_service
from src.storage import init_db
from src.tts.router import TTSRouter
from src.voice_library import VoiceLibraryManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("open-speech")

__version__ = "0.7.0"
STATIC_DIR = Path(__file__).parent / "static"


def get_runtime_version() -> str:
    """Return the running app version from package metadata, with a safe fallback."""
    try:
        return package_version("open-speech")
    except PackageNotFoundError:
        return __version__


def _suffix_from_filename(filename: str) -> str | None:
    """Extract audio suffix from filename."""
    return stt_service.suffix_from_filename(filename)


tts_router = TTSRouter(device=settings.tts_effective_device)
model_manager = ModelManager(stt_router=backend_router, tts_router=tts_router)
tts_cache = TTSCache(settings.tts_cache_dir, settings.tts_cache_max_mb, settings.tts_cache_enabled)
pronunciation_dict = PronunciationDictionary(settings.tts_pronunciation_dict or None)
voice_library = VoiceLibraryManager(settings.voice_library_path, max_count=settings.voice_library_max_count)
profile_manager = ProfileManager()
history_manager = HistoryManager()
batch_store = BatchJobStore()
batch_worker: BatchWorker | None = None
progress_service = model_service.progress_service
_download_progress = progress_service.download_progress
_download_progress_lock = progress_service.download_progress_lock
_model_operation_lock = progress_service.model_operation_lock
DEFAULT_VOICE_PRESETS = tts_service.DEFAULT_VOICE_PRESETS


def _load_voice_presets() -> list[dict]:
    """Load voice presets from config file or defaults."""
    return tts_service.load_voice_presets()


def _synthesize_array(*, text: str, model: str, voice: str, speed: float, sample_rate: int = 24000, language: str | None = None):
    return tts_service.synthesize_array(
        text=text,
        model=model,
        voice=voice,
        speed=speed,
        sample_rate=sample_rate,
        language=language,
        tts_router=tts_router,
        settings=settings,
    )


conversation_manager = ConversationManager(profile_manager=profile_manager, synthesize_fn=_synthesize_array)
composer_manager = MultiTrackComposer()


def _tts_backend_name(model_id: str) -> str:
    return tts_service.tts_backend_name(tts_router=tts_router, model_id=model_id)


def _tts_capabilities(model_id: str) -> dict:
    return tts_service.tts_capabilities(tts_router=tts_router, model_id=model_id)


def _validate_tts_feature_support(*, model_id: str, voice_design: str | None = None, reference_audio: bytes | str | None = None) -> str | None:
    return tts_service.validate_tts_feature_support(
        tts_router=tts_router,
        model_id=model_id,
        voice_design=voice_design,
        reference_audio=reference_audio,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Open Speech v%s starting up", get_runtime_version())
    logger.info("Default STT model: %s", settings.stt_model)
    logger.info("Default TTS model: %s", settings.tts_model)
    logger.info("Device: %s, Compute: %s", settings.stt_device, settings.stt_compute_type)

    init_db()

    global batch_worker
    batch_worker = BatchWorker(batch_store, backend_router, max_concurrent=settings.os_batch_workers)

    zombie_jobs = batch_store.list_jobs(limit=200, status="running")
    for zombie in zombie_jobs:
        logger.warning("Recovering zombie batch job %s (was running on previous server instance)", zombie.job_id)
        batch_store.update(zombie.job_id, status="failed", finished_at=time.time(), error="Server restarted during processing")

    try:
        Path(settings.os_composer_dir).mkdir(parents=True, exist_ok=True)
    except PermissionError:
        Path("data/composer").mkdir(parents=True, exist_ok=True)

    if not settings.os_api_key:
        logger.warning("⚠️ No API key set — all endpoints are unauthenticated. Set OS_API_KEY for production use.")
        if settings.os_auth_required:
            raise RuntimeError("OS_AUTH_REQUIRED=true but OS_API_KEY is not set")

    lifecycle = ModelLifecycleManager(backend_router)
    lifecycle.start()
    logger.info(
        "Model lifecycle manager started (TTL=%ds, max_loaded=%d)",
        settings.os_model_ttl,
        settings.os_max_loaded_models,
    )

    cleanup_task = None
    if settings.tts_cache_enabled:

        async def _cleanup_loop():
            while True:
                await asyncio.sleep(30)
                await asyncio.get_running_loop().run_in_executor(None, tts_cache.evict_if_needed)

        cleanup_task = asyncio.create_task(_cleanup_loop(), name="tts-cache-cleanup")

    wyoming_task = None
    if settings.os_wyoming_enabled:
        from src.wyoming.server import start_wyoming_server

        wyoming_task = await start_wyoming_server(
            host=settings.os_wyoming_host,
            port=settings.os_wyoming_port,
            stt_router=backend_router,
            tts_router=tts_router,
        )
        logger.info("Wyoming protocol server enabled on %s:%d", settings.os_wyoming_host, settings.os_wyoming_port)

    if settings.stt_preload_models:
        for model_id in settings.stt_preload_models.split(","):
            model_id = model_id.strip()
            if model_id:
                try:
                    logger.info("Preloading STT model: %s", model_id)
                    backend_router.load_model(model_id)
                except Exception as exc:
                    logger.warning("Failed to preload STT model %s: %s", model_id, exc)

    if settings.tts_preload_models:
        for model_id in settings.tts_preload_models.split(","):
            model_id = model_id.strip()
            if model_id:
                try:
                    logger.info("Preloading TTS model: %s", model_id)
                    tts_router.load_model(model_id)
                except Exception as exc:
                    logger.warning("Failed to preload TTS model %s: %s", model_id, exc)

    yield

    if wyoming_task is not None:
        wyoming_task.cancel()
        try:
            await wyoming_task
        except asyncio.CancelledError:
            pass
    if cleanup_task is not None:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

    if batch_worker and batch_worker._tasks:
        logger.info("Cancelling %d in-flight batch jobs on shutdown", len(batch_worker._tasks))
        for task in list(batch_worker._tasks.values()):
            task.cancel()
        if batch_worker._tasks:
            await asyncio.gather(*list(batch_worker._tasks.values()), return_exceptions=True)

    await lifecycle.stop()


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(_request: Request, exc: StarletteHTTPException):
        detail = exc.detail
        if isinstance(detail, dict):
            message = str(detail.get("message") or detail.get("detail") or detail)
            code = str(detail.get("code") or "http_error")
        else:
            message = str(detail)
            code = "http_error"
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"message": message, "code": code}},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={"error": {"message": str(exc), "code": "validation_error"}},
        )


def _configure_middleware(app: FastAPI) -> None:
    app.add_middleware(SecurityMiddleware)

    cors_origins = [origin.strip() for origin in settings.os_cors_origins.split(",") if origin.strip()]
    allow_creds = "*" not in cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=allow_creds,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Open Speech",
        description="OpenAI-compatible speech server",
        version=get_runtime_version(),
        lifespan=lifespan,
    )

    _register_exception_handlers(app)
    _configure_middleware(app)

    app.include_router(
        stt_routes.create_router(
            get_settings=lambda: settings,
            get_backend_router=lambda: backend_router,
            get_history_manager=lambda: history_manager,
            get_diarizer_cls=lambda: PyannoteDiarizer,
            get_attach_speakers_fn=lambda: attach_text_to_speakers,
        )
    )
    app.include_router(streaming_routes.create_router())
    app.include_router(
        realtime_routes.create_router(
            get_settings=lambda: settings,
            get_tts_router=lambda: tts_router,
        )
    )
    app.include_router(
        batch_routes.create_router(
            get_settings=lambda: settings,
            get_batch_worker=lambda: batch_worker,
            get_batch_store=lambda: batch_store,
        )
    )
    app.include_router(
        model_routes.create_router(
            get_settings=lambda: settings,
            get_backend_router=lambda: backend_router,
            get_tts_router=lambda: tts_router,
            get_model_manager=lambda: model_manager,
            get_progress_service=lambda: progress_service,
        )
    )
    app.include_router(
        tts_routes.create_router(
            get_settings=lambda: settings,
            get_tts_router=lambda: tts_router,
            get_tts_cache=lambda: tts_cache,
            get_pronunciation_dict=lambda: pronunciation_dict,
            get_history_manager=lambda: history_manager,
            get_voice_library=lambda: voice_library,
        )
    )
    app.include_router(
        studio_routes.create_router(
            get_settings=lambda: settings,
            get_voice_library=lambda: voice_library,
            get_profile_manager=lambda: profile_manager,
            get_history_manager=lambda: history_manager,
            get_conversation_manager=lambda: conversation_manager,
            get_composer_manager=lambda: composer_manager,
        )
    )
    app.include_router(
        health_routes.create_router(
            get_runtime_version=get_runtime_version,
            get_backend_router=lambda: backend_router,
        )
    )

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.include_router(web_routes.create_router(static_dir=STATIC_DIR))
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    from src.ssl_utils import DEFAULT_CERT_FILE, DEFAULT_KEY_FILE, ensure_ssl_certs

    kwargs: dict = dict(host=settings.os_host, port=settings.os_port)

    if settings.os_ssl_enabled:
        cert = settings.os_ssl_certfile or DEFAULT_CERT_FILE
        key = settings.os_ssl_keyfile or DEFAULT_KEY_FILE
        ensure_ssl_certs(cert, key)
        kwargs["ssl_certfile"] = cert
        kwargs["ssl_keyfile"] = key
        logger.info("Listening on https://%s:%d", settings.os_host, settings.os_port)
    else:
        logger.info("Listening on http://%s:%d", settings.os_host, settings.os_port)

    uvicorn.run(app, **kwargs)
