"""Open Speech — OpenAI-compatible speech server (STT + TTS)."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import yaml

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

from src.config import settings
from src.lifecycle import ModelLifecycleManager
from src.middleware import SecurityMiddleware, verify_ws_api_key, verify_ws_origin
from src.model_manager import ModelManager, ModelState
from src.tts.router import TTSRouter
from src.tts.models import TTSSpeechRequest, VoiceObject, VoiceListResponse, ModelLoadRequest, ModelUnloadRequest
from src.tts.pipeline import encode_audio, encode_audio_streaming, get_content_type
from src.cache.tts_cache import TTSCache
from src.audio.preprocessing import preprocess_stt_audio
from src.audio.postprocessing import process_tts_chunks
from src.diarization.pyannote_diarizer import PyannoteDiarizer, attach_text_to_speakers
from src.pronunciation.dictionary import PronunciationDictionary, parse_ssml
from src.formatters import format_transcription
from src.models import (
    HealthResponse,
    LoadedModelsResponse,
    ModelListResponse,
    ModelObject,
    PullResponse,
)
from src.router import router as backend_router
from src.streaming import streaming_endpoint
from src.utils.audio import convert_to_wav, get_suffix_from_content_type

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("open-speech")

__version__ = "0.5.1"


def _suffix_from_filename(filename: str) -> str | None:
    """Extract audio suffix from filename."""
    ext_map = {
        ".wav": ".wav", ".mp3": ".mp3", ".ogg": ".ogg",
        ".flac": ".flac", ".m4a": ".m4a", ".webm": ".webm",
        ".opus": ".ogg", ".aac": ".m4a",
    }
    for ext, suffix in ext_map.items():
        if filename.lower().endswith(ext):
            return suffix
    return None


tts_router = TTSRouter(device=settings.tts_effective_device)
model_manager = ModelManager(stt_router=backend_router, tts_router=tts_router)
tts_cache = TTSCache(settings.tts_cache_dir, settings.tts_cache_max_mb, settings.tts_cache_enabled)
pronunciation_dict = PronunciationDictionary(settings.tts_pronunciation_dict or None)


def _tts_backend_name(model_id: str) -> str:
    backend = tts_router.get_backend(model_id)
    return getattr(backend, "name", model_id)


def _tts_capabilities(model_id: str) -> dict:
    backend = tts_router.get_backend(model_id)
    caps = getattr(backend, "capabilities", {})
    return dict(caps)


def _validate_tts_feature_support(*, model_id: str, voice_design: str | None = None, reference_audio: bytes | str | None = None) -> str | None:
    backend_name = _tts_backend_name(model_id)
    caps = _tts_capabilities(model_id)
    if voice_design and not caps.get("voice_design", False):
        if backend_name == "kokoro":
            return "voice_design is not supported by the kokoro backend. Use qwen3 for instruction-controlled speech."
        return f"voice_design is not supported by the {backend_name} backend."

    if reference_audio is not None and not caps.get("voice_clone", False):
        if backend_name == "piper":
            return "Voice cloning is not supported by the piper backend. Use qwen3 or fish-speech."
        return f"Voice cloning is not supported by the {backend_name} backend."
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Open Speech v%s starting up", __version__)
    logger.info("Default STT model: %s", settings.stt_model)
    logger.info("Default TTS model: %s", settings.tts_model)
    logger.info("Device: %s, Compute: %s", settings.stt_device, settings.stt_compute_type)

    if not settings.os_api_key:
        logger.warning("⚠️ No API key set — all endpoints are unauthenticated. Set OS_API_KEY for production use.")
        if settings.os_auth_required:
            raise RuntimeError("OS_AUTH_REQUIRED=true but OS_API_KEY is not set")

    # Preload STT models
    models_to_load = set()
    models_to_load.add(settings.stt_model)
    if settings.stt_preload_models:
        for m in settings.stt_preload_models.split(","):
            m = m.strip()
            if m:
                models_to_load.add(m)

    for model_id in models_to_load:
        try:
            logger.info("Preloading model: %s", model_id)
            start = time.time()
            backend_router.load_model(model_id)
            elapsed = time.time() - start
            logger.info("Model %s loaded in %.1fs", model_id, elapsed)
        except Exception as e:
            logger.error("Failed to preload model %s: %s", model_id, e)

    # Preload TTS models
    if settings.tts_enabled and settings.tts_preload_models:
        for m in settings.tts_preload_models.split(","):
            m = m.strip()
            if m:
                try:
                    logger.info("Preloading TTS model: %s", m)
                    start = time.time()
                    tts_router.load_model(m)
                    elapsed = time.time() - start
                    logger.info("TTS model %s loaded in %.1fs", m, elapsed)
                except Exception as e:
                    logger.error("Failed to preload TTS model %s: %s", m, e)

    # Start lifecycle manager
    lifecycle = ModelLifecycleManager(backend_router)
    lifecycle.start()
    logger.info("Model lifecycle manager started (TTL=%ds, max_loaded=%d)",
                settings.os_model_ttl, settings.os_max_loaded_models)

    # Start background cache cleanup
    cleanup_task = None
    if settings.tts_cache_enabled:
        async def _cleanup_loop():
            while True:
                await asyncio.sleep(30)
                await asyncio.get_running_loop().run_in_executor(None, tts_cache.evict_if_needed)
        cleanup_task = asyncio.create_task(_cleanup_loop(), name="tts-cache-cleanup")

    # Start Wyoming server if enabled
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
    await lifecycle.stop()


app = FastAPI(
    title="Open Speech",
    description="OpenAI-compatible speech-to-text server",
    version=__version__,
    lifespan=lifespan,
)


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

# --- Security Middleware ---
app.add_middleware(SecurityMiddleware)

# --- CORS ---
cors_origins = [o.strip() for o in settings.os_cors_origins.split(",") if o.strip()]
allow_creds = "*" not in cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_creds,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- OpenAI-compatible endpoints ---


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: Annotated[UploadFile, File()],
    model: Annotated[str, Form()] = settings.stt_model,
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[str, Form()] = "json",
    temperature: Annotated[float, Form()] = 0.0,
    diarize: bool = False,
):
    """Transcribe audio to text (OpenAI-compatible)."""
    audio_bytes = await file.read()
    max_bytes = settings.os_max_upload_mb * 1024 * 1024
    if len(audio_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Upload too large. Max: {settings.os_max_upload_mb}MB")
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")
    suffix = get_suffix_from_content_type(file.content_type)
    if suffix == ".ogg" and file.filename:
        ext_suffix = _suffix_from_filename(file.filename)
        if ext_suffix:
            suffix = ext_suffix
    if diarize and not settings.stt_diarize_enabled:
        raise HTTPException(status_code=400, detail="Diarization is disabled. Set STT_DIARIZE_ENABLED=true")

    audio_wav = convert_to_wav(audio_bytes, suffix=suffix)
    audio_wav = preprocess_stt_audio(
        audio_wav,
        noise_reduce=settings.stt_noise_reduce,
        normalize=settings.stt_normalize,
    )

    backend_format = "verbose_json" if response_format in ("srt", "vtt", "json", "verbose_json") else response_format
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: backend_router.transcribe(
                audio=audio_wav,
                model=model,
                language=language,
                response_format=backend_format,
                temperature=temperature,
                prompt=prompt,
            ),
        )
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(e))

    if diarize:
        try:
            diarizer = PyannoteDiarizer()
            dsegs = await loop.run_in_executor(None, lambda: diarizer.diarize(audio_wav))
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Diarization failed: {e}")
        text = result.get("text", "")
        return JSONResponse({"text": text, "segments": attach_text_to_speakers(text, dsegs)})

    if response_format in ("text", "srt", "vtt"):
        content, content_type = format_transcription(result, response_format)
        return PlainTextResponse(content, media_type=content_type)

    if result.get("raw_text"):
        return PlainTextResponse(result["text"])

    return JSONResponse(result)


@app.post("/v1/audio/translations")
async def translate(
    file: Annotated[UploadFile, File()],
    model: Annotated[str, Form()] = settings.stt_model,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[str, Form()] = "json",
    temperature: Annotated[float, Form()] = 0.0,
):
    """Translate audio to English text (OpenAI-compatible)."""
    audio_bytes = await file.read()
    max_bytes = settings.os_max_upload_mb * 1024 * 1024
    if len(audio_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Upload too large. Max: {settings.os_max_upload_mb}MB")
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")
    suffix = get_suffix_from_content_type(file.content_type)
    audio_wav = convert_to_wav(audio_bytes, suffix=suffix)
    audio_wav = preprocess_stt_audio(
        audio_wav,
        noise_reduce=settings.stt_noise_reduce,
        normalize=settings.stt_normalize,
    )

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: backend_router.translate(
                audio=audio_wav,
                model=model,
                response_format=response_format,
                temperature=temperature,
                prompt=prompt,
            ),
        )
    except Exception as e:
        logger.exception("Translation failed")
        raise HTTPException(status_code=500, detail=str(e))

    if result.get("raw_text"):
        return PlainTextResponse(result["text"])

    return JSONResponse(result)


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    loaded = backend_router.loaded_models()
    models = [
        ModelObject(id=m.model, owned_by=f"open-speech/{m.backend}")
        for m in loaded
    ]
    loaded_ids = {m.model for m in loaded}
    if settings.stt_model not in loaded_ids:
        models.append(ModelObject(id=settings.stt_model))
    if settings.tts_enabled:
        tts_loaded = tts_router.loaded_models()
        tts_loaded_ids = {m.model for m in tts_loaded}
        for m in tts_loaded:
            models.append(ModelObject(id=m.model, owned_by=f"open-speech/{m.backend}"))
        if settings.tts_model not in tts_loaded_ids:
            models.append(ModelObject(id=settings.tts_model, owned_by="open-speech/tts"))
    return ModelListResponse(data=models)


@app.get("/v1/models/{model:path}")
async def get_model(model: str):
    """Get model details."""
    return ModelObject(id=model)


# --- Legacy management endpoints (kept for backwards compat) ---


@app.get("/api/ps")
async def list_loaded_models():
    """List currently loaded models."""
    models = backend_router.loaded_models()
    return LoadedModelsResponse(models=models)


@app.post("/api/ps/{model:path}")
async def load_model_legacy(model: str):
    """Load a model into memory (legacy endpoint)."""
    try:
        backend_router.load_model(model)
    except Exception as e:
        logger.exception("Failed to load model %s", model)
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "loaded", "model": model}


@app.delete("/api/ps/{model:path}")
async def unload_model_legacy(model: str):
    """Unload a model from memory (legacy endpoint)."""
    if model == settings.stt_model:
        raise HTTPException(status_code=409, detail="Cannot unload default model")
    if not backend_router.is_model_loaded(model):
        raise HTTPException(status_code=404, detail=f"Model {model} is not loaded")
    backend_router.unload_model(model)
    return {"status": "unloaded", "model": model}


# --- Unified model management endpoints (Phase 3a) ---


@app.get("/api/models")
async def list_all_models():
    """List all models (available + downloaded + loaded) from unified ModelManager."""
    models = model_manager.list_all()
    response_models = [m.to_dict() for m in models]
    for model in response_models:
        if model.get("type") == "tts":
            try:
                model["capabilities"] = _tts_capabilities(model["id"])
            except Exception:
                model["capabilities"] = {}
    return {"models": response_models}


@app.get("/api/tts/capabilities")
async def get_tts_capabilities(model: str | None = None):
    """Return capabilities of selected (or default) TTS backend."""
    if not settings.tts_enabled:
        raise HTTPException(status_code=404, detail="TTS is disabled")

    model_id = model or settings.tts_model
    backend_name = _tts_backend_name(model_id)
    return {"backend": backend_name, "capabilities": _tts_capabilities(model_id)}


@app.get("/api/models/{model_id:path}/status")
async def get_model_status(model_id: str):
    """Get status of a specific model."""
    info = model_manager.status(model_id)
    return info.to_dict()


_download_progress: dict[str, dict] = {}
_download_progress_lock = asyncio.Lock()
_model_operation_lock = asyncio.Lock()


@app.get("/api/models/{model_id:path}/progress")
async def get_model_progress(model_id: str):
    """Get download/load progress for a model."""
    async with _download_progress_lock:
        if model_id in _download_progress:
            return _download_progress[model_id]
    # Check if already loaded
    info = model_manager.status(model_id)
    if info.state == ModelState.LOADED:
        return {"status": "ready", "progress": 1.0}
    return {"status": "idle", "progress": 0.0}


@app.post("/api/models/{model_id:path}/load")
async def load_model_unified(model_id: str):
    """Load a model (download if needed)."""
    async with _model_operation_lock:
        async with _download_progress_lock:
            _download_progress[model_id] = {"status": "downloading", "progress": 0.5}
        try:
            info = model_manager.load(model_id)
            async with _download_progress_lock:
                _download_progress[model_id] = {"status": "ready", "progress": 1.0}
        except Exception as e:
            async with _download_progress_lock:
                _download_progress.pop(model_id, None)
            logger.exception("Failed to load model %s", model_id)
            raise HTTPException(status_code=500, detail=str(e))
    return info.to_dict()


@app.delete("/api/models/{model_id:path}")
async def unload_model_unified(model_id: str):
    """Unload a model from RAM."""
    info = model_manager.status(model_id)
    if info.state != ModelState.LOADED:
        raise HTTPException(status_code=404, detail=f"Model {model_id} is not loaded")
    if info.is_default:
        raise HTTPException(status_code=409, detail="Cannot unload default model")
    async with _model_operation_lock:
        model_manager.unload(model_id)
    return {"status": "unloaded", "model": model_id}


@app.post("/api/pull/{model:path}")
async def pull_model(model: str):
    """Download a model (triggers HuggingFace download via load)."""
    try:
        backend_router.load_model(model)
        backend_router.unload_model(model)
    except Exception as e:
        logger.exception("Failed to pull model %s", model)
        raise HTTPException(status_code=500, detail=str(e))
    return PullResponse(status="downloaded", model=model)


@app.get("/health")
async def health():
    """Health check."""
    loaded = backend_router.loaded_models()
    return HealthResponse(models_loaded=len(loaded))


# --- Streaming WebSocket ---


@app.websocket("/v1/audio/stream")
async def ws_stream(
    websocket: WebSocket,
    model: str | None = None,
    language: str | None = None,
    sample_rate: int = 16000,
    encoding: str = "pcm_s16le",
    interim_results: bool = True,
    endpointing: int = 300,
    vad: bool | None = None,
):
    """Real-time streaming transcription via WebSocket."""
    if not verify_ws_origin(websocket):
        await websocket.close(code=1008, reason="Origin not allowed")
        return
    if not verify_ws_api_key(websocket):
        await websocket.close(code=4001, reason="Invalid or missing API key")
        return
    await streaming_endpoint(
        websocket,
        model=model,
        language=language,
        sample_rate=sample_rate,
        encoding=encoding,
        interim_results=interim_results,
        endpointing=endpointing,
        vad=vad,
    )


# --- OpenAI Realtime API ---


@app.websocket("/v1/realtime")
async def ws_realtime(
    websocket: WebSocket,
    model: str | None = None,
):
    """OpenAI Realtime API compatible WebSocket endpoint (audio I/O only)."""
    if not settings.os_realtime_enabled:
        await websocket.close(code=4004, reason="Realtime API is disabled")
        return
    if not verify_ws_origin(websocket):
        await websocket.close(code=1008, reason="Origin not allowed")
        return
    if not verify_ws_api_key(websocket):
        await websocket.close(code=4001, reason="Invalid or missing API key")
        return
    from src.realtime.server import realtime_endpoint
    await realtime_endpoint(websocket, tts_router=tts_router, model=model or "")


# --- TTS endpoints ---


@app.post("/v1/audio/speech")
async def synthesize_speech(
    request: TTSSpeechRequest,
    stream: bool = False,
    cache: bool = True,
):
    """Synthesize speech from text (OpenAI-compatible)."""
    if not settings.tts_enabled:
        raise HTTPException(status_code=404, detail="TTS is disabled")

    if len(request.input) > settings.tts_max_input_length:
        raise HTTPException(
            status_code=400,
            detail=f"Input too long. Max: {settings.tts_max_input_length} characters",
        )

    if not request.input.strip():
        raise HTTPException(status_code=400, detail="Input text is empty")

    feature_error = _validate_tts_feature_support(
        model_id=request.model,
        voice_design=request.voice_design,
        reference_audio=request.reference_audio,
    )
    if feature_error:
        raise HTTPException(status_code=400, detail=feature_error)

    valid_formats = {"mp3", "opus", "aac", "flac", "wav", "pcm"}
    if request.response_format not in valid_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid response_format. Must be one of: {', '.join(sorted(valid_formats))}",
        )

    content_type = get_content_type(request.response_format)

    synth_input = request.input
    if request.input_type == "ssml":
        synth_input = parse_ssml(synth_input)
    synth_input = pronunciation_dict.apply(synth_input)

    # Build extended kwargs for backends that support voice_design/reference_audio
    has_extended = bool(request.voice_design or request.reference_audio)

    def _do_synthesize():
        if has_extended:
            backend = tts_router.get_backend(request.model)
            kwargs: dict = dict(text=synth_input, voice=request.voice, speed=request.speed, lang_code=request.language)
            import inspect
            sig = inspect.signature(backend.synthesize)
            if request.voice_design and "voice_design" in sig.parameters:
                kwargs["voice_design"] = request.voice_design
            if request.reference_audio and "reference_audio" in sig.parameters:
                try:
                    ref_bytes = base64.b64decode(request.reference_audio)
                except Exception:
                    ref_bytes = request.reference_audio.encode()
                kwargs["reference_audio"] = ref_bytes
            if request.clone_transcript and "clone_transcript" in sig.parameters:
                kwargs["clone_transcript"] = request.clone_transcript
            return backend.synthesize(**kwargs)
        return tts_router.synthesize(
            text=synth_input,
            model=request.model,
            voice=request.voice,
            speed=request.speed,
            lang_code=request.language,
        )

    if stream:
        def _generate():
            chunks = process_tts_chunks(
                _do_synthesize(),
                trim=settings.tts_trim_silence,
                normalize=settings.tts_normalize_output,
            )
            yield from encode_audio_streaming(chunks, fmt=request.response_format, sample_rate=24000)

        return StreamingResponse(
            _generate(),
            media_type=content_type,
            headers={"Transfer-Encoding": "chunked"},
        )

    loop = asyncio.get_running_loop()

    if cache and settings.tts_cache_enabled and not stream:
        cached = tts_cache.get(text=synth_input, voice=request.voice, speed=request.speed, fmt=request.response_format)
        if cached is not None:
            return StreamingResponse(
                iter([cached]),
                media_type=content_type,
                headers={"Content-Length": str(len(cached)), "X-Cache": "HIT"},
            )

    try:
        processed_chunks = process_tts_chunks(
            _do_synthesize(),
            trim=settings.tts_trim_silence,
            normalize=settings.tts_normalize_output,
        )
        audio_bytes = await loop.run_in_executor(
            None,
            lambda: encode_audio(
                processed_chunks,
                fmt=request.response_format,
                sample_rate=24000,
            ),
        )
        if cache and settings.tts_cache_enabled and not stream:
            await loop.run_in_executor(None, lambda: tts_cache.set(
                text=synth_input,
                voice=request.voice,
                speed=request.speed,
                fmt=request.response_format,
                audio=audio_bytes,
            ))
    except Exception as e:
        logger.exception("TTS synthesis failed")
        raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(
        iter([audio_bytes]),
        media_type=content_type,
        headers={"Content-Length": str(len(audio_bytes))},
    )


@app.post("/v1/audio/models/load")
async def load_tts_model(request: ModelLoadRequest | None = None):
    """Load a TTS model into memory."""
    if not settings.tts_enabled:
        raise HTTPException(status_code=404, detail="TTS is disabled")

    model_id = request.model if request else settings.tts_model
    try:
        tts_router.load_model(model_id)
    except Exception as e:
        logger.exception("Failed to load TTS model %s", model_id)
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "loaded", "model": model_id}


@app.post("/v1/audio/models/unload")
async def unload_tts_model(request: ModelUnloadRequest | None = None):
    """Unload a TTS model from memory."""
    if not settings.tts_enabled:
        raise HTTPException(status_code=404, detail="TTS is disabled")

    model_id = request.model if request else settings.tts_model
    if not tts_router.is_model_loaded(model_id):
        raise HTTPException(status_code=404, detail=f"TTS model {model_id} is not loaded")
    tts_router.unload_model(model_id)
    return {"status": "unloaded", "model": model_id}


@app.get("/v1/audio/models")
async def list_tts_models():
    """List TTS models and their status."""
    if not settings.tts_enabled:
        raise HTTPException(status_code=404, detail="TTS is disabled")

    loaded = tts_router.loaded_models()
    loaded_ids = {m.model for m in loaded}
    models = []
    for m in loaded:
        models.append({
            "model": m.model,
            "backend": m.backend,
            "device": m.device,
            "status": "loaded",
            "loaded_at": m.loaded_at,
            "last_used_at": m.last_used_at,
        })
    if settings.tts_model not in loaded_ids:
        models.append({
            "model": settings.tts_model,
            "backend": "kokoro",
            "status": "not_loaded",
        })
    return {"models": models}


@app.get("/v1/audio/voices")
async def list_voices():
    """List available TTS voices."""
    if not settings.tts_enabled:
        raise HTTPException(status_code=404, detail="TTS is disabled")

    voices = tts_router.list_voices()
    return VoiceListResponse(
        voices=[
            VoiceObject(id=v.id, name=v.name, language=v.language, gender=v.gender)
            for v in voices
        ]
    )


# --- Voice Presets ---

DEFAULT_VOICE_PRESETS = [
    {"name": "Will", "voice": "am_puck(1)+am_liam(1)+am_onyx(0.5)", "speed": 1.2, "description": "Dry wit genius blend — Puck + Liam + Onyx"},
    {"name": "Female", "voice": "af_jessica(1)+af_heart(1)", "speed": 1.2, "description": "Warm female blend — Jessica + Heart"},
    {"name": "British Butler", "voice": "bm_george", "speed": 0.9, "description": "Refined British male"},
]


def _load_voice_presets() -> list[dict]:
    """Load voice presets from config file or defaults."""
    config_path = os.environ.get("TTS_VOICES_CONFIG")
    if config_path and Path(config_path).exists():
        try:
            with open(config_path) as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and "presets" in data:
                return data["presets"]
            if isinstance(data, list):
                return data
        except Exception as e:
            logger.warning("Failed to load voice presets from %s: %s", config_path, e)
    return DEFAULT_VOICE_PRESETS


@app.get("/api/voice-presets")
async def get_voice_presets():
    """Return voice presets for the web UI."""
    return {"presets": _load_voice_presets()}


# --- Voice Cloning Endpoint ---


@app.post("/v1/audio/speech/clone")
async def clone_speech(
    input: Annotated[str, Form()],
    model: Annotated[str, Form()] = "qwen3-tts/1.7B-Base",
    reference_audio: Annotated[UploadFile, File()] = None,
    voice: Annotated[str, Form()] = "Ryan",
    speed: Annotated[float, Form()] = 1.0,
    response_format: Annotated[str, Form()] = "mp3",
    transcript: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
):
    """Synthesize speech with voice cloning via multipart upload."""
    if not settings.tts_enabled:
        raise HTTPException(status_code=404, detail="TTS is disabled")

    if not input.strip():
        raise HTTPException(status_code=400, detail="Input text is empty")

    ref_bytes = None
    if reference_audio:
        feature_error = _validate_tts_feature_support(model_id=model, reference_audio=b"provided")
        if feature_error:
            raise HTTPException(status_code=400, detail=feature_error)
        ref_bytes = await reference_audio.read()
        max_bytes = settings.os_max_upload_mb * 1024 * 1024
        if len(ref_bytes) > max_bytes:
            raise HTTPException(status_code=413, detail=f"Upload too large. Max: {settings.os_max_upload_mb}MB")
        if len(ref_bytes) == 0:
            raise HTTPException(status_code=400, detail="Reference audio is empty")

    content_type = get_content_type(response_format)
    loop = asyncio.get_running_loop()

    try:
        # Pass reference_audio as kwargs — backends that support it will use it
        def _synth():
            backend = tts_router.get_backend(model)
            synth_kwargs = dict(text=input, voice=voice, speed=speed, lang_code=language)
            # Pass reference_audio if backend supports it
            import inspect
            sig = inspect.signature(backend.synthesize)
            if "reference_audio" in sig.parameters:
                synth_kwargs["reference_audio"] = ref_bytes
            if transcript and "clone_transcript" in sig.parameters:
                synth_kwargs["clone_transcript"] = transcript
            return encode_audio(
                process_tts_chunks(
                    backend.synthesize(**synth_kwargs),
                    trim=settings.tts_trim_silence,
                    normalize=settings.tts_normalize_output,
                ),
                fmt=response_format,
                sample_rate=24000,
            )

        audio_bytes = await loop.run_in_executor(None, _synth)
    except Exception as e:
        logger.exception("Voice cloning synthesis failed")
        raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(
        iter([audio_bytes]),
        media_type=content_type,
        headers={"Content-Length": str(len(audio_bytes))},
    )


# --- Web UI ---

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/web", response_class=HTMLResponse)
async def web_ui():
    """Serve the web UI."""
    index = STATIC_DIR / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text())
    return HTMLResponse("<h1>Web UI not found</h1>", status_code=404)


if __name__ == "__main__":
    import uvicorn
    from src.ssl_utils import ensure_ssl_certs, DEFAULT_CERT_FILE, DEFAULT_KEY_FILE

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
