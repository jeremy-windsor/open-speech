"""Open Speech — OpenAI-compatible speech server (STT + TTS)."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse

from src.config import settings
from src.lifecycle import ModelLifecycleManager
from src.middleware import SecurityMiddleware, verify_ws_api_key
from src.tts.router import TTSRouter
from src.tts.models import TTSSpeechRequest, VoiceObject, VoiceListResponse, ModelLoadRequest, ModelUnloadRequest
from src.tts.pipeline import encode_audio, encode_audio_streaming, get_content_type
from src.tts.voices import OPENAI_VOICE_MAP
from src.formatters import format_transcription
from src.models import (
    HealthResponse,
    LoadedModelsResponse,
    ModelListResponse,
    ModelObject,
    PullResponse,
    TranscriptionResponse,
)
from src.router import router as backend_router
from src.streaming import streaming_endpoint
from src.utils.audio import convert_to_wav, get_suffix_from_content_type

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("open-speech")


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Open Speech v0.1.0 starting up")
    logger.info("Default model: %s", settings.stt_default_model)
    logger.info("Device: %s, Compute: %s", settings.stt_device, settings.stt_compute_type)

    # Preload models
    models_to_load = set()
    models_to_load.add(settings.stt_default_model)
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
                settings.stt_model_ttl, settings.stt_max_loaded_models)

    yield

    await lifecycle.stop()


app = FastAPI(
    title="Open Speech",
    description="OpenAI-compatible speech-to-text server",
    version="0.1.0",
    lifespan=lifespan,
)

# --- Security Middleware ---
app.add_middleware(SecurityMiddleware)

# --- CORS ---
cors_origins = [o.strip() for o in settings.stt_cors_origins.split(",") if o.strip()]
# Don't allow credentials with wildcard origins (browser security risk)
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
    model: Annotated[str, Form()] = settings.stt_default_model,
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[str, Form()] = "json",
    temperature: Annotated[float, Form()] = 0.0,
):
    """Transcribe audio to text (OpenAI-compatible)."""
    audio_bytes = await file.read()
    max_bytes = settings.stt_max_upload_mb * 1024 * 1024
    if len(audio_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Upload too large. Max: {settings.stt_max_upload_mb}MB")
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")
    # Detect format from content type or filename extension
    suffix = get_suffix_from_content_type(file.content_type)
    if suffix == ".ogg" and file.filename:
        # Content-type may be generic; try filename extension
        ext_suffix = _suffix_from_filename(file.filename)
        if ext_suffix:
            suffix = ext_suffix
    audio_wav = convert_to_wav(audio_bytes, suffix=suffix)

    # Always request verbose JSON from backend for segment data
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

    # Format output based on response_format
    if response_format in ("text", "srt", "vtt"):
        content, content_type = format_transcription(result, response_format)
        return PlainTextResponse(content, media_type=content_type)

    if result.get("raw_text"):
        return PlainTextResponse(result["text"])

    return JSONResponse(result)


@app.post("/v1/audio/translations")
async def translate(
    file: Annotated[UploadFile, File()],
    model: Annotated[str, Form()] = settings.stt_default_model,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[str, Form()] = "json",
    temperature: Annotated[float, Form()] = 0.0,
):
    """Translate audio to English text (OpenAI-compatible)."""
    audio_bytes = await file.read()
    max_bytes = settings.stt_max_upload_mb * 1024 * 1024
    if len(audio_bytes) > max_bytes:
        raise HTTPException(status_code=413, detail=f"Upload too large. Max: {settings.stt_max_upload_mb}MB")
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file")
    suffix = get_suffix_from_content_type(file.content_type)
    audio_wav = convert_to_wav(audio_bytes, suffix=suffix)

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
    # Always include the default model even if not loaded
    loaded_ids = {m.model for m in loaded}
    if settings.stt_default_model not in loaded_ids:
        models.append(ModelObject(id=settings.stt_default_model))
    # Include TTS models with status
    if settings.tts_enabled:
        tts_loaded = tts_router.loaded_models()
        tts_loaded_ids = {m.model for m in tts_loaded}
        for m in tts_loaded:
            models.append(ModelObject(id=m.model, owned_by=f"open-speech/{m.backend}"))
        if settings.tts_default_model not in tts_loaded_ids:
            models.append(ModelObject(id=settings.tts_default_model, owned_by="open-speech/tts"))
    return ModelListResponse(data=models)


@app.get("/v1/models/{model:path}")
async def get_model(model: str):
    """Get model details."""
    return ModelObject(id=model)


# --- Management endpoints ---


@app.get("/api/ps")
async def list_loaded_models():
    """List currently loaded models."""
    models = backend_router.loaded_models()
    return LoadedModelsResponse(models=models)


@app.post("/api/ps/{model:path}")
async def load_model(model: str):
    """Load a model into memory."""
    try:
        backend_router.load_model(model)
    except Exception as e:
        logger.exception("Failed to load model %s", model)
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "loaded", "model": model}


@app.delete("/api/ps/{model:path}")
async def unload_model(model: str):
    """Unload a model from memory."""
    if model == settings.stt_default_model:
        raise HTTPException(status_code=409, detail="Cannot unload default model")
    if not backend_router.is_model_loaded(model):
        raise HTTPException(status_code=404, detail=f"Model {model} is not loaded")
    backend_router.unload_model(model)
    return {"status": "unloaded", "model": model}


@app.get("/api/models")
async def list_all_models():
    """List all models on disk (cached) with their loaded status."""
    models = backend_router.list_cached_models()
    return {"models": models}


@app.delete("/api/models/{model:path}")
async def delete_model(model: str):
    """Delete a model from disk (HuggingFace cache)."""
    if model == settings.stt_default_model:
        raise HTTPException(status_code=409, detail="Cannot delete default model")

    # Unload from memory first if loaded
    if backend_router.is_model_loaded(model):
        backend_router.unload_model(model)

    # Delete from cache
    if not backend_router.delete_cached_model(model):
        raise HTTPException(status_code=404, detail=f"Model {model} not found on disk")

    return {"status": "deleted", "model": model}


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
):
    """Real-time streaming transcription via WebSocket."""
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
    )


# --- TTS endpoints ---


@app.post("/v1/audio/speech")
async def synthesize_speech(
    request: TTSSpeechRequest,
    stream: bool = False,
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

    valid_formats = {"mp3", "opus", "aac", "flac", "wav", "pcm"}
    if request.response_format not in valid_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid response_format. Must be one of: {', '.join(sorted(valid_formats))}",
        )

    content_type = get_content_type(request.response_format)

    if stream:
        # Streaming response — audio starts before full generation
        def _generate():
            chunks = tts_router.synthesize(
                text=request.input,
                model=request.model,
                voice=request.voice,
                speed=request.speed,
            )
            yield from encode_audio_streaming(chunks, fmt=request.response_format, sample_rate=24000)

        return StreamingResponse(
            _generate(),
            media_type=content_type,
            headers={"Transfer-Encoding": "chunked"},
        )

    loop = asyncio.get_running_loop()

    try:
        audio_bytes = await loop.run_in_executor(
            None,
            lambda: encode_audio(
                tts_router.synthesize(
                    text=request.input,
                    model=request.model,
                    voice=request.voice,
                    speed=request.speed,
                ),
                fmt=request.response_format,
                sample_rate=24000,
            ),
        )
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

    model_id = request.model if request else settings.tts_default_model
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

    model_id = request.model if request else settings.tts_default_model
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
    if settings.tts_default_model not in loaded_ids:
        models.append({
            "model": settings.tts_default_model,
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

    kwargs: dict = dict(host=settings.stt_host, port=settings.stt_port)

    if settings.stt_ssl_enabled:
        cert = settings.stt_ssl_certfile or DEFAULT_CERT_FILE
        key = settings.stt_ssl_keyfile or DEFAULT_KEY_FILE
        ensure_ssl_certs(cert, key)
        kwargs["ssl_certfile"] = cert
        kwargs["ssl_keyfile"] = key
        logger.info("Listening on https://%s:%d", settings.stt_host, settings.stt_port)
    else:
        logger.info("Listening on http://%s:%d", settings.stt_host, settings.stt_port)

    uvicorn.run(app, **kwargs)
