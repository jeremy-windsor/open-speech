"""Open Speech — OpenAI-compatible STT server."""

from __future__ import annotations

import logging
import time
from typing import Annotated

from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.config import settings
from src.lifecycle import ModelLifecycleManager
from src.middleware import SecurityMiddleware, verify_ws_api_key
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
    suffix = get_suffix_from_content_type(file.content_type)
    audio_wav = convert_to_wav(audio_bytes, suffix=suffix)

    import asyncio
    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: backend_router.transcribe(
                audio=audio_wav,
                model=model,
                language=language,
                response_format=response_format,
                temperature=temperature,
                prompt=prompt,
            ),
        )
    except Exception as e:
        logger.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(e))

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

    import asyncio
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
    uvicorn.run(app, host=settings.stt_host, port=settings.stt_port)
