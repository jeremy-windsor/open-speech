# OpenAI-Compatible STT Server — Project Plan

## Problem Statement

The self-hosted STT landscape is fragmented:
- **faster-whisper-server** — Works, OpenAI-compatible, but infrequently updated. Limited to Whisper models only.
- **Speaches** — Tries to do STT + TTS, buggy, poor voice handling.
- **whisper-asr-webservice** — Swagger UI, decent, but not OpenAI-compatible endpoints.
- **NVIDIA NeMo** — Enterprise-grade models (Canary, Parakeet) but no simple Docker deployment.

**The gap**: No clean, maintained Docker container that provides OpenAI-compatible STT endpoints with pluggable backends (Whisper, NVIDIA, future models).

Kokoro solved this for TTS — FastAPI, OpenAI `/v1/audio/speech`, clean Docker image. We need the STT equivalent.

## Goals

1. OpenAI-compatible API (`/v1/audio/transcriptions`, `/v1/audio/translations`)
2. Pluggable STT backends behind a common interface
3. GPU-accelerated Docker container
4. Model hot-swap without container restart
5. Minimal dependencies, fast startup
6. Drop-in replacement for faster-whisper-server (or OpenAI's API)

## API Endpoints

### Core (OpenAI-compatible)
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/audio/transcriptions` | Transcribe audio → text |
| `POST` | `/v1/audio/translations` | Translate audio → English text |
| `GET` | `/v1/models` | List available models |
| `GET` | `/v1/models/{model}` | Model details |

### Management (Ollama-style)
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/ps` | List loaded models |
| `POST` | `/api/ps/{model}` | Load model into memory |
| `DELETE` | `/api/ps/{model}` | Unload model from memory |
| `POST` | `/api/pull/{model}` | Download model |
| `GET` | `/health` | Health check |

### Optional (future)
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/audio/transcriptions/stream` | Streaming transcription (WebSocket) |
| `GET` | `/docs` | Swagger UI |

## Backend Architecture

```
┌─────────────────────────────────────────┐
│           FastAPI Application            │
│                                         │
│  /v1/audio/transcriptions               │
│  /v1/audio/translations                 │
│  /v1/models                             │
│  /api/ps, /api/pull                     │
└──────────────┬──────────────────────────┘
               │
       ┌───────▼───────┐
       │ Backend Router │
       │ (model → impl)│
       └───────┬───────┘
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼───┐ ┌───▼───┐ ┌───▼───┐
│Faster │ │NeMo   │ │Future │
│Whisper│ │Canary │ │Models │
│       │ │Parakeet│ │       │
└───────┘ └───────┘ └───────┘
```

### Backend Interface

Each backend implements:
```python
class STTBackend(Protocol):
    name: str
    
    def load_model(self, model_id: str) -> None: ...
    def unload_model(self, model_id: str) -> None: ...
    def list_models(self) -> list[ModelInfo]: ...
    def loaded_models(self) -> list[str]: ...
    def transcribe(self, audio: bytes, model: str, 
                   language: str = None, 
                   response_format: str = "json",
                   temperature: float = 0.0) -> TranscriptionResult: ...
    def translate(self, audio: bytes, model: str) -> TranslationResult: ...
```

### Backend #1: faster-whisper (ctranslate2)
- Models: All Whisper variants (tiny → large-v3, turbo, distil)
- Source: HuggingFace ctranslate2 repos
- GPU: CUDA via ctranslate2
- Priority: **Phase 1** — this is what we run today

### Backend #2: NVIDIA NeMo
- Models: Canary Qwen 2.5B, Parakeet TDT 1.1B
- Source: NVIDIA NGC / HuggingFace
- GPU: CUDA via NeMo toolkit
- Priority: **Phase 2** — requires NeMo dependencies

### Backend #3: Moonshine (edge)
- Models: Moonshine Tiny, Base
- Source: HuggingFace
- GPU/CPU: ONNX runtime
- Priority: **Phase 3** — nice to have for CPU-only deployments

## Docker Architecture

```dockerfile
# Base: NVIDIA CUDA runtime
FROM nvidia/cuda:12.4-runtime-ubuntu22.04

# Core deps: Python, FastAPI, ffmpeg (audio conversion)
# Backend deps: faster-whisper, (optional) nemo_toolkit

# Env vars for config:
#   STT_DEFAULT_MODEL=deepdml/faster-whisper-large-v3-turbo-ct2
#   STT_DEFAULT_BACKEND=faster-whisper
#   STT_DEVICE=cuda (or cpu)
#   STT_COMPUTE_TYPE=float16
#   STT_MODEL_DIR=/models (volume mount for persistence)
```

### Image variants:
- `stt-server:latest` — faster-whisper only (~2GB image)
- `stt-server:nemo` — faster-whisper + NeMo (~8GB image)
- `stt-server:cpu` — CPU-only, no CUDA (~1GB image)

## File Structure

```
openai-stt-server/
├── Dockerfile
├── Dockerfile.nemo
├── Dockerfile.cpu
├── docker-compose.yml
├── pyproject.toml
├── README.md
├── src/
│   ├── main.py              # FastAPI app, endpoint definitions
│   ├── config.py             # Environment-based config
│   ├── models.py             # Pydantic models (request/response)
│   ├── router.py             # Backend router (model → implementation)
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── base.py           # STTBackend protocol
│   │   ├── faster_whisper.py # ctranslate2 backend
│   │   ├── nemo.py           # NeMo backend (Phase 2)
│   │   └── moonshine.py      # Moonshine backend (Phase 3)
│   └── utils/
│       ├── audio.py          # ffmpeg audio conversion
│       └── download.py       # HuggingFace model downloader
└── tests/
    ├── test_api.py
    ├── test_backends.py
    └── fixtures/
        └── test_audio.ogg
```

## Phases

### Phase 1 — MVP (faster-whisper parity)
**Goal**: Drop-in replacement for faster-whisper-server with cleaner code.

- [ ] FastAPI app with OpenAI-compatible endpoints
- [ ] faster-whisper backend via ctranslate2
- [ ] Model management (load/unload/list/pull)
- [ ] Audio format handling (ffmpeg conversion)
- [ ] GPU support (CUDA)
- [ ] Docker image + compose
- [ ] Health endpoint
- [ ] Swagger docs
- [ ] Basic tests

**Success criteria**: Can replace faster-whisper-server on 192.0.2.24 with identical API behavior.

### Phase 2 — NVIDIA Models
**Goal**: Add Canary and Parakeet as selectable backends.

- [ ] NeMo backend adapter
- [ ] Canary Qwen 2.5B support
- [ ] Parakeet TDT support
- [ ] Model-to-backend auto-routing (whisper models → faster-whisper, NeMo models → NeMo)
- [ ] Separate Docker image with NeMo deps
- [ ] Benchmark suite (WER comparison across backends)

### Phase 3 — Polish & Extras
**Goal**: Production hardening and nice-to-haves.

- [ ] Streaming transcription (WebSocket)
- [ ] Moonshine backend for CPU
- [ ] Batch inference support
- [ ] Prometheus metrics
- [ ] Web UI (simple upload → transcribe page)
- [ ] Rate limiting
- [ ] API key auth (optional)
- [ ] PyPI package

## Decisions

1. **Project name**: `open-speech` — repo at `will-assistant/open-speech`
2. **Public repo**: Open source from day one, MIT license
3. **NeMo**: Deferred — too heavy for Phase 1. Focus on faster-whisper, add lightweight backends later.
4. **Model storage**: `~/.cache/huggingface/hub` (standard HF cache dir) — volume mount in Docker for persistence
5. **Builder**: Forge agent handles Phase 1 implementation
6. **Default port**: 8100 (configurable via env var)
7. **Python**: 3.11+
8. **Future**: Cloud STT proxy (ElevenLabs, Cartesia, Deepgram) behind same endpoint — local-first with cloud fallback

## Tech Stack

- **Python 3.11+** — FastAPI, uvicorn
- **faster-whisper** — ctranslate2 inference
- **ffmpeg** — Audio format conversion
- **huggingface_hub** — Model downloads
- **Docker** — NVIDIA CUDA base image
- **pytest** — Testing

## References

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — ctranslate2 Whisper
- [faster-whisper-server](https://github.com/fedirz/faster-whisper-server) — Current server we use
- [kokoro-fastapi](https://github.com/remsky/kokoro-fastapi) — Inspiration for clean TTS server
- [OpenAI Audio API](https://platform.openai.com/docs/api-reference/audio) — API spec to implement
- [NVIDIA Canary](https://huggingface.co/nvidia/canary-qwen-2.5b) — Best WER model
- [NVIDIA Parakeet](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — Fastest streaming model
