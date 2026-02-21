# Phase 3 — Unified Model Architecture

## Vision

Open Speech is a **framework**, not a monolith. One container, any provider, any model. The image ships with the engine — models and providers are runtime choices controlled by configuration.

Think of it like a browser. Chrome doesn't ship with every website baked in. It ships with the rendering engine and you point it at what you want.

---

## Design Principles

1. **One image** — `jwindsor1/open-speech:latest` (~2.5–4GB). No CPU/GPU split. Both runtimes included.
2. **Zero models baked in** — Every model downloads at first boot into persistent volumes.
3. **Config drives everything** — `docker-compose.yml` declares what models to use. Change config, restart, done.
4. **Provider = backend code** — Installed in the image but dormant until a model needs it.
5. **Model = weights** — Downloaded at runtime, cached in volumes, never in the image.

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Open Speech                         │
│                                                       │
│  ┌─────────────┐    ┌──────────────────────────────┐ │
│  │  Unified     │    │  Provider Registry            │ │
│  │  Model       │───▶│                               │ │
│  │  Manager     │    │  STT:                         │ │
│  │              │    │    faster-whisper (installed)  │ │
│  │  • load      │    │                               │ │
│  │  • unload    │    │  TTS:                         │ │
│  │  • lifecycle │    │    kokoro (installed)          │ │
│  │  • eviction  │    │    piper (installed)           │ │
│  └──────┬───────┘    │    qwen3 (installed)           │ │
│         │            │                               │ │
│         ▼            │                               │ │
│  ┌─────────────┐    └──────────────────────────────┘ │
│  │  Model       │    └──────────────────────────────┘ │
│  │  Cache       │                                     │
│  │  (volume)    │    Providers are code. Always there. │
│  │              │    Models are weights. Downloaded    │
│  │  /models/    │    on demand into the volume.       │
│  │    stt/      │                                     │
│  │    tts/      │                                     │
│  └─────────────┘                                     │
└──────────────────────────────────────────────────────┘
```

---

## Configuration

### Environment Variables

```bash
# ── Server ──────────────────────────────────────────
OS_PORT=8100                    # Listen port
OS_HOST=0.0.0.0                 # Bind address
OS_API_KEY=                     # API auth (empty = disabled)
OS_CORS_ORIGINS=*               # CORS origins
OS_TRUST_PROXY=false            # X-Forwarded-For trust
OS_MAX_UPLOAD_MB=100            # Upload limit
OS_RATE_LIMIT=0                 # Requests/min/IP (0 = off)
OS_RATE_LIMIT_BURST=0           # Burst allowance

# ── SSL ─────────────────────────────────────────────
OS_SSL_ENABLED=true             # HTTPS on/off
OS_SSL_CERTFILE=                # Custom cert path (auto-gen if empty)
OS_SSL_KEYFILE=                 # Custom key path (auto-gen if empty)

# ── Speech-to-Text ──────────────────────────────────
STT_MODEL=Systran/faster-whisper-base       # Default STT model
STT_DEVICE=cpu                              # cpu or cuda
STT_COMPUTE_TYPE=int8                       # float16, int8, int8_float16

# ── Text-to-Speech ──────────────────────────────────
TTS_MODEL=kokoro                            # Default TTS model
TTS_DEVICE=cpu                              # cpu or cuda
TTS_VOICE=af_heart                          # Default voice
TTS_SPEED=1.0                               # Default speed

# ── Model Lifecycle ─────────────────────────────────
OS_MODEL_TTL=300                # Seconds idle before auto-unload (0 = never)
OS_MAX_LOADED_MODELS=0          # Max in RAM (0 = unlimited), LRU eviction

# ── Streaming ───────────────────────────────────────
OS_STREAM_CHUNK_MS=2000         # WebSocket STT chunk size
OS_STREAM_VAD_THRESHOLD=0.5     # VAD confidence threshold
OS_STREAM_ENDPOINTING_MS=300    # Silence before finalizing
OS_STREAM_MAX_CONNECTIONS=10    # Max concurrent WebSocket streams
```

### Backwards Compatibility

Old `STT_PORT`, `STT_HOST`, `STT_API_KEY` etc. still work as aliases. If both `OS_PORT` and `STT_PORT` are set, `OS_` wins. Deprecation warning logged on startup if old names detected.

---

## Docker Compose Examples

### Minimal (Raspberry Pi, low-RAM)

```yaml
services:
  open-speech:
    image: jwindsor1/open-speech:latest
    ports: ["8100:8100"]
    environment:
      - STT_MODEL=Systran/faster-whisper-tiny
      - STT_DEVICE=cpu
      - TTS_MODEL=piper/en_US-lessac-medium
      - TTS_DEVICE=cpu
    volumes:
      - models:/root/.cache/open-speech
```
**First boot download: ~110MB. RAM: ~500MB.**

### Balanced (Home server, CPU)

```yaml
services:
  open-speech:
    image: jwindsor1/open-speech:latest
    ports: ["8100:8100"]
    environment:
      - STT_MODEL=Systran/faster-whisper-base
      - STT_DEVICE=cpu
      - STT_COMPUTE_TYPE=int8
      - TTS_MODEL=kokoro
      - TTS_DEVICE=cpu
    volumes:
      - models:/root/.cache/open-speech
```
**First boot download: ~480MB. RAM: ~1GB.**

### GPU Workstation (Jeremy's Windows PC)

```yaml
services:
  open-speech:
    image: jwindsor1/open-speech:latest
    ports: ["8100:8100"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - STT_MODEL=deepdml/faster-whisper-large-v3-turbo-ct2
      - STT_DEVICE=cuda
      - STT_COMPUTE_TYPE=float16
      - TTS_MODEL=kokoro
      - TTS_DEVICE=cuda
    volumes:
      - models:/root/.cache/open-speech
```
**First boot download: ~1.8GB. RAM: ~3GB.**

### Voice Cloning (future)

```yaml
    environment:
      - STT_MODEL=deepdml/faster-whisper-large-v3-turbo-ct2
      - STT_DEVICE=cuda
      - TTS_MODEL=qwen3-tts-0.6b
      - TTS_DEVICE=cuda
      - TTS_VOICE=default
```
**Same image. Different model. Voice cloning enabled.**

### Split Device (CPU transcription, GPU synthesis)

```yaml
    environment:
      - STT_MODEL=Systran/faster-whisper-base
      - STT_DEVICE=cpu
      - TTS_MODEL=qwen3-tts-0.6b
      - TTS_DEVICE=cuda
```
**STT stays cheap on CPU. TTS gets GPU power for quality.**

---

## Provider Registry

### What Ships in the Image

Every provider's **code** is installed. No models, just the Python packages that know how to load and run them.

| Provider | Package | Supports | Size in image |
|----------|---------|----------|---------------|
| **faster-whisper** | `faster-whisper`, `ctranslate2` | STT | ~50MB |
| **kokoro** | `kokoro` | TTS | ~15MB |
| **piper** | `piper-tts` | TTS | ~10MB |
| **qwen3** | `transformers`, `accelerate` | TTS | ~200MB (shared with other HF models) |

Total provider overhead: ~330MB on top of PyTorch base.

### How Routing Works

The model name determines the provider:

| Model pattern | Provider | Type |
|---------------|----------|------|
| `Systran/faster-whisper-*`, `deepdml/faster-whisper-*` | faster-whisper | STT |
| `kokoro` | kokoro | TTS |
| `piper/*` | piper | TTS |
| `qwen3-tts-*` | qwen3 | TTS |

No config needed. Name the model, Open Speech knows what provider to use.

### Model Discovery

Each provider implements a `list_available()` method:

```python
class TTSBackend(Protocol):
    def list_available(self) -> list[ModelInfo]:
        """Return models this provider can serve, with sizes and capabilities."""
        ...
```

This powers the web UI model browser — shows what's available, what's downloaded, what's loaded.

---

## Unified Model Manager

One system manages all models regardless of type (STT/TTS):

```python
class ModelManager:
    """Unified model lifecycle for STT and TTS."""

    def load(self, model_id: str, device: str = None) -> None:
        """Download (if needed) and load model into memory."""
        provider = self.resolve_provider(model_id)
        if not self.is_downloaded(model_id):
            provider.download(model_id)    # → volume
        provider.load(model_id, device)    # → RAM/VRAM

    def unload(self, model_id: str) -> None:
        """Remove model from memory. Keeps on disk."""

    def evict_lru(self) -> None:
        """Unload least recently used non-default model."""

    def status(self) -> list[LoadedModel]:
        """All loaded models with type, device, RAM, last_used."""
```

### Lifecycle

```
Available (provider knows about it, not downloaded)
    │
    ▼  download triggered by config or API
Downloaded (weights on disk in volume, not in RAM)
    │
    ▼  load triggered by config, API, or first request
Loaded (in RAM/VRAM, serving requests)
    │
    ▼  TTL expiry or manual unload or LRU eviction
Downloaded (back to disk only)
```

### Unified API

Old endpoints kept for compatibility. New unified endpoints added:

| Endpoint | Description |
|----------|-------------|
| `GET /api/models` | All models: available, downloaded, loaded (unified) |
| `POST /api/models/{id}/load` | Download + load a model |
| `DELETE /api/models/{id}` | Unload from RAM |
| `DELETE /api/models/{id}/cache` | Delete from disk |
| `GET /api/models/{id}/status` | Single model status |

Old `/api/ps`, `/v1/audio/models/*` still work, mapped to unified manager.

---

## Web UI — Model Browser

New tab or section in Models tab:

```
┌─────────────────────────────────────────────────┐
│  MODELS                                          │
│                                                   │
│  ┌─ Loaded ─────────────────────────────────────┐│
│  │ faster-whisper-large-v3-turbo  STT  cuda  ▋▋ ││
│  │ kokoro                         TTS  cuda  ▋  ││
│  └──────────────────────────────────────────────┘│
│                                                   │
│  ┌─ Available ──────────────────────────────────┐│
│  │                                               ││
│  │  STT Models                                   ││
│  │  ┌─────────────────────────────────────────┐ ││
│  │  │ faster-whisper-tiny     75MB   [Load]   │ ││
│  │  │ faster-whisper-base    150MB   [Load]   │ ││
│  │  │ faster-whisper-small   500MB   [Load]   │ ││
│  │  │ faster-whisper-large   1.5GB   [Load]   │ ││
│  │  └─────────────────────────────────────────┘ ││
│  │                                               ││
│  │  TTS Models                                   ││
│  │  ┌─────────────────────────────────────────┐ ││
│  │  │ kokoro                 330MB   [Load]   │ ││
│  │  │ piper/en_US-lessac      35MB   [Load]   │ ││
│  │  │ qwen3-tts-0.6b        1.2GB   [Load]   │ ││
│  │  │ qwen3-tts-1.7b        3.4GB   [Load]   │ ││
│  │  └─────────────────────────────────────────┘ ││
│  └──────────────────────────────────────────────┘│
└─────────────────────────────────────────────────┘
```

Download progress shown inline. Load button becomes Unload after loading. Device selector per model if GPU available.

---

## Dockerfile (Single)

```dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Python + system deps
RUN apt-get update && apt-get install -y python3.12 python3-pip openssl ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Install all providers (code only, no models)
COPY pyproject.toml .
RUN pip install .[all]
# Installs: faster-whisper, kokoro, piper-tts,
#           transformers, accelerate

# App code
COPY src/ src/

# No models downloaded here. Zero. Nada.
# Models download at runtime based on STT_MODEL and TTS_MODEL env vars.

EXPOSE 8100
VOLUME ["/root/.cache/open-speech"]
CMD ["python", "-m", "src.main"]
```

**Image size breakdown:**
- CUDA base: ~1.8GB
- Python + system: ~200MB
- PyTorch (CPU+CUDA): ~500MB (shared libs with CUDA base)
- All provider packages: ~330MB
- App code: ~1MB
- **Total: ~2.8GB**

CPU-only machines: PyTorch detects no CUDA and falls back automatically. Same image works.

---

## Migration Path

### Phase 3a — Foundation (do first)
1. Rename env vars (`STT_PORT` → `OS_PORT` with backwards compat aliases)
2. Unified `ModelManager` class
3. Single `STT_MODEL` / `TTS_MODEL` config (replaces `_PRELOAD_MODELS`, `_DEFAULT_MODEL`)
4. Unified `/api/models` endpoints
5. Single Dockerfile
6. Update web UI models tab
7. Update all docker-compose files
8. Update README

### Phase 3b — New Providers (after foundation)
1. Piper TTS backend
2. Qwen3-TTS backend
3. Fish Speech backend
4. Model browser in web UI (available/downloaded/loaded)

### Phase 3c — Polish
1. Download progress API + UI
2. Model size estimates in browser
3. Voice presets config file
4. Replace Speaches + Kokoro-FastAPI in production
5. README badges, screenshots, CHANGELOG

---

## What This Means for Users

**Before (today):**
- Pick CPU or GPU image
- Hope the defaults work
- Manually edit env vars you don't understand
- Two separate model management systems

**After (Phase 3):**
- One image: `docker pull jwindsor1/open-speech:latest`
- Set `STT_MODEL` and `TTS_MODEL` in compose
- `docker compose up -d`
- Models download automatically
- Swap models by changing one line and restarting
- Browse and load new models from the web UI
- Works on CPU, GPU, or mixed — same image

---

## Open Questions

1. **Model cache location** — `/root/.cache/open-speech` (unified) or keep HuggingFace's default `~/.cache/huggingface`? Unified is cleaner but means re-downloading if user already has HF models cached.

2. **Version pinning** — Should `STT_MODEL=faster-whisper-base` resolve to a specific version, or always pull latest? Suggest: pin to specific HF revision by default, allow override.

3. **Multi-model loading** — Allow `STT_MODEL=model1,model2` for preloading multiple? Or keep it single and use the web UI / API for extras?

4. **Community model registry** — Long-term, a curated list of tested models with compatibility badges? Like Docker Hub but for speech models.
