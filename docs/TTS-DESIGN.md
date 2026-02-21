# Open Speech — TTS Integration Design

> **Status:** Design Draft  
> **Date:** 2026-02-16  
> **Author:** Forge (automated design)  
> **Goal:** Add TTS to Open Speech, creating a unified voice server that replaces both Speaches (STT) and Kokoro-FastAPI (TTS) with a single container.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [API Design](#2-api-design)
3. [Voice System](#3-voice-system)
4. [Backend Abstraction](#4-backend-abstraction)
5. [Audio Pipeline](#5-audio-pipeline)
6. [Configuration](#6-configuration)
7. [Docker & Deployment](#7-docker--deployment)
8. [Dependencies & Compatibility](#8-dependencies--compatibility)
9. [Migration Path](#9-migration-path)
10. [Phase Plan](#10-phase-plan)
11. [Testing Strategy](#11-testing-strategy)

---

## 1. Architecture Overview

### Current State

Open Speech is an STT-only server built on FastAPI with a pluggable backend system:

```
src/
├── main.py           # FastAPI app, endpoints
├── config.py         # pydantic-settings (STT_* env vars)
├── router.py         # BackendRouter — routes to STT backends
├── models.py         # Pydantic request/response models
├── middleware.py      # Auth, rate limiting, upload limits
├── streaming.py       # WebSocket real-time STT
├── backends/
│   ├── base.py       # STTBackend Protocol
│   └── faster_whisper.py
└── utils/
    └── audio.py      # WAV conversion helpers
```

The `BackendRouter` pattern is already designed for extensibility — it maps model IDs to backend implementations. We'll mirror this pattern for TTS.

### Proposed Architecture

```
src/
├── main.py           # FastAPI app (add TTS endpoints)
├── config.py         # Unified settings (STT_* + TTS_* env vars)
├── router.py         # BackendRouter (STT — unchanged)
├── tts/
│   ├── __init__.py
│   ├── router.py     # TTSRouter — routes to TTS backends
│   ├── models.py     # TTS request/response models
│   ├── pipeline.py   # Audio encoding pipeline (wav→mp3/opus/etc)
│   ├── voices.py     # Voice registry, blending, presets
│   └── backends/
│       ├── base.py   # TTSBackend Protocol
│       ├── kokoro.py # Kokoro-82M backend
│       ├── piper.py  # Piper backend
│       └── qwen3.py  # Qwen3-TTS backend
├── backends/         # STT backends (unchanged)
│   ├── base.py
│   └── faster_whisper.py
├── middleware.py      # Shared security (unchanged)
├── streaming.py       # STT streaming (unchanged)
└── utils/
    └── audio.py      # Shared audio utils
```

Key principle: **STT and TTS are peers, not parent/child.** They share the FastAPI app, middleware, and config system but have independent backend registries and model lifecycles.

---

## 2. API Design

### OpenAI-Compatible Speech Endpoint

#### `POST /v1/audio/speech`

**Request body** (JSON):

```json
{
  "model": "kokoro",
  "input": "Hello world!",
  "voice": "af_heart",
  "response_format": "mp3",
  "speed": 1.0
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | yes | — | TTS model identifier (e.g. `kokoro`) |
| `input` | string | yes | — | Text to synthesize. Max 4096 chars. |
| `voice` | string | yes | — | Voice ID, Kokoro voice name, or blended combo (e.g. `af_bella+af_sky`) |
| `response_format` | string | no | `mp3` | Output format: `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm` |
| `speed` | float | no | `1.0` | Playback speed (0.25–4.0) |

**Response:** Raw audio bytes with appropriate `Content-Type` header.

| Format | Content-Type | Notes |
|--------|-------------|-------|
| `mp3` | `audio/mpeg` | Default, widely compatible |
| `opus` | `audio/opus` | Smallest, best for streaming |
| `aac` | `audio/aac` | Apple ecosystem |
| `flac` | `audio/flac` | Lossless |
| `wav` | `audio/wav` | Uncompressed PCM |
| `pcm` | `audio/pcm` | Raw 24kHz 16-bit LE mono |

**Streaming:** Support `Transfer-Encoding: chunked` for the same endpoint. Clients using the OpenAI SDK's `with_streaming_response` pattern should work transparently.

#### `GET /v1/audio/voices`

Non-OpenAI-standard but used by Kokoro-FastAPI and useful for discovery:

```json
{
  "voices": [
    {"id": "af_heart", "name": "Heart", "language": "en-us", "gender": "female"},
    {"id": "af_bella", "name": "Bella", "language": "en-us", "gender": "female"},
    {"id": "am_adam", "name": "Adam", "language": "en-us", "gender": "male"}
  ]
}
```

#### OpenAI Voice Mapping

Map OpenAI's standard voice names to Kokoro voices for drop-in compatibility:

```json
{
  "alloy": "af_heart",
  "echo": "am_adam",
  "fable": "bf_emma",
  "onyx": "am_michael",
  "nova": "af_nova",
  "shimmer": "af_bella"
}
```

This mapping should be configurable via a JSON file or env var.

#### Model Management

Extend existing `/api/ps` endpoints to include TTS models:

- `GET /api/ps` — list all loaded models (STT + TTS, with `type` field)
- `POST /api/ps/{model}` — load model (auto-detect STT vs TTS by model ID pattern)
- `DELETE /api/ps/{model}` — unload model

---

## 3. Voice System

### Voice Types

1. **Built-in voices** — Ship with the Kokoro model. ~50+ voices across languages. Each is a `.pt` tensor file (~350KB each).

2. **OpenAI aliases** — Map standard OpenAI voice names to built-in voices for compatibility.

3. **Blended voices** — Runtime-generated by mixing voice tensors with weights:
   - `af_bella+af_sky` → 50/50 mix
   - `af_bella(2)+af_sky(1)` → 67/33 mix
   - Weights are normalized to sum to 1.0

4. **Custom voices** — User-provided `.pt` files placed in the voices directory.

### Voice Registry

```python
class VoiceRegistry:
    """Manages voice discovery, loading, caching, and blending."""
    
    def get_voice(self, voice_id: str) -> VoiceTensor:
        """Load a voice by ID. Handles blending syntax."""
        
    def list_voices(self) -> list[VoiceInfo]:
        """List all available voices."""
        
    def blend_voices(self, spec: str) -> VoiceTensor:
        """Parse 'voice1(w1)+voice2(w2)' and return blended tensor."""
        
    def save_blend(self, spec: str, path: Path) -> None:
        """Cache a blended voice to disk for reuse."""
```

### Voice Blending Algorithm

1. Parse voice spec string: `af_bella(2)+af_sky(1)` → `[("af_bella", 2), ("af_sky", 1)]`
2. Normalize weights: `[2, 1]` → `[0.667, 0.333]`
3. Load each voice tensor
4. Weighted average: `result = sum(weight * tensor for weight, tensor in zip(weights, tensors))`
5. Cache result for reuse

---

## 4. Backend Abstraction

### TTSBackend Protocol

Mirror the existing `STTBackend` pattern:

```python
@runtime_checkable
class TTSBackend(Protocol):
    """Protocol that all TTS backends must implement."""
    
    name: str
    sample_rate: int  # Native output sample rate (e.g. 24000 for Kokoro)

    def load_model(self, model_id: str) -> None: ...
    def unload_model(self, model_id: str) -> None: ...
    def loaded_models(self) -> list[LoadedModelInfo]: ...
    def is_model_loaded(self, model_id: str) -> bool: ...
    
    def synthesize(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        lang_code: str | None = None,
    ) -> Iterator[np.ndarray]:
        """Generate audio chunks as numpy arrays (float32, mono, native sample rate).
        
        Yields chunks for streaming support. Each chunk is a sentence/segment.
        """
        ...
    
    def list_voices(self) -> list[VoiceInfo]: ...
```

Key design decisions:
- **Yield chunks** — Kokoro naturally generates per-sentence. This enables streaming.
- **Return raw numpy** — Let the audio pipeline handle encoding. Backends stay format-agnostic.
- **Voice string passthrough** — Backend receives the raw voice string (including blend syntax). The backend or a shared voice registry handles parsing.

### Kokoro Backend Implementation Notes

The `kokoro` Python package (≥0.9.4) provides `KPipeline`:

```python
from kokoro import KPipeline

pipeline = KPipeline(lang_code='a')  # 'a' = American English
generator = pipeline(text, voice='af_heart')
for gs, ps, audio in generator:
    # audio is numpy array, 24kHz
    yield audio
```

- Model auto-downloads from HuggingFace on first use (~330MB)
- Voices auto-download as needed (~350KB each)
- Requires `espeak-ng` system package for phoneme fallback
- Uses PyTorch for GPU inference, can fall back to CPU
- Sample rate: 24,000 Hz

### Future Backend Sketches

**Piper:**
- ONNX-based, very fast on CPU
- Many voices/languages via downloadable voice packs
- Lower quality than Kokoro but much lighter
- Good for high-throughput, low-latency scenarios

**Qwen3-TTS:**
- Voice cloning and voice design from text descriptions
- Multiple model sizes (0.6B, 1.7B)
- GPU recommended

---

## 5. Audio Pipeline

### Encoding Pipeline

TTS backends output raw numpy float32 arrays at their native sample rate. The pipeline converts to the requested format:

```
Backend (numpy float32 @ 24kHz)
  → Resample (if target rate differs)
  → Normalize / Volume adjustment
  → Encode (mp3/opus/wav/flac/aac/pcm)
  → Stream chunks to response
```

### Implementation

```python
class AudioEncoder:
    """Encode raw audio to various output formats."""
    
    def encode(
        self,
        audio_chunks: Iterator[np.ndarray],
        format: str,
        sample_rate: int = 24000,
        speed: float = 1.0,
    ) -> Iterator[bytes]:
        """Encode audio chunks to the requested format, yielding output bytes."""
```

**Encoding libraries:**
- `wav` / `pcm` — Python stdlib `wave` module or raw bytes
- `mp3` — `lameenc` or `pydub` (wraps ffmpeg)
- `opus` — `opuslib` or ffmpeg subprocess
- `flac` — `soundfile` (already common dependency)
- `aac` — ffmpeg subprocess

**Recommended approach:** Use `soundfile` for wav/flac, `ffmpeg` subprocess for mp3/opus/aac. FFmpeg is already commonly available in Docker images and handles all formats reliably.

### Streaming Strategy

For streaming responses:
1. Kokoro generates per-sentence audio chunks
2. Each chunk is independently encoded
3. Chunks are yielded as `StreamingResponse` body

For MP3 streaming, use CBR mode so chunks are independently decodable. For Opus, use OGG container with page boundaries at chunk edges.

---

## 6. Configuration

### New Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_ENABLED` | `true` | Enable/disable TTS endpoints |
| `TTS_DEFAULT_MODEL` | `kokoro` | Default TTS model |
| `TTS_DEFAULT_VOICE` | `af_heart` | Default voice when none specified |
| `TTS_DEVICE` | `${STT_DEVICE}` | Device for TTS inference (`cuda`/`cpu`). Falls back to STT_DEVICE. |
| `TTS_VOICES_DIR` | `/app/voices` | Directory for voice files |
| `TTS_MODEL_DIR` | `` | Model storage path (empty = HF cache) |
| `TTS_MAX_INPUT_LENGTH` | `4096` | Max input text length (chars) |
| `TTS_SAMPLE_RATE` | `24000` | Output sample rate |
| `TTS_DEFAULT_FORMAT` | `mp3` | Default output format |
| `TTS_DEFAULT_SPEED` | `1.0` | Default speech speed |
| `TTS_PRELOAD_MODELS` | `` | Comma-separated TTS models to preload |
| `TTS_VOICE_MAPPING` | `` | Path to JSON file mapping OpenAI voice names to backend voices |
| `TTS_STREAM_CHUNK_SENTENCES` | `1` | Number of sentences per streaming chunk |

### Updated Settings Class

```python
class Settings(BaseSettings):
    # Existing STT settings (unchanged)...
    
    # TTS settings
    tts_enabled: bool = True
    tts_default_model: str = "kokoro"
    tts_default_voice: str = "af_heart"
    tts_device: str | None = None  # Falls back to stt_device
    tts_voices_dir: str = "/app/voices"
    tts_model_dir: str | None = None
    tts_max_input_length: int = 4096
    tts_sample_rate: int = 24000
    tts_default_format: str = "mp3"
    tts_default_speed: float = 1.0
    tts_preload_models: str = ""
    tts_voice_mapping: str = ""
    tts_stream_chunk_sentences: int = 1
    
    @property
    def tts_effective_device(self) -> str:
        return self.tts_device or self.stt_device
```

---

## 7. Docker & Deployment

### Single Container Strategy

The Dockerfile adds TTS dependencies alongside existing STT ones:

```dockerfile
# Base: NVIDIA CUDA runtime (same as current)
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y \
    python3.11 espeak-ng ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python deps (both STT + TTS)
COPY pyproject.toml .
RUN pip install .[tts]

# App
COPY src/ src/
```

### GPU Memory Budget

| Component | VRAM (approx) | Notes |
|-----------|---------------|-------|
| faster-whisper large-v3-turbo (float16) | ~1.5 GB | STT model |
| faster-whisper base | ~0.2 GB | Smaller STT model |
| Kokoro-82M (float32) | ~0.4 GB | 82M params × 4 bytes |
| Kokoro-82M (float16) | ~0.2 GB | If we cast to fp16 |
| Silero VAD | ~0.05 GB | Tiny |
| CUDA overhead | ~0.5 GB | Context, kernels |
| **Total (typical)** | **~2.5–3 GB** | Well within 8GB GPUs |

**Memory sharing is natural** — both models are loaded into the same GPU process. PyTorch and CTranslate2 (used by faster-whisper) can coexist on the same CUDA device. No special sharing mechanism needed.

**Concurrent inference:** STT and TTS will rarely run simultaneously in typical usage (user speaks → STT → process → TTS → play). If they do overlap, the GPU handles both since total VRAM is well under budget.

### Updated docker-compose.gpu.yml

```yaml
services:
  open-speech:
    build: .
    image: jwindsor1/open-speech:latest
    ports:
      - "8100:8100"
    environment:
      # STT
      - STT_DEVICE=cuda
      - STT_COMPUTE_TYPE=float16
      - STT_DEFAULT_MODEL=deepdml/faster-whisper-large-v3-turbo-ct2
      - STT_PRELOAD_MODELS=deepdml/faster-whisper-large-v3-turbo-ct2
      # TTS
      - TTS_DEFAULT_MODEL=kokoro
      - TTS_DEFAULT_VOICE=af_heart
      - TTS_PRELOAD_MODELS=kokoro
    volumes:
      - hf-cache:/root/.cache/huggingface
      - vad-cache:/root/.cache/silero-vad
      - voices:/app/voices
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  hf-cache:
  vad-cache:
  voices:
```

---

## 8. Dependencies & Compatibility

### New Python Dependencies for TTS

| Package | Version | Purpose | Size |
|---------|---------|---------|------|
| `kokoro` | ≥0.9.4 | Kokoro TTS inference library | ~50KB (pulls deps) |
| `misaki` | (kokoro dep) | G2P / phonemization | ~100KB |
| `torch` | ≥2.0 | Neural network runtime | ~2GB (already needed by kokoro) |
| `transformers` | ≥4.30 | Model loading (kokoro dep) | ~5MB |
| `soundfile` | ≥0.12 | Audio file I/O (wav, flac) | ~5MB |
| `scipy` | ≥1.10 | Signal processing, resampling | ~30MB |

### System Dependencies

| Package | Purpose |
|---------|---------|
| `espeak-ng` | Phoneme fallback for unknown words |
| `ffmpeg` | Audio format conversion (mp3, opus, aac) |

### Compatibility Analysis

**PyTorch vs CTranslate2 (faster-whisper):**
- faster-whisper uses CTranslate2, which has its own CUDA bindings separate from PyTorch
- Both can coexist — CTranslate2 doesn't conflict with PyTorch's CUDA usage
- The current Dockerfile likely doesn't include PyTorch (faster-whisper doesn't need it)
- **Adding `kokoro` will pull in PyTorch (~2GB)** — this is the biggest new dependency
- Ensure CUDA versions match: CTranslate2 and PyTorch must target the same CUDA major version (12.x)

**Key risk:** PyTorch ships with its own CUDA libraries (`nvidia-*` packages). If faster-whisper's CTranslate2 was compiled against a different CUDA version, there could be conflicts. Mitigation: pin compatible versions in pyproject.toml and test thoroughly.

**ONNX alternative:** `kokoro-onnx` exists and uses ONNX Runtime instead of PyTorch. Since we already have `onnxruntime` as a dependency, this would be lighter. However:
- `kokoro-onnx` is a separate project (thewh1teagle), not from the official kokoro author
- May lag behind on features/voices
- The official `kokoro` package is better maintained
- **Recommendation:** Use official `kokoro` with PyTorch. The size cost is worth the compatibility and features.

### Updated pyproject.toml

```toml
[project.optional-dependencies]
tts = [
    "kokoro>=0.9.4",
    "soundfile>=0.12.0",
]
gpu = [
    "torch>=2.0",
]
cpu = [
    "torch>=2.0",  # CPU-only wheel
]
```

---

## 9. Migration Path

### From Kokoro-FastAPI to Open Speech TTS

Current setup (two containers):
```
Speaches/open-speech (STT) → port 8100
Kokoro-FastAPI (TTS)        → port 8880
```

Target (one container):
```
Open Speech (STT + TTS) → port 8100
```

### Migration Steps

1. **Update Open Speech** to include TTS support
2. **Test** that `/v1/audio/speech` works identically to Kokoro-FastAPI's endpoint
3. **Update clients** to point TTS requests to port 8100 instead of 8880
4. **Verify** OpenAI SDK compatibility: `client.audio.speech.create()` should work unchanged except for `base_url`
5. **Remove** Kokoro-FastAPI container from docker-compose

### API Compatibility Checklist

| Kokoro-FastAPI Endpoint | Open Speech Equivalent | Status |
|------------------------|----------------------|--------|
| `POST /v1/audio/speech` | `POST /v1/audio/speech` | Phase 1 |
| `GET /v1/audio/voices` | `GET /v1/audio/voices` | Phase 1 |
| `POST /v1/audio/voices/combine` | `POST /v1/audio/voices/combine` | Phase 1 |
| `GET /v1/models` | `GET /v1/models` (extended) | Phase 1 |
| `GET /web` | `GET /web` (extended) | Phase 2 |

---

## 10. Phase Plan

### Phase 1: Core TTS (MVP)

**Goal:** Drop-in replacement for Kokoro-FastAPI.

- [ ] Create `src/tts/` package structure
- [ ] Implement `TTSBackend` protocol
- [ ] Implement `KokoroBackend` using the `kokoro` Python package
- [ ] Implement `AudioEncoder` (mp3, wav, opus, flac, pcm via ffmpeg)
- [ ] Add `POST /v1/audio/speech` endpoint (non-streaming)
- [ ] Add `GET /v1/audio/voices` endpoint
- [ ] Voice blending support (Kokoro `+` syntax with weights)
- [ ] OpenAI voice name mapping
- [ ] Configuration: `TTS_*` env vars
- [ ] Update Dockerfile with `espeak-ng`, `ffmpeg`, PyTorch
- [ ] Update `docker-compose.gpu.yml`
- [ ] Unit tests for TTS pipeline
- [ ] Integration test: OpenAI SDK speech creation

### Phase 2: Streaming & Web UI

**Goal:** Full streaming support and unified web interface.

- [ ] Streaming response for `/v1/audio/speech` (chunked transfer)
- [ ] Update web UI: add TTS tab (text input → audio playback)
- [ ] Voice preview in web UI
- [ ] `POST /v1/audio/voices/combine` endpoint
- [ ] Performance benchmarks (latency, throughput)

### Phase 3: Additional Backends

**Goal:** Pluggable engine support.

- [x] Piper backend (CPU-optimized, many voices)
- [x] Backend selection via model name prefix or config
- [ ] Voice management API (upload, delete custom voices)

### Phase 4: Advanced Features

- [x] Qwen3-TTS backend (voice cloning + design)
- [ ] WebSocket streaming TTS (bidirectional: text in → audio out)
- [ ] SSML support
- [ ] Per-word timestamps / caption generation
- [ ] Multi-language auto-detection

---

## 11. Testing Strategy

### Unit Tests

```
tests/
├── test_api.py              # Existing STT tests
├── test_tts_api.py           # TTS endpoint tests
├── test_tts_pipeline.py      # Audio encoding pipeline
├── test_voice_registry.py    # Voice loading, blending, listing
├── test_voice_blending.py    # Weight parsing, tensor mixing
├── test_audio_encoder.py     # Format conversion (mp3/wav/opus/flac)
└── test_openai_compat.py     # OpenAI SDK integration
```

### Test Categories

**1. API Contract Tests** (mock backend):
- Correct status codes and content types
- Request validation (missing fields, too-long input, invalid format)
- OpenAI voice name mapping
- Streaming response headers

**2. Backend Tests** (require model):
- Kokoro model loads/unloads correctly
- Generates non-empty audio for simple text
- Voice loading and blending produces valid tensors
- GPU and CPU inference paths both work

**3. Audio Pipeline Tests** (no model needed):
- Synthetic numpy → each output format
- Sample rate conversion
- Speed adjustment
- Chunked encoding for streaming

**4. Integration Tests:**
- OpenAI Python SDK `client.audio.speech.create()` end-to-end
- Streaming with `with_streaming_response`
- File upload STT → TTS round-trip
- Concurrent STT + TTS requests

**5. Voice Quality Validation** (manual/CI):
- Reference audio comparison (cosine similarity)
- Ensure no silence/corruption in output
- Verify blended voices sound like expected mix

### CI Considerations

- Unit tests and API contract tests run without GPU (mock backends)
- Backend and integration tests need a GPU runner or use CPU-mode Kokoro (slower but functional)
- Voice quality tests can be semi-automated with waveform analysis (amplitude, duration, silence detection)

---

## Appendix A: OpenAI TTS API Reference

From the OpenAI API specification:

```
POST /v1/audio/speech
Content-Type: application/json

{
  "model": "tts-1",          // required
  "input": "Hello world",    // required, max 4096 chars
  "voice": "alloy",          // required
  "response_format": "mp3",  // optional, default "mp3"
  "speed": 1.0               // optional, 0.25-4.0, default 1.0
}

Response: audio binary with Content-Type based on format
```

## Appendix B: Kokoro Voice Inventory (Partial)

| Voice ID | Language | Gender | Notes |
|----------|----------|--------|-------|
| `af_heart` | en-us | F | Default, warm |
| `af_bella` | en-us | F | Clear, professional |
| `af_sky` | en-us | F | Bright |
| `af_nova` | en-us | F | — |
| `am_adam` | en-us | M | Deep |
| `am_michael` | en-us | M | — |
| `bf_emma` | en-gb | F | British |
| `bm_george` | en-gb | M | British |

Full list available via `GET /v1/audio/voices` at runtime (auto-discovered from voice files).

## Appendix C: Resource Requirements Summary

| Resource | Current (STT only) | With TTS | Delta |
|----------|-------------------|----------|-------|
| Docker image size | ~3 GB | ~6 GB | +3 GB (PyTorch) |
| VRAM (typical) | ~2 GB | ~2.5 GB | +0.4 GB |
| RAM | ~1 GB | ~2 GB | +1 GB |
| Disk (models) | ~1.5 GB | ~2 GB | +0.5 GB |
| System deps | — | espeak-ng, ffmpeg | New |
