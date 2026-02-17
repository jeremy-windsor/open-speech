# Changelog

All notable changes to Open Speech are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Security
- Security hardening (Sentinel review): safe Vosk zip extraction (Zip Slip prevention), realtime buffer limits + idle timeout, startup auth warning + `OS_AUTH_REQUIRED`, WebSocket origin allowlist (`OS_WS_ALLOWED_ORIGINS`), Wyoming bind host (`OS_WYOMING_HOST` default `127.0.0.1`), voice clone upload size guard, query-string API key deprecation warning, non-root Docker user, hardened TLS cert dir/permissions, and model-management locking improvements.

### Added
- Python client SDK upgrades:
  - `stream_transcribe()` now uses `/v1/audio/stream?vad=true` WebSocket protocol with sync + async support, event streaming, and reconnect handling.
  - `realtime_session()` now returns fully functional sync + async realtime session objects (`send_audio`, `commit`, `create_response`, `on_transcript`, `on_audio`, `on_vad`, `close`).
- New JS/TS client package scaffold at `client-js/` (`@open-speech/client`) with STT/TTS helpers, stream transcription, realtime session API, PCM conversion utilities, reconnection flow, and usage docs.
- Added `requirements.lock` with pinned core dependencies for reproducible installs.
- Web UI Transcribe tab improvements:
  - live audio level meter,
  - partial transcript stream panel,
  - VAD status indicator,
  - clearly separated final transcript + copy button,
  - duration and processing-time metrics.
- Added client SDK WebSocket unit tests for sync/async streaming + realtime protocol messages.
- **OpenAI Realtime API** — WebSocket endpoint at `/v1/realtime` for drop-in compatibility
  with OpenAI Realtime API clients (audio I/O only — STT + TTS, no LLM)
  - Session management (`session.create`, `session.update`)
  - Input audio buffer with append/commit/clear and server VAD auto-commit
  - Transcription via active STT backend on audio commit
  - TTS response streaming via `response.create` → `response.audio.delta` events
  - Audio format negotiation: pcm16 (24kHz), g711_ulaw, g711_alaw (8kHz)
  - Server VAD mode using Silero VAD from Phase 5b
  - Enable/disable with `OS_REALTIME_ENABLED` (default: true)
- **Voice Activity Detection (VAD)** — Silero VAD integration for speech detection
  - ONNX-based (<2MB model, MIT licensed), no PyTorch dependency
  - VAD-gated WebSocket STT — only forwards speech to backend, saving compute
  - `speech_start` / `speech_end` events sent to WebSocket clients
  - Configurable: `STT_VAD_ENABLED`, `STT_VAD_THRESHOLD`, `STT_VAD_MIN_SPEECH_MS`, `STT_VAD_SILENCE_MS`
  - `vad=true/false` query parameter on WebSocket endpoint
  - Web UI mic now shows recording states: listening → speech detected → processing
  - Wyoming STT handler uses VAD to filter silence before transcription
  - New module: `src/vad/` with `SileroVAD` wrapper, `is_speech()`, `get_speech_segments()`
- **Wyoming Protocol Support** — async TCP server (port 10400) for Home Assistant integration
  - Open Speech is now a drop-in STT + TTS provider for Home Assistant voice pipelines
  - Enable with `OS_WYOMING_ENABLED=true`, configure port with `OS_WYOMING_PORT`
  - Supports `Describe`, `Transcribe`, and `Synthesize` events
  - Audio resampled to Wyoming standard (16kHz, 16-bit, mono)
  - Runs alongside existing HTTP/WebSocket API on separate port

## [0.5.0] - 2026-02-17

### Added
- Phase 6 production hardening:
  - TTS response caching (LRU, configurable max size, cache bypass query param)
  - Speaker diarization endpoint support via optional `pyannote.audio`
  - Audio preprocessing (noise reduction + gain normalization) for STT
  - Audio postprocessing (silence trimming + normalization) for TTS
  - Python client SDK (`OpenSpeechClient`) with sync/async helpers
  - Pronunciation dictionary + SSML subset support (`input_type=ssml`)
- New optional extras: `diarize`, `noise`, `client`
- New docs: `docs/PHASE-6.md`
- Extensive test coverage for Phase 6 modules and APIs

### Changed
- Version bumped to 0.5.0

## [0.4.0] - 2026-02-17

### Added
- **Qwen3-TTS Backend** — voice design via text description, voice cloning, streaming, 0.6B and 1.7B models
- **Fish Speech Backend** — high-quality synthesis with zero-shot voice cloning
- **Extended TTS API** — `voice_design` and `reference_audio` fields on `/v1/audio/speech`
- **Voice Cloning Endpoint** — `POST /v1/audio/speech/clone` with multipart file upload
- **Voice Presets** — `GET /api/voice-presets` endpoint, presets dropdown in web UI Speak tab
- **Voice presets config** — load from YAML via `TTS_VOICES_CONFIG` env var
- Model registry: 3 new models (qwen3-tts-0.6b, qwen3-tts-1.7b, fish-speech-1.5)
- 23 new tests (332 total)

### Fixed
- Health endpoint now reports correct version (was hardcoded 0.1.0)
- Version sourced from `__version__` variable
- Speed slider: changed from 0.25 steps to 5% increments, min 0.5x (`75fb457`)
- Kokoro no longer appears in STT model dropdown (`c128533`)
- Kokoro-82M no longer listed as STT in Models tab (`c128533`)
- Moonshine models show "provider not installed" instead of Download when `useful-moonshine-onnx` is missing (`c128533`)
- Version badge now loads dynamically from `/health` endpoint (`c128533`)
- Voice presets updated to match actual available voices (`75fb457`)
- Mic transcription partial fix for WebSocket format issue (`c128533`)
- Moonshine package renamed from `moonshine-onnx` to `useful-moonshine-onnx` (`773095f`)

### Changed
- Version bumped to 0.4.0
- Optional dependency groups: `qwen` (transformers+accelerate+torch), `fish` (fish-speech)
- TTS history entries now have download and delete buttons (`c128533`)
- Stream toggle has tooltip explaining behavior (`c128533`)

## [0.3.0] - 2026-02-17

### Added
- **Phase 3: Unified Model Architecture**
  - Single image strategy — one container for CPU + GPU
  - Unified ModelManager for STT and TTS lifecycle
  - Environment variable rename: OS_* (server), STT_* (speech-to-text), TTS_* (text-to-speech)
  - Backwards compatibility for all old env var names
  - Unified /api/models endpoints (list, load, unload, status)
  - Model registry with 14 curated models and metadata
  - Model browser in web UI (available/downloaded/loaded states)
  - Download progress API
- **Piper TTS Backend**
  - 6 curated English voices (US + GB)
  - ONNX inference, lightweight (~35MB per model)
  - Auto-download from HuggingFace
- 36 new tests (309 total)

## [0.2.0] - 2026-02-16

### Added
- **Professional Web UI overhaul**
  - Light/dark theme with OS auto-detection
  - Inter font, card-based layout, pill tabs
  - Custom audio player with progress bar
  - Voice blend builder (visual tag-based UI)
  - Toast notifications (no more alert())
  - Mobile responsive design
- **HTTPS support** — auto-generated self-signed certificates
- **TTS environment variables** in Docker configs
- `.env.example` with all configuration documented
- Generic `docker-compose.yml` for easy testing

### Fixed
- Generate button spinner stuck after completion
- Kokoro showing as both STT and TTS in models tab
- STT models not showing device (cuda/cpu)
- Default model unload button showing when it shouldn't
- Streaming TTS torch tensor → numpy conversion

## [0.1.0] - 2026-02-14

### Added
- **Phase 1: Core STT Server**
  - OpenAI-compatible `/v1/audio/transcriptions` and `/v1/audio/translations`
  - faster-whisper backend with GPU (CUDA float16) and CPU (int8) support
  - Silero VAD for voice activity detection
  - Model lifecycle: TTL eviction, max models, LRU
  - Self-signed HTTPS with auto-generated certificates
- **Phase 2: TTS + Streaming**
  - Kokoro TTS backend (82M params, 52 voices, voice blending)
  - `/v1/audio/speech` endpoint (OpenAI-compatible)
  - Streaming TTS (`?stream=true`, chunked transfer)
  - WebSocket real-time STT (`/v1/audio/stream`)
  - Moonshine and Vosk STT backends
  - SRT/VTT subtitle formatters
  - 3-tab web UI (Transcribe, Speak, Models)
- **Security** — API key auth, rate limiting, CORS, upload limits
- Docker images: CPU and GPU
- 230 tests
