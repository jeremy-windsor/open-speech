# Changelog

All notable changes to Open Speech are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

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
