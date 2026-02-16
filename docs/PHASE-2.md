# Open Speech — Phase 2: Multi-Model + Streaming + Web UI

## Goal
A fully working STT/TTS server with multiple engines, streaming support,
file upload/download, and a web UI that exposes everything. CPU-first.
GPU support comes in Phase 3.

## STT Requirements

### Backends (all OpenAI-compatible `/v1/audio/transcriptions`)
- **faster-whisper** — default, proven, high accuracy
- **Moonshine** — fast CPU, English-only, tiny models
- **Vosk** — ultra-lightweight, offline, Kaldi-based

### Features
- [x] Batch transcription (upload file → get text)
- [ ] **Streaming STT via WebSocket** — `ws://host/v1/audio/stream`
  - Send audio chunks, receive partial transcripts in real-time
  - Moonshine and Vosk both support streaming natively
  - faster-whisper: use chunked VAD approach
- [ ] Model switching via `model` parameter
- [ ] File upload: WAV, MP3, OGG, FLAC, M4A, WEBM
- [ ] Response formats: json, text, srt, vtt (OpenAI spec)
- [ ] Download transcript as file (.txt, .srt, .vtt)

## TTS Requirements

### Backends (all via `/v1/audio/speech`)
- **Kokoro** — 82M, fast, 52 voices, blending

### Features
- [x] Batch synthesis (text → audio file)
- [ ] **Streaming TTS** — `POST /v1/audio/speech` with chunked response
  - Audio starts playing before full generation completes
  - Use persistent ffmpeg pipe (already built)
- [ ] Text file upload → bulk synthesis
- [ ] Voice blending via API
- [ ] Speed control (0.25x–4.0x)
- [ ] Output formats: mp3, wav, opus, flac, pcm
- [ ] Download generated audio

## Web UI Requirements

### Transcribe Tab (STT)
- [ ] Model selector dropdown (faster-whisper / moonshine / vosk)
- [ ] File upload (drag & drop)
- [ ] **Live microphone recording with real-time transcription**
- [ ] Streaming display (words appear as spoken)
- [ ] Output format selector (text / srt / vtt)
- [ ] Download transcript button
- [ ] History of recent transcriptions

### Speak Tab (TTS)
- [x] Text input area
- [x] Voice selector + blend input
- [x] Speed slider + format selector
- [x] Audio player with download
- [ ] **Text file upload** (paste or upload .txt → synthesize)
- [ ] **Bulk mode** — synthesize paragraph by paragraph
- [ ] Streaming audio playback (starts playing before done)
- [ ] History of recent generations

### Models Tab (new)
- [ ] Show all available STT + TTS backends
- [ ] Model status: loaded / not loaded / downloading
- [ ] Load / unload buttons
- [ ] Device info (CPU/GPU)
- [ ] Model size + memory usage

## API Endpoints (OpenAI-compatible)

### STT
- `POST /v1/audio/transcriptions` — batch transcribe (existing)
- `POST /v1/audio/translations` — translate to English (existing)
- `WS /v1/audio/stream` — streaming transcription (new)

### TTS
- `POST /v1/audio/speech` — synthesize (existing, add streaming)
- `GET /v1/audio/voices` — list voices (existing)

### Models
- `GET /v1/models` — list all models (existing)
- `POST /v1/audio/models/load` — load model (existing)
- `POST /v1/audio/models/unload` — unload model (existing)
- `GET /v1/audio/models` — TTS model status (existing)

### Utility
- `GET /health` — health check
- `GET /web` — web UI

## Constraints
- **English only** — no multilingual complexity
- **CPU-first** — must work without GPU
- **No auth** — local/private network use
- **Docker-ready** — everything works in the container
