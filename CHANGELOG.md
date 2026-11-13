# Changelog

All notable changes to Open Speech are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Upgrade notes
- **The default STT model is now Parakeet** (`nvidia/parakeet-tdt-0.6b-v2`) in both Docker images and the Compose
  files, int8 on a CPU and fp16 on a GPU. It transcribes English only: a request in another language gets
  `400 language_unsupported`, and Wyoming lists English alone. For other languages set `STT_MODEL` to a Whisper
  model (such as `deepdml/faster-whisper-large-v3-turbo-ct2`, the previous GPU default, or
  `Systran/faster-whisper-base`, the previous CPU default), or name one per request. `whisper-1` translations need
  `STT_TRANSLATE_MODEL` set to a Whisper model that translates (`Systran/faster-whisper-medium` or `-large-v3`).
- On a GPU the first load downloads the 2.5 GB fp32 export and converts it to fp16 once (about a minute and 4 GB
  of RAM). Press Download on it in Models beforehand, or set `STT_PRELOAD_MODELS=nvidia/parakeet-tdt-0.6b-v2` to
  do this while the server starts.

### Added
- **NVIDIA Parakeet TDT 0.6B v2 speech-to-text** (`nvidia/parakeet-tdt-0.6b-v2`) through onnx-asr: English with
  punctuation, several times faster than Whisper large-v3-turbo on a GPU at a lower word error rate, and about as
  fast as Whisper base on a CPU. Long audio is cut with Silero VAD into pieces of at most 20 s; `verbose_json`,
  `srt` and `vtt` carry sentence segments and word times. `prompt` and `temperature` are accepted and ignored.
- `STT_PARAKEET_GPU_MEM_LIMIT_MB` (default `2304`) caps Parakeet's GPU memory pool, so it stays loaded next to
  Kokoro on an 8 GB card.

### Changed
- The Docker images and Compose files set `STT_MODEL=nvidia/parakeet-tdt-0.6b-v2` (was
  `deepdml/faster-whisper-large-v3-turbo-ct2` on the GPU image and the base Compose file, and
  `Systran/faster-whisper-base` on the CPU image and `docker-compose.cpu.yml`). The source default is unchanged.
- ONNX Runtime 1.24.4 (was 1.24.1); the CUDA image installs `onnxruntime-gpu` in its place.
- Streaming STT ends a Parakeet utterance at 25 s (30 s for Whisper).
- The web app's Transcribe view lists Parakeet next to the Whisper models, and a model picked there loads next to
  the pinned `STT_MODEL` instead of unloading it (`POST /api/models/{id}/load?keep_pinned=true`).

## [0.9.0] - 2026-11-13

### Upgrade notes
- **Voice worker images have new tags:** `open-speech-chatterbox:0.2.1`, `open-speech-cosyvoice:0.2.4` and
  `open-speech-qwen3:0.2.1` (they were `0.1.0`, `0.1.0` and `0.1.1`), and the new Fish Audio S2 worker is
  `open-speech-fish:0.1.0`. Each worker is now built with the `providers/` context and `<name>/Dockerfile`, so start
  it once with `--build`; a hand-written build is `docker build -f providers/<name>/Dockerfile providers`. A
  deployment that pins `OPEN_SPEECH_*_IMAGE` keeps its own tag.
- **The GPU voice workers are off by default.** Each worker in `docker-compose.voice-models.yml` has a Compose profile
  of its own, `chatterbox`, `cosyvoice`, `qwen3` and `fish`, and layering the file starts none of them. Name the ones
  you want with `--profile qwen3` (and so on, or `--profile "*"`) before `up`, or set
  `COMPOSE_PROFILES=qwen3,chatterbox` in the shell or `.env`, which also covers `down`. A worker that isn't turned
  on reads "Worker offline" in Models. Worker containers an earlier `up` started keep running, on their old images,
  until stopped; stop them once with
  `docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.voice-models.yml --profile "*" down`.
- **A voice worker on a GPU its torch build can't run** now reports `failed` with `cuda_arch_unsupported` on
  `/health` (it said `ok` while every load failed), so Compose shows it unhealthy and Models says why.
- **Host header check.** Requests get `400` (WebSocket handshakes `400` or close code `1008`) unless `Host` is an IP
  address, a single-label name, `localhost`, a name under `.localhost`, `.local`, `.lan`, `.home.arpa` or
  `.internal`, this machine's own hostname or FQDN (in Docker, the container's), or a name in the new `OS_ALLOWED_HOSTS`
  (`*.example.com` matches subdomains; `*` turns the check off). Add every other name that clients, monitors or a
  reverse proxy use, such as `*.fritz.box` for router-assigned LAN names.
- **Same-origin CORS and a cross-site write guard.** `OS_CORS_ORIGINS` now defaults to empty (no CORS headers), and
  `docker-compose.yml` no longer sets `*`. `POST`, `PUT`, `PATCH` and `DELETE` from a page of another origin (or a
  `null` one) get `403`; clients that send no `Origin` (SDKs, curl) are not affected. List cross-origin pages'
  exact origins in `OS_CORS_ORIGINS`, and in `OS_WS_ALLOWED_ORIGINS` for WebSockets, whose empty default now admits
  only this server's own pages. Behind a proxy, pass `Host` through with its port (nginx:
  `proxy_set_header Host $http_host;`) or list the public origin.
- **Security headers and API docs.** Responses carry `nosniff`, `Referrer-Policy`, `X-Frame-Options: DENY`, COOP and
  `Permissions-Policy`, and the web UI a strict `Content-Security-Policy`; embedding the UI in a frame (a Home
  Assistant webpage panel) needs `OS_SECURITY_HEADERS=false`. `/docs` and `/redoc` answer `404` unless
  `OS_API_DOCS=true`; `/openapi.json` is still served.
- **Failed-auth lockout and WebSocket rate limit.** With `OS_API_KEY` set, an address that sends
  `OS_AUTH_FAILURE_LIMIT` (default `10`) wrong keys within 60 seconds gets `429` for 60 seconds, even with the right
  key (`0` turns it off). WebSocket handshakes now count against `OS_RATE_LIMIT`. With `OS_TRUST_PROXY=true`, the
  client address is read from the right end of `X-Forwarded-For`; behind more than one proxy set
  `OS_TRUSTED_PROXY_HOPS`.
- **Refused WebSockets open, then close.** `/v1/audio/stream`, `/v1/audio/speech/stream` and `/v1/realtime` accept a
  socket they refuse and close it at once with a code and reason: `4001` a missing or wrong key or ticket, `1008` an
  Origin not allowed or an invalid query value, `4004` the feature turned off. They used to fail the handshake
  (`1006` in a browser). Read the close code and don't reconnect on these three.
- **API answers that changed.** `/v1/audio/transcriptions` with `response_format=json` answers only `{"text": ...}`,
  as OpenAI's does; diarized transcriptions follow `response_format`; `/v1/audio/translations` on a model that can't
  translate (large-v3-turbo, `*.en` and distil models) answers `400 translation_unsupported`, and the new
  `STT_TRANSLATE_MODEL` (empty by default) sets the model translations use. `GET /api/history` items carry `meta`
  as an object instead of `meta_json`. Model routes answer `500` or `503` for load failures where they said `400`.
- **CosyVoice no longer offers Japanese.** Its library reads Japanese only as katakana, so numbers and kanji came out
  garbled. A `ja` hint is refused before the worker is asked: `400` with code `http_error` and the model's languages.
  Text with kana and no hint gets `400 language_unsupported` with the model's languages and a pointer to Qwen3 Base
  (`qwen3/0.6b-base`).
- **Models and caches.** New `OS_PIN_DEFAULT_MODELS` (default `true`) keeps `STT_MODEL` and `TTS_MODEL` loaded
  through `OS_MODEL_TTL` and LRU eviction, outside the `OS_MAX_LOADED_MODELS` count. New `TTS_PRELOAD_DEFAULT`
  (default `true`) loads `TTS_MODEL` in the background at startup, which may download it on a first start. TTS cache
  keys and Reader part ids now include `TTS_TRIM_SILENCE` and `TTS_NORMALIZE_OUTPUT`, so cached audio is made once
  more after the upgrade.
- **Compose.** `docker-compose.yml` passes most settings through empty, so the application default or a legacy name
  in `.env` now applies: check `.env` for `STT_API_KEY` (clients then need that key) and `STT_SSL_ENABLED=false`. It
  ignores `OS_HOST`, `OS_PORT` and `OS_WYOMING_PORT`; change only the host side of `ports:` to move the server.
- **Removed settings and files.** `OS_HISTORY_RETAIN_AUDIO` and `OS_HISTORY_MAX_MB` never had an effect and now log a
  warning; `OS_HISTORY_MAX_ENTRIES` bounds history. `OS_STREAM_VAD_THRESHOLD`, `OS_PROVIDERS_DIR`,
  `docker-compose.qwen3.yml` (use `docker-compose.voice-models.yml --profile qwen3`) and `Dockerfile.cpu` (use the
  `DEVICE` build argument) are gone.
- **Clients.** The Python `client` extra requires `websockets>=14.0`. The Python and TypeScript clients default to
  `https://localhost:8100`; with `apiKey` set, the TypeScript client connects WebSockets with one-time tickets, so
  `RealtimeSession` and `LiveSpeechSession` open asynchronously (await `ready`).
- **TLS, Wyoming and the AI assistant.** A certificate Open Speech generated is renewed at startup within 30 days of
  expiry, so clients that pinned it must trust the new one; when only one of a certificate and key you supplied
  exists, startup stops with an error. Wyoming closes a connection after an `error` event or
  `OS_WYOMING_IDLE_TIMEOUT_S` (default `120`) seconds without an event, and at `OS_WYOMING_MAX_CONNECTIONS` (default
  `32`) a new connection closes the one that has waited longest. Once an AI
  server URL is set, `/v1/realtime` answers each turn by default; send `create_response: false` for transcripts only,
  and on a shared network set `OS_API_KEY` or `OS_ASSISTANT_CONNECTION_LOCKED=true`.
- **Reader documents.** Japanese, Chinese and Hindi text is now cut at its own sentence ends, and long sentences are no
  longer cut inside sound tags such as `[clear throat]`, for new imports; a document already in your library keeps its
  stored sentences until it is imported again.

### Added
- **Web app redesign.** `/web` is a new UI of plain static files and native JavaScript modules, with light and dark
  themes, a desktop rail and a mobile bottom bar, and an installable app manifest. Old hash addresses still work. See
  [Web UI](README.md#web-ui).
- **Reader.** Playback that keeps going across views with a mini-player and lock-screen controls, a sleep timer,
  speeds from 0.5× to 3.5×, Reading appearance (text size, typeface, spacing, paper themes), sentence skips, and the
  Atkinson Hyperlegible Next face bundled.
- **Speak, Voices and Studio.** A voice picker with search, favorites and previews, a visual blend editor, presets,
  copy as curl, Python or JavaScript; named voices with per-model versions (`/api/voices/identities`), a guided
  three-step cloning flow with quality checks, editing a preset or voice in Speak; reorderable Studio turns.
- **AI assistant.** Point Open Speech at an OpenAI-compatible AI server (`OS_ASSISTANT_*`, or Settings › AI assistant
  with no restart) and edit AI presets with thinking controls, token limits, waiting phrases and spoken error
  sentences. **Talk** (`#talk`) holds a spoken conversation with it; **Ask AI** sends a transcript, a Reader
  selection, sentence, paragraph, chapter or book, the editor's text or a Studio script, with an optional question,
  and shows, speaks or saves the answer; long inputs run as jobs. See [AI assistant](docs/AI-ASSISTANT.md).
- **`/v1/realtime` with the AI assistant:** replies streamed and spoken sentence by sentence, barge-in that tells an
  interruption from "mm-hmm", typed questions, and per-turn timing logs.
- **Voice workers.** Optional GPU workers for Chatterbox (regular and Turbo), CosyVoice 2 and 3, Qwen3-TTS 0.6B and
  Fish Audio S2 Pro (`fish/s2-pro`, 4-bit, about 3.3 GB of GPU memory), each an opt-in Compose profile, with pinned
  model revisions. See [Cloning models](docs/CLONING-MODELS.md).
- **Security settings:** `OS_ALLOWED_HOSTS`, `OS_SECURITY_HEADERS`, `OS_API_DOCS`, `OS_AUTH_FAILURE_LIMIT`,
  `OS_TRUSTED_PROXY_HOPS`, and `POST /api/auth/ws-ticket` for one-time WebSocket tickets.
- **OpenAI compatibility:** OpenAI TTS, STT and realtime model names map to `TTS_MODEL` and `STT_MODEL`, and the voices
  `ash`, `ballad`, `coral`, `sage`, `verse`, `marin` and `cedar` map to Kokoro voices.
- **API:** `GET /api/history?search=`, `max_input_length` in `GET /api/tts/capabilities`, `encoding=linear16` on
  `/v1/audio/stream`, more transcription `language` forms (`auto`, `en-US`), `HEAD` on Reader media, `OS_LOG_LEVEL`.
- **Docs:** what Open Speech is for, a Troubleshooting section, [AI agents](docs/AI-AGENTS.md),
  [AI assistant](docs/AI-ASSISTANT.md) and [Cloning models](docs/CLONING-MODELS.md) guides, and
  `scripts/smoke_openai_api.py`. A GitHub Actions workflow runs the test suite.

### Changed
- Streamed `/v1/audio/speech` sends audio as the backend produces it, and `/v1/realtime` reply audio streams as each
  chunk is synthesized.
- `/v1/audio/speech` fills omitted fields from settings (`TTS_MODEL`, `TTS_SPEED`, `TTS_DEFAULT_FORMAT`, `TTS_VOICE`)
  and ignores unknown JSON fields.
- Transcription uses Whisper's temperature fallback at `temperature` 0, as OpenAI does; diarization follows Whisper's
  segment timestamps and runs on CUDA when available; batch files are preprocessed like single uploads and wait on
  disk.
- SSML `<break>` strengths and time units, structural tags and comments follow the SSML spec more closely.
- Model loads, unloads, downloads and evictions are serialized per model type; downloads no longer load the model.
- Kokoro shares its weights across languages; Piper's `speed` scales each voice's own pace.
- Installs and images use a frozen `uv.lock`; `httpx` is a core dependency.
- GPU workers return freed memory after unload, the Chatterbox worker restarts itself after an unload, and a worker's
  CUDA out-of-memory error is `503 insufficient_gpu_memory`.
- Docker's healthcheck allows a 180 s start period, and Compose on Linux gives `./data` to the server's user when it
  can't write it.
- Speak › Live is a developer option (Settings › Developer); `/v1/audio/speech/stream` is unchanged.
- Transcripts stay out of the INFO log; session logs give sizes and levels only.
- US spelling throughout the web app and docs; API values such as `cancelled` keep their spelling.

### Fixed
- **Speech:** clear `400` errors for languages, effects and reference audio a model can't take; NaN samples, empty
  clips, ffmpeg failures and dictionary values with backslashes no longer fail requests; the `podcast_eq` effect keeps
  the speech band without clipping; Kokoro speaks long Chinese, Japanese, Hindi and Romance-language text to the end.
- **Transcription:** subtitle cue text and timestamps, unknown languages, empty audio, `STT_NORMALIZE` clipping, VAD
  trimming at odd sample rates and the first sound of an utterance.
- **Streaming and realtime:** odd-length frames, sample-rate conversion, close codes, `session.update` edge cases,
  VAD failures, lone UTF-16 surrogates in events, and large frames that stalled the event loop.
- **Live Reader:** Markdown fences, links and list markers mid-line, segment caps, edits to text already sent, and
  sentences the pronunciation dictionary empties.
- **Reader:** CommonMark-conformant Markdown import in linear time, WHATWG charset handling, EPUB paths, chapter names
  and reading order, re-rendering after a changed voice, dictionary or trimming setting, and two listeners at once.
- **Studio:** validation of effects, turns and lengths, renders that fail or race an edit, and database writes that
  could interleave.
- **Models API:** status codes, unknown models, download state, cache discovery and on-disk sizes; slow or failing
  workers no longer hold up listings or in-process synthesis.
- **Voice workers:** CosyVoice text normalization, streaming latency and error handling, Qwen3 GPU memory after
  unload and cut-off generations, and loading on every GPU of a supported family.
- **Wyoming:** error events, language normalization, VAD trimming for every sample format, and connection cleanup.
- **AI assistant:** the API key is sent only to the origin it was set for, extra bodies are bounded, AI server errors
  never echo their body, and long answers no longer stall the server.
- **Clients:** the Python and TypeScript clients report refused streams (`StreamRefusedError`) and clean up on early
  exit.
- **Web app:** accessible tabs and cards, storage failures, mobile layouts, microphone silence notices, and messages
  that say why a WebSocket was refused or a worker can't run.
- **Config:** empty variables count as unset, legacy `STT_*` names apply under Compose, `.env.example` works with
  `docker run --env-file`, and the docs say how to write `$` and `'` in values.

### Security
- Fixed an API-key bypass through a crafted `Host` header; upgrade any deployment that relies on `OS_API_KEY`.
- A legacy `STT_API_KEY` in `.env` is enforced under Compose (it was silently dropped).
- The Host check, same-origin CORS, the write guard and WebSocket origin checks stop other web pages, including DNS
  rebinding, from driving a LAN server through a visitor's browser.
- Request bodies, audio length and sample rate, SSML pauses, Wyoming events and JSON nesting are bounded before
  they are processed.
- Logs mask `api_key`, `ticket` and signatures in query strings; TLS setup never changes certificate files it did not
  generate; artifact deletion and composer paths can't reach outside their roots.
- Voice workers are addressed by private network aliases, so a LAN device named like a worker can't receive requests.

### Removed
- The Ming-3.0 TTS 0.5B (`ming/3-0.5b`) and VoxCPM2 (`voxcpm/2`) workers, which ran slower than real time; saved
  voices, presets or `TTS_MODEL` values that name them stop working. Remove their leftovers with
  `docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.voice-models.yml --profile "*" up -d --remove-orphans`,
  `docker volume rm open-speech_ming-hf-cache open-speech_voxcpm-hf-cache` and
  `docker rmi open-speech-ming:0.1.0 open-speech-voxcpm:0.1.0`.
- `OS_HISTORY_RETAIN_AUDIO`, `OS_HISTORY_MAX_MB`, `OS_STREAM_VAD_THRESHOLD`, `OS_PROVIDERS_DIR`,
  `docker-compose.qwen3.yml` and `Dockerfile.cpu`.

## [0.8.0] - 2026-09-12

### Added
- Live Reader WebSocket at `/v1/audio/speech/stream` for incremental text input and PCM16 audio output.
- Browser controls for reading typed or pasted text from the cursor, reading a complete document,
  pausing, resuming, stopping, and clearing.
- Book, responsive phrase, and completed-word reading modes.
- Playback acknowledgements, bounded buffering, cancellation generations, source offsets, idle flushing,
  Markdown cleanup, HTML entity decoding, and safe fenced-code handling.
- Live speech support in the TypeScript client and a raw Python streaming example.

### Changed
- Natural book mode now keeps comma-separated clauses together and retains a short pause at synthesis
  boundaries instead of treating every comma as a standalone utterance.
- Docker dependency layers are ordered ahead of application source so ordinary code rebuilds reuse the
  expensive CUDA, Torch, and provider layers.
- Live Reader controls now remain next to the text input at normal laptop resolutions.
- Maintained documentation now describes current behavior only; obsolete phase plans, machine-specific
  test plans, and point-in-time reviews were removed from the working tree.

### Fixed
- Live Reader flow control now counts Unicode code points consistently between JavaScript and Python, so
  emoji and other supplementary characters cannot leave browser input permanently blocked.
- Piper Live Reader sessions now advertise the selected model's actual sample rate.
- HTTP requests carrying WebSocket upgrade headers can no longer bypass HTTP authentication middleware.
- Voice blends and speech input handling received additional validation and resource bounds.

## [0.7.0] - 2026-02-27

### Added
- **Batch Transcription API** — async multi-file transcription with job queue
  - `POST /v1/audio/transcriptions/batch` — submit up to 20 audio files
  - `GET /v1/audio/jobs` — list jobs with optional status filter
  - `GET /v1/audio/jobs/{id}` — full job detail including results
  - `GET /v1/audio/jobs/{id}/result` — results array (409 if not done yet)
  - `DELETE /v1/audio/jobs/{id}` — cancel or delete jobs
- SQLite-backed batch job store (`src/batch/store.py`)
- Async batch worker (`src/batch/worker.py`) with configurable concurrency
- Per-file error isolation — one file failing doesn't abort the batch
- History integration — completed batch files auto-logged to STT history
- New config: `OS_BATCH_WORKERS` (default 2) — max concurrent batch jobs

## [0.6.1] - 2026-02-23

### Removed
- Moonshine and Vosk STT backends, and the Fish Speech, XTTS, F5-TTS, and in-process Qwen3 TTS
  backends with their dependencies. (Recorded after the fact; Qwen3 later returned as an isolated
  provider worker.)

### Fixed
- Stabilized model availability, loading, download progress, and provider-missing states.
- Improved the model browser, audio waveform, mobile layout, and Piper catalog handling.

## [0.6.0] - 2026-02-20

### Added
- Persistent voice profiles and generation history.
- Conversation rendering, voice effects, and the multi-track composer.
- OpenAI-style realtime audio, Silero VAD, and Wyoming protocol support.
- Python and TypeScript client libraries.

### Security
- Added bounded realtime buffers, WebSocket origin controls, upload limits, non-root containers, and
  hardened model-management concurrency.

## [0.5.1] - 2026-02-17

### Fixed
- Standardized HTTP error envelope across API and middleware responses to `{"error": "..."}`.
- Replaced ad-hoc `JSONResponse({"error": ...})` branches in TTS endpoints with `HTTPException` plus a unified exception handler.
- Synced README API endpoint table with all implemented routes in `src/main.py`.
- Added full OS_*/STT_*/TTS_* environment variable reference in README.
- Corrected README Qwen3 model IDs to match `src/model_registry.py` (`qwen3-tts/...`).
- Added missing `STT_MODEL_DIR` documentation entry to `.env.example`.
- Harmonized `docker-compose.yml`, `docker-compose.cpu.yml`, and `docker-compose.gpu.yml` env coverage, ports, and volume mounts using non-root container paths.
- Expanded runtime tests for `src/streaming.py` session lifecycle, VAD flow, mid-stream disconnects, and error propagation.

### Changed
- Version bumped to 0.5.1.

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
