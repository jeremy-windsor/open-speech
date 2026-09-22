# Open Speech

**Self-hosted speech-model harness with OpenAI-compatible APIs.**

[![Version](https://img.shields.io/badge/version-0.8.0-blue?style=flat-square)]()
[![Docker Hub](https://img.shields.io/docker/pulls/jwindsor1/open-speech?style=flat-square&logo=docker)](https://hub.docker.com/r/jwindsor1/open-speech)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue?style=flat-square&logo=python)](https://python.org)

## What is Open Speech?

Open Speech is a self-hosted harness for running, comparing, and controlling speech models through
OpenAI-style endpoints. It currently provides:

- **Speech-to-text** via `faster-whisper`
- **Text-to-speech** via local backends such as **Kokoro**, **Piper**, and **Pocket-TTS**, plus isolated GPU providers
- **Streaming STT** over WebSocket
- **Audio-focused realtime I/O** over `/v1/realtime`
- **Batch jobs**, a **web UI**, **history/profiles/conversations/composer**, and **Wyoming** integration

This repo is not a universal provider gateway. It is a pragmatic local speech-model harness with a
compatible API surface. Models and provider-specific controls remain explicit so the harness does not
pretend that every engine supports the same features.

## Features

### Speech-to-Text
- OpenAI-compatible `/v1/audio/transcriptions` and `/v1/audio/translations`
- Real-time streaming transcription via WebSocket at `/v1/audio/stream`
- Silero VAD support for streaming sessions
- `json`, `verbose_json`, `text`, `srt`, and `vtt` response formats
- Optional diarization (`STT_DIARIZE_ENABLED=true` + pyannote extra)
- Optional preprocessing (noise reduction + normalization)
- Async batch transcription jobs

### Text-to-Speech
- OpenAI-compatible `/v1/audio/speech`
- Incremental Live Reader over `/v1/audio/speech/stream`
- Local TTS backends: Kokoro, Piper, Pocket-TTS
- Disk-backed TTS cache
- Pronunciation dictionary + basic SSML parsing
- Output post-processing (trim silence, normalize)
- Voice presets for the web UI
- Voice Lab for recording, uploading, previewing, correcting, and profile-linking local clone references
- Kokoro voice blending using the `voice` field, e.g. `af_bella(2)+af_sky(1)`
- Model-specific capabilities and voice catalogs so controls only appear when the selected model supports them
- Isolated Qwen3, Chatterbox, and CosyVoice GPU providers without adding their conflicting
  Torch and Transformers pins to the core harness
- Machine-readable TTS conformance report for provider metadata, controls, voices, and opt-in audio checks

### Runtime / Platform
- Unified model browser + load/unload/download endpoints
- Audio-focused `/v1/realtime` WebSocket endpoint
- SQLite-backed studio features: profiles, history, conversations, composer
- Wyoming protocol support for Home Assistant-style integrations
- Self-signed HTTPS support
- API key auth, CORS, rate limiting, WebSocket origin checks

## Quick Start

```bash
docker run -d -p 8100:8100 jwindsor1/open-speech:cpu
```

Open **https://localhost:8100/web** and accept the self-signed cert.

GPU example:

```bash
docker run -d -p 8100:8100 --gpus all jwindsor1/open-speech:latest
```

## Installation (from source)

```bash
git clone https://github.com/jeremy-windsor/open-speech.git
cd open-speech
pip install -e .                  # Core runtime
pip install -e ".[tts]"          # Kokoro TTS
pip install -e ".[piper]"        # Piper TTS
pip install -e ".[diarize]"      # Speaker diarization
pip install -e ".[noise]"        # Noise reduction preprocessing
pip install -e ".[client]"       # Client SDK deps
pip install -e ".[dev]"          # pytest, ruff, httpx
pip install -e ".[all]"          # Core + common optional backends
pip install -r requirements.lock  # Fully pinned core runtime deps
```

## Models

Models are downloaded on demand and cached on disk.

### STT Models

| Model | Size | Backend | Languages |
|---|---:|---|---|
| `deepdml/faster-whisper-large-v3-turbo-ct2` | ~800MB | faster-whisper | 99+ |
| `Systran/faster-whisper-large-v3` | ~1.5GB | faster-whisper | 99+ |
| `Systran/faster-whisper-medium` | ~800MB | faster-whisper | 99+ |
| `Systran/faster-whisper-small` | ~250MB | faster-whisper | 99+ |
| `Systran/faster-whisper-base` | ~150MB | faster-whisper | 99+ |
| `Systran/faster-whisper-tiny` | ~75MB | faster-whisper | 99+ |

### TTS Models

| Model | Size | Backend | Notes |
|---|---:|---|---|
| `kokoro` | ~82MB | Kokoro | default backend, many voices, blend syntax in `voice` |
| `pocket-tts` | ~220MB | Pocket-TTS | built-in voices, backend advertises streaming support |
| `piper/en_US-lessac-medium` | ~35MB | Piper | one voice per model |
| `piper/en_US-joe-medium` | ~35MB | Piper | one voice per model |
| `piper/en_US-amy-medium` | ~35MB | Piper | one voice per model |
| `piper/en_US-arctic-medium` | ~35MB | Piper | one voice per model |
| `piper/en_GB-alan-medium` | ~35MB | Piper | one voice per model |
| `qwen3/0.6b-base` | ~1.8GB | isolated Qwen3 provider | reference cloning with an exact transcript |
| `chatterbox/regular` | ~8.6GiB | isolated Chatterbox provider | English reference cloning |
| `chatterbox/turbo` | ~5.4GiB | isolated Chatterbox provider | faster English cloning, native speech tags |
| `cosyvoice/2-0.5b` | ~4.6GiB | isolated CosyVoice provider | multilingual cloning, instructions, speed control |
| `cosyvoice/3-0.5b` | ~9.4GiB | isolated CosyVoice provider | multilingual cloning, instructions, native streaming |

The optional model sizes are approximate Windows cache additions observed during the September 2026
RTX 2070 SUPER validation. Chatterbox Regular plus Turbo and CosyVoice 2 plus 3 each occupied about
14GiB in their shared provider cache. Cache size is disk use, not model VRAM. First load includes model
download; use a cached reload to measure startup. See [TTS Backends](docs/TTS-BACKENDS.md) for the
provider bundle and validation matrix.

## API Reference

All endpoints use Bearer auth when `OS_API_KEY` is set.
Interactive docs are available at `/docs`.

### Speech-to-Text

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/audio/transcriptions` | Transcribe audio |
| `POST` | `/v1/audio/translations` | Translate audio to English |
| `POST` | `/v1/audio/transcriptions/batch` | Submit a batch job |
| `GET` | `/v1/audio/jobs` | List jobs |
| `GET` | `/v1/audio/jobs/{job_id}` | Job detail |
| `GET` | `/v1/audio/jobs/{job_id}/result` | Completed job results |
| `DELETE` | `/v1/audio/jobs/{job_id}` | Cancel/delete a job |
| `GET` | `/v1/audio/stream` | Returns `426` telling HTTP clients to use WebSocket |
| `WS` | `/v1/audio/stream` | Real-time streaming transcription |

#### `POST /v1/audio/transcriptions`

Multipart form upload.

**Fields:**
- `file` (required)
- `model`
- `language`
- `prompt`
- `response_format` = `json | verbose_json | text | srt | vtt`
- `temperature`
- `diarize`

```bash
curl -sk https://localhost:8100/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=deepdml/faster-whisper-large-v3-turbo-ct2" \
  -F "response_format=json"
```

#### `WS /v1/audio/stream`

Send PCM16 audio chunks, receive transcript/VAD events.

```javascript
const ws = new WebSocket("wss://localhost:8100/v1/audio/stream?vad=true");
ws.onmessage = (e) => console.log(JSON.parse(e.data));
ws.send(audioChunkArrayBuffer);
```

### Text-to-Speech

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/audio/speech` | Synthesize speech |
| `GET` | `/v1/audio/speech/stream` | Returns `426` telling HTTP clients to use WebSocket |
| `WS` | `/v1/audio/speech/stream` | Incremental text in, PCM16 audio out |
| `POST` | `/v1/audio/speech/clone` | Multipart reference-audio endpoint for compatible backends |
| `GET` | `/v1/audio/voices` | List voices |
| `GET` | `/v1/audio/models` | List TTS models and load state |
| `POST` | `/v1/audio/models/load` | Load TTS model |
| `POST` | `/v1/audio/models/unload` | Unload TTS model |
| `GET` | `/api/tts/capabilities` | TTS backend capabilities |
| `GET` | `/api/voice-presets` | Voice presets for the UI |

#### `POST /v1/audio/speech`

JSON body.

**Fields:**
- `model`
- `input`
- `voice`
- `speed`
- `response_format` = `mp3 | opus | aac | flac | wav | pcm | m4a`
- `language`
- `input_type` = `text | ssml`
- `voice_design` *(backend-gated)*
- `reference_audio` *(backend-gated)*
- `clone_transcript` *(backend-gated)*
- `effects`

**Query params:** `stream`, `cache`

```bash
curl -sk https://localhost:8100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro","input":"Hello world","voice":"af_heart"}' \
  -o output.mp3
```

> **Voice blending note:** the API uses the `voice` field for Kokoro blends. There is **no** separate `voice_blend` request field in the harness contract.

#### `POST /v1/audio/speech/clone`

Multipart form endpoint that forwards reference audio to backends that support it.
The route exists, but the built-in local backends in this tree do not currently provide a broad, production-ready voice cloning story.

#### Live Reader: `WS /v1/audio/speech/stream`

Live Reader accepts text deltas and immediately streams PCM16 audio back to the requester. With HTTPS,
the address is `wss://HOST:8100/v1/audio/speech/stream`; WSS is WebSocket traffic protected by HTTPS.
The Open Speech machine performs synthesis only. The browser, laptop, or other requesting client plays
or saves the returned audio.

The web UI at `/web` includes Start at cursor, Read all, Pause/Resume, Stop, Clear, full-sentence,
responsive-phrase, and word-by-word controls. A browser requires one Start click before it may play
audio. Existing one-shot TTS and `/v1/realtime` behavior are unchanged.

Client events:

```json
{"type":"session.update","session":{"model":"kokoro","voice":"af_heart","latency_mode":"natural"}}
{"type":"session.update","session":{"model":"chatterbox/turbo","voice":"reference","voice_library_ref":"narrator","latency_mode":"natural"}}
{"type":"input_text.append","text":"Words from a user or an AI token stream. "}
{"type":"input_text.commit"}
{"type":"playback.ack","response_id":"resp_...","sequence":0}
{"type":"response.cancel"}
```

Server audio arrives as `response.output_audio.delta` events with base64 PCM16LE mono, the backend's
actual `sample_rate`, a frame `sequence`, and the source range on the first frame. Acknowledge a frame
after the requester has played or otherwise consumed it. `input_text.done` means all text from the most
recent commit has been synthesized. Applications that generate AI text should forward their received
text deltas to this socket; Open Speech does not contact an AI provider itself.

Natural mode is the book-reading default: it prefers full sentences and paragraphs, releases incomplete
prose after its 1,500 ms input idle window, and remains bounded by separate sentence-mode limits.
`responsive` also releases comma, colon, and semicolon clauses for lower-latency AI text streams.
`instant_word` speaks each completed word sooner but sounds more choppy. Markdown prose markers are
removed, HTML entities are decoded, link labels are spoken without URLs, inline code is spoken, and
fenced code blocks are replaced with "Code block skipped." Dictionary matches that span two synthesized
segments may not apply. When silence trimming is enabled, a stateful edge trimmer retains a short natural
pause while preserving progressive backend chunks and interior pauses. Live Reader does not normalize
each segment, because that would cause sentence-to-sentence gain changes. Effects, encoding, and cache
writes remain outside the live path.

For a raw Python client example:

```bash
printf 'This text is streamed to the remote voice.\n' | \
  python examples/live_tts_client.py --url https://HOST:8100 --insecure
```

### Realtime Audio

| Method | Path | Description |
|---|---|---|
| `WS` | `/v1/realtime` | OpenAI-style realtime audio WebSocket |

`/v1/realtime` is **audio I/O only**: transcription, audio output, session events, VAD-style flow. It is not full OpenAI Realtime feature parity with tool calling and conversation orchestration.

### Model Management

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/models` | OpenAI-style model list |
| `GET` | `/v1/models/{model}` | OpenAI-style model detail |
| `GET` | `/api/ps` | Legacy loaded STT models |
| `POST` | `/api/ps/{model}` | Legacy STT load |
| `DELETE` | `/api/ps/{model}` | Legacy STT unload |
| `POST` | `/api/pull/{model}` | Legacy download/load+unload |
| `GET` | `/api/models` | Unified model inventory |
| `GET` | `/api/models/{id}/status` | Model state |
| `GET` | `/api/models/{id}/progress` | Download/load progress |
| `POST` | `/api/models/{id}/load` | Load model |
| `POST` | `/api/models/{id}/download` | Download artifacts |
| `POST` | `/api/models/{id}/prefetch` | Alias for download |
| `DELETE` | `/api/models/{id}` | Unload model |
| `DELETE` | `/api/models/{id}/artifacts` | Delete cached artifacts |

### Studio / Persistence APIs

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/voices/library` | Store a named voice reference |
| `GET` | `/api/voices/library` | List voice refs |
| `GET` | `/api/voices/library-config` | Get the reference-duration limit used by Voice Lab |
| `GET` | `/api/voices/library/{name}` | Get voice ref metadata |
| `GET` | `/api/voices/library/{name}/audio` | Preview stored voice ref audio |
| `PATCH` | `/api/voices/library/{name}` | Correct or clear a voice ref transcript |
| `DELETE` | `/api/voices/library/{name}` | Delete voice ref |
| `POST` | `/api/profiles` | Create profile |
| `GET` | `/api/profiles` | List profiles |
| `GET` | `/api/profiles/{id}` | Get profile |
| `PUT` | `/api/profiles/{id}` | Update profile |
| `DELETE` | `/api/profiles/{id}` | Delete profile |
| `POST` | `/api/profiles/{id}/default` | Set default profile |
| `GET` | `/api/history` | List history |
| `DELETE` | `/api/history/{id}` | Delete history entry |
| `DELETE` | `/api/history` | Clear history |
| `POST` | `/api/conversations` | Create conversation |
| `GET` | `/api/conversations` | List conversations |
| `GET` | `/api/conversations/{id}` | Get conversation |
| `POST` | `/api/conversations/{id}/turns` | Add turn |
| `DELETE` | `/api/conversations/{id}/turns/{turn_id}` | Delete turn |
| `POST` | `/api/conversations/{id}/render` | Render conversation |
| `GET` | `/api/conversations/{id}/audio` | Fetch rendered audio |
| `DELETE` | `/api/conversations/{id}` | Delete conversation |
| `POST` | `/api/composer/render` | Render composition |
| `GET` | `/api/composer/renders` | List renders |
| `GET` | `/api/composer/render/{id}/audio` | Fetch rendered composition |
| `DELETE` | `/api/composer/render/{id}` | Delete composition |

### Health / UI

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/web` | Web UI |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc |

## Web UI

Open **https://localhost:8100/web**.

Current UI areas:
- **Transcribe** — upload files, microphone input, streaming STT
- **Speak** — one-shot synthesis plus Live Reader for typed, pasted, or streamed text
- **Voice Lab** — record or upload a reference, verify its exact transcript, preview it, run a clone test, and save a profile
- **Models** — load/unload/download known models
- **History / Settings** — runtime convenience features

The web UI has a Kokoro blend builder, but the harness contract is still the plain `voice` string. In other words: the UI helps compose `af_bella(2)+af_sky(1)`, and the API only knows about `voice="af_bella(2)+af_sky(1)"`.

Speak opens on the configured reading-default model (normally Kokoro), not whichever optional model
was most recently loaded. Switching to another TTS model for Generate or Live Reader confirms that
the currently loaded TTS model will be unloaded. **Restore Kokoro** reselects and, if needed, loads the reading
default; it does not keep two TTS models resident on an 8 GB GPU.

`Save as Profile` stores the current provider, model, voice or Kokoro blend, speed, and output format
in the server-side Studio database. Profiles survive browser storage clearing and container replacement when
the `/home/openspeech/data` volume is preserved. The Speak tab's `Preset` selector applies these saved
profiles; the Settings tab can mark one as the default or delete it. `TTS_VOICES_CONFIG` is a separate
administrator-supplied YAML preset source and is not browser local storage.

## Voice Blending

Kokoro supports weighted voice syntax:

```bash
curl -sk https://localhost:8100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro","input":"Hello","voice":"af_bella(2)+af_sky(1)"}'
```

## Docker Compose

### Default CPU Launch

Plain Compose is CPU-safe and does not request NVIDIA GPU passthrough:

```bash
docker compose up -d
```

It defaults to `STT_DEVICE=cpu`, `STT_COMPUTE_TYPE=int8`, and `TTS_DEVICE=cpu`.

The checked-in `docker-compose.cpu.yml` remains available when you specifically want the CPU image:

```bash
docker compose -f docker-compose.cpu.yml up -d
```

### GPU Launch

GPU users must include the GPU override. Plain `docker compose up -d` is not a GPU launch.

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --force-recreate
```

The GPU override requests NVIDIA passthrough with `gpus: all`, includes the NVIDIA device reservation block, and sets `STT_DEVICE=cuda`, `STT_COMPUTE_TYPE=float16`, and `TTS_DEVICE=cuda`.

For Voice Lab cloning, launch the supported provider bundle as part of the harness:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  -f docker-compose.voice-models.yml up -d --build
```

The provider services remain ready while their model weights stay unloaded. In **Models**, use
**Download** to cache a model or **Load to GPU** to activate it. In **Voice Lab**, selecting a clone
model and clicking **Clone test** performs the load automatically. Open Speech unloads the previous
TTS model before loading the selected one, so an 8 GB GPU holds only one speech model at a time.

### Volumes

Model caches live under `/home/openspeech/.cache/huggingface` inside the container.
Persist that path unless you enjoy re-downloading large things for sport. The checked-in Compose files also persist Silero VAD, TLS certs, app data, and TTS cache volumes.

## Development and validation

Run ordinary tests from an isolated Python environment:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Build CUDA images on the Windows GPU host with immutable revision tags. Validate that tag before
moving `latest`:

```powershell
$Tag = "jwindsor1/open-speech:cuda-$(git rev-parse --short HEAD)"
docker build -f Dockerfile -t $Tag .
docker run -d --rm --name open-speech-canary --gpus all -p 8110:8100 `
  -e OS_SSL_ENABLED=false -e OS_WYOMING_ENABLED=false `
  -e STT_DEVICE=cuda -e STT_COMPUTE_TYPE=float16 -e TTS_DEVICE=cuda $Tag
```

The GPU gate covers the repository suite, every advertised STT and TTS model, uncached WAV output,
Live Reader, all Kokoro voices, and long-form STT. Keep recordings, transcripts, generated audio,
and raw results outside Git. Current backend-specific validation commands and known limits are in
[docs/TTS-BACKENDS.md](docs/TTS-BACKENDS.md#validation).

## Security

```bash
# API key
OS_API_KEY=my-secret-key docker compose up -d
curl -sk -H "Authorization: Bearer my-secret-key" https://localhost:8100/health

# Fail fast if API key missing
OS_AUTH_REQUIRED=true

# WebSocket origin allowlist
OS_WS_ALLOWED_ORIGINS=https://myapp.com,https://staging.myapp.com

# Rate limiting
OS_RATE_LIMIT=60
OS_RATE_LIMIT_BURST=10

# CORS
OS_CORS_ORIGINS=https://myapp.com,https://staging.myapp.com

# Custom TLS cert
OS_SSL_CERTFILE=/certs/cert.pem
OS_SSL_KEYFILE=/certs/key.pem

# Extra SANs for auto-generated self-signed certs
OS_TLS_EXTRA_SANS=192.0.2.24,openspeech.local
```

If you change `OS_TLS_EXTRA_SANS` after a cert has already been generated, remove the existing cert/key volume or files so Open Speech can regenerate them. With the checked-in Compose files, that usually means removing the generated `ssl-certs` Docker volume for the project.

## Environment Variables

Defaults come from `src/config.py`. The checked-in base Compose file additionally pins CPU-safe device settings; the GPU override changes those device settings to CUDA.

### `OS_*` — harness / shared

| Variable | Default | Description |
|---|---|---|
| `OS_PORT` | `8100` | HTTP bind port |
| `OS_HOST` | `0.0.0.0` | HTTP bind host |
| `OS_API_KEY` | `""` | Bearer API key; empty disables auth |
| `OS_AUTH_REQUIRED` | `false` | Fail startup if API key is missing |
| `OS_CORS_ORIGINS` | `*` | Comma-separated CORS origins |
| `OS_WS_ALLOWED_ORIGINS` | `""` | Allowed WebSocket `Origin` values |
| `OS_TRUST_PROXY` | `false` | Trust `X-Forwarded-For` headers |
| `OS_MAX_UPLOAD_MB` | `100` | Max upload size |
| `OS_RATE_LIMIT` | `0` | Requests/min/IP; `0` disables |
| `OS_RATE_LIMIT_BURST` | `0` | Burst bucket size |
| `OS_SSL_ENABLED` | `true` | Enable HTTPS |
| `OS_SSL_CERTFILE` | `""` | TLS cert path; auto-generated if empty |
| `OS_SSL_KEYFILE` | `""` | TLS key path; auto-generated if empty |
| `OS_TLS_EXTRA_SANS` | `""` | Extra comma-separated DNS/IP names for generated self-signed certs |
| `OS_VOICE_LIBRARY_PATH` | `/home/openspeech/data/voices` | Stored voice reference directory |
| `OS_VOICE_LIBRARY_MAX_COUNT` | `100` | Max stored voice refs; `0` = unlimited |
| `OS_VOICE_LIBRARY_MAX_SECONDS` | `60` | Max reference duration; `0` = unlimited |
| `OS_STUDIO_DB_PATH` | `/home/openspeech/data/studio.db` | SQLite DB for studio metadata |
| `OS_HISTORY_ENABLED` | `true` | Enable history logging |
| `OS_HISTORY_MAX_ENTRIES` | `1000` | Max retained history rows |
| `OS_HISTORY_RETAIN_AUDIO` | `true` | Retain output-path metadata where available |
| `OS_HISTORY_MAX_MB` | `2000` | Audio/history storage budget |
| `OS_EFFECTS_ENABLED` | `true` | Enable effects processing |
| `OS_CONVERSATIONS_DIR` | `/home/openspeech/data/conversations` | Conversation storage directory |
| `OS_COMPOSER_DIR` | `/home/openspeech/data/composer` | Composer storage directory |
| `OS_PROVIDERS_DIR` | `/home/openspeech/data/providers` | Provider package directory |
| `OS_BATCH_WORKERS` | `2` | Concurrent batch worker count |
| `OS_BATCH_MAX_PENDING` | `10` | Max queued + running batch jobs |
| `OS_BATCH_MAX_TOTAL_MB` | `500` | Max aggregate upload size per batch request |
| `OS_WYOMING_ENABLED` | `false` | Enable Wyoming TCP server |
| `OS_WYOMING_HOST` | `127.0.0.1` | Wyoming bind host |
| `OS_WYOMING_PORT` | `10400` | Wyoming port |
| `OS_REALTIME_ENABLED` | `true` | Enable `/v1/realtime` |
| `OS_REALTIME_MAX_BUFFER_MB` | `50` | Max realtime audio buffer per session |
| `OS_REALTIME_IDLE_TIMEOUT_S` | `120` | Realtime idle timeout |
| `OS_MODEL_TTL` | `300` | Auto-unload idle STT/TTS model TTL, including defaults; `0` = never |
| `OS_MAX_LOADED_MODELS` | `0` | Max loaded STT+TTS models; `0` = unlimited |
| `OS_STREAM_CHUNK_MS` | `100` | Streaming chunk window |
| `OS_STREAM_VAD_THRESHOLD` | `0.5` | Streaming VAD threshold |
| `OS_STREAM_ENDPOINTING_MS` | `300` | Silence to finalize utterance |
| `OS_STREAM_MAX_CONNECTIONS` | `10` | Max concurrent streaming WS sessions |

### `STT_*` — speech-to-text

| Variable | Default | Description |
|---|---|---|
| `STT_MODEL` | `Systran/faster-whisper-base` | Default STT model |
| `STT_DEVICE` | `cpu` | STT inference device |
| `STT_COMPUTE_TYPE` | `int8` | Compute precision |
| `STT_MODEL_DIR` | `None` | Optional local model directory |
| `STT_PRELOAD_MODELS` | `""` | Comma-separated models to preload |
| `STT_VAD_ENABLED` | `true` | Enable VAD by default for streaming |
| `STT_VAD_THRESHOLD` | `0.5` | VAD speech probability threshold |
| `STT_VAD_MIN_SPEECH_MS` | `250` | Minimum speech duration |
| `STT_VAD_SILENCE_MS` | `800` | Silence duration before speech end |
| `STT_DIARIZE_ENABLED` | `false` | Enable diarization support |
| `STT_NOISE_REDUCE` | `false` | Enable denoise preprocessing |
| `STT_NORMALIZE` | `true` | Normalize input audio |

### `TTS_*` — text-to-speech

| Variable | Default | Description |
|---|---|---|
| `TTS_ENABLED` | `true` | Enable TTS endpoints |
| `TTS_MODEL` | `kokoro` | Default TTS model |
| `TTS_VOICE` | `af_heart` | Default voice |
| `TTS_DEVICE` | `None` | TTS device override; falls back to STT device. Base Compose sets `cpu`; GPU override sets `cuda` |
| `TTS_MAX_INPUT_LENGTH` | `4096` | Max text length |
| `TTS_DEFAULT_FORMAT` | `mp3` | Default output format |
| `TTS_SPEED` | `1.0` | Default speed |
| `TTS_PRELOAD_MODELS` | `""` | Comma-separated TTS models to preload |
| `TTS_EXTERNAL_PROVIDERS` | `""` | JSON map of isolated provider IDs to private worker URLs |
| `TTS_VOICES_CONFIG` | `""` | YAML voice preset path |
| `TTS_CACHE_ENABLED` | `false` | Enable on-disk cache |
| `TTS_CACHE_MAX_MB` | `500` | Cache size budget |
| `TTS_CACHE_DIR` | `/var/lib/open-speech/cache` | Cache directory |
| `TTS_TRIM_SILENCE` | `true` | Trim generated silence |
| `TTS_NORMALIZE_OUTPUT` | `true` | Normalize output loudness |
| `TTS_PRONUNCIATION_DICT` | `""` | Pronunciation dictionary path |
| `TTS_LIVE_ENABLED` | `true` | Enable the incremental Live Reader WebSocket |
| `TTS_LIVE_MAX_CONNECTIONS` | `1` | Concurrent Live Reader sessions per process |
| `TTS_LIVE_IDLE_TIMEOUT_S` | `300` | Close a connection without client events after this many seconds |
| `TTS_LIVE_MAX_BUFFER_CHARS` | `8192` | Text waiting in the session's segmenter-input queue |
| `TTS_LIVE_MAX_PENDING_SEGMENTS` | `3` | Synthesizable text segments waiting behind the active segment |
| `TTS_LIVE_MAX_UNACKED_SECONDS` | `15` | Pause outgoing audio after this much unconsumed playback |
| `TTS_LIVE_AUDIO_FRAME_MS` | `50` | Approximate PCM duration in each audio delta |
| `TTS_LIVE_SEGMENT_IDLE_MS` | `250` | Input idle delay for responsive and word modes |
| `TTS_LIVE_MAX_SEGMENT_CHARS` | `200` | Responsive/word mode character cap per synthesis call |
| `TTS_LIVE_MAX_SEGMENT_WORDS` | `20` | Responsive/word mode word cap per synthesis call |
| `TTS_LIVE_SENTENCE_IDLE_MS` | `1500` | Input idle delay before full-sentence mode releases incomplete prose |
| `TTS_LIVE_SENTENCE_MAX_CHARS` | `500` | Full-sentence mode character cap per synthesis call |
| `TTS_LIVE_SENTENCE_MAX_WORDS` | `80` | Full-sentence mode word cap per synthesis call |
| `TTS_LIVE_MAX_PAUSE_S` | `900` | Maximum continuous client playback pause |
| `TTS_LIVE_PLAYBACK_STALL_TIMEOUT_S` | `30` | Close an unpaused client that stops acknowledging audio |
| `TTS_LIVE_SHUTDOWN_TIMEOUT_S` | `5` | Maximum teardown wait before a stuck worker drains in background |

## Wyoming

Open Speech can expose STT/TTS over the [Wyoming protocol](https://github.com/rhasspy/wyoming).

```bash
OS_WYOMING_ENABLED=true
OS_WYOMING_HOST=127.0.0.1
OS_WYOMING_PORT=10400
```

Example Home Assistant config:

```yaml
wyoming:
  - host: "YOUR_OPEN_SPEECH_IP"
    port: 10400
```

## Python SDK Example

```python
import httpx
from openai import OpenAI

client = OpenAI(
    base_url="https://localhost:8100/v1",
    api_key="not-needed",
    http_client=httpx.Client(verify=False),
)

with open("audio.wav", "rb") as f:
    result = client.audio.transcriptions.create(
        model="deepdml/faster-whisper-large-v3-turbo-ct2",
        file=f,
    )
print(result.text)

speech = client.audio.speech.create(
    model="kokoro",
    input="Hello world",
    voice="af_heart",
)
speech.stream_to_file("output.mp3")
```

## Status

Current release: **v0.8.0**. The running version is available from `GET /health`; run `pytest -q` for
the current test result instead of relying on a documentation snapshot.

## License

[MIT](LICENSE) © 2026 Jeremy Windsor
