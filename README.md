# Open Speech

**OpenAI-compatible speech server for faster-whisper STT and local TTS backends.**

[![Version](https://img.shields.io/badge/version-0.7.0-blue?style=flat-square)]()
[![Docker Hub](https://img.shields.io/docker/pulls/jwindsor1/open-speech?style=flat-square&logo=docker)](https://hub.docker.com/r/jwindsor1/open-speech)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-585%20passing-brightgreen?style=flat-square)]()
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue?style=flat-square&logo=python)](https://python.org)

## What is Open Speech?

Open Speech is a self-hosted speech API that exposes OpenAI-style endpoints for:

- **Speech-to-text** via `faster-whisper`
- **Text-to-speech** via local backends such as **Kokoro**, **Piper**, and **Pocket-TTS**
- **Streaming STT** over WebSocket
- **Audio-focused realtime I/O** over `/v1/realtime`
- **Batch jobs**, a **web UI**, **history/profiles/conversations/composer**, and **Wyoming** integration

This repo is not a universal provider gateway. It is a pragmatic local speech server with a compatible API surface. Software should be allowed to tell the truth once in a while.

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
- Local TTS backends: Kokoro, Piper, Pocket-TTS
- Disk-backed TTS cache
- Pronunciation dictionary + basic SSML parsing
- Output post-processing (trim silence, normalize)
- Voice presets for the web UI
- Kokoro voice blending using the `voice` field, e.g. `af_bella(2)+af_sky(1)`

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

> **Voice blending note:** the API uses the `voice` field for Kokoro blends. There is **no** separate `voice_blend` request field on the server.

#### `POST /v1/audio/speech/clone`

Multipart form endpoint that forwards reference audio to backends that support it.
The route exists, but the built-in local backends in this tree do not currently provide a broad, production-ready voice cloning story.

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
| `GET` | `/api/voices/library/{name}` | Get voice ref metadata |
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
- **Speak** — text input, voice selection, synthesis
- **Models** — load/unload/download known models
- **History / Settings** — runtime convenience features

The web UI has a Kokoro blend builder, but the server contract is still the plain `voice` string. In other words: the UI helps compose `af_bella(2)+af_sky(1)`, and the API only knows about `voice="af_bella(2)+af_sky(1)"`.

## Voice Blending

Kokoro supports weighted voice syntax:

```bash
curl -sk https://localhost:8100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro","input":"Hello","voice":"af_bella(2)+af_sky(1)"}'
```

## Docker Compose

### CPU

```yaml
services:
  open-speech:
    image: jwindsor1/open-speech:cpu
    ports: ["8100:8100"]
    environment:
      - STT_MODEL=Systran/faster-whisper-base
      - STT_DEVICE=cpu
      - TTS_MODEL=kokoro
      - TTS_DEVICE=cpu
    volumes:
      - hf-cache:/root/.cache/huggingface
volumes:
  hf-cache:
```

### GPU

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
      - hf-cache:/root/.cache/huggingface
volumes:
  hf-cache:
```

### Volumes

Model caches live under `/root/.cache/huggingface` inside the container.
Persist that path unless you enjoy re-downloading large things for sport.

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
```

## Environment Variables

Defaults come from `src/config.py`.

### `OS_*` — server / shared

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
| `OS_VOICE_LIBRARY_PATH` | `/home/openspeech/data/voices` | Stored voice reference directory |
| `OS_VOICE_LIBRARY_MAX_COUNT` | `100` | Max stored voice refs; `0` = unlimited |
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
| `OS_MODEL_TTL` | `300` | Auto-unload idle model TTL |
| `OS_MAX_LOADED_MODELS` | `0` | Max loaded models; `0` = unlimited |
| `OS_STREAM_CHUNK_MS` | `100` | Streaming chunk window |
| `OS_STREAM_VAD_THRESHOLD` | `0.5` | Streaming VAD threshold |
| `OS_STREAM_ENDPOINTING_MS` | `300` | Silence to finalize utterance |
| `OS_STREAM_MAX_CONNECTIONS` | `10` | Max concurrent streaming WS sessions |

### `STT_*` — speech-to-text

| Variable | Default | Description |
|---|---|---|
| `STT_MODEL` | `deepdml/faster-whisper-large-v3-turbo-ct2` | Default STT model |
| `STT_DEVICE` | `cuda` | STT inference device |
| `STT_COMPUTE_TYPE` | `float16` | Compute precision |
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
| `TTS_DEVICE` | `None` | TTS device override; falls back to STT device |
| `TTS_MAX_INPUT_LENGTH` | `4096` | Max text length |
| `TTS_DEFAULT_FORMAT` | `mp3` | Default output format |
| `TTS_SPEED` | `1.0` | Default speed |
| `TTS_PRELOAD_MODELS` | `""` | Comma-separated TTS models to preload |
| `TTS_VOICES_CONFIG` | `""` | YAML voice preset path |
| `TTS_CACHE_ENABLED` | `false` | Enable on-disk cache |
| `TTS_CACHE_MAX_MB` | `500` | Cache size budget |
| `TTS_CACHE_DIR` | `/var/lib/open-speech/cache` | Cache directory |
| `TTS_TRIM_SILENCE` | `true` | Trim generated silence |
| `TTS_NORMALIZE_OUTPUT` | `true` | Normalize output loudness |
| `TTS_PRONUNCIATION_DICT` | `""` | Pronunciation dictionary path |

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

Current code reports **v0.7.0** and the test suite currently passes **585 tests**.

## License

[MIT](LICENSE) © 2026 Jeremy Windsor
