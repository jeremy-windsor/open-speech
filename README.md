# Open Speech

**Self-hosted speech-model harness with OpenAI-compatible APIs.**

[![Version](https://img.shields.io/badge/version-0.8.0-blue?style=flat-square)](CHANGELOG.md)
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
- Optional diarization (`STT_DIARIZE_ENABLED=true` + the `diarize` extra, which installs pyannote)
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
- Named voices with explicit per-model realizations, separate reading presets, and preserved legacy profiles
- Kokoro voice blending using the `voice` field, e.g. `af_bella(2)+af_sky(1)`
- Model-specific capabilities and voice catalogs so controls only appear when the selected model supports them
- Isolated Qwen3, Chatterbox, and CosyVoice GPU providers without adding their conflicting
  Torch and Transformers pins to the core harness
- Machine-readable TTS conformance report for provider metadata, controls, voices, and opt-in audio checks

See [Named voices and reading presets](docs/VOICE-IDENTITIES.md) for migration behavior
and speech API usage.

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

The image sets some defaults of its own that differ from the source defaults in `src/config.py`;
see [Docker image defaults](#docker-image-defaults).

## Use it from an OpenAI app

Apps and SDKs written for OpenAI's audio API can use Open Speech without code changes. Point them at:

| Setting | Value |
|---|---|
| Base URL | `https://HOST:8100/v1` |
| API key | Any non-empty string. If `OS_API_KEY` is set on the server, use that key. |
| TTS model | `tts-1`, `tts-1-hd`, and `gpt-4o-mini-tts` use the configured `TTS_MODEL` (default `kokoro`). An Open Speech ID such as `kokoro` or `piper/en_US-lessac-medium` picks that model. |
| STT model | `whisper-1`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, or no model at all use the configured `STT_MODEL`. An Open Speech ID such as `deepdml/faster-whisper-large-v3-turbo-ct2` picks that model. |
| Voice | Kokoro maps OpenAI voice names to its own voices (see below). Other models need their own voice IDs. |

The STT name mapping applies to `/v1/audio/transcriptions`, `/v1/audio/translations`,
`/v1/audio/transcriptions/batch`, and the `/v1/audio/stream` WebSocket.

OpenAI voice names map to Kokoro voices: `alloy` → `af_heart`, `echo` → `am_adam`, `fable` → `bf_emma`,
`onyx` → `am_michael`, `nova` → `af_nova`, `shimmer` → `af_bella`. Piper models also accept `alloy` as
their default speaker. Other OpenAI voice names, and other models (Pocket-TTS, Qwen3, Chatterbox,
CosyVoice), need a voice ID from `GET /v1/audio/voices?model=MODEL`. If you change `TTS_MODEL` away
from Kokoro, apps that send `voice: "alloy"` will get an error unless the new model accepts it.

`/v1/audio/speech` fills in fields the client leaves out from the server settings: `model` from
`TTS_MODEL`, `speed` from `TTS_SPEED`, and `response_format` from `TTS_DEFAULT_FORMAT`. `voice` comes
from `TTS_VOICE` only when the model also came from `TTS_MODEL` (left out, or an OpenAI model name),
because a voice ID belongs to one model. Anything the client sends explicitly wins. Unknown JSON fields
are ignored, so newer OpenAI parameters do not break requests; the retired `voice_blend` field is still
rejected with `422`.

**HTTPS note:** Open Speech uses a self-signed certificate by default, and most third-party apps refuse
self-signed certificates. Pick one:

- On a trusted home network, turn HTTPS off with `OS_SSL_ENABLED=false` and use `http://HOST:8100/v1`.
- Put Open Speech behind a reverse proxy (Caddy, nginx, Traefik) that has a real certificate.
- Give Open Speech a trusted certificate with `OS_SSL_CERTFILE` and `OS_SSL_KEYFILE`.

## Installation (from source)

Open Speech supports **Python 3.12 only**. Install [uv](https://docs.astral.sh/uv/), then:

```bash
git clone https://github.com/jeremy-windsor/open-speech.git
cd open-speech
uv sync --frozen --extra cpu --extra all  # Locked CPU runtime and core TTS providers
# Use --extra cuda instead of --extra cpu for CUDA 12.8.
# Add --extra dev for development, or select tts, piper, or pocket individually.
```

The source defaults store data under `/home/openspeech/data`, the TTS cache under
`/var/lib/open-speech/cache`, and generated TLS certs under `/var/lib/open-speech/certs`. Those paths
suit the Docker image but usually are not writable on your own machine, so point them at a local
folder. The server does not read `.env` by itself; `uv run --env-file` loads it. Save this as `.env`
in the repository root:

```bash
OS_SSL_ENABLED=false
OS_STUDIO_DB_PATH=data/studio.db
OS_VOICE_LIBRARY_PATH=data/voices
OS_CONVERSATIONS_DIR=data/conversations
OS_COMPOSER_DIR=data/composer
TTS_CACHE_DIR=data/cache
```

Then start the server:

```bash
uv run --frozen --env-file .env python -m src.main
```

It listens on `http://localhost:8100` (web UI at `/web`). To keep HTTPS, remove `OS_SSL_ENABLED=false`
and set `OS_SSL_CERTFILE` and `OS_SSL_KEYFILE` to a writable cert/key path; Open Speech generates a
self-signed pair there if the files do not exist yet.

## Models

Models are downloaded on demand and cached on disk.
The tables below highlight common IDs rather than the complete curated catalog. `src/model_registry.py`
contains all 14 STT and 36 TTS entries; `/api/models` reports the catalog available to the configured
running harness.

### STT Models

| Model | Size | Backend | Languages |
|---|---:|---|---|
| `deepdml/faster-whisper-large-v3-turbo-ct2` | ~1.6GB | faster-whisper | 99+ |
| `Systran/faster-whisper-large-v3` | ~3.0GB | faster-whisper | 99+ |
| `Systran/faster-whisper-medium` | ~1.5GB | faster-whisper | 99+ |
| `Systran/faster-whisper-small` | ~500MB | faster-whisper | 99+ |
| `Systran/faster-whisper-base` | ~150MB | faster-whisper | 99+ |
| `Systran/faster-whisper-tiny` | ~75MB | faster-whisper | 99+ |

### TTS Models

| Model | Size | Backend | Notes |
|---|---:|---|---|
| `kokoro` | ~330MB | Kokoro | default backend, many voices, blend syntax in `voice` |
| `pocket-tts` | ~220MB | Pocket-TTS | built-in voices, backend advertises streaming support |
| `piper/en_US-lessac-medium` | ~35MB | Piper | one voice per model |
| `piper/en_US-joe-medium` | ~35MB | Piper | one voice per model |
| `piper/en_US-amy-medium` | ~35MB | Piper | one voice per model |
| `piper/en_US-arctic-medium` | ~35MB | Piper | one voice per model |
| `piper/en_GB-alan-medium` | ~35MB | Piper | one voice per model |
| `qwen3/0.6b-custom-voice` | ~1.8GB | isolated Qwen3 provider | 9 official preset voices |
| `qwen3/0.6b-base` | ~1.8GB | isolated Qwen3 provider | reference cloning with an exact transcript; hidden unless `QWEN3_ENABLE_BASE=true` (the voice-models Compose file sets it) |
| `chatterbox/regular` | ~8.6GiB | isolated Chatterbox provider | English reference cloning |
| `chatterbox/turbo` | ~5.4GiB | isolated Chatterbox provider | faster English cloning, native speech tags |
| `cosyvoice/2-0.5b` | ~4.6GiB | isolated CosyVoice provider | multilingual cloning, instructions, speed control |
| `cosyvoice/3-0.5b` | ~9.4GiB | isolated CosyVoice provider | multilingual cloning, instructions, native streaming |

The optional model sizes are approximate disk cache usage, not model VRAM. First load may
include model downloads. See [TTS Backends](docs/TTS-BACKENDS.md) for the provider bundle
and conformance commands.

## API Reference

API endpoints use Bearer auth when `OS_API_KEY` is set. `/health`, `/docs`, `/openapi.json`,
`/redoc`, `/web`, and web static assets remain unauthenticated. The bundled web UI does not currently
collect or attach an API key, so its shell loads but its API and WebSocket features do not work when
`OS_API_KEY` is enabled; use an authenticated client for that deployment mode.
For clients that cannot set headers (such as browser WebSockets), the server also accepts a
deprecated `?api_key=KEY` query parameter. It logs a warning each time and can leak the key into
logs and browser history, so prefer the `Authorization: Bearer` header.
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

**Query params:** `model` (default `STT_MODEL`; OpenAI names map to it), `language`, `sample_rate`
(default `16000`), `encoding` (default `pcm_s16le`), `interim_results` (default `true`), `endpointing`
(milliseconds of silence that end an utterance; default `OS_STREAM_ENDPOINTING_MS`), and `vad`
(default `STT_VAD_ENABLED`).

```javascript
const ws = new WebSocket("wss://localhost:8100/v1/audio/stream?vad=true");
ws.onmessage = (e) => console.log(JSON.parse(e.data));
ws.onopen = () => ws.send(audioChunkArrayBuffer);
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
- `voice` — voice ID string or an object with `id`; named-voice UUIDs resolve their model realization
- `speed`
- `response_format` = `mp3 | opus | aac | flac | wav | pcm | m4a`
- `stream_format` = `audio | sse` — enables byte streaming or base64 `speech.audio.delta` events
- `language`
- `input_type` = `text | ssml`
- `instructions` *(backend-gated)*
- `voice_design` *(backend-gated)*
- `reference_audio` *(base64-encoded; backend-gated)*
- `voice_library_ref` *(stored provider-neutral reference; backend-gated)*
- `clone_transcript` *(backend-gated)*
- `effects`

**Query params:** `stream`, `cache`

Left-out `model`, `voice`, `speed`, and `response_format` come from the `TTS_*` settings, and unknown
fields are ignored; see [Use it from an OpenAI app](#use-it-from-an-openai-app) for the exact rules.

Speech PCM is always signed 16-bit little-endian mono at 24 kHz, including Piper
models with a different native rate. SSE ends with `speech.audio.done`; local
providers do not supply OpenAI token usage counts. Piper speaker IDs come from
`/v1/audio/voices?model=piper/MODEL` and are validated against that model.

```bash
curl -sk https://localhost:8100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro","input":"Hello world","voice":"af_heart"}' \
  -o output.mp3
```

> **Voice blending note:** the API uses the `voice` field for Kokoro blends. There is **no** separate `voice_blend` request field in the harness contract.

#### `POST /v1/audio/speech/clone`

Multipart form endpoint that forwards reference audio to backends that support it.
Cloning needs a clone-capable model: Chatterbox, CosyVoice, or Qwen3 Base, run as isolated providers
with `docker-compose.voice-models.yml`. The built-in Kokoro, Piper, and Pocket-TTS backends do not clone.
It accepts required `input` plus `model`, `reference_audio`, `voice_library_ref`, `voice`, `speed`,
`response_format`, `transcript`, and `language` form fields. Supply either an uploaded reference or a
stored `voice_library_ref` when the selected backend requires one.

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

The current OpenAI Python SDK connects without a WebSocket subprotocol override.
GA sessions use nested `session.audio.input` and `session.audio.output` settings
and `response.output_audio.delta` events. PCM at 24 kHz, PCMU and PCMA are supported;
input transcription selects its own model through `audio.input.transcription`.
Existing project clients offering the `realtime` subprotocol retain legacy events
until they send a nested GA session update. Response voice overrides are honored.

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
| `POST` | `/api/voices/identities` | Create a named voice |
| `GET` | `/api/voices/identities` | List named voices; `?model=` reports availability for one model, `?check_availability=false` skips provider checks |
| `PATCH` | `/api/voices/identities/{id}` | Rename a named voice |
| `DELETE` | `/api/voices/identities/{id}` | Delete a named voice |
| `POST` | `/api/voices/identities/{id}/realizations` | Add a model-specific realization |
| `PUT` | `/api/voices/identities/{id}/realizations/{model}` | Update a realization |
| `DELETE` | `/api/voices/identities/{id}/realizations/{model}` | Remove a realization |
| `POST` | `/api/profiles` | Create profile (reading preset) |
| `GET` | `/api/profiles` | List profiles |
| `GET` | `/api/profiles/{id}` | Get profile |
| `GET` | `/api/profiles/{id}/resolve` | Resolve a profile's exact voice recipe |
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

See [Named voices and reading presets](docs/VOICE-IDENTITIES.md) for request bodies and examples.

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
- **History** — search, page through, re-generate, or delete saved STT/TTS entries
- **Studio** — assemble conversations and multi-track compositions, then render and download them
- **Settings** — manage voice profiles and review server-side history settings

The web UI has a Kokoro blend builder, but the harness contract is still the plain `voice` string. In other words: the UI helps compose `af_bella(2)+af_sky(1)`, and the API only knows about `voice="af_bella(2)+af_sky(1)"`.

Speak opens on the configured reading-default model (normally Kokoro), not whichever optional model
was most recently loaded. Switching to another TTS model for Generate or Live Reader confirms that
the currently loaded TTS model will be unloaded. **Restore Kokoro** reselects and, if needed, loads the reading
default. Open Speech keeps only one TTS model loaded at a time, whatever the GPU size.

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

It defaults to `STT_DEVICE=cpu`, `STT_COMPUTE_TYPE=int8`, and `TTS_DEVICE=cpu`. The base
Compose launch also selects the Turbo STT model, enables Wyoming on `0.0.0.0`, limits the shared
model manager to two loaded models, and uses a 2,000 ms streaming STT chunk window. These are
Compose launch defaults, not the source defaults listed under Environment Variables below.

`docker-compose.cpu.yml` extends the base file with the same CPU image, ports, and volumes, but changes
four settings: `STT_MODEL=Systran/faster-whisper-base` (smaller and faster on CPU), Wyoming off
(`OS_WYOMING_ENABLED=false`, host `127.0.0.1`), and no cap on loaded models (`OS_MAX_LOADED_MODELS=0`):

```bash
docker compose -f docker-compose.cpu.yml up -d
```

### Docker image defaults

The Docker image sets its own environment defaults, which differ from `src/config.py`. A plain
`docker run` uses these unless you pass `-e` overrides (the Compose files set most of them again):

| Setting | Image default | Source default |
|---|---|---|
| `OS_WYOMING_ENABLED` | `true` | `false` |
| `OS_WYOMING_HOST` | `0.0.0.0` | `127.0.0.1` |
| `OS_MAX_LOADED_MODELS` | `2` | `0` (unlimited) |
| `STT_MODEL` | `Systran/faster-whisper-base` (`:cpu`), `deepdml/faster-whisper-large-v3-turbo-ct2` (`:latest`, CUDA) | `Systran/faster-whisper-base` |
| `STT_DEVICE` / `STT_COMPUTE_TYPE` / `TTS_DEVICE` | `cpu`/`int8`/`cpu` (`:cpu`), `cuda`/`float16`/`cuda` (`:latest`) | `cpu`/`int8`/unset |

The container entrypoint always sets `HF_HOME=/home/openspeech/.cache/huggingface` and
`STT_MODEL_DIR=/home/openspeech/.cache/huggingface/hub`, so values you pass for those two are replaced.
Wyoming listens on port `10400`; add `-p 10400:10400` to `docker run` if you use it.

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
model and clicking **Clone test** performs the load automatically. Loading a TTS model always
unloads the previously loaded one first, so only one TTS model is loaded at a time.

The Qwen3 provider's `qwen3/0.6b-base` cloning model is hidden unless `QWEN3_ENABLE_BASE=true`;
`docker-compose.voice-models.yml` sets it to `true`.

#### Voice provider settings

These go in `.env` next to the Compose files. The Compose files and the provider workers in
`providers/*/app.py` read them; they are not Open Speech server settings.

| Variable | Default | Description |
|---|---|---|
| `OPEN_SPEECH_IMAGE` | `jwindsor1/open-speech:cpu` (`:latest` with the GPU override) | Core harness image tag |
| `OPEN_SPEECH_REVISION` | `unknown` | Git revision recorded by local image builds |
| `OPEN_SPEECH_CHATTERBOX_IMAGE` | `open-speech-chatterbox:0.1.0` | Chatterbox worker image tag |
| `OPEN_SPEECH_COSYVOICE_IMAGE` | `open-speech-cosyvoice:0.1.0` | CosyVoice worker image tag |
| `OPEN_SPEECH_QWEN3_IMAGE` | `open-speech-qwen3:0.1.1` | Qwen3 worker image tag |
| `HF_TOKEN` | `""` | Optional Hugging Face token for gated downloads (core and workers) |
| `QWEN3_ENABLE_BASE` | `false` in the worker; `true` in the voice-models Compose file | Show the `qwen3/0.6b-base` cloning model |
| `QWEN3_DTYPE` | `auto` | `auto`, `float32`, `float16`, or `bfloat16` |
| `QWEN3_MAX_SEGMENT_UNITS` | `400` | Script-weighted text per Qwen segment |
| `QWEN3_MAX_NEW_TOKENS_CEILING` | `1200` | Qwen generation ceiling (minimum 192) |
| `QWEN3_PROMPT_CACHE_SIZE` | `8` | Cached reference prompts per worker |
| `QWEN3_STREAM_QUEUE_TIMEOUT_S` | `30` | Qwen streaming queue timeout |
| `CHATTERBOX_MAX_INPUT_CHARS` | `350` | Chatterbox per-request text limit |
| `COSYVOICE_MAX_INPUT_CHARS` | `1500` | CosyVoice per-request text limit |
| `COSYVOICE_FP16` | `true` | CosyVoice half precision on CUDA |
| `QWEN3_MAX_REFERENCE_MB`, `CHATTERBOX_MAX_REFERENCE_MB`, `COSYVOICE_MAX_REFERENCE_MB` | `100` | Max reference upload size; Compose sets all three from `OS_MAX_UPLOAD_MB` |

### Volumes

Model caches live under `/home/openspeech/.cache/huggingface` inside the container.
Persist that path unless you enjoy re-downloading large things for sport. The checked-in Compose files also persist Silero VAD, TLS certs, app data, and TTS cache volumes.

## Development and validation

Run ordinary tests from the repository root. Open Speech supports **Python 3.12 only**:

```bash
uv sync --frozen --extra cpu --extra dev
uv run --frozen --extra cpu --extra dev python -m pytest -q -rs
```

Native Windows uses the same suite:

```powershell
uv sync --frozen --extra cpu --extra dev
uv run --frozen --extra cpu --extra dev python -m pytest -q -rs
```

Put Node.js and ffmpeg on `PATH` to exercise browser behavior and real audio encoding.
Two Kokoro unit tests require the optional Torch runtime. Windows skips the two symlink
tests only when the account lacks symbolic-link privilege; Linux still exercises them.
Run the separate JavaScript SDK tests with `bun test client-js/tests`.
GitHub Actions (`.github/workflows/tests.yml`) runs the Python test suite on Python 3.12 for pushes
to `main` and for pull requests.

To check a running server end to end the way an OpenAI app would use it, run the smoke test. It needs
only the Python standard library (Python 3.9 or newer), so it works without installing anything:

```bash
python scripts/smoke_openai_api.py --url https://localhost:8100 --insecure
```

It checks `/health` and `/v1/models`, synthesizes a Kokoro WAV through `/v1/audio/speech`, transcribes
it back, checks the `text`, `srt`, `vtt`, and `verbose_json` formats, and tries the OpenAI model names
`tts-1` and `whisper-1`. `--insecure` accepts the self-signed certificate. If `OS_API_KEY` is set in
your shell, the script sends it. The first run can take minutes while model weights download; exit
code `0` means every check passed.

CPU and CUDA use the same Dockerfile and frozen `uv.lock`. Builds install only the selected
providers and download weights on first use into persistent model caches. `BAKED_TTS_MODELS`
can explicitly prefetch models. The UI ships in the image. `/health` reports its build revision.
After changing dependencies, run `uv lock` and export the pip-compatible core requirements
with `uv export --frozen --no-dev --no-emit-project --no-hashes --no-annotate --no-header -o requirements.lock`.

Build and validate an immutable CUDA tag on the GPU host:

```powershell
$Tag = "jwindsor1/open-speech:cuda-$(git rev-parse --short HEAD)"
docker build --build-arg DEVICE=cuda --build-arg REVISION=$(git rev-parse HEAD) -t $Tag .
docker run -d --rm --name open-speech-canary --gpus all -p 8110:8100 `
  -e OS_SSL_ENABLED=false -e OS_WYOMING_ENABLED=false `
  -e STT_DEVICE=cuda -e STT_COMPUTE_TYPE=float16 -e TTS_DEVICE=cuda $Tag
```

For CPU, build with `DEVICE=cpu`; `docker-compose.cpu.yml` is the CPU profile with smaller defaults.
Deploy a verified tag with `OPEN_SPEECH_IMAGE`; GPU Compose layers
`docker-compose.yml` and `docker-compose.gpu.yml`.

Backend-specific conformance commands are in
[docs/TTS-BACKENDS.md](docs/TTS-BACKENDS.md#validation).

## Security

```bash
# API key
OS_API_KEY=my-secret-key docker compose up -d
curl -sk https://localhost:8100/health  # public liveness check
curl -sk -H "Authorization: Bearer my-secret-key" https://localhost:8100/v1/models
# Deprecated fallback for clients that cannot send headers (logs a warning):
#   wss://localhost:8100/v1/audio/stream?api_key=my-secret-key

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

Defaults come from `src/config.py`. The Docker image and the checked-in Compose files override several
of them (see [Docker image defaults](#docker-image-defaults) and [Docker Compose](#docker-compose));
the GPU override changes the device settings to CUDA. A source install reads only the process
environment, so use `uv run --env-file .env` or export the variables yourself.

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
| `OS_MAX_UPLOAD_MB` | `100` | Max upload size; checked-in provider Compose files also use it for voice references |
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
| `OS_BATCH_WORKERS` | `2` | Concurrent batch worker count |
| `OS_BATCH_MAX_PENDING` | `10` | Max queued + running batch jobs |
| `OS_BATCH_MAX_TOTAL_MB` | `500` | Max aggregate upload size per batch request |
| `OS_WYOMING_ENABLED` | `false` | Enable Wyoming TCP server (Docker image and base Compose: `true`) |
| `OS_WYOMING_HOST` | `127.0.0.1` | Wyoming bind host (Docker image and base Compose: `0.0.0.0`) |
| `OS_WYOMING_PORT` | `10400` | Wyoming port |
| `OS_REALTIME_ENABLED` | `true` | Enable `/v1/realtime` |
| `OS_REALTIME_MAX_BUFFER_MB` | `50` | Max realtime audio buffer per session |
| `OS_REALTIME_IDLE_TIMEOUT_S` | `120` | Realtime idle timeout |
| `OS_MODEL_TTL` | `300` | Auto-unload idle STT/TTS model TTL, including defaults; `0` = never |
| `OS_MAX_LOADED_MODELS` | `0` | Max loaded STT+TTS models; `0` = unlimited |
| `OS_STREAM_CHUNK_MS` | `100` | Streaming chunk window |
| `OS_STREAM_ENDPOINTING_MS` | `300` | Silence (ms) that ends an utterance; default for the `/v1/audio/stream` `endpointing` query param |
| `OS_STREAM_MAX_CONNECTIONS` | `10` | Max concurrent streaming WS sessions |

### `STT_*` — speech-to-text

| Variable | Default | Description |
|---|---|---|
| `STT_MODEL` | `Systran/faster-whisper-base` | STT model used when a request omits `model` or sends an OpenAI name such as `whisper-1`; loaded on first use |
| `STT_DEVICE` | `cpu` | STT inference device |
| `STT_COMPUTE_TYPE` | `int8` | Compute precision |
| `STT_MODEL_DIR` | `None` | Optional local model directory |
| `STT_PRELOAD_MODELS` | `""` | Comma-separated models to preload |
| `STT_VAD_ENABLED` | `true` | Enable VAD by default for streaming |
| `STT_VAD_THRESHOLD` | `0.5` | VAD speech probability threshold, including streaming |
| `STT_VAD_MIN_SPEECH_MS` | `250` | Minimum speech duration |
| `STT_VAD_SILENCE_MS` | `800` | Silence duration before speech end |
| `STT_DIARIZE_ENABLED` | `false` | Enable diarization support |
| `STT_NOISE_REDUCE` | `false` | Enable denoise preprocessing |
| `STT_NORMALIZE` | `true` | Normalize input audio |

### `TTS_*` — text-to-speech

| Variable | Default | Description |
|---|---|---|
| `TTS_ENABLED` | `true` | Enable TTS endpoints |
| `TTS_MODEL` | `kokoro` | TTS model used when `/v1/audio/speech` omits `model` or sends `tts-1`, `tts-1-hd`, or `gpt-4o-mini-tts`; also the web UI reading default |
| `TTS_VOICE` | `af_heart` | Voice used when `/v1/audio/speech` omits `voice` and the model came from `TTS_MODEL`; must be valid for `TTS_MODEL` |
| `TTS_DEVICE` | `None` | TTS device override; falls back to STT device. Base Compose sets `cpu`; GPU override sets `cuda` |
| `TTS_MAX_INPUT_LENGTH` | `4096` | Max text length |
| `TTS_DEFAULT_FORMAT` | `mp3` | Output format used when `/v1/audio/speech` omits `response_format` |
| `TTS_SPEED` | `1.0` | Speed used when `/v1/audio/speech` omits `speed` |
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
OS_WYOMING_HOST=0.0.0.0   # inside Docker; use 127.0.0.1 to allow only local clients on a source install
OS_WYOMING_PORT=10400
```

The Docker image and base Compose file already enable Wyoming on `0.0.0.0:10400`;
`docker-compose.cpu.yml` and a source install leave it off. In Docker, publish port `10400`.

Example Home Assistant config:

```yaml
wyoming:
  - host: "YOUR_OPEN_SPEECH_IP"
    port: 10400
```

## Python SDK Example

This compatibility example uses the external `openai` package; install it separately with
`pip install openai`. The `client` project extra installs dependencies for the built-in
`src.client.OpenSpeechClient` instead.

```python
import httpx
from openai import OpenAI

client = OpenAI(
    base_url="https://localhost:8100/v1",
    api_key="not-needed",  # use your OS_API_KEY if one is set
    http_client=httpx.Client(verify=False),  # accept the self-signed cert
)

with open("audio.wav", "rb") as f:
    result = client.audio.transcriptions.create(
        model="deepdml/faster-whisper-large-v3-turbo-ct2",  # or "whisper-1" for STT_MODEL
        file=f,
    )
print(result.text)

with client.audio.speech.with_streaming_response.create(
    model="kokoro",  # or "tts-1" for TTS_MODEL
    input="Hello world",
    voice="af_heart",
) as response:
    response.stream_to_file("output.mp3")
```

## Status

Current release: **v0.8.0**. The running version is available from `GET /health`; run `pytest -q` for
the current test result instead of relying on a documentation snapshot.

## License

[MIT](LICENSE) © 2026 Jeremy Windsor
