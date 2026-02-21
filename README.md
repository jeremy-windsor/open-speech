# Open Speech

**OpenAI-compatible speech server — any STT/TTS provider, one container.**

[![Version](https://img.shields.io/badge/version-0.6.0-blue?style=flat-square)]()
[![Docker Hub](https://img.shields.io/docker/pulls/jwindsor1/open-speech?style=flat-square&logo=docker)](https://hub.docker.com/r/jwindsor1/open-speech)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-593%20passing-brightgreen?style=flat-square)]()
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue?style=flat-square&logo=python)](https://python.org)

## What is Open Speech?

Open Speech is a self-hosted speech server that speaks the OpenAI API. Plug in any STT or TTS backend, swap models at runtime, and hit the same endpoints your apps already use. One Docker image, CPU or GPU, no vendor lock-in.

## Features

**🎙️ Speech-to-Text**
- OpenAI-compatible `/v1/audio/transcriptions` and `/v1/audio/translations`
- Real-time streaming via WebSocket (`/v1/audio/stream`)
- Silero VAD — only transcribe when someone's talking
- SRT/VTT subtitle output
- Optional speaker diarization (`?diarize=true`) via pyannote
- Optional input audio preprocessing (noise reduction + normalization)

**🔊 Text-to-Speech**
- OpenAI-compatible `/v1/audio/speech`
- Streaming TTS with chunked transfer
- 50+ voices across backends
- Voice blending — mix voices with `af_bella(2)+af_sky(1)` syntax
- TTS response caching (configurable disk LRU)
- Pronunciation dictionary + SSML subset (`input_type=ssml`)
- Output postprocessing (silence trim + normalize)

**🧩 Qwen3-TTS Deep Integration (Phase 7a)**
- Official `qwen-tts` backend integration (`Qwen3TTSModel`)
- Three-model auto-selection per request:
  - `CustomVoice` for named premium speakers
  - `VoiceDesign` for instruction-only voice creation
  - `Base` for voice cloning from reference audio
- 9 premium speakers: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee
- Instruction control via `voice_design` field
- In-memory reusable clone-prompt cache (`create_voice_clone_prompt()`)
- On-demand model loading only (no startup model load)
- Practical GPU note: 8GB VRAM can run one 1.7B model comfortably; loading two 1.7B models is usually tight

**🧠 Model Management**
- Nothing baked in — models download at runtime
- Unified model browser in the web UI
- Load, unload, hot-swap models via API
- TTL eviction and LRU lifecycle

**🔌 OpenAI Realtime API**
- Drop-in replacement for OpenAI's `/v1/realtime` WebSocket endpoint
- Audio I/O only — STT + TTS, bring your own LLM
- Server VAD mode with Silero VAD for automatic speech detection
- Audio format negotiation: pcm16, g711_ulaw, g711_alaw
- Works with existing OpenAI Realtime API client libraries

**🐳 Deployment**
- Single image for CPU + GPU (NVIDIA CUDA)
- Web UI with light/dark mode at `/web`
- Self-signed HTTPS out of the box
- API key auth, rate limiting, CORS

## Quick Start

```bash
docker run -d -p 8100:8100 jwindsor1/open-speech:cpu
```

Open **https://localhost:8100/web** — accept the self-signed cert, and you're in.

> For GPU: `docker run -d -p 8100:8100 --gpus all jwindsor1/open-speech:latest`

## Installation (from source)

```bash
git clone https://github.com/will-assistant/open-speech.git
cd open-speech
pip install -e .                    # Core (faster-whisper STT + Kokoro TTS)
pip install -e ".[piper]"           # + Piper TTS
pip install -e ".[qwen]"            # + Qwen3-TTS deep integration (qwen-tts)
pip install -e ".[all]"             # All core backends (keeps heavy optional extras separate)
pip install -e ".[diarize]"         # + Speaker diarization (pyannote)
pip install -e ".[noise]"           # + Noise reduction preprocessing
pip install -e ".[client]"          # + Python client SDK deps
pip install -e ".[dev]"             # Development tools (pytest, ruff, httpx)
pip install -r requirements.lock      # Reproducible pinned core dependencies
```

## Configuration

All config via environment variables. `OS_` for server, `STT_` for speech-to-text, `TTS_` for text-to-speech.

### Server (`OS_`)

| Variable | Default | Description |
|----------|---------|-------------|
| `OS_HOST` | `0.0.0.0` | Bind address |
| `OS_PORT` | `8100` | Listen port |
| `OS_API_KEY` | | API key (empty = no auth) |
| `OS_CORS_ORIGINS` | `*` | Comma-separated CORS origins |
| `OS_SSL_ENABLED` | `true` | Enable HTTPS |
| `OS_MODEL_TTL` | `300` | Idle seconds before auto-unload |
| `OS_MAX_LOADED_MODELS` | `0` | Max models in memory (0 = unlimited) |
| `OS_RATE_LIMIT` | `0` | Requests/min per IP (0 = off) |
| `OS_MAX_UPLOAD_MB` | `100` | Max upload size |

### Speech-to-Text (`STT_`)

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_MODEL` | `deepdml/faster-whisper-large-v3-turbo-ct2` | Default STT model |
| `STT_DEVICE` | `cuda` | `cuda` or `cpu` |
| `STT_COMPUTE_TYPE` | `float16` | `float16`, `int8`, `int8_float16` |
| `STT_PRELOAD_MODELS` | | Comma-separated models to preload |

### Text-to-Speech (`TTS_`)

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_ENABLED` | `true` | Enable TTS endpoints |
| `TTS_MODEL` | `kokoro` | Default TTS model |
| `TTS_VOICE` | `af_heart` | Default voice |
| `TTS_SPEED` | `1.0` | Default speed |
| `TTS_DEVICE` | *(inherits STT)* | `cuda` or `cpu` |
| `TTS_MAX_INPUT_LENGTH` | `4096` | Max input text length |
| `TTS_VOICES_CONFIG` | | Path to voice presets YAML |
| `TTS_QWEN3_SIZE` | `1.7B` | Qwen3 model size: `1.7B` or `0.6B` |
| `TTS_QWEN3_FLASH_ATTN` | `false` | Enable flash-attention-2 when installed |
| `TTS_QWEN3_DEVICE` | `cuda:0` | Device override for qwen-tts model loading |

> **Backwards compatibility:** Old env var names (`STT_PORT`, `STT_HOST`, etc.) still work but log deprecation warnings.

### Voice Activity Detection (`STT_VAD_`)

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_VAD_ENABLED` | `true` | Enable Silero VAD for speech detection |
| `STT_VAD_THRESHOLD` | `0.5` | Speech probability threshold (0.0–1.0) |
| `STT_VAD_MIN_SPEECH_MS` | `250` | Minimum speech duration before triggering |
| `STT_VAD_SILENCE_MS` | `800` | Silence duration to trigger speech_end |

VAD (Voice Activity Detection) uses the [Silero VAD](https://github.com/snakers4/silero-vad) ONNX model (<2MB) to detect when someone is speaking. When enabled:

- **WebSocket streaming** only forwards speech to the STT backend, saving compute
- **VAD events** (`speech_start` / `speech_end`) are sent to WebSocket clients
- **Web UI mic** shows visual indicators for speech detection state
- **Wyoming protocol** filters silence before transcription

The VAD model downloads automatically on first use. It runs on CPU with negligible overhead (<5% CPU).

**WebSocket VAD query parameter:**

```
ws://host:8100/v1/audio/stream?vad=true    # force enable
ws://host:8100/v1/audio/stream?vad=false   # force disable
ws://host:8100/v1/audio/stream             # use STT_VAD_ENABLED default
```

### Wyoming Protocol (`OS_WYOMING_`)

| Variable | Default | Description |
|----------|---------|-------------|
| `OS_WYOMING_ENABLED` | `false` | Enable Wyoming TCP server |
| `OS_WYOMING_PORT` | `10400` | Wyoming protocol port |

Open Speech supports the [Wyoming protocol](https://github.com/rhasspy/wyoming), making it a drop-in STT + TTS provider for **Home Assistant**.

**Enable it:**

```bash
OS_WYOMING_ENABLED=true
OS_WYOMING_PORT=10400  # default
```

**Home Assistant setup:**

Add to your Home Assistant `configuration.yaml`:

```yaml
wyoming:
  - host: "YOUR_OPEN_SPEECH_IP"
    port: 10400
```

Or add via **Settings → Devices & Services → Add Integration → Wyoming Protocol** and enter your Open Speech server's IP and port 10400.

Once connected, Open Speech appears as both an STT and TTS provider in your voice pipeline configuration.

### OpenAI Realtime API (`OS_REALTIME_`)

| Variable | Default | Description |
|----------|---------|-------------|
| `OS_REALTIME_ENABLED` | `true` | Enable `/v1/realtime` WebSocket endpoint |

Open Speech implements the [OpenAI Realtime API](https://platform.openai.com/docs/api-reference/realtime) WebSocket protocol for **audio I/O only** (STT + TTS). Any app built for OpenAI's voice mode works as a drop-in replacement — just point it at your Open Speech server.

**We handle:** audio input → transcription, text → speech output, VAD-based turn detection.
**We don't handle:** LLM conversation, function calling, text generation. Bring your own brain.

**Usage example (Python):**

```python
import asyncio, base64, json, websockets

async def main():
    uri = "ws://localhost:8100/v1/realtime?model=whisper-large-v3-turbo"
    async with websockets.connect(uri, subprotocols=["realtime"]) as ws:
        event = json.loads(await ws.recv())  # session.created
        print(f"Session: {event['session']['id']}")

        # Send audio (PCM16, 24kHz, mono)
        with open("audio.raw", "rb") as f:
            audio = f.read()
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(audio).decode(),
        }))
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        # Receive transcription
        while True:
            event = json.loads(await ws.recv())
            if event["type"] == "conversation.item.input_audio_transcription.completed":
                print(f"Transcript: {event['transcript']}")
                break

asyncio.run(main())
```

**Drop-in replacement for OpenAI clients** — just change the base URL:

```python
# Point any OpenAI Realtime client at Open Speech
REALTIME_URL = "ws://localhost:8100/v1/realtime"
```

## Models

Models are **not baked into the image** — they download on first use and persist in the Docker volume.

### STT Models

| Model | Size | Backend | Languages |
|-------|------|---------|-----------|
| `deepdml/faster-whisper-large-v3-turbo-ct2` | ~800MB | faster-whisper | 99+ |
| `Systran/faster-whisper-large-v3` | ~1.5GB | faster-whisper | 99+ |
| `Systran/faster-whisper-medium` | ~800MB | faster-whisper | 99+ |
| `Systran/faster-whisper-small` | ~250MB | faster-whisper | 99+ |
| `Systran/faster-whisper-base` | ~150MB | faster-whisper | 99+ |
| `Systran/faster-whisper-tiny` | ~75MB | faster-whisper | 99+ |

### TTS Models

| Model | Size | Backend | Voices |
|-------|------|---------|--------|
| `kokoro` | ~82MB | Kokoro | 52 voices, blending |
| `pocket-tts` | ~220MB | Pocket TTS | 8 built-in voices, streaming |
| `piper/en_US-lessac-medium` | ~35MB | Piper | 1 |
| `piper/en_US-joe-medium` | ~35MB | Piper | 1 |
| `piper/en_US-amy-medium` | ~35MB | Piper | 1 |
| `piper/en_US-arctic-medium` | ~35MB | Piper | 1 |
| `piper/en_GB-alan-medium` | ~35MB | Piper | 1 |
| `qwen3-tts/0.6B-CustomVoice` | ~1.2GB | Qwen3-TTS | 4 + voice design |
| `qwen3-tts/1.7B-CustomVoice` | ~3.4GB | Qwen3-TTS | 4 + voice design |

Switch models by changing `STT_MODEL` / `TTS_MODEL` and restarting, or use the API:

```bash
curl -sk -X POST https://localhost:8100/api/models/Systran%2Ffaster-whisper-small/load
```

## API Reference

All endpoints return JSON unless noted. Authentication via `Authorization: Bearer <key>` header when `OS_API_KEY` is set. Interactive docs at `/docs` (Swagger UI).

### Speech-to-Text (STT)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/audio/transcriptions` | Transcribe audio to text (OpenAI-compatible) |
| `POST` | `/v1/audio/translations` | Translate audio to English text (OpenAI-compatible) |
| `WS` | `/v1/audio/stream` | Real-time streaming transcription via WebSocket |
| `GET` | `/v1/audio/stream` | Returns `426` — instructs HTTP clients to use WebSocket |

<details>
<summary><strong>POST /v1/audio/transcriptions</strong></summary>

Multipart form upload. Returns transcription in the requested format.

**Params** (form fields): `file` (required), `model`, `language`, `prompt`, `response_format` (`json`|`text`|`verbose_json`|`srt`|`vtt`), `temperature`, `diarize` (bool)

```bash
curl -sk https://localhost:8100/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=deepdml/faster-whisper-large-v3-turbo-ct2" \
  -F "response_format=json"
```

**Response** (`json`): `{ "text": "..." }`
**Response** (`verbose_json`): `{ "text", "segments": [{ "start", "end", "text" }], "language" }`
**Response** (`diarize=true`): `{ "text", "segments": [{ "speaker", "start", "end", "text" }] }`
</details>

<details>
<summary><strong>POST /v1/audio/translations</strong></summary>

Translates non-English audio to English. Same multipart interface as transcriptions.

**Params**: `file` (required), `model`, `prompt`, `response_format`, `temperature`

**Response**: `{ "text": "..." }`
</details>

<details>
<summary><strong>WS /v1/audio/stream</strong></summary>

Real-time streaming STT via WebSocket. Send raw audio chunks, receive transcript events.

**Query params**: `model`, `language`, `sample_rate` (default 16000), `encoding` (`pcm_s16le`), `interim_results` (bool), `endpointing` (ms), `vad` (bool)

```javascript
const ws = new WebSocket("wss://localhost:8100/v1/audio/stream?vad=true");
ws.onmessage = (e) => console.log(JSON.parse(e.data));
ws.send(audioChunkArrayBuffer); // PCM16 LE mono
```

**Events**: `{ "type": "transcript", "text", "is_final" }`, `{ "type": "speech_start" }`, `{ "type": "speech_end" }`
</details>

### Text-to-Speech (TTS)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/audio/speech` | Synthesize speech from text (OpenAI-compatible) |
| `POST` | `/v1/audio/speech/clone` | Voice cloning via multipart upload |
| `GET` | `/v1/audio/voices` | List available TTS voices |
| `GET` | `/v1/audio/models` | List TTS models and their load status |
| `POST` | `/v1/audio/models/load` | Load a TTS model into memory |
| `POST` | `/v1/audio/models/unload` | Unload a TTS model from memory |
| `GET` | `/api/tts/capabilities` | Backend capabilities for a TTS model |
| `GET` | `/api/voice-presets` | List configured voice presets |

<details>
<summary><strong>POST /v1/audio/speech</strong></summary>

JSON body. Returns audio bytes in the requested format.

**Body**: `model` (string), `input` (text to speak), `voice`, `speed` (float), `response_format` (`mp3`|`opus`|`aac`|`flac`|`wav`|`pcm`), `language`, `input_type` (`text`|`ssml`), `voice_design` (string, Qwen3 only), `reference_audio` (base64, Qwen3 only), `clone_transcript`, `effects` (array)
**Query**: `stream` (bool), `cache` (bool)

```bash
curl -sk https://localhost:8100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro","input":"Hello world","voice":"af_heart"}' \
  -o output.mp3
```

**Response**: Audio bytes with `Content-Type` matching format. Header `X-Cache: HIT` on cache hit.
</details>

<details>
<summary><strong>POST /v1/audio/speech/clone</strong></summary>

Multipart form upload for voice cloning with a reference audio sample.

**Params**: `input` (required), `model`, `reference_audio` (file upload), `voice_library_ref` (string — use stored voice instead of upload), `voice`, `speed`, `response_format`, `transcript`, `language`

```bash
curl -sk https://localhost:8100/v1/audio/speech/clone \
  -F "input=Hello, I sound like the reference!" \
  -F "model=qwen3-tts/0.6B-CustomVoice" \
  -F "reference_audio=@reference.wav" \
  -o cloned.mp3
```

**Response**: Audio bytes.
</details>

<details>
<summary><strong>GET /v1/audio/voices</strong></summary>

List available TTS voices. Optionally filter by model/provider.

**Query**: `model` (optional — filter by provider, e.g. `kokoro`, `piper`)

**Response**: `{ "voices": [{ "id", "name", "language", "gender" }] }`
</details>

<details>
<summary><strong>GET /v1/audio/models</strong></summary>

List TTS models and their runtime status (loaded/not loaded).

**Response**: `{ "models": [{ "model", "backend", "device", "status", "loaded_at", "last_used_at" }] }`
</details>

<details>
<summary><strong>POST /v1/audio/models/load</strong></summary>

**Body**: `{ "model": "kokoro" }` (optional — defaults to `TTS_MODEL`)

**Response**: `{ "status": "loaded", "model": "kokoro" }`
</details>

<details>
<summary><strong>POST /v1/audio/models/unload</strong></summary>

**Body**: `{ "model": "kokoro" }` (optional — defaults to `TTS_MODEL`)

**Response**: `{ "status": "unloaded", "model": "kokoro" }`
</details>

<details>
<summary><strong>GET /api/tts/capabilities</strong></summary>

Returns capability flags for the selected TTS backend.

**Query**: `model` (optional — defaults to `TTS_MODEL`)

**Response**: `{ "backend": "kokoro", "capabilities": { "voice_design": false, "voice_clone": false, ... } }`
</details>

### OpenAI Realtime API

| Method | Path | Description |
|--------|------|-------------|
| `WS` | `/v1/realtime` | OpenAI Realtime API-compatible audio WebSocket |

<details>
<summary><strong>WS /v1/realtime</strong></summary>

Drop-in replacement for OpenAI's Realtime API. Audio I/O only (STT + TTS). Requires `OS_REALTIME_ENABLED=true` (default).

**Query**: `model` (optional)
**Subprotocol**: `realtime`

**Client events**: `input_audio_buffer.append` (`{ "audio": "<base64>" }`), `input_audio_buffer.commit`, `response.create`
**Server events**: `session.created`, `conversation.item.input_audio_transcription.completed` (`{ "transcript" }`), `response.audio.delta`, `response.audio.done`
</details>

### Models (OpenAI-compatible)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/models` | List all available models (OpenAI format) |
| `GET` | `/v1/models/{model}` | Get model details |

<details>
<summary><strong>GET /v1/models</strong></summary>

Lists both STT and TTS models. Includes loaded models and configured defaults.

**Response**: `{ "object": "list", "data": [{ "id", "object": "model", "owned_by" }] }`
</details>

### Unified Model Management

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/models` | List all models (available + downloaded + loaded) |
| `GET` | `/api/models/{id}/status` | Get model state (loaded/downloaded/available) |
| `GET` | `/api/models/{id}/progress` | Download/load progress |
| `POST` | `/api/models/{id}/load` | Load a model into memory |
| `POST` | `/api/models/{id}/download` | Download model weights only |
| `POST` | `/api/models/{id}/prefetch` | Cache model weights (alias for download) |
| `DELETE` | `/api/models/{id}` | Unload a model from memory |
| `DELETE` | `/api/models/{id}/artifacts` | Delete cached model weights/artifacts |

<details>
<summary><strong>GET /api/models</strong></summary>

Returns all known models across STT and TTS with state info and TTS capabilities.

**Response**: `{ "models": [{ "id", "type", "backend", "state", "capabilities": {} }] }`
</details>

<details>
<summary><strong>POST /api/models/{id}/load</strong></summary>

```bash
curl -sk -X POST https://localhost:8100/api/models/kokoro/load
```

**Response**: `{ "id", "type", "backend", "state": "loaded" }`
**Error**: `{ "error": { "message", "code" } }` (400 or 500)
</details>

<details>
<summary><strong>GET /api/models/{id}/progress</strong></summary>

Poll download/load progress for a model.

**Response**: `{ "status": "loading"|"downloading"|"ready"|"idle", "progress": 0.0–1.0 }`
</details>

### Provider Management

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/providers/install` | Install provider Python dependencies (async) |
| `GET` | `/api/providers/install/{job_id}` | Poll provider install job status |

<details>
<summary><strong>POST /api/providers/install</strong></summary>

Starts an async pip install for a provider's dependencies. Returns a job ID to poll.

**Body**: `{ "provider": "piper", "model": null }`

**Response**: `{ "job_id": "uuid", "status": "installing" }`
</details>

<details>
<summary><strong>GET /api/providers/install/{job_id}</strong></summary>

**Response**: `{ "job_id", "status": "installing"|"done"|"failed", "output", "error" }`
</details>

### Voice Library

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/voices/library` | Upload a named voice reference for cloning |
| `GET` | `/api/voices/library` | List all stored voice references |
| `GET` | `/api/voices/library/{name}` | Get voice reference metadata |
| `DELETE` | `/api/voices/library/{name}` | Delete a stored voice reference |

<details>
<summary><strong>POST /api/voices/library</strong></summary>

Multipart upload. Stores a named voice reference audio for use with `voice_library_ref` in clone requests.

**Params**: `name` (form), `audio` (file upload — WAV format)

**Response** (201): `{ "name", "content_type", "size_bytes", "created_at" }`
</details>

### Voice Profiles (Studio)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/profiles` | Create a voice profile |
| `GET` | `/api/profiles` | List all profiles + default |
| `GET` | `/api/profiles/{id}` | Get a profile |
| `PUT` | `/api/profiles/{id}` | Update a profile |
| `DELETE` | `/api/profiles/{id}` | Delete a profile |
| `POST` | `/api/profiles/{id}/default` | Set as default profile |

<details>
<summary><strong>POST /api/profiles</strong></summary>

**Body**: `{ "name", "backend", "model", "voice", "speed", "format", "blend", "reference_audio_id", "effects": [] }`

**Response** (201): `{ "id", "name", "backend", "model", "voice", "speed", "format", ... }`
</details>

<details>
<summary><strong>GET /api/profiles</strong></summary>

**Response**: `{ "profiles": [{ "id", "name", "backend", "voice", ... }], "default_profile_id": "..." }`
</details>

### History

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/history` | List generation history entries |
| `DELETE` | `/api/history/{id}` | Delete a history entry |
| `DELETE` | `/api/history` | Clear all history |

<details>
<summary><strong>GET /api/history</strong></summary>

**Query**: `type` (`stt`|`tts`|null), `limit` (default 50), `offset` (default 0)

**Response**: `{ "items": [{ "id", "type", "model", "created_at", ... }], "total", "limit", "offset" }`
</details>

### Conversations

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/conversations` | Create a conversation |
| `GET` | `/api/conversations` | List conversations |
| `GET` | `/api/conversations/{id}` | Get a conversation |
| `POST` | `/api/conversations/{id}/turns` | Add a turn to a conversation |
| `DELETE` | `/api/conversations/{id}/turns/{turn_id}` | Delete a turn |
| `POST` | `/api/conversations/{id}/render` | Render conversation to audio |
| `GET` | `/api/conversations/{id}/audio` | Get rendered audio file |
| `DELETE` | `/api/conversations/{id}` | Delete a conversation |

<details>
<summary><strong>POST /api/conversations</strong></summary>

**Body**: `{ "name": "Demo", "turns": [{ "speaker", "text", "profile_id", "effects" }] }`

**Response** (201): `{ "id", "name", "turns": [...], "created_at" }`
</details>

<details>
<summary><strong>POST /api/conversations/{id}/render</strong></summary>

**Body**: `{ "format": "wav", "sample_rate": 24000, "save_turn_audio": true }`

**Response**: `{ "id", "render_output_path", "format", "duration_s" }`
</details>

### Composer (Multi-track)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/composer/render` | Render a multi-track composition |
| `GET` | `/api/composer/renders` | List all composition renders |
| `GET` | `/api/composer/render/{id}/audio` | Get rendered composition audio |
| `DELETE` | `/api/composer/render/{id}` | Delete a composition render |

<details>
<summary><strong>POST /api/composer/render</strong></summary>

**Body**: `{ "name", "format": "wav", "sample_rate": 24000, "tracks": [{ "source_path", "offset_s", "volume", "muted", "solo", "effects" }] }`

**Response**: `{ "id", "render_output_path", "format", "tracks_count" }`
</details>

### Legacy Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/ps` | List loaded STT models |
| `POST` | `/api/ps/{model}` | Load an STT model |
| `DELETE` | `/api/ps/{model}` | Unload an STT model |
| `POST` | `/api/pull/{model}` | Download a model (load + unload) |

> These legacy endpoints are kept for backwards compatibility. Prefer the unified `/api/models/*` endpoints.

### Health & UI

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/web` | Web UI (HTML) |
| `GET` | `/docs` | OpenAPI/Swagger interactive docs |

<details>
<summary><strong>GET /health</strong></summary>

**Response**: `{ "status": "ok", "version": "0.6.0", "models_loaded": 2 }`
</details>
<details>
<summary><strong>Transcribe audio</strong></summary>

```bash
curl -sk https://localhost:8100/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=deepdml/faster-whisper-large-v3-turbo-ct2" \
  -F "response_format=json"
```

Formats: `json`, `text`, `verbose_json`, `srt`, `vtt`
</details>

<details>
<summary><strong>Text-to-speech</strong></summary>

```bash
curl -sk https://localhost:8100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro","input":"Hello world","voice":"af_heart"}' \
  -o output.mp3
```

Formats: `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`
</details>

<details>
<summary><strong>OpenAI Python SDK</strong></summary>

```python
import httpx
from openai import OpenAI

client = OpenAI(
    base_url="https://localhost:8100/v1",
    api_key="not-needed",
    http_client=httpx.Client(verify=False),
)

# STT
with open("audio.wav", "rb") as f:
    result = client.audio.transcriptions.create(
        model="deepdml/faster-whisper-large-v3-turbo-ct2", file=f
    )
print(result.text)

# TTS
response = client.audio.speech.create(
    model="kokoro", input="Hello world", voice="af_heart"
)
response.stream_to_file("output.mp3")
```
</details>

<details>
<summary><strong>Open Speech Python client SDK</strong></summary>

```python
import asyncio
from src.client import OpenSpeechClient

client = OpenSpeechClient(base_url="http://localhost:8100")

# Sync helpers
result = client.transcribe(open("audio.wav", "rb").read())
print(result["text"])

# Realtime session
rt = client.realtime_session()
rt.on_transcript(lambda ev: print("tx", ev))
rt.on_vad(lambda ev: print("vad", ev))
rt.send_audio(b"\x00\x00" * 2400)
rt.commit()
rt.create_response("Hello from Open Speech", voice="alloy")
rt.close()

# Async streaming transcription
async def run():
    async def chunks():
        yield b"\x00\x00" * 3200
    async for event in client.async_stream_transcribe(chunks()):
        print(event)

asyncio.run(run())
```
</details>

<details>
<summary><strong>Open Speech JS/TS client SDK</strong></summary>

```ts
import { OpenSpeechClient } from "@open-speech/client";

const client = new OpenSpeechClient({ baseUrl: "http://localhost:8100" });
const transcript = await client.transcribe(await (await fetch("/audio.wav")).arrayBuffer());
console.log(transcript.text);

const rt = client.realtimeSession();
rt.onTranscript((ev) => console.log(ev));
rt.sendAudio(pcmChunk);
rt.commit();
rt.createResponse("Hello there", "alloy");
```
</details>

<details>
<summary><strong>Real-time streaming (WebSocket)</strong></summary>

```javascript
const ws = new WebSocket("wss://localhost:8100/v1/audio/stream?model=deepdml/faster-whisper-large-v3-turbo-ct2");
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "transcript") {
        console.log(data.is_final ? "FINAL:" : "partial:", data.text);
    }
};
// Send PCM16 LE mono 16kHz chunks
ws.send(audioChunkArrayBuffer);
```
</details>

## Web UI

Open **https://localhost:8100/web** for a three-tab interface:

- **Transcribe** — Upload files, record from mic, or stream in real-time
- **Speak** — Enter text, pick a voice, generate audio
- **Models** — Browse available models, download, load/unload

Light and dark themes with OS auto-detection.

## Docker

### Docker Compose — CPU

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

### Docker Compose — GPU

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

### Build-time baked providers (GPU image)

Default GPU image bakes `kokoro,qwen3` — both providers are ready to use immediately with zero "Install Provider" step. You can customize which providers and model weights are baked at build time:

```bash
docker build \
  --build-arg BAKED_PROVIDERS=kokoro,qwen3,pocket-tts,piper \
  --build-arg BAKED_TTS_MODELS=kokoro,pocket-tts,piper/en_US-ryan-medium \
  -t jwindsor1/open-speech:full .
```

- `BAKED_PROVIDERS`: controls which Python packages are pre-installed at build time (fast, adds ~200MB per provider). Default: `kokoro,qwen3`.
- `BAKED_TTS_MODELS`: controls which model weights are baked into the image (slow, adds GB per model). Default: `kokoro`.

> **Note:** Qwen3 model weights (~3-4GB for 1.7B) are **not** baked by default — they download on first Load click. Only the Python packages (torchaudio, transformers, qwen-tts, etc.) are pre-installed so the provider is instantly available without a pip install step.

### Windows GPU (WSL2)

Ensure [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is installed in WSL2, then use the GPU compose file above.

### Volume Strategy

Models download to `/root/.cache/huggingface` inside the container. Mount a named volume to persist across restarts — no re-downloading.

## STT Backends

| Backend | Best for | Languages | Model prefix |
|---------|----------|-----------|-------------|
| **faster-whisper** | High accuracy, GPU | 99+ | `Systran/faster-whisper-*`, `deepdml/faster-whisper-*` |

## TTS Backends

| Backend | Best for | Voices | Status |
|---------|----------|--------|--------|
| **Kokoro** | Quality + variety | 52 voices, blending | ✅ Stable |
| **Pocket TTS** | CPU-first, low-latency | 8 built-in voices, streaming | ✅ Stable |
| **Piper** | Lightweight, fast | Per-model voices | ✅ Stable |
| **Qwen3-TTS** | Voice design + cloning | 4 built-in + custom | ✅ Stable |

See [TTS-BACKENDS.md](docs/TTS-BACKENDS.md) for the backend roadmap.

## Voice Blending

Kokoro supports mixing voices with weighted syntax:

```bash
curl -sk https://localhost:8100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro","input":"Hello","voice":"af_bella(2)+af_sky(1)"}'
```

This blends 2 parts `af_bella` with 1 part `af_sky`. See `voice-presets.example.yml` for saving named presets.

## Voice Cloning

Clone any voice from a reference audio sample (Qwen3-TTS):

```bash
# Via multipart upload
curl -sk https://localhost:8100/v1/audio/speech/clone \
  -F "input=Hello, I sound like the reference!" \
  -F "model=qwen3-tts/0.6B-CustomVoice" \
  -F "reference_audio=@reference.wav" \
  -o cloned.mp3

# Via JSON with base64
curl -sk https://localhost:8100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-tts/0.6B-CustomVoice","input":"Hello","voice":"default","reference_audio":"<base64>"}' \
  -o cloned.mp3
```

## Voice Design

Describe a voice in natural language and Qwen3-TTS will generate it:

```bash
curl -sk https://localhost:8100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-tts/0.6B-CustomVoice",
    "input": "Good morning!",
    "voice": "default",
    "voice_design": "A warm, deep male voice with a slight British accent"
  }' \
  -o designed.mp3
```

> `voice_design` and `reference_audio` are ignored by backends that don't support them (Kokoro, Piper).

## Security

```bash
# API key (recommended for all non-local deployments)
OS_API_KEY=my-secret-key docker compose up -d
curl -sk -H "Authorization: Bearer my-secret-key" https://localhost:8100/health

# Enforce auth at startup (fails fast when API key missing)
OS_AUTH_REQUIRED=true

# Realtime/WebSocket origin allowlist (optional)
OS_WS_ALLOWED_ORIGINS=https://myapp.com,https://staging.myapp.com

# Wyoming bind host (default localhost)
OS_WYOMING_HOST=127.0.0.1

# Rate limiting (60 req/min, burst of 10)
OS_RATE_LIMIT=60 OS_RATE_LIMIT_BURST=10

# CORS
OS_CORS_ORIGINS=https://myapp.com,https://staging.myapp.com

# Custom SSL cert
OS_SSL_CERTFILE=/certs/cert.pem OS_SSL_KEYFILE=/certs/key.pem
```

## Environment Variables Reference

Defaults from `src/config.py` (grouped by prefix).

### OS_* (server / shared)

| Variable | Default | Description |
|---|---|---|
| `OS_PORT` | `8100` | HTTP bind port |
| `OS_HOST` | `0.0.0.0` | HTTP bind host |
| `OS_API_KEY` | `""` | Bearer API key (empty disables auth) |
| `OS_AUTH_REQUIRED` | `false` | Fail startup if API key missing |
| `OS_CORS_ORIGINS` | `*` | Comma-separated CORS origins |
| `OS_WS_ALLOWED_ORIGINS` | `""` | Allowed WebSocket `Origin` values |
| `OS_TRUST_PROXY` | `false` | Trust `X-Forwarded-For` headers |
| `OS_MAX_UPLOAD_MB` | `100` | Max upload size in MB |
| `OS_RATE_LIMIT` | `0` | Requests/min/IP (`0` disables) |
| `OS_RATE_LIMIT_BURST` | `0` | Burst bucket size |
| `OS_SSL_ENABLED` | `true` | Enable HTTPS |
| `OS_SSL_CERTFILE` | `""` | TLS cert path (auto-generated if empty) |
| `OS_SSL_KEYFILE` | `""` | TLS key path (auto-generated if empty) |
| `OS_VOICE_LIBRARY_PATH` | `/home/openspeech/data/voices` | Stored voice reference directory |
| `OS_VOICE_LIBRARY_MAX_COUNT` | `100` | Max stored voice references (`0` = unlimited) |
| `OS_STUDIO_DB_PATH` | `/home/openspeech/data/studio.db` | SQLite database for profiles/history |
| `OS_HISTORY_ENABLED` | `true` | Enable TTS/STT history logging |
| `OS_HISTORY_MAX_ENTRIES` | `1000` | Maximum history rows retained |
| `OS_HISTORY_RETAIN_AUDIO` | `true` | Keep audio output path metadata |
| `OS_HISTORY_MAX_MB` | `2000` | Max retained audio footprint in MB |
| `OS_EFFECTS_ENABLED` | `true` | Enable TTS effects processing |
| `OS_CONVERSATIONS_DIR` | `/home/openspeech/data/conversations` | Conversation storage directory |
| `OS_COMPOSER_DIR` | `/home/openspeech/data/composer` | Composer storage directory |
| `OS_PROVIDERS_DIR` | `/home/openspeech/data/providers` | User-installed provider package directory |
| `OS_WYOMING_ENABLED` | `false` | Enable Wyoming TCP server |
| `OS_WYOMING_HOST` | `127.0.0.1` | Wyoming bind host |
| `OS_WYOMING_PORT` | `10400` | Wyoming port |
| `OS_REALTIME_ENABLED` | `true` | Enable `/v1/realtime` |
| `OS_REALTIME_MAX_BUFFER_MB` | `50` | Max realtime audio buffer per session |
| `OS_REALTIME_IDLE_TIMEOUT_S` | `120` | Realtime idle timeout seconds |
| `OS_MODEL_TTL` | `300` | Auto-unload idle model TTL (seconds) |
| `OS_MAX_LOADED_MODELS` | `0` | Max loaded models (`0` = unlimited) |
| `OS_STREAM_CHUNK_MS` | `100` | Streaming chunk window (ms) |
| `OS_STREAM_VAD_THRESHOLD` | `0.5` | Streaming VAD threshold |
| `OS_STREAM_ENDPOINTING_MS` | `300` | Silence to finalize utterance (ms) |
| `OS_STREAM_MAX_CONNECTIONS` | `10` | Max concurrent streaming WS sessions |

### STT_* (speech-to-text)

| Variable | Default | Description |
|---|---|---|
| `STT_MODEL` | `deepdml/faster-whisper-large-v3-turbo-ct2` | Default STT model ID |
| `STT_DEVICE` | `cuda` | STT inference device |
| `STT_COMPUTE_TYPE` | `float16` | STT compute precision |
| `STT_MODEL_DIR` | `None` | Optional local model directory |
| `STT_PRELOAD_MODELS` | `""` | Comma-separated STT models to preload |
| `STT_VAD_ENABLED` | `true` | Enable VAD by default on streaming endpoint |
| `STT_VAD_THRESHOLD` | `0.5` | VAD speech probability threshold |
| `STT_VAD_MIN_SPEECH_MS` | `250` | Min speech duration before activation |
| `STT_VAD_SILENCE_MS` | `800` | Silence duration for endpointing |
| `STT_DIARIZE_ENABLED` | `false` | Enable diarization support |
| `STT_NOISE_REDUCE` | `false` | Enable STT denoise preprocessing |
| `STT_NORMALIZE` | `true` | Enable STT input normalization |

### TTS_* (text-to-speech)

| Variable | Default | Description |
|---|---|---|
| `TTS_ENABLED` | `true` | Enable TTS endpoints |
| `TTS_MODEL` | `kokoro` | Default TTS model ID |
| `TTS_VOICE` | `af_heart` | Default TTS voice |
| `TTS_DEVICE` | `None` | TTS device override (falls back to STT device) |
| `TTS_MAX_INPUT_LENGTH` | `4096` | Max text chars per synthesis request |
| `TTS_DEFAULT_FORMAT` | `mp3` | Default output audio format |
| `TTS_SPEED` | `1.0` | Default synthesis speed |
| `TTS_PRELOAD_MODELS` | `""` | Comma-separated TTS models to preload |
| `TTS_VOICES_CONFIG` | `""` | YAML voice presets file path |
| `TTS_CACHE_ENABLED` | `false` | Enable on-disk synthesis cache |
| `TTS_CACHE_MAX_MB` | `500` | Max cache size in MB |
| `TTS_CACHE_DIR` | `/var/lib/open-speech/cache` | Cache directory |
| `TTS_TRIM_SILENCE` | `true` | Trim generated silence |
| `TTS_NORMALIZE_OUTPUT` | `true` | Normalize output loudness |
| `TTS_PRONUNCIATION_DICT` | `""` | Pronunciation dictionary path |
| `TTS_QWEN3_SIZE` | `1.7B` | Qwen3 model size selector |
| `TTS_QWEN3_FLASH_ATTN` | `false` | Enable Qwen3 flash attention |
| `TTS_QWEN3_DEVICE` | `cuda:0` | Qwen3 backend device override |


## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full phase breakdown and current status.

### Current status (v0.5.1)
- ✅ Phases 2–6 complete (multi-model, Docker, advanced TTS, Wyoming, production hardening)
- ✅ Phase 8a+8b — Voice Profiles + Generation History
- ✅ Web UI — ground-up rewrite: Transcribe / Speak / Models / History / Settings
- 🚧 Phase 8c–8e — Conversation mode, Voice effects, Composer
- 🔴 B6, B9 — critical backend bugs (provider install path, streaming event loop)

## Contributing

Open Speech uses a pluggable backend system. To add a new STT or TTS backend:

1. Create a new file in `src/tts/` or `src/stt/`
2. Implement the backend interface (see existing backends for examples)
3. Register it in the model registry
4. Add tests
5. Submit a PR

See [TTS-BACKENDS.md](docs/TTS-BACKENDS.md) for the TTS backend roadmap and design patterns.

## License

[MIT](LICENSE) © 2026 Jeremy Windsor
