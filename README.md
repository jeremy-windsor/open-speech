# Open Speech

OpenAI-compatible speech-to-text server with pluggable backends.

Drop-in replacement for faster-whisper-server / Speaches with a cleaner architecture, web UI, and real-time streaming.

## Features

- **OpenAI API compatible** — `POST /v1/audio/transcriptions`, `POST /v1/audio/translations`
- **Real-time streaming** — `WS /v1/audio/stream` (Deepgram-compatible protocol)
- **Web UI** — Upload files, record from mic, stream live, or synthesize speech at `/web`
- **Text-to-speech** — `POST /v1/audio/speech` (OpenAI-compatible, Kokoro-82M backend)
- **Voice blending** — Mix voices with `af_bella(2)+af_sky(1)` syntax
- **Multiple STT backends** — faster-whisper (GPU/CPU), Moonshine (fast CPU, English), Vosk (tiny, offline)
- **Pluggable backends** — select via model name: `faster-whisper-*`, `moonshine/*`, `vosk-*`
- **TTS Web UI** — Text-to-speech tab with voice selector, blending, speed control, history, and model management
- **Model hot-swap** — Load/unload models via `/api/ps` and `/v1/audio/models/load`
- **GPU + CPU** — CUDA float16 or CPU int8
- **Self-signed HTTPS** — Auto-generated cert, browser mic works out of the box
- **Silero VAD** — Voice activity detection prevents transcribing silence
- **Docker ready** — GPU and CPU compose files included

## Quick Start

### One-liner (Docker Hub)

```bash
# GPU (NVIDIA)
docker run -d -p 8100:8100 --gpus all jwindsor1/open-speech:latest

# CPU
docker run -d -p 8100:8100 jwindsor1/open-speech:cpu
```

Open **https://localhost:8100/web** — accept the self-signed cert warning, then upload audio or use the mic.

### Docker Compose (recommended for persistent setups)

```bash
git clone https://github.com/will-assistant/open-speech.git
cd open-speech

# GPU
docker compose -f docker-compose.gpu.yml up -d

# CPU
docker compose -f docker-compose.cpu.yml up -d
```

Compose uses persistent volumes for model cache — models survive container rebuilds.

### Custom Configuration

```bash
cp .env.example .env    # edit as needed
docker compose -f docker-compose.gpu.yml up -d
```

All settings work as environment variables, in `.env`, or inline in compose. See [Configuration](#configuration) for the full list.

### Windows with GPU (WSL2 + Docker Desktop)

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) with WSL2 backend
2. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) in WSL2
3. Run:

```powershell
docker run -d -p 8100:8100 --gpus all jwindsor1/open-speech:latest
```

Or clone the repo and use `docker compose -f docker-compose.gpu.yml up -d` for persistent config.

## STT Backends

Open Speech supports multiple speech-to-text backends. The backend is selected automatically based on the `model` parameter:

| Backend | Model prefix | Best for | Languages |
|---------|-------------|----------|-----------|
| **faster-whisper** | `deepdml/faster-whisper-*`, etc. | High accuracy, GPU | 99+ languages |
| **Moonshine** | `moonshine/tiny`, `moonshine/base` | Fast CPU inference, edge | English only |
| **Vosk** | `vosk-model-*` | Tiny models, fully offline | Many (per model) |

### Install optional backends

```bash
pip install 'open-speech[moonshine]'  # Moonshine (moonshine-onnx)
pip install 'open-speech[vosk]'       # Vosk
```

### Usage examples

```bash
# faster-whisper (default)
curl -sk https://localhost:8100/v1/audio/transcriptions \
  -F "file=@audio.wav" -F "model=deepdml/faster-whisper-large-v3-turbo-ct2"

# Moonshine — 5x faster than Whisper on CPU, English only
curl -sk https://localhost:8100/v1/audio/transcriptions \
  -F "file=@audio.wav" -F "model=moonshine/tiny"

# Vosk — tiny offline model
curl -sk https://localhost:8100/v1/audio/transcriptions \
  -F "file=@audio.wav" -F "model=vosk-model-small-en-us-0.15"
```

Set a default backend via environment variable:
```bash
STT_DEFAULT_MODEL=moonshine/tiny  # Use Moonshine by default
```

## API Usage

### Transcribe a file

```bash
curl -sk https://localhost:8100/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=deepdml/faster-whisper-large-v3-turbo-ct2" \
  -F "response_format=json"
```

### OpenAI Python SDK

```python
import httpx
from openai import OpenAI

client = OpenAI(
    base_url="https://localhost:8100/v1",
    api_key="not-needed",
    http_client=httpx.Client(verify=False),  # self-signed cert
)

with open("audio.wav", "rb") as f:
    result = client.audio.transcriptions.create(
        model="deepdml/faster-whisper-large-v3-turbo-ct2",
        file=f,
    )
print(result.text)
```

### Text-to-Speech

```bash
# Generate speech (saves as MP3)
curl -sk https://localhost:8100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro","input":"Hello world","voice":"alloy"}' \
  -o output.mp3
```

```python
# OpenAI Python SDK
speech = client.audio.speech.create(
    model="kokoro",
    input="Hello world!",
    voice="alloy",  # or "af_bella", "af_bella(2)+af_sky(1)"
    response_format="mp3",
)
speech.stream_to_file("output.mp3")
```

**Voice options:**
- OpenAI names: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`
- Kokoro voices: `af_heart`, `af_bella`, `am_adam`, etc.
- Blends: `af_bella(2)+af_sky(1)` (weighted mix)

**Formats:** `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`

### Streaming TTS

```bash
# Stream audio as it's generated (chunked transfer)
curl -sk "https://localhost:8100/v1/audio/speech?stream=true" \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro","input":"Long text here...","voice":"af_heart"}' \
  --output - | mpv -
```

### Transcript Formats (SRT/VTT)

```bash
# Get SRT subtitles
curl -sk https://localhost:8100/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=srt" -o transcript.srt

# Get WebVTT subtitles
curl -sk https://localhost:8100/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=vtt" -o transcript.vtt

# Plain text only
curl -sk https://localhost:8100/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=text"
```

### Real-time streaming (WebSocket)

```javascript
const ws = new WebSocket("wss://localhost:8100/v1/audio/stream?model=deepdml/faster-whisper-large-v3-turbo-ct2");

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "transcript") {
        console.log(data.is_final ? "FINAL:" : "partial:", data.text);
    }
};

// Send PCM16 LE mono 16kHz audio as binary frames
ws.send(audioChunkArrayBuffer);

// Stop gracefully
ws.send(JSON.stringify({ type: "stop" }));
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Server health + loaded model count |
| `GET` | `/v1/models` | List available models (STT + TTS) |
| `GET` | `/api/ps` | Show loaded models with details |
| `POST` | `/api/ps/{model}` | Load a model |
| `POST` | `/v1/audio/transcriptions` | Transcribe audio file |
| `POST` | `/v1/audio/translations` | Translate audio to English |
| `POST` | `/v1/audio/speech` | Synthesize speech from text (TTS) |
| `GET` | `/v1/audio/voices` | List available TTS voices |
| `WS` | `/v1/audio/stream` | Real-time streaming transcription |
| `GET` | `/web` | Web UI |
| `GET` | `/docs` | Swagger/OpenAPI docs |

## Configuration

All config via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_HOST` | `0.0.0.0` | Bind address |
| `STT_PORT` | `8100` | Listen port |
| `STT_DEVICE` | `cuda` | `cuda` or `cpu` |
| `STT_COMPUTE_TYPE` | `float16` | `float16`, `int8`, `int8_float16` |
| `STT_DEFAULT_MODEL` | `deepdml/faster-whisper-large-v3-turbo-ct2` | Default whisper model |
| `STT_PRELOAD_MODELS` | `` | Comma-separated models to download and load on startup |
| `STT_MODEL_TTL` | `300` | Seconds idle before auto-unload (0 = never). Default model exempt |
| `STT_MAX_LOADED_MODELS` | `0` | Max models in memory (0 = unlimited). LRU eviction, default exempt |
| `STT_STREAM_CHUNK_MS` | `2000` | Streaming chunk size (ms) |
| `STT_STREAM_VAD_THRESHOLD` | `0.5` | VAD speech detection threshold |
| `STT_STREAM_ENDPOINTING_MS` | `300` | Silence before finalizing utterance |
| `STT_STREAM_MAX_CONNECTIONS` | `10` | Max concurrent WebSocket streams |
| `STT_API_KEY` | `` | API key for authentication (empty = auth disabled) |
| `STT_RATE_LIMIT` | `0` | Max requests/min per IP (0 = disabled) |
| `STT_RATE_LIMIT_BURST` | `0` | Burst allowance (0 = same as rate limit) |
| `STT_MAX_UPLOAD_MB` | `100` | Maximum upload file size in MB |
| `STT_CORS_ORIGINS` | `*` | Comma-separated allowed CORS origins |
| `STT_TRUST_PROXY` | `false` | Trust X-Forwarded-For for rate limiting (set true behind reverse proxy) |
| **TTS Settings** | | |
| `TTS_ENABLED` | `true` | Enable/disable TTS endpoints |
| `TTS_DEFAULT_MODEL` | `kokoro` | Default TTS model |
| `TTS_DEFAULT_VOICE` | `af_heart` | Default voice |
| `TTS_DEVICE` | _(inherits STT_DEVICE)_ | Device for TTS inference (`cuda`/`cpu`) |
| `TTS_MAX_INPUT_LENGTH` | `4096` | Max input text length (chars) |
| `TTS_DEFAULT_FORMAT` | `mp3` | Default output audio format |
| `TTS_DEFAULT_SPEED` | `1.0` | Default speech speed |
| `TTS_PRELOAD_MODELS` | `` | Comma-separated TTS models to preload |
| `TTS_VOICES_CONFIG` | `` | Path to custom voice presets YAML |

## Model Lifecycle

Open Speech supports Ollama-style automatic model eviction to manage memory:

- **TTL eviction** — Models idle longer than `STT_MODEL_TTL` seconds are automatically unloaded. The default model is exempt.
- **Max models** — When `STT_MAX_LOADED_MODELS` is set, the least recently used non-default model is evicted when the limit is exceeded.
- **Manual unload** — `DELETE /api/ps/{model}` to immediately unload a model (409 for default, 404 if not loaded).
- **Enriched status** — `GET /api/ps` returns `last_used_at`, `is_default`, and `ttl_remaining` per model.

```bash
# Keep models for 10 minutes, max 3 loaded
STT_MODEL_TTL=600 STT_MAX_LOADED_MODELS=3 docker compose up -d
```

## Security

### API Key Authentication

Set `STT_API_KEY` to require authentication on all API endpoints. Health (`/health`) and web UI (`/web`) are always exempt.

```bash
# Enable auth
STT_API_KEY=my-secret-key docker compose up -d

# Use with curl
curl -sk https://localhost:8100/v1/audio/transcriptions \
  -H "Authorization: Bearer my-secret-key" \
  -F "file=@audio.wav"

# Use with OpenAI SDK
client = OpenAI(base_url="https://localhost:8100/v1", api_key="my-secret-key")

# WebSocket auth via query param
ws = new WebSocket("wss://localhost:8100/v1/audio/stream?api_key=my-secret-key");
```

### Rate Limiting

Per-IP token bucket rate limiter. Set `STT_RATE_LIMIT` to enable.

```bash
STT_RATE_LIMIT=60          # 60 requests/min per IP
STT_RATE_LIMIT_BURST=10    # Allow bursts up to 10
```

Rate limit info is returned in response headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `Retry-After` (on 429).

### Upload Limits

`STT_MAX_UPLOAD_MB` (default 100) caps file upload size. Empty files are rejected with 400.

### CORS

`STT_CORS_ORIGINS` controls allowed origins (default `*`). Set to specific origins for production:

```bash
STT_CORS_ORIGINS=https://myapp.com,https://staging.myapp.com
```

## Response Formats

`response_format` parameter supports: `json`, `text`, `verbose_json`, `srt`, `vtt`

## License

MIT
