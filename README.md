# Open Speech

OpenAI-compatible speech server with pluggable backends — STT and TTS in one container.

Drop-in replacement for faster-whisper-server / Speaches with a cleaner architecture, web UI, and real-time streaming.

## Features

- **OpenAI API compatible** — `POST /v1/audio/transcriptions`, `POST /v1/audio/translations`
- **Real-time streaming** — `WS /v1/audio/stream` (Deepgram-compatible protocol)
- **Web UI** — Upload files, record from mic, stream live, or synthesize speech at `/web`
- **Text-to-speech** — `POST /v1/audio/speech` (OpenAI-compatible, Kokoro-82M backend)
- **Voice blending** — Mix voices with `af_bella(2)+af_sky(1)` syntax
- **Multiple STT backends** — faster-whisper (GPU/CPU), Moonshine (fast CPU, English), Vosk (tiny, offline)
- **Pluggable backends** — select via model name: `faster-whisper-*`, `moonshine/*`, `vosk-*`
- **Unified model management** — `GET /api/models` for all models (STT + TTS), load/unload via API
- **Model hot-swap** — Load/unload models via `/api/models/{id}/load` and `DELETE /api/models/{id}`
- **GPU + CPU** — CUDA float16 or CPU int8, same image
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

### Docker Compose (recommended)

```yaml
# docker-compose.yml
services:
  open-speech:
    image: jwindsor1/open-speech:cpu
    ports: ["8100:8100"]
    environment:
      - OS_PORT=8100
      - STT_MODEL=Systran/faster-whisper-base
      - STT_DEVICE=cpu
      - TTS_MODEL=kokoro
      - TTS_DEVICE=cpu
    volumes:
      - hf-cache:/root/.cache/huggingface

volumes:
  hf-cache:
```

GPU version:

```yaml
# docker-compose.gpu.yml
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
      - OS_PORT=8100
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

### Custom Configuration

```bash
cp .env.example .env    # edit as needed
docker compose up -d
```

## STT Backends

| Backend | Model prefix | Best for | Languages |
|---------|-------------|----------|-----------|
| **faster-whisper** | `deepdml/faster-whisper-*`, etc. | High accuracy, GPU | 99+ languages |
| **Moonshine** | `moonshine/tiny`, `moonshine/base` | Fast CPU inference, edge | English only |
| **Vosk** | `vosk-model-*` | Tiny models, fully offline | Many (per model) |

### Install optional backends

```bash
pip install 'open-speech[moonshine]'  # Moonshine (moonshine-onnx)
pip install 'open-speech[vosk]'       # Vosk
pip install 'open-speech[piper]'      # Piper TTS
pip install 'open-speech[all]'        # All optional backends
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
curl -sk https://localhost:8100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro","input":"Hello world","voice":"alloy"}' \
  -o output.mp3
```

**Voice options:** `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`, or Kokoro voices like `af_heart`, `af_bella`, `am_adam`. Blends: `af_bella(2)+af_sky(1)`.

**Formats:** `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`

### Model Management

```bash
# List all models (loaded + downloaded + available)
curl -sk https://localhost:8100/api/models | jq

# Load a model
curl -sk -X POST https://localhost:8100/api/models/Systran%2Ffaster-whisper-base/load

# Check model status
curl -sk https://localhost:8100/api/models/Systran%2Ffaster-whisper-base/status

# Unload a model
curl -sk -X DELETE https://localhost:8100/api/models/Systran%2Ffaster-whisper-base
```

### Transcript Formats (SRT/VTT)

```bash
curl -sk https://localhost:8100/v1/audio/transcriptions \
  -F "file=@audio.wav" -F "response_format=srt" -o transcript.srt
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
ws.send(audioChunkArrayBuffer);  // PCM16 LE mono 16kHz
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Server health + loaded model count |
| `GET` | `/v1/models` | List available models (OpenAI-compatible) |
| `POST` | `/v1/audio/transcriptions` | Transcribe audio file |
| `POST` | `/v1/audio/translations` | Translate audio to English |
| `POST` | `/v1/audio/speech` | Synthesize speech from text (TTS) |
| `GET` | `/v1/audio/voices` | List available TTS voices |
| `WS` | `/v1/audio/stream` | Real-time streaming transcription |
| `GET` | `/api/models` | All models: available, downloaded, loaded (unified) |
| `POST` | `/api/models/{id}/load` | Load a model |
| `DELETE` | `/api/models/{id}` | Unload a model |
| `GET` | `/api/models/{id}/status` | Model status |
| `GET` | `/api/ps` | Loaded STT models (legacy) |
| `POST` | `/api/ps/{model}` | Load STT model (legacy) |
| `GET` | `/web` | Web UI |
| `GET` | `/docs` | Swagger/OpenAPI docs |

## Configuration

All config via environment variables. Uses `OS_` prefix for server-level, `STT_` for speech-to-text, `TTS_` for text-to-speech.

> **Backwards compatibility:** Old names like `STT_PORT`, `STT_HOST`, `STT_API_KEY`, `STT_DEFAULT_MODEL` etc. still work but log deprecation warnings. Migrate to new names.

| Variable | Default | Description |
|----------|---------|-------------|
| **Server** | | |
| `OS_HOST` | `0.0.0.0` | Bind address |
| `OS_PORT` | `8100` | Listen port |
| `OS_API_KEY` | `` | API key for auth (empty = disabled) |
| `OS_CORS_ORIGINS` | `*` | Comma-separated CORS origins |
| `OS_TRUST_PROXY` | `false` | Trust X-Forwarded-For |
| `OS_MAX_UPLOAD_MB` | `100` | Max upload size in MB |
| `OS_RATE_LIMIT` | `0` | Requests/min per IP (0 = disabled) |
| `OS_RATE_LIMIT_BURST` | `0` | Burst allowance |
| `OS_SSL_ENABLED` | `true` | Enable HTTPS |
| `OS_SSL_CERTFILE` | `` | Custom cert path (auto-gen if empty) |
| `OS_SSL_KEYFILE` | `` | Custom key path (auto-gen if empty) |
| **Model Lifecycle** | | |
| `OS_MODEL_TTL` | `300` | Seconds idle before auto-unload (0 = never) |
| `OS_MAX_LOADED_MODELS` | `0` | Max models in memory (0 = unlimited) |
| **Streaming** | | |
| `OS_STREAM_CHUNK_MS` | `2000` | Streaming chunk size (ms) |
| `OS_STREAM_VAD_THRESHOLD` | `0.5` | VAD speech detection threshold |
| `OS_STREAM_ENDPOINTING_MS` | `300` | Silence before finalizing utterance |
| `OS_STREAM_MAX_CONNECTIONS` | `10` | Max concurrent WebSocket streams |
| **STT** | | |
| `STT_MODEL` | `deepdml/faster-whisper-large-v3-turbo-ct2` | Default STT model |
| `STT_DEVICE` | `cuda` | `cuda` or `cpu` |
| `STT_COMPUTE_TYPE` | `float16` | `float16`, `int8`, `int8_float16` |
| `STT_PRELOAD_MODELS` | `` | Comma-separated models to preload |
| **TTS** | | |
| `TTS_ENABLED` | `true` | Enable/disable TTS endpoints |
| `TTS_MODEL` | `kokoro` | Default TTS model |
| `TTS_VOICE` | `af_heart` | Default voice |
| `TTS_SPEED` | `1.0` | Default speech speed |
| `TTS_DEVICE` | _(inherits STT_DEVICE)_ | Device for TTS (`cuda`/`cpu`) |
| `TTS_MAX_INPUT_LENGTH` | `4096` | Max input text length (chars) |
| `TTS_PRELOAD_MODELS` | `` | TTS models to preload |
| `TTS_VOICES_CONFIG` | `` | Path to custom voice presets YAML |

## Model Lifecycle

- **TTL eviction** — Models idle longer than `OS_MODEL_TTL` seconds are auto-unloaded. Default model exempt.
- **Max models** — When `OS_MAX_LOADED_MODELS` is set, LRU eviction kicks in. Default exempt.
- **Manual unload** — `DELETE /api/models/{id}` to immediately unload.

```bash
OS_MODEL_TTL=600 OS_MAX_LOADED_MODELS=3 docker compose up -d
```

## Security

### API Key Authentication

```bash
OS_API_KEY=my-secret-key docker compose up -d

curl -sk https://localhost:8100/v1/audio/transcriptions \
  -H "Authorization: Bearer my-secret-key" \
  -F "file=@audio.wav"
```

### Rate Limiting

```bash
OS_RATE_LIMIT=60 OS_RATE_LIMIT_BURST=10
```

### CORS

```bash
OS_CORS_ORIGINS=https://myapp.com,https://staging.myapp.com
```

## Response Formats

`response_format` parameter supports: `json`, `text`, `verbose_json`, `srt`, `vtt`

## License

MIT
