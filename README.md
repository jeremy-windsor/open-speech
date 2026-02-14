# Open Speech

OpenAI-compatible speech-to-text server with pluggable backends.

Drop-in replacement for faster-whisper-server / Speaches with a cleaner architecture, web UI, and real-time streaming.

## Features

- **OpenAI API compatible** — `POST /v1/audio/transcriptions`, `POST /v1/audio/translations`
- **Real-time streaming** — `WS /v1/audio/stream` (Deepgram-compatible protocol)
- **Web UI** — Upload files, record from mic, or stream live at `/web`
- **Pluggable backends** — faster-whisper (Phase 1), more coming
- **Model hot-swap** — Load/unload models via `/api/ps`
- **GPU + CPU** — CUDA float16 or CPU int8
- **Self-signed HTTPS** — Auto-generated cert, browser mic works out of the box
- **Silero VAD** — Voice activity detection prevents transcribing silence
- **Docker ready** — GPU and CPU compose files included

## Quick Start

### GPU (NVIDIA)

```bash
git clone https://github.com/will-assistant/open-speech.git
cd open-speech
docker compose -f docker-compose.gpu.yml up -d
```

### CPU

```bash
docker compose -f docker-compose.cpu.yml up -d
```

Open `https://localhost:8100/web` in your browser.
Accept the self-signed cert warning, then upload audio or use the microphone.

### Windows with GPU (WSL2 + Docker Desktop)

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) with WSL2 backend
2. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) in WSL2
3. Clone and run:

```powershell
git clone https://github.com/will-assistant/open-speech.git
cd open-speech
docker compose -f docker-compose.gpu.yml up -d
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
| `GET` | `/v1/models` | List available models |
| `GET` | `/api/ps` | Show loaded models with details |
| `POST` | `/api/ps/{model}` | Load a model |
| `POST` | `/v1/audio/transcriptions` | Transcribe audio file |
| `POST` | `/v1/audio/translations` | Translate audio to English |
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
| `STT_STREAM_CHUNK_MS` | `2000` | Streaming chunk size (ms) |
| `STT_STREAM_VAD_THRESHOLD` | `0.5` | VAD speech detection threshold |
| `STT_STREAM_ENDPOINTING_MS` | `300` | Silence before finalizing utterance |
| `STT_STREAM_MAX_CONNECTIONS` | `10` | Max concurrent WebSocket streams |

## Response Formats

`response_format` parameter supports: `json`, `text`, `verbose_json`, `srt`, `vtt`

## License

MIT
