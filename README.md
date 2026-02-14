# 🎙️ Open Speech

OpenAI-compatible speech-to-text server with pluggable backends.

Drop-in replacement for [faster-whisper-server](https://github.com/fedirz/faster-whisper-server) with a cleaner architecture, better model management, and support for multiple STT backends.

## Features

- **OpenAI-compatible API** — `/v1/audio/transcriptions`, `/v1/audio/translations`
- **Pluggable backends** — faster-whisper (ctranslate2) today, more coming
- **Model management** — Load, unload, hot-swap models without restart
- **GPU accelerated** — CUDA support via ctranslate2
- **Docker-first** — Production-ready container images
- **Inspired by [kokoro-fastapi](https://github.com/remsky/kokoro-fastapi)** — Clean, focused, well-packaged

## Quick Start

```bash
docker run -d --gpus all -p 8100:8100 \
  -v open-speech-models:/root/.cache/huggingface \
  jwindsor1/open-speech:latest
```

```bash
curl -X POST http://localhost:8100/v1/audio/transcriptions \
  -F "file=@audio.ogg" \
  -F "model=deepdml/faster-whisper-large-v3-turbo-ct2" \
  -F "response_format=json"
```

## API

### OpenAI-compatible
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/audio/transcriptions` | Transcribe audio to text |
| `POST` | `/v1/audio/translations` | Translate audio to English |
| `GET` | `/v1/models` | List available models |

### Model Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/ps` | List loaded models |
| `POST` | `/api/ps/{model}` | Load a model |
| `DELETE` | `/api/ps/{model}` | Unload a model |
| `POST` | `/api/pull/{model}` | Download a model |
| `GET` | `/health` | Health check |

## Supported Backends

### Phase 1 (Current)
- **faster-whisper** — All Whisper models via ctranslate2 (tiny → large-v3-turbo, distil variants)

### Planned
- Cloud STT proxy (ElevenLabs, Cartesia, Deepgram) — unified endpoint, local-first with cloud fallback
- Additional local models as the ecosystem evolves

## Configuration

| Env Var | Default | Description |
|---------|---------|-------------|
| `STT_DEFAULT_MODEL` | `deepdml/faster-whisper-large-v3-turbo-ct2` | Default model |
| `STT_DEVICE` | `cuda` | Device (`cuda` or `cpu`) |
| `STT_COMPUTE_TYPE` | `float16` | Compute type for inference |
| `STT_HOST` | `0.0.0.0` | Bind address |
| `STT_PORT` | `8100` | Listen port |

## Development

```bash
# Clone
git clone https://github.com/will-assistant/open-speech.git
cd open-speech

# Install
pip install -e ".[dev]"

# Run
uvicorn src.main:app --host 0.0.0.0 --port 8100

# Test
pytest
```

## License

MIT

## Credits

Inspired by [kokoro-fastapi](https://github.com/remsky/kokoro-fastapi) — the gold standard for self-hosted TTS servers.
