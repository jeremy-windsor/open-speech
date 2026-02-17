# Open Speech

**OpenAI-compatible speech server — any STT/TTS provider, one container.**

[![Version](https://img.shields.io/badge/version-0.5.0-blue?style=flat-square)]()
[![Docker Hub](https://img.shields.io/docker/pulls/jwindsor1/open-speech?style=flat-square&logo=docker)](https://hub.docker.com/r/jwindsor1/open-speech)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-332%20passing-brightgreen?style=flat-square)]()
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
pip install -e ".[moonshine]"       # + Moonshine STT (uses useful-moonshine-onnx)
pip install -e ".[vosk]"            # + Vosk STT
pip install -e ".[piper]"           # + Piper TTS
pip install -e ".[qwen]"            # + Qwen3-TTS deep integration (qwen-tts)
pip install -e ".[fish]"            # + Fish Speech TTS
pip install -e ".[all]"             # All core backends (keeps heavy optional extras separate)
pip install -e ".[diarize]"         # + Speaker diarization (pyannote)
pip install -e ".[noise]"           # + Noise reduction preprocessing
pip install -e ".[client]"          # + Python client SDK deps
pip install -e ".[dev]"             # Development tools (pytest, ruff, httpx)
pip install -r requirements.lock      # Reproducible pinned core dependencies
```

> **Note:** The Moonshine STT package was renamed from `moonshine-onnx` to `useful-moonshine-onnx`. If you see import errors for Moonshine, run `pip install useful-moonshine-onnx`.

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
| `moonshine/tiny` | ~60MB | Moonshine | English |
| `moonshine/base` | ~200MB | Moonshine | English |
| `vosk-model-small-en-us-0.15` | ~40MB | Vosk | English |

### TTS Models

| Model | Size | Backend | Voices |
|-------|------|---------|--------|
| `kokoro` | ~82MB | Kokoro | 52 voices, blending |
| `piper/en_US-lessac-medium` | ~35MB | Piper | 1 |
| `piper/en_US-joe-medium` | ~35MB | Piper | 1 |
| `piper/en_US-amy-medium` | ~35MB | Piper | 1 |
| `piper/en_US-arctic-medium` | ~35MB | Piper | 1 |
| `piper/en_GB-alan-medium` | ~35MB | Piper | 1 |
| `qwen3-tts-0.6b` | ~1.2GB | Qwen3-TTS | 4 + voice design |
| `qwen3-tts-1.7b` | ~3.4GB | Qwen3-TTS | 4 + voice design |
| `fish-speech-1.5` | ~500MB | Fish Speech | Zero-shot cloning |

Switch models by changing `STT_MODEL` / `TTS_MODEL` and restarting, or use the API:

```bash
curl -sk -X POST https://localhost:8100/api/models/Systran%2Ffaster-whisper-small/load
```

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/audio/transcriptions` | Transcribe audio |
| `POST` | `/v1/audio/translations` | Translate audio → English |
| `POST` | `/v1/audio/speech` | Text-to-speech |
| `GET` | `/v1/audio/voices` | List TTS voices |
| `GET` | `/v1/models` | List models (OpenAI format) |
| `WS` | `/v1/audio/stream` | Real-time streaming STT |
| `GET` | `/api/models` | All models (available/downloaded/loaded) |
| `POST` | `/api/models/{id}/load` | Load a model |
| `DELETE` | `/api/models/{id}` | Unload a model |
| `GET` | `/api/models/{id}/status` | Model status + download progress |
| `POST` | `/v1/audio/speech/clone` | Voice cloning (multipart) |
| `GET` | `/api/voice-presets` | Voice presets list |
| `GET` | `/health` | Health check |
| `GET` | `/web` | Web UI |
| `GET` | `/docs` | OpenAPI/Swagger docs |

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

### Windows GPU (WSL2)

Ensure [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is installed in WSL2, then use the GPU compose file above.

### Volume Strategy

Models download to `/root/.cache/huggingface` inside the container. Mount a named volume to persist across restarts — no re-downloading.

## STT Backends

| Backend | Best for | Languages | Model prefix |
|---------|----------|-----------|-------------|
| **faster-whisper** | High accuracy, GPU | 99+ | `Systran/faster-whisper-*`, `deepdml/faster-whisper-*` |
| **Moonshine** | Fast CPU inference | English | `moonshine/*` |
| **Vosk** | Tiny, fully offline | Per model | `vosk-model-*` |

## TTS Backends

| Backend | Best for | Voices | Status |
|---------|----------|--------|--------|
| **Kokoro** | Quality + variety | 52 voices, blending | ✅ Stable |
| **Piper** | Lightweight, fast | Per-model voices | ✅ Stable |
| **Qwen3-TTS** | Voice design + cloning | 4 built-in + custom | ✅ Stable |
| **Fish Speech** | Voice cloning | Zero-shot cloning | ✅ Stable |

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

Clone any voice from a reference audio sample (Qwen3-TTS and Fish Speech):

```bash
# Via multipart upload
curl -sk https://localhost:8100/v1/audio/speech/clone \
  -F "input=Hello, I sound like the reference!" \
  -F "model=qwen3-tts-0.6b" \
  -F "reference_audio=@reference.wav" \
  -o cloned.mp3

# Via JSON with base64
curl -sk https://localhost:8100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-tts-0.6b","input":"Hello","voice":"default","reference_audio":"<base64>"}' \
  -o cloned.mp3
```

## Voice Design

Describe a voice in natural language and Qwen3-TTS will generate it:

```bash
curl -sk https://localhost:8100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-tts-0.6b",
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

## Roadmap

### ✅ Phase 4 (complete)

- Qwen3-TTS and Fish Speech backends with voice design + cloning
- Extended TTS API (`voice_design`, `reference_audio`, `/v1/audio/speech/clone`)
- Voice presets in web UI with YAML config support
- Version reporting fix, 332 tests

### Phase 5 (next)

- **Community model registry** — user-contributed model configs and voice presets
- **Benchmark dashboard** — CPU/GPU latency and quality comparison across backends
- **Multi-language expansion** — CJK, European language packs for Piper and Qwen3
- **Production cutover** — replace legacy split stack, migration tooling
- **Streaming voice cloning** — real-time clone with chunked reference audio

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
