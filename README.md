# Open Speech

**Self-hosted speech-to-text and text-to-speech with OpenAI-compatible APIs and a web app.**

[![Version](https://img.shields.io/badge/version-0.18.1-blue?style=flat-square)](CHANGELOG.md)
[![Docker Hub](https://img.shields.io/docker/pulls/jwindsor1/open-speech?style=flat-square&logo=docker)](https://hub.docker.com/r/jwindsor1/open-speech)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue?style=flat-square&logo=python)](https://python.org)

Open Speech runs speech models on your own machine, CPU or NVIDIA GPU, and serves them through OpenAI-style
endpoints, a WebSocket streaming API, Wyoming for Home Assistant, and a web app for reading, speaking,
transcribing and making voices. Current release: **v0.18.1** ([changelog](CHANGELOG.md)).

Use it in three ways:

1. **On its own**: text to speech and speech to text in the [web app](docs/WEB-APP.md) and the
   [API](docs/API.md). Nothing is sent to an AI.
2. **As the ears and mouth of another AI agent**: your agent sends audio to be written down and text to be spoken,
   and does its own thinking ([AI-AGENTS.md](docs/AI-AGENTS.md)).
3. **As the front end of your own AI**: Open Speech sends what you say, type or read to your OpenAI-compatible AI
   server and speaks or saves the reply ([AI-ASSISTANT.md](docs/AI-ASSISTANT.md)).

## Features

- **Speech-to-text**: NVIDIA Parakeet TDT 0.6B v2 by default (English, fast), and Whisper models through
  faster-whisper for 99+ languages and translation. Files, batch jobs, and live streaming over WebSocket with
  voice activity detection; `json`, `verbose_json`, `text`, `srt` and `vtt`; optional diarization.
- **Text-to-speech**: Kokoro by default, with voice blends such as `af_bella(2)+af_sky(1)`, plus Supertonic, Piper
  and Pocket-TTS. Streamed audio, the Live Reader for text that arrives token by token, a pronunciation
  dictionary, SSML and effects.
- **Optional GPU voice workers**: Breeze TTS 2, Qwen3, Chatterbox and Fish Audio S2, each in its own container.
- **Voice cloning and Design a voice**: clone a voice from a recording, or make one from a description, and save it
  as a named voice for any model that can speak it.
- **Web app**: Reader (books and documents read aloud), Speak, Transcribe, Talk, Voices, Studio, History, Models
  and Settings, in desktop and mobile browsers or installed as an app.
- **AI assistant**: talk to your own AI by voice, ask it about a transcript or a document, or have it rework a
  whole book.
- **OpenAI-compatible API**: `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/audio/translations`,
  `/v1/models` and `/v1/realtime`.
- **Wyoming** for Home Assistant voice pipelines.
- **GPU memory guard**: checks free GPU memory before each load and unloads idle models to make room.
- **Secure defaults**: HTTPS, an optional API key, Host and Origin checks, and rate limits.

## Quick start

### Docker run

```bash
docker run -d --name open-speech -p 8100:8100 \
  -v open-speech-data:/home/openspeech/data \
  -v open-speech-models:/home/openspeech/.cache/huggingface \
  -v open-speech-certs:/var/lib/open-speech/certs \
  jwindsor1/open-speech:cpu
```

With an NVIDIA GPU:

```bash
docker run -d --name open-speech -p 8100:8100 --gpus all \
  -v open-speech-data:/home/openspeech/data \
  -v open-speech-models:/home/openspeech/.cache/huggingface \
  -v open-speech-certs:/var/lib/open-speech/certs \
  jwindsor1/open-speech:latest
```

Keep the `-v` volumes: your voices, recordings, Reader documents, history and AI settings are written under
`/home/openspeech/data`, and downloaded models under `/home/openspeech/.cache/huggingface`. Without them both are
lost when the container is removed, which is how you upgrade: `docker pull jwindsor1/open-speech:cpu` (or
`:latest`), `docker rm -f open-speech`, then the same `docker run`.

### Docker Compose

Compose keeps the data in the checkout's `./data` folder and can add the GPU voice workers:

```bash
git clone https://github.com/jeremy-windsor/open-speech.git
cd open-speech
cp .env.example .env
docker compose up -d                                                   # CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d   # NVIDIA GPU
```

Name the files once in `.env` and every `docker compose` command (`up -d`, `logs -f`, `down`) uses them; separate
them with `;` instead of `:` on Windows:

```bash
COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml
# Optional: the core image (default :cpu, or :latest with the GPU file)
OPEN_SPEECH_IMAGE=jwindsor1/open-speech:latest
```

`.env.example` lists every setting with its default; [Configuration](docs/CONFIGURATION.md) explains them.

### First run

Open **https://localhost:8100/web** and accept the self-signed certificate. The first start downloads the
default text-to-speech model, Kokoro (about 330 MB), in the background. The first transcription downloads Parakeet;
on a GPU that first load, with a one-time conversion, takes about 7 minutes, so press **Download** on it in
**Models** to have it ready sooner.
Parakeet is English-only; for other languages pick a Whisper model in Transcribe or set `STT_MODEL`.

Requests addressed to an IP address, `localhost` or a LAN name such as `lan-box.local` work as they are. A DNS name
such as `speech.example.com` gets `400 Host not allowed` until you list it in `OS_ALLOWED_HOSTS`
([Security](docs/SECURITY.md)).

## Use it from other apps

Point an app or SDK written for OpenAI's audio API at `https://HOST:8100/v1`, with any API key (or `OS_API_KEY`
when the server has one). `tts-1` and `whisper-1` use the server's default models, and an Open Speech model ID
such as `kokoro` picks that model:

```python
import httpx
from openai import OpenAI

client = OpenAI(
    base_url="https://localhost:8100/v1",
    api_key="not-needed",  # use your OS_API_KEY if one is set
    http_client=httpx.Client(verify=False),  # accept the self-signed cert
)

with open("audio.wav", "rb") as f:
    result = client.audio.transcriptions.create(model="whisper-1", file=f)
print(result.text)

with client.audio.speech.with_streaming_response.create(
    model="kokoro", input="Hello world", voice="af_heart",
) as response:
    response.stream_to_file("output.mp3")
```

- **Voices.** OpenAI voice names such as `alloy` and `nova` map to Kokoro voices; other models take their own
  voice IDs from `GET /v1/audio/voices?model=MODEL` ([OpenAI voice names](docs/API.md#openai-voice-names)).
- **Open WebUI.** In Admin Panel › Settings › Audio, choose OpenAI for speech-to-text and text-to-speech, with the
  base URL above, the model `whisper-1` for STT, and `kokoro` (or `tts-1`) with a voice such as `af_heart` for TTS.
- **Home Assistant.** Add the Wyoming Protocol integration with the Open Speech host and port `10400`
  ([Wyoming](docs/API.md#wyoming)). Wyoming has no authentication.
- **Self-signed certificate.** Most apps other than browsers refuse it: turn HTTPS off on a trusted network
  (`OS_SSL_ENABLED=false`, then `http://HOST:8100/v1`), or give Open Speech a trusted certificate
  ([HTTPS](docs/API.md#https)).

The whole API, streaming and realtime included, is in [docs/API.md](docs/API.md).

## Models

| Model | Type | Default | Runs in | Notes |
|---|---|---|---|---|
| `nvidia/parakeet-tdt-0.6b-v2` | STT | Yes (images and Compose) | Core, GPU or CPU | English only, the fastest |
| `deepdml/faster-whisper-large-v3-turbo-ct2`, `Systran/faster-whisper-*` | STT | `Systran/faster-whisper-base` on a source install | Core, GPU or CPU | 99+ languages, and translation ([which models translate](docs/API.md#post-v1audiotranslations)) |
| `kokoro` | TTS | Yes | Core, GPU or CPU | Many voices, blends, OpenAI voice names |
| `supertonic` | TTS | | Core, GPU or CPU | Ten voices, blends, quality steps |
| `piper/*` | TTS | | Core | Small voices, one or more speakers each |
| `pocket-tts` | TTS | | Core | Built-in voices, streams |
| `qwen3/*` | TTS | | Qwen3 GPU worker | Built-in voices (Custom Voice), a voice from a description (VoiceDesign), cloning (Base) |
| `chatterbox/regular`, `chatterbox/turbo` | TTS | | Chatterbox GPU worker | English cloning |
| `fish/s2-pro` | TTS | | Fish Audio S2 GPU worker | Multilingual cloning |
| `breeze/tts-2` | TTS | | Breeze TTS 2 GPU worker | English and Chinese: a voice from a description, and cloning |

Sizes, languages, which models stay loaded and the GPU memory guard are in [docs/MODELS.md](docs/MODELS.md); how
each cloning model gets a voice is in [docs/CLONING-MODELS.md](docs/CLONING-MODELS.md).

## Optional voice workers

The GPU voice workers are off by default. Each has a Compose profile of its own in
`docker-compose.voice-models.yml`; name it to start that worker:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  -f docker-compose.voice-models.yml --profile qwen3 up -d --build
```

| Profile | Worker | What it adds |
|---|---|---|
| `--profile breeze` | Breeze TTS 2 | Design a voice from a description, and cloning, in English and Chinese |
| `--profile qwen3` | Qwen3 | Nine built-in voices, VoiceDesign, and cloning (Base) |
| `--profile chatterbox` | Chatterbox | English cloning, regular and Turbo |
| `--profile fish` | Fish Audio S2 | Multilingual cloning |

Or list the profiles in `.env`, as in `COMPOSE_PROFILES=qwen3,breeze`, together with
`COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml:docker-compose.voice-models.yml`. Only Qwen3 Custom Voice
has built-in voices; the others clone from a recording or design a voice in the web app's Voices view
([Cloning models](docs/CLONING-MODELS.md)). Stopping workers, and the settings each one reads, are in
[GPU voice workers](docs/CONFIGURATION.md#gpu-voice-workers).

## Configuration

Settings are environment variables, set in `.env`. The ones most people change:

| Setting | Default | What it does |
|---|---|---|
| `OS_API_KEY` | empty (no key) | Require `Authorization: Bearer KEY` on the API ([API key](docs/SECURITY.md#api-key)) |
| `OS_SSL_ENABLED` | `true` | HTTPS with a generated self-signed certificate; `false` serves plain HTTP |
| `OS_ALLOWED_HOSTS` | empty | DNS names served besides IP addresses and LAN names |
| `STT_MODEL` | Parakeet in the images | The speech-to-text model for requests that name none, or `whisper-1` |
| `TTS_MODEL`, `TTS_VOICE` | `kokoro`, `af_heart` | The voice for requests that name none, or `tts-1` |
| `COMPOSE_FILE`, `COMPOSE_PROFILES` | `docker-compose.yml`, none | The Compose files and GPU voice workers `docker compose` uses |
| `READER_MAX_UPLOAD_MB`, `READER_MAX_DOCUMENT_CHARS` | `50`, `5000000` | The largest file and text the Reader imports |
| `HF_TOKEN` | empty | A Hugging Face token for gated downloads |
| `OS_LOG_LEVEL` | `INFO` | `DEBUG` for [troubleshooting](docs/TROUBLESHOOTING.md#the-server-log) |

The AI assistant's server is set in the web app under Settings › AI assistant, or with `OS_ASSISTANT_URL`. Every
setting, the `.env` quoting rules and the Docker defaults are in [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

## Documentation

| Document | What it covers |
|---|---|
| [Web app](docs/WEB-APP.md) | Each section of the web app, shortcuts, presets, installing it as an app |
| [API reference](docs/API.md) | Every endpoint, errors and limits, streaming, Live Reader, realtime, Wyoming |
| [Models](docs/MODELS.md) | The model catalog, Parakeet, GPU worker models, residency, the GPU memory guard |
| [Configuration](docs/CONFIGURATION.md) | Every setting, `.env` rules, Docker Compose, image defaults, volumes |
| [Cloning models](docs/CLONING-MODELS.md) | The cloning models, giving one a voice, Design a voice, starting the workers |
| [Named voices and reading presets](docs/VOICE-IDENTITIES.md) | Named voices, their versions per model, and presets |
| [AI assistant](docs/AI-ASSISTANT.md) | Using your own AI from Talk, Ask AI and book jobs, and its reference |
| [AI agents](docs/AI-AGENTS.md) | Open Speech as the speech layer of an agent that does its own thinking |
| [Local listener](docs/LOCAL-LISTENER.md) | An always-on microphone listener smoke test |
| [TTS backends](docs/TTS-BACKENDS.md) | The backend contract, voice worker deployment, conformance checks |
| [Security](docs/SECURITY.md) | API key, host names, browser origins, reverse proxies, TLS |
| [Troubleshooting](docs/TROUBLESHOOTING.md) | The server log and common problems |
| [Development](docs/DEVELOPMENT.md) | Installing from source, tests, building images |
| [Changelog](CHANGELOG.md) | What changed in each release |

## Development

Open Speech needs Python 3.12, [uv](https://docs.astral.sh/uv/) and `ffmpeg` on `PATH`:

```bash
uv sync --frozen --extra cpu --extra all --extra dev
uv run --frozen --extra cpu --extra all --extra dev python -m pytest -q -rs tests
```

The CUDA install, running the server from source and building images are in
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

## License

[MIT](LICENSE) © 2026 Jeremy Windsor
