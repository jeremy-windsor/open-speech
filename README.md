# Open Speech

**Self-hosted speech-model harness with OpenAI-compatible APIs.**

[![Version](https://img.shields.io/badge/version-0.13.0-blue?style=flat-square)](CHANGELOG.md)
[![Docker Hub](https://img.shields.io/docker/pulls/jwindsor1/open-speech?style=flat-square&logo=docker)](https://hub.docker.com/r/jwindsor1/open-speech)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue?style=flat-square&logo=python)](https://python.org)

## What is Open Speech?

Open Speech is a self-hosted harness for running, comparing, and controlling speech models through
OpenAI-style endpoints. It currently provides:

- **Speech-to-text** via `faster-whisper`
- **Text-to-speech** via local backends such as **Kokoro**, **Supertonic**, **Piper**, and **Pocket-TTS**, plus isolated GPU providers
- **Streaming STT** over WebSocket
- **Realtime voice** over `/v1/realtime`, and an optional **AI assistant**: what you say or type, or a Reader
  document, goes to your own OpenAI-compatible AI server, and the reply is spoken or saved
  ([AI assistant](#ai-assistant))
- **Batch jobs**, a **web UI**, **history/profiles/conversations/composer**, and **Wyoming** integration

This repo is not a universal provider gateway. It is a pragmatic local speech-model harness with a
compatible API surface. Models and provider-specific controls remain explicit so the harness does not
pretend that every engine supports the same features.

## What it's for

Open Speech is used in three ways:

1. **On its own, with no AI.** Text to speech and speech to text in the [web app](#web-ui) (Speak, Transcribe,
   the Reader, Studio) and through the plain APIs below. Nothing is sent to an AI.
2. **The speech layer of another AI assistant or agent**: its ears and its mouth. Your agent sends audio to be
   written down and text to be spoken, and does its own thinking; Open Speech never calls an AI for it. Which
   endpoint fits which job, with the smallest working request for each, is in
   [docs/AI-AGENTS.md](docs/AI-AGENTS.md).
3. **The front end of an AI assistant.** Open Speech sends what you say, type or read to your own
   OpenAI-compatible AI server and speaks or saves its reply: the web app's **Talk** view, **Ask AI**, Reader
   book jobs, and `/v1/realtime` with the assistant on. It is set up in Settings › AI assistant; see
   [AI assistant](#ai-assistant) and [docs/AI-ASSISTANT.md](docs/AI-ASSISTANT.md).

| Job | Endpoint | Notes |
|---|---|---|
| Transcribe a file | `POST /v1/audio/transcriptions` | OpenAI's API: the whole file in, one answer out ([details](#post-v1audiotranscriptions)). `/v1/audio/translations` translates to English, and `/v1/audio/transcriptions/batch` queues up to 20 files |
| Transcribe a microphone as you speak | `WS /v1/audio/stream` | PCM16 in, interim and final transcripts out. Silero VAD ends an utterance after `OS_STREAM_ENDPOINTING_MS` (300 ms) of silence. `examples/listen.py` ([details](#ws-v1audiostream)) |
| Speak a reply | `POST /v1/audio/speech` | OpenAI's API. `?stream=true` with `response_format: "pcm"` sends audio as it is made ([details](#post-v1audiospeech)) |
| Speak a reply while the AI is still writing it | `WS /v1/audio/speech/stream` | The Live Reader: send each token as it arrives, and audio comes back sentence by sentence (`latency_mode` `natural`), clause by clause (`responsive`) or word by word (`instant_word`). One session at a time by default (`TTS_LIVE_MAX_CONNECTIONS`). `examples/live_tts_client.py` ([details](#live-reader-ws-v1audiospeechstream)) |
| Listen and speak at once | `WS /v1/realtime` | OpenAI's Realtime protocol: audio in, each turn transcribed, replies spoken word for word as your agent sends them, and talking over a reply stops it ([details](#realtime-audio)) |
| A voice satellite for Home Assistant | Wyoming, TCP port `10400` | The Wyoming protocol, with no authentication ([details](#wyoming)) |
| Let Open Speech ask your AI and speak the answer | `WS /v1/realtime` with the assistant on, `POST /api/assistant/ask`, `POST /v1/reader/documents/{id}/assistant` | Waiting phrases while the AI thinks, spoken errors, AI presets. The web app's Talk view and Ask AI use these ([details](#ai-assistant)) |

## Features

### Speech-to-Text
- OpenAI-compatible `/v1/audio/transcriptions` and `/v1/audio/translations`
- NVIDIA Parakeet TDT 0.6B v2 as the default model in the images: English only, several times faster than Whisper
  large-v3-turbo with better accuracy; Whisper models for 99+ languages and translation, loaded on demand
- Real-time streaming transcription via WebSocket at `/v1/audio/stream`
- Silero VAD support for streaming sessions
- `json`, `verbose_json`, `text`, `srt`, and `vtt` response formats
- ISO-639-1 language codes, with case, region suffixes, and `auto` accepted
- Optional diarization (`STT_DIARIZE_ENABLED=true` + the `diarize` extra, which installs pyannote),
  with speaker labels in every response format
- Optional preprocessing (noise reduction + normalization)
- Async batch transcription jobs, with uploads queued on disk

### Text-to-Speech
- OpenAI-compatible `/v1/audio/speech`
- Incremental Live Reader over `/v1/audio/speech/stream`
- Local TTS backends: Kokoro, Supertonic, Piper, Pocket-TTS
- Disk-backed TTS cache
- Pronunciation dictionary + basic SSML parsing
- Output post-processing (trim silence, normalize)
- Voice presets for the web UI
- Voices: a guided clone flow for recording or uploading local clone references, with quality checks,
  transcript verification, previews, clone tests, and presets, and a design flow that makes a voice from a
  description and saves it as a recording
- Named voices with explicit per-model realizations, separate reading presets, and preserved legacy profiles
- Kokoro voice blending using the `voice` field, e.g. `af_bella(2)+af_sky(1)`
- OpenAI voice names, including the newer `ash`, `coral`, `marin`, and `cedar`, mapped to Kokoro voices
- Reader: import epub, text, Markdown, or HTML documents and listen to them as rendered audio parts
- Model-specific capabilities and voice catalogs so controls only appear when the selected model supports them
- Isolated Qwen3, Chatterbox, Fish Audio S2, and Breeze TTS 2 GPU providers without adding their conflicting
  Torch and Transformers pins to the core harness
- Machine-readable TTS conformance report for provider metadata, controls, voices, and opt-in audio checks

See [Named voices and reading presets](docs/VOICE-IDENTITIES.md) for migration behavior
and speech API usage.

### Runtime / Platform
- Unified model browser + load/unload/download endpoints
- Default models pinned in memory, with idle (TTL) and least-recently-used eviction for the others
- `/v1/realtime` WebSocket endpoint: transcription and spoken replies, answered by your own AI server when the
  [AI assistant](#ai-assistant) is on, with waiting phrases, spoken errors and editable AI presets
- SQLite-backed studio features: profiles, history, conversations, composer
- Wyoming protocol support for Home Assistant-style integrations
- HTTPS with a generated self-signed certificate, renewed at startup once it expires within 30 days
- Secure defaults: Host-header validation, same-origin CORS and WebSockets, a cross-site write guard,
  and security headers
- API key auth with one-time WebSocket tickets, a failed-auth lockout, and rate limiting

## Quick Start

```bash
docker run -d --name open-speech -p 8100:8100 \
  -v open-speech-data:/home/openspeech/data \
  -v open-speech-models:/home/openspeech/.cache/huggingface \
  -v open-speech-certs:/var/lib/open-speech/certs \
  jwindsor1/open-speech:cpu
```

Open **https://localhost:8100/web** and accept the self-signed cert.

Keep the `-v` volumes. Everything you make, such as voices, recordings, Reader documents, history and AI
settings, is written under `/home/openspeech/data`, and the downloaded models under
`/home/openspeech/.cache/huggingface`. Without the volumes they stay inside the container and are lost when it is
removed, and removing it is how you run a newer image. To upgrade, run `docker pull jwindsor1/open-speech:cpu` (or
`:latest` for the GPU image), then `docker rm -f open-speech`, then the same `docker run` command.
[Docker Compose](#docker-compose) keeps this data in the checkout's `./data` folder instead.

Open Speech answers requests addressed to an IP address, `localhost`, or a LAN name such as `lan-box`
or `lan-box.local`. Any other DNS name, such as `speech.example.com`, gets `400 Host not allowed`
until you list it in `OS_ALLOWED_HOSTS`; see [Security](#security).

GPU example:

```bash
docker run -d --name open-speech -p 8100:8100 --gpus all \
  -v open-speech-data:/home/openspeech/data \
  -v open-speech-models:/home/openspeech/.cache/huggingface \
  -v open-speech-certs:/var/lib/open-speech/certs \
  jwindsor1/open-speech:latest
```

The image sets some defaults of its own that differ from the source defaults in `src/config.py`;
see [Docker image defaults](#docker-image-defaults). Both images transcribe with
[Parakeet](#parakeet), which is English-only: for other languages pick a Whisper model in Transcribe, send its ID as
`model`, or set `STT_MODEL` to it.

## Use it from an OpenAI app

Apps and SDKs written for OpenAI's audio API can use Open Speech without code changes. Point them at:

| Setting | Value |
|---|---|
| Base URL | `https://HOST:8100/v1` |
| API key | Any non-empty string. If `OS_API_KEY` is set on the server, use that key. |
| TTS model | `tts-1`, `tts-1-hd`, and `gpt-4o-mini-tts` use the configured `TTS_MODEL` (default `kokoro`). An Open Speech ID such as `kokoro` or `piper/en_US-lessac-medium` picks that model. |
| STT model | `whisper-1`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, or no model at all use the configured `STT_MODEL` (translations use `STT_TRANSLATE_MODEL` when it is set: [which models translate](#post-v1audiotranslations)). An Open Speech ID such as `deepdml/faster-whisper-large-v3-turbo-ct2` picks that model. |
| Realtime model | `gpt-realtime`, `gpt-4o-realtime-preview`, and `gpt-4o-mini-realtime-preview` (with any `-...` suffix, such as a date) speak with `TTS_MODEL` on `/v1/realtime`. |
| Voice | Kokoro maps OpenAI voice names to its own voices (see below). Other models need their own voice IDs. |

The STT name mapping applies to `/v1/audio/transcriptions`, `/v1/audio/translations`,
`/v1/audio/transcriptions/batch`, the `/v1/audio/stream` WebSocket, and `/v1/realtime` input
transcription.

OpenAI voice names map to Kokoro voices:

| OpenAI voice | Kokoro voice | OpenAI voice | Kokoro voice |
|---|---|---|---|
| `alloy` | `af_heart` | `ash` | `am_fenrir` |
| `echo` | `am_adam` | `ballad` | `bm_george` |
| `fable` | `bf_emma` | `coral` | `af_aoede` |
| `onyx` | `am_michael` | `sage` | `af_sarah` |
| `nova` | `af_nova` | `verse` | `am_puck` |
| `shimmer` | `af_bella` | `marin` | `af_heart` |
| | | `cedar` | `am_michael` |

`marin` and `cedar` share the voices of `alloy` and `onyx`. Kokoro applies the mapping wherever it
speaks, including `/v1/realtime` replies. A mapped name is not resolved inside a blend. Piper models
also accept `alloy` as their default speaker. Supertonic maps each name to a voice of the same gender as on Kokoro:
`alloy`/`marin` F1, `nova` F2, `shimmer` F3, `coral` F4, `sage`/`fable` F5, `onyx`/`cedar` M1, `echo` M2, `ash` M3,
`ballad` M4, `verse` M5. Other OpenAI voice names, and other models (Pocket-TTS,
Qwen3, Chatterbox, Fish, Breeze), need a voice ID from `GET /v1/audio/voices?model=MODEL`. If you change
`TTS_MODEL` away from Kokoro, apps that send `voice: "alloy"` will get an error unless the new model
accepts it.

`/v1/audio/speech` fills in fields the client leaves out from the server settings: `model` from
`TTS_MODEL`, `speed` from `TTS_SPEED` (only on a model with speed control; a fixed-speed model such as
Pocket-TTS or Chatterbox speaks at 1.0), and `response_format` from `TTS_DEFAULT_FORMAT`. `voice` comes
from `TTS_VOICE` only when the model also came from `TTS_MODEL` (left out, or an OpenAI model name),
because a voice ID belongs to one model. Anything the client sends explicitly wins. Unknown JSON fields
are ignored, so newer OpenAI parameters do not break requests; the retired `voice_blend` field is still
rejected with `422`. When an OpenAI model name (`tts-1`, `tts-1-hd`, `gpt-4o-mini-tts`) arrives with
`instructions` and `TTS_MODEL` cannot follow instructions, the instructions are ignored; an Open Speech
model ID that cannot follow them gets `400`.

**HTTPS note:** Open Speech uses a self-signed certificate by default, and most third-party apps refuse
self-signed certificates. Pick one:

- On a trusted home network, turn HTTPS off with `OS_SSL_ENABLED=false` and use `http://HOST:8100/v1`.
- Put Open Speech behind a reverse proxy (Caddy, nginx, Traefik) that has a real certificate; see
  [Reverse proxies](#reverse-proxies) for the settings it needs.
- Give Open Speech a trusted certificate with `OS_SSL_CERTFILE` and `OS_SSL_KEYFILE` (set both).

## Installation (from source)

Open Speech supports **Python 3.12 only**. The server also runs the `ffmpeg` program, which `uv sync` doesn't
install, so put it on `PATH` first: `sudo apt install ffmpeg`, `brew install ffmpeg`, or
`winget install Gyan.FFmpeg`. Without it every mp3, opus, aac and flac answer fails with `500 ffmpeg not found`,
and mp3 is what an OpenAI app gets by default, what the web app's Speak and voice previews ask for, and what every
Reader part and Composer mix is made of. Install [uv](https://docs.astral.sh/uv/), then:

```bash
git clone https://github.com/jeremy-windsor/open-speech.git
cd open-speech
uv sync --frozen --extra cpu --extra all  # Locked CPU runtime and core TTS providers
# Add --extra dev for development, or select tts, piper, pocket, or supertonic individually.
```

For CUDA 12.8, install with this command instead, and set the devices in `.env` (below):

```bash
uv sync --frozen --extra cuda --extra all --no-install-package onnxruntime
```

The cuda extra brings `onnxruntime-gpu`, which provides the same `onnxruntime` Python package as the CPU build that
faster-whisper and Piper require; with both installed they overwrite each other's files, so ONNX Runtime can lose
its CUDA provider or break outright. `--no-install-package onnxruntime` leaves the CPU build out. For the same
reason, run commands in a CUDA install with `uv run --no-sync` (or set `UV_NO_SYNC=1` in the shell): a `uv run`
that syncs the environment installs the CPU build again.

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
READER_CACHE_DIR=data/reader
```

For a CUDA install (`--extra cuda`), add these lines too. Without them Whisper and every TTS model run on the CPU,
the source default:

```bash
STT_DEVICE=cuda
STT_COMPUTE_TYPE=float16
TTS_DEVICE=cuda
```

Batch job uploads and the batch job database go next to `OS_STUDIO_DB_PATH`.

`uv run --env-file` reads `.env` by rules of its own. It expands a `$` outside single quotes, and `$$` is no
escape there, so `Xk9$$mP2q` arrives as `Xk9` and `$ecretKey123` as nothing (an empty `OS_API_KEY` turns
authentication off). It drops each double quote, so `{"top_k":20}` arrives as `{top_k:20}`, which is not JSON. And
a blank or a backslash in a value that isn't quoted makes it stop reading `.env` at that line, with only a warning:
that setting and every one below it stay unset. So single-quote such a value, as in `OS_API_KEY='Xk9$mP2q'`,
`OS_ASSISTANT_EXTRA_BODY='{"top_k": 20}'` or `TTS_PRONUNCIATION_DICT='C:\Users\me\dict.yaml'`, which Compose also
reads as written, or export it in the shell instead. An apostrophe (`'`) in a value that isn't quoted starts a quote
that runs to the next apostrophe, even lines below. One apostrophe makes uv stop reading `.env` at that line, so
`OS_API_KEY=Xk9'mP2q` leaves the key and `OS_AUTH_REQUIRED` below it unset, with authentication off; two are dropped
(`it's-Bob's` arrives as `its-Bobs`); `OS_ASSISTANT_API_KEY` is cut the same way. Single quotes can't hold an
apostrophe, so write it as `\'`, as in `OS_API_KEY=Xk9\'mP2q`, or export the value in the shell. Compose reads
`Xk9'mP2q` as written but keeps the backslash of `\'`, so a `.env` that both read needs a key without an apostrophe.
A key of letters, digits, `-` and `_` avoids all of this:
`python -c "import secrets; print(secrets.token_urlsafe(32))"` prints one.

Then start the server:

```bash
uv run --frozen --env-file .env python -m src.main
# A CUDA install: uv run --no-sync --env-file .env python -m src.main
```

It listens on `http://localhost:8100` (web UI at `/web`). To keep HTTPS, remove `OS_SSL_ENABLED=false`
and set both `OS_SSL_CERTFILE` and `OS_SSL_KEYFILE` to a writable cert/key path; Open Speech generates a
self-signed pair there if neither file exists yet. If only one of the two files exists, startup stops
instead of overwriting it. Generating the pair runs the `openssl` program, which must then be on `PATH` (Linux and
macOS have one; on Windows, Git for Windows ships one in its `usr\bin` folder), or startup stops with
`openssl not found`; a pair of your own needs no `openssl`.

At startup the server also loads `TTS_MODEL` (Kokoro by default) in the background when its provider
is installed. On the first start this downloads the model's weights. Set `TTS_PRELOAD_DEFAULT=false`
to load it on first use instead.

## Models

Models are downloaded on demand and cached on disk.
The tables below highlight common IDs rather than the complete curated catalog. `src/model_registry.py`
contains all 15 STT and 37 TTS entries; `/api/models` reports the catalog available to the configured
running harness.

### STT Models

| Model | Size | Backend | Languages |
|---|---:|---|---|
| `nvidia/parakeet-tdt-0.6b-v2` | ~3.7GB (GPU), ~0.7GB (CPU) | Parakeet (ONNX) | English |
| `deepdml/faster-whisper-large-v3-turbo-ct2` | ~1.6GB | faster-whisper | 99+ |
| `Systran/faster-whisper-large-v3` | ~3.0GB | faster-whisper | 99+ |
| `Systran/faster-whisper-medium` | ~1.5GB | faster-whisper | 99+ |
| `Systran/faster-whisper-small` | ~500MB | faster-whisper | 99+ |
| `Systran/faster-whisper-base` | ~150MB | faster-whisper | 99+ |
| `Systran/faster-whisper-tiny` | ~75MB | faster-whisper | 99+ |

### Parakeet

`nvidia/parakeet-tdt-0.6b-v2` is the default STT model of the Docker images and the Compose files. It runs NVIDIA's
Parakeet TDT 0.6B v2 through [onnx-asr](https://github.com/istupakov/onnx-asr) and ONNX Runtime, from the
published ONNX export `istupakov/parakeet-tdt-0.6b-v2-onnx` at a pinned revision.

- **English only.** A request with another `language` gets `400 language_unsupported`, naming a Whisper model to
  use instead; no language, `auto` and `en` (`en-US`, `en-GB`, ...) are accepted. Translations get
  `400 translation_unsupported` (set `STT_TRANSLATE_MODEL` to a Whisper model that translates). Wyoming lists it
  with English only. Punctuation and capitals are part of its output.
- **Fastest.** On an 8 GB RTX 2070 SUPER a 3 s command takes about 0.04 s and a 10 s sentence 0.09 s, against
  0.29 s and 0.44 s for Whisper large-v3-turbo, at a lower word error rate on English. On an 8-core CPU (int8) it
  is about as fast as Whisper base with turbo's accuracy.
- **Files.** On a GPU the 2.5 GB fp32 export is downloaded and converted to fp16 once, on the first load: about
  7 minutes in all, most of it the download, and up to about 6 GB of free RAM for the conversion, which runs in a
  process of its own (if it runs out of memory, the load fails with a message saying so and the server keeps
  running). The 1.2 GB result is kept beside the download in the model cache, so the cache holds about 3.7 GB, and
  later loads read it directly. On a CPU the published int8 files (0.7 GB) are used as they are.
  `STT_COMPUTE_TYPE` applies to Whisper only. So that the first request doesn't wait for this, press Download on
  it in Models once the server is up, or set `STT_PRELOAD_MODELS=nvidia/parakeet-tdt-0.6b-v2`: the preload runs in
  the background after the server starts, so the server answers meanwhile, and a request for Parakeet before it
  finishes waits for that same download and load instead of starting another.
- **Memory.** About 1.5 GB of GPU memory once loaded. ONNX Runtime's memory pool is capped at
  `STT_PARAKEET_GPU_MEM_LIMIT_MB` (2304 MB), so Parakeet and Kokoro stay loaded together on an 8 GB card with
  room for a Whisper model on demand. Long audio is cut with Silero VAD into pieces of at most 20 s (a stretch with
  no pause is cut at its quietest point), and streaming STT ends an utterance at about 24 s.
- **Requests.** `prompt` and `temperature` are accepted and ignored: its decoder is greedy and takes no prompt.
  `verbose_json` has one segment per sentence, with `start`, `end`, `avg_logprob` and word times in `words`; `srt`
  and `vtt` cues follow the segments.

To make it the default on a source install, or to choose another default, set `STT_MODEL` (in `.env`, or in Compose
through `.env`). A request that names a Whisper model loads it next to the pinned Parakeet, and `OS_MODEL_TTL`
unloads it after it has been idle.

### TTS Models

| Model | Size | Backend | Notes |
|---|---:|---|---|
| `kokoro` | ~330MB | Kokoro | default backend, many voices, blend syntax in `voice` |
| `pocket-tts` | ~220MB | Pocket-TTS | built-in voices, backend advertises streaming support |
| `supertonic` | ~383MB | Supertonic 3 (ONNX) | voices `M1`–`M5` and `F1`–`F5`, blends, optional `steps` (1–32, default 8), speed 0.7–1.9 (others are held to it, named in `X-Speed-Applied`); GPU when ONNX Runtime has CUDA, else CPU (about 0.6 GB of RAM) |
| `piper/en_US-lessac-medium` | ~60MB | Piper | one voice per model |
| `piper/en_US-joe-medium` | ~60MB | Piper | one voice per model |
| `piper/en_US-amy-medium` | ~60MB | Piper | one voice per model |
| `piper/en_US-arctic-medium` | ~73MB | Piper | 18 speakers (`piper/en_US-arctic-medium/<speaker>`; list them with `/v1/audio/voices?model=piper/en_US-arctic-medium`) |
| `piper/en_GB-alan-medium` | ~60MB | Piper | one voice per model |
| `qwen3/0.6b-custom-voice` | ~1.8GB | isolated Qwen3 provider | 9 official preset voices |
| `qwen3/0.6b-base` | ~1.8GB | isolated Qwen3 provider | reference cloning with an exact transcript; hidden unless `QWEN3_ENABLE_BASE=true` (the voice-models Compose file sets it) |
| `chatterbox/regular` | ~8.6GiB | isolated Chatterbox provider | English reference cloning |
| `chatterbox/turbo` | ~5.4GiB | isolated Chatterbox provider | faster English cloning, native speech tags |
| `fish/s2-pro` | ~4.7GiB | isolated Fish Audio S2 provider | multilingual cloning with an exact transcript, 4-bit (NF4) S2 Pro |
| `breeze/tts-2` | ~3.3GiB | isolated Breeze TTS 2 provider | English and Chinese voices designed from a description, and cloning with an exact transcript; Q8_0 GGUF through a C++ port |

The optional model sizes are approximate disk cache usage, not model VRAM. First load may
include model downloads. See [TTS Backends](docs/TTS-BACKENDS.md) for the provider bundle
and conformance commands. Chatterbox, Qwen3 Base, Fish Audio S2 and Breeze TTS 2 are cloning models, with no voices of their own:
[Cloning models](docs/CLONING-MODELS.md) gives each one's languages, how to give it a voice, and how to start
its worker.

### Model residency

- A request loads the model it needs on first use (transcription, speech, Live Reader, Reader,
  realtime, Wyoming). That model stays loaded next to the ones already loaded.
- An explicit load (`POST /api/models/{id}/load`, `POST /v1/audio/models/load`, or the legacy STT
  `POST /api/ps/{model}`) first unloads the other loaded models of the same type, STT or TTS.
  `POST /api/models/{id}/load?keep_pinned=true` leaves the pinned default of that type loaded (see below), as a
  request that loads a model on demand does. The web app loads STT models this way, from Transcribe and from
  Models › Load, so a Whisper model loads next to the pinned Parakeet; the Load row says which models it unloads
  and which stay loaded. Each `GET /api/models` row's `pinned` says whether it is a pinned default.
- A legacy `POST /api/ps/{model}` whose load fails loads back the STT models it unloaded first, the pinned default
  among them, so a mistyped or broken model doesn't leave the server without one.
- Download (`POST /api/models/{id}/download`) fetches files without loading them for faster-whisper, Parakeet,
  Kokoro, Supertonic, and Piper. Pocket-TTS and the isolated workers (Qwen3, Chatterbox, Fish, Breeze) have no
  download-only path, so they load once and unload again.
- `STT_MODEL` and `TTS_MODEL` are pinned by default (`OS_PIN_DEFAULT_MODELS=true`): idle and LRU
  eviction never unload them, and they load on top of `OS_MAX_LOADED_MODELS`, which then caps the
  other models. An explicit unload, or an explicit load of another model of the same type, still
  replaces them. `TTS_PRELOAD_DEFAULT=true` loads `TTS_MODEL` in the background at startup, and
  `STT_PRELOAD_MODELS` loads its models in the background once the server has started, next to the ones loaded.
- A `/v1/realtime` session whose transcription `language` its STT model can't transcribe (any but English for
  Parakeet) gets one `error` event with code `language_unsupported` after its `session.updated`, and its turns are
  transcribed in English.
- Every 30 seconds, models idle longer than `OS_MODEL_TTL` are unloaded, then the least recently used
  models beyond `OS_MAX_LOADED_MODELS`. A model that is transcribing, synthesizing, streaming, or
  loading is skipped and retried on the next pass, so the cap can be exceeded while only busy models
  are left beyond it.

#### GPU memory guard

Before a model loads onto the GPU, the server checks the card's free memory, whatever starts the load: Models › Load,
a request's first use, a preload or a worker. The model's size, `OS_GPU_HEADROOM_MB`, and the room the pinned models
need to grow while they run (Parakeet up to `STT_PARAKEET_GPU_MEM_LIMIT_MB`, Kokoro about 300 MiB) must all fit. When
they don't, idle models that aren't pinned are unloaded, least recently used first. When you load a speech model
yourself, the idle pinned STT model may be unloaded too; the next transcription loads it back. If there is still not
enough room, the load fails with `503 insufficient_gpu_memory`: the message says what to unload, and `detail.gpu`
gives the figures in MiB. Models on the CPU aren't checked, and neither is anything on a server without an NVIDIA GPU.

- `OS_GPU_MEMORY_GUARD` (`true`): turns the guard on or off.
- `OS_GPU_HEADROOM_MB` (`600`): MiB kept free on top of a model's size.
- `OS_GPU_MAX_BIG_MODELS` (`auto`): how many models of 2 GiB or more (GPU worker models, large Whisper) can be loaded at
  once besides the pinned defaults. Loading one more first unloads the least recently used idle one. With `auto`, an
  8 GB card takes 1, a 12 to 16 GB card 2, and a 24 GB card has no cap.

## API Reference

API endpoints use Bearer auth (`Authorization: Bearer KEY`) when `OS_API_KEY` is set. `/health`,
`/openapi.json`, `/web`, the web UI's static files, the signed Reader audio links under
`/v1/reader/media/`, and `/docs` and `/redoc` when they are on, need no key. WebSocket endpoints also accept a one-time `?ticket=` from
`POST /api/auth/ws-ticket`, for browsers that cannot set headers on a WebSocket. The server still
accepts a deprecated `?api_key=KEY` query parameter as well, and logs a warning each time. Open Speech's
own log shows the values of `api_key`, `ticket`, `exp` and `sig` in a query string as `***` (in uvicorn's
request lines, at every `OS_LOG_LEVEL`), but a reverse proxy's logs and browser history still keep the key,
so prefer the header or a ticket. See [Security](#security) for
the Host, Origin, lockout, and rate-limit checks that run before a route.

Errors are JSON: `{"error": {"message": "...", "code": "..."}}`. Request validation errors are `422`
with code `validation_error`; refusals from the security checks may carry only `message`. A `422`'s
message lists at most the first 10 errors, in FastAPI's format (`N validation errors:`, then one per
line), with each value cut short, and counts the rest (`... and N more`). JSON lists are bounded: a
request holds at most 500 conversation `turns`, 64 composer `tracks`, and 16 `effects` in a speech
request, a preset, a turn or a track; a longer list is one `too_long` error. The AI assistant's routes
also name the field in `param`.
The OpenAPI schema is at `/openapi.json`. The interactive pages, Swagger UI at `/docs` and ReDoc at
`/redoc`, are off by default because they load scripts from a CDN on the server's own origin, where the web
UI keeps its API key; set `OS_API_DOCS=true` to serve them.

### Authentication

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/auth/ws-ticket` | Returns `{"ticket": "...", "expires_in": 60}` for one WebSocket connection |

The endpoint itself needs the API key. Connect within 60 seconds with `?ticket=TICKET` on any
WebSocket endpoint; a ticket works once and lives in the server's memory, so a restart discards it.
Fetch a new ticket for every connection, reconnects included. A socket without a valid ticket or
key is opened and then closed at once with `4001 Invalid or missing API key`
([Refused WebSockets](#refused-websockets)). A wrong, expired, or reused ticket or key also counts toward the
[failed-auth lockout](#api-key); a handshake that sends no credentials does not. A page on another
origin also needs that origin in `OS_CORS_ORIGINS`, because its ticket request is a cross-site write
(see [Browser origins](#browser-origins)).

```bash
curl -sk -X POST -H "Authorization: Bearer my-secret-key" https://localhost:8100/api/auth/ws-ticket
# {"ticket":"TICKET","expires_in":60}  ->  wss://localhost:8100/v1/audio/stream?ticket=TICKET
```

### Refused WebSockets

A WebSocket that is refused is accepted and then closed at once, with no event first, so the client, a browser
included, gets the close code and the reason:

| Close code | Reason | When |
|---|---|---|
| `4001` | `Invalid or missing API key` | `OS_API_KEY` is set, and the socket has no key, a wrong one, or a used, expired or unknown `?ticket=` |
| `1008` | `Origin not allowed. List it in OS_WS_ALLOWED_ORIGINS to allow it.` | The page's `Origin` isn't allowed ([Browser origins](#browser-origins)) |
| `4004` | `Live TTS is disabled` | `/v1/audio/speech/stream` with `TTS_ENABLED` or `TTS_LIVE_ENABLED` false |
| `4004` | `Realtime API is disabled` | `/v1/realtime` with `OS_REALTIME_ENABLED=false` |

`/v1/audio/stream` has no switch of its own; it also closes with `1008` for a query value it can't take, with a
reason that names it (`Invalid sample_rate: ...`, `Invalid endpointing: ...`, `Invalid vad: ...` or
`Invalid interim_results: ...`, see [`WS /v1/audio/stream`](#ws-v1audiostream)). Nothing is read and no session starts. Don't reconnect after one
of these closes: the same socket gets the same answer.

**Upgrade note:** these refusals used to fail the handshake, which a client that reads it saw as HTTP `403` and
a browser only as close code `1006`, with no reason. A client that took a failed handshake as the refusal now sees
the socket open and then close, so it must read the close code.

Two refusals still answer the handshake with HTTP, before any route runs: a `Host` that isn't accepted gets `400`
(`OS_ALLOWED_HOSTS`, [Host names](#host-names)), and the rate limit or the failed-auth lockout gets `429` with
`Retry-After` ([Rate limiting](#rate-limiting), [API key](#api-key)).

The Python client's `stream_transcribe` and `async_stream_transcribe` raise `src.client.StreamRefusedError` (a
`RuntimeError` with `.code` and `.reason`) on a `4001`, `1008` or `4004` close, and don't reconnect. The
JavaScript SDK does the same with its own `StreamRefusedError` (`client-js/README.md`, "Refused sockets").
`examples/listen.py` and `examples/voice_chat.py` print one `[refused]` line with the code, the reason and what to do.

### Upload and body limits

The server refuses an oversized request body while it arrives. A declared `Content-Length` over the
limit gets `413` before anything is read, and a body that grows past it is cut off with
`413 Request body too large. Max: NMB`. The limit is `READER_MAX_UPLOAD_MB` for
`POST /v1/reader/documents`, the larger of `OS_MAX_UPLOAD_MB` and `OS_BATCH_MAX_TOTAL_MB` for
`POST /v1/audio/transcriptions/batch`, and `OS_MAX_UPLOAD_MB` for every other route. Multipart and
other bodies get 1 MB on top for form fields. A JSON body may be the larger of 4/3 of the limit plus
64 KB (room for base64) and the limit plus 1 MB, so with the default 100 MB any JSON request is capped
at about 133 MB. Routes then apply their exact limits, such as `413 Upload too large. Max: 100MB` for
an audio file.

### Speech-to-Text

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/audio/transcriptions` | Transcribe audio |
| `POST` | `/v1/audio/translations` | Translate audio to English ([which models translate](#post-v1audiotranslations)) |
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
- `file` (required) — up to `OS_MAX_UPLOAD_MB` (`413` above it) and 4 hours of audio, at a sample rate
  of 8,000–192,000 Hz
- `model` — default `STT_MODEL`; OpenAI names map to it
- `language` — an ISO-639-1 code such as `en`. Case and a region suffix are accepted (`EN`, `en-US`,
  `pt_BR`), as are `auto` (auto-detect) and the older codes `nb`, `jv`, `iw`, `in`, and `ji`. Anything
  else gets `400 Invalid language '...'. Use an ISO-639-1 code such as 'en'`. Parakeet transcribes English only:
  another language gets `400 language_unsupported` ([Parakeet](#parakeet)).
- `prompt` — Whisper's initial prompt; Parakeet ignores it
- `response_format` = `json | verbose_json | text | srt | vtt`
- `temperature` — default `0`, which uses faster-whisper's temperature fallback (0.0, 0.2, … 1.0);
  other values are used as given. Parakeet ignores it
- `diarize` — form field or `?diarize=true`; needs `STT_DIARIZE_ENABLED=true` (`400` otherwise)

Audio longer than 4 hours gets `400 Audio is longer than 14400 seconds; split it into shorter files`,
and a sample rate outside the range gets `400 Unsupported sample rate N Hz; audio must be 8000-192000 Hz`.
Both limits also apply to `/v1/audio/translations` and to each batch file. An empty upload, or one that decodes
to no audio samples (such as a WAV header with no frames), gets `400 Empty audio file`, on
`/v1/audio/translations` too.

With `diarize`, every `response_format` carries speakers: `text` gives one `SPEAKER_XX: ...` line per
speaker run, `srt` prefixes each cue with `SPEAKER_XX: `, `vtt` uses `<v SPEAKER_XX>` voice spans,
`verbose_json` adds `speaker` to each Whisper segment, and `json` returns
`{text, segments: [{speaker, start, end, text}]}`. Pyannote's models are gated: if their download is
refused, the response is a `400` that names the models to accept on Hugging Face and `HF_TOKEN`.
Diarization runs on CUDA when `STT_DEVICE` is `cuda` or `auto` and CUDA is available.

Send `X-History: true` to log the transcription in `/api/history` (when `OS_HISTORY_ENABLED=true`).

```bash
curl -sk https://localhost:8100/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=deepdml/faster-whisper-large-v3-turbo-ct2" \
  -F "response_format=json"
```

#### `POST /v1/audio/translations`

The same upload as a transcription, without `language` and `diarize`; the text comes back in English, and
`verbose_json` says `"language": "en"` (OpenAI spells it `english`). `model`
`whisper-1`, another OpenAI name, or no model uses `STT_TRANSLATE_MODEL`, or `STT_MODEL` while that is empty.

Not every model translates. Parakeet (the default `STT_MODEL` of the images and Compose) transcribes English only,
large-v3-turbo answers in the speaker's own language, and English-only models (`*.en`, the distil models) only
transcribe, so a translation on one of them gets
`400 translation_unsupported` ("MODEL can't translate to English. Use Systran/faster-whisper-medium or
Systran/faster-whisper-large-v3, ..."), before anything is decoded. To make `whisper-1` translate on such a
deployment, set `STT_TRANSLATE_MODEL=Systran/faster-whisper-medium` (1.5 GB, downloaded on the first translation)
or `Systran/faster-whisper-large-v3` (3.0 GB); it loads next to `STT_MODEL` while in use.

```bash
curl -sk https://localhost:8100/v1/audio/translations \
  -F "file=@spanish.wav" \
  -F "model=Systran/faster-whisper-medium"
```

#### `POST /v1/audio/transcriptions/batch`

Multipart form with up to 20 `file` parts, each up to `OS_MAX_UPLOAD_MB` and together up to
`OS_BATCH_MAX_TOTAL_MB` (`413` otherwise), plus `model`, `language`, `response_format`, and
`temperature` as above. It answers with `job_id`, `status: "queued"`, `file_count`, and `created_at`,
or with `429` while `OS_BATCH_MAX_PENDING` jobs are queued or running. Queued uploads wait on disk until their
job ends. A restart fails every queued or running job with the reason
`Server restarted before batch processing completed` and deletes their uploads.

Up to `OS_BATCH_WORKERS` jobs run at once, but only one batch file is transcribed at a time, so live
STT never waits behind more than one batch file. A file that fails (for example, one longer than 4
hours) fails on its own, and the job's other files still run. A file that decodes to no audio samples fails with
`status: "failed"` and `error: "Empty audio file"`; a 0-byte file is refused when the job is submitted, with
`400 File 'NAME' is empty`. `GET /v1/audio/jobs` takes `limit`
(1–200, default 50) and `status`. `GET /v1/audio/jobs/{job_id}/result` answers `409` until the job is
done: `{"error": "Job not complete", "status": ..., "retry_after": 5}`, or for a failed job
`{"error": "Job failed", "status": "failed", "reason": ...}`. `DELETE` on a running job stops it after
the file in progress.

#### `WS /v1/audio/stream`

Send PCM16 audio chunks, receive transcript/VAD events.

**Query params:** `model` (default `STT_MODEL`; OpenAI names map to it), `language` (codes as on
`/v1/audio/transcriptions`), `sample_rate` (default `16000`; 8,000–192,000, otherwise the socket is opened and
closed at once with `1008 Invalid sample_rate: must be 8000-192000`), `encoding` (`pcm_s16le`, the default, or
Deepgram's `linear16`), `interim_results` (default
`true`), `endpointing` (milliseconds of silence that end an utterance, 0–30,000; default
`OS_STREAM_ENDPOINTING_MS`), and `vad` (default `STT_VAD_ENABLED`). An `interim_results`, `endpointing` or `vad`
the server can't read, or an `endpointing` outside its range, closes the socket the same way, with `1008` and a
reason that names it, such as `Invalid vad: must be true or false`.

With VAD on, an utterance starts 300 ms before the first window that Silero scores as speech, but never
before the previous utterance ends. That lead-in is sent to the model, and the transcript's `start`
includes it.

Any other `encoding` gets an `error` event and close code `1003 Unsupported encoding` before any audio
is read. The first event is `session.begin`, with `session_id`, `model`, `sample_rate`,
`internal_sample_rate`, `vad_enabled`, and `language` (the Whisper code in use, or `null` for
auto-detect). An `error` event before `session.begin` means the stream did not start: the server then
closes the socket with `1011` and `Failed to load model` or `Failed to load VAD model`, and a reconnect
would only try the same load again. An unknown
`language` is reported by an `error` event right after `session.begin`, and the stream continues with
auto-detect. Parakeet, which is English only, reports any other language the same way and continues in
English. Send `{"type":"config","sample_rate":N}` to change the rate and `{"type":"stop"}` to end;
the server answers with `session.end` and closes the socket with `1000 Stream ended`. A server failure
during the stream also sends `session.end`, then closes with `1011 Streaming session failed`. An utterance
is finalized after 30 seconds at most (about 24 s for Parakeet). Above
`OS_STREAM_MAX_CONNECTIONS` streams, a new one is closed with `1013`. A missing or wrong key closes the socket
with `4001`, and a page's `Origin` that isn't allowed with `1008`, right after it opens
([Refused WebSockets](#refused-websockets)).

Each `transcript` event has `text`, `is_final`, `speech_final`, `start` and `end` (seconds into the stream).
`is_final: false` is interim text, which can still change; `is_final: true` with `speech_final: false` holds
the words confirmed so far; `speech_final: true` holds the whole utterance once it has ended. With VAD on,
`vad` events (`state`: `speech_start`, `speech_end`) frame each utterance.

At INFO, the server's default log level, the server log holds no transcript text. Each decode is logged by
its size, such as `Transcription #3: 47 chars (0.5s audio)` (`0 chars` when it came back empty),
`Confirmed: 9 words` and `Final utterance: 47 chars (0.6s audio)`. The text goes only to DEBUG lines
(`OS_LOG_LEVEL=DEBUG`). When a session ends, one INFO line says how much audio arrived and how it scored;
see [Nothing transcribed?](#nothing-transcribed).

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
| `GET` | `/v1/audio/voices` | List voices; for a cloning model, the named voices that have a version for it |
| `GET` | `/v1/audio/models` | List the usable TTS models and their load state (`?loaded=true`: loaded only) |
| `POST` | `/v1/audio/models/load` | Load TTS model |
| `POST` | `/v1/audio/models/unload` | Unload TTS model |
| `GET` | `/api/tts/capabilities` | TTS backend capabilities |
| `GET` | `/api/voice-presets` | Voice presets for the UI |

#### `POST /v1/audio/speech`

JSON body.

**Fields:**
- `model`
- `input` — up to `TTS_MAX_INPUT_LENGTH` characters (`400 Input too long. Max: N characters`)
- `voice` — voice ID string or an object with `id`; named-voice UUIDs resolve their model realization
- `speed` — 0.25–4.0
- `response_format` = `mp3 | opus | aac | flac | wav | pcm | m4a`
- `stream_format` = `audio | sse` — enables byte streaming or base64 `speech.audio.delta` events
- `language` — a hint; one the model cannot speak (for example `ko` on Kokoro) gets `400` before
  synthesis. Kokoro and Qwen3 read a regional tag such as `zh-CN` or `pt-PT` as its primary language.
- `input_type` = `text | ssml`
- `instructions` *(backend-gated)*
- `voice_design` *(backend-gated)* — a description of the voice, for a model that designs one (Breeze TTS 2)
- `reference_audio` *(base64-encoded; backend-gated)* — decodes to at most `OS_MAX_UPLOAD_MB`
  (`413 Reference audio too large. Max: NMB`). Invalid base64 gets
  `400 reference_audio must be base64-encoded audio`, and one that decodes to nothing gets
  `400 Reference audio is empty`. Send it or `voice_library_ref`, not both.
- `voice_library_ref` *(stored provider-neutral reference; backend-gated)*
- `clone_transcript` *(backend-gated)*
- `effects` — at most 16 (`422`); not with streaming (`400`); invalid parameters get `400`; skipped
  when `OS_EFFECTS_ENABLED=false`

**Query params:** `stream`, `cache`

Left-out `model`, `voice`, `speed`, and `response_format` come from the `TTS_*` settings, and unknown
fields are ignored; see [Use it from an OpenAI app](#use-it-from-an-openai-app) for the exact rules.

`TTS_PRONUNCIATION_DICT` applies after the length check. Its expansions are held only to the model's
own limit (`400 Input too long for MODEL. Max: N characters`), while SSML is held to
`TTS_MAX_INPUT_LENGTH` after its pauses are expanded. A stream (`?stream=true` or `stream_format`)
sends audio as the backend produces it, except `wav`, which is sent as one complete file once synthesis
ends (use `pcm` for the lowest latency); it is never peak-normalized, and only its edges are trimmed.
The on-disk cache (`TTS_CACHE_ENABLED=true`; `?cache=false` skips it) serves only non-streamed
requests without effects, instructions, voice design, or reference audio. Send `X-History: true` to
log the request in `/api/history`.

Errors: with no TTS provider installed, speech, clone, voice-list, capability, and model load/unload
requests get `503` with code `provider_missing`. An isolated worker that cannot be reached gets `503`.
A worker's `400`, `404`, `409`, `413`, `429`, and `503` keep their status, a worker `422` (a value the
model rejects) becomes `400`, and any other worker status becomes `502`. A busy worker (`429`
`worker_busy`) is retried for about 2 seconds, honoring its `Retry-After`, before its `429` is
returned. A worker that runs out of GPU memory (its HTTP `507`) is reported as `503` with a message
that suggests unloading other GPU models or choosing a smaller one.

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

`GET /api/tts/capabilities?model=ID` (default `TTS_MODEL`) returns `backend`, `model`, `sample_rate`,
`capabilities`, `max_input_length` (`TTS_MAX_INPUT_LENGTH`), and `max_input_chars` when the model has
its own, usually lower, limit. `POST /v1/audio/models/load` and `/unload` act on `TTS_MODEL` when the
body is empty or names no `model`; a load first unloads the other loaded TTS models.

#### `GET /v1/audio/models`

Lists every TTS model this server can use, loaded or not: each one `/api/models` lists whose provider is installed
and, for an isolated worker, answers and can run it, plus `TTS_MODEL` whatever its state. `?loaded=true` lists
only the loaded models. Each row has `id` and `name`, as OpenAI-style clients such as Open WebUI read them (Open
WebUI lists the `id`s), and this server's own fields:

```json
{"models": [
  {"id": "kokoro", "name": "Kokoro", "object": "model", "model": "kokoro", "backend": "kokoro",
   "device": "cuda", "status": "loaded", "state": "loaded", "loaded_at": 1794661071.4,
   "last_used_at": 1794661115.5, "is_default": true, "description": "Fast, 52 voices, voice blending"},
  {"id": "piper/en_US-amy-medium", "name": "Piper Amy (US, medium)", "object": "model",
   "model": "piper/en_US-amy-medium", "backend": "piper", "device": null, "status": "not_loaded",
   "state": "downloaded", "loaded_at": null, "last_used_at": null, "is_default": false,
   "description": "US English - Amy voice"}
]}
```

`id` and `model` are the model ID to send as `model` on `/v1/audio/speech`. `status` is `loaded` or `not_loaded`;
`state` is the `/api/models` state (`provider_installed`, `downloaded` or `loaded`; for `TTS_MODEL` also
`provider_missing` or `provider_unavailable`, and `unknown` when no provider takes it).
`device`, `loaded_at` and `last_used_at` are `null` for a model that is not loaded. `GET /v1/audio/voices`
(`?model=ID`, default `TTS_MODEL`) returns `{"voices": [{"id", "name", "language", "gender"}]}`. A cloning model
has no voices of its own, so for one it lists the named voices that have a version for it, each `id` a
`voice:<uuid>` that `/v1/audio/speech` takes as `voice`: an OpenAI app such as Open WebUI can then pick a cloned or
designed voice by name.

#### `POST /v1/audio/speech/clone`

Multipart form endpoint that forwards reference audio to backends that support it.
Cloning needs a clone-capable model: Chatterbox, Qwen3 Base, Fish Audio S2, or Breeze TTS 2, run as isolated providers
with `docker-compose.voice-models.yml`. Those GPU workers are optional and off by default: each starts only when
its Compose profile is named (`--profile qwen3`, or `COMPOSE_PROFILES`), as [GPU Launch](#gpu-launch) shows. The
built-in Kokoro, Piper, and Pocket-TTS backends do not clone.
It accepts required `input` plus `model`, `reference_audio`, `voice_library_ref`, `voice`, `speed`,
`response_format`, `transcript`, and `language` form fields. Supply either an uploaded reference or a
stored `voice_library_ref` when the selected backend requires one. A left-out `model` means
`TTS_MODEL`, and a left-out `response_format` means `TTS_DEFAULT_FORMAT`. `input` is held to
`TTS_MAX_INPUT_LENGTH`, and the pronunciation dictionary applies as on `/v1/audio/speech`. The upload
may be up to `OS_MAX_UPLOAD_MB` (`413 Upload too large. Max: NMB`); an empty one gets
`400 Reference audio is empty`. `speed` outside 0.25–4.0 gets `422`.

#### `POST /api/voices/design`

Speaks a short sample in a voice made from a description, with a model whose capabilities have `voice_design`
(Breeze TTS 2, which must be loaded). JSON body: `model`, `description` (up to 1,000 characters), and optional `text`
(up to 500 characters; a short greeting when left out) and `language`. It answers a WAV at the model's rate; a model
that can't design a voice gets `400`. To keep the voice, save the sample with its words as the transcript
(`POST /api/voices/library` with `name`, `audio` and `transcript`); it then clones like any recording, with any
cloning model, by `voice_library_ref` or as a named voice's version (`"voice": "voice:<uuid>"`). The web UI's
**Voices › Design a voice** does both.

```bash
curl -sk https://localhost:8100/api/voices/design -H 'Content-Type: application/json' \
  -d '{"model": "breeze/tts-2", "description": "A warm, calm man in his fifties, a slight rasp, unhurried",
       "text": "Hello! This is the voice you described."}' -o sample.wav
```

#### Live Reader: `WS /v1/audio/speech/stream`

Live Reader accepts text deltas and immediately streams PCM16 audio back to the requester. With HTTPS,
the address is `wss://HOST:8100/v1/audio/speech/stream`; WSS is WebSocket traffic protected by HTTPS.
The Open Speech machine performs synthesis only. The browser, laptop, or other requesting client plays
or saves the returned audio.

The web UI's Speak › **Live** mode (a developer test, shown by Settings › Developer) uses it: Start reads
from the cursor, Read all from start reads everything, and Pause, Resume and Stop control the session. Its
modes are Book (whole sentences),
Responsive (shorter phrases) and Each word. A browser requires one Start click before it may play
audio. Existing one-shot TTS and `/v1/realtime` behavior are unchanged.

Client events:

```json
{"type":"session.update","session":{"model":"kokoro","voice":"af_heart","latency_mode":"natural"}}
{"type":"session.update","session":{"model":"chatterbox/turbo","voice":"reference","voice_library_ref":"narrator","latency_mode":"natural"}}
{"type":"input_text.append","text":"Words from a user or an AI token stream. "}
{"type":"input_text.commit"}
{"type":"input_text.context","text":"Sales rose 5% to"}
{"type":"playback.ack","response_id":"resp_...","sequence":0}
{"type":"response.cancel"}
```

Server audio arrives as `response.output_audio.delta` events with base64 PCM16LE mono, the backend's
actual `sample_rate`, a frame `sequence`, and the source range on the first frame. Acknowledge a frame
after the requester has played or otherwise consumed it. `input_text.done` means all text from the most
recent commit has been synthesized. `input_text.commit` releases the text held so far but doesn't start a
new line: text appended after it continues the same line unless the text before it ended with a line break.
So a client that commits after one reply and sends another that starts with `1. ` hears "1." spoken; when
the next text starts a line, end the text before it with a line break, or send `\n` first.

`input_text.context` `{text}` comes before a generation's first `input_text.append` (after `session.updated`
or `response.cancelled`): the source text just before where the next text starts, from its line's start. It
is not spoken or counted (`accepted_chars` and the source ranges count from the next text), and the next text
is cleaned as its continuation, so a leading `12.` or `>` there is prose. Send only the text after the last
line break, and `""` or nothing when the next text starts a line. It is held to the size of one
`input_text.append` (4 × `TTS_LIVE_MAX_BUFFER_CHARS` UTF-8 bytes). A code fence or link left open at that point
isn't carried over. There is no reply unless it is refused: sent after the generation's first append, larger
than that, or with text that isn't a string, it gets an `error` with code `invalid_event`. A client that
resends from mid-line after `response.cancel`, or starts from the middle of a line, uses it, as Speak › Live
does.

Applications that generate AI text can forward their received
text deltas to this socket; the Live Reader itself never contacts an AI server. For Open Speech to ask
your AI server and speak its reply, see [AI assistant](#ai-assistant).

A `session.update` the model cannot serve (an unknown voice or reference, a feature the model lacks,
or a language it cannot speak, such as `ko` on Kokoro) gets an `error` event with code `invalid_event`,
and the previous settings stay in effect. A binary frame also gets `invalid_event`, and the session
continues. Segments are capped at the model's own `max_input_chars` when it has one. When the
pronunciation dictionary lengthens a segment past that limit, the segment is split but still spoken as
one response. A connection beyond `TTS_LIVE_MAX_CONNECTIONS` sessions gets an `error` event with code
`session_busy` and is closed with `1013`. With `TTS_ENABLED` or `TTS_LIVE_ENABLED` false the socket is closed
with `4004 Live TTS is disabled` right after it opens, and a missing or wrong key with `4001`
([Refused WebSockets](#refused-websockets)).

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

`/v1/realtime` transcribes each turn and speaks replies. A `response.create` that carries text speaks
that text word for word. With the [AI assistant](#ai-assistant) on, each turn's transcript, and a
`response.create` with no text, goes to your own AI server: short waiting phrases are spoken while it
works, then its reply, and its errors are spoken too. The conversation lasts as long as the connection,
so a reconnect starts fresh. Without the assistant, nothing answers a turn by itself. It is not full
OpenAI Realtime parity: there are no tools (function calling) and no text-only output, and
`conversation.item.create` takes only a user's typed message ([Typed questions](#talking-to-it)).

The current OpenAI Python SDK connects without a WebSocket subprotocol override.
GA sessions use nested `session.audio.input` and `session.audio.output` settings
and `response.output_audio.delta` events. PCM at 24 kHz, PCMU and PCMA are supported;
input transcription selects its own model through `audio.input.transcription`.
Existing project clients offering the `realtime` subprotocol retain legacy events
until they send a nested GA session update. Response voice overrides are honored.

Reply audio streams as it is synthesized, and the server keeps listening while it talks.
When server VAD hears the user over a reply (`turn_detection.interrupt_response`, on by
default as in OpenAI), the remaining audio is held until the overlap is transcribed:
an acknowledgement such as "mm-hmm" or "yeah okay" lets the reply continue, real words
cancel it with `status_details.reason: "turn_detected"`. Speech longer than
`OS_REALTIME_BARGE_IN_MAX_MS` cancels without waiting. While barge-in is on, reply audio is
sent at playback speed, at most `OS_REALTIME_AUDIO_LEAD_MS` ahead, so there is still a reply
to interrupt and the client never has more than that buffered. Echo cancellation is up to the
client (browsers do it); without it the server can hear its own voice and stop itself.

**Models and voices.** OpenAI realtime model names (`gpt-realtime`, `gpt-4o-realtime-preview`, and
`gpt-4o-mini-realtime-preview`, with any `-...` suffix) and OpenAI TTS names speak with `TTS_MODEL`,
whether they arrive in `?model=`, a `session.update`, or a `response.create`; `session.created` and
`session.updated` still echo the client's name. Input transcription uses
`audio.input.transcription.model` (GA) or `input_audio_transcription.model` (beta, and legacy sessions
whose `?model=` is a realtime name), with OpenAI STT names mapped to `STT_MODEL`. Project clients that
put an STT model ID in `?model=` still transcribe with it. A `realtime`-subprotocol session that a
nested GA `session.update` upgrades keeps a `?model=` the TTS router can serve (such as `kokoro` or a
Piper ID) and otherwise switches to `TTS_MODEL`. Voices go to the TTS model as given, so Kokoro maps
the OpenAI voice names listed under [Use it from an OpenAI app](#use-it-from-an-openai-app). Replies
speak `TTS_PRONUNCIATION_DICT` replacements; transcripts keep the original text.

**Limits and errors.** An `error` caused by a client event carries that event's `event_id` in
`error.event_id`, as in OpenAI's API; that includes a reply's `tts_error` or AI error, which names the
`response.create` that asked for it. A reply that server VAD started carries none, and neither do the errors
no client event caused: the session's own idle timeout, a frame that isn't JSON or is binary, and a failed
transcription (reported as `conversation.item.input_audio_transcription.failed` with its `item_id`).

- A transcription `language` is normalized as on `/v1/audio/transcriptions`; an unknown one gets an
  `error` with code `invalid_request`, and the previous language stays in effect.
- Text with a lone UTF-16 surrogate (half an emoji) is refused with `invalid_request`, and the session stays
  as it was: in a `session.update`'s `instructions`, `voice`, `model` or transcription settings, in a
  `response.create`'s text to speak, and in a typed question. A whole emoji is fine.
- A `response.create` whose text is longer than `TTS_MAX_INPUT_LENGTH` gets `string_above_max_length`,
  and nothing is synthesized. When the dictionary-expanded text is longer than the TTS model's own
  `max_input_chars` (`Input too long for MODEL. Max: N characters`), or the dictionary leaves it blank
  (`Input text is empty`), the reply fails with an `error` of code `tts_error` and a `response.done`
  whose status is `failed`.
- Each session buffers at most `OS_REALTIME_MAX_BUFFER_MB` of input audio (`buffer_overflow` beyond
  it). Under `server_vad`, a manual `input_audio_buffer.commit` transcribes up to 30 seconds of audio
  since the last commit or clear. A commit of less than 50 ms gets `input_audio_buffer_commit_empty`.
  As in OpenAI, `input_audio_buffer.speech_started.audio_start_ms` includes `prefix_padding_ms`.
- A binary frame gets `invalid_event`, and the session continues.
- Realtime sessions are capped at `OS_STREAM_MAX_CONNECTIONS` (close code `1013` above it). A session
  that sends nothing for `OS_REALTIME_IDLE_TIMEOUT_S`, counted from the end of the last reply, is closed
  with `4008`; so is a manual-commit session that appends audio but has not committed for that long.
- A refused socket is opened and closed at once, before any event: `4004 Realtime API is disabled` with
  `OS_REALTIME_ENABLED=false`, `1008` for a page's `Origin` that isn't allowed, and `4001` for a missing or
  wrong key or ticket ([Refused WebSockets](#refused-websockets)). Read the close code after the socket opens,
  and don't reconnect on these three.

### AI assistant API

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/assistant` | Whether the assistant is on, the AI server's host and default model, which connection settings are set and where each comes from, and `locked` |
| `GET` | `/api/assistant/connection` | The connection: `url`, `model`, `api_key_set` (never the key), `api_key_tls_checked`, `verify_tls`, `template_kwargs`, `max_tokens_field`, `extra_body`, `stream`, `sources`, `enabled`, `locked` |
| `PUT` | `/api/assistant/connection` | Save some connection fields; `403` while locked |
| `DELETE` | `/api/assistant/connection` | Forget the saved connection, so `.env` applies again; `403` while locked |
| `POST` | `/api/assistant/connection/check` | One `GET {url}/models` with these fields, unsaved; answers the model ids |
| `GET` | `/api/assistant/presets` | The AI presets, built-ins first |
| `POST` | `/api/assistant/presets` | Create a preset (`201`); its id is made from its name |
| `GET` | `/api/assistant/presets/{id}` | One preset |
| `PUT` | `/api/assistant/presets/{id}` | Replace its editable fields |
| `DELETE` | `/api/assistant/presets/{id}` | Delete yours (`204`), or restore a built-in's defaults (`200`) |
| `POST` | `/api/assistant/test` | One real call with a preset: `{preset, text?, overrides?}` |
| `POST` | `/api/assistant/ask` | Send text, a transcript or a Reader document (all of it, some chapters or blocks, a selection or one sentence) to the AI with a preset and an optional question of your own; a document's text goes with its name and chapter. The reply as text, spoken (base64 audio) or saved as a Reader document; a longer input answers `202` with a book job ([Ask AI](#ask-ai-send-anything-to-the-ai)) |

The book jobs' routes are in the [Reader](#reader) table; they also follow and cancel the jobs that
`/api/assistant/ask` starts. Every route but `GET /api/assistant` and the
`/api/assistant/connection` ones answers `404` `assistant_disabled` while the assistant is off; the
connection routes are how you turn it on. See [AI assistant](#ai-assistant) for the bodies, the answers
and the error codes.

### Model Management

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/models` | OpenAI-style model list |
| `GET` | `/v1/models/{model}` | OpenAI-style model detail |
| `GET` | `/api/ps` | Legacy loaded STT models |
| `POST` | `/api/ps/{model}` | Legacy STT load |
| `DELETE` | `/api/ps/{model}` | Legacy STT unload |
| `POST` | `/api/pull/{model}` | Legacy STT download |
| `GET` | `/api/models` | Unified model inventory |
| `GET` | `/api/models/{id}/status` | Model state |
| `GET` | `/api/models/{id}/progress` | Download/load progress |
| `POST` | `/api/models/{id}/load` | Load model, unloading the others of its type (`?keep_pinned=true` keeps the pinned default) |
| `POST` | `/api/models/{id}/download` | Download artifacts |
| `POST` | `/api/models/{id}/prefetch` | Alias for download |
| `DELETE` | `/api/models/{id}` | Unload model |
| `DELETE` | `/api/models/{id}/artifacts` | Delete cached artifacts |

See [Model residency](#model-residency) for what load, download, and eviction keep in memory.
Legacy `POST /api/pull/{model}` fetches an STT model without loading it. The unified routes answer
errors with these codes:

| Status | Code | Meaning |
|---|---|---|
| `404` | `unknown_model` | Not a model any installed or configured provider serves |
| `404` | `not_loaded` | `DELETE /api/models/{id}` on a model that is not loaded |
| `404` | `tts_disabled` | With `TTS_ENABLED=false`, the status, progress, load, download, or prefetch of a TTS model |
| `400` | `provider_missing` | The provider is not installed; for Qwen3, Chatterbox, Fish, and Breeze the message says to run the isolated worker and list it in `TTS_EXTERNAL_PROVIDERS` |
| `503` | `provider_unavailable` | An isolated worker is configured but doesn't answer, or answers that it can't run (its `/health` is `failed`); then the message gives the worker's reason, as in `Provider 'chatterbox' is configured but its worker can't run: Torch wheel has no code that runs on sm_120 (built for: ...).` |
| `503` | `insufficient_gpu_memory` | The model doesn't fit in the GPU's free memory, even after unloading the idle models that aren't pinned (the [GPU memory guard](#gpu-memory-guard)), or a worker ran out of GPU memory; the message names what to unload |
| `409` | `model_fetching` | `DELETE /api/models/{id}/artifacts` while the model's files are being downloaded (a first use or a preload); delete them once the download has finished |
| `500` | `load_failed`, `download_failed`, `unload_failed` | Any other failure, such as CUDA out of memory in process |

Each `GET /api/models` row, and a model's status, has `pinned`: `true` for `STT_MODEL` and `TTS_MODEL` while
`OS_PIN_DEFAULT_MODELS` is on (idle and LRU eviction leave them, and a load with `keep_pinned=true` loads next to
them). Each also has `provider_available` and `provider_failure`. The
second is the worker's own `{"code": ..., "message": ...}` for a TTS model whose isolated worker answers
that it can't run (for example `cuda_arch_unsupported`), and `null` otherwise, also for a worker that
doesn't answer.

`GET /v1/models/{model}` answers `404` for a catalog TTS model no request can use: TTS is disabled,
its provider is not installed, or its isolated worker is not configured or does not advertise it.

### Studio / Persistence APIs

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/voices/design` | Speak a short sample in a voice made from a description (see below) |
| `POST` | `/api/voices/library` | Store a named voice reference |
| `GET` | `/api/voices/library` | List voice refs |
| `GET` | `/api/voices/library-config` | Get the reference-duration limit used by the web UI's voice cloning |
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

- **History.** `GET /api/history` takes `type` (`tts` or `stt`), `limit` (default 50), `offset`, and
  `search`. `search` matches a literal, case-insensitive substring of an entry's full text or input
  file name, and `total` counts only the matches. Each item carries `meta`, an object; TTS entries
  record `reference_voice`, `instructions`, `effects`, `input_type`, and `language` there. Only
  requests that send `X-History: true` are logged, and `OS_HISTORY_MAX_ENTRIES` bounds the history.
- **Conversations.** `POST /api/conversations` takes at most 500 `turns` (`422`). A turn's text is held
  to `TTS_MAX_INPUT_LENGTH` (`400`, for example `Turn 2 is too long. Max: 4096 characters`). Render
  `sample_rate` must be 8000–48000 (`422`). A turn
  with a preset is spoken by the preset's `model`, or its `backend` when it has none; one without a
  preset uses `TTS_MODEL`.
  Adding or deleting a turn clears the rendered output, so `GET /api/conversations/{id}/audio` is `404`
  until the next render. A render during which a turn was added or deleted answers
  `409 Conversation turns changed during the render. Render it again.` and does not publish its audio.
- **Composer.** A render takes at most 64 tracks, each with `offset_s` up to 7200, and a
  `sample_rate` of 8000–48000 (`422` otherwise). A composition longer than 2 hours gets
  `400 Composition is too long. Max: 7200 seconds`. Track sources must sit under the data directory,
  `OS_VOICE_LIBRARY_PATH`, `OS_CONVERSATIONS_DIR`, or `OS_COMPOSER_DIR` (`403` otherwise). Renders run
  one at a time.
- **Effects.** A preset, a turn and a track take at most 16 `effects` (`422`). Invalid effect parameters
  on presets, turns, and tracks get `400` (`Turn N: ...`, `Track N: ...`). `OS_EFFECTS_ENABLED=false`
  turns those effects off in renders.

### Reader

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/reader/documents` | Import a document: multipart `file` (`.epub`, `.txt`, `.md`, `.markdown`, `.html`, `.htm`, `.xhtml`) with optional `title`, or JSON `{"type": "text", "text": "...", "title": "..."}` |
| `GET` | `/v1/reader/documents` | List documents |
| `GET` | `/v1/reader/documents/{id}` | Get a document |
| `DELETE` | `/v1/reader/documents/{id}` | Delete a document and its audio |
| `PUT` | `/v1/reader/documents/{id}/progress` | Save `block_index`, `offset_ms`, `model`, `voice`, and `speed` |
| `POST` | `/v1/reader/documents/{id}/render` | Render audio parts from `from_block` with a `model` and `voice` |
| `GET` | `/v1/reader/documents/{id}/parts` | Part status for `from_block`, `model`, and `voice` |
| `GET`, `HEAD` | `/v1/reader/media/{part}.mp3` | Signed audio link (`exp`, `sig`); needs no API key |
| `POST` | `/v1/reader/documents/{id}/assistant` | Send the document to the AI assistant: `preset` (default `book`), `overrides`, `chapters` (`[first, last]`, from 0) and `title`; `202` with the job ([Book jobs](#book-jobs)) |
| `GET` | `/v1/reader/assistant-jobs` | Book jobs, newest first |
| `GET` | `/v1/reader/assistant-jobs/{job_id}` | A book job, with each chapter's status and its text once done |
| `DELETE` | `/v1/reader/assistant-jobs/{job_id}` | Cancel a book job; its finished pieces stay saved |

The Reader routes exist only with `READER_ENABLED=true`. An upload may be up to `READER_MAX_UPLOAD_MB`
and a document up to `READER_MAX_DOCUMENT_CHARS` characters (`413` otherwise). Importing the same
content again returns the existing document with `200` instead of `201`. Audio links expire after
`READER_MEDIA_URL_TTL_S` (`403` afterwards). Without `OS_API_KEY`, the links are signed with a key made
for each server process, so a restart also invalidates them; the parts endpoint returns new ones.
With a named voice (`voice:<id>`), parts rendered before its version for the model, its recording or that
recording's transcript changed are rendered again; a deleted named voice or recording gets `404` from
`/render` and `/parts`, and one with no version for the model `400`, as on `/v1/audio/speech`.

The server renders one job at a time, sentence by sentence: the parts the latest `/render` asked for. A
new `/render` replaces the job, so a skip drops parts no one needs any more. A job's first part is the
exception: when a request for another document, model or voice replaces the job, that part is finished and
saved first, so two listeners waiting at once each get their first part. A voice or model change made
before the first part is ready therefore waits for that part in the old voice.

A sentence with nothing left to say (no letter or digit, such as a `* * *` or `______` rule, or one the
pronunciation dictionary removes) is not sent to the model; it keeps its place in the part's timings. A
sentence the voice refuses for its language (`language_unsupported`, from a worker) is left out and
the part plays on, because the Reader sends no language hint and one word would otherwise stop the book; a part
in which every sentence is refused that way fails with the refusal, which names the languages and model to use.
Any other synthesis error fails the part.

### Health / UI

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/web` | Web UI |
| `GET` | `/openapi.json` | OpenAPI schema |
| `GET` | `/docs` | Swagger UI (only with `OS_API_DOCS=true`) |
| `GET` | `/redoc` | ReDoc (only with `OS_API_DOCS=true`) |

## Web UI

Open **https://localhost:8100/web** and accept the self-signed certificate the first time. The web UI is
plain static files and native JavaScript modules: no build step, nothing from a CDN, and it runs under a
strict Content Security Policy. It talks only to the server it came from, so it works on a LAN without
internet access.

### Navigation

- **Desktop** (960 px wide and up): a rail on the left groups the sections as **Listen** (Reader, Speak),
  **Capture** (Transcribe, Talk), **Create** (Voices, Studio) and **Manage** (History, Models, Settings).
- **Phone and tablet**: a bottom bar holds Reader, Speak, Transcribe and Voices; **More** opens Talk, Studio,
  History, Models and Settings.
- The **status chip** (top bar on phones, foot of the rail on desktops) shows the speech model in use, a
  download or load in progress, and connection or API key problems. Select it for details.
- The first visit opens the Reader; after that the app opens on the section you used last.
- Every section has its own address, so Back, bookmarks and links work: `#reader`, `#reader/<document>`,
  `#speak`, `#speak/live`, `#transcribe`, `#talk`, `#voices`, `#voices/recordings`,
  `#voices/clone`, `#studio/conversations`, `#studio/composer`, `#history`, `#models/tts`, `#models/stt` and
  `#settings`. `#speak/live` opens Generate, with the address `#speak`, while Speak › Live is hidden (Settings
  › Developer). `#voices/named` and `#voices/presets` (the old Named voices and Presets tabs) open `#voices`.
  Older bookmarks (`#voicelab`, `#tts`, `#stt`) still open the matching section.
- Press `?` for the keyboard shortcuts. `g` then `r`, `s`, `t`, `a`, `v`, `d`, `h`, `m` or `,` goes to a section;
  Space or `k` plays and pauses. Settings › Shortcuts turns the single-key shortcuts off.

### Sections

- **Reader**: the **Library** of documents (EPUB, text, Markdown, HTML, or pasted text) and the reading view,
  which highlights the sentence being read. The player has play and pause, 15 s back and 30 s forward,
  previous and next paragraph, a chapter-wide scrubber, speeds from 0.5× to 3.5×, a sleep timer, the
  reading voice, and **Reading appearance** (text size, typeface, spacing, width, paper and highlight
  style). Start over and Mark as finished are in a document's menu. **Ask AI** in the reading view's top bar
  sends this sentence (the one being read, or the last one read), this paragraph, the selected paragraphs, the
  current chapter or the whole document to the AI, with the document's name and chapter and an optional question
  of your own.
- **Speak**: type or open a `.txt` file, then **Generate** a take. The inspector sets the model, the voice
  (a picker with search, favorites and previews), a Kokoro blend (a visual editor), a reference recording
  for cloning models, speed, format, effects and instructions. Each take shows its first-audio time and
  real-time factor, plays and downloads, and its request can be copied as curl, Python or JavaScript.
  **Ask AI**, after Open .txt, sends the editor's text to the AI and speaks the answer. **Live**, a streaming
  test for developers, reads text as it arrives, the way an AI app streams its reply (see
  [Live Reader](#live-reader-ws-v1audiospeechstream)); it is hidden until Settings › Developer › Show Speak ›
  Live turns it on.
- **Transcribe**: upload a file (with progress and Cancel) or record from the microphone. Transcripts list
  their segments with timestamps and export as TXT, SRT, VTT or JSON. **Ask AI**, beside Copy, sends the
  transcript to the AI.
- **Talk**: talk to the [AI assistant](#ai-assistant) and hear it answer in a voice, over `/v1/realtime`.
  **Start talking** opens the microphone; speak, then pause. Your words, the waiting phrases, the thinking
  (dimmed and folded) and the answer appear in the conversation while the answer is read aloud. An answer
  that an error or the token limit cut off keeps what was said, with a note under it ("The answer was cut
  off: the AI reached its token limit."). **Type instead** sends a typed question (Enter sends, Shift+Enter starts a line, up to 8,000 characters), with the
  microphone on or off. Talking over an answer stops it; with speakers rather than headphones, turn off
  **Interrupt by talking**, so the answer's own sound doesn't stop it. **Stop reply**, **Resume sound** and
  **New conversation** sit beside the main button. **Delete chat**, beside the conversation's heading, clears the
  chat: while Talk runs it starts a new conversation, so the AI forgets the old one; after Stop it only clears the
  screen, with Undo for 5 seconds. The AI preset, the reply voice ("Same as Speak" by default)
  and Interrupt by talking are remembered in this browser; one that no longer exists falls back to the
  default. The AI remembers the conversation while Talk stays connected: after a dropped connection Talk
  reconnects by itself and says the conversation started over. With the assistant off, Talk says to set the
  AI server's address in Settings › AI assistant. To turn speech into text without the AI, use Transcribe.
- **Voices**: named voices and presets in one list, and recordings. A named voice is who speaks (one version per
  model); a preset is how it reads (speed, format, effects, instructions). Each named voice is a card with its
  versions and, under them, its presets and **New preset**; presets that use a plain voice, a blend or a recording
  are under **Presets without a named voice**. **Edit in Speak** on a version and **Edit** on a preset open it in
  Speak with an editing bar: **Update voice** or **Update preset** saves it, **Cancel** puts Speak back.
  **Clone a voice** is a guided three-step flow (Capture, Verify words, Test & save) with quality checks for
  duration, clipping, loudness, background noise and sample rate. **Design a voice** (Describe, Listen, Name &
  save) makes a voice from a description with a model that can design one (Breeze TTS 2), speaks an editable
  sample in it, and saves the sample as a recording with its words as the transcript, so any cloning model can
  speak in it.
- **Studio**: conversations made of turns, rendered to WAV or MP3, and the **Composer**, which mixes tracks
  from rendered conversations and earlier mixes. A conversation's More actions menu has **Ask AI about the
  script…**.
- **History**: every speech and transcription entry the server kept, with search, the All, Speech and
  Transcripts filters, Load more, **Open in Speak** (the text with the settings it was spoken with), Copy
  text, Delete and Clear all. An entry's More actions menu has **Ask AI…**.
- **Models**: the TTS and STT models with their state, and Load, Unload, Download and Delete files. An
  isolated worker that doesn't answer reads "Worker offline", not "Not installed". One that answers but
  can't run, such as a worker whose torch build has no code for the GPU, reads "Worker can't run" with the
  worker's reason.
- **Settings**: theme (System, Light or Dark) and Reduce motion; playback defaults (reading speed, skip
  intervals, what the previous and next buttons and the lock-screen track buttons move by, the sleep-timer
  fade, keeping the screen on); the API key; server status; the **AI assistant** (its connection, AI
  presets with their Ask AI fields, Preview and Test; see [Settings › AI assistant](#settings--ai-assistant));
  history; shortcuts; Developer (Show Speak › Live).

**Ask AI** shows only while the AI assistant is on: in Transcribe (the transcript), the Reader (this sentence,
this paragraph, the selected paragraphs, the current chapter or the whole document; it starts with this sentence
once something has been read, or with a selection you just made), Speak (the editor's text, spoken by default), History
(an entry's text) and Studio (a conversation's script, a line per turn). It opens one sheet: what is sent and
its size, **Your question** (optional: quick picks "Explain this", "Summarize", "Define the hard words", or your own
words, up to 2,000 characters), **Ask it to** (the presets offered for that kind of input, by their action names), and **Then**:
Show the answer, Speak it in a voice, Save it as a document. A line says which server it is sent to; for a Reader part, another says that the document's name and
chapter go with it. The answer comes back as text, with its thinking folded, Play, Copy, Save as a document, Open in the Reader and
Send again; the AI's errors are said in plain words, with Try again. A whole book or another long input runs
as a job, with progress and Cancel, and its answers are saved in the Reader as a new document; a long text or
transcript is kept in the Reader too, where you can delete it. Closing the sheet leaves a job running, and a
notice says when it has finished (with Open) or failed. The sheet remembers, in this browser, the preset you last
chose for each kind of input, your answer choices, the voice and your last three questions; a preset or voice that is gone falls back to the
default. Each preset's action name, the inputs it is offered for and its first answer choice are set in Settings › AI
assistant. The sheet uses [`POST /api/assistant/ask`](#ask-ai-send-anything-to-the-ai).

The UI uses one name per idea:

| Term | Meaning |
|---|---|
| Document, Library | A book, article or pasted text in the Reader, and the list of them |
| Voice | One provider voice, such as `af_heart` |
| Blend | A weighted Kokoro mix, shown as `af_bella(2)+af_sky(1)` |
| Named voice, version | A saved voice identity, and its setup for one model ([Named voices](docs/VOICE-IDENTITIES.md)) |
| Recording, reference recording | A clip and its transcript for cloning models |
| Preset, default preset | Saved delivery settings (`/api/profiles`) |
| Take | One result of Generate |
| Live | Speak's streaming test for developers (`/v1/audio/speech/stream`), shown by Settings › Developer |
| Talk | The view for talking to the AI assistant, by voice or typed (`/v1/realtime`) |
| Ask AI | Sending a transcript, a document, a selection or text to the AI with a preset (`POST /api/assistant/ask`) |
| Segment, interim text | A committed piece of a transcript, and the words still being recognized |
| Conversation, turn, render | A Studio script, its lines, and its audio |
| Mix, track | A Composer output, and one of its sources |

### Speech models, blends and presets

Speak opens with the default preset when one is set. Otherwise it opens on the model last picked in this
browser, or on the configured default (`TTS_MODEL`, normally Kokoro); an optional model that merely happens
to be loaded never takes its place. **Restore Kokoro** (named after the default) selects the default again
and loads it if needed.
When the chosen model is loaded, Generate sends its request at once; otherwise it loads the model first.
Loading from Speak or Models is an explicit load, which unloads the other loaded TTS model (see
[Model residency](#model-residency)), so Speak asks before it switches.

A cloning model such as Chatterbox has no built-in voices: it speaks in the voice of a reference recording.
For such a model every voice picker says so, and named voices that have no version for that model are shown
last, disabled. Speak's picker also lists your recordings first, with **Clone a voice…** to make one. See
[Cloning models](docs/CLONING-MODELS.md).

The blend editor only composes the voice string. The API contract is still the plain `voice` field: the
UI builds `af_bella(2)+af_sky(1)`, and the API only sees `voice="af_bella(2)+af_sky(1)"`. Paste a blend
string into the editor to edit it, or copy it out.

**Save preset…** in Speak stores the model, voice or blend, speed, format, effects, reference recording and
instructions in the server-side Studio database, so presets survive browser storage clearing and container
replacement when the `/home/openspeech/data` volume is kept. Selecting a preset chip in Speak applies it.
In **Voices**, a preset's **Use in Speak** applies it, **Edit** opens it in Speak where **Update preset** saves
Speak's setup into it (**Save as new…** keeps it as another preset), and it can be set as the default, renamed or
deleted. A preset that can't be used with the current models says why and changes nothing. `TTS_VOICES_CONFIG` is a separate,
administrator-supplied YAML preset source.

### API key and WebSocket tickets

With `OS_API_KEY` set, enter the key in **Settings › Connection** (Save, Test and Clear are next to it). The
key stays in this browser's storage, and every request the page sends to its own server carries it as
`Authorization: Bearer KEY`. A browser cannot set headers on a WebSocket, so Live, live transcription and Talk
first ask `POST /api/auth/ws-ticket` for a one-time [ticket](#authentication), a fresh one for every
connection and reconnect. The key never appears in a URL, in copied code (which reads
`$OPEN_SPEECH_API_KEY` or the environment instead) or anywhere on the page but the key field. A key the server refused is not sent
again until you enter another one, so a typo doesn't trip the [failed-auth lockout](#api-key).

### iPhone: lock screen and Home Screen

- The Reader keeps playing on a locked iPhone. The lock screen shows the chapter as the track and the
  document as the album, with a chapter-wide scrubber, skip buttons, and previous and next track. Settings ›
  Playback › **Lock-screen previous and next** chooses whether those move by paragraph or by chapter.
- Safari needs one tap on Play before a page may play audio; after a reload the Reader asks for it.
- To install the app, open `/web` in Safari, tap **Share**, then **Add to Home Screen**. It opens full screen
  with its own icon. A Home Screen app may refuse a certificate the phone doesn't trust, so give Open
  Speech a trusted certificate first ([TLS certificates](#tls-certificates)). The microphone (Transcribe,
  Talk, voice cloning) also needs HTTPS, or `localhost`.
- **Talk** with the microphone open may play the answer through the earpiece on iOS (not yet tried on a
  device); headphones avoid it.
- **Keep the screen on while reading** (Settings › Playback) holds the screen awake where the browser
  supports it.

### Changes from the previous web UI

The web UI was redesigned in this release. Changes you will notice:

- The Reader keeps playing when you leave it. A **mini-player** on every other screen gives the controls
  back; its Stop button ends playback. Finishing a document ends the session, so the mini-player goes too.
- Starting any other audio stops Live; it doesn't pause it, because a paused Live session would hold up the
  Reader's rendering. Recording from the microphone pauses whatever is playing.
- History replaces the "Recent" lists in Speak and Transcribe, with one Clear all. Entries have no inline
  player, because the server keeps no audio for them.
- Transcribe always asks for segments, and exports TXT, SRT, VTT and JSON. A file transcription shows its
  progress and can be canceled, and server errors during a recording are shown.
- Named voices and presets moved from Settings to Voices, where they share one list. Saving a cloned recording
  opens Voices ›
  Recordings with the new card; its **Use in Speak** button selects it in Speak.
- The theme follows the system unless you choose Light or Dark.
- Streaming is on by default for MP3 takes without effects; other formats, and takes with effects, play
  when they are ready.
- The Live mode (Book, Responsive or Each word) is remembered across reloads.
- Studio turns move with up and down buttons. The Composer lists rendered conversations and earlier mixes
  as sources; recordings aren't listed, because the server exposes no file path for them.
- The Models list marks the server's configured STT default, and reads an isolated worker that stopped
  answering as "Worker offline", and one that answers but can't run as "Worker can't run", with its reason.

## Voice Blending

Kokoro supports weighted voice syntax:

```bash
curl -sk https://localhost:8100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro","input":"Hello","voice":"af_bella(2)+af_sky(1)"}'
```

Join Kokoro voice IDs with `+`. A number in parentheses sets a voice's relative weight (default `1`,
decimals allowed), and the weights are normalized, so `af_bella+af_sky` is an even mix. OpenAI voice
names such as `alloy` are not resolved inside a blend; use Kokoro IDs. A blend that names an unknown
voice gets `400 Unknown Kokoro voice: ID`. The same `voice` string works on `/v1/audio/speech`, Live
Reader, `/v1/realtime`, and Wyoming whenever the model is Kokoro.

Supertonic blends its own voices the same way, for example `"model":"supertonic","voice":"M1(0.6)+F2(0.4)"`.
A blend mixes voices of one model only. Supertonic also takes an optional `steps` field (quality steps: fewer is
faster, more is cleaner); other models ignore it, so plain OpenAI clients work unchanged. Numbers, money, times,
dates and titles in English text are spelled out before Supertonic reads them.

## AI assistant

Open Speech can send what you say or type, a transcript, or a Reader document to your own AI server through
its OpenAI-compatible Chat Completions API (`POST {url}/chat/completions`), and speak the reply in any voice
or save it as a new Reader document. A voice turn streams the reply (`stream: true`) and starts speaking on
its first sentence; Ask AI and book jobs wait for the whole reply (`stream: false`). It is off until you give it
the server's address. (To use Open Speech as the speech layer of an AI agent that does its own thinking,
leave it off and see [docs/AI-AGENTS.md](docs/AI-AGENTS.md).)

| Way in | How | What comes back |
|---|---|---|
| **Talk** | The web app's [Talk](#sections) view, or any `WS /v1/realtime` client, such as `examples/voice_chat.py`: speak, or type a question ([Talking to it](#talking-to-it)) | Short phrases such as "Hmm." while the AI works, then the spoken reply. Talking over it stops it. |
| **Ask AI** | The web app's [Ask AI](#sections), or `POST /api/assistant/ask` ([Ask AI](#ask-ai-send-anything-to-the-ai)), with typed text, a transcript, or a Reader document or part of one | The reply as text, and on request spoken (base64 audio) or saved as a Reader document. A long input runs as a book job |
| **A book** | Import an EPUB or text file into the Reader, then start a [book job](#book-jobs) on it | The AI's answer for each piece of the book, saved as a new Reader document you play like any other |

The thinking is never spoken. When the AI can't answer, Open Speech says so out loud. A step-by-step guide
is in [docs/AI-ASSISTANT.md](docs/AI-ASSISTANT.md).

### Turn it on

Give it the server's base URL, ending in `/v1`: in Settings › AI assistant › Connection, which needs no
restart, or in `.env` with `OS_ASSISTANT_URL`, then restart. For example, a llama.cpp-style server on your
network, with low thinking:

```bash
OS_ASSISTANT_URL=http://ai-server.lan:8080/v1
OS_ASSISTANT_MODEL=your-model
OS_ASSISTANT_TEMPLATE_KWARGS=true
OS_ASSISTANT_REASONING_EFFORT=low
```

Such a server's chat template reads the thinking controls as template variables, so
`OS_ASSISTANT_TEMPLATE_KWARGS=true` sends them there too. Some templates think at length unless the effort
arrives as a template variable.

**Your words and your books go to that server.** With a server on your own network they stay there;
pointing `OS_ASSISTANT_URL` at a cloud API sends them out.

From inside the Docker container, a server on the Docker host is `http://host.docker.internal:PORT/v1`
(Docker Desktop provides that name; on Linux add `extra_hosts: ["host.docker.internal:host-gateway"]` to the
service). Another machine on your network is reached by its name or IP address, as long as the container
can resolve it. `docker-compose.yml` passes every `OS_ASSISTANT_*` setting through from `.env`.

### Settings, presets and the three layers

Every field has three layers, with the same names everywhere: `.env` (`OS_ASSISTANT_*`, listed under
[Environment Variables](#environment-variables)) gives the defaults, an AI preset overrides them, and one
request overrides the preset: a realtime `session.update`'s `assistant`, a book job's `overrides`, or an
`/api/assistant/ask` request's `overrides`. A field left empty falls back to the layer below.

The **connection** is apart: the URL, the API key, the default model, the TLS check, the template switch,
the token-field name, the extra body and Stream replies come from `.env` or from Settings › AI assistant › Connection
([The connection](#the-connection)), never from a preset or a session. A preset's own `model` still wins
over the default model.

Two **AI presets** are built in: `voice`, the default for `/v1/realtime` and for asking about text or a
transcript, and `book`, the default for book jobs and for asking about a document. You can edit both and add
your own, such as "Translate to Spanish", up to 50 in all.

| Field | Used by | Values |
|---|---|---|
| `name` | both | Up to 80 characters. A new preset's `id` is made from it; `voice` and `book` keep theirs |
| `label`, `inputs`, `output` | Ask AI | How the web app's Ask AI offers the preset, never sent to the AI and never overridden by a request: `label` is the action's name (up to 80 characters; default the preset's name, "Answer" for `voice`, "Rewrite for listening" for `book`), `inputs` the kinds of input it suits (`text`, `transcript`, `document`; `[]` is the default: `voice` `text` and `transcript`, `book` `document`, yours all three), and `output` its first answer choice (`text`, `speech` or `document`; default `speech` for `voice`, `document` for `book`, `text` for yours). A `PUT` that leaves one out keeps it, and `null` or an empty value is the default again |
| `model` | both | Up to 200 characters; empty uses the connection's model (`OS_ASSISTANT_MODEL`), and with none set the AI server picks |
| `system_prompt` | both | Up to 8,000 characters; empty uses `OS_ASSISTANT_SYSTEM_PROMPT` or the built-in voice prompt for a voice turn, and the built-in book prompt for a book job |
| `user_template` | both | Wraps your words or a piece of the book; must contain `{text}`; up to 8,000 characters; default `{text}` ([Prompts](#prompts)) |
| `thinking` | both | `default` (sends nothing), `on` or `off` ([Thinking controls](#thinking-controls)) |
| `reasoning_effort` | both | Empty (not sent), `none`, `minimal`, `low`, `medium`, `high` or `xhigh` |
| `thinking_budget` | both | 0 to 1,000,000 tokens of thinking, sent as `reasoning_budget_tokens`; 0 sends none ([Thinking controls](#thinking-controls)) |
| `show_thinking` | both | Send the AI's thinking to the client as text |
| `max_tokens` | both | 0 to 32,768; 0 sends no limit. It cuts the answer off ([Thinking controls](#thinking-controls)) |
| `temperature`, `top_p` | both | 0 to 2 and 0 to 1; empty sends none, so the AI server's own default applies |
| `timeout_s` | both | 5 to 3,600 seconds; empty uses `OS_ASSISTANT_TIMEOUT_S` for a voice turn and `OS_ASSISTANT_JOB_TIMEOUT_S` for a book job |
| `history_turns` | voice | 0 to 100 earlier exchanges sent with each turn |
| `fillers`, `filler_after_ms`, `filler_every_ms`, `filler_max` | voice | [Waiting phrases](#waiting-phrases) |
| `speak_errors`, `error_sentences` | voice | [Errors, spoken](#errors-spoken) |
| `chunk_chars` | book | 1,000 to 200,000: the most book text in one request |

Presets are kept in `assistant-presets.json` next to `OS_STUDIO_DB_PATH`. The file is written only when a
preset is saved, and a built-in you haven't changed isn't in it. A missing or unreadable file means only
the built-ins, with a warning in the log. Each preset in the API carries its effective values plus
`builtin`, `customized` and `inherited` (the fields that come from `.env` or the code). A `PUT` replaces
the editable fields: one left out or empty follows `.env` or the code again (except `label`, `inputs` and
`output`, which a `PUT` that leaves them out keeps), and the read-only fields may be sent back and are
ignored. `DELETE` on a built-in restores its defaults. Refusals: `422`
`validation_error` with the field in `param`, `404` `preset_not_found`, and `409` `preset_exists` or
`too_many_presets`.

### Prompts

The system prompt is the system message, and the user template wraps your words or a piece of the book
into the user message. The built-in prompts are:

- **voice:** "You are a helpful voice assistant. Your replies are read aloud. Answer in short, plain spoken
  sentences. Don't use Markdown, lists, tables, code, links or emoji."
- **book:** "You rework book text for listening, following the user's instructions. Reply only with plain
  prose paragraphs, because your reply is read aloud. Don't use Markdown, lists or headings, and don't
  comment on what you did."

A template must contain `{text}`. Book jobs also fill `{title}`, `{author}` and `{chapter}` (the chapter's
title). The built-in `voice` preset takes only `{text}`; your own presets may use all four, and a voice turn
leaves the other three empty. For example, a book preset with the template
`Summarize this part of "{chapter}":\n\n{text}` sends each piece with that instruction in front of it.
Rendering is one pass of literal replacement: other braces are kept as they are, and the text put in for
`{text}` is never searched for placeholders, so a book that contains `{title}` stays as written. An unknown
placeholder is refused when the preset is saved. In a realtime session, `instructions` replaces the
preset's system prompt.

### Thinking controls

Servers don't agree on how to switch or shorten thinking, so there are four controls. Each is sent only when
set, because some servers refuse fields they don't know.

| Control | What it sends | Who reads it |
|---|---|---|
| `thinking`: `on` or `off` | `"chat_template_kwargs": {"enable_thinking": true}` (or `false`) | Qwen3 and other models that can think or not, on vLLM, SGLang and the llama.cpp server |
| `reasoning_effort` | `"reasoning_effort"`; with `OS_ASSISTANT_TEMPLATE_KWARGS=true` also `chat_template_kwargs.reasoning_effort` | OpenAI's field; gpt-oss and similar reasoning models, and templates that read it |
| `thinking_budget`: 1 to 1,000,000 | `"reasoning_budget_tokens"`, the most tokens of thinking | Servers that read `reasoning_budget_tokens` (llama.cpp-style). Other servers ignore or refuse it; Settings' Test shows whether yours takes it |
| `OS_ASSISTANT_EXTRA_BODY` (connection only) | Whatever JSON object it holds, merged last; its `chat_template_kwargs` are merged key by key. The preset's thinking controls win where both set one: inside `chat_template_kwargs`, and at the top level a preset's `reasoning_effort` and `thinking_budget` over the extra body's `reasoning_effort` and `reasoning_budget_tokens`, which go only when the preset sends none. It can't set `messages` or `stream` | Anything else your server wants, such as Ollama's `think` or `top_k` |

The token limit goes as `max_tokens`, or as `max_completion_tokens` with
`OS_ASSISTANT_MAX_TOKENS_FIELD=max_completion_tokens`, for a server that follows OpenAI's newer API.

**A token limit cuts the answer off; a thinking budget doesn't.** `max_tokens` is a hard cut-off the model
isn't told about: an answer that reaches it stops mid-sentence (`finish_reason: "length"`), and on most
servers the thinking counts towards it. For short answers, ask for them in the prompt ("Answer in 3
sentences."), and keep the limit as a safety net. To shorten the thinking, set `thinking_budget`: when the
thinking reaches it, a server that reads `reasoning_budget_tokens` closes the thinking and the model writes a
complete answer, based on shorter thinking. How such a server treats a budget that doesn't fit inside
`max_tokens` is up to the server. By default neither is sent (`OS_ASSISTANT_THINKING_BUDGET=0`,
`OS_ASSISTANT_MAX_TOKENS=0`).

The thinking comes back in `message.reasoning_content`, `message.reasoning` or `<think>…</think>` at the
start of `content`; all three are read, and the answer is whatever is left. The thinking is never spoken,
and never sent back in the conversation history. With `show_thinking` it is sent to the client as text.
**The token limit includes the thinking** on most servers: a low `max_tokens` can stop the model while it
is still thinking, which is reported as `assistant_reply_cut_off` ("raise the token limit or lower the
thinking"), not as an empty reply.

**A streamed voice turn** (Stream replies, `OS_ASSISTANT_STREAM`, on by default) reads the same places as
they arrive: `delta.reasoning_content` (or `delta.reasoning`) and a leading `<think>…</think>` in
`delta.content` are the thinking, the rest of `content` the answer. The request also carries
`"stream_options": {"include_usage": true}`, so the server's last piece holds the token counts for the
[timing line](#the-ai-assistant); a server that refuses that with HTTP 400 is asked once more without it,
and not again until the connection is saved or Open Speech restarts. The thinking streams first, and an
answer cut off by `max_tokens` (`finish_reason: "length"`) after some of it was spoken keeps what was said and
says nothing more ([Talking to it](#talking-to-it)). A chat template that opens `<think>` itself, so the reply
has a `</think>` but no `<think>`, can't be told apart from the answer while it streams: turn the server's
reasoning parser on (llama.cpp's `--reasoning-format`, vLLM's `--reasoning-parser`), or turn Stream replies
off.

### Waiting phrases

While a voice turn waits for the AI, Open Speech says short phrases, so you know it heard you.

| Preset field | Default | Meaning |
|---|---|---|
| `fillers` | `["Hmm.", "OK, just a second.", "Let me think about that."]` | Up to 10 phrases of up to 80 characters; an empty list turns them off |
| `filler_after_ms` | `1200` | No phrase if the reply comes sooner (0 to 30,000) |
| `filler_every_ms` | `4000` | The gap before the next phrase while it still waits (0 to 30,000) |
| `filler_max` | `3` | The most phrases in one turn (0 to 10) |

- They apply only to voice turns that go to the AI, not to book jobs or to word-for-word replies.
- They're chosen at random, never the same one twice in a row, and spoken in the session's voice. Each is
  made once per session, ahead of time after `session.update`, so the first one plays at once.
- Each phrase's audio follows an `open_speech.assistant.filler` event with its text. Phrases aren't part of
  the transcript, the history or `response.done`.
- A phrase that has started is finished, and the reply follows straight after it. No phrase starts once
  the reply has arrived. Talking over a phrase works as talking over a reply does.
- Some voices read "Hmm" oddly, for example letter by letter. Try your phrases with Preview in Settings, and
  respell them as they sound ("Hmmm.", "Mm.") or use the pronunciation dictionary.

### Errors, spoken

When the AI can't answer a voice turn, the preset's sentence for the error is spoken, unless
`speak_errors` is off. `error_sentences` maps a code to its sentence (up to 300 characters); a code left out
keeps the default.

| Code | When | Default sentence |
|---|---|---|
| `assistant_unreachable` | Connection refused, DNS or TLS failure | "I couldn't reach the AI server." |
| `assistant_timeout` | No whole reply within `timeout_s` | "The AI took too long to answer." |
| `assistant_http_error` | A status other than 200; only the status is reported, never the server's text | "The AI server returned an error." |
| `assistant_bad_response` | Not JSON, no `choices[0].message`, or a reply over 8 MiB | "The AI didn't send back an answer." |
| `assistant_empty_reply` | No answer left once the thinking is removed | "The AI didn't send back an answer." |
| `assistant_reply_cut_off` | An empty answer with `finish_reason: "length"`, or in a voice turn one cut off before its first sentence was complete (after that, [nothing more is said](#talking-to-it)) | "The answer was cut off before it started." |

### Talking to it

The web app's **Talk** view does all of this for you ([Sections](#sections)). A client of your own connects to
`wss://HOST:8100/v1/realtime` (no subprotocol gives a GA session) and sends a `session.update`:

```json
{"type": "session.update", "session": {
  "type": "realtime",
  "instructions": "Answer like a pirate.",
  "assistant": {"preset": "voice", "reasoning_effort": "low", "fillers": ["Mm."]},
  "audio": {
    "input": {"format": {"type": "audio/pcm", "rate": 24000},
              "turn_detection": {"type": "server_vad", "create_response": true}},
    "output": {"voice": "af_bella(2)+af_sky(1)"}}}}
```

- `assistant`: `preset` picks the AI preset (default `voice`). Every other key overrides that preset field
  for this session; an empty one is skipped, and `null` removes the override. `"assistant": null` goes back
  to the voice preset with no overrides. A value that isn't valid refuses the whole update (`error` with
  code `invalid_request`), and the session stays as it was.
- `instructions` is OpenAI's name for the system prompt, and replaces the preset's for this session.
- `session.created` and `session.updated` report both; `assistant` is `null` while the assistant is off,
  and also with `TTS_ENABLED=false`, because nothing could be spoken.
- `turn_detection.create_response` defaults to whether the assistant is on. As in OpenAI's API, `true`
  means "reply after each turn". A turn is one that server VAD ends: a client's own
  `input_audio_buffer.commit` starts no reply, as in OpenAI's API, and its `response.create` asks for one.
  Without the assistant the session reports `false`, even when asked for `true`.

A turn then goes like this:

1. Server VAD hears you stop, and faster-whisper writes down what you said:
   `conversation.item.input_audio_transcription.completed`.
2. Your words, wrapped in the preset's user template, join the conversation. With `create_response`, a
   response starts for a turn server VAD ended: `response.created`.
3. The AI is asked with `stream: true` (Stream replies, on by default). While it thinks,
   `open_speech.assistant.filler` and the phrase's audio, every few seconds, until the first sentence of its
   answer is ready.
4. `open_speech.assistant.thinking` with the thinking as text, once the answer has started, when the server
   sent some and `show_thinking` is on.
5. The answer is cut into sentences as they arrive (the Live Reader's sentence cutting), and each is spoken at
   once: a `response.output_audio_transcript.delta` with that sentence's text, then its audio
   (`response.output_audio.delta`), while the AI is still writing the rest. Legacy (`realtime` subprotocol)
   sessions get `response.audio_transcript.delta` and `response.audio.delta`. The deltas joined are the
   answer's text.
6. `response.output_audio_transcript.done` with the whole answer, and `response.done`.

With Stream replies off (`OS_ASSISTANT_STREAM=false`, or `"stream": false` saved through
[`PUT /api/assistant/connection`](#the-connection)), the AI's whole reply is awaited: the phrases play until it has all arrived, and one
`response.output_audio_transcript.delta` carries the whole text, before its audio. A server that answers a
streamed request with one plain JSON reply is read the same way, so it works either way.

More about a turn:

- **Manual turns.** Every transcript joins the conversation, also with `create_response: false` or manual
  commits; a `response.create` with no text then answers it, as in OpenAI's manual flow. A
  `response.create` that carries text (`instructions` or an `input_text` item) is still spoken word for
  word, with no AI call and no phrases. After a commit of your own, wait for its transcript (a non-blank
  `transcript` in `conversation.item.input_audio_transcription.completed`) before that `response.create`:
  after `input_audio_buffer_commit_empty`, a failed transcription or a blank one, nothing joined the
  conversation, and the AI would be asked about the conversation as it was. Words committed while a reply
  runs don't join it either, so send `response.cancel` first.
- **Typed questions.** Send `conversation.item.create` with `item: {type: "message", role: "user", content:
  [{type: "input_text", text}]}` (up to 8,000 characters; several `input_text` parts are joined with line
  breaks), then a `response.create` with no text. The item is acknowledged with `conversation.item.added` and
  `conversation.item.done` (`conversation.item.created` in legacy sessions), joins the conversation through
  the preset's user template, and is answered as a spoken turn is, waiting phrases, errors, cancel and
  barge-in included. It starts no reply by itself, even with `create_response`, so send the `response.create`
  once the item is acknowledged. `item.id` (1 to 32 characters) is echoed, or the server makes one;
  `previous_item_id` must be left out or `null`. Other items, roles and content are refused with an
  `invalid_request_error` (code `invalid_request`), as are blank text and text with a lone UTF-16 surrogate
  (half an emoji), and longer text gets `string_above_max_length`. A typed message while a reply runs gets
  `response_in_progress`: send `response.cancel` first (events are handled in order, so the three can go back
  to back). Without the assistant it gets `assistant_disabled`, and with `TTS_ENABLED=false` `tts_disabled`.
  A refused item isn't kept, so don't send the `response.create` after a refusal.
- **The reply** is cleaned as the Live Reader cleans text (Markdown removed, links read by their label,
  code blocks replaced with "Code block skipped.") and spoken in segments within the voice model's own
  limit, so an AI reply is never refused for its length.
- **Talking over it.** Real words over the wait or the reply cancel the AI request and the phrases, and
  become the next question; an acknowledgement such as "mm-hmm" lets it continue (see
  [Realtime Audio](#realtime-audio)). `response.cancel` and closing the socket cancel it too, with a
  `response.done` whose status is `cancelled`. Each closes a streamed reply's connection, so the AI server can stop
  generating. The history keeps the answer as far as it was handed to the voice (the sentences spoken or
  being spoken; all of it once it had all arrived); an answer interrupted before its first sentence leaves the
  question unanswered in the history.
- **Errors.** An `AssistantError` before any of the answer was spoken (including a dropped stream, an error
  in it, or the token limit while it still thinks) gives an `error` event (type `server_error`, with the
  code), then the spoken sentence, then a `response.done` with status `failed` and the code in
  `status_details.error.code`. The conversation keeps only your question.
- **Cut off after it started.** An error or the token limit (`finish_reason: "length"`) after some of a
  streamed answer was spoken keeps what was said and says nothing more: not the sentence it was in the middle
  of, and no error sentence. `response.done` then has status `incomplete`, `status_details`
  `{"type": "incomplete", "error": {"type": "server_error", "code": ...}}` (such as `assistant_reply_cut_off`
  or `assistant_unreachable`), and the transcript and the history hold the part that was spoken. There is no
  `error` event.
- **The conversation** sends the last `history_turns` exchanges (the answers only, never the thinking),
  plus any unanswered question. It lasts as long as the connection: a reconnect starts fresh.
- **Voices.** Kokoro voices and blends work as they are, and named voices (`voice:<id>`) are resolved. A
  named voice that doesn't resolve refuses `session.update`; one whose version is a reference recording
  (a cloned voice) can't be used here yet.
- **Waiting is busy.** The time the AI takes doesn't count toward `OS_REALTIME_IDLE_TIMEOUT_S`. A client
  that only types sends nothing between questions; an empty `{"type": "session.update", "session": {}}`
  changes nothing and keeps the session open, as Talk does after 30 seconds of sending nothing.

To talk through it from a terminal, use `examples/voice_chat.py` (the `examples` extra, with PortAudio):

```bash
python examples/voice_chat.py --url https://HOST:8100 --insecure --voice af_heart --preset voice
```

It streams the microphone as 24 kHz PCM16, plays the reply, and prints your words, the phrases, the
thinking (dimmed) and the reply, on one `[ai]` line that grows a sentence at a time; a reply cut off after
some of it was spoken (a `response.done` with status `incomplete`) adds `[cut off: CODE]`, such as
`[cut off: assistant_reply_cut_off]` for the token limit. **Use headphones:** the script has no echo cancellation, so through
speakers the microphone hears the reply and the reply interrupts itself. Without headphones, add
`--no-interrupt`, which sends `interrupt_response: false`. `--api-key` defaults to the shell's
`OS_API_KEY`, and `--list-devices`, `--device` and `--output-device` choose the audio devices.

### Book jobs

Import the file into the Reader as usual (`POST /v1/reader/documents`: an EPUB, HTML, Markdown or text
file, or pasted text; there is no import from a URL), then start a job on the document:

```bash
curl -sk -X POST https://localhost:8100/v1/reader/documents/DOCUMENT_ID/assistant \
  -H "Content-Type: application/json" \
  -d '{"preset": "book", "overrides": {"user_template": "Summarize this part of \"{chapter}\":\n\n{text}"}, "chapters": [0, 2], "title": "Summary"}'
```

- **The body:** `preset` (default `book`), `overrides` (preset fields for this job; an empty one is
  skipped), `chapters` (`[first, last]`, the Reader's chapter indexes from 0, or `null` for all) and
  `title`. It answers `202` with the job. A bad field gets `422` with `param`; an unknown document `404`.
- **How it runs.** The settings are worked out once and kept in the job, so a later change to the preset
  doesn't change it. Each chapter is cut into pieces of at most `chunk_chars`, at block boundaries (a
  longer block at its sentences), and each piece is sent on its own, with no history. A chapter's opening
  headings aren't sent; the result keeps them as they are, so each chapter still opens with its title.
- **Saved as it goes.** Each answer is saved as soon as it comes back, so a restart loses none. An error is
  retried once; a second one stops the job as `failed` with the code in `error` (and `error_message`),
  keeping the finished pieces. Starting the same job again (the same document, and the same settings that
  reach the AI: model, prompts, thinking, effort, token limit and sampling) sends only what's missing.
- **One job at a time.** Later jobs wait as `queued`. A shutdown, or a crash found at the next start,
  leaves the job `failed` with the code `interrupted`; start it again to finish it.
- **The result** is a new Reader document named "<document title>: <your title, or the preset's name>",
  with one chapter per chapter sent and the answers as paragraphs (split at blank lines). The job's
  `result_document_id` names it. The thinking is kept in the job, not in the document.
- **The job** (`GET /v1/reader/assistant-jobs/{job_id}`) has `id`, `document_id`, `preset`, `title`,
  `chapters_requested`, `blocks_requested` (`[first, last]` for a job that
  [Ask AI](#ask-ai-send-anything-to-the-ai) started over a range of blocks, `null` otherwise), `status`
  (`queued`, `running`, `done`, `failed` or `cancelled`), `error`,
  `error_message`, `result_document_id`, `created`, `updated`, `settings` (the preset fields in effect,
  nothing server-only) and `chapters`: each with `index`, `title`, `status`, `pieces`, `pieces_done`, its
  `text` once done, its `thinking` when shown, and `error`. `GET /v1/reader/assistant-jobs` lists the jobs
  without their `settings` and `chapters`.
- **Canceling and deleting.** `DELETE /v1/reader/assistant-jobs/{job_id}` cancels a queued or running job;
  its finished pieces stay saved. Deleting the source document deletes its jobs and their pieces, and stops
  a running one; the result document stays.
- **Listening before the whole book is done.** A finished chapter's `text` is in the job straight away; any
  client can send it to the [Live Reader](#live-reader-ws-v1audiospeechstream) and hear it at once.

The book-job routes answer `404` `assistant_disabled` while the assistant is off, and `503`
`assistant_not_started` if its client isn't running. The AI server is asked for one book at a time, but a
book job and a voice turn can still wait for each other at a server with one slot.

### Ask AI: send anything to the AI

`POST /api/assistant/ask` sends one input to the AI with a preset and gives the reply back. The web app's Ask AI
uses it ([Sections](#sections)).

```bash
curl -sk -X POST https://localhost:8100/api/assistant/ask \
  -H "Content-Type: application/json" \
  -d '{"input": {"kind": "text", "text": "Why is the sky blue?"}, "output": {"speech": {"voice": "af_heart"}}}'
```

- **The input** is typed text (`{"kind": "text", "text": …}`), a transcript (`{"kind": "transcript", "text":
  …}`, or `"history_id"` for a history entry, read on the server), or a Reader document (`{"kind":
  "document", "document_id": …}`: all of it, or `"chapters": [first, last]`, `"blocks": [first, last]`, a
  selection's `"block_start"` and `"block_end"` (whole blocks are sent), or one sentence's `"block"` and
  `"sentence"`). The preset is `book` for a document and `voice` otherwise, and `overrides` takes its fields as
  Test does. Text may be up to `READER_MAX_DOCUMENT_CHARS` characters.
- **One sentence:** `"sentence"` is its index in block `"block"`, from 0, as the Reader reads and highlights the
  block (its `sentences` offsets in `GET /v1/reader/documents/{id}`; a block without any is one sentence). It is
  always answered at once.
- **Your own question:** `"question": "Explain this to me."` (up to 2,000 characters, not blank) goes with any
  input, before its text: the message is the question, a blank line, then the text (or, with a template other
  than `{text}`, the filled-in template). Every piece of a job gets it, and a different question sends every
  piece again.
- **The document's name and chapter** go with a document's text: its message starts with `From "<title>",
  chapter "<chapter>":` and a blank line (after the question, if any). The title is the Reader document's, which
  for an imported file is its file name unless the file gives a title. The chapter is named when the document has
  more than one and it isn't the document's title; each piece of a job names its own chapter (a piece never spans
  two). A preset template that uses `{title}` or `{chapter}` gets no such line, so nothing says it twice. Text and
  transcripts get none.
- **The reply** always comes back as text: `{reply, thinking, finish_reason, elapsed_ms, preset, input: {kind,
  chars}}`, and `input` also has `question_chars` with a question, and `block` and `sentence` for a sentence
  (never their text). `output.text` (`true` or `false`, `true` by default) says whether you want the text; the
  reply comes back as text whatever it says. A value that is not `true` or `false` is refused (`param`
  `output.text`), and so is `false` with neither `speech` nor `document`, which asks for no output (`param`
  `output`). `"output": {"speech": {…}}` also speaks it (`model`, `voice`, `format` and `speed`, by default
  `TTS_MODEL`, `TTS_VOICE`, `mp3` and `1.0`), cleaned as the Live Reader speaks, and returns the audio in base64
  (`audio: {format, content_type, base64, bytes}`): up to `TTS_MAX_INPUT_LENGTH` characters, and a longer reply
  comes back with `audio_error` (`speech_too_long`) instead. The speech settings are checked before the AI is
  asked: TTS off, or a model, voice or speed the TTS router refuses, is answered with `param` `output.speech`. `"output": {"document": {"title": …}}` saves it as a Reader
  document (`document_id`). Nothing else is kept on the server.
- **When the AI fails**, the answer is `502` (`504` for a timeout) with the error code and `elapsed_ms`, and,
  with speech asked and the preset's `speak_errors` on, the preset's sentence for it (`sentence`) and its
  audio.
- **A long input**, more than one piece of the preset's `chunk_chars`, runs as a book job instead: the answer
  is `202` with the `job`, which you follow (`GET /v1/reader/assistant-jobs/{id}`, each chapter's
  `pieces_done` of `pieces`) and cancel (`DELETE`) with the Reader's job routes, and whose answers become a new
  Reader document (`result_document_id`). A job is never spoken here; play its result in the Reader. A long text
  or transcript is kept in the Reader as a document first, so the job can resume and you can listen to it;
  the AI still gets the text exactly as it was sent, in pieces cut at blank lines (a longer paragraph at
  line breaks, then at sentences), each filling only `{text}`.
- **Refusals:** `422` `validation_error` with the field in `param` (every field outside these, `stream: true`,
  which is kept for a streamed reply later, and an input the Reader would need while `READER_ENABLED=false`),
  `404` `document_not_found` or `history_not_found`, and `503` `assistant_not_started`.

Presets also have a `label` (the action's name), `inputs` (the input kinds it suits) and `output` (its default
output) for Ask AI ([the presets](#settings-presets-and-the-three-layers)). The web app shows them; they never
reach the AI, and no request can override them.

### Settings › AI assistant

The web UI's Settings has an **AI assistant** section:

- **Status:** whether it is on, the AI server's host and the default model. Off, it says how to turn it
  on: enter the address under Connection, then Check and Save; when the connection is locked, set
  `OS_ASSISTANT_URL` in `.env` and restart. In the list of connection settings in words (shown when the
  Connection area can't be), "Certificate check: On (default)" means `OS_ASSISTANT_VERIFY_TLS` is unset, so
  the certificate is checked.
- **Connection:** the server URL, the model (a text field, with a list filled by Check), the API key, "Check
  TLS certificate", "Send thinking as template variables", "Stream replies", the token-limit field name and
  the extra body.
  - Each field says where its value comes from: from `.env`, saved here, or the default. A saved one has
    **Use .env**, which puts it back to `.env`.
  - The key is never shown, only "Set" or "Not set", with **Clear key**. It is sent only to the server it
    was saved for, so when the URL moves to another server (another scheme, host or port), the area says
    "Enter the key again for a new server", and saving without a key clears it. A change of path keeps it.
    It also stays with the certificate check: turning **Check TLS certificate** off for an https server
    needs the key again in the same save, which the key field says before you save; saving without it clears
    the key. Turning the check back on keeps the key.
  - **Check** asks the server for its models with the form as it is, unsaved, and generates nothing.
    **Save connection** sends only the fields you changed. **Revert to .env** forgets everything saved here,
    after asking.
  - Saving a URL turns the assistant on at once, with no restart. With
    `OS_ASSISTANT_CONNECTION_LOCKED=true` the area is a read-only list that points at `.env`.
- **Presets:** the picker (the built-ins first), **New**, **Duplicate** (a copy of the preset as it is on
  screen, unsaved changes included), and **Delete** for yours or **Restore defaults** for a built-in, each
  after a confirm. The fields are grouped as Ask AI, Prompts, Thinking, Limits, Sampling, While waiting and
  Errors; the voice preset hides the book's piece size, and the book preset the voice-only groups. **Ask AI**
  holds how the web app's Ask AI offers the preset, none of which reaches the AI: **Action name** (`label`),
  **First answer choice** (`output`) and **Offered for** (`inputs`: text, transcript, document).
  **Thinking** holds the mode, the effort and **Thinking budget (tokens)** (`thinking_budget`); **Limits**
  holds **Max tokens**, whose hint says that it cuts the answer off ([Thinking controls](#thinking-controls)).
  - **Save** sends the name, the preset's own fields and the ones you changed, so a field that comes from
    `.env` and that you didn't touch keeps following `.env`.
  - A refused field is marked with the server's reason. Unsaved changes are marked and can be discarded;
    switching presets asks first, and leaving the page warns.
- **Preview** speaks a phrase or an error sentence in the voice chosen in Speak. It can't use a reference
  recording chosen in Speak; choose a voice there first.
- **Test** sends one message ("Say hello in one sentence." unless you type one) with the preset as it is on
  screen, saved or not, and shows the reply, the thinking (folded), the time taken and the finish reason,
  or the failure in plain words. The reply isn't spoken, and nothing is saved. A field you emptied is tried
  as its `.env` or built-in default, as it would be once saved.

### The connection

`GET /api/assistant/connection` answers `{enabled, locked, url, model, api_key_set, api_key_tls_checked,
verify_tls, template_kwargs, max_tokens_field, extra_body, stream, sources}`; `sources` maps each of the eight
fields to `env`, `saved` or `default`. The key itself is never in an answer, a log, a preset or a job.
`api_key_tls_checked` is read-only: `true` or `false` says whether the key in use was set with the certificate
check on (for the `.env` key, the value of `OS_ASSISTANT_VERIFY_TLS`), and it is `null` when no key is in use. A
`PUT` that sends it back is not refused, and changes nothing.

- **`PUT`** takes any of the eight fields and saves each one sent, so send only what changes: `url` (`""`
  saves "off"), `api_key` (left out keeps the key, `""` clears it; stripped, then printable ASCII of up to
  4,096 characters), `model` (up to 200 characters), `verify_tls`, `template_kwargs` and `stream` (`true` or
  `false`; `stream` is Stream replies, `OS_ASSISTANT_STREAM`),
  `max_tokens_field` (`max_tokens` or `max_completion_tokens`), and `extra_body` (an object, or JSON text, up to
  16,000 characters, nesting objects and lists at most 32 levels deep, with no `NaN` or `Infinity`, which JSON
  doesn't have). `null` for any field forgets its saved value, so it follows `.env` again. It answers the new
  view; a bad field gets `422` with `param`, and nothing is saved.
  Saving rebuilds the shared client: a request already on its way finishes on the old one.
- **`DELETE`** forgets the saved connection, so every field follows `.env` again.
- **`POST /api/assistant/connection/check`** takes the same fields as an unsaved candidate over the
  connection in use (`{}` checks the one in use) and makes one `GET {url}/models`, within 10 seconds. It
  answers `200` `{reachable: true, status, models, listed, elapsed_ms}`: `models` holds only the model ids of
  an OpenAI-style list (`listed`), and another HTTP status gives `models: []`. A server it can't reach gets
  `502` `assistant_unreachable`, or `504` `assistant_timeout`, with `reachable: false`; no URL gets `422`
  with `param` `url`.
- **The key stays with its server.** A key is sent only to the origin (scheme, host and port, with 80 and
  443 filled in) of the URL it was saved for, and `OS_ASSISTANT_API_KEY` only to `OS_ASSISTANT_URL`'s
  origin. A `PUT` that moves the URL to another origin without `api_key` clears the saved key
  (`api_key_set: false`); a change of path, such as `/v1` to `/api/v1`, keeps it. A check of another origin
  sends no key unless its body has `api_key`.
- **The key stays with its certificate check.** Whether the certificate check was on when a key was saved
  is kept beside it (shown as `api_key_tls_checked`). A key saved with the check on isn't sent to an https URL
  once the check
  is off: a `PUT` that turns `verify_tls` off for an https URL without `api_key` clears the saved key
  (`api_key_set: false`), so give the key again in the same `PUT`, and a check with `verify_tls: false` sends
  no stored key. `OS_ASSISTANT_API_KEY` counts as set with `OS_ASSISTANT_VERIFY_TLS`, so
  `OS_ASSISTANT_VERIFY_TLS=false` with a key in `.env` keeps working, but turning the check off in Settings
  withholds the `.env` key from an https URL. Turning the check back on keeps a key, and an http URL, which
  has no certificate, isn't affected.
- **The key's own rule:** it is stripped (so a trailing line break or tab from a secret file isn't sent), and
  must then be printable ASCII of up to 4,096 characters. `OS_ASSISTANT_API_KEY` that breaks it stops startup
  with `OS_ASSISTANT_API_KEY must be printable ASCII of up to 4096 characters`, never quoting the key; a saved
  one gets `422` naming `api_key`. No error text holds the key.
- **Guards, for every URL from `.env` or saved:** `http` or `https` with a valid host and port (a host such
  as `192.168.1.300`, which isn't an address, is refused); no user name or password, query or fragment;
  redirects aren't followed; only a `200`'s body is read, up to 8 MiB; and an error names only the HTTP
  status, never the server's text, or says that the server's answer isn't valid HTTP. `OS_ASSISTANT_URL`
  that breaks one stops startup with a clear message, as does an `OS_ASSISTANT_EXTRA_BODY` that isn't a JSON
  object, nests deeper than 32 levels or holds `NaN` or `Infinity`. A startup error names the setting and the
  reason, never its value.
- **Locked.** With `OS_ASSISTANT_CONNECTION_LOCKED=true` the saved file is ignored, `PUT` and `DELETE`
  answer `403` `connection_locked`, and so does a check whose body sets any field.

`GET /api/assistant` reports `enabled`, the AI server's `host` (no path, query or key), the default
`model`, `server_settings` (for each connection setting, whether it differs from its default, never its
value), `locked`, and `sources` keyed by the `.env` names. `POST /api/assistant/test` takes
`{preset, text?, overrides?}` and answers `{reply, thinking, finish_reason, elapsed_ms}`, or `502` (`504`
for a timeout) with the error code and `elapsed_ms`. Its `overrides` are the preset as it is on screen: a
field sent as `null` or empty is tried as its `.env` or built-in default, and a field left out keeps the
saved value.

### Security and privacy

- **Your words and your books go to the AI server.** With a server on your network they stay there; a
  cloud API's URL sends them out.
- **Anyone who can use the API can change the presets and the connection**, and `/check` makes the server
  call any http or https address its network can reach (answering only the status and model ids). Set
  `OS_API_KEY` when others can reach the server, or `OS_ASSISTANT_CONNECTION_LOCKED=true` to keep the
  connection in `.env`.
- **The saved key** is kept in plain text in `assistant-connection.json` next to `OS_STUDIO_DB_PATH` (the
  `./data` volume under Compose), and is never returned or logged.
- **What is kept.** A voice conversation lasts as long as its connection. Ask AI keeps nothing on the server
  unless you save the answer, except for a long input: it runs as a book job, a long text or transcript is
  kept in the Reader as a document, and the answers are saved as a new one, both of which you can delete.
- **The AI's reply is only spoken or saved as text.** There are no tools and nothing it can run, so a
  malicious reply or book can only make the AI say something.
- **Limits:** `OS_STREAM_MAX_CONNECTIONS` caps realtime sessions, book jobs run one at a time, the preset
  fields have the caps above, and a reply is read up to 8 MiB.

### What to expect for speed

A voice turn's delay is roughly the end-of-turn silence (`silence_duration_ms`, 500 ms by default), plus
the faster-whisper decode, plus the AI's time for its thinking and the first sentence of its answer, plus
the voice model's first audio. With Stream replies off it is the AI's whole reply instead of its first
sentence. The AI part dominates. Low thinking keeps it short, keep voice replies short through the prompt,
and the waiting phrases fill the silence from 1.2 seconds on. Book jobs have no one waiting on each piece,
so more thinking is fine there. An AI server on another machine also leaves this machine's GPU to
faster-whisper and the voice model.

Each voice turn that goes to the AI logs one INFO line of its timings, in numbers only (never a word said,
answered or thought, and never the key):

```
Realtime turn of session sess_…: silence_ms=500 stt_ms=640 ai_first_token_ms=180 ai_first_answer_ms=1450 ai_done_ms=2300 first_audio_ms=1620 reply_chars=212 tokens_in=812 tokens_out=96 streamed=1
```

| Field | Meaning |
|---|---|
| `silence_ms` | The end-of-turn silence server VAD waited for; `-` for a typed question or a client's own commit |
| `stt_ms` | How long faster-whisper took on the turn; `-` for a typed question |
| `ai_first_token_ms` | From the request to the AI's first piece, thinking or answer |
| `ai_first_answer_ms` | From the request to the first piece of the answer: the gap from `ai_first_token_ms` is the thinking |
| `ai_done_ms` | From the request to the reply's end (or its failure) |
| `first_audio_ms` | From the request to the reply's first audio sent (not a waiting phrase's): what is left after `ai_first_answer_ms` is the voice's share (the first sentence completing, then the voice model) |
| `reply_chars` | The answer's length as spoken, in characters |
| `tokens_in`, `tokens_out` | The server's token counts, when it sends them (`stream_options.include_usage`) |
| `streamed` | `1` when the reply arrived as a stream, `0` when it came whole |

`-` marks what wasn't measured or reached. What you heard waiting is about `silence_ms + stt_ms +
first_audio_ms`. When `ai_first_answer_ms` is close to `ai_done_ms` on a streamed turn, the answer arrived in one
go: see [The AI assistant](#the-ai-assistant) under Troubleshooting.

## Docker Compose

### Default CPU Launch

Plain Compose is CPU-safe and does not request NVIDIA GPU passthrough:

```bash
docker compose up -d
```

It defaults to `STT_DEVICE=cpu`, `STT_COMPUTE_TYPE=int8`, and `TTS_DEVICE=cpu`. The base
Compose launch also selects the [Parakeet](#parakeet) STT model (English only; int8 on a CPU, fp16 with the GPU
override; set `STT_MODEL` in `.env` to a Whisper model for other languages, and `STT_TRANSLATE_MODEL` for
translations), enables Wyoming on `0.0.0.0`, caps the models loaded
besides the pinned defaults at two (`OS_MAX_LOADED_MODELS=2`), and uses a 2,000 ms streaming STT chunk
window. These are Compose launch defaults, not the source defaults listed under Environment Variables
below.

Compose reads `.env` next to the Compose files and passes most other `OS_*` settings through empty
when `.env` does not set them. An empty value counts as unset, so the source default applies. A
setting that the Compose file's `environment:` list does not name, such as `OS_PIN_DEFAULT_MODELS`,
`OS_WYOMING_MAX_CONNECTIONS`, `OS_BATCH_*`, `OS_HISTORY_*`, or `TTS_PRELOAD_DEFAULT`, reaches the
container only from `.env`; a shell variable such as `OS_PIN_DEFAULT_MODELS=false docker compose up`
has no effect.

Compose also expands a `$` in a `.env` value that isn't single-quoted: `$NAME` and `${NAME}` become
that variable, blank when it is unset (Compose then warns `The "NAME" variable is not set`). So
`OS_API_KEY=$ecretKey123` reaches the server empty, which turns authentication off, and
`OS_API_KEY=Xk9$mP2q` arrives as `Xk9`; `OS_ASSISTANT_API_KEY` and `HF_TOKEN` are cut the same way.
Single-quote the whole value (`OS_API_KEY='Xk9$mP2q'`): Compose and a source install's `uv run --env-file`
both read it as written. Under Compose alone, `$$` also writes a `$` (`OS_API_KEY=Xk9$$mP2q`), but uv cuts the
key at the `$$` just the same. Also single-quote a value that holds a double quote, a blank or a backslash, such as
`OS_ASSISTANT_EXTRA_BODY='{"top_k": 20}'`: Compose reads it either way, but uv drops each double quote and stops
reading `.env` at a blank or a backslash (see [Installation (from source)](#installation-from-source)). An apostrophe
is the one character the two can't share a spelling for: Compose reads `OS_API_KEY=Xk9'mP2q` as written, while uv
stops reading `.env` there unless it is written `Xk9\'mP2q`, and Compose would keep that backslash. Under Compose,
leave it as written; for both, use a key without an apostrophe (see [API key](#api-key)).

Legacy names in `.env`, such as `STT_API_KEY`, `STT_CORS_ORIGINS`, `STT_DEFAULT_MODEL`, or
`TTS_DEFAULT_VOICE`, still work under the Compose files and log a deprecation warning. Only
`STT_HOST` and `STT_PORT` are ignored there, because the container must listen on `0.0.0.0:8100`.
For the same reason the Compose files ignore `OS_HOST`, `OS_PORT`, and `OS_WYOMING_PORT` in `.env`:
the container listens on `0.0.0.0:8100` and Wyoming on `10400`, matching `ports:`. To publish other
host ports, change the host side of `ports:`, for example `"9000:8100"`.

`docker-compose.cpu.yml` extends the base file with the same CPU image, ports, and volumes, but changes
four settings: `STT_MODEL=nvidia/parakeet-tdt-0.6b-v2` (the base file's model, set again), Wyoming off
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
| `OS_MAX_LOADED_MODELS` | `2` (besides the pinned defaults) | `0` (unlimited) |
| `STT_MODEL` | `nvidia/parakeet-tdt-0.6b-v2` (int8 on `:cpu`, fp16 on `:latest`, CUDA) | `Systran/faster-whisper-base` |
| `STT_DEVICE` / `STT_COMPUTE_TYPE` / `TTS_DEVICE` | `cpu`/`int8`/`cpu` (`:cpu`), `cuda`/`float16`/`cuda` (`:latest`) | `cpu`/`int8`/unset |

The image also sets `TTS_MODEL=kokoro` and `TTS_VOICE=af_heart`, the source defaults. Because the
image sets `STT_MODEL`, `TTS_MODEL`, `TTS_VOICE`, and `OS_MAX_LOADED_MODELS` itself, a plain
`docker run --env-file .env` ignores their legacy names (`STT_DEFAULT_MODEL`, `TTS_DEFAULT_MODEL`,
`TTS_DEFAULT_VOICE`, `STT_MAX_LOADED_MODELS`); use the new names there. `docker run --env-file`
also takes everything after `=` as the value, a trailing `# comment`, blanks and quotes included, so
keep each comment on its own line, as `.env.example` does. `docker run --env-file` expands nothing and
keeps a `$$` and the quotes as written, so one `.env` can't give a value holding a `$` to both it and
Compose (see [Docker Compose](#docker-compose)).

The container entrypoint always sets `HF_HOME=/home/openspeech/.cache/huggingface` and
`STT_MODEL_DIR=/home/openspeech/.cache/huggingface/hub`, so values you pass for those two are replaced.
Wyoming listens on port `10400`; add `-p 10400:10400` to `docker run` if you use it.

### GPU Launch

GPU users must include the GPU override. Plain `docker compose up -d` is not a GPU launch.

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --force-recreate
```

The GPU override requests NVIDIA passthrough with `gpus: all`, includes the NVIDIA device reservation block, and sets `STT_DEVICE=cuda`, `STT_COMPUTE_TYPE=float16`, and `TTS_DEVICE=cuda`.

For voice cloning and voice design in the web UI (Voices), add one of the optional GPU voice workers: Chatterbox,
Qwen3, Fish Audio S2 or Breeze TTS 2. They are off by default. `docker-compose.voice-models.yml` gives each worker a
Compose profile of its own, named after it (`chatterbox`, `qwen3`, `fish`, `breeze`), so layering the file alone
starts and builds none of them. Turn one
on by naming its profile:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  -f docker-compose.voice-models.yml --profile qwen3 up -d --build
```

Use `--profile chatterbox`, `--profile fish` or `--profile breeze` for the others; for two, repeat the flag
(`--profile qwen3 --profile chatterbox`), and `--profile "*"` turns on all four. `COMPOSE_PROFILES=qwen3,chatterbox`, in the shell
or in `.env`, does the same as the flags, and also covers later commands such as `down`. The core still lists every
worker's models: one whose worker isn't turned on reads "Worker offline" in Models, and the core runs without it.
The core reaches each worker by a network alias only that worker holds, such as
`http://chatterbox.voice-workers.internal.:8210`, so a device on your network called `chatterbox` can't stand in for
a worker that is off (see [Voice provider deployment](docs/TTS-BACKENDS.md#voice-provider-deployment)).

A worker container that an earlier `up` started keeps running, on the image it started from, until it is stopped:
`up` leaves alone a worker whose profile isn't named, and `down` without its profile doesn't stop it. A host that
ran this file before the profiles has its workers running, so stop them once before the first `up` that names a
profile (this stops the core too; the `up` starts it again):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  -f docker-compose.voice-models.yml --profile "*" down
```

Stop one worker you no longer want with the same `-f` files and `stop chatterbox` (or `rm -s -f chatterbox` to
remove its container). A `--profile` flag replaces `COMPOSE_PROFILES` rather than adding to it: with
`COMPOSE_PROFILES=qwen3` in `.env`, `--profile chatterbox up -d` starts Chatterbox and leaves Qwen3 running as it
was, and a later `down` (which reads `.env`) leaves Chatterbox running. To audition another worker, stop the
one before (`stop qwen3`) or change `COMPOSE_PROFILES` itself.
Only Qwen3 Custom Voice (`qwen3/0.6b-custom-voice`) has built-in voices; Chatterbox, Qwen3 Base, Fish
Audio S2 and Breeze TTS 2 clone from a recording.

Fish Audio S2 (`fish/s2-pro`) runs the S2 Pro checkpoint quantized to 4 bits (NF4), about 3.3 GB of GPU memory. Its
first load fetches about 5 GB of weights and compiles for a few minutes (`FISH_COMPILE=false` skips the compile), and
after each unload the worker restarts to free its memory.

Breeze TTS 2 (`breeze/tts-2`) runs through the community C++ port Breeze-TTS-2.cpp with its Q8_0 weights, about
4.4 GB of GPU memory. Its first load fetches about 3.6 GB of weights. It designs a voice from a description and clones
one from a recording with its exact transcript; inline cues such as `(laugh)` are part of the text. On an 8 GB card
it fits next to Kokoro once Parakeet is unloaded, which a load from Models does when it must (see
[GPU memory guard](#gpu-memory-guard)). Its image is built for Turing (RTX 20) GPUs with code that newer ones compile
at their first load; `--build-arg BREEZE_CUDA_ARCHITECTURES=86` (RTX 30, or `89`, `120`) builds it for your card.

The provider services remain ready while their model weights stay unloaded. In **Models**, use
**Download** to cache a model or **Load** to activate it. In **Voices › Recordings**, choosing a clone
model and clicking **Test clone** on a recording performs the load automatically. An explicit load unloads the other
loaded TTS models first. A TTS model that a later request loads on first use, such as Kokoro for a
speech request, then stays loaded beside it: the pinned `TTS_MODEL` until an explicit unload or load,
and any other model until idle or LRU eviction unloads it. See [Model residency](#model-residency).

The Qwen3 provider's `qwen3/0.6b-base` cloning model is hidden unless `QWEN3_ENABLE_BASE=true`;
`docker-compose.voice-models.yml` sets it to `true`. What each cloning model needs and speaks is in
[Cloning models](docs/CLONING-MODELS.md).

Each worker image builds with `providers/` as its context and `<name>/Dockerfile` as its Dockerfile,
so it can copy the shared `providers/_common` code. Rebuild existing worker images after an upgrade
with `up -d --build` as above. To build one by hand, run
`docker build -f providers/chatterbox/Dockerfile providers` (or `qwen3`, `fish`, `breeze`);
`docker build providers/<name>` no longer works. A worker that runs out of GPU memory answers HTTP
`507`, frees its CUDA cache, and stays healthy; Open Speech reports it as `503` (code
`insufficient_gpu_memory`). A worker whose torch build has no code for the GPU (for example an RTX
50-series card, `sm_120`) can never load a model: its `/health` reports `status: "failed"` with
`failure.code` `cuda_arch_unsupported` and a message naming the architecture, and Compose shows it
unhealthy. Open Speech then answers its models' loads with `503 provider_unavailable` and the worker's
reason, and the Models view reads "Worker can't run" (see
[docs/TTS-BACKENDS.md](docs/TTS-BACKENDS.md#voice-provider-deployment)).

#### Voice provider settings

These go in `.env` next to the Compose files. The Compose files and the provider workers in
`providers/*/app.py` read them; they are not Open Speech server settings.

| Variable | Default | Description |
|---|---|---|
| `OPEN_SPEECH_IMAGE` | `jwindsor1/open-speech:cpu` (`:latest` with the GPU override) | Core harness image tag |
| `OPEN_SPEECH_REVISION` | `unknown` | Git revision recorded by local image builds |
| `COMPOSE_PROFILES` | `""` (no worker) | Which optional GPU voice workers `docker-compose.voice-models.yml` starts, comma-separated: `chatterbox`, `qwen3`, `fish`, `breeze` (read by Docker Compose, the same as `--profile`; each worker is off by default) |
| `OPEN_SPEECH_CHATTERBOX_IMAGE` | `open-speech-chatterbox:0.2.1` | Chatterbox worker image tag |
| `OPEN_SPEECH_QWEN3_IMAGE` | `open-speech-qwen3:0.2.1` | Qwen3 worker image tag |
| `OPEN_SPEECH_FISH_IMAGE` | `open-speech-fish:0.1.0` | Fish Audio S2 worker image tag |
| `OPEN_SPEECH_BREEZE_IMAGE` | `open-speech-breeze:0.1.0` | Breeze TTS 2 worker image tag |
| `HF_TOKEN` | `""` | Optional Hugging Face token for gated downloads (core and workers) |
| `QWEN3_ENABLE_BASE` | `false` in the worker; `true` in the voice-models Compose file | Show the `qwen3/0.6b-base` cloning model |
| `QWEN3_DTYPE` | `auto` | `auto`, `float32`, `float16`, or `bfloat16` |
| `QWEN3_MAX_SEGMENT_UNITS` | `400` | Script-weighted text per Qwen segment |
| `QWEN3_MAX_NEW_TOKENS_CEILING` | `1200` | Qwen generation ceiling (minimum 192) |
| `QWEN3_PROMPT_CACHE_SIZE` | `8` | Cached reference prompts per Qwen3 worker |
| `QWEN3_STREAM_QUEUE_TIMEOUT_S` | `30` | Qwen streaming queue timeout |
| `CHATTERBOX_MAX_INPUT_CHARS` | `350` | Chatterbox per-request text limit |
| `CHATTERBOX_RESTART_AFTER_UNLOAD_S` | `0` in the worker; `10` in the voice-models Compose file | After an unload, restart the Chatterbox worker once it has been idle this long, so it releases about 1.5 GiB of host RAM (`0` disables) |
| `CHATTERBOX_REFERENCE_CACHE_SIZE` | `8` | Cached reference conditionals per Chatterbox worker, cleared on unload |
| `FISH_MAX_INPUT_CHARS` | `1500` | Fish Audio S2 per-request text limit |
| `FISH_SEGMENT_CHARS` | `300` | Most text Fish Audio S2 speaks in one generation; longer text goes in sentence segments |
| `FISH_MAX_REFERENCE_SECONDS` | `30` | Longest reference recording Fish Audio S2 takes |
| `FISH_COMPILE` | `true` | `torch.compile` the Fish model at load; `false` if the compile fails on your GPU (slower) |
| `FISH_MAX_SEQ_LEN` | `2048` | Fish context length in tokens (reference plus segment) |
| `FISH_REFERENCE_CACHE_SIZE` | `8` | Cached encoded references per Fish worker, cleared on unload |
| `FISH_TEMPERATURE`, `FISH_TOP_P` | `0.7`, `0.7` | Fish sampling |
| `FISH_SEED` | `42` | Fish sampling seed, the same for every segment so a text keeps one voice |
| `FISH_RESTART_AFTER_UNLOAD_S` | `1` | Restart the Fish worker this long after an unload or a failed load, since it can't free its model otherwise (`0` disables) |
| `FISH_OFFLOAD_EMBEDDINGS`, `FISH_DECODE_CHUNK_FRAMES` | `1`, `64` | Read by the Fish fork itself: keep token embeddings in host RAM, and decode 64 codec frames at a time |
| `BREEZE_MAX_INPUT_CHARS` | `4096` | Breeze TTS 2 per-request text limit; the port speaks longer text in pieces of whole sentences |
| `BREEZE_MAX_REFERENCE_SECONDS` | `30` | Longest reference recording Breeze TTS 2 takes |
| `BREEZE_DESIGN_CFG`, `BREEZE_CLONE_CFG`, `BREEZE_CUE_CFG` | `3.0`, `1.0`, `2.5` | Breeze guidance for a designed voice, for a cloned one, and the least for text with an inline cue such as `(laugh)`, which needs it to be voiced (past about 3 it sounds harsh) |
| `BREEZE_SEED` | `42` | Breeze sampling seed for a cloned voice; a designed one gets a new seed each time unless the request names one |
| `BREEZE_TEXT_ENCODER_CPU` | `false` | Run Breeze's text encoder on the CPU: about 1 GB less GPU memory, about 0.7 s more per piece of text |
| `QWEN3_MAX_REFERENCE_MB`, `CHATTERBOX_MAX_REFERENCE_MB`, `FISH_MAX_REFERENCE_MB`, `BREEZE_MAX_REFERENCE_MB` | `100` | Max reference upload size; Compose sets all four from `OS_MAX_UPLOAD_MB` (or the legacy `STT_MAX_UPLOAD_MB`) |

### Volumes

Model caches live under `/home/openspeech/.cache/huggingface` inside the container.
Persist that path unless you enjoy re-downloading large things for sport. The checked-in Compose files persist:

| Volume | Container path | Holds |
|---|---|---|
| `hf-cache` | `/home/openspeech/.cache/huggingface` | STT and TTS model weights |
| `vad-cache` | `/home/openspeech/.cache/silero-vad` | Silero VAD model |
| `ssl-certs` | `/var/lib/open-speech/certs` | Generated TLS certificate and key |
| `tts-cache` | `/var/lib/open-speech/cache` | TTS cache (`TTS_CACHE_ENABLED=true`) |
| `./data` (bind mount) | `/home/openspeech/data` | Studio database, voice library, conversations, composer renders, Reader audio, batch jobs and their queued uploads |

The server in the container runs as the `openspeech` user, uid 1000, and writes to `./data`. On Linux a bind mount
keeps the checkout's owner, so when the server can't write `./data` (a checkout cloned with `sudo`, or by another
account), the container's start gives the folder to uid 1000 (`chown -R`) before the server starts, and the host
then shows uid 1000 as its owner. `sudo chown -R 1000:1000 data` before the first start does the same by hand.
Docker Desktop on Windows and macOS maps the owner, so the server can write `./data` there and nothing changes.

`docker-compose.voice-models.yml` adds `chatterbox-hf-cache`, `qwen3-hf-cache`, `fish-hf-cache` and `breeze-hf-cache` for the workers' model weights, `fish-compile-cache` for Fish Audio S2's compiled kernels, and `breeze-cuda-cache` for the CUDA code a GPU newer than the Breeze build compiles at its first load.

## Troubleshooting

### The server log

When something doesn't work, set `OS_LOG_LEVEL=DEBUG` in `.env`, restart with the same `-f` files and profiles
that started the server, and watch `docker compose logs -f open-speech`. That restart is `docker compose up -d`
for the plain CPU launch only; for the `docker-compose.cpu.yml` launch it is
`docker compose -f docker-compose.cpu.yml up -d`, and on a GPU host it is
`docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d`, plus
`-f docker-compose.voice-models.yml --profile qwen3` (your workers' profiles) when you use them. A bare
`docker compose up -d` recreates any other launch from `docker-compose.yml` alone: a GPU server as the CPU image
without the voice workers, and a `docker-compose.cpu.yml` server with Wyoming on `0.0.0.0`, where it has no
authentication, the Turbo STT model and a two-model cap. DEBUG lines can
hold what was said (transcript text), so set it back to `INFO` afterwards. No level logs the API key or a WebSocket ticket from Open Speech's own code, and uvicorn's
request lines show the values of `api_key`, `ticket`, `exp` and `sig` in a query string as `***` at every
level. A reverse proxy's logs and browser history still keep a `?api_key=`, so prefer the `Authorization`
header or a ticket to the deprecated `?api_key=`. At `DEBUG`, httpx, httpcore, urllib3, websockets and other
chatty libraries stay at `INFO`.

### Nothing transcribed?

When a `/v1/audio/stream` or `/v1/realtime` session ends, the server logs one INFO line, such as
`Streaming session … ended (transcriptions=0, errors=0, audio=6.9 s, peak=silent, vad_max=0.00, speech=0)` or
`Realtime session … ended (audio=6.9 s, peak=-23.1 dBFS, vad_max=0.97, speech=2)`:

- `audio=0.0 s`: no audio arrived.
- `peak=silent`: the client sent only zeros (a muted or wrong microphone).
- A peak with `speech=0`: the VAD heard no speech. `vad_max` stayed below `STT_VAD_THRESHOLD`, or below
  `turn_detection.threshold` on `/v1/realtime`.
- `speech` above 0 with no transcript: the speech was transcribed to nothing.
- `vad_max=none`: the VAD scored no audio (it was off, or less than one 32 ms window arrived).

The line holds no audio and no text.

### The AI assistant

[docs/AI-ASSISTANT.md](docs/AI-ASSISTANT.md#expected-behavior) lists what each spoken error means and what to
check. Settings › AI assistant › Connection's **Check** asks the AI server for its models without generating
anything, and **Test** makes one real call.

**Is the wait the AI or the voice?** Each voice turn's [timing line](#what-to-expect-for-speed) splits it:
a large `ai_first_answer_ms` is the AI thinking (lower the effort or the thinking budget), a large gap from
there to `first_audio_ms` is the first sentence being long or the voice model being slow, and `silence_ms`
plus `stt_ms` is Open Speech hearing you.

**The voice only starts when the AI has finished, though `streamed=1`.** Something between Open Speech and
the AI server holds the stream back and sends it all at the end: the timing line shows `ai_first_token_ms`
close to `ai_done_ms`. A reverse proxy such as nginx, or a front end such as Open WebUI, must pass server-sent
events through unbuffered; for nginx, `proxy_buffering off;` in the location that proxies `/v1/`. A server
that can't stream at all answers in one piece (`streamed=0`) and still works; with `OS_ASSISTANT_STREAM=false`
Open Speech doesn't ask it to.

## Development and validation

Run ordinary tests from the repository root. Open Speech supports **Python 3.12 only**. Keep the extras that
[Installation (from source)](#installation-from-source) used (for a CUDA install, `--extra cuda` in place of
`--extra cpu`, with `--no-install-package onnxruntime` on `uv sync` and `uv run --no-sync` in place of
`uv run --frozen --extra ...`): `uv sync` removes every package the extras it is given don't need, so leaving
out `--extra all` uninstalls Kokoro, Supertonic, Piper and Pocket-TTS.

```bash
uv sync --frozen --extra cpu --extra all --extra dev
uv run --frozen --extra cpu --extra all --extra dev python -m pytest -q -rs tests
```

Native Windows uses the same suite:

```powershell
uv sync --frozen --extra cpu --extra all --extra dev
uv run --frozen --extra cpu --extra all --extra dev python -m pytest -q -rs tests
```

Pass `tests` as the path, not `.`: the `./data` directory that Compose mounts can hold third-party
packages with test files of their own, which break collection.

The suite runs with a home folder and a Hugging Face cache of its own, under its temporary folder (`HOME`,
`USERPROFILE`, `XDG_CACHE_HOME`, `HF_HOME` and `HF_HUB_CACHE` are set for it, and `STT_MODEL_DIR` and
`HUGGINGFACE_HUB_CACHE` from your shell are dropped), so it never reads or deletes the models you downloaded.

Put Node.js and ffmpeg on `PATH` to exercise browser behavior and real audio encoding.
Node.js 22.18 or newer also runs the client-js checks in `tests/test_client_sdk.py`, which load
`client-js/src/index.ts` without a build step; older versions skip them.
Six Kokoro unit tests require the optional
Torch runtime. Windows skips the four symlink tests only when the account lacks symbolic-link privilege; Linux
still exercises them.
The JavaScript SDK's own tests (`client-js/tests/index.test.ts`) use the Node.js test runner: the
suite runs them through `tests/test_client_sdk.py`, and `node --test client-js/tests/index.test.ts`
runs them alone.
GitHub Actions (`.github/workflows/tests.yml`) runs the Python test suite on Python 3.12, with
Node.js 24, for pushes to `main` and for pull requests.

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

Open Speech's defaults are locked down. A deployment reached by IP address, `localhost`, or a LAN name
works without changing any security setting; set `OS_API_KEY` to require a key. A public name, a
reverse proxy, a page on another site, or an iframe needs one of the opt-outs below. The Wyoming port
is separate and has no authentication; see [Wyoming](#wyoming). When upgrading, note that these
defaults are new and that `docker-compose.yml` no longer sets `OS_CORS_ORIGINS=*`; see the upgrade
notes in [CHANGELOG.md](CHANGELOG.md).

### API key

Set the key in `.env`. A key of letters, digits, `-` and `_` reads the same everywhere;
`python -c "import secrets; print(secrets.token_urlsafe(32))"` prints one. Single-quote a key that holds a `$`, as
in `OS_API_KEY='Xk9$mP2q'`: Compose and a source install's `uv run --env-file` both read it as written, while `$$`
works under Compose only (see [Docker Compose](#docker-compose)). Single quotes can't hold an apostrophe: in a source
install's `.env`, write it as `\'` (`OS_API_KEY=Xk9\'mP2q`) or export the key in the shell, because uv stops reading
`.env` at a bare one and drops the key and `OS_AUTH_REQUIRED` below it, which leaves authentication off (see
[Installation (from source)](#installation-from-source)). For example:

```bash
OS_API_KEY=my-secret-key

# Stop at startup when OS_API_KEY is empty
OS_AUTH_REQUIRED=true
```

Then restart with the same `-f` files and profiles that started the server: `docker compose up -d` for the plain
CPU launch, `docker compose -f docker-compose.cpu.yml up -d` for the `docker-compose.cpu.yml` one, or
`docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d` for the GPU one. A bare
`docker compose up -d` recreates any other launch from `docker-compose.yml` alone: a GPU server as the CPU image (see
[GPU Launch](#gpu-launch)), and a `docker-compose.cpu.yml` server with Wyoming on `0.0.0.0`, where it has no
authentication, the Turbo STT model and a two-model cap. Then check:

```bash
curl -sk https://localhost:8100/health  # public liveness check
curl -sk -H "Authorization: Bearer my-secret-key" https://localhost:8100/v1/models
```

- With `OS_API_KEY` set, API requests need `Authorization: Bearer KEY`; the scheme is
  case-insensitive. The paths that need no key are listed under [API Reference](#api-reference).
- WebSocket clients that can set headers send the same header. Browsers cannot, so they first call
  `POST /api/auth/ws-ticket` with the key and then connect with `?ticket=TICKET`. A ticket works for
  one connection within 60 seconds; see [Authentication](#authentication) for what a bad ticket
  gets. The deprecated `?api_key=KEY` query parameter still works and logs a warning each time. Open
  Speech's own log shows the values of `api_key`, `ticket`, `exp` and `sig` in a query string as `***`, but a
  reverse proxy's logs and browser history still keep the key.
- **Failed-auth lockout.** After `OS_AUTH_FAILURE_LIMIT` (default `10`) wrong keys from one client
  address within 60 seconds, counting `Authorization` headers, `?api_key=` values, and WebSocket
  tickets, that address gets `429` with `Retry-After` for 60 seconds on HTTP requests and WebSocket
  handshakes, even with the right key. Requests without any key do not count. Clients that share one
  address, such as clients behind NAT, through Docker Desktop's port publishing, or behind a reverse
  proxy without `OS_TRUST_PROXY=true`, are locked out together. Behind a proxy, set
  `OS_TRUST_PROXY=true`; otherwise `OS_AUTH_FAILURE_LIMIT=0` turns the lockout off. The lockout
  applies only when `OS_API_KEY` is set.

### Host names

Every HTTP request and WebSocket handshake must name an accepted host in its `Host` header, which
blocks DNS rebinding attacks from web pages. Accepted without configuration:

- IP addresses, and names without a dot (`lan-box`, `localhost`)
- `*.localhost`, `*.local`, `*.lan`, `*.home.arpa`, and `*.internal`
- the server's own hostname and FQDN (inside Docker, the container's)

Any other name gets `400 Host not allowed. Add it to OS_ALLOWED_HOSTS on the server.` until you list
it in `OS_ALLOWED_HOSTS`, a comma-separated list. `*.example.com` matches subdomains of `example.com`
but not `example.com` itself, and `*` turns the check off. Router names that end in `.fritz.box`,
`.localdomain`, or `.home` need listing too:

```bash
OS_ALLOWED_HOSTS=speech.example.com,*.fritz.box,*.localdomain,*.home
```

A malformed `Host` header (one with a path, `?`, `#`, `@`, or spaces, a port above 65535, or a request
with two `Host` lines) gets `400 Invalid Host header`. The first refused name is logged once with a
hint about `OS_ALLOWED_HOSTS`.

### Browser origins

- **CORS.** `OS_CORS_ORIGINS` is empty by default, so the server sends no CORS headers and only its own
  pages can read API responses. A comma-separated list of origins gets CORS headers, with
  credentials, and may send writes; `*` sends CORS headers to any origin, without credentials. The
  answer to an unexpected server error (`500`, code `internal_error`) carries the same headers, so the page
  can read it.
- **Cross-site write guard.** A `POST`, `PUT`, `PATCH`, or `DELETE` whose `Origin` is neither this
  server's own (the same host and port as the `Host` header) nor listed in `OS_CORS_ORIGINS` gets
  `403`. So does `Origin: null`, and a request without an `Origin` that sends
  `Sec-Fetch-Site: cross-site`. Other requests without an `Origin`, such as SDK and curl calls, are not
  affected. `OS_CORS_ORIGINS=*` lets every origin through the guard.
- **WebSockets.** `OS_WS_ALLOWED_ORIGINS` is empty by default, which accepts only this server's own
  pages. List other origins to accept them too, or set `*` for any. Clients that send no `Origin`
  (SDKs, Home Assistant, CLI tools) are always accepted. A refused page's socket is opened and closed at once
  with `1008` ([Refused WebSockets](#refused-websockets)). With `OS_API_KEY` set, a page on another origin also
  needs that origin in `OS_CORS_ORIGINS`:
  its `POST /api/auth/ws-ticket` is a cross-site write, and without CORS it cannot read the ticket.
- **Wildcards.** Without `OS_API_KEY`, `OS_CORS_ORIGINS=*` lets any web page the user opens send
  writes (synthesis, history deletes, model loads) and read the answers, `OS_WS_ALLOWED_ORIGINS=*`
  lets it open WebSockets, and `OS_ALLOWED_HOSTS=*` lets a DNS rebinding page do the same. Set
  `OS_API_KEY` before using any of these `*` values.

```bash
OS_CORS_ORIGINS=https://myapp.com,https://staging.myapp.com
OS_WS_ALLOWED_ORIGINS=https://myapp.com,https://staging.myapp.com
```

### Reverse proxies

1. Point the proxy at `https://HOST:8100`. Open Speech serves HTTPS with a self-signed certificate by
   default, so turn off the proxy's certificate verification for this upstream, or give Open Speech a
   certificate the proxy trusts (see [TLS certificates](#tls-certificates)). When only the proxy can
   reach Open Speech, `OS_SSL_ENABLED=false` serves plain HTTP to it instead.
2. Pass the client's `Host` header through to Open Speech, port included, and list the public name in
   `OS_ALLOWED_HOSTS`. nginx sends the upstream's name by default; set
   `proxy_set_header Host $http_host;`. nginx's `$host` drops the port, so it works only when the
   proxy listens on the default port (443 or 80). The same-origin checks compare the host and port of
   the browser's `Origin` with `Host`.
3. If the proxy rewrites `Host` instead, same-origin checks no longer match the browser's `Origin`:
   list the public origin, such as `https://speech.example.com`, in both `OS_CORS_ORIGINS` and
   `OS_WS_ALLOWED_ORIGINS`.
4. Set `OS_TRUST_PROXY=true`, so rate limits and the failed-auth lockout see each client's own address
   from `X-Forwarded-For`. The client is the entry `OS_TRUSTED_PROXY_HOPS` (default `1`) from the right,
   counting every `X-Forwarded-For` line; raise it when several proxies each append an entry. Leave
   `OS_TRUST_PROXY=false` when clients can reach Open Speech directly, because they could forge the
   header.

### Security headers

With `OS_SECURITY_HEADERS=true` (the default), every HTTP response carries
`X-Content-Type-Options: nosniff`, `Referrer-Policy: same-origin`, `X-Frame-Options: DENY`,
`Cross-Origin-Opener-Policy: same-origin`, and
`Permissions-Policy: microphone=(self), camera=(), geolocation=()`. HTML from `/web` and `/static` also
gets a strict `Content-Security-Policy` (`script-src 'self'`, `frame-ancestors 'none'`); `/docs` and
`/redoc` do not. To embed the web UI in an iframe, such as a Home Assistant webpage panel, set
`OS_SECURITY_HEADERS=false`, which turns all of these headers off.

`/docs` and `/redoc` are off unless `OS_API_DOCS=true`. FastAPI's Swagger UI and ReDoc pages load their
scripts from a CDN, unpinned, on the server's own origin, where the web UI keeps its API key in browser
storage, so a compromised CDN could read it. Turn them on only where that is acceptable, such as a
trusted LAN without `OS_API_KEY`.

### Rate limiting

```bash
# Requests per minute per client address; 0 = off
OS_RATE_LIMIT=60
OS_RATE_LIMIT_BURST=10
```

WebSocket handshakes share the per-address bucket with HTTP requests. A limited request gets `429` with
`Retry-After`, and responses carry `X-RateLimit-Limit` and `X-RateLimit-Remaining`. Request bodies are
limited separately; see [Upload and body limits](#upload-and-body-limits).

### TLS certificates

```bash
# Your own certificate: set both, to files that exist
OS_SSL_CERTFILE=/certs/cert.pem
OS_SSL_KEYFILE=/certs/key.pem

# Extra SANs for the generated self-signed certificate
OS_TLS_EXTRA_SANS=192.0.2.24,openspeech.local
```

- With neither set, Open Speech generates a self-signed pair in `/var/lib/open-speech/certs`, valid
  for 365 days for `localhost`, `127.0.0.1`, `0.0.0.0`, and the `OS_TLS_EXTRA_SANS` entries. At
  startup it generates a new pair once the certificate expires within 30 days. Clients that trusted
  the old certificate must then trust the new one.
- Set `OS_SSL_CERTFILE` and `OS_SSL_KEYFILE` together. Open Speech never replaces your own files or
  changes their permissions. When neither file exists, it generates a self-signed pair at those paths.
  When only one of them exists, startup stops with an error rather than overwrite it.
- The SANs are read only when a pair is generated. If you change `OS_TLS_EXTRA_SANS` after a cert has
  been generated, remove the generated cert and key so Open Speech can generate them again. With the
  checked-in Compose files, that usually means removing the project's `ssl-certs` Docker volume.

## Environment Variables

Defaults come from `src/config.py`. The Docker image and the checked-in Compose files override several
of them (see [Docker image defaults](#docker-image-defaults) and [Docker Compose](#docker-compose));
the GPU override changes the device settings to CUDA. A source install reads only the process
environment, so use `uv run --env-file .env` or export the variables yourself; in that `.env`, single-quote a
value that holds a `$`, a double quote, a blank or a backslash, as in `OS_API_KEY='Xk9$mP2q'`, and write an
apostrophe as `\'`, as in `OS_API_KEY=Xk9\'mP2q` (see
[Installation (from source)](#installation-from-source)). An empty value counts
as unset, so the default applies. A value that isn't valid stops startup with an error that names the
setting and the reason, never the value, because it may be a secret.

Older names still work and log a deprecation warning at startup; each applies when its new name is
unset or empty: `STT_PORT`, `STT_HOST`, `STT_API_KEY`, `STT_CORS_ORIGINS`, `STT_TRUST_PROXY`,
`STT_MAX_UPLOAD_MB`, `STT_RATE_LIMIT`, `STT_RATE_LIMIT_BURST`, `STT_SSL_ENABLED`, `STT_SSL_CERTFILE`,
`STT_SSL_KEYFILE`, `STT_MODEL_TTL`, `STT_MAX_LOADED_MODELS`, `STT_STREAM_CHUNK_MS`,
`STT_STREAM_ENDPOINTING_MS`, and `STT_STREAM_MAX_CONNECTIONS` (now `OS_*`), plus `STT_DEFAULT_MODEL`,
`TTS_DEFAULT_MODEL`, `TTS_DEFAULT_VOICE`, and `TTS_DEFAULT_SPEED` (now `STT_MODEL`, `TTS_MODEL`,
`TTS_VOICE`, and `TTS_SPEED`). `OS_HISTORY_RETAIN_AUDIO` and `OS_HISTORY_MAX_MB` were removed, because
history never stored audio files; setting either logs a startup warning and has no effect.

### `OS_*` — harness / shared

| Variable | Default | Description |
|---|---|---|
| `OS_PORT` | `8100` | HTTP bind port |
| `OS_HOST` | `0.0.0.0` | HTTP bind host |
| `OS_API_KEY` | `""` | Bearer API key; empty disables auth |
| `OS_AUTH_REQUIRED` | `false` | Fail startup if `OS_API_KEY` is empty |
| `OS_AUTH_FAILURE_LIMIT` | `10` | Wrong API keys or WebSocket tickets per client address within 60 s before a 60 s lockout (`429`, even with the right key); `0` = off. Applies only when `OS_API_KEY` is set |
| `OS_ALLOWED_HOSTS` | `""` | Host names served besides IP addresses, dotless names, `*.localhost`, `*.local`, `*.lan`, `*.home.arpa`, `*.internal`, and the server's own names; comma-separated, `*.example.com` wildcards, `*` turns the check off |
| `OS_CORS_ORIGINS` | `""` | Empty = same-origin only, with no CORS headers; listed origins (comma-separated) get CORS headers with credentials and may send writes; `*` sends CORS headers without credentials and lets every origin through the cross-site write guard (set `OS_API_KEY` first) |
| `OS_WS_ALLOWED_ORIGINS` | `""` | Browser origins allowed on WebSockets besides this server's own; `*` = any. Clients that send no `Origin` are always allowed |
| `OS_SECURITY_HEADERS` | `true` | Send the nosniff, frame, referrer, opener, and permissions headers, plus a strict CSP on the web UI; `false` allows iframe embedding |
| `OS_API_DOCS` | `false` | Serve Swagger UI at `/docs` and ReDoc at `/redoc` (they load scripts from a CDN); `/openapi.json` is always served |
| `OS_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR` (any case): the server's log level, which uvicorn's access log follows. `DEBUG` adds diagnostic lines, among them the text of `/v1/audio/stream` transcripts; httpx, httpcore, urllib3, websockets and other chatty libraries stay at `INFO`. Any other value stops startup ([The server log](#the-server-log)) |
| `OS_TRUST_PROXY` | `false` | Take the client address from `X-Forwarded-For` |
| `OS_TRUSTED_PROXY_HOPS` | `1` | With `OS_TRUST_PROXY=true`: proxies that append to `X-Forwarded-For`; the client is the entry this many places from the right |
| `OS_MAX_UPLOAD_MB` | `100` | Max upload and request body size (see [Upload and body limits](#upload-and-body-limits)); also caps base64 `reference_audio` and Wyoming audio; checked-in provider Compose files also use it for voice references |
| `OS_RATE_LIMIT` | `0` | Requests per minute per client address, WebSocket handshakes included; `0` disables |
| `OS_RATE_LIMIT_BURST` | `0` | Burst bucket size; `0` = `OS_RATE_LIMIT` |
| `OS_SSL_ENABLED` | `true` | Enable HTTPS |
| `OS_SSL_CERTFILE` | `""` | TLS cert path; empty uses a generated cert in `/var/lib/open-speech/certs`. Set together with `OS_SSL_KEYFILE` |
| `OS_SSL_KEYFILE` | `""` | TLS key path; empty uses the generated key. Set together with `OS_SSL_CERTFILE` |
| `OS_TLS_EXTRA_SANS` | `""` | Extra comma-separated DNS/IP names for generated self-signed certs |
| `OS_VOICE_LIBRARY_PATH` | `/home/openspeech/data/voices` | Stored voice reference directory |
| `OS_VOICE_LIBRARY_MAX_COUNT` | `100` | Max stored voice refs; `0` = unlimited |
| `OS_VOICE_LIBRARY_MAX_SECONDS` | `60` | Max reference duration; `0` = unlimited |
| `OS_STUDIO_DB_PATH` | `/home/openspeech/data/studio.db` | SQLite DB for studio metadata; batch jobs and their queued uploads go in the same directory |
| `OS_HISTORY_ENABLED` | `true` | Log requests that send `X-History: true` |
| `OS_HISTORY_MAX_ENTRIES` | `1000` | Max retained history rows; `0` = unlimited |
| `OS_EFFECTS_ENABLED` | `true` | Apply `effects` on speech requests and on Studio presets, turns, and tracks |
| `OS_CONVERSATIONS_DIR` | `/home/openspeech/data/conversations` | Conversation storage directory |
| `OS_COMPOSER_DIR` | `/home/openspeech/data/composer` | Composer storage directory |
| `OS_BATCH_WORKERS` | `2` | Batch jobs that run at once; only one batch file is transcribed at a time |
| `OS_BATCH_MAX_PENDING` | `10` | Max queued + running batch jobs; more get `429` |
| `OS_BATCH_MAX_TOTAL_MB` | `500` | Max aggregate upload size per batch request |
| `OS_WYOMING_ENABLED` | `false` | Enable Wyoming TCP server (Docker image and base Compose: `true`) |
| `OS_WYOMING_HOST` | `127.0.0.1` | Wyoming bind host (Docker image and base Compose: `0.0.0.0`) |
| `OS_WYOMING_PORT` | `10400` | Wyoming port |
| `OS_WYOMING_IDLE_TIMEOUT_S` | `120` | Close a Wyoming connection whose next event takes longer than this many seconds to arrive, or that stops reading replies this long; `0` = never |
| `OS_WYOMING_MAX_CONNECTIONS` | `32` | Max open Wyoming connections; at the cap a new connection replaces the one idle longest, and is refused only while all are mid-request; `0` = unlimited |
| `OS_REALTIME_ENABLED` | `true` | Enable `/v1/realtime` |
| `OS_REALTIME_MAX_BUFFER_MB` | `50` | Max realtime input audio buffer per session |
| `OS_REALTIME_IDLE_TIMEOUT_S` | `120` | Close a realtime session idle this long, counted from the end of the last reply |
| `OS_REALTIME_BARGE_IN_MAX_MS` | `1500` | Talking over a reply this long stops it without waiting for the transcript |
| `OS_REALTIME_AUDIO_LEAD_MS` | `1000` | With barge-in on, how far ahead of playback reply audio is sent (`0` sends it as fast as it is made) |
| `OS_ASSISTANT_CONNECTION_LOCKED` | `false` | `true` keeps the AI connection `.env`-only: the connection saved in Settings › AI assistant is ignored, and changing it answers `403` |
| `OS_ASSISTANT_URL` | `""` | The [AI assistant](#ai-assistant)'s server: a base URL ending in `/v1`, to which `/chat/completions` and `/models` are appended; `http` or `https` with a valid host and port (not `192.168.1.300`), with no user name, password, query or fragment; anything else stops startup. Empty = off, unless a URL is saved in Settings |
| `OS_ASSISTANT_API_KEY` | `""` | Sent as `Authorization: Bearer KEY`, only to `OS_ASSISTANT_URL`'s origin; never logged or returned. It is stripped, and must then be printable ASCII of up to 4,096 characters, or startup stops |
| `OS_ASSISTANT_MODEL` | `""` | Model name sent when a preset names none; empty sends none, so the AI server picks |
| `OS_ASSISTANT_VERIFY_TLS` | `true` | Check the AI server's certificate; `false` for a server on your network with a self-signed one. `OS_ASSISTANT_API_KEY` counts as set with this value: turning the check off later in Settings, for an https server, withholds the key until it is given again ([The connection](#the-connection)) |
| `OS_ASSISTANT_TEMPLATE_KWARGS` | `false` | Also send the thinking controls as chat-template variables (`chat_template_kwargs`), for llama.cpp-style servers; OpenAI's own API refuses the field |
| `OS_ASSISTANT_MAX_TOKENS_FIELD` | `max_tokens` | The name the token limit is sent under: `max_tokens` or `max_completion_tokens`; anything else stops startup |
| `OS_ASSISTANT_EXTRA_BODY` | `""` | A JSON object merged into every request, such as `{"top_k": 20}`, written single-quoted in `.env` (`OS_ASSISTANT_EXTRA_BODY='{"top_k": 20}'`); it can't set `messages` or `stream`, nests objects and lists at most 32 levels deep, and holds no `NaN` or `Infinity`. Anything else stops startup |
| `OS_ASSISTANT_STREAM` | `true` | Stream replies: a voice turn asks with `stream: true` and speaks each sentence as it arrives ([Talking to it](#talking-to-it)); `false` waits for the whole reply, for a server that can't stream. Ask AI and book jobs never stream |
| `OS_ASSISTANT_SYSTEM_PROMPT` | `""` | System prompt of a voice turn whose preset leaves it empty; empty = the built-in voice prompt |
| `OS_ASSISTANT_THINKING` | `default` | Presets' default `thinking`: `default` sends nothing, `on` or `off` send `enable_thinking` |
| `OS_ASSISTANT_REASONING_EFFORT` | `""` | Presets' default `reasoning_effort`: `none`, `minimal`, `low`, `medium`, `high` or `xhigh`; empty sends none (`low` suits a server whose template thinks at length) |
| `OS_ASSISTANT_THINKING_BUDGET` | `0` | Presets' default `thinking_budget`, 0–1000000 tokens of thinking, sent as `reasoning_budget_tokens` (for servers that read it); `0` sends none ([Thinking controls](#thinking-controls)) |
| `OS_ASSISTANT_MAX_TOKENS` | `0` | Presets' default token limit, 0–32768; `0` sends no limit. An answer that reaches it is cut off mid-sentence, and the thinking counts towards it |
| `OS_ASSISTANT_TEMPERATURE` | unset | Presets' default `temperature`, 0–2; unset sends none |
| `OS_ASSISTANT_TOP_P` | unset | Presets' default `top_p`, 0–1; unset sends none |
| `OS_ASSISTANT_TIMEOUT_S` | `120` | Seconds to wait for the whole reply to a voice turn, 5–3600 |
| `OS_ASSISTANT_JOB_TIMEOUT_S` | `600` | Seconds to wait for each piece of a book job, 5–3600 |
| `OS_ASSISTANT_HISTORY_TURNS` | `20` | Earlier exchanges sent with each voice turn, 0–100 |
| `OS_ASSISTANT_CHUNK_CHARS` | `12000` | The most book text in one request, 1000–200000 characters |
| `OS_ASSISTANT_SHOW_THINKING` | `true` | Send the AI's thinking to the client as text; it is never spoken |
| `OS_ASSISTANT_SPEAK_ERRORS` | `true` | Speak a sentence when the AI can't answer a voice turn |
| `OS_MODEL_TTL` | `300` | Unload a model idle this many seconds; `0` = never. Pinned defaults are exempt |
| `OS_MAX_LOADED_MODELS` | `0` | Max loaded models besides the pinned defaults (every model when `OS_PIN_DEFAULT_MODELS=false`); the least recently used idle models beyond it are unloaded; `0` = unlimited |
| `OS_PIN_DEFAULT_MODELS` | `true` | Keep `STT_MODEL` and `TTS_MODEL` loaded: idle and LRU eviction skip them, and they do not count toward `OS_MAX_LOADED_MODELS`. An explicit unload, or an explicit load of another model of the same type, still replaces them |
| `OS_GPU_MEMORY_GUARD` | `true` | Check the GPU's free memory before each load onto it, unload idle models that aren't pinned to make room, and refuse a load that still doesn't fit ([GPU memory guard](#gpu-memory-guard)) |
| `OS_GPU_HEADROOM_MB` | `600` | GPU memory, in MiB, kept free on top of a model's size when the guard checks a load |
| `OS_GPU_MAX_BIG_MODELS` | `auto` | Big models (2 GiB or more of GPU memory) loaded at once besides the pinned defaults: `auto` takes it from the card (under 11 GiB: 1, under 22 GiB: 2, else no cap), a number sets it, `0` = no cap |
| `OS_STREAM_CHUNK_MS` | `100` | Streaming chunk window; values below 32 act as 32 |
| `OS_STREAM_ENDPOINTING_MS` | `300` | Silence (ms) that ends an utterance; default for the `/v1/audio/stream` `endpointing` query param |
| `OS_STREAM_MAX_CONNECTIONS` | `10` | Max concurrent `/v1/audio/stream` sessions, and separately max `/v1/realtime` sessions |

### `STT_*` — speech-to-text

| Variable | Default | Description |
|---|---|---|
| `STT_MODEL` | `Systran/faster-whisper-base` | STT model used when a request omits `model` or sends an OpenAI name such as `whisper-1`; loaded on first use, then pinned (`OS_PIN_DEFAULT_MODELS`). The Docker images and Compose set `nvidia/parakeet-tdt-0.6b-v2` ([Parakeet](#parakeet), English only) |
| `STT_DEVICE` | `cpu` | STT inference device |
| `STT_COMPUTE_TYPE` | `int8` | Compute precision of Whisper models; Parakeet runs fp16 on a GPU and int8 on a CPU |
| `STT_MODEL_DIR` | `None` | Optional local model directory; unset or empty uses the Hugging Face cache. The Docker image always sets it to `/home/openspeech/.cache/huggingface/hub` |
| `STT_PRELOAD_MODELS` | `""` | Comma-separated models to load at startup. They load one after another in the background once the server has started, so it answers meanwhile; a request for a model still loading waits for that load instead of starting another. The log says when each starts, finishes or fails (a first Parakeet load on a GPU: about 7 minutes) |
| `STT_TRANSLATE_MODEL` | `""` | Model `/v1/audio/translations` uses for `whisper-1` or no model; empty uses `STT_MODEL`. Parakeet, large-v3-turbo and English-only models can't translate (`400 translation_unsupported`): set `Systran/faster-whisper-medium` or `-large-v3` ([details](#post-v1audiotranslations)). Nothing is downloaded until it is set |
| `STT_PARAKEET_GPU_MEM_LIMIT_MB` | `2304` | Most GPU memory Parakeet's ONNX Runtime memory pool may take, in MB; `0` = no cap. Past the card's memory, Windows pages the other loaded models out instead of failing, and their next request is seconds slower ([Parakeet](#parakeet)) |
| `STT_VAD_ENABLED` | `true` | Enable VAD by default for streaming, and Wyoming's trimming of silence around speech. Both keep 300 ms before the first window over `STT_VAD_THRESHOLD`, as `/v1/realtime`'s default `prefix_padding_ms` does |
| `STT_VAD_THRESHOLD` | `0.5` | VAD speech probability threshold, including streaming. Speech starts at the first 32 ms window at or above it, and keeps the 300 ms before it |
| `STT_VAD_MIN_SPEECH_MS` | `250` | Minimum speech duration Wyoming keeps (streaming ends speech after `OS_STREAM_ENDPOINTING_MS`) |
| `STT_VAD_SILENCE_MS` | `800` | Silence that ends a speech segment in Wyoming's trimming |
| `STT_DIARIZE_ENABLED` | `false` | Enable diarization support (needs the `diarize` extra, and `HF_TOKEN` for pyannote's gated models) |
| `STT_NOISE_REDUCE` | `false` | Enable denoise preprocessing |
| `STT_NORMALIZE` | `true` | Normalize input audio, boosting quiet input by at most 30 dB and only until its peak reaches 0.95, so peaks never clip |

### `TTS_*` — text-to-speech

| Variable | Default | Description |
|---|---|---|
| `TTS_ENABLED` | `true` | Enable TTS endpoints |
| `TTS_MODEL` | `kokoro` | TTS model used when `/v1/audio/speech` omits `model` or sends `tts-1`, `tts-1-hd`, or `gpt-4o-mini-tts`, for OpenAI realtime names, and for Wyoming; also the web UI reading default. Loaded at startup (`TTS_PRELOAD_DEFAULT`) and pinned (`OS_PIN_DEFAULT_MODELS`) |
| `TTS_VOICE` | `af_heart` | Voice used when `/v1/audio/speech` omits `voice` and the model came from `TTS_MODEL`; must be valid for `TTS_MODEL` |
| `TTS_DEVICE` | `None` | TTS device override; falls back to STT device. Base Compose sets `cpu`; GPU override sets `cuda` |
| `TTS_MAX_INPUT_LENGTH` | `4096` | Max input characters for speech, clone, the text of a realtime `response.create`, Wyoming, conversation turns, and Ask AI's spoken reply (an AI reply on `/v1/realtime` is spoken in segments, whatever its length); reported by `/api/tts/capabilities` |
| `TTS_DEFAULT_FORMAT` | `mp3` | Output format used when `/v1/audio/speech` omits `response_format` |
| `TTS_SPEED` | `1.0` | Speed used when `/v1/audio/speech` or a Live Reader `session.update` omits `speed`, on a model with speed control; a fixed-speed model (Pocket-TTS, Chatterbox, Qwen3) speaks at 1.0 instead of refusing the request |
| `TTS_PRELOAD_MODELS` | `""` | Comma-separated TTS models to preload |
| `TTS_PRELOAD_DEFAULT` | `true` | Load `TTS_MODEL` in the background at startup when its provider is installed, downloading its weights on first start; while an isolated worker is still offline, retry every 15 s, up to 20 times. Separate from `TTS_PRELOAD_MODELS` |
| `TTS_EXTERNAL_PROVIDERS` | `""` | JSON map of isolated provider IDs to private worker URLs; a URL with an invalid port is rejected at startup. `docker-compose.voice-models.yml` sets it to each worker's network alias, such as `http://qwen3.voice-workers.internal.:8200`; a bare single-label name that no running container holds is looked up on your network's DNS |
| `TTS_VOICES_CONFIG` | `""` | YAML voice preset path; an entry without a name or voice, or with a value `POST /api/profiles` would refuse, is skipped with a warning that names the field, and is imported once the file is fixed |
| `TTS_CACHE_ENABLED` | `false` | Enable the on-disk cache, which `/v1/audio/speech` and Wyoming share |
| `TTS_CACHE_MAX_MB` | `500` | Cache size budget |
| `TTS_CACHE_DIR` | `/var/lib/open-speech/cache` | Cache directory |
| `TTS_TRIM_SILENCE` | `true` | Trim generated silence |
| `TTS_NORMALIZE_OUTPUT` | `true` | Normalize output loudness; streams and Live Reader are never normalized |
| `TTS_PRONUNCIATION_DICT` | `""` | Pronunciation dictionary path: YAML for `.yaml` or `.yml`, JSON otherwise, mapping text as written to the text to speak (`AWS: A W S`); applies to speech, clone, Live Reader, Reader, realtime replies, Wyoming, and conversations. Every key and value is read as the text written (`NO`, `12:30`, `0.50`, `true`), and an empty value (`""`, or `um:` in YAML) removes the word; a JSON `null` value leaves its entry out, with a warning |
| `TTS_LIVE_ENABLED` | `true` | Enable the incremental Live Reader WebSocket |
| `TTS_LIVE_MAX_CONNECTIONS` | `1` | Concurrent Live Reader sessions per process |
| `TTS_LIVE_IDLE_TIMEOUT_S` | `300` | Close a connection without client events after this many seconds |
| `TTS_LIVE_MAX_BUFFER_CHARS` | `8192` | Text waiting in the session's segmenter-input queue; queued `input_text.commit` events have the same bound |
| `TTS_LIVE_MAX_PENDING_SEGMENTS` | `3` | Synthesizable text segments waiting behind the active segment |
| `TTS_LIVE_MAX_UNACKED_SECONDS` | `15` | Pause outgoing audio after this much unconsumed playback |
| `TTS_LIVE_AUDIO_FRAME_MS` | `50` | Approximate PCM duration in each audio delta |
| `TTS_LIVE_SEGMENT_IDLE_MS` | `250` | Input idle delay for responsive and word modes |
| `TTS_LIVE_MAX_SEGMENT_CHARS` | `200` | Responsive/word mode character cap per synthesis call |
| `TTS_LIVE_MAX_SEGMENT_WORDS` | `20` | Responsive/word mode word cap per synthesis call |
| `TTS_LIVE_SENTENCE_IDLE_MS` | `1500` | Input idle delay before full-sentence mode releases incomplete prose |
| `TTS_LIVE_SENTENCE_MAX_CHARS` | `500` | Full-sentence mode character cap per synthesis call; also for each segment of a spoken AI reply |
| `TTS_LIVE_SENTENCE_MAX_WORDS` | `80` | Full-sentence mode word cap per synthesis call; also for each segment of a spoken AI reply |
| `TTS_LIVE_MAX_PAUSE_S` | `900` | Maximum continuous client playback pause |
| `TTS_LIVE_PLAYBACK_STALL_TIMEOUT_S` | `30` | Close an unpaused client that stops acknowledging audio |
| `TTS_LIVE_SHUTDOWN_TIMEOUT_S` | `5` | Maximum teardown wait before a stuck worker drains in background |
| `READER_ENABLED` | `true` | Enable the Reader tab and /v1/reader API |
| `READER_CACHE_DIR` | `/home/openspeech/data/reader` | Rendered Reader audio (inside the ./data volume) |
| `READER_CACHE_MAX_MB` | `2000` | Least recently played audio is removed above this size |
| `READER_MAX_UPLOAD_MB` | `50` | Largest epub/text/HTML upload |
| `READER_MAX_DOCUMENT_CHARS` | `5000000` | Largest imported document, in characters |
| `READER_LOOKAHEAD_PARTS` | `3` | Audio parts rendered ahead of the listener |
| `READER_PART_MAX_CHARS` | `450` | Characters grouped into one audio part |
| `READER_MEDIA_URL_TTL_S` | `21600` | Lifetime of signed Reader audio links |

## Wyoming

Open Speech can expose STT/TTS over the [Wyoming protocol](https://github.com/rhasspy/wyoming).

```bash
OS_WYOMING_ENABLED=true
# 0.0.0.0 inside Docker; 127.0.0.1 allows only local clients on a source install
OS_WYOMING_HOST=0.0.0.0
OS_WYOMING_PORT=10400
```

The Docker image and base Compose file already enable Wyoming on `0.0.0.0:10400`;
`docker-compose.cpu.yml` and a source install leave it off. In Docker, publish port `10400`. The
Compose files keep the container port at `10400` and ignore `OS_WYOMING_PORT`; to use another host
port, change the host side of `ports:`.

The Wyoming port has no authentication: `OS_API_KEY`, the Host and Origin checks, and rate limits do not
apply to it. These limits do:

- **STT.** Audio must be PCM at 8,000–192,000 Hz, with 1–4 byte samples and at least one channel
  (error code `unsupported_audio`). 8-bit samples are unsigned, as in WAV; wider ones are signed
  little-endian. One transcription buffers at most `OS_MAX_UPLOAD_MB` of audio
  (`audio_too_large`). One event's payload is held to the same limit, and its JSON data section to
  1 MiB + 12 × `TTS_MAX_INPUT_LENGTH` bytes (`event_too_large`).
- **Languages.** A Transcribe language is normalized as on `/v1/audio/transcriptions` (`de-DE` becomes
  `de`; empty or `auto` means auto-detect); an unknown one gets `invalid_language`. The Info (Describe)
  reply lists Whisper's languages for multilingual models and `en` for English-only ones, so
  non-English Home Assistant pipelines can select Open Speech STT. Parakeet, the images' default, lists `en`
  alone, and another language gets `language_unsupported`: for a non-English pipeline set `STT_MODEL` to a
  multilingual Whisper model.
- **TTS.** Synthesis uses `TTS_MODEL` with the requested voice (default `TTS_VOICE`) and answers 16 kHz
  16-bit mono PCM. Text longer than `TTS_MAX_INPUT_LENGTH`, or longer than the model's own limit after
  the pronunciation dictionary, gets `input_too_long`.
- **Errors.** A failed request gets an `error` event. The server then half-closes the connection,
  discards what the client still sends for up to 2 seconds so the client can read the error, and
  closes it.
- **Idle connections.** `OS_WYOMING_IDLE_TIMEOUT_S` (default `120`) closes a connection whose next event
  takes longer than that to arrive, or that stops reading replies that long. Time the server spends
  transcribing or synthesizing does not count.
- **Connection cap.** `OS_WYOMING_MAX_CONNECTIONS` (default `32`) caps open connections. At the cap, a
  new connection closes the one that has waited longest for its next event; it is refused only while
  every open connection is mid-request.

Example Home Assistant config:

```yaml
wyoming:
  - host: "YOUR_OPEN_SPEECH_IP"
    port: 10400
```

## Python SDK Example

This compatibility example uses the external `openai` package; install it separately with
`pip install openai`. The `client` project extra installs dependencies for the built-in
`src.client.OpenSpeechClient` instead: `httpx` and `websockets` 14 or newer, which its WebSocket
helpers need. Its `ssl_verify` option applies to those helpers too: `False` accepts the self-signed
certificate, and a CA bundle path or an `ssl.SSLContext` is used as given. Its `stream_transcribe`
and `async_stream_transcribe` raise `RuntimeError` when the connection ends before the server
confirms the end of the stream.

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

Current release: **v0.13.0**. The running version is available from `GET /health`; run
`python -m pytest -q tests` for the current test result instead of relying on a documentation snapshot.

## License

[MIT](LICENSE) © 2026 Jeremy Windsor
