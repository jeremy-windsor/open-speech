# Configuration

Open Speech is configured with environment variables. Docker Compose reads them from `.env` next to the Compose
files: copy `.env.example` to `.env`, where every setting is listed, commented out, with its default and a
description. A source install loads `.env` with `uv run --env-file .env`
([Install from source](DEVELOPMENT.md#install-from-source)). What the web app saves, such as the AI assistant's
connection and presets, voices and reading presets, is kept in the data folder, not in `.env`.

- [How settings are read](#how-settings-are-read)
- [Writing values in .env](#writing-values-in-env)
- [Docker Compose](#docker-compose): [CPU launch](#cpu-launch), [GPU launch](#gpu-launch),
  [GPU voice workers](#gpu-voice-workers), [Docker image defaults](#docker-image-defaults), [Volumes](#volumes)
- [Compose and voice worker settings](#compose-and-voice-worker-settings)
- [Settings](#settings): [server (`OS_*`)](#server-settings), [speech-to-text (`STT_*`)](#speech-to-text-settings),
  [text-to-speech (`TTS_*`)](#text-to-speech-settings), [Reader (`READER_*`)](#reader-settings),
  [legacy names](#legacy-names)

## How settings are read

Defaults come from `src/config.py`; the tables under [Settings](#settings) give them. The Docker image and the
Compose files set several of them differently ([Docker image defaults](#docker-image-defaults),
[CPU launch](#cpu-launch)), and the GPU override sets the devices to CUDA. A source install reads only the process
environment, so use `uv run --env-file .env` or export the variables yourself. An empty value counts as unset, so
the default applies. A value that isn't valid stops startup with an error that names the setting and the reason,
never the value, because it may be a secret.

## Writing values in .env

Set a value by removing the `# ` in front of its `NAME=value` line. Keep each comment on its own line and do not
quote values, except as below: `docker run --env-file` takes everything after `=` as the value, a trailing
`# comment`, blanks and quotes included.

Each program that reads `.env` has rules of its own:

- **Docker Compose** expands a `$` in a value that isn't single-quoted: `$NAME` and `${NAME}` become that
  variable, blank when it is unset (Compose then warns `The "NAME" variable is not set`). So
  `OS_API_KEY=$ecretKey123` reaches the server empty, which turns authentication off, and `OS_API_KEY=Xk9$mP2q`
  arrives as `Xk9`; `OS_ASSISTANT_API_KEY` and `HF_TOKEN` are cut the same way. Under Compose alone, `$$` also
  writes a `$` (`OS_API_KEY=Xk9$$mP2q`).
- **`uv run --env-file`** (a source install) expands a `$` outside single quotes too, and `$$` is no escape there:
  `Xk9$$mP2q` arrives as `Xk9`. It drops each double quote, so `{"top_k":20}` arrives as `{top_k:20}`, which is
  not JSON. A blank or a backslash in a value that isn't quoted makes it stop reading `.env` at that line, with only
  a warning: that setting and every one below it stay unset. An apostrophe (`'`) in a value that isn't quoted
  starts a quote that runs to the next apostrophe, even lines below. One apostrophe makes uv stop reading `.env`
  at that line, so `OS_API_KEY=Xk9'mP2q` leaves the key and `OS_AUTH_REQUIRED` below it unset, with
  authentication off; two are dropped (`it's-Bob's` arrives as `its-Bobs`).
- **`docker run --env-file`** expands nothing and keeps a `$$` and the quotes as written.

So single-quote a value that holds a `$`, a double quote, a blank or a backslash, as in `OS_API_KEY='Xk9$mP2q'`,
`OS_ASSISTANT_EXTRA_BODY='{"top_k": 20}'` or `TTS_PRONUNCIATION_DICT='C:\Users\me\dict.yaml'`: Compose and uv both
read it as written. `docker run --env-file` would keep the quotes, so one `.env` can't give such a value to it and
to the other two. Single quotes can't hold an apostrophe: for uv, write it as `\'` (`OS_API_KEY=Xk9\'mP2q`) or
export the value in the shell. Compose reads `Xk9'mP2q` as written but keeps the backslash of `\'`, so a `.env`
that both read needs a key without an apostrophe. A key of letters, digits, `-` and `_` avoids all of this:
`python -c "import secrets; print(secrets.token_urlsafe(32))"` prints one.

## Docker Compose

Compose reads `.env` next to the Compose files and passes most other `OS_*` settings through empty when `.env`
does not set them, so the source default applies. A setting that the Compose file's `environment:` list does not
name, such as `OS_PIN_DEFAULT_MODELS`, `OS_WYOMING_MAX_CONNECTIONS`, `OS_BATCH_*`, `OS_HISTORY_*`, or
`TTS_PRELOAD_DEFAULT`, reaches the container only from `.env`, not from the shell that runs `docker compose`.

Legacy names in `.env`, such as `STT_API_KEY`, `STT_CORS_ORIGINS`, `STT_DEFAULT_MODEL`, or `TTS_DEFAULT_VOICE`,
work under the Compose files and log a deprecation warning. The Compose files ignore `OS_HOST`, `OS_PORT`,
`OS_WYOMING_PORT`, `STT_HOST` and `STT_PORT` in `.env`: the container listens on `0.0.0.0:8100` and Wyoming on
`10400`, matching `ports:`. To publish other host ports, change the host side of `ports:`, for example
`"9000:8100"`.

Name the Compose files and profiles once in `.env`, and every `docker compose` command (`up`, `down`, `logs`)
uses them. Separate the files with `:` on Linux and macOS and with `;` on Windows:

```bash
COMPOSE_FILE=docker-compose.yml:docker-compose.gpu.yml:docker-compose.voice-models.yml
COMPOSE_PROFILES=qwen3
```

Without `COMPOSE_FILE`, give the same `-f` files to every command, as below.

### CPU launch

Plain Compose is CPU-safe and does not request NVIDIA GPU passthrough:

```bash
docker compose up -d
```

It defaults to `STT_DEVICE=cpu`, `STT_COMPUTE_TYPE=int8`, and `TTS_DEVICE=cpu`. The base Compose launch also
selects the [Parakeet](MODELS.md#parakeet) STT model (English only; int8 on a CPU, fp16 with the GPU override; set
`STT_MODEL` in `.env` to a Whisper model for other languages, and `STT_TRANSLATE_MODEL` for translations), enables
Wyoming on `0.0.0.0`, caps the models loaded besides the pinned defaults at two (`OS_MAX_LOADED_MODELS=2`), and uses
a 2,000 ms streaming STT chunk window. These are Compose launch defaults, not the source defaults listed under
[Settings](#settings).

`docker-compose.cpu.yml` extends the base file with the same CPU image, ports, and volumes, but changes four
settings: `STT_MODEL=nvidia/parakeet-tdt-0.6b-v2` (the base file's model, set again), Wyoming off
(`OS_WYOMING_ENABLED=false`, host `127.0.0.1`), and no cap on loaded models (`OS_MAX_LOADED_MODELS=0`):

```bash
docker compose -f docker-compose.cpu.yml up -d
```

### GPU launch

A GPU launch layers the GPU override on the base file; plain `docker compose up -d` is not a GPU launch unless
`COMPOSE_FILE` in `.env` names both files.

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --force-recreate
```

The GPU override requests NVIDIA passthrough with `gpus: all`, includes the NVIDIA device reservation block, and
sets `STT_DEVICE=cuda`, `STT_COMPUTE_TYPE=float16`, and `TTS_DEVICE=cuda`. It needs an NVIDIA GPU and the NVIDIA
Container Toolkit (Linux) or Docker Desktop with WSL2 GPU support (Windows).

### GPU voice workers

For voice cloning and voice design in the web app (Voices), add one of the optional GPU voice workers: Chatterbox,
Qwen3, Fish Audio S2 or Breeze TTS 2. They are off by default. `docker-compose.voice-models.yml` gives each worker a
Compose profile of its own, named after it (`chatterbox`, `qwen3`, `fish`, `breeze`), so layering the file alone
starts and builds none of them. Turn one on by naming its profile:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  -f docker-compose.voice-models.yml --profile qwen3 up -d --build
```

Use `--profile chatterbox`, `--profile fish` or `--profile breeze` for the others; for two, repeat the flag
(`--profile qwen3 --profile chatterbox`), and `--profile "*"` turns on all four. `COMPOSE_PROFILES=qwen3,chatterbox`,
in the shell or in `.env`, does the same as the flags, and also covers later commands such as `down`. The core still
lists every worker's models: one whose worker isn't turned on reads "Worker offline" in Models, and the core runs
without it. The core reaches each worker by a network alias only that worker holds, such as
`http://chatterbox.voice-workers.internal.:8210`, so a device on your network called `chatterbox` can't stand in for
a worker that is off (see [Voice provider deployment](TTS-BACKENDS.md#voice-provider-deployment)).

A worker container that an earlier `up` started keeps running, on the image it started from, until it is stopped:
`up` leaves alone a worker whose profile isn't named, and `down` without its profile doesn't stop it. This stops
every worker, and the core with them (the next `up` starts the core again):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  -f docker-compose.voice-models.yml --profile "*" down
```

Stop one worker you no longer want with the same `-f` files and `stop chatterbox` (or `rm -s -f chatterbox` to
remove its container). A `--profile` flag replaces `COMPOSE_PROFILES` rather than adding to it: with
`COMPOSE_PROFILES=qwen3` in `.env`, `--profile chatterbox up -d` starts Chatterbox and leaves Qwen3 running as it
was, and a later `down` (which reads `.env`) leaves Chatterbox running. To audition another worker, stop the one
before (`stop qwen3`) or change `COMPOSE_PROFILES` itself.

Only Qwen3 Custom Voice (`qwen3/0.6b-custom-voice`, `qwen3/1.7b-custom-voice`) has built-in voices; Chatterbox,
Qwen3 Base, Fish Audio S2 and Breeze TTS 2 clone from a recording, and Breeze TTS 2 and Qwen3 VoiceDesign
(`qwen3/1.7b-voice-design`) make a voice from a description. What each model needs is in
[Cloning models](CLONING-MODELS.md), and its memory and first load in
[GPU voice worker models](MODELS.md#gpu-voice-worker-models). Rebuild the worker images with `up -d --build` after
updating the checkout.

### Docker image defaults

The Docker image sets its own environment defaults, which differ from `src/config.py`. A plain `docker run` uses
these unless you pass `-e` overrides (the Compose files set most of them again):

| Setting | Image default | Source default |
|---|---|---|
| `OS_WYOMING_ENABLED` | `true` | `false` |
| `OS_WYOMING_HOST` | `0.0.0.0` | `127.0.0.1` |
| `OS_MAX_LOADED_MODELS` | `2` (besides the pinned defaults) | `0` (unlimited) |
| `STT_MODEL` | `nvidia/parakeet-tdt-0.6b-v2` (int8 on `:cpu`, fp16 on `:latest`, CUDA) | `Systran/faster-whisper-base` |
| `STT_DEVICE` / `STT_COMPUTE_TYPE` / `TTS_DEVICE` | `cpu`/`int8`/`cpu` (`:cpu`), `cuda`/`float16`/`cuda` (`:latest`) | `cpu`/`int8`/unset |

The image also sets `TTS_MODEL=kokoro` and `TTS_VOICE=af_heart`, the source defaults. Because the image sets
`STT_MODEL`, `TTS_MODEL`, `TTS_VOICE`, and `OS_MAX_LOADED_MODELS` itself, a plain `docker run --env-file .env`
ignores their legacy names (`STT_DEFAULT_MODEL`, `TTS_DEFAULT_MODEL`, `TTS_DEFAULT_VOICE`, `STT_MAX_LOADED_MODELS`);
use the new names there.

The container entrypoint always sets `HF_HOME=/home/openspeech/.cache/huggingface` and
`STT_MODEL_DIR=/home/openspeech/.cache/huggingface/hub`, so values you pass for those two are replaced. Wyoming
listens on port `10400`; add `-p 10400:10400` to `docker run` if you use it.

### Volumes

Model caches live under `/home/openspeech/.cache/huggingface` inside the container; persist that path so models
aren't downloaded again. The Compose files persist:

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

## Compose and voice worker settings

These go in `.env` next to the Compose files. Docker Compose, the Compose files and the provider workers in
`providers/*/app.py` read them; they are not Open Speech server settings.

| Variable | Default | Description |
|---|---|---|
| `COMPOSE_FILE` | `docker-compose.yml` | The Compose files every `docker compose` command uses, separated by `:` (`;` on Windows), such as `docker-compose.yml:docker-compose.gpu.yml` (read by Docker Compose, the same as `-f`) |
| `OPEN_SPEECH_IMAGE` | `jwindsor1/open-speech:cpu` (`:latest` with the GPU override) | Core harness image tag |
| `OPEN_SPEECH_REVISION` | `unknown` | Git revision recorded by local image builds |
| `COMPOSE_PROFILES` | `""` (no worker) | Which optional GPU voice workers `docker-compose.voice-models.yml` starts, comma-separated: `chatterbox`, `qwen3`, `fish`, `breeze` (read by Docker Compose, the same as `--profile`; each worker is off by default) |
| `OPEN_SPEECH_CHATTERBOX_IMAGE` | `open-speech-chatterbox:0.2.1` | Chatterbox worker image tag |
| `OPEN_SPEECH_QWEN3_IMAGE` | `open-speech-qwen3:0.3.0` | Qwen3 worker image tag |
| `OPEN_SPEECH_FISH_IMAGE` | `open-speech-fish:0.1.0` | Fish Audio S2 worker image tag |
| `OPEN_SPEECH_BREEZE_IMAGE` | `open-speech-breeze:0.2.0` | Breeze TTS 2 worker image tag |
| `HF_TOKEN` | `""` | Optional Hugging Face token for gated downloads (core and workers) |
| `QWEN3_ENABLE_BASE` | `false` in the worker; `true` in the voice-models Compose file | Show the `qwen3/0.6b-base` and `qwen3/1.7b-base` cloning models |
| `QWEN3_DTYPE` | `auto` | `auto`, `float32`, `float16`, or `bfloat16`; `auto` is bfloat16 from sm_80, and before it float32 for the 0.6B models and float16 for the 1.7B |
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

## Settings

The server's own settings, with their source defaults.

### Server settings

`OS_*`: the server, security, storage, streaming, Wyoming, realtime, the AI assistant and model residency.

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
| `OS_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING` or `ERROR` (any case): the server's log level, which uvicorn's access log follows. `DEBUG` adds diagnostic lines, among them the text of `/v1/audio/stream` transcripts; httpx, httpcore, urllib3, websockets and other chatty libraries stay at `INFO`. Any other value stops startup ([The server log](TROUBLESHOOTING.md#the-server-log)) |
| `OS_TRUST_PROXY` | `false` | Take the client address from `X-Forwarded-For` |
| `OS_TRUSTED_PROXY_HOPS` | `1` | With `OS_TRUST_PROXY=true`: proxies that append to `X-Forwarded-For`; the client is the entry this many places from the right |
| `OS_MAX_UPLOAD_MB` | `100` | Max upload and request body size (see [Upload and body limits](API.md#upload-and-body-limits)); also caps base64 `reference_audio` and Wyoming audio; checked-in provider Compose files also use it for voice references |
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
| `OS_ASSISTANT_URL` | `""` | The [AI assistant](AI-ASSISTANT.md)'s server: a base URL ending in `/v1`, to which `/chat/completions` and `/models` are appended; `http` or `https` with a valid host and port (not `192.168.1.300`), with no user name, password, query or fragment; anything else stops startup. Empty = off, unless a URL is saved in Settings |
| `OS_ASSISTANT_API_KEY` | `""` | Sent as `Authorization: Bearer KEY`, only to `OS_ASSISTANT_URL`'s origin; never logged or returned. It is stripped, and must then be printable ASCII of up to 4,096 characters, or startup stops |
| `OS_ASSISTANT_MODEL` | `""` | Model name sent when a preset names none; empty sends none, so the AI server picks |
| `OS_ASSISTANT_VERIFY_TLS` | `true` | Check the AI server's certificate; `false` for a server on your network with a self-signed one. `OS_ASSISTANT_API_KEY` counts as set with this value: turning the check off later in Settings, for an https server, withholds the key until it is given again ([The connection](AI-ASSISTANT.md#the-connection)) |
| `OS_ASSISTANT_TEMPLATE_KWARGS` | `false` | Also send the thinking controls as chat-template variables (`chat_template_kwargs`), for llama.cpp-style servers; OpenAI's own API refuses the field |
| `OS_ASSISTANT_MAX_TOKENS_FIELD` | `max_tokens` | The name the token limit is sent under: `max_tokens` or `max_completion_tokens`; anything else stops startup |
| `OS_ASSISTANT_EXTRA_BODY` | `""` | A JSON object merged into every request, such as `{"top_k": 20}`, written single-quoted in `.env` (`OS_ASSISTANT_EXTRA_BODY='{"top_k": 20}'`); it can't set `messages` or `stream`, nests objects and lists at most 32 levels deep, and holds no `NaN` or `Infinity`. Anything else stops startup |
| `OS_ASSISTANT_STREAM` | `true` | Stream replies: a voice turn asks with `stream: true` and speaks each sentence as it arrives ([Talking to it](AI-ASSISTANT.md#talking-to-it)); `false` waits for the whole reply, for a server that can't stream. Ask AI and book jobs never stream |
| `OS_ASSISTANT_SYSTEM_PROMPT` | `""` | System prompt of a voice turn whose preset leaves it empty; empty = the built-in voice prompt |
| `OS_ASSISTANT_THINKING` | `default` | Presets' default `thinking`: `default` sends nothing, `on` or `off` send `enable_thinking` |
| `OS_ASSISTANT_REASONING_EFFORT` | `""` | Presets' default `reasoning_effort`: `none`, `minimal`, `low`, `medium`, `high` or `xhigh`; empty sends none (`low` suits a server whose template thinks at length) |
| `OS_ASSISTANT_THINKING_BUDGET` | `0` | Presets' default `thinking_budget`, 0–1000000 tokens of thinking, sent as `reasoning_budget_tokens` (for servers that read it); `0` sends none ([Thinking controls](AI-ASSISTANT.md#thinking-controls)) |
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
| `OS_GPU_MEMORY_GUARD` | `true` | Check the GPU's free memory before each load onto it, unload idle models that aren't pinned to make room, and refuse a load that still doesn't fit ([GPU memory guard](MODELS.md#gpu-memory-guard)) |
| `OS_GPU_HEADROOM_MB` | `600` | GPU memory, in MiB, kept free on top of a model's size when the guard checks a load |
| `OS_GPU_MAX_BIG_MODELS` | `auto` | Big models (2 GiB or more of GPU memory) loaded at once besides the pinned defaults: `auto` takes it from the card (under 11 GiB: 1, under 22 GiB: 2, else no cap), a number sets it, `0` = no cap |
| `OS_STREAM_CHUNK_MS` | `100` | Streaming chunk window; values below 32 act as 32 |
| `OS_STREAM_ENDPOINTING_MS` | `300` | Silence (ms) that ends an utterance; default for the `/v1/audio/stream` `endpointing` query param |
| `OS_STREAM_MAX_CONNECTIONS` | `10` | Max concurrent `/v1/audio/stream` sessions, and separately max `/v1/realtime` sessions |

### Speech-to-text settings

| Variable | Default | Description |
|---|---|---|
| `STT_MODEL` | `Systran/faster-whisper-base` | STT model used when a request omits `model` or sends an OpenAI name such as `whisper-1`; loaded on first use, then pinned (`OS_PIN_DEFAULT_MODELS`). The Docker images and Compose set `nvidia/parakeet-tdt-0.6b-v2` ([Parakeet](MODELS.md#parakeet), English only) |
| `STT_DEVICE` | `cpu` | STT inference device |
| `STT_COMPUTE_TYPE` | `int8` | Compute precision of Whisper models; Parakeet runs fp16 on a GPU and int8 on a CPU |
| `STT_MODEL_DIR` | `None` | Optional local model directory; unset or empty uses the Hugging Face cache. The Docker image always sets it to `/home/openspeech/.cache/huggingface/hub` |
| `STT_PRELOAD_MODELS` | `""` | Comma-separated models to load at startup. They load one after another in the background once the server has started, so it answers meanwhile; a request for a model still loading waits for that load instead of starting another. The log says when each starts, finishes or fails (a first Parakeet load on a GPU: about 7 minutes) |
| `STT_TRANSLATE_MODEL` | `""` | Model `/v1/audio/translations` uses for `whisper-1` or no model; empty uses `STT_MODEL`. Parakeet, large-v3-turbo and English-only models can't translate (`400 translation_unsupported`): set `Systran/faster-whisper-medium` or `-large-v3` ([details](API.md#post-v1audiotranslations)). Nothing is downloaded until it is set |
| `STT_PARAKEET_GPU_MEM_LIMIT_MB` | `2304` | Most GPU memory Parakeet's ONNX Runtime memory pool may take, in MB; `0` = no cap. Past the card's memory, Windows pages the other loaded models out instead of failing, and their next request is seconds slower ([Parakeet](MODELS.md#parakeet)) |
| `STT_VAD_ENABLED` | `true` | Enable VAD by default for streaming, and Wyoming's trimming of silence around speech. Both keep 300 ms before the first window over `STT_VAD_THRESHOLD`, as `/v1/realtime`'s default `prefix_padding_ms` does |
| `STT_VAD_THRESHOLD` | `0.5` | VAD speech probability threshold, including streaming. Speech starts at the first 32 ms window at or above it, and keeps the 300 ms before it |
| `STT_VAD_MIN_SPEECH_MS` | `250` | Minimum speech duration Wyoming keeps (streaming ends speech after `OS_STREAM_ENDPOINTING_MS`) |
| `STT_VAD_SILENCE_MS` | `800` | Silence that ends a speech segment in Wyoming's trimming |
| `STT_DIARIZE_ENABLED` | `false` | Enable diarization support (needs the `diarize` extra, and `HF_TOKEN` for pyannote's gated models) |
| `STT_NOISE_REDUCE` | `false` | Enable denoise preprocessing |
| `STT_NORMALIZE` | `true` | Normalize input audio, boosting quiet input by at most 30 dB and only until its peak reaches 0.95, so peaks never clip |

### Text-to-speech settings

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


### Reader settings

| Variable | Default | Description |
|---|---|---|
| `READER_ENABLED` | `true` | Enable the Reader tab and /v1/reader API |
| `READER_CACHE_DIR` | `/home/openspeech/data/reader` | Rendered Reader audio (inside the ./data volume) |
| `READER_CACHE_MAX_MB` | `2000` | Least recently played audio is removed above this size |
| `READER_MAX_UPLOAD_MB` | `50` | Largest epub/text/HTML upload |
| `READER_MAX_DOCUMENT_CHARS` | `5000000` | Largest imported document, in characters |
| `READER_LOOKAHEAD_PARTS` | `3` | Audio parts rendered ahead of the listener |
| `READER_PART_MAX_CHARS` | `450` | Characters grouped into one audio part |
| `READER_MEDIA_URL_TTL_S` | `21600` | Lifetime of signed Reader audio links |

### Legacy names

Older names work and log a deprecation warning at startup; each applies when its new name is unset or empty:
`STT_PORT`, `STT_HOST`, `STT_API_KEY`, `STT_CORS_ORIGINS`, `STT_TRUST_PROXY`, `STT_MAX_UPLOAD_MB`, `STT_RATE_LIMIT`,
`STT_RATE_LIMIT_BURST`, `STT_SSL_ENABLED`, `STT_SSL_CERTFILE`, `STT_SSL_KEYFILE`, `STT_MODEL_TTL`,
`STT_MAX_LOADED_MODELS`, `STT_STREAM_CHUNK_MS`, `STT_STREAM_ENDPOINTING_MS`, and `STT_STREAM_MAX_CONNECTIONS` (now
`OS_*`), plus `STT_DEFAULT_MODEL`, `TTS_DEFAULT_MODEL`, `TTS_DEFAULT_VOICE`, and `TTS_DEFAULT_SPEED` (now
`STT_MODEL`, `TTS_MODEL`, `TTS_VOICE`, and `TTS_SPEED`). `OS_HISTORY_RETAIN_AUDIO` and `OS_HISTORY_MAX_MB` have no
effect, because history stores no audio; setting either logs a startup warning.
