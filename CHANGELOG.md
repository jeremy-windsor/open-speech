# Changelog

All notable changes to Open Speech are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/).

## [0.17.0] - 2026-11-16

### Added
- Home: the app opens on a Home screen with shortcuts, the document to pick up and recent History; Settings › Appearance › Open on chooses Home or the last view.
- Motion: view entrances, a lively Home header, playing indicators in the mini-player, Speak and Talk (off with Reduce motion).

### Changed
- Phones: a compact layout (smaller buttons, bars, titles and spacing; text boxes start at three lines and grow).
- A refreshed look: accent gradient, softer control borders and shadows.

### Fixed
- iPhone: no double-tap zoom on controls; larger tap areas for Talk's Thinking and the narrow-screen logo; Talk's log no longer traps a swipe.

## [0.16.4] - 2026-11-16

### Fixed
- Talk with Interrupt by talking off: the microphone listens again after a turn with no words, a failed transcription, or a typed question that couldn't be sent.
- Talk: a session whose assistant was deleted elsewhere can still change its settings, and a renamed assistant keeps the session even when another one takes its old name.

## [0.16.3] - 2026-11-16

### Fixed
- Talk: a session keeps working after its assistant is deleted (it moves to .env) or renamed.
- Talk: a session whose assistant was deleted can still change its preset and instructions.
- Switching or deleting assistants quickly no longer leaves the wrong one in use.

## [0.16.2] - 2026-11-15

### Fixed
- Designed voices: Save named voice in Speak no longer replaces a designed voice's sample with a stock voice, and the server only accepts a designed voice's own sample for its versions.
- Deleting a designed voice that a preset still uses is refused while Recordings is full.
- History takes made with a deleted designed voice can be generated again from the recording it became.

## [0.16.1] - 2026-11-15

### Fixed
- Talk: with Interrupt by talking off, the microphone pauses while the AI answers and listens again when it's done.

## [0.16.0] - 2026-11-15

### Added
- Saved AI assistants: keep several AI servers in Settings › AI assistant and choose the one in use.
- Talk has an assistant switcher; Ask AI shows which assistant answers.
- Presets can name an assistant; `/v1/realtime` `session.update` takes `assistant`.

### Changed
- On narrow screens the tab bar is Reader, Speak, Transcribe and Talk; Voices is under More.

## [0.15.0] - 2026-11-15

### Added
- Designed voices are named voices: Voices › Design a voice saves one, its card shows the description and sample,
  and Edit description makes it again. Cloning models speak from its sample; it isn't listed in Recordings.
- Speak: a take designed from the voice description has Save as voice….
- Breeze TTS 2 options Guidance and Variation in Speak, presets and Design a voice (image `open-speech-breeze:0.2.0`).
- Clone a voice also saves a named voice.
- The Reader's voice sheet links to Design a voice for a model with no voices.

### Changed
- Presets: Set as default is now Default in Speak.

### Fixed
- Speak: Save named voice… no longer fails while designing a voice.

## [0.14.0] - 2026-11-14

### Added
- Qwen3-TTS 1.7B in the Qwen3 worker (image `open-speech-qwen3:0.3.0`): `qwen3/1.7b-custom-voice` (the nine
  voices, with an optional style in `instructions`), `qwen3/1.7b-voice-design` (a voice from a description, in Speak
  and Voices › Design a voice; the Reader, Studio and Talk leave it out) and `qwen3/1.7b-base` (cloning). Best on
  newer or bigger GPUs: about 5.2 GB of GPU memory each, and about a third of real time on an RTX 2070 SUPER.
- Models › Download fetches a Qwen3 model without loading it, and a first load downloads before making GPU room.

### Changed
- `QWEN3_DTYPE=auto` runs the 1.7B models in float16 on GPUs older than sm_80; the 0.6B models stay in float32.

## [0.13.0] - 2026-11-14

### Added
- Breeze TTS 2 (`breeze/tts-2`), an optional GPU worker (`--profile breeze`, image `open-speech-breeze:0.1.0`):
  English and Chinese voices designed from a description, or cloned from a recording and its transcript, through
  the Breeze-TTS-2.cpp port with Q8_0 weights (about 3.6 GB download, 4.4 GB of GPU memory). Inline cues such as
  `(laugh)` are part of the text.
- Voices › Design a voice: describe a voice, listen to a sample, and save it as a recording that any cloning model
  can speak in. `POST /api/voices/design` makes the sample.
- `/v1/audio/voices` lists, for a cloning model, the named voices that have a version for it (`voice:<uuid>`).

### Fixed
- Loading a TTS model (Models › Load or `POST /api/models/{id}/load`) no longer unloads the pinned `TTS_MODEL`.

## [0.12.0] - 2026-11-14

### Added
- Supertonic 3 (`supertonic`), built into both images: ten voices (`M1`–`M5`, `F1`–`F5`), blends such as
  `M1(0.6)+F2(0.4)`, streaming per sentence; it runs on the GPU when ONNX Runtime has CUDA, else on the CPU.
  About 383 MB downloaded on first use.
- `steps` on `/v1/audio/speech`: Supertonic's quality steps (1–32, default 8); models without it ignore it. Speak
  shows the control only for a model that has it, and presets keep it.
- English numbers, money, times, dates and titles are spelled out before Supertonic reads them.

### Changed
- Voice pickers head each model's voices with its name, and voices shown elsewhere carry it ("Kokoro · Heart",
  "Supertonic · M1", "Piper · Amy (US)"). `/v1/audio/voices` rows name their `model`.

## [0.11.0] - 2026-11-14

### Upgrade notes
- A model load that doesn't fit in free GPU memory is refused with `503 insufficient_gpu_memory` (after idle,
  unpinned models are unloaded). Set `OS_GPU_MEMORY_GUARD=false` to turn this off.

### Added
- GPU memory guard: loads are checked against free GPU memory; idle unpinned models are unloaded to make room, and
  the pinned `STT_MODEL` and `TTS_MODEL` stay loaded, with room kept for them to grow as they run (loading a big
  speech model yourself may unload the idle `STT_MODEL` until the next transcription). Settings:
  `OS_GPU_MEMORY_GUARD`, `OS_GPU_HEADROOM_MB`, `OS_GPU_MAX_BIG_MODELS`.
- Workers report their GPU memory (`gpu_memory_mb` in the manifest, `gpu_memory` in `/health`).

## [0.10.2] - 2026-11-14

### Fixed
- Open WebUI and other OpenAI-style clients showed blank TTS model names: `GET /v1/audio/models` rows had no `id`.
  Each row now has `id` and a readable `name` (such as `Kokoro` or `Piper Amy (US, medium)`), and the list holds
  every TTS model the server can use, with its `state` (`provider_installed`, `downloaded`, `loaded`), instead of
  only the loaded ones and `TTS_MODEL`. The existing fields (`model`, `backend`, `device`, `status`, `loaded_at`,
  `last_used_at`) stay; `?loaded=true` lists only the loaded models.

## [0.10.1] - 2026-11-14

### Fixed
- Parakeet's one-time fp16 conversion on a GPU runs in a process of its own. When it runs out of memory, only that
  process stops: the load fails with a message saying the conversion needs about 6 GB of free RAM (or
  `STT_DEVICE=cpu`), and the server keeps running instead of being restarted. A warning is logged first when less
  than that is free.
- A conversion stopped together with its server (out of memory, `docker stop`) no longer leaves its temporary
  folder and its 2.4 GB copy of the encoder in the model cache: the next load removes them, also after a restart
  that gives the server the same process ID, and also once the converted folder exists.
- Two servers converting on one model cache at once: the one that finishes second uses the other's folder instead
  of failing. The converted files are flushed to disk before they are put in place, and an empty file left by a
  hard stop no longer counts as converted.
- The README's CUDA source install now runs `uv sync --frozen --extra cuda --extra all --no-install-package
  onnxruntime`, and `uv run --no-sync` afterwards: the CPU build of ONNX Runtime, which faster-whisper and Piper
  require, and `onnxruntime-gpu` install the same files and overwrote each other.
- The documentation gives the first GPU load of Parakeet as about 7 minutes and up to about 6 GB of RAM (it said
  a minute and 4 GB), its cache as about 3.7 GB on a GPU, and Parakeet's streaming limits and its answer to a
  stream in another language.
- `STT_PRELOAD_MODELS` no longer holds up startup: its models load one after another in the background once the
  server has started, and the log says when each starts, finishes or fails. A first GPU start of Parakeet (about
  7 minutes) answered nothing until the load finished, and Docker's healthcheck could mark the container unhealthy
  meanwhile. HTTP and Wyoming requests are served while it runs; one for the model being preloaded waits for that
  same download and load instead of starting another. Shutdown no longer waits for it.
- Parakeet on a GPU: 21–25 s clips could fail with a 500 once ONNX Runtime's memory pool had fragmented. The pool
  now gives memory back after each run, and a piece that still doesn't fit is split at its quietest point and
  decoded in halves instead of failing the request.
- Parakeet's live transcription ends an utterance at 24 s (was 25 s), so its final text is decoded in one piece
  and matches the words already confirmed.
- Long audio is cut for Parakeet without cutting words in half, and without tiny leftover pieces, from which it
  made up words ("Yeah.").
- Very short audio returns empty text with Parakeet instead of an error or `<unk>`.
- Parakeet's subtitle cues and word end times no longer stretch through the pause after a word.
- Scoring speech with Silero VAD before a Parakeet decode no longer holds the GPU lock, so other decodes go on
  meanwhile.
- A failed Silero VAD download is not tried again on every request: for 5 minutes long audio for Parakeet is cut
  into equal pieces at once instead of each request waiting for another download.
- The GPU image started with no GPU visible runs Parakeet int8 on the CPU and reports `cpu` as its device, also
  when ONNX Runtime silently started its session on the CPU.
- Models shows Parakeet as Downloaded only when this device's files are in the cache (fp16 on a GPU, int8 on a
  CPU), and gives its size for this device: about 3.7 GB on a GPU, 0.7 GB on a CPU (it said 1.3 GB).
- A transcription with Parakeet's files missing while Hugging Face can't be reached answers
  `503 model_files_unavailable` with a message saying so. Error messages sent to clients (HTTP, live
  transcription, `/v1/realtime`, Wyoming, model loads) no longer name the server's folders; the log still does.
- Deleting a model's files while its download runs (a first use or a preload) is refused with
  `409 model_fetching`, which Models shows on the row, instead of answering that nothing was found and leaving
  the download to write them back.
- Models › Load of a speech-to-text model keeps the pinned `STT_MODEL` loaded, as Transcribe does, and the row says
  which models it unloads and which stay loaded. `GET /api/models` rows carry `pinned`.
- A `/v1/realtime` session asking Parakeet for another language gets one `language_unsupported` error event and
  is transcribed in English, instead of failing every turn.
- A legacy `POST /api/ps/{model}` whose load fails loads back the STT models it unloaded first, the pinned default
  among them.
- Transcribe names each speech-to-text model with its family ("Whisper base", "Parakeet TDT 0.6B v2").

## [0.10.0] - 2026-11-13

### Upgrade notes
- **The default STT model is now Parakeet** (`nvidia/parakeet-tdt-0.6b-v2`) in both Docker images and the Compose
  files, int8 on a CPU and fp16 on a GPU. It transcribes English only: a request in another language gets
  `400 language_unsupported`, and Wyoming lists English alone. For other languages set `STT_MODEL` to a Whisper
  model (such as `deepdml/faster-whisper-large-v3-turbo-ct2`, the previous GPU default, or
  `Systran/faster-whisper-base`, the previous CPU default), or name one per request. `whisper-1` translations need
  `STT_TRANSLATE_MODEL` set to a Whisper model that translates (`Systran/faster-whisper-medium` or `-large-v3`).
- On a GPU the first load downloads the 2.5 GB fp32 export and converts it to fp16 once (about 7 minutes in all,
  and up to about 6 GB of RAM during the conversion). Press Download on it in Models once the server is up.
  `STT_PRELOAD_MODELS=nvidia/parakeet-tdt-0.6b-v2` also works, but the server answers nothing until that load
  finishes (from 0.10.1 the preload runs in the background after the server starts).

### Added
- **NVIDIA Parakeet TDT 0.6B v2 speech-to-text** (`nvidia/parakeet-tdt-0.6b-v2`) through onnx-asr: English with
  punctuation, several times faster than Whisper large-v3-turbo on a GPU at a lower word error rate, and about as
  fast as Whisper base on a CPU. Long audio is cut with Silero VAD into pieces of at most 20 s; `verbose_json`,
  `srt` and `vtt` carry sentence segments and word times. `prompt` and `temperature` are accepted and ignored.
- `STT_PARAKEET_GPU_MEM_LIMIT_MB` (default `2304`) caps Parakeet's GPU memory pool, so it stays loaded next to
  Kokoro on an 8 GB card.

### Changed
- The Docker images and Compose files set `STT_MODEL=nvidia/parakeet-tdt-0.6b-v2` (was
  `deepdml/faster-whisper-large-v3-turbo-ct2` on the GPU image and the base Compose file, and
  `Systran/faster-whisper-base` on the CPU image and `docker-compose.cpu.yml`). The source default is unchanged.
- ONNX Runtime 1.24.4 (was 1.24.1); the CUDA image installs `onnxruntime-gpu` in its place.
- Streaming STT ends a Parakeet utterance at 25 s (30 s for Whisper).
- The web app's Transcribe view lists Parakeet next to the Whisper models, and a model picked there loads next to
  the pinned `STT_MODEL` instead of unloading it (`POST /api/models/{id}/load?keep_pinned=true`).

## [0.9.0] - 2026-11-13

### Upgrade notes
- **Voice worker images have new tags:** `open-speech-chatterbox:0.2.1` and
  `open-speech-qwen3:0.2.1` (they were `0.1.0` and `0.1.1`), and the new Fish Audio S2 worker is
  `open-speech-fish:0.1.0`. Each worker is now built with the `providers/` context and `<name>/Dockerfile`, so start
  it once with `--build`; a hand-written build is `docker build -f providers/<name>/Dockerfile providers`. A
  deployment that pins `OPEN_SPEECH_*_IMAGE` keeps its own tag.
- **The GPU voice workers are off by default.** Each worker in `docker-compose.voice-models.yml` has a Compose profile
  of its own, `chatterbox`, `qwen3` and `fish`, and layering the file starts none of them. Name the ones
  you want with `--profile qwen3` (and so on, or `--profile "*"`) before `up`, or set
  `COMPOSE_PROFILES=qwen3,chatterbox` in the shell or `.env`, which also covers `down`. A worker that isn't turned
  on reads "Worker offline" in Models. Worker containers an earlier `up` started keep running, on their old images,
  until stopped; stop them once with
  `docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.voice-models.yml --profile "*" down`.
- **A voice worker on a GPU its torch build can't run** now reports `failed` with `cuda_arch_unsupported` on
  `/health` (it said `ok` while every load failed), so Compose shows it unhealthy and Models says why.
- **Host header check.** Requests get `400` (WebSocket handshakes `400` or close code `1008`) unless `Host` is an IP
  address, a single-label name, `localhost`, a name under `.localhost`, `.local`, `.lan`, `.home.arpa` or
  `.internal`, this machine's own hostname or FQDN (in Docker, the container's), or a name in the new `OS_ALLOWED_HOSTS`
  (`*.example.com` matches subdomains; `*` turns the check off). Add every other name that clients, monitors or a
  reverse proxy use, such as `*.fritz.box` for router-assigned LAN names.
- **Same-origin CORS and a cross-site write guard.** `OS_CORS_ORIGINS` now defaults to empty (no CORS headers), and
  `docker-compose.yml` no longer sets `*`. `POST`, `PUT`, `PATCH` and `DELETE` from a page of another origin (or a
  `null` one) get `403`; clients that send no `Origin` (SDKs, curl) are not affected. List cross-origin pages'
  exact origins in `OS_CORS_ORIGINS`, and in `OS_WS_ALLOWED_ORIGINS` for WebSockets, whose empty default now admits
  only this server's own pages. Behind a proxy, pass `Host` through with its port (nginx:
  `proxy_set_header Host $http_host;`) or list the public origin.
- **Security headers and API docs.** Responses carry `nosniff`, `Referrer-Policy`, `X-Frame-Options: DENY`, COOP and
  `Permissions-Policy`, and the web UI a strict `Content-Security-Policy`; embedding the UI in a frame (a Home
  Assistant webpage panel) needs `OS_SECURITY_HEADERS=false`. `/docs` and `/redoc` answer `404` unless
  `OS_API_DOCS=true`; `/openapi.json` is still served.
- **Failed-auth lockout and WebSocket rate limit.** With `OS_API_KEY` set, an address that sends
  `OS_AUTH_FAILURE_LIMIT` (default `10`) wrong keys within 60 seconds gets `429` for 60 seconds, even with the right
  key (`0` turns it off). WebSocket handshakes now count against `OS_RATE_LIMIT`. With `OS_TRUST_PROXY=true`, the
  client address is read from the right end of `X-Forwarded-For`; behind more than one proxy set
  `OS_TRUSTED_PROXY_HOPS`.
- **Refused WebSockets open, then close.** `/v1/audio/stream`, `/v1/audio/speech/stream` and `/v1/realtime` accept a
  socket they refuse and close it at once with a code and reason: `4001` a missing or wrong key or ticket, `1008` an
  Origin not allowed or an invalid query value, `4004` the feature turned off. They used to fail the handshake
  (`1006` in a browser). Read the close code and don't reconnect on these three.
- **API answers that changed.** `/v1/audio/transcriptions` with `response_format=json` answers only `{"text": ...}`,
  as OpenAI's does; diarized transcriptions follow `response_format`; `/v1/audio/translations` on a model that can't
  translate (large-v3-turbo, `*.en` and distil models) answers `400 translation_unsupported`, and the new
  `STT_TRANSLATE_MODEL` (empty by default) sets the model translations use. `GET /api/history` items carry `meta`
  as an object instead of `meta_json`. Model routes answer `500` or `503` for load failures where they said `400`.
- **Models and caches.** New `OS_PIN_DEFAULT_MODELS` (default `true`) keeps `STT_MODEL` and `TTS_MODEL` loaded
  through `OS_MODEL_TTL` and LRU eviction, outside the `OS_MAX_LOADED_MODELS` count. New `TTS_PRELOAD_DEFAULT`
  (default `true`) loads `TTS_MODEL` in the background at startup, which may download it on a first start. TTS cache
  keys and Reader part ids now include `TTS_TRIM_SILENCE` and `TTS_NORMALIZE_OUTPUT`, so cached audio is made once
  more after the upgrade.
- **Compose.** `docker-compose.yml` passes most settings through empty, so the application default or a legacy name
  in `.env` now applies: check `.env` for `STT_API_KEY` (clients then need that key) and `STT_SSL_ENABLED=false`. It
  ignores `OS_HOST`, `OS_PORT` and `OS_WYOMING_PORT`; change only the host side of `ports:` to move the server.
- **Removed settings and files.** `OS_HISTORY_RETAIN_AUDIO` and `OS_HISTORY_MAX_MB` never had an effect and now log a
  warning; `OS_HISTORY_MAX_ENTRIES` bounds history. `OS_STREAM_VAD_THRESHOLD`, `OS_PROVIDERS_DIR`,
  `docker-compose.qwen3.yml` (use `docker-compose.voice-models.yml --profile qwen3`) and `Dockerfile.cpu` (use the
  `DEVICE` build argument) are gone.
- **Clients.** The Python `client` extra requires `websockets>=14.0`. The Python and TypeScript clients default to
  `https://localhost:8100`; with `apiKey` set, the TypeScript client connects WebSockets with one-time tickets, so
  `RealtimeSession` and `LiveSpeechSession` open asynchronously (await `ready`).
- **TLS, Wyoming and the AI assistant.** A certificate Open Speech generated is renewed at startup within 30 days of
  expiry, so clients that pinned it must trust the new one; when only one of a certificate and key you supplied
  exists, startup stops with an error. Wyoming closes a connection after an `error` event or
  `OS_WYOMING_IDLE_TIMEOUT_S` (default `120`) seconds without an event, and at `OS_WYOMING_MAX_CONNECTIONS` (default
  `32`) a new connection closes the one that has waited longest. Once an AI
  server URL is set, `/v1/realtime` answers each turn by default; send `create_response: false` for transcripts only,
  and on a shared network set `OS_API_KEY` or `OS_ASSISTANT_CONNECTION_LOCKED=true`.
- **Reader documents.** Japanese, Chinese and Hindi text is now cut at its own sentence ends, and long sentences are no
  longer cut inside sound tags such as `[clear throat]`, for new imports; a document already in your library keeps its
  stored sentences until it is imported again.

### Added
- **Web app redesign.** `/web` is a new UI of plain static files and native JavaScript modules, with light and dark
  themes, a desktop rail and a mobile bottom bar, and an installable app manifest. Old hash addresses still work. See
  [Web UI](docs/WEB-APP.md).
- **Reader.** Playback that keeps going across views with a mini-player and lock-screen controls, a sleep timer,
  speeds from 0.5× to 3.5×, Reading appearance (text size, typeface, spacing, paper themes), sentence skips, and the
  Atkinson Hyperlegible Next face bundled.
- **Speak, Voices and Studio.** A voice picker with search, favorites and previews, a visual blend editor, presets,
  copy as curl, Python or JavaScript; named voices with per-model versions (`/api/voices/identities`), a guided
  three-step cloning flow with quality checks, editing a preset or voice in Speak; reorderable Studio turns.
- **AI assistant.** Point Open Speech at an OpenAI-compatible AI server (`OS_ASSISTANT_*`, or Settings › AI assistant
  with no restart) and edit AI presets with thinking controls, token limits, waiting phrases and spoken error
  sentences. **Talk** (`#talk`) holds a spoken conversation with it; **Ask AI** sends a transcript, a Reader
  selection, sentence, paragraph, chapter or book, the editor's text or a Studio script, with an optional question,
  and shows, speaks or saves the answer; long inputs run as jobs. See [AI assistant](docs/AI-ASSISTANT.md).
- **`/v1/realtime` with the AI assistant:** replies streamed and spoken sentence by sentence, barge-in that tells an
  interruption from "mm-hmm", typed questions, and per-turn timing logs.
- **Voice workers.** Optional GPU workers for Chatterbox (regular and Turbo), Qwen3-TTS 0.6B and
  Fish Audio S2 Pro (`fish/s2-pro`, 4-bit, about 3.3 GB of GPU memory), each an opt-in Compose profile, with pinned
  model revisions. See [Cloning models](docs/CLONING-MODELS.md).
- **Security settings:** `OS_ALLOWED_HOSTS`, `OS_SECURITY_HEADERS`, `OS_API_DOCS`, `OS_AUTH_FAILURE_LIMIT`,
  `OS_TRUSTED_PROXY_HOPS`, and `POST /api/auth/ws-ticket` for one-time WebSocket tickets.
- **OpenAI compatibility:** OpenAI TTS, STT and realtime model names map to `TTS_MODEL` and `STT_MODEL`, and the voices
  `ash`, `ballad`, `coral`, `sage`, `verse`, `marin` and `cedar` map to Kokoro voices.
- **API:** `GET /api/history?search=`, `max_input_length` in `GET /api/tts/capabilities`, `encoding=linear16` on
  `/v1/audio/stream`, more transcription `language` forms (`auto`, `en-US`), `HEAD` on Reader media, `OS_LOG_LEVEL`.
- **Docs:** what Open Speech is for, a Troubleshooting section, [AI agents](docs/AI-AGENTS.md),
  [AI assistant](docs/AI-ASSISTANT.md) and [Cloning models](docs/CLONING-MODELS.md) guides, and
  `scripts/smoke_openai_api.py`. A GitHub Actions workflow runs the test suite.

### Changed
- Streamed `/v1/audio/speech` sends audio as the backend produces it, and `/v1/realtime` reply audio streams as each
  chunk is synthesized.
- `/v1/audio/speech` fills omitted fields from settings (`TTS_MODEL`, `TTS_SPEED`, `TTS_DEFAULT_FORMAT`, `TTS_VOICE`)
  and ignores unknown JSON fields.
- Transcription uses Whisper's temperature fallback at `temperature` 0, as OpenAI does; diarization follows Whisper's
  segment timestamps and runs on CUDA when available; batch files are preprocessed like single uploads and wait on
  disk.
- SSML `<break>` strengths and time units, structural tags and comments follow the SSML spec more closely.
- Model loads, unloads, downloads and evictions are serialized per model type; downloads no longer load the model.
- Kokoro shares its weights across languages; Piper's `speed` scales each voice's own pace.
- Installs and images use a frozen `uv.lock`; `httpx` is a core dependency.
- GPU workers return freed memory after unload, the Chatterbox worker restarts itself after an unload, and a worker's
  CUDA out-of-memory error is `503 insufficient_gpu_memory`.
- Docker's healthcheck allows a 180 s start period, and Compose on Linux gives `./data` to the server's user when it
  can't write it.
- Speak › Live is a developer option (Settings › Developer); `/v1/audio/speech/stream` is unchanged.
- Transcripts stay out of the INFO log; session logs give sizes and levels only.
- US spelling throughout the web app and docs; API values such as `cancelled` keep their spelling.

### Fixed
- **Speech:** clear `400` errors for languages, effects and reference audio a model can't take; NaN samples, empty
  clips, ffmpeg failures and dictionary values with backslashes no longer fail requests; the `podcast_eq` effect keeps
  the speech band without clipping; Kokoro speaks long Chinese, Japanese, Hindi and Romance-language text to the end.
- **Transcription:** subtitle cue text and timestamps, unknown languages, empty audio, `STT_NORMALIZE` clipping, VAD
  trimming at odd sample rates and the first sound of an utterance.
- **Streaming and realtime:** odd-length frames, sample-rate conversion, close codes, `session.update` edge cases,
  VAD failures, lone UTF-16 surrogates in events, and large frames that stalled the event loop.
- **Live Reader:** Markdown fences, links and list markers mid-line, segment caps, edits to text already sent, and
  sentences the pronunciation dictionary empties.
- **Reader:** CommonMark-conformant Markdown import in linear time, WHATWG charset handling, EPUB paths, chapter names
  and reading order, re-rendering after a changed voice, dictionary or trimming setting, and two listeners at once.
- **Studio:** validation of effects, turns and lengths, renders that fail or race an edit, and database writes that
  could interleave.
- **Models API:** status codes, unknown models, download state, cache discovery and on-disk sizes; slow or failing
  workers no longer hold up listings or in-process synthesis.
- **Voice workers:** streaming latency and error handling, Qwen3 GPU memory after
  unload and cut-off generations, and loading on every GPU of a supported family.
- **Wyoming:** error events, language normalization, VAD trimming for every sample format, and connection cleanup.
- **AI assistant:** the API key is sent only to the origin it was set for, extra bodies are bounded, AI server errors
  never echo their body, and long answers no longer stall the server.
- **Clients:** the Python and TypeScript clients report refused streams (`StreamRefusedError`) and clean up on early
  exit.
- **Web app:** accessible tabs and cards, storage failures, mobile layouts, microphone silence notices, and messages
  that say why a WebSocket was refused or a worker can't run.
- **Config:** empty variables count as unset, legacy `STT_*` names apply under Compose, `.env.example` works with
  `docker run --env-file`, and the docs say how to write `$` and `'` in values.

### Security
- Fixed an API-key bypass through a crafted `Host` header; upgrade any deployment that relies on `OS_API_KEY`.
- A legacy `STT_API_KEY` in `.env` is enforced under Compose (it was silently dropped).
- The Host check, same-origin CORS, the write guard and WebSocket origin checks stop other web pages, including DNS
  rebinding, from driving a LAN server through a visitor's browser.
- Request bodies, audio length and sample rate, SSML pauses, Wyoming events and JSON nesting are bounded before
  they are processed.
- Logs mask `api_key`, `ticket` and signatures in query strings; TLS setup never changes certificate files it did not
  generate; artifact deletion and composer paths can't reach outside their roots.
- Voice workers are addressed by private network aliases, so a LAN device named like a worker can't receive requests.

### Removed
- The Ming-3.0 TTS 0.5B (`ming/3-0.5b`) and VoxCPM2 (`voxcpm/2`) workers, which ran slower than real time; saved
  voices, presets or `TTS_MODEL` values that name them stop working. Remove their leftovers with
  `docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.voice-models.yml --profile "*" up -d --remove-orphans`,
  `docker volume rm open-speech_ming-hf-cache open-speech_voxcpm-hf-cache` and
  `docker rmi open-speech-ming:0.1.0 open-speech-voxcpm:0.1.0`.
- `OS_HISTORY_RETAIN_AUDIO`, `OS_HISTORY_MAX_MB`, `OS_STREAM_VAD_THRESHOLD`, `OS_PROVIDERS_DIR`,
  `docker-compose.qwen3.yml` and `Dockerfile.cpu`.

## [0.8.0] - 2026-09-12

### Added
- Live Reader WebSocket at `/v1/audio/speech/stream` for incremental text input and PCM16 audio output.
- Browser controls for reading typed or pasted text from the cursor, reading a complete document,
  pausing, resuming, stopping, and clearing.
- Book, responsive phrase, and completed-word reading modes.
- Playback acknowledgements, bounded buffering, cancellation generations, source offsets, idle flushing,
  Markdown cleanup, HTML entity decoding, and safe fenced-code handling.
- Live speech support in the TypeScript client and a raw Python streaming example.

### Changed
- Natural book mode now keeps comma-separated clauses together and retains a short pause at synthesis
  boundaries instead of treating every comma as a standalone utterance.
- Docker dependency layers are ordered ahead of application source so ordinary code rebuilds reuse the
  expensive CUDA, Torch, and provider layers.
- Live Reader controls now remain next to the text input at normal laptop resolutions.
- Maintained documentation now describes current behavior only; obsolete phase plans, machine-specific
  test plans, and point-in-time reviews were removed from the working tree.

### Fixed
- Live Reader flow control now counts Unicode code points consistently between JavaScript and Python, so
  emoji and other supplementary characters cannot leave browser input permanently blocked.
- Piper Live Reader sessions now advertise the selected model's actual sample rate.
- HTTP requests carrying WebSocket upgrade headers can no longer bypass HTTP authentication middleware.
- Voice blends and speech input handling received additional validation and resource bounds.

## [0.7.0] - 2026-02-27

### Added
- **Batch Transcription API** — async multi-file transcription with job queue
  - `POST /v1/audio/transcriptions/batch` — submit up to 20 audio files
  - `GET /v1/audio/jobs` — list jobs with optional status filter
  - `GET /v1/audio/jobs/{id}` — full job detail including results
  - `GET /v1/audio/jobs/{id}/result` — results array (409 if not done yet)
  - `DELETE /v1/audio/jobs/{id}` — cancel or delete jobs
- SQLite-backed batch job store (`src/batch/store.py`)
- Async batch worker (`src/batch/worker.py`) with configurable concurrency
- Per-file error isolation — one file failing doesn't abort the batch
- History integration — completed batch files auto-logged to STT history
- New config: `OS_BATCH_WORKERS` (default 2) — max concurrent batch jobs

## [0.6.1] - 2026-02-23

### Removed
- Moonshine and Vosk STT backends, and the Fish Speech, XTTS, F5-TTS, and in-process Qwen3 TTS
  backends with their dependencies. (Recorded after the fact; Qwen3 later returned as an isolated
  provider worker.)

### Fixed
- Stabilized model availability, loading, download progress, and provider-missing states.
- Improved the model browser, audio waveform, mobile layout, and Piper catalog handling.

## [0.6.0] - 2026-02-20

### Added
- Persistent voice profiles and generation history.
- Conversation rendering, voice effects, and the multi-track composer.
- OpenAI-style realtime audio, Silero VAD, and Wyoming protocol support.
- Python and TypeScript client libraries.

### Security
- Added bounded realtime buffers, WebSocket origin controls, upload limits, non-root containers, and
  hardened model-management concurrency.

## [0.5.1] - 2026-02-17

### Fixed
- Standardized HTTP error envelope across API and middleware responses to `{"error": "..."}`.
- Replaced ad-hoc `JSONResponse({"error": ...})` branches in TTS endpoints with `HTTPException` plus a unified exception handler.
- Synced README API endpoint table with all implemented routes in `src/main.py`.
- Added full OS_*/STT_*/TTS_* environment variable reference in README.
- Corrected README Qwen3 model IDs to match `src/model_registry.py` (`qwen3-tts/...`).
- Added missing `STT_MODEL_DIR` documentation entry to `.env.example`.
- Harmonized `docker-compose.yml`, `docker-compose.cpu.yml`, and `docker-compose.gpu.yml` env coverage, ports, and volume mounts using non-root container paths.
- Expanded runtime tests for `src/streaming.py` session lifecycle, VAD flow, mid-stream disconnects, and error propagation.

### Changed
- Version bumped to 0.5.1.

## [0.5.0] - 2026-02-17

### Added
- Phase 6 production hardening:
  - TTS response caching (LRU, configurable max size, cache bypass query param)
  - Speaker diarization endpoint support via optional `pyannote.audio`
  - Audio preprocessing (noise reduction + gain normalization) for STT
  - Audio postprocessing (silence trimming + normalization) for TTS
  - Python client SDK (`OpenSpeechClient`) with sync/async helpers
  - Pronunciation dictionary + SSML subset support (`input_type=ssml`)
- New optional extras: `diarize`, `noise`, `client`
- Extensive test coverage for Phase 6 modules and APIs

### Changed
- Version bumped to 0.5.0

## [0.4.0] - 2026-02-17

### Added
- **Qwen3-TTS Backend** — voice design via text description, voice cloning, streaming, 0.6B and 1.7B models
- **Fish Speech Backend** — high-quality synthesis with zero-shot voice cloning
- **Extended TTS API** — `voice_design` and `reference_audio` fields on `/v1/audio/speech`
- **Voice Cloning Endpoint** — `POST /v1/audio/speech/clone` with multipart file upload
- **Voice Presets** — `GET /api/voice-presets` endpoint, presets dropdown in web UI Speak tab
- **Voice presets config** — load from YAML via `TTS_VOICES_CONFIG` env var
- Model registry: 3 new models (qwen3-tts-0.6b, qwen3-tts-1.7b, fish-speech-1.5)
- 23 new tests (332 total)

### Fixed
- Health endpoint now reports correct version (was hardcoded 0.1.0)
- Version sourced from `__version__` variable
- Speed slider: changed from 0.25 steps to 5% increments, min 0.5x (`75fb457`)
- Kokoro no longer appears in STT model dropdown (`c128533`)
- Kokoro-82M no longer listed as STT in Models tab (`c128533`)
- Moonshine models show "provider not installed" instead of Download when `useful-moonshine-onnx` is missing (`c128533`)
- Version badge now loads dynamically from `/health` endpoint (`c128533`)
- Voice presets updated to match actual available voices (`75fb457`)
- Mic transcription partial fix for WebSocket format issue (`c128533`)
- Moonshine package renamed from `moonshine-onnx` to `useful-moonshine-onnx` (`773095f`)

### Changed
- Version bumped to 0.4.0
- Optional dependency groups: `qwen` (transformers+accelerate+torch), `fish` (fish-speech)
- TTS history entries now have download and delete buttons (`c128533`)
- Stream toggle has tooltip explaining behavior (`c128533`)

## [0.3.0] - 2026-02-17

### Added
- **Phase 3: Unified Model Architecture**
  - Single image strategy — one container for CPU + GPU
  - Unified ModelManager for STT and TTS lifecycle
  - Environment variable rename: OS_* (server), STT_* (speech-to-text), TTS_* (text-to-speech)
  - Backwards compatibility for all old env var names
  - Unified /api/models endpoints (list, load, unload, status)
  - Model registry with 14 curated models and metadata
  - Model browser in web UI (available/downloaded/loaded states)
  - Download progress API
- **Piper TTS Backend**
  - 6 curated English voices (US + GB)
  - ONNX inference, lightweight (~35MB per model)
  - Auto-download from HuggingFace
- 36 new tests (309 total)

## [0.2.0] - 2026-02-16

### Added
- **Professional Web UI overhaul**
  - Light/dark theme with OS auto-detection
  - Inter font, card-based layout, pill tabs
  - Custom audio player with progress bar
  - Voice blend builder (visual tag-based UI)
  - Toast notifications (no more alert())
  - Mobile responsive design
- **HTTPS support** — auto-generated self-signed certificates
- **TTS environment variables** in Docker configs
- `.env.example` with all configuration documented
- Generic `docker-compose.yml` for easy testing

### Fixed
- Generate button spinner stuck after completion
- Kokoro showing as both STT and TTS in models tab
- STT models not showing device (cuda/cpu)
- Default model unload button showing when it shouldn't
- Streaming TTS torch tensor → numpy conversion

## [0.1.0] - 2026-02-14

### Added
- **Phase 1: Core STT Server**
  - OpenAI-compatible `/v1/audio/transcriptions` and `/v1/audio/translations`
  - faster-whisper backend with GPU (CUDA float16) and CPU (int8) support
  - Silero VAD for voice activity detection
  - Model lifecycle: TTL eviction, max models, LRU
  - Self-signed HTTPS with auto-generated certificates
- **Phase 2: TTS + Streaming**
  - Kokoro TTS backend (82M params, 52 voices, voice blending)
  - `/v1/audio/speech` endpoint (OpenAI-compatible)
  - Streaming TTS (`?stream=true`, chunked transfer)
  - WebSocket real-time STT (`/v1/audio/stream`)
  - Moonshine and Vosk STT backends
  - SRT/VTT subtitle formatters
  - 3-tab web UI (Transcribe, Speak, Models)
- **Security** — API key auth, rate limiting, CORS, upload limits
- Docker images: CPU and GPU
- 230 tests
