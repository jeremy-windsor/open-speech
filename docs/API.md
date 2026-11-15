# API reference

Open Speech's HTTP and WebSocket API. Apps written for OpenAI's audio API work without changes; see
[OpenAI clients](#openai-clients). For which endpoint suits an AI agent, with the smallest working request for each,
see [AI-AGENTS.md](AI-AGENTS.md).

- [Authentication](#authentication), [Refused WebSockets](#refused-websockets),
  [Upload and body limits](#upload-and-body-limits), [OpenAI clients](#openai-clients)
- [Speech-to-text](#speech-to-text): transcriptions, translations, batch jobs, streaming
- [Text-to-speech](#text-to-speech): speech, [voice blending](#voice-blending), voices, cloning, voice design
- [Live Reader](#live-reader-ws-v1audiospeechstream), [Realtime audio](#realtime-audio),
  [AI assistant](#ai-assistant-api)
- [Model management](#model-management), [Studio and persistence](#studio-and-persistence), [Reader](#reader),
  [Health and web app](#health-and-web-app)
- [Wyoming](#wyoming), [Python client](#python-client)

API endpoints use Bearer auth (`Authorization: Bearer KEY`) when `OS_API_KEY` is set. `/health`,
`/openapi.json`, `/web`, the web UI's static files, the signed Reader audio links under
`/v1/reader/media/`, and `/docs` and `/redoc` when they are on, need no key. WebSocket endpoints also accept a one-time `?ticket=` from
`POST /api/auth/ws-ticket`, for browsers that cannot set headers on a WebSocket. The server also
accepts a deprecated `?api_key=KEY` query parameter as well, and logs a warning each time. Open Speech's
own log shows the values of `api_key`, `ticket`, `exp` and `sig` in a query string as `***` (in uvicorn's
request lines, at every `OS_LOG_LEVEL`), but a reverse proxy's logs and browser history still keep the key,
so prefer the header or a ticket. See [Security](SECURITY.md) for
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

## Authentication

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/auth/ws-ticket` | Returns `{"ticket": "...", "expires_in": 60}` for one WebSocket connection |

The endpoint itself needs the API key. Connect within 60 seconds with `?ticket=TICKET` on any
WebSocket endpoint; a ticket works once and lives in the server's memory, so a restart discards it.
Fetch a new ticket for every connection, reconnects included. A socket without a valid ticket or
key is opened and then closed at once with `4001 Invalid or missing API key`
([Refused WebSockets](#refused-websockets)). A wrong, expired, or reused ticket or key also counts toward the
[failed-auth lockout](SECURITY.md#api-key); a handshake that sends no credentials does not. A page on another
origin also needs that origin in `OS_CORS_ORIGINS`, because its ticket request is a cross-site write
(see [Browser origins](SECURITY.md#browser-origins)).

```bash
curl -sk -X POST -H "Authorization: Bearer my-secret-key" https://localhost:8100/api/auth/ws-ticket
# {"ticket":"TICKET","expires_in":60}  ->  wss://localhost:8100/v1/audio/stream?ticket=TICKET
```

## Refused WebSockets

A WebSocket that is refused is accepted and then closed at once, with no event first, so the client, a browser
included, gets the close code and the reason:

| Close code | Reason | When |
|---|---|---|
| `4001` | `Invalid or missing API key` | `OS_API_KEY` is set, and the socket has no key, a wrong one, or a used, expired or unknown `?ticket=` |
| `1008` | `Origin not allowed. List it in OS_WS_ALLOWED_ORIGINS to allow it.` | The page's `Origin` isn't allowed ([Browser origins](SECURITY.md#browser-origins)) |
| `4004` | `Live TTS is disabled` | `/v1/audio/speech/stream` with `TTS_ENABLED` or `TTS_LIVE_ENABLED` false |
| `4004` | `Realtime API is disabled` | `/v1/realtime` with `OS_REALTIME_ENABLED=false` |

`/v1/audio/stream` has no switch of its own; it also closes with `1008` for a query value it can't take, with a
reason that names it (`Invalid sample_rate: ...`, `Invalid endpointing: ...`, `Invalid vad: ...` or
`Invalid interim_results: ...`, see [`WS /v1/audio/stream`](#ws-v1audiostream)). Nothing is read and no session starts. Don't reconnect after one
of these closes: the same socket gets the same answer.

A refused socket is opened before it is closed so that the client gets the code and the reason: a failed
handshake gives a browser only close code `1006`, with no reason, and other clients an HTTP `403`. A client must
therefore read the close code after the socket opens.

Two refusals answer the handshake with HTTP, before any route runs: a `Host` that isn't accepted gets `400`
(`OS_ALLOWED_HOSTS`, [Host names](SECURITY.md#host-names)), and the rate limit or the failed-auth lockout gets `429` with
`Retry-After` ([Rate limiting](SECURITY.md#rate-limiting), [API key](SECURITY.md#api-key)).

The Python client's `stream_transcribe` and `async_stream_transcribe` raise `src.client.StreamRefusedError` (a
`RuntimeError` with `.code` and `.reason`) on a `4001`, `1008` or `4004` close, and don't reconnect. The
JavaScript SDK does the same with its own `StreamRefusedError` (`client-js/README.md`, "Refused sockets").
`examples/listen.py` and `examples/voice_chat.py` print one `[refused]` line with the code, the reason and what to do.

## Upload and body limits

The server refuses an oversized request body while it arrives. A declared `Content-Length` over the
limit gets `413` before anything is read, and a body that grows past it is cut off with
`413 Request body too large. Max: NMB`. The limit is `READER_MAX_UPLOAD_MB` for
`POST /v1/reader/documents`, the larger of `OS_MAX_UPLOAD_MB` and `OS_BATCH_MAX_TOTAL_MB` for
`POST /v1/audio/transcriptions/batch`, and `OS_MAX_UPLOAD_MB` for every other route. Multipart and
other bodies get 1 MB on top for form fields. A JSON body may be the larger of 4/3 of the limit plus
64 KB (room for base64) and the limit plus 1 MB, so with the default 100 MB any JSON request is capped
at about 133 MB. Routes then apply their exact limits, such as `413 Upload too large. Max: 100MB` for
an audio file.

## OpenAI clients

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

### OpenAI voice names

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
are ignored, so newer OpenAI parameters do not break requests; a `voice_blend` field is rejected with `422`
(a blend goes in `voice`). When an OpenAI model name (`tts-1`, `tts-1-hd`, `gpt-4o-mini-tts`) arrives with
`instructions` and `TTS_MODEL` cannot follow instructions, the instructions are ignored; an Open Speech
model ID that cannot follow them gets `400`.

### HTTPS

Open Speech uses a self-signed certificate by default, and most third-party apps refuse
self-signed certificates. Pick one:

- On a trusted home network, turn HTTPS off with `OS_SSL_ENABLED=false` and use `http://HOST:8100/v1`.
- Put Open Speech behind a reverse proxy (Caddy, nginx, Traefik) that has a real certificate; see
  [Reverse proxies](SECURITY.md#reverse-proxies) for the settings it needs.
- Give Open Speech a trusted certificate with `OS_SSL_CERTFILE` and `OS_SSL_KEYFILE` (set both).

## Speech-to-text

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

### `POST /v1/audio/transcriptions`

Multipart form upload.

**Fields:**
- `file` (required) — up to `OS_MAX_UPLOAD_MB` (`413` above it) and 4 hours of audio, at a sample rate
  of 8,000–192,000 Hz
- `model` — default `STT_MODEL`; OpenAI names map to it
- `language` — an ISO-639-1 code such as `en`. Case and a region suffix are accepted (`EN`, `en-US`,
  `pt_BR`), as are `auto` (auto-detect) and the older codes `nb`, `jv`, `iw`, `in`, and `ji`. Anything
  else gets `400 Invalid language '...'. Use an ISO-639-1 code such as 'en'`. Parakeet transcribes English only:
  another language gets `400 language_unsupported` ([Parakeet](MODELS.md#parakeet)).
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

### `POST /v1/audio/translations`

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

### `POST /v1/audio/transcriptions/batch`

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

### `WS /v1/audio/stream`

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
see [Nothing transcribed?](TROUBLESHOOTING.md#nothing-transcribed).

```javascript
const ws = new WebSocket("wss://localhost:8100/v1/audio/stream?vad=true");
ws.onmessage = (e) => console.log(JSON.parse(e.data));
ws.onopen = () => ws.send(audioChunkArrayBuffer);
```

## Text-to-speech

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

### `POST /v1/audio/speech`

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
- `instructions` *(backend-gated)* — how to speak (Breeze TTS 2, Qwen3 1.7B Custom Voice), or the voice's
  description on Qwen3 1.7B VoiceDesign
- `voice_design` *(backend-gated)* — a description of the voice, for a model that designs one (Breeze TTS 2,
  Qwen3 1.7B VoiceDesign)
- `reference_audio` *(base64-encoded; backend-gated)* — decodes to at most `OS_MAX_UPLOAD_MB`
  (`413 Reference audio too large. Max: NMB`). Invalid base64 gets
  `400 reference_audio must be base64-encoded audio`, and one that decodes to nothing gets
  `400 Reference audio is empty`. Send it or `voice_library_ref`, not both.
- `voice_library_ref` *(stored provider-neutral reference; backend-gated)*
- `clone_transcript` *(backend-gated)*
- `steps`, `cfg_scale`, `temperature` *(model-gated)* — the model's own options, each sent only to a model whose
  `capabilities.options` declares it (Supertonic `steps`; Breeze TTS 2 `cfg_scale` and `temperature`) and held to
  its `min` and `max` (`400`); other models ignore them
- `effects` — at most 16 (`422`); not with streaming (`400`); invalid parameters get `400`; skipped
  when `OS_EFFECTS_ENABLED=false`

**Query params:** `stream`, `cache`

Left-out `model`, `voice`, `speed`, and `response_format` come from the `TTS_*` settings, and unknown
fields are ignored; see [Use it from an OpenAI app](#openai-clients) for the exact rules.

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
its own, usually lower, limit. `capabilities.options` maps each of the model's own options to
`{type, label, min, max, step, default, description}`; `type` is `integer` or `number`, and a `null` default means
the model chooses per request. `POST /v1/audio/models/load` and `/unload` act on `TTS_MODEL` when the
body is empty or names no `model`; a load first unloads the other loaded TTS models, except the pinned `TTS_MODEL`.

### Voice blending

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

### `GET /v1/audio/models`

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

### `POST /v1/audio/speech/clone`

Multipart form endpoint that forwards reference audio to backends that support it.
Cloning needs a clone-capable model: Chatterbox, Qwen3 Base, Fish Audio S2, or Breeze TTS 2, run as isolated providers
with `docker-compose.voice-models.yml`. Those GPU workers are optional and off by default: each starts only when
its Compose profile is named (`--profile qwen3`, or `COMPOSE_PROFILES`), as [GPU voice workers](CONFIGURATION.md#gpu-voice-workers) shows. The
built-in Kokoro, Piper, and Pocket-TTS backends do not clone.
It accepts required `input` plus `model`, `reference_audio`, `voice_library_ref`, `voice`, `speed`,
`response_format`, `transcript`, and `language` form fields. Supply either an uploaded reference or a
stored `voice_library_ref` when the selected backend requires one. A left-out `model` means
`TTS_MODEL`, and a left-out `response_format` means `TTS_DEFAULT_FORMAT`. `input` is held to
`TTS_MAX_INPUT_LENGTH`, and the pronunciation dictionary applies as on `/v1/audio/speech`. The upload
may be up to `OS_MAX_UPLOAD_MB` (`413 Upload too large. Max: NMB`); an empty one gets
`400 Reference audio is empty`. `speed` outside 0.25–4.0 gets `422`.

### `POST /api/voices/design`

Speaks a short sample in a voice made from a description, with a model whose capabilities have `voice_design`
(Breeze TTS 2 or Qwen3 1.7B VoiceDesign, which must be loaded). JSON body: `model`, `description` (up to 1,000
characters), and optional `text` (up to 300 characters; a short greeting when left out), `language`, `seed`
(0 to 2³¹−1) and `options` (the model's own options, as `/v1/audio/speech` takes them, such as
`{"cfg_scale": 2.5, "temperature": 0.8}`; options the model doesn't declare are ignored). It answers a WAV at the
model's rate, with the seed it was made with in `X-Voice-Seed`: without a `seed` each sample is a new voice, and the
same seed, description, text and options give the same sample back. A model that can't design a voice gets `400`,
and a sample longer than 30 seconds gets `400`. To keep the voice, save the sample as a designed voice
(`POST /api/voices/designed`, below).

```bash
curl -sk https://localhost:8100/api/voices/design -H 'Content-Type: application/json' \
  -d '{"model": "breeze/tts-2", "description": "A warm, calm man in his fifties, a slight rasp, unhurried",
       "text": "Hello! This is the voice you described."}' -o sample.wav
```

### `POST /api/voices/designed`

Saves a designed sample as a named voice. Multipart form: `name`, `model` (the model that designed it),
`description`, `transcript` (the sample's words), `audio` (the sample, a WAV of at most 30 seconds), and optional
`seed`, `options` (a JSON object of the model's own options) and `speak_with` (cloning model IDs, repeated or
comma-separated). It creates the named voice, its sample, and a version for each `speak_with` model that clones
the sample (`{"model", "voice": "<sample>", "reference_audio_id": "<sample>"}`). Without `speak_with`, the design
model is used when it clones (Breeze TTS 2), else the downloaded cloning models of the same provider (Qwen3 1.7B
VoiceDesign: `qwen3/1.7b-base`, and `qwen3/0.6b-base` when it is downloaded). It answers `201` with the named voice;
a name in use gets `409`, a model that can't design or a `speak_with` model that can't clone the sample `400`, and a
sample that isn't a WAV or is longer than 30 seconds `422`. A refused request saves nothing.

```json
{"id": "6f1c…", "name": "Warm narrator", "created_at": "2026-11-15T09:12:03+00:00", "preset_count": 0,
 "voice": "voice:6f1c…",
 "realizations": [{"id": "…", "voice_identity_id": "6f1c…", "model": "breeze/tts-2",
                   "voice": "designed_6f1c…", "reference_audio_id": "designed_6f1c…",
                   "created_at": "2026-11-15T09:12:03+00:00"}],
 "design": {"description": "A warm, calm man in his fifties", "model": "breeze/tts-2", "seed": 1234567,
            "options": {"temperature": 0.8}, "sample": "designed_6f1c…",
            "transcript": "Hello! This is the voice you described.", "duration_s": 3.4}}
```

Every named voice (`GET /api/voices/identities`, create, rename) carries `design`: `null`, or for a designed voice
what it was designed from and its sample (`sample` is the sample's voice-library name). The sample is not listed by
`GET /api/voices/library`, but `GET /api/voices/library/{sample}/audio` plays it. It can't be overwritten,
edited or deleted through `/api/voices/library` (`409`); deleting the named voice deletes it.

`PUT /api/voices/designed/{id}` takes the same fields without `name` and `speak_with` and replaces the sample, its
transcript and its design in place; the versions keep speaking from it. A named voice that isn't designed gets
`404`.

## Live Reader: `WS /v1/audio/speech/stream`

Live Reader accepts text deltas and immediately streams PCM16 audio back to the requester. With HTTPS,
the address is `wss://HOST:8100/v1/audio/speech/stream`; WSS is WebSocket traffic protected by HTTPS.
The Open Speech machine performs synthesis only. The browser, laptop, or other requesting client plays
or saves the returned audio.

The web UI's Speak › **Live** mode (a developer test, shown by Settings › Developer) uses it: Start reads
from the cursor, Read all from start reads everything, and Pause, Resume and Stop control the session. Its
modes are Book (whole sentences),
Responsive (shorter phrases) and Each word. A browser requires one Start click before it may play
audio.

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
your AI server and speak its reply, see [AI assistant](AI-ASSISTANT.md).

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

## Realtime audio

| Method | Path | Description |
|---|---|---|
| `WS` | `/v1/realtime` | OpenAI-style realtime audio WebSocket |

`/v1/realtime` transcribes each turn and speaks replies. A `response.create` that carries text speaks
that text word for word. With the [AI assistant](AI-ASSISTANT.md) on, each turn's transcript, and a
`response.create` with no text, goes to your own AI server: short waiting phrases are spoken while it
works, then its reply, and its errors are spoken too. The conversation lasts as long as the connection,
so a reconnect starts fresh. Without the assistant, nothing answers a turn by itself. It is not full
OpenAI Realtime parity: there are no tools (function calling) and no text-only output, and
`conversation.item.create` takes only a user's typed message ([Typed questions](AI-ASSISTANT.md#talking-to-it)).

The current OpenAI Python SDK connects without a WebSocket subprotocol override.
GA sessions use nested `session.audio.input` and `session.audio.output` settings
and `response.output_audio.delta` events. PCM at 24 kHz, PCMU and PCMA are supported;
input transcription selects its own model through `audio.input.transcription`.
Clients that offer the `realtime` subprotocol get the legacy (beta) events
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
the OpenAI voice names listed under [Use it from an OpenAI app](#openai-clients). Replies
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

## AI assistant API

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
| `POST` | `/api/assistant/ask` | Send text, a transcript or a Reader document (all of it, some chapters or blocks, a selection or one sentence) to the AI with a preset and an optional question of your own; a document's text goes with its name and chapter. The reply as text, spoken (base64 audio) or saved as a Reader document; a longer input answers `202` with a book job ([Ask AI](AI-ASSISTANT.md#ask-ai-send-anything-to-the-ai)) |

The book jobs' routes are in the [Reader](#reader) table; they also follow and cancel the jobs that
`/api/assistant/ask` starts. Every route but `GET /api/assistant` and the
`/api/assistant/connection` ones answers `404` `assistant_disabled` while the assistant is off; the
connection routes are how you turn it on. See [AI assistant](AI-ASSISTANT.md) for the bodies, the answers
and the error codes.

## Model management

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
| `POST` | `/api/models/{id}/load` | Load model, unloading the others of its type (a TTS load keeps the pinned default; `?keep_pinned=true` does for STT) |
| `POST` | `/api/models/{id}/download` | Download artifacts |
| `POST` | `/api/models/{id}/prefetch` | Alias for download |
| `DELETE` | `/api/models/{id}` | Unload model |
| `DELETE` | `/api/models/{id}/artifacts` | Delete cached artifacts |

See [Model residency](MODELS.md#model-residency) for what load, download, and eviction keep in memory.
Legacy `POST /api/pull/{model}` fetches an STT model without loading it. The unified routes answer
errors with these codes:

| Status | Code | Meaning |
|---|---|---|
| `404` | `unknown_model` | Not a model any installed or configured provider serves |
| `404` | `not_loaded` | `DELETE /api/models/{id}` on a model that is not loaded |
| `404` | `tts_disabled` | With `TTS_ENABLED=false`, the status, progress, load, download, or prefetch of a TTS model |
| `400` | `provider_missing` | The provider is not installed; for Qwen3, Chatterbox, Fish, and Breeze the message says to run the isolated worker and list it in `TTS_EXTERNAL_PROVIDERS` |
| `503` | `provider_unavailable` | An isolated worker is configured but doesn't answer, or answers that it can't run (its `/health` is `failed`); then the message gives the worker's reason, as in `Provider 'chatterbox' is configured but its worker can't run: Torch wheel has no code that runs on sm_120 (built for: ...).` |
| `503` | `insufficient_gpu_memory` | The model doesn't fit in the GPU's free memory, even after unloading the idle models that aren't pinned (the [GPU memory guard](MODELS.md#gpu-memory-guard)), or a worker ran out of GPU memory; the message names what to unload |
| `409` | `model_fetching` | `DELETE /api/models/{id}/artifacts` while the model's files are being downloaded (a first use or a preload); delete them once the download has finished |
| `500` | `load_failed`, `download_failed`, `unload_failed` | Any other failure, such as CUDA out of memory in process |

Each `GET /api/models` row, and a model's status, has `pinned`: `true` for `STT_MODEL` and `TTS_MODEL` while
`OS_PIN_DEFAULT_MODELS` is on (idle and LRU eviction leave them, a TTS load leaves the pinned TTS model, and an STT
load with `keep_pinned=true` loads next to the pinned STT model). Each also has `provider_available` and `provider_failure`. The
second is the worker's own `{"code": ..., "message": ...}` for a TTS model whose isolated worker answers
that it can't run (for example `cuda_arch_unsupported`), and `null` otherwise, also for a worker that
doesn't answer.

`GET /v1/models/{model}` answers `404` for a catalog TTS model no request can use: TTS is disabled,
its provider is not installed, or its isolated worker is not configured or does not advertise it.

## Studio and persistence

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/voices/design` | Speak a short sample in a voice made from a description (see [above](#post-apivoicesdesign)) |
| `POST` | `/api/voices/designed` | Save a designed sample as a named voice (see [above](#post-apivoicesdesigned)) |
| `PUT` | `/api/voices/designed/{id}` | Replace a designed voice's sample and design |
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
| `DELETE` | `/api/voices/identities/{id}` | Delete a named voice; returns `{"moved_to": ...}` (a recording name or null): a designed sample still used by presets or other voices becomes that recording, else it is deleted |
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

See [Named voices and reading presets](VOICE-IDENTITIES.md) for request bodies and examples.

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

## Reader

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
| `POST` | `/v1/reader/documents/{id}/assistant` | Send the document to the AI assistant: `preset` (default `book`), `overrides`, `chapters` (`[first, last]`, from 0) and `title`; `202` with the job ([Book jobs](AI-ASSISTANT.md#book-jobs)) |
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

## Health and web app

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/web` | Web UI |
| `GET` | `/openapi.json` | OpenAPI schema |
| `GET` | `/docs` | Swagger UI (only with `OS_API_DOCS=true`) |
| `GET` | `/redoc` | ReDoc (only with `OS_API_DOCS=true`) |

## Wyoming

Open Speech can expose STT/TTS over the [Wyoming protocol](https://github.com/rhasspy/wyoming).

```bash
OS_WYOMING_ENABLED=true
# 0.0.0.0 inside Docker; 127.0.0.1 allows only local clients on a source install
OS_WYOMING_HOST=0.0.0.0
OS_WYOMING_PORT=10400
```

The Docker image and base Compose file enable Wyoming on `0.0.0.0:10400`;
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

In Home Assistant, add the **Wyoming Protocol** integration (Settings › Devices & services › Add integration) with
the Open Speech host and port `10400`. Its speech-to-text and text-to-speech then show up for a voice assistant's
pipeline.

## Python client

`src.client.OpenSpeechClient` is the built-in Python client. The `client` project extra installs what it needs:
`httpx`, and `websockets` 14 or newer for its WebSocket helpers. Its `ssl_verify` option applies to those helpers
too: `False` accepts the self-signed certificate, and a CA bundle path or an `ssl.SSLContext` is used as given.
Its `stream_transcribe` and `async_stream_transcribe` raise `RuntimeError` when the connection ends before the
server confirms the end of the stream, and `src.client.StreamRefusedError` on a refused socket
([Refused WebSockets](#refused-websockets)). The JavaScript SDK is in `client-js/`
([client-js/README.md](../client-js/README.md)).
