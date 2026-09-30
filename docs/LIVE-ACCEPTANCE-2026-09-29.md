# Live speech acceptance — 2026-09-29

Actual inference succeeded for all **36 registered TTS models** and **14 registered
STT models**. Full acceptance **failed**: Piper speaker selection and artifact
deletion fail, and the current OpenAI Realtime contract is incompatible.

This audit exercised HTTP, WebSocket, TCP, SDK and browser operations against real
models. It did not use mock inference or count the unit suite as live acceptance.
Temporary callers were disposable; no new test source was retained.

[Curated observations](LIVE-ACCEPTANCE-2026-09-29.json) contain the model IDs,
measured audio output, transcripts, contract failures and final runtime checks.

## Runtime being qualified

- Core source: `384e617cf77a208c36f296d3e24186a829e06fb4` (Slice 6).
- Current worker source was mounted into isolated copies of the installed worker
  images. Current core source was mounted into an isolated copy of the installed
  core image. This qualifies that source/dependency combination, **not a newly
  rebuilt release image**.
- Windows-host Docker, Linux application containers, RTX 2070 SUPER with 8 GiB
  VRAM; Docker VM had about 7.69 GiB RAM.
- Official clients: Python `openai==3.22.1`; JavaScript `openai==7.25.0` on
  Node `v24.21.0`. Project Python and TypeScript clients also exercised real audio.
- Synthetic speech supplied all reference recordings. Browser microphone capture
  used a synthetic WAV through an isolated Chrome profile.
- The original four production containers remained running and healthy with their
  original start times. No production deployment, restart, voice-library update,
  model load or database migration was performed.

SDK calls used a placeholder key and a local Open Speech base URL. No OpenAI cloud
request was made. These are client compatibility results, not a comparison with
the acoustic quality of OpenAI's hosted models.

## Provider results

All six HTTP output formats were exercised with real output: WAV, MP3, Opus, AAC,
FLAC and PCM. WAV headers and sample data were inspected; compressed formats were
decoded with ffmpeg and checked for nonempty, non-silent output. Raw PCM has no
header, so its native sampling rate remains part of the provider contract.

| Provider | Actual coverage | Result and qualification limits |
| --- | --- | --- |
| Kokoro | All 52 voices with native-language text across nine language codes; six formats; blend; SSML; speed; five effects; HTTP streaming; named voice; Live Reader | All 52 generated valid audio. Japanese/Chinese initially lacked offline assets; the online retry succeeded for every voice. This is generation evidence, not a listening score in every language. |
| Pocket TTS | All eight voices; six formats; named voice; Live Reader; unsupported-control rejection | Real output succeeded. Unsupported speed and instructions returned explicit errors. |
| Piper | All 28 registered models; six formats per model; speed; HTTP streaming; Live Reader; ASR round trips; explicit per-model named realizations | The main matrix completed 421 operations. All 28 default models and the later 28 explicit named defaults generated audio. Speaker selection and cache deletion failed deeper checks, described below. |
| Qwen3 | 0.6B CustomVoice and 0.6B Base; all nine CustomVoice choices; six formats; named voice; Base clone; CustomVoice Live Reader | Actual generation succeeded for both models. Base does not advertise Live Reader. Unsupported speed/instructions were rejected. |
| Chatterbox | Regular and Turbo; six formats; HTTP streaming; Live Reader; multipart clone; named clone; ASR round trip | Both models and both saved named clones generated valid audio. Initial disposable database ownership errors were corrected and the actual named calls rerun. |
| CosyVoice | 2-0.5B and 3-0.5B; six formats; speed; instructions; HTTP streaming; Live Reader; multipart clone; named clone; ASR round trip | Both models succeeded. CosyVoice 3's isolated retry completed all 17 operations after the initial audit timeout/OOM, rather than counting the first attempt as a pass. |

The Piper short phrase matched the input words in 27 of 28 ASR round trips after
normalizing case and punctuation. `en_GB-cori-high` was recognized as “provided”
instead of “provider.” That observation does not isolate whether TTS or the ASR
decoder caused the substitution.

Pocket's additional English round trip matched the input exactly. Qwen Base also
matched its English input; Qwen CustomVoice's Ryan output was recognized with an
extra “Uh” and the initial “A” omitted. Both Qwen models returned real HTTP-streamed
audio, with first bytes observed after 12.420 and 8.069 seconds respectively.
Configuring Qwen Base Live Reader returned its explicit unsupported-feature error;
Base synthesis without the required reference transcript returned HTTP 400.

Example measured Live Reader time to first audio: Chatterbox Turbo 1.153 seconds,
Qwen CustomVoice 10.168 seconds, CosyVoice 2 4.403 seconds, CosyVoice 3 3.150 seconds.
These are individual observed calls, not stable performance benchmarks.

## Speech recognition and transport

All 14 registered Whisper/Distil-Whisper models transcribed the same 12.1195-second
synthetic English recording on CUDA. Every model recovered the input words after
normalizing case and punctuation. The accompanying result artifact lists the exact
model IDs and observed text; this single clean recording is not a corpus benchmark.

Actual additional coverage included:

- Official Python transcription and translation in JSON, verbose JSON, text, SRT
  and VTT; all five batch result formats with two real files.
- WAV, MP3, Opus, AAC and FLAC uploads, plus 48 kHz stereo float32 WAV. All recovered
  the expected English text.
- A Spanish synthetic recording: Spanish transcription recovered the
  sentence; translation returned the correct English sentence.
- Streaming STT at 16 kHz with VAD off/on and at 48 kHz with VAD on; actual transcript
  events and session termination.
- Legacy Realtime PCM16, G.711 μ-law and A-law input/output, transcription, audio
  generation and buffer clear. Server VAD also detected an actual speech turn,
  committed it and transcribed the first sentence without a manual commit.
- Wyoming Describe, synthesis and transcription over its actual TCP protocol;
  synthesis returned 16 kHz audio.
- CPU Whisper base/int8, CPU Kokoro and Pocket generation; trusted HTTPS and WSS
  with certificate verification; a configured 1 MiB limit rejected 1 MiB + 1 byte
  with HTTP 413.
- Denoising with the missing optional dependency installed only in the disposable
  environment. It ran, but one recognition changed “fox” to “box”; no quality
  improvement is claimed.
- Real model-load LRU behavior with a one-model limit, and background TTL eviction
  to an empty model list. The one-second-TTL scenario took 51.364 seconds including
  server startup and model loading; background eviction runs every 30 seconds.

The same short text at Kokoro's API speed boundaries produced 12.0917 seconds of
audio at 0.25× and 0.8297 seconds at 4×. Deleting a running 20-file batch returned
204 and the next lookup returned 404. A malformed-audio batch reached overall
`done` with its individual file marked `failed`; the decoder error was recorded.

Live Reader's three modes generated actual audio. The long passage submitted
11,160 characters and received 11,923 audio frames, 187 responses and 595.155 seconds
of PCM. A client that withheld playback acknowledgements received
`playback_stalled` after 30.64 seconds. A new session subsequently generated audio.
Pause/resume, keepalive, malformed events/JSON, cancellation after actual audio and
reconfiguration were also exercised.

The long passage was acknowledged promptly by the caller: this measures generated
audio receipt, not ten minutes of physical playback or word-by-word alignment.

## Slice 6, persistence, Studio and browser

The named Kokoro blend generated audio through HTTP, all three Live Reader modes
and a conversation reading preset. Pocket, Qwen, Chatterbox, CosyVoice and every
Piper default model also generated through a saved exact-model realization.
Unknown identities, unrealized models and missing library references failed
explicitly. Duplicate identities/realizations were rejected. Preset create, read,
update, delete, default, resolution and the shared preset catalog were exercised.

Stopping the isolated Chatterbox worker made its saved named voice return HTTP 503
with a JSON error and no audio. Restarting and reloading the worker restored audio
from the same identity. The failure did not silently generate with another model.

A read-only backup of the production database was migrated only in a disposable
copy. All four existing profiles retained their 13 original columns' values.
Four identities and four realizations were linked, version 6 was recorded, and a
second initialization was idempotent. The original copied database's digest stayed
unchanged. A separate container recreation preserved the synthetic named identity,
preset, speed and reference bytes.

Conversation rendering exercised WAV, MP3, Opus, AAC, FLAC, PCM and M4A. Composer
exercised WAV/MP3 output, real stereo input, solo, mute, offsets, volume, effects,
resampling and source-path containment. Library transcript update, same-name
replacement, audio read and delete; conversation/turn lifecycle; history opt-in,
pagination, deletion and clearing were exercised through actual endpoints.

Browser actions covered microphone/VAD transcription of the synthetic fixture,
WAV upload transcription, Voice Lab recording, suggested transcript, saving without
a transcript, library preview to completion, named-voice save, reading-preset save,
identity selection preserving 1.3× speed, speech generation/playback/download,
three Live Reader modes and pause/resume/stop/clear, model catalog expansion,
default-preset settings, History, saved conversation loading/rendering and Composer
rendering/playback. History showed the named blend's resolved voice recipe.

Authentication, query-token WebSocket authentication, spoofed Upgrade handling,
allowed/denied origins, CORS and rate limiting were exercised against a disposable
server with its own temporary key. Missing/wrong keys returned 401; the configured
burst returned three successes followed by 429 responses.

## Confirmed defects and compatibility gaps

| Finding | Actual observation | Source location |
| --- | --- | --- |
| Current OpenAI Realtime handshake fails | Python SDK 3.22.1's normal `realtime.connect` failed with `no subprotocols supported`. Forcing `realtime` connected. | `src/routes/realtime.py`, `src/realtime/server.py`, `src/realtime/session.py` |
| Realtime GA settings/events incompatible | With a forced connection, requested nested `session.audio` input/output settings were ignored: the reply had `audio: null`, legacy PCM format and default voice. Output used `response.audio.delta`, not GA `response.output_audio.delta`. Actual audio was produced, but the requested contract was not fulfilled. | `src/realtime/server.py`, `src/realtime/session.py` |
| Speech streaming/voice-object compatibility incomplete | `stream_format: audio`, `stream_format: sse` and a voice object with `id` each returned 422. The project's `?stream=true` byte streaming succeeded. | `src/tts/models.py`, `src/services/tts.py` |
| Piper speaker IDs are ignored | Loaded LibriTTS advertised 904 speakers. Two valid IDs and a nonexistent speaker all returned real audio with HTTP 200. The router substitutes the model ID for the requested speaker because Piper declares itself single-speaker, and synthesis does not pass a speaker ID. | `src/tts/router.py`, `src/tts/backends/piper_backend.py` |
| Piper voice listing ignores requested model | Asking for Lessac's voices while LibriTTS was loaded returned LibriTTS's 904 IDs. | `src/tts/router.py`, `src/tts/backends/piper_backend.py` |
| Piper artifact deletion is ineffective | Lessac had six existing cache files; deletion returned `not_found` with no deleted paths; every file remained. | `src/model_manager.py` |
| Piper raw PCM is not OpenAI's 24 kHz PCM contract | Piper produced native 16 kHz or 22.05 kHz audio. Headerless PCM retains that rate instead of the documented OpenAI speech rate. | `src/services/tts.py`, `src/tts/pipeline.py` |
| Malformed transcription input returns 500 | Bytes containing `not-a-wav` were passed to the decoder and returned HTTP 500, not a client validation error. | `src/services/stt.py`, `src/backends/faster_whisper.py` |
| Project Realtime voice override ignored | `create_response(..., 'does-not-exist')` still generated 140,400 audio bytes; synthesis used the session voice. | `src/client/__init__.py`, `src/realtime/server.py`, `src/realtime/session.py` |
| Unknown model lookup fabricates a result | `/v1/models/does-not-exist` returned HTTP 200 with a model object. | `src/services/models.py` |
| Unknown legacy pull model returns 500 | `/api/pull/does-not-exist` exposed the backend's invalid-model ValueError as HTTP 500. | `src/services/models.py` |
| Composer failure leaves a stale spinner | An actual missing-source 400 left the browser showing `Rendering...`; retrying with the real path rendered successfully. | `src/static/app.js`, `renderComposerMix` |

No source repairs were made during this requested acceptance audit.

OpenAI contract references: [speech API](https://developers.openai.com/api/reference/python/resources/audio/subresources/speech/methods/create),
[raw PCM](https://developers.openai.com/api/docs/guides/text-to-speech),
[Realtime conversations](https://developers.openai.com/api/docs/guides/realtime-conversations).

## Cleanup and production state

The five audit containers, two owned anonymous volumes, newly pulled Node image,
isolated browser profile, synthetic download, recordings, reference library,
database copies, temporary certificates, model caches and caller programs were
removed. Their absence was checked. No new test programs were committed.

All four original production containers were `running` and `healthy` after
cleanup. Each retained its original image ID and its original start time from
`2026-09-29T03:47:27Z`. The result artifact records these exact observations.

## Limits and environment issues

- Speaker diarization did not run actual speaker inference: the stock environment
  lacks the optional diarization dependency. Enabling it returned the explicit
  dependency error. No gated-model agreement was accepted.
- Enabling denoising in the stock image without `noisereduce` returned 500. The
  disposable dependency-installed rerun succeeded. This is a deployment dependency
  gate, not proof that the stock image supports that configuration.
- Two audit-only OOM events occurred under the roughly 7.69 GiB Docker RAM limit:
  an early core run with duplicate audit workers/many STT models, and the initial
  CosyVoice 3 load. Fresh isolated retries succeeded. This does not establish safe
  simultaneous residency of all providers.
- Disposable database ownership, a retained read-only SQLite connection, initial
  missing offline assets and an SDK runner attached to an earlier network namespace
  were corrected before their affected operations were rerun. Probe assumptions
  about list shapes, batch terminal states, library filenames and composition IDs
  were corrected. These are not counted as product defects.
- No physical microphone, human listening panel, clone-similarity measurement,
  mobile-browser run, long-lived production load test, native Windows application
  deployment, freshly rebuilt release image or OpenAI-cloud comparison was qualified.

The supported models do perform real inference on this host. A blanket statement
that every advertised feature works, or that the application is a complete current
OpenAI voice API replacement, would be contradicted by the observations above.
