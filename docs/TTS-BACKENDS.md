# TTS Backends

Open Speech discovers backend classes in `src/tts/backends/` and routes requests by the `model`
field. Small compatible backends are baked into the core image with `BAKED_PROVIDERS`. Providers with
conflicting Python or CUDA dependencies use the shipped voice-model bundle. Model weights remain in
persistent cache volumes and can be downloaded or loaded independently.

## Shipped backends

| Backend | Model form | Native sample rate | Native progressive output | Notes |
|---|---|---:|---|---|
| Kokoro | `kokoro` | 24 kHz | No | Default high-quality local voice; supports weighted voice blends |
| Piper | `piper/<voice>` | 16 or 22.05 kHz | No | Small ONNX voices; sample rate is selected from the requested model |
| Pocket-TTS | `pocket-tts` | 24 kHz | Yes | Lightweight backend with built-in voices |
| Supertonic 3 | `supertonic` | 44.1 kHz | Yes (per sentence) | ONNX on GPU or CPU; voices M1–M5, F1–F5, blends; optional `steps` |

The GPU voice-model bundle runs larger providers as private services. Their incompatible ML
dependencies remain outside the core harness image, while model lifecycle stays under the same API
controls and the web UI's Models and Voices views:

| Provider | Model | Voices / identity | Numeric speed | Live Reader |
|---|---|---|---|---|
| Qwen3 | `qwen3/0.6b-custom-voice` | 9 official preset voices | No | Yes |
| Qwen3 | `qwen3/0.6b-base` | Provider-neutral reference WAV + transcript (hidden unless `QWEN3_ENABLE_BASE=true`) | No | No |
| Chatterbox | `chatterbox/regular` | Reference audio | No | Yes |
| Chatterbox | `chatterbox/turbo` | Reference audio longer than 5 seconds | No | Yes |
| Fish Audio S2 | `fish/s2-pro` | Reference audio of at most 30 seconds + exact transcript | No | Yes |
| Breeze TTS 2 | `breeze/tts-2` | Reference audio of at most 30 seconds + exact transcript, or a voice description | No | Yes |

The 1.7B CustomVoice, Base, and VoiceDesign variants remain intentionally hidden until the 0.6B
contract and memory behavior on smaller GPUs are proven. Qwen's 0.6B models do not document a numeric
speed argument, so the harness disables the speed slider instead of silently translating or ignoring it.
The Qwen3 worker hides the 0.6B Base cloning model unless `QWEN3_ENABLE_BASE=true`.
`docker-compose.voice-models.yml` sets it to `true` so the web UI's Voices view can offer every supported
cloning model.
The core catalog still follows the exact model IDs advertised by each provider.

Chatterbox regular and Turbo are distinct upstream runtimes. Both clone an English voice from reference
audio and neither exposes a numeric speed parameter. Turbo additionally accepts native speech tags and
requires a reference longer than five seconds. Fish Audio S2 runs the S2 Pro checkpoint
through a community fork of fish-speech that quantizes it to 4-bit NF4 (`scarxity/fish-speech-int4-patch`,
weights `scarxity/fish-speech-s2-pro-nf4`), about 3.3 GB of VRAM, in float16 so Turing cards can run it. It
outputs 44.1 kHz audio, clones from a reference of at most 30 seconds and its exact transcript, encodes each
reference once on the CPU, and speaks text in segments of up to `FISH_SEGMENT_CHARS` (300) characters. It
refuses a speed other than 1.0, `instructions` and a voice design with `400`. Its first load fetches about
5 GB of weights and runs `torch.compile` for a few minutes; `FISH_COMPILE=false` skips the compile if it
fails on a GPU. Breeze TTS 2 runs through the community C++/ggml port `HoppouAI/Breeze-TTS-2.cpp` (tag `v0.1.7`,
built with its CUDA backend) and its Q8_0 GGUF weights, about 4.4 GB of VRAM; the worker starts the port's
`breeze-server` on a load and stops it on an unload. It speaks English and Chinese at 24 kHz, designs a voice from a
description (`voice_design`, or `instructions` without a recording) or clones one from a recording of at most 30
seconds and its exact transcript, and with a recording takes `instructions` as how to speak. Inline cues such as
`(laugh)` or `[笑]` are part of the text; the worker raises the guidance for text that has one, as the port needs
it to voice them. It refuses a speed other than 1.0, a recording together with a description, and a request with
neither (`voice_required`) with `400`. Provider and model repository revisions are
pinned in the worker source; an upstream branch move cannot silently change a tested image.

`native progressive output` describes the backend capability reported as `streaming`. It does not
indicate whether Live Reader can use a model. Models whose manifests advertise `live_reader` accept
incremental text, segment it according to the selected reading mode, and stream each completed synthesis
result to the client as PCM16 frames. Qwen Base intentionally does not advertise that capability.

The default CPU and CUDA Docker builds bake all four providers. To customize the image:

```bash
docker build \
  --build-arg BAKED_PROVIDERS="kokoro,piper,pocket-tts,supertonic" \
  --build-arg BAKED_TTS_MODELS="kokoro" \
  -t open-speech:local .
```

Keep `/home/openspeech/.cache/huggingface` persistent so model files are not downloaded again after
container replacement.

## Backend contract

Each in-process backend implements the protocol in `src/tts/backends/base.py`:

- `load_model()`, `unload_model()`, `is_model_loaded()`, and `loaded_models()` manage lifecycle.
- `synthesize()` yields mono float32 NumPy arrays.
- `list_voices()` reports selectable voices.
- `sample_rate` provides the default output rate.
- `get_sample_rate(model_id)`, when present, returns the selected model's actual output rate.
- `capabilities` tells the API which controls are valid for backends whose models share one contract.
- `download_model(model_id)`, when present, fetches the model's files into the cache `load_model()`
  reads, without loading the model. Kokoro and Piper have it. `ModelManager` calls it outside the
  router's locks, possibly while the backend is synthesizing, so it must not touch the loaded state.
  Without it, a download loads the model once and unloads it again.
- `validate_voice(voice)`, when present, raises `ValueError` for a voice the backend cannot speak,
  without loading a model. A backend that sets `requires_model_id` gets
  `validate_voice(voice, model_id)`. The router calls it again before each synthesis.
- `validate_language(language)`, when present, raises `ValueError` for a language hint the model
  cannot speak, without loading the model. A backend that sets `requires_model_id` gets
  `validate_language(language, model_id)`, as for `validate_voice`. A backend without it accepts any
  hint.

The router runs `load_model()`, `unload_model()`, and `synthesize()` for one backend one at a time,
but `is_model_loaded()`, `loaded_models()`, and `list_voices()` run alongside them, so model listings
and voice lists do not wait for a running synthesis or stream. A backend therefore copies its loaded
state for those reads under a small lock of its own; iterating a dict that a load is changing can
raise mid-load. This applies to a community backend added with `TTSRouter.register_backend()` too.

The API calls `validate_voice()` and `validate_language()` before any synthesis. `/v1/audio/speech`
(streamed or not) and `/v1/audio/speech/clone` return their message as HTTP 400; Live Reader rejects
the `session.update` with an `invalid_event` error and keeps the previous settings.

Kokoro, Piper (per voice model), Pocket-TTS and the isolated workers, such as Chatterbox, Fish
and Qwen3, refuse a hint whose primary subtag the model does not declare, with
`Unsupported language '<tag>' for <model>. Supported: ...`. A regional tag counts as its primary
subtag (`en-GB` and `EN_us` are `en`). The workers are checked against their manifest's `languages`.
Kokoro maps a regional tag to its own pipeline (`zh-CN` is Mandarin, `pt-PT` is Portuguese; English
tags such as `en-AU` keep the voice's accent). Any hint is accepted, and the check left to the worker,
in two cases: a backend without the hook, and a manifest that declares no languages or lists a name the
core has no code for. Qwen3's manifest names its languages ("Auto", "English", ...). The core counts
each name as its code, so `en`, `zh-CN`, `English` (any case) and `auto` are accepted, and `nl` gets the
same `Unsupported language` message before the worker is asked. Listing "Auto" admits the hint `auto`,
not every hint.

An isolated worker implements worker contract v1: `GET /v1/manifest`, `GET /health`,
`POST /v1/models/load`, `POST /v1/models/unload`, and `POST /v1/audio/speech`, which streams float32
little-endian PCM. The manifest is versioned (`schema_version` 1) and declares capabilities, voices,
language support, sample rate, input limit, and cancellation granularity per model. The core rejects
identity and schema mismatches. A speech response must carry `X-Audio-Format: float32le` and an
`X-Audio-Sample-Rate` equal to the model's manifest `sample_rate`; the core fails any other with
`audio_format_mismatch` or `sample_rate_mismatch`. The shared worker module also sends
`X-Cancellation-Granularity`, which the core does not check. A worker holds one model at a time and
synthesizes only the loaded one. Worker errors keep their status and `code` when the status is 400,
404, 409, 413, 429, or 503; a 422 (request validation) becomes 400 `invalid_request` naming the field;
a 507, or an `out_of_memory` or `cuda_out_of_memory` code, becomes 503 `insufficient_gpu_memory`;
anything else becomes 502. A worker answers 429 `worker_busy` while it finishes the chunk in flight
after a canceled stream, so the core retries a `worker_busy` request for up to about 2 seconds,
honoring `Retry-After` in seconds, before it fails.

A worker also says how much GPU memory its models take, for the core's GPU memory guard (README, "GPU memory
guard"). Each manifest model may give `gpu_memory_mb`, the GPU memory in MiB the model holds once loaded, and
`/health` gives `gpu_memory`: `{"model_mb": ..., "need_mb": ...}`, what the loaded model holds now (torch's reserved
memory) and its expected need, both `null` with no model loaded. Before a load the core checks the card's free memory
against `gpu_memory_mb`, and counts `model_mb` for a loaded model it may unload to make room. Both fields are
optional: for a worker that leaves them out the core uses the registry's size for the model (`gpu_mb` in
`src/model_registry.py`), else 2,000 MiB.

A worker's `max_input_chars` narrows the core input limit for that model; the core checks it before
returning cached audio and the adapter checks again before forwarding. `TTS_MAX_INPUT_LENGTH` covers
the client's input (for SSML, its parsed text with pauses); the pronunciation dictionary's expansions
answer only to the model's own limit. `GET /api/tts/capabilities` reports `max_input_length`
(`TTS_MAX_INPUT_LENGTH`) and, when the model has a limit of its own, `max_input_chars`. Live Reader
caps its segments at that limit and splits a segment that the dictionary lengthened past it into
several syntheses of one response; the Reader splits an over-limit sentence the same way. Speech,
clone, Studio, Wyoming, and realtime (for the text of a `response.create`) refuse over-limit text
with `Input too long for <model>. Max: N characters`. An AI reply on `/v1/realtime` is not refused:
it is spoken in segments, each within the model's limit. Provider URLs come only from
`TTS_EXTERNAL_PROVIDERS`; requests cannot supply or override them. A URL that is not HTTP(S),
carries credentials, a query, or a fragment, or has an invalid port is rejected at startup.

`TTSRouter` in `src/tts/router.py` resolves exact backend names and model prefixes. The audio pipeline
handles PCM conversion, file encoding, and non-live post-processing. Live Reader deliberately bypasses
per-segment normalization and retains bounded edge silence so consecutive sentences do not pump in volume
or run together unnaturally. Streamed `/v1/audio/speech` responses work the same way: they are never
peak-normalized, and with `TTS_TRIM_SILENCE` on they keep up to 50 ms of lead-in and 150 ms of tail. The
stream's synthesis never waits for the client: it runs to the end, unless the client disconnects, and
releases the provider however slowly the client reads.

## Model selection

- Use Kokoro for the normal reading voice and voice blending.
- Use Piper when model size and CPU latency matter more than natural prosody.
- Use Pocket-TTS when its installed voices fit the use case and native progressive output is useful.

The curated core catalog is `src/model_registry.py`. External models appear only if their exact
ID is advertised by the provider. If a configured provider is temporarily unreachable, its known catalog
rows remain marked unavailable rather than becoming selectable. A configured default omitted from the
manifest is likewise shown as unavailable so the misconfiguration is visible. The web UI's Models view
reads a configured worker that does not answer as "Worker offline", one that answers `/health` with
`status: "failed"` and a `failure` as "Worker can't run" with the worker's reason (the row's
`provider_failure`), a worker model whose worker is not in
`TTS_EXTERNAL_PROVIDERS` as "Worker not configured" (with the voice-models Compose file as the hint), and
an in-process provider missing from the image as "Not installed" (with the rebuild command). The running
harness exposes
installed, downloaded, and loaded state plus `default_tts_model` through `GET /api/models`.

The web UI's Speak view starts with the default preset when one is set, otherwise on the model last
picked in that browser or on the configured TTS default, normally Kokoro; an experimental model that was
left loaded never takes its place. Generating, or starting Live, on a TTS model that is not loaded asks
first when another TTS model is loaded, because loading it unloads that model; canceling leaves it
untouched. The **Restore Kokoro** button selects the reading default and loads it if needed.
An explicit load (**Load** in Models, `POST /api/models/{id}/load`, or `POST /v1/audio/models/load`)
unloads every other loaded TTS model first, whatever the GPU size. A speech request (HTTP, Live
Reader, Reader, Studio, Wyoming, or realtime) for an in-process model that is not loaded loads it
next to the loaded ones; TTL and LRU eviction (`OS_MODEL_TTL`, `OS_MAX_LOADED_MODELS`) unload it
later, while it is idle. An isolated worker synthesizes only the model it has loaded; a request for
another one gets HTTP 409 `model_not_loaded`. With `OS_PIN_DEFAULT_MODELS=true` (the default), TTL
and LRU eviction never unload `TTS_MODEL` or `STT_MODEL`, and these pinned defaults do not count
toward `OS_MAX_LOADED_MODELS`; an explicit unload, or the explicit load of another model of the same
type, still unloads them. `TTS_PRELOAD_DEFAULT=true` (the default) loads `TTS_MODEL` in the
background at startup when its provider is installed. Eviction never unloads a model while it is
synthesizing, streaming, or loading; it tries again on a later pass. **Download**
(`POST /api/models/{id}/download`) fetches Kokoro and Piper files through `download_model()` and
does not load them. Pocket-TTS and the isolated workers still load the model once and unload it,
because worker contract v1 has no download-only call; a model that was already loaded stays loaded.
Unknown model IDs on the unified `/api/models` lifecycle routes return `unknown_model` instead of
being guessed to be STT. Legacy `/api/ps` STT routes retain their existing behavior.

## Voice provider deployment

The voice provider workers are optional and off by default. Each one in `docker-compose.voice-models.yml` has a
Compose profile of its own, named after it (`chatterbox`, `qwen3`, `fish`, `breeze`), so layering the file on the core
GPU harness starts and builds none of them until its profile is named:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  -f docker-compose.voice-models.yml --profile qwen3 up -d --build
```

Name `--profile chatterbox`, `--profile fish` or `--profile breeze` for the others, repeat the flag for more than
one, or use `--profile "*"` for all four. `COMPOSE_PROFILES=qwen3,chatterbox` (in the shell or in `.env`) does the same as the
flags and also covers later commands such as `down`. The core's `TTS_EXTERNAL_PROVIDERS` names all four workers,
so a worker whose profile is off reads "Worker offline" and its models can't be loaded (`503
provider_unavailable`); the core starts and serves without it. It names each by a network alias that only that
worker's container holds, `http://qwen3.voice-workers.internal.:8200`, `http://chatterbox.voice-workers.internal.:8210`,
`http://fish.voice-workers.internal.:8230` and `http://breeze.voice-workers.internal.:8240`, not by the bare service name: with a profile off no container
holds the name, and Docker's DNS then asks the host's DNS servers, where a device on the LAN called `chatterbox` (a
router registers each DHCP client's name, bare and as `chatterbox.lan`) would answer and be sent the clone
recordings. No LAN registers a name under `voice-workers.internal`, the public DNS has no `.internal`, and the
trailing dot keeps the host's search domains from being appended. Only Qwen3 Custom Voice
(`qwen3/0.6b-custom-voice`) has built-in voices; Chatterbox, Qwen3 Base, Fish Audio S2 and Breeze TTS 2 clone
from a recording.

A worker container that an earlier `up` started keeps running, on its old image, until it is stopped: `up` leaves
alone a worker whose profile isn't named (it isn't an orphan), and `down` without its profile doesn't stop it. A
`--profile` flag replaces `COMPOSE_PROFILES` rather than adding to it. So on a host that ran the file before the
profiles, stop all of them once before the first `up` that names a profile (the core stops too, and that `up` starts
it again), and stop one no longer wanted with the same `-f` files and `stop chatterbox`:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  -f docker-compose.voice-models.yml --profile "*" down
```

This bundle replaces the older standalone Qwen3 Compose file. The provider settings (image tags,
`QWEN3_*`, `CHATTERBOX_*`, `FISH_*`, `BREEZE_*`, `HF_TOKEN`) are listed in the README under
[Voice provider settings](../README.md#voice-provider-settings).

Qwen3, Chatterbox, Fish Audio S2 and Breeze TTS 2 remain reachable only from the private Compose network. Each has its
own persistent model cache. Keeping the provider services running does not load model weights onto the
GPU. In the web UI's Models view, **Download** caches weights on disk (by loading the model once and
unloading it) and **Load** activates a model; **Test clone** in Voices › Recordings performs the load
automatically. An explicit load unloads the
current TTS model before activating a different one, including switches between providers.

Core-to-provider HTTP bypasses environment proxies and refuses redirects so text and reference audio
stay on the configured origin. Torch/CUDA versions and model repository commits are pinned.

The four workers share their HTTP scaffolding (request models, typed errors, `/health`, model
load/unload, float32 streaming, the reference-conditioning cache, sentence rules, and language-tag
lookup) through `providers/_common/worker.py`. Each image therefore builds with `providers/` as its
context and `<name>/Dockerfile` as its Dockerfile, and runs `providers.<name>.app:app`; the build
imports the worker once so a broken layout fails the build instead of the container. To build one by
hand, run `docker build -f providers/<name>/Dockerfile providers`; `docker build providers/<name>`
no longer works. Rebuild existing worker images after an upgrade (`up -d --build`).

If CUDA runs out of memory while a worker loads a model or synthesizes, the worker frees the CUDA
cache, stays healthy, and answers HTTP 507 with `{"error": {"code": "out_of_memory", ...}}`. A failed
load leaves no model loaded; a failed synthesis keeps the model loaded for the next request. The core
reports both as `insufficient_gpu_memory` (HTTP 503) naming the model. A synthesis error can set the
HTTP status only before the first audio chunk is sent; a later one ends the stream early. When a request ends
early (an error or a client disconnect), the worker finishes or stops its model work before it accepts
the next request.

If the worker's torch build has no code for the GPU (for example an RTX 50-series card, `sm_120`), every
load the worker gets answers `503 cuda_arch_unsupported`, and its `/health` reports `status: "failed"`
with the same `failure`:
`{"code": "cuda_arch_unsupported", "message": "Torch wheel has no code that runs on sm_120 (built for: ...)"}`.
`gpu.capability` and `gpu.arch_list` stay in the gpu block. Compose shows the worker unhealthy, and the
core counts the provider unavailable. A load, download or prefetch of its models then answers
`503 provider_unavailable` with the worker's reason in the message
(`Provider 'chatterbox' is configured but its worker can't run: Torch wheel has no code ...`), and the
`/api/models` rows carry it as `provider_failure: {code, message}`, which the web UI's Models view shows
as "Worker can't run". The core passes on any failed health's `failure` this way, as one printable line
of at most 300 characters: also `cuda_query_failed`, and for Qwen3 a poisoned CUDA context or a
`QWEN3_DTYPE` that is invalid or that the GPU can't run.

Qwen generation is always non-streaming at the model API because its alternate mode only simulates
streaming text input. The worker generates each sentence separately and applies the harness's sentence
rules, so a period after an honorific, an initial, or a list number (`Mr.`, `Dr.`, `J. R. R.`, `e.g.`,
`1.`) does not end the sentence. Both Qwen3 0.6B models receive a per-segment codec-token limit instead
of Qwen's 2048-token default. Segments are bounded with script-weighted units so CJK, kana, Hangul, and
full-width text receive more budget than Latin text. `QWEN3_MAX_SEGMENT_UNITS` defaults to 400 and the
host-level `QWEN3_MAX_NEW_TOKENS_CEILING` defaults to 1200 and must be at least 192. Lowering the
token ceiling also lowers the effective segment-unit limit, so each segment retains the same duration
headroom. If output reaches that safety limit, the worker returns
`generation_limit_reached` and discards the partial audio rather than presenting a cut-off sentence as
successful synthesis.

`QWEN3_DTYPE=auto` is the safe default. It uses float32 on pre-Ampere GPUs (compute capability below
8.0, such as Turing cards) because Qwen sampling produced non-finite fp16 probabilities in live
validation, and bfloat16 on compute capability 8.0 or newer. `float32`, `float16`, and `bfloat16` may be selected explicitly,
but float16 is experimental and bfloat16 is rejected below capability 8.0. The resolved dtype and
the loaded model component dtypes appear in `/health`. Float32 costs more VRAM and throughput.
If CUDA reports a poisoned context (for example, a device-side assertion), the worker returns the
typed `cuda_context_failed` error, marks health failed, and exits after the response so Compose can
start a clean process.

The CustomVoice model accepts only its official voice IDs: `Vivian`, `Serena`, `Uncle_Fu`, `Dylan`,
`Eric`, `Ryan`, `Aiden`, `Ono_Anna`, and `Sohee`. No OpenAI voice aliases are mapped silently.

Qwen's `language` hint accepts the model's language names (`English`, any case), ISO 639-1 codes
(`en`), and regional tags, which fall back to their primary subtag (`zh-CN` is Chinese, `pt-BR` is
Portuguese). An omitted hint means `Auto`. The worker rejects a language Qwen does not support with
HTTP 400 `language_unsupported` before any generation; through Open Speech, the core's check (see above)
refuses the same hints first.

For isolated performance measurement, stop providers that are not under test. This is a benchmark
procedure only; it is not required for normal voice-cloning use because an idle provider has no model
weights resident on the GPU.

An idle provider can still hold host RAM. After unloading, the Chatterbox worker keeps its CUDA
context and libraries, about 1.5 GiB, and on Docker Desktop's default VM (about 7.7 GiB) that was
enough to get another worker OOM-killed while it loaded. The voice-models Compose file therefore sets
`CHATTERBOX_RESTART_AFTER_UNLOAD_S=10`: ten seconds after an unload with no new load, the worker exits
and Compose restarts it empty. Switching between `chatterbox/regular` and `chatterbox/turbo` sends the
new load within that window, so it does not restart. While it restarts, Chatterbox loads return
`provider_unavailable` until it is healthy again: about 35 seconds after the unload on an RTX 2070
SUPER host, and about a minute if another model is loading at the same time.

The Fish Audio S2 worker can't free its model at all: the fork keeps it on a thread of its own and has
no teardown. So it exits `FISH_RESTART_AFTER_UNLOAD_S` (1) seconds after every unload, and after a
failed load, and Compose restarts it empty. A load in that second answers `503 worker_restarting`,
since a second copy of the model would not fit beside the first.

The Breeze TTS 2 worker runs no CUDA code of its own: the port's `breeze-server` holds the model, as a child
process that runs only while the model is loaded. An unload stops it, which hands its GPU memory back at once,
with no restart. The worker reads the GPU with `nvidia-smi`, and `/health` reports as `gpu_memory.model_mb` the
memory the card lost across the load (the manifest's 4450 MiB when that can't be read), so the core's GPU memory
guard counts it. A server that runs out of memory while loading answers `507 out_of_memory`; one that exits while
loaded leaves the worker empty, and the next use loads it again. Its image is built for the CUDA architectures in
the build argument `BREEZE_CUDA_ARCHITECTURES` (`75`, Turing, with code newer GPUs compile at their first load); on
a GPU the build can't run, `/health` is `failed` with `cuda_arch_unsupported` and names the argument to rebuild with.

Save the reference once, as a recording in the web UI's Voices view. Chatterbox uses the stored WAV but
does not consume the
transcript. Qwen3 Base and Fish Audio S2 consume both and should not be scored until the transcript is
checked word for word. Breeze TTS 2 consumes both too, and also takes delivery `instructions`; no other worker
does. Chatterbox Turbo tags and Breeze cues belong in the synthesis text.

In the web UI, **Voices › Clone a voice** records or uploads a reference, checks its quality, verifies the
exact transcript, can test it before saving, and saves it to the library. Browser recordings and
supported uploads are converted to mono PCM16 WAV before storage. The optional STT suggestion stays marked
as an unverified draft until the user confirms it word for word. **Voices › Design a voice** describes a voice,
has a model that designs voices (Breeze TTS 2) speak an editable sample through `POST /api/voices/design`, and saves
the sample to the library with its words as the transcript. In **Voices › Recordings**, a saved
reference can be previewed, tested with a clone model, saved as a preset, or used in Speak, whose
**Reference recording** control shows whenever the selected model supports cloning.

The same provider-neutral flow is available through the API:

```bash
curl -k -X POST https://localhost:8100/api/voices/library \
  -F 'name=narrator' \
  -F 'transcript=These are the exact words spoken in the reference clip.' \
  -F 'audio=@reference.wav;type=audio/wav'

curl -k https://localhost:8100/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"qwen3/0.6b-base",
    "input":"This is a cloning test.",
    "voice":"narrator",
    "voice_library_ref":"narrator",
    "response_format":"wav"
  }' \
  -o qwen-clone.wav
```

The logical library asset is not owned by Qwen or Kokoro. Each worker caches the conditioning it
derives from a reference (Qwen clone prompts, Chatterbox conditionals,
Fish prompt tokens),
so Live Reader does not re-analyze the reference for every sentence. The cache is a per-worker LRU,
8 entries by default (`QWEN3_PROMPT_CACHE_SIZE`, `CHATTERBOX_REFERENCE_CACHE_SIZE`,
`FISH_REFERENCE_CACHE_SIZE`), keyed by the model, the audio hash, and the hash of the text the
conditioning depends on (the transcript), and it is cleared when the
model unloads. Requests using reference audio remain ineligible for the shared TTS output cache.

Named voices have stable identities and explicit realizations for each exact model. A realization
contains its provider voice/blend and optional provider-neutral reference asset. Studio profiles are
reading presets with a model, optional named-voice link, speed, effects, instructions, and format.
Neither a Kokoro blend nor a reference recording creates a realization for another model automatically.
Built-in voice packs appear only after that provider/model is selected. See
[Named voices and reading presets](VOICE-IDENTITIES.md) for migration and API use.

Use `scripts/benchmark_tts.py` for comparable one-shot completion/RTF measurements or Live Reader TTFA.
TTFA is reported only from the first Live Reader PCM delta; one-shot HTTP reports time-to-complete.

## Validation

The registry contains 15 STT and 37 TTS IDs: Kokoro, Pocket-TTS, Supertonic, 28 Piper voices, two Qwen
IDs, two Chatterbox IDs, one Fish Audio S2 ID, and one Breeze TTS 2 ID. External rows become selectable only when their provider
advertises the exact ID. Hidden, unavailable, unloaded, and intentionally skipped models are not
inference passes.

Run TTS contract checks from the core service so the command can reach the private providers; the worker a
check names must be turned on with its profile (see [Voice provider deployment](#voice-provider-deployment)):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.voice-models.yml \
  exec open-speech python scripts/tts_conformance.py \
  --url https://localhost:8100 --insecure --probe-rejections --synthesize \
  --worker-url http://qwen3.voice-workers.internal.:8200 --provider qwen3 \
  --model kokoro --model pocket-tts --model piper/en_US-lessac-medium \
  --model qwen3/0.6b-custom-voice
```

The command prints JSON and exits nonzero when a check fails. Rejection probes use minimal uncached
inputs; omit `--probe-rejections` if a provider must not receive deliberate invalid requests.
`--synthesize` requests short uncached WAV output. Unloaded external models and checks that require a
controlled fixture remain explicit skips. Live Reader, cancellation, worker outages, incompatible
manifests, and forced generation limits require separate gates.

Run the same contract against each provider after loading the selected model and saving a recording
named `narrator` (Voices in the web UI, or `POST /api/voices/library`):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  -f docker-compose.voice-models.yml exec open-speech \
  python scripts/tts_conformance.py --url https://localhost:8100 --insecure \
  --probe-rejections --synthesize --voice-library-ref narrator \
  --worker-url http://chatterbox.voice-workers.internal.:8210 --provider chatterbox \
  --model chatterbox/regular --model chatterbox/turbo

docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  -f docker-compose.voice-models.yml exec open-speech \
  python scripts/tts_conformance.py --url https://localhost:8100 --insecure \
  --probe-rejections --synthesize --voice-library-ref narrator \
  --worker-url http://fish.voice-workers.internal.:8230 --provider fish \
  --model fish/s2-pro

docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  -f docker-compose.voice-models.yml exec open-speech \
  python scripts/tts_conformance.py --url https://localhost:8100 --insecure \
  --probe-rejections --synthesize --voice-library-ref narrator \
  --worker-url http://breeze.voice-workers.internal.:8240 --provider breeze \
  --model breeze/tts-2
```

## Adding a backend

1. Add `src/tts/backends/<name>.py` implementing the backend protocol, including the thread-safe
   status reads and, where they apply, the optional `download_model()` and `validate_language()` hooks.
2. Add its dependency to `pyproject.toml` and the Docker provider installation map.
3. Register its model identifiers in `src/model_registry.py`.
4. Add backend, routing, capability, sample-rate, and API tests.
5. Update this file and the backend table in `README.md` with verified behavior only.
