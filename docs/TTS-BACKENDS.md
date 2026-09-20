# TTS Backends

Open Speech discovers backend classes in `src/tts/backends/` and routes requests by the `model`
field. Backends are baked into the image with `BAKED_PROVIDERS`; model weights remain in persistent
cache volumes and can be downloaded or loaded independently.

## Shipped backends

| Backend | Model form | Native sample rate | Native progressive output | Notes |
|---|---|---:|---|---|
| Kokoro | `kokoro` | 24 kHz | No | Default high-quality local voice; supports weighted voice blends |
| Piper | `piper/<voice>` | 16 or 22.05 kHz | No | Small ONNX voices; sample rate is selected from the requested model |
| Pocket-TTS | `pocket-tts` | 24 kHz | Yes | Lightweight backend with built-in voices |

Optional providers can run as private HTTP workers. Their ML dependencies, model cache, and process
lifecycle remain outside the core harness image. The first canary is Qwen3-TTS:

| Provider | Model | Voices / identity | Numeric speed | Live Reader |
|---|---|---|---|---|
| Qwen3 | `qwen3/0.6b-custom-voice` | 9 official preset voices | No | Yes |
| Qwen3 | `qwen3/0.6b-base` (opt-in) | Provider-neutral reference WAV + transcript | No | No |

The 1.7B CustomVoice, Base, and VoiceDesign variants remain intentionally hidden until the 0.6B
contract and RTX 2070 memory behavior are proven. Qwen's 0.6B models do not document a numeric speed
argument, so the harness disables the speed slider instead of silently translating or ignoring it.
The 0.6B Base cloning model is also hidden by default because it has not produced successful audio on
the tested RTX 2070 SUPER. Set `QWEN3_ENABLE_BASE=true` only on a deployment where you intend to test
it; restarting the worker changes its manifest, and the core catalog follows the exact advertised IDs.

`native progressive output` describes the backend capability reported as `streaming`. It does not
indicate whether Live Reader can use the backend. Live Reader accepts incremental text for every shipped
backend, segments it according to the selected reading mode, and streams each completed synthesis result
to the client as PCM16 frames.

The default CPU and CUDA Docker builds bake all three providers. To customize the image:

```bash
docker build \
  --build-arg BAKED_PROVIDERS="kokoro,piper,pocket-tts" \
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

An isolated worker exposes `GET /v1/manifest`, `/health`, model load/unload, and float32 PCM synthesis.
The manifest is versioned and declares capabilities, voices, language support, sample rate, input limit,
and cancellation granularity per model. The core rejects identity, schema, sample-rate, and audio-format
mismatches. A worker's `max_input_chars` narrows the core input limit for that model; the core checks it
before returning cached audio and the adapter checks again before forwarding. Live Reader checks each
completed synthesis segment, not the whole accumulating text. Provider URLs come only from
`TTS_EXTERNAL_PROVIDERS`; requests cannot supply or override them.

`TTSRouter` in `src/tts/router.py` resolves exact backend names and model prefixes. The audio pipeline
handles PCM conversion, file encoding, and non-live post-processing. Live Reader deliberately bypasses
per-segment normalization and retains bounded edge silence so consecutive sentences do not pump in volume
or run together unnaturally.

## Model selection

- Use Kokoro for the normal reading voice and voice blending.
- Use Piper when model size and CPU latency matter more than natural prosody.
- Use Pocket-TTS when its installed voices fit the use case and native progressive output is useful.

The curated core catalog is `src/model_registry.py`. Optional external models appear only if their exact
ID is advertised by the worker. If a configured worker is temporarily unreachable, its known catalog
rows remain marked unavailable rather than becoming selectable. A configured default omitted from the
manifest is likewise shown as unavailable so the misconfiguration is visible. The Models tab distinguishes
"Worker unavailable" from an in-process provider that is not installed. The running harness exposes
installed, downloaded, and loaded state plus `default_tts_model` through `GET /api/models`.

The Speak tab starts on the configured TTS default, normally Kokoro, even if an experimental model
was left loaded. A browser selection remains selected for that session. Generating or starting Live
Reader on a different TTS model confirms that the loaded model will be unloaded; canceling leaves it
untouched. The **Restore Kokoro** button selects the reading default and loads it if needed.
On the tested 8 GB GPU only one TTS model is kept loaded at a time; the default is not pinned in VRAM.
Unknown model IDs on the unified `/api/models` lifecycle routes return `unknown_model` instead of
being guessed to be STT. Legacy `/api/ps` STT routes retain their existing behavior.

## Isolated Qwen3 canary

The normal Compose stack contains no Qwen worker and shows no Qwen choices. Start the canary explicitly:

```bash
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  -f docker-compose.qwen3.yml \
  --profile qwen3 up -d --build
```

The worker has no published host port. Open Speech reaches it only as `http://qwen3:8200` on the
Compose network. It uses its own named Hugging Face cache and loads only one Qwen model at a time.
Removing `docker-compose.qwen3.yml` removes the canary while leaving the external-provider harness
contract intact. Core-to-worker HTTP bypasses environment proxies and refuses redirects so text and
reference audio stay on the configured origin. Torch/CUDA versions and both model repository commits
are pinned; the worker releases its operation lock after a bounded abandoned-stream wait.

Qwen generation is always non-streaming at the model API because its alternate mode only simulates
streaming text input. Both canary models receive a per-segment codec-token limit instead of Qwen's
2048-token default. Segments are bounded with script-weighted units so CJK, kana, Hangul, and full-width
text receive more budget than Latin text. `QWEN3_MAX_SEGMENT_UNITS` defaults to 400 and the host-level
`QWEN3_MAX_NEW_TOKENS_CEILING` defaults to 1200 and must be at least 192. Lowering the token ceiling
also lowers the effective segment-unit limit, so each segment retains the same duration headroom.
If output reaches that safety limit, the worker returns
`generation_limit_reached` and discards the partial audio rather than presenting a cut-off sentence as
successful synthesis.

`QWEN3_DTYPE=auto` is the safe default. It uses float32 on pre-Ampere GPUs such as the RTX 2070
because Qwen sampling produced non-finite fp16 probabilities in live validation, and bfloat16 on
compute capability 8.0 or newer. `float32`, `float16`, and `bfloat16` may be selected explicitly,
but float16 is experimental and bfloat16 is rejected below capability 8.0. The resolved dtype and
the loaded model component dtypes appear in `/health`. Float32 costs more VRAM and throughput;
the 0.6B canary is the supported 8 GB Turing target, while 1.7B models remain deferred there.
If CUDA reports a poisoned context (for example, a device-side assertion), the worker returns the
typed `cuda_context_failed` error, marks health failed, and exits after the response so Compose can
start a clean process.

The CustomVoice model accepts only its official voice IDs: `Vivian`, `Serena`, `Uncle_Fu`, `Dylan`,
`Eric`, `Ryan`, `Aiden`, `Ono_Anna`, and `Sohee`. No OpenAI voice aliases are mapped silently.

If Base is explicitly enabled, upload a provider-neutral WAV with the exact spoken transcript:

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

The logical library asset is not owned by Qwen or Kokoro. Provider-specific prompt caches are keyed by
the model, audio hash, and transcript hash inside the disposable worker. Requests using reference audio
remain ineligible for the shared TTS output cache.

Studio profiles remain render presets: they may select a provider, model, voice ID, speed, effects, and
an optional provider-neutral reference asset. A profile named `Will` therefore does not make `Will` a
Kokoro-owned voice identity; the reusable identity is the library asset, while each profile describes one
provider's rendering of it. Built-in provider voice packs appear only after that provider/model is selected.

Use `scripts/benchmark_tts.py` for comparable one-shot completion/RTF measurements or Live Reader TTFA.
TTFA is reported only from the first Live Reader PCM delta; one-shot HTTP reports time-to-complete.

## Harness conformance report

Run the report from the core container so it can reach both the harness and a private worker. It
prints JSON and exits nonzero if a tested check fails. By default it makes metadata-only GET
requests and does not request synthesis. The command below also opts into invalid-request probes:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.qwen3.yml \
  --profile qwen3 exec open-speech python scripts/tts_conformance.py \
  --url https://localhost:8100 --insecure --probe-rejections \
  --worker-url http://qwen3:8200 --provider qwen3 \
  --model kokoro --model qwen3/0.6b-custom-voice
```

Invalid-request probes should reject before inference, but a broken provider could accept one and
start synthesis. Control probes use one character and disable the output cache. The input-limit probe
sends the advertised limit plus one character (up to 8,192) and also disables the cache; if validation
is broken, that request could be expensive. This tests the core rejection; worker-side validation is
covered by focused tests. Omit `--probe-rejections` when this risk is unacceptable.
Add `--synthesize` to test short, uncached WAV output.
An in-process backend may load lazily; an unloaded external model is reported as skipped. For a clone-only
model, also supply an existing `--voice-library-ref`; otherwise its audio check is explicitly
skipped. Real inference can take minutes and may fail on a model that advertises a capability but
cannot deliver it on the current hardware. The report does not exercise Live Reader, cancellation,
worker outages, truncated streams, incompatible worker versions, or forced generation limits; those
checks are marked `skip` rather than `pass`. The report does verify a worker's advertised generation
ceiling and segment bounds as configuration evidence, not proof that a real generation was cut off.
Forced-limit behavior remains covered by the focused worker test or a controlled manual run.
Qwen Base cloning has not produced successful audio on the tested RTX 2070 SUPER; the worker returned
`generation_limit_reached` at its safety floor. Kokoro remains the proven fast reading baseline.

## Validating the harness

The registry contains 14 STT and 34 TTS IDs: Kokoro, Pocket-TTS, 30 Piper voices, and two optional
Qwen IDs. The Windows deployment tested on 2026-09-19 advertised 14 STT and 33 TTS IDs; the Qwen
0.6B Base clone was correctly absent because its worker did not advertise it. A hidden or unavailable
model is **not** an inference pass. The conformance script takes explicit `--model` arguments and
only checks TTS. It does not enumerate this inventory or test STT.

### Slices 1–5: observed acceptance

The Windows checkout was `22d93d7` on an RTX 2070 SUPER. The served Speak UI matched that source
after accounting for Windows line endings. The core-to-worker report for Qwen CustomVoice found
10 passes, no failures, and six explicit skips. A separate four-model core report with rejection
probes found 17 passes, no failures, and 27 skips. These reports alone do not prove audio or Live
Reader: the checks below used real synthesis through the deployed API.

| Slice | Contract checked | Result |
|---|---|---|
| 1: routing and capabilities | Exact model routing, voice catalogs, unsupported controls | Focused tests passed; the four active providers reported capabilities and rejected invalid controls where applicable |
| 2: isolated Qwen worker | CustomVoice can load and synthesize without changing the core dependency set | WAV and Live Reader PCM produced; the Base clone remained hidden and was not retested |
| 3: conformance | Metadata, rejections, worker manifest, generation policy | Reports above had zero failures; intentional skips remain visible |
| 4: exact model gating | Base absent unless advertised, worker input-limit rejection | Base was absent from the live catalog; the core-to-worker report passed its worker checks |
| 5: reading safeguards | Kokoro default, explicit switch warning, cancellation, restore | Canceling a Qwen selection left Kokoro loaded; Restore Kokoro selected and loaded it; Kokoro was reloaded after each provider and produced audio again after Qwen |

Single-run timings below use different short sentences and are functional measurements, not a
quality ranking. RTF is generation time divided by audio duration; below 1 means faster than real
time. One-shot HTTP does not report time to first audio.

| Model | One-shot audio / completion / RTF | Live Reader first PCM / RTF | Status |
|---|---|---|---|
| Kokoro | 7.74 s / 0.42 s / 0.054 | 0.31 s / 0.068 | Baseline; a 22.82 s blended reading at 1.2× speed also succeeded |
| Piper `en_US-lessac-medium` | 2.08 s / 0.36 s / 0.173 | 0.12 s / 0.064 | One of 30 Piper models tested |
| Pocket-TTS | 2.76 s / 1.97 s / 0.715 | 0.39 s / 0.536 | One built-in voice tested |
| Qwen `0.6b-custom-voice` | 3.61 s / 17.41 s / 4.816 | 14.50 s / 4.985 | Works, but too slow here for uninterrupted reading |

The sampled WAV files contained nonzero mono PCM16 audio at the advertised rate; this is not a
listening-quality judgment. Live Reader produced PCM for all four providers. Kokoro cancellation
returned `response.cancelled`, and another utterance completed in the same session. The default and
tiny faster-whisper models transcribed a generated test clip; recognizable words were returned, but
neither run was an accuracy benchmark. STT models loaded for the check were unloaded afterward, and
Kokoro was the only model left loaded. An existing saved profile could be selected in Speak; its
rendered audio and unavailable-model behavior were not tested live.

An isolated full repository test run of `22d93d7` with the temporary-data fixture below reached
**865 passed, 2 failed, 2 skipped**. Both
failures are stale expectations in `tests/test_unified_api.py`: the two unknown-model endpoints
return HTTP 404 and `error.code=unknown_model`, matching the central HTTP error handler, while the
tests still expect `detail.code`. The skips require Torch for optional Kokoro checks. Do not describe
the full suite as green until the assertions are aligned and the suite is rerun.

### Completing the model matrix

1. Record the Git revision, container image identity, worker manifest, hardware, and the live
   `/api/models` inventory. Use exact advertised IDs; treat hidden Qwen Base and any missing
   provider as skipped with a reason, not passed.
2. Run the isolated full suite first. Run the conformance report with `--probe-rejections` and,
   after loading each model, `--synthesize`. A 409 or any `skip` is not successful inference.
   From inside the core container, include `--worker-url` and `--provider` for worker checks.
3. For **each advertised TTS ID**, load one at a time, synthesize an uncached short sample, check
   sample rate, duration, non-silent audio and the voice/control contract, then exercise Live Reader
   when advertised. Repeat a representative long reading, cancellation/recovery, and a listening
   comparison for contenders; measure cold and warm latency, RTF, and GPU memory. The 29 untested
   Piper variants and other built-in/preset voices still need this pass. Restore Kokoro and prove
   reading after every switch.
4. For **each of the 14 STT IDs**, use a known WAV fixture with a reference transcript, check
   transcription, format and timestamps, and record accuracy and latency by device. Large model
   downloads and load times require a separate budget; an installed provider is not a downloaded
   or validated model.
5. Check unavailable-worker behavior, failed unload, aborted worker streams, incompatible
   manifests, forced generation limits, saved-profile behavior, and UI model-switch races with
   controlled fixtures. A live worker-outage drill requires a planned service interruption. Keep
   those cases marked unverified until exercised. Finally confirm the restored Kokoro default and
   a short STT request before promoting an image.

Use `scripts/benchmark_tts.py` with explicit `--model`, `--voice`, `--output`, and `--results` paths
for one-shot and `--mode live` runs. It writes WAV and JSON files; keep them in a disposable test
directory. Test artifacts and cloned voice references must not be added to the repository. The
conformance report does not listen to audio, verify a saved profile, or qualify continuous playback.

## Adding a backend

1. Add `src/tts/backends/<name>.py` implementing the backend protocol.
2. Add its dependency to `pyproject.toml` and the Docker provider installation map.
3. Register its model identifiers in `src/model_registry.py`.
4. Add backend, routing, capability, sample-rate, and API tests.
5. Update this file and the backend table in `README.md` with verified behavior only.
