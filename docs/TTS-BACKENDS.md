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
The 0.6B Base cloning model remains hidden by default. A scripted reference produced one successful
clone on the tested RTX 2070 SUPER on 2026-09-20; that single result does not qualify voice quality or
reliability. Set `QWEN3_ENABLE_BASE=true` only on a deployment where you intend to test it; restarting
the worker changes its manifest, and the core catalog follows the exact advertised IDs.

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

If Base is explicitly enabled, use the **Voice Lab** tab to record or upload a reference, verify the
exact transcript, save it, preview the stored audio, and run a clone test. Voice Lab converts browser
recordings and supported uploads to mono PCM16 WAV before storage. Its optional STT suggestion remains
unverified until the user confirms it word-for-word. Saved references can be linked to Studio profiles
or selected directly in Speak; the reference control stays visible whenever the selected model supports
cloning.

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

The logical library asset is not owned by Qwen or Kokoro. Provider-specific prompt caches are keyed by
the model, audio hash, and transcript hash inside the disposable worker. Requests using reference audio
remain ineligible for the shared TTS output cache.

Studio profiles remain render presets: they may select a provider, model, voice ID, speed, effects, and
an optional provider-neutral reference asset. A profile named `Will` therefore does not make `Will` a
Kokoro-owned voice identity; the reusable identity is the library asset, while each profile describes one
provider's rendering of it. Built-in provider voice packs appear only after that provider/model is selected.

Use `scripts/benchmark_tts.py` for comparable one-shot completion/RTF measurements or Live Reader TTFA.
TTFA is reported only from the first Live Reader PCM delta; one-shot HTTP reports time-to-complete.

## Validation

The registry contains 14 STT and 32 TTS IDs: Kokoro, Pocket-TTS, 28 Piper voices, and two optional
Qwen IDs. With Qwen CustomVoice available and Qwen Base disabled, the normal live catalog contains
14 STT and 31 TTS IDs. Hidden, unavailable, and intentionally skipped models are not inference
passes.

Run TTS contract checks from the core container so the command can reach the private Qwen worker:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f docker-compose.qwen3.yml \
  --profile qwen3 exec open-speech python scripts/tts_conformance.py \
  --url https://localhost:8100 --insecure --probe-rejections --synthesize \
  --worker-url http://qwen3:8200 --provider qwen3 \
  --model kokoro --model pocket-tts --model piper/en_US-lessac-medium \
  --model qwen3/0.6b-custom-voice
```

The command prints JSON and exits nonzero when a check fails. Rejection probes use minimal uncached
inputs; omit `--probe-rejections` if a provider must not receive deliberate invalid requests.
`--synthesize` requests short uncached WAV output. Unloaded external models and checks that require a
controlled fixture remain explicit skips. Live Reader, cancellation, worker outages, incompatible
manifests, and forced generation limits require separate gates.

The Windows RTX 2070 SUPER baseline is:

- Full repository suite: 883 passed, 9 skipped, 0 failed.
- STT: 14/14 advertised models produced timestamped text from the short reference recording.
- Long-form STT: Turbo and both distilled English models covered the full 274.67-second recording.
- TTS: 31/31 live advertised IDs produced non-silent mono PCM16 WAVs and completed Live Reader.
- Kokoro: 52/52 voices, including Japanese and Chinese, synthesized from the immutable image.
- Recovery: a cancelled Live Reader session accepted and completed another utterance; Kokoro was
  restored after the matrix and the health endpoint returned 200.
- Voice Lab: Chrome converted a 14.549-second stereo FLAC reference to 48 kHz mono PCM16 WAV, and
  Faster Whisper on CUDA/float16 returned the complete 46-word reference draft.
- Qwen Base cloning: cold model load took 74.4 seconds. The first 3.588-second clone completed in
  42.9 seconds (RTF 11.95); a warm 3.221-second clone completed in 28.4 seconds (RTF 8.80).
- Clone intelligibility: Windows Faster Whisper scored the two known generated texts at 8.3% and
  18.2% WER. Both WAVs were non-silent mono 24 kHz PCM16 with no clipped samples.
- Kokoro baseline: after a 44.3-second cold load, a warm 4.324-second render completed in 0.48
  seconds (RTF 0.11).

These checks establish packaging, routing, inference, output format, long-file coverage, clone-text
intelligibility, and basic recovery. They do not establish subjective voice likeness or human-speech
accuracy. The reference transcript was produced and rechecked by the same STT model, so its zero-error
self-check is not independent ground truth. A defensible human-speech WER comparison still needs an
audited transcript, and Qwen Base still needs a listening comparison. Sustained playback,
saved-profile rendering, model-switch races, forced
generation limits, incompatible manifests, and controlled worker outage behavior still require
separate qualification.

Use `scripts/benchmark_tts.py` with explicit `--model`, `--voice`, `--output`, and `--results` paths
for one-shot and Live Reader timing. Keep recordings, transcripts, generated audio, and raw results
outside Git.

## Adding a backend

1. Add `src/tts/backends/<name>.py` implementing the backend protocol.
2. Add its dependency to `pyproject.toml` and the Docker provider installation map.
3. Register its model identifiers in `src/model_registry.py`.
4. Add backend, routing, capability, sample-rate, and API tests.
5. Update this file and the backend table in `README.md` with verified behavior only.
