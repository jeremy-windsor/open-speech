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
| Qwen3 | `qwen3/0.6b-base` | Provider-neutral reference WAV + transcript | No | Not in this slice |

The 1.7B CustomVoice, Base, and VoiceDesign variants remain intentionally hidden until the 0.6B
contract and RTX 2070 memory behavior are proven. Qwen's 0.6B models do not document a numeric speed
argument, so the harness disables the speed slider instead of silently translating or ignoring it.

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
mismatches. Provider URLs come only from `TTS_EXTERNAL_PROVIDERS`; requests cannot supply or override
them.

`TTSRouter` in `src/tts/router.py` resolves exact backend names and model prefixes. The audio pipeline
handles PCM conversion, file encoding, and non-live post-processing. Live Reader deliberately bypasses
per-segment normalization and retains bounded edge silence so consecutive sentences do not pump in volume
or run together unnaturally.

## Model selection

- Use Kokoro for the normal reading voice and voice blending.
- Use Piper when model size and CPU latency matter more than natural prosody.
- Use Pocket-TTS when its installed voices fit the use case and native progressive output is useful.

The authoritative core catalog is `src/model_registry.py`. The running harness exposes its installed,
downloaded, and loaded state through `GET /api/models`.

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

The CustomVoice model accepts only its official voice IDs: `Vivian`, `Serena`, `Uncle_Fu`, `Dylan`,
`Eric`, `Ryan`, `Aiden`, `Ono_Anna`, and `Sohee`. No OpenAI voice aliases are mapped silently.

For Base cloning, upload a provider-neutral WAV with the exact spoken transcript:

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

## Adding a backend

1. Add `src/tts/backends/<name>.py` implementing the backend protocol.
2. Add its dependency to `pyproject.toml` and the Docker provider installation map.
3. Register its model identifiers in `src/model_registry.py`.
4. Add backend, routing, capability, sample-rate, and API tests.
5. Update this file and the backend table in `README.md` with verified behavior only.
