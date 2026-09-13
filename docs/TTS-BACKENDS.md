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

Each backend implements the protocol in `src/tts/backends/base.py`:

- `load_model()`, `unload_model()`, `is_model_loaded()`, and `loaded_models()` manage lifecycle.
- `synthesize()` yields mono float32 NumPy arrays.
- `list_voices()` reports selectable voices.
- `sample_rate` provides the default output rate.
- `get_sample_rate(model_id)`, when present, returns the selected model's actual output rate.
- `capabilities` tells the API which backend-specific controls are valid.

`TTSRouter` in `src/tts/router.py` resolves exact backend names and model prefixes. The audio pipeline
handles PCM conversion, file encoding, and non-live post-processing. Live Reader deliberately bypasses
per-segment normalization and retains bounded edge silence so consecutive sentences do not pump in volume
or run together unnaturally.

## Model selection

- Use Kokoro for the normal reading voice and voice blending.
- Use Piper when model size and CPU latency matter more than natural prosody.
- Use Pocket-TTS when its installed voices fit the use case and native progressive output is useful.

The authoritative model catalog is `src/model_registry.py`. The running harness exposes its installed,
downloaded, and loaded state through `GET /api/models`.

## Adding a backend

1. Add `src/tts/backends/<name>.py` implementing the backend protocol.
2. Add its dependency to `pyproject.toml` and the Docker provider installation map.
3. Register its model identifiers in `src/model_registry.py`.
4. Add backend, routing, capability, sample-rate, and API tests.
5. Update this file and the backend table in `README.md` with verified behavior only.
