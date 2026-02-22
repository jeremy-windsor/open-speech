# Open Speech — Fixes & Feature Requests

Quick intake for bugs, fixes, and feature ideas. Will triages, Forge builds.

For bigger items, open a [GitHub Issue](https://github.com/will-assistant/open-speech/issues).

## Status Key
- 🔴 Open
- 🟡 In Progress (dispatched to Forge)
- 🟢 Done (commit linked)

---

## Bugs

| # | Description | Status | Commit |
|---|-------------|--------|--------|
| B32 | **qwen3-tts still fails after B26 fix** — `qwen-tts` pulls `torchaudio` from PyPI (CPU-only) which conflicts with CUDA torch in GPU image. Fix: pre-install torchaudio from PyTorch CUDA index before qwen-tts install, both in Dockerfile bake step and runtime UI install path. Also added qwen deps to `[all]` extra. | 🟢 | 1bdd3d4 |
| B1 | Mic transcription captures nothing (WebSocket/format issue) | 🟢 | 9070140 |
| B2 | Piper backend passes `length_scale` kwarg rejected by current piper-tts API → synthesis fails with "# channels not specified" | 🟢 | e19eea3 |
| B3 | WebSocket library missing — uvicorn starts without `websockets`/`wsproto`, mic streaming broken. Fix: add `websockets` to Dockerfile deps + pyproject extras. Log: `WARNING: No supported WebSocket library detected` | 🟢 | — |
| B4 | `/v1/audio/stream` endpoint returns 404 (GET with query params). Likely route not registered or WebSocket upgrade fails silently before route match. Related to B3. | 🟢 | 8bfe6f9 |
| B5 | Piper `synthesize()` ignores requested model — always uses first loaded. `voice` param = model_id but code just grabs first key from `self._loaded` dict (line 187). Should match `voice` to loaded model key. | 🟢 | c2f47b0 |
| B6 | Provider install (`/api/providers/install`) installs packages to `~/.local` instead of system site-packages. Server can't import them. Cause: `pip install --user` default when not root. Fix: use `pip install --target` or `sys.executable -m pip install` pointing at the right site-packages dir. **Also:** when kokoro is installed, must run `python -m spacy download en_core_web_sm` (misaki phonemizer dep) — baked into CPU image but not GPU image. | 🟢 | ee712e4 |
| B7 | Inconsistent API error envelope — some endpoints return `{"error":"..."}` others `{"detail":"..."}` (FastAPI HTTPException default). Standardize all to `{"error":{"message":"..."}}`. See PROJECT-REVIEW.md section 1. | 🟢 | f3bde86 |
| B8 | README API table missing many implemented endpoints (`/v1/realtime`, `/api/ps`, `/v1/audio/models/*`, `/api/tts/capabilities`, etc). README env var docs missing ~15 active config knobs. See PROJECT-REVIEW.md sections 2+4+8. | 🟢 | Replaced flat API table with categorized reference (11 sections, 55+ endpoints) with request params, response shapes, and curl examples. All env vars already documented in prior update. |
| B9 | **Streaming TTS blocks event loop** — `_generate()` calls `_do_synthesize()` synchronously when `stream=True` (no `run_in_executor`). Non-streaming path correctly uses executor (line 803). Heavy models (Qwen3) will stall all concurrent requests during synthesis. **File:** `src/main.py:772-779`. Fix: wrap streaming synthesis in `asyncio.get_event_loop().run_in_executor()`. | 🟢 | already implemented |
| B10 | **TTS cache key missing model** — Cache key is `(text, voice, speed, format)` but omits the active model. Switching TTS backends (e.g. kokoro → piper) with same voice name returns stale cached audio from wrong backend. **File:** `src/cache/tts_cache.py` + `src/main.py:797-800`. Fix: add model/backend name to cache key. | 🟢 | 6b2aab0 |
| B11 | **`inspect.signature` called on every TTS request** — `_do_synthesize()` calls `inspect.signature()` at runtime for every request with `voice_design` or `reference_audio`. Slow (reflection) and fragile. **File:** `src/main.py:756-770`. Fix: use backend `capabilities` dict instead. | 🟢 | a56d033 |
| B12 | Frontend model auto-prepare race: provider install API was fire-and-forget polling, so Generate/Transcribe could continue before install completed and fail with confusing state errors. | 🟢 | pending |
| B40 | **`download()` calls `load()` which triggers auto-unload of existing loaded model** — Prefetching/downloading any model caused `_auto_unload_type` to unload already-loaded models of the same type because `download()` internally calls `load()`. Fix: add `_evict_others=False` parameter to `load()`, pass it from `download()`. | 🟢 | pending |
| B41 | **UI poll loop never terminates on failure/revert states** — After a failed download, the poll loop checking model status ran indefinitely (80+ polls over 5+ minutes) because the only break condition was `state === 'loaded'/'downloaded'`. Fix: add breaks for `'available'` and `'provider_missing'` states, plus a 60-iteration (3 minute) hard timeout. | 🟢 | pending |
| B43 | **Missing piper model variants** — `piper/en_US-ryan-high`, `en_US-amy-high`, `en_US-lessac-low`, `en_GB-alan-low`, `en_GB-cori-high` not in PIPER_MODELS dict. Users couldn't load/download high-quality ryan voice. Added all missing variants. | 🟢 | <commit> |
| B44 | **Wyoming binds to 127.0.0.1 — Home Assistant can't connect** — Wyoming server default host was `127.0.0.1`, refusing external connections from HA. Changed default to `0.0.0.0` in Dockerfile ENV and docker-compose.yml. | 🟢 | <commit> |
| B45 | **Unbaked backends show "Load to GPU" button and nuke working models** — pocket-tts and piper registered in the model list even when not installed. Clicking Load to GPU auto-unloaded the working model (F16), then failed. Fix: add `is_available()` classmethod to optional backends; TTSRouter skips unavailable ones; model_manager marks known models as `provider_missing` when backend not registered. | 🟢 | <commit> |
| B46 | **Download status poll loop stuck — concurrent downloads never terminate** — Status endpoint ignored `_download_progress`; models showed `provider_installed` during download so B41 break conditions never fired. Also queued downloads had no progress entry before acquiring lock. Fixed: status endpoint overlays `_download_progress`; queued status set before lock; B41 resets poll timeout on active progress. | 🟢 | <commit> |
| B47 | **Web UI audio streaming buffered full response before playback** — Generate button used `res.blob()` which collects all audio before playing. Fixed with MediaSource API + streaming fetch: audio plays as first chunks arrive. Falls back to blob for non-mp3 formats. | 🟢 | <commit> |

## Fixes

| # | Description | Status | Commit |
|---|-------------|--------|--------|
| F1 | Speed slider 0.25 step → 5% increments | 🟢 | 75fb457 |
| F2 | Kokoro showing in STT dropdown | 🟢 | c128533 |
| F3 | Kokoro-82M listed as STT in Models tab | 🟢 | c128533 |
| F4 | Models show Download but provider not installed | 🟢 | c128533 |
| F5 | Version badge showed v1.0 | 🟢 | c128533 |
| F6 | Voice presets didn't match actual voices | 🟢 | 75fb457 |
| F7 | *(removed — vosk backend removed)* | 🟢 | — |
| F8 | Realtime audio buffer limit + idle timeout protections | 🟢 | pending |
| F9 | Auth hardening (`OS_AUTH_REQUIRED`, startup warning, query-key deprecation) | 🟢 | pending |
| F10 | WS origin allowlist + Wyoming localhost default bind | 🟢 | pending |
| F11 | Voice clone upload size limit + TLS cert dir hardening | 🟢 | pending |
| F12 | Docker non-root user + cache/cert path updates | 🟢 | pending |
| F13 | Model manager concurrency locks + realtime model resolution fix | 🟢 | pending |
| F14 | Manual model lifecycle: provider install/download/load/unload/delete with actionable errors | 🟢 | pending |
| F15 | UX/model-flow rescue pass: auto-prepare on Generate/Transcribe, `ensureModelReady(modelId)`, capability-aware advanced controls, focused defaults (kokoro/piper + faster-whisper), and cleaner status-driven UI | 🟢 | pending |

## Features

| # | Description | Status | Commit |
|---|-------------|--------|--------|
| T1 | TTS history — download + delete buttons | 🟢 | c128533 |
| T2 | Stream toggle tooltip | 🟢 | c128533 |
| T3 | Voice presets dropdown in Speak tab | 🟢 | 0ea4d4c |
| T4 | Voice cloning endpoint (`/v1/audio/speech/clone`) | 🟢 | 0ea4d4c |
| T5 | Qwen3-TTS backend | 🟢 | 0ea4d4c |
| T6 | *(removed — fish-speech backend removed)* | 🟢 | — |
| T7 | TTS response cache with LRU + cache bypass | 🟢 | pending |
| T8 | STT diarization option (`diarize=true`) | 🟢 | pending |
| T9 | STT/TTS audio pre/post processing pipeline | 🟢 | pending |
| T10 | Python client SDK + example | 🟢 | pending |
| T11 | Pronunciation dictionary + SSML subset | 🟢 | pending |

---

*To add: just tell Will in Discord. He'll add it here and batch dispatch to Forge.*
| B30 | **qwen3-tts Install button resets — no feedback, reverts to Install** — Clicking Install on qwen3-tts in the Providers list shows no spinner, no progress, no error — button just bounces back to "Install". Root cause: B26 (missing `qwen-tts>=0.1.0` in `PROVIDER_INSTALL_SPECS["qwen3"]`) causes the pip install subprocess to fail silently. The UI now keeps an error-state retry button and shows inline error text on failure. | 🟢 | f8a9585 |
| B29 | **Voice Effects checkboxes misaligned — controls appear but are non-functional** — Voice Effects section now uses tight inline label+checkbox controls with full clickable labels; layout no longer spreads checkbox and text apart. | 🟢 | f8a9585 |
| B28 | **Models tab empty on load, 30s timeout before populating** — `/api/models` is fetched once at init; STT/TTS loaders consume cache, and non-critical startup loaders are non-blocking via settled promises. Added simple loading placeholders to STT/TTS selectors. | 🟢 | f8a9585 |
| B27 | **Unloading model doesn't release VRAM** — Added `gc.collect()` + CUDA cache emptying in affected unload paths (kokoro, qwen3 backend, faster-whisper) to mirror working backends. | 🟢 | f8a9585 |
| B26 | **Qwen3-TTS install missing `qwen-tts` package** — Added `qwen-tts>=0.1.0` install spec and corrected install guidance message in qwen3 backend import error. | 🟢 | f8a9585 |
| B13 | **Speak tab model selector confusing** — Model dropdown shows full registry paths (`piper/en_US-ryan-medium`) mixing provider+model+voice into one field. Should be three separate dropdowns: Provider (installed TTS only) → Voice (for that provider) → Preset. Defaults to wrong/missing model instead of loaded model. | 🟢 | f21169c |
| B33 | **Remove provider management UI and runtime pip install — bake-first architecture** | 🟢 | d9267a3 |
| B14 | **TTS model state never reaches `downloaded`** — `list_all()` checks HF cache for STT models via `list_cached_models()` but always passes `is_downloaded=False` for TTS models. After provider install, kokoro/piper stays `provider_installed` forever even if weights are already cached. Fix: (1) backend — wire `_candidate_artifact_paths()` into `list_all()` for TTS models, set `is_downloaded=True` if any artifact path exists. (2) UX follow-up — explicit **Cache (prefetch)** action + clearer model-state messaging (installed vs cached vs loaded), plus bake-time provider/model controls in Docker. | 🟢 | 6cfb90d + UX follow-up |
| B34 | **qwen3 install: transformers 4.x requires huggingface-hub<1.0, conflicts with qwen-tts 0.1.x** — pinned transformers>=5.0.0 (which dropped the huggingface-hub<1.0 constraint), added --upgrade to BAKED_PROVIDERS install, and added minimum version pins for accelerate, soundfile, librosa in all qwen3 specs. | 🟢 | 2e3cdcc |
| B35 | **kokoro + qwen-tts transformers conflict** — qwen-tts hard-pins transformers==4.57.3; cannot be installed alongside transformers>=5.0.0 in one pip call. Fix: split into two sequential installs — qwen-tts first (gets 4.57.3), then kokoro separately (uses whatever is installed). Removed transformers>=5.0.0 from all qwen3 specs. | 🟢 | 931167e |
| B36 | **torch version upgrade breaks torchaudio** — second pip call included bare `torch` with `--upgrade`, upgrading from CUDA-built 2.5.1+cu121 to PyPI 2.10.0; ABI mismatch breaks torchaudio. Fix: removed `torch` from second pip call entirely — it is already installed from CUDA index. | 🟢 | 1d76891 |
| B37 | **qwen-tts poisons entire environment via transformers==4.57.3 hard huggingface-hub<1.0 check** — transformers==4.57.3 (hard-pinned by qwen-tts) raises `ImportError` at module import time if huggingface-hub>=1.0 is installed. This killed kokoro (which imports AlbertModel from transformers) and faster-whisper. Fix: (1) remove qwen3 from default BAKED_PROVIDERS (now kokoro only); (2) when qwen3 is explicitly baked, install qwen-tts with `--no-deps` then manually wire transformers>=5.0.0 + huggingface-hub>=1.0. | 🟢 | pending |
| B38 | **Non-baked backends show generic Python ImportError instead of clean "not installed" message** — piper not in BAKED_PROVIDERS returns `No module named 'piper'` (raw 500) instead of a user-friendly error. The provider_missing check should catch this and return a clean 400 with rebuild instructions. | 🟢 | d3c58d6 |
| F16 | **Allow loading multiple STT models simultaneously — should enforce 1 STT + 1 TTS max** — Currently `OS_MAX_LOADED_MODELS=0` (unlimited). User expects single-model-per-type enforcement: loading a second STT should auto-unload the first. Fix: add per-type limits (STT max=1, TTS max=1) enforced at load time, or use `OS_MAX_LOADED_MODELS=2` as a stopgap. | 🟢 | 8a6ba03 |
| B39 | **"Cache" and "Load" button labels are confusing** — Cache = download weights to disk; Load = move weights into GPU VRAM. Users click Cache, skip Load, then get 30s generation because the model loads inline. Fix: rename to "Download" + "Load to GPU", add tooltip, auto-load on first Generate if only cached. | 🟢 | 82594cc |
| B40 | **Docker image layer bloat from package reinstalls across layers** — bake script installs unpinned transitive deps (numpy, onnxruntime, huggingface-hub, scipy) then requirements.lock reinstalls pinned versions in a later layer; Docker stores both versions (~200MB dead bytes). Fix: removed onnxruntime/huggingface-hub from qwen3 explicit deps, added version alignment step at end of bake script to pre-pin packages to match requirements.lock. | 🟢 | 10b25c8 |
| B39 | **"Cache" and "Load" button labels are confusing** — Users click Cache, expect audio to work fast, then get 30s wait because Cache ≠ ready-to-run. Cache = download to disk; Load = move to GPU VRAM. Labels should read "Download" and "Load to GPU" (or at minimum, a tooltip explaining the difference). Auto-load on first Generate if model is only cached would also help. | 🟢 | 82594cc |
| F17 | **Piper not in default BAKED_PROVIDERS** — piper backend is fully implemented but `piper-tts` package is not installed in the default image. Added `piper` to default `BAKED_PROVIDERS` in Dockerfile + docker-compose.yml. | 🟢 | 82594cc |
| F18 | **Wyoming protocol disabled by default** — Wyoming server is fully implemented but `OS_WYOMING_ENABLED` defaulted to `false`. Flipped to `true` in Dockerfile ENV + docker-compose.yml so Home Assistant can connect without manual env override. | 🟢 | 82594cc |
| F19 | Full Piper English catalog expanded in backend + registry (en_US + en_GB low/medium/high variants) | 🟢 | <commit> |
| F20 | Added distil-whisper model metadata entries (small.en, medium.en, large-v3) | 🟢 | <commit> |
| F21 | Added m4a format support to TTS API + ffmpeg pipeline/content-type map | 🟢 | <commit> |
| F22 | Added Kokoro CUDA warmup after pipeline init to eliminate first-request kernel compile stall | 🟢 | <commit> |
| F23 | Enabled BuildKit pip cache mounts in Dockerfile and Dockerfile.cpu (`# syntax=` + `--mount=type=cache`) | 🟢 | <commit> |
| F24 | Reworked streaming TTS into true queue-based async streaming (no full-buffer `list()` call) | 🟢 | <commit> |
| F25 | Added HF token passthrough (`HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`) in Docker + compose, and wired startup preloads after Wyoming start | 🟢 | <commit> |
| F26 | UI now renders `provider_missing`/`provider_available=false` as grayed-out “Not installed” rows with blocked actions | 🟢 | <commit> |
