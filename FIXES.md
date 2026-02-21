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
| B1 | Mic transcription captures nothing (WebSocket/format issue) | 🟡 | c128533 (partial) |
| B2 | Piper backend passes `length_scale` kwarg rejected by current piper-tts API → synthesis fails with "# channels not specified" | 🟢 | e19eea3 |
| B3 | WebSocket library missing — uvicorn starts without `websockets`/`wsproto`, mic streaming broken. Fix: add `websockets` to Dockerfile deps + pyproject extras. Log: `WARNING: No supported WebSocket library detected` | 🟢 | — |
| B4 | `/v1/audio/stream` endpoint returns 404 (GET with query params). Likely route not registered or WebSocket upgrade fails silently before route match. Related to B3. | 🟢 | 8bfe6f9 |
| B5 | Piper `synthesize()` ignores requested model — always uses first loaded. `voice` param = model_id but code just grabs first key from `self._loaded` dict (line 187). Should match `voice` to loaded model key. | 🟢 | c2f47b0 |
| B6 | Provider install (`/api/providers/install`) installs packages to `~/.local` instead of system site-packages. Server can't import them. Cause: `pip install --user` default when not root. Fix: use `pip install --target` or `sys.executable -m pip install` pointing at the right site-packages dir. **Also:** when kokoro is installed, must run `python -m spacy download en_core_web_sm` (misaki phonemizer dep) — baked into CPU image but not GPU image. | 🟢 | ee712e4 |
| B7 | Inconsistent API error envelope — some endpoints return `{"error":"..."}` others `{"detail":"..."}` (FastAPI HTTPException default). Standardize all to `{"error":{"message":"..."}}`. See PROJECT-REVIEW.md section 1. | 🟢 | f3bde86 |
| B8 | README API table missing many implemented endpoints (`/v1/realtime`, `/api/ps`, `/v1/audio/models/*`, `/api/tts/capabilities`, etc). README env var docs missing ~15 active config knobs. See PROJECT-REVIEW.md sections 2+4+8. | 🟡 | — |
| B9 | **Streaming TTS blocks event loop** — `_generate()` calls `_do_synthesize()` synchronously when `stream=True` (no `run_in_executor`). Non-streaming path correctly uses executor (line 803). Heavy models (Qwen3, XTTS) will stall all concurrent requests during synthesis. **File:** `src/main.py:772-779`. Fix: wrap streaming synthesis in `asyncio.get_event_loop().run_in_executor()`. | 🟢 | already implemented |
| B10 | **TTS cache key missing model** — Cache key is `(text, voice, speed, format)` but omits the active model. Switching TTS backends (e.g. kokoro → piper) with same voice name returns stale cached audio from wrong backend. **File:** `src/cache/tts_cache.py` + `src/main.py:797-800`. Fix: add model/backend name to cache key. | 🟢 | 6b2aab0 |
| B11 | **`inspect.signature` called on every TTS request** — `_do_synthesize()` calls `inspect.signature()` at runtime for every request with `voice_design` or `reference_audio`. Slow (reflection) and fragile. **File:** `src/main.py:756-770`. Fix: use backend `capabilities` dict instead. | 🟢 | a56d033 |
| B12 | Frontend model auto-prepare race: provider install API was fire-and-forget polling, so Generate/Transcribe could continue before install completed and fail with confusing state errors. | 🟢 | pending |

## Fixes

| # | Description | Status | Commit |
|---|-------------|--------|--------|
| F1 | Speed slider 0.25 step → 5% increments | 🟢 | 75fb457 |
| F2 | Kokoro showing in STT dropdown | 🟢 | c128533 |
| F3 | Kokoro-82M listed as STT in Models tab | 🟢 | c128533 |
| F4 | Moonshine models show Download but provider not installed | 🟢 | c128533 |
| F5 | Version badge showed v1.0 | 🟢 | c128533 |
| F6 | Voice presets didn't match actual voices | 🟢 | 75fb457 |
| F7 | Vosk Zip Slip safe extraction + validation | 🟢 | pending |
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
| T6 | Fish Speech backend | 🟢 | 0ea4d4c |
| T7 | TTS response cache with LRU + cache bypass | 🟢 | pending |
| T8 | STT diarization option (`diarize=true`) | 🟢 | pending |
| T9 | STT/TTS audio pre/post processing pipeline | 🟢 | pending |
| T10 | Python client SDK + example | 🟢 | pending |
| T11 | Pronunciation dictionary + SSML subset | 🟢 | pending |

---

*To add: just tell Will in Discord. He'll add it here and batch dispatch to Forge.*
| B30 | **qwen3-tts Install button resets — no feedback, reverts to Install** — Clicking Install on qwen3-tts in the Providers list shows no spinner, no progress, no error — button just bounces back to "Install". Root cause: B26 (missing `qwen-tts>=0.1.0` in `PROVIDER_INSTALL_SPECS["qwen3"]`) causes the pip install subprocess to fail silently. The UI doesn't surface the error from the install endpoint — it likely receives a 500/non-200 or the install "succeeds" (pip runs) but the package is wrong so the provider check still fails on verify. Fix: (1) apply B26 qwen-tts package fix in `model_manager.py`, (2) have the Install button show a spinner + display the error message if install fails instead of silently reverting. | 🔴 | — |
| B29 | **Voice Effects checkboxes misaligned — controls appear but are non-functional** — Voice Effects section renders checkboxes far to the left of their labels with large whitespace gaps. "Podcast EQ" label wraps across two lines. Layout uses a wide grid/flexbox structure that spaces checkbox elements far from their text. Controls appear visible but Jeremy reports effects are "still off" — checkboxes may not be togglable or the effect isn't applying even when checked. Appears related to B20 (effects grid CSS fix) — the 2-column grid was applied but the checkbox+label pairing within each cell is still broken. Fix: wrap each effect as a single `<label>` element containing `<input type="checkbox">` + text so click target = full label; verify `grid-template-columns` allocates correct widths so labels aren't clipped. | 🔴 | — |
| B28 | **Models tab empty on load, 30s timeout before populating** — `init()` runs 6 async functions in `Promise.all()` including `loadTTSProviders()`, `loadSTTModels()`, and `refreshModels()` which all independently call `GET /api/models`. Race condition: slow response overwrites `state.modelsCache` with stale data after `renderModelsView()` already ran, leaving Models tab empty until a subsequent refresh. Fix: (1) fetch `/api/models` once at init start, populate `state.modelsCache`, then call `renderModelsView()` immediately; (2) have `loadTTSProviders()` + `loadSTTModels()` use cache if already populated instead of re-fetching; (3) use `Promise.allSettled` for non-critical loaders (conversations, composer history) so a slow endpoint doesn't stall init; (4) add "Loading models..." skeleton text in STT/TTS sections until data arrives. | 🔴 | — |
| B27 | **Unloading model doesn't release VRAM** — `kokoro.py` and `qwen3_backend.py` `unload_model()` delete the model reference but don't call `gc.collect()` + `torch.cuda.empty_cache()`. VRAM stays allocated until Python GC runs. The working backends (xtts, f5tts, fish_speech) all do the proper cleanup. Fix: add `import gc`, then after `del self._pipeline` / `del self._loaded_models[model_id]` call `gc.collect()` + `if torch.cuda.is_available(): torch.cuda.empty_cache()`. Affects: `kokoro.py`, `qwen3_backend.py`, `src/backends/faster_whisper.py` (STT). The working backends xtts/f5tts/fish_speech already do this correctly. | 🔴 | — |
| B26 | **Qwen3-TTS install missing `qwen-tts` package** — `PROVIDER_INSTALL_SPECS["qwen3"]` installs `transformers, accelerate, soundfile, librosa` but omits `qwen-tts>=0.1.0` (the package providing `from qwen_tts import Qwen3TTSModel`). Load fails with: `Qwen3-TTS requires optional dependency group [qwen]. Install with: pip install -e '.[qwen]'`. Two fixes: (1) add `"qwen-tts>=0.1.0"` to `PROVIDER_INSTALL_SPECS["qwen3"]` in `model_manager.py`, (2) fix the error message to say `pip install qwen-tts>=0.1.0` not the dev-install command. | 🔴 | — |
| B13 | **Speak tab model selector confusing** — Model dropdown shows full registry paths (`piper/en_US-ryan-medium`) mixing provider+model+voice into one field. Should be three separate dropdowns: Provider (installed TTS only) → Voice (for that provider) → Preset. Defaults to wrong/missing model instead of loaded model. | 🟢 | f21169c |
| B14 | **TTS model state never reaches `downloaded`** — `list_all()` checks HF cache for STT models via `list_cached_models()` but always passes `is_downloaded=False` for TTS models. After provider install, kokoro/piper stays `provider_installed` forever even if weights are already cached. Fix: (1) backend — wire `_candidate_artifact_paths()` into `list_all()` for TTS models, set `is_downloaded=True` if any artifact path exists. (2) UX follow-up — explicit **Cache (prefetch)** action + clearer model-state messaging (installed vs cached vs loaded), plus bake-time provider/model controls in Docker. | 🟢 | 6cfb90d + UX follow-up |
