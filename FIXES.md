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
| B6 | Provider install (`/api/providers/install`) installs packages to `~/.local` instead of system site-packages. Server can't import them. Cause: `pip install --user` default when not root. Fix: use `pip install --target` or `sys.executable -m pip install` pointing at the right site-packages dir. | 🔴 | — |
| B7 | Inconsistent API error envelope — some endpoints return `{"error":"..."}` others `{"detail":"..."}` (FastAPI HTTPException default). Standardize all to `{"error":{"message":"..."}}`. See PROJECT-REVIEW.md section 1. | 🔴 | — |
| B8 | README API table missing many implemented endpoints (`/v1/realtime`, `/api/ps`, `/v1/audio/models/*`, `/api/tts/capabilities`, etc). README env var docs missing ~15 active config knobs. See PROJECT-REVIEW.md sections 2+4+8. | 🔴 | — |
| B9 | **Streaming TTS blocks event loop** — `_generate()` calls `_do_synthesize()` synchronously when `stream=True` (no `run_in_executor`). Non-streaming path correctly uses executor (line 803). Heavy models (Qwen3, XTTS) will stall all concurrent requests during synthesis. **File:** `src/main.py:772-779`. Fix: wrap streaming synthesis in `asyncio.get_event_loop().run_in_executor()`. | 🔴 | — |
| B10 | **TTS cache key missing model** — Cache key is `(text, voice, speed, format)` but omits the active model. Switching TTS backends (e.g. kokoro → piper) with same voice name returns stale cached audio from wrong backend. **File:** `src/cache/tts_cache.py` + `src/main.py:797-800`. Fix: add model/backend name to cache key. | 🔴 | — |
| B11 | **`inspect.signature` called on every TTS request** — `_do_synthesize()` calls `inspect.signature()` at runtime for every request with `voice_design` or `reference_audio`. Slow (reflection) and fragile. **File:** `src/main.py:756-770`. Fix: use backend `capabilities` dict instead. | 🔴 | — |
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
| B13 | **Speak tab model selector confusing** — Model dropdown shows full registry paths (`piper/en_US-ryan-medium`) mixing provider+model+voice into one field. Should be three separate dropdowns: Provider (installed TTS only) → Voice (for that provider) → Preset. Defaults to wrong/missing model instead of loaded model. | 🔴 | — |
