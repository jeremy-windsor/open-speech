# Open Speech — Project Review (Production Readiness)

Repo: `/home/claude/repos/open-speech`  
Date: 2026-02-17

## What I executed
- `python3 -m pytest tests/ -q` → **499 passed**
- `python3 -m pytest tests/ --cov=src --cov-report=term-missing -q` → **499 passed**, overall **76%** line coverage
- `python3 -m ruff check src` → clean after quick-fix cleanup
- `docker build -f Dockerfile.cpu -t open-speech:review-cpu .` → **passes**
- `docker build -f Dockerfile -t open-speech:review-gpu .` → started but aborted due very large base/layer pull duration in this environment (not fully validated end-to-end)

## Quick fixes applied (and pushed)
I fixed low-risk cleanup items directly and pushed them in a separate commit:
- **Commit:** `3ad74f8`
- **Summary:** dead imports/unused locals cleanup in `src/`, `.env.example` completeness improvements, and `Dockerfile.cpu` correctness updates.

Notable direct fixes:
- `Dockerfile.cpu` now creates non-root `openspeech` user before `USER openspeech` (was previously invalid at runtime) (`Dockerfile.cpu:11-13`, `Dockerfile.cpu:42`)
- `Dockerfile.cpu` now uses `OS_HOST/OS_PORT` + modern `STT_MODEL/TTS_MODEL/TTS_VOICE` defaults and exposes Wyoming port (`Dockerfile.cpu:26-37`)
- `.env.example` now includes missing realtime/VAD/TTS advanced config knobs (`.env.example:54+`)

---

## 1) Code Consistency

### 🔴 Must fix before release
1. **Inconsistent API error envelope (`detail` vs `error`)**  
   - **Type:** Real bug (API contract inconsistency)
   - `src/main.py:570`, `src/main.py:799` return `{"error": ...}` via `JSONResponse`.
   - Most other failures use `HTTPException(..., detail=...)` (e.g. `src/main.py:553-559`, `src/main.py:664`, many others), resulting in `{"detail": ...}`.
   - This creates client-side branching and violates “consistent JSON error” expectation.

### 🟡 Should fix
2. **Backend behavior consistency is mostly good, but capability handling is duplicated in endpoint logic**  
   - **Type:** Cleanup/design debt
   - Feature gating happens in endpoint glue (`_validate_tts_feature_support`) instead of stricter backend-level interface guarantees (`src/main.py:80-94`, `src/main.py:564-570`, `src/main.py:797-799`).
   - Consider standardizing on backend capability checks in router/backend adapters and returning typed exceptions.

3. **`src/streaming.py` remains very large and mixed-responsibility**  
   - **Type:** Cleanup/maintainability
   - Large single module (`src/streaming.py`) combines session state, VAD, transcript agreement, transport, and execution policy.
   - Harder to reason about regressions and per-feature tests.

### 🔵 Nice to have
4. **Enforce lint in CI for source+tests**  
   - **Type:** Cleanup
   - Source lint is now clean, but tests still contain many stylistic unused imports if linted strictly.

---

## 2) Integration Gaps

### 🔴 Must fix before release
1. **README API table is incomplete vs actual server surface**  
   - **Type:** Missing feature in docs
   - Implemented but not listed in API table:
     - `/v1/realtime` (`src/main.py:523`)
     - `/v1/models/{model:path}` (`src/main.py:346`)
     - `/api/ps` legacy endpoints (`src/main.py:355`, `362`, `373`)
     - `/api/models/{id}/progress` (`src/main.py:424`)
     - `/api/pull/{model}` (`src/main.py:468`)
     - `/v1/audio/models/*` (`src/main.py:673`, `688`, `701`)
     - `/api/tts/capabilities` (`src/main.py:401`)
   - README table currently at `README.md:284-300`.

### 🟡 Should fix
2. **Model naming mismatch in docs vs registry for Qwen3 IDs**  
   - **Type:** Real bug (docs/config mismatch)
   - README lists `qwen3-tts-0.6b` / `qwen3-tts-1.7b` (`README.md:272-273`), but registry and endpoints use slash/case IDs like `qwen3-tts/1.7B-CustomVoice` (`src/model_registry.py:29-35`).

3. **Feature wiring check summary**  
   - **Type:** Informational (mostly good)
   - Wyoming: wired via lifespan startup/teardown (`src/main.py:159-176`) + compose exposes `10400` in all compose variants.
   - VAD: wired for streaming and realtime paths.
   - Realtime: gated by `OS_REALTIME_ENABLED` and mounted at `/v1/realtime` (`src/main.py:523-539`).
   - Caching, diarization, pronunciation, capabilities: all wired into main request paths.

---

## 3) Test Coverage Gaps

### ✅ Result
- `python3 -m pytest tests/ -q` → **499 passed**.

### 🟡 Should fix
1. **Critical-path coverage gaps remain despite good breadth**  
   - **Type:** Missing tests
   - No explicit OOM/memory-pressure behavior tests found for STT/TTS inference paths.
   - Realtime/WebSocket has many tests, but failure-mode matrix can still expand (e.g., partial frame corruption + disconnect timing edges).

2. **Very low coverage in key streaming STT module**  
   - **Type:** Missing tests
   - `src/streaming.py` at ~28% in coverage run.
   - This is a high-risk runtime path and should be raised significantly.

### 🔵 Nice to have
3. **No non-empty source file with literal 0% coverage found**  
   - Mostly `__init__.py` files are empty and naturally at 100/0 statements.

---

## 4) Configuration Audit (OS_*, STT_*, TTS_*)

### Audit summary
- Config fields defined in `Settings`: **55** OS/STT/TTS vars.
- `.env.example`: now includes all except `STT_MODEL_DIR`.
- README: missing several active vars (see below).

### 🔴 Must fix before release
1. **README does not document many active production knobs**  
   - **Type:** Missing docs
   - Missing in README but present in `src/config.py` include:
     - `OS_REALTIME_MAX_BUFFER_MB`, `OS_REALTIME_IDLE_TIMEOUT_S`
     - `OS_STREAM_CHUNK_MS`, `OS_STREAM_VAD_THRESHOLD`, `OS_STREAM_ENDPOINTING_MS`, `OS_STREAM_MAX_CONNECTIONS`
     - `OS_TRUST_PROXY`
     - `STT_DIARIZE_ENABLED`, `STT_NOISE_REDUCE`, `STT_NORMALIZE`, `STT_MODEL_DIR`
     - `TTS_DEFAULT_FORMAT`, `TTS_CACHE_ENABLED`, `TTS_CACHE_MAX_MB`, `TTS_CACHE_DIR`, `TTS_TRIM_SILENCE`, `TTS_NORMALIZE_OUTPUT`, `TTS_PRONUNCIATION_DICT`, `TTS_PRELOAD_MODELS`

### 🟡 Should fix
2. **`.env.example` still missing `STT_MODEL_DIR`**  
   - **Type:** Missing docs/config
   - Present in config (`src/config.py`, `stt_model_dir`) but absent in `.env.example`.

3. **Defaults mostly sensible for first-time user**  
   - **Type:** Informational
   - Good defaults for local startup (HTTPS on, model defaults, auth optional with explicit warning).
   - Consider whether `OS_SSL_ENABLED=true` as default is ideal for all first-time non-browser API users.

---

## 5) API Completeness

### 🔴 Must fix before release
1. **Error response schema is not consistent across endpoints**  
   - **Type:** Real bug
   - Mixed `{"detail": ...}` and `{"error": ...}` shapes (see section 1).

### 🟡 Should fix
2. **README API docs lag actual implementation**  
   - **Type:** Missing docs
   - See section 2 endpoint mismatches.

3. **OpenAI compatibility: partial but not strict**  
   - **Type:** Missing feature/spec gap
   - `/v1/audio/transcriptions`, `/v1/audio/speech`, `/v1/models` are implemented.
   - Compatibility is practical, but not strict spec parity in all fields/validation semantics.

### 🔵 Nice to have
4. **Document response schemas and error schemas explicitly in README**  
   - **Type:** Docs improvement

---

## 6) Web UI Review (`src/static/index.html`)

### Findings
- UI endpoint references found: `/v1/audio/transcriptions`, `/v1/audio/speech`, `/api/models`, `/api/tts/capabilities`, `/api/voice-presets`, `/health` — all implemented.
- Tab structure and styling are consistent (single-file CSS/JS style, coherent theme system).

### 🟡 Should fix
1. **No automated UI test coverage for tab-level regressions**  
   - **Type:** Missing tests
   - Functionality appears wired, but no browser-level integration tests to catch runtime JS regressions.

### 🔵 Nice to have
2. **Move large inline JS/CSS into modular assets**  
   - **Type:** Cleanup/maintainability
   - `index.html` is very large; modularization would improve maintainability and lintability.

---

## 7) Docker Review

### ✅ What I verified
- `Dockerfile.cpu` now builds cleanly in this environment after fix.
- Compose files all expose HTTP + Wyoming ports.

### 🔴 Must fix before release
1. **GPU Dockerfile build not fully validated in this run**  
   - **Type:** Validation gap
   - Build was initiated and progressed deeply but intentionally aborted due environment/time constraints with huge base/image layers.
   - Release checklist should include successful CI build for both Dockerfiles.

### 🟡 Should fix
2. **Compose consistency differs between default and cpu/gpu variants**  
   - **Type:** Cleanup
   - `docker-compose.yml` uses prebuilt image and fewer env options than cpu/gpu compose files.
   - Consider harmonizing env coverage and comments to reduce drift.

---

## 8) Documentation Gaps

### 🔴 Must fix before release
1. **README lacks complete env-var documentation for active config surface**  
   - **Type:** Missing docs
   - See section 4 list.

2. **README API table not synchronized with implemented endpoints**  
   - **Type:** Missing docs
   - See section 2 list.

### 🟡 Should fix
3. **Model docs use stale Qwen IDs vs actual registry IDs**  
   - **Type:** Real bug/docs mismatch
   - `README.md:272-273` vs `src/model_registry.py:29-35`.

4. **Getting-started path is mostly clear, but advanced features are fragmented**  
   - **Type:** Docs quality
   - Advanced config/features (realtime tuning, cache tuning, diarization toggles) should be consolidated in one “production config” section.

---

## Final release recommendation
**Current status: Not release-ready without doc/API contract cleanup.**

Primary blockers:
1. Standardize error response envelope (`error` vs `detail`).
2. Bring README API + config docs fully in sync with real server behavior.
3. Confirm GPU Dockerfile build in CI before release.

Secondary priorities:
- Increase test depth on streaming/realtime failure modes and memory-pressure scenarios.
- Improve maintainability of large monolithic modules (`streaming.py`, `static/index.html`).
