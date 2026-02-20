# Open Speech — Roadmap

Current version: **v0.5.1**  
Next target: **v0.6.0 "Studio"**

---

## ✅ Phases Complete

### Phase 2 — Multi-Model + Streaming
- Multiple simultaneous TTS/STT model support
- Streaming STT via WebSocket (`/v1/audio/stream`)
- File upload transcription
- Response format selection (text, json, srt, vtt)
- Faster-Whisper, Moonshine, Vosk STT backends

### Phase 3 — Unified Model Architecture
- Single Docker image (GPU + CPU Dockerfiles)
- Unified `ModelManager` — download, load, unload, status
- Provider install system — pip install at runtime from UI
- Model registry (`src/model_registry.py`) — curated model catalog
- Config-driven via environment variables

### Phase 4 — Advanced TTS
- **Kokoro** — 52 voices, voice blending, high quality
- **Piper** — lightweight, fast, per-model voices
- **Qwen3-TTS** — voice design + zero-shot cloning
- **Fish Speech** — zero-shot voice cloning
- **F5-TTS** — flow matching voice cloning
- **XTTS v2** — multilingual cloning (16 languages)
- Extended TTS API: `voice_design`, `reference_audio`, `/v1/audio/speech/clone`
- Voice Reference Library — upload named voices, reuse by name

### Phase 5 — Voice Assistant Integration
- **Wyoming protocol** server (`src/wyoming/`) — Home Assistant drop-in
- **Silero VAD** (`src/vad/silero.py`) — voice activity detection
- **Real-time API** (`src/realtime/`) — OpenAI Realtime protocol compatible
- **Live mic streaming** — browser WebSocket → VAD → STT → transcript
- Anti-aliasing resampler (48kHz→16kHz, scipy `resample_poly`)

### Phase 6 — Production Hardening
- **TTS response cache** (`src/cache/tts_cache.py`) — SHA256-keyed, LRU eviction
- **Speaker diarization** (`src/diarization/`) — pyannote.audio, optional dep
- **Audio preprocessing** — noise reduction, gain normalization
- **Audio postprocessing** — silence trim, output normalization
- **Python client SDK** (`src/client/`) — sync/async transcribe + speak
- **Pronunciation dictionary** (`src/pronunciation/`) — JSON/YAML substitutions, SSML subset

### Phase 8a — Voice Profiles *(shipped in v0.5.1)*
- Named persistent voice profiles (`src/profiles.py`)
- Full CRUD API: `POST/GET/PUT/DELETE /api/profiles`
- Default profile support
- Profile selector in Speak tab → one-click restore all settings
- Persisted to `studio.db` (SQLite, WAL mode)

### Phase 8b — Generation History *(shipped in v0.5.1)*
- TTS + STT history log (`src/history.py`)
- Auto-logged after every successful generation/transcription
- Streamed requests: metadata-only (no audio file)
- API: `GET /api/history`, `DELETE /api/history/{id}`, `DELETE /api/history`
- History tab in web UI — paginated, re-generate, delete
- Configurable retention: `OS_HISTORY_MAX_ENTRIES`, `OS_HISTORY_MAX_MB`

### Phase 8e — Multi-Track Composer *(shipped in v0.5.1)*
- Track mixer manager (`src/composer.py`) with per-track offset/volume/mute/solo/effects
- Composer APIs: `POST /api/composer/render`, `GET /api/composer/renders`, `GET /api/composer/render/{id}/audio`, `DELETE /api/composer/render/{id}`
- Studio tab Composer card with track rows, render playback, and history
- Secure source-path validation (data roots only) + persisted compositions in `studio.db`

---

## 🔧 Current Web UI (v0.5.1)

Full 3-tab redesign (ground-up rewrite, 2026-02-20):

| Tab | Status |
|-----|--------|
| **Transcribe** | ✅ File upload + live mic, VAD indicator, partial + final results |
| **Speak** | ✅ Provider → Model → Voice → Preset cascade, auto-load flow, Generate state machine |
| **Models** | ✅ Loaded Models, STT/TTS columns (no cross-contamination), Providers section with install/uninstall |
| **History** | ✅ Paginated TTS+STT log, re-generate, delete, clear all |
| **Settings** | ✅ Profile CRUD, history settings |

---

## 🚧 In Progress / Upcoming

### Bug Fixes (prioritized)
| ID | Issue | Priority |
|----|-------|----------|
| B6 | Provider install (`pip`) writes to wrong path in Docker — Install Provider button broken | 🔴 Critical |
| B9 | Streaming TTS runs synchronously — blocks event loop for heavy models | 🔴 Critical |
| B7 | Inconsistent error envelopes (`{"detail":...}` vs `{"error":...}`) | 🟠 High |
| B11 | `inspect.signature()` called on every TTS request — use capabilities dict instead | 🟠 High |
| B10 | TTS cache key missing model — wrong cached audio after backend switch | 🟡 Medium |
| B8 | README API table still missing some endpoints | 🟡 Medium |

### Phase 8c — Conversation Mode
- Multi-turn conversation builder in Studio tab
- Turn list: speaker, profile, text → sequential render
- Export as single WAV/MP3 or per-turn ZIP
- REST API: `POST/GET/DELETE /api/conversations`, `POST /api/conversations/{id}/render`

### Phase 8d — Voice Effects
- Effects chain (`src/effects/chain.py`) — scipy-based
- Effects: normalize, pitch shift, room reverb, podcast EQ, robot
- Per-request `effects` parameter on `/v1/audio/speech`
- Effects panel in Speak tab (collapsible, capability-gated)

### Phase 7b-7d — Qwen3 Advanced *(deferred)*
- Voice design → clone workflow
- Native streaming (sub-100ms first chunk)
- Batch inference + vLLM backend

---

## Not Planned
- LLM conversation / function calling (bring your own brain)
- Multi-language UI (English only)
- CI/CD pipelines
- Cloud provider integrations

---

## Version History

| Version | Highlights |
|---------|-----------|
| v0.5.1 | XTTS v2, Voice Library, Phase 8a+8b (Profiles+History), UI rewrite, Models tab redesign, Speak tab Provider/Model/Voice cascade |
| v0.5.0 | Phase 6 production hardening (cache, diarization, audio processing, client SDK) |
| v0.4.x | Phase 5: Wyoming, VAD, Realtime API, live mic streaming |
| v0.3.x | Phase 4: Advanced TTS backends (Qwen3, Fish, F5, XTTS) |
| v0.2.x | Phase 3: Unified Docker image, ModelManager, provider install |
| v0.1.x | Phase 2: Multi-model, streaming STT, file upload |
