# Phase 8 Design Review

## Executive Summary
The current codebase is well-positioned for Phase 8 at the API-layer level: `FastAPI` routes are centralized in `src/main.py`, TTS/STT processing is already modularized, and there is existing persistence precedent via `VoiceLibraryManager`. However, Phase 8 introduces durable server-side state (profiles/history/conversations/compositions) that the app does not currently model, so this requires a small persistence architecture addition rather than incremental endpoint-only work. The biggest implementation risk is not synthesis itself, but keeping long-running audio operations (render/mix/effects) from blocking the event loop and from creating uncontrolled disk growth. I recommend shipping in two waves: (1) profiles + history + server-side history-backed UI, then (2) conversation + effects + composer, with shared storage and file lifecycle primitives added up front.

## Current Architecture Summary
- **Entrypoint / API surface**: `src/main.py` is the central API module (OpenAI-compatible STT/TTS endpoints + model/provider management + voice library + static UI serving).
- **Config**: `src/config.py` uses `pydantic-settings`, with strong env-var compatibility patterns and existing OS/STT/TTS namespacing.
- **Model orchestration**: `src/model_manager.py` provides unified STT/TTS model lifecycle abstraction and provider install flow.
- **TTS pipeline**: `src/tts/pipeline.py` handles final encoding (`wav`/`pcm` native, compressed via ffmpeg), and `src/audio/postprocessing.py` already applies trim + normalize in one place.
- **Current persistence**:
  - Voice library persists files + metadata (`src/voice_library.py`, API in `main.py`).
  - UI “history” is currently **browser localStorage only** (`src/static/app.js`), not server-persistent.
- **Frontend**:
  - Tabs: `Transcribe`, `Speak`, `Models` only.
  - No History tab, no Studio tab, no Settings tab.
  - TTS form has model/voice/speed/format and capability-based advanced controls.

Architectural implication: adding Phase 8 cleanly means introducing dedicated manager modules (as spec proposes) and keeping `main.py` as routing composition layer.

## Gap Analysis by Sub-phase

### 8a — Voice Profiles
**What exists / reusable**
- Config/env pattern in `src/config.py` already supports adding `OS_PROFILES_PATH` cleanly.
- Existing voice-related concepts (`TTSSpeechRequest`, voice presets endpoint, voice library references) are useful primitives.

**What to build**
- Persistent profile store (JSON for v1 simplicity, SQLite if wanting unified DB strategy with history/conversations).
- CRUD API routes under `/api/profiles`.
- UI profile selector in Speak tab and profile manager in settings area.

**Architecture conflicts / notes**
- Current voice presets are static/default and not user CRUD; profiles overlap but are conceptually different (user-owned persisted objects).
- `TTSSpeechRequest` currently has no `profile_id`; must either:
  - resolve profile in UI and submit expanded fields (simplest), or
  - accept `profile_id` server-side and resolve there.

**Risk areas**
- Backward-compat between legacy voice preset behavior and new profiles.
- Concurrency for profile file writes if JSON backend is chosen.

---

### 8b — History
**What exists / reusable**
- STT and TTS endpoints are centralized, so insertion points for logging are obvious:
  - `/v1/audio/speech`
  - `/v1/audio/transcriptions`
- UI already has local history rendering widgets and item cards.

**What to build**
- `HistoryManager` with SQLite (`data/history.db`) and APIs (`GET`, `DELETE by id`, `DELETE all`).
- Server-side pagination and filtering (`type=tts|stt`, limit/offset).
- Disk retention policies (`OS_HISTORY_MAX_ENTRIES`, `OS_HISTORY_RETAIN_AUDIO`, `OS_HISTORY_MAX_MB`).
- Re-generate support metadata in history rows (enough to recreate TTS request payload).

**Architecture conflicts / notes**
- Current UI history uses localStorage keys (`open-speech-tts-history`, `open-speech-stt-history`); this must be replaced or dual-mode migrated.
- Streaming TTS path complicates output-path logging: audio may not be stored unless explicitly buffered to file.

**Risk areas**
- SQLite I/O from async endpoints (must use thread offload or async driver discipline).
- File retention race conditions (deleting files while user plays/downloads).
- History table growth without strict pruning job.

---

### 8c — Conversation Mode
**What exists / reusable**
- TTS synthesis pipeline can render turns sequentially.
- Existing realtime module mentions conversation events, but it is not persistent conversation/studio data.

**What to build**
- Conversation data model + persistence (conversation + turns).
- APIs for create/list/get/delete and render-all.
- Combined render output handling + optional per-turn export paths.
- UI builder in new Studio tab for turn editing + profile assignment.

**Architecture conflicts / notes**
- No current job abstraction for long-running multi-turn render; synchronous request may become too long.
- Need stable relation with profiles (`profile_id` FK or nullable external reference).

**Risk areas**
- Long sequential synthesis in request/response cycle (timeouts).
- Partial failure handling (one bad turn should not corrupt entire conversation state).

---

### 8d — Voice Effects
**What exists / reusable**
- `src/audio/postprocessing.py` already applies normalize/trim; can be generalized into chain architecture.
- `process_tts_chunks` currently provides a single post-generation insertion point.

**What to build**
- Effects chain abstraction (`src/effects/chain.py`) with deterministic ordered effect execution.
- Extend `TTSSpeechRequest` schema to accept `effects` array.
- Effect parameter validation and capability gating via `OS_EFFECTS_ENABLED`.
- UI effect panel with presets/toggles/sliders + request serialization.

**Architecture conflicts / notes**
- Current output pipeline concatenates chunks then processes once; this is fine for offline effects, but not real-time low-latency preview.
- Streaming mode + complex effects may force full-buffer processing (latency or incompatibility).

**Risk areas**
- CPU load and latency spikes on larger audio.
- Dependency portability (ffmpeg + python DSP libs across Docker images).
- Non-deterministic loudness results if normalization + compression order is not fixed.

---

### 8e — Multi-Track Composer
**What exists / reusable**
- Encoding/export path from `src/tts/pipeline.py` reusable for final output.
- Existing TTS generated assets and voice library can be valid track sources.

**What to build**
- Composer model/schema (composition + tracks).
- Mixer engine (offset, gain, mute/solo, optional track effects).
- `/api/composer/render` endpoint and composition save/load APIs (spec says save JSON).
- Studio timeline UI and track controls.

**Architecture conflicts / notes**
- No current abstraction for project assets (uploaded files vs generated files) with lifecycle ownership.
- Browser-side timeline is currently nonexistent; app.js is single-file and will become hard to maintain unless modularized.

**Risk areas**
- Memory consumption while mixing many long tracks.
- Sample-rate/channel normalization edge cases across sources.
- Export format codec availability.

## Recommended Implementation Order
1. **Foundation first (before 8a): storage + media paths + cleanup utilities**
   - Add shared persistence utilities (SQLite connection helper, file root structure, retention worker hooks).
   - Rationale: 8b/8c/8e all need this; avoids rewrite.
2. **8a Voice Profiles**
   - Smallest user-visible win; also required dependency for 8c speaker/profile assignment.
3. **8b History**
   - High-value persistence and operational observability; should land before Studio features.
4. **8d Effects (basic chain: normalize + pitch + reverb first)**
   - Enables richer quality and is reusable by 8e track effects.
5. **8c Conversation Mode**
   - Build on profiles + history + effects-aware TTS request shape.
6. **8e Composer**
   - Largest scope; depends on stable asset management and optional effects pipeline.

This is slightly different from spec ordering (8c before 8d) because implementing effects earlier reduces duplicated processing logic between conversation rendering and final composer mixing.

## Data Models

### Option chosen: SQLite-first (recommended)
Use one DB file (`data/studio.db`) with tables below. (Alternative is split DBs, but one DB simplifies joins and migrations.)

```sql
-- profiles
CREATE TABLE profiles (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  backend TEXT NOT NULL,
  model TEXT,
  voice TEXT NOT NULL,
  speed REAL NOT NULL DEFAULT 1.0,
  format TEXT NOT NULL DEFAULT 'mp3',
  blend TEXT,
  reference_audio_id TEXT,
  effects_json TEXT,
  is_default INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

-- history entries (tts + stt)
CREATE TABLE history_entries (
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL CHECK(type IN ('tts','stt')),
  created_at TEXT NOT NULL,
  model TEXT,
  voice TEXT,
  speed REAL,
  format TEXT,
  text_preview TEXT,
  full_text TEXT,
  input_filename TEXT,
  output_path TEXT,
  output_bytes INTEGER,
  meta_json TEXT
);
CREATE INDEX idx_history_type_created ON history_entries(type, created_at DESC);

-- conversations
CREATE TABLE conversations (
  id TEXT PRIMARY KEY,
  name TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  render_output_path TEXT,
  meta_json TEXT
);

CREATE TABLE conversation_turns (
  id TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  turn_index INTEGER NOT NULL,
  speaker TEXT NOT NULL,
  profile_id TEXT,
  text TEXT NOT NULL,
  audio_path TEXT,
  duration_ms INTEGER,
  effects_json TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
  FOREIGN KEY(profile_id) REFERENCES profiles(id) ON DELETE SET NULL
);
CREATE INDEX idx_turns_conversation_idx ON conversation_turns(conversation_id, turn_index);

-- compositions
CREATE TABLE compositions (
  id TEXT PRIMARY KEY,
  name TEXT,
  sample_rate INTEGER NOT NULL DEFAULT 24000,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  render_output_path TEXT,
  meta_json TEXT
);

CREATE TABLE composition_tracks (
  id TEXT PRIMARY KEY,
  composition_id TEXT NOT NULL,
  track_index INTEGER NOT NULL,
  source_type TEXT NOT NULL CHECK(source_type IN ('upload','tts','conversation_turn')),
  source_ref TEXT,
  source_path TEXT NOT NULL,
  offset_s REAL NOT NULL DEFAULT 0,
  volume REAL NOT NULL DEFAULT 1.0,
  muted INTEGER NOT NULL DEFAULT 0,
  solo INTEGER NOT NULL DEFAULT 0,
  effects_json TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(composition_id) REFERENCES compositions(id) ON DELETE CASCADE
);
```

### JSON shape for effects chain
```json
[
  { "type": "normalize", "target_lufs": -16 },
  { "type": "pitch", "semitones": 2 },
  { "type": "reverb", "room": "small", "mix": 0.2 }
]
```

## API Design

### Profiles
- `POST /api/profiles`
  - Req:
    ```json
    {
      "name": "Narrator",
      "backend": "kokoro",
      "model": "kokoro",
      "voice": "af_heart",
      "speed": 1.0,
      "format": "mp3",
      "blend": null,
      "reference_audio_id": null,
      "effects": []
    }
    ```
  - Res `201`: created profile object with `id`, timestamps.
- `GET /api/profiles` -> `{ "items": [...], "default_profile_id": "..." }`
- `GET /api/profiles/{id}` -> profile object
- `PUT /api/profiles/{id}` -> updated profile object
- `DELETE /api/profiles/{id}` -> `204`
- `POST /api/profiles/{id}/default` -> `{ "status": "ok", "default_profile_id": "..." }`

### History
- `GET /api/history?type=tts|stt&limit=50&offset=0`
  - Res:
    ```json
    {
      "items": [{"id":"...","type":"tts","created_at":"...","text_preview":"...","output_path":"..."}],
      "total": 123,
      "limit": 50,
      "offset": 0
    }
    ```
- `DELETE /api/history/{id}` -> `204`
- `DELETE /api/history` -> `{ "deleted": 123 }`

### Conversations
- `POST /api/conversations`
  - Req: `{ "name": "Scene 1", "turns": [{"speaker":"A","text":"Hi","profile_id":"..."}] }`
  - Res `201`: conversation with turns.
- `GET /api/conversations?limit=50&offset=0`
- `GET /api/conversations/{id}`
- `POST /api/conversations/{id}/render`
  - Req: `{ "format": "wav", "sample_rate": 24000, "save_turn_audio": true }`
  - Res: `{ "conversation_id":"...","output_path":"...","download_url":"...","duration_ms":12345 }`
- `DELETE /api/conversations/{id}` -> `204`

### Composer
- `POST /api/composer/render`
  - Req:
    ```json
    {
      "name": "Episode Intro",
      "format": "mp3",
      "tracks": [
        {"source_path":"/data/.../voice1.wav","offset_s":0,"volume":1.0,"muted":false,"solo":false,"effects":[]},
        {"source_path":"/data/.../music.wav","offset_s":1.2,"volume":0.25,"muted":false,"solo":false,"effects":[]}
      ]
    }
    ```
  - Res: `{ "composition_id":"...","output_path":"...","download_url":"..." }`

### Existing endpoint changes
- `POST /v1/audio/speech`
  - Add optional request fields:
    - `profile_id: string | null`
    - `effects: Effect[]`
    - `save_history: bool = true`
  - Behavior: resolve profile defaults, apply effects chain, log history.
- `POST /v1/audio/transcriptions`
  - Add `save_history: bool = true` (form field).
  - Log stt history (filename/model/text).

## UI Changes

### Global navigation (`index.html`)
- Add tabs:
  - `History`
  - `Studio`
  - `Settings` (or extend existing Models tab with settings sections; separate tab is cleaner for Phase 8 requirements)
- Keep `Transcribe`, `Speak`, `Models`.

### Speak tab changes
- **Top section**: Profile selector dropdown + “Apply” + quick “Save as profile”.
- **Effects panel**: collapsible card under speed/format controls.
  - Toggle per effect: Normalize, Pitch, Reverb, Podcast EQ, Robot.
  - Per-effect sliders/selectors.
- **Post-generate actions**:
  - Existing download remains.
  - Add “Save to Studio track” shortcut.

### New History tab
- Filter bar: type (All/TTS/STT), text search (optional), limit.
- Paginated list/table with row actions:
  - Play (if audio exists)
  - Download
  - Re-generate (for TTS items)
  - Delete
- Footer controls: Prev/Next page, Clear All.

### New Studio tab (split sections)
1. **Conversation Builder**
   - Turn list (speaker, profile, text, delete/reorder).
   - “Add Turn”, “Render All”, “Play Conversation”, “Export WAV/MP3”, “Export ZIP per turn”.
2. **Composer**
   - Track list/timeline row per track (offset, volume, mute/solo, effects).
   - Add source from upload/history/conversation turn.
   - Render mix + download.

### Settings changes
- Profiles manager (create/edit/delete/set default).
- History settings (retention count/size, retain-audio toggle, clear all).
- Effects presets manager (save/load named chains).

### CSS / JS implementation notes
- `app.js` is currently monolithic (518 lines); Phase 8 should split into modules:
  - `ui-tabs.js`, `ui-speak.js`, `ui-history.js`, `ui-studio.js`, `ui-settings.js`, `api-client.js`, `state-store.js`.
- Add new reusable styles for:
  - table/list controls
  - timeline rows and rulers
  - modal/dialog for profile CRUD.
- Replace localStorage history with API-backed state; keep localStorage only for UI prefs (theme, last selected tab/profile).

## Dependency Recommendations
- **pedalboard vs scipy/librosa**
  - Recommendation: **start with scipy/numpy implementation for v1 effects**, add `pedalboard` as optional enhancement only if quality/perf warrants.
  - Why:
    - `scipy` is already declared core dependency.
    - `librosa` is not in core deps/lock and can be heavy; avoid making it hard requirement unless pitch-shift quality mandates it.
    - `pedalboard` is strong for music FX quality and ergonomics, but adds platform/packaging complexity and should be optional (`studio` extra).
- **pydub for mixing**
  - Recommendation: **do not use pydub as primary mixer**.
  - Use numpy/scipy for deterministic sample-accurate mixing and only rely on ffmpeg for final encoding (already established in pipeline).
  - Keep `pydub` optional for convenience import/export utilities if needed, not core render engine.

## Open Questions
1. **Persistence strategy**: keep spec split (`profiles.json` + `history.db`) or move everything to single `studio.db` now?
2. **History audio retention**: if `OS_HISTORY_RETAIN_AUDIO=false`, should TTS history still keep regeneratable parameter snapshots only?
3. **Streaming TTS logging**: should streamed outputs be fully buffered and saved for history, or store metadata-only for streamed requests?
4. **Effects in streaming mode**: support only stateless effects, or disable effects when `stream=true` initially?
5. **Conversation render execution model**: synchronous request vs background job endpoint with polling/status.
6. **Composer source policy**: permit arbitrary filesystem paths or only managed asset roots for security/safety.
7. **Profile schema scope**: include `model` and `effects` now, even though spec minimal fields don’t require both?
8. **UI scope for v0.7.0**: whether Settings becomes a dedicated tab now vs incremental sections inside existing tabs.
9. **Dependency posture**: approve adding `pedalboard` and/or `librosa` to optional `studio` extra, with Docker optional install path.
10. **Migration behavior**: should existing localStorage history be imported into server history on first load (best-effort) or ignored.

## File-by-File Implementation Plan

### Create
- `src/profiles.py`
  - `ProfileManager` with CRUD + default profile semantics.
- `src/history.py`
  - `HistoryManager` (SQLite), pruning, delete/clear/list.
- `src/conversation.py`
  - `ConversationManager`, turn persistence, render orchestration.
- `src/composer.py`
  - `MultiTrackComposer` mix/render logic.
- `src/effects/chain.py`
  - `EffectsChain`, validators, effect implementations.
- `src/storage.py` (recommended helper)
  - shared sqlite connection helpers + app data paths + safe file write/delete.
- New tests:
  - `tests/test_profiles_api.py`
  - `tests/test_history_api.py`
  - `tests/test_conversation_api.py`
  - `tests/test_effects_chain.py`
  - `tests/test_composer.py`

### Modify
- `src/config.py`
  - Add `os_profiles_path`, `os_history_enabled`, `os_history_max_entries`, `os_history_retain_audio`, `os_history_max_mb`, `os_effects_enabled`.
- `src/tts/models.py`
  - Extend `TTSSpeechRequest` with `profile_id`, `effects`, optional `save_history`.
- `src/main.py`
  - Wire managers at module init and lifespan hooks.
  - Add all new `/api/profiles`, `/api/history`, `/api/conversations`, `/api/composer/render` routes.
  - Update `/v1/audio/speech` and `/v1/audio/transcriptions` for history/effects integration.
- `src/static/index.html`
  - Add History/Studio/Settings panels and controls.
- `src/static/app.css`
  - Add styles for lists/tables/timeline/effects forms/settings cards.
- `src/static/app.js`
  - Add API-backed profile/history/studio flows; remove localStorage history dependency.

### Test Strategy
- **Unit tests**
  - Profile validation + default behavior.
  - History pruning policy (count and disk cap).
  - Effects chain ordering and parameter bounds.
  - Composer math (offset/volume/mute/solo) with deterministic short arrays.
- **API tests (FastAPI TestClient)**
  - Full CRUD for profiles/history/conversations.
  - Speech/transcription endpoints verify history side effects.
  - Composer render endpoint returns valid media and metadata.
- **Integration tests**
  - Render conversation then use output as composer track.
  - Retention cleanup removes DB rows + files consistently.
- **UI sanity tests (existing style)**
  - Capability gating for effects panel.
  - History tab pagination rendering.
  - Profile apply/load on Speak tab.
