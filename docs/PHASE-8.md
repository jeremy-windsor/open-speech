# Phase 8 — Studio (v0.7.0)

Phase 8 transforms Open Speech into a full voice production studio. The focus is persistent identity, generative history, creative tooling, and a dramatically improved web UI.

---

## 8a — Voice Profiles (Persistent Identity)

**Why:** Users want to define a "character" once and reuse it across sessions, not re-enter settings every time.

**What:**
- Named voice profiles stored server-side: `{ name, backend, voice, speed, format, blend, reference_audio_id }`
- CRUD API: `POST/GET/PUT/DELETE /api/profiles`
- Profile selector in Speak tab — pick a profile, all settings populate
- Optional: mark a profile as "default" — loads on app open
- Profiles persist across restarts (SQLite or JSON file)

**Config:**
- `OS_PROFILES_PATH` — where profiles are stored (default: `/home/openspeech/data/profiles.json`)

---

## 8b — Generation History

**Why:** "What did I generate last Tuesday?" — currently lost on page refresh.

**What:**
- Every TTS synthesis logged: timestamp, text (truncated 200 chars), model, voice, speed, format, output path
- STT transcriptions also logged: timestamp, filename, model, result text
- History API: `GET /api/history?type=tts|stt&limit=50&offset=0`
- Delete entry: `DELETE /api/history/{id}`
- Clear all: `DELETE /api/history`
- Web UI History tab: paginated list with play/download/re-generate buttons
- Persist to SQLite (`data/history.db`)

**Config:**
- `OS_HISTORY_ENABLED=true`
- `OS_HISTORY_MAX_ENTRIES=1000`
- `OS_HISTORY_RETAIN_AUDIO=true` — keep generated audio on disk (subject to `OS_HISTORY_MAX_MB`)
- `OS_HISTORY_MAX_MB=2000`

---

## 8c — Conversation Mode

**Why:** Multi-turn voice interactions — generate a script, hear it back, iterate.

**What:**
- Conversation is a list of turns: `{ speaker, text, profile_id, audio_path }`
- Build turns in UI: type text, assign speaker/profile, add to queue
- "Render All" button — synthesizes all turns sequentially, produces combined audio
- Playback: click any turn to hear just that segment, or play full conversation
- Export: download as single WAV/MP3 or per-turn ZIP
- Save/load conversations as JSON
- API: `POST /api/conversations` + `GET/DELETE /api/conversations/{id}`

---

## 8d — Voice Effects

**Why:** Post-processing for creative and practical use (podcast compression, pitch shift, room reverb).

**What:**
- Effects chain applied to TTS output before delivery
- Built on `scipy` / `librosa` (already available) or `pedalboard` (Spotify, MIT license)
- Effects:
  - **Normalize** — loudness normalization (already partially done in 6c)
  - **Pitch shift** — semitone adjustment (-12 to +12)
  - **Room reverb** — small/medium/large room simulation
  - **Podcast EQ** — high-pass 80Hz + presence boost + gentle compression
  - **Robot** — vocoder-style effect
- Effect chain config per-request: `effects=[{"type":"reverb","room":"small"},{"type":"pitch","semitones":2}]`
- Web UI: effects panel with toggles + sliders, preview before committing
- Config: `OS_EFFECTS_ENABLED=true`

---

## 8e — Multi-Track Composer

**Why:** Create layered audio — narration over music bed, multiple speakers, sound effects.

**What (v1 — simple):**
- Track list: each track = audio source (TTS generated or uploaded file) + start time offset + volume + effects
- Timeline view in browser (CSS-based, no canvas complexity)
- Mix to single output: `POST /api/composer/render` returns mixed audio
- Per-track controls: volume slider, mute, solo, offset (seconds)
- Save composition as JSON
- Export as WAV/MP3

**Out of scope for v1:** real-time preview, MIDI, plug-in architecture

---

## UI Changes Required for Phase 8

### New tabs:
- **History** — TTS + STT log with playback and re-generate
- **Studio** — Conversation builder + composer (combined)

### Updated tabs:
- **Speak** — Profile selector at top; after generate, auto-save to history
- **Transcribe** — Auto-save transcriptions to history

### New settings section:
- Profiles manager (create/edit/delete)
- History settings (retention, clear all)
- Effects presets (save/load effect chains)

---

## Backend Additions

| File | Purpose |
|------|---------|
| `src/history.py` | HistoryManager — SQLite backend for TTS/STT log |
| `src/profiles.py` | ProfileManager — JSON or SQLite profile CRUD |
| `src/conversation.py` | ConversationManager — turn-based synthesis queue |
| `src/effects/chain.py` | EffectsChain — pedalboard or scipy effects pipeline |
| `src/composer.py` | MultiTrackComposer — mix tracks to output audio |

## New API Endpoints

```
POST   /api/profiles                    Create profile
GET    /api/profiles                    List profiles
GET    /api/profiles/{id}               Get profile
PUT    /api/profiles/{id}               Update profile
DELETE /api/profiles/{id}               Delete profile

GET    /api/history                     List history entries (?type=tts|stt&limit=50)
DELETE /api/history/{id}               Delete one entry
DELETE /api/history                    Clear all history

POST   /api/conversations              Create conversation
GET    /api/conversations              List conversations
GET    /api/conversations/{id}         Get conversation + turns
POST   /api/conversations/{id}/render  Render all turns to audio
DELETE /api/conversations/{id}         Delete conversation

POST   /api/composer/render            Mix tracks to output audio

POST   /v1/audio/speech                (updated) — accepts `effects` param, logs to history
POST   /v1/audio/transcriptions        (updated) — logs result to history
```

---

## Dependencies

```toml
[project.optional-dependencies]
studio = [
    "pedalboard>=0.9.0",   # voice effects
    "pydub>=0.25.0",       # audio mixing for composer
]
```

Add to `Dockerfile` optional install path. Not baked in by default.

---

## Phasing

| Sub-phase | Scope | Est. effort |
|-----------|-------|-------------|
| 8a | Voice profiles API + UI selector | Small |
| 8b | History API + UI tab | Medium |
| 8c | Conversation mode | Medium |
| 8d | Voice effects | Medium |
| 8e | Multi-track composer | Large |

Recommend shipping 8a + 8b first (high value, low complexity), then 8c, then 8d + 8e as a bundle.

---

## Version Target

v0.7.0 — "Studio"
