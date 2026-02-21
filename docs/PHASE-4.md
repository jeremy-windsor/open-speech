# Phase 4 — Advanced TTS: Voice Cloning, Design & New Backends

## Summary

Phase 4 added two new TTS backends (Qwen3-TTS and Fish Speech), extended the TTS API with voice cloning and voice design capabilities, and added voice presets to the web UI.

**Commit:** `0ea4d4c` (Phase 4 core)
**Follow-up fixes:** `c128533`, `75fb457`, `773095f`, `878d803`

---

## What Shipped

### New TTS Backends

#### Qwen3-TTS
- Two model sizes: 0.6B (~1.2GB) and 1.7B (~3.4GB)
- 4 built-in voices + voice design (describe a voice in natural language)
- Zero-shot voice cloning from reference audio
- Streaming support
- Optional dependency: `pip install open-speech[qwen]`

### Extended TTS API

| Endpoint | Description |
|----------|-------------|
| `POST /v1/audio/speech` | Added `voice_design` and `reference_audio` fields |
| `POST /v1/audio/speech/clone` | New multipart endpoint for voice cloning |
| `GET /api/voice-presets` | List configured voice presets |

- `voice_design`: text description of desired voice (Qwen3-TTS only)
- `reference_audio`: base64-encoded audio for cloning (Qwen3-TTS)
- Fields are ignored by backends that don't support them (Kokoro, Piper)

### Voice Presets

- YAML-based configuration via `TTS_VOICES_CONFIG` env var
- Dropdown in web UI Speak tab
- Default presets: Will, Female, British Butler
- See `voice-presets.example.yml`

### Web UI Updates

- Voice presets dropdown in Speak tab
- TTS history entries with download + delete buttons
- Stream toggle tooltip
- Dynamic version badge from `/health` endpoint

---

## Post-Phase 4 Fixes

| Fix | Commit |
|-----|--------|
| Speed slider: 0.25 steps → 5% increments, min 0.5x | `75fb457` |
| Voice presets matched to actual available voices | `75fb457` |
| Kokoro filtered from STT model dropdown | `c128533` |
| Kokoro-82M removed from STT listing in Models tab | `c128533` |
| Provider check (show "not installed" vs Download) | `c128533` |
| Version badge dynamic loading | `c128533` |
| TTS history download + delete buttons | `c128533` |
| Stream toggle tooltip | `c128533` |
| Minor package fixes | `773095f` |
| FIXES.md intake tracker added | `878d803` |

---

## Test Coverage

- 23 new tests added in Phase 4
- **332 total tests passing**
- Coverage includes: Qwen3-TTS backend, voice cloning endpoint, voice presets API, voice design parameter validation

---

## Configuration

```bash
# Use Qwen3-TTS as default TTS backend
TTS_MODEL=qwen3-tts-0.6b
TTS_DEVICE=cuda              # GPU strongly recommended

# Voice presets
TTS_VOICES_CONFIG=voice-presets.yml
```

---

## Dependencies

| Group | Packages | Size |
|-------|----------|------|
| `[qwen]` | transformers, accelerate, torch | ~2GB+ |

Qwen is **not** included in `[all]` due to size. Install separately as needed.
