# Phase 6 — Production Hardening (v0.5.0)

## Implemented

### 6a — TTS Response Caching
- Added `src/cache/tts_cache.py` with SHA256 keying on `(text + voice + speed + format)`.
- File-backed cache with LRU eviction by mtime and configurable max size.
- New config:
  - `TTS_CACHE_ENABLED=false`
  - `TTS_CACHE_MAX_MB=500`
  - `TTS_CACHE_DIR=/var/lib/open-speech/cache`
- `/v1/audio/speech` now checks cache before generation and stores generated audio after encoding.
- Cache bypass supported via query param: `cache=false`.
- Wyoming TTS path also uses cache (PCM cache payloads).
- Added periodic cleanup task in app lifespan.

### 6b — Speaker Diarization
- Added `src/diarization/pyannote_diarizer.py`.
- Optional dependency group `diarize = ["pyannote.audio>=3.0"]`.
- New config: `STT_DIARIZE_ENABLED=false`.
- `/v1/audio/transcriptions?diarize=true` returns:
  - `{"text": "...", "segments": [{"speaker": "SPEAKER_00", "start": 0.0, "end": 2.5, "text": "..."}]}`
- If pyannote is missing, API returns install guidance error.

### 6c — Audio Pre/Post Processing
- Added `src/audio/preprocessing.py`:
  - Optional noise reduction (`noisereduce`) via `STT_NOISE_REDUCE=false`
  - Gain normalization via `STT_NORMALIZE=true`
- Added `src/audio/postprocessing.py`:
  - Silence trimming via `TTS_TRIM_SILENCE=true`
  - Output normalization via `TTS_NORMALIZE_OUTPUT=true`
- Wired into HTTP STT/TTS and Wyoming STT/TTS handlers.

### 6d — Python Client SDK
- Added `src/client/__init__.py` with `OpenSpeechClient`.
- Includes sync and async methods for:
  - `transcribe(audio)`
  - `speak(text, voice, speed)`
  - `stream_transcribe(audio_stream)` (placeholder streaming interface)
  - `realtime_session()` (placeholder async session interface)
- Added optional dependency group `client` and example `examples/python_client.py`.

### 6e — Pronunciation Control
- Added `src/pronunciation/dictionary.py`:
  - JSON/YAML dictionary loading using `TTS_PRONUNCIATION_DICT`
  - Text substitution before synthesis
- Added SSML subset handling (`input_type=ssml`) for:
  - `<break time="...ms"/>`
  - `<emphasis>`
  - `<phoneme>`
- TTS pipeline applies pronunciation substitutions and SSML conversion before synthesis.

## Packaging / Versioning
- Bumped project version to `0.5.0` in `pyproject.toml`.
- Added optional extras:
  - `diarize`
  - `noise`
  - `client`
- Left `[all]` unchanged for heavyweight optional isolation.
