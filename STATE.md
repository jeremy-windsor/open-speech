# Open Speech — STATE.md

## Current Status
Phase 2 TTS COMPLETE. 221 tests passing. 70% coverage.

## Last Session: 2026-02-17 Nightly (4:00 AM)
**Task:** Test coverage improvements — streaming unit tests + SSL tests

### Changes Made
1. **`tests/test_streaming_units.py`** — 33 new tests:
   - `resample_pcm16`: passthrough, upsample, downsample, empty, single sample, valid range, DC offset
   - `LocalAgreement2`: first input, prefix agreement, disagreement, case insensitive, flush, reset, progressive, no double-confirm
   - `StreamingSession._pcm_to_wav`: valid WAV header, sample rates, empty audio, round-trip data
   - Constants sanity checks
   - `SileroVAD` (mocked ONNX): probability, empty audio, short audio, multi-window max, reset

2. **`tests/test_ssl_utils.py`** — 7 new tests:
   - Skip when both exist, generate when missing, create parent dirs, openssl not found, openssl failure, one missing

### Coverage Changes
- ssl_utils.py: 0% → 100%
- streaming.py: 30% → 36% (pure functions covered; session logic needs integration tests with WebSocket)
- Total: 68% → 70%, 182 → 221 tests

### Phase 2 TTS Status (COMPLETE)
All checklist items done prior to this session:
- ✅ Kokoro engine embedded (auto-discovery backend)
- ✅ `/v1/audio/speech` endpoint (OpenAI-compatible)
- ✅ Voice presets + blending (`af_bella(2)+af_sky(1)` syntax)
- ✅ Multiple output formats (mp3, opus, aac, flac, wav, pcm via ffmpeg)
- ✅ Streaming TTS (chunked audio response)
- ✅ Web UI TTS tab (voice selector, blending, speed control, history, model management)
- ✅ Self-signed HTTPS (auto-generated certs)
- ✅ Additional STT backends (Moonshine, Vosk)
- ✅ Model lifecycle management (TTL eviction, max models, load/unload API)

### Next Steps
- Phase 3: Voice cloning backends (XTTS, F5-TTS) — needs GPU for testing
- Phase 3: Real-time bidirectional voice conversation (WebSocket)
- Improve streaming.py coverage (integration tests with mock WebSocket)
- Improve main.py coverage (more endpoint tests)
- Improve backend coverage (faster_whisper 43%, vosk 58%)
