# Open Speech — STATE.md

## Current Status
Phase 2 streaming bug fix in progress (2026-02-14)

## Last Session: 2026-02-14 Nightly
**Task:** Fix streaming transcription bug — Live tab VAD fires but no transcript reaches browser

### Root Cause
Client forced `AudioContext({ sampleRate: 16000 })` but browsers often silently fall back to system default (44100/48000 Hz). Audio sent at the system rate but both client and server assumed 16kHz. WAV header said 16kHz → faster-whisper processed garbled audio → empty text → no results.

### Changes Made
1. **`src/streaming.py`** — Major overhaul:
   - Added `resample_pcm16()` — linear interpolation resampler (any rate → 16kHz)
   - Added `INTERNAL_SAMPLE_RATE = 16000` constant
   - `StreamingSession` now tracks `client_sample_rate` vs internal rate
   - `_process_chunk()` resamples before VAD and stores audio at 16kHz
   - `_pcm_to_wav()` now takes explicit sample_rate parameter
   - Model pre-validation at session start (errors sent to client)
   - Error events sent to client (`{"type": "error", "message": "..."}`)
   - All `asyncio.get_event_loop()` → `asyncio.get_running_loop()`
   - INFO-level logging for speech start/end, transcription results, errors

2. **`src/static/index.html`** — Client fixes:
   - AudioContext uses system default rate (no forced 16kHz)
   - Actual sample rate reported to server in WebSocket URL
   - Console logging for debugging
   - Error event handling from server
   - Status display shows actual sample rate

3. **`src/main.py`** — `get_event_loop()` → `get_running_loop()`

4. **`tests/test_streaming.py`** — 18 new unit tests:
   - LocalAgreement2: 9 tests (agreement, flush, reset, edge cases)
   - resample_pcm16: 6 tests (noop, downsample, upsample, common rates)
   - _pcm_to_wav: 3 tests (header validity, sample rate, data integrity)

5. **`tests/test_ws_client.py`** — Manual WebSocket test client for CLI debugging

### Test Results
24/24 passing (6 existing API + 18 new streaming)

### Next Steps
- Build GPU Docker image and push to Docker Hub
- Deploy to 192.0.2.24 and test with real browser/mic
- Phase 3 features (model management UI, export, auth)
