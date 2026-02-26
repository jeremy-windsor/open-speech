# Open Speech — Manual Test Plan

**Version:** 0.5.1+  
**Environment:** Web GUI at `https://203.0.113.10:8100/web` + JW laptop with headset  
**Tester:** Jeremy Windsor  
**Last updated:** 2026-02-18

---

## Test Environment Setup

- **Browser:** Chrome on jw-laptop (accept self-signed cert warning)
- **Audio input:** Headset mic
- **Audio output:** Headset speakers
- **GPU:** RTX 2060 8GB (CUDA)
- **Container:** `docker compose -f docker-compose.gpu.yml up -d`

---

## Phase 1: Smoke Tests (run after every rebuild)

| # | Test | Expected | Pass/Fail |
|---|------|----------|-----------|
| S1 | Browse to `https://203.0.113.10:8100/web` | Page loads, version badge shows `0.5.1` | |
| S2 | `GET /health` via browser | `{"status":"ok","version":"0.5.1","models_loaded":N}` | |
| S3 | Models tab loads | STT + TTS model list visible, no JS errors in console | |
| S4 | STT model shows "Loaded" | faster-whisper-large-v3-turbo-ct2 state = loaded | |
| S5 | No permission errors in docker logs | `docker logs open-speech --tail 20` = clean startup | |

---

## Phase 2: STT — Speech-to-Text

### 2.1 Basic Transcription
**Setup:** Headset mic plugged in, browser mic permission granted

| # | Test | Steps | Expected | Pass/Fail |
|---|------|-------|----------|-----------|
| T1 | Short phrase | Click mic, say "Hello, this is a test" | Transcript appears within 2s | |
| T2 | Longer sentence | Say 10+ word sentence clearly | Full accurate transcript | |
| T3 | Technical terms | Say "MPLS, SRv6, BGP, NETCONF" | Acronyms transcribed correctly | |
| T4 | Background noise | Say phrase with ambient noise | Transcript still accurate | |
| T5 | API endpoint | `POST /v1/audio/transcriptions` with WAV file | Returns `{"text":"..."}` | |

### 2.2 Model Switching
| # | Test | Steps | Expected | Pass/Fail |
|---|------|-------|----------|-----------|
| T6 | Load tiny model | Models tab → Systran/faster-whisper-tiny → Download → Load | Loads successfully | |
| T7 | Transcribe on tiny | Switch to tiny, transcribe phrase | Faster, slightly less accurate | |
| T8 | Unload model | Unload tiny | Memory freed, model gone from loaded list | |
| T9 | Switch back to large | Load large-v3-turbo again | Loads, transcription quality restored | |

---

## Phase 3: TTS — Text-to-Speech

### 3.1 Kokoro Backend
**Setup:** Install Provider → Load kokoro in Models tab

| # | Test | Steps | Expected | Pass/Fail |
|---|------|-------|----------|-----------|
| K1 | Basic synthesis | Speak tab → kokoro → type "Hello world" → Generate | Audio plays in browser | |
| K2 | Voice selection | Switch voice dropdown (alloy, echo, fable, onyx, nova, shimmer) | Different voices audibly distinct | |
| K3 | Voice blend UI | Check Speak tab | Blend controls visible for kokoro | |
| K4 | Speed control | Set speed to 0.8x, generate | Speech noticeably slower | |
| K5 | Speed control | Set speed to 1.5x, generate | Speech noticeably faster | |
| K6 | Long text | Paste 500+ word paragraph | Full audio generated, no cutoff | |
| K7 | Download audio | Generate then click download | MP3/WAV file downloads | |

### 3.2 Piper Backend
**Setup:** Install Provider → Load a piper model

| # | Test | Steps | Expected | Pass/Fail |
|---|------|-------|----------|-----------|
| P1 | Load piper model | Models → en_US-lessac-medium → Install Provider → Download → Load | Loads successfully | |
| P2 | Basic synthesis | Switch backend to piper → type text → Generate | Audio plays | |
| P3 | Blend controls hidden | Check Speak tab with piper active | Voice blend UI NOT visible | |
| P4 | British voice | Load en_GB-alan-medium → Generate | British accent audible | |

### 3.3 Pocket TTS Backend (pending Forge build)
**Setup:** Install Provider → Load pocket-tts

| # | Test | Steps | Expected | Pass/Fail |
|---|------|-------|----------|-----------|
| PT1 | Install provider | Install Provider for pocket-tts | Completes without error | |
| PT2 | Load model | Download → Load | Loads on CPU | |
| PT3 | Synthesis speed | Generate 30s of speech, time it | < 15s generation time | |
| PT4 | No GPU needed | Check docker logs | No CUDA errors, runs on CPU | |
| PT5 | Voice options | Check speaker dropdown | Available voices listed | |

### 3.4 Qwen3 Backend (GPU required)
**Setup:** Install Provider → Load qwen3 model

| # | Test | Steps | Expected | Pass/Fail |
|---|------|-------|----------|-----------|
| Q1 | Load model | qwen3-tts/1.7B-Base → Install Provider → Load | Loads on GPU | |
| Q2 | Basic synthesis | Generate English text | Clear English speech | |
| Q3 | Voice design visible | Check Speak tab | Voice design controls visible | |
| Q4 | Instructions field | Enter "Speak with excitement" → Generate | Noticeably more energetic | |
| Q5 | Language selection | Set language to zh → generate Chinese text | Chinese speech output | |

---

## Phase 4: Voice Cloning

| # | Test | Steps | Expected | Pass/Fail |
|---|------|-------|----------|-----------|
| VC1 | Reference audio upload | Fish/F5 backend → upload 10s WAV of voice → Generate | Cloned voice output | |
| VC2 | Clone quality | Listen to reference vs output | Recognizable similarity | |
| VC3 | Backend gating | Switch to kokoro | Voice clone UI NOT visible | |
| VC4 | API clone endpoint | `POST /v1/audio/speech/clone` with reference file | Returns cloned audio | |

---

## Phase 5: Real-Time Voice Loop (UC1)

**Goal:** Speak → transcribe → process → hear response

| # | Test | Steps | Expected | Pass/Fail |
|---|------|-------|----------|-----------|
| RT1 | STT latency | Speak phrase, measure time to transcript | < 2s for short phrase | |
| RT2 | TTS latency | Submit text, measure time to first audio | < 3s with kokoro | |
| RT3 | TTS latency (pocket) | Same test with pocket-tts | < 1.5s target | |
| RT4 | Full loop time | Voice in → text → TTS out, total time | < 5s end-to-end | |
| RT5 | Wyoming protocol | Point HA/other client at port 10400 | STT + TTS both work | |

---

## Phase 6: API Compatibility

| # | Test | Steps | Expected | Pass/Fail |
|---|------|-------|----------|-----------|
| A1 | OpenAI TTS compat | `POST /v1/audio/speech` with `model=tts-1` | Returns audio | |
| A2 | OpenAI STT compat | `POST /v1/audio/transcriptions` with `model=whisper-1` | Returns transcript | |
| A3 | Models list | `GET /v1/models` | Lists available models | |
| A4 | Streaming response | `POST /v1/audio/speech` with `stream=true` | Chunked audio response | |

---

## Phase 7: UI/UX Checks

| # | Test | Steps | Expected | Pass/Fail |
|---|------|-------|----------|-----------|
| U1 | Backend switch | Change TTS backend dropdown | Capabilities refresh, UI adapts | |
| U2 | Feature gating | Switch between backends | Only relevant controls visible | |
| U3 | Error display | Load a model that fails | Clear error message, not spinner | |
| U4 | Progress feedback | Download large model | Download progress shown | |
| U5 | Version badge | Check top of page | Shows correct version (0.5.1) | |
| U6 | Mobile layout | Open on phone browser | Usable (not broken) | |
| U7 | History tab | Generate several clips | History entries appear | |
| U8 | Download from history | Click download on past generation | File downloads | |

---

## Known Issues / Out of Scope

- Mobile layout not optimized (tracked for Phase 8)
- Qwen3 streaming (Phase 7c — not yet built)
- Real-time duplex voice (Phase 8 — Studio features)
- Voice profile persistence (Phase 8)

---

## Test Execution Log

| Date | Tester | Version | Phase | Result | Notes |
|------|--------|---------|-------|--------|-------|
| 2026-02-18 | JW | 0.5.1 | S1-S5 | — | First formal test run |

---

## Reporting Bugs

Add to `FIXES.md` in repo root with format:
```
| Fxx | [Bug description] | [backend/component] | open |
```
