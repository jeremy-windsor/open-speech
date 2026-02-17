# Phase 5: Voice Assistant Integration

Phase 5 transforms Open Speech from a dev tool into a production voice assistant backend. The focus is real-time audio pipelines, Home Assistant integration, and API compatibility with the emerging voice AI ecosystem.

## Phase 5a — Wyoming Protocol Support 🏠

**Why:** Home Assistant is the largest open-source smart home platform. Wyoming is its native voice protocol. Supporting it makes Open Speech a drop-in replacement for separate Piper + Whisper containers — one unified server instead of two.

**What:**
- Wyoming TCP server (default port 10400)
- `Describe` event → advertise STT + TTS capabilities
- `Transcribe` event → route to active STT backend
- `Synthesize` event → route to active TTS backend  
- `AudioChunk` / `AudioStart` / `AudioStop` event handling
- Auto-discovery via zeroconf/mDNS (optional but nice)
- Works alongside existing HTTP/WebSocket API (separate port)

**Implementation:**
- New module: `src/wyoming/server.py` — async TCP server using `wyoming` Python package
- `src/wyoming/stt_handler.py` — bridges Wyoming audio events → Open Speech STT pipeline
- `src/wyoming/tts_handler.py` — bridges Wyoming synthesize events → Open Speech TTS pipeline
- Config: `OS_WYOMING_ENABLED=true`, `OS_WYOMING_PORT=10400`
- The `wyoming` package is MIT licensed, pure Python, minimal deps

**Testing:**
- Unit tests for event parsing/generation
- Integration test with `wyoming-cli` tool
- HA discovery test (zeroconf advertisement)

**Acceptance:**
- Home Assistant auto-discovers Open Speech as both STT and TTS provider
- Voice pipeline works end-to-end: wake word → STT (Open Speech) → intent → TTS (Open Speech)
- Latency competitive with standalone Piper/Whisper (<500ms STT, <300ms TTS first chunk)

---

## Phase 5b — Voice Activity Detection (VAD) 🎙️

**Why:** Every real-time voice system needs to know when someone starts and stops talking. Without VAD, you're either streaming silence (wasting compute) or requiring push-to-talk (bad UX). VAD also fixes the broken mic transcription in the web UI.

**What:**
- Silero VAD integration (ONNX, <2MB model, MIT licensed)
- VAD-gated WebSocket STT endpoint
- Configurable speech thresholds (sensitivity, min speech duration, silence timeout)
- VAD state events on WebSocket (speech_start, speech_end, vad_confidence)
- Web UI mic button uses VAD for automatic start/stop

**Implementation:**
- New module: `src/vad/silero.py` — Silero VAD wrapper
- Update `src/streaming/websocket_stt.py` — VAD gate before sending to STT
- New endpoint: `GET /v1/audio/stream/vad` — VAD-enabled WebSocket (or flag on existing)
- Config: `STT_VAD_ENABLED=true`, `STT_VAD_THRESHOLD=0.5`, `STT_VAD_SILENCE_MS=800`
- Web UI: update mic handler to use VAD events for record state

**Testing:**
- VAD accuracy tests with known speech/silence samples
- WebSocket integration with VAD gating
- Latency measurement (VAD processing overhead)

**Acceptance:**
- Web UI mic captures speech only (no dead air)
- WebSocket clients receive speech_start/speech_end events
- CPU overhead of VAD < 5% additional load

---

## Phase 5c — OpenAI Realtime API Compatibility 🔌

**Why:** OpenAI's Realtime API (`/v1/realtime`) is becoming the de facto standard for voice AI apps. Supporting it means instant compatibility with ChatGPT voice mode clients, coding assistants, and dozens of third-party apps — without those apps changing a line of code.

**What:**
- WebSocket endpoint: `GET /v1/realtime` with model parameter
- Session management (create, update, close)
- Input audio buffer handling (append, commit, clear)
- Server VAD mode (using our Silero VAD from 5b)
- Audio format negotiation (pcm16, g711_ulaw, g711_alaw)
- Response generation with streaming audio output
- Conversation item management

**Implementation:**
- New module: `src/realtime/server.py` — WebSocket handler implementing OpenAI Realtime protocol
- `src/realtime/session.py` — session state management
- `src/realtime/audio_buffer.py` — input audio buffering with VAD
- Wire to existing STT + TTS pipelines for actual processing
- Config: `OS_REALTIME_ENABLED=true`

**Note:** We implement the *audio transport and STT/TTS* portion. The "conversation" / LLM portion is out of scope — clients bring their own LLM. We're the ears and mouth, not the brain.

**Testing:**
- Protocol conformance tests against OpenAI Realtime API spec
- Session lifecycle tests
- Audio round-trip tests (send audio → get transcription + TTS response)

**Acceptance:**
- OpenAI Realtime API client libraries connect and work
- Audio input → transcription → TTS output round-trip < 1s
- Compatible with at least 2 existing OpenAI Realtime client apps

---

## Phase 6 — Production Hardening

### 6a — TTS Response Caching
- Hash(text + voice + speed + format) → cached audio file
- LRU eviction with configurable max cache size
- Config: `TTS_CACHE_ENABLED=true`, `TTS_CACHE_MAX_MB=500`
- Serves cached responses in <10ms vs 200ms+ generation

### 6b — Speaker Diarization
- `pyannote.audio` integration for "who said what"
- Optional flag on `/v1/audio/transcriptions`: `diarize=true`
- Returns segments with speaker labels
- GPU recommended, falls back to CPU

### 6c — Audio Pre/Post Processing
- **Input:** Noise reduction (noisereduce or RNNoise), gain normalization
- **Output:** Silence trimming, volume normalization, crossfade for streamed chunks
- Config: `STT_NOISE_REDUCE=true`, `TTS_NORMALIZE=true`

### 6d — Client SDKs
- `open-speech-client` Python package: streaming STT, TTS, VAD, Wyoming
- `@open-speech/client` npm package: browser WebSocket client with audio handling
- Both handle: audio encoding, WebSocket management, reconnection, VAD events

### 6e — Pronunciation Control
- Custom pronunciation dictionary (JSON/YAML)
- SSML subset support for Kokoro/Piper (pause, emphasis, phoneme)
- IP addresses, acronyms, technical terms read correctly

---

## Priority & Dependencies

```
5a Wyoming ──────────────────────────► HA integration
5b VAD ──────────┬──────────────────► Real-time STT, web UI mic fix
                 │
5c Realtime API ─┘ (depends on 5b) ► OpenAI ecosystem compatibility

6a Caching ─────────────────────────► Production latency
6b Diarization ─────────────────────► Multi-speaker use cases
6c Audio Processing ────────────────► Audio quality
6d Client SDKs ─────────────────────► Developer adoption
6e Pronunciation ───────────────────► TTS quality
```

## Competitive Landscape

| Feature | Open Speech | Piper+Whisper | Speaches | Coqui |
|---------|------------|---------------|----------|-------|
| Unified STT+TTS | ✅ | ❌ (separate) | ✅ (dead) | ❌ (TTS only) |
| Wyoming Protocol | 5a | ✅ (separate) | ❌ | ❌ |
| OpenAI API compat | ✅ | ❌ | ✅ | ❌ |
| Realtime API | 5c | ❌ | ❌ | ❌ |
| VAD integrated | 5b | ❌ | ❌ | ❌ |
| Voice cloning | ✅ | ❌ | ❌ | ✅ |
| Model hot-swap | ✅ | ❌ | Partial | ❌ |
| Web UI | ✅ | ❌ | ✅ | ❌ |
| GPU + CPU | ✅ | ✅ | ✅ | ✅ |

After Phase 5, Open Speech is the only open-source project covering all columns. That's the goal.
