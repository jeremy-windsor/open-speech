# Qwen3-TTS Deep Integration — Design Document

## Overview

Our current `qwen3_backend.py` is a thin wrapper. The real Qwen3-TTS ecosystem has **5 models** across **3 capabilities** with features we should fully exploit. This doc covers what exists, what we should integrate, and how it maps to Open Speech.

## Qwen3-TTS Model Family

| Model | Size | Capability | Streaming | Instruction Control |
|-------|------|-----------|-----------|-------------------|
| `Qwen3-TTS-12Hz-1.7B-CustomVoice` | ~3.5GB | 9 premium voices + style control via instructions | ✅ | ✅ |
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | ~3.5GB | Create any voice from text description | ✅ | ✅ |
| `Qwen3-TTS-12Hz-1.7B-Base` | ~3.5GB | 3-second voice cloning + fine-tuning base | ✅ | ❌ |
| `Qwen3-TTS-12Hz-0.6B-CustomVoice` | ~1.2GB | 9 premium voices (lighter, no instruction control) | ✅ | ❌ |
| `Qwen3-TTS-12Hz-0.6B-Base` | ~1.2GB | Voice cloning (lighter) | ✅ | ❌ |
| `Qwen3-TTS-Tokenizer-12Hz` | ~200MB | Audio codec — encode/decode speech to tokens | N/A | N/A |

**Key insight:** These aren't variants of one model — they're **different models for different jobs**. Our backend should load the right one for the task.

## Features to Integrate

### 1. Instruction-Controlled Speech (HIGH PRIORITY)

**What:** Natural language control over how text is spoken.
- "Speak angrily" → angry delivery
- "Whisper this" → whispered speech
- "Read like a news anchor" → broadcast style
- "Speak slowly with sadness" → emotional + pace control

**Current state:** Our API accepts `voice_design` text but it's underutilized.

**Integration plan:**
- Map the `instruct` parameter to our existing `voice_design` field on `/v1/audio/speech`
- When `voice_design` is set AND model is CustomVoice → pass as `instruct` to `generate_custom_voice()`
- When `voice_design` is set AND model is VoiceDesign → pass as `instruct` to `generate_voice_design()`
- This makes instruction control work through the existing API with zero breaking changes

**API example:**
```json
POST /v1/audio/speech
{
  "input": "I can't believe you did that.",
  "voice": "Ryan",
  "voice_design": "Speak with disbelief and slight anger"
}
```

### 2. Three-Model Architecture (HIGH PRIORITY)

**What:** Load the right model for the right task automatically.

**Model selection logic:**
| User intent | Model loaded | Method called |
|------------|-------------|--------------|
| Named voice (Ryan, Vivian, etc.) | CustomVoice 1.7B | `generate_custom_voice()` |
| Named voice + instruction | CustomVoice 1.7B | `generate_custom_voice(instruct=...)` |
| Voice from text description | VoiceDesign 1.7B | `generate_voice_design(instruct=...)` |
| Clone from reference audio | Base 1.7B | `generate_voice_clone()` |
| Lightweight named voice | CustomVoice 0.6B | `generate_custom_voice()` |
| Lightweight clone | Base 0.6B | `generate_voice_clone()` |

**Config:**
```env
TTS_QWEN3_MODEL=1.7B          # or 0.6B for resource-constrained
TTS_QWEN3_MODELS_DIR=/models/qwen3  # shared cache for all 3 models
```

**Implementation:** The backend loads models on-demand. First CustomVoice request loads CustomVoice. First clone request loads Base. VoiceDesign loads only when voice_design is used without a named speaker. Models share the tokenizer.

### 3. Voice Design → Clone Workflow (HIGH PRIORITY)

**What:** The killer workflow. Design a voice with words → generate reference clip → reuse forever.

**Flow:**
1. User describes a voice: "Deep male baritone, calm and authoritative, British accent"
2. VoiceDesign model generates a reference clip
3. We create a reusable `voice_clone_prompt` from that clip
4. All future requests with that voice use the cached prompt (no re-extraction)
5. Save as a named preset

**API:**
```json
POST /api/voice-presets/create
{
  "name": "Commander",
  "description": "Deep male baritone, calm and authoritative, British accent",
  "sample_text": "The mission parameters have been updated. All teams proceed to waypoint alpha."
}
```

Response creates a persistent preset with the clone prompt cached. All subsequent `/v1/audio/speech` calls with `voice: "Commander"` use the pre-computed prompt.

**This replaces our static voice presets with dynamically generated ones.** Users can create any voice they can describe.

### 4. 9 Premium Built-in Voices (MEDIUM)

**What:** Qwen3 ships 9 high-quality voices with native language specialties.

| Speaker | Description | Native Language |
|---------|------------|-----------------|
| Vivian | Bright, slightly edgy young female | Chinese |
| Serena | Warm, gentle young female | Chinese |
| Uncle_Fu | Seasoned male, low mellow timbre | Chinese |
| Dylan | Youthful Beijing male, clear natural | Chinese (Beijing) |
| Eric | Lively Chengdu male, slightly husky | Chinese (Sichuan) |
| Ryan | Dynamic male, strong rhythmic drive | English |
| Aiden | Sunny American male, clear midrange | English |
| Ono_Anna | Playful Japanese female, light nimble | Japanese |
| Sohee | Warm Korean female, rich emotion | Korean |

**Integration:** Expose these as selectable voices in the web UI Speak tab dropdown. Auto-detect language from speaker selection as a default.

### 5. Streaming Generation — 97ms First Packet (HIGH PRIORITY)

**What:** Qwen3-TTS supports native streaming via its dual-track architecture. First audio packet after just one character of input.

**Integration:**
- When `stream=true` on `/v1/audio/speech`, use Qwen3's streaming generation
- Pipe chunks directly to HTTP chunked transfer encoding
- Wire into Wyoming TTS handler for streaming Home Assistant responses
- Wire into Realtime API for `/v1/realtime` audio output

**This is the biggest latency win.** Our current streaming chunks audio after full generation. Native streaming means audio starts before generation completes.

### 6. Batch Inference (MEDIUM)

**What:** All `generate_*` methods accept lists for batch processing.

**Integration:**
- Add batch endpoint: `POST /v1/audio/speech/batch`
- Accept array of requests, return array of audio files
- Useful for: audiobook generation, bulk voiceover, subtitle dubbing
- Single model load, multiple generations

### 7. 10-Language Support (LOW — future)

**What:** Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian.

**Integration:** We're English-only for now, but the model supports all 10 natively. When we're ready for multilingual:
- Add `language` parameter to `/v1/audio/speech` (default "Auto" for detection)
- Auto-detect from text if not specified
- Voice/language compatibility validation

### 8. vLLM Backend (MEDIUM — production scaling)

**What:** Qwen3-TTS works with vLLM for production-scale inference with batching, paging, and GPU memory optimization.

**Integration:** Add `qwen3-vllm` as an alternative backend that connects to a vLLM server instead of loading models directly. For users with heavy TTS workloads, this is the scaling path.

```env
TTS_QWEN3_BACKEND=native    # direct model loading (default)
TTS_QWEN3_BACKEND=vllm      # connect to vLLM server
TTS_QWEN3_VLLM_URL=http://localhost:8000
```

### 9. Audio Tokenizer (LOW — advanced)

**What:** `Qwen3TTSTokenizer` can encode any audio to discrete tokens and decode back. Useful for:
- Audio compression/transport
- Training data preparation
- Audio manipulation in token space

**Integration:** Expose as utility endpoint:
- `POST /api/audio/encode` → returns tokens
- `POST /api/audio/decode` → returns audio from tokens

---

## Implementation Plan

### Phase 7a — Core Qwen3 Upgrade (1 session)
1. Rewrite `src/tts/backends/qwen3_backend.py` to use `qwen-tts` package properly
2. Implement 3-model auto-selection (CustomVoice / VoiceDesign / Base)
3. Wire `voice_design` field → `instruct` parameter
4. Expose 9 premium voices in model registry
5. Add `language` parameter (default "Auto")
6. Update web UI Speak tab with Qwen3 voices

### Phase 7b — Voice Design → Clone Workflow (1 session)
1. Implement `/api/voice-presets/create` with VoiceDesign → clone prompt caching
2. Persistent preset storage (file-based, survives restart)
3. Web UI preset builder: describe voice → preview → save
4. Reusable `voice_clone_prompt` for zero-overhead repeated generation

### Phase 7c — Native Streaming (1 session)
1. Implement Qwen3 streaming generation in backend
2. Wire to chunked HTTP responses
3. Wire to Wyoming TTS handler
4. Wire to Realtime API audio output
5. Benchmark: measure time-to-first-audio

### Phase 7d — Production Features (1 session)
1. Batch inference endpoint
2. vLLM backend option
3. Audio tokenizer utility endpoints

---

## Dependencies

```
qwen-tts>=0.1.0          # Official Qwen3-TTS package
flash-attn>=2.0           # Optional, recommended for GPU memory
```

**Note:** `qwen-tts` pulls in `transformers`, `torch`, `soundfile`. These overlap with existing deps. Add as optional group: `qwen3 = ["qwen-tts>=0.1.0"]`.

Flash attention is optional but strongly recommended for the 1.7B models. Without it, VRAM usage roughly doubles.

## Hardware Requirements

| Config | VRAM | Models |
|--------|------|--------|
| Minimum (0.6B) | ~4GB | CustomVoice 0.6B OR Base 0.6B |
| Standard (1.7B single) | ~8GB | One 1.7B model at a time |
| Full (1.7B all three) | ~16GB | CustomVoice + VoiceDesign + Base loaded simultaneously |
| Production (vLLM) | ~12GB | vLLM manages memory efficiently |

Your RTX 2060 (6GB) can run one 1.7B model at a time with bfloat16, or both 0.6B models simultaneously. The model manager handles load/unload automatically.

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| `qwen-tts` package breaking changes | Pin version, test on update |
| VRAM exhaustion with multiple models | Model manager TTL eviction (already built) |
| Flash attention compatibility | Optional dep, graceful fallback to standard attention |
| Tokenizer download size (~200MB shared) | Cached in Docker volume, one-time download |
| Quality difference between 0.6B and 1.7B | Default to 1.7B, let users opt into 0.6B for speed |

---

## Voicebox Feature Parity (voicebox.sh)

Voicebox is a Tauri desktop app built on Qwen3-TTS + Whisper. Desktop-first creative studio. We're API-first voice assistant backend. Different niches, but their UX features are worth adopting.

### Phase 8 — Studio Features

#### 8a — Persistent Voice Profiles
- Create profiles from audio samples or voice design descriptions
- Store as files (JSON metadata + cached clone prompts + reference audio)
- Import/export profiles (zip with metadata + audio)
- Multi-sample support — combine multiple reference clips for higher quality
- API: `POST /api/profiles`, `GET /api/profiles`, `DELETE /api/profiles/{id}`
- Web UI: Profile manager tab

#### 8b — Persistent Generation History
- SQLite database for all generated audio (text, voice, params, audio path)
- Search by text, voice, date range
- Re-generate with one click (same params)
- Filter/sort in web UI
- Auto-cleanup by age/size configurable
- API: `GET /api/history`, `DELETE /api/history/{id}`

#### 8c — Conversation Mode
- Multi-speaker dialogue generation from script
- Auto turn-taking with configurable pause between speakers
- Input format: `[Speaker1] line\n[Speaker2] line\n...`
- Output: single audio file with all speakers mixed sequentially
- API: `POST /v1/audio/conversation`
- Web UI: Conversation tab with speaker assignment

#### 8d — Voice Effects Pipeline
- Post-processing effects chain on TTS output
- Pitch shift (semitones)
- Reverb (room size, wet/dry)
- Speed change (without pitch shift, time-stretch)
- EQ (basic high/low shelf)
- API: `effects` parameter on `/v1/audio/speech`
- Use `soundfile` + `scipy.signal` or `pedalboard` library

#### 8e — Multi-Track Composition API
- Combine multiple audio clips with timing offsets
- Simple mix: sequential or overlapping with crossfade
- API: `POST /v1/audio/compose` with array of {audio, offset_ms, volume}
- Output: single mixed audio file
- Not a full DAW — just programmatic composition

---

## Competitive Impact

With full Qwen3 integration, Open Speech becomes:
- **The only open-source server** with instruction-controlled TTS ("speak angrily")
- **The only unified platform** with voice design from text description
- **The first** to offer the VoiceDesign → Clone workflow via REST API
- **97ms streaming** competitive with commercial APIs

This is the differentiator. Kokoro is good for lightweight TTS. Qwen3 is the premium engine.
