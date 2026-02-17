# TTS Backends — Design & Roadmap

## Overview

Open Speech uses a pluggable backend architecture for text-to-speech. Each backend implements the `TTSBackend` protocol defined in `src/tts/backends/base.py`. Users select backends via model name in the API, and only the backends they install get loaded.

This document covers the existing backend, planned additions, implementation patterns, and the plugin interface.

---

## Architecture

```
API Request (model="kokoro", voice="af_heart")
    │
    ▼
┌──────────────┐
│  TTS Router  │  ← resolves model name → backend
│  (pipeline)  │
└──────┬───────┘
       │
       ├── kokoro.py      (built-in, default)
       ├── piper.py       (planned — lightweight, offline)
       ├── fish_speech.py  (planned — voice cloning)
       ├── cosyvoice.py   (planned — streaming, emotion)
       ├── qwen3_tts.py   (planned — voice cloning, design)
       └── indextts.py    (planned — emotion, duration control)
```

**Routing logic:** The `model` parameter in the API determines which backend handles the request. Each backend registers the model names it handles. If no model is specified, `TTS_DEFAULT_MODEL` is used.

---

## Backend Protocol

Every TTS backend must implement this interface (`src/tts/backends/base.py`):

```python
class TTSBackend(Protocol):
    name: str           # e.g. "kokoro", "piper", "fish-speech"
    sample_rate: int    # native output sample rate (e.g. 24000)

    def load_model(self, model_id: str) -> None: ...
    def unload_model(self, model_id: str) -> None: ...
    def is_model_loaded(self, model_id: str) -> bool: ...
    def loaded_models(self) -> list[TTSLoadedModelInfo]: ...

    def synthesize(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        lang_code: str | None = None,
    ) -> Iterator[np.ndarray]:
        """Yield float32 numpy audio chunks at native sample rate."""
        ...

    def list_voices(self) -> list[VoiceInfo]: ...
```

**Key design points:**
- `synthesize()` is a **generator** — yield chunks for streaming support
- Audio is always **float32 numpy arrays** — the pipeline handles format conversion (MP3, WAV, etc.)
- Backends manage their own model loading/unloading — the pipeline handles lifecycle (TTL, eviction)

---

## Current Backend

### Kokoro (built-in, default)

| Property | Value |
|----------|-------|
| **Package** | `kokoro` (≥0.9.4) |
| **Model size** | 82M parameters, ~330MB download |
| **Sample rate** | 24kHz |
| **Languages** | English (US/GB), Spanish, French, Hindi, Italian, Japanese, Portuguese, Mandarin |
| **Voice cloning** | No |
| **Streaming** | Yes (per-sentence chunking) |
| **CPU performance** | 2.9x realtime on CPU (tested: Xeon E5-2680 v4) |
| **GPU performance** | ~10x realtime on CUDA |
| **License** | MIT (Apache 2.0 for model weights) |

**Why it's the default:** Small, fast, MIT licensed, great quality for its size, runs well on CPU. No voice cloning, but has 52 built-in voices with blending support (`af_bella(2)+af_sky(1)`).

**Install:** Included by default. No optional dependency needed.

---

## Planned Backends

### Tier 1 — High Priority

These are mature, well-documented, and have clear Python packages or inference APIs.

#### 1. Piper

| Property | Value |
|----------|-------|
| **Package** | `piper-tts` (v1.4.1, Feb 2026) |
| **Model sizes** | 15MB – 75MB per voice (ONNX) |
| **Sample rate** | 22.05kHz (model-dependent) |
| **Languages** | 30+ languages, hundreds of voices |
| **Voice cloning** | No |
| **Streaming** | Yes (very low latency) |
| **Best for** | Lightweight, offline, edge devices, Pi |
| **License** | MIT |

**Why:** The fastest local TTS available. Tiny models, runs on anything including Raspberry Pi. ONNX runtime means no PyTorch dependency. Perfect complement to Kokoro — where Kokoro is "good quality, reasonable speed," Piper is "reasonable quality, extreme speed."

**Implementation plan:**
- Backend file: `src/tts/backends/piper.py`
- Model discovery: scan `~/.cache/piper/` for `.onnx` files, or download from Piper's model repo
- Voice listing: parse model metadata JSON (each model = one voice)
- Model name format: `piper/en_US-lessac-medium`, `piper/en_GB-alan-medium`
- Optional dep group: `pip install open-speech[piper]`

```python
# Example usage
curl -X POST http://localhost:8100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"piper","input":"Hello world","voice":"en_US-lessac-medium"}'
```

---

#### 2. Fish Speech 1.5

| Property | Value |
|----------|-------|
| **Package** | `fish-audio-sdk` (API) or local via `fish-speech` repo |
| **Model size** | ~500MB |
| **Sample rate** | 24kHz |
| **Languages** | English, Chinese, Japanese, Korean + more |
| **Voice cloning** | Yes — zero-shot from audio reference |
| **Streaming** | Yes |
| **Best for** | Highest quality, voice cloning |
| **License** | Apache 2.0 |

**Why:** Top of the TTS Arena leaderboard (ELO 1339). DualAR architecture delivers near-human quality. Zero-shot voice cloning from a short audio sample. The quality king.

**Implementation plan:**
- Backend file: `src/tts/backends/fish_speech.py`
- Two modes: local inference (GPU required, ~4GB VRAM) or Fish Audio API (cloud, API key needed)
- Voice cloning: accept reference audio via extended API parameter
- Model name format: `fish-speech` (local) or `fish-audio` (cloud API)
- Optional dep group: `pip install open-speech[fish]`

```python
# Built-in voice
curl -X POST http://localhost:8100/v1/audio/speech \
  -d '{"model":"fish-speech","input":"Hello","voice":"default"}'

# Clone from reference audio (extended API)
curl -X POST http://localhost:8100/v1/audio/speech \
  -F "input=Hello world" -F "model=fish-speech" \
  -F "reference_audio=@my_voice.wav"
```

---

#### 3. Qwen3-TTS

| Property | Value |
|----------|-------|
| **Package** | `transformers` + custom pipeline |
| **Model sizes** | 0.6B and 1.7B |
| **Sample rate** | 24kHz |
| **Languages** | Multilingual (strong English + Chinese) |
| **Voice cloning** | Yes — zero-shot |
| **Streaming** | Yes — bidirectional, first packet after 1 character |
| **Best for** | Voice design, expressive speech, streaming |
| **License** | Apache 2.0 |

**Why:** Released Jan 2026 by Alibaba's Qwen team. Two killer features: (1) **free-form voice design** — describe the voice you want in text ("a warm female voice with slight British accent") and it generates it, and (2) **extreme streaming** — first audio packet after processing just one character. The 0.6B model is surprisingly capable.

**Implementation plan:**
- Backend file: `src/tts/backends/qwen3_tts.py`
- Model loading via HuggingFace transformers or ModelScope
- Voice design: extended API parameter for voice description text
- Voice cloning: reference audio parameter (same as Fish Speech)
- Model name format: `qwen3-tts-0.6b`, `qwen3-tts-1.7b`
- GPU strongly recommended (0.6B fits in ~2GB VRAM, 1.7B needs ~4GB)
- Optional dep group: `pip install open-speech[qwen]`

```python
# Standard voice
curl -X POST http://localhost:8100/v1/audio/speech \
  -d '{"model":"qwen3-tts-0.6b","input":"Hello","voice":"default"}'

# Voice design (describe the voice you want)
curl -X POST http://localhost:8100/v1/audio/speech \
  -d '{"model":"qwen3-tts-0.6b","input":"Hello","voice_design":"A deep male voice, calm and professional, slight Southern accent"}'
```

---

### Tier 2 — Future Additions

These are promising but have more complex dependencies or are less mature.

#### 4. CosyVoice2

| Property | Value |
|----------|-------|
| **Source** | `github.com/FunAudioLLM/CosyVoice` |
| **Model size** | 0.5B |
| **Languages** | Chinese, English, Japanese, Korean, Cantonese |
| **Voice cloning** | Yes — zero-shot, cross-lingual |
| **Streaming** | Yes — 150ms first-packet latency |
| **Best for** | Low-latency streaming, emotion/dialect control |
| **License** | Apache 2.0 |

**Why:** Fastest streaming latency of any open model (150ms). Emotion and dialect control. From Alibaba's SpeechLab (same team as Qwen audio). Complex dependency tree (custom Matcha-TTS fork) makes integration harder.

**Implementation complexity:** HIGH — requires custom fork of Matcha-TTS as `third_party/`, specific torch versions. Not a simple pip install. Best done as a separate Docker layer or sidecar.

---

#### 5. IndexTTS-2

| Property | Value |
|----------|-------|
| **Source** | `github.com/index-tts/index-tts` |
| **Model size** | Large (multi-component: GPT + vocoder) |
| **Languages** | English, Chinese |
| **Voice cloning** | Yes — zero-shot |
| **Streaming** | Limited |
| **Best for** | Emotion-controlled speech, audiovisual dubbing |
| **License** | Apache 2.0 |

**Why:** Unique emotion disentanglement — clone a voice without cloning the emotion, then apply any emotion independently. Duration control for dubbing. Newer (2025), less community adoption than Fish Speech or CosyVoice.

**Implementation complexity:** MEDIUM — has pip-installable components but requires model download orchestration.

---

## Dependency Strategy

Each backend is an optional dependency group. Base Open Speech has zero TTS deps beyond Kokoro:

```toml
# pyproject.toml
[project.optional-dependencies]
piper = ["piper-tts>=1.4.0"]
fish = ["fish-speech>=1.5.0"]
qwen = ["transformers>=4.40.0", "accelerate"]
cosyvoice = ["cosyvoice @ git+https://github.com/FunAudioLLM/CosyVoice"]
indextts = ["indextts>=2.0.0"]
all-tts = ["open-speech[piper,fish,qwen]"]  # convenience bundle (Tier 1 only)
```

**Docker strategy:**
- `:cpu` — Kokoro only (default, small image)
- `:latest` (GPU) — Kokoro + Piper (both lightweight)
- `:full` (future) — All Tier 1 backends pre-installed
- Users can extend any image: `pip install open-speech[fish]` inside the container

---

## Model Selection Matrix

Help users pick the right backend:

| Need | Recommended | Why |
|------|-------------|-----|
| Fast, good enough, CPU | **Kokoro** | 82M, 2.9x realtime on CPU |
| Fastest possible, edge/Pi | **Piper** | 15MB models, ONNX, runs anywhere |
| Highest quality | **Fish Speech** | TTS Arena leader |
| Voice cloning (easy) | **Fish Speech** or **Qwen3** | Zero-shot from audio sample |
| Design a voice from text | **Qwen3-TTS** | Unique "describe the voice" feature |
| Lowest streaming latency | **CosyVoice2** | 150ms first packet |
| Emotion control | **IndexTTS-2** | Emotion disentangled from speaker |
| Multilingual | **Qwen3-TTS** or **CosyVoice2** | Strongest multi-language support |

---

## Implementation Checklist (per backend)

- [ ] Create `src/tts/backends/{name}.py` implementing `TTSBackend`
- [ ] Add optional dependency group in `pyproject.toml`
- [ ] Register backend in `src/tts/pipeline.py` router
- [ ] Add model name pattern to routing logic
- [ ] Write tests in `tests/test_tts_{name}.py` (mock model, test interface)
- [ ] Update `docs/TTS-BACKENDS.md` with real benchmarks
- [ ] Update README.md with backend table
- [ ] Test in Docker (CPU and GPU where applicable)
- [ ] Update `.env.example` if new env vars needed

---

## Build Order

1. **Piper** — simplest integration (pip install, ONNX, no PyTorch conflict), broadens CPU story
2. **Qwen3-TTS (0.6B)** — voice design is a killer feature nobody else has, reasonable VRAM
3. **Fish Speech** — quality king, voice cloning, but heavier deps
4. **CosyVoice2** — complex deps, save for later
5. **IndexTTS-2** — niche (emotion/dubbing), save for later

---

## API Extensions

For backends that support voice cloning or voice design, the standard OpenAI `/v1/audio/speech` API needs optional extensions:

```json
{
  "model": "qwen3-tts-0.6b",
  "input": "Hello world",
  "voice": "default",
  "voice_design": "A warm female voice with slight British accent",
  "reference_audio_url": "https://example.com/my_voice.wav",
  "emotion": "happy",
  "speed": 1.0
}
```

These fields are **optional** and **ignored by backends that don't support them** (e.g., Kokoro ignores `voice_design`). This keeps backward compatibility with the OpenAI SDK while exposing advanced features.

For reference audio upload (multipart):
```
POST /v1/audio/speech
Content-Type: multipart/form-data

input=Hello world
model=fish-speech
reference_audio=@voice_sample.wav
```

---

## Contributing a Backend

1. Create `src/tts/backends/your_backend.py`
2. Implement the `TTSBackend` protocol (see `base.py`)
3. Add your optional deps to `pyproject.toml`
4. Register in `src/tts/pipeline.py`
5. Add tests
6. Submit a PR

The key contract: `synthesize()` yields `np.ndarray` chunks of float32 audio at your `sample_rate`. Everything else (format conversion, streaming, caching) is handled by the pipeline.
