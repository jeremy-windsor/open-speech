# Models

Models are downloaded on first use, or with Download in the web app's Models view, and cached on disk. The tables
list the common IDs: `src/model_registry.py` contains all 15 STT and 40 TTS entries, and `/api/models` lists what
the running server can use. Which models stay loaded is in [Model residency](#model-residency).

## Speech-to-text models

| Model | Size | Backend | Languages |
|---|---:|---|---|
| `nvidia/parakeet-tdt-0.6b-v2` | ~3.7GB (GPU), ~0.7GB (CPU) | Parakeet (ONNX) | English |
| `deepdml/faster-whisper-large-v3-turbo-ct2` | ~1.6GB | faster-whisper | 99+ |
| `Systran/faster-whisper-large-v3` | ~3.0GB | faster-whisper | 99+ |
| `Systran/faster-whisper-medium` | ~1.5GB | faster-whisper | 99+ |
| `Systran/faster-whisper-small` | ~500MB | faster-whisper | 99+ |
| `Systran/faster-whisper-base` | ~150MB | faster-whisper | 99+ |
| `Systran/faster-whisper-tiny` | ~75MB | faster-whisper | 99+ |

## Parakeet

`nvidia/parakeet-tdt-0.6b-v2` is the default STT model of the Docker images and the Compose files. It runs NVIDIA's
Parakeet TDT 0.6B v2 through [onnx-asr](https://github.com/istupakov/onnx-asr) and ONNX Runtime, from the
published ONNX export `istupakov/parakeet-tdt-0.6b-v2-onnx` at a pinned revision.

- **English only.** A request with another `language` gets `400 language_unsupported`, naming a Whisper model to
  use instead; no language, `auto` and `en` (`en-US`, `en-GB`, ...) are accepted. Translations get
  `400 translation_unsupported` (set `STT_TRANSLATE_MODEL` to a Whisper model that translates). Wyoming lists it
  with English only. Punctuation and capitals are part of its output.
- **Fastest.** On an 8 GB RTX 2070 SUPER a 3 s command takes about 0.04 s and a 10 s sentence 0.09 s, against
  0.29 s and 0.44 s for Whisper large-v3-turbo, at a lower word error rate on English. On an 8-core CPU (int8) it
  is about as fast as Whisper base with turbo's accuracy.
- **Files.** On a GPU the 2.5 GB fp32 export is downloaded and converted to fp16 once, on the first load: about
  7 minutes in all, most of it the download, and up to about 6 GB of free RAM for the conversion, which runs in a
  process of its own (if it runs out of memory, the load fails with a message saying so and the server keeps
  running). The 1.2 GB result is kept beside the download in the model cache, so the cache holds about 3.7 GB, and
  later loads read it directly. On a CPU the published int8 files (0.7 GB) are used as they are.
  `STT_COMPUTE_TYPE` applies to Whisper only. So that the first request doesn't wait for this, press Download on
  it in Models once the server is up, or set `STT_PRELOAD_MODELS=nvidia/parakeet-tdt-0.6b-v2`: the preload runs in
  the background after the server starts, so the server answers meanwhile, and a request for Parakeet before it
  finishes waits for that same download and load instead of starting another.
- **Memory.** About 1.5 GB of GPU memory once loaded. ONNX Runtime's memory pool is capped at
  `STT_PARAKEET_GPU_MEM_LIMIT_MB` (2304 MB), so Parakeet and Kokoro stay loaded together on an 8 GB card with
  room for a Whisper model on demand. Long audio is cut with Silero VAD into pieces of at most 20 s (a stretch with
  no pause is cut at its quietest point), and streaming STT ends an utterance at about 24 s.
- **Requests.** `prompt` and `temperature` are accepted and ignored: its decoder is greedy and takes no prompt.
  `verbose_json` has one segment per sentence, with `start`, `end`, `avg_logprob` and word times in `words`; `srt`
  and `vtt` cues follow the segments.

To make it the default on a source install, or to choose another default, set `STT_MODEL` (in `.env`, or in Compose
through `.env`). A request that names a Whisper model loads it next to the pinned Parakeet, and `OS_MODEL_TTL`
unloads it after it has been idle.

## Text-to-speech models

| Model | Size | Backend | Notes |
|---|---:|---|---|
| `kokoro` | ~330MB | Kokoro | default backend, many voices, blend syntax in `voice` |
| `pocket-tts` | ~220MB | Pocket-TTS | built-in voices, backend advertises streaming support |
| `supertonic` | ~383MB | Supertonic 3 (ONNX) | voices `M1`–`M5` and `F1`–`F5`, blends, optional `steps` (1–32, default 8), speed 0.7–1.9 (others are held to it, named in `X-Speed-Applied`); GPU when ONNX Runtime has CUDA, else CPU (about 0.6 GB of RAM) |
| `piper/en_US-lessac-medium` | ~60MB | Piper | one voice per model |
| `piper/en_US-joe-medium` | ~60MB | Piper | one voice per model |
| `piper/en_US-amy-medium` | ~60MB | Piper | one voice per model |
| `piper/en_US-arctic-medium` | ~73MB | Piper | 18 speakers (`piper/en_US-arctic-medium/<speaker>`; list them with `/v1/audio/voices?model=piper/en_US-arctic-medium`) |
| `piper/en_GB-alan-medium` | ~60MB | Piper | one voice per model |
| `qwen3/0.6b-custom-voice` | ~1.8GB | isolated Qwen3 provider | 9 official preset voices |
| `qwen3/0.6b-base` | ~1.8GB | isolated Qwen3 provider | reference cloning with an exact transcript; hidden unless `QWEN3_ENABLE_BASE=true` (the voice-models Compose file sets it) |
| `qwen3/1.7b-custom-voice` | ~4.2GiB | isolated Qwen3 provider | the same 9 voices, with an optional style in `instructions`; best on newer or bigger GPUs |
| `qwen3/1.7b-voice-design` | ~4.2GiB | isolated Qwen3 provider | a voice made from a description (`instructions` or `voice_design`); best on newer or bigger GPUs |
| `qwen3/1.7b-base` | ~4.2GiB | isolated Qwen3 provider | reference cloning with an exact transcript, hidden like the 0.6B Base; best on newer or bigger GPUs |
| `chatterbox/regular` | ~8.6GiB | isolated Chatterbox provider | English reference cloning |
| `chatterbox/turbo` | ~5.4GiB | isolated Chatterbox provider | faster English cloning, native speech tags |
| `fish/s2-pro` | ~4.7GiB | isolated Fish Audio S2 provider | multilingual cloning with an exact transcript, 4-bit (NF4) S2 Pro |
| `breeze/tts-2` | ~3.3GiB | isolated Breeze TTS 2 provider | English and Chinese voices designed from a description, and cloning with an exact transcript; Q8_0 GGUF through a C++ port |

Sizes are approximate disk cache usage, not GPU memory. See [TTS backends](TTS-BACKENDS.md) for each backend's
details and the conformance commands. Chatterbox, Qwen3 Base, Fish Audio S2 and Breeze TTS 2 are cloning models,
with no voices of their own: [Cloning models](CLONING-MODELS.md) gives each one's languages, how to give it a
voice, and how to start its worker.

## GPU voice worker models

The Qwen3, Chatterbox, Fish Audio S2 and Breeze TTS 2 models run in optional GPU workers, off by default; how to
turn one on is in [GPU voice workers](CONFIGURATION.md#gpu-voice-workers).

Qwen3's 1.7B models (`qwen3/1.7b-custom-voice`, `qwen3/1.7b-voice-design`, `qwen3/1.7b-base`) are best on newer or
bigger GPUs: each holds about 5.2 GB of GPU memory, and on an RTX 2070 SUPER they speak at about a third of real
time. The worker holds one Qwen3 model at a time, so loading another unloads the one before; on an 8 GB card a
1.7B load may unload the idle Parakeet to make room, and Kokoro stays loaded. Each first load fetches about 4.5 GB.

Fish Audio S2 (`fish/s2-pro`) runs the S2 Pro checkpoint quantized to 4 bits (NF4), about 3.3 GB of GPU memory. Its
first load fetches about 5 GB of weights and compiles for a few minutes (`FISH_COMPILE=false` skips the compile), and
after each unload the worker restarts to free its memory.

Breeze TTS 2 (`breeze/tts-2`) runs through the community C++ port Breeze-TTS-2.cpp with its Q8_0 weights, about
4.4 GB of GPU memory. Its first load fetches about 3.6 GB of weights. It designs a voice from a description and clones
one from a recording with its exact transcript; inline cues such as `(laugh)` are part of the text. It has two
options of its own (`capabilities.options`): **Guidance** (`cfg_scale`, 1–5) is how closely it follows the
description, chosen per request when left out (3 for a description, 1 for a clone, 2.5 with an inline cue; the
`BREEZE_DESIGN_CFG`, `BREEZE_CLONE_CFG` and `BREEZE_CUE_CFG` settings), and above 3 can sound harsh; **Variation**
(`temperature`, 0.1–1.3, default 0.9) is lower for a steadier voice and higher for a livelier one. On an 8 GB card
it fits next to Kokoro once Parakeet is unloaded, which a load from Models does when it must (see
[GPU memory guard](#gpu-memory-guard)). Its image is built for Turing (RTX 20) GPUs with code that newer ones compile
at their first load; `--build-arg BREEZE_CUDA_ARCHITECTURES=86` (RTX 30, or `89`, `120`) builds it for your card.

A worker stays up while its model is unloaded. In **Models**, use
**Download** to cache a model or **Load** to activate it. In **Voices › Recordings**, choosing a clone
model and clicking **Test clone** on a recording performs the load automatically. An explicit load unloads the other
loaded TTS models first. A TTS model that a later request loads on first use, such as Kokoro for a
speech request, then stays loaded beside it: the pinned `TTS_MODEL` until an explicit unload or load,
and any other model until idle or LRU eviction unloads it. See [Model residency](#model-residency).

## Model residency

- A request loads the model it needs on first use (transcription, speech, Live Reader, Reader,
  realtime, Wyoming). That model stays loaded next to the ones already loaded.
- An explicit load (`POST /api/models/{id}/load`, `POST /v1/audio/models/load`, or the legacy STT
  `POST /api/ps/{model}`) first unloads the other loaded models of the same type, STT or TTS.
  A TTS load always leaves the pinned `TTS_MODEL` (Kokoro) loaded; only the pinned STT model may be unloaded to
  make GPU room (see [GPU memory guard](#gpu-memory-guard)).
  `POST /api/models/{id}/load?keep_pinned=true` leaves the pinned default STT model loaded (see below), as a
  request that loads a model on demand does. The web app loads STT models this way, from Transcribe and from
  Models › Load, so a Whisper model loads next to the pinned Parakeet; the Load row says which models it unloads
  and which stay loaded. Each `GET /api/models` row's `pinned` says whether it is a pinned default.
- A legacy `POST /api/ps/{model}` whose load fails loads back the STT models it unloaded first, the pinned default
  among them, so a mistyped or broken model doesn't leave the server without one.
- Download (`POST /api/models/{id}/download`) fetches files without loading them for faster-whisper, Parakeet,
  Kokoro, Supertonic, Piper, and the Qwen3 and Breeze workers. Pocket-TTS and the Chatterbox and Fish workers have
  no download-only path, so they load once and unload again.
- `STT_MODEL` and `TTS_MODEL` are pinned by default (`OS_PIN_DEFAULT_MODELS=true`): idle and LRU
  eviction never unload them, and they load on top of `OS_MAX_LOADED_MODELS`, which then caps the
  other models. An explicit unload, or an explicit load of another model of the same type, still
  replaces them. `TTS_PRELOAD_DEFAULT=true` loads `TTS_MODEL` in the background at startup, and
  `STT_PRELOAD_MODELS` loads its models in the background once the server has started, next to the ones loaded.
- A `/v1/realtime` session whose transcription `language` its STT model can't transcribe (any but English for
  Parakeet) gets one `error` event with code `language_unsupported` after its `session.updated`, and its turns are
  transcribed in English.
- Every 30 seconds, models idle longer than `OS_MODEL_TTL` are unloaded, then the least recently used
  models beyond `OS_MAX_LOADED_MODELS`. A model that is transcribing, synthesizing, streaming, or
  loading is skipped and retried on the next pass, so the cap can be exceeded while only busy models
  are left beyond it.

### GPU memory guard

Before a model loads onto the GPU, the server checks the card's free memory, whatever starts the load: Models › Load,
a request's first use, a preload or a worker. The model's size, `OS_GPU_HEADROOM_MB`, and the room the pinned models
need to grow while they run (Parakeet up to `STT_PARAKEET_GPU_MEM_LIMIT_MB`, Kokoro about 300 MiB) must all fit. When
they don't, idle models that aren't pinned are unloaded, least recently used first. When you load a speech model
yourself, the idle pinned STT model may be unloaded too; the next transcription loads it back. If there is still not
enough room, the load fails with `503 insufficient_gpu_memory`: the message says what to unload, and `detail.gpu`
gives the figures in MiB. Models on the CPU aren't checked, and neither is anything on a server without an NVIDIA GPU.

- `OS_GPU_MEMORY_GUARD` (`true`): turns the guard on or off.
- `OS_GPU_HEADROOM_MB` (`600`): MiB kept free on top of a model's size.
- `OS_GPU_MAX_BIG_MODELS` (`auto`): how many models of 2 GiB or more (GPU worker models, large Whisper) can be loaded at
  once besides the pinned defaults. Loading one more first unloads the least recently used idle one. With `auto`, an
  8 GB card takes 1, a 12 to 16 GB card 2, and a 24 GB card has no cap.
