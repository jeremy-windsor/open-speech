"""Curated registry of known models with metadata."""

from __future__ import annotations

# faster_whisper.tokenizer._LANGUAGE_CODES, in its order: what a multilingual Whisper
# checkpoint transcribes. Copied so that listing languages does not import faster_whisper.
WHISPER_LANGUAGES: tuple[str, ...] = tuple("""
    af am ar as az ba be bg bn bo br bs ca cs cy da de el en es et eu fa fi fo fr gl gu
    ha haw he hi hr ht hu hy id is it ja jw ka kk km kn ko la lb ln lo lt lv mg mi mk ml
    mn mr ms mt my ne nl nn no oc pa pl ps pt ro ru sa sd si sk sl sn so sq sr su sv sw
    ta te tg th tk tl tr tt uk ur uz vi yi yo zh yue
""".split())

# An entry's optional "languages" lists what the model transcribes or speaks when that is
# narrower than its family's (distil-large-v3 is trained on English only).
#
# "gpu_mb": MiB a model holds once loaded on a GPU (CUDA libraries included); none means it runs on the CPU. Measured on
# an RTX 2070 SUPER for Parakeet, Kokoro, Whisper large-v3-turbo and Qwen3 (float32); the rest are estimates. A
# worker's own figure (its manifest's gpu_memory_mb) overrides it. "gpu_growth_mb": how much more a model may take
# while it runs (Kokoro: an estimate; Parakeet's comes from STT_PARAKEET_GPU_MEM_LIMIT_MB unless that is 0).
KNOWN_MODELS: list[dict] = [
    # STT — faster-whisper
    {"id": "Systran/faster-whisper-tiny", "type": "stt", "provider": "faster-whisper", "source": "Systran", "model_format": "CT2", "size_mb": 75, "gpu_mb": 400, "description": "Fastest, lowest quality"},
    {"id": "Systran/faster-whisper-base", "type": "stt", "provider": "faster-whisper", "source": "Systran", "model_format": "CT2", "size_mb": 150, "gpu_mb": 450, "description": "Good balance for CPU"},
    {"id": "Systran/faster-whisper-small", "type": "stt", "provider": "faster-whisper", "source": "Systran", "model_format": "CT2", "size_mb": 500, "gpu_mb": 900, "description": "Better accuracy"},
    {"id": "Systran/faster-whisper-medium", "type": "stt", "provider": "faster-whisper", "source": "Systran", "model_format": "CT2", "size_mb": 1500, "gpu_mb": 2000, "description": "High accuracy"},
    {"id": "Systran/faster-whisper-tiny.en", "type": "stt", "provider": "faster-whisper", "source": "Systran", "model_format": "CT2", "size_mb": 75, "gpu_mb": 400, "description": "English-only tiny model"},
    {"id": "Systran/faster-whisper-base.en", "type": "stt", "provider": "faster-whisper", "source": "Systran", "model_format": "CT2", "size_mb": 150, "gpu_mb": 450, "description": "English-only base model"},
    {"id": "Systran/faster-whisper-small.en", "type": "stt", "provider": "faster-whisper", "source": "Systran", "model_format": "CT2", "size_mb": 500, "gpu_mb": 900, "description": "English-only small model"},
    {"id": "Systran/faster-whisper-medium.en", "type": "stt", "provider": "faster-whisper", "source": "Systran", "model_format": "CT2", "size_mb": 1500, "gpu_mb": 2000, "description": "English-only medium model"},
    {"id": "Systran/faster-whisper-large-v2", "type": "stt", "provider": "faster-whisper", "source": "Systran", "model_format": "CT2", "size_mb": 2900, "gpu_mb": 3500, "description": "Large-v2, high accuracy"},
    {"id": "Systran/faster-whisper-large-v3", "type": "stt", "provider": "faster-whisper", "source": "Systran", "model_format": "CT2", "size_mb": 3000, "gpu_mb": 3500, "description": "Large-v3, high accuracy"},
    {"id": "deepdml/faster-whisper-large-v3-turbo-ct2", "type": "stt", "provider": "faster-whisper", "source": "deepdml", "model_format": "CT2", "size_mb": 1600, "gpu_mb": 2100, "description": "Large-v3-turbo, near large-v3 accuracy at 3-4x speed"},
    # distil-whisper (smaller/faster variants)
    {"id": "Systran/faster-distil-whisper-small.en", "type": "stt", "provider": "faster-whisper", "source": "Systran", "model_format": "CT2", "size_mb": 250, "gpu_mb": 700, "description": "Distil-small English-only, fast CPU"},
    {"id": "Systran/faster-distil-whisper-medium.en", "type": "stt", "provider": "faster-whisper", "source": "Systran", "model_format": "CT2", "size_mb": 750, "gpu_mb": 1300, "description": "Distil-medium English-only, balanced"},
    {"id": "Systran/faster-distil-whisper-large-v3", "type": "stt", "provider": "faster-whisper", "source": "Systran", "model_format": "CT2", "size_mb": 1500, "gpu_mb": 2000, "description": "Distil-large-v3, near large-v3 quality at half size", "languages": ["en"]},
    # STT — NVIDIA Parakeet (onnx-asr). What it takes on disk depends on the device it runs on (size_mb_by_device,
    # "cuda" or "cpu"): on a GPU the 2.5 GB fp32 export it downloads plus the 1.2 GB fp16 copy its first load
    # makes, about 3.7 GB; on a CPU the 0.65 GB int8 files, about 0.7 GB. size_mb is the GPU figure.
    {"id": "nvidia/parakeet-tdt-0.6b-v2", "type": "stt", "provider": "parakeet", "source": "NVIDIA", "model_format": "ONNX", "size_mb": 3700, "gpu_mb": 1500, "gpu_growth_mb": 800, "size_mb_by_device": {"cuda": 3700, "cpu": 700}, "description": "Parakeet TDT 0.6B v2: English only, fastest, Whisper turbo accuracy (fp16 on GPU, int8 on CPU)", "languages": ["en"]},
    # TTS — kokoro
    {"id": "kokoro", "type": "tts", "provider": "kokoro", "size_mb": 330, "gpu_mb": 1230, "gpu_growth_mb": 300, "description": "Fast, 52 voices, voice blending"},
    # TTS — supertonic. size_mb: its four ONNX files and ten voices; about 0.6 GB of RAM on a CPU. gpu_mb and
    # gpu_growth_mb are estimates.
    {"id": "supertonic", "type": "tts", "provider": "supertonic", "source": "Supertone", "model_format": "ONNX", "size_mb": 383, "gpu_mb": 1000, "gpu_growth_mb": 300, "description": "Fast ONNX voices on a GPU or CPU, 10 voices, voice blending"},
    # TTS — pocket-tts
    {"id": "pocket-tts", "type": "tts", "provider": "pocket-tts", "size_mb": 220, "description": "CPU-first low-latency TTS with streaming and multiple voices"},
    # TTS — isolated Qwen3 canary (hidden unless its worker is configured)
    {"id": "qwen3/0.6b-custom-voice", "type": "tts", "provider": "qwen3", "source": "Qwen", "model_format": "external", "size_mb": 1800, "gpu_mb": 3500, "description": "Experimental Qwen3 0.6B preset voice model", "optional_provider": True},
    {"id": "qwen3/0.6b-base", "type": "tts", "provider": "qwen3", "source": "Qwen", "model_format": "external", "size_mb": 1800, "gpu_mb": 3500, "description": "Experimental Qwen3 0.6B voice cloning model", "optional_provider": True},
    # TTS — isolated voice-cloning workers (hidden unless configured)
    {"id": "chatterbox/regular", "type": "tts", "provider": "chatterbox", "source": "Resemble AI", "model_format": "external", "size_mb": 8800, "gpu_mb": 3000, "description": "Chatterbox 500M English voice cloning model", "optional_provider": True},
    {"id": "chatterbox/turbo", "type": "tts", "provider": "chatterbox", "source": "Resemble AI", "model_format": "external", "size_mb": 5500, "gpu_mb": 2500, "description": "Chatterbox Turbo 350M English voice cloning model", "optional_provider": True},
    # size_mb: model.pth 3.12 GB + codec.pth 1.87 GB + the tokenizer files of
    # scarxity/fish-speech-s2-pro-nf4 (providers/fish/app.py, MODEL_FILES), about 5.0 GB, in MiB.
    {"id": "fish/s2-pro", "type": "tts", "provider": "fish", "source": "Fish Audio", "model_format": "external", "size_mb": 4770, "gpu_mb": 3300, "description": "Fish Audio S2 Pro (NF4) multilingual voice cloning model", "optional_provider": True},
    # TTS — piper (US). size_mb is the voice's .onnx in MiB: a "low" voice is a 16 kHz model as large as a
    # "medium" one.
    {"id": "piper/en_US-lessac-low", "type": "tts", "provider": "piper", "size_mb": 60, "description": "US English - Lessac, low quality"},
    {"id": "piper/en_US-lessac-medium", "type": "tts", "provider": "piper", "size_mb": 60, "description": "US English - Lessac voice"},
    {"id": "piper/en_US-lessac-high", "type": "tts", "provider": "piper", "size_mb": 109, "description": "US English - Lessac, high quality"},
    {"id": "piper/en_US-amy-medium", "type": "tts", "provider": "piper", "size_mb": 60, "description": "US English - Amy voice"},
    {"id": "piper/en_US-arctic-medium", "type": "tts", "provider": "piper", "size_mb": 73, "description": "US English - Arctic voice"},
    {"id": "piper/en_US-bryce-medium", "type": "tts", "provider": "piper", "size_mb": 61, "description": "US English - Bryce voice"},
    {"id": "piper/en_US-danny-low", "type": "tts", "provider": "piper", "size_mb": 60, "description": "US English - Danny, low quality"},
    {"id": "piper/en_US-hfc_female-medium", "type": "tts", "provider": "piper", "size_mb": 60, "description": "US English - HFC female voice"},
    {"id": "piper/en_US-hfc_male-medium", "type": "tts", "provider": "piper", "size_mb": 60, "description": "US English - HFC male voice"},
    {"id": "piper/en_US-joe-medium", "type": "tts", "provider": "piper", "size_mb": 60, "description": "US English - Joe voice"},
    {"id": "piper/en_US-john-medium", "type": "tts", "provider": "piper", "size_mb": 61, "description": "US English - John voice"},
    {"id": "piper/en_US-kathleen-low", "type": "tts", "provider": "piper", "size_mb": 60, "description": "US English - Kathleen, low quality"},
    {"id": "piper/en_US-kusal-medium", "type": "tts", "provider": "piper", "size_mb": 60, "description": "US English - Kusal voice"},
    {"id": "piper/en_US-libritts_r-medium", "type": "tts", "provider": "piper", "size_mb": 75, "description": "US English - LibriTTS-R voice"},
    {"id": "piper/en_US-ljspeech-medium", "type": "tts", "provider": "piper", "size_mb": 61, "description": "US English - LJSpeech voice"},
    {"id": "piper/en_US-ljspeech-high", "type": "tts", "provider": "piper", "size_mb": 109, "description": "US English - LJSpeech, high quality"},
    {"id": "piper/en_US-norman-medium", "type": "tts", "provider": "piper", "size_mb": 61, "description": "US English - Norman voice"},
    {"id": "piper/en_US-ryan-low", "type": "tts", "provider": "piper", "size_mb": 60, "description": "US English - Ryan, low quality"},
    {"id": "piper/en_US-ryan-medium", "type": "tts", "provider": "piper", "size_mb": 60, "description": "US English - Ryan voice"},
    {"id": "piper/en_US-ryan-high", "type": "tts", "provider": "piper", "size_mb": 115, "description": "US English - Ryan, high quality"},
    # TTS — piper (GB)
    {"id": "piper/en_GB-alan-low", "type": "tts", "provider": "piper", "size_mb": 60, "description": "British English - Alan, low quality"},
    {"id": "piper/en_GB-alan-medium", "type": "tts", "provider": "piper", "size_mb": 60, "description": "British English - Alan voice"},
    {"id": "piper/en_GB-cori-medium", "type": "tts", "provider": "piper", "size_mb": 61, "description": "British English - Cori voice"},
    {"id": "piper/en_GB-cori-high", "type": "tts", "provider": "piper", "size_mb": 109, "description": "British English - Cori, high quality"},
    {"id": "piper/en_GB-jenny_dioco-medium", "type": "tts", "provider": "piper", "size_mb": 60, "description": "British English - Jenny Dioco voice"},
    {"id": "piper/en_GB-northern_english_male-medium", "type": "tts", "provider": "piper", "size_mb": 60, "description": "British English - Northern English male voice"},
    {"id": "piper/en_GB-semaine-medium", "type": "tts", "provider": "piper", "size_mb": 73, "description": "British English - Semaine voice"},
    {"id": "piper/en_GB-southern_english_female-low", "type": "tts", "provider": "piper", "size_mb": 60, "description": "British English - Southern English female, low quality"},
]


def get_known_models() -> list[dict]:
    """Return a copy of the known models list."""
    return [m.copy() for m in KNOWN_MODELS]


def get_known_model(model_id: str) -> dict | None:
    """Look up a known model by ID."""
    for m in KNOWN_MODELS:
        if m["id"] == model_id:
            return m.copy()
    return None


# Provider display names, as the web app shows them (src/static/js/core/models.js, PROVIDER_DISPLAY).
PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "kokoro": "Kokoro", "piper": "Piper", "pocket-tts": "Pocket TTS", "supertonic": "Supertonic", "qwen3": "Qwen3 TTS",
    "chatterbox": "Chatterbox", "fish": "Fish Audio S2", "f5-tts": "F5 TTS",
    "xtts": "XTTS v2", "faster-whisper": "faster-whisper", "parakeet": "Parakeet",
}

_PIPER_QUALITY = {"x_low": "extra low", "low": "low", "medium": "medium", "high": "high"}


def model_display_name(model_id: str, provider: str | None = None) -> str:
    """A model's readable name: the provider's display name and the model's variant.

    "kokoro" -> "Kokoro", "chatterbox/turbo" -> "Chatterbox turbo", "piper/en_US-amy-medium" -> "Piper Amy (US,
    medium)". A model of an unknown provider is named by its id.
    """
    model_id = str(model_id or "")
    provider = provider or (model_id.split("/", 1)[0] if "/" in model_id else model_id)
    display = PROVIDER_DISPLAY_NAMES.get(provider)
    if not display:
        return model_id
    variant = model_id.split("/", 1)[1] if "/" in model_id else ""
    if provider == "piper" and variant:
        parts = variant.split("-")
        if len(parts) >= 3 and "_" in parts[0] and parts[-1] in _PIPER_QUALITY:
            speaker = " ".join(word.capitalize() for word in "-".join(parts[1:-1]).split("_"))
            region = parts[0].split("_", 1)[1]
            return f"{display} {speaker} ({region}, {_PIPER_QUALITY[parts[-1]]})"
    return f"{display} {variant}" if variant else display


def known_model_size_mb(entry: dict, device: str | None) -> int | None:
    """A registry entry's size on disk for the device it runs on ("cuda..." or "cpu"; None: unknown).

    An entry whose files differ by device (Parakeet: fp16 on a GPU, int8 on a CPU) has size_mb_by_device;
    any other has one size_mb.
    """
    by_device = entry.get("size_mb_by_device")
    if by_device and device:
        size = by_device.get("cuda" if str(device).lower().startswith("cuda") else "cpu")
        if size:
            return size
    return entry.get("size_mb")


def stt_model_languages(model_id: str) -> list[str]:
    """Return the languages an STT model transcribes, as Whisper's language codes.

    A registry entry's "languages" (Parakeet's ["en"]) decides; otherwise the model is a
    Whisper checkpoint: an English-only one ("*.en") knows "en", any other Whisper's full list.
    """
    known = get_known_model(model_id)
    if known and known.get("languages"):
        return list(known["languages"])
    if model_id.lower().endswith(".en"):
        return ["en"]
    return list(WHISPER_LANGUAGES)


# Models /v1/audio/translations names when a model can't translate.
TRANSLATING_STT_MODELS: tuple[str, ...] = ("Systran/faster-whisper-medium", "Systran/faster-whisper-large-v3")


def stt_model_can_translate(model_id: str) -> bool:
    """Whether a model translates to English with ``task="translate"``.

    Parakeet transcribes English only. Large-v3-turbo was not trained to translate and answers in the speaker's language
    (OpenAI's Whisper README); an English-only checkpoint ("*.en", distil-large-v3) has no
    task token, so faster-whisper only transcribes; distil-whisper models are English-only.
    """
    name = model_id.lower()
    if "turbo" in name or "distil" in name:
        return False
    return stt_model_languages(model_id) != ["en"]
