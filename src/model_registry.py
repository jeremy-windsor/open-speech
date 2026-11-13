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
KNOWN_MODELS: list[dict] = [
    # STT — faster-whisper
    {"id": "Systran/faster-whisper-tiny", "type": "stt", "provider": "faster-whisper", "source": "Systran", "model_format": "CT2", "size_mb": 75, "description": "Fastest, lowest quality"},
    {"id": "Systran/faster-whisper-base", "type": "stt", "provider": "faster-whisper", "source": "Systran", "model_format": "CT2", "size_mb": 150, "description": "Good balance for CPU"},
    {"id": "Systran/faster-whisper-small", "type": "stt", "provider": "faster-whisper", "source": "Systran", "model_format": "CT2", "size_mb": 500, "description": "Better accuracy"},
    {"id": "Systran/faster-whisper-medium", "type": "stt", "provider": "faster-whisper", "source": "Systran", "model_format": "CT2", "size_mb": 1500, "description": "High accuracy"},
    {"id": "Systran/faster-whisper-tiny.en", "type": "stt", "provider": "faster-whisper", "source": "Systran", "model_format": "CT2", "size_mb": 75, "description": "English-only tiny model"},
    {"id": "Systran/faster-whisper-base.en", "type": "stt", "provider": "faster-whisper", "source": "Systran", "model_format": "CT2", "size_mb": 150, "description": "English-only base model"},
    {"id": "Systran/faster-whisper-small.en", "type": "stt", "provider": "faster-whisper", "source": "Systran", "model_format": "CT2", "size_mb": 500, "description": "English-only small model"},
    {"id": "Systran/faster-whisper-medium.en", "type": "stt", "provider": "faster-whisper", "source": "Systran", "model_format": "CT2", "size_mb": 1500, "description": "English-only medium model"},
    {"id": "Systran/faster-whisper-large-v2", "type": "stt", "provider": "faster-whisper", "source": "Systran", "model_format": "CT2", "size_mb": 2900, "description": "Large-v2, high accuracy"},
    {"id": "Systran/faster-whisper-large-v3", "type": "stt", "provider": "faster-whisper", "source": "Systran", "model_format": "CT2", "size_mb": 3000, "description": "Large-v3, high accuracy"},
    {"id": "deepdml/faster-whisper-large-v3-turbo-ct2", "type": "stt", "provider": "faster-whisper", "source": "deepdml", "model_format": "CT2", "size_mb": 1600, "description": "Large-v3-turbo, near large-v3 accuracy at 3-4x speed"},
    # distil-whisper (smaller/faster variants)
    {"id": "Systran/faster-distil-whisper-small.en", "type": "stt", "provider": "faster-whisper", "source": "Systran", "model_format": "CT2", "size_mb": 250, "description": "Distil-small English-only, fast CPU"},
    {"id": "Systran/faster-distil-whisper-medium.en", "type": "stt", "provider": "faster-whisper", "source": "Systran", "model_format": "CT2", "size_mb": 750, "description": "Distil-medium English-only, balanced"},
    {"id": "Systran/faster-distil-whisper-large-v3", "type": "stt", "provider": "faster-whisper", "source": "Systran", "model_format": "CT2", "size_mb": 1500, "description": "Distil-large-v3, near large-v3 quality at half size", "languages": ["en"]},
    # TTS — kokoro
    {"id": "kokoro", "type": "tts", "provider": "kokoro", "size_mb": 330, "description": "Fast, 52 voices, voice blending"},
    # TTS — pocket-tts
    {"id": "pocket-tts", "type": "tts", "provider": "pocket-tts", "size_mb": 220, "description": "CPU-first low-latency TTS with streaming and multiple voices"},
    # TTS — isolated Qwen3 canary (hidden unless its worker is configured)
    {"id": "qwen3/0.6b-custom-voice", "type": "tts", "provider": "qwen3", "source": "Qwen", "model_format": "external", "size_mb": 1800, "description": "Experimental Qwen3 0.6B preset voice model", "optional_provider": True},
    {"id": "qwen3/0.6b-base", "type": "tts", "provider": "qwen3", "source": "Qwen", "model_format": "external", "size_mb": 1800, "description": "Experimental Qwen3 0.6B voice cloning model", "optional_provider": True},
    # TTS — isolated voice-cloning workers (hidden unless configured)
    {"id": "chatterbox/regular", "type": "tts", "provider": "chatterbox", "source": "Resemble AI", "model_format": "external", "size_mb": 8800, "description": "Chatterbox 500M English voice cloning model", "optional_provider": True},
    {"id": "chatterbox/turbo", "type": "tts", "provider": "chatterbox", "source": "Resemble AI", "model_format": "external", "size_mb": 5500, "description": "Chatterbox Turbo 350M English voice cloning model", "optional_provider": True},
    {"id": "cosyvoice/2-0.5b", "type": "tts", "provider": "cosyvoice", "source": "FunAudioLLM", "model_format": "external", "size_mb": 4700, "description": "CosyVoice 2 0.5B multilingual voice cloning model", "optional_provider": True},
    {"id": "cosyvoice/3-0.5b", "type": "tts", "provider": "cosyvoice", "source": "FunAudioLLM", "model_format": "external", "size_mb": 9600, "description": "CosyVoice 3 0.5B multilingual voice cloning model", "optional_provider": True},
    # size_mb: model.pth 3.12 GB + codec.pth 1.87 GB + the tokenizer files of
    # scarxity/fish-speech-s2-pro-nf4 (providers/fish/app.py, MODEL_FILES), about 5.0 GB, in MiB.
    {"id": "fish/s2-pro", "type": "tts", "provider": "fish", "source": "Fish Audio", "model_format": "external", "size_mb": 4770, "description": "Fish Audio S2 Pro (NF4) multilingual voice cloning model", "optional_provider": True},
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


def stt_model_languages(model_id: str) -> list[str]:
    """Return the languages an STT model transcribes, as Whisper's language codes.

    Every STT backend is faster-whisper: an English-only checkpoint ("*.en") knows
    "en", any other knows Whisper's full list unless its registry entry says otherwise.
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
    """Whether a Whisper model translates to English with ``task="translate"``.

    Large-v3-turbo was not trained to translate and answers in the speaker's language
    (OpenAI's Whisper README); an English-only checkpoint ("*.en", distil-large-v3) has no
    task token, so faster-whisper only transcribes; distil-whisper models are English-only.
    """
    name = model_id.lower()
    if "turbo" in name or "distil" in name:
        return False
    return stt_model_languages(model_id) != ["en"]
