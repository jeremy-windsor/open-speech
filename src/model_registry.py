"""Curated registry of known models with metadata."""

from __future__ import annotations

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
    {"id": "Systran/faster-distil-whisper-large-v3", "type": "stt", "provider": "faster-whisper", "source": "Systran", "model_format": "CT2", "size_mb": 1500, "description": "Distil-large-v3, near large-v3 quality at half size"},
    # TTS — kokoro
    {"id": "kokoro", "type": "tts", "provider": "kokoro", "size_mb": 330, "description": "Fast, 52 voices, voice blending"},
    # TTS — pocket-tts
    {"id": "pocket-tts", "type": "tts", "provider": "pocket-tts", "size_mb": 220, "description": "CPU-first low-latency TTS with streaming and multiple voices"},
    # TTS — piper (US)
    {"id": "piper/en_US-lessac-low", "type": "tts", "provider": "piper", "size_mb": 6, "description": "US English - Lessac, low quality"},
    {"id": "piper/en_US-lessac-medium", "type": "tts", "provider": "piper", "size_mb": 35, "description": "US English - Lessac voice"},
    {"id": "piper/en_US-lessac-high", "type": "tts", "provider": "piper", "size_mb": 75, "description": "US English - Lessac, high quality"},
    {"id": "piper/en_US-amy-medium", "type": "tts", "provider": "piper", "size_mb": 35, "description": "US English - Amy voice"},
    {"id": "piper/en_US-amy-high", "type": "tts", "provider": "piper", "size_mb": 75, "description": "US English - Amy, high quality"},
    {"id": "piper/en_US-arctic-medium", "type": "tts", "provider": "piper", "size_mb": 35, "description": "US English - Arctic voice"},
    {"id": "piper/en_US-bryce-medium", "type": "tts", "provider": "piper", "size_mb": 35, "description": "US English - Bryce voice"},
    {"id": "piper/en_US-danny-low", "type": "tts", "provider": "piper", "size_mb": 6, "description": "US English - Danny, low quality"},
    {"id": "piper/en_US-hfc_female-medium", "type": "tts", "provider": "piper", "size_mb": 35, "description": "US English - HFC female voice"},
    {"id": "piper/en_US-hfc_male-medium", "type": "tts", "provider": "piper", "size_mb": 35, "description": "US English - HFC male voice"},
    {"id": "piper/en_US-joe-medium", "type": "tts", "provider": "piper", "size_mb": 35, "description": "US English - Joe voice"},
    {"id": "piper/en_US-john-medium", "type": "tts", "provider": "piper", "size_mb": 35, "description": "US English - John voice"},
    {"id": "piper/en_US-kathleen-low", "type": "tts", "provider": "piper", "size_mb": 6, "description": "US English - Kathleen, low quality"},
    {"id": "piper/en_US-kusal-medium", "type": "tts", "provider": "piper", "size_mb": 35, "description": "US English - Kusal voice"},
    {"id": "piper/en_US-libritts_r-medium", "type": "tts", "provider": "piper", "size_mb": 35, "description": "US English - LibriTTS-R voice"},
    {"id": "piper/en_US-ljspeech-medium", "type": "tts", "provider": "piper", "size_mb": 35, "description": "US English - LJSpeech voice"},
    {"id": "piper/en_US-ljspeech-high", "type": "tts", "provider": "piper", "size_mb": 75, "description": "US English - LJSpeech, high quality"},
    {"id": "piper/en_US-norman-medium", "type": "tts", "provider": "piper", "size_mb": 35, "description": "US English - Norman voice"},
    {"id": "piper/en_US-ryan-low", "type": "tts", "provider": "piper", "size_mb": 6, "description": "US English - Ryan, low quality"},
    {"id": "piper/en_US-ryan-medium", "type": "tts", "provider": "piper", "size_mb": 35, "description": "US English - Ryan voice"},
    {"id": "piper/en_US-ryan-high", "type": "tts", "provider": "piper", "size_mb": 75, "description": "US English - Ryan, high quality"},
    # TTS — piper (GB)
    {"id": "piper/en_GB-alan-low", "type": "tts", "provider": "piper", "size_mb": 6, "description": "British English - Alan, low quality"},
    {"id": "piper/en_GB-alan-medium", "type": "tts", "provider": "piper", "size_mb": 35, "description": "British English - Alan voice"},
    {"id": "piper/en_GB-cori-medium", "type": "tts", "provider": "piper", "size_mb": 35, "description": "British English - Cori voice"},
    {"id": "piper/en_GB-cori-high", "type": "tts", "provider": "piper", "size_mb": 75, "description": "British English - Cori, high quality"},
    {"id": "piper/en_GB-jenny_dioco-medium", "type": "tts", "provider": "piper", "size_mb": 35, "description": "British English - Jenny Dioco voice"},
    {"id": "piper/en_GB-northern_english_male-medium", "type": "tts", "provider": "piper", "size_mb": 35, "description": "British English - Northern English male voice"},
    {"id": "piper/en_GB-semaine-medium", "type": "tts", "provider": "piper", "size_mb": 35, "description": "British English - Semaine voice"},
    {"id": "piper/en_GB-southern_english_female-low", "type": "tts", "provider": "piper", "size_mb": 6, "description": "British English - Southern English female, low quality"},
    {"id": "piper/en_GB-southern_english_female-medium", "type": "tts", "provider": "piper", "size_mb": 35, "description": "British English - Southern English female voice"},
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
