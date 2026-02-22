"""Curated registry of known models with metadata."""

from __future__ import annotations

KNOWN_MODELS: list[dict] = [
    # STT — faster-whisper
    {"id": "Systran/faster-whisper-tiny", "type": "stt", "provider": "faster-whisper", "size_mb": 75, "description": "Fastest, lowest quality"},
    {"id": "Systran/faster-whisper-base", "type": "stt", "provider": "faster-whisper", "size_mb": 150, "description": "Good balance for CPU"},
    {"id": "Systran/faster-whisper-small", "type": "stt", "provider": "faster-whisper", "size_mb": 500, "description": "Better accuracy"},
    {"id": "Systran/faster-whisper-medium", "type": "stt", "provider": "faster-whisper", "size_mb": 1500, "description": "High accuracy"},
    {"id": "Systran/faster-whisper-tiny.en", "type": "stt", "provider": "faster-whisper", "size_mb": 75, "description": "English-only tiny model"},
    {"id": "Systran/faster-whisper-base.en", "type": "stt", "provider": "faster-whisper", "size_mb": 150, "description": "English-only base model"},
    {"id": "Systran/faster-whisper-small.en", "type": "stt", "provider": "faster-whisper", "size_mb": 500, "description": "English-only small model"},
    {"id": "Systran/faster-whisper-medium.en", "type": "stt", "provider": "faster-whisper", "size_mb": 1500, "description": "English-only medium model"},
    {"id": "Systran/faster-whisper-large-v2", "type": "stt", "provider": "faster-whisper", "size_mb": 2900, "description": "Large-v2, high accuracy"},
    {"id": "Systran/faster-whisper-large-v3", "type": "stt", "provider": "faster-whisper", "size_mb": 3000, "description": "Large-v3, high accuracy"},
    {"id": "deepdml/faster-whisper-large-v3-turbo-ct2", "type": "stt", "provider": "faster-whisper", "size_mb": 1500, "description": "Best quality, GPU recommended"},
    # TTS — kokoro
    {"id": "kokoro", "type": "tts", "provider": "kokoro", "size_mb": 330, "description": "Fast, 52 voices, voice blending"},
    # TTS — piper
    {"id": "piper/en_US-lessac-medium", "type": "tts", "provider": "piper", "size_mb": 35, "description": "Lightweight, fast, good quality"},
    # TTS — pocket-tts
    {"id": "pocket-tts", "type": "tts", "provider": "pocket-tts", "size_mb": 220, "description": "CPU-first low-latency TTS with streaming and multiple voices"},
    {"id": "piper/en_US-lessac-high", "type": "tts", "provider": "piper", "size_mb": 75, "description": "Higher quality, still fast"},
    {"id": "piper/en_US-amy-medium", "type": "tts", "provider": "piper", "size_mb": 35, "description": "Female voice, natural"},
    {"id": "piper/en_US-ryan-medium", "type": "tts", "provider": "piper", "size_mb": 35, "description": "Male voice"},
    {"id": "piper/en_GB-alan-medium", "type": "tts", "provider": "piper", "size_mb": 35, "description": "British male"},
    {"id": "piper/en_GB-cori-medium", "type": "tts", "provider": "piper", "size_mb": 35, "description": "British female"},
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
