"""Curated registry of known models with metadata."""

from __future__ import annotations

KNOWN_MODELS: list[dict] = [
    # STT — faster-whisper
    {"id": "Systran/faster-whisper-tiny", "type": "stt", "provider": "faster-whisper", "size_mb": 75, "description": "Fastest, lowest quality"},
    {"id": "Systran/faster-whisper-base", "type": "stt", "provider": "faster-whisper", "size_mb": 150, "description": "Good balance for CPU"},
    {"id": "Systran/faster-whisper-small", "type": "stt", "provider": "faster-whisper", "size_mb": 500, "description": "Better accuracy"},
    {"id": "Systran/faster-whisper-medium", "type": "stt", "provider": "faster-whisper", "size_mb": 1500, "description": "High accuracy"},
    {"id": "deepdml/faster-whisper-large-v3-turbo-ct2", "type": "stt", "provider": "faster-whisper", "size_mb": 1500, "description": "Best quality, GPU recommended"},
    # STT — moonshine
    {"id": "moonshine/tiny", "type": "stt", "provider": "moonshine", "size_mb": 35, "description": "Fast CPU, English only"},
    {"id": "moonshine/base", "type": "stt", "provider": "moonshine", "size_mb": 70, "description": "Better accuracy, English only"},
    # TTS — kokoro
    {"id": "kokoro", "type": "tts", "provider": "kokoro", "size_mb": 330, "description": "Fast, 52 voices, voice blending"},
    # TTS — piper
    {"id": "piper/en_US-lessac-medium", "type": "tts", "provider": "piper", "size_mb": 35, "description": "Lightweight, fast, good quality"},
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
