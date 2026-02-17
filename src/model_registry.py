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
    # TTS — qwen3
    {"id": "qwen3-tts/1.7B-CustomVoice", "type": "tts", "provider": "qwen3", "size_mb": 3500, "description": "Premium 9-speaker model with instruction control"},
    {"id": "qwen3-tts/1.7B-VoiceDesign", "type": "tts", "provider": "qwen3", "size_mb": 3500, "description": "Create voices from text descriptions (instruction-driven)"},
    {"id": "qwen3-tts/1.7B-Base", "type": "tts", "provider": "qwen3", "size_mb": 3500, "description": "Voice cloning model for reference-audio synthesis"},
    {"id": "qwen3-tts/0.6B-CustomVoice", "type": "tts", "provider": "qwen3", "size_mb": 1200, "description": "Lighter premium-speaker model"},
    {"id": "qwen3-tts/0.6B-Base", "type": "tts", "provider": "qwen3", "size_mb": 1200, "description": "Lighter voice cloning model"},
    {"id": "qwen3-tts/Tokenizer-12Hz", "type": "tts", "provider": "qwen3", "size_mb": 200, "description": "Shared Qwen3 audio tokenizer"},
    # TTS — fish-speech
    {"id": "fish-speech-1.5", "type": "tts", "provider": "fish-speech", "size_mb": 500, "description": "Top TTS quality, zero-shot voice cloning"},
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
