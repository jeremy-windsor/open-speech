"""Voice Activity Detection (VAD) module."""

from src.vad.silero import SileroVAD, get_vad_model

__all__ = ["SileroVAD", "get_vad_model"]
