"""Kokoro TTS backend — uses the kokoro Python package."""

from __future__ import annotations

import logging
import time
from typing import Iterator

import numpy as np

from src.tts.backends.base import TTSBackend, TTSLoadedModelInfo, VoiceInfo
from src.tts.voices import parse_voice_spec

logger = logging.getLogger(__name__)


class KokoroBackend:
    """TTS backend using the Kokoro-82M model via the kokoro package."""

    name: str = "kokoro"
    sample_rate: int = 24000

    def __init__(self, device: str = "auto") -> None:
        self._device = device
        self._pipeline = None  # Lazy-loaded
        self._loaded_at: float | None = None
        self._last_used: float | None = None
        self._model_id: str | None = None

    def _get_device(self) -> str:
        if self._device != "auto":
            return self._device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _ensure_loaded(self, model_id: str = "kokoro") -> None:
        """Lazy-load the pipeline on first use."""
        if self._pipeline is not None:
            return

        logger.info("Loading Kokoro model (device=%s)...", self._get_device())
        start = time.time()
        from kokoro import KPipeline
        self._pipeline = KPipeline(lang_code="a", device=self._get_device())
        self._model_id = model_id
        self._loaded_at = time.time()
        elapsed = time.time() - start
        logger.info("Kokoro model loaded in %.1fs", elapsed)

    def load_model(self, model_id: str) -> None:
        self._ensure_loaded(model_id)

    def unload_model(self, model_id: str) -> None:
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            self._model_id = None
            self._loaded_at = None
            self._last_used = None
            logger.info("Kokoro model unloaded")

    def is_model_loaded(self, model_id: str) -> bool:
        return self._pipeline is not None

    def loaded_models(self) -> list[TTSLoadedModelInfo]:
        if self._pipeline is None:
            return []
        return [TTSLoadedModelInfo(
            model=self._model_id or "kokoro",
            backend=self.name,
            device=self._get_device(),
            loaded_at=self._loaded_at or 0,
            last_used_at=self._last_used,
        )]

    def synthesize(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        lang_code: str | None = None,
    ) -> Iterator[np.ndarray]:
        """Generate audio chunks from text.
        
        Yields numpy float32 arrays at 24kHz.
        """
        self._ensure_loaded()
        self._last_used = time.time()

        spec = parse_voice_spec(voice)

        if spec.is_blend:
            voice_tensor = self._blend_voices(spec)
        else:
            voice_tensor = spec.primary_id

        pipeline_lang = lang_code or "a"
        # Re-create pipeline if lang changed (kokoro requires lang at init)
        pipeline = self._pipeline

        for _gs, _ps, audio in pipeline(text, voice=voice_tensor, speed=speed):
            if audio is not None and len(audio) > 0:
                yield audio

    def _blend_voices(self, spec):
        """Blend multiple voice tensors according to weights."""
        import torch

        weights = spec.normalized_weights()
        tensors = []
        for comp in spec.components:
            # Load voice tensor from kokoro's voice pack
            t = self._pipeline.load_voice(comp.voice_id)
            tensors.append(t)

        # Weighted average
        result = torch.zeros_like(tensors[0])
        for w, t in zip(weights, tensors):
            result += w * t
        return result

    def list_voices(self) -> list[VoiceInfo]:
        """List available Kokoro voices."""
        # Return known built-in voices
        voices = [
            VoiceInfo(id="af_heart", name="Heart", language="en-us", gender="female"),
            VoiceInfo(id="af_bella", name="Bella", language="en-us", gender="female"),
            VoiceInfo(id="af_sky", name="Sky", language="en-us", gender="female"),
            VoiceInfo(id="af_nova", name="Nova", language="en-us", gender="female"),
            VoiceInfo(id="af_nicole", name="Nicole", language="en-us", gender="female"),
            VoiceInfo(id="af_sarah", name="Sarah", language="en-us", gender="female"),
            VoiceInfo(id="am_adam", name="Adam", language="en-us", gender="male"),
            VoiceInfo(id="am_michael", name="Michael", language="en-us", gender="male"),
            VoiceInfo(id="bf_emma", name="Emma", language="en-gb", gender="female"),
            VoiceInfo(id="bf_isabella", name="Isabella", language="en-gb", gender="female"),
            VoiceInfo(id="bm_george", name="George", language="en-gb", gender="male"),
            VoiceInfo(id="bm_lewis", name="Lewis", language="en-gb", gender="male"),
        ]
        return voices
