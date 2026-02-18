"""Fish Speech backend — high-quality TTS with zero-shot voice cloning."""

from __future__ import annotations

import logging
import time
from typing import Iterator

import numpy as np

from src.tts.backends.base import TTSLoadedModelInfo, VoiceInfo

logger = logging.getLogger(__name__)

FISH_MODELS: dict[str, dict] = {
    "fish-speech-1.5": {
        "hf_id": "fishaudio/fish-speech-1.5",
        "size_gb": 0.5,
        "min_vram_gb": 4,
    },
}


class FishSpeechBackend:
    """TTS backend using Fish Speech for high-quality synthesis and voice cloning."""

    name = "fish-speech"
    sample_rate = 24000
    capabilities: dict = {
        "voice_blend": False,
        "voice_design": False,
        "voice_clone": True,
        "streaming": False,
        "instructions": False,
        "speakers": [],
        "languages": ["en"],
        "speed_control": True,
        "ssml": False,
        "batch": False,
    }

    def __init__(self, device: str = "auto") -> None:
        self._device = device
        self._models: dict[str, dict] = {}

    def _resolve_device(self) -> str:
        if self._device == "cpu":
            return "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def load_model(self, model_id: str) -> None:
        if model_id in self._models:
            return

        if model_id not in FISH_MODELS:
            raise ValueError(f"Unknown Fish Speech model: {model_id}. Available: {list(FISH_MODELS)}")

        try:
            import fish_speech  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "Fish Speech requires the 'fish-speech' package. "
                "Install manually — see https://github.com/fishaudio/fish-speech"
            )

        hf_id = FISH_MODELS[model_id]["hf_id"]
        device = self._resolve_device()

        logger.info("Loading Fish Speech model %s (%s) on %s", model_id, hf_id, device)
        start = time.time()

        try:
            from fish_speech.inference import TTSInference
            engine = TTSInference(model_path=hf_id, device=device)
        except Exception as e:
            raise RuntimeError(f"Failed to load Fish Speech model: {e}")

        elapsed = time.time() - start
        logger.info("Fish Speech model %s loaded in %.1fs on %s", model_id, elapsed, device)

        self._models[model_id] = {
            "engine": engine,
            "device": device,
            "loaded_at": time.time(),
            "last_used_at": None,
        }

    def unload_model(self, model_id: str) -> None:
        if model_id in self._models:
            del self._models[model_id]
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("Unloaded Fish Speech model %s", model_id)

    def is_model_loaded(self, model_id: str) -> bool:
        return model_id in self._models

    def loaded_models(self) -> list[TTSLoadedModelInfo]:
        return [
            TTSLoadedModelInfo(
                model=mid,
                backend="fish-speech",
                device=info["device"],
                loaded_at=info["loaded_at"],
                last_used_at=info["last_used_at"],
            )
            for mid, info in self._models.items()
        ]

    def synthesize(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        lang_code: str | None = None,
        *,
        reference_audio: bytes | None = None,
    ) -> Iterator[np.ndarray]:
        """Synthesize text to audio.

        Extended parameters:
            reference_audio: Raw audio bytes for zero-shot voice cloning.
        """
        if not self._models:
            raise RuntimeError("No Fish Speech model loaded. Load one first.")

        model_id = next(iter(self._models))
        info = self._models[model_id]
        info["last_used_at"] = time.time()

        engine = info["engine"]

        try:
            kwargs = {"text": text}
            if reference_audio:
                kwargs["reference_audio"] = reference_audio

            audio = engine.synthesize(**kwargs)

            # Ensure float32 numpy
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio, dtype=np.float32)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Normalize
            peak = np.abs(audio).max()
            if peak > 0:
                audio = audio / peak

            # Apply speed
            if speed != 1.0 and speed > 0:
                indices = np.arange(0, len(audio), speed)
                indices = indices[indices < len(audio)].astype(int)
                audio = audio[indices]

            yield audio

        except Exception as e:
            logger.exception("Fish Speech synthesis failed")
            raise RuntimeError(f"Fish Speech synthesis failed: {e}")

    def list_voices(self) -> list[VoiceInfo]:
        # Fish Speech uses zero-shot cloning, no built-in voice list
        return [
            VoiceInfo(id="default", name="Default", language="en-us", gender="unknown"),
        ]
