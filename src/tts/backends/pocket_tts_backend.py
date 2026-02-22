"""Pocket TTS backend — CPU-friendly low-latency TTS by Kyutai."""

from __future__ import annotations

import logging
import time
from typing import Any, Iterator

import numpy as np

from src.tts.backends.base import TTSLoadedModelInfo, VoiceInfo

logger = logging.getLogger(__name__)

POCKET_TTS_DEFAULT_MODEL_ID = "pocket-tts"
POCKET_TTS_SPEAKERS: list[dict[str, str]] = [
    {"name": "alba", "description": "Default female voice", "language": "en"},
    {"name": "marius", "description": "Male voice", "language": "en"},
    {"name": "javert", "description": "Male voice", "language": "en"},
    {"name": "jean", "description": "Male voice", "language": "en"},
    {"name": "fantine", "description": "Female voice", "language": "en"},
    {"name": "cosette", "description": "Female voice", "language": "en"},
    {"name": "eponine", "description": "Female voice", "language": "en"},
    {"name": "azelma", "description": "Female voice", "language": "en"},
]


class PocketTTSBackend:
    """TTS backend using pocket-tts with optional streaming chunks."""

    name = "pocket-tts"
    sample_rate = 24000
    capabilities: dict[str, Any] = {
        "voice_blend": False,
        "voice_design": False,
        "voice_clone": False,
        "streaming": True,
        "instructions": False,
        "speakers": POCKET_TTS_SPEAKERS,
        "languages": ["en"],
        "speed_control": False,
        "ssml": False,
        "batch": False,
    }

    @classmethod
    def is_available(cls) -> bool:
        try:
            import pocket_tts  # noqa: F401
            return True
        except ImportError:
            return False

    def __init__(self, device: str = "auto") -> None:
        self._device = device
        self._models: dict[str, dict[str, Any]] = {}

    def _ensure_loaded(self, model_id: str = POCKET_TTS_DEFAULT_MODEL_ID) -> dict[str, Any]:
        if model_id not in self._models:
            self.load_model(model_id)
        return self._models[model_id]

    def _resolve_model_id(self, model_id: str) -> str:
        return model_id or POCKET_TTS_DEFAULT_MODEL_ID

    def _resolve_voice(self, voice: str) -> str:
        normalized = (voice or "").strip().lower()
        available = {s["name"] for s in POCKET_TTS_SPEAKERS}
        if normalized in available:
            return normalized
        return "alba"

    def load_model(self, model_id: str) -> None:
        model_id = self._resolve_model_id(model_id)
        if model_id in self._models:
            return

        try:
            from pocket_tts import TTSModel
        except ImportError as e:
            raise RuntimeError(
                "pocket-tts package is not installed. "
                "Rebuild the image with BAKED_PROVIDERS=kokoro,pocket-tts. "
                "Example: docker build --build-arg BAKED_PROVIDERS=kokoro,pocket-tts ."
            ) from e

        logger.info("Loading Pocket TTS model %s...", model_id)
        start = time.time()

        try:
            model = TTSModel.load_model()
        except Exception as e:
            raise RuntimeError(f"Failed to load Pocket TTS model {model_id}: {e}") from e

        self.sample_rate = int(getattr(model, "sample_rate", self.sample_rate))
        self._models[model_id] = {
            "model": model,
            "voice_states": {},
            "device": getattr(model, "device", "cpu"),
            "loaded_at": time.time(),
            "last_used_at": None,
        }
        logger.info("Pocket TTS model loaded in %.1fs", time.time() - start)

    def unload_model(self, model_id: str) -> None:
        model_id = self._resolve_model_id(model_id)
        if model_id in self._models:
            del self._models[model_id]
            logger.info("Pocket TTS model %s unloaded", model_id)

    def is_model_loaded(self, model_id: str) -> bool:
        model_id = self._resolve_model_id(model_id)
        return model_id in self._models

    def loaded_models(self) -> list[TTSLoadedModelInfo]:
        return [
            TTSLoadedModelInfo(
                model=mid,
                backend=self.name,
                device=str(info.get("device", "cpu")),
                loaded_at=info["loaded_at"],
                last_used_at=info.get("last_used_at"),
            )
            for mid, info in self._models.items()
        ]

    def _voice_state_for(self, model_info: dict[str, Any], voice: str) -> Any:
        voice_states = model_info["voice_states"]
        if voice in voice_states:
            return voice_states[voice]

        state = model_info["model"].get_state_for_audio_prompt(voice)
        voice_states[voice] = state
        return state

    def synthesize(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        lang_code: str | None = None,
    ) -> Iterator[np.ndarray]:
        del speed, lang_code  # Not currently supported by pocket-tts API.

        if not text or not text.strip():
            raise ValueError("Text must not be empty for Pocket TTS synthesis.")

        model_id = next(iter(self._models), POCKET_TTS_DEFAULT_MODEL_ID)
        model_info = self._ensure_loaded(model_id)
        model_info["last_used_at"] = time.time()

        resolved_voice = self._resolve_voice(voice)
        model = model_info["model"]

        try:
            voice_state = self._voice_state_for(model_info, resolved_voice)
            stream = model.generate_audio_stream(voice_state, text)
            for chunk in stream:
                if hasattr(chunk, "detach"):
                    chunk = chunk.detach().cpu().numpy()
                elif not isinstance(chunk, np.ndarray):
                    chunk = np.array(chunk)
                if chunk.dtype != np.float32:
                    chunk = chunk.astype(np.float32)
                if chunk.size > 0:
                    yield chunk
        except Exception as e:
            logger.exception("Pocket TTS synthesis failed")
            raise RuntimeError(f"Pocket TTS synthesis failed: {e}") from e

    def list_voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(
                id=v["name"],
                name=v["name"].title(),
                language="en",
                gender=("female" if v["name"] in {"alba", "fantine", "cosette", "eponine", "azelma"} else "male"),
            )
            for v in POCKET_TTS_SPEAKERS
        ]
