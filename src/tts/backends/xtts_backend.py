"""XTTS v2 backend — multilingual zero-shot voice cloning via Coqui TTS.

XTTS v2 requires reference audio for voice cloning and supports 16 languages.
It does not expose native speed control in this backend integration; `speed`
is accepted for protocol compatibility but intentionally ignored.

Requires: pip install TTS>=0.22.0
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from src.tts.backends.base import DEFAULT_TTS_CAPABILITIES, TTSLoadedModelInfo, VoiceInfo

logger = logging.getLogger(__name__)

XTTS_MODELS: dict[str, dict[str, Any]] = {
    "xtts/v2": {
        "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
        "description": "XTTS v2 — high-quality multilingual voice cloning (16 languages)",
        "size_gb": 1.8,
        "min_vram_gb": 4,
        "sample_rate": 24000,
    },
}


class XTTSBackend:
    """TTS backend using Coqui XTTS v2 for multilingual voice cloning.

    Notes:
        - Reference audio is required for voice cloning (`speaker_wav`).
        - No native speed control is applied here; `speed` is ignored.
    """

    name = "xtts"
    sample_rate = 24000
    capabilities: dict[str, Any] = {
        **DEFAULT_TTS_CAPABILITIES,
        "voice_clone": True,
        "streaming": False,
        "languages": [
            "en", "fr", "de", "es", "it", "pt", "nl", "tr",
            "ru", "pl", "cs", "ar", "zh-cn", "ja", "hu", "ko",
        ],
        "speed_control": False,
    }

    def __init__(self, device: str = "auto") -> None:
        self._device = device
        self._models: dict[str, dict[str, Any]] = {}

    def _resolve_device(self) -> str:
        if self._device == "cpu":
            return "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return "xpu"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def load_model(self, model_id: str) -> None:
        if model_id in self._models:
            return

        if model_id not in XTTS_MODELS:
            raise ValueError(f"Unknown XTTS model: {model_id}. Available: {list(XTTS_MODELS)}")

        try:
            from TTS.api import TTS  # noqa: N812
        except ImportError:
            raise RuntimeError(
                "XTTS requires the 'TTS' package. Install with: pip install TTS>=0.22.0"
            )

        device = self._resolve_device()
        model_name = XTTS_MODELS[model_id]["model_name"]

        logger.info("Loading XTTS model %s on %s...", model_id, device)
        start = time.time()

        try:
            engine = TTS(model_name, gpu=(device != "cpu"))
        except Exception as e:
            raise RuntimeError(f"Failed to load XTTS model {model_id}: {e}")

        elapsed = time.time() - start
        logger.info("XTTS model %s loaded in %.1fs on %s", model_id, elapsed, device)

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
            logger.info("Unloaded XTTS model %s", model_id)

    def is_model_loaded(self, model_id: str) -> bool:
        return model_id in self._models

    def loaded_models(self) -> list[TTSLoadedModelInfo]:
        return [
            TTSLoadedModelInfo(
                model=mid,
                backend=self.name,
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
        reference_text: str | None = None,
    ) -> Iterator[np.ndarray]:
        """Synthesize speech with XTTS v2 using reference audio voice cloning."""
        del voice, reference_text

        if not self._models:
            raise RuntimeError("No XTTS model loaded. Load one first.")

        if not text or not text.strip():
            raise ValueError("Text must not be empty for XTTS synthesis.")

        if not reference_audio:
            raise RuntimeError("XTTS requires reference audio for voice cloning.")

        if speed != 1.0:
            logger.debug("XTTS speed control is not supported; ignoring speed=%s", speed)

        model_id = next(iter(self._models))
        info = self._models[model_id]
        info["last_used_at"] = time.time()
        engine = info["engine"]

        language = lang_code or "en"
        temp_path: str | None = None

        try:
            fd = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            fd.write(reference_audio)
            fd.flush()
            fd.close()
            temp_path = fd.name

            wav = engine.tts(
                text=text,
                speaker_wav=temp_path,
                language=language,
            )
        except Exception as e:
            logger.exception("XTTS synthesis failed")
            raise RuntimeError(f"XTTS synthesis failed: {e}")
        finally:
            if temp_path is not None:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception:
                    pass

        if not isinstance(wav, np.ndarray):
            wav = np.array(wav, dtype=np.float32)
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)

        if wav.size == 0:
            raise RuntimeError("XTTS returned empty audio.")

        peak = np.abs(wav).max()
        if peak > 1.0:
            wav = wav / peak

        yield wav

    def list_voices(self) -> list[VoiceInfo]:
        """XTTS has no built-in voice list; expose default clone target voice."""
        return [
            VoiceInfo(
                id="default",
                name="Default (reference cloning)",
                language="multilingual",
                gender="unknown",
            )
        ]
