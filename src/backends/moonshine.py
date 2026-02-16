"""Moonshine STT backend — fast, CPU-friendly, English-only."""

from __future__ import annotations

import logging
import tempfile
import time
from typing import Any

from src.models import LoadedModelInfo

logger = logging.getLogger(__name__)

MOONSHINE_MODELS = {"moonshine/tiny", "moonshine/base"}


class MoonshineBackend:
    """STT backend using moonshine-onnx (English-only, CPU-optimized)."""

    name = "moonshine"

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._loaded_at: dict[str, float] = {}
        self._last_used: dict[str, float] = {}

    def load_model(self, model_id: str) -> None:
        if model_id in self._models:
            return

        try:
            from moonshine_onnx import MoonshineOnnxModel
        except ImportError:
            raise ImportError(
                "moonshine-onnx is not installed. Install with: pip install 'open-speech[moonshine]'"
            )

        # model_id is like "moonshine/tiny" or "moonshine/base"
        model_name = model_id.split("/")[-1] if "/" in model_id else model_id
        logger.info("Loading Moonshine model: %s", model_name)
        model = MoonshineOnnxModel(model_name=model_name)
        self._models[model_id] = model
        self._loaded_at[model_id] = time.time()
        self._last_used[model_id] = time.time()
        logger.info("Moonshine model %s loaded", model_id)

    def unload_model(self, model_id: str) -> None:
        if model_id in self._models:
            del self._models[model_id]
            del self._loaded_at[model_id]
            self._last_used.pop(model_id, None)

    def loaded_models(self) -> list[LoadedModelInfo]:
        from src.config import settings
        now = time.time()
        return [
            LoadedModelInfo(
                model=mid,
                backend=self.name,
                device="cpu",
                compute_type="float32",
                loaded_at=self._loaded_at[mid],
                last_used_at=self._last_used.get(mid),
                is_default=(mid == settings.stt_default_model),
                ttl_remaining=None,
            )
            for mid in self._models
        ]

    def is_model_loaded(self, model_id: str) -> bool:
        return model_id in self._models

    def _ensure_model(self, model_id: str) -> Any:
        if model_id not in self._models:
            self.load_model(model_id)
        self._last_used[model_id] = time.time()
        return self._models[model_id]

    def transcribe(
        self,
        audio: bytes,
        model: str,
        language: str | None = None,
        response_format: str = "json",
        temperature: float = 0.0,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        if language and language not in ("en", "english", None):
            raise ValueError(
                f"Moonshine only supports English. Requested language: {language}"
            )

        import numpy as np

        moonshine_model = self._ensure_model(model)

        # Decode audio bytes to numpy array
        audio_array = self._decode_audio(audio)

        # Moonshine returns list of strings (one per segment)
        tokens = moonshine_model.generate(audio_array)
        full_text = " ".join(tokens).strip() if tokens else ""

        if response_format == "verbose_json":
            duration = len(audio_array) / 16000.0
            return {
                "task": "transcribe",
                "language": "en",
                "duration": duration,
                "text": full_text,
                "segments": [
                    {
                        "id": 0,
                        "seek": 0,
                        "start": 0.0,
                        "end": duration,
                        "text": full_text,
                        "tokens": [],
                        "temperature": temperature,
                        "avg_logprob": 0.0,
                        "compression_ratio": 0.0,
                        "no_speech_prob": 0.0,
                    }
                ],
            }
        elif response_format == "text":
            return {"text": full_text, "raw_text": True}
        elif response_format == "srt":
            duration = len(audio_array) / 16000.0
            srt = f"1\n00:00:00,000 --> {_format_ts_srt(duration)}\n{full_text}\n"
            return {"text": srt, "raw_text": True}
        elif response_format == "vtt":
            duration = len(audio_array) / 16000.0
            vtt = f"WEBVTT\n\n00:00:00.000 --> {_format_ts_vtt(duration)}\n{full_text}\n"
            return {"text": vtt, "raw_text": True}
        else:
            return {"text": full_text}

    def translate(
        self,
        audio: bytes,
        model: str,
        response_format: str = "json",
        temperature: float = 0.0,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError("Moonshine does not support translation (English-only model)")

    @staticmethod
    def _decode_audio(audio_bytes: bytes) -> Any:
        """Decode audio bytes to 16kHz mono float32 numpy array."""
        import numpy as np

        try:
            import soundfile as sf
            import io
            audio_array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        except Exception:
            # Fallback: try raw PCM 16-bit
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            sr = 16000

        # Convert to mono if stereo
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)

        # Resample to 16kHz if needed
        if sr != 16000:
            length = int(len(audio_array) * 16000 / sr)
            audio_array = np.interp(
                np.linspace(0, len(audio_array), length, endpoint=False),
                np.arange(len(audio_array)),
                audio_array,
            )

        return audio_array


def _format_ts_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_ts_vtt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
