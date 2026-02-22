"""Piper TTS backend — lightweight, fast neural TTS using ONNX models."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from src.tts.backends.base import TTSLoadedModelInfo, VoiceInfo

logger = logging.getLogger(__name__)

# HuggingFace repo for Piper ONNX voices
PIPER_HF_REPO = "rhasspy/piper-voices"

# Curated available models: model_key → (relative path in HF repo, sample_rate)
PIPER_MODELS: dict[str, dict] = {
    "piper/en_US-lessac-medium": {
        "name": "en_US-lessac-medium",
        "lang": "en_US",
        "quality": "medium",
        "sample_rate": 22050,
    },
    "piper/en_US-lessac-high": {
        "name": "en_US-lessac-high",
        "lang": "en_US",
        "quality": "high",
        "sample_rate": 22050,
    },
    "piper/en_US-lessac-low": {
        "name": "en_US-lessac-low",
        "lang": "en_US",
        "quality": "low",
        "sample_rate": 16000,
    },
    "piper/en_US-amy-medium": {
        "name": "en_US-amy-medium",
        "lang": "en_US",
        "quality": "medium",
        "sample_rate": 22050,
    },
    "piper/en_US-amy-high": {
        "name": "en_US-amy-high",
        "lang": "en_US",
        "quality": "high",
        "sample_rate": 22050,
    },
    "piper/en_US-arctic-medium": {
        "name": "en_US-arctic-medium",
        "lang": "en_US",
        "quality": "medium",
        "sample_rate": 22050,
    },
    "piper/en_US-bryce-medium": {
        "name": "en_US-bryce-medium",
        "lang": "en_US",
        "quality": "medium",
        "sample_rate": 22050,
    },
    "piper/en_US-danny-low": {
        "name": "en_US-danny-low",
        "lang": "en_US",
        "quality": "low",
        "sample_rate": 16000,
    },
    "piper/en_US-hfc_female-medium": {
        "name": "en_US-hfc_female-medium",
        "lang": "en_US",
        "quality": "medium",
        "sample_rate": 22050,
    },
    "piper/en_US-hfc_male-medium": {
        "name": "en_US-hfc_male-medium",
        "lang": "en_US",
        "quality": "medium",
        "sample_rate": 22050,
    },
    "piper/en_US-joe-medium": {
        "name": "en_US-joe-medium",
        "lang": "en_US",
        "quality": "medium",
        "sample_rate": 22050,
    },
    "piper/en_US-john-medium": {
        "name": "en_US-john-medium",
        "lang": "en_US",
        "quality": "medium",
        "sample_rate": 22050,
    },
    "piper/en_US-kathleen-low": {
        "name": "en_US-kathleen-low",
        "lang": "en_US",
        "quality": "low",
        "sample_rate": 16000,
    },
    "piper/en_US-kusal-medium": {
        "name": "en_US-kusal-medium",
        "lang": "en_US",
        "quality": "medium",
        "sample_rate": 22050,
    },
    "piper/en_US-libritts_r-medium": {
        "name": "en_US-libritts_r-medium",
        "lang": "en_US",
        "quality": "medium",
        "sample_rate": 22050,
    },
    "piper/en_US-ljspeech-high": {
        "name": "en_US-ljspeech-high",
        "lang": "en_US",
        "quality": "high",
        "sample_rate": 22050,
    },
    "piper/en_US-ljspeech-medium": {
        "name": "en_US-ljspeech-medium",
        "lang": "en_US",
        "quality": "medium",
        "sample_rate": 22050,
    },
    "piper/en_US-norman-medium": {
        "name": "en_US-norman-medium",
        "lang": "en_US",
        "quality": "medium",
        "sample_rate": 22050,
    },
    "piper/en_US-ryan-low": {
        "name": "en_US-ryan-low",
        "lang": "en_US",
        "quality": "low",
        "sample_rate": 16000,
    },
    "piper/en_US-ryan-medium": {
        "name": "en_US-ryan-medium",
        "lang": "en_US",
        "quality": "medium",
        "sample_rate": 22050,
    },
    "piper/en_US-ryan-high": {
        "name": "en_US-ryan-high",
        "lang": "en_US",
        "quality": "high",
        "sample_rate": 22050,
    },
    "piper/en_GB-alan-low": {
        "name": "en_GB-alan-low",
        "lang": "en_GB",
        "quality": "low",
        "sample_rate": 16000,
    },
    "piper/en_GB-alan-medium": {
        "name": "en_GB-alan-medium",
        "lang": "en_GB",
        "quality": "medium",
        "sample_rate": 22050,
    },
    "piper/en_GB-cori-medium": {
        "name": "en_GB-cori-medium",
        "lang": "en_GB",
        "quality": "medium",
        "sample_rate": 22050,
    },
    "piper/en_GB-cori-high": {
        "name": "en_GB-cori-high",
        "lang": "en_GB",
        "quality": "high",
        "sample_rate": 22050,
    },
    "piper/en_GB-jenny_dioco-medium": {
        "name": "en_GB-jenny_dioco-medium",
        "lang": "en_GB",
        "quality": "medium",
        "sample_rate": 22050,
    },
    "piper/en_GB-northern_english_male-medium": {
        "name": "en_GB-northern_english_male-medium",
        "lang": "en_GB",
        "quality": "medium",
        "sample_rate": 22050,
    },
    "piper/en_GB-semaine-medium": {
        "name": "en_GB-semaine-medium",
        "lang": "en_GB",
        "quality": "medium",
        "sample_rate": 22050,
    },
    "piper/en_GB-southern_english_female-low": {
        "name": "en_GB-southern_english_female-low",
        "lang": "en_GB",
        "quality": "low",
        "sample_rate": 16000,
    },
    "piper/en_GB-southern_english_female-medium": {
        "name": "en_GB-southern_english_female-medium",
        "lang": "en_GB",
        "quality": "medium",
        "sample_rate": 22050,
    },
}


def _get_cache_dir() -> Path:
    """Get piper model cache directory."""
    from huggingface_hub import constants as hf_constants
    return Path(hf_constants.HF_HUB_CACHE)


def _hf_path_for_model(model_name: str) -> tuple[str, str]:
    """Return (onnx_path, json_path) relative paths within the HF repo."""
    # e.g. en_US-lessac-medium → en/en_US/lessac/medium/en_US-lessac-medium.onnx
    parts = model_name.split("-")
    lang = parts[0]  # en_US
    lang_short = lang.split("_")[0]  # en
    voice_name = parts[1] if len(parts) > 1 else "unknown"
    quality = parts[2] if len(parts) > 2 else "medium"
    if quality not in {"low", "medium", "high"}:
        quality = "medium"

    base = f"{lang_short}/{lang}/{voice_name}/{quality}/{model_name}"
    return f"{base}.onnx", f"{base}.onnx.json"


class PiperBackend:
    """TTS backend using Piper ONNX models."""

    name: str = "piper"
    single_speaker: bool = True  # Model_id selects the voice, not a voice name
    sample_rate: int = 22050  # Default; varies per model
    capabilities: dict = {
        "voice_blend": False,
        "voice_design": False,
        "voice_clone": False,
        "streaming": False,
        "instructions": False,
        "speakers": [
            {"name": meta["name"], "description": f"{meta['lang']} {meta['quality']}", "language": meta["lang"].replace("_", "-").lower()}
            for meta in PIPER_MODELS.values()
        ],
        "languages": sorted({meta["lang"].split("_")[0].lower() for meta in PIPER_MODELS.values()}),
        "speed_control": True,
        "ssml": False,
        "batch": False,
    }

    @classmethod
    def is_available(cls) -> bool:
        try:
            import piper  # noqa: F401
            return True
        except ImportError:
            return False

    def __init__(self, device: str = "auto") -> None:
        self._device = device
        self._loaded: dict[str, dict] = {}  # model_id → {"voice": PiperVoice, "info": {...}}

    def _download_model(self, model_id: str) -> tuple[str, str]:
        """Download model files from HuggingFace. Returns (onnx_path, json_path)."""
        from huggingface_hub import hf_hub_download

        meta = PIPER_MODELS.get(model_id)
        if not meta:
            raise ValueError(f"Unknown Piper model: {model_id}")

        model_name = meta["name"]
        onnx_rel, json_rel = _hf_path_for_model(model_name)

        logger.info("Downloading Piper model %s from HuggingFace...", model_id)
        onnx_path = hf_hub_download(repo_id=PIPER_HF_REPO, filename=onnx_rel)
        json_path = hf_hub_download(repo_id=PIPER_HF_REPO, filename=json_rel)
        logger.info("Piper model %s downloaded", model_id)
        return onnx_path, json_path

    def load_model(self, model_id: str) -> None:
        """Load a Piper model (download if needed)."""
        if model_id in self._loaded:
            return

        onnx_path, json_path = self._download_model(model_id)

        logger.info("Loading Piper model %s...", model_id)
        start = time.time()

        try:
            from piper import PiperVoice
        except ImportError:
            raise RuntimeError(
                "piper-tts package is not installed. "
                "Rebuild the image with BAKED_PROVIDERS=piper (or kokoro,piper). "
                "Example: docker build --build-arg BAKED_PROVIDERS=kokoro,piper ."
            )

        voice = PiperVoice.load(onnx_path, config_path=json_path)

        meta = PIPER_MODELS.get(model_id, {})
        sr = meta.get("sample_rate", 22050)

        self._loaded[model_id] = {
            "voice": voice,
            "onnx_path": onnx_path,
            "json_path": json_path,
            "sample_rate": sr,
            "loaded_at": time.time(),
            "last_used": None,
        }
        elapsed = time.time() - start
        logger.info("Piper model %s loaded in %.1fs (sample_rate=%d)", model_id, elapsed, sr)

    def unload_model(self, model_id: str) -> None:
        if model_id in self._loaded:
            del self._loaded[model_id]
            logger.info("Piper model %s unloaded", model_id)

    def is_model_loaded(self, model_id: str) -> bool:
        return model_id in self._loaded

    def loaded_models(self) -> list[TTSLoadedModelInfo]:
        result = []
        for mid, info in self._loaded.items():
            result.append(TTSLoadedModelInfo(
                model=mid,
                backend=self.name,
                device="cpu",
                loaded_at=info["loaded_at"],
                last_used_at=info.get("last_used"),
            ))
        return result

    def synthesize(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        lang_code: str | None = None,
    ) -> Iterator[np.ndarray]:
        """Generate audio from text using the loaded Piper model.

        The `voice` param is the model_id for Piper (e.g. piper/en_US-lessac-medium).
        Piper models are single-speaker, so voice is used to select the model.
        Yields float32 numpy chunks (one per sentence).
        """
        from piper.config import SynthesisConfig

        # Find which loaded model to use.
        # `voice` may be the model_id (e.g. "piper/en_US-lessac-medium") or a
        # generic voice name like "alloy".  Try exact match first, then fall
        # back to the first loaded model.
        if not self._loaded:
            # Auto-load the requested model (or first known model)
            auto_model = voice if voice in PIPER_MODELS else None
            if not auto_model:
                # Fall back to first known model
                auto_model = next(iter(PIPER_MODELS), None)
            if auto_model:
                logger.info("Auto-loading Piper model: %s", auto_model)
                self.load_model(auto_model)
            if not self._loaded:
                raise RuntimeError("No Piper model loaded")

        if voice in self._loaded:
            model_id = voice
        else:
            model_id = next(iter(self._loaded))
            if voice and voice != "alloy":
                logger.warning(
                    "Requested voice %r not found in loaded Piper models; "
                    "falling back to %s",
                    voice,
                    model_id,
                )

        info = self._loaded[model_id]
        info["last_used"] = time.time()
        piper_voice = info["voice"]
        sr = info["sample_rate"]

        # Build synthesis config — length_scale < 1.0 is faster, > 1.0 is slower
        length_scale = (1.0 / speed) if speed > 0 else 1.0
        syn_config = SynthesisConfig(length_scale=length_scale)

        # Update sample_rate from model metadata
        self.sample_rate = sr

        # synthesize() returns Iterable[AudioChunk]; each chunk has audio_float_array
        for chunk in piper_voice.synthesize(text, syn_config):
            audio_float32 = np.asarray(chunk.audio_float_array, dtype=np.float32)
            if audio_float32.ndim > 1:
                audio_float32 = audio_float32.flatten()
            yield audio_float32

    def list_voices(self) -> list[VoiceInfo]:
        """List voices from loaded models' metadata."""
        voices = []
        for model_id, info in self._loaded.items():
            meta = PIPER_MODELS.get(model_id, {})
            model_name = meta.get("name", model_id)
            lang = meta.get("lang", "en_US").replace("_", "-").lower()

            # Try reading config JSON for speaker info
            json_path = info.get("json_path")
            if json_path:
                try:
                    with open(json_path) as f:
                        config = json.load(f)
                    # Multi-speaker models have speaker_id_map
                    speaker_map = config.get("speaker_id_map", {})
                    if speaker_map:
                        for speaker_name in speaker_map:
                            voices.append(VoiceInfo(
                                id=f"{model_id}/{speaker_name}",
                                name=speaker_name,
                                language=lang,
                            ))
                        continue
                except Exception:
                    pass

            # Single-speaker model
            voices.append(VoiceInfo(
                id=model_id,
                name=model_name,
                language=lang,
            ))

        return voices

    def get_sample_rate(self, model_id: str) -> int:
        """Get sample rate for a specific loaded model."""
        if model_id in self._loaded:
            return self._loaded[model_id]["sample_rate"]
        meta = PIPER_MODELS.get(model_id, {})
        return meta.get("sample_rate", 22050)
