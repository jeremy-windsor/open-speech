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
    requires_model_id: bool = True
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

    @staticmethod
    def supports_model(model_id: str) -> bool:
        return model_id in PIPER_MODELS

    def _resolve_voice_model(self, voice: str, model_id: str | None) -> str:
        selected = model_id or (voice if voice in PIPER_MODELS else voice.rpartition("/")[0])
        if selected not in PIPER_MODELS:
            raise ValueError(f"Unknown Piper model: {selected}")
        return selected

    def _speaker_map(self, model_id: str, *, download: bool = False) -> dict[str, int]:
        from huggingface_hub import hf_hub_download, try_to_load_from_cache

        filename = _hf_path_for_model(PIPER_MODELS[model_id]["name"])[1]
        json_path = self._loaded.get(model_id, {}).get("json_path")
        if not json_path:
            json_path = try_to_load_from_cache(PIPER_HF_REPO, filename)
        if not isinstance(json_path, str) or not Path(json_path).is_file():
            if not download:
                return {}
            json_path = hf_hub_download(repo_id=PIPER_HF_REPO, filename=filename)
        with open(json_path, encoding="utf-8") as config_file:
            return json.load(config_file).get("speaker_id_map", {})

    def _speaker_id(self, voice: str, model_id: str) -> int | None:
        if voice in {model_id, "alloy", "default"}:
            return None
        prefix = f"{model_id}/"
        if not voice.startswith(prefix):
            raise ValueError(f"Unknown Piper voice: {voice}")
        speaker = voice[len(prefix):]
        speaker_map = self._speaker_map(model_id, download=True)
        if speaker not in speaker_map:
            raise ValueError(f"Unknown Piper voice: {voice}")
        return int(speaker_map[speaker])

    def validate_voice(self, voice: str, model_id: str | None = None) -> None:
        model_id = self._resolve_voice_model(voice, model_id)
        self._speaker_id(voice, model_id)

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
        model_id: str | None = None,
    ) -> Iterator[np.ndarray]:
        """Generate audio from text using the loaded Piper model.

        Voice IDs select the model's default voice or a named speaker.
        Yields float32 numpy chunks (one per sentence).
        """
        from piper.config import SynthesisConfig

        model_id = self._resolve_voice_model(voice, model_id)
        speaker_id = self._speaker_id(voice, model_id)
        if model_id not in self._loaded:
            logger.info("Auto-loading Piper model: %s", model_id)
            self.load_model(model_id)

        info = self._loaded[model_id]
        info["last_used"] = time.time()
        piper_voice = info["voice"]

        # Build synthesis config — length_scale < 1.0 is faster, > 1.0 is slower
        length_scale = (1.0 / speed) if speed > 0 else 1.0
        syn_config = SynthesisConfig(length_scale=length_scale, speaker_id=speaker_id)

        # synthesize() returns Iterable[AudioChunk]; each chunk has audio_float_array
        for chunk in piper_voice.synthesize(text, syn_config):
            audio_float32 = np.asarray(chunk.audio_float_array, dtype=np.float32)
            if audio_float32.ndim > 1:
                audio_float32 = audio_float32.flatten()
            yield audio_float32

    def list_voices(self, model_id: str | None = None) -> list[VoiceInfo]:
        """List speakers for the selected model, or all loaded models."""
        if model_id is not None and model_id not in PIPER_MODELS:
            raise ValueError(f"Unknown Piper model: {model_id}")
        voices = []
        models = [model_id] if model_id else list(self._loaded)
        for selected_model in models:
            meta = PIPER_MODELS.get(selected_model, {})
            model_name = meta.get("name", selected_model)
            lang = meta.get("lang", "en_US").replace("_", "-").lower()

            speaker_map = self._speaker_map(selected_model, download=True)
            if speaker_map:
                for speaker_name in speaker_map:
                    voices.append(VoiceInfo(
                        id=f"{selected_model}/{speaker_name}",
                        name=speaker_name,
                        language=lang,
                    ))
                continue

            # Single-speaker model
            voices.append(VoiceInfo(
                id=selected_model,
                name=model_name,
                language=lang,
            ))

        return voices

    def get_sample_rate(self, model_id: str) -> int:
        """Get sample rate for a specific loaded model."""
        if model_id in self._loaded:
            return self._loaded[model_id]["sample_rate"]
        meta = PIPER_MODELS.get(model_id)
        if meta is None:
            raise ValueError(f"Unknown Piper model: {model_id}")
        return meta["sample_rate"]
