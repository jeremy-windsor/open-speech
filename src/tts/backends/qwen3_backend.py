"""Qwen3-TTS backend with deep qwen-tts integration."""

from __future__ import annotations

import hashlib
import inspect
import logging
import os
import tempfile
import time
from typing import Any, Iterator

import numpy as np

from src.config import settings
from src.tts.backends.base import TTSLoadedModelInfo, VoiceInfo

logger = logging.getLogger(__name__)

DEFAULT_SPEAKER = "Ryan"

# speaker -> (display name, language, gender)
QWEN3_PREMIUM_VOICES: dict[str, tuple[str, str, str]] = {
    "Vivian": ("Vivian", "zh", "female"),
    "Serena": ("Serena", "zh", "female"),
    "Uncle_Fu": ("Uncle Fu", "zh", "male"),
    "Dylan": ("Dylan", "zh", "male"),
    "Eric": ("Eric", "zh", "male"),
    "Ryan": ("Ryan", "en", "male"),
    "Aiden": ("Aiden", "en", "male"),
    "Ono_Anna": ("Ono Anna", "ja", "female"),
    "Sohee": ("Sohee", "ko", "female"),
}

# OpenAI voices mapped to sensible Qwen defaults
OPENAI_VOICE_ALIASES = {
    "alloy": "Ryan",
    "echo": "Aiden",
    "fable": "Ryan",
    "onyx": "Ryan",
    "nova": "Vivian",
    "shimmer": "Serena",
}


def _qwen3_hf_id(size: str, variant: str) -> str:
    return f"Qwen/Qwen3-TTS-12Hz-{size}-{variant}"


class Qwen3Backend:
    """TTS backend using official qwen-tts package."""

    name = "qwen3"
    sample_rate = 24000
    capabilities: dict = {
        "voice_blend": False,
        "voice_design": True,
        "voice_clone": True,
        "streaming": True,
        "instructions": True,
        "speakers": [
            {"name": "Vivian", "description": "Warm female narrator", "language": "zh"},
            {"name": "Serena", "description": "Bright female conversational", "language": "zh"},
            {"name": "Uncle_Fu", "description": "Mature male storyteller", "language": "zh"},
            {"name": "Dylan", "description": "Balanced male assistant", "language": "zh"},
            {"name": "Eric", "description": "Clear male professional", "language": "zh"},
            {"name": "Ryan", "description": "Neutral male general", "language": "en"},
            {"name": "Aiden", "description": "Energetic male English", "language": "en"},
            {"name": "Ono_Anna", "description": "Natural Japanese female", "language": "ja"},
            {"name": "Sohee", "description": "Natural Korean female", "language": "ko"},
        ],
        "languages": ["Auto", "en-us", "en", "zh", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"],
        "speed_control": True,
        "ssml": False,
        "batch": True,
    }

    def __init__(self, device: str = "auto") -> None:
        self._device = settings.tts_qwen3_device or device
        self._size = settings.tts_qwen3_size
        self._flash_attn = settings.tts_qwen3_flash_attn
        self._loaded_models: dict[str, dict[str, Any]] = {}
        self._by_hf_id: dict[str, Any] = {}
        self._clone_prompt_cache: dict[str, Any] = {}

    def _resolve_size(self, model_id: str) -> str:
        mid = model_id.lower()
        if "0.6" in mid:
            return "0.6B"
        if "1.7" in mid:
            return "1.7B"
        return self._size

    def _resolve_device(self) -> str:
        return self._device if self._device not in ("", "auto", None) else "cuda:0"

    def _resolve_variant(self, *, voice: str, voice_design: str | None, reference_audio: bytes | None, size: str) -> str:
        if reference_audio is not None:
            return "Base"
        speaker = self._normalize_speaker(voice)
        if speaker:
            return "CustomVoice"
        if voice_design:
            if size == "0.6B":
                logger.warning("VoiceDesign is unavailable for 0.6B; falling back to CustomVoice")
                return "CustomVoice"
            return "VoiceDesign"
        return "CustomVoice"

    def _normalize_speaker(self, voice: str | None) -> str | None:
        if not voice:
            return None
        if voice in QWEN3_PREMIUM_VOICES:
            return voice
        mapped = OPENAI_VOICE_ALIASES.get(voice.lower())
        if mapped:
            return mapped
        return None

    def _validate_speaker(self, voice: str) -> str:
        speaker = self._normalize_speaker(voice)
        if speaker:
            return speaker
        raise ValueError(
            f"Unsupported Qwen3 speaker '{voice}'. Supported: {', '.join(QWEN3_PREMIUM_VOICES)}"
        )

    def _infer_language(self, speaker: str, lang_code: str | None) -> str | None:
        if lang_code:
            normalized = lang_code.strip().lower().replace("_", "-")
            # Qwen3 expects short tags (en/zh/ja/...) or Auto.
            if normalized in {"en-us", "en-gb", "en-au", "en-ca"}:
                return "en"
            if normalized == "auto":
                return "Auto"
            return normalized
        meta = QWEN3_PREMIUM_VOICES.get(speaker)
        return meta[1] if meta else None

    def _load_hf_model(self, hf_id: str) -> Any:
        if hf_id in self._by_hf_id:
            return self._by_hf_id[hf_id]

        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as e:
            raise RuntimeError(
                "Qwen3-TTS requires optional dependency group [qwen]. Install with: pip install -e '.[qwen]'"
            ) from e

        dtype = getattr(torch, "bfloat16", None) or getattr(torch, "float16", None)
        kwargs: dict[str, Any] = {
            "device_map": self._resolve_device(),
            "dtype": dtype,
        }
        if self._flash_attn:
            kwargs["attn_implementation"] = "flash_attention_2"

        start = time.time()
        logger.info("Loading Qwen3 model %s", hf_id)
        model = Qwen3TTSModel.from_pretrained(hf_id, **kwargs)
        logger.info("Loaded Qwen3 model %s in %.1fs", hf_id, time.time() - start)
        self._by_hf_id[hf_id] = model
        return model

    def _load_variant(self, variant: str, size: str, model_id: str) -> Any:
        hf_id = _qwen3_hf_id(size, variant)
        model = self._load_hf_model(hf_id)
        self._loaded_models[model_id] = {
            "model": model,
            "device": self._resolve_device(),
            "loaded_at": time.time(),
            "last_used_at": None,
            "hf_id": hf_id,
        }
        return model

    def _find_loaded_variant(self, variant: str, size: str) -> tuple[str, dict[str, Any]] | None:
        suffix = f"-{size}-{variant}"
        for mid, info in self._loaded_models.items():
            if info.get("hf_id", "").endswith(suffix):
                return mid, info
        return None

    def load_model(self, model_id: str) -> None:
        if self.is_model_loaded(model_id):
            return

        size = self._resolve_size(model_id)
        # explicit IDs map to variants; generic qwen3 loads CustomVoice
        variant = "CustomVoice"
        lower = model_id.lower()
        if "voicedesign" in lower:
            variant = "VoiceDesign"
        elif "-base" in lower or lower.endswith("/base"):
            variant = "Base"

        if size == "0.6B" and variant == "VoiceDesign":
            raise ValueError("0.6B Qwen3 does not provide VoiceDesign model")

        self._load_variant(variant, size, model_id)

    def unload_model(self, model_id: str) -> None:
        if model_id in self._loaded_models:
            del self._loaded_models[model_id]
            logger.info("Unloaded Qwen3 model handle %s", model_id)

    def is_model_loaded(self, model_id: str) -> bool:
        return model_id in self._loaded_models

    def loaded_models(self) -> list[TTSLoadedModelInfo]:
        return [
            TTSLoadedModelInfo(
                model=mid,
                backend="qwen3",
                device=info["device"],
                loaded_at=info["loaded_at"],
                last_used_at=info["last_used_at"],
            )
            for mid, info in self._loaded_models.items()
        ]

    def _get_runtime_model(self, *, variant: str, size: str) -> tuple[str, dict[str, Any]]:
        found = self._find_loaded_variant(variant, size)
        if found:
            return found

        runtime_id = f"qwen3-tts/{size}-{variant}"
        self.load_model(runtime_id)
        return runtime_id, self._loaded_models[runtime_id]

    def _supports_kw(self, fn: Any, key: str) -> bool:
        sig = inspect.signature(fn)
        if key in sig.parameters:
            return True
        return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    def _call(self, fn: Any, **kwargs: Any) -> Any:
        sig = inspect.signature(fn)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            return fn(**kwargs)
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters and v is not None}
        return fn(**filtered)

    def _prepare_ref_audio(self, reference_audio: bytes) -> str:
        fd, path = tempfile.mkstemp(prefix="qwen3_ref_", suffix=".wav")
        with os.fdopen(fd, "wb") as f:
            f.write(reference_audio)
        return path

    def _clone_prompt_key(self, ref_audio: bytes, ref_text: str | None) -> str:
        h = hashlib.sha256()
        h.update(ref_audio)
        h.update((ref_text or "").encode("utf-8"))
        return h.hexdigest()

    def synthesize(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        lang_code: str | None = None,
        *,
        voice_design: str | None = None,
        reference_audio: bytes | None = None,
        clone_transcript: str | None = None,
    ) -> Iterator[np.ndarray]:
        size = self._size
        variant = self._resolve_variant(
            voice=voice,
            voice_design=voice_design,
            reference_audio=reference_audio,
            size=size,
        )
        _, info = self._get_runtime_model(variant=variant, size=size)
        info["last_used_at"] = time.time()
        model = info["model"]

        tmp_ref_path: str | None = None
        try:
            if variant == "Base":
                if reference_audio is None:
                    raise ValueError("reference_audio is required for voice cloning")
                tmp_ref_path = self._prepare_ref_audio(reference_audio)
                prompt_key = self._clone_prompt_key(reference_audio, clone_transcript)
                clone_prompt = self._clone_prompt_cache.get(prompt_key)
                if clone_prompt is None and hasattr(model, "create_voice_clone_prompt"):
                    clone_prompt = self._call(
                        model.create_voice_clone_prompt,
                        ref_audio=tmp_ref_path,
                        ref_text=clone_transcript,
                    )
                    self._clone_prompt_cache[prompt_key] = clone_prompt

                kwargs: dict[str, Any] = {
                    "text": text,
                    "ref_audio": tmp_ref_path,
                    "ref_text": clone_transcript,
                    "speed": speed,
                }
                if clone_prompt is not None:
                    if self._supports_kw(model.generate_voice_clone, "clone_prompt"):
                        kwargs["clone_prompt"] = clone_prompt
                    elif self._supports_kw(model.generate_voice_clone, "voice_clone_prompt"):
                        kwargs["voice_clone_prompt"] = clone_prompt
                audio = self._call(model.generate_voice_clone, **kwargs)
            elif variant == "VoiceDesign":
                audio = self._call(
                    model.generate_voice_design,
                    text=text,
                    instruct=voice_design,
                    speed=speed,
                )
            else:
                speaker = self._validate_speaker(voice) if voice else DEFAULT_SPEAKER
                language = self._infer_language(speaker, lang_code)
                audio = self._call(
                    model.generate_custom_voice,
                    text=text,
                    speaker=speaker,
                    instruct=voice_design,
                    language=language,
                    speed=speed,
                )

            if isinstance(audio, list):
                arrays = [np.asarray(a, dtype=np.float32) for a in audio]
                combined = np.concatenate(arrays) if arrays else np.zeros(0, dtype=np.float32)
            else:
                combined = np.asarray(audio, dtype=np.float32)

            if combined.ndim > 1:
                combined = combined.squeeze()
            if combined.size and np.max(np.abs(combined)) > 1.0:
                combined = combined / np.max(np.abs(combined))
            yield combined.astype(np.float32)
        finally:
            if tmp_ref_path:
                try:
                    os.unlink(tmp_ref_path)
                except OSError:
                    pass

    def list_voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(id=speaker, name=meta[0], language=meta[1], gender=meta[2])
            for speaker, meta in QWEN3_PREMIUM_VOICES.items()
        ]
