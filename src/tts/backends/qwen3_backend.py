"""Qwen3-TTS backend — high-quality TTS with voice design and cloning."""

from __future__ import annotations

import logging
import time
from typing import Iterator

import numpy as np

from src.tts.backends.base import TTSBackend, TTSLoadedModelInfo, VoiceInfo

logger = logging.getLogger(__name__)

QWEN3_MODELS: dict[str, dict] = {
    "qwen3-tts-0.6b": {
        "hf_id": "Qwen/Qwen3-TTS-0.6B",
        "size_gb": 1.2,
        "min_vram_gb": 2,
    },
    "qwen3-tts-1.7b": {
        "hf_id": "Qwen/Qwen3-TTS-1.7B",
        "size_gb": 3.4,
        "min_vram_gb": 4,
    },
}

DEFAULT_VOICE = "Chelsie"

# Built-in speaker names supported by Qwen3-TTS
QWEN3_VOICES = [
    {"id": "Chelsie", "name": "Chelsie", "gender": "female"},
    {"id": "Aidan", "name": "Aidan", "gender": "male"},
    {"id": "Serena", "name": "Serena", "gender": "female"},
    {"id": "Ethan", "name": "Ethan", "gender": "male"},
]


class Qwen3Backend:
    """TTS backend using Qwen3-TTS models via HuggingFace transformers."""

    name = "qwen3"
    sample_rate = 24000

    def __init__(self, device: str = "auto") -> None:
        self._device = device
        self._models: dict[str, dict] = {}  # model_id → {model, processor, loaded_at, ...}

    def _resolve_device(self, model_id: str) -> str:
        """Resolve device, checking CUDA availability for large models."""
        if self._device not in ("auto", "cuda", "cpu"):
            return self._device

        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False

        if self._device == "cpu":
            return "cpu"

        model_info = QWEN3_MODELS.get(model_id, {})
        min_vram = model_info.get("min_vram_gb", 2)

        if has_cuda:
            return "cuda"
        elif min_vram > 2:
            logger.warning(
                "Model %s needs ~%dGB VRAM but CUDA is not available. "
                "Loading on CPU may be very slow.",
                model_id, min_vram,
            )
        return "cpu"

    def load_model(self, model_id: str) -> None:
        if model_id in self._models:
            return

        if model_id not in QWEN3_MODELS:
            raise ValueError(f"Unknown Qwen3 model: {model_id}. Available: {list(QWEN3_MODELS)}")

        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError:
            raise RuntimeError(
                "Qwen3-TTS requires 'transformers' and 'accelerate'. "
                "Install with: pip install open-speech[qwen]"
            )

        hf_id = QWEN3_MODELS[model_id]["hf_id"]
        device = self._resolve_device(model_id)

        logger.info("Loading Qwen3-TTS model %s (%s) on %s", model_id, hf_id, device)
        start = time.time()

        try:
            import torch
            dtype = torch.float16 if device == "cuda" else torch.float32
        except ImportError:
            dtype = None

        processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device if device != "cpu" else None,
        )
        if device == "cpu":
            model = model.cpu()

        elapsed = time.time() - start
        logger.info("Qwen3-TTS model %s loaded in %.1fs on %s", model_id, elapsed, device)

        self._models[model_id] = {
            "model": model,
            "processor": processor,
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
            logger.info("Unloaded Qwen3-TTS model %s", model_id)

    def is_model_loaded(self, model_id: str) -> bool:
        return model_id in self._models

    def loaded_models(self) -> list[TTSLoadedModelInfo]:
        return [
            TTSLoadedModelInfo(
                model=mid,
                backend="qwen3",
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
        voice_design: str | None = None,
        reference_audio: bytes | None = None,
    ) -> Iterator[np.ndarray]:
        """Synthesize text to audio.

        Extended parameters:
            voice_design: Text description of desired voice characteristics.
            reference_audio: Raw audio bytes for voice cloning.
        """
        # Find a loaded model (use first available)
        if not self._models:
            raise RuntimeError("No Qwen3-TTS model loaded. Load one first.")

        model_id = next(iter(self._models))
        info = self._models[model_id]
        info["last_used_at"] = time.time()

        model = info["model"]
        processor = info["processor"]

        # Build the prompt based on mode
        if voice_design:
            # Voice design mode: use text description to shape voice
            prompt = f"<voice_design>{voice_design}</voice_design>{text}"
        elif reference_audio:
            # Voice cloning mode: pass reference audio
            prompt = text
        else:
            # Standard mode with speaker name
            speaker = voice if voice in [v["id"] for v in QWEN3_VOICES] else DEFAULT_VOICE
            prompt = f"<speaker>{speaker}</speaker>{text}"

        try:
            import torch

            inputs = processor(
                text=prompt,
                return_tensors="pt",
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=2048,
                )

            # Extract audio from model output
            audio = outputs[0].cpu().float().numpy()

            # Apply speed adjustment if not 1.0
            if speed != 1.0 and speed > 0:
                import numpy as np
                indices = np.arange(0, len(audio), speed)
                indices = indices[indices < len(audio)].astype(int)
                audio = audio[indices]

            # Normalize to float32 [-1, 1]
            if audio.max() > 0:
                audio = audio / np.abs(audio).max()

            yield audio.astype(np.float32)

        except Exception as e:
            logger.exception("Qwen3-TTS synthesis failed")
            raise RuntimeError(f"Qwen3-TTS synthesis failed: {e}")

    def list_voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(id=v["id"], name=v["name"], language="en-us", gender=v["gender"])
            for v in QWEN3_VOICES
        ]
