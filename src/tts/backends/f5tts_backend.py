"""F5-TTS backend — flow-matching TTS with zero-shot voice cloning.

F5-TTS uses a reference audio clip (5-15s) plus its transcript to clone
any voice in a single forward pass. No fine-tuning required.

Models:
  - F5TTS_v1_Base: 330M params, 24kHz, Vocos vocoder (~1.5GB VRAM)
  - E2TTS_Base: 330M params, 24kHz (older, kept for compatibility)

Requires: pip install f5-tts
  Dependencies: torch, torchaudio, vocos, transformers, cached-path, pydub, soundfile
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

F5_MODELS: dict[str, dict[str, Any]] = {
    "f5-tts/v1-base": {
        "f5_model_name": "F5TTS_v1_Base",
        "description": "F5-TTS v1 Base — flow-matching, Vocos vocoder",
        "size_gb": 1.5,
        "min_vram_gb": 4,
        "sample_rate": 24000,
    },
    "f5-tts/e2-base": {
        "f5_model_name": "E2TTS_Base",
        "description": "E2-TTS Base — flat-U-net variant",
        "size_gb": 1.5,
        "min_vram_gb": 4,
        "sample_rate": 24000,
    },
}

# Default reference audio for when no clone source is provided.
# F5-TTS *requires* reference audio — there's no "default voice" mode.
# We ship a small built-in reference so the /v1/audio/speech endpoint
# works without requiring the caller to provide reference audio every time.
_BUILTIN_REF_TEXT = "Some call me nature, others call me mother nature."


def _get_builtin_ref_audio_path() -> str | None:
    """Get path to the built-in reference audio bundled with f5-tts."""
    try:
        from importlib.resources import files
        ref_path = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
        if Path(ref_path).exists():
            return ref_path
    except Exception:
        pass
    return None


class F5TTSBackend:
    """TTS backend using F5-TTS for zero-shot voice cloning via flow matching."""

    name = "f5-tts"
    sample_rate = 24000
    capabilities: dict[str, Any] = {
        **DEFAULT_TTS_CAPABILITIES,
        "voice_clone": True,
        "streaming": False,
        # F5-TTS v1 trained primarily on English + Chinese; other languages
        # work via zero-shot cloning but quality varies.
        "languages": ["en", "zh"],
        "speed_control": True,
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
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return "xpu"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def load_model(self, model_id: str) -> None:
        """Load an F5-TTS model (downloads from HuggingFace on first use)."""
        if model_id in self._models:
            return

        if model_id not in F5_MODELS:
            raise ValueError(
                f"Unknown F5-TTS model: {model_id}. "
                f"Available: {list(F5_MODELS)}"
            )

        try:
            from f5_tts.api import F5TTS  # noqa: F811
        except ImportError:
            raise RuntimeError(
                "F5-TTS requires the 'f5-tts' package. "
                "Install with: pip install f5-tts"
            )

        meta = F5_MODELS[model_id]
        device = self._resolve_device()

        logger.info("Loading F5-TTS model %s on %s...", model_id, device)
        start = time.time()

        try:
            engine = F5TTS(
                model=meta["f5_model_name"],
                device=device,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load F5-TTS model {model_id}: {e}")

        elapsed = time.time() - start
        logger.info("F5-TTS model %s loaded in %.1fs on %s", model_id, elapsed, device)

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
            logger.info("Unloaded F5-TTS model %s", model_id)

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
        """Synthesize text, optionally cloning a voice from reference audio.

        Args:
            text: Text to synthesize.
            voice: Voice identifier. Currently unused — F5-TTS uses reference
                audio for voice selection, not named voices. Kept for protocol compat.
            speed: Playback speed multiplier (clamped to 0.25-4.0).
            lang_code: Language code. Currently unused — F5-TTS infers language
                from reference audio. Kept for protocol compatibility.
            reference_audio: Raw audio bytes for zero-shot voice cloning.
                If not provided, uses the built-in default reference voice.
            reference_text: Transcript of the reference audio.
                If not provided and reference_audio is given, F5-TTS will
                auto-transcribe using Whisper (uses extra GPU memory).
        """
        if not self._models:
            raise RuntimeError("No F5-TTS model loaded. Load one first.")

        if not text or not text.strip():
            raise ValueError("Text must not be empty for F5-TTS synthesis.")

        # Clamp speed to safe range
        speed = max(0.25, min(speed, 4.0))

        model_id = next(iter(self._models))
        info = self._models[model_id]
        info["last_used_at"] = time.time()
        engine = info["engine"]

        # Resolve reference audio — do all I/O before yielding to avoid
        # generator-inside-try/except issues (GeneratorExit, cleanup timing).
        ref_file = None
        ref_text = reference_text or ""
        temp_path: str | None = None

        try:
            if reference_audio:
                # Write reference audio bytes to a temp file (F5-TTS needs a file path)
                fd = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                fd.write(reference_audio)
                fd.flush()
                fd.close()
                temp_path = fd.name
                ref_file = temp_path
            else:
                # Use built-in reference
                ref_file = _get_builtin_ref_audio_path()
                if ref_file is None:
                    raise RuntimeError(
                        "No reference audio provided and built-in reference not found. "
                        "F5-TTS requires reference audio for synthesis."
                    )
                ref_text = ref_text or _BUILTIN_REF_TEXT

            # F5-TTS inference — run to completion before yielding
            wav, sr, _spec = engine.infer(
                ref_file=ref_file,
                ref_text=ref_text,
                gen_text=text,
                speed=speed,
                show_info=lambda msg: logger.debug("F5-TTS: %s", msg),
                progress=None,  # No progress bar in server mode
            )
        except Exception as e:
            logger.exception("F5-TTS synthesis failed")
            raise RuntimeError(f"F5-TTS synthesis failed: {e}")
        finally:
            # Clean up temp file immediately after inference completes
            if temp_path is not None:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception:
                    pass

        # Post-process and yield outside try/except
        if not isinstance(wav, np.ndarray):
            wav = np.array(wav, dtype=np.float32)
        if wav.dtype != np.float32:
            wav = wav.astype(np.float32)

        if wav.size == 0:
            raise RuntimeError("F5-TTS returned empty audio.")

        # Normalize peak to avoid clipping
        peak = np.abs(wav).max()
        if peak > 1.0:
            wav = wav / peak

        yield wav

    def list_voices(self) -> list[VoiceInfo]:
        """F5-TTS uses zero-shot cloning — no built-in voice list.

        Returns a single "default" voice that uses the built-in reference.
        Users should provide their own reference audio for custom voices.
        """
        voices = [
            VoiceInfo(
                id="default",
                name="Default (built-in reference)",
                language="en-us",
                gender="female",
            ),
        ]
        return voices
