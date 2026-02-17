"""faster-whisper backend using ctranslate2."""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from src.config import settings
from src.models import LoadedModelInfo

logger = logging.getLogger(__name__)


class FasterWhisperBackend:
    """STT backend using faster-whisper (ctranslate2)."""

    name = "faster-whisper"

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}  # model_id -> WhisperModel
        self._loaded_at: dict[str, float] = {}
        self._last_used: dict[str, float] = {}

    def load_model(self, model_id: str) -> None:
        """Load a faster-whisper model into memory."""
        if model_id in self._models:
            logger.info("Model %s already loaded", model_id)
            return

        from faster_whisper import WhisperModel

        logger.info("Loading model %s (device=%s, compute=%s)", 
                     model_id, settings.stt_device, settings.stt_compute_type)

        model = WhisperModel(
            model_id,
            device=settings.stt_device,
            compute_type=settings.stt_compute_type,
            download_root=settings.stt_model_dir,
        )

        self._models[model_id] = model
        self._loaded_at[model_id] = time.time()
        self._last_used[model_id] = time.time()
        logger.info("Model %s loaded successfully", model_id)

    def unload_model(self, model_id: str) -> None:
        """Unload a model from memory."""
        if model_id in self._models:
            del self._models[model_id]
            del self._loaded_at[model_id]
            self._last_used.pop(model_id, None)
            logger.info("Model %s unloaded", model_id)

    def loaded_models(self) -> list[LoadedModelInfo]:
        now = time.time()
        ttl = settings.stt_model_ttl
        default_model = settings.stt_default_model
        return [
            LoadedModelInfo(
                model=mid,
                backend=self.name,
                device=settings.stt_device,
                compute_type=settings.stt_compute_type,
                loaded_at=self._loaded_at[mid],
                last_used_at=self._last_used.get(mid),
                is_default=(mid == default_model),
                ttl_remaining=(
                    None if (mid == default_model or ttl == 0)
                    else max(0.0, ttl - (now - self._last_used.get(mid, now)))
                ),
            )
            for mid in self._models
        ]

    def is_model_loaded(self, model_id: str) -> bool:
        return model_id in self._models

    # --- Cache management ---

    def _get_cache_dir(self) -> Path:
        """Return the HuggingFace hub cache directory used by faster-whisper."""
        if settings.stt_model_dir:
            return Path(settings.stt_model_dir)
        # Default HF hub cache
        import os
        return Path(os.environ.get("HF_HUB_CACHE",
                    os.environ.get("HUGGINGFACE_HUB_CACHE",
                    Path.home() / ".cache" / "huggingface" / "hub")))

    def list_cached_models(self) -> list[dict[str, Any]]:
        """List models found in the HuggingFace cache directory."""
        cache_dir = self._get_cache_dir()
        results = []
        loaded_ids = set(self._models.keys())

        if not cache_dir.exists():
            # Still return loaded models (they may use a custom path)
            for mid in loaded_ids:
                results.append({
                    "model": mid,
                    "loaded": True,
                    "is_default": mid == settings.stt_default_model,
                    "size_mb": 0,
                })
            return results

        seen = set()

        if settings.stt_model_dir:
            # Custom model dir: models stored as models--Org--Name or directly
            for p in cache_dir.iterdir():
                if not p.is_dir():
                    continue
                if p.name.startswith("models--"):
                    # Convert models--Org--Name -> Org/Name
                    parts = p.name.split("--", 2)
                    if len(parts) == 3:
                        model_id = f"{parts[1]}/{parts[2]}"
                    else:
                        model_id = p.name
                else:
                    model_id = p.name
                seen.add(model_id)
                size_mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 * 1024)
                results.append({
                    "model": model_id,
                    "loaded": model_id in loaded_ids,
                    "is_default": model_id == settings.stt_default_model,
                    "size_mb": round(size_mb, 1),
                })
        else:
            # Default HF cache: directories named models--Org--Name
            for p in cache_dir.iterdir():
                if not p.is_dir() or not p.name.startswith("models--"):
                    continue
                parts = p.name.split("--", 2)
                if len(parts) != 3:
                    continue
                model_id = f"{parts[1]}/{parts[2]}"
                seen.add(model_id)
                size_mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 * 1024)
                results.append({
                    "model": model_id,
                    "loaded": model_id in loaded_ids,
                    "is_default": model_id == settings.stt_default_model,
                    "size_mb": round(size_mb, 1),
                })

        # Include loaded models not found in cache scan
        for mid in loaded_ids:
            if mid not in seen:
                results.append({
                    "model": mid,
                    "loaded": True,
                    "is_default": mid == settings.stt_default_model,
                    "size_mb": 0,
                })

        return results

    def _find_cache_path(self, model_id: str) -> Path | None:
        """Find the cache directory for a model."""
        cache_dir = self._get_cache_dir()
        if not cache_dir.exists():
            return None

        if settings.stt_model_dir:
            # Check models--Org--Name format
            safe_name = "models--" + model_id.replace("/", "--")
            p = cache_dir / safe_name
            if p.exists():
                return p
            # Check direct name
            p = cache_dir / model_id.split("/")[-1]
            if p.exists():
                return p
        else:
            safe_name = "models--" + model_id.replace("/", "--")
            p = cache_dir / safe_name
            if p.exists():
                return p
        return None

    def delete_cached_model(self, model_id: str) -> bool:
        """Delete a model from the cache. Returns True if found and deleted."""
        p = self._find_cache_path(model_id)
        if p and p.exists():
            shutil.rmtree(p)
            logger.info("Deleted cached model %s from %s", model_id, p)
            return True
        return False

    def is_model_cached(self, model_id: str) -> bool:
        """Check if a model exists in the cache."""
        return self._find_cache_path(model_id) is not None

    def _ensure_model(self, model_id: str) -> Any:
        """Ensure model is loaded, auto-load if not."""
        if model_id not in self._models:
            self.load_model(model_id)
        self._last_used[model_id] = time.time()
        return self._models[model_id]

    def _run_inference(
        self,
        audio: bytes,
        model_id: str,
        task: str = "transcribe",
        language: str | None = None,
        response_format: str = "json",
        temperature: float = 0.0,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        """Run transcription/translation inference."""
        whisper_model = self._ensure_model(model_id)

        # Write audio to temp file (faster-whisper needs a file path or ndarray)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            f.write(audio)
            f.flush()

            kwargs: dict[str, Any] = {
                "task": task,
                "beam_size": 5,
                "temperature": temperature,
            }
            if language and task == "transcribe":
                kwargs["language"] = language
            if prompt:
                kwargs["initial_prompt"] = prompt

            segments_gen, info = whisper_model.transcribe(f.name, **kwargs)
            segments = list(segments_gen)

        # Build response based on format
        full_text = "".join(s.text for s in segments).strip()

        if response_format in ("verbose_json",):
            return {
                "task": task,
                "language": info.language,
                "duration": info.duration,
                "text": full_text,
                "segments": [
                    {
                        "id": i,
                        "seek": int(s.seek),
                        "start": s.start,
                        "end": s.end,
                        "text": s.text,
                        "tokens": list(s.tokens) if s.tokens else [],
                        "temperature": s.temperature,
                        "avg_logprob": s.avg_logprob,
                        "compression_ratio": s.compression_ratio,
                        "no_speech_prob": s.no_speech_prob,
                    }
                    for i, s in enumerate(segments)
                ],
            }
        elif response_format == "text":
            return {"text": full_text, "raw_text": True}
        elif response_format == "srt":
            return {"text": self._to_srt(segments), "raw_text": True}
        elif response_format == "vtt":
            return {"text": self._to_vtt(segments), "raw_text": True}
        else:
            # json (default)
            return {"text": full_text}

    def transcribe(
        self,
        audio: bytes,
        model: str,
        language: str | None = None,
        response_format: str = "json",
        temperature: float = 0.0,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        return self._run_inference(
            audio, model, task="transcribe",
            language=language, response_format=response_format,
            temperature=temperature, prompt=prompt,
        )

    def translate(
        self,
        audio: bytes,
        model: str,
        response_format: str = "json",
        temperature: float = 0.0,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        return self._run_inference(
            audio, model, task="translate",
            response_format=response_format,
            temperature=temperature, prompt=prompt,
        )

    @staticmethod
    def _to_srt(segments: list) -> str:
        lines = []
        for i, s in enumerate(segments, 1):
            start = _format_timestamp_srt(s.start)
            end = _format_timestamp_srt(s.end)
            lines.append(f"{i}\n{start} --> {end}\n{s.text.strip()}\n")
        return "\n".join(lines)

    @staticmethod
    def _to_vtt(segments: list) -> str:
        lines = ["WEBVTT\n"]
        for s in segments:
            start = _format_timestamp_vtt(s.start)
            end = _format_timestamp_vtt(s.end)
            lines.append(f"{start} --> {end}\n{s.text.strip()}\n")
        return "\n".join(lines)


def _format_timestamp_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
