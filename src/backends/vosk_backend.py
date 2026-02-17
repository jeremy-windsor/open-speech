"""Vosk STT backend — lightweight, offline, Kaldi-based."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path, PurePosixPath
from typing import Any

from src.models import LoadedModelInfo

logger = logging.getLogger(__name__)

VOSK_MODEL_URLS = {
    "vosk-model-small-en-us-0.15": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    "vosk-model-en-us-0.22": "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip",
}

DEFAULT_VOSK_MODEL = "vosk-model-small-en-us-0.15"


def _safe_extract_zip(zf: Any, dest: Path) -> None:
    """Safely extract zip members to dest, preventing path traversal."""
    dest_resolved = dest.resolve()
    for member in zf.infolist():
        name = member.filename
        p = PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"Unsafe zip member path: {name}")

        target = (dest / name).resolve()
        if dest_resolved not in target.parents and target != dest_resolved:
            raise ValueError(f"Zip member escapes destination: {name}")

    for member in zf.infolist():
        zf.extract(member, dest)


class VoskBackend:
    """STT backend using Vosk (Kaldi-based, offline)."""

    name = "vosk"

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}  # model_id -> vosk.Model
        self._loaded_at: dict[str, float] = {}
        self._last_used: dict[str, float] = {}
        self._model_dir = os.environ.get("VOSK_MODEL_DIR", str(Path.home() / ".cache" / "vosk"))

    def load_model(self, model_id: str) -> None:
        if model_id in self._models:
            return

        try:
            import vosk
        except ImportError:
            raise ImportError(
                "vosk is not installed. Install with: pip install 'open-speech[vosk]'"
            )

        vosk.SetLogLevel(-1)  # Suppress Vosk's verbose logging

        model_path = self._resolve_model_path(model_id)
        logger.info("Loading Vosk model from %s", model_path)
        model = vosk.Model(str(model_path))
        self._models[model_id] = model
        self._loaded_at[model_id] = time.time()
        self._last_used[model_id] = time.time()
        logger.info("Vosk model %s loaded", model_id)

    def _resolve_model_path(self, model_id: str) -> Path:
        """Find or download a Vosk model."""
        model_dir = Path(self._model_dir) / model_id
        if model_dir.exists():
            return model_dir

        # Auto-download if known model
        if model_id in VOSK_MODEL_URLS:
            return self._download_model(model_id)

        # Try as direct path
        if Path(model_id).exists():
            return Path(model_id)

        raise FileNotFoundError(
            f"Vosk model '{model_id}' not found. Known models: {list(VOSK_MODEL_URLS.keys())}"
        )

    def _download_model(self, model_id: str) -> Path:
        """Download and extract a Vosk model."""
        import urllib.request
        import zipfile

        url = VOSK_MODEL_URLS[model_id]
        dest = Path(self._model_dir)
        dest.mkdir(parents=True, exist_ok=True)

        zip_path = dest / f"{model_id}.zip"
        logger.info("Downloading Vosk model %s from %s", model_id, url)
        urllib.request.urlretrieve(url, zip_path)

        logger.info("Extracting %s", zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            _safe_extract_zip(zf, dest)
        zip_path.unlink()

        model_path = dest / model_id
        if not model_path.exists():
            # Some zips have a different root dir
            extracted = [d for d in dest.iterdir() if d.is_dir() and d.name.startswith(model_id)]
            if extracted:
                model_path = extracted[0]

        return model_path

    def unload_model(self, model_id: str) -> None:
        if model_id in self._models:
            del self._models[model_id]
            del self._loaded_at[model_id]
            self._last_used.pop(model_id, None)

    def loaded_models(self) -> list[LoadedModelInfo]:
        from src.config import settings
        return [
            LoadedModelInfo(
                model=mid,
                backend=self.name,
                device="cpu",
                compute_type="int8",
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

        vosk_model = self._ensure_model(model)

        # Decode audio to 16kHz mono PCM
        audio_pcm, sr = self._decode_to_pcm(audio)

        from vosk import KaldiRecognizer

        rec = KaldiRecognizer(vosk_model, sr)
        rec.SetWords(True)

        # Feed audio in chunks
        chunk_size = 4000
        for i in range(0, len(audio_pcm), chunk_size):
            rec.AcceptWaveform(audio_pcm[i : i + chunk_size])

        final = json.loads(rec.FinalResult())
        full_text = final.get("text", "").strip()

        # Build segments from word-level results if available
        segments = []
        if "result" in final:
            words = final["result"]
            if words:
                segments.append(
                    {
                        "id": 0,
                        "seek": 0,
                        "start": words[0].get("start", 0.0),
                        "end": words[-1].get("end", 0.0),
                        "text": full_text,
                        "tokens": [],
                        "temperature": 0.0,
                        "avg_logprob": words[0].get("conf", 0.0),
                        "compression_ratio": 0.0,
                        "no_speech_prob": 0.0,
                    }
                )

        duration = len(audio_pcm) / (sr * 2)  # 16-bit = 2 bytes per sample

        if response_format == "verbose_json":
            return {
                "task": "transcribe",
                "language": language or "en",
                "duration": duration,
                "text": full_text,
                "segments": segments,
            }
        elif response_format == "text":
            return {"text": full_text, "raw_text": True}
        elif response_format == "srt":
            end = segments[0]["end"] if segments else duration
            srt = f"1\n{_format_ts_srt(0)}" f" --> {_format_ts_srt(end)}\n{full_text}\n"
            return {"text": srt, "raw_text": True}
        elif response_format == "vtt":
            end = segments[0]["end"] if segments else duration
            vtt = f"WEBVTT\n\n{_format_ts_vtt(0)} --> {_format_ts_vtt(end)}\n{full_text}\n"
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
        raise NotImplementedError("Vosk does not support translation")

    @staticmethod
    def _decode_to_pcm(audio_bytes: bytes) -> tuple[bytes, int]:
        """Decode audio bytes to 16kHz mono 16-bit PCM bytes. Returns (pcm_bytes, sample_rate)."""
        import numpy as np

        try:
            import soundfile as sf
            import io

            audio_array, sr = sf.read(io.BytesIO(audio_bytes), dtype="int16")
        except Exception:
            # Fallback: assume raw 16-bit PCM at 16kHz
            return audio_bytes, 16000

        # Convert to mono
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1).astype(np.int16)

        # Resample to 16kHz if needed
        if sr != 16000:
            float_arr = audio_array.astype(np.float32)
            length = int(len(float_arr) * 16000 / sr)
            resampled = np.interp(
                np.linspace(0, len(float_arr), length, endpoint=False),
                np.arange(len(float_arr)),
                float_arr,
            ).astype(np.int16)
            return resampled.tobytes(), 16000

        return audio_array.tobytes(), sr


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
