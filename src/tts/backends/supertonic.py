"""Supertonic 3 backend: Supertone's ONNX voices on ONNX Runtime, on the GPU when its CUDA provider is there."""

from __future__ import annotations

import gc
import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from src.live_tts.segmenter import split_sentences
from src.tts.backends.base import TTSLoadedModelInfo, VoiceInfo, check_declared_language
from src.tts.text_normalizer import normalize_english
from src.tts.voices import parse_voice_spec

logger = logging.getLogger(__name__)

SUPERTONIC_MODEL_ID = "supertonic"
SUPERTONIC_REPO_ID = "Supertone/supertonic-3"
# The revision supertonic 1.3.1 was released against (supertonic.config.MODEL_CONFIGS).
SUPERTONIC_REVISION = "724fb5abbf5502583fb520898d45929e62f02c0b"
SUPERTONIC_VOICES = ("M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5")
DEFAULT_VOICE = "M1"
_SESSIONS = ("duration_predictor", "text_encoder", "vector_estimator", "vocoder")
_FILES = ("onnx/tts.json", "onnx/unicode_indexer.json", *(f"onnx/{name}.onnx" for name in _SESSIONS),
          "voice_styles/*.json")
SUPERTONIC_LANGUAGES = [
    "en", "ar", "bg", "cs", "da", "de", "el", "es", "et", "fi", "fr", "hi", "hr", "hu", "id", "it", "ja", "ko",
    "lt", "lv", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "tr", "uk", "vi",
]
DEFAULT_STEPS = 8
MAX_STEPS = 32
# Speed 1 is the package's default pace (1.05); the model takes 0.7 to 2.0, so speed 0.7 to 1.9.
BASE_PACE, MIN_PACE, MAX_PACE = 1.05, 0.7, 2.0
MIN_SPEED, MAX_SPEED = 0.7, 1.9
# OpenAI voice names, by the gender of the Kokoro voice each one maps to (src.tts.voices.OPENAI_VOICE_MAP).
OPENAI_VOICES = {
    "alloy": "F1", "marin": "F1", "nova": "F2", "shimmer": "F3", "coral": "F4", "sage": "F5", "fable": "F5",
    "onyx": "M1", "cedar": "M1", "echo": "M2", "ash": "M3", "ballad": "M4", "verse": "M5",
}
# The package's chunk length; longer text is read sentence by sentence.
MAX_SENTENCE_CHARS = 300
SENTENCE_GAP_S = 0.3


def _load_style(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    return tuple(
        np.asarray(data[key]["data"], dtype=np.float32).reshape(data[key]["dims"])
        for key in ("style_ttl", "style_dp")
    )


def blend_styles(styles: dict[str, tuple[np.ndarray, np.ndarray]],
                 weights: list[tuple[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    """The weighted average of the voices' style vectors (weights summing to 1)."""
    ttl = sum(weight * styles[voice][0] for voice, weight in weights)
    dp = sum(weight * styles[voice][1] for voice, weight in weights)
    return ttl.astype(np.float32), dp.astype(np.float32)


class SupertonicBackend:
    """TTS backend for Supertonic 3 (ten preset voices, blendable)."""

    name = "supertonic"
    sample_rate = 44100
    capabilities: dict[str, Any] = {
        "voice_blend": True,
        "voice_design": False,
        "voice_clone": False,
        "streaming": True,
        "instructions": False,
        "speakers": [
            {"name": voice, "description": f"{'Male' if voice[0] == 'M' else 'Female'} voice", "language": "en"}
            for voice in SUPERTONIC_VOICES
        ],
        "languages": SUPERTONIC_LANGUAGES,
        "speed_control": True,
        "speed_range": {"min": MIN_SPEED, "max": MAX_SPEED},
        "ssml": False,
        "batch": False,
        # Request fields only this model reads; others ignore them.
        "options": {
            "steps": {
                "type": "integer", "label": "Quality steps", "min": 1, "max": MAX_STEPS, "step": 1,
                "default": DEFAULT_STEPS,
                "description": "Fewer is faster, more is cleaner.",
            },
        },
    }

    @classmethod
    def is_available(cls) -> bool:
        try:
            import onnxruntime  # noqa: F401
            import supertonic  # noqa: F401
            return True
        except ImportError:
            return False

    def __init__(self, device: str = "auto") -> None:
        self._device = device
        self._loaded: dict[str, Any] | None = None
        # The router serializes load, unload and synthesis but not status reads.
        self._state_lock = threading.Lock()

    @staticmethod
    def supports_model(model_id: str) -> bool:
        return model_id == SUPERTONIC_MODEL_ID

    def _check_model(self, model_id: str) -> None:
        if model_id != SUPERTONIC_MODEL_ID:
            raise ValueError(f"Unknown Supertonic model: {model_id}")

    def _get_device(self) -> str:
        """"cuda" when TTS_DEVICE allows it and ONNX Runtime has its CUDA provider, else "cpu"."""
        loaded = self._loaded
        if loaded is not None:
            return loaded["device"]
        wanted = str(self._device or "auto").strip().lower()
        if not (wanted == "auto" or wanted.startswith("cuda")):
            return "cpu"
        try:
            import onnxruntime as ort
        except ImportError:
            return "cpu"
        return "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"

    def _providers(self, device: str) -> list:
        if device != "cuda":
            return ["CPUExecutionProvider"]
        # EXHAUSTIVE would search again for every new sentence length.
        options: dict[str, Any] = {"cudnn_conv_algo_search": "HEURISTIC", "arena_extend_strategy": "kSameAsRequested"}
        wanted = str(self._device or "").strip().lower()
        if wanted.startswith("cuda:") and wanted[5:].isdigit():
            options["device_id"] = int(wanted[5:])
        return [("CUDAExecutionProvider", options), "CPUExecutionProvider"]

    @staticmethod
    def _files() -> Path:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(
            repo_id=SUPERTONIC_REPO_ID, revision=SUPERTONIC_REVISION, allow_patterns=list(_FILES),
        ))

    def download_model(self, model_id: str) -> None:
        """Fetch the ONNX files and voices into the Hugging Face cache, without loading them."""
        self._check_model(model_id)
        self._files()

    def load_model(self, model_id: str) -> None:
        self._check_model(model_id)
        if self._loaded is not None:
            return
        try:
            import onnxruntime as ort
            from supertonic.core import Supertonic, UnicodeProcessor
        except ImportError as e:
            raise RuntimeError(
                "supertonic package is not installed. Rebuild the image with supertonic in BAKED_PROVIDERS."
            ) from e

        logger.info("Loading Supertonic 3...")
        start = time.time()
        root = self._files()
        device = self._get_device()
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sessions = [
            ort.InferenceSession(str(root / "onnx" / f"{name}.onnx"), sess_options=options,
                                 providers=self._providers(device))
            for name in _SESSIONS
        ]
        if device == "cuda" and any("CUDAExecutionProvider" not in s.get_providers() for s in sessions):
            logger.warning("ONNX Runtime couldn't start Supertonic on the GPU: it runs on the CPU")
            device = "cpu"
            sessions = [
                ort.InferenceSession(str(root / "onnx" / f"{name}.onnx"), sess_options=options,
                                     providers=self._providers(device))
                for name in _SESSIONS
            ]
        with open(root / "onnx" / "tts.json", encoding="utf-8") as file:
            config = json.load(file)
        processor = UnicodeProcessor(str(root / "onnx" / "unicode_indexer.json"))
        engine = Supertonic(config, processor, *sessions)
        styles = {voice: _load_style(root / "voice_styles" / f"{voice}.json") for voice in SUPERTONIC_VOICES}
        self.sample_rate = int(engine.sample_rate)
        if device == "cuda":
            # The first run sets up cuDNN; doing it here keeps it off the first request.
            try:
                from supertonic.core import Style

                engine(["Hello."], Style(*styles[DEFAULT_VOICE]), 2, BASE_PACE, "en")
            except Exception as e:
                logger.warning("Supertonic GPU warm-up failed (non-fatal): %s", e)
        with self._state_lock:
            self._loaded = {
                "engine": engine, "processor": processor, "styles": styles, "device": device,
                "loaded_at": time.time(), "last_used_at": None,
            }
        logger.info("Supertonic 3 loaded on %s in %.1fs", device, time.time() - start)

    def unload_model(self, model_id: str) -> None:
        self._check_model(model_id)
        with self._state_lock:
            unloaded, self._loaded = self._loaded is not None, None
        if unloaded:
            gc.collect()
            logger.info("Supertonic 3 unloaded")

    def is_model_loaded(self, model_id: str) -> bool:
        return model_id == SUPERTONIC_MODEL_ID and self._loaded is not None

    def loaded_models(self) -> list[TTSLoadedModelInfo]:
        with self._state_lock:
            loaded = self._loaded
        if loaded is None:
            return []
        return [TTSLoadedModelInfo(
            model=SUPERTONIC_MODEL_ID, backend=self.name, device=loaded["device"],
            loaded_at=loaded["loaded_at"], last_used_at=loaded["last_used_at"],
        )]

    @staticmethod
    def voice_weights(voice: str) -> list[tuple[str, float]]:
        """A voice or blend ("M1", "m1(0.6)+F2(0.4)") as (voice, weight) pairs; an OpenAI name is OPENAI_VOICES'."""
        raw = (voice or "").strip()
        if not raw or raw.lower() == "default":
            return [(DEFAULT_VOICE, 1.0)]
        if raw.lower() in OPENAI_VOICES:
            return [(OPENAI_VOICES[raw.lower()], 1.0)]
        spec = parse_voice_spec(raw)
        for component in spec.components:
            if component.voice_id.upper() not in SUPERTONIC_VOICES:
                if spec.is_blend:
                    raise ValueError(
                        f"Unknown Supertonic voice in the blend: {component.voice_id}. A blend must use one model's "
                        f"voices: {', '.join(SUPERTONIC_VOICES)}"
                    )
                raise ValueError(
                    f"Unknown Supertonic voice: {component.voice_id}. Voices: {', '.join(SUPERTONIC_VOICES)}"
                )
        return list(zip([c.voice_id.upper() for c in spec.components], spec.normalized_weights()))

    def validate_voice(self, voice: str) -> None:
        self.voice_weights(voice)

    def validate_language(self, language: str | None) -> None:
        check_declared_language(language, SUPERTONIC_LANGUAGES, SUPERTONIC_MODEL_ID)

    @staticmethod
    def validate_text(text: str) -> None:
        if not any(char.isalnum() for char in text):
            raise ValueError("Input text is empty")

    @staticmethod
    def _language(lang_code: str | None) -> str:
        if not lang_code or not lang_code.strip():
            return "en"
        check_declared_language(lang_code, SUPERTONIC_LANGUAGES, SUPERTONIC_MODEL_ID)
        return lang_code.strip().lower().replace("_", "-").split("-", 1)[0]

    @staticmethod
    def _steps(steps: Any) -> int:
        if steps is None:
            return DEFAULT_STEPS
        if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= MAX_STEPS:
            raise ValueError(f"steps must be a whole number from 1 to {MAX_STEPS}")
        return steps

    def synthesize(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        lang_code: str | None = None,
        steps: int | None = None,
    ) -> Iterator[np.ndarray]:
        """Yield one chunk per line or sentence, with a short pause between them."""
        weights = self.voice_weights(voice)
        lang = self._language(lang_code)
        total_steps = self._steps(steps)
        if self._loaded is None:
            self.load_model(SUPERTONIC_MODEL_ID)
        loaded = self._loaded
        loaded["last_used_at"] = time.time()
        from supertonic.core import Style

        style = Style(*blend_styles(loaded["styles"], weights))
        pace = min(MAX_PACE, max(MIN_PACE, BASE_PACE * speed))
        if lang == "en":
            text = normalize_english(text)
        gap = np.zeros(int(SENTENCE_GAP_S * self.sample_rate), dtype=np.float32)
        spoke = False
        lines = re.split(r"\n+", text)
        sentences = (line[start:end] for line in lines for start, end in split_sentences(line, MAX_SENTENCE_CHARS))
        for sentence in sentences:
            sentence = self._speakable(loaded["processor"], sentence)
            if not sentence:
                continue
            wav, duration = loaded["engine"]([sentence], style, total_steps, pace, lang)
            audio = np.asarray(wav, dtype=np.float32).reshape(-1)
            audio = audio[: int(self.sample_rate * float(np.asarray(duration).reshape(-1)[0]))]
            if not audio.size:
                continue
            if spoke:
                yield gap
            spoke = True
            loaded["last_used_at"] = time.time()
            yield audio

    @staticmethod
    def _speakable(processor: Any, sentence: str) -> str:
        """The sentence without characters the model has no embedding for; '' when nothing is left to say."""
        _valid, unsupported = processor.validate_text(sentence)
        if unsupported:
            sentence = sentence.translate({ord(char): " " for char in unsupported})
        sentence = " ".join(sentence.split())
        return sentence if any(char.isalnum() for char in sentence) else ""

    def list_voices(self) -> list[VoiceInfo]:
        return [
            VoiceInfo(id=voice, name=voice, language="en", gender="male" if voice[0] == "M" else "female")
            for voice in SUPERTONIC_VOICES
        ]
