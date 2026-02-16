"""Kokoro TTS backend — uses the kokoro Python package."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from src.tts.backends.base import TTSBackend, TTSLoadedModelInfo, VoiceInfo
from src.tts.voices import parse_voice_spec

logger = logging.getLogger(__name__)

# Voice ID prefix → lang_code mapping
VOICE_PREFIX_TO_LANG: dict[str, str] = {
    "a": "a",  # American English (af_, am_)
    "b": "b",  # British English (bf_, bm_)
    "e": "e",  # Spanish (ef_, em_)
    "f": "f",  # French (ff_)
    "h": "h",  # Hindi (hf_, hm_)
    "i": "i",  # Italian (if_, im_)
    "j": "j",  # Japanese (jf_, jm_)
    "p": "p",  # Portuguese (pf_)
    "z": "z",  # Mandarin (zf_, zm_)
}

LANG_CODE_TO_LANGUAGE: dict[str, str] = {
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr",
    "h": "hi",
    "i": "it",
    "j": "ja",
    "p": "pt-br",
    "z": "zh",
}

# All known Kokoro voices (comprehensive list)
ALL_KOKORO_VOICES: list[dict[str, str]] = [
    # American English - Female
    {"id": "af_heart", "name": "Heart", "lang": "a", "gender": "female"},
    {"id": "af_alloy", "name": "Alloy", "lang": "a", "gender": "female"},
    {"id": "af_aoede", "name": "Aoede", "lang": "a", "gender": "female"},
    {"id": "af_bella", "name": "Bella", "lang": "a", "gender": "female"},
    {"id": "af_jessica", "name": "Jessica", "lang": "a", "gender": "female"},
    {"id": "af_kore", "name": "Kore", "lang": "a", "gender": "female"},
    {"id": "af_nicole", "name": "Nicole", "lang": "a", "gender": "female"},
    {"id": "af_nova", "name": "Nova", "lang": "a", "gender": "female"},
    {"id": "af_river", "name": "River", "lang": "a", "gender": "female"},
    {"id": "af_sarah", "name": "Sarah", "lang": "a", "gender": "female"},
    {"id": "af_sky", "name": "Sky", "lang": "a", "gender": "female"},
    # American English - Male
    {"id": "am_adam", "name": "Adam", "lang": "a", "gender": "male"},
    {"id": "am_echo", "name": "Echo", "lang": "a", "gender": "male"},
    {"id": "am_eric", "name": "Eric", "lang": "a", "gender": "male"},
    {"id": "am_fenrir", "name": "Fenrir", "lang": "a", "gender": "male"},
    {"id": "am_liam", "name": "Liam", "lang": "a", "gender": "male"},
    {"id": "am_michael", "name": "Michael", "lang": "a", "gender": "male"},
    {"id": "am_onyx", "name": "Onyx", "lang": "a", "gender": "male"},
    {"id": "am_puck", "name": "Puck", "lang": "a", "gender": "male"},
    {"id": "am_santa", "name": "Santa", "lang": "a", "gender": "male"},
    # British English - Female
    {"id": "bf_alice", "name": "Alice", "lang": "b", "gender": "female"},
    {"id": "bf_emma", "name": "Emma", "lang": "b", "gender": "female"},
    {"id": "bf_isabella", "name": "Isabella", "lang": "b", "gender": "female"},
    {"id": "bf_lily", "name": "Lily", "lang": "b", "gender": "female"},
    # British English - Male
    {"id": "bm_daniel", "name": "Daniel", "lang": "b", "gender": "male"},
    {"id": "bm_fable", "name": "Fable", "lang": "b", "gender": "male"},
    {"id": "bm_george", "name": "George", "lang": "b", "gender": "male"},
    {"id": "bm_lewis", "name": "Lewis", "lang": "b", "gender": "male"},
    # Spanish
    {"id": "ef_dora", "name": "Dora", "lang": "e", "gender": "female"},
    {"id": "em_alex", "name": "Alex", "lang": "e", "gender": "male"},
    {"id": "em_santa", "name": "Santa (ES)", "lang": "e", "gender": "male"},
    # French
    {"id": "ff_siwis", "name": "Siwis", "lang": "f", "gender": "female"},
    # Hindi
    {"id": "hf_alpha", "name": "Alpha", "lang": "h", "gender": "female"},
    {"id": "hf_beta", "name": "Beta", "lang": "h", "gender": "female"},
    {"id": "hm_omega", "name": "Omega", "lang": "h", "gender": "male"},
    {"id": "hm_psi", "name": "Psi", "lang": "h", "gender": "male"},
    # Italian
    {"id": "if_sara", "name": "Sara", "lang": "i", "gender": "female"},
    {"id": "im_nicola", "name": "Nicola", "lang": "i", "gender": "male"},
    # Japanese
    {"id": "jf_alpha", "name": "Alpha (JA)", "lang": "j", "gender": "female"},
    {"id": "jf_gongitsune", "name": "Gongitsune", "lang": "j", "gender": "female"},
    {"id": "jf_nezumi", "name": "Nezumi", "lang": "j", "gender": "female"},
    {"id": "jf_tebukuro", "name": "Tebukuro", "lang": "j", "gender": "female"},
    {"id": "jm_kumo", "name": "Kumo", "lang": "j", "gender": "male"},
    # Portuguese
    {"id": "pf_dora", "name": "Dora (PT)", "lang": "p", "gender": "female"},
    # Mandarin
    {"id": "zf_xiaobei", "name": "Xiaobei", "lang": "z", "gender": "female"},
    {"id": "zf_xiaoni", "name": "Xiaoni", "lang": "z", "gender": "female"},
    {"id": "zf_xiaoxiao", "name": "Xiaoxiao", "lang": "z", "gender": "female"},
    {"id": "zf_xiaoyi", "name": "Xiaoyi", "lang": "z", "gender": "female"},
    {"id": "zm_yunjian", "name": "Yunjian", "lang": "z", "gender": "male"},
    {"id": "zm_yunxi", "name": "Yunxi", "lang": "z", "gender": "male"},
    {"id": "zm_yunxia", "name": "Yunxia", "lang": "z", "gender": "male"},
    {"id": "zm_yunyang", "name": "Yunyang", "lang": "z", "gender": "male"},
]


def lang_code_from_voice_id(voice_id: str) -> str:
    """Derive Kokoro lang_code from voice ID prefix.
    
    Voice IDs follow the pattern: {lang_char}{gender_char}_{name}
    The first character maps to the language.
    """
    if voice_id and len(voice_id) >= 2:
        prefix = voice_id[0]
        if prefix in VOICE_PREFIX_TO_LANG:
            return VOICE_PREFIX_TO_LANG[prefix]
    return "a"  # Default to American English


def _discover_voices_from_package() -> list[VoiceInfo] | None:
    """Try to discover voices from the kokoro package at runtime.
    
    Scans the HuggingFace cache for downloaded voice .pt files.
    Returns None if discovery fails.
    """
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if "Kokoro" in repo.repo_id:
                voices = []
                for revision in repo.revisions:
                    for f in revision.files:
                        if f.file_path.name.endswith(".pt") and "/voices/" in str(f.file_path):
                            voice_id = f.file_path.stem
                            lang = lang_code_from_voice_id(voice_id)
                            language = LANG_CODE_TO_LANGUAGE.get(lang, "en-us")
                            gender = "female" if len(voice_id) >= 2 and voice_id[1] == "f" else "male"
                            name = voice_id.split("_", 1)[1].title() if "_" in voice_id else voice_id
                            voices.append(VoiceInfo(
                                id=voice_id, name=name,
                                language=language, gender=gender,
                            ))
                if voices:
                    return sorted(voices, key=lambda v: v.id)
    except Exception as e:
        logger.debug("Voice auto-discovery failed: %s", e)
    return None


class KokoroBackend:
    """TTS backend using the Kokoro-82M model via the kokoro package."""

    name: str = "kokoro"
    sample_rate: int = 24000

    def __init__(self, device: str = "auto") -> None:
        self._device = device
        self._pipeline = None  # Lazy-loaded
        self._current_lang_code: str = "a"
        self._loaded_at: float | None = None
        self._last_used: float | None = None
        self._model_id: str | None = None

    def _get_device(self) -> str:
        if self._device != "auto":
            return self._device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _ensure_loaded(self, model_id: str = "kokoro", lang_code: str = "a") -> None:
        """Lazy-load the pipeline on first use, or reload if lang changes."""
        if self._pipeline is not None and self._current_lang_code == lang_code:
            return

        if self._pipeline is not None:
            logger.info("Switching Kokoro lang_code from '%s' to '%s'",
                        self._current_lang_code, lang_code)

        logger.info("Loading Kokoro model (device=%s, lang=%s)...",
                     self._get_device(), lang_code)
        start = time.time()
        from kokoro import KPipeline
        self._pipeline = KPipeline(lang_code=lang_code, device=self._get_device())
        self._current_lang_code = lang_code
        self._model_id = model_id
        self._loaded_at = time.time()
        elapsed = time.time() - start
        logger.info("Kokoro model loaded in %.1fs", elapsed)

    def load_model(self, model_id: str) -> None:
        self._ensure_loaded(model_id)

    def unload_model(self, model_id: str) -> None:
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
            self._model_id = None
            self._loaded_at = None
            self._last_used = None
            self._current_lang_code = "a"
            logger.info("Kokoro model unloaded")

    def is_model_loaded(self, model_id: str) -> bool:
        return self._pipeline is not None

    def loaded_models(self) -> list[TTSLoadedModelInfo]:
        if self._pipeline is None:
            return []
        return [TTSLoadedModelInfo(
            model=self._model_id or "kokoro",
            backend=self.name,
            device=self._get_device(),
            loaded_at=self._loaded_at or 0,
            last_used_at=self._last_used,
        )]

    def synthesize(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        lang_code: str | None = None,
    ) -> Iterator[np.ndarray]:
        """Generate audio chunks from text.
        
        Yields numpy float32 arrays at 24kHz.
        """
        spec = parse_voice_spec(voice)

        # Derive lang code from voice ID if not explicitly provided
        derived_lang = lang_code or lang_code_from_voice_id(spec.primary_id)
        self._ensure_loaded(lang_code=derived_lang)
        self._last_used = time.time()

        if spec.is_blend:
            voice_tensor = self._blend_voices(spec)
        else:
            voice_tensor = spec.primary_id

        pipeline = self._pipeline

        for _gs, _ps, audio in pipeline(text, voice=voice_tensor, speed=speed):
            if audio is not None and len(audio) > 0:
                yield audio

    def _blend_voices(self, spec):
        """Blend multiple voice tensors according to weights.
        
        Uses KPipeline.load_voice() which loads voice .pt files from HuggingFace
        and returns torch.FloatTensor voice packs.
        """
        import torch

        weights = spec.normalized_weights()
        tensors = []
        for comp in spec.components:
            # KPipeline.load_voice() loads and caches voice tensors
            t = self._pipeline.load_voice(comp.voice_id)
            tensors.append(t)

        # Weighted average
        result = torch.zeros_like(tensors[0])
        for w, t in zip(weights, tensors):
            result += w * t
        return result

    def list_voices(self) -> list[VoiceInfo]:
        """List available Kokoro voices.
        
        First tries to auto-discover from downloaded voice files.
        Falls back to comprehensive static list of all known voices.
        """
        # Try runtime discovery first
        discovered = _discover_voices_from_package()
        if discovered:
            return discovered

        # Fallback: comprehensive static list
        return [
            VoiceInfo(
                id=v["id"],
                name=v["name"],
                language=LANG_CODE_TO_LANGUAGE.get(v["lang"], "en-us"),
                gender=v["gender"],
            )
            for v in ALL_KOKORO_VOICES
        ]
