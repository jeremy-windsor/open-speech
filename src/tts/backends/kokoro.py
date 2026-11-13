"""Kokoro TTS backend — uses the kokoro Python package."""

from __future__ import annotations

import gc
import logging
import re
import threading
import time
from typing import Any, Iterator

import numpy as np

from src.tts.backends.base import TTSLoadedModelInfo, VoiceInfo
from src.tts.voices import parse_voice_spec

logging.getLogger("phonemizer").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

VOICE_DISCOVERY_TTL_SECONDS = 60.0
KOKORO_REPO_ID = "hexgrad/Kokoro-82M"
# What KModel and KPipeline fetch from KOKORO_REPO_ID, every voice included, so a download
# leaves no weights to fetch on first use.
_KOKORO_FILES = ("config.json", "kokoro-v1_0.pth", "voices/*.pt")

# kokoro 0.9.4's KPipeline cuts a chunk's phonemes past this many (ps = ps[:510]). For every
# language but English it splits text only at newlines and ASCII .!?, so the backend cuts longer
# text into pieces that fit before it reaches KPipeline.
KPIPELINE_MAX_PHONEMES = 510
_CLOSERS = re.escape("\"'”’»」』）)】]")
# Kana, CJK ideographs (Extension A, unified, compatibility) and half-width katakana.
_CJK = "぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ"
# Where a line that is too long may be cut, tried in order: sentence ends, then clause marks,
# then spaces. A cut keeps the punctuation (and any closing quote) with the text before it.
# An ASCII mark counts before whitespace, or straight before a Han or kana character, as Chinese
# and Japanese are often written ("家门.河边"; UAX #29 SB11), so "3.5" and "1,000" stay whole.
_CUT_POINTS = (
    re.compile(rf"(?:[.!?]+[{_CLOSERS}]*(?=[\s{_CJK}]|$)|[。！？．｡…।॥]+[{_CLOSERS}]*)\s*"),
    re.compile(rf"(?:[,;:]+[{_CLOSERS}]*(?=[\s{_CJK}])|[，、；：﹐﹑､]+)\s*"),
    re.compile(r"\s+"),
)

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

LANGUAGE_TO_LANG_CODE: dict[str, str] = {
    language: lang_code for lang_code, language in LANG_CODE_TO_LANGUAGE.items()
}
LANGUAGE_TO_LANG_CODE["pt"] = "p"
# Plain "en" takes the voice's accent (a or b); see normalize_lang_code.
SUPPORTED_LANGUAGE_TAGS: list[str] = sorted({"en", *LANGUAGE_TO_LANG_CODE})

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


def normalize_lang_code(language: str | None, voice_id: str) -> str:
    """Adapt public language tags to the one-letter codes Kokoro expects.

    A regional tag Kokoro has no entry for falls back to its primary subtag (BCP 47
    lookup), so zh-CN is Mandarin and en-AU is English in the voice's accent. A language
    Kokoro has no pipeline for raises ValueError instead of reaching KPipeline's assert.
    """
    normalized = (language or "").strip().lower().replace("_", "-")
    if not normalized:
        return lang_code_from_voice_id(voice_id)
    if normalized in VOICE_PREFIX_TO_LANG:
        return normalized
    if normalized in LANGUAGE_TO_LANG_CODE:
        return LANGUAGE_TO_LANG_CODE[normalized]
    primary = normalized.split("-", 1)[0]
    if primary == "en":
        voice_lang = lang_code_from_voice_id(voice_id)
        return voice_lang if voice_lang in {"a", "b"} else "a"
    if primary in LANGUAGE_TO_LANG_CODE:
        return LANGUAGE_TO_LANG_CODE[primary]
    raise ValueError(
        f"Unsupported language '{language}' for Kokoro. "
        f"Supported: {', '.join(SUPPORTED_LANGUAGE_TAGS)}"
    )


def _cut_after(pattern: re.Pattern[str], text: str) -> list[str]:
    """Split text after each match of pattern; the parts join back to text."""
    parts: list[str] = []
    start = 0
    for match in pattern.finditer(text):
        if start < match.end() < len(text):
            parts.append(text[start:match.end()])
            start = match.end()
    parts.append(text[start:])
    return parts


def _fit_line(count, text: str, level: int = 0) -> list[str]:
    """Cut text into pieces of at most KPIPELINE_MAX_PHONEMES phonemes, packing parts back together."""
    if count(text) <= KPIPELINE_MAX_PHONEMES:
        return [text]
    if level >= len(_CUT_POINTS):
        # No sentence end, clause mark or space left: cut at a character boundary.
        if len(text) < 2:
            return [text]
        middle = len(text) // 2
        return _fit_line(count, text[:middle], level) + _fit_line(count, text[middle:], level)
    parts = _cut_after(_CUT_POINTS[level], text)
    if len(parts) < 2:
        return _fit_line(count, text, level + 1)
    # Pack by the parts' own counts (plus a space between them), so G2P runs on each part once;
    # each packed piece is counted again as a whole and cut finer if it is still too long.
    pieces: list[str] = []
    current, current_count = "", 0
    for part in parts:
        part_count = count(part)
        if current and current_count + 1 + part_count > KPIPELINE_MAX_PHONEMES:
            pieces.extend(_fit_line(count, current, level + 1))
            current, current_count = part, part_count
        else:
            current += part
            current_count += part_count + (1 if current_count else 0)
    pieces.extend(_fit_line(count, current, level + 1))
    return pieces


def split_for_kpipeline(text: str, g2p) -> list[str]:
    """The pieces of non-English text to hand KPipeline, so that none loses phonemes past 510.

    Lines stay separate, as KPipeline's own newline split keeps them; a line whose phonemes
    (by the pipeline's own G2P) fit is one piece, as before.
    """
    counts: dict[str, int] = {}

    def count(piece: str) -> int:
        piece = piece.strip()
        if piece not in counts:
            result = g2p(piece) if piece else ""
            phonemes = result[0] if isinstance(result, tuple) else result
            counts[piece] = len(phonemes) if isinstance(phonemes, str) else 0
        return counts[piece]

    pieces: list[str] = []
    for line in re.split(r"\n+", text.strip()):
        if line.strip():
            pieces.extend(piece.strip() for piece in _fit_line(count, line) if piece.strip())
    return pieces


def _discover_voices_from_package() -> list[VoiceInfo] | None:
    """Try to discover voices from the kokoro package at runtime.
    
    Scans the HuggingFace cache for downloaded voice .pt files.
    Returns None if discovery fails.
    """
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            # The pipeline only loads voices from its own repo; other Kokoro repos ship
            # voice packs it cannot fetch.
            if repo.repo_id == KOKORO_REPO_ID:
                # Keyed by ID because every cached revision lists its own copy of a voice.
                voices: dict[str, VoiceInfo] = {}
                for revision in repo.revisions:
                    for f in revision.files:
                        if f.file_path.name.endswith(".pt") and f.file_path.parent.name == "voices":
                            voice_id = f.file_path.stem
                            lang = lang_code_from_voice_id(voice_id)
                            language = LANG_CODE_TO_LANGUAGE.get(lang, "en-us")
                            gender = "female" if len(voice_id) >= 2 and voice_id[1] == "f" else "male"
                            name = voice_id.split("_", 1)[1].title() if "_" in voice_id else voice_id
                            voices[voice_id] = VoiceInfo(
                                id=voice_id, name=name,
                                language=language, gender=gender,
                            )
                if voices:
                    return sorted(voices.values(), key=lambda v: v.id)
    except Exception as e:
        logger.debug("Voice auto-discovery failed: %s", e)
    return None


class KokoroBackend:
    """TTS backend using the Kokoro-82M model via the kokoro package."""

    name: str = "kokoro"
    sample_rate: int = 24000
    capabilities: dict = {
        "voice_blend": True,
        "voice_design": False,
        "voice_clone": False,
        "streaming": False,
        "instructions": False,
        "speakers": [
            {"name": v["id"], "description": f"{v['name']} ({'Female' if v['gender']=='female' else 'Male'})", "language": LANG_CODE_TO_LANGUAGE.get(v["lang"], "en-us")}
            for v in ALL_KOKORO_VOICES
        ],
        "languages": sorted({LANG_CODE_TO_LANGUAGE.get(v["lang"], "en-us") for v in ALL_KOKORO_VOICES}),
        "speed_control": True,
        "ssml": "partial",
        "batch": False,
    }

    @classmethod
    def is_available(cls) -> bool:
        try:
            import kokoro  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def supports_model(model_id: str) -> bool:
        return model_id == "kokoro"

    def validate_voice(self, voice: str) -> None:
        spec = parse_voice_spec(voice)
        available = {item["id"] for item in ALL_KOKORO_VOICES}
        unknown = [
            component.voice_id
            for component in spec.components
            if component.voice_id not in available
        ]
        if unknown:
            # Preserve locally cached/custom Kokoro voice packs while still
            # rejecting typos instead of handing them to the model loader.
            now = time.monotonic()
            if now - self._voice_discovery_checked_at >= VOICE_DISCOVERY_TTL_SECONDS:
                discovered = _discover_voices_from_package() or []
                self._discovered_voice_ids = {item.id for item in discovered}
                self._voice_discovery_checked_at = now
            available.update(self._discovered_voice_ids)
            unknown = [
                component.voice_id
                for component in spec.components
                if component.voice_id not in available
            ]
        if unknown:
            raise ValueError(f"Unknown Kokoro voice: {unknown[0]}")

    def validate_language(self, language: str | None) -> None:
        """Reject a language hint Kokoro has no pipeline for, without loading the model."""
        normalize_lang_code(language, "")

    def __init__(self, device: str = "auto") -> None:
        self._device = device
        self._pipeline = None  # Lazy-loaded; the pipeline for _current_lang_code
        # One pipeline per language used so far. They all share the first one's KModel.
        self._pipelines: dict[str, Any] = {}
        self._current_lang_code: str = "a"
        self._loaded_at: float | None = None
        self._last_used: float | None = None
        self._model_id: str | None = None
        self._discovered_voice_ids: set[str] = set()
        self._voice_discovery_checked_at = float("-inf")
        # The router serializes load, unload and synthesis but not status or voice reads, so
        # the loaded state is swapped under this lock and never held during a load.
        self._state_lock = threading.Lock()

    def _get_device(self) -> str:
        if self._device != "auto":
            return self._device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _ensure_loaded(self, model_id: str = "kokoro", lang_code: str = "a") -> None:
        """Lazy-load the model on first use, and each language's pipeline on its first use.

        A language switch reuses the loaded KModel and only builds that language's text
        front end, so moving between af_ and bf_ voices does not reload the weights.
        """
        if self._pipeline is not None and self._current_lang_code == lang_code:
            return

        if self._pipelines:
            pipeline = self._pipelines.get(lang_code)
            if pipeline is None:
                logger.info("Adding Kokoro lang_code '%s' to the loaded model", lang_code)
                from kokoro import KPipeline
                shared_model = next(iter(self._pipelines.values())).model
                pipeline = KPipeline(lang_code=lang_code, repo_id=KOKORO_REPO_ID, model=shared_model)
            with self._state_lock:
                self._pipelines[lang_code] = pipeline
                self._pipeline = pipeline
                self._current_lang_code = lang_code
            return

        logger.info("Loading Kokoro model (device=%s, lang=%s)...",
                     self._get_device(), lang_code)
        start = time.time()
        from kokoro import KPipeline
        pipeline = KPipeline(lang_code=lang_code, device=self._get_device(), repo_id=KOKORO_REPO_ID)
        with self._state_lock:
            self._pipelines = {lang_code: pipeline}
            self._pipeline = pipeline
            self._current_lang_code = lang_code
            self._model_id = model_id
            self._loaded_at = time.time()
        elapsed = time.time() - start
        logger.info("Kokoro model loaded in %.1fs", elapsed)

        # Warmup: run a silent synthesis to pre-compile CUDA kernels
        # This turns 30s first-generate into ~instant on subsequent calls
        if self._get_device() == "cuda":
            try:
                logger.info("Warming up Kokoro CUDA kernels...")
                warmup_start = time.time()
                for _gs, _ps, _audio in self._pipeline(".", voice="af_heart", speed=1.0):
                    break  # consume just one chunk to trigger kernel compilation
                logger.info("CUDA warmup done in %.1fs", time.time() - warmup_start)
            except Exception as e:
                logger.warning("CUDA warmup failed (non-fatal): %s", e)

    def download_model(self, model_id: str) -> None:
        """Fetch the weights and every voice into the Hugging Face cache, without loading them."""
        from huggingface_hub import snapshot_download

        logger.info("Downloading Kokoro weights and voices")
        snapshot_download(repo_id=KOKORO_REPO_ID, allow_patterns=list(_KOKORO_FILES))

    def load_model(self, model_id: str) -> None:
        self._ensure_loaded(model_id)

    def unload_model(self, model_id: str) -> None:
        if self._pipeline is not None:
            with self._state_lock:
                self._pipelines = {}
                self._pipeline = None
                self._model_id = None
                self._loaded_at = None
                self._last_used = None
                self._current_lang_code = "a"
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info("Kokoro model unloaded")

    def is_model_loaded(self, model_id: str) -> bool:
        return self._pipeline is not None

    def loaded_models(self) -> list[TTSLoadedModelInfo]:
        with self._state_lock:
            if self._pipeline is None:
                return []
            model_id, loaded_at, last_used = self._model_id, self._loaded_at, self._last_used
        return [TTSLoadedModelInfo(
            model=model_id or "kokoro",
            backend=self.name,
            device=self._get_device(),
            loaded_at=loaded_at or 0,
            last_used_at=last_used,
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
        self.validate_voice(voice)
        spec = parse_voice_spec(voice)

        # Derive lang code from voice ID, or adapt a public ISO-style hint.
        derived_lang = normalize_lang_code(lang_code, spec.primary_id)
        self._ensure_loaded(lang_code=derived_lang)
        self._last_used = time.time()

        if spec.is_blend:
            voice_tensor = self._blend_voices(spec)
        else:
            voice_tensor = spec.primary_id

        pipeline = self._pipeline
        # English chunks at 510 phonemes inside KPipeline; other languages are cut here first
        # (kokoro 0.9.4 would drop every phoneme of a chunk past the 510th).
        pieces = text if derived_lang in ("a", "b") else split_for_kpipeline(text, pipeline.g2p)

        for _gs, _ps, audio in pipeline(pieces, voice=voice_tensor, speed=speed):
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
        # Try runtime discovery first — but only use it if it finds
        # every voice in our static list (otherwise discovery only
        # returned the subset of downloaded voices)
        discovered = _discover_voices_from_package()
        if discovered and {v["id"] for v in ALL_KOKORO_VOICES} <= {v.id for v in discovered}:
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
