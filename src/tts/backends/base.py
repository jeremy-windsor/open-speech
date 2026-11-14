"""Base protocol for TTS backends."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterator, Protocol, runtime_checkable

import numpy as np

# A worker's GPU memory figure past this (MiB) is not a size.
MAX_GPU_MEMORY_MB = 64 * 1024


@dataclass
class VoiceInfo:
    """Metadata about an available voice."""
    id: str
    name: str
    language: str = "en-us"
    gender: str = "unknown"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VoiceInfo":
        if not isinstance(payload, dict):
            raise ValueError("Voice manifest entries must be objects")
        voice_id = payload.get("id")
        if not isinstance(voice_id, str) or not voice_id.strip():
            raise ValueError("Voice manifest entries require a non-empty id")
        return cls(
            id=voice_id,
            name=str(payload.get("name") or voice_id),
            language=str(payload.get("language") or "en-us"),
            gender=str(payload.get("gender") or "unknown"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "name": self.name,
            "language": self.language,
            "gender": self.gender,
        }


@dataclass(frozen=True)
class TTSModelManifest:
    """Model-specific controls exposed by one TTS provider."""

    id: str
    provider: str
    sample_rate: int
    capabilities: dict[str, Any]
    voices: tuple[VoiceInfo, ...] = ()
    max_input_chars: int | None = None
    revision: str = "1"
    # MiB the model holds once loaded (optional); None: the registry's gpu_mb.
    gpu_memory_mb: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TTSModelManifest":
        model_id = payload.get("id")
        provider = payload.get("provider")
        sample_rate = payload.get("sample_rate")
        capabilities = payload.get("capabilities")
        voices = payload.get("voices", [])
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("Model manifest requires a non-empty id")
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("Model manifest requires a non-empty provider")
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError("Model manifest requires a positive integer sample_rate")
        if not isinstance(capabilities, dict):
            raise ValueError("Model manifest requires a capabilities object")
        if not isinstance(voices, list):
            raise ValueError("Model manifest voices must be a list")
        max_input_chars = payload.get("max_input_chars")
        if max_input_chars is not None and (
            not isinstance(max_input_chars, int) or max_input_chars <= 0
        ):
            raise ValueError("Model manifest max_input_chars must be a positive integer")
        gpu_memory_mb = payload.get("gpu_memory_mb")
        if (isinstance(gpu_memory_mb, bool) or not isinstance(gpu_memory_mb, int)
                or not 0 < gpu_memory_mb <= MAX_GPU_MEMORY_MB):
            gpu_memory_mb = None
        return cls(
            id=model_id,
            provider=provider,
            sample_rate=sample_rate,
            capabilities=dict(capabilities),
            voices=tuple(VoiceInfo.from_dict(item) for item in voices),
            max_input_chars=max_input_chars,
            revision=str(payload.get("revision") or "1"),
            gpu_memory_mb=gpu_memory_mb,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "provider": self.provider,
            "sample_rate": self.sample_rate,
            "capabilities": dict(self.capabilities),
            "voices": [voice.to_dict() for voice in self.voices],
            "revision": self.revision,
        }
        if self.max_input_chars is not None:
            payload["max_input_chars"] = self.max_input_chars
        if self.gpu_memory_mb is not None:
            payload["gpu_memory_mb"] = self.gpu_memory_mb
        return payload


@dataclass
class TTSLoadedModelInfo:
    """Info about a loaded TTS model."""
    model: str
    backend: str
    device: str
    loaded_at: float
    last_used_at: float | None = None
    # MiB a worker says its model holds; None for an in-process model.
    gpu_memory_mb: float | None = None




DEFAULT_TTS_CAPABILITIES: dict[str, Any] = {
    "voice_blend": False,
    "voice_design": False,
    "voice_clone": False,
    "streaming": False,
    "instructions": False,
    "speakers": [],
    "languages": ["en"],
    "speed_control": True,
    "ssml": False,
    "batch": False,
}

_LANGUAGE_CODE = re.compile(r"[a-z]{2,3}")

# Language names a manifest may list instead of codes (Qwen3's, providers/qwen3/app.py), with
# their ISO 639-1 codes. "Auto" has no code: it admits the hint auto, not every hint.
_LANGUAGE_NAME_CODES: dict[str, str | None] = {
    "auto": None,
    "chinese": "zh",
    "english": "en",
    "japanese": "ja",
    "korean": "ko",
    "german": "de",
    "french": "fr",
    "russian": "ru",
    "portuguese": "pt",
    "spanish": "es",
    "italian": "it",
}


def _primary_subtag(tag: str) -> str:
    """A language tag's primary subtag: en-US, EN_us and en all give en."""
    return tag.strip().lower().replace("_", "-").split("-", 1)[0]


def check_declared_language(language: str | None, languages: Any, model_id: str) -> None:
    """Reject a language hint whose primary subtag is not among the model's declared languages.

    A regional tag counts by its primary subtag, in any case (en-GB and EN_us are en). A list may
    name languages instead of giving codes (Qwen3's "Auto", "English", ...). A listed name is
    then accepted as a hint itself, in any case, and counts as its code, so English admits en-US
    and Auto admits auto. Any hint is accepted when the model declares no languages, or when its
    list has an entry that is neither a code nor one of those names: that model's worker checks
    the hint itself.
    """
    if not language or not language.strip():
        return
    if not isinstance(languages, (list, tuple)) or not languages:
        return
    declared: set[str] = set()
    for entry in languages:
        if not isinstance(entry, str):
            return
        primary = _primary_subtag(entry)
        name = entry.strip().lower()
        if _LANGUAGE_CODE.fullmatch(primary):
            declared.add(primary)
        elif name in _LANGUAGE_NAME_CODES:
            declared.add(name)
            code = _LANGUAGE_NAME_CODES[name]
            if code is not None:
                declared.add(code)
        else:
            return
    if _primary_subtag(language) not in declared:
        raise ValueError(
            f"Unsupported language '{language}' for {model_id}. "
            f"Supported: {', '.join(languages)}"
        )


@runtime_checkable
class TTSBackend(Protocol):
    """Protocol that all TTS backends must implement.

    The router runs load_model, unload_model and synthesize for one backend one at a time,
    but is_model_loaded, loaded_models and list_voices run concurrently with them, so a
    backend snapshots its loaded state for those reads under a small lock of its own.

    Optional hooks, which callers look up with getattr:

    download_model(model_id) -> None
        Fetch the model's files into the cache load_model reads, without loading it, so a
        download leaves the loaded models as they are. ModelManager calls it outside the
        router's locks, while the backend may be synthesizing, so it must not touch the
        loaded state. Without it, a download loads the model once and unloads it again.

    validate_language(language) -> None
        Raise ValueError for a language hint the model cannot speak, without loading it. A
        backend that sets requires_model_id gets validate_language(language, model_id).
        Without it, any hint is accepted.
    """

    name: str
    sample_rate: int
    capabilities: dict[str, Any]

    @classmethod
    def is_available(cls) -> bool:
        """Return True if this backend's required packages are installed."""
        return True

    def load_model(self, model_id: str) -> None: ...
    def unload_model(self, model_id: str) -> None: ...
    def is_model_loaded(self, model_id: str) -> bool: ...
    def loaded_models(self) -> list[TTSLoadedModelInfo]: ...

    def synthesize(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        lang_code: str | None = None,
        **backend_options: Any,
    ) -> Iterator[np.ndarray]:
        """Generate audio chunks as float32 numpy arrays at native sample rate.

        Provider-specific options are passed only after capability validation.
        Yields chunks (typically per-sentence) for streaming support.
        """
        ...

    def list_voices(self) -> list[VoiceInfo]: ...
