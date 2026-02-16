"""Voice registry, presets, and blend parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class VoiceComponent:
    """A single voice with its weight in a blend."""
    voice_id: str
    weight: float = 1.0


@dataclass
class VoiceSpec:
    """Parsed voice specification — one or more weighted voices."""
    components: list[VoiceComponent]

    @property
    def is_blend(self) -> bool:
        return len(self.components) > 1

    @property
    def primary_id(self) -> str:
        return self.components[0].voice_id

    def normalized_weights(self) -> list[float]:
        total = sum(c.weight for c in self.components)
        if total == 0:
            return [1.0 / len(self.components)] * len(self.components)
        return [c.weight / total for c in self.components]


# OpenAI voice name → Kokoro voice ID
OPENAI_VOICE_MAP: dict[str, str] = {
    "alloy": "af_heart",
    "echo": "am_adam",
    "fable": "bf_emma",
    "onyx": "am_michael",
    "nova": "af_nova",
    "shimmer": "af_bella",
}

# Pattern: voice_id or voice_id(weight)
_COMPONENT_RE = re.compile(r"([a-zA-Z0-9_]+)(?:\((\d+(?:\.\d+)?)\))?")


def resolve_voice_name(voice: str) -> str:
    """Resolve OpenAI voice aliases to backend voice IDs.
    
    Returns the input unchanged if not an alias.
    """
    return OPENAI_VOICE_MAP.get(voice, voice)


def parse_voice_spec(voice: str) -> VoiceSpec:
    """Parse a voice string like 'af_bella(2)+af_sky(1)' into a VoiceSpec.
    
    Supports:
    - Single voice: 'af_bella'
    - OpenAI alias: 'alloy'
    - Blend: 'af_bella+af_sky' (equal weights)
    - Weighted blend: 'af_bella(2)+af_sky(1)'
    """
    # Resolve OpenAI aliases first (only for non-blend single names)
    if "+" not in voice and "(" not in voice:
        voice = resolve_voice_name(voice)

    parts = voice.split("+")
    components = []
    for part in parts:
        part = part.strip()
        m = _COMPONENT_RE.fullmatch(part)
        if not m:
            raise ValueError(f"Invalid voice spec component: {part!r}")
        voice_id = m.group(1)
        weight = float(m.group(2)) if m.group(2) else 1.0
        components.append(VoiceComponent(voice_id=voice_id, weight=weight))

    return VoiceSpec(components=components)
