"""Realtime session state and configuration."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


VALID_AUDIO_FORMATS = {"pcm16", "g711_ulaw", "g711_alaw"}

# Sample rates per format
FORMAT_SAMPLE_RATES = {
    "pcm16": 24000,
    "g711_ulaw": 8000,
    "g711_alaw": 8000,
}


@dataclass
class TurnDetectionConfig:
    type: str = "server_vad"
    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500
    create_response: bool = False  # We don't generate LLM responses


@dataclass
class SessionConfig:
    """Realtime session configuration."""
    id: str = field(default_factory=lambda: f"sess_{uuid.uuid4().hex[:24]}")
    model: str = ""
    voice: str = "alloy"
    input_audio_format: str = "pcm16"
    output_audio_format: str = "pcm16"
    input_audio_transcription: dict[str, Any] | None = field(default_factory=lambda: {"model": "whisper-1"})
    turn_detection: TurnDetectionConfig = field(default_factory=TurnDetectionConfig)

    def to_dict(self) -> dict[str, Any]:
        td = None
        if self.turn_detection:
            td = {
                "type": self.turn_detection.type,
                "threshold": self.turn_detection.threshold,
                "prefix_padding_ms": self.turn_detection.prefix_padding_ms,
                "silence_duration_ms": self.turn_detection.silence_duration_ms,
                "create_response": self.turn_detection.create_response,
            }
        return {
            "id": self.id,
            "object": "realtime.session",
            "model": self.model,
            "voice": self.voice,
            "input_audio_format": self.input_audio_format,
            "output_audio_format": self.output_audio_format,
            "input_audio_transcription": self.input_audio_transcription,
            "turn_detection": td,
            "modalities": ["audio", "text"],
        }

    def update_from(self, data: dict[str, Any]) -> None:
        """Update session config from a session.update event payload."""
        session = data.get("session", data)

        if "model" in session and session["model"]:
            self.model = str(session["model"])
        if "voice" in session:
            self.voice = session["voice"]
        if "input_audio_format" in session:
            fmt = session["input_audio_format"]
            if fmt in VALID_AUDIO_FORMATS:
                self.input_audio_format = fmt
        if "output_audio_format" in session:
            fmt = session["output_audio_format"]
            if fmt in VALID_AUDIO_FORMATS:
                self.output_audio_format = fmt
        if "input_audio_transcription" in session:
            self.input_audio_transcription = session["input_audio_transcription"]

        if "turn_detection" in session:
            td = session["turn_detection"]
            if td is None:
                self.turn_detection = None  # type: ignore[assignment]
            else:
                if self.turn_detection is None:
                    self.turn_detection = TurnDetectionConfig()
                if "type" in td:
                    self.turn_detection.type = td["type"]
                if "threshold" in td:
                    self.turn_detection.threshold = float(td["threshold"])
                if "prefix_padding_ms" in td:
                    self.turn_detection.prefix_padding_ms = int(td["prefix_padding_ms"])
                if "silence_duration_ms" in td:
                    self.turn_detection.silence_duration_ms = int(td["silence_duration_ms"])
                if "create_response" in td:
                    self.turn_detection.create_response = bool(td["create_response"])

    @property
    def vad_enabled(self) -> bool:
        return (
            self.turn_detection is not None
            and self.turn_detection.type == "server_vad"
        )
