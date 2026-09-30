"""Realtime session state and configuration."""

from __future__ import annotations

import copy
import math
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

VALID_AUDIO_FORMATS = {"pcm16", "g711_ulaw", "g711_alaw"}

# Sample rates per format
FORMAT_SAMPLE_RATES = {
    "pcm16": 24000,
    "g711_ulaw": 8000,
    "g711_alaw": 8000,
}


def parse_audio_format(value: Any) -> str:
    if not isinstance(value, dict):
        raise TypeError("Audio format must be an object")
    formats = {"audio/pcm": "pcm16", "audio/pcmu": "g711_ulaw", "audio/pcma": "g711_alaw"}
    fmt = formats.get(value.get("type"))
    if fmt is None or (fmt == "pcm16" and value.get("rate", 24000) != 24000):
        raise ValueError("Supported audio formats are PCM at 24000 Hz, PCMU, and PCMA")
    return fmt


def audio_format_object(fmt: str) -> dict[str, Any]:
    if fmt == "pcm16":
        return {"type": "audio/pcm", "rate": 24000}
    return {"type": "audio/pcmu" if fmt == "g711_ulaw" else "audio/pcma"}


def parse_voice(value: Any) -> str:
    if isinstance(value, dict) and set(value) == {"id"}:
        value = value["id"]
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Voice must be a nonempty string or an object with an id")
    return value


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
    ga: bool = False
    type: str = "realtime"
    speed: float = 1.0
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
        if self.ga:
            return {
                "id": self.id,
                "type": self.type,
                "model": self.model,
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": audio_format_object(self.input_audio_format),
                        "transcription": self.input_audio_transcription,
                        "turn_detection": td,
                    },
                    "output": {
                        "format": audio_format_object(self.output_audio_format),
                        "voice": self.voice,
                        "speed": self.speed,
                    },
                },
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
        if not isinstance(session, dict):
            raise TypeError("Session must be an object")
        candidate = copy.deepcopy(self)
        candidate._apply_update(session)
        self.__dict__.update(candidate.__dict__)

    def _apply_update(self, session: dict[str, Any]) -> None:
        session = dict(session)
        if "audio" in session:
            self.ga = True
            audio = session["audio"]
            if not isinstance(audio, dict):
                raise ValueError("Session audio must be an object")
            for direction in ("input", "output"):
                config = audio.get(direction, {})
                if not isinstance(config, dict):
                    raise TypeError(f"Audio {direction} must be an object")
                if "format" in config:
                    session[f"{direction}_audio_format"] = parse_audio_format(config["format"])
                if direction == "input":
                    for nested, flat in (("transcription", "input_audio_transcription"), ("turn_detection", "turn_detection")):
                        if nested in config:
                            session[flat] = config[nested]
                else:
                    for key in ("voice", "speed"):
                        if key in config:
                            session[key] = config[key]
        if "type" in session:
            if session["type"] not in {"realtime", "transcription"}:
                raise ValueError("Unsupported session type")
            self.type = session["type"]
        if "output_modalities" in session and session["output_modalities"] != ["audio"]:
            raise ValueError("Only audio output is supported")
        if "speed" in session:
            speed = float(session["speed"])
            if not math.isfinite(speed) or not 0.25 <= speed <= 1.5:
                raise ValueError("Realtime audio speed must be between 0.25 and 1.5")
            self.speed = speed

        if "model" in session and session["model"]:
            self.model = str(session["model"])
        if "voice" in session:
            self.voice = parse_voice(session["voice"])
        if "input_audio_format" in session:
            fmt = session["input_audio_format"]
            if fmt in VALID_AUDIO_FORMATS:
                self.input_audio_format = fmt
        if "output_audio_format" in session:
            fmt = session["output_audio_format"]
            if fmt in VALID_AUDIO_FORMATS:
                self.output_audio_format = fmt
        if "input_audio_transcription" in session:
            transcription = session["input_audio_transcription"]
            if transcription is not None and not isinstance(transcription, dict):
                raise ValueError("Transcription must be an object or null")
            self.input_audio_transcription = session["input_audio_transcription"]

        if "turn_detection" in session:
            td = session["turn_detection"]
            if td is None:
                self.turn_detection = None  # type: ignore[assignment]
            else:
                if not isinstance(td, dict):
                    raise ValueError("Turn detection must be an object or null")
                if td.get("type", "server_vad") != "server_vad":
                    raise ValueError("Only server_vad turn detection is supported")
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
                if not math.isfinite(self.turn_detection.threshold) or not 0 <= self.turn_detection.threshold <= 1:
                    raise ValueError("VAD threshold must be between 0 and 1")
                if self.turn_detection.prefix_padding_ms < 0 or self.turn_detection.silence_duration_ms < 0:
                    raise ValueError("VAD durations must not be negative")

    @property
    def turn_detection_dict(self) -> dict[str, Any] | None:
        return asdict(self.turn_detection) if self.turn_detection else None

    @property
    def vad_enabled(self) -> bool:
        return (
            self.turn_detection is not None
            and self.turn_detection.type == "server_vad"
        )
