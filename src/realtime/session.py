"""Realtime session state and configuration."""

from __future__ import annotations

import copy
import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from src.assistant import presets as assistant_presets
from src.assistant.presets import EffectiveSettings

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


def has_lone_surrogate(text: str) -> bool:
    """Whether ``text`` holds a UTF-16 surrogate code point: half of a pair, such as half an emoji (the JSON escape
    ``"\\ud83d"``), which JavaScript's ``slice`` can leave. UTF-8 can't encode one (RFC 3629, section 3), so such
    text can be neither sent to the AI server nor sent back to the client."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return True
    return False


def _lone_surrogate_message(what: str) -> str:
    return (
        f"{what} has a lone surrogate (half of a UTF-16 pair, such as half an emoji), which can't be sent "
        "back; send whole characters"
    )


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
    # As in OpenAI's API, true means "reply automatically after each turn": the transcript goes to the AI
    # assistant and its reply is spoken. It defaults to whether the assistant is on: without one,
    # SessionConfig keeps it false even when a client asks for true, which is the session's honest state.
    create_response: bool = True
    interrupt_response: bool = True  # stop a reply when the user talks over it


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
    # OpenAI's name for the session's system prompt; with the AI assistant on it overrides the preset's.
    instructions: str | None = None
    # The AI preset this session's turns use, and its overrides (the preset fields' names).
    assistant_preset: str = assistant_presets.BUILTIN_IDS[0]
    assistant_overrides: dict[str, Any] = field(default_factory=dict)
    # Whether this connection can reply with the AI assistant. Set by the server, never by a client.
    assistant_on: bool = False

    def __post_init__(self) -> None:
        if self.turn_detection is not None and not self.assistant_on:
            self.turn_detection.create_response = False

    def assistant_settings(self) -> EffectiveSettings:
        """The AI settings for this session's next turn: the preset, its overrides, then ``instructions``.

        Raises ``src.assistant.presets.PresetError`` (a ValueError) for a value that isn't valid.
        """
        overrides = dict(self.assistant_overrides)
        if self.instructions:
            overrides["system_prompt"] = self.instructions
        return assistant_presets.effective(self.assistant_preset, overrides, purpose="voice")

    def assistant_dict(self) -> dict[str, Any] | None:
        """The preset id and the overrides (never a server-only setting), or None with no assistant."""
        if not self.assistant_on:
            return None
        return {"preset": self.assistant_preset, **copy.deepcopy(self.assistant_overrides)}

    def to_dict(self) -> dict[str, Any]:
        td = None
        if self.turn_detection:
            td = {
                "type": self.turn_detection.type,
                "threshold": self.turn_detection.threshold,
                "prefix_padding_ms": self.turn_detection.prefix_padding_ms,
                "silence_duration_ms": self.turn_detection.silence_duration_ms,
                "create_response": self.turn_detection.create_response,
                "interrupt_response": self.turn_detection.interrupt_response,
            }
        if self.ga:
            return {
                "id": self.id,
                "type": self.type,
                "model": self.model,
                "output_modalities": ["audio"],
                "instructions": self.instructions,
                "assistant": self.assistant_dict(),
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
            "instructions": self.instructions,
            "assistant": self.assistant_dict(),
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

        # The voice, the model and the transcription settings are echoed in session.updated, which couldn't be sent with
        # a lone surrogate in them.
        if "model" in session and session["model"]:
            self.model = str(session["model"])
            if has_lone_surrogate(self.model):
                raise ValueError(_lone_surrogate_message("The model"))
        if "voice" in session:
            self.voice = parse_voice(session["voice"])
            if has_lone_surrogate(self.voice):
                raise ValueError(_lone_surrogate_message("The voice"))
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
            if transcription is not None and has_lone_surrogate(json.dumps(transcription, ensure_ascii=False)):
                raise ValueError(_lone_surrogate_message("The transcription settings (such as its prompt)"))
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
                    self.turn_detection = TurnDetectionConfig(create_response=self.assistant_on)
                if "type" in td:
                    self.turn_detection.type = td["type"]
                if "threshold" in td:
                    self.turn_detection.threshold = float(td["threshold"])
                if "prefix_padding_ms" in td:
                    self.turn_detection.prefix_padding_ms = int(td["prefix_padding_ms"])
                if "silence_duration_ms" in td:
                    self.turn_detection.silence_duration_ms = int(td["silence_duration_ms"])
                if "create_response" in td:
                    # Without the AI assistant nothing replies, so the session reports false.
                    self.turn_detection.create_response = bool(td["create_response"]) and self.assistant_on
                if "interrupt_response" in td:
                    self.turn_detection.interrupt_response = bool(td["interrupt_response"])
                if not math.isfinite(self.turn_detection.threshold) or not 0 <= self.turn_detection.threshold <= 1:
                    raise ValueError("VAD threshold must be between 0 and 1")
                if self.turn_detection.prefix_padding_ms < 0 or self.turn_detection.silence_duration_ms < 0:
                    raise ValueError("VAD durations must not be negative")

        if "instructions" in session:
            instructions = session["instructions"]
            if instructions is not None and not isinstance(instructions, str):
                raise ValueError("Instructions must be a string or null")
            if instructions is not None and has_lone_surrogate(instructions):
                # It could be neither sent to the AI nor echoed in session.updated.
                raise ValueError(
                    "Instructions have a lone surrogate (half of a UTF-16 pair, such as half an emoji), which "
                    "can't be sent to the AI; send whole characters"
                )
            self.instructions = instructions if instructions and instructions.strip() else None
        if "assistant" in session:
            self._apply_assistant(session["assistant"])
        if self.assistant_on and ("instructions" in session or "assistant" in session):
            try:
                self.assistant_settings()  # a bad value refuses the whole update
            except assistant_presets.PresetError as exc:
                raise ValueError(f"Invalid AI assistant setting: {exc.message}") from None

    def _apply_assistant(self, block: Any) -> None:
        """``assistant``: ``preset`` picks the AI preset; every other key overrides that preset field, and a
        null removes the override; its ``assistant`` key picks a saved AI assistant by id or name. A text is short
        for ``{"assistant": text}``. ``assistant: null`` goes back to the voice preset with no overrides."""
        if isinstance(block, str):
            block = {assistant_presets.ASSISTANT_FIELD: block}
        if block is not None and not isinstance(block, dict):
            raise ValueError("assistant must be an object, a saved AI assistant's id or name, or null")
        if not self.assistant_on:
            return  # nothing would use it; an unchecked block isn't kept
        if block is None:
            self.assistant_preset = assistant_presets.BUILTIN_IDS[0]
            self.assistant_overrides = {}
            return
        block = dict(block)
        if "preset" in block:
            preset = block.pop("preset")
            self.assistant_preset = assistant_presets.BUILTIN_IDS[0] if preset is None else preset
        for name, value in block.items():
            if value is None:
                self.assistant_overrides.pop(name, None)
            else:
                self.assistant_overrides[name] = copy.deepcopy(value)

    @property
    def turn_detection_dict(self) -> dict[str, Any] | None:
        return asdict(self.turn_detection) if self.turn_detection else None

    @property
    def vad_enabled(self) -> bool:
        return (
            self.turn_detection is not None
            and self.turn_detection.type == "server_vad"
        )
