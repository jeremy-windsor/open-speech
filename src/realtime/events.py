"""OpenAI Realtime API event type definitions and serialization.

Reference: https://platform.openai.com/docs/api-reference/realtime
"""

from __future__ import annotations

import uuid
from typing import Any


def _event_id() -> str:
    return f"evt_{uuid.uuid4().hex[:24]}"


def _item_id() -> str:
    return f"item_{uuid.uuid4().hex[:20]}"


def _response_id() -> str:
    return f"resp_{uuid.uuid4().hex[:20]}"


# ---------------------------------------------------------------------------
# Server → Client events
# ---------------------------------------------------------------------------


def session_created(session: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": _event_id(),
        "type": "session.created",
        "session": session,
    }


def session_updated(session: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": _event_id(),
        "type": "session.updated",
        "session": session,
    }


def error(message: str, error_type: str = "invalid_request_error", code: str | None = None,
          event_id: str | None = None) -> dict[str, Any]:
    err: dict[str, Any] = {"type": error_type, "message": message}
    if code:
        err["code"] = code
    if event_id:
        err["event_id"] = event_id
    return {
        "event_id": _event_id(),
        "type": "error",
        "error": err,
    }


def input_audio_buffer_speech_started(audio_start_ms: int, item_id: str) -> dict[str, Any]:
    return {
        "event_id": _event_id(),
        "type": "input_audio_buffer.speech_started",
        "audio_start_ms": audio_start_ms,
        "item_id": item_id,
    }


def input_audio_buffer_speech_stopped(audio_end_ms: int, item_id: str) -> dict[str, Any]:
    return {
        "event_id": _event_id(),
        "type": "input_audio_buffer.speech_stopped",
        "audio_end_ms": audio_end_ms,
        "item_id": item_id,
    }


def input_audio_buffer_committed(item_id: str, previous_item_id: str | None = None) -> dict[str, Any]:
    return {
        "event_id": _event_id(),
        "type": "input_audio_buffer.committed",
        "previous_item_id": previous_item_id,
        "item_id": item_id,
    }


def input_audio_buffer_cleared() -> dict[str, Any]:
    return {
        "event_id": _event_id(),
        "type": "input_audio_buffer.cleared",
    }


def conversation_item_created(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": _event_id(),
        "type": "conversation.item.created",
        "previous_item_id": None,
        "item": item,
    }


def conversation_item_input_audio_transcription_completed(
    item_id: str, content_index: int, transcript: str
) -> dict[str, Any]:
    return {
        "event_id": _event_id(),
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": item_id,
        "content_index": content_index,
        "transcript": transcript,
    }


def response_created(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": _event_id(),
        "type": "response.created",
        "response": response,
    }


def response_audio_delta(response_id: str, item_id: str, output_index: int,
                          content_index: int, delta: str) -> dict[str, Any]:
    return {
        "event_id": _event_id(),
        "type": "response.audio.delta",
        "response_id": response_id,
        "item_id": item_id,
        "output_index": output_index,
        "content_index": content_index,
        "delta": delta,
    }


def response_audio_done(response_id: str, item_id: str, output_index: int,
                         content_index: int) -> dict[str, Any]:
    return {
        "event_id": _event_id(),
        "type": "response.audio.done",
        "response_id": response_id,
        "item_id": item_id,
        "output_index": output_index,
        "content_index": content_index,
    }


def response_done(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": _event_id(),
        "type": "response.done",
        "response": response,
    }
