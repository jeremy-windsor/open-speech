"""Tests for the repeatable TTS benchmark helper."""

from __future__ import annotations

import base64
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from scripts import benchmark_tts


class _FakeWebSocket:
    def __init__(self, events: list[dict]) -> None:
        self._events = iter(events)
        self.sent: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def recv(self, timeout: float):
        del timeout
        return json.dumps(next(self._events))

    def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))


def test_live_benchmark_collects_every_committed_segment(monkeypatch, tmp_path):
    first_pcm = b"\x01\x00\x02\x00"
    second_pcm = b"\x03\x00\x04\x00\x05\x00"
    websocket = _FakeWebSocket(
        [
            {"type": "session.created"},
            {"type": "session.updated"},
            {
                "type": "response.output_audio.delta",
                "response_id": "response-1",
                "sequence": 0,
                "sample_rate": 24000,
                "delta": base64.b64encode(first_pcm).decode(),
            },
            {
                "type": "response.done",
                "response": {"id": "response-1", "status": "completed"},
            },
            {
                "type": "response.output_audio.delta",
                "response_id": "response-2",
                "sequence": 0,
                "sample_rate": 24000,
                "delta": base64.b64encode(second_pcm).decode(),
            },
            {
                "type": "response.done",
                "response": {"id": "response-2", "status": "completed"},
            },
            {"type": "input_text.done", "generation": 0},
        ]
    )
    websockets_module = ModuleType("websockets")
    sync_module = ModuleType("websockets.sync")
    client_module = ModuleType("websockets.sync.client")
    client_module.connect = lambda *_args, **_kwargs: websocket
    monkeypatch.setitem(sys.modules, "websockets", websockets_module)
    monkeypatch.setitem(sys.modules, "websockets.sync", sync_module)
    monkeypatch.setitem(sys.modules, "websockets.sync.client", client_module)
    args = SimpleNamespace(
        url="https://localhost:8100",
        api_key="",
        insecure=True,
        model="qwen3/0.6b-custom-voice",
        voice="Ryan",
        speed=1.0,
        language="en",
        text="First sentence. Second sentence.",
        output=str(tmp_path / "live.wav"),
    )

    audio, metrics = benchmark_tts._live(args)

    assert metrics["sample_rate"] == 24000
    assert metrics["audio_duration_s"] == pytest.approx(5 / 24000)
    assert metrics["ttfa_source"] == "live_reader_first_pcm_delta"
    assert audio.startswith(b"RIFF")
    acknowledgements = [
        event for event in websocket.sent if event["type"] == "playback.ack"
    ]
    assert [event["response_id"] for event in acknowledgements] == [
        "response-1",
        "response-2",
    ]


@pytest.mark.parametrize(
    ("terminal_event", "message"),
    [
        (
            {"type": "response.done", "response": {"status": "failed"}},
            "ended with status: failed",
        ),
        ({"type": "response.cancelled"}, "was cancelled"),
    ],
)
def test_live_benchmark_fails_fast_on_unsuccessful_response(
    monkeypatch, tmp_path, terminal_event, message
):
    websocket = _FakeWebSocket(
        [
            {"type": "session.created"},
            {"type": "session.updated"},
            terminal_event,
        ]
    )
    websockets_module = ModuleType("websockets")
    sync_module = ModuleType("websockets.sync")
    client_module = ModuleType("websockets.sync.client")
    client_module.connect = lambda *_args, **_kwargs: websocket
    monkeypatch.setitem(sys.modules, "websockets", websockets_module)
    monkeypatch.setitem(sys.modules, "websockets.sync", sync_module)
    monkeypatch.setitem(sys.modules, "websockets.sync.client", client_module)
    args = SimpleNamespace(
        url="https://localhost:8100",
        api_key="",
        insecure=True,
        model="qwen3/0.6b-custom-voice",
        voice="Ryan",
        speed=1.0,
        language="en",
        text="Test failure handling.",
        output=str(tmp_path / "failed.wav"),
    )

    with pytest.raises(RuntimeError, match=message):
        benchmark_tts._live(args)
