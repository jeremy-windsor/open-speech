"""Routing regression for the distilled English long-form STT path."""

from __future__ import annotations

import io
import sys
import wave
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.backends.faster_whisper import FasterWhisperBackend


def _silent_wav(seconds: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\0\0" * 16000 * seconds)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("model_id", "duration", "use_chunks"),
    [
        ("Systran/faster-distil-whisper-small.en", 31, True),
        ("Systran/faster-distil-whisper-medium.en", 31, True),
        ("Systran/faster-distil-whisper-medium.en", 30, False),
        ("Systran/faster-distil-whisper-large-v3", 31, False),
    ],
)
def test_only_affected_long_recordings_use_global_timestamp_chunks(
    model_id, duration, use_chunks
):
    segment = SimpleNamespace(
        seek=0, start=duration - 2, end=duration - 1, text=" Final words.",
        tokens=[], temperature=0.0, avg_logprob=-0.1,
        compression_ratio=1.0, no_speech_prob=0.0,
    )
    result = ([segment], SimpleNamespace(language="en", duration=duration))
    model = MagicMock()
    model.transcribe.return_value = result
    pipeline = MagicMock()
    pipeline.transcribe.return_value = result
    faster_whisper = ModuleType("faster_whisper")
    faster_whisper.BatchedInferencePipeline = MagicMock(return_value=pipeline)
    backend = FasterWhisperBackend()
    backend._models[model_id] = model

    with patch.dict(sys.modules, {"faster_whisper": faster_whisper}):
        response = backend.transcribe(
            _silent_wav(duration), model_id, response_format="verbose_json"
        )

    assert response["text"] == "Final words."
    assert response["segments"][0]["start"] == duration - 2
    assert response["segments"][0]["end"] == duration - 1
    if use_chunks:
        faster_whisper.BatchedInferencePipeline.assert_called_once_with(model)
        assert pipeline.transcribe.call_args.kwargs["chunk_length"] == 15
        assert pipeline.transcribe.call_args.kwargs["batch_size"] == 1
        model.transcribe.assert_not_called()
    else:
        faster_whisper.BatchedInferencePipeline.assert_not_called()
        model.transcribe.assert_called_once()
