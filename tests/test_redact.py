"""Client-facing error messages name no filesystem path (src/utils/redact.py); the log keeps the full text."""

from __future__ import annotations

import pytest

from src.model_manager import ModelLifecycleError, ModelManager
from src.utils.redact import redact_paths
from tests.test_model_manager import FakeSTTRouter, FakeTTSRouter

CACHED = "/root/.cache/huggingface/hub/models--istupakov--parakeet-tdt-0.6b-v2-onnx/snapshots/abc/encoder-model.onnx"


def test_redact_paths_keeps_file_names_and_leaves_urls_and_source_locations():
    assert redact_paths("Load model from /root/.cache/huggingface/hub/models--x/snapshots/abc/encoder.onnx failed") \
        == "Load model from encoder.onnx failed"
    assert redact_paths(r"No file C:\Users\me\cache\vocab.txt here") == "No file vocab.txt here"
    assert redact_paths(r"at \\host\share\models\x.onnx.") == "at x.onnx."
    kept = "See https://huggingface.co/api/models/x and onnxruntime/core/framework/bfc_arena.cc:359 (and/or)"
    assert redact_paths(kept) == kept


def test_a_failed_fetch_names_the_file_not_its_folder():
    manager = ModelManager(stt_router=FakeSTTRouter(), tts_router=FakeTTSRouter())
    error = manager.stt_fetch_failed("Systran/faster-whisper-base", OSError(f"No such file: {CACHED}"))
    assert error.message == "Failed to load model 'Systran/faster-whisper-base': No such file: encoder-model.onnx"


def test_a_failed_load_names_the_file_not_its_folder():
    stt = FakeSTTRouter()

    def broken(model_id):
        raise RuntimeError(f"Load model from {CACHED} failed: Protobuf parsing failed")

    stt.load_model = broken
    manager = ModelManager(stt_router=stt, tts_router=FakeTTSRouter())
    with pytest.raises(ModelLifecycleError) as caught:
        manager.load("Systran/faster-whisper-base")
    assert caught.value.message == (
        "Failed to load model 'Systran/faster-whisper-base': Load model from encoder-model.onnx failed: "
        "Protobuf parsing failed"
    )
