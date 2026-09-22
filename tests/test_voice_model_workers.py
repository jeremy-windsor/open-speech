"""Contract tests for Chatterbox and CosyVoice isolated workers."""

from __future__ import annotations

import base64
import importlib
import io
import sys
import wave
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest


def _import_worker(monkeypatch, module_name: str):
    class FakeOutOfMemoryError(Exception):
        pass

    cuda = SimpleNamespace(
        OutOfMemoryError=FakeOutOfMemoryError,
        is_available=lambda: False,
        empty_cache=lambda: None,
    )
    torch_stub = SimpleNamespace(
        cuda=cuda,
        inference_mode=lambda: nullcontext(),
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def _wav_base64(duration: float = 6.1, sample_rate: int = 16000) -> str:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * int(duration * sample_rate))
    return base64.b64encode(output.getvalue()).decode("ascii")


@pytest.mark.parametrize(
    ("module_name", "provider", "model_ids"),
    [
        (
            "providers.chatterbox.app",
            "chatterbox",
            {"chatterbox/regular", "chatterbox/turbo"},
        ),
        (
            "providers.cosyvoice.app",
            "cosyvoice",
            {"cosyvoice/2-0.5b", "cosyvoice/3-0.5b"},
        ),
    ],
)
def test_worker_manifests_pin_all_models(monkeypatch, module_name, provider, model_ids):
    worker = _import_worker(monkeypatch, module_name)

    assert worker.MANIFEST["schema_version"] == 1
    assert worker.MANIFEST["provider"] == provider
    assert set(worker.MODEL_IDS) == model_ids == set(worker.MODEL_REVISIONS)
    assert {model["id"] for model in worker.MANIFEST["models"]} == model_ids
    assert all(len(revision) == 40 for revision in worker.MODEL_REVISIONS.values())
    assert all(model["sample_rate"] == 24000 for model in worker.MANIFEST["models"])


def test_chatterbox_capabilities_match_regular_and_turbo(monkeypatch):
    worker = _import_worker(monkeypatch, "providers.chatterbox.app")
    manifests = {model["id"]: model for model in worker.MANIFEST["models"]}

    for model in manifests.values():
        capabilities = model["capabilities"]
        assert capabilities["voice_clone"] is True
        assert capabilities["reference_audio"] is True
        assert capabilities["clone_transcript_required"] is False
        assert capabilities["speed_control"] is False
        assert capabilities["streaming"] is False
    assert manifests["chatterbox/regular"]["capabilities"]["native_tags"] is False
    assert manifests["chatterbox/turbo"]["capabilities"]["native_tags"] is True


def test_chatterbox_generates_clone_and_rejects_unsupported_speed(monkeypatch):
    worker = _import_worker(monkeypatch, "providers.chatterbox.app")
    calls = []

    class Model:
        def generate(self, text, *, audio_prompt_path):
            calls.append((text, audio_prompt_path))
            return np.array([[0.1, -0.2]], dtype=np.float32)

    worker.runtime.model = Model()
    worker.runtime.model_id = "chatterbox/regular"
    payload = worker.SynthesisRequest(
        model="chatterbox/regular",
        text="Open Speech clone test.",
        reference_audio=_wav_base64(),
    )

    result = worker.runtime.generate(payload)

    assert result.tolist() == pytest.approx([0.1, -0.2])
    assert calls[0][0] == payload.text
    assert not worker.os.path.exists(calls[0][1])

    payload.speed = 1.2
    with pytest.raises(worker.WorkerFailure) as unsupported:
        worker.runtime.generate(payload)
    assert unsupported.value.code == "speed_unsupported"


def test_chatterbox_turbo_enforces_reference_duration(monkeypatch):
    worker = _import_worker(monkeypatch, "providers.chatterbox.app")
    worker.runtime.model = SimpleNamespace(generate=lambda *_args, **_kwargs: np.zeros(8))
    worker.runtime.model_id = "chatterbox/turbo"
    payload = worker.SynthesisRequest(
        model="chatterbox/turbo",
        text="Turbo reference check.",
        reference_audio=_wav_base64(duration=5.0),
    )

    with pytest.raises(worker.WorkerFailure) as too_short:
        worker.runtime.generate(payload)

    assert too_short.value.code == "reference_audio_too_short"


def test_cosyvoice_capabilities_expose_clone_controls(monkeypatch):
    worker = _import_worker(monkeypatch, "providers.cosyvoice.app")

    for model in worker.MANIFEST["models"]:
        capabilities = model["capabilities"]
        assert capabilities["voice_clone"] is True
        assert capabilities["clone_transcript_required"] is True
        assert capabilities["instructions"] is True
        assert capabilities["speed_control"] is True
        assert capabilities["streaming"] is True


def test_cosyvoice_zero_shot_uses_transcript_speed_and_stream_mode(monkeypatch):
    worker = _import_worker(monkeypatch, "providers.cosyvoice.app")
    calls = []

    class Model:
        def inference_zero_shot(self, *args, **kwargs):
            calls.append((args, kwargs))
            yield {"tts_speech": np.array([[0.25, -0.25]], dtype=np.float32)}

    worker.runtime.model = Model()
    worker.runtime.model_id = "cosyvoice/3-0.5b"
    payload = worker.SynthesisRequest(
        model="cosyvoice/3-0.5b",
        text="Open Speech clone test.",
        speed=1.2,
        reference_audio=_wav_base64(),
        clone_transcript="Exact reference words.",
    )

    chunks = list(worker.runtime.generate(payload))

    assert chunks[0].tolist() == pytest.approx([0.25, -0.25])
    args, kwargs = calls[0]
    assert args[0] == payload.text
    assert args[1] == "You are a helpful assistant.<|endofprompt|>Exact reference words."
    assert not worker.os.path.exists(args[2])
    assert kwargs == {"stream": False, "speed": 1.2}


def test_cosyvoice_instructions_use_model_specific_prompt(monkeypatch):
    worker = _import_worker(monkeypatch, "providers.cosyvoice.app")
    calls = []

    class Model:
        def inference_instruct2(self, *args, **kwargs):
            calls.append((args, kwargs))
            yield {"tts_speech": np.ones(2, dtype=np.float32)}

    worker.runtime.model = Model()
    worker.runtime.model_id = "cosyvoice/2-0.5b"
    payload = worker.SynthesisRequest(
        model="cosyvoice/2-0.5b",
        text="Follow the direction.",
        instructions="calm and precise",
        reference_audio=_wav_base64(),
        clone_transcript="Exact reference words.",
    )

    assert list(worker.runtime.generate(payload))
    args, kwargs = calls[0]
    assert args[1] == "calm and precise<|endofprompt|>"
    assert kwargs == {"stream": True, "speed": 1.0}


def test_cosyvoice_requires_exact_transcript_and_bounded_reference(monkeypatch):
    worker = _import_worker(monkeypatch, "providers.cosyvoice.app")
    worker.runtime.model = object()
    worker.runtime.model_id = "cosyvoice/2-0.5b"

    missing_transcript = worker.SynthesisRequest(
        model="cosyvoice/2-0.5b",
        text="Missing transcript.",
        reference_audio=_wav_base64(),
    )
    with pytest.raises(worker.WorkerFailure) as missing:
        list(worker.runtime.generate(missing_transcript))
    assert missing.value.code == "clone_transcript_required"

    long_reference = worker.SynthesisRequest(
        model="cosyvoice/2-0.5b",
        text="Long reference.",
        reference_audio=_wav_base64(duration=30.1),
        clone_transcript="Exact words.",
    )
    with pytest.raises(worker.WorkerFailure) as too_long:
        list(worker.runtime.generate(long_reference))
    assert too_long.value.code == "reference_audio_too_long"


@pytest.mark.asyncio
async def test_cosyvoice_response_streams_float32_and_releases_lock(monkeypatch):
    worker = _import_worker(monkeypatch, "providers.cosyvoice.app")

    def generate(_payload):
        yield np.array([0.1], dtype="<f4")
        yield np.array([0.2], dtype="<f4")

    monkeypatch.setattr(worker.runtime, "generate", generate)

    class Request:
        async def is_disconnected(self):
            return False

    payload = worker.SynthesisRequest(
        model="cosyvoice/2-0.5b",
        text="Streaming check.",
        reference_audio="ignored",
        clone_transcript="Exact words.",
    )
    response = await worker.synthesize(payload, Request())
    body = b"".join([chunk async for chunk in response.body_iterator])

    assert response.headers["x-audio-format"] == "float32le"
    assert response.headers["x-native-streaming"] == "true"
    assert np.frombuffer(body, dtype="<f4").tolist() == pytest.approx([0.1, 0.2])
    assert not worker.operation_lock.locked()


@pytest.mark.parametrize(
    ("module_name", "model_id"),
    [
        ("providers.chatterbox.app", "chatterbox/regular"),
        ("providers.cosyvoice.app", "cosyvoice/2-0.5b"),
    ],
)
@pytest.mark.asyncio
async def test_worker_response_cleanup_releases_only_its_own_lock(
    monkeypatch,
    module_name,
    model_id,
):
    worker = _import_worker(monkeypatch, module_name)

    if module_name == "providers.chatterbox.app":
        monkeypatch.setattr(
            worker.runtime,
            "generate",
            lambda _payload: np.array([0.1], dtype="<f4"),
        )
    else:
        def generate(_payload):
            yield np.array([0.1], dtype="<f4")

        monkeypatch.setattr(worker.runtime, "generate", generate)

    class Request:
        async def is_disconnected(self):
            return False

    payload = worker.SynthesisRequest(
        model=model_id,
        text="Lock ownership check.",
        reference_audio="ignored",
        clone_transcript="Exact words.",
    )
    response = await worker.synthesize(payload, Request())
    _ = [chunk async for chunk in response.body_iterator]
    assert not worker.operation_lock.locked()

    await worker.operation_lock.acquire()
    try:
        assert response.background is not None
        await response.background()
        assert worker.operation_lock.locked()
    finally:
        if worker.operation_lock.locked():
            worker.operation_lock.release()
