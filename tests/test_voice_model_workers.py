"""Contract tests for the isolated voice-cloning workers."""

from __future__ import annotations

import asyncio
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
    ("module_name", "provider", "model_ids", "sample_rate"),
    [
        (
            "providers.chatterbox.app",
            "chatterbox",
            {"chatterbox/regular", "chatterbox/turbo"},
            24000,
        ),
        (
            "providers.cosyvoice.app",
            "cosyvoice",
            {"cosyvoice/2-0.5b", "cosyvoice/3-0.5b"},
            24000,
        ),
        ("providers.ming.app", "ming", {"ming/3-0.5b"}, 44100),
        ("providers.voxcpm.app", "voxcpm", {"voxcpm/2"}, 48000),
    ],
)
def test_worker_manifests_pin_all_models(
    monkeypatch, module_name, provider, model_ids, sample_rate
):
    worker = _import_worker(monkeypatch, module_name)

    assert worker.MANIFEST["schema_version"] == 1
    assert worker.MANIFEST["provider"] == provider
    assert set(worker.MODEL_IDS) == model_ids == set(worker.MODEL_REVISIONS)
    assert {model["id"] for model in worker.MANIFEST["models"]} == model_ids
    assert all(len(revision) == 40 for revision in worker.MODEL_REVISIONS.values())
    assert all(model["sample_rate"] == sample_rate for model in worker.MANIFEST["models"])


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
        ("providers.ming.app", "ming/3-0.5b"),
        ("providers.voxcpm.app", "voxcpm/2"),
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


@pytest.mark.asyncio
async def test_chatterbox_restarts_after_unload_when_idle(monkeypatch):
    worker = _import_worker(monkeypatch, "providers.chatterbox.app")
    monkeypatch.setattr(worker, "RESTART_AFTER_UNLOAD_S", 0.01)
    worker.runtime.cuda_used = True
    kills = []
    monkeypatch.setattr(worker.os, "kill", lambda pid, sig: kills.append(sig))

    await worker.unload_model(worker.LoadRequest(model="chatterbox/regular"))
    await worker.restart_task

    assert kills == [worker.signal.SIGTERM]


@pytest.mark.asyncio
async def test_chatterbox_load_after_unload_cancels_restart(monkeypatch):
    worker = _import_worker(monkeypatch, "providers.chatterbox.app")
    monkeypatch.setattr(worker, "RESTART_AFTER_UNLOAD_S", 0.05)
    worker.runtime.cuda_used = True
    kills = []
    monkeypatch.setattr(worker.os, "kill", lambda pid, sig: kills.append(sig))

    def fake_load(model_id):
        worker.runtime.model = object()
        worker.runtime.model_id = model_id

    monkeypatch.setattr(worker.runtime, "load", fake_load)

    await worker.unload_model(worker.LoadRequest(model="chatterbox/regular"))
    pending = worker.restart_task
    await worker.load_model(worker.LoadRequest(model="chatterbox/turbo"))
    await asyncio.sleep(0.1)

    assert pending.cancelled()
    assert kills == []


@pytest.mark.asyncio
async def test_chatterbox_never_restarts_without_cuda_or_when_disabled(monkeypatch):
    worker = _import_worker(monkeypatch, "providers.chatterbox.app")
    monkeypatch.setattr(worker.os, "kill", lambda pid, sig: pytest.fail("worker exited"))

    monkeypatch.setattr(worker, "RESTART_AFTER_UNLOAD_S", 0.01)
    await worker.unload_model(worker.LoadRequest(model="chatterbox/regular"))
    assert worker.restart_task is None

    worker.runtime.cuda_used = True
    monkeypatch.setattr(worker, "RESTART_AFTER_UNLOAD_S", 0)
    await worker.unload_model(worker.LoadRequest(model="chatterbox/regular"))
    assert worker.restart_task is None


class _Embedding:
    def clone(self):
        return self


def _fake_ming(worker, *, overhead=100, lengths=None):
    """Install a fake Ming pipeline; returns the calls it records."""
    calls = {"load_prompt": [], "prepare_text": [], "generate": []}
    lengths = list(lengths or [])

    class Model:
        tokenizer = object()
        model_type = "dense"
        patch_size = 1

        def encode_prompt_latent(self, _waveform):
            return np.zeros((1, 75, 64), dtype=np.float32)

        def generate(self, **kwargs):
            calls["generate"].append(kwargs)
            steps = lengths.pop(0) if lengths else 10
            return np.full(steps * 3528, 0.1, dtype=np.float32)

    class Tts:
        model = Model()

        def _load_prompt(self, paths):
            calls["load_prompt"].append(paths)
            assert worker.os.path.exists(paths[0])
            return "waveform", [_Embedding()]

        def _pad_align(self):
            return 3528

        def prepare_text(self, text, taskname, language, max_length):
            calls["prepare_text"].append((text, taskname, language, max_length))
            return text.split("|")

    def build_input_ids(_tokenizer, _model_type, prompt, text, **kwargs):
        calls.setdefault("layouts", []).append((prompt, kwargs))
        return list(range(overhead + len(text.split()))), [], -1

    worker.runtime.model = Tts()
    worker.runtime.model_id = "ming/3-0.5b"
    worker.runtime.tasks = SimpleNamespace(
        build_tts_prompt=lambda language: f"prompt:{language}",
        build_task_instruction=lambda task, **controls: f"{task}:{controls['style']}",
    )
    worker.runtime.contract = SimpleNamespace(build_input_ids=build_input_ids)
    worker.runtime.trim = lambda speech, _rate: speech
    return calls


def test_ming_clone_uses_transcript_and_sizes_fragments_to_the_reference(monkeypatch):
    worker = _import_worker(monkeypatch, "providers.ming.app")
    calls = _fake_ming(worker, overhead=300)
    payload = worker.SynthesisRequest(
        model="ming/3-0.5b",
        text="First part.|Second part.",
        reference_audio=_wav_base64(),
        clone_transcript="  Exact reference words. ",
        language="en-US",
    )

    chunks = list(worker.runtime.generate(payload))

    assert len(chunks) == 2
    # (512 - 300 positions already used) // 8 positions per word.
    assert calls["prepare_text"] == [("First part.|Second part.", "TTS", None, 26)]
    first = calls["generate"][0]
    assert first["prompt"] == "prompt:None"
    assert first["prompt_text"] == "Exact reference words."
    assert first["use_zero_spk_emb"] is False
    assert first["instruction"] is None
    assert first["prompt_latent"].shape == (1, 75, 64)
    assert calls["layouts"][0][1]["prompt_latent_steps"] == 75
    assert not worker.os.path.exists(calls["load_prompt"][0][0])

    list(worker.runtime.generate(payload))
    assert len(calls["load_prompt"]) == 1  # the encoded reference is reused


def test_ming_regenerates_a_fragment_in_halves_when_it_fills_the_cache(monkeypatch):
    worker = _import_worker(monkeypatch, "providers.ming.app")
    # The first attempt uses every one of the 512 - 104 free positions.
    calls = _fake_ming(worker, overhead=100, lengths=[408, 10, 10])
    payload = worker.SynthesisRequest(
        model="ming/3-0.5b",
        text="one two three four",
        reference_audio=_wav_base64(),
        clone_transcript="Exact words.",
    )

    chunks = list(worker.runtime.generate(payload))

    assert [call["text"] for call in calls["generate"]] == [
        "one two three four",
        "one two",
        "three four",
    ]
    assert len(chunks) == 2


def test_ming_voice_design_and_named_languages(monkeypatch):
    worker = _import_worker(monkeypatch, "providers.ming.app")
    monkeypatch.setattr(worker.torch, "zeros", lambda *_shape: _Embedding(), raising=False)
    calls = _fake_ming(worker)
    payload = worker.SynthesisRequest(
        model="ming/3-0.5b",
        text="Bonjour.",
        voice_design="A calm narrator",
        language="fr",
    )

    assert list(worker.runtime.generate(payload))

    call = calls["generate"][0]
    assert call["prompt"] == "prompt:fr"
    assert call["instruction"] == "VOICE_DESIGN:A calm narrator"
    assert call["use_zero_spk_emb"] is True
    assert call["prompt_latent"] is None and call["prompt_text"] is None
    assert calls["prepare_text"][0][1:3] == ("TTS", "fr")


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"clone_transcript": None}, "clone_transcript_required"),
        ({"reference_audio": None}, "reference_audio_required"),
        ({"voice_design": "Bright voice"}, "voice_conflict"),
        ({"speed": 1.2}, "speed_unsupported"),
        ({"instructions": "Whisper"}, "instructions_unsupported"),
        ({"reference_audio": _wav_base64(duration=20.5)}, "reference_audio_too_long"),
    ],
)
def test_ming_rejects_requests_it_cannot_voice(monkeypatch, changes, code):
    worker = _import_worker(monkeypatch, "providers.ming.app")
    _fake_ming(worker)
    fields = {
        "model": "ming/3-0.5b",
        "text": "Check.",
        "reference_audio": _wav_base64(),
        "clone_transcript": "Exact words.",
    }
    fields.update(changes)

    with pytest.raises(worker.WorkerFailure) as failure:
        list(worker.runtime.generate(worker.SynthesisRequest(**fields)))

    assert failure.value.code == code


def test_ming_runs_bf16_on_every_card(monkeypatch):
    worker = _import_worker(monkeypatch, "providers.ming.app")

    assert worker._resolve_dtype_name("auto") == "bfloat16"
    assert worker._resolve_dtype_name("float32") == "float32"
    with pytest.raises(worker.WorkerFailure):
        worker._resolve_dtype_name("int8")


def test_voxcpm_runs_fp16_on_cards_without_bf16(monkeypatch):
    worker = _import_worker(monkeypatch, "providers.voxcpm.app")

    assert worker._resolve_dtype_name("auto", (7, 5)) == "float16"
    assert worker._resolve_dtype_name("auto", (8, 6)) == "bfloat16"
    assert worker._resolve_dtype_name("float32", (7, 5)) == "float32"
    with pytest.raises(worker.WorkerFailure):
        worker._resolve_dtype_name("int8", (7, 5))


def _fake_voxcpm(worker):
    calls = {"prompt": [], "stream": []}

    class Model:
        def build_prompt_cache(self, **kwargs):
            calls["prompt"].append(kwargs)
            for path in kwargs.values():
                if path and str(path).endswith(".wav"):
                    assert worker.os.path.exists(path)
            return {"mode": "cached"}

        def generate_with_prompt_cache_streaming(self, **kwargs):
            calls["stream"].append(kwargs)
            yield np.array([[0.1, 0.2]], dtype=np.float32), None, None
            yield np.array([[0.3]], dtype=np.float32), None, None

    worker.runtime.model = Model()
    worker.runtime.model_id = "voxcpm/2"
    return calls


def test_voxcpm_clone_with_transcript_continues_from_the_clip(monkeypatch):
    worker = _import_worker(monkeypatch, "providers.voxcpm.app")
    monkeypatch.setattr(worker, "MAX_SEGMENT_CHARS", 30)
    calls = _fake_voxcpm(worker)
    payload = worker.SynthesisRequest(
        model="voxcpm/2",
        text="First sentence here. Second one.\nThird sentence is longer than the rest!",
        reference_audio=_wav_base64(),
        clone_transcript="Exact reference words.",
    )

    chunks = list(worker.runtime.generate(payload))

    prompt = calls["prompt"][0]
    assert prompt["prompt_text"] == "Exact reference words."
    assert prompt["prompt_wav_path"] == prompt["reference_wav_path"]
    assert not worker.os.path.exists(prompt["reference_wav_path"])
    assert [call["target_text"] for call in calls["stream"]] == [
        "First sentence here.",
        "Second one.",
        "Third sentence is longer than the rest!",
    ]
    assert all(call["prompt_cache"] == {"mode": "cached"} for call in calls["stream"])
    assert np.concatenate(chunks).tolist() == pytest.approx([0.1, 0.2, 0.3] * 3)

    list(worker.runtime.generate(payload))
    assert len(calls["prompt"]) == 1  # the prompt cache is reused for the same clip


def test_voxcpm_segments_group_whole_sentences():
    from providers.voxcpm.app import _segments

    text = "One. Two! Three? A much longer fourth sentence."
    assert _segments(text, 15) == ["One. Two!", "Three?", "A much longer fourth sentence."]
    assert _segments(text, 300) == [text]


def test_voxcpm_style_directions_use_reference_mode(monkeypatch):
    worker = _import_worker(monkeypatch, "providers.voxcpm.app")
    calls = _fake_voxcpm(worker)
    payload = worker.SynthesisRequest(
        model="voxcpm/2",
        text="Read this.",
        reference_audio=_wav_base64(),
        clone_transcript="Exact reference words.",
        instructions="slightly faster (cheerful)",
    )

    assert list(worker.runtime.generate(payload))

    assert set(calls["prompt"][0]) == {"reference_wav_path"}
    assert calls["stream"][0]["target_text"] == "(slightly faster cheerful)Read this."


def test_voxcpm_voice_design_keeps_text_whole(monkeypatch):
    worker = _import_worker(monkeypatch, "providers.voxcpm.app")
    monkeypatch.setattr(worker, "MAX_SEGMENT_CHARS", 10)
    calls = _fake_voxcpm(worker)
    payload = worker.SynthesisRequest(
        model="voxcpm/2",
        text="One sentence. Another sentence.",
        voice_design="A warm older man",
    )

    assert list(worker.runtime.generate(payload))

    assert calls["prompt"] == []
    assert [call["target_text"] for call in calls["stream"]] == [
        "(A warm older man)One sentence. Another sentence."
    ]
    assert calls["stream"][0]["prompt_cache"] is None


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"reference_audio": None}, "reference_audio_required"),
        ({"voice_design": "Bright voice"}, "voice_conflict"),
        ({"speed": 0.8}, "speed_unsupported"),
        ({"reference_audio": _wav_base64(duration=30.5)}, "reference_audio_too_long"),
    ],
)
def test_voxcpm_rejects_requests_it_cannot_voice(monkeypatch, changes, code):
    worker = _import_worker(monkeypatch, "providers.voxcpm.app")
    _fake_voxcpm(worker)
    fields = {"model": "voxcpm/2", "text": "Check.", "reference_audio": _wav_base64()}
    fields.update(changes)

    with pytest.raises(worker.WorkerFailure) as failure:
        list(worker.runtime.generate(worker.SynthesisRequest(**fields)))

    assert failure.value.code == code


@pytest.mark.parametrize(
    ("module_name", "model_id"),
    [("providers.ming.app", "ming/3-0.5b"), ("providers.voxcpm.app", "voxcpm/2")],
)
@pytest.mark.asyncio
async def test_new_workers_restart_after_unload_when_idle(monkeypatch, module_name, model_id):
    worker = _import_worker(monkeypatch, module_name)
    monkeypatch.setattr(worker, "RESTART_AFTER_UNLOAD_S", 0.01)
    worker.runtime.cuda_used = True
    kills = []
    monkeypatch.setattr(worker.os, "kill", lambda pid, sig: kills.append(sig))

    await worker.unload_model(worker.LoadRequest(model=model_id))
    await worker.restart_task

    assert kills == [worker.signal.SIGTERM]
