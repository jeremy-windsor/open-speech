"""Dependency-free contract tests for the disposable Qwen worker."""

from __future__ import annotations

import asyncio
import importlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest


def _import_worker(monkeypatch):
    class FakeOutOfMemoryError(Exception):
        pass

    cuda = SimpleNamespace(
        OutOfMemoryError=FakeOutOfMemoryError,
        is_available=lambda: False,
        empty_cache=lambda: None,
    )
    torch_stub = SimpleNamespace(
        cuda=cuda,
        float16=object(),
        float32=object(),
        bfloat16=object(),
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    sys.modules.pop("providers.qwen3.app", None)
    return importlib.import_module("providers.qwen3.app")


@pytest.mark.asyncio
async def test_unconsumed_stream_releases_worker_lock(monkeypatch):
    worker = _import_worker(monkeypatch)
    worker.STREAM_QUEUE_TIMEOUT_S = 0.02
    monkeypatch.setattr(worker, "_split_text", lambda _text: ["one", "two", "three", "four"])
    monkeypatch.setattr(
        worker.runtime,
        "generate",
        lambda _payload, _segment: np.array([0.1], dtype=np.float32),
    )

    class Request:
        async def is_disconnected(self):
            return False

    payload = worker.SynthesisRequest(
        model="qwen3/0.6b-custom-voice",
        text="ignored",
        voice="Ryan",
    )
    response = await worker.synthesize(payload, Request())

    assert response.status_code == 200
    assert worker.operation_lock.locked()
    await asyncio.sleep(0.1)
    assert not worker.operation_lock.locked()
    assert await anext(response.body_iterator) == np.array([0.1], dtype=np.float32).tobytes()
    with pytest.raises(worker.WorkerFailure, match="stopped reading"):
        await anext(response.body_iterator)


def test_worker_pins_model_revisions(monkeypatch):
    worker = _import_worker(monkeypatch)

    assert set(worker.MODEL_IDS) == set(worker.MODEL_REVISIONS)
    assert all(len(revision) == 40 for revision in worker.MODEL_REVISIONS.values())
    assert {
        manifest["id"]: manifest["revision"] for manifest in worker.MANIFEST["models"]
    } == worker.MODEL_REVISIONS


@pytest.mark.parametrize(
    ("setting", "capability", "expected"),
    [
        ("auto", (7, 5), "float32"),
        ("auto", (8, 6), "bfloat16"),
        ("float32", (7, 5), "float32"),
        ("float16", (7, 5), "float16"),
        ("bfloat16", (8, 6), "bfloat16"),
    ],
)
def test_dtype_policy(monkeypatch, setting, capability, expected):
    worker = _import_worker(monkeypatch)

    assert worker._resolve_dtype_name(setting, capability) == expected


def test_dtype_policy_rejects_invalid_or_unsupported_values(monkeypatch):
    worker = _import_worker(monkeypatch)

    with pytest.raises(worker.WorkerFailure) as unsupported:
        worker._resolve_dtype_name("bfloat16", (7, 5))
    assert unsupported.value.code == "dtype_unsupported"

    with pytest.raises(worker.WorkerFailure) as invalid:
        worker._resolve_dtype_name("magic", (8, 6))
    assert invalid.value.code == "dtype_invalid"


@pytest.mark.asyncio
async def test_cuda_context_failure_marks_health_failed_and_schedules_restart(monkeypatch):
    worker = _import_worker(monkeypatch)

    class BrokenModel:
        def generate_custom_voice(self, **_kwargs):
            raise RuntimeError(
                "probability tensor contains either inf, nan or element < 0"
            )

    class Request:
        async def is_disconnected(self):
            return False

    worker.runtime.model = BrokenModel()
    worker.runtime.model_id = "qwen3/0.6b-custom-voice"
    scheduled = []
    monkeypatch.setattr(worker, "_schedule_process_restart", lambda: scheduled.append(True))
    payload = worker.SynthesisRequest(
        model="qwen3/0.6b-custom-voice",
        text="In the beginning.",
        voice="Ryan",
    )

    with pytest.raises(worker.HTTPException) as exc_info:
        await worker.synthesize(payload, Request())

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["code"] == "cuda_context_failed"
    assert scheduled == [True]
    health = await worker.health()
    assert health["status"] == "failed"
    assert health["failure"]["code"] == "cuda_context_failed"
    assert not worker.operation_lock.locked()


def test_accelerator_error_is_a_cuda_context_failure(monkeypatch):
    worker = _import_worker(monkeypatch)
    AcceleratorError = type("AcceleratorError", (RuntimeError,), {})

    assert worker._is_cuda_context_failure(AcceleratorError("kernel failed"))


@pytest.mark.asyncio
async def test_failed_runtime_rejects_generate_and_unload_without_cuda_calls(monkeypatch):
    worker = _import_worker(monkeypatch)
    cuda_calls = []
    worker.torch.cuda.empty_cache = lambda: cuda_calls.append(True)
    worker.runtime.failed_code = "cuda_context_failed"

    class Request:
        async def is_disconnected(self):
            return False

    payload = worker.SynthesisRequest(
        model="qwen3/0.6b-custom-voice",
        text="In the beginning.",
        voice="Ryan",
    )

    with pytest.raises(worker.HTTPException) as generate_error:
        await worker.synthesize(payload, Request())
    with pytest.raises(worker.HTTPException) as unload_error:
        await worker.unload_model(worker.LoadRequest(model=payload.model))

    assert generate_error.value.status_code == 503
    assert unload_error.value.status_code == 503
    assert worker.runtime.failed_code == "cuda_context_failed"
    assert cuda_calls == []


@pytest.mark.asyncio
async def test_invalid_dtype_marks_health_failed(monkeypatch):
    worker = _import_worker(monkeypatch)
    worker.DTYPE_SETTING = "magic"
    worker.torch.cuda.is_available = lambda: True
    worker.torch.cuda.get_device_capability = lambda _index: (8, 6)
    worker.torch.cuda.get_device_name = lambda _index: "Fake GPU"
    worker.torch.cuda.get_arch_list = lambda: ["sm_86"]
    worker.torch.cuda.memory_allocated = lambda _index: 0
    worker.torch.cuda.memory_reserved = lambda _index: 0

    result = await worker.health()

    assert result["status"] == "failed"
    assert result["failure"]["code"] == "dtype_invalid"


@pytest.mark.asyncio
async def test_cuda_query_failure_marks_health_failed(monkeypatch):
    worker = _import_worker(monkeypatch)

    def fail_query(_index):
        raise RuntimeError("driver query failed")

    worker.torch.cuda.is_available = lambda: True
    worker.torch.cuda.get_device_capability = fail_query

    result = await worker.health()

    assert result["status"] == "failed"
    assert result["failure"]["code"] == "cuda_query_failed"


def test_parameter_dtype_summary_includes_wrapper_components(monkeypatch):
    worker = _import_worker(monkeypatch)

    class Module:
        def __init__(self, dtype):
            self.dtype = dtype

        def parameters(self):
            return [SimpleNamespace(dtype=self.dtype)]

    root_child = Module("torch.float32")
    root = SimpleNamespace(named_children=lambda: [("talker", root_child)])
    wrapper = SimpleNamespace(model=root, speech_tokenizer=Module("torch.float32"))

    assert worker._parameter_dtype_summary(wrapper) == {
        "model.talker": ["float32"],
        "speech_tokenizer": ["float32"],
    }


def test_parameter_dtype_diagnostics_do_not_break_runtime(monkeypatch):
    worker = _import_worker(monkeypatch)
    worker.runtime.parameter_dtypes = {"previous": ["float32"]}

    def fail_diagnostic(_model):
        raise RuntimeError("diagnostic failed")

    monkeypatch.setattr(worker, "_parameter_dtype_summary", fail_diagnostic)

    worker.runtime._refresh_parameter_dtypes()

    assert worker.runtime.parameter_dtypes == {"previous": ["float32"]}


@pytest.mark.asyncio
async def test_cuda_context_failure_during_load_schedules_restart(monkeypatch):
    worker = _import_worker(monkeypatch)
    worker.torch.cuda.is_available = lambda: True
    worker.torch.cuda.get_device_capability = lambda _index: (7, 5)
    worker.torch.cuda.get_arch_list = lambda: ["sm_75"]

    class BrokenLoader:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            raise RuntimeError("probability tensor contains either inf or nan")

    monkeypatch.setitem(
        sys.modules,
        "qwen_tts",
        SimpleNamespace(Qwen3TTSModel=BrokenLoader),
    )
    scheduled = []
    monkeypatch.setattr(worker, "_schedule_process_restart", lambda: scheduled.append(True))

    with pytest.raises(worker.HTTPException) as exc_info:
        await worker.load_model(worker.LoadRequest(model="qwen3/0.6b-custom-voice"))

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail["code"] == "cuda_context_failed"
    assert scheduled == [True]


def test_benchmark_allows_runtime_without_git():
    from scripts import benchmark_tts

    with patch("subprocess.run", side_effect=FileNotFoundError("git")):
        assert benchmark_tts._git_sha() is None
