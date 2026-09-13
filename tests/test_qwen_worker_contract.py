"""Dependency-free contract tests for the disposable Qwen worker."""

from __future__ import annotations

import asyncio
import importlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def _import_worker(monkeypatch):
    cuda = SimpleNamespace(
        OutOfMemoryError=RuntimeError,
        is_available=lambda: False,
        empty_cache=lambda: None,
    )
    torch_stub = SimpleNamespace(cuda=cuda, float16=object())
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
