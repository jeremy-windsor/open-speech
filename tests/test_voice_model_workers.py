"""Contract tests for the isolated Chatterbox and Fish Audio S2 workers and their shared scaffolding."""

from __future__ import annotations

import asyncio
import base64
import importlib
import importlib.util
import io
import json
import math
import re
import sys
import types
import urllib.error
import urllib.parse
import wave
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from src.tts.external import ExternalProviderError, ExternalTTSBackend

ROOT = Path(__file__).resolve().parents[1]


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


def _fake_gpu(worker) -> list[bool]:
    """Report a usable GPU on the stubbed torch and record CUDA cache flushes."""
    emptied: list[bool] = []
    worker.torch.cuda.is_available = lambda: True
    worker.torch.cuda.get_device_capability = lambda _index: (7, 5)
    worker.torch.cuda.get_device_name = lambda _index: "Fake GPU"
    worker.torch.cuda.get_arch_list = lambda: ["sm_75"]
    worker.torch.cuda.memory_allocated = lambda _index: 0
    worker.torch.cuda.memory_reserved = lambda _index: 0
    worker.torch.cuda.empty_cache = lambda: emptied.append(True)
    return emptied


# Official torch wheels list one cubin per compute-capability
# family, not every GPU they run on. A cubin for X.y runs on any X.z with z >= y (CUDA C
# Programming Guide, Binary Compatibility; torch's own _check_cubins: "backward compatible
# within major version"), and PTX (compute_XY) is JIT-compiled onto any equal or newer GPU.
# require_cuda() asked for the exact sm_XY, so every model load answered 503
# cuda_arch_unsupported on RTX 40-series / L4 / L40 (8.9), GTX 10xx (6.1) and Maxwell 5.2
# cards while /health said ok.
CU128_ARCH_LIST = ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]
CU12X_WORKER_IMAGE_ARCH_LIST = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]


def _fake_gpu_of(worker, capability, arch_list) -> list[bool]:
    emptied = _fake_gpu(worker)
    worker.torch.cuda.get_device_capability = lambda _index: capability
    worker.torch.cuda.get_arch_list = lambda: list(arch_list)
    return emptied


@pytest.mark.parametrize(
    ("capability", "arch_list", "runs"),
    [
        pytest.param((8, 9), CU128_ARCH_LIST, True, id="ada-8.9-runs-the-sm_86-cubin"),
        pytest.param((8, 9), CU12X_WORKER_IMAGE_ARCH_LIST, True, id="ada-8.9-worker-image"),
        pytest.param((6, 1), CU12X_WORKER_IMAGE_ARCH_LIST, True, id="pascal-6.1-runs-the-sm_60-cubin"),
        pytest.param((5, 2), CU12X_WORKER_IMAGE_ARCH_LIST, True, id="maxwell-5.2-runs-the-sm_50-cubin"),
        pytest.param((7, 5), ["sm_75"], True, id="exact-cubin"),
        pytest.param((12, 0), ["sm_80", "sm_90", "compute_90"], True, id="newer-gpu-jits-the-ptx"),
        pytest.param((10, 3), ["sm_90", "sm_100f"], True, id="family-cubin-on-a-later-minor"),
        pytest.param((10, 0), ["sm_90", "sm_100a"], True, id="arch-specific-cubin-on-its-gpu"),
        pytest.param((3, 7), CU12X_WORKER_IMAGE_ARCH_LIST, False, id="kepler-3.7-older-than-the-build"),
        pytest.param((6, 1), CU128_ARCH_LIST, False, id="pascal-6.1-dropped-by-cu128"),
        pytest.param((12, 0), CU12X_WORKER_IMAGE_ARCH_LIST, False, id="newer-major-without-ptx"),
        pytest.param((8, 6), ["sm_75", "sm_89"], False, id="cubin-for-a-higher-minor"),
        pytest.param((8, 9), ["sm_75", "compute_90"], False, id="ptx-for-a-newer-gpu"),
        pytest.param((10, 3), ["sm_90", "sm_100a"], False, id="arch-specific-cubin-elsewhere"),
        pytest.param((8, 9), [], False, id="no-architectures"),
    ],
)
def test_require_cuda_accepts_every_gpu_the_torch_build_has_code_for(
    monkeypatch, capability, arch_list, runs
):
    worker = _import_worker(monkeypatch, "providers.chatterbox.app")
    _fake_gpu_of(worker, capability, arch_list)

    if runs:
        assert tuple(worker.common.require_cuda()) == capability
        return
    with pytest.raises(worker.common.WorkerFailure) as caught:
        worker.common.require_cuda()
    assert (caught.value.code, caught.value.status_code) == ("cuda_arch_unsupported", 503)
    assert f"sm_{capability[0]}{capability[1]}" in str(caught.value)


@pytest.mark.parametrize(
    ("capability", "arch_list", "loads"),
    [
        pytest.param((8, 9), CU128_ARCH_LIST, True, id="ada-8.9-cu128"),
        pytest.param((6, 1), CU12X_WORKER_IMAGE_ARCH_LIST, True, id="pascal-6.1-worker-image"),
        pytest.param((3, 7), CU12X_WORKER_IMAGE_ARCH_LIST, False, id="kepler-3.7-worker-image"),
    ],
)
def test_model_load_matches_what_health_reports_on_a_same_major_gpu(
    monkeypatch, capability, arch_list, loads
):
    worker = _import_worker(monkeypatch, "providers.chatterbox.app")
    monkeypatch.setattr(worker, "RESTART_AFTER_UNLOAD_S", 0)
    _fake_gpu_of(worker, capability, arch_list)
    built = []

    def build_model(model_id, _capability):
        built.append(model_id)
        return SimpleNamespace(sr=24000)

    monkeypatch.setattr(worker.runtime, "build_model", build_model)
    client = TestClient(worker.app)
    health = client.get("/health").json()
    # /health fails on a GPU no load can run on.
    assert health["status"] == ("ok" if loads else "failed"), health

    response = client.post("/v1/models/load", json={"model": "chatterbox/regular"})

    if loads:
        assert response.status_code == 200, response.text
        assert len(built) == 1
        assert worker.runtime.model_id == "chatterbox/regular"
        assert health["failure"] is None
    else:
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == "cuda_arch_unsupported"
        assert built == []
        assert worker.runtime.model is None
        # /health gives the reason a load gets.
        assert health["failure"] == response.json()["detail"]


@pytest.mark.parametrize(
    ("module_name", "provider", "model_id"),
    [
        pytest.param("providers.chatterbox.app", "chatterbox", "chatterbox/regular", id="chatterbox"),
        # The Fish Audio S2 worker is on the same contract.
        pytest.param("providers.fish.app", "fish", "fish/s2-pro", id="fish"),
    ],
)
@pytest.mark.parametrize(
    ("capability", "arch_list", "runs"),
    [
        pytest.param((12, 0), CU12X_WORKER_IMAGE_ARCH_LIST, False, id="blackwell-12.0-worker-image"),
        pytest.param((10, 0), CU12X_WORKER_IMAGE_ARCH_LIST, False, id="blackwell-10.0-worker-image"),
        pytest.param((6, 1), CU128_ARCH_LIST, False, id="pascal-6.1-cu128"),
        pytest.param((7, 5), CU12X_WORKER_IMAGE_ARCH_LIST, True, id="turing-7.5-worker-image"),
        pytest.param((8, 9), CU128_ARCH_LIST, True, id="ada-8.9-cu128"),
    ],
)
def test_health_fails_when_the_torch_build_has_no_code_for_the_gpu(
    monkeypatch, module_name, provider, model_id, capability, arch_list, runs
):
    # docs/TTS-BACKENDS.md uses worker health for
    # "can do its GPU work": a poisoned CUDA context marks health failed, and loads are gated on
    # health. On a GPU the torch build has no code for, every load answers 503
    # cuda_arch_unsupported, yet /health said ok, so Compose, the core and the web UI counted the
    # worker usable. Now /health fails with the load's code and message.
    worker = _import_worker(monkeypatch, module_name)
    if hasattr(worker, "RESTART_AFTER_UNLOAD_S"):
        monkeypatch.setattr(worker, "RESTART_AFTER_UNLOAD_S", 0)
    _fake_gpu_of(worker, capability, arch_list)
    client = TestClient(worker.app)
    backend = ExternalTTSBackend(provider=provider, base_url=f"http://{provider}:8200")
    monkeypatch.setattr(backend._opener, "open", _bridge_to_worker(client))

    health = client.get("/health").json()

    # The diagnostics stay in the gpu block either way.
    assert health["gpu"]["available"] is True
    assert health["gpu"]["capability"] == list(capability)
    assert health["gpu"]["arch_list"] == list(arch_list)
    if runs:
        assert (health["status"], health["failure"]) == ("ok", None), health
        assert backend.is_provider_available() is True
        return
    architecture = f"sm_{capability[0]}{capability[1]}"
    assert health["status"] == "failed", health
    assert health["failure"] == {
        "code": "cuda_arch_unsupported",
        "message": (
            f"Torch wheel has no code that runs on {architecture} "
            f"(built for: {' '.join(arch_list)})"
        ),
    }
    assert health["device"] == "cuda:0"
    # The core no longer offers its models (ExternalTTSBackend reads status == "ok").
    assert backend.is_provider_available() is False
    # A load still answers with the same reason.
    load = client.post("/v1/models/load", json={"model": model_id})
    assert load.status_code == 503, load.text
    assert load.json()["detail"] == health["failure"]
    assert worker.runtime.model is None


class _BridgedResponse(io.BytesIO):
    def __init__(self, data, headers):
        super().__init__(data)
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _bridge_to_worker(client: TestClient):
    """Serve ExternalTTSBackend's urllib requests from the worker's ASGI app."""

    def open_worker(request, timeout=None):
        response = client.request(
            request.get_method(),
            urllib.parse.urlsplit(request.full_url).path,
            content=request.data,
            headers=dict(request.header_items()),
        )
        if response.status_code >= 400:
            raise urllib.error.HTTPError(
                request.full_url,
                response.status_code,
                response.reason_phrase,
                response.headers,
                io.BytesIO(response.content),
            )
        return _BridgedResponse(response.content, response.headers)

    return open_worker


@pytest.mark.parametrize(
    ("module_name", "provider", "model_ids"),
    [
        (
            "providers.chatterbox.app",
            "chatterbox",
            {"chatterbox/regular", "chatterbox/turbo"},
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


@pytest.mark.parametrize(
    ("module_name", "provider", "options", "refused", "accepted"),
    [
        pytest.param(
            "providers.chatterbox.app",
            "chatterbox",
            {},
            [("chatterbox/turbo", "ko"), ("chatterbox/regular", "fr")],
            [("chatterbox/turbo", "en"), ("chatterbox/regular", "en-GB")],
            id="chatterbox",
        ),
        pytest.param(
            "providers.fish.app",
            "fish",
            {"clone_transcript": "Exact words."},
            # The model card's tiers 1 and 2, and no other language.
            [("fish/s2-pro", "it"), ("fish/s2-pro", "hi-IN")],
            [("fish/s2-pro", "en-US"), ("fish/s2-pro", "ja"), ("fish/s2-pro", "pt-BR")],
            id="fish",
        ),
    ],
)
def test_worker_models_refuse_a_language_their_manifest_does_not_declare(
    monkeypatch, module_name, provider, options, refused, accepted
):
    # README.md (POST /v1/audio/speech): a language hint "the model
    # cannot speak ... gets 400 before synthesis", and Live Reader answers such a session.update
    # with invalid_event; both run this check. Neither worker reads the hint, so Korean on
    # Chatterbox (English only) was synthesized with the default handling.
    from src.services.tts import validate_tts_feature_support
    from src.tts.router import TTSRouter

    worker = _import_worker(monkeypatch, module_name)
    monkeypatch.setattr("src.tts.router._discover_backends", lambda *_args, **_kwargs: {})
    router = TTSRouter(device="cpu", external_providers="")
    backend = ExternalTTSBackend(provider=provider, base_url=f"http://{provider}:8210")
    bridge = _bridge_to_worker(TestClient(worker.app))
    requested: list[str] = []

    def open_worker(request, timeout=None):
        requested.append(urllib.parse.urlsplit(request.full_url).path)
        return bridge(request, timeout=timeout)

    monkeypatch.setattr(backend._opener, "open", open_worker)
    router.register_backend(provider, backend)

    def check(model_id, language):
        return validate_tts_feature_support(
            tts_router=router,
            model_id=model_id,
            reference_audio=b"reference",
            language=language,
            **options,
        )

    for model_id, language in refused:
        error = check(model_id, language)
        assert error is not None, f"{model_id} accepted language={language!r}"
        assert error.startswith(f"Unsupported language '{language}' for {model_id}."), error
    for model_id, language in accepted:
        assert check(model_id, language) is None, (model_id, language)
    # Only the manifest was read: nothing was loaded or spoken.
    assert set(requested) == {"/v1/manifest"}


@pytest.mark.parametrize(
    ("module_name", "model_id"),
    [
        ("providers.chatterbox.app", "chatterbox/regular"),
        ("providers.fish.app", "fish/s2-pro"),
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


def test_chatterbox_reuses_reference_conditionals_until_unload(monkeypatch):
    worker = _import_worker(monkeypatch, "providers.chatterbox.app")
    first, second = _wav_base64(duration=2.0), _wav_base64(duration=3.0)
    prepared = []
    used = []

    class Model:
        conds = None

        def generate(self, text, audio_prompt_path=None):
            # Like Chatterbox: a prompt path rebuilds self.conds, otherwise the current
            # conds are used.
            if audio_prompt_path:
                with open(audio_prompt_path, "rb") as reference:
                    self.conds = SimpleNamespace(audio=reference.read())
                prepared.append(text)
            used.append((text, self.conds.audio))
            return np.zeros(4, dtype=np.float32)

    def say(text, reference):
        worker.runtime.generate(
            worker.SynthesisRequest(model="chatterbox/regular", text=text, reference_audio=reference)
        )

    worker.runtime.model = Model()
    worker.runtime.model_id = "chatterbox/regular"
    for text, reference in [("One.", first), ("Two.", first), ("Three.", second), ("Four.", first)]:
        say(text, reference)

    assert prepared == ["One.", "Three."]
    first_audio, second_audio = base64.b64decode(first), base64.b64decode(second)
    assert used == [
        ("One.", first_audio),
        ("Two.", first_audio),
        ("Three.", second_audio),
        ("Four.", first_audio),
    ]

    worker.runtime.unload()
    assert len(worker.runtime.references) == 0
    worker.runtime.model = Model()
    worker.runtime.model_id = "chatterbox/regular"
    say("Five.", first)
    assert prepared == ["One.", "Three.", "Five."]


def test_chatterbox_caches_only_conditionals_built_from_the_reference(monkeypatch):
    # A Chatterbox that reads the prompt without storing it in model.conds leaves the
    # built-in voice there; caching that would speak later requests in the wrong voice.
    worker = _import_worker(monkeypatch, "providers.chatterbox.app")
    builtin = SimpleNamespace(audio=b"built-in voice")
    prompts = []

    class Model:
        conds = builtin

        def generate(self, text, audio_prompt_path=None):
            prompts.append(audio_prompt_path is not None)
            return np.zeros(4, dtype=np.float32)

    worker.runtime.model = Model()
    worker.runtime.model_id = "chatterbox/regular"
    reference = _wav_base64(duration=2.0)
    for text in ("One.", "Two."):
        worker.runtime.generate(
            worker.SynthesisRequest(model="chatterbox/regular", text=text, reference_audio=reference)
        )

    assert prompts == [True, True]
    assert len(worker.runtime.references) == 0


@pytest.mark.parametrize(
    ("module_name", "model_id", "message"),
    [
        (
            "providers.chatterbox.app",
            "chatterbox/regular",
            "CUDA out of memory during Chatterbox synthesis",
        ),
    ],
)
def test_out_of_memory_during_synthesis_is_a_typed_507_and_the_worker_stays_healthy(
    monkeypatch, module_name, model_id, message
):
    worker = _import_worker(monkeypatch, module_name)
    emptied = _fake_gpu(worker)
    attempts = []

    def audio_after_one_out_of_memory():
        attempts.append(True)
        if len(attempts) == 1:
            raise worker.torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB")
        return np.array([0.25, -0.25], dtype=np.float32)

    class Model:
        def generate(self, _text, audio_prompt_path=None):
            return audio_after_one_out_of_memory()

    worker.runtime.model = Model()
    worker.runtime.model_id = model_id
    client = TestClient(worker.app)
    body = {
        "model": model_id,
        "text": "Hello there.",
        "reference_audio": _wav_base64(duration=2.0),
        "clone_transcript": "Exact words.",
    }

    response = client.post("/v1/audio/speech", json=body)

    assert response.status_code == 507
    assert response.json() == {"error": {"code": "out_of_memory", "message": message}}
    assert emptied
    health = client.get("/health").json()
    assert (health["status"], health["loaded_model"], health["busy"]) == ("ok", model_id, False)
    retry = client.post("/v1/audio/speech", json=body)
    assert retry.status_code == 200
    assert np.frombuffer(retry.content, dtype="<f4").tolist() == pytest.approx([0.25, -0.25])


def test_out_of_memory_while_loading_is_a_typed_507_and_leaves_the_worker_empty(monkeypatch):
    worker = _import_worker(monkeypatch, "providers.chatterbox.app")
    monkeypatch.setattr(worker, "RESTART_AFTER_UNLOAD_S", 0)
    emptied = _fake_gpu(worker)

    def build_model(_model_id, _capability):
        # Some CUDA paths raise a plain RuntimeError with the allocator's message.
        raise RuntimeError("CUDA out of memory. Tried to allocate 1.50 GiB")

    monkeypatch.setattr(worker.runtime, "build_model", build_model)
    client = TestClient(worker.app)

    response = client.post("/v1/models/load", json={"model": "chatterbox/regular"})

    assert response.status_code == 507
    assert response.json() == {
        "error": {"code": "out_of_memory", "message": "CUDA out of memory while loading Chatterbox"}
    }
    assert emptied
    assert worker.runtime.model is None
    health = client.get("/health").json()
    assert (health["status"], health["loaded_model"], health["busy"]) == ("ok", None, False)


def test_release_memory_frees_the_cublas_workspaces_before_emptying_the_cache(monkeypatch):
    # A cuBLAS workspace is a live block, and empty_cache()
    # keeps every segment holding one, so an unloaded model's segment stayed reserved.
    worker = _import_worker(monkeypatch, "providers.chatterbox.app")
    calls = []
    worker.torch.cuda.is_available = lambda: True
    worker.torch.cuda.empty_cache = lambda: calls.append("empty_cache")
    worker.torch._C = SimpleNamespace(_cuda_clearCublasWorkspaces=lambda: calls.append("clear"))

    worker.common.release_memory()
    assert calls == ["clear", "empty_cache"]

    # A torch without the private hook still empties the cache.
    calls.clear()
    worker.torch._C = SimpleNamespace()
    worker.common.release_memory()
    assert calls == ["empty_cache"]

    calls.clear()
    worker.torch.cuda.is_available = lambda: False
    worker.common.release_memory()
    assert calls == []


def test_core_reports_worker_out_of_memory_as_insufficient_gpu_memory(monkeypatch):
    worker = _import_worker(monkeypatch, "providers.chatterbox.app")
    _fake_gpu(worker)

    class Model:
        def generate(self, _text, audio_prompt_path=None):
            raise worker.torch.cuda.OutOfMemoryError("CUDA out of memory.")

    worker.runtime.model = Model()
    worker.runtime.model_id = "chatterbox/regular"
    backend = ExternalTTSBackend(provider="chatterbox", base_url="http://chatterbox:8210")
    monkeypatch.setattr(backend._opener, "open", _bridge_to_worker(TestClient(worker.app)))

    with pytest.raises(ExternalProviderError) as caught:
        list(
            backend.synthesize(
                "Hello",
                "reference",
                model_id="chatterbox/regular",
                reference_audio=base64.b64decode(_wav_base64(duration=2.0)),
            )
        )

    assert (caught.value.status_code, caught.value.code) == (503, "insufficient_gpu_memory")


@pytest.mark.parametrize("name", ["chatterbox", "qwen3", "fish"])
def test_worker_images_build_from_providers_with_the_shared_module(name):
    compose = yaml.safe_load((ROOT / "docker-compose.voice-models.yml").read_text(encoding="utf-8"))
    build = compose["services"][name]["build"]
    context = ROOT / build["context"]
    dockerfile = (context / build["dockerfile"]).read_text(encoding="utf-8").splitlines()

    copies = [line.split()[1:] for line in dockerfile if line.startswith("COPY ")]
    for *sources, _destination in copies:
        for source in sources:
            assert (context / source).exists(), source
    assert ["_common/", "providers/_common/"] in copies
    assert [f"{name}/app.py", f"providers/{name}/"] in copies
    command = json.loads(next(line for line in dockerfile if line.startswith("CMD "))[4:])
    assert command[:2] == ["uvicorn", f"providers.{name}.app:app"]
    assert importlib.util.find_spec(f"providers.{name}.app") is not None


# ── The Fish Audio S2 Pro worker (the NF4 fork of fish-speech) ───────────────────────────────────────────────────
# What an optional provider must do: it loads, speaks a sentence and refuses what it can't do. The fork is faked at
# its boundary (huggingface_hub, tools.server.model_manager, fish_speech.utils.schema and the CPU reference codec), so
# no torch, fork or weights are needed.

FISH_MODEL = "fish/s2-pro"
FISH_SNAPSHOT = "/models/fish-s2-pro-nf4"
FISH_TRANSCRIPT = "These are the exact words of the recording."


class _FakeFishEngine:
    """The fork's TTSInferenceEngine: a header, then one ``final`` result (or an ``error`` one) per request."""

    def __init__(self, *, sample_rate=44100, seconds_per_char=0.01, error=None):
        self.decoder_model = SimpleNamespace(sample_rate=sample_rate)
        self.ref_by_id: dict = {}
        self.requests: list = []
        self.sample_rate = sample_rate
        self.seconds_per_char = seconds_per_char
        self.error = error

    def samples_for(self, text: str) -> int:
        return max(1, round(len(text) * self.seconds_per_char * self.sample_rate))

    def inference(self, request):
        self.requests.append(request)
        yield SimpleNamespace(code="header", audio=None, error=None)
        if self.error is not None:
            yield SimpleNamespace(code="error", audio=None, error=self.error)
            return
        audio = np.full(self.samples_for(request.text), 0.1, dtype=np.float32)
        yield SimpleNamespace(code="final", audio=(self.sample_rate, audio), error=None)


def _fish(monkeypatch, tmp_path, *, engine=None, manager_error=None):
    """The Fish worker with a fake GPU and a fake fork; returns (worker, client, engine, calls, kills)."""
    worker = _import_worker(monkeypatch, "providers.fish.app")
    _fake_gpu(worker)
    engine = engine or _FakeFishEngine()
    calls: dict[str, list] = {"snapshot": [], "manager": [], "codec": [], "encoded": []}

    def snapshot_download(**kwargs):
        calls["snapshot"].append(kwargs)
        return FISH_SNAPSHOT

    class ModelManager:
        def __init__(self, **kwargs):
            calls["manager"].append(kwargs)
            if manager_error is not None:
                raise manager_error
            self.tts_inference_engine = engine

    def encode_reference(codec, wav_path, tokens_path):
        wav_path = Path(wav_path)
        calls["encoded"].append(
            (codec, wav_path.read_bytes(), wav_path.with_suffix(".lab").read_text(encoding="utf-8"), Path(tokens_path))
        )
        return f"tokens-{len(calls['encoded'])}"

    def load_reference_codec(snapshot):
        calls["codec"].append(snapshot)
        return "cpu-codec"

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))
    model_manager = types.ModuleType("tools.server.model_manager")
    model_manager.ModelManager = ModelManager
    schema = types.ModuleType("fish_speech.utils.schema")
    schema.ServeTTSRequest = lambda **fields: SimpleNamespace(**fields)
    for name, module in (
        ("tools", types.ModuleType("tools")),
        ("tools.server", types.ModuleType("tools.server")),
        ("tools.server.model_manager", model_manager),
        ("fish_speech", types.ModuleType("fish_speech")),
        ("fish_speech.utils", types.ModuleType("fish_speech.utils")),
        ("fish_speech.utils.schema", schema),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(worker, "_load_reference_codec", load_reference_codec)
    monkeypatch.setattr(worker, "_encode_reference", encode_reference)
    monkeypatch.setattr(worker, "REFERENCES_DIR", tmp_path / "references")
    # The restart after an unload is tested on its own, with a short wait.
    monkeypatch.setattr(worker, "RESTART_AFTER_UNLOAD_S", 0)
    kills: list = []
    monkeypatch.setattr(worker.os, "kill", lambda _pid, sig: kills.append(sig))
    return worker, TestClient(worker.app), engine, calls, kills


def _fish_body(**changes) -> dict:
    body = {
        "model": FISH_MODEL,
        "text": "Hello from a cloned voice.",
        "reference_audio": _wav_base64(duration=2.0),
        "clone_transcript": FISH_TRANSCRIPT,
    }
    body.update(changes)
    return {key: value for key, value in body.items() if value is not None}


def test_fish_manifest_offers_one_cloning_model_with_no_voices(monkeypatch):
    worker = _import_worker(monkeypatch, "providers.fish.app")
    backend = ExternalTTSBackend(provider="fish", base_url="http://fish.voice-workers.internal.:8230")
    monkeypatch.setattr(backend._opener, "open", _bridge_to_worker(TestClient(worker.app)))

    assert (worker.MANIFEST["schema_version"], worker.MANIFEST["provider"]) == (1, "fish")
    [model] = worker.MANIFEST["models"]
    assert (model["id"], model["provider"], model["sample_rate"], model["max_input_chars"]) == (
        FISH_MODEL, "fish", 44100, 1500)
    assert model["revision"] == "279f07f238504911b6724157cc9f07fb7e0a23d2" == worker.MODEL_REVISION
    assert model["voices"] == [] and backend.list_voices(FISH_MODEL) == []
    assert backend.get_sample_rate(FISH_MODEL) == 44100
    capabilities = backend.get_capabilities(FISH_MODEL)
    assert {name: capabilities[name] for name in (
        "voice_clone", "reference_audio", "clone_transcript", "clone_transcript_required", "voice_blend",
        "voice_design", "instructions", "speed_control", "speed_mode", "streaming", "cancellation", "live_reader",
        "speakers")} == {
        "voice_clone": True, "reference_audio": True, "clone_transcript": True, "clone_transcript_required": True,
        "voice_blend": False, "voice_design": False, "instructions": False, "speed_control": False,
        "speed_mode": "none", "streaming": True, "cancellation": "between_segments", "live_reader": True,
        "speakers": []}
    assert sorted(capabilities["languages"]) == sorted(["en", "zh", "ja", "ko", "es", "pt", "ar", "ru", "fr", "de"])


def test_fish_loads_the_pinned_nf4_snapshot_in_half_precision_with_a_decode_only_codec(monkeypatch, tmp_path):
    worker, client, _engine, calls, kills = _fish(monkeypatch, tmp_path)

    response = client.post("/v1/models/load", json={"model": FISH_MODEL})

    assert response.status_code == 200, response.text
    [snapshot] = calls["snapshot"]
    assert (snapshot["repo_id"], snapshot["revision"]) == (
        "scarxity/fish-speech-s2-pro-nf4", "279f07f238504911b6724157cc9f07fb7e0a23d2")
    assert {"model.pth", "codec.pth", "tokenizer.json", "config.json"} <= set(snapshot["allow_patterns"])
    assert calls["manager"] == [{
        "mode": "tts", "device": "cuda", "half": True, "compile": True, "bnb4": True,
        "llama_checkpoint_path": FISH_SNAPSHOT, "decoder_checkpoint_path": str(Path(FISH_SNAPSHOT) / "codec.pth"),
        "decoder_config_name": "modded_dac_vq", "max_seq_len": 2048, "lazy_load": False, "codec_decode_only": True,
    }]
    health = client.get("/health").json()
    assert (health["status"], health["provider"], health["loaded_model"], health["busy"]) == (
        "ok", "fish", FISH_MODEL, False)
    assert {name: health[name] for name in ("quantization", "precision", "codec", "reference_encoder")} == {
        "quantization": "nf4", "precision": "float16", "codec": "decode-only", "reference_encoder": "cpu"}
    assert re.fullmatch(r"[0-9a-f]{40}", health["fork_commit"])
    assert calls["codec"] == [], "the CPU reference codec loads with the first recording, not with the model"
    # Loading the loaded model again is a no-op.
    assert client.post("/v1/models/load", json={"model": FISH_MODEL}).status_code == 200
    assert len(calls["manager"]) == 1 and kills == []


@pytest.mark.parametrize(
    ("engine_rate", "manager_error", "status", "code"),
    [
        pytest.param(24000, None, 500, "sample_rate_mismatch", id="sample-rate"),
        pytest.param(
            44100,
            type("InductorError", (RuntimeError,), {"__module__": "torch._inductor.exc"})("backend compiler failed"),
            503,
            "compile_failed",
            id="torch-compile",
        ),
        pytest.param(44100, RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"), 507, "out_of_memory",
                     id="out-of-memory"),
    ],
)
def test_fish_failed_load_is_typed_and_leaves_no_model(monkeypatch, tmp_path, engine_rate, manager_error, status,
                                                       code):
    worker, client, _engine, _calls, _kills = _fish(
        monkeypatch, tmp_path, engine=_FakeFishEngine(sample_rate=engine_rate), manager_error=manager_error)

    response = client.post("/v1/models/load", json={"model": FISH_MODEL})

    assert response.status_code == status, response.text
    body = response.json()
    detail = body["error"] if status == 507 else body["detail"]
    assert detail["code"] == code
    if code == "compile_failed":
        assert "FISH_COMPILE=false" in detail["message"]
    assert worker.runtime.model is None and worker.runtime.manager is None
    assert client.get("/health").json()["loaded_model"] is None


@pytest.mark.asyncio
async def test_fish_restarts_after_an_unload_and_refuses_a_load_until_then(monkeypatch, tmp_path):
    worker, _client, _engine, calls, kills = _fish(monkeypatch, tmp_path)
    monkeypatch.setattr(worker, "RESTART_AFTER_UNLOAD_S", 0.05)
    await worker.load_model(worker.LoadRequest(model=FISH_MODEL))

    await worker.unload_model(worker.LoadRequest(model=FISH_MODEL))
    # The fork's thread still holds the unloaded model on the GPU: a second one would not fit beside it.
    with pytest.raises(worker.HTTPException) as caught:
        await worker.load_model(worker.LoadRequest(model=FISH_MODEL))
    assert (caught.value.status_code, caught.value.detail["code"]) == (503, "worker_restarting")
    assert (await worker.health())["restart_pending"] is True
    await asyncio.wait_for(worker.restart_task, 5)

    assert kills == [worker.signal.SIGTERM]
    assert worker.runtime.model is None and len(calls["manager"]) == 1


@pytest.mark.asyncio
async def test_fish_restarts_after_a_failed_load_but_not_when_turned_off(monkeypatch, tmp_path):
    worker, _client, _engine, _calls, kills = _fish(
        monkeypatch, tmp_path, manager_error=RuntimeError("CUDA out of memory."))
    monkeypatch.setattr(worker, "RESTART_AFTER_UNLOAD_S", 0.01)
    with pytest.raises(worker.HTTPException):
        await worker.load_model(worker.LoadRequest(model=FISH_MODEL))
    await asyncio.wait_for(worker.restart_task, 5)
    assert kills == [worker.signal.SIGTERM]

    worker, _client, _engine, _calls, kills = _fish(monkeypatch, tmp_path)
    await worker.load_model(worker.LoadRequest(model=FISH_MODEL))
    await worker.unload_model(worker.LoadRequest(model=FISH_MODEL))  # FISH_RESTART_AFTER_UNLOAD_S=0
    assert worker.restart_task is None and kills == []


def test_fish_restarts_one_second_after_an_unload_by_default(monkeypatch):
    monkeypatch.delenv("FISH_RESTART_AFTER_UNLOAD_S", raising=False)
    worker = _import_worker(monkeypatch, "providers.fish.app")
    assert worker.RESTART_AFTER_UNLOAD_S == 1.0


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        pytest.param({"speed": 1.5}, "speed_unsupported", id="speed"),
        pytest.param({"instructions": "Whisper it."}, "instructions_unsupported", id="instructions"),
        pytest.param({"voice_design": "A deep, warm voice."}, "instructions_unsupported", id="voice-design"),
        pytest.param({"language": "xx"}, "language_unsupported", id="language"),
        pytest.param({"reference_audio": None}, "reference_audio_required", id="no-reference"),
        pytest.param({"clone_transcript": None}, "clone_transcript_required", id="no-transcript"),
        pytest.param({"clone_transcript": "   "}, "clone_transcript_required", id="blank-transcript"),
        pytest.param({"reference_audio": _wav_base64(duration=45.0)}, "reference_audio_too_long",
                     id="45-second-reference"),
    ],
)
def test_fish_refuses_what_it_cannot_do_before_any_generation(monkeypatch, tmp_path, changes, code):
    worker, client, engine, calls, _kills = _fish(monkeypatch, tmp_path)
    assert client.post("/v1/models/load", json={"model": FISH_MODEL}).status_code == 200

    response = client.post("/v1/audio/speech", json=_fish_body(**changes))

    assert response.status_code == 400, response.text
    assert response.json()["detail"]["code"] == code
    assert engine.requests == [] and calls["encoded"] == [], "nothing was encoded or generated"
    health = client.get("/health").json()
    assert (health["loaded_model"], health["busy"]) == (FISH_MODEL, False)
    # The same request within the limits is spoken.
    assert client.post("/v1/audio/speech", json=_fish_body()).status_code == 200


def test_fish_takes_a_30_second_reference_and_refuses_1501_characters(monkeypatch, tmp_path):
    worker, client, engine, _calls, _kills = _fish(monkeypatch, tmp_path)
    assert client.post("/v1/models/load", json={"model": FISH_MODEL}).status_code == 200
    backend = ExternalTTSBackend(provider="fish", base_url="http://fish.voice-workers.internal.:8230")
    monkeypatch.setattr(backend._opener, "open", _bridge_to_worker(client))

    assert client.post("/v1/audio/speech", json=_fish_body(reference_audio=_wav_base64(duration=30.0))).status_code \
        == 200
    assert client.post("/v1/audio/speech", json=_fish_body(text="a" * 1501)).status_code == 422
    with pytest.raises(ValueError, match=r"Input too long for fish/s2-pro\. Max: 1500 characters"):
        list(backend.synthesize("a" * 1501, "reference", model_id=FISH_MODEL, reference_audio=b"RIFF",
                                clone_transcript=FISH_TRANSCRIPT))
    with pytest.raises(ValueError, match=r"Unsupported language 'xx' for fish/s2-pro"):
        backend.validate_language("xx", FISH_MODEL)
    assert len(engine.requests) == 1


def test_fish_speaks_a_cloned_sentence_through_the_core(monkeypatch, tmp_path):
    worker, client, engine, calls, _kills = _fish(monkeypatch, tmp_path)
    assert client.post("/v1/models/load", json={"model": FISH_MODEL}).status_code == 200
    backend = ExternalTTSBackend(provider="fish", base_url="http://fish.voice-workers.internal.:8230")
    monkeypatch.setattr(backend._opener, "open", _bridge_to_worker(client))
    reference = base64.b64decode(_wav_base64(duration=3.0))
    text = "Hello from the cloned voice."

    audio = np.concatenate(list(backend.synthesize(
        text, "reference", model_id=FISH_MODEL, reference_audio=reference, clone_transcript=FISH_TRANSCRIPT)))

    assert audio.dtype == np.float32 and audio.size == engine.samples_for(text)
    assert audio.tolist() == pytest.approx([0.1] * audio.size)
    key = worker.runtime.references.key(FISH_MODEL, reference, FISH_TRANSCRIPT)
    [request] = engine.requests
    assert vars(request) == {
        "text": text, "reference_id": key, "use_memory_cache": "on", "streaming": False, "format": "wav",
        "max_new_tokens": min(1024, math.ceil(len(text) * 2.2) + 64), "chunk_length": 200,
        "top_p": 0.7, "temperature": 0.7, "seed": worker.SEED,
    }
    # The recording was encoded once, on the CPU codec, and handed to the engine with its transcript.
    folder = tmp_path / "references" / key
    assert calls["codec"] == [FISH_SNAPSHOT]
    assert calls["encoded"] == [("cpu-codec", reference, FISH_TRANSCRIPT, folder / "ref.tokens.pt")]
    assert engine.ref_by_id[key] == (["tokens-1"], [FISH_TRANSCRIPT])
    assert (folder / "ref.wav").read_bytes() == reference
    assert (folder / "ref.lab").read_text(encoding="utf-8") == FISH_TRANSCRIPT

    # The next sentence in the same voice reuses the tokens; another transcript is another reference.
    list(backend.synthesize("A second sentence.", "reference", model_id=FISH_MODEL, reference_audio=reference,
                            clone_transcript=FISH_TRANSCRIPT))
    list(backend.synthesize("A third sentence.", "reference", model_id=FISH_MODEL, reference_audio=reference,
                            clone_transcript="Other words."))
    assert len(calls["encoded"]) == 2 and calls["codec"] == [FISH_SNAPSHOT]
    assert [item.reference_id for item in engine.requests][:2] == [key, key]
    assert engine.requests[2].reference_id != key

    # An unload forgets the references, their files and the CPU codec.
    assert client.post("/v1/models/unload", json={"model": FISH_MODEL}).status_code == 200
    assert not folder.exists() and list((tmp_path / "references").iterdir()) == []
    assert (worker.runtime.model, worker.runtime.codec, len(worker.runtime.references)) == (None, None, 0)


def test_fish_evicts_the_oldest_reference_with_its_files(monkeypatch, tmp_path):
    worker, client, engine, _calls, _kills = _fish(monkeypatch, tmp_path)
    monkeypatch.setattr(worker.runtime.references, "size", 1)
    assert client.post("/v1/models/load", json={"model": FISH_MODEL}).status_code == 200

    assert client.post("/v1/audio/speech", json=_fish_body(clone_transcript="First words.")).status_code == 200
    first = engine.requests[0].reference_id
    assert client.post("/v1/audio/speech", json=_fish_body(clone_transcript="Second words.")).status_code == 200
    second = engine.requests[1].reference_id

    assert first != second and set(engine.ref_by_id) == {second}
    assert sorted(path.name for path in (tmp_path / "references").iterdir()) == [second]


def test_fish_speaks_a_long_text_in_segments_of_whole_sentences_each_with_its_token_budget(monkeypatch, tmp_path):
    worker, client, engine, _calls, _kills = _fish(monkeypatch, tmp_path)
    assert client.post("/v1/models/load", json={"model": FISH_MODEL}).status_code == 200
    sentences = [f"Sentence number {index} has a few more words in it." for index in range(30)]
    text = " ".join(sentences)
    assert len(text) <= 1500

    response = client.post("/v1/audio/speech", json=_fish_body(text=text))

    assert response.status_code == 200, response.text
    assert response.headers["X-Audio-Sample-Rate"] == "44100"
    assert response.headers["X-Cancellation-Granularity"] == "between_segments"
    segments = [request.text for request in engine.requests]
    assert len(segments) > 1 and all(len(segment) <= 300 for segment in segments)
    assert " ".join(segments) == text and all(segment.endswith(".") for segment in segments)
    assert [request.max_new_tokens for request in engine.requests] == [
        min(1024, math.ceil(len(segment) * 2.2) + 64) for segment in segments]
    assert {request.reference_id for request in engine.requests} == {engine.requests[0].reference_id}
    assert np.frombuffer(response.content, dtype="<f4").size == sum(engine.samples_for(s) for s in segments)

    # A sentence longer than a segment is cut between words.
    engine.requests.clear()
    words = " ".join(["word"] * 140)
    assert client.post("/v1/audio/speech", json=_fish_body(text=words)).status_code == 200
    pieces = [request.text for request in engine.requests]
    assert len(pieces) == 3 and all(len(piece) <= 300 for piece in pieces) and " ".join(pieces) == words


def test_fish_gives_wide_characters_the_budget_of_their_speech_time(monkeypatch):
    # A Chinese character takes about as long to say as three Latin ones; counted as one, a 300-character Chinese
    # segment would have a third of the tokens it needs and end at the generation limit.
    worker = _import_worker(monkeypatch, "providers.fish.app")
    chinese = "今天天气很好我们去公园散步吧" * 10  # 140 wide characters, no spaces or sentence ends

    segments = worker._segments(chinese)

    assert "".join(segments) == chinese
    assert all(len(segment) * 3.2 <= 300 for segment in segments)
    assert worker._max_new_tokens("你好") == math.ceil(2 * 3.2 * 2.2) + 64
    assert worker._max_new_tokens(segments[0]) <= 1024


@pytest.mark.parametrize(
    ("engine", "status", "code"),
    [
        pytest.param(_FakeFishEngine(seconds_per_char=1.0), 500, "generation_limit_reached", id="token-limit"),
        pytest.param(_FakeFishEngine(error=RuntimeError("decoder exploded")), 500, "generation_failed", id="error"),
    ],
)
def test_fish_generation_failures_are_typed_and_the_worker_stays_usable(monkeypatch, tmp_path, engine, status, code):
    worker, client, engine, _calls, _kills = _fish(monkeypatch, tmp_path, engine=engine)
    assert client.post("/v1/models/load", json={"model": FISH_MODEL}).status_code == 200

    response = client.post("/v1/audio/speech", json=_fish_body(text="Hello there."))

    assert response.status_code == status, response.text
    assert response.json()["detail"]["code"] == code
    health = client.get("/health").json()
    assert (health["status"], health["loaded_model"], health["busy"]) == ("ok", FISH_MODEL, False)


def test_fish_out_of_memory_during_synthesis_is_a_typed_507(monkeypatch, tmp_path):
    worker, client, engine, _calls, _kills = _fish(monkeypatch, tmp_path)
    assert client.post("/v1/models/load", json={"model": FISH_MODEL}).status_code == 200
    engine.error = worker.torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1 GiB")

    response = client.post("/v1/audio/speech", json=_fish_body())

    assert response.status_code == 507
    assert response.json() == {"error": {"code": "out_of_memory", "message": worker.OUT_OF_MEMORY_MESSAGE}}
    engine.error = None
    assert client.post("/v1/audio/speech", json=_fish_body()).status_code == 200


# ── Worker contract: the GPU memory a model needs and holds (the core's GPU memory guard reads both) ────────────────


def _import_any_worker(monkeypatch, module_name: str):
    """A worker with torch stubbed, the Qwen3 dtypes included, and Qwen3 Base shown as the voice-models file does."""
    monkeypatch.setenv("QWEN3_ENABLE_BASE", "true")
    if module_name != "providers.qwen3.app":
        return _import_worker(monkeypatch, module_name)
    _import_worker(monkeypatch, "providers._common.worker")  # installs the torch stub
    torch_stub = sys.modules["torch"]
    torch_stub.float16, torch_stub.float32, torch_stub.bfloat16 = object(), object(), object()
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.mark.parametrize(
    ("module_name", "provider", "model_id"),
    [
        ("providers.chatterbox.app", "chatterbox", "chatterbox/turbo"),
        ("providers.fish.app", "fish", "fish/s2-pro"),
        ("providers.qwen3.app", "qwen3", "qwen3/0.6b-custom-voice"),
    ],
)
def test_worker_health_reports_the_gpu_memory_its_model_holds_and_needs(monkeypatch, module_name, provider, model_id):
    worker = _import_any_worker(monkeypatch, module_name)
    _fake_gpu(worker)
    worker.torch.cuda.memory_reserved = lambda _index: 2900 * 1024 * 1024
    client = TestClient(worker.app)
    assert client.get("/health").json()["gpu_memory"] == {"model_mb": None, "need_mb": None}

    worker.runtime.model, worker.runtime.model_id = object(), model_id
    health = client.get("/health").json()
    manifest = {model["id"]: model for model in client.get("/v1/manifest").json()["models"]}

    assert health["gpu_memory"] == {"model_mb": 2900.0, "need_mb": manifest[model_id]["gpu_memory_mb"]}
    backend = ExternalTTSBackend(provider=provider, base_url=f"http://{provider}:8200")
    monkeypatch.setattr(backend._opener, "open", _bridge_to_worker(client))
    assert backend.gpu_memory_need_mb(model_id) == manifest[model_id]["gpu_memory_mb"]
    assert [(model.model, model.gpu_memory_mb) for model in backend.loaded_models()] == [(model_id, 2900.0)]


def test_each_worker_model_states_the_gpu_memory_the_registry_gives_it(monkeypatch):
    from src.model_registry import get_known_model

    stated = {}
    for module_name in ("providers.chatterbox.app", "providers.fish.app", "providers.qwen3.app"):
        worker = _import_any_worker(monkeypatch, module_name)
        stated.update({model["id"]: model["gpu_memory_mb"] for model in worker.MANIFEST["models"]})
    assert set(stated) == {"chatterbox/regular", "chatterbox/turbo", "fish/s2-pro", "qwen3/0.6b-custom-voice",
                           "qwen3/0.6b-base"}
    assert stated == {model_id: get_known_model(model_id)["gpu_mb"] for model_id in stated}
