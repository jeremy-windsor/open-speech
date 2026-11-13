"""Contract tests for the isolated Chatterbox and CosyVoice workers and their shared scaffolding."""

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
import threading
import time
import types
import unicodedata
import urllib.error
import urllib.parse
import uuid
import wave
from concurrent.futures import ThreadPoolExecutor
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
    worker = _import_worker(monkeypatch, "providers.cosyvoice.app")
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
    worker = _import_worker(monkeypatch, "providers.cosyvoice.app")
    _fake_gpu_of(worker, capability, arch_list)
    built = []

    def auto_model(**kwargs):
        built.append(kwargs)
        return SimpleNamespace(sample_rate=24000, frontend=SimpleNamespace(text_frontend="wetext"))

    _stub_cosyvoice_library(monkeypatch, auto_model)
    client = TestClient(worker.app)
    health = client.get("/health").json()
    # /health fails on a GPU no load can run on.
    assert health["status"] == ("ok" if loads else "failed"), health

    response = client.post("/v1/models/load", json={"model": "cosyvoice/2-0.5b"})

    if loads:
        assert response.status_code == 200, response.text
        assert len(built) == 1
        assert worker.runtime.model_id == "cosyvoice/2-0.5b"
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
        pytest.param("providers.cosyvoice.app", "cosyvoice", "cosyvoice/2-0.5b", id="cosyvoice"),
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


def test_cosyvoice_languages_are_declared_per_model(monkeypatch):
    # docs/TTS-BACKENDS.md: the manifest "declares capabilities, voices, language support ... per
    # model". CosyVoice 2 speaks Chinese, English and Korean; German, French, Russian, Spanish and
    # Italian came with Fun-CosyVoice3, so the core must not offer them for 2-0.5b.
    # Neither model lists
    # Japanese, which the pinned library reads only as katakana and the image can't produce.
    worker = _import_worker(monkeypatch, "providers.cosyvoice.app")
    backend = ExternalTTSBackend(provider="cosyvoice", base_url="http://cosyvoice:8220")
    monkeypatch.setattr(backend._opener, "open", _bridge_to_worker(TestClient(worker.app)))
    cosyvoice3_only = {"de", "fr", "ru", "es", "it"}

    cosyvoice2 = set(backend.get_capabilities("cosyvoice/2-0.5b")["languages"])
    cosyvoice3 = set(backend.get_capabilities("cosyvoice/3-0.5b")["languages"])

    assert cosyvoice2 == {"zh", "en", "ko"}
    assert cosyvoice3 == {"zh", "en", "ko"} | cosyvoice3_only


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
            "providers.cosyvoice.app",
            "cosyvoice",
            {"clone_transcript": "Exact words."},
            # "ja" is refused on both models.
            [("cosyvoice/2-0.5b", "de"), ("cosyvoice/2-0.5b", "ja"), ("cosyvoice/3-0.5b", "ja-JP")],
            [("cosyvoice/2-0.5b", "ko"), ("cosyvoice/2-0.5b", "zh-CN"), ("cosyvoice/3-0.5b", "de")],
            id="cosyvoice",
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
    # Chatterbox (English only) or German on CosyVoice 2 was synthesized with the default handling.
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


class _HopKeepingCosyVoiceEngine:
    """CosyVoice2Model's streaming chunk size at the pinned commit 074ca6dc
    (cosyvoice/cli/model.py:258 and 345-362), with the LLM adding 5 tokens per 0.1 s poll.

    The hop lives on the instance and doubles after each streamed chunk, up to 100 tokens.
    """

    pre_lookahead_len = 3  # cosyvoice/flow/flow.py

    def __init__(self) -> None:
        self.token_hop_len = 25
        self.token_max_hop_len = 4 * self.token_hop_len
        self.stream_scale_factor = 2
        self.requests: list[dict] = []

    def tts(self, *, prompt_tokens: int, speech_tokens: int, stream: bool):
        record = {"start_hop": self.token_hop_len, "first_chunk_at": None, "chunks": 0}
        self.requests.append(record)
        generated = token_offset = 0
        pad = math.ceil(prompt_tokens / self.token_hop_len) * self.token_hop_len - prompt_tokens
        for _poll in range(1000):
            generated = min(speech_tokens, generated + 5)
            hop = self.token_hop_len + pad if token_offset == 0 else self.token_hop_len
            if stream and generated - token_offset >= hop + self.pre_lookahead_len:
                if record["first_chunk_at"] is None:
                    record["first_chunk_at"] = generated
                record["chunks"] += 1
                token_offset += hop
                self.token_hop_len = min(
                    self.token_max_hop_len, self.token_hop_len * self.stream_scale_factor
                )
                yield {"tts_speech": np.full(hop, 0.1, dtype=np.float32)}
            if generated == speech_tokens and generated - token_offset < hop + self.pre_lookahead_len:
                break
        else:
            raise AssertionError("the fake tts() loop did not end")
        if record["first_chunk_at"] is None:
            record["first_chunk_at"] = generated
        yield {"tts_speech": np.full(generated - token_offset, 0.1, dtype=np.float32)}


def test_cosyvoice_every_streamed_request_starts_from_the_initial_hop(monkeypatch):
    # CosyVoice2Model.tts never resets its instance hop, so
    # from the worker's second streamed request on, the first chunk waited for 100 tokens
    # plus the prompt pad; a Live Reader sentence of 3 s came out whole, at its end.
    worker = _import_worker(monkeypatch, "providers.cosyvoice.app")
    engine = _HopKeepingCosyVoiceEngine()

    class Model:
        sample_rate = 24000
        model = engine

        def inference_zero_shot(self, tts_text, prompt_text, prompt_wav, stream=False, speed=1.0):
            # A 10 s reference (250 prompt tokens) and a 3 s sentence (75 speech tokens).
            yield from self.model.tts(prompt_tokens=250, speech_tokens=75, stream=stream)

    worker.runtime.model = Model()
    worker.runtime.model_id = "cosyvoice/2-0.5b"
    worker.runtime.after_load()

    for sentence in ("The first sentence.", "The second sentence.", "The third sentence."):
        payload = worker.SynthesisRequest(
            model="cosyvoice/2-0.5b",
            text=sentence,
            reference_audio=_wav_base64(duration=10.0),
            clone_transcript="Exact reference words.",
        )
        assert list(worker.runtime.generate(payload))

    seen = [(r["start_hop"], r["first_chunk_at"], r["chunks"]) for r in engine.requests]
    # Hop 25 plus a pad of 0 and the lookahead of 3: the first chunk after 30 tokens (1.2 s).
    assert seen == [(25, 30, 1)] * 3, seen


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


# CosyVoiceFrontEnd.text_normalize, at the library's default
# text_frontend=True, knows two kinds of text: one with a CJK ideograph goes to wetext's Chinese
# normalizer, any other to its English one, after which inflect spells every digit run left in
# English. These are its outputs for these strings in open-speech-cosyvoice:0.2.1
# (replayed in the container); text_frontend=False, and a text
# holding "<|...|>", leave the text as it is. The models declare zh, en, ja, ko (and de, fr, ru, es,
# it for 3-0.5b; docs/CLONING-MODELS.md), and Fun-CosyVoice3 reads numbers "without a traditional
# frontend module" (its README in the image).
_COSYVOICE_FRONTEND_OUTPUT = {
    "Ich habe 3 Katzen und 25 Bücher.": "Ich habe three Katzen und twenty five Bücher.",
    "Il est 8 heures, et il a 42 ans.": "Il est eight heures, et il a forty two ans.",
    "Tengo 3 gatos y 25 libros.": "Tengo three gatos y twenty five libros.",
    "Ho 3 gatti e 25 libri.": "Ho three gatti e twenty five libri.",
    "У меня 3 кошки и 25 книг.": "У меня three кошки и twenty five книг.",
    "사과 3개를 샀어요.": "사과 three 개를 샀어요.",
    "りんごを3個買いました。": "りんごを三个买いました。",
    "Chapter 12 has 25 pages.": "Chapter twelve has twenty five pages.",
    "第3章有25页。": "第三章有二十五页。",
    "I have 2 cats.": "I have two cats.",
    # The 0.2.2 frontend's outputs for these, from a CPU replay ("cafe" was replayed without its
    # accent; wetext leaves the word as it is).
    "Ich habe 3 Katzen und 25 Hunde.": "Ich habe three Katzen und twenty five Hunde.",
    "Kapitel 12 hat 25 Seiten.": "Kapitel twelve hat twenty five Seiten.",
    "Il a 3 chats et 25 livres.": "Il a three chats et twenty five livres.",
    "I paid $5.20 for 2 coffees at the café.": "I paid five point two dollars for two coffees at the café.",
    # The baked wetext Chinese normalizer's outputs, from the CPU
    # replay in open-speech-cosyvoice:0.2.3.
    "我买了 iPhone SE，花了 3299 元。": "我买了 iPhone SE,花了 三千两百九十九元.",
    "他 2019 年毕业于 MIT，月薪 3 万元。": "他 两千零一十九 年毕业于 MIT,月薪 三万元.",
    "我在 café 喝了 2 杯咖啡，花了 35 元。": "我在 café 喝了 两杯咖啡,花了 三十五元.",
    "产品通过了 CE 认证，售价 299 元。": "产品通过了 CE 认证,售价 两百九十九元.",
}
_COSYVOICE_TRANSCRIPT = "I have 2 cats."


def _only_punctuation_or_symbols(text: str) -> bool:
    # cosyvoice/utils/frontend_utils.py:133-136 at 074ca6dc: regex.fullmatch(r'^[\p{P}\p{S}]*$', text)
    return all(unicodedata.category(character)[0] in "PS" for character in text)


class _NormalizingCosyVoiceFrontend:
    def __init__(self) -> None:
        self.spk2info: dict = {}

    def text_normalize(self, text, split=True, text_frontend=True):
        if "<|" in text and "|>" in text:
            text_frontend = False
        if text_frontend:
            text = _COSYVOICE_FRONTEND_OUTPUT.get(text, text)
            if split:
                # cosyvoice/cli/frontend.py:159 drops each piece of only punctuation or symbols.
                return [piece for piece in [text.strip()] if not _only_punctuation_or_symbols(piece)]
        return [text] if split else text


class _NormalizingCosyVoice:
    """The pinned library's signatures (cosyvoice/cli/cosyvoice.py); records what the LLM is given."""

    sample_rate = 24000

    def __init__(self) -> None:
        self.frontend = _NormalizingCosyVoiceFrontend()
        self.registered: list[str] = []
        self.calls: list[dict] = []
        self.spoken: list[str] = []

    def add_zero_shot_spk(self, prompt_text, prompt_wav, zero_shot_spk_id):
        self.frontend.spk2info[zero_shot_spk_id] = prompt_text
        self.registered.append(prompt_text)
        return True

    def _speak(self, tts_text, zero_shot_spk_id, text_frontend):
        # With a cached speaker the conditioning, prompt text included, comes from spk2info.
        assert zero_shot_spk_id in self.frontend.spk2info
        self.calls.append({"text": tts_text, "text_frontend": text_frontend})
        for segment in self.frontend.text_normalize(tts_text, split=True, text_frontend=text_frontend):
            self.spoken.append(segment)
            yield {"tts_speech": np.full((1, 4), 0.1, dtype=np.float32)}

    def inference_zero_shot(
        self, tts_text, prompt_text, prompt_wav, zero_shot_spk_id="", stream=False, speed=1.0,
        text_frontend=True,
    ):
        yield from self._speak(tts_text, zero_shot_spk_id, text_frontend)

    def inference_instruct2(
        self, tts_text, instruct_text, prompt_wav, zero_shot_spk_id="", stream=False, speed=1.0,
        text_frontend=True,
    ):
        yield from self._speak(tts_text, zero_shot_spk_id, text_frontend)


def _speak_through_the_cosyvoice_frontend(
    worker, model, model_id, language, text, *, instructions=None, reference=None
):
    worker.runtime.model = model
    worker.runtime.model_id = model_id
    assert worker._supports_speaker_cache(model), "the fake takes the real library's cached path"
    payload = worker.SynthesisRequest(
        model=model_id,
        text=text,
        language=language,
        instructions=instructions,
        reference_audio=reference or _wav_base64(),
        clone_transcript=_COSYVOICE_TRANSCRIPT,
    )
    return list(worker.runtime.generate(payload))


@pytest.mark.parametrize(
    ("model_id", "language", "text", "instructions"),
    [
        pytest.param("cosyvoice/3-0.5b", "de", "Ich habe 3 Katzen und 25 Bücher.", None, id="3-de"),
        pytest.param("cosyvoice/3-0.5b", "fr", "Il est 8 heures, et il a 42 ans.", None, id="3-fr"),
        pytest.param("cosyvoice/3-0.5b", "es", "Tengo 3 gatos y 25 libros.", None, id="3-es"),
        pytest.param("cosyvoice/3-0.5b", "it", "Ho 3 gatti e 25 libri.", None, id="3-it"),
        pytest.param("cosyvoice/3-0.5b", "ru", "У меня 3 кошки и 25 книг.", None, id="3-ru"),
        pytest.param("cosyvoice/3-0.5b", "ko", "사과 3개를 샀어요.", None, id="3-ko"),
        pytest.param("cosyvoice/2-0.5b", "ko", "사과 3개를 샀어요.", None, id="2-ko"),
        pytest.param(
            "cosyvoice/3-0.5b", "de", "Ich habe 3 Katzen und 25 Bücher.", "ruhig", id="3-de-instructed"
        ),
        # Without a hint the text tells: Cyrillic, Hangul, or Latin words that read as German,
        # French, Spanish or Italian rather than English.
        pytest.param("cosyvoice/3-0.5b", None, "Ich habe 3 Katzen und 25 Bücher.", None, id="3-latin"),
        pytest.param("cosyvoice/3-0.5b", None, "У меня 3 кошки и 25 книг.", None, id="3-cyrillic"),
        pytest.param("cosyvoice/2-0.5b", None, "사과 3개를 샀어요.", None, id="2-hangul"),
        # The web app sends no hint, and an unaccented German,
        # Spanish, Italian or French sentence went to the English normalizer ("three Katzen").
        pytest.param("cosyvoice/3-0.5b", None, "Ich habe 3 Katzen und 25 Hunde.", None, id="3-no-hint-de"),
        pytest.param("cosyvoice/3-0.5b", None, "Kapitel 12 hat 25 Seiten.", None, id="3-no-hint-de-chapter"),
        pytest.param("cosyvoice/3-0.5b", None, "Tengo 3 gatos y 25 libros.", None, id="3-no-hint-es"),
        pytest.param("cosyvoice/3-0.5b", None, "Ho 3 gatti e 25 libri.", None, id="3-no-hint-it"),
        pytest.param("cosyvoice/3-0.5b", None, "Il a 3 chats et 25 livres.", None, id="3-no-hint-fr"),
        pytest.param(
            "cosyvoice/3-0.5b", None, "Il est 8 heures, et il a 42 ans.", "calme", id="3-no-hint-fr-instructed"
        ),
    ],
)
def test_cosyvoice_gives_its_other_languages_to_the_model_as_written(
    monkeypatch, model_id, language, text, instructions
):
    # The worker left text_frontend at its default, so "3 Katzen"
    # was spoken as "three Katzen" and Japanese kanji turned into simplified Chinese ("個" to "个").
    worker = _import_worker(monkeypatch, "providers.cosyvoice.app")
    model = _NormalizingCosyVoice()

    assert _speak_through_the_cosyvoice_frontend(
        worker, model, model_id, language, text, instructions=instructions
    )

    assert model.spoken == [text], f"{language} text {text!r} was given to the model as {model.spoken!r}"
    assert [call["text_frontend"] for call in model.calls] == [False]
    if not instructions:
        # The cached conditioning holds the transcript as this call would have normalized it.
        assert len(model.registered) == 1 and _COSYVOICE_TRANSCRIPT in model.registered[0]


@pytest.mark.parametrize(
    ("model_id", "language", "text"),
    [
        pytest.param("cosyvoice/3-0.5b", "en", "Chapter 12 has 25 pages.", id="3-en"),
        pytest.param("cosyvoice/3-0.5b", "zh", "第3章有25页。", id="3-zh"),
        pytest.param("cosyvoice/2-0.5b", "en-GB", "Chapter 12 has 25 pages.", id="2-en-GB"),
        pytest.param("cosyvoice/2-0.5b", "zh-CN", "第3章有25页。", id="2-zh-CN"),
        pytest.param("cosyvoice/2-0.5b", None, "Chapter 12 has 25 pages.", id="2-ascii"),
        pytest.param("cosyvoice/2-0.5b", None, "第3章有25页。", id="2-han"),
        pytest.param("cosyvoice/3-0.5b", None, "Chapter 12 has 25 pages.", id="3-no-hint-en"),
        # One accented word made English skip its normalizer, so
        # "$5.20" and "2" reached the model as written.
        pytest.param(
            "cosyvoice/3-0.5b", None, "I paid $5.20 for 2 coffees at the café.", id="3-no-hint-en-cafe"
        ),
        # With no hint, one listed ("SE", "MIT", "CE") or accented
        # ("café") Latin word outvoted a Chinese sentence, so its money and numbers reached the model
        # as raw digits. Text with a CJK ideograph is Chinese and keeps the normalizer.
        pytest.param("cosyvoice/3-0.5b", None, "我买了 iPhone SE，花了 3299 元。", id="3-no-hint-zh-SE"),
        pytest.param("cosyvoice/3-0.5b", None, "他 2019 年毕业于 MIT，月薪 3 万元。", id="3-no-hint-zh-MIT"),
        pytest.param("cosyvoice/3-0.5b", None, "我在 café 喝了 2 杯咖啡，花了 35 元。", id="3-no-hint-zh-cafe"),
        pytest.param("cosyvoice/3-0.5b", None, "产品通过了 CE 认证，售价 299 元。", id="3-no-hint-zh-CE"),
        pytest.param("cosyvoice/2-0.5b", None, "我买了 iPhone SE，花了 3299 元。", id="2-no-hint-zh-SE"),
        pytest.param("cosyvoice/2-0.5b", None, "我在 café 喝了 2 杯咖啡，花了 35 元。", id="2-no-hint-zh-cafe"),
    ],
)
def test_cosyvoice_keeps_its_text_frontend_for_english_and_chinese(
    monkeypatch, model_id, language, text
):
    # A control: English and Chinese keep the normalizer.
    worker = _import_worker(monkeypatch, "providers.cosyvoice.app")
    model = _NormalizingCosyVoice()

    assert _speak_through_the_cosyvoice_frontend(worker, model, model_id, language, text)

    assert model.spoken == [_COSYVOICE_FRONTEND_OUTPUT[text]]
    assert [call["text_frontend"] for call in model.calls] == [True]
    if model_id == "cosyvoice/2-0.5b":
        assert model.registered == ["I have two cats."]


@pytest.mark.parametrize(
    ("text", "text_frontend"),
    [
        # The vote's documented limit (docs/CLONING-MODELS.md "text the words don't settle"): a short English title
        # whose only listed word is foreign is outvoted and goes to the model as written. Deliberate: a language hint
        # settles it.
        pytest.param("Le Mans 2024", False, id="title-le-mans"),
        pytest.param("Die Hard 2", False, id="title-die-hard"),
        # One English word against one foreign one is a tie, and a tie keeps the normalizer.
        pytest.param("The iPhone SE costs $429.", True, id="tie-iphone-se"),
        # In Chinese text the Latin words are not weighed at all, so
        # two listed words ("La La"), or a name ("Da Vinci"), leave the Chinese normalizer on.
        pytest.param("《La La Land》在 2016 年上映，票房 4.4 亿美元。", True, id="zh-la-la-land"),
        pytest.param("会议定于 2024年3月5日 下午 3:30 在 Da Vinci 厅举行。", True, id="zh-da-vinci"),
    ],
)
def test_cosyvoice_no_hint_vote_near_ties(monkeypatch, text, text_frontend):
    # Pins the word vote's edges with no hint, so a change to them
    # is deliberate.
    worker = _import_worker(monkeypatch, "providers.cosyvoice.app")
    model = _NormalizingCosyVoice()

    assert _speak_through_the_cosyvoice_frontend(worker, model, "cosyvoice/3-0.5b", None, text)

    assert [call["text_frontend"] for call in model.calls] == [text_frontend], model.calls
    assert model.spoken == [text]


def test_cosyvoice_registers_a_reference_per_text_frontend_setting(monkeypatch):
    # A CosyVoice 2 transcript is normalized with the call's
    # setting, so an English and a Korean request on one reference need two conditionings.
    worker = _import_worker(monkeypatch, "providers.cosyvoice.app")
    model = _NormalizingCosyVoice()
    reference = _wav_base64()

    for language, text in [("en", "Chapter 12 has 25 pages."), ("ko", "사과 3개를 샀어요."),
                           ("en", "Chapter 12 has 25 pages."), ("ko", "사과 3개를 샀어요.")]:
        assert _speak_through_the_cosyvoice_frontend(
            worker, model, "cosyvoice/2-0.5b", language, text, reference=reference
        )

    assert model.registered == ["I have two cats.", "I have 2 cats."]


def test_cosyvoice_splits_text_it_gives_without_the_frontend_at_sentence_ends(monkeypatch):
    # With text_frontend=False CosyVoice also skips its own
    # sentence split and would speak a whole long text as one segment.
    worker = _import_worker(monkeypatch, "providers.cosyvoice.app")
    model = _NormalizingCosyVoice()
    text = "Ich habe 3 Katzen. Du hast 25 Bücher! りんごを3個買いました。Wer hat mehr?"

    chunks = _speak_through_the_cosyvoice_frontend(worker, model, "cosyvoice/3-0.5b", "de", text)

    sentences = ["Ich habe 3 Katzen.", "Du hast 25 Bücher!", "りんごを3個買いました。", "Wer hat mehr?"]
    assert model.spoken == sentences
    assert [call["text_frontend"] for call in model.calls] == [False] * 4
    assert len(chunks) == 4
    assert len(model.registered) == 1


@pytest.mark.parametrize(
    ("model_id", "language", "text"),
    [
        pytest.param("cosyvoice/3-0.5b", "ja", "りんごを3個買いました。", id="3-ja"),
        pytest.param("cosyvoice/2-0.5b", "ja-JP", "りんごを3個買いました。", id="2-ja-JP"),
        pytest.param("cosyvoice/3-0.5b", None, "りんごを3個買いました。", id="3-kana"),
        pytest.param("cosyvoice/2-0.5b", None, "今日はとても良い天気ですね。", id="2-kana"),
        pytest.param("cosyvoice/3-0.5b", None, "The word is ありがとう.", id="3-kana-in-english"),
    ],
)
def test_cosyvoice_refuses_japanese_with_the_languages_it_speaks(monkeypatch, model_id, language, text):
    # The pinned library reads
    # Japanese only as katakana (example.py:98 at 074ca6dc), which the image can't produce, so "3個"
    # and 今日 were garbled on both models. A ja hint, or kana with no hint (every web-app request),
    # is refused with the supported list and Qwen3 Base named, before the model is asked.
    worker = _import_worker(monkeypatch, "providers.cosyvoice.app")
    model = _NormalizingCosyVoice()

    with pytest.raises(worker.WorkerFailure) as refused:
        _speak_through_the_cosyvoice_frontend(worker, model, model_id, language, text)

    supported = ", ".join(worker.MODEL_LANGUAGES[model_id])
    message = str(refused.value)
    assert refused.value.status_code == 400
    assert refused.value.code == "language_unsupported"
    assert f"Supported: {supported}." in message, message
    assert "qwen3/0.6b-base" in message
    assert "ja" not in worker.MODEL_LANGUAGES[model_id]
    assert model.calls == [] and model.registered == []


def test_cosyvoice_japanese_refusal_reaches_the_client_as_a_400(monkeypatch):
    # Through the core's backend, as the web app sends it (no hint).
    worker = _import_worker(monkeypatch, "providers.cosyvoice.app")
    model = _NormalizingCosyVoice()
    worker.runtime.model = model
    worker.runtime.model_id = "cosyvoice/3-0.5b"
    backend = ExternalTTSBackend(provider="cosyvoice", base_url="http://cosyvoice:8220")
    monkeypatch.setattr(backend._opener, "open", _bridge_to_worker(TestClient(worker.app)))

    with pytest.raises(ExternalProviderError) as refused:
        list(
            backend.synthesize(
                "りんごを3個買いました。", "reference", model_id="cosyvoice/3-0.5b",
                reference_audio=base64.b64decode(_wav_base64()), clone_transcript=_COSYVOICE_TRANSCRIPT,
            )
        )

    assert (refused.value.status_code, refused.value.code) == (400, "language_unsupported")
    assert "Japanese" in refused.value.message and "qwen3/0.6b-base" in refused.value.message
    assert model.calls == []


@pytest.mark.parametrize("text", ["?", "...", "______", "—", "★★★", "!!!", "  ?  "])
def test_cosyvoice_answers_punctuation_only_text_with_a_short_silence(monkeypatch, text):
    # The library's frontend drops a piece of only punctuation or
    # symbols (cosyvoice/cli/frontend.py:159), so the model yielded nothing and the worker's 500
    # reached the client as 502 "Model returned no audio". A Live Reader, Ask AI or Talk segment
    # such as "...", or a Reader rule line "______", failed its whole answer or part.
    worker = _import_worker(monkeypatch, "providers.cosyvoice.app")
    model = _NormalizingCosyVoice()
    worker.runtime.model = model
    worker.runtime.model_id = "cosyvoice/3-0.5b"
    backend = ExternalTTSBackend(provider="cosyvoice", base_url="http://cosyvoice:8220")
    monkeypatch.setattr(backend._opener, "open", _bridge_to_worker(TestClient(worker.app)))

    chunks = list(
        backend.synthesize(
            text, "reference", model_id="cosyvoice/3-0.5b",
            reference_audio=base64.b64decode(_wav_base64()), clone_transcript=_COSYVOICE_TRANSCRIPT,
        )
    )

    audio = np.concatenate([np.asarray(chunk, dtype=np.float32).reshape(-1) for chunk in chunks])
    assert 0 < audio.size <= worker.SAMPLE_RATE // 4
    assert not audio.any(), "a text with nothing to say is answered with silence"
    assert model.spoken == []


def test_cosyvoice_text_with_words_that_yields_no_audio_is_still_a_model_failure(monkeypatch):
    # A control: only text with nothing to say gets silence.
    worker = _import_worker(monkeypatch, "providers.cosyvoice.app")

    class Silent(_NormalizingCosyVoice):
        def _speak(self, tts_text, zero_shot_spk_id, text_frontend):
            self.calls.append({"text": tts_text, "text_frontend": text_frontend})
            yield from ()

    model = Silent()
    worker.runtime.model = model
    worker.runtime.model_id = "cosyvoice/3-0.5b"
    backend = ExternalTTSBackend(provider="cosyvoice", base_url="http://cosyvoice:8220")
    monkeypatch.setattr(backend._opener, "open", _bridge_to_worker(TestClient(worker.app)))

    with pytest.raises(ExternalProviderError) as failed:
        list(
            backend.synthesize(
                "Chapter 12 has 25 pages.", "reference", model_id="cosyvoice/3-0.5b",
                reference_audio=base64.b64decode(_wav_base64()), clone_transcript=_COSYVOICE_TRANSCRIPT,
            )
        )

    assert (failed.value.status_code, failed.value.code) == (502, "invalid_audio")
    assert model.calls == [{"text": "Chapter 12 has 25 pages.", "text_frontend": True}]


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


@pytest.mark.asyncio
async def test_cosyvoice_disconnect_keeps_worker_busy_until_current_chunk_finishes(monkeypatch):
    worker = _import_worker(monkeypatch, "providers.cosyvoice.app")
    second_chunk_started = threading.Event()
    finish_second_chunk = threading.Event()
    model_running = threading.Event()

    def generate(_payload):
        yield np.array([0.1], dtype="<f4")
        model_running.set()
        second_chunk_started.set()
        finish_second_chunk.wait(10)
        model_running.clear()
        yield np.array([0.2], dtype="<f4")

    monkeypatch.setattr(worker.runtime, "generate", generate)

    class Request:
        async def is_disconnected(self):
            return False

    payload = worker.SynthesisRequest(
        model="cosyvoice/2-0.5b",
        text="Disconnect check.",
        reference_audio="ignored",
        clone_transcript="Exact words.",
    )
    first_body = asyncio.Event()
    disconnect = asyncio.Event()

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            first_body.set()

    async def receive():
        await disconnect.wait()
        return {"type": "http.disconnect"}

    task = None
    try:
        response = await worker.synthesize(payload, Request())
        # uvicorn reports ASGI HTTP spec 2.3, so Starlette cancels the body iterator
        # as soon as receive() delivers http.disconnect.
        scope = {"type": "http", "asgi": {"spec_version": "2.3"}}
        task = asyncio.create_task(response(scope, receive, send))
        await asyncio.wait_for(first_body.wait(), 5)
        assert await asyncio.to_thread(second_chunk_started.wait, 5)
        disconnect.set()
        # Give a premature cleanup time to run; a correct worker is still in the chunk.
        await asyncio.wait({task}, timeout=1)

        assert model_running.is_set()
        assert (await worker.health())["busy"] is True
        with pytest.raises(worker.HTTPException) as busy:
            await worker.synthesize(payload, Request())
        assert busy.value.status_code == 429

        finish_second_chunk.set()
        await asyncio.wait_for(task, 5)
        assert not worker.operation_lock.locked()
    finally:
        finish_second_chunk.set()
        if task is not None and not task.done():
            task.cancel()
            await asyncio.wait({task}, timeout=5)


def test_cosyvoice_speed_outside_worker_range_is_a_client_error(monkeypatch):
    worker = _import_worker(monkeypatch, "providers.cosyvoice.app")

    class Model:
        def inference_zero_shot(self, *_args, **_kwargs):
            yield {"tts_speech": np.array([0.25, -0.25], dtype=np.float32)}

    worker.runtime.model = Model()
    worker.runtime.model_id = "cosyvoice/2-0.5b"

    backend = ExternalTTSBackend(provider="cosyvoice", base_url="http://cosyvoice:8220")
    monkeypatch.setattr(backend._opener, "open", _bridge_to_worker(TestClient(worker.app)))
    options = {
        "reference_audio": base64.b64decode(_wav_base64(duration=2.0)),
        "clone_transcript": "Exact words.",
    }

    chunks = list(
        backend.synthesize("Hello", "reference", speed=1.5, model_id="cosyvoice/2-0.5b", **options)
    )
    assert np.concatenate(chunks).tolist() == pytest.approx([0.25, -0.25])

    # The public API accepts 0.25-4.0; CosyVoice itself stops at 2.0.
    with pytest.raises(ExternalProviderError) as caught:
        list(
            backend.synthesize(
                "Hello", "reference", speed=3.0, model_id="cosyvoice/2-0.5b", **options
            )
        )
    assert caught.value.status_code == 400
    assert "speed" in caught.value.message


@pytest.mark.parametrize(
    ("module_name", "model_id"),
    [
        ("providers.chatterbox.app", "chatterbox/regular"),
        ("providers.cosyvoice.app", "cosyvoice/2-0.5b"),
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


def test_cosyvoice_enters_inference_mode_per_chunk_on_the_thread_that_pulls_it(monkeypatch):
    # The HTTP stream pulls each chunk with its own asyncio.to_thread call, so consecutive
    # chunks run on different pool threads. torch.inference_mode is thread-local: held
    # across a yield, it covers only the first chunk and stays on in that pool thread.
    torch = pytest.importorskip("torch")
    monkeypatch.setitem(sys.modules, "torch", torch)
    sys.modules.pop("providers.cosyvoice.app", None)
    worker = importlib.import_module("providers.cosyvoice.app")
    events = []

    class Model:
        def inference_zero_shot(self, *_args, **_kwargs):
            try:
                for value in (0.1, 0.2, 0.3):
                    events.append(torch.is_inference_mode_enabled())
                    yield {"tts_speech": torch.tensor([value])}
            finally:
                events.append("closed")

    worker.runtime.model = Model()
    worker.runtime.model_id = "cosyvoice/2-0.5b"
    payload = worker.SynthesisRequest(
        model="cosyvoice/2-0.5b",
        text="Three chunks.",
        reference_audio=_wav_base64(duration=2.0),
        clone_transcript="Exact words.",
    )
    chunks = worker.runtime.generate(payload)
    pool = [ThreadPoolExecutor(max_workers=1) for _ in range(3)]
    try:
        assert pool[0].submit(worker._next_chunk, chunks).result(10)[0] is False
        assert pool[1].submit(worker._next_chunk, chunks).result(10)[0] is False
        # A client disconnect closes the stream from yet another thread.
        pool[2].submit(chunks.close).result(10)
        left_on = [thread.submit(torch.is_inference_mode_enabled).result(10) for thread in pool]
    finally:
        for thread in pool:
            thread.shutdown(wait=True)

    assert events == [True, True, "closed"]
    assert left_on == [False, False, False]


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


def test_cosyvoice_registers_each_reference_once_as_a_zero_shot_speaker(monkeypatch):
    monkeypatch.setenv("COSYVOICE_REFERENCE_CACHE_SIZE", "2")
    worker = _import_worker(monkeypatch, "providers.cosyvoice.app")
    first, second = _wav_base64(duration=2.0), _wav_base64(duration=3.0)
    added = []
    used = []

    class Frontend:
        def __init__(self):
            self.spk2info = {"builtin": "shipped speaker"}

        def text_normalize(self, text, split=True, text_frontend=True):
            assert (split, text_frontend) == (False, True)
            return f"normalized {text}"

    class Model:
        def __init__(self):
            self.frontend = Frontend()

        def add_zero_shot_spk(self, prompt_text, prompt_wav, zero_shot_spk_id):
            with open(prompt_wav, "rb") as reference:
                self.frontend.spk2info[zero_shot_spk_id] = (prompt_text, reference.read())
            added.append(prompt_text)
            return True

        def inference_zero_shot(
            self, tts_text, prompt_text, prompt_wav, zero_shot_spk_id="", stream=False, speed=1.0
        ):
            used.append(self.frontend.spk2info[zero_shot_spk_id])
            yield {"tts_speech": np.zeros(2, dtype=np.float32)}

        inference_instruct2 = inference_zero_shot

    model = Model()
    worker.runtime.model = model
    worker.runtime.model_id = "cosyvoice/2-0.5b"

    def say(reference, instructions=None):
        payload = worker.SynthesisRequest(
            model="cosyvoice/2-0.5b",
            text="Hello there.",
            reference_audio=reference,
            clone_transcript="Exact words.",
            instructions=instructions,
        )
        assert list(worker.runtime.generate(payload))

    say(first)
    say(first)
    say(first, instructions="calm")
    say(second)  # a third conditioning evicts the least recently used one
    say(first)

    # Zero-shot conditioning uses the transcript as inference_zero_shot normalizes it;
    # instructed conditioning uses the instruction as inference_instruct2 receives it.
    transcript, instruction = "normalized Exact words.", "calm<|endofprompt|>"
    first_audio, second_audio = base64.b64decode(first), base64.b64decode(second)
    assert added == [transcript, instruction, transcript, transcript]
    assert used == [
        (transcript, first_audio),
        (transcript, first_audio),
        (instruction, first_audio),
        (transcript, second_audio),
        (transcript, first_audio),
    ]
    # Evicted speakers leave the model; its shipped speakers stay.
    assert len(model.frontend.spk2info) == 3
    assert model.frontend.spk2info["builtin"] == "shipped speaker"

    worker.runtime.unload()
    assert len(worker.runtime.references) == 0


def test_cosyvoice_without_the_speaker_cache_parameters_synthesizes_uncached(monkeypatch):
    # The speaker cache calls text_normalize(..., split=False, text_frontend=True); a
    # CosyVoice whose text_normalize lacks those parameters failed every request.
    worker = _import_worker(monkeypatch, "providers.cosyvoice.app")
    calls = []

    class Frontend:
        def __init__(self):
            self.spk2info = {}

        def text_normalize(self, text):
            return text

    class Model:
        def __init__(self):
            self.frontend = Frontend()

        def add_zero_shot_spk(self, prompt_text, prompt_wav, zero_shot_spk_id):
            raise AssertionError("the speaker cache needs text_normalize's parameters")

        def inference_zero_shot(
            self, tts_text, prompt_text, prompt_wav, zero_shot_spk_id="", stream=False, speed=1.0
        ):
            calls.append(zero_shot_spk_id)
            yield {"tts_speech": np.zeros(2, dtype=np.float32)}

        inference_instruct2 = inference_zero_shot

    worker.runtime.model = Model()
    worker.runtime.model_id = "cosyvoice/2-0.5b"
    payload = worker.SynthesisRequest(
        model="cosyvoice/2-0.5b",
        text="Hello there.",
        reference_audio=_wav_base64(duration=2.0),
        clone_transcript="Exact words.",
    )

    assert list(worker.runtime.generate(payload))
    assert calls == [""]


@pytest.mark.parametrize(
    ("module_name", "model_id", "message"),
    [
        (
            "providers.chatterbox.app",
            "chatterbox/regular",
            "CUDA out of memory during Chatterbox synthesis",
        ),
        (
            "providers.cosyvoice.app",
            "cosyvoice/2-0.5b",
            "CUDA out of memory during CosyVoice synthesis",
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

        def inference_zero_shot(self, *_args, **_kwargs):
            yield {"tts_speech": audio_after_one_out_of_memory()}

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


def _stub_cosyvoice_library(monkeypatch, auto_model) -> None:
    cosyvoice_cli = types.ModuleType("cosyvoice.cli.cosyvoice")
    cosyvoice_cli.AutoModel = auto_model
    monkeypatch.setitem(sys.modules, "cosyvoice", types.ModuleType("cosyvoice"))
    monkeypatch.setitem(sys.modules, "cosyvoice.cli", types.ModuleType("cosyvoice.cli"))
    monkeypatch.setitem(sys.modules, "cosyvoice.cli.cosyvoice", cosyvoice_cli)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **_kwargs: "/models/cosyvoice"),
    )


def test_out_of_memory_while_loading_is_a_typed_507_and_leaves_the_worker_empty(monkeypatch):
    worker = _import_worker(monkeypatch, "providers.cosyvoice.app")
    emptied = _fake_gpu(worker)

    def auto_model(**_kwargs):
        # Some CUDA paths raise a plain RuntimeError with the allocator's message.
        raise RuntimeError("CUDA out of memory. Tried to allocate 1.50 GiB")

    _stub_cosyvoice_library(monkeypatch, auto_model)
    client = TestClient(worker.app)

    response = client.post("/v1/models/load", json={"model": "cosyvoice/2-0.5b"})

    assert response.status_code == 507
    assert response.json() == {
        "error": {"code": "out_of_memory", "message": "CUDA out of memory while loading CosyVoice"}
    }
    assert emptied
    assert worker.runtime.model is None
    health = client.get("/health").json()
    assert (health["status"], health["loaded_model"], health["busy"]) == ("ok", None, False)


WETEXT_FILES = ("en/tn/tagger.fst", "en/tn/verbalizer.fst", "zh/tn/tagger.fst", "zh/tn/verbalizer.fst")


def _stub_cosyvoice_text_frontend(monkeypatch, modelscope_download) -> None:
    """CosyVoice's AutoModel with its real text-frontend choice, over wetext 0.0.4.

    - CosyVoice 074ca6dc ``cosyvoice/cli/frontend.py:56-75``: with no ttsfrd, build two wetext
      Normalizers; any failure sets ``text_frontend = ''`` and logs only at INFO.
    - wetext 0.0.4 ``wetext/wetext.py``: the module imports ``snapshot_download`` from
      modelscope, calls it with ``"pengzhendong/wetext"`` and no revision, and loads
      ``{repo}/{lang}/tn/{tagger,verbalizer}.fst`` with kaldifst, which fails on a missing file.
    """
    modelscope = types.ModuleType("modelscope")
    modelscope.snapshot_download = modelscope_download
    monkeypatch.setitem(sys.modules, "modelscope", modelscope)

    wetext_module = types.ModuleType("wetext.wetext")
    wetext_module.snapshot_download = modelscope.snapshot_download

    class Normalizer:
        def __init__(self, tagger_path=None, verbalizer_path=None, lang="auto", operator="tn",
                     remove_erhua=False, enable_0_to_9=False):
            repo_dir = wetext_module.snapshot_download("pengzhendong/wetext")
            self.files = [f"{repo_dir}/{name}" for name in WETEXT_FILES]
            for path in self.files:
                if not Path(path).is_file():
                    raise RuntimeError(f"kaldifst: cannot open {path}")

    wetext_module.Normalizer = Normalizer
    wetext_package = types.ModuleType("wetext")
    wetext_package.wetext = wetext_module
    wetext_package.Normalizer = Normalizer
    monkeypatch.setitem(sys.modules, "wetext", wetext_package)
    monkeypatch.setitem(sys.modules, "wetext.wetext", wetext_module)
    monkeypatch.setitem(sys.modules, "ttsfrd", None)  # absent from the image

    class FrontEnd:
        def __init__(self):
            try:
                import ttsfrd  # noqa: F401

                self.text_frontend = "ttsfrd"
            except ImportError:
                try:  # upstream catches everything with a bare except
                    from wetext import Normalizer as EnNormalizer
                    from wetext import Normalizer as ZhNormalizer

                    self.zh_tn_model = ZhNormalizer(remove_erhua=False)
                    self.en_tn_model = EnNormalizer()
                    self.text_frontend = "wetext"
                except (ImportError, RuntimeError):
                    self.text_frontend = ""

    _stub_cosyvoice_library(
        monkeypatch, lambda **_kwargs: SimpleNamespace(sample_rate=24000, frontend=FrontEnd())
    )


def test_cosyvoice_load_reads_the_text_normalizer_baked_into_the_image(monkeypatch, tmp_path):
    # wetext fetched its FSTs at every load in a new container,
    # from ModelScope's unpinned master. The image now holds them at a pinned commit.
    worker = _import_worker(monkeypatch, "providers.cosyvoice.app")
    _fake_gpu(worker)
    for name in WETEXT_FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(b"fst")
    monkeypatch.setattr(worker, "WETEXT_DIR", str(tmp_path), raising=False)
    fetches = []

    def download(model_id, revision=None, **_kwargs):
        fetches.append((model_id, revision))
        raise RuntimeError("no network in this test")

    _stub_cosyvoice_text_frontend(monkeypatch, download)
    client = TestClient(worker.app)

    response = client.post("/v1/models/load", json={"model": "cosyvoice/2-0.5b"})

    assert response.status_code == 200, response.text
    assert fetches == []
    frontend = worker.runtime.model.frontend
    assert frontend.text_frontend == "wetext"
    assert frontend.en_tn_model.files == [f"{tmp_path}/{name}" for name in WETEXT_FILES]


def test_cosyvoice_load_without_its_text_normalizer_fails_with_a_typed_error(monkeypatch, tmp_path):
    # ModelScope refused the fetch (403, too many requests),
    # CosyVoice fell back to no normalizer, and the load still answered 200 with no word of it:
    # "It costs $5.20 at 3:30." was read as "It costs $five.twenty at three:thirty.".
    worker = _import_worker(monkeypatch, "providers.cosyvoice.app")
    _fake_gpu(worker)
    monkeypatch.setattr(worker, "WETEXT_DIR", str(tmp_path / "missing"), raising=False)

    def refused(model_id, revision=None, **_kwargs):
        raise RuntimeError("[E3002] [403] too many requests, try again later")

    _stub_cosyvoice_text_frontend(monkeypatch, refused)
    client = TestClient(worker.app)

    response = client.post("/v1/models/load", json={"model": "cosyvoice/2-0.5b"})

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["code"] == "text_frontend_unavailable"
    assert "normalizer" in response.json()["detail"]["message"]
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


class _FakeCosyVoiceLlm:
    """Token generator whose calls each follow a plan of (token count, error at the end)."""

    def __init__(self, *plans):
        self.plans = list(plans)
        self.produced = []
        self.finished = []

    def inference(self, text, uuid):
        count, failure = self.plans.pop(0)
        finished = threading.Event()
        self.finished.append(finished)
        produced = 0
        try:
            for token in range(count):
                time.sleep(0.002)
                yield token
                produced += 1
            if failure is not None:
                raise failure
        finally:
            self.produced.append(produced)
            finished.set()


class _FakeCosyVoiceEngine:
    """CosyVoice2Model's threading (cosyvoice/cli/model.py): llm_job fills the token list on
    its own thread and marks llm_end_dict only after its loop; tts() waits for that mark."""

    hop = 4

    def __init__(self, llm):
        self.llm = llm
        self.llm_end_dict = {}
        self.tts_speech_token_dict = {}
        self.token2wav_failure = None

    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        for token in self.llm.inference(text=text, uuid=uuid):
            self.tts_speech_token_dict[uuid].append(token)
        self.llm_end_dict[uuid] = True

    def token2wav(self, tokens):
        if self.token2wav_failure is not None:
            failure, self.token2wav_failure = self.token2wav_failure, None
            raise failure
        return {"tts_speech": np.full(len(tokens), 0.1, dtype=np.float32)}

    def tts(self, text, stream):
        this_uuid = uuid.uuid4().hex
        self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
        job = threading.Thread(target=self.llm_job, args=(text, None, None, None, this_uuid))
        job.start()
        tokens = self.tts_speech_token_dict[this_uuid]
        offset = 0
        if stream:
            # Bounded so that a worker which never ends the wait fails instead of hanging.
            deadline = time.monotonic() + 5
            while True:
                time.sleep(0.01)
                assert time.monotonic() < deadline, "tts() is still waiting for the LLM thread"
                if len(tokens) - offset >= self.hop:
                    yield self.token2wav(tokens[offset : offset + self.hop])
                    offset += self.hop
                if self.llm_end_dict[this_uuid] and len(tokens) - offset < self.hop:
                    break
        job.join()
        yield self.token2wav(tokens[offset:])


class _FakeCosyVoice:
    sample_rate = 24000

    def __init__(self, llm):
        self.model = _FakeCosyVoiceEngine(llm)
        self.frontend = SimpleNamespace(text_frontend="wetext")

    def inference_zero_shot(self, tts_text, prompt_text, prompt_wav, stream=False, speed=1.0):
        yield from self.model.tts(tts_text, stream)


def _load_fake_cosyvoice(monkeypatch, worker, model):
    emptied = _fake_gpu(worker)
    _stub_cosyvoice_library(monkeypatch, lambda **_kwargs: model)
    client = TestClient(worker.app)
    assert client.post("/v1/models/load", json={"model": "cosyvoice/2-0.5b"}).status_code == 200
    emptied.clear()
    return client, emptied


def _cosyvoice_body(speed: float = 1.0) -> dict:
    return {
        "model": "cosyvoice/2-0.5b",
        "text": "Hello there.",
        "speed": speed,
        "reference_audio": _wav_base64(duration=2.0),
        "clone_transcript": "Exact words.",
    }


@pytest.mark.parametrize("speed", [1.0, 1.2], ids=["streamed", "whole"])
def test_cosyvoice_out_of_memory_on_the_llm_thread_is_a_typed_507_and_frees_the_worker(
    monkeypatch, speed
):
    # CosyVoice generates the LLM tokens on its own thread, which never marks the end when
    # it fails. A streamed request then waited forever with the worker locked, and a whole
    # one vocoded the tokens produced so far and returned the cut-off clip as success.
    worker = _import_worker(monkeypatch, "providers.cosyvoice.app")
    out_of_memory = worker.torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 1 GiB")
    llm = _FakeCosyVoiceLlm((2, out_of_memory), (6, None))
    client, emptied = _load_fake_cosyvoice(monkeypatch, worker, _FakeCosyVoice(llm))

    response = client.post("/v1/audio/speech", json=_cosyvoice_body(speed))

    assert response.status_code == 507
    assert response.json() == {
        "error": {"code": "out_of_memory", "message": "CUDA out of memory during CosyVoice synthesis"}
    }
    assert emptied
    health = client.get("/health").json()
    assert (health["status"], health["loaded_model"], health["busy"]) == (
        "ok",
        "cosyvoice/2-0.5b",
        False,
    )
    retry = client.post("/v1/audio/speech", json=_cosyvoice_body(speed))
    assert retry.status_code == 200
    assert np.frombuffer(retry.content, dtype="<f4").tolist() == pytest.approx([0.1] * 6)


@pytest.mark.parametrize("ending", ["out_of_memory", "disconnect"])
def test_cosyvoice_stops_the_llm_thread_before_a_request_that_ends_early_frees_the_worker(
    monkeypatch, ending
):
    # The LLM thread outlives a request that ends before it: the vocoder running out of
    # memory, or the client going away. The next request then shared the GPU with it.
    worker = _import_worker(monkeypatch, "providers.cosyvoice.app")
    llm = _FakeCosyVoiceLlm((500, None), (6, None))
    model = _FakeCosyVoice(llm)
    client, _emptied = _load_fake_cosyvoice(monkeypatch, worker, model)

    try:
        if ending == "out_of_memory":
            model.model.token2wav_failure = worker.torch.cuda.OutOfMemoryError("CUDA out of memory.")
            assert client.post("/v1/audio/speech", json=_cosyvoice_body()).status_code == 507
        else:
            chunks = worker.runtime.generate(worker.SynthesisRequest(**_cosyvoice_body()))
            assert worker._next_chunk(chunks)[0] is False
            chunks.close()

        assert llm.finished[0].is_set()
        assert llm.produced[0] < 500
    finally:
        # A worker that leaves the thread running lets it finish its 500 tokens here.
        assert llm.finished[0].wait(10)

    retry = client.post("/v1/audio/speech", json=_cosyvoice_body())
    assert retry.status_code == 200
    assert np.frombuffer(retry.content, dtype="<f4").tolist() == pytest.approx([0.1] * 6)


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


@pytest.mark.parametrize("name", ["chatterbox", "cosyvoice", "qwen3", "fish"])
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
