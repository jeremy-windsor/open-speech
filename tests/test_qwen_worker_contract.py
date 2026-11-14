"""Dependency-free contract tests for the disposable Qwen worker."""

from __future__ import annotations

import asyncio
import importlib
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.live_tts.segmenter import split_sentences


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
    worker = importlib.import_module("providers.qwen3.app")
    # A load reads the checkpoint from its cached snapshot; no test reaches the Hub.
    monkeypatch.setattr(worker, "_snapshot", lambda model_id, **_kwargs: f"/hf/{model_id}")
    return worker


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
    monkeypatch.delenv("QWEN3_ENABLE_BASE", raising=False)
    worker = _import_worker(monkeypatch)

    assert set(worker.MODEL_IDS) == set(worker.MODEL_REVISIONS)
    assert all(len(revision) == 40 for revision in worker.MODEL_REVISIONS.values())
    assert {
        manifest["id"]: manifest["revision"] for manifest in worker.MANIFEST["models"]
    } == {
        model_id: worker.MODEL_REVISIONS[model_id]
        for model_id in ("qwen3/0.6b-custom-voice", "qwen3/1.7b-custom-voice", "qwen3/1.7b-voice-design")
    }


def test_base_is_hidden_and_rejected_by_default_but_can_be_enabled(monkeypatch):
    monkeypatch.delenv("QWEN3_ENABLE_BASE", raising=False)
    worker = _import_worker(monkeypatch)
    assert [model["id"] for model in worker.MANIFEST["models"]] == [
        "qwen3/0.6b-custom-voice", "qwen3/1.7b-custom-voice", "qwen3/1.7b-voice-design"]
    for base in ("qwen3/0.6b-base", "qwen3/1.7b-base"):
        with pytest.raises(worker.WorkerFailure) as caught:
            worker.runtime.load(base)
        assert caught.value.code == "unknown_model"
        with pytest.raises(worker.WorkerFailure) as caught:
            worker.runtime.download(base)
        assert caught.value.code == "unknown_model"

    monkeypatch.setenv("QWEN3_ENABLE_BASE", "true")
    enabled = _import_worker(monkeypatch)
    assert {model["id"] for model in enabled.MANIFEST["models"]} == set(enabled.MODEL_IDS)


def test_worker_input_limit_matches_advertised_limit(monkeypatch):
    worker = _import_worker(monkeypatch)
    limit = worker.MANIFEST["models"][0]["max_input_chars"]
    assert len(worker.SynthesisRequest(model="qwen3/0.6b-custom-voice", text="x" * limit).text) == limit
    with pytest.raises(ValueError):
        worker.SynthesisRequest(model="qwen3/0.6b-custom-voice", text="x" * (limit + 1))


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


_WORKER_IMAGE_ARCH_LIST = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
_CU128_ARCH_LIST = ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("capability", "arch_list", "dtype_setting", "runs", "dtype"),
    [
        pytest.param((12, 0), _WORKER_IMAGE_ARCH_LIST, "auto", False, "bfloat16", id="blackwell-12.0"),
        pytest.param((6, 1), _CU128_ARCH_LIST, "auto", False, "float32", id="pascal-6.1-cu128"),
        # bfloat16 is refused below 8.0 too; the load checks the architecture first, and so
        # does /health.
        pytest.param((6, 1), _CU128_ARCH_LIST, "bfloat16", False, None, id="pascal-6.1-cu128-bf16"),
        pytest.param((7, 5), _WORKER_IMAGE_ARCH_LIST, "auto", True, "float32", id="turing-7.5"),
    ],
)
async def test_health_fails_when_the_torch_build_has_no_code_for_the_gpu(
    monkeypatch, capability, arch_list, dtype_setting, runs, dtype
):
    # Every load on a GPU the torch build has no
    # code for answers 503 cuda_arch_unsupported, yet /health said ok. Now it fails with the
    # load's code and message, as for a poisoned CUDA context (docs/TTS-BACKENDS.md).
    worker = _import_worker(monkeypatch)
    _fake_gpu(worker)
    worker.torch.cuda.get_device_capability = lambda _index: capability
    worker.torch.cuda.get_arch_list = lambda: list(arch_list)
    monkeypatch.setattr(worker, "DTYPE_SETTING", dtype_setting)

    result = await worker.health()

    assert result["gpu"]["capability"] == list(capability)
    assert result["gpu"]["arch_list"] == list(arch_list)
    assert result["dtype"] == dtype
    if runs:
        assert (result["status"], result["failure"]) == ("ok", None), result
        return
    architecture = f"sm_{capability[0]}{capability[1]}"
    assert result["status"] == "failed", result
    assert result["failure"] == {
        "code": "cuda_arch_unsupported",
        "message": (
            f"Torch wheel has no code that runs on {architecture} "
            f"(built for: {' '.join(arch_list)})"
        ),
    }
    with pytest.raises(worker.WorkerFailure) as load:
        worker.runtime.load("qwen3/0.6b-custom-voice")
    assert worker.common.error_detail(load.value) == result["failure"]
    assert worker.runtime.model is None


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


@pytest.mark.parametrize("character", ["界", "あ", "한", "Ａ"])
def test_wide_scripts_receive_more_speech_units(monkeypatch, character):
    worker = _import_worker(monkeypatch)

    assert worker._character_units("A") == 1.0
    assert worker._character_units(character) == worker.WIDE_CHARACTER_UNITS


def test_generation_token_limit_has_floor_weighting_and_ceiling(monkeypatch):
    worker = _import_worker(monkeypatch)

    assert worker._max_new_tokens("A") == 192
    assert worker._max_new_tokens("A" * 400) == 948
    assert worker._text_units("界" * 125) == pytest.approx(
        worker.WIDE_CHARACTER_UNITS * 125
    )
    assert worker._max_new_tokens("界" * 125) == 948

    worker.MAX_NEW_TOKENS_CEILING = 300
    assert worker._max_new_tokens("A" * 400) == 300


def test_split_text_respects_script_weighted_unit_limit(monkeypatch):
    worker = _import_worker(monkeypatch)
    worker.MAX_SEGMENT_UNITS = 10
    text = "天地玄黃宇宙洪荒"

    segments = worker._split_text(text)

    assert "".join(segments) == text
    assert all(worker._text_units(segment) <= 10 for segment in segments)
    assert segments == ["天地玄", "黃宇宙", "洪荒"]


def test_split_text_respects_token_ceiling_and_cjk_punctuation(monkeypatch):
    worker = _import_worker(monkeypatch)
    worker.MAX_SEGMENT_UNITS = 400
    worker.MAX_NEW_TOKENS_CEILING = 192

    assert worker._split_text("A" * 100) == ["A" * 64, "A" * 36]

    worker.MAX_NEW_TOKENS_CEILING = 1200
    assert worker._split_text("第一句。第二句！第三句？") == [
        "第一句。",
        "第二句！",
        "第三句？",
    ]
    assert worker._split_text("「こんにちは。」\n「元気？」") == [
        "「こんにちは。」",
        "「元気？」",
    ]
    assert worker._split_text("他说：“你好。”然后走了。") == [
        "他说：“你好。”",
        "然后走了。",
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Mr. Bennet waited on Mr. Bingley.", ["Mr. Bennet waited on Mr. Bingley."]),
        (
            "J. R. R. Tolkien wrote it. It sold well.",
            ["J. R. R. Tolkien wrote it.", "It sold well."],
        ),
        ("It rained. Dr. Smith left.", ["It rained.", "Dr. Smith left."]),
        (
            "Read e.g. the notes on St. Ives. Then stop!",
            ["Read e.g. the notes on St. Ives.", "Then stop!"],
        ),
        ("1. Preheat the oven.\n2. Mix.", ["1. Preheat the oven.", "2. Mix."]),
        # A number that starts a sentence is a list number on a shared line too.
        ("Add the flour. 2. Mix well.", ["Add the flour.", "2. Mix well."]),
        (
            "Chapter one ends here. 3. The next part begins.",
            ["Chapter one ends here.", "3. The next part begins."],
        ),
        ("Pi is 3.14. Really? Yes!", ["Pi is 3.14.", "Really?", "Yes!"]),
    ],
)
def test_split_text_keeps_abbreviations_and_initials_inside_their_sentence(
    monkeypatch, text, expected
):
    # Each part is a separate generation, so a cut after "Mr." is read as two sentences.
    worker = _import_worker(monkeypatch)

    assert worker._split_text(text) == expected
    # The worker copies the harness's sentence rules (Live Reader and Reader).
    assert [text[start:end].strip() for start, end in split_sentences(text)] == expected


def _custom_voice_worker(monkeypatch):
    worker = _import_worker(monkeypatch)
    calls = []

    class Model:
        def generate_custom_voice(self, **kwargs):
            calls.append(kwargs)
            return [np.zeros(2400, dtype=np.float32)], worker.SAMPLE_RATE

    worker.runtime.model = Model()
    worker.runtime.model_id = "qwen3/0.6b-custom-voice"
    return worker, calls


@pytest.mark.parametrize(
    ("hint", "expected"),
    [
        ("zh-CN", "Chinese"),
        ("pt-BR", "Portuguese"),
        ("es-MX", "Spanish"),
        ("en-AU", "English"),
        ("EN_us", "English"),
        ("ja", "Japanese"),
        ("english", "English"),
        ("  ", "Auto"),
        (None, "Auto"),
    ],
)
def test_language_hints_reach_qwen_as_its_language_names(monkeypatch, hint, expected):
    worker, calls = _custom_voice_worker(monkeypatch)
    request = worker.SynthesisRequest(
        model="qwen3/0.6b-custom-voice", text="Hello there.", voice="Ryan", language=hint
    )

    worker.runtime.generate(request, request.text)

    assert calls[0]["language"] == expected
    assert expected in worker.MANIFEST["models"][0]["capabilities"]["languages"]


@pytest.mark.asyncio
async def test_unsupported_language_is_a_client_error_before_generation(monkeypatch):
    worker, calls = _custom_voice_worker(monkeypatch)

    class Request:
        async def is_disconnected(self):
            return False

    payload = worker.SynthesisRequest(
        model="qwen3/0.6b-custom-voice", text="Hallo daar.", voice="Ryan", language="nl-NL"
    )

    with pytest.raises(worker.HTTPException) as caught:
        await worker.synthesize(payload, Request())

    assert caught.value.status_code == 400
    assert caught.value.detail["code"] == "language_unsupported"
    assert calls == []
    assert not worker.operation_lock.locked()


def test_the_cores_language_check_refuses_exactly_what_the_worker_refuses(monkeypatch):
    # The core checks a hint against this manifest's languages
    # before the worker is asked, so Live Reader refuses a session.update Qwen3 can't speak
    # (README.md: invalid_event, previous settings kept). The manifest names its languages, and
    # the core accepted every hint for it, so each segment failed at the worker instead.
    from src.tts.backends.base import check_declared_language

    worker = _import_worker(monkeypatch)
    hints = [
        *worker.LANGUAGES,
        *(name.upper() for name in worker.LANGUAGES),
        *worker.LANGUAGE_NAMES,
        "zh-CN", "pt-BR", "EN_us", "en-AU", "  English  ", "auto-detect",
        "nl", "nl-NL", "Dutch", "sv", "pl-PL", "xx", "x", "zzzz",
    ]
    for model in worker.MANIFEST["models"]:
        languages = model["capabilities"]["languages"]
        for hint in hints:
            try:
                worker._normalize_language(hint)
            except worker.WorkerFailure as exc:
                assert exc.code == "language_unsupported"
                worker_takes = False
            else:
                worker_takes = True
            try:
                check_declared_language(hint, languages, model["id"])
            except ValueError as exc:
                assert str(exc).startswith(f"Unsupported language '{hint}' for {model['id']}.")
                core_takes = False
            else:
                core_takes = True
            assert core_takes == worker_takes, (model["id"], hint, worker_takes)


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


def test_out_of_memory_during_generation_is_a_typed_507_and_the_worker_stays_healthy(
    monkeypatch,
):
    worker = _import_worker(monkeypatch)
    emptied = _fake_gpu(worker)
    attempts = []

    class Model:
        def generate_custom_voice(self, **_kwargs):
            attempts.append(True)
            if len(attempts) == 1:
                raise worker.torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB")
            return [np.zeros(2400, dtype=np.float32)], worker.SAMPLE_RATE

    worker.runtime.model = Model()
    worker.runtime.model_id = "qwen3/0.6b-custom-voice"
    client = TestClient(worker.app)
    body = {"model": "qwen3/0.6b-custom-voice", "text": "Hello there.", "voice": "Ryan"}

    response = client.post("/v1/audio/speech", json=body)

    assert response.status_code == 507
    assert response.json() == {
        "error": {
            "code": "out_of_memory",
            "message": "CUDA out of memory during Qwen3-TTS generation",
        }
    }
    assert emptied
    health = client.get("/health").json()
    assert (health["status"], health["failure"], health["busy"]) == ("ok", None, False)
    assert health["loaded_model"] == "qwen3/0.6b-custom-voice"
    assert client.post("/v1/audio/speech", json=body).status_code == 200


def test_out_of_memory_while_loading_is_a_typed_507_and_leaves_the_worker_empty(monkeypatch):
    worker = _import_worker(monkeypatch)
    emptied = _fake_gpu(worker)

    class OutOfMemoryLoader:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            # Some CUDA paths raise a plain RuntimeError with the allocator's message.
            raise RuntimeError("CUDA out of memory. Tried to allocate 1.50 GiB")

    monkeypatch.setitem(sys.modules, "qwen_tts", SimpleNamespace(Qwen3TTSModel=OutOfMemoryLoader))
    client = TestClient(worker.app)

    response = client.post("/v1/models/load", json={"model": "qwen3/0.6b-custom-voice"})

    assert response.status_code == 507
    assert response.json() == {
        "error": {"code": "out_of_memory", "message": "CUDA out of memory while loading Qwen3-TTS"}
    }
    assert emptied
    assert (worker.runtime.model, worker.runtime.failed_code) == (None, None)
    health = client.get("/health").json()
    assert (health["status"], health["loaded_model"], health["busy"]) == ("ok", None, False)


MIB = 1024 * 1024


class _CachingAllocator:
    """torch's CUDA caching allocator, reduced to what an unload depends on.

    ``empty_cache()`` releases only segments with no live block ("Release all unoccupied cached
    memory", torch/cuda/memory.py). cuBLAS keeps its workspace as a live block until
    ``torch._C._cuda_clearCublasWorkspaces()`` (torch/_inductor/cudagraph_trees.py).
    """

    def __init__(self) -> None:
        self.segments: dict[int, dict[int, int]] = {}
        self.sizes: dict[int, int] = {}
        self.workspaces: list[tuple[int, int]] = []
        self._next_id = 0

    def _id(self) -> int:
        self._next_id += 1
        return self._next_id

    def reserve(self, size: int) -> int:
        segment = self._id()
        self.segments[segment], self.sizes[segment] = {}, size
        return segment

    def allocate(self, segment: int, size: int) -> int:
        block = self._id()
        self.segments[segment][block] = size
        return block

    def free(self, segment: int, block: int) -> None:
        self.segments.get(segment, {}).pop(block, None)

    def clear_cublas_workspaces(self) -> None:
        while self.workspaces:
            self.free(*self.workspaces.pop())

    def empty_cache(self) -> None:
        for segment in [segment for segment, blocks in self.segments.items() if not blocks]:
            del self.segments[segment], self.sizes[segment]

    def allocated(self, _index: int = 0) -> int:
        return sum(sum(blocks.values()) for blocks in self.segments.values())

    def reserved(self, _index: int = 0) -> int:
        return sum(self.sizes.values())


@pytest.mark.parametrize("model_id", ["qwen3/0.6b-base", "qwen3/0.6b-custom-voice"])
def test_unload_returns_the_whole_model_segment_to_the_gpu(monkeypatch, model_id):
    # Transformers' caching_allocator_warmup puts Qwen3's
    # weights in one model-sized segment, and a synthesis's matmul takes cuBLAS's 8 MiB
    # workspace from the same segment. release_memory() only emptied the cache, so the idle
    # worker kept 3490 MiB reserved with no model loaded (as its /health showed).
    monkeypatch.setenv("QWEN3_ENABLE_BASE", "true")
    worker = _import_worker(monkeypatch)
    _fake_gpu(worker)
    allocator = _CachingAllocator()
    worker.torch.cuda.memory_allocated = allocator.allocated
    worker.torch.cuda.memory_reserved = allocator.reserved
    worker.torch.cuda.empty_cache = allocator.empty_cache
    worker.torch._C = SimpleNamespace(_cuda_clearCublasWorkspaces=allocator.clear_cublas_workspaces)

    class Model:
        def __init__(self) -> None:
            self.segment = allocator.reserve(3490 * MIB)
            self.weights = allocator.allocate(self.segment, 3474 * MIB)

        def __del__(self) -> None:
            allocator.free(self.segment, self.weights)

        def matmul(self) -> None:
            if not allocator.workspaces:
                block = allocator.allocate(self.segment, 8 * MIB + 128 * 1024)
                allocator.workspaces.append((self.segment, block))

    monkeypatch.setattr(worker.runtime, "build_model", lambda _model_id, _capability: Model())
    monkeypatch.setattr(worker.runtime, "after_load", lambda: None)
    client = TestClient(worker.app)

    assert client.post("/v1/models/load", json={"model": model_id}).status_code == 200
    worker.runtime.model.matmul()
    assert client.get("/health").json()["gpu"]["memory_reserved_mb"] == 3490.0
    assert client.post("/v1/models/unload", json={"model": model_id}).status_code == 200

    health = client.get("/health").json()
    assert health["loaded_model"] is None
    assert (health["gpu"]["memory_allocated_mb"], health["gpu"]["memory_reserved_mb"]) == (0.0, 0.0)


@pytest.mark.parametrize(
    ("model_id", "expected_method"),
    [
        ("qwen3/0.6b-custom-voice", "generate_custom_voice"),
        ("qwen3/0.6b-base", "generate_voice_clone"),
    ],
)
def test_generation_forwards_non_streaming_mode_and_token_limit(
    monkeypatch, model_id, expected_method
):
    worker = _import_worker(monkeypatch)
    calls = []

    class Model:
        def generate_custom_voice(self, **kwargs):
            calls.append(("generate_custom_voice", kwargs))
            return [np.zeros(2400, dtype=np.float32)], worker.SAMPLE_RATE

        def generate_voice_clone(self, **kwargs):
            calls.append(("generate_voice_clone", kwargs))
            return [np.zeros(2400, dtype=np.float32)], worker.SAMPLE_RATE

    worker.runtime.model = Model()
    worker.runtime.model_id = model_id
    monkeypatch.setattr(worker.runtime, "_decode_reference", lambda _encoded: b"wav")
    monkeypatch.setattr(worker.runtime, "_clone_prompt", lambda _audio, _text: "prompt")
    request = worker.SynthesisRequest(
        model=model_id,
        text="In the beginning.",
        voice="Ryan",
        reference_audio="ignored",
        clone_transcript="Reference words.",
    )

    worker.runtime.generate(request, request.text)

    assert len(calls) == 1
    method_name, kwargs = calls[0]
    assert method_name == expected_method
    assert kwargs["non_streaming_mode"] is True
    assert kwargs["max_new_tokens"] == worker._max_new_tokens(request.text)


# The Qwen3-TTS-12Hz speech tokenizer decodes 1920 samples per codec token at 24 kHz
# (speech_tokenizer/config.json: decode_upsample_rate 1920, output_sample_rate 24000,
# encoder_config._frame_rate 12.5), so a run that spends its whole max_new_tokens budget
# returns max_new_tokens * 1920 samples.
QWEN_TOKENIZER_SAMPLES_PER_TOKEN = 1920
QWEN_TOKENIZER_SAMPLE_RATE = 24000
_SENTENCE = "The quick brown fox jumps over the lazy dog near the riverbank. "


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(_SENTENCE[:39].strip() + ".", id="latin-40"),
        pytest.param((_SENTENCE * 2)[:119].strip() + ".", id="latin-120"),
        pytest.param((_SENTENCE * 7)[:399].strip() + ".", id="default-400-unit-segment"),
        pytest.param("今天天气很好" * 7, id="cjk-42"),
    ],
)
def test_generation_limit_discards_partial_audio(monkeypatch, text):
    # docs/TTS-BACKENDS.md: "If output reaches that safety limit, the worker returns
    # `generation_limit_reached` and discards the partial audio rather than presenting a cut-off
    # sentence as successful synthesis." The budget is the worker's own for each text.
    worker = _import_worker(monkeypatch)
    budgets = []

    class Model:
        def generate_custom_voice(self, *, max_new_tokens, **_kwargs):
            # Never emits end-of-speech: every codec token of the budget is decoded.
            budgets.append(max_new_tokens)
            samples = max_new_tokens * QWEN_TOKENIZER_SAMPLES_PER_TOKEN
            return [np.zeros(samples, dtype=np.float32)], QWEN_TOKENIZER_SAMPLE_RATE

    worker.runtime.model = Model()
    worker.runtime.model_id = "qwen3/0.6b-custom-voice"
    request = worker.SynthesisRequest(model="qwen3/0.6b-custom-voice", text=text, voice="Ryan")

    with pytest.raises(worker.WorkerFailure) as caught:
        worker.runtime.generate(request, request.text)

    assert caught.value.code == "generation_limit_reached"
    assert budgets == [worker._max_new_tokens(text)]


def test_audio_below_generation_limit_is_returned(monkeypatch):
    worker = _import_worker(monkeypatch)
    token_limit = 24

    class Model:
        def generate_custom_voice(self, **_kwargs):
            return [np.zeros(20000, dtype=np.float32)], worker.SAMPLE_RATE

    worker.runtime.model = Model()
    worker.runtime.model_id = "qwen3/0.6b-custom-voice"
    monkeypatch.setattr(worker, "_max_new_tokens", lambda _text: token_limit)
    request = worker.SynthesisRequest(
        model="qwen3/0.6b-custom-voice",
        text="Test below the limit.",
        voice="Ryan",
    )

    audio = worker.runtime.generate(request, request.text)

    assert audio.size == 20000


def test_generation_api_validation_accepts_kwargs_and_rejects_missing_controls(monkeypatch):
    worker = _import_worker(monkeypatch)

    class Compatible:
        def generate_voice_clone(self, non_streaming_mode=True, **kwargs):
            return non_streaming_mode, kwargs

    class Incompatible:
        def generate_voice_clone(self, text):
            return text

    class KwargsOnly:
        def generate_voice_clone(self, **kwargs):
            return kwargs

    worker._validate_generation_api(Compatible(), "qwen3/0.6b-base")
    with pytest.raises(worker.WorkerFailure) as caught:
        worker._validate_generation_api(Incompatible(), "qwen3/0.6b-base")
    with pytest.raises(worker.WorkerFailure) as kwargs_only:
        worker._validate_generation_api(KwargsOnly(), "qwen3/0.6b-base")

    assert caught.value.code == "provider_api_mismatch"
    assert kwargs_only.value.code == "provider_api_mismatch"


def test_generation_settings_reject_invalid_limits(monkeypatch):
    worker = _import_worker(monkeypatch)

    worker.MAX_SEGMENT_UNITS = 0
    with pytest.raises(RuntimeError, match="MAX_SEGMENT_UNITS"):
        worker._validate_generation_settings()

    worker.MAX_SEGMENT_UNITS = 400
    worker.MAX_NEW_TOKENS_CEILING = worker.GENERATION_TOKEN_FLOOR - 1
    with pytest.raises(RuntimeError, match="MAX_NEW_TOKENS_CEILING"):
        worker._validate_generation_settings()


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
