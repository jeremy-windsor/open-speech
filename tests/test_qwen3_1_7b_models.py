"""The Qwen3-TTS 1.7B models in the Qwen3 worker (CustomVoice, VoiceDesign, Base), with qwen-tts and the GPU faked:
the manifest, loading one checkpoint at a time, each model's request to qwen-tts, refusals, GPU memory, the
download-only route, and the OpenAI-compatible endpoints and voice design through the core."""

from __future__ import annotations

import base64
import gc
import importlib
import inspect
import io
import json
import sqlite3
import sys
import wave
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import main as main_module
from src import storage
from src.model_manager import ModelManager
from src.model_registry import get_known_model
from src.tts.external import ExternalTTSBackend
from src.tts.router import TTSRouter
from src.voice_library import VoiceLibraryManager
from tests.test_voice_model_workers import _bridge_to_worker

CUSTOM = "qwen3/1.7b-custom-voice"
DESIGN = "qwen3/1.7b-voice-design"
BASE = "qwen3/1.7b-base"
LARGE = (CUSTOM, DESIGN, BASE)
MIB = 1024 * 1024
DESCRIPTION = "A warm, calm man in his fifties, a slight rasp, unhurried."
TRANSCRIPT = "These are the exact words of the recording."


class FakeQwen:
    """qwen-tts's Qwen3TTSModel: its from_pretrained and the generate methods, as the pinned 0.1.1 names them."""

    loads: list = []
    calls: list = []
    live = 0

    def __init__(self, path: str) -> None:
        self.path = path
        FakeQwen.live += 1

    def __del__(self) -> None:
        FakeQwen.live -= 1

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.loads.append((path, kwargs))
        return cls(path)

    def _speak(self, method, kwargs):
        FakeQwen.calls.append((method, kwargs))
        return [np.full(2400, 0.25, dtype=np.float32)], 24000

    def generate_custom_voice(self, text, speaker, language=None, instruct=None, non_streaming_mode=True, **kwargs):
        return self._speak("generate_custom_voice", {
            "text": text, "speaker": speaker, "language": language, "instruct": instruct,
            "non_streaming_mode": non_streaming_mode, **kwargs})

    def generate_voice_design(self, text, instruct, language=None, non_streaming_mode=True, **kwargs):
        return self._speak("generate_voice_design", {
            "text": text, "instruct": instruct, "language": language, "non_streaming_mode": non_streaming_mode,
            **kwargs})

    def generate_voice_clone(self, text, language=None, ref_audio=None, ref_text=None, x_vector_only_mode=False,
                             voice_clone_prompt=None, non_streaming_mode=False, **kwargs):
        return self._speak("generate_voice_clone", {
            "text": text, "language": language, "voice_clone_prompt": voice_clone_prompt,
            "non_streaming_mode": non_streaming_mode, **kwargs})

    def create_voice_clone_prompt(self, ref_audio, ref_text=None, x_vector_only_mode=False):
        FakeQwen.calls.append(("create_voice_clone_prompt", {"ref_text": ref_text,
                                                             "x_vector_only_mode": x_vector_only_mode}))
        return {"prompt": ref_text}


@pytest.fixture
def qwen(monkeypatch):
    """The worker with torch stubbed, a Turing GPU whose reserved memory is what the loaded fakes hold, and the
    Hugging Face snapshots faked: a cached model returns its folder, the others must be fetched."""
    monkeypatch.setenv("QWEN3_ENABLE_BASE", "true")
    monkeypatch.delenv("QWEN3_DTYPE", raising=False)
    cuda = SimpleNamespace(
        OutOfMemoryError=type("FakeOutOfMemoryError", (Exception,), {}),
        is_available=lambda: True,
        empty_cache=lambda: None,
        get_device_capability=lambda _index: (7, 5),
        get_device_name=lambda _index: "NVIDIA GeForce RTX 2070 SUPER",
        get_arch_list=lambda: ["sm_75"],
        memory_allocated=lambda _index: 0,
        memory_reserved=lambda _index: FakeQwen.live * 5200 * MIB,
    )
    seeds: list[int] = []
    torch_stub = SimpleNamespace(cuda=cuda, float16="float16", float32="float32", bfloat16="bfloat16",
                                 manual_seed=seeds.append)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "qwen_tts", SimpleNamespace(Qwen3TTSModel=FakeQwen))
    monkeypatch.setattr(FakeQwen, "loads", [])
    monkeypatch.setattr(FakeQwen, "calls", [])
    sys.modules.pop("providers.qwen3.app", None)
    worker = importlib.import_module("providers.qwen3.app")
    state = {"cached": set(), "fetches": [], "seeds": seeds, "fail": None}

    def snapshot(model_id, *, local_files_only=False):
        if state["fail"]:
            raise OSError(state["fail"])
        if local_files_only and model_id not in state["cached"]:
            raise FileNotFoundError(f"{model_id} is not cached")
        if not local_files_only:
            state["fetches"].append(model_id)
            state["cached"].add(model_id)
        return f"/hf/snapshots/{model_id}"

    monkeypatch.setattr(worker, "_snapshot", snapshot)
    monkeypatch.setattr(worker, "_random_seed", lambda: 777)
    client = TestClient(worker.app)
    yield worker, client, state
    worker.runtime.unload()
    gc.collect()


def load(client, model_id):
    response = client.post("/v1/models/load", json={"model": model_id})
    assert response.status_code == 200, response.text


def speak(client, **body):
    return client.post("/v1/audio/speech", json={"text": "Hello there.", **body})


def last_call(method):
    return [kwargs for name, kwargs in FakeQwen.calls if name == method][-1]


# ── Manifest and registry ──────────────────────────────────────────────────────────────────────────────────────────


def test_the_manifest_offers_the_three_1_7b_models_next_to_the_0_6b(qwen):
    worker, client, _state = qwen
    models = {model["id"]: model for model in client.get("/v1/manifest").json()["models"]}

    assert list(models) == ["qwen3/0.6b-custom-voice", "qwen3/0.6b-base", CUSTOM, DESIGN, BASE]
    for model_id in LARGE:
        assert models[model_id]["revision"] == worker.MODEL_REVISIONS[model_id] and len(worker.MODEL_REVISIONS[model_id]) == 40
        assert worker.MODEL_IDS[model_id].startswith("Qwen/Qwen3-TTS-12Hz-1.7B-")
        assert models[model_id]["gpu_memory_mb"] == 5200
        assert models[model_id]["sample_rate"] == 24000
        assert models[model_id]["capabilities"]["speed_control"] is False

    custom = models[CUSTOM]["capabilities"]
    assert [voice["id"] for voice in models[CUSTOM]["voices"]] == [voice["id"] for voice in worker.VOICES]
    assert {voice["id"] for voice in models[CUSTOM]["voices"] if voice["language"] == "en"} == {"Ryan", "Aiden"}
    assert (custom["instructions"], custom["voice_design"], custom["voice_clone"], custom["live_reader"]) == (
        True, False, False, True)
    assert models["qwen3/0.6b-custom-voice"]["capabilities"]["instructions"] is False

    design = models[DESIGN]["capabilities"]
    assert models[DESIGN]["voices"] == []
    assert (design["voice_design"], design["instructions"], design["voice_clone"], design["live_reader"]) == (
        True, True, False, False)

    base = models[BASE]["capabilities"]
    assert models[BASE]["voices"] == []
    assert (base["voice_clone"], base["clone_transcript_required"], base["voice_design"], base["live_reader"]) == (
        True, True, False, False)


def test_the_registry_rows_say_where_the_1_7b_models_are_best(qwen):
    worker, _client, _state = qwen
    for model in worker.MANIFEST["models"]:
        if model["id"] not in LARGE:
            continue
        row = get_known_model(model["id"])
        assert (row["provider"], row["gpu_mb"], row["optional_provider"]) == ("qwen3", model["gpu_memory_mb"], True)
        assert "best on newer or bigger GPUs" in row["description"]
        assert 4200 < row["size_mb"] < 4400
    assert "voice design" in get_known_model(DESIGN)["description"]
    assert "voice cloning" in get_known_model(BASE)["description"]


def test_without_base_turned_on_neither_base_model_is_offered(monkeypatch, qwen):
    monkeypatch.setenv("QWEN3_ENABLE_BASE", "false")
    sys.modules.pop("providers.qwen3.app", None)
    worker = importlib.import_module("providers.qwen3.app")
    client = TestClient(worker.app)

    assert [model["id"] for model in worker.MANIFEST["models"]] == ["qwen3/0.6b-custom-voice", CUSTOM, DESIGN]
    for path in ("/v1/models/load", "/v1/models/download"):
        response = client.post(path, json={"model": BASE})
        assert (response.status_code, response.json()["detail"]["code"]) == (400, "unknown_model")
    assert speak(client, model=BASE).json()["detail"]["code"] == "unknown_model"


# ── Loading one checkpoint at a time ───────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("model_id", "capability", "dtype"),
    [
        (CUSTOM, (7, 5), "float16"),
        (DESIGN, (7, 5), "float16"),
        (BASE, (7, 5), "float16"),
        (CUSTOM, (8, 6), "bfloat16"),
        ("qwen3/0.6b-custom-voice", (7, 5), "float32"),
        ("qwen3/0.6b-base", (8, 9), "bfloat16"),
    ],
)
def test_a_load_reads_the_pinned_snapshot_in_the_dtype_the_gpu_runs(qwen, model_id, capability, dtype):
    worker, client, state = qwen
    worker.torch.cuda.get_device_capability = lambda _index: capability
    worker.torch.cuda.get_arch_list = lambda: ["sm_75", "sm_80", "sm_86"]

    load(client, model_id)

    path, kwargs = FakeQwen.loads[-1]
    assert path == f"/hf/snapshots/{model_id}", "loaded from the local folder, not by repository ID"
    assert kwargs == {"device_map": "cuda:0", "dtype": dtype, "attn_implementation": "sdpa"}
    assert state["fetches"] == [model_id], "not cached yet, so the load fetched it"
    health = client.get("/health").json()
    assert (health["loaded_model"], health["dtype"]) == (model_id, dtype)


def test_an_explicit_dtype_still_applies_to_the_1_7b(monkeypatch, qwen):
    monkeypatch.setenv("QWEN3_DTYPE", "float32")
    sys.modules.pop("providers.qwen3.app", None)
    worker = importlib.import_module("providers.qwen3.app")
    monkeypatch.setattr(worker, "_snapshot", lambda model_id, **_kwargs: f"/hf/{model_id}")
    worker.runtime.load(CUSTOM)
    assert FakeQwen.loads[-1][1]["dtype"] == "float32"
    worker.runtime.unload()


def test_switching_models_unloads_the_one_before_and_health_reports_the_memory(qwen):
    _worker, client, _state = qwen

    for model_id in ("qwen3/0.6b-custom-voice", CUSTOM, DESIGN, BASE, "qwen3/0.6b-base"):
        load(client, model_id)
        gc.collect()
        assert FakeQwen.live == 1, f"{model_id}: one checkpoint on the GPU at a time"
        health = client.get("/health").json()
        assert health["loaded_model"] == model_id
        need = 5200 if model_id in LARGE else 3500
        assert health["gpu_memory"] == {"model_mb": 5200.0, "need_mb": need}

    load(client, CUSTOM)
    response = client.post("/v1/models/unload", json={"model": CUSTOM})
    assert response.status_code == 200
    gc.collect()
    health = client.get("/health").json()
    assert FakeQwen.live == 0
    assert (health["loaded_model"], health["gpu_memory"]) == (None, {"model_mb": None, "need_mb": None})
    assert health["gpu"]["memory_reserved_mb"] == 0.0


def test_the_core_reads_the_memory_each_model_needs_and_holds(monkeypatch, qwen):
    _worker, client, _state = qwen
    backend = ExternalTTSBackend(provider="qwen3", base_url="http://qwen3.voice-workers.internal.:8200")
    monkeypatch.setattr(backend._opener, "open", _bridge_to_worker(client))

    assert {model_id: backend.gpu_memory_need_mb(model_id) for model_id in LARGE} == dict.fromkeys(LARGE, 5200)
    backend.load_model(DESIGN)
    assert [(model.model, model.gpu_memory_mb) for model in backend.loaded_models()] == [(DESIGN, 5200.0)]
    backend.load_model(BASE)
    assert [model.model for model in backend.loaded_models()] == [BASE]


# ── The download-only route ────────────────────────────────────────────────────────────────────────────────────────


def test_a_download_fetches_the_snapshot_without_loading_it(qwen):
    worker, client, state = qwen

    response = client.post("/v1/models/download", json={"model": DESIGN})

    assert response.status_code == 200
    assert response.json() == {"status": "downloaded", "model": DESIGN}
    assert state["fetches"] == [DESIGN]
    assert FakeQwen.loads == [] and worker.runtime.model is None
    assert client.get("/health").json()["gpu_memory"]["model_mb"] is None
    load(client, DESIGN)
    assert state["fetches"] == [DESIGN], "the load read the downloaded snapshot"


def test_a_download_runs_while_another_model_is_loaded_and_keeps_it(qwen):
    worker, client, state = qwen
    load(client, "qwen3/0.6b-custom-voice")
    assert client.post("/v1/models/download", json={"model": BASE}).status_code == 200
    assert worker.runtime.model_id == "qwen3/0.6b-custom-voice"
    assert state["fetches"] == ["qwen3/0.6b-custom-voice", BASE]


def test_a_failed_or_unknown_download_is_a_typed_failure(qwen):
    _worker, client, state = qwen
    response = client.post("/v1/models/download", json={"model": "qwen3/9b"})
    assert (response.status_code, response.json()["detail"]["code"]) == (400, "unknown_model")
    state["fail"] = "Network is unreachable"
    response = client.post("/v1/models/download", json={"model": CUSTOM})
    assert (response.status_code, response.json()["detail"]["code"]) == (500, "model_download_failed")
    assert "Network is unreachable" in response.json()["detail"]["message"]


# ── Each model's request to qwen-tts ───────────────────────────────────────────────────────────────────────────────


def test_custom_voice_speaks_a_built_in_voice_with_an_optional_style(qwen):
    worker, client, _state = qwen
    load(client, CUSTOM)

    response = speak(client, model=CUSTOM, voice="Aiden", language="en-US", instructions="  Speak softly.  ")
    assert response.status_code == 200
    assert response.headers["X-Audio-Sample-Rate"] == "24000"
    assert np.frombuffer(response.content, dtype="<f4").size == 2400
    assert last_call("generate_custom_voice") == {
        "text": "Hello there.", "speaker": "Aiden", "language": "English", "instruct": "Speak softly.",
        "non_streaming_mode": True, "max_new_tokens": worker._max_new_tokens("Hello there.")}

    assert speak(client, model=CUSTOM, voice="Ryan").status_code == 200
    assert last_call("generate_custom_voice")["instruct"] is None


def test_voice_design_speaks_every_segment_from_one_seed(qwen):
    _worker, client, state = qwen
    load(client, DESIGN)

    response = speak(client, model=DESIGN, text="First sentence here. Second sentence here.",
                     voice_design=DESCRIPTION, instructions="Cheerful.")
    assert response.status_code == 200
    designs = [kwargs for name, kwargs in FakeQwen.calls if name == "generate_voice_design"]
    assert [call["text"] for call in designs] == ["First sentence here.", "Second sentence here."]
    assert {call["instruct"] for call in designs} == {f"{DESCRIPTION} Cheerful."}
    assert all(call["non_streaming_mode"] is True and call["language"] == "Auto" for call in designs)
    assert state["seeds"] == [777, 777], "a new voice for the request, the same for each of its segments"

    assert speak(client, model=DESIGN, instructions=DESCRIPTION, seed=123).status_code == 200
    assert last_call("generate_voice_design")["instruct"] == DESCRIPTION
    assert state["seeds"][-1] == 123


def test_base_clones_from_the_recording_and_its_transcript(qwen):
    _worker, client, state = qwen
    load(client, BASE)
    reference = base64.b64encode(b"RIFF-reference-audio").decode("ascii")

    for _ in range(2):
        response = speak(client, model=BASE, voice="reference", reference_audio=reference,
                         clone_transcript=TRANSCRIPT, language="English")
        assert response.status_code == 200
    prompts = [kwargs for name, kwargs in FakeQwen.calls if name == "create_voice_clone_prompt"]
    assert prompts == [{"ref_text": TRANSCRIPT, "x_vector_only_mode": False}], "the prompt is built once"
    clone = last_call("generate_voice_clone")
    assert (clone["voice_clone_prompt"], clone["language"], clone["non_streaming_mode"]) == (
        {"prompt": TRANSCRIPT}, "English", True)
    assert state["seeds"] == [], "no seed unless the request names one"


@pytest.mark.parametrize(
    ("model_id", "changes", "status", "code"),
    [
        pytest.param("qwen3/0.6b-custom-voice", {"instructions": "Whisper."}, 400, "instructions_unsupported",
                     id="0.6b-custom-instructions"),
        pytest.param(CUSTOM, {"voice": "alloy"}, 400, "voice_unsupported", id="custom-unknown-voice"),
        pytest.param(CUSTOM, {"voice_design": DESCRIPTION}, 400, "voice_design_unsupported", id="custom-design"),
        pytest.param(CUSTOM, {"speed": 1.25}, 400, "speed_unsupported", id="custom-speed"),
        pytest.param(DESIGN, {"voice": "alloy"}, 400, "voice_required", id="design-without-description"),
        pytest.param(DESIGN, {"voice_design": "   "}, 400, "voice_required", id="design-blank-description"),
        pytest.param(DESIGN, {"instructions": DESCRIPTION, "language": "nl"}, 400, "language_unsupported",
                     id="design-language"),
        pytest.param(BASE, {"clone_transcript": TRANSCRIPT}, 400, "reference_audio_required", id="base-no-recording"),
        pytest.param(BASE, {"reference_audio": "UklGRg=="}, 400, "clone_transcript_required", id="base-no-transcript"),
        pytest.param(BASE, {"reference_audio": "UklGRg==", "clone_transcript": TRANSCRIPT,
                            "instructions": "Slowly."}, 400, "instructions_unsupported", id="base-instructions"),
        pytest.param(BASE, {"reference_audio": "UklGRg==", "clone_transcript": TRANSCRIPT,
                            "voice_design": DESCRIPTION}, 400, "voice_design_unsupported", id="base-design"),
    ],
)
def test_what_a_model_cannot_do_is_refused_before_generation(qwen, model_id, changes, status, code):
    worker, client, _state = qwen
    load(client, model_id)
    body = {"model": model_id, "voice": "Ryan", **changes}

    response = speak(client, **body)

    assert (response.status_code, response.json()["detail"]["code"]) == (status, code)
    assert [name for name, _kwargs in FakeQwen.calls if name.startswith("generate")] == []
    assert not worker.operation_lock.locked()


def test_a_request_for_a_model_that_is_not_loaded_is_refused(qwen):
    _worker, client, _state = qwen
    load(client, CUSTOM)
    response = speak(client, model=DESIGN, instructions=DESCRIPTION)
    assert (response.status_code, response.json()["detail"]["code"]) == (409, "model_not_loaded")
    assert FakeQwen.calls == []


def test_the_pinned_api_must_take_a_style_instruction_where_the_model_follows_one(qwen):
    worker, _client, _state = qwen

    class NoInstruct:
        def generate_custom_voice(self, text, speaker, non_streaming_mode=True, **kwargs):
            return text

        def generate_voice_design(self, text, non_streaming_mode=True, **kwargs):
            return text

    worker._validate_generation_api(NoInstruct(), "qwen3/0.6b-custom-voice")
    for model_id in (CUSTOM, DESIGN):
        with pytest.raises(worker.WorkerFailure) as caught:
            worker._validate_generation_api(NoInstruct(), model_id)
        assert caught.value.code == "provider_api_mismatch" and "instruct" in str(caught.value)
    worker._validate_generation_api(FakeQwen("x"), DESIGN)
    assert "instruct" in inspect.signature(FakeQwen.generate_voice_design).parameters


def test_the_image_checks_the_pinned_package_has_each_method():
    from pathlib import Path

    dockerfile = (Path(__file__).resolve().parents[1] / "providers" / "qwen3" / "Dockerfile").read_text(
        encoding="utf-8")
    assert '"generate_voice_design"' in dockerfile and '"instruct" in parameters' in dockerfile
    requirements = (Path(__file__).resolve().parents[1] / "providers" / "qwen3" / "requirements.txt").read_text(
        encoding="utf-8")
    assert "qwen-tts==0.1.1" in requirements.split()


# ── Through the core: OpenAI clients, the voices list and voice design ────────────────────────────────────────────


class _Plain:
    name = "plain"
    sample_rate = 24000
    capabilities = {"speed_control": True}

    def load_model(self, model_id): pass
    def unload_model(self, model_id): pass
    def is_model_loaded(self, model_id): return True
    def loaded_models(self): return []
    def list_voices(self): return []

    def synthesize(self, text, voice, speed=1.0, lang_code=None, **options):
        yield np.zeros(2400, dtype=np.float32)


@pytest.fixture
def core(qwen, tmp_path, monkeypatch):
    _worker, worker_client, _state = qwen
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    monkeypatch.setattr(storage, "_conn", connection)
    storage.init_db()
    with patch("src.tts.router._discover_backends", return_value={}):
        router = TTSRouter(device="cpu",
                           external_providers=json.dumps({"qwen3": "http://qwen3.voice-workers.internal.:8200"}))
    router.register_backend("plain", _Plain())
    posts: list[str] = []
    bridge = _bridge_to_worker(worker_client)

    def open_worker(request, timeout=None):
        if request.get_method() == "POST":
            posts.append(request.full_url.rsplit(":8200", 1)[1])
        return bridge(request, timeout)

    router._backends["qwen3"]._opener.open = open_worker
    stt = MagicMock(_backends={})
    stt.loaded_models.return_value = []
    stt.list_cached_models.return_value = []
    manager = ModelManager(stt_router=stt, tts_router=router)
    with (
        patch.object(main_module, "tts_router", router),
        patch.object(main_module, "model_manager", manager),
        patch.object(main_module, "voice_library", VoiceLibraryManager(tmp_path / "voices")),
        patch.object(main_module.settings, "tts_cache_enabled", False),
        patch.object(main_module.settings, "tts_trim_silence", False),
        patch("src.tts.pipeline.encode_with_ffmpeg",
              side_effect=lambda audio, fmt, sample_rate=24000: f"{fmt}:{sample_rate}".encode()),
    ):
        yield TestClient(main_module.app), posts, main_module
    connection.close()


def core_load(client, model_id):
    response = client.post(f"/api/models/{model_id}/load")
    assert response.status_code == 200, response.text


OPENAI_SDK = {"Authorization": "Bearer sk-anything", "User-Agent": "OpenAI/Python 1.99.0"}


def test_a_first_load_downloads_before_it_loads(core):
    client, posts, _main = core
    core_load(client, CUSTOM)
    assert posts == ["/v1/models/download", "/v1/models/load"]


def test_openai_clients_list_the_models_and_the_built_in_voices(core):
    client, _posts, _main = core
    core_load(client, CUSTOM)
    rows = {row["id"]: row for row in client.get("/v1/audio/models").json()["models"]}
    assert rows[CUSTOM]["name"] == "Qwen3 TTS 1.7b-custom-voice"
    assert (rows[CUSTOM]["status"], rows[CUSTOM]["backend"]) == ("loaded", "qwen3")
    voices = client.get(f"/v1/audio/voices?model={CUSTOM}").json()["voices"]
    assert {"Ryan", "Aiden"} <= {voice["id"] for voice in voices}
    assert len(voices) == 9 and all(voice["model"] == CUSTOM for voice in voices)
    assert client.get(f"/v1/audio/voices?model={DESIGN}").json()["voices"] == []


def test_an_openai_client_speaks_a_built_in_voice_with_its_instructions(core):
    client, _posts, main = core
    core_load(client, CUSTOM)
    # openai-python's audio.speech.create(model=..., voice=..., input=..., instructions=..., response_format="wav").
    response = client.post("/v1/audio/speech", json={
        "model": CUSTOM, "voice": "Ryan", "input": "Good morning.", "instructions": "Sound excited.",
        "response_format": "wav"}, headers=OPENAI_SDK)
    assert response.status_code == 200, response.text
    with wave.open(io.BytesIO(response.content)) as clip:
        assert clip.getframerate() == 24000
    assert (last_call("generate_custom_voice")["speaker"], last_call("generate_custom_voice")["instruct"]) == (
        "Ryan", "Sound excited.")
    # Open WebUI posts {input, model, voice} and plays the default format.
    response = client.post("/v1/audio/speech", json={"input": "Hi.", "model": CUSTOM, "voice": "Aiden"})
    assert response.status_code == 200
    assert response.content == f"{main.settings.tts_default_format}:24000".encode()
    assert last_call("generate_custom_voice")["instruct"] is None


def test_an_openai_client_designs_a_voice_with_instructions(core):
    client, _posts, _main = core
    core_load(client, DESIGN)
    response = client.post("/v1/audio/speech", json={
        "model": DESIGN, "voice": "alloy", "input": "Reading in a voice I described.", "instructions": DESCRIPTION,
        "response_format": "wav"}, headers=OPENAI_SDK)
    assert response.status_code == 200, response.text
    assert last_call("generate_voice_design")["instruct"] == DESCRIPTION
    response = client.post("/v1/audio/speech", json={"model": DESIGN, "voice": "alloy", "input": "Hi."})
    assert response.status_code == 400 and response.json()["error"]["code"] == "voice_required"


def test_the_design_wizard_offers_voice_design_and_its_sample_clones_with_base(core, qwen):
    client, _posts, _main = core
    seeds = qwen[2]["seeds"]
    capabilities = client.get(f"/api/tts/capabilities?model={DESIGN}").json()["capabilities"]
    assert capabilities["voice_design"] is True and capabilities["voice_clone"] is False
    core_load(client, DESIGN)

    sample = client.post("/api/voices/design", json={"model": DESIGN, "description": DESCRIPTION,
                                                     "text": "Hello! This is the voice you described.", "seed": 42})
    assert sample.status_code == 200, sample.text
    assert sample.headers["content-type"] == "audio/wav"
    with wave.open(io.BytesIO(sample.content)) as clip:
        assert (clip.getframerate(), clip.getnframes()) == (24000, 4800)
    designs = [kwargs for name, kwargs in FakeQwen.calls if name == "generate_voice_design"]
    assert [(call["text"], call["instruct"]) for call in designs] == [
        ("Hello!", DESCRIPTION), ("This is the voice you described.", DESCRIPTION)]
    assert seeds == [42, 42], "the seed the wizard sends gives the same voice back"

    saved = client.post("/api/voices/library", data={"name": "warm_narrator", "transcript": "Hello there."},
                        files={"audio": ("warm_narrator.wav", sample.content, "audio/wav")})
    assert saved.status_code == 201, saved.text
    identity = client.post("/api/voices/identities", json={"name": "Warm narrator"}).json()
    realization = client.post(f"/api/voices/identities/{identity['id']}/realizations",
                              json={"model": BASE, "voice": "reference", "reference_audio_id": "warm_narrator"})
    assert realization.status_code == 201, realization.text
    voices = client.get(f"/v1/audio/voices?model={BASE}").json()["voices"]
    assert {voice["id"]: voice["name"] for voice in voices} == {identity["voice"]: "Warm narrator"}

    core_load(client, BASE)
    response = client.post("/v1/audio/speech", json={
        "model": BASE, "voice": identity["voice"], "input": "Now in the saved voice.", "response_format": "wav"},
        headers=OPENAI_SDK)
    assert response.status_code == 200, response.text
    assert last_call("create_voice_clone_prompt")["ref_text"] == "Hello there."
    assert last_call("generate_voice_clone")["text"] == "Now in the saved voice."


def test_the_design_api_refuses_a_model_that_cannot_design(core):
    client, _posts, _main = core
    response = client.post("/api/voices/design", json={"model": CUSTOM, "description": DESCRIPTION})
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "voice_design is not supported by the qwen3 backend."
