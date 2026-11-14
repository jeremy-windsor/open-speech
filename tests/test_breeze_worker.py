"""The Breeze TTS 2 worker (providers/breeze/app.py) on the worker contract, with the port's breeze-server faked at its
boundary: its process, its HTTP API, nvidia-smi and the weights download. No build, download or GPU is needed."""

from __future__ import annotations

import base64
import http.client
import importlib
import io
import json
import subprocess
import sys
import wave

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient

from src.model_registry import get_known_model
from src.tts.external import ExternalProviderError, ExternalTTSBackend
from tests.test_voice_model_workers import ROOT, _bridge_to_worker, _wav_base64

MODEL = "breeze/tts-2"
WEIGHTS = "/home/breezeworker/.cache/huggingface/hub/models--HoppouAI--Breeze-TTS-2.cpp/snapshots/e4fae64/breeze-tts-2-q8_0.gguf"
TRANSCRIPT = "These are the exact words of the recording."
DESCRIPTION = "a warm, calm middle-aged male narrator with a slight rasp"


class FakeProcess:
    """breeze-server: running until it is signalled, or until it exits by itself with ``exit_code``."""

    def __init__(self, command, *, log="", exit_code=None):
        self.command = command
        self.pid = 4242
        self.stdout = io.StringIO(log)
        self.returncode = exit_code
        self.signals = []

    def poll(self):
        return self.returncode

    def send_signal(self, signum):
        self.signals.append(signum)
        self.returncode = -signum

    def wait(self, timeout=None):
        return self.returncode


class FakeResponse:
    def __init__(self, status, body=b"", headers=None, *, chunks=None, fail_after=None):
        self.status = status
        self._headers = headers or {}
        self._chunks = list(chunks) if chunks is not None else [body]
        self._fail_after = fail_after
        self.reads = 0

    def getheader(self, name, default=None):
        return self._headers.get(name, default)

    def read(self, amount=None):
        return b"".join(self._chunks)

    def read1(self, amount=None):
        if self._fail_after is not None and self.reads >= self._fail_after:
            raise http.client.IncompleteRead(b"")
        self.reads += 1
        return self._chunks.pop(0) if self._chunks else b""


class FakeServer:
    """The port's HTTP API on 127.0.0.1: /health once loaded, and /v1/audio/speech answering ``response``."""

    def __init__(self):
        self.requests = []
        self.response = None
        self.health = {"status": "ok", "sample_rate": 24000, "ws_port": 0}

    def connection(self, timeout):
        server = self

        class Connection:
            def request(self, method, path, body=None, headers=None):
                server.requests.append({"method": method, "path": path, "body": body, "headers": headers or {}})

            def getresponse(self):
                return server.response() if callable(server.response) else server.response

            def close(self):
                pass

        return Connection()


def pcm(samples) -> bytes:
    return np.asarray(samples, dtype="<i2").tobytes()


def speech(samples=(1000, -1000, 16384, -32768), *, chunks=None, **headers):
    data = pcm(samples)
    return FakeResponse(200, headers={"X-Sample-Rate": "24000", "X-Sample-Format": "s16le", **headers},
                        chunks=chunks if chunks is not None else [data])


def form_fields(body: bytes, content_type: str) -> dict:
    """The multipart form the worker sent: field -> text, and ref_audio -> bytes."""
    boundary = content_type.split("boundary=", 1)[1].encode()
    fields = {}
    for part in body.split(b"--" + boundary)[1:-1]:
        head, _, value = part.strip(b"\r\n").partition(b"\r\n\r\n")
        name = head.split(b'name="', 1)[1].split(b'"', 1)[0].decode()
        fields[name] = value if b"filename=" in head else value.decode("utf-8")
    return fields


def last_form(server: FakeServer) -> dict:
    request = server.requests[-1]
    return form_fields(request["body"], request["headers"]["Content-Type"])


@pytest.fixture
def breeze(monkeypatch):
    monkeypatch.setenv("BREEZE_CUDA_ARCHITECTURES", "75")
    sys.modules.pop("providers.breeze.app", None)
    worker = importlib.import_module("providers.breeze.app")
    server = FakeServer()
    started = []
    state = {"gpu": ["NVIDIA GeForce RTX 2070 SUPER", "7.5", "8192", "1500"], "used": [1500.0, 5950.0],
             "next_log": "", "next_exit": None, "downloads": []}

    def start(command):
        process = FakeProcess(command, log=state["next_log"], exit_code=state["next_exit"])
        started.append(process)
        return process

    def download():
        state["downloads"].append(True)
        return WEIGHTS

    def used():
        return state["used"].pop(0) if state["used"] else 5950.0

    monkeypatch.setattr(worker, "_nvidia_smi", lambda *query: list(state["gpu"]))
    monkeypatch.setattr(worker, "_gpu_used_mb", used)
    monkeypatch.setattr(worker, "_weights_path", download)
    monkeypatch.setattr(worker, "_start_server", start)
    monkeypatch.setattr(worker, "_server_health", lambda timeout=2.0: dict(server.health) if server.health else None)
    monkeypatch.setattr(worker, "_connection", server.connection)
    state["pauses"] = []
    monkeypatch.setattr(worker, "_pause", state["pauses"].append)
    monkeypatch.setattr(worker, "BUSY_WAIT_S", 0.0)
    client = TestClient(worker.app)
    yield worker, client, server, started, state
    worker.runtime.unload()


def core_backend(monkeypatch, client) -> ExternalTTSBackend:
    backend = ExternalTTSBackend(provider="breeze", base_url="http://breeze.voice-workers.internal.:8240",
                                 busy_retry_s=0.0)
    monkeypatch.setattr(backend._opener, "open", _bridge_to_worker(client))
    return backend


def load(client):
    response = client.post("/v1/models/load", json={"model": MODEL})
    assert response.status_code == 200, response.text
    return response.json()


# ── Manifest, registry and health ──────────────────────────────────────────────────────────────────────────────────


def test_the_manifest_offers_one_model_that_designs_and_clones_voices(breeze):
    worker, client, *_ = breeze
    manifest = client.get("/v1/manifest").json()
    assert (manifest["schema_version"], manifest["provider"]) == (1, "breeze")
    [model] = manifest["models"]
    assert (model["id"], model["provider"], model["sample_rate"], model["voices"]) == (MODEL, "breeze", 24000, [])
    assert model["revision"] == worker.WEIGHTS_REVISION and len(model["revision"]) == 40
    assert model["max_input_chars"] == 4096
    capabilities = model["capabilities"]
    for name in ("voice_design", "voice_clone", "reference_audio", "clone_transcript", "clone_transcript_required",
                 "streaming", "instructions", "live_reader", "native_tags"):
        assert capabilities[name] is True, name
    assert (capabilities["speed_control"], capabilities["voice_blend"], capabilities["languages"]) == (
        False, False, ["en", "zh"])
    # The core's GPU memory guard reads the need from here, and the registry holds the same figure.
    row = get_known_model(MODEL)
    assert model["gpu_memory_mb"] == row["gpu_mb"] == 4450
    assert (row["provider"], row["model_format"], row["optional_provider"]) == ("breeze", "external", True)
    assert "voice design" in row["description"] and "voice cloning" in row["description"]
    # breeze-tts-2-q8_0.gguf is 3,568,844,480 bytes.
    assert row["size_mb"] == round(3_568_844_480 / 1024 / 1024)


def test_the_image_pins_the_port_and_the_weights(breeze):
    worker, *_ = breeze
    dockerfile = (ROOT / "providers" / "breeze" / "Dockerfile").read_text(encoding="utf-8")
    assert f"ARG BREEZE_REF={worker.PORT_REF}" in dockerfile
    assert f"ARG BREEZE_COMMIT={worker.PORT_COMMIT}" in dockerfile
    assert 'test "$(git rev-parse HEAD)" = "${BREEZE_COMMIT}"' in dockerfile
    assert "-DBREEZE_CUDA=ON" in dockerfile and "-DBREEZE_VULKAN=OFF" in dockerfile
    assert "lib64/stubs/libcuda.so" in dockerfile, "libggml-cuda links against the driver stub"
    assert "libgomp1" in dockerfile
    assert "--mount=type=cache" not in dockerfile, "PIP_NO_CACHE_DIR leaves pip nothing to cache"
    assert "/home/breezeworker/.nv/ComputeCache" in dockerfile, "the volume mounts over a directory the worker owns"
    assert "BREEZE_CUDA_ARCHITECTURES=${BREEZE_CUDA_ARCHITECTURES}" in dockerfile, "the worker checks the built arch"
    assert dockerfile.rstrip().splitlines()[-1] == (
        'CMD ["uvicorn", "providers.breeze.app:app", "--host", "0.0.0.0", "--port", "8240"]')
    assert (worker.WEIGHTS_REPO, worker.WEIGHTS_FILE) == ("HoppouAI/Breeze-TTS-2.cpp", "breeze-tts-2-q8_0.gguf")
    assert len(worker.WEIGHTS_REVISION) == 40 and len(worker.PORT_COMMIT) == 40
    requirements = (ROOT / "providers" / "breeze" / "requirements.txt").read_text(encoding="utf-8").split()
    assert all("==" in line for line in requirements), "every requirement is pinned"


def test_health_reads_the_gpu_with_nvidia_smi_and_reports_the_memory_the_server_holds(breeze, monkeypatch):
    worker, client, server, started, state = breeze
    health = client.get("/health").json()
    assert (health["status"], health["provider"], health["loaded_model"], health["device"]) == (
        "ok", "breeze", None, "cuda:0")
    assert health["gpu"]["name"] == "NVIDIA GeForce RTX 2070 SUPER" and health["gpu"]["capability"] == [7, 5]
    assert health["gpu_memory"] == {"model_mb": None, "need_mb": None}
    assert (health["port_commit"], health["failure"]) == (worker.PORT_COMMIT, None)

    load(client)
    health = client.get("/health").json()
    assert health["loaded_model"] == MODEL
    # Measured across the load: 5950 MiB used after it, 1500 before.
    assert health["gpu_memory"] == {"model_mb": 4450.0, "need_mb": 4450}
    backend = core_backend(monkeypatch, client)
    assert backend.gpu_memory_need_mb(MODEL) == 4450
    assert [(m.model, m.gpu_memory_mb) for m in backend.loaded_models()] == [(MODEL, 4450.0)]

    # nvidia-smi can't tell (or nothing changed): the manifest's figure stands in, so the guard still counts it.
    worker.runtime.model_mb = None
    assert client.get("/health").json()["gpu_memory"]["model_mb"] == 4450.0

    state["gpu"] = []
    assert client.get("/health").json()["status"] == "unavailable", "no GPU to read: the core reads it as offline"


@pytest.mark.parametrize(
    ("built", "capability", "status"),
    [
        pytest.param("75", "8.6", "ok", id="ptx-for-newer-gpus"),
        pytest.param("75-real", "8.6", "failed", id="turing-only-on-ampere"),
        pytest.param("75-real;86-real", "8.6", "ok", id="its-own-arch"),
        pytest.param("86", "7.5", "failed", id="newer-build-on-turing"),
    ],
)
def test_a_gpu_the_build_has_no_code_for_fails_health_and_loads(breeze, monkeypatch, built, capability, status):
    worker, client, _server, started, state = breeze
    monkeypatch.setattr(worker, "CUDA_ARCHITECTURES", built)
    state["gpu"][1] = capability
    health = client.get("/health").json()
    assert health["status"] == status
    if status == "failed":
        assert health["failure"]["code"] == "cuda_arch_unsupported"
        assert f"BREEZE_CUDA_ARCHITECTURES={capability.replace('.', '')}" in health["failure"]["message"]
        response = client.post("/v1/models/load", json={"model": MODEL})
        assert response.status_code == 503 and response.json()["detail"]["code"] == "cuda_arch_unsupported"
        assert started == []


# ── Load and unload ────────────────────────────────────────────────────────────────────────────────────────────────


def test_a_load_starts_the_server_on_the_pinned_weights_and_an_unload_stops_it(breeze):
    worker, client, _server, started, state = breeze
    answer = load(client)
    assert (answer["status"], answer["model"]) == ("loaded", MODEL)
    [process] = started
    assert process.command == [
        "breeze-server", WEIGHTS, "--host", "127.0.0.1", "--port", "7863", "--ws-port", "-1",
        "--voices-dir", "/tmp/breeze-voices", "--parallel", "1",
    ]
    assert load(client) and len(started) == 1, "a second load of the loaded model is a no-op"

    response = client.post("/v1/models/unload", json={"model": MODEL})
    assert response.json() == {"status": "unloaded", "model": MODEL}
    assert process.signals, "the server was stopped, which hands its GPU memory back"
    health = client.get("/health").json()
    assert (health["loaded_model"], health["gpu_memory"]["model_mb"]) == (None, None)

    load(client)
    assert len(started) == 2 and len(state["downloads"]) == 2, "a new server; hf_hub_download answers from its cache"


def test_the_text_encoder_can_run_on_the_cpu(breeze, monkeypatch):
    worker, client, _server, started, _state = breeze
    monkeypatch.setattr(worker, "TEXT_ENCODER_CPU", True)
    load(client)
    assert started[0].command[-1] == "--te-cpu"


@pytest.mark.parametrize(
    ("log", "status", "code"),
    [
        pytest.param("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 3400 MiB on device 0: cudaMalloc failed: "
                     "out of memory\n", 507, "out_of_memory", id="out-of-memory"),
        pytest.param("error: failed to open model file\n", 500, "model_load_failed", id="bad-weights"),
        pytest.param("WARNING: failed to allocate 512.00 MB of pinned memory: out of memory\n"
                     "error: failed to open model file\n", 500, "model_load_failed", id="pinned-memory-warning"),
        pytest.param("ggml_cuda: cudaMalloc failed: out of memory\n" + "loading tensors\n" * 6
                     + "error: failed to open model file\n", 500, "model_load_failed", id="oom-long-before-the-exit"),
    ],
)
def test_a_server_that_exits_while_loading_is_a_typed_failure(breeze, log, status, code):
    worker, client, server, started, state = breeze
    state["next_log"], state["next_exit"] = log, 1
    server.health = None
    response = client.post("/v1/models/load", json={"model": MODEL})
    assert response.status_code == status, response.text
    body = response.json()
    detail = body.get("detail") or body.get("error")
    assert detail["code"] == code
    if code == "model_load_failed":
        assert "error: failed to open model file" in detail["message"]
    assert worker.runtime.model_id is None and worker.runtime.process is None


def test_a_server_that_never_comes_up_is_stopped(breeze, monkeypatch):
    worker, client, server, started, _state = breeze
    server.health = None
    monkeypatch.setattr(worker, "LOAD_TIMEOUT_S", 0.3)
    response = client.post("/v1/models/load", json={"model": MODEL})
    assert response.status_code == 504 and response.json()["detail"]["code"] == "model_load_failed"
    assert started[0].signals and worker.runtime.process is None


def test_a_download_fetches_the_weights_without_starting_the_server(breeze):
    worker, client, _server, started, state = breeze
    response = client.post("/v1/models/download", json={"model": MODEL})
    assert response.json() == {"status": "downloaded", "model": MODEL}
    assert state["downloads"] and started == [], "no server, so no GPU memory, while it downloads"
    assert client.get("/health").json()["loaded_model"] is None
    response = client.post("/v1/models/download", json={"model": "breeze/f16"})
    assert response.status_code == 400 and response.json()["detail"]["code"] == "unknown_model"


def test_a_failed_download_is_a_typed_failure(breeze, monkeypatch):
    worker, client, *_ = breeze

    def fail():
        raise OSError("disk full")

    monkeypatch.setattr(worker, "_weights_path", fail)
    response = client.post("/v1/models/download", json={"model": MODEL})
    assert response.status_code == 500
    assert response.json()["detail"]["code"] == "model_download_failed"
    assert "disk full" in response.json()["detail"]["message"]


def test_a_server_that_will_not_stop_is_kept_and_stopped_again(breeze):
    worker, client, _server, started, _state = breeze
    load(client)
    [process] = started

    def stuck(timeout=None):
        raise subprocess.TimeoutExpired("breeze-server", timeout)

    process.wait = stuck
    process.send_signal = process.signals.append  # it ignores them
    worker.runtime.unload()
    assert len(process.signals) == 2, "asked to stop, then killed"
    assert worker.runtime.model_id is None
    assert worker.runtime.process is process, "not orphaned: it still holds its GPU memory"
    worker.runtime.unload()
    assert len(process.signals) == 4, "the next unload stops it again"
    process.returncode = -9


def test_an_exit_seen_late_does_not_clear_the_server_that_replaced_it(breeze):
    worker, client, _server, started, _state = breeze
    load(client)
    started[0].returncode = 139
    replacement = FakeProcess(["breeze-server"])

    class Drainer:
        def join(self, timeout=None):
            # Meanwhile another thread has started a new server.
            worker.runtime.process = replacement

    worker.runtime.drainer = Drainer()
    assert worker.runtime.server_exited() is True
    assert worker.runtime.process is replacement


def test_a_server_at_another_sample_rate_is_refused(breeze):
    worker, client, server, started, _state = breeze
    server.health = {"status": "ok", "sample_rate": 48000}
    response = client.post("/v1/models/load", json={"model": MODEL})
    assert response.status_code == 500 and response.json()["detail"]["code"] == "sample_rate_mismatch"
    assert started[0].signals


def test_a_server_that_dies_while_loaded_leaves_the_worker_empty(breeze, monkeypatch):
    worker, client, _server, started, _state = breeze
    load(client)
    started[0].returncode = 139
    health = client.get("/health").json()
    assert (health["loaded_model"], health["last_exit_code"]) == (None, 139)
    backend = core_backend(monkeypatch, client)
    assert backend.loaded_models() == [], "the core loads it again on the next use"


# ── Speech ─────────────────────────────────────────────────────────────────────────────────────────────────────────


def test_a_designed_voice_speaks_through_the_core_as_float32(breeze, monkeypatch):
    worker, client, server, _started, _state = breeze
    load(client)
    # An odd split between chunks: a sample's two bytes arrive apart.
    data = pcm([1000, -1000, 16384, -32768, 32767])
    server.response = lambda: speech(chunks=[data[:3], data[3:7], data[7:]])
    backend = core_backend(monkeypatch, client)
    monkeypatch.setattr(worker, "_random_seed", lambda: 7)

    audio = np.concatenate(list(backend.synthesize("Hello there.", "designed", model_id=MODEL,
                                                   voice_design=DESCRIPTION)))

    assert audio.dtype == np.float32
    assert audio.tolist() == pytest.approx([1000 / 32768, -1000 / 32768, 0.5, -1.0, 32767 / 32768])
    request = server.requests[-1]
    assert (request["method"], request["path"]) == ("POST", "/v1/audio/speech")
    assert last_form(server) == {"text": "Hello there.", "seed": "7", "instruction": DESCRIPTION, "cfg_scale": "3"}


def test_a_description_gives_a_new_voice_each_time_and_a_seed_gives_one_back(breeze, monkeypatch):
    worker, client, server, _started, _state = breeze
    load(client)
    server.response = lambda: speech()
    seeds = iter([1234, 98765])
    monkeypatch.setattr(worker, "_random_seed", lambda: next(seeds))
    backend = core_backend(monkeypatch, client)
    sent = []
    for options in ({}, {}, {"seed": 555}):
        list(backend.synthesize("Hi.", "designed", model_id=MODEL, voice_design=DESCRIPTION, **options))
        sent.append(last_form(server)["seed"])
    assert sent == ["1234", "98765", "555"]
    clone = {"reference_audio": base64.b64decode(_wav_base64(duration=2.0)), "clone_transcript": TRANSCRIPT}
    list(backend.synthesize("Hi.", "reference", model_id=MODEL, **clone))
    assert last_form(server)["seed"] == "42", "a clone keeps its one seed, so it sounds the same each time"


def test_instructions_alone_design_the_voice_and_join_a_description(breeze, monkeypatch):
    worker, client, server, _started, _state = breeze
    load(client)
    server.response = lambda: speech()
    backend = core_backend(monkeypatch, client)
    list(backend.synthesize("Hi.", "x", model_id=MODEL, instructions="An old sea captain, gruff."))
    assert last_form(server)["instruction"] == "An old sea captain, gruff."
    list(backend.synthesize("Hi.", "x", model_id=MODEL, voice_design=DESCRIPTION, instructions="Whisper it."))
    assert last_form(server)["instruction"] == f"{DESCRIPTION} Whisper it."


def test_a_cloned_voice_sends_the_recording_as_pcm_wav_with_its_transcript(breeze, monkeypatch):
    worker, client, server, _started, _state = breeze
    load(client)
    server.response = lambda: speech()
    backend = core_backend(monkeypatch, client)
    reference = base64.b64decode(_wav_base64(duration=3.0, sample_rate=44100))

    list(backend.synthesize("A cloned sentence.", "reference", model_id=MODEL, reference_audio=reference,
                            clone_transcript=TRANSCRIPT, instructions="Slowly, with care."))

    form = last_form(server)
    assert {k: v for k, v in form.items() if k != "ref_audio"} == {
        "text": "A cloned sentence.", "seed": "42", "ref_text": TRANSCRIPT, "instruction": "Slowly, with care.",
        "cfg_scale": "1",
    }
    with wave.open(io.BytesIO(form["ref_audio"])) as clip:
        assert (clip.getnchannels(), clip.getsampwidth(), clip.getframerate()) == (1, 2, 44100)
        assert clip.getnframes() == 3 * 44100


@pytest.mark.parametrize(
    ("text", "cloning", "scale"),
    [
        pytest.param("He said, (laugh) that was close.", True, "2.5", id="clone-cue"),
        pytest.param("He said, (clears throat) well.", False, "3", id="design-cue-keeps-its-guidance"),
        pytest.param("他说[笑]好的。", True, "2.5", id="chinese-cue"),
        pytest.param("Page (3) of the report.", True, "1", id="a-number-is-not-a-cue"),
        pytest.param("The results (see above) were fine.", True, "1", id="an-aside-is-not-a-cue"),
        pytest.param("Call me (optional) tomorrow.", True, "1", id="a-word-is-not-a-cue"),
        pytest.param("Ship it to the USA (United States).", True, "1", id="a-name-is-not-a-cue"),
        pytest.param("Well (sighs) fine.", True, "2.5", id="sighs"),
        pytest.param("Oh (nervous chuckle) right.", True, "2.5", id="nervous-chuckle"),
        pytest.param("他说[叹气]好吧。", True, "2.5", id="chinese-sigh"),
        pytest.param("见第[三]章。", True, "1", id="a-chinese-number-is-not-a-cue"),
    ],
)
def test_an_inline_cue_gets_the_guidance_it_needs_to_be_voiced(breeze, monkeypatch, text, cloning, scale):
    worker, client, server, _started, _state = breeze
    load(client)
    server.response = lambda: speech()
    backend = core_backend(monkeypatch, client)
    options = ({"reference_audio": base64.b64decode(_wav_base64(duration=2.0)), "clone_transcript": TRANSCRIPT}
               if cloning else {"voice_design": DESCRIPTION})
    list(backend.synthesize(text, "v", model_id=MODEL, **options))
    assert last_form(server)["cfg_scale"] == scale


@pytest.mark.parametrize(
    ("changes", "status", "code"),
    [
        pytest.param({"speed": 1.25}, 400, "speed_unsupported", id="speed"),
        pytest.param({"language": "fr"}, 400, "language_unsupported", id="french"),
        pytest.param({"voice_design": DESCRIPTION}, 400, "voice_design_with_reference", id="both-voices"),
        pytest.param({"clone_transcript": "  "}, 400, "clone_transcript_required", id="no-transcript"),
        pytest.param({"reference_audio": _wav_base64(duration=31.0)}, 400, "reference_audio_too_long", id="31-seconds"),
        pytest.param({"reference_audio": None, "clone_transcript": None}, 400, "voice_required", id="no-voice"),
        pytest.param({"reference_audio": "not base64!"}, 400, "invalid_reference_audio", id="bad-base64"),
    ],
)
def test_what_the_model_cannot_do_is_refused_before_the_server_is_asked(breeze, changes, status, code):
    worker, client, server, _started, _state = breeze
    load(client)
    body = {"model": MODEL, "text": "Hello.", "voice": "reference", "reference_audio": _wav_base64(duration=2.0),
            "clone_transcript": TRANSCRIPT, **changes}
    response = client.post("/v1/audio/speech", json={k: v for k, v in body.items() if v is not None})
    assert response.status_code == status, response.text
    assert response.json()["detail"]["code"] == code
    assert server.requests == []
    assert client.get("/health").json()["busy"] is False


def test_chinese_and_regional_english_are_spoken(breeze):
    worker, client, server, _started, _state = breeze
    load(client)
    server.response = lambda: speech()
    for language in ("zh", "zh-TW", "en-GB"):
        response = client.post("/v1/audio/speech", json={"model": MODEL, "text": "你好", "voice_design": DESCRIPTION,
                                                          "language": language})
        assert response.status_code == 200, (language, response.text)
        assert response.headers["X-Audio-Format"] == "float32le"
        assert response.headers["X-Audio-Sample-Rate"] == "24000"


def test_a_request_before_the_load_is_refused(breeze):
    worker, client, server, _started, _state = breeze
    response = client.post("/v1/audio/speech", json={"model": MODEL, "text": "Hi.", "voice_design": DESCRIPTION})
    assert response.status_code == 409 and response.json()["detail"]["code"] == "model_not_loaded"
    assert server.requests == []


@pytest.mark.parametrize(
    ("answer", "status", "code"),
    [
        pytest.param(FakeResponse(409, json.dumps({"error": "busy"}).encode()), 429, "worker_busy", id="busy"),
        pytest.param(FakeResponse(400, json.dumps({"error": "text is required"}).encode()), 400, "invalid_request",
                     id="refused"),
        pytest.param(FakeResponse(500, b"boom"), 502, "generation_failed", id="server-error"),
        pytest.param(FakeResponse(200, headers={"X-Sample-Rate": "48000"}, chunks=[b"\x00\x01"]), 500,
                     "sample_rate_mismatch", id="sample-rate"),
        pytest.param(FakeResponse(200, headers={"X-Sample-Rate": "24000"}, chunks=[]), 500, "invalid_audio",
                     id="no-audio"),
    ],
)
def test_server_answers_before_any_audio_are_typed_errors(breeze, answer, status, code):
    worker, client, server, _started, _state = breeze
    load(client)
    server.response = answer
    response = client.post("/v1/audio/speech", json={"model": MODEL, "text": "Hi.", "voice_design": DESCRIPTION})
    assert response.status_code == status, response.text
    assert response.json()["detail"]["code"] == code
    assert client.get("/health").json()["busy"] is False, "the worker is free again"


def test_a_request_waits_while_a_cancelled_stream_still_holds_the_servers_slot(breeze, monkeypatch):
    worker, client, server, _started, state = breeze
    monkeypatch.setattr(worker, "BUSY_WAIT_S", 2.0)
    load(client)
    answers = [FakeResponse(409, b"{\"error\":\"busy\"}"), FakeResponse(409, b"{\"error\":\"busy\"}"), speech()]
    server.response = lambda: answers.pop(0)
    response = client.post("/v1/audio/speech", json={"model": MODEL, "text": "Hi.", "voice_design": DESCRIPTION})
    assert response.status_code == 200
    assert len(server.requests) == 3 and state["pauses"] == [0.1, 0.2], "asked again with a growing wait"


def test_a_stream_cut_short_is_a_failure_not_a_quiet_end(breeze):
    worker, client, server, _started, _state = breeze
    load(client)
    server.response = lambda: FakeResponse(200, headers={"X-Sample-Rate": "24000"}, chunks=[pcm([1, 2]), pcm([3])],
                                           fail_after=1)
    payload = worker.SynthesisRequest(model=MODEL, text="Hi.", voice_design=DESCRIPTION)
    chunks = worker.runtime.generate(payload)
    assert next(chunks).tolist() == pytest.approx([1 / 32768, 2 / 32768])
    with pytest.raises(worker.WorkerFailure) as caught:
        next(chunks)
    assert (caught.value.code, caught.value.status_code) == ("generation_failed", 502)
    # The server crashed under it: the failure says why, and the worker holds no model any more.
    server.response = lambda: FakeResponse(200, headers={"X-Sample-Rate": "24000"}, chunks=[pcm([1])], fail_after=1)
    chunks = worker.runtime.generate(payload)
    next(chunks)
    worker.runtime.process.returncode = -9
    worker.runtime.log_tail.append("CUDA error: out of memory")
    with pytest.raises(worker.WorkerFailure) as caught:
        next(chunks)
    assert caught.value.code == "out_of_memory"
    assert worker.runtime.model_id is None


# ── Compose, the core's provider list, and the GPU memory guard ────────────────────────────────────────────────────


def test_the_compose_service_is_an_opt_in_gpu_profile_on_the_private_network():
    compose = yaml.safe_load((ROOT / "docker-compose.voice-models.yml").read_text(encoding="utf-8"))
    breeze = compose["services"]["breeze"]
    assert breeze["profiles"] == ["breeze"]
    assert breeze["networks"]["default"]["aliases"] == ["breeze.voice-workers.internal"]
    assert "ports" not in breeze, "the worker publishes no port"
    assert breeze["build"] == {"context": "providers", "dockerfile": "breeze/Dockerfile",
                               "args": {"REVISION": "${OPEN_SPEECH_REVISION:-unknown}"}}
    assert breeze["image"] == "${OPEN_SPEECH_BREEZE_IMAGE:-open-speech-breeze:0.1.0}"
    assert breeze["gpus"] == "all" and breeze["init"] is True
    assert breeze["volumes"] == ["breeze-hf-cache:/home/breezeworker/.cache/huggingface",
                                 "breeze-cuda-cache:/home/breezeworker/.nv/ComputeCache"]
    assert "breeze-hf-cache" in compose["volumes"] and "breeze-cuda-cache" in compose["volumes"]
    health = " ".join(breeze["healthcheck"]["test"])
    assert "http://localhost:8240/health" in health and "data['provider']=='breeze'" in health
    environment = dict(item.split("=", 1) for item in breeze["environment"])
    assert environment["BREEZE_MAX_REFERENCE_MB"] == "${OS_MAX_UPLOAD_MB:-${STT_MAX_UPLOAD_MB:-100}}"
    core = dict(item.split("=", 1) for item in compose["services"]["open-speech"]["environment"])
    assert json.loads(core["TTS_EXTERNAL_PROVIDERS"])["breeze"] == "http://breeze.voice-workers.internal.:8240"


def test_an_unknown_model_is_refused(breeze):
    worker, client, *_ = breeze
    assert client.post("/v1/models/load", json={"model": "breeze/f16"}).status_code == 400
    response = client.post("/v1/audio/speech", json={"model": "breeze/f16", "text": "Hi."})
    assert response.status_code == 400 and response.json()["detail"]["code"] == "unknown_model"


def test_the_core_reports_a_worker_refusal_with_its_code(breeze, monkeypatch):
    worker, client, server, _started, _state = breeze
    load(client)
    backend = core_backend(monkeypatch, client)
    with pytest.raises(ExternalProviderError) as caught:
        list(backend.synthesize("Hi.", "x", model_id=MODEL))
    assert (caught.value.status_code, caught.value.code) == (400, "voice_required")
