"""Parakeet 0.10.1 fixes: the GPU arena (shrinking, and halves for a piece that doesn't fit), the stream's cut, how
long audio is cut, scraps of audio, word ends, the VAD outside the inference lock, Silero's failed download, the
device check, what counts as downloaded, error messages without paths, and deleting during a fetch. Nothing is
downloaded and no GPU is needed: onnx-asr, ONNX Runtime's sessions and the hub are fakes."""

from __future__ import annotations

import io
import json
import math
import random
import sys
import threading
import wave
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import router as router_module
from src.backends import parakeet as parakeet_module
from src.backends import parakeet_fp16
from src.backends.parakeet import (
    MAX_PIECE_S,
    SAMPLE_RATE,
    VAD_WINDOW,
    ModelFetchRunningError,
    ModelFilesUnavailableError,
    ParakeetBackend,
    build_result,
    plan_pieces,
)
from src.config import settings
from src.main import app
from tests.test_parakeet import (  # noqa: F401 - fixtures
    MODEL,
    SPEC,
    _backend,
    _probs,
    _result,
    _snapshot,
    _wav,
    cache,
    fake_hub,
    fake_onnx_asr,
)

MAX = int(MAX_PIECE_S * SAMPLE_RATE)
REPO = "models--" + SPEC.onnx_repo.replace("/", "--")


def _wav_of(samples: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return buf.getvalue()


class _Session:
    """An ONNX Runtime session as onnx-asr holds it: its providers, and the run options of each run."""

    def __init__(self, providers, device_id="0"):
        self.providers = providers
        self.device_id = device_id
        self.runs = []

    def get_providers(self):
        return list(self.providers)

    def get_provider_options(self):
        return {"CUDAExecutionProvider": {"device_id": self.device_id}} if "CUDAExecutionProvider" in self.providers \
            else {}

    def run(self, output_names, input_feed, run_options=None):
        self.runs.append(run_options)
        return ["out"]


class _Adapter:
    """onnx-asr's adapter with timestamps around a session; ``decode`` gives each waveform's result."""

    def __init__(self, session, decode=None, calls=None):
        self.asr = SimpleNamespace(_encoder=session)
        self.decode = decode or (lambda w: _result(" one", [" one"], [0.5], [-0.1]))
        self.calls = calls if calls is not None else []

    def recognize(self, waveforms, sample_rate=16000):
        self.calls.append([len(w) for w in waveforms])
        return [self.decode(w) for w in waveforms]


def _gpu(monkeypatch, device="cuda"):
    monkeypatch.setattr(settings, "stt_device", device)
    monkeypatch.setattr(ParakeetBackend, "_cuda_available", staticmethod(lambda: True))


def _fake_fp16(monkeypatch):
    def ensure_fp16(source_dir, repo_dir, revision):
        folder = Path(repo_dir) / parakeet_fp16.FP16_DIR_NAME / parakeet_fp16.fp16_folder_name(revision)
        folder.mkdir(parents=True, exist_ok=True)
        for name in parakeet_fp16.FP16_FILES:
            (folder / name).write_bytes(b"x")
        return folder

    monkeypatch.setattr(parakeet_fp16, "ensure_fp16", ensure_fp16)


# --- 1. The GPU arena -----------------------------------------------------------------------------------------------


def test_a_cuda_load_makes_the_encoder_shrink_its_arena_after_each_run(fake_onnx_asr, cache, monkeypatch):
    _gpu(monkeypatch, "cuda:1")
    session = _Session(["CUDAExecutionProvider", "CPUExecutionProvider"], device_id="1")
    fake_onnx_asr.load_model = lambda name, path=None, **kw: SimpleNamespace(with_timestamps=lambda: _Adapter(session))
    backend = _backend()

    backend.load_model(MODEL, str(cache))

    encoder = backend._models[MODEL].asr._encoder
    assert encoder.get_providers()[0] == "CUDAExecutionProvider"  # the session's own methods pass through
    encoder.run(["outputs", "encoded_lengths"], {"audio_signal": None})
    options = session.runs[-1]
    assert options.get_run_config_entry("memory.enable_memory_arena_shrinkage") == "gpu:1"
    assert backend.loaded_models()[0].device == "cuda"


def test_the_arena_cap_and_its_growth_strategy_stay(monkeypatch):
    monkeypatch.setattr(settings, "stt_parakeet_gpu_mem_limit_mb", 2304)
    options = _backend()._providers("cuda")[0][1]
    assert options["arena_extend_strategy"] == "kSameAsRequested"
    assert options["gpu_mem_limit"] == 2304 * 1024 * 1024


def _allocation_failure():
    return RuntimeError(
        "[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned while running Conv node. "
        "Status Message: /onnxruntime_src/onnxruntime/core/framework/bfc_arena.cc:359 void* "
        "onnxruntime::BFCArena::AllocateRawInternal(size_t, bool) Available memory of 12345 is smaller than "
        "requested bytes of 67890"
    )


def _loaded(backend, adapter, monkeypatch):
    monkeypatch.setitem(backend._models, MODEL, adapter)
    monkeypatch.setitem(backend._loaded_at, MODEL, 0.0)
    return backend


def test_a_piece_the_gpu_arena_has_no_room_for_is_decoded_in_halves_split_at_its_quietest(monkeypatch):
    samples = np.full(22 * SAMPLE_RATE, 3000, dtype=np.int16)
    samples[int(7.9 * SAMPLE_RATE):int(8.1 * SAMPLE_RATE)] = 0  # a pause at 8 s

    def decode(waveform):
        if len(waveform) > 15 * SAMPLE_RATE:
            raise _allocation_failure()
        return _result(" one two", [" one", " two"], [0.5, 1.0], [-0.1, -0.2])

    calls = []
    backend = _loaded(_backend(), _Adapter(None, decode, calls), monkeypatch)

    result = backend.transcribe(_wav_of(samples), MODEL, response_format="verbose_json")

    assert [len(c) for c in calls] == [1, 1, 1]
    first, second = calls[1][0], calls[2][0]
    assert first + second == len(samples) and abs(first / SAMPLE_RATE - 8.0) < 0.1
    assert result["text"] == "one two one two"
    starts = [w["start"] for w in result["words"]]
    assert starts[:2] == [0.5, 1.0] and abs(starts[2] - (first / SAMPLE_RATE + 0.5)) < 1e-3


def test_a_failure_that_is_not_for_memory_still_fails_the_request(monkeypatch):
    def decode(waveform):
        raise RuntimeError("Invalid input shape")

    calls = []
    backend = _loaded(_backend(), _Adapter(None, decode, calls), monkeypatch)

    with pytest.raises(RuntimeError, match="Invalid input shape"):
        backend.transcribe(_wav(10.0), MODEL)
    assert len(calls) == 1


def test_splitting_stops_after_max_splits_and_the_failure_is_raised(monkeypatch):
    def decode(waveform):
        raise _allocation_failure()

    calls = []
    backend = _loaded(_backend(), _Adapter(None, decode, calls), monkeypatch)

    with pytest.raises(RuntimeError, match="Available memory"):
        backend.transcribe(_wav(24.0), MODEL)
    assert len(calls) <= 2 ** (parakeet_module.MAX_SPLITS + 1) - 1


def test_a_failed_batch_whose_pieces_dont_fit_alone_splits_them(monkeypatch):
    sizes_seen = []

    def decode(waveform):
        sizes_seen.append(len(waveform))
        return _result(" a", [" a"], [0.1], [-0.1])

    class Adapter(_Adapter):
        def recognize(self, waveforms, sample_rate=16000):
            if len(waveforms) > 1 or len(waveforms[0]) > 5 * SAMPLE_RATE:
                raise _allocation_failure()
            return super().recognize(waveforms, sample_rate)

    # Three 6 s pieces make one batch (18 s); it fails, and so does each piece alone.
    results = ParakeetBackend._recognize(Adapter(None, decode), [np.zeros(6 * SAMPLE_RATE, np.float32)] * 3)

    assert [r.text for r in results] == ["a a"] * 3
    assert sizes_seen and all(size <= 5 * SAMPLE_RATE for size in sizes_seen)


# --- 3. Cutting long audio ------------------------------------------------------------------------------------------


def _check(pieces, probs, n, threshold=0.5):
    """Pieces in order, apart, within the limit, each with speech, and covering every speech window."""
    assert all(0 <= lo < hi <= n for lo, hi in pieces)
    assert all(a[1] <= b[0] for a, b in zip(pieces, pieces[1:]))
    assert all(hi - lo <= MAX for lo, hi in pieces)
    for lo, hi in pieces:
        assert any(probs[i] >= threshold for i in range(lo // VAD_WINDOW, min(len(probs), -(-hi // VAD_WINDOW))))
    for i, prob in enumerate(probs):
        if prob >= threshold:  # covered, by one piece or by two meeting at a cut
            start, end = i * VAD_WINDOW, (i + 1) * VAD_WINDOW
            assert sum(max(0, min(hi, end) - max(lo, start)) for lo, hi in pieces) == VAD_WINDOW, i


def test_a_cut_never_leaves_a_scrap_of_a_few_samples():
    n = MAX + 1  # speech running to the end of the audio, one sample over 20 s
    probs = [0.95] * (n // VAD_WINDOW)
    probs[MAX // VAD_WINDOW - 1] = 0.6  # a slightly quieter window just before 20 s

    pieces = plan_pieces(probs, n, threshold=0.5)

    assert min(hi - lo for lo, hi in pieces) >= parakeet_module.MIN_PIECE_S * SAMPLE_RATE
    _check(pieces, probs, n)


@pytest.mark.parametrize("speech_s", [19.7, 19.8, 19.9, 19.98])
def test_speech_just_under_twenty_seconds_then_silence_is_one_piece(speech_s):
    probs = _probs([(speech_s, 0.9), (8, 0.0)])
    n = len(probs) * VAD_WINDOW

    pieces = plan_pieces(probs, n, threshold=0.5)

    assert len(pieces) == 1, pieces
    _check(pieces, probs, n)


@pytest.mark.parametrize("speech_s", [20.1, 20.3, 20.5, 21.0])
def test_speech_just_over_twenty_seconds_leaves_no_piece_under_a_second(speech_s):
    probs = _probs([(2, 0.0), (speech_s, 0.9), (8, 0.0)])
    n = len(probs) * VAD_WINDOW

    pieces = plan_pieces(probs, n, threshold=0.5)

    assert len(pieces) == 2
    assert min(hi - lo for lo, hi in pieces) >= SAMPLE_RATE
    _check(pieces, probs, n)


def test_a_cut_goes_in_the_middle_of_the_quietest_stretch():
    dip = [0.35, 0.3, 0.32, 0.31, 0.34]  # under the threshold, but too short to be a pause
    probs = _probs([(14, 0.9)]) + dip + _probs([(14, 0.9)])
    n = len(probs) * VAD_WINDOW
    first = len(_probs([(14, 0.9)]))

    pieces = plan_pieces(probs, n, threshold=0.5)

    middle = (first * VAD_WINDOW + (first + len(dip)) * VAD_WINDOW) // 2
    assert abs(pieces[0][1] - middle) < VAD_WINDOW // 4
    _check(pieces, probs, n)


def test_random_speech_is_cut_into_valid_pieces():
    rng = random.Random(7)
    for _ in range(300):
        pattern = []
        for _ in range(rng.randint(1, 12)):
            pattern.append((rng.choice([0.05, 0.2, 0.5, 2, 8, 15, 19.9, 20.2, 26, 41]), 0.9 if rng.random() < 0.6
                            else rng.choice([0.0, 0.3, 0.6])))
        probs = [min(1.0, max(0.0, p + rng.uniform(-0.05, 0.05))) for p in _probs(pattern)]
        n = len(probs) * VAD_WINDOW + rng.randint(0, VAD_WINDOW - 1)

        _check(plan_pieces(probs, n, threshold=0.5), probs, n)


# --- 4. Scraps of audio ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("samples", [1, 2, 160, 400, 800, 1599])
@pytest.mark.parametrize("response_format", ["json", "text", "srt", "vtt", "verbose_json"])
def test_audio_under_a_tenth_of_a_second_gives_empty_text_without_decoding(
    fake_onnx_asr, fake_hub, cache, samples, response_format,
):
    result = _backend().transcribe(_wav_of(np.full(samples, 3000)), MODEL, response_format=response_format)

    assert fake_onnx_asr.batches == []
    if response_format == "verbose_json":
        assert result["text"] == "" and result["segments"] == [] and result["words"] == []
        json.dumps(result, allow_nan=False)
    elif response_format == "vtt":
        assert result["text"].strip() == "WEBVTT"
    else:
        assert result["text"] == ""


def test_unknown_tokens_are_left_out_and_nan_logprobs_give_a_finite_average():
    noise = _result(" <unk><unk><unk>", ["<unk>", "<unk>", "<unk>"], [0.0, 0.08, 0.16], [math.nan] * 3)
    mixed = _result(" hello <unk> world", [" hello", " <unk>", "wor", "ld"], [0.1, 0.4, 0.5, 0.6],
                    [-0.1, math.nan, -0.3, math.nan])

    result = build_result([(0.0, 1.0, noise), (5.0, 6.0, mixed)], duration=6.0)

    assert result["text"] == "hello world"
    assert [w["word"] for w in result["words"]] == ["hello", "world"]
    assert result["segments"][0]["avg_logprob"] == pytest.approx(-0.2)
    json.dumps(result, allow_nan=False)
    nan_only = build_result([(0.0, 1.0, _result(" hm", [" hm"], [0.1], [math.nan]))], duration=1.0)
    assert nan_only["segments"][0]["avg_logprob"] == 0.0
    json.dumps(nan_only, allow_nan=False)


@pytest.mark.parametrize("response_format", ["json", "text", "srt", "vtt", "verbose_json"])
def test_http_answers_a_one_sample_clip_with_200_and_no_text(fake_onnx_asr, fake_hub, cache, monkeypatch,
                                                             response_format):
    backend = _backend()
    backend.load_model(MODEL)
    monkeypatch.setitem(router_module.router._backends, "parakeet", backend)
    client = TestClient(app)

    resp = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", _wav_of(np.full(1, 3000)), "audio/wav")},
                       data={"model": MODEL, "response_format": response_format})

    assert resp.status_code == 200, resp.text
    if response_format in ("json", "verbose_json"):
        assert resp.json()["text"] == ""
    assert fake_onnx_asr.batches == []


# --- 5. Word ends ---------------------------------------------------------------------------------------------------


def test_a_word_before_a_pause_ends_with_its_last_token_not_at_the_next_word():
    piece = _result(" Calico. Next", [" Cal", "ico", ".", " Next"], [2.16, 2.32, 2.48, 13.6], [-0.1] * 4)

    result = build_result([(0.0, 14.5, piece)], duration=14.5)

    assert [(w["word"], w["start"], w["end"]) for w in result["words"]] == [
        ("Calico.", 2.16, 2.88), ("Next", 13.6, 14.0),
    ]
    assert [(s["start"], s["end"]) for s in result["segments"]] == [(2.16, 2.88), (13.6, 14.0)]
    srt = ParakeetBackend._to_srt(result["segments"])
    assert "00:00:02,160 --> 00:00:02,880\nCalico." in srt


# --- 6. The VAD outside the inference lock --------------------------------------------------------------------------


def test_long_audio_is_scored_by_the_vad_while_another_decode_holds_the_lock(fake_onnx_asr, fake_hub, cache):
    scored = threading.Event()

    def score(samples):
        scored.set()
        return [0.9] * (len(samples) // VAD_WINDOW)

    backend = _backend(score=score)
    backend.load_model(MODEL)
    holding, release = threading.Event(), threading.Event()

    def hold():
        with backend._inference_lock:
            holding.set()
            release.wait(10)

    holder = threading.Thread(target=hold)
    holder.start()
    assert holding.wait(5)
    worker = threading.Thread(target=lambda: backend.transcribe(_wav(30.0), MODEL))
    worker.start()
    try:
        assert scored.wait(5), "the VAD waited for the inference lock"
    finally:
        release.set()
        holder.join(5)
        worker.join(10)
    assert fake_onnx_asr.batches


# --- 7. Silero's failed download ------------------------------------------------------------------------------------


@pytest.fixture
def no_silero(tmp_path, monkeypatch):
    from src.vad import silero

    monkeypatch.setattr(silero, "SILERO_CACHE_DIR", tmp_path / "silero")
    monkeypatch.setattr(silero, "_vad_model", None)
    monkeypatch.setattr(silero, "_last_failure", None)
    attempts = []

    def download(model_path):
        attempts.append(model_path)
        raise TimeoutError("Silero VAD model download took longer than 60 s")

    monkeypatch.setattr(silero, "_download_model", download)
    return SimpleNamespace(module=silero, attempts=attempts)


def test_a_failed_blocking_download_is_remembered_and_not_retried_at_once(no_silero, monkeypatch):
    silero = no_silero.module

    with pytest.raises(TimeoutError):
        silero.load_vad_model_blocking()
    with pytest.raises(RuntimeError, match="download failed"):
        silero.load_vad_model_blocking()
    assert len(no_silero.attempts) == 1

    when, error = silero._last_failure
    monkeypatch.setattr(silero, "_last_failure", (when - silero.SILERO_RETRY_AFTER_S - 1, error))
    with pytest.raises(TimeoutError):
        silero.load_vad_model_blocking()
    assert len(no_silero.attempts) == 2


@pytest.mark.asyncio
async def test_a_failed_async_download_spares_the_blocking_path_a_wait(no_silero):
    silero = no_silero.module

    with pytest.raises(TimeoutError):
        await silero.get_vad_model()
    with pytest.raises(RuntimeError, match="download failed"):
        silero.load_vad_model_blocking()
    assert len(no_silero.attempts) == 1


def test_long_audio_without_silero_is_cut_evenly_and_only_tries_the_download_once(fake_onnx_asr, fake_hub, cache,
                                                                                 no_silero):
    backend = ParakeetBackend()  # the real score_speech

    for _ in range(3):
        assert backend.transcribe(_wav(50.0), MODEL)["text"]
    assert len(no_silero.attempts) == 1


# --- 8. The device ------------------------------------------------------------------------------------------------


@pytest.fixture
def cuda_provider(monkeypatch):
    """ONNX Runtime listing its CUDA provider (as the GPU package always does); .devices is CTranslate2's count."""
    import onnxruntime as ort

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(parakeet_module, "_cuda_device_count", None)
    module = ModuleType("ctranslate2")
    module.devices = 0
    module.get_cuda_device_count = lambda: module.devices
    monkeypatch.setitem(sys.modules, "ctranslate2", module)
    monkeypatch.setattr(settings, "stt_device", "cuda")
    return module


def test_the_gpu_image_with_no_gpu_visible_runs_on_the_cpu(cuda_provider):
    assert _backend().runtime() == ("cpu", "int8")


def test_a_visible_gpu_runs_fp16(cuda_provider):
    cuda_provider.devices = 1
    assert _backend().runtime() == ("cuda", "float16")


def test_no_ctranslate2_trusts_onnx_runtime_and_leaves_the_check_to_the_load(cuda_provider, monkeypatch):
    def broken():
        raise RuntimeError("no CUDA driver")

    cuda_provider.get_cuda_device_count = broken
    assert _backend().runtime() == ("cuda", "float16")


def test_a_cuda_session_that_fell_back_to_the_cpu_is_reloaded_from_the_int8_files(fake_onnx_asr, fake_hub, cache,
                                                                                    monkeypatch):
    _gpu(monkeypatch)
    _fake_fp16(monkeypatch)
    loads = []

    def load_model(name, path=None, **kwargs):
        loads.append((kwargs["quantization"], Path(path).name))
        # ORT gives a CPU session when CUDA can't start, whatever it was asked for.
        return SimpleNamespace(with_timestamps=lambda: _Adapter(_Session(["CPUExecutionProvider"])))

    fake_onnx_asr.load_model = load_model
    backend = _backend()

    backend.load_model(MODEL)

    assert [q for q, _ in loads] == ["fp16", "int8"]
    assert [call["allow_patterns"] for call in fake_hub] == [list(parakeet_fp16.SOURCE_FILES),
                                                            list(parakeet_module.INT8_FILES)]
    info = backend.loaded_models()[0]
    assert (info.device, info.compute_type) == ("cpu", "int8")
    assert backend.runtime() == ("cpu", "int8")
    assert not isinstance(backend._models[MODEL].asr._encoder, parakeet_module._ArenaShrinkingSession)


# --- 9. What counts as downloaded -----------------------------------------------------------------------------------


def _listed(backend):
    return [m["model"] for m in backend.list_cached_models()]


def test_on_the_cpu_only_the_int8_files_count(cache):
    backend = _backend()
    (cache / REPO).mkdir(parents=True)
    assert not backend.is_model_cached(MODEL) and _listed(backend) == []

    _snapshot(cache, parakeet_fp16.SOURCE_FILES)
    assert not backend.is_model_cached(MODEL) and _listed(backend) == []

    _snapshot(cache, parakeet_module.INT8_FILES)
    assert backend.is_model_cached(MODEL) and _listed(backend) == [MODEL]


def test_on_the_gpu_the_fp16_folder_or_every_fp32_file_counts(cache, monkeypatch):
    _gpu(monkeypatch)
    backend = _backend()
    _snapshot(cache, parakeet_module.INT8_FILES)
    assert not backend.is_model_cached(MODEL) and _listed(backend) == []

    fp32_only = [name for name in parakeet_fp16.SOURCE_FILES if name not in parakeet_module.INT8_FILES]
    _snapshot(cache, fp32_only[:-1])
    assert not backend.is_model_cached(MODEL)
    _snapshot(cache, parakeet_fp16.SOURCE_FILES)
    assert backend.is_model_cached(MODEL) and _listed(backend) == [MODEL]

    for path in (cache / REPO / "snapshots").rglob("*"):
        if path.is_file():
            path.unlink()
    assert not backend.is_model_cached(MODEL)
    _fake_fp16(monkeypatch)
    parakeet_fp16.ensure_fp16(None, cache / REPO, SPEC.revision)
    assert backend.is_model_cached(MODEL) and _listed(backend) == [MODEL]


# --- 10. Error messages ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("offline", ["raises", "empty_snapshot"])
def test_files_missing_while_the_hub_is_unreachable_say_so_without_paths(fake_onnx_asr, cache, monkeypatch, offline):
    import huggingface_hub
    from huggingface_hub.errors import LocalEntryNotFoundError

    def snapshot_download(repo_id, revision=None, allow_patterns=None, cache_dir=None, **kwargs):
        if offline == "raises":
            raise LocalEntryNotFoundError(f"Cannot find an appropriate cached snapshot folder in {cache_dir}")
        return str(_snapshot(Path(cache_dir), []))  # what the hub hands back offline: the folder, as it is

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)

    with pytest.raises(ModelFilesUnavailableError) as info:
        _backend().load_model(MODEL)

    message = str(info.value)
    assert "aren't downloaded and Hugging Face can't be reached" in message
    assert str(cache) not in message and "/" not in message.replace(MODEL, "")
    assert info.value.code == "model_files_unavailable"
    assert fake_onnx_asr.loads == []


def test_a_load_error_naming_a_path_is_passed_on_without_it(fake_onnx_asr, fake_hub, cache):
    def load_model(name, path=None, **kwargs):
        raise FileNotFoundError(f"File {Path(path) / 'encoder-model.int8.onnx'} not found")

    fake_onnx_asr.load_model = load_model

    with pytest.raises(Exception) as info:
        _backend().load_model(MODEL)

    assert "encoder-model.int8.onnx not found" in str(info.value)
    assert str(cache) not in str(info.value)
    assert str(cache) in str(info.value.__cause__)  # the log's copy keeps it


@pytest.fixture
def failing_http(monkeypatch):
    class Failing(ParakeetBackend):
        error: Exception = RuntimeError("x")

        def is_model_loaded(self, model_id):
            return True

        def transcribe(self, audio, model, **kwargs):
            raise self.error

    backend = Failing()
    monkeypatch.setitem(router_module.router._backends, "parakeet", backend)
    client = TestClient(app)
    client.backend = backend
    return client


def _post(client):
    return client.post("/v1/audio/transcriptions", files={"file": ("a.wav", _wav(1.0), "audio/wav")},
                       data={"model": MODEL})


def test_http_500_leaves_the_servers_paths_out(failing_http):
    failing_http.backend.error = RuntimeError(
        "Load model from /root/.cache/huggingface/hub/models--istupakov--x/fp16/rev-c1/encoder-model.fp16.onnx failed"
    )

    resp = _post(failing_http)

    assert resp.status_code == 500
    assert "encoder-model.fp16.onnx failed" in resp.text and "/root" not in resp.text


def test_http_answers_files_missing_offline_with_503(failing_http):
    failing_http.backend.error = ModelFilesUnavailableError("Parakeet's files aren't downloaded and Hugging Face "
                                                            "can't be reached")

    resp = _post(failing_http)

    assert resp.status_code == 503, resp.text
    assert resp.json()["error"]["code"] == "model_files_unavailable"


# --- 11. Deleting during a fetch ------------------------------------------------------------------------------------


def test_the_cache_is_not_deleted_while_a_fetch_writes_it(cache, monkeypatch):
    started, release = threading.Event(), threading.Event()

    def download_files(self, model_id, device):
        started.set()
        release.wait(5)
        return _snapshot(cache, parakeet_module.INT8_FILES)

    monkeypatch.setattr(ParakeetBackend, "_download_files", download_files)
    backend = _backend()
    (cache / REPO).mkdir(parents=True)

    future = backend.start_fetch(MODEL)
    assert started.wait(5)
    try:
        with pytest.raises(ModelFetchRunningError, match="being downloaded"):
            backend.delete_cached_model(MODEL)
        assert (cache / REPO).is_dir()
    finally:
        release.set()
    future.result(5)

    assert backend.delete_cached_model(MODEL) is True
    assert not (cache / REPO).exists()
