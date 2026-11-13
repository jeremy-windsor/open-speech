"""NVIDIA Parakeet through onnx-asr: the backend with a fake onnx_asr, cutting long audio, timestamps, the requests it
refuses, the fp16 converter on a tiny graph, and the documentation of its model and setting. Nothing is downloaded."""

from __future__ import annotations

import io
import itertools
import re
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
from src.backends.faster_whisper import NetworkModelPathError
from src.backends.parakeet import (
    MAX_PIECE_S,
    SAMPLE_RATE,
    LanguageUnsupportedError,
    ParakeetBackend,
    build_result,
    check_language,
    even_pieces,
    is_english_or_auto,
    plan_pieces,
)
from src.config import settings
from src.main import app

ROOT = Path(__file__).resolve().parents[1]
MODEL = "nvidia/parakeet-tdt-0.6b-v2"
SPEC = parakeet_module.PARAKEET_MODELS[MODEL]
WINDOW_S = parakeet_module.VAD_WINDOW / SAMPLE_RATE


def _wav(seconds: float, rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.ones(int(seconds * rate), dtype=np.int16) * 100).tobytes())
    return buf.getvalue()


def _result(text: str, tokens=None, timestamps=None, logprobs=None):
    return SimpleNamespace(text=text, tokens=tokens, timestamps=timestamps, logprobs=logprobs)


class _FakeAdapter:
    """onnx-asr's adapter with timestamps: records each batch it is given."""

    def __init__(self, calls: list, fail_batches_over: int | None = None):
        self.calls = calls
        self.fail_batches_over = fail_batches_over

    def recognize(self, waveforms, sample_rate=16000):
        self.calls.append([len(w) for w in waveforms])
        if self.fail_batches_over is not None and len(waveforms) > self.fail_batches_over:
            raise RuntimeError("Failed to allocate memory for requested buffer")
        return [
            _result(" hello world.", [" hello", " world", "."], [0.0, 0.48, 0.8], [-0.1, -0.2, -0.3])
            for _ in waveforms
        ]


@pytest.fixture
def fake_onnx_asr(monkeypatch):
    """A fake onnx_asr module; .loads records each load_model call, .batches each recognize batch."""
    module = ModuleType("onnx_asr")
    module.loads = []
    module.batches = []
    module.fail_batches_over = None

    def load_model(name, path=None, **kwargs):
        module.loads.append({"name": name, "path": path, **kwargs})
        return SimpleNamespace(with_timestamps=lambda: _FakeAdapter(module.batches, module.fail_batches_over))

    module.load_model = load_model
    monkeypatch.setitem(sys.modules, "onnx_asr", module)
    return module


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """STT_MODEL_DIR in a temporary folder, CPU device."""
    monkeypatch.setattr(settings, "stt_model_dir", str(tmp_path / "hub"))
    monkeypatch.setattr(settings, "stt_device", "cpu")
    return tmp_path / "hub"


def _snapshot(cache_dir: Path, files) -> Path:
    snapshot = cache_dir / ("models--" + SPEC.onnx_repo.replace("/", "--")) / "snapshots" / SPEC.revision
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in files:
        (snapshot / name).write_bytes(b"x")
    return snapshot


@pytest.fixture
def fake_hub(monkeypatch, cache):
    """huggingface_hub.snapshot_download writing the files it is asked for; .calls records each call."""
    import huggingface_hub

    calls = []

    def snapshot_download(repo_id, revision=None, allow_patterns=None, cache_dir=None, **kwargs):
        calls.append({"repo_id": repo_id, "revision": revision, "allow_patterns": list(allow_patterns),
                      "cache_dir": cache_dir})
        return str(_snapshot(Path(cache_dir), allow_patterns))

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    return calls


def _backend(**kwargs) -> ParakeetBackend:
    return ParakeetBackend(score_speech_fn=kwargs.pop("score", None) or (lambda samples: []))


# --- Routing, registry, languages --------------------------------------------------------------------------------


def test_the_router_sends_parakeet_ids_to_parakeet_and_every_other_to_faster_whisper():
    router = router_module.BackendRouter()
    assert router.get_backend(MODEL).name == "parakeet"
    assert router.get_backend("Systran/faster-whisper-base").name == "faster-whisper"
    assert router.get_backend("nvidia/parakeet-unknown").name == "faster-whisper"


def test_the_registry_lists_parakeet_as_an_english_stt_model_that_cannot_translate():
    from src.model_manager import ModelManager
    from src.model_registry import get_known_model, stt_model_can_translate, stt_model_languages

    row = get_known_model(MODEL)
    assert (row["type"], row["provider"], row["model_format"]) == ("stt", "parakeet", "ONNX")
    assert stt_model_languages(MODEL) == ["en"]
    assert stt_model_can_translate(MODEL) is False
    manager = ModelManager(stt_router=router_module.BackendRouter(), tts_router=SimpleNamespace(_backends={}))
    assert manager.resolve_provider(MODEL) == "parakeet"


@pytest.mark.parametrize("language", [None, "", " ", "auto", "AUTO", "en", "EN", "en-US", "en_gb"])
def test_english_and_no_language_are_accepted(language):
    assert is_english_or_auto(language)
    check_language(MODEL, language)


@pytest.mark.parametrize("language", ["de", "fr-FR", "eng", "es"])
def test_other_languages_are_refused_naming_a_whisper_model(language):
    assert not is_english_or_auto(language)
    with pytest.raises(LanguageUnsupportedError, match="English only") as info:
        check_language(MODEL, language)
    assert info.value.code == "language_unsupported"
    assert "faster-whisper" in str(info.value)
    check_language("Systran/faster-whisper-base", language)  # Whisper models are not affected


# --- HTTP ---------------------------------------------------------------------------------------------------------


@pytest.fixture
def http(monkeypatch):
    """The app with a recording Parakeet backend: .calls holds each transcribe's kwargs."""
    calls = []

    class Recording(ParakeetBackend):
        def is_model_loaded(self, model_id):
            return True

        def transcribe(self, audio, model, **kwargs):
            calls.append({"model": model, **kwargs})
            parakeet_module.check_language(model, kwargs.get("language"))
            return {"text": "hello"}

    backend = Recording()
    monkeypatch.setitem(router_module.router._backends, "parakeet", backend)
    monkeypatch.setattr(settings, "stt_model", MODEL)
    client = TestClient(app)
    client.calls = calls
    return client


def _post(client, path="/v1/audio/transcriptions", **data):
    return client.post(path, files={"file": ("a.wav", _wav(1.0), "audio/wav")}, data=data)


@pytest.mark.parametrize("model", [MODEL, "whisper-1"])
def test_http_refuses_another_language_with_400_language_unsupported_before_decoding(http, model):
    resp = _post(http, language="de", model=model)

    assert resp.status_code == 400, resp.text
    error = resp.json()["error"]
    assert error["code"] == "language_unsupported"
    assert "English only" in error["message"] and "faster-whisper" in error["message"]
    assert http.calls == []


def test_http_accepts_english_and_ignores_prompt_and_temperature(http):
    resp = _post(http, model=MODEL, language="en-US", prompt="Kubernetes, Grafana", temperature="0.4")

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"text": "hello"}
    assert http.calls[0]["language"] == "en"


@pytest.mark.parametrize("model", [MODEL, "whisper-1", None])
def test_http_refuses_translation_on_parakeet_naming_the_models_that_translate(http, model):
    data = {} if model is None else {"model": model}
    resp = _post(http, "/v1/audio/translations", **data)

    assert resp.status_code == 400, resp.text
    error = resp.json()["error"]
    assert error["code"] == "translation_unsupported"
    assert "Systran/faster-whisper-medium" in error["message"] and "STT_TRANSLATE_MODEL" in error["message"]


def test_http_translation_with_a_translate_model_set_goes_to_whisper(http, monkeypatch):
    whisper = SimpleNamespace(
        name="faster-whisper", is_model_loaded=lambda m: True, loaded_models=lambda: [],
        translate=lambda audio, model, **kw: {"text": f"translated by {model}"},
    )
    monkeypatch.setattr(router_module.router, "_default_backend", whisper)
    monkeypatch.setattr(settings, "stt_translate_model", "Systran/faster-whisper-medium")

    resp = _post(http, "/v1/audio/translations", model="whisper-1")

    assert resp.status_code == 200, resp.text
    assert resp.json() == {"text": "translated by Systran/faster-whisper-medium"}


def test_a_refusal_raised_by_the_backend_is_a_400(monkeypatch):
    from src.services import stt as stt_service

    exc = LanguageUnsupportedError("no")
    refusal = stt_service._client_error(exc)
    assert refusal.status_code == 400 and refusal.detail["code"] == "language_unsupported"
    assert stt_service._client_error(RuntimeError("boom")) is None


# --- Load, unload, fetch ------------------------------------------------------------------------------------------


def test_cpu_fetches_the_int8_files_at_the_pinned_revision_and_loads_them(fake_onnx_asr, fake_hub, cache):
    backend = _backend()

    backend.load_model(MODEL)

    assert fake_hub == [{
        "repo_id": SPEC.onnx_repo, "revision": SPEC.revision, "allow_patterns": list(parakeet_module.INT8_FILES),
        "cache_dir": str(cache),
    }]
    load = fake_onnx_asr.loads[0]
    assert (load["name"], load["quantization"], load["providers"]) == (
        "nemo-parakeet-tdt-0.6b-v2", "int8", ["CPUExecutionProvider"],
    )
    assert Path(load["path"]) == cache / f"models--{SPEC.onnx_repo.replace('/', '--')}" / "snapshots" / SPEC.revision
    info = backend.loaded_models()[0]
    assert (info.model, info.backend, info.device, info.compute_type) == (MODEL, "parakeet", "cpu", "int8")
    assert backend.is_model_loaded(MODEL)


def test_files_already_in_the_cache_load_without_the_network(fake_onnx_asr, fake_hub, cache):
    _snapshot(cache, parakeet_module.INT8_FILES)

    _backend().load_model(MODEL)

    assert fake_hub == []
    assert len(fake_onnx_asr.loads) == 1


def test_gpu_fetches_fp32_converts_once_and_loads_fp16_with_the_capped_cuda_session(
    fake_onnx_asr, fake_hub, cache, monkeypatch,
):
    monkeypatch.setattr(settings, "stt_device", "cuda")
    monkeypatch.setattr(settings, "stt_parakeet_gpu_mem_limit_mb", 2304)
    monkeypatch.setattr(ParakeetBackend, "_cuda_available", staticmethod(lambda: True))
    conversions = []

    def ensure_fp16(source_dir, repo_dir, revision):
        conversions.append((Path(source_dir), Path(repo_dir), revision))
        folder = Path(repo_dir) / parakeet_fp16.FP16_DIR_NAME / parakeet_fp16.fp16_folder_name(revision)
        folder.mkdir(parents=True)
        for name in parakeet_fp16.FP16_FILES:
            (folder / name).write_bytes(b"x")
        return folder

    monkeypatch.setattr(parakeet_fp16, "ensure_fp16", ensure_fp16)

    backend = _backend()
    backend.load_model(MODEL)
    backend.unload_model(MODEL)
    backend.load_model(MODEL)

    assert [call["allow_patterns"] for call in fake_hub] == [list(parakeet_fp16.SOURCE_FILES)]
    assert len(conversions) == 1 and conversions[0][2] == SPEC.revision
    load = fake_onnx_asr.loads[-1]
    assert load["quantization"] == "fp16"
    cuda, options = load["providers"][0]
    assert cuda == "CUDAExecutionProvider" and load["providers"][1] == "CPUExecutionProvider"
    assert options == {
        "cudnn_conv_algo_search": "HEURISTIC", "arena_extend_strategy": "kSameAsRequested",
        "gpu_mem_limit": 2304 * 1024 * 1024,
    }
    assert backend.loaded_models()[0].compute_type == "float16"


def test_cuda_without_the_cuda_provider_falls_back_to_int8_on_the_cpu(monkeypatch):
    monkeypatch.setattr(ParakeetBackend, "_cuda_available", staticmethod(lambda: False))
    for device in ("cuda", "auto", "cpu"):
        monkeypatch.setattr(settings, "stt_device", device)
        assert _backend().runtime() == ("cpu", "int8")


def test_no_gpu_cap_when_the_setting_is_zero(monkeypatch):
    monkeypatch.setattr(settings, "stt_parakeet_gpu_mem_limit_mb", 0)
    assert "gpu_mem_limit" not in _backend()._providers("cuda")[0][1]


def test_concurrent_fetches_of_one_model_share_one_download(fake_onnx_asr, monkeypatch, cache):
    started, release = threading.Event(), threading.Event()
    downloads = []

    def download_files(self, model_id, device):
        downloads.append(model_id)
        started.set()
        release.wait(5)
        return _snapshot(cache, parakeet_module.INT8_FILES)

    monkeypatch.setattr(ParakeetBackend, "_download_files", download_files)
    backend = _backend()

    first = backend.start_fetch(MODEL)
    assert started.wait(5)
    second = backend.start_fetch(MODEL)
    release.set()

    assert first is second
    assert first.result(5) == second.result(5)
    assert downloads == [MODEL]


def test_a_failed_fetch_raises_loads_nothing_and_the_next_fetch_tries_again(fake_onnx_asr, monkeypatch, cache):
    attempts = []

    def download_files(self, model_id, device):
        attempts.append(model_id)
        raise OSError("network down")

    monkeypatch.setattr(ParakeetBackend, "_download_files", download_files)
    backend = _backend()

    for _ in range(2):
        with pytest.raises(OSError, match="network down"):
            backend.load_model(MODEL)
    assert attempts == [MODEL, MODEL]
    assert fake_onnx_asr.loads == [] and not backend.is_model_loaded(MODEL)


@pytest.mark.parametrize("model_id", ["//host/share", "\\\\host\\share\\model"])
def test_a_network_path_is_refused_before_anything_reads_it(model_id):
    with pytest.raises(NetworkModelPathError):
        _backend().start_fetch(model_id)


def test_an_unknown_model_is_refused():
    with pytest.raises(ValueError, match="Unknown Parakeet model"):
        _backend().load_model("nvidia/parakeet-other")


def test_unload_if_idle_leaves_a_model_that_is_serving(fake_onnx_asr, fake_hub, cache):
    backend = _backend()
    backend.load_model(MODEL)
    holding, release = threading.Event(), threading.Event()

    def hold():
        with backend._inference_lock:
            holding.set()
            release.wait(5)

    worker = threading.Thread(target=hold)
    worker.start()
    assert holding.wait(5)
    try:
        assert backend.unload_if_idle(MODEL) is False
        assert backend.is_model_loaded(MODEL)
    finally:
        release.set()
        worker.join(5)
    assert backend.unload_if_idle(MODEL) is True
    assert not backend.is_model_loaded(MODEL) and backend.loaded_models() == []


def test_unload_drops_the_session_so_its_memory_can_go(fake_onnx_asr, fake_hub, cache):
    import gc
    import weakref

    backend = _backend()
    backend.load_model(MODEL)
    session = weakref.ref(backend._models[MODEL])

    backend.unload_model(MODEL)
    gc.collect()

    assert session() is None


@pytest.mark.parametrize(("pinned", "expected"), [(True, None), (False, 300.0)])
def test_the_pinned_default_reports_no_idle_timeout(fake_onnx_asr, fake_hub, cache, monkeypatch, pinned, expected):
    monkeypatch.setattr(settings, "os_model_ttl", 300)
    monkeypatch.setattr(settings, "os_pin_default_models", pinned)
    monkeypatch.setattr(settings, "stt_model", MODEL)
    backend = _backend()
    backend.load_model(MODEL)

    ttl = backend.loaded_models()[0].ttl_remaining
    assert ttl == expected if expected is None else ttl == pytest.approx(expected, abs=2)


def test_the_cache_lists_and_deletes_the_model_under_its_own_id(fake_onnx_asr, fake_hub, cache):
    backend = _backend()
    assert backend.list_cached_models() == [] and not backend.is_model_cached(MODEL)
    backend.download_model(MODEL)

    listed = backend.list_cached_models()
    assert [(m["model"], m["backend"], m["loaded"]) for m in listed] == [(MODEL, "parakeet", False)]
    assert backend.is_model_cached(MODEL)
    assert backend.delete_cached_model(MODEL) is True
    assert not backend.is_model_cached(MODEL) and backend.list_cached_models() == []
    assert backend.delete_cached_model(MODEL) is False


# --- Transcription ------------------------------------------------------------------------------------------------


def test_short_audio_is_one_piece_with_no_vad(fake_onnx_asr, fake_hub, cache):
    scored = []
    backend = _backend(score=lambda samples: scored.append(len(samples)) or [])

    result = backend.transcribe(_wav(10.0), MODEL, language="en", prompt="ignored", temperature=0.7)

    assert result == {"text": "hello world."}
    assert scored == [] and fake_onnx_asr.batches == [[10 * SAMPLE_RATE]]


def _within_batch_limits(batches) -> bool:
    return all(
        len(batch) <= parakeet_module.BATCH_SIZE
        and max(batch) * len(batch) <= parakeet_module.BATCH_MAX_S * SAMPLE_RATE
        for batch in batches
    )


def test_long_audio_is_cut_with_the_vad_into_pieces_of_at_most_20_seconds(fake_onnx_asr, fake_hub, cache,
                                                                         monkeypatch):
    monkeypatch.setattr(settings, "stt_vad_threshold", 0.5)
    seconds = 125.0
    windows = int(seconds * SAMPLE_RATE) // parakeet_module.VAD_WINDOW
    # Speech everywhere but a 1 s pause every 10 s.
    probs = [0.0 if (i * WINDOW_S) % 10 > 9 else 0.9 for i in range(windows)]
    backend = _backend(score=lambda samples: probs)

    result = backend.transcribe(_wav(seconds), MODEL, response_format="verbose_json")

    sizes = [size for batch in fake_onnx_asr.batches for size in batch]
    assert all(size <= MAX_PIECE_S * SAMPLE_RATE for size in sizes)
    assert _within_batch_limits(fake_onnx_asr.batches)
    assert len(sizes) == len(result["segments"]) >= 6
    starts = [segment["start"] for segment in result["segments"]]
    assert starts == sorted(starts) and starts[0] == 0.0 and starts[1] > 15


def _short_utterances(count: int) -> tuple[list[float], float]:
    """Speech of 3 s, then 20 s of silence, ``count`` times: pieces too far apart to join."""
    pattern = [(3, 0.9), (20, 0.0)] * count
    return _probs(pattern), sum(seconds for seconds, _ in pattern)


def test_short_pieces_are_decoded_four_at_a_time_and_kept_in_order(fake_onnx_asr, fake_hub, cache):
    probs, seconds = _short_utterances(6)

    result = _backend(score=lambda samples: probs).transcribe(_wav(seconds), MODEL, response_format="verbose_json")

    assert [len(batch) for batch in fake_onnx_asr.batches] == [4, 2]
    assert _within_batch_limits(fake_onnx_asr.batches)
    starts = [segment["start"] for segment in result["segments"]]
    assert len(starts) == 6 and starts == sorted(starts)


def test_batches_hold_at_most_25_seconds_of_padded_audio():
    second = SAMPLE_RATE
    batches = ParakeetBackend._batches([20 * second, 3 * second, 19 * second, 2 * second, 6 * second, 5 * second])
    assert batches == [[3, 1, 5, 4], [2], [0]]
    assert ParakeetBackend._batches([25 * second]) == [[0]]
    assert ParakeetBackend._batches([]) == []


def test_a_failed_batch_is_decoded_one_piece_at_a_time(fake_onnx_asr, fake_hub, cache):
    fake_onnx_asr.fail_batches_over = 1
    probs, seconds = _short_utterances(6)

    result = _backend(score=lambda samples: probs).transcribe(_wav(seconds), MODEL)

    assert result["text"].count("hello world.") == 6
    assert [len(batch) for batch in fake_onnx_asr.batches] == [4, 1, 1, 1, 1, 2, 1, 1]


def test_long_audio_is_cut_evenly_when_the_vad_cannot_load(fake_onnx_asr, fake_hub, cache):
    def broken(samples):
        raise RuntimeError("no Silero")

    _backend(score=broken).transcribe(_wav(50.0), MODEL)

    sizes = sorted(size for batch in fake_onnx_asr.batches for size in batch)
    assert sum(sizes) == 50 * SAMPLE_RATE and len(sizes) == 3 and sizes[-1] - sizes[0] <= 1


def test_another_language_is_refused_before_the_model_loads(fake_onnx_asr, fake_hub, cache):
    with pytest.raises(LanguageUnsupportedError):
        _backend().transcribe(_wav(1.0), MODEL, language="de")
    assert fake_hub == [] and fake_onnx_asr.loads == []


def test_translate_is_refused():
    with pytest.raises(ValueError) as info:
        _backend().translate(_wav(1.0), MODEL)
    assert info.value.code == "translation_unsupported"


@pytest.mark.parametrize("rate", [8000, 44100])
def test_audio_at_another_rate_is_resampled_to_16_khz(rate):
    samples = parakeet_module.decode_audio(_wav(1.0, rate=rate))
    assert samples.dtype == np.float32 and abs(len(samples) - SAMPLE_RATE) < 50


# --- Cutting ------------------------------------------------------------------------------------------------------


def _probs(seconds_pattern: list[tuple[float, float]]) -> list[float]:
    probs: list[float] = []
    for seconds, prob in seconds_pattern:
        probs.extend([prob] * round(seconds / WINDOW_S))
    return probs


def _cut(probs: list[float]):
    n = len(probs) * parakeet_module.VAD_WINDOW
    return plan_pieces(probs, n, threshold=0.5), n


def test_pieces_join_speech_up_to_twenty_seconds():
    pieces, n = _cut(_probs([(5, 0.9), (2, 0.0), (5, 0.9), (2, 0.0), (5, 0.9), (2, 0.0), (5, 0.9)]))

    assert all(hi - lo <= MAX_PIECE_S * SAMPLE_RATE for lo, hi in pieces)
    assert len(pieces) == 2
    assert pieces[0][0] == 0 and pieces[-1][1] == n


def test_speech_with_no_pause_is_hard_cut_at_its_quietest_window():
    pattern = [(14, 0.9), (0.064, 0.4), (30, 0.9)]  # a dip under 0.9 that is still speech at 14 s
    pieces, n = _cut(_probs(pattern))

    assert all(hi - lo <= MAX_PIECE_S * SAMPLE_RATE for lo, hi in pieces)
    assert abs(pieces[0][1] / SAMPLE_RATE - 14.06) < 0.1
    assert pieces[-1][1] == n and len(pieces) == 3


def test_flat_speech_is_cut_at_exactly_twenty_seconds():
    pieces, n = _cut([0.9] * round(45 / WINDOW_S))

    assert [hi - lo for lo, hi in pieces][:2] == [int(MAX_PIECE_S * SAMPLE_RATE)] * 2
    assert pieces[-1][1] == n


def test_silence_gives_no_pieces_and_a_long_silence_is_left_out():
    assert _cut([0.0] * 1000)[0] == []
    pieces, _ = _cut(_probs([(3, 0.9), (60, 0.0), (3, 0.9)]))
    assert len(pieces) == 2
    assert sum(hi - lo for lo, hi in pieces) < 8 * SAMPLE_RATE


def test_even_pieces_cover_the_audio_within_the_limit():
    pieces = even_pieces(int(61 * SAMPLE_RATE))
    assert len(pieces) == 4 and pieces[0][0] == 0 and pieces[-1][1] == int(61 * SAMPLE_RATE)
    assert all(b[0] == a[1] for a, b in itertools.pairwise(pieces))
    assert even_pieces(0) == []


# --- Timestamps and formats ---------------------------------------------------------------------------------------


def _two_pieces():
    first = _result(
        " He has grave doubts. Truly?",
        [" He", " has", " g", "ra", "ve", " doubts", ".", " Tr", "uly", "?"],
        [0.32, 0.64, 0.88, 0.96, 1.12, 1.6, 2.0, 2.4, 2.56, 2.8],
        [-0.1] * 10,
    )
    second = _result(" Yes", [" Yes"], [0.16], [-0.5])
    return [(0.0, 3.5, first), (30.0, 31.0, second)]


def test_words_and_sentence_segments_come_from_token_times_and_piece_offsets():
    result = build_result(_two_pieces(), duration=31.0)

    assert result["text"] == "He has grave doubts. Truly? Yes"
    assert [(w["word"], w["start"], w["end"]) for w in result["words"]] == [
        ("He", 0.32, 0.64), ("has", 0.64, 0.88), ("grave", 0.88, 1.6), ("doubts.", 1.6, 2.4),
        ("Truly?", 2.4, 3.2), ("Yes", 30.16, 30.56),
    ]
    assert [(s["id"], s["start"], s["end"], s["text"]) for s in result["segments"]] == [
        (0, 0.32, 2.4, " He has grave doubts."), (1, 2.4, 3.2, " Truly?"), (2, 30.16, 30.56, " Yes"),
    ]
    segment = result["segments"][0]
    assert (result["task"], result["language"], result["duration"]) == ("transcribe", "en", 31.0)
    assert segment["avg_logprob"] == pytest.approx(-0.1)
    assert set(segment) >= {"seek", "tokens", "temperature", "compression_ratio", "no_speech_prob"}


def test_a_piece_without_token_times_is_one_segment_over_the_piece():
    result = build_result([(20.0, 40.0, _result(" text only"))], duration=40.0)
    assert [(s["start"], s["end"], s["text"]) for s in result["segments"]] == [(20.0, 40.0, " text only")]
    assert result["words"] == []


def test_srt_and_vtt_come_from_the_segments(fake_onnx_asr, fake_hub, cache, monkeypatch):
    from src.formatters import format_transcription

    backend = _backend()
    monkeypatch.setattr(ParakeetBackend, "_recognize", staticmethod(lambda model, waveforms: [_two_pieces()[0][2]]))

    srt = backend.transcribe(_wav(3.5), MODEL, response_format="srt")["text"]
    vtt = backend.transcribe(_wav(3.5), MODEL, response_format="vtt")["text"]
    verbose = backend.transcribe(_wav(3.5), MODEL, response_format="verbose_json")

    assert srt.startswith("1\n00:00:00,320 --> 00:00:02,400\nHe has grave doubts.\n")
    assert "2\n00:00:02,400 --> 00:00:03,200\nTruly?" in srt
    assert vtt.startswith("WEBVTT\n") and "00:00:00.320 --> 00:00:02.400\nHe has grave doubts." in vtt
    # The routes render subtitles from verbose_json with the shared formatter: the same cues.
    assert format_transcription(verbose, "srt")[0].strip() == srt.strip()
    assert backend.transcribe(_wav(3.5), MODEL, response_format="text") == {
        "text": "He has grave doubts. Truly?", "raw_text": True,
    }


# --- Streaming ----------------------------------------------------------------------------------------------------


def test_a_parakeet_stream_finalizes_an_utterance_at_25_seconds():
    from src import streaming

    def session(model):
        return streaming.StreamingSession(
            ws=None, model=model, language=None, sample_rate=16000, interim_results=True, endpointing_ms=300,
        )

    assert session(MODEL).max_utterance_bytes == 25 * SAMPLE_RATE * 2
    assert session("Systran/faster-whisper-base").max_utterance_bytes == streaming.MAX_UTTERANCE_BYTES
    assert parakeet_module.STREAM_MAX_UTTERANCE_S * SAMPLE_RATE <= parakeet_module.SINGLE_PIECE_MAX_S * SAMPLE_RATE


@pytest.mark.asyncio
async def test_a_parakeet_stream_finalizes_long_speech_in_pieces_of_25_seconds(monkeypatch):
    from src import streaming

    decoded = []

    class Router:
        def is_model_loaded(self, model):
            return True

        def transcribe(self, audio, model, **kwargs):
            decoded.append(len(audio) - 44)
            return {"text": "words"}

    class AlwaysSpeech:
        def __call__(self, samples):
            return 1.0

    sent = []

    class WS:
        async def send_text(self, text):
            sent.append(text)

    monkeypatch.setattr(streaming, "backend_router", Router())
    session = streaming.StreamingSession(
        ws=WS(), model=MODEL, language=None, sample_rate=16000, interim_results=False, endpointing_ms=300,
    )
    session.vad_state = AlwaysSpeech()
    for _ in range(60):  # 60 s of speech, 1 s at a time
        await session._process_16k(b"\x10\x00" * SAMPLE_RATE, transcribe=False)

    finals = [len(s) for s in sent if '"speech_final": true' in s]
    assert len(finals) == 2
    # Checked after each 512-sample VAD window: at most one window past 25 s.
    assert max(decoded) <= (25 * SAMPLE_RATE + parakeet_module.VAD_WINDOW) * 2


# --- fp16 converter -----------------------------------------------------------------------------------------------


def _tiny_encoder(folder: Path) -> None:
    """An 'encoder' in the export's shape: a /layers.* MatMul, a Softmax inside the layers, and a node outside the
    layers, with its weights in an external .data file."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    # Large enough for the converter's external data file (tensors over 1 KB).
    w1 = numpy_helper.from_array((rng.standard_normal((64, 64)) / 8).astype(np.float32), "w1")
    w2 = numpy_helper.from_array((rng.standard_normal((64, 64)) / 8).astype(np.float32), "w2")
    nodes = [
        helper.make_node("MatMul", ["audio_signal", "w1"], ["pre"], name="/pre_encode/MatMul"),
        helper.make_node("MatMul", ["pre", "w2"], ["h"], name="/layers.0/MatMul"),
        helper.make_node("Softmax", ["h"], ["s"], axis=-1, name="/layers.0/Softmax"),
        helper.make_node("Relu", ["s"], ["outputs"], name="/layers.0/Relu"),
        helper.make_node("Identity", ["length"], ["encoded_lengths"], name="/Identity"),
    ]
    graph = helper.make_graph(
        nodes, "encoder",
        [helper.make_tensor_value_info("audio_signal", TensorProto.FLOAT, ["b", 64]),
         helper.make_tensor_value_info("length", TensorProto.INT64, ["b"])],
        [helper.make_tensor_value_info("outputs", TensorProto.FLOAT, ["b", 64]),
         helper.make_tensor_value_info("encoded_lengths", TensorProto.INT64, ["b"])],
        [w1, w2],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    folder.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(folder / parakeet_fp16.ENCODER), save_as_external_data=True,
                    all_tensors_to_one_file=True, location=parakeet_fp16.ENCODER + ".data", size_threshold=0)
    for name in (parakeet_fp16.DECODER, *parakeet_fp16.COPIED_FILES):
        (folder / name).write_bytes(b"decoder" if name == parakeet_fp16.DECODER else b"x")


def test_the_converter_keeps_fp32_islands_and_gives_the_source_output(tmp_path):
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxconverter_common")
    import onnxruntime as ort
    from onnx import TensorProto

    source = tmp_path / "snapshot"
    _tiny_encoder(source)

    folder = parakeet_fp16.ensure_fp16(source, tmp_path / "repo", "rev1")

    assert folder == tmp_path / "repo" / parakeet_fp16.FP16_DIR_NAME / "rev1-c1"
    assert sorted(p.name for p in folder.iterdir()) == sorted(parakeet_fp16.FP16_FILES)
    assert (folder / parakeet_fp16.DECODER_FP16).read_bytes() == b"decoder"
    model = onnx.load(str(folder / parakeet_fp16.ENCODER_FP16))
    types = {init.name: init.data_type for init in model.graph.initializer}
    assert types["w2"] == TensorProto.FLOAT16  # inside the conformer layers
    assert types["w1"] == TensorProto.FLOAT  # outside them: kept fp32
    assert [o.type.tensor_type.elem_type for o in model.graph.output] == [TensorProto.FLOAT, TensorProto.INT64]

    feed = {"audio_signal": np.random.default_rng(1).standard_normal((2, 64)).astype(np.float32),
            "length": np.array([8, 8], dtype=np.int64)}
    want = ort.InferenceSession(str(source / parakeet_fp16.ENCODER), providers=["CPUExecutionProvider"]).run(None, feed)
    got = ort.InferenceSession(str(folder / parakeet_fp16.ENCODER_FP16), providers=["CPUExecutionProvider"]).run(None, feed)
    assert got[0].dtype == np.float32
    np.testing.assert_allclose(got[0], want[0], atol=2e-2)
    np.testing.assert_array_equal(got[1], want[1])


def test_the_converter_reads_a_hugging_face_snapshot_whose_files_link_into_blobs(tmp_path):
    """The hub cache keeps snapshot files as symbolic links into ../../blobs; onnx refuses external data whose real
    path leaves the model's folder, so the encoder is staged as plain files first."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxconverter_common")
    built = tmp_path / "built"
    _tiny_encoder(built)
    repo = tmp_path / "models--x--y"
    blobs, snapshot = repo / "blobs", repo / "snapshots" / "rev1"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    try:
        for index, path in enumerate(sorted(built.iterdir())):
            blob = blobs / f"blob{index}"
            path.replace(blob)
            (snapshot / path.name).symlink_to(Path("..", "..", "blobs", blob.name))
    except OSError:
        pytest.skip("this host can't create symbolic links without a privilege")

    folder = parakeet_fp16.ensure_fp16(snapshot, repo, "rev1")

    assert sorted(p.name for p in folder.iterdir()) == sorted(parakeet_fp16.FP16_FILES)
    assert not [p for p in folder.parent.iterdir() if p.name.startswith(parakeet_fp16.TEMP_PREFIX)]


def test_the_encoder_is_converted_from_plain_files_in_a_staging_folder_that_is_removed(tmp_path):
    source, repo, seen = _source(tmp_path), tmp_path / "repo", []

    def convert(staged: Path, target: Path) -> None:
        seen.append(staged)
        assert staged.parent != source
        for name in (parakeet_fp16.ENCODER, parakeet_fp16.ENCODER + ".data"):
            assert (staged.parent / name).is_file() and not (staged.parent / name).is_symlink()
        _counting_convert([])(staged, target)

    folder = parakeet_fp16.ensure_fp16(source, repo, "rev1", convert=convert)

    assert seen and seen[0].name == parakeet_fp16.ENCODER
    assert not seen[0].parent.exists()
    assert (source / parakeet_fp16.ENCODER).read_bytes() == b"x"  # the source is untouched
    assert parakeet_fp16.is_complete(folder)


def _counting_convert(calls: list, fail: bool = False):
    def convert(source: Path, target: Path) -> None:
        calls.append(target)
        target.write_bytes(b"graph")
        Path(str(target) + ".data").write_bytes(b"weights")
        if fail:
            raise MemoryError("killed half way")
    return convert


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "snapshot"
    source.mkdir(exist_ok=True)
    for name in parakeet_fp16.SOURCE_FILES:
        (source / name).write_bytes(b"x")
    return source


def test_the_converted_cache_is_reused_and_rebuilt_for_a_new_revision(tmp_path):
    source, repo, calls = _source(tmp_path), tmp_path / "repo", []

    first = parakeet_fp16.ensure_fp16(source, repo, "rev1", convert=_counting_convert(calls))
    again = parakeet_fp16.ensure_fp16(source, repo, "rev1", convert=_counting_convert(calls))
    assert first == again and len(calls) == 1
    assert parakeet_fp16.cached_fp16_dir(repo, "rev1") == first

    newer = parakeet_fp16.ensure_fp16(source, repo, "rev2", convert=_counting_convert(calls))
    assert len(calls) == 2 and newer.name == "rev2-c1"
    assert sorted(p.name for p in (repo / parakeet_fp16.FP16_DIR_NAME).iterdir()) == ["rev2-c1"]


def test_a_new_converter_version_rebuilds_the_cache(tmp_path, monkeypatch):
    source, repo, calls = _source(tmp_path), tmp_path / "repo", []
    parakeet_fp16.ensure_fp16(source, repo, "rev1", convert=_counting_convert(calls))
    monkeypatch.setattr(parakeet_fp16, "CONVERTER_VERSION", 2)

    folder = parakeet_fp16.ensure_fp16(source, repo, "rev1", convert=_counting_convert(calls))

    assert folder.name == "rev1-c2" and len(calls) == 2


def test_a_failed_conversion_leaves_nothing_a_later_load_trusts(tmp_path):
    source, repo, calls = _source(tmp_path), tmp_path / "repo", []

    with pytest.raises(MemoryError):
        parakeet_fp16.ensure_fp16(source, repo, "rev1", convert=_counting_convert(calls, fail=True))

    assert list((repo / parakeet_fp16.FP16_DIR_NAME).iterdir()) == []
    assert parakeet_fp16.cached_fp16_dir(repo, "rev1") is None
    parakeet_fp16.ensure_fp16(source, repo, "rev1", convert=_counting_convert(calls))
    assert parakeet_fp16.cached_fp16_dir(repo, "rev1") is not None


def test_a_half_written_final_folder_is_not_trusted_and_is_replaced(tmp_path):
    source, repo, calls = _source(tmp_path), tmp_path / "repo", []
    partial = repo / parakeet_fp16.FP16_DIR_NAME / "rev1-c1"
    partial.mkdir(parents=True)
    (partial / parakeet_fp16.ENCODER_FP16).write_bytes(b"half")

    assert parakeet_fp16.cached_fp16_dir(repo, "rev1") is None
    folder = parakeet_fp16.ensure_fp16(source, repo, "rev1", convert=_counting_convert(calls))

    assert folder == partial and len(calls) == 1 and parakeet_fp16.is_complete(folder)


def test_missing_source_files_are_named(tmp_path):
    source = _source(tmp_path)
    (source / "encoder-model.onnx.data").unlink()
    with pytest.raises(FileNotFoundError, match="encoder-model.onnx.data"):
        parakeet_fp16.ensure_fp16(source, tmp_path / "repo", "rev1", convert=_counting_convert([]))


# --- Settings, packaging and documentation ------------------------------------------------------------------------


def test_the_gpu_memory_setting_defaults_to_2304_mb_and_is_documented():
    from src.config import Settings

    assert Settings.model_fields["stt_parakeet_gpu_mem_limit_mb"].default == 2304
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert re.search(r"^\| `STT_PARAKEET_GPU_MEM_LIMIT_MB` \| `2304` \|", readme, re.MULTILINE)
    assert "\n# STT_PARAKEET_GPU_MEM_LIMIT_MB=2304\n" in example


def test_both_images_and_compose_default_to_parakeet():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert dockerfile.count("STT_MODEL=nvidia/parakeet-tdt-0.6b-v2") == 2
    for name in ("docker-compose.yml", "docker-compose.cpu.yml"):
        assert "STT_DEFAULT_MODEL:-nvidia/parakeet-tdt-0.6b-v2}}" in (ROOT / name).read_text(encoding="utf-8")
    assert type(settings).model_fields["stt_model"].default == "Systran/faster-whisper-base"


def test_the_cuda_image_installs_onnxruntime_gpu_in_place_of_onnxruntime():
    import tomllib

    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert '["--no-install-package", "onnxruntime"]' in dockerfile
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert "onnxruntime==1.24.4" in project["dependencies"] and "onnx-asr==0.12.0" in project["dependencies"]
    assert "onnxruntime-gpu==1.24.4" in project["optional-dependencies"]["cuda"]
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    for package, version in (("onnxruntime", "1.24.4"), ("onnxruntime-gpu", "1.24.4"), ("onnx-asr", "0.12.0")):
        assert f'name = "{package}"\nversion = "{version}"' in lock
    assert "onnx-asr==0.12.0" in (ROOT / "requirements.lock").read_text(encoding="utf-8")


def test_the_readme_lists_parakeet_in_the_stt_models_table():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert re.search(r"^\| `nvidia/parakeet-tdt-0\.6b-v2` \|.*\| Parakeet \(ONNX\) \| English \|", readme, re.MULTILINE)


def test_the_vad_scores_every_window_with_the_shared_silero_session(monkeypatch):
    from src.vad import silero

    class Session:
        def run(self, outputs, feeds):
            assert feeds["input"].shape == (1, silero.VAD_CONTEXT_SIZE + parakeet_module.VAD_WINDOW)
            return np.array([[0.75]], dtype=np.float32), feeds["state"]

    monkeypatch.setattr(silero, "_vad_model", silero.SileroVAD(Session()))

    probs = parakeet_module.score_speech(np.zeros(10 * parakeet_module.VAD_WINDOW + 100, dtype=np.float32))

    assert probs == [0.75] * 10
    assert silero.load_vad_model_blocking() is silero._vad_model


def test_a_title_or_an_initial_does_not_end_a_segment():
    tokens = [" Mr", ".", " J", ".", " Quilter", " is", " here", ".", " Yes", "."]
    piece = _result(" Mr. J. Quilter is here. Yes.", tokens, [0.1 * i for i in range(len(tokens))], [-0.1] * len(tokens))

    result = build_result([(0.0, 2.0, piece)], duration=2.0)

    assert [s["text"] for s in result["segments"]] == [" Mr. J. Quilter is here.", " Yes."]


# --- Residency: a Whisper model loads next to the pinned Parakeet -------------------------------------------------


@pytest.mark.parametrize(("keep_pinned", "left"), [
    (True, {MODEL, "Systran/faster-whisper-small"}),
    (False, {"Systran/faster-whisper-small"}),
])
def test_an_stt_load_that_keeps_the_pinned_default_unloads_only_the_others(monkeypatch, keep_pinned, left):
    from src.model_manager import ModelManager
    from tests.test_model_manager import FakeSTTRouter, FakeTTSRouter

    monkeypatch.setattr(settings, "stt_model", MODEL)
    monkeypatch.setattr(settings, "os_pin_default_models", True)
    stt = FakeSTTRouter()
    stt._backends["parakeet"] = stt._default_backend
    for model_id in (MODEL, "deepdml/faster-whisper-large-v3-turbo-ct2"):
        stt.load_model(model_id)
    manager = ModelManager(stt_router=stt, tts_router=FakeTTSRouter())

    manager.load("Systran/faster-whisper-small", **({"keep_pinned": True} if keep_pinned else {}))

    assert {m.model for m in stt.loaded_models()} == left


def test_the_load_route_passes_keep_pinned(monkeypatch):
    from src.services.models import ModelProgressService

    calls = []

    async def load(self, *, model_id, model_manager, keep_pinned=False):
        calls.append((model_id, keep_pinned))
        return {"id": model_id}

    monkeypatch.setattr(ModelProgressService, "load", load)
    client = TestClient(app)

    assert client.post("/api/models/Systran%2Ffaster-whisper-small/load?keep_pinned=true").status_code == 200
    assert client.post("/api/models/Systran%2Ffaster-whisper-small/load").status_code == 200
    assert calls == [("Systran/faster-whisper-small", True), ("Systran/faster-whisper-small", False)]
