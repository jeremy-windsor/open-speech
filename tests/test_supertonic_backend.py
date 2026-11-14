"""Supertonic 3 backend, with fake ONNX sessions and a fake supertonic package (no downloads)."""

from __future__ import annotations

import json
import sys
import types
from unittest.mock import patch

import numpy as np
import pytest

from src.tts.backends import supertonic as st
from src.tts.backends.supertonic import SupertonicBackend, blend_styles

TEST_SENTENCE = (
    "Dr. Okafor's flight, BA 2049, leaves Heathrow at 6:45 p.m. on March 3rd, 2027; the fare was $1,284.50."
)


class FakeStyle:
    def __init__(self, ttl, dp):
        self.ttl, self.dp = ttl, dp


class FakeProcessor:
    def __init__(self, path):
        self.path = path

    def validate_text(self, text):
        bad = sorted({c for c in text if c in "☃"})
        return not bad, bad


class FakeEngine:
    sample_rate = 44100

    def __init__(self, config, processor, *sessions):
        self.config, self.processor, self.sessions = config, processor, sessions
        self.calls = []

    def __call__(self, texts, style, total_step, speed, lang):
        self.calls.append({"text": texts[0], "style": style, "steps": total_step, "speed": speed, "lang": lang})
        # Padding past the predicted duration, which the backend trims.
        return np.full((1, 6000), 0.5, dtype=np.float32), np.array([0.1], dtype=np.float32)


class FakeSession:
    def __init__(self, path, sess_options=None, providers=None):
        self.path, self.providers = path, providers
        names = [p[0] if isinstance(p, tuple) else p for p in providers]
        self._active = names if FakeSession.cuda_works else [n for n in names if n != "CUDAExecutionProvider"]

    cuda_works = True

    def get_providers(self):
        return self._active


@pytest.fixture
def fake_package(monkeypatch):
    core = types.ModuleType("supertonic.core")
    core.Style, core.Supertonic, core.UnicodeProcessor = FakeStyle, FakeEngine, FakeProcessor
    package = types.ModuleType("supertonic")
    package.core = core
    monkeypatch.setitem(sys.modules, "supertonic", package)
    monkeypatch.setitem(sys.modules, "supertonic.core", core)
    import onnxruntime

    monkeypatch.setattr(onnxruntime, "InferenceSession", FakeSession)
    FakeSession.cuda_works = True
    return core


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    (tmp_path / "onnx").mkdir()
    (tmp_path / "voice_styles").mkdir()
    (tmp_path / "onnx" / "tts.json").write_text(json.dumps({"ae": {"sample_rate": 44100}}), encoding="utf-8")
    (tmp_path / "onnx" / "unicode_indexer.json").write_text("[0]", encoding="utf-8")
    for i, voice in enumerate(st.SUPERTONIC_VOICES):
        style = {
            "style_ttl": {"dims": [1, 2, 3], "data": np.full((1, 2, 3), i, dtype=float).tolist()},
            "style_dp": {"dims": [1, 2], "data": [[i, -i]]},
        }
        (tmp_path / "voice_styles" / f"{voice}.json").write_text(json.dumps(style), encoding="utf-8")
    monkeypatch.setattr(SupertonicBackend, "_files", staticmethod(lambda: tmp_path))
    return tmp_path


def loaded(device="cpu"):
    backend = SupertonicBackend(device=device)
    backend.load_model("supertonic")
    return backend


def test_capabilities_declare_blends_streaming_speed_and_the_steps_option():
    caps = SupertonicBackend.capabilities
    assert caps["voice_blend"] and caps["streaming"] and caps["speed_control"]
    assert not caps["voice_clone"] and not caps["instructions"]
    assert caps["options"]["steps"] == {
        "type": "integer", "label": "Quality steps", "min": 1, "max": 32, "default": 8,
        "description": "Fewer is faster, more is cleaner.",
    }
    assert "en" in caps["languages"] and len(caps["languages"]) == 31
    assert [s["name"] for s in caps["speakers"]] == list(st.SUPERTONIC_VOICES)


def test_is_available_needs_the_supertonic_package():
    with patch.dict(sys.modules, {"supertonic": None}):
        assert SupertonicBackend.is_available() is False


def test_load_without_the_package_says_how_to_install_it(model_dir):
    with (
        patch.dict(sys.modules, {"supertonic": None, "supertonic.core": None}),
        pytest.raises(RuntimeError, match="BAKED_PROVIDERS"),
    ):
        SupertonicBackend(device="cpu").load_model("supertonic")


def test_load_on_the_cpu_and_unload(fake_package, model_dir):
    backend = loaded("cpu")
    engine = backend._loaded["engine"]
    assert [s.path.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for s in engine.sessions] == [
        "duration_predictor.onnx", "text_encoder.onnx", "vector_estimator.onnx", "vocoder.onnx",
    ]
    assert all(s.providers == ["CPUExecutionProvider"] for s in engine.sessions)
    assert engine.calls == [], "no warm-up on the CPU"
    [info] = backend.loaded_models()
    assert (info.model, info.backend, info.device) == ("supertonic", "supertonic", "cpu")
    assert backend.is_model_loaded("supertonic")
    assert backend.sample_rate == 44100
    backend.unload_model("supertonic")
    assert backend.loaded_models() == [] and not backend.is_model_loaded("supertonic")


def test_loads_on_the_gpu_with_cuda_and_warms_up(fake_package, model_dir, monkeypatch):
    import onnxruntime

    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    backend = SupertonicBackend(device="cuda:1")
    assert backend._get_device() == "cuda", "the memory guard asks before the load"
    backend.load_model("supertonic")
    engine = backend._loaded["engine"]
    cuda, cpu = engine.sessions[0].providers
    assert cuda[0] == "CUDAExecutionProvider" and cuda[1]["device_id"] == 1 and cpu == "CPUExecutionProvider"
    assert cuda[1]["cudnn_conv_algo_search"] == "HEURISTIC"
    assert backend.loaded_models()[0].device == "cuda"
    assert len(engine.calls) == 1 and engine.calls[0]["steps"] == 2, "one short warm-up run"


def test_a_gpu_that_cannot_start_falls_back_to_the_cpu(fake_package, model_dir, monkeypatch):
    import onnxruntime

    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    FakeSession.cuda_works = False
    backend = loaded("cuda")
    assert backend.loaded_models()[0].device == "cpu"
    assert backend._get_device() == "cpu"
    assert all(s.providers == ["CPUExecutionProvider"] for s in backend._loaded["engine"].sessions)


def test_device_without_cuda_or_with_tts_device_cpu_is_the_cpu(monkeypatch):
    import onnxruntime

    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: ["CPUExecutionProvider"])
    assert SupertonicBackend(device="cuda")._get_device() == "cpu"
    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert SupertonicBackend(device="cpu")._get_device() == "cpu"
    assert SupertonicBackend(device="auto")._get_device() == "cuda"


def test_download_fetches_only_the_needed_files_at_the_pinned_revision():
    with patch("huggingface_hub.snapshot_download", return_value="/cache/snap") as download:
        SupertonicBackend(device="cpu").download_model("supertonic")
    download.assert_called_once_with(
        repo_id="Supertone/supertonic-3", revision=st.SUPERTONIC_REVISION,
        allow_patterns=[
            "onnx/tts.json", "onnx/unicode_indexer.json", "onnx/duration_predictor.onnx", "onnx/text_encoder.onnx",
            "onnx/vector_estimator.onnx", "onnx/vocoder.onnx", "voice_styles/*.json",
        ],
    )
    with pytest.raises(ValueError, match="Unknown Supertonic model"):
        SupertonicBackend(device="cpu").download_model("supertonic/other")


def test_synthesis_reads_sentence_by_sentence_with_pauses_trimmed_to_each_duration(fake_package, model_dir):
    backend = loaded()
    chunks = list(backend.synthesize("First sentence here. Second one now! And a third?", "M1"))
    engine = backend._loaded["engine"]
    assert [c["text"] for c in engine.calls] == ["First sentence here.", "Second one now!", "And a third?"]
    assert all((c["steps"], c["speed"], c["lang"]) == (8, 1.05, "en") for c in engine.calls)
    assert [len(c) for c in chunks] == [4410, int(0.3 * 44100), 4410, int(0.3 * 44100), 4410]
    assert all(c.dtype == np.float32 for c in chunks)
    assert not chunks[1].any(), "the pause is silence"


def test_long_sentences_are_cut_to_the_models_chunk_length(fake_package, model_dir):
    backend = loaded()
    words = " ".join(["word"] * 200) + "."
    list(backend.synthesize(words, "F1"))
    texts = [c["text"] for c in backend._loaded["engine"].calls]
    assert len(texts) > 1 and all(len(t) <= st.MAX_SENTENCE_CHARS for t in texts)


def test_english_text_is_normalized_before_the_model_reads_it(fake_package, model_dir):
    backend = loaded()
    list(backend.synthesize(TEST_SENTENCE, "M1"))
    spoken = " ".join(c["text"] for c in backend._loaded["engine"].calls)
    assert spoken == (
        "Doctor Okafor's flight, B A twenty forty-nine, leaves Heathrow at six forty-five P M on March third, "
        "twenty twenty-seven; the fare was one thousand two hundred eighty-four dollars and fifty cents."
    )
    assert not any(ch.isdigit() for ch in spoken) and "$" not in spoken


def test_another_language_is_passed_on_and_not_normalized_as_english(fake_package, model_dir):
    backend = loaded()
    list(backend.synthesize("Es kostet 5 Euro.", "F3", lang_code="de-DE"))
    [call] = backend._loaded["engine"].calls
    assert (call["lang"], call["text"]) == ("de", "Es kostet 5 Euro.")
    with pytest.raises(ValueError, match="Unsupported language"):
        list(backend.synthesize("Hi.", "M1", lang_code="xx"))
    backend.validate_language("en-GB")
    with pytest.raises(ValueError):
        backend.validate_language("zh")


def test_steps_and_speed(fake_package, model_dir):
    backend = loaded()
    list(backend.synthesize("Hello there.", "M2", speed=1.5, steps=4))
    list(backend.synthesize("Hello there.", "M2", speed=0.25))
    fast, slow = backend._loaded["engine"].calls
    assert (fast["steps"], fast["speed"]) == (4, pytest.approx(1.575))
    assert (slow["steps"], slow["speed"]) == (8, 0.7), "held to the model's range"
    for bad in (0, 33, 2.5, True):
        with pytest.raises(ValueError, match="steps"):
            list(backend.synthesize("Hello.", "M1", steps=bad))


def test_voices_list_and_validation():
    backend = SupertonicBackend(device="cpu")
    voices = backend.list_voices()
    assert [(v.id, v.name, v.gender) for v in voices[:1] + voices[5:6]] == [("M1", "M1", "male"), ("F1", "F1", "female")]
    assert len(voices) == 10
    backend.validate_voice("m3")
    backend.validate_voice("M1(0.6)+F2(0.4)")
    with pytest.raises(ValueError, match="Unknown Supertonic voice: AF_HEART"):
        backend.validate_voice("af_heart")
    with pytest.raises(ValueError, match="Unknown Supertonic voice"):
        backend.validate_voice("M1+M9")
    assert SupertonicBackend.voice_weights("alloy") == [("M1", 1.0)], "an OpenAI voice name means the default"
    assert SupertonicBackend.voice_weights("") == [("M1", 1.0)]


def test_blending_averages_the_style_vectors_by_weight(fake_package, model_dir):
    styles = {"A": (np.ones((1, 2)), np.zeros((1, 1))), "B": (np.full((1, 2), 3.0), np.ones((1, 1)))}
    ttl, dp = blend_styles(styles, [("A", 0.25), ("B", 0.75)])
    assert ttl.tolist() == [[2.5, 2.5]] and dp.tolist() == [[0.75]] and ttl.dtype == np.float32

    backend = loaded()
    assert SupertonicBackend.voice_weights("M1(0.6)+F2(0.4)") == [("M1", 0.6), ("F2", 0.4)]
    list(backend.synthesize("Blend.", "M1(3)+F2(1)"))
    style = backend._loaded["engine"].calls[0]["style"]
    m1, f2 = backend._loaded["styles"]["M1"], backend._loaded["styles"]["F2"]
    np.testing.assert_allclose(style.ttl, 0.75 * m1[0] + 0.25 * f2[0])
    np.testing.assert_allclose(style.dp, 0.75 * m1[1] + 0.25 * f2[1])


def test_characters_the_model_cannot_read_are_dropped_and_empty_sentences_skipped(fake_package, model_dir):
    backend = loaded()
    chunks = list(backend.synthesize("Snow ☃ falls. ... ☃.", "M1"))
    assert [c["text"] for c in backend._loaded["engine"].calls] == ["Snow falls."]
    assert len(chunks) == 1


def test_the_router_discovers_it_and_passes_steps_only_to_it(fake_package, model_dir):
    from src.tts.router import TTSRouter

    router = TTSRouter(device="cpu", external_providers="")
    assert "supertonic" in router.list_backends()
    assert router.get_backend("supertonic").name == "supertonic"
    assert router.sample_rate_for("supertonic") == 44100
    chunks = list(router.synthesize(text="Hi there.", model="supertonic", voice="F1", steps=3))
    assert chunks and router.get_backend("supertonic")._loaded["engine"].calls[-1]["steps"] == 3


def test_the_images_bake_supertonic_from_its_extra():
    import re
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r'ARG BAKED_PROVIDERS="([^"]*)"', dockerfile).group(1) == "kokoro,piper,pocket-tts,supertonic"
    assert json.loads(re.search(r"extras = (\{[^}]*\})", dockerfile).group(1))["supertonic"] == "supertonic"
    compose = (root / "docker-compose.yml").read_text(encoding="utf-8")
    assert 'BAKED_PROVIDERS: "kokoro,piper,pocket-tts,supertonic"' in compose
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["optional-dependencies"]["supertonic"] == ["supertonic==1.3.1"]
    assert "supertonic" in project["optional-dependencies"]["all"][0]
