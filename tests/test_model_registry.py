"""Tests for the curated model registry."""

from __future__ import annotations

import importlib
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import model_registry
from src.model_registry import (
    KNOWN_MODELS,
    WHISPER_LANGUAGES,
    get_known_model,
    get_known_models,
    stt_model_languages,
)


class TestKnownModels:
    def test_registry_not_empty(self):
        assert len(KNOWN_MODELS) > 0

    def test_all_have_required_fields(self):
        for m in KNOWN_MODELS:
            assert "id" in m
            assert "type" in m
            assert m["type"] in ("stt", "tts")
            assert "provider" in m
            assert "size_mb" in m
            assert "description" in m
            assert isinstance(m["size_mb"], int)

    def test_has_stt_models(self):
        stt = [m for m in KNOWN_MODELS if m["type"] == "stt"]
        assert len(stt) >= 5

    def test_has_tts_models(self):
        tts = [m for m in KNOWN_MODELS if m["type"] == "tts"]
        assert len(tts) >= 5

    def test_has_piper_models(self):
        piper = [m for m in KNOWN_MODELS if m["provider"] == "piper"]
        assert len(piper) >= 5

    def test_has_kokoro(self):
        kokoro = [m for m in KNOWN_MODELS if m["id"] == "kokoro"]
        assert len(kokoro) == 1

    def test_has_pocket_tts(self):
        pocket = [m for m in KNOWN_MODELS if m["id"] == "pocket-tts"]
        assert len(pocket) == 1
        assert pocket[0]["provider"] == "pocket-tts"

    def test_has_optional_voice_model_workers(self):
        expected = {
            "chatterbox": {"chatterbox/regular", "chatterbox/turbo"},
            "fish": {"fish/s2-pro"},
        }
        for provider, model_ids in expected.items():
            models = [m for m in KNOWN_MODELS if m["provider"] == provider]
            assert {model["id"] for model in models} == model_ids
            assert all(model["optional_provider"] is True for model in models)

    def test_unique_ids(self):
        ids = [m["id"] for m in KNOWN_MODELS]
        assert len(ids) == len(set(ids))


class TestGetKnownModels:
    def test_returns_copy(self):
        models = get_known_models()
        models[0]["id"] = "mutated"
        assert KNOWN_MODELS[0]["id"] != "mutated"


class TestGetKnownModel:
    def test_found(self):
        m = get_known_model("kokoro")
        assert m is not None
        assert m["type"] == "tts"

    def test_not_found(self):
        assert get_known_model("nonexistent") is None

    def test_piper_model(self):
        m = get_known_model("piper/en_US-lessac-medium")
        assert m is not None
        assert m["provider"] == "piper"
        assert m["size_mb"] == 60  # en_US-lessac-medium.onnx is 63 201 294 bytes


# The registry gave every Piper voice 6 (low), 35 (medium) or 75 MB (high), which
# /api/models and the Models view show as its size on disk. Piper's "low" is a 16 kHz model as large as "medium".
# The .onnx sizes in bytes, rhasspy/piper-voices snapshot c10ece1a, read with `stat -L` in the core image's Hugging
# Face cache; each .onnx.json adds about 5 KB. size_mb is MiB, as the measured
# Whisper rows are.
PIPER_ONNX_BYTES = {
    "en_GB-alan-low": 63104526,
    "en_GB-alan-medium": 63201294,
    "en_GB-cori-high": 114219352,
    "en_GB-cori-medium": 63531379,
    "en_GB-jenny_dioco-medium": 63201294,
    "en_GB-northern_english_male-medium": 63201294,
    "en_GB-semaine-medium": 76737711,
    "en_GB-southern_english_female-low": 63104526,
    "en_US-amy-medium": 63201294,
    "en_US-arctic-medium": 76766385,
    "en_US-bryce-medium": 63531379,
    "en_US-danny-low": 63104526,
    "en_US-hfc_female-medium": 63201294,
    "en_US-hfc_male-medium": 63201294,
    "en_US-joe-medium": 63201294,
    "en_US-john-medium": 63531379,
    "en_US-kathleen-low": 63104526,
    "en_US-kusal-medium": 63201294,
    "en_US-lessac-high": 113895201,
    "en_US-lessac-low": 63201294,
    "en_US-lessac-medium": 63201294,
    "en_US-libritts_r-medium": 78580914,
    "en_US-ljspeech-high": 114199011,
    "en_US-ljspeech-medium": 63531379,
    "en_US-norman-medium": 63531379,
    "en_US-ryan-high": 120786792,
    "en_US-ryan-low": 63104526,
    "en_US-ryan-medium": 63201294,
}


def test_every_piper_voice_size_is_its_files_size():
    from src.tts.backends.piper_backend import PIPER_MODELS

    piper = {m["id"].removeprefix("piper/"): m["size_mb"] for m in KNOWN_MODELS if m["provider"] == "piper"}
    assert sorted(piper) == sorted(PIPER_ONNX_BYTES) == sorted(m.removeprefix("piper/") for m in PIPER_MODELS)
    wrong = {
        voice: (size_mb, round(PIPER_ONNX_BYTES[voice] / 1024 / 1024, 1))
        for voice, size_mb in piper.items()
        if abs(size_mb - PIPER_ONNX_BYTES[voice] / 1024 / 1024) > 1
    }
    assert wrong == {}, f"registry size_mb against the files' MiB: {wrong}"


ROOT = Path(__file__).resolve().parents[1]
GPU_WORKERS = ("chatterbox", "qwen3", "fish", "breeze")


def _worker_manifest_models(monkeypatch) -> list[dict]:
    """Each model the GPU workers' manifests (providers/*/app.py) advertise, Qwen3 Base included as the voice-models
    Compose file turns it on. torch is stubbed out, as in tests/test_env_compat.py's _worker_models."""
    cuda = SimpleNamespace(OutOfMemoryError=type("FakeOutOfMemoryError", (Exception,), {}),
                           is_available=lambda: False, empty_cache=lambda: None)
    torch_stub = SimpleNamespace(cuda=cuda, inference_mode=lambda: nullcontext(),
                                 float16=object(), float32=object(), bfloat16=object())
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setenv("QWEN3_ENABLE_BASE", "true")
    models = []
    for provider in GPU_WORKERS:
        name = f"providers.{provider}.app"
        monkeypatch.delitem(sys.modules, name, raising=False)
        models += importlib.import_module(name).MANIFEST["models"]
    return models


class TestWorkerModelsAreRegistered:
    """Without a registry row, /api/models lists a worker's model only once it is loaded,
    so the Models view, Speak and Voices › Test clone don't offer it (the Fish worker shipped without one)."""

    def test_fish_s2_pro_is_a_registered_cloning_model_of_the_fish_worker(self, monkeypatch):
        row = get_known_model("fish/s2-pro")
        assert row is not None, "src/model_registry.py has no row for fish/s2-pro"
        assert row["type"] == "tts"
        assert row["provider"] == "fish"
        assert row["model_format"] == "external"
        assert row["optional_provider"] is True
        assert "voice cloning" in row["description"]
        # providers/fish/app.py's model.pth + codec.pth + tokenizer files, about 5.0 GB (4,768 MiB).
        assert 4700 <= row["size_mb"] <= 4800
        manifest = next(model for model in _worker_manifest_models(monkeypatch) if model["id"] == "fish/s2-pro")
        assert manifest["provider"] == row["provider"]
        assert manifest["capabilities"]["voice_clone"] is True
        assert manifest["voices"] == []
        assert manifest["sample_rate"] == 44100

    def test_every_worker_model_has_a_row_with_its_provider(self, monkeypatch):
        wrong = []
        for model in _worker_manifest_models(monkeypatch):
            row = get_known_model(model["id"])
            if row is None:
                wrong.append(f"{model['id']}: no row")
                continue
            if (row["type"], row["provider"], row.get("model_format"), row.get("optional_provider")) != (
                    "tts", model["provider"], "external", True):
                wrong.append(f"{model['id']}: {row}")
            cloning = model["capabilities"].get("voice_clone") and not model["voices"]
            if cloning and "voice cloning" not in row["description"]:
                wrong.append(f"{model['id']}: the description doesn't say voice cloning")
        assert wrong == [], "src/model_registry.py against the workers' manifests: " + "; ".join(wrong)

    def test_a_missing_row_fails_the_check(self, monkeypatch):
        monkeypatch.setattr(model_registry, "KNOWN_MODELS", [m for m in KNOWN_MODELS if m["id"] != "fish/s2-pro"])
        with pytest.raises(AssertionError, match="no row for fish/s2-pro"):
            self.test_fish_s2_pro_is_a_registered_cloning_model_of_the_fish_worker(monkeypatch)
        with pytest.raises(AssertionError, match="fish/s2-pro: no row"):
            self.test_every_worker_model_has_a_row_with_its_provider(monkeypatch)

    def test_the_docs_count_the_registry(self):
        stt = sum(m["type"] == "stt" for m in KNOWN_MODELS)
        tts = sum(m["type"] == "tts" for m in KNOWN_MODELS)
        models = " ".join((ROOT / "docs" / "MODELS.md").read_text(encoding="utf-8").split())
        backends = " ".join((ROOT / "docs" / "TTS-BACKENDS.md").read_text(encoding="utf-8").split())
        assert re.search(r"contains all (\d+) STT and (\d+) TTS entries", models).groups() == (str(stt), str(tts))
        assert re.search(r"registry contains (\d+) STT and (\d+) TTS IDs", backends).groups() == (str(stt), str(tts))


class TestSttModelLanguages:
    def test_multilingual_whisper_models_list_whisper_languages(self):
        # README: the faster-whisper models transcribe 99+ languages.
        for model_id in ("deepdml/faster-whisper-large-v3-turbo-ct2", "Systran/faster-whisper-small", "someone/custom-ct2"):
            languages = stt_model_languages(model_id)
            assert languages == list(WHISPER_LANGUAGES)
            assert {"en", "de", "fr", "es", "ja", "zh"} <= set(languages)
        assert len(WHISPER_LANGUAGES) == len(set(WHISPER_LANGUAGES)) >= 99

    def test_english_only_models_list_english(self):
        assert stt_model_languages("Systran/faster-whisper-base.en") == ["en"]
        assert stt_model_languages("Systran/faster-distil-whisper-small.en") == ["en"]
        assert stt_model_languages("Systran/faster-distil-whisper-large-v3") == ["en"]

    def test_matches_faster_whisper_language_codes(self):
        tokenizer = pytest.importorskip("faster_whisper.tokenizer")
        assert WHISPER_LANGUAGES == tuple(tokenizer._LANGUAGE_CODES)
