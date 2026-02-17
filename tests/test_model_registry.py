"""Tests for the curated model registry."""

from __future__ import annotations

import pytest

from src.model_registry import KNOWN_MODELS, get_known_models, get_known_model


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
        assert m["size_mb"] == 35
