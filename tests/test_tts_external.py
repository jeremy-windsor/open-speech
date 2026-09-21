"""Contract tests for isolated external TTS providers."""

from __future__ import annotations

import http.client
import io
import json
import urllib.error
import urllib.request

import numpy as np
import pytest

from src.services.tts import validate_tts_feature_support
from src.tts.external import (
    ExternalProviderError,
    ExternalTTSBackend,
    parse_external_provider_urls,
)


MANIFEST = {
    "schema_version": 1,
    "provider": "qwen3",
    "models": [
        {
            "id": "qwen3/0.6b-custom-voice",
            "provider": "qwen3",
            "sample_rate": 24000,
            "revision": "test-1",
            "max_input_chars": 8,
            "capabilities": {
                "voice_clone": False,
                "instructions": False,
                "speed_control": False,
            },
            "voices": [
                {"id": "Ryan", "name": "Ryan", "language": "en", "gender": "male"}
            ],
        },
        {
            "id": "qwen3/0.6b-base",
            "provider": "qwen3",
            "sample_rate": 24000,
            "revision": "test-1",
            "capabilities": {
                "voice_clone": True,
                "clone_transcript_required": True,
                "instructions": False,
                "speed_control": False,
            },
            "voices": [],
        },
    ],
}


class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes, headers: dict[str, str] | None = None):
        super().__init__(data)
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _urlopen(request, **_kwargs):
    if request.full_url.endswith("/v1/manifest"):
        return FakeResponse(json.dumps(MANIFEST).encode())
    if request.full_url.endswith("/health"):
        return FakeResponse(json.dumps({"status": "ok", "provider": "qwen3"}).encode())
    if request.full_url.endswith("/v1/audio/speech"):
        payload = json.loads(request.data)
        assert payload["model"] == "qwen3/0.6b-custom-voice"
        return FakeResponse(
            np.array([0.25, -0.25], dtype="<f4").tobytes(),
            {"X-Audio-Sample-Rate": "24000", "X-Audio-Format": "float32le"},
        )
    return FakeResponse(b"{}")


def _backend(monkeypatch, open_fn=_urlopen):
    backend = ExternalTTSBackend(provider="qwen3", base_url="http://qwen3:8200")
    monkeypatch.setattr(backend._opener, "open", open_fn)
    return backend


def test_external_model_load_uses_cold_start_timeout(monkeypatch):
    backend = _backend(monkeypatch)
    calls = []

    def tracked_open(request, **kwargs):
        calls.append((request.full_url, kwargs.get("timeout")))
        return _urlopen(request, **kwargs)

    monkeypatch.setattr(backend._opener, "open", tracked_open)
    backend.load_model("qwen3/0.6b-custom-voice")

    assert calls[-1][0].endswith("/v1/models/load")
    assert calls[-1][1] == 1800.0


def test_external_provider_config_is_explicit_and_rejects_credentials():
    assert parse_external_provider_urls('{"qwen3":"http://qwen3:8200"}') == {
        "qwen3": "http://qwen3:8200"
    }
    with pytest.raises(ValueError, match="cannot contain credentials"):
        parse_external_provider_urls('{"qwen3":"http://user:secret@qwen3:8200"}')


def test_model_specific_manifest_controls_voices_and_speed(monkeypatch):
    backend = _backend(monkeypatch)

    assert backend.get_capabilities("qwen3/0.6b-custom-voice")["voice_clone"] is False
    assert backend.get_capabilities("qwen3/0.6b-base")["voice_clone"] is True
    assert [voice.id for voice in backend.list_voices("qwen3/0.6b-custom-voice")] == [
        "Ryan"
    ]
    assert backend.list_voices("qwen3/0.6b-base") == []


def test_external_audio_requires_manifest_sample_rate(monkeypatch):
    def wrong_rate_urlopen(request, **kwargs):
        response = _urlopen(request, **kwargs)
        if request.full_url.endswith("/v1/audio/speech"):
            response.headers["X-Audio-Sample-Rate"] = "22050"
        return response

    backend = _backend(monkeypatch, wrong_rate_urlopen)

    with pytest.raises(ExternalProviderError, match="22050 Hz"):
        list(
            backend.synthesize(
                "Hello", "Ryan", model_id="qwen3/0.6b-custom-voice"
            )
        )


def test_external_audio_yields_float32(monkeypatch):
    backend = _backend(monkeypatch)

    chunks = list(
        backend.synthesize("Hello", "Ryan", model_id="qwen3/0.6b-custom-voice")
    )

    np.testing.assert_allclose(chunks[0], [0.25, -0.25])
    assert chunks[0].dtype == np.float32


def test_external_adapter_enforces_model_input_limit_before_worker_post(monkeypatch):
    calls = []

    def tracked_open(request, **kwargs):
        calls.append(request.full_url)
        return _urlopen(request, **kwargs)

    backend = _backend(monkeypatch, tracked_open)
    with pytest.raises(ValueError, match="Max: 8 characters"):
        list(backend.synthesize("nine chars", "Ryan", model_id="qwen3/0.6b-custom-voice"))

    assert not any(url.endswith("/v1/audio/speech") for url in calls)


def test_catalog_membership_uses_exact_manifest_and_handles_worker_outage(monkeypatch):
    backend = _backend(monkeypatch)
    assert backend.advertises_model("qwen3/0.6b-custom-voice") is True
    assert backend.advertises_model("qwen3/not-offered") is False

    backend._opener.open = lambda *_args, **_kwargs: (_ for _ in ()).throw(urllib.error.URLError("offline"))
    backend._manifest_checked_at = 0
    assert backend.advertises_model("qwen3/0.6b-custom-voice") is True

    cold_backend = _backend(monkeypatch, backend._opener.open)
    assert cold_backend.advertises_model("qwen3/0.6b-custom-voice") is None


def test_aborted_external_audio_stream_is_typed(monkeypatch):
    class AbortedResponse(FakeResponse):
        def read1(self, _size=-1):
            raise http.client.IncompleteRead(b"partial")

    def aborted_stream(request, **kwargs):
        if request.full_url.endswith("/v1/audio/speech"):
            return AbortedResponse(
                b"",
                {"X-Audio-Sample-Rate": "24000", "X-Audio-Format": "float32le"},
            )
        return _urlopen(request, **kwargs)

    backend = _backend(monkeypatch, aborted_stream)

    with pytest.raises(ExternalProviderError) as caught:
        list(backend.synthesize("Hello", "Ryan", model_id="qwen3/0.6b-custom-voice"))
    assert caught.value.code == "provider_stream_aborted"


def test_worker_down_is_typed_provider_unavailable(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise urllib.error.URLError("offline")

    backend = _backend(monkeypatch, unavailable)

    assert backend.is_provider_available() is False
    with pytest.raises(ExternalProviderError) as caught:
        backend.get_capabilities("qwen3/0.6b-custom-voice")
    assert caught.value.code == "provider_unavailable"
    assert caught.value.status_code == 503


def test_manifest_schema_version_is_rejected(monkeypatch):
    bad_manifest = {**MANIFEST, "schema_version": 99}
    backend = _backend(
        monkeypatch,
        lambda *_args, **_kwargs: FakeResponse(json.dumps(bad_manifest).encode()),
    )

    with pytest.raises(ExternalProviderError) as caught:
        backend.get_capabilities("qwen3/0.6b-custom-voice")
    assert caught.value.code == "manifest_version_unsupported"


def test_feature_validation_uses_selected_external_model(monkeypatch):
    backend = _backend(monkeypatch)

    class Router:
        def get_backend(self, _model):
            return backend

        def get_capabilities(self, model):
            return backend.get_capabilities(model)

        def validate_voice(self, model, voice):
            backend.validate_voice(voice, model)

    router = Router()
    assert "Speed control is not supported" in validate_tts_feature_support(
        tts_router=router,
        model_id="qwen3/0.6b-custom-voice",
        voice="Ryan",
        speed=1.2,
    )
    assert "clone_transcript is required" in validate_tts_feature_support(
        tts_router=router,
        model_id="qwen3/0.6b-base",
        voice="logical-voice",
        reference_audio=b"wav",
    )


def test_external_provider_bypasses_proxies_and_rejects_redirects(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:9999")
    backend = ExternalTTSBackend(provider="qwen3", base_url="http://qwen3:8200")

    assert not any(
        isinstance(handler, urllib.request.ProxyHandler)
        for handler in backend._opener.handlers
    )
    assert any(
        handler.__class__.__name__ == "_RejectRedirects"
        for handler in backend._opener.handlers
    )


def test_live_reader_capability_is_enforced(monkeypatch):
    backend = _backend(monkeypatch)

    class Router:
        def get_backend(self, _model):
            return backend

        def get_capabilities(self, model):
            capabilities = backend.get_capabilities(model)
            capabilities["live_reader"] = False
            return capabilities

    assert "Live Reader is not supported" in validate_tts_feature_support(
        tts_router=Router(),
        model_id="qwen3/0.6b-base",
        live_reader=True,
    )
