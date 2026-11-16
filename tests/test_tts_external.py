"""Contract tests for isolated external TTS providers."""

from __future__ import annotations

import http.client
import io
import json
import socket
import time
import urllib.error
import urllib.request
from unittest.mock import patch

import numpy as np
import pytest

from src.services.tts import validate_tts_feature_support
from src.tts import external as external_module
from src.tts.external import (
    ExternalProviderError,
    ExternalTTSBackend,
    parse_external_provider_urls,
)
from src.tts.router import TTSRouter

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


def _health_open(health, *, speech_error=None):
    def open_fn(request, **kwargs):
        if request.full_url.endswith("/health"):
            return FakeResponse(json.dumps(health).encode())
        if speech_error is not None and request.full_url.endswith("/v1/audio/speech"):
            raise speech_error
        return _urlopen(request, **kwargs)

    return open_fn


LOADED_HEALTH = {
    "status": "ok",
    "provider": "qwen3",
    "loaded_model": "qwen3/0.6b-custom-voice",
    "loaded_at": 1000.0,
    "last_used_at": None,
}


def test_running_and_failed_requests_count_as_model_use(monkeypatch):
    backend = _backend(monkeypatch, _health_open(LOADED_HEALTH))
    started = time.time()

    chunks = backend.synthesize("Hello", "Ryan", model_id="qwen3/0.6b-custom-voice")
    next(chunks)
    assert backend.loaded_models()[0].last_used_at >= started
    list(chunks)
    assert backend.loaded_models()[0].last_used_at >= started

    failing = _backend(
        monkeypatch,
        _health_open(LOADED_HEALTH, speech_error=urllib.error.URLError("worker went away")),
    )
    with pytest.raises(ExternalProviderError):
        list(failing.synthesize("Hello", "Ryan", model_id="qwen3/0.6b-custom-voice"))
    assert failing.loaded_models()[0].last_used_at >= started


def test_busy_worker_counts_as_model_use(monkeypatch):
    backend = _backend(monkeypatch, _health_open({**LOADED_HEALTH, "busy": True}))
    started = time.time()

    assert backend.loaded_models()[0].last_used_at >= started


def test_idle_worker_reports_its_own_last_use(monkeypatch):
    backend = _backend(monkeypatch, _health_open({**LOADED_HEALTH, "last_used_at": 1500.0}))

    assert backend.loaded_models()[0].last_used_at == 1500.0


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


def _timing_out_worker(clock):
    """A worker that accepted the connection and never answered: each read runs out its timeout."""
    calls = []

    def open_fn(request, *, timeout, **_kwargs):
        calls.append(request.full_url)
        clock.now += timeout
        raise urllib.error.URLError(TimeoutError("timed out"))

    return open_fn, calls


def test_a_timed_out_manifest_read_is_served_from_the_failure_cache(monkeypatch):
    # A failed manifest read is kept for 2 s so an outage costs one timeout per window, not one
    # per read: GET /api/models alone reads each worker model's manifest several times.
    clock = _FakeClock()
    monkeypatch.setattr(external_module, "time", clock)
    open_fn, calls = _timing_out_worker(clock)
    backend = _backend(monkeypatch, open_fn)

    for _ in range(3):
        with pytest.raises(ExternalProviderError) as caught:
            backend.get_capabilities("qwen3/0.6b-custom-voice")
        assert caught.value.code == "provider_unavailable"
    assert calls == ["http://qwen3:8200/v1/manifest"]
    assert clock.now == 103.0  # one 3 s timeout in all

    # A failure that took 3 s is remembered for four times as long: once that has
    # passed, the worker is asked again.
    clock.now += 12.0
    with pytest.raises(ExternalProviderError):
        backend.get_capabilities("qwen3/0.6b-custom-voice")
    assert len(calls) == 2


def test_a_timed_out_health_read_is_served_from_the_failure_cache(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(external_module, "time", clock)
    open_fn, calls = _timing_out_worker(clock)
    backend = _backend(monkeypatch, open_fn)

    assert [backend.is_provider_available() for _ in range(3)] == [False, False, False]
    with pytest.raises(ExternalProviderError):
        backend.loaded_models()
    assert calls == ["http://qwen3:8200/health"]
    assert clock.now == 103.0

    clock.now += 12.0  # four times the 3 s the failed read took
    assert backend.is_provider_available() is False
    assert len(calls) == 2


def test_a_health_identity_mismatch_is_cached_like_other_health_failures(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(external_module, "time", clock)
    calls = []

    def other_provider(request, **_kwargs):
        calls.append(request.full_url)
        return FakeResponse(json.dumps({"status": "ok", "provider": "chatterbox"}).encode())

    backend = _backend(monkeypatch, other_provider)

    assert [backend.is_provider_available() for _ in range(3)] == [False, False, False]
    with pytest.raises(ExternalProviderError) as caught:
        backend.loaded_models()
    assert caught.value.code == "provider_identity_mismatch"
    assert calls == ["http://qwen3:8200/health"]


# One GET /api/models reads each worker's health and manifest once per
# registry model and once per TTS row. A read that failed slowly (a hung worker's 3 s timeout, a slow name
# lookup for a worker that isn't running) outlived its 1 s / 2 s failure window before the next read, so
# the listing paid the failure on almost every read: 18 s for one hung worker, past the web app's 15 s
# timeout. CHANGELOG: an unreachable worker costs one timeout per window, not one per read.

class _NoSTT:
    _backends = {"faster-whisper": object()}

    def loaded_models(self):
        return []

    def list_cached_models(self):
        return []


def _one_listing(monkeypatch, providers: dict[str, str], cost: float | None):
    """One GET /api/models listing on a fake clock, every worker failing after `cost` s (None: hung)."""
    from src.model_manager import ModelManager
    from src.services import models as model_service
    from src.services import tts as tts_service

    clock = _FakeClock()
    monkeypatch.setattr(external_module, "time", clock)
    monkeypatch.setattr("src.tts.router._discover_backends", lambda *_args, **_kwargs: {})
    router = TTSRouter(device="cpu", external_providers=json.dumps(providers))
    calls: dict[str, list[str]] = {}

    def failing_worker(name):
        calls[name] = []

        def open_fn(request, *, timeout, **_kwargs):
            calls[name].append(request.full_url)
            if cost is None:  # accepted the connection and never answered
                clock.now += timeout
                raise TimeoutError("timed out")
            clock.now += cost  # the name lookup for a worker with no container
            raise urllib.error.URLError(socket.gaierror(socket.EAI_NONAME, "Name or service not known"))

        return open_fn

    for name in providers:
        monkeypatch.setattr(router._backends[name]._opener, "open", failing_worker(name))
    manager = ModelManager(stt_router=_NoSTT(), tts_router=router)
    start = clock.now
    result = model_service.list_all_models(
        model_manager=manager,
        tts_capabilities_for=lambda model_id: tts_service.tts_capabilities(tts_router=router, model_id=model_id),
        default_stt_model="stt-default",
        default_tts_model="kokoro",
        tts_enabled=True,
    )
    rows = {row["id"]: row["state"] for row in result["models"] if row["provider"] in providers}
    assert rows and set(rows.values()) == {"provider_unavailable"}, rows
    return calls, clock.now - start


def test_one_listing_reads_a_hung_worker_once_per_window_not_once_per_model(monkeypatch):
    calls, elapsed = _one_listing(monkeypatch, {"qwen3": "http://qwen3:8200"}, None)

    assert len(calls["qwen3"]) <= 2, calls
    assert elapsed <= 6.0, f"the listing took {elapsed:.1f} s ({len(calls['qwen3'])} reads of 3 s)"


@pytest.mark.parametrize("lookup_s", [0.4, 1.5, 4.0])
def test_one_listing_reads_each_absent_worker_once_per_window_however_slowly_it_fails(monkeypatch, lookup_s):
    providers = {"qwen3": "http://qwen3:8200", "chatterbox": "http://chatterbox:8210", "fish": "http://fish:8230"}
    calls, elapsed = _one_listing(monkeypatch, providers, lookup_s)

    assert {name: len(reads) <= 2 for name, reads in calls.items()} == dict.fromkeys(providers, True), calls
    assert elapsed <= 2 * len(providers) * lookup_s


def test_a_worker_read_that_failed_slowly_answers_both_reads_until_it_is_asked_again(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(external_module, "time", clock)
    open_fn, calls = _timing_out_worker(clock)
    backend = _backend(monkeypatch, open_fn)

    assert backend.is_provider_available() is False
    # The worker didn't answer its health read, so the manifest read in the same window isn't sent.
    with pytest.raises(ExternalProviderError) as caught:
        backend.get_capabilities("qwen3/0.6b-custom-voice")
    assert caught.value.code == "provider_unavailable"
    assert backend.advertises_model("qwen3/0.6b-custom-voice") is None
    assert calls == ["http://qwen3:8200/health"]

    clock.now += 11.9
    assert backend.is_provider_available() is False
    assert calls == ["http://qwen3:8200/health"]

    # Four times the 3 s read has passed, and the worker is back.
    clock.now += 0.2
    monkeypatch.setattr(backend._opener, "open", _urlopen)
    assert backend.is_provider_available() is True
    assert backend.advertises_model("qwen3/0.6b-custom-voice") is True


@pytest.mark.parametrize("answer", ["health", "load"])
def test_a_read_that_began_failing_before_the_worker_answered_does_not_hold_it_offline(monkeypatch, answer):
    # Listing A's health read is still in its slow name lookup when the
    # worker comes up; another listing's health read (or a Load) gets an answer meanwhile. A's
    # failure, returning last, was remembered for four times its 7.5 s: the worker read as offline and Load
    # answered 503 for 30 s, though it had just answered. A failure older than the last answer is not kept.
    clock = _FakeClock()
    monkeypatch.setattr(external_module, "time", clock)
    calls = []
    backend = _backend(monkeypatch)
    other_reads = []

    def open_fn(request, **kwargs):
        calls.append(request.full_url.rsplit("8200", 1)[1])
        if len(calls) == 1:  # listing A's /health: the name lookup of a worker still starting
            clock.now += 7.0
            # The worker is up; while A's lookup runs on, another thread gets its answers.
            if answer == "health":
                other_reads.append(backend.is_provider_available())
            else:
                backend.load_model("qwen3/0.6b-custom-voice")
                other_reads.append("loaded")
            clock.now += 0.5
            raise urllib.error.URLError(socket.gaierror(socket.EAI_NONAME, "Name or service not known"))
        return _urlopen(request, **kwargs)

    monkeypatch.setattr(backend._opener, "open", open_fn)

    assert backend.is_provider_available() is False  # listing A's own read failed
    assert other_reads == ([True] if answer == "health" else ["loaded"])

    clock.now += 1.2  # past the 1 s health failure window, well inside the memo's 30 s
    assert backend.is_provider_available() is True
    assert calls[-1] == "/health", calls
    assert backend.advertises_model("qwen3/0.6b-custom-voice") is True


def test_a_worker_that_answers_with_an_error_is_not_taken_for_unreachable(monkeypatch):
    calls = []

    def health_fails(request, **kwargs):
        calls.append(request.full_url)
        if request.full_url.endswith("/health"):
            raise urllib.error.HTTPError(request.full_url, 500, "Internal Server Error", {}, io.BytesIO(b"{}"))
        return _urlopen(request, **kwargs)

    backend = _backend(monkeypatch, health_fails)

    assert backend.is_provider_available() is False
    assert backend.advertises_model("qwen3/0.6b-custom-voice") is True
    assert calls == ["http://qwen3:8200/health", "http://qwen3:8200/v1/manifest"]


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


def test_worker_request_validation_is_a_client_error_naming_the_field(monkeypatch):
    # FastAPI's 422 body for a speed outside the worker's own range.
    body = {
        "detail": [
            {
                "type": "less_than_equal",
                "loc": ["body", "speed"],
                "msg": "Input should be less than or equal to 2",
                "input": 3.0,
            }
        ]
    }
    rejected = urllib.error.HTTPError(
        "http://qwen3:8200/v1/audio/speech",
        422,
        "Unprocessable Entity",
        {},
        io.BytesIO(json.dumps(body).encode()),
    )
    backend = _backend(monkeypatch, _health_open(LOADED_HEALTH, speech_error=rejected))

    with pytest.raises(ExternalProviderError) as caught:
        list(backend.synthesize("Hello", "Ryan", model_id="qwen3/0.6b-custom-voice"))
    assert caught.value.status_code == 400
    assert caught.value.code == "invalid_request"
    assert "speed: Input should be less than or equal to 2" in caught.value.message


def test_manifest_with_non_object_voice_entries_is_invalid_manifest(monkeypatch):
    drifted = {
        **MANIFEST,
        "models": [{**MANIFEST["models"][0], "voices": ["Ryan", "Aiden"]}],
    }
    backend = _backend(
        monkeypatch,
        lambda *_args, **_kwargs: FakeResponse(json.dumps(drifted).encode()),
    )

    with pytest.raises(ExternalProviderError) as caught:
        backend.get_capabilities("qwen3/0.6b-custom-voice")
    assert caught.value.code == "invalid_manifest"
    assert backend.advertises_model("qwen3/0.6b-custom-voice") is None


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


@pytest.mark.parametrize("value", ["2026-10-21T08:00:00Z", "inf", True])
@pytest.mark.parametrize("field", ["loaded_at", "last_used_at"])
def test_health_timestamp_drift_is_typed_and_skipped_by_the_router(monkeypatch, field, value):
    # docs/TTS-BACKENDS.md: "The core rejects identity, schema, ... mismatches". Like manifest
    # drift, a /health timestamp that is not epoch seconds is a typed provider error, which the
    # router treats as one unavailable worker instead of failing every model listing.
    drifted = {**LOADED_HEALTH, field: value}
    with patch("src.tts.router._discover_backends", return_value={}):
        router = TTSRouter(device="cpu", external_providers='{"qwen3": "http://qwen3:8200"}')
    backend = router._backends["qwen3"]
    monkeypatch.setattr(backend._opener, "open", _health_open(drifted))

    with pytest.raises(ExternalProviderError) as caught:
        backend.loaded_models()
    assert caught.value.code == "invalid_provider_response"
    assert field in caught.value.message
    assert router.loaded_models() == []


def test_health_timestamps_sent_as_numeric_strings_are_still_read(monkeypatch):
    # Epoch seconds sent as a numeric string were always read as a number; such a worker must
    # stay listed, or its model is never idle-evicted.
    health = {**LOADED_HEALTH, "loaded_at": "1000.5", "last_used_at": " 1200 "}
    backend = _backend(monkeypatch, _health_open(health))

    [info] = backend.loaded_models()
    assert (info.loaded_at, info.last_used_at) == (1000.5, 1200.0)


@pytest.mark.parametrize(
    "url", ["http://qwen3:abc", "http://qwen3:70000", "http://qwen3:0", "http://qwen3:+80"]
)
def test_external_provider_url_with_an_invalid_port_is_rejected(url):
    # RFC 3986: port = *DIGIT, and a TCP destination port is 1-65535. urllib would wrap 70000 to
    # another port at connect time, sending text and reference audio off the configured origin.
    with pytest.raises(ValueError, match="invalid port"):
        parse_external_provider_urls(json.dumps({"qwen3": url}))


def test_external_provider_url_port_is_optional():
    assert parse_external_provider_urls(
        '{"qwen3": "http://qwen3:8200", "chatterbox": "https://chatterbox"}'
    ) == {"qwen3": "http://qwen3:8200", "chatterbox": "https://chatterbox"}


class _FakeClock:
    """Stands in for the time module so busy-retry waits are counted, not slept."""

    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _busy_error(url: str, headers: dict[str, str] | None = None) -> urllib.error.HTTPError:
    body = {"detail": {"code": "worker_busy", "message": "Worker is busy"}}
    return urllib.error.HTTPError(
        url, 429, "Too Many Requests", headers or {}, io.BytesIO(json.dumps(body).encode())
    )


def _busy_then_ready(busy_answers: int, headers: dict[str, str] | None = None):
    calls = []

    def open_fn(request, **kwargs):
        if request.full_url.endswith(("/v1/audio/speech", "/v1/models/load")):
            calls.append(request.full_url)
            if len(calls) <= busy_answers:
                raise _busy_error(request.full_url, headers)
        return _urlopen(request, **kwargs)

    return open_fn, calls


def test_worker_busy_after_a_cancel_is_retried_before_failing(monkeypatch):
    # After a cancelled stream the worker answers 429 worker_busy until its current chunk ends;
    # the next Live Reader segment must not fail because it arrived inside that window.
    clock = _FakeClock()
    monkeypatch.setattr(external_module, "time", clock)
    open_fn, calls = _busy_then_ready(2)
    backend = _backend(monkeypatch, open_fn)

    chunks = list(backend.synthesize("Hello", "Ryan", model_id="qwen3/0.6b-custom-voice"))

    np.testing.assert_allclose(chunks[0], [0.25, -0.25])
    assert len(calls) == 3
    assert clock.sleeps == [0.1, 0.2]


def test_model_load_retries_worker_busy_too(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(external_module, "time", clock)
    open_fn, calls = _busy_then_ready(1)
    backend = _backend(monkeypatch, open_fn)

    backend.load_model("qwen3/0.6b-custom-voice")

    assert len(calls) == 2


def test_worker_busy_retries_stop_within_the_bounded_wait(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(external_module, "time", clock)
    open_fn, calls = _busy_then_ready(1000)
    backend = _backend(monkeypatch, open_fn)

    with pytest.raises(ExternalProviderError) as caught:
        list(backend.synthesize("Hello", "Ryan", model_id="qwen3/0.6b-custom-voice"))

    assert (caught.value.status_code, caught.value.code) == (429, "worker_busy")
    assert 1 < len(calls) < 10
    assert sum(clock.sleeps) <= external_module.WORKER_BUSY_RETRY_S


def test_worker_busy_retry_honours_retry_after(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(external_module, "time", clock)
    open_fn, calls = _busy_then_ready(1, {"Retry-After": "0.75"})
    backend = _backend(monkeypatch, open_fn)

    list(backend.synthesize("Hello", "Ryan", model_id="qwen3/0.6b-custom-voice"))

    assert clock.sleeps == [0.75]


def test_worker_busy_with_a_retry_after_beyond_the_bound_fails_at_once(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(external_module, "time", clock)
    open_fn, calls = _busy_then_ready(1, {"Retry-After": "30"})
    backend = _backend(monkeypatch, open_fn)

    with pytest.raises(ExternalProviderError) as caught:
        list(backend.synthesize("Hello", "Ryan", model_id="qwen3/0.6b-custom-voice"))

    assert caught.value.code == "worker_busy"
    assert (len(calls), clock.sleeps) == (1, [])


def test_other_worker_errors_are_not_retried(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(external_module, "time", clock)
    calls = []

    def open_fn(request, **kwargs):
        if request.full_url.endswith("/v1/audio/speech"):
            calls.append(request.full_url)
            body = {"detail": {"code": "unknown_model", "message": "Unknown model"}}
            raise urllib.error.HTTPError(
                request.full_url, 400, "Bad Request", {}, io.BytesIO(json.dumps(body).encode())
            )
        return _urlopen(request, **kwargs)

    backend = _backend(monkeypatch, open_fn)

    with pytest.raises(ExternalProviderError) as caught:
        list(backend.synthesize("Hello", "Ryan", model_id="qwen3/0.6b-custom-voice"))

    assert (caught.value.status_code, caught.value.code) == (400, "unknown_model")
    assert (len(calls), clock.sleeps) == (1, [])


@pytest.mark.parametrize(
    ("status", "body"),
    [
        (507, {"error": {"code": "out_of_memory", "message": "CUDA out of memory"}}),
        (503, {"detail": {"code": "cuda_out_of_memory", "message": "CUDA out of memory"}}),
    ],
    ids=["507_error_envelope", "503_detail_envelope"],
)
@pytest.mark.parametrize("operation", ["load", "synthesis"])
def test_worker_out_of_memory_is_a_typed_insufficient_gpu_memory_error(
    monkeypatch, status, body, operation
):
    def open_fn(request, **kwargs):
        if request.full_url.endswith(("/v1/audio/speech", "/v1/models/load")):
            raise urllib.error.HTTPError(
                request.full_url, status, "Out of memory", {}, io.BytesIO(json.dumps(body).encode())
            )
        return _urlopen(request, **kwargs)

    backend = _backend(monkeypatch, open_fn)

    with pytest.raises(ExternalProviderError) as caught:
        if operation == "load":
            backend.load_model("qwen3/0.6b-custom-voice")
        else:
            list(backend.synthesize("Hello", "Ryan", model_id="qwen3/0.6b-custom-voice"))

    assert (caught.value.status_code, caught.value.code) == (503, "insufficient_gpu_memory")
    assert "GPU memory" in caught.value.message
    assert "qwen3/0.6b-custom-voice" in caught.value.message


# Worker answer to /v1/audio/speech -> what the client of Open Speech gets (docs/TTS-BACKENDS.md).
_WORKER_ERRORS = {
    "507_out_of_memory": (
        507, {"error": {"code": "out_of_memory", "message": "CUDA out of memory"}},
        503, "insufficient_gpu_memory",
    ),
    "422_validation": (
        422, {"detail": [{"loc": ["body", "speed"], "msg": "out of range", "type": "value_error"}]},
        400, "invalid_request",
    ),
    "409_model_not_loaded": (
        409, {"detail": {"code": "model_not_loaded", "message": "Load qwen3/0.6b-base first"}},
        409, "model_not_loaded",
    ),
    "429_worker_busy": (
        429, {"detail": {"code": "worker_busy", "message": "Worker is busy"}},
        429, "worker_busy",
    ),
    "503_provider_unavailable": (
        503, {"detail": {"code": "provider_unavailable", "message": "Worker is starting"}},
        503, "provider_unavailable",
    ),
}


@pytest.fixture
def worker_app(monkeypatch):
    """The app with a real router and ExternalTTSBackend; only the worker's HTTP is faked."""
    from fastapi.testclient import TestClient

    from src import main as main_module

    with patch("src.tts.router._discover_backends", return_value={}):
        router = TTSRouter(device="cpu", external_providers='{"qwen3": "http://qwen3:8200"}')
    backend = router._backends["qwen3"]
    backend._busy_retry_s = 0.0  # worker_busy fails at once instead of being retried

    def answer_speech_with(status, body):
        def open_fn(request, **kwargs):
            if request.full_url.endswith("/v1/audio/speech"):
                raise urllib.error.HTTPError(
                    request.full_url, status, "worker error", {}, io.BytesIO(json.dumps(body).encode())
                )
            return _urlopen(request, **kwargs)

        monkeypatch.setattr(backend._opener, "open", open_fn)

    monkeypatch.setattr(main_module, "tts_router", router)
    monkeypatch.setattr(main_module.settings, "tts_cache_enabled", False)
    yield TestClient(main_module.app), answer_speech_with  # no lifespan: nothing preloads


@pytest.mark.parametrize("case", list(_WORKER_ERRORS))
@pytest.mark.parametrize("route", ["speech", "speech_stream", "clone"])
def test_speech_routes_keep_the_worker_error_code(worker_app, route, case):
    client, answer_speech_with = worker_app
    worker_status, worker_body, status, code = _WORKER_ERRORS[case]
    answer_speech_with(worker_status, worker_body)

    if route == "clone":
        resp = client.post(
            "/v1/audio/speech/clone",
            data={
                "input": "Hello", "model": "qwen3/0.6b-base", "voice": "default",
                "transcript": "Reference words", "response_format": "pcm",
            },
            files={"reference_audio": ("ref.wav", b"RIFF-reference", "audio/wav")},
        )
    else:
        resp = client.post(
            "/v1/audio/speech" + ("?stream=true" if route == "speech_stream" else ""),
            json={"model": "qwen3/0.6b-custom-voice", "input": "Hello", "voice": "Ryan", "response_format": "pcm"},
        )

    assert resp.status_code == status, resp.text
    assert resp.json()["error"]["code"] == code, resp.text


def test_synthesis_forwards_only_the_options_the_model_declares(monkeypatch):
    manifest = json.loads(json.dumps(MANIFEST))
    manifest["models"][0]["capabilities"]["options"] = {
        "temperature": {"type": "number", "min": 0.1, "max": 1.3, "default": 0.9},
        "model": {"type": "number"},  # a payload field is never an option
    }
    sent = []

    def open_fn(request, **kwargs):
        if request.full_url.endswith("/v1/manifest"):
            return FakeResponse(json.dumps(manifest).encode())
        if request.full_url.endswith("/v1/audio/speech"):
            sent.append(json.loads(request.data))
        return _urlopen(request, **kwargs)

    backend = _backend(monkeypatch, open_fn)
    list(backend.synthesize("Hello", "Ryan", model_id="qwen3/0.6b-custom-voice", temperature=0.5, steps=4,
                            cfg_scale=2.0, model="other"))
    list(backend.synthesize("Hello", "Ryan", model_id="qwen3/0.6b-custom-voice", temperature=None))
    assert sent[0]["temperature"] == 0.5 and sent[0]["model"] == "qwen3/0.6b-custom-voice"
    assert "steps" not in sent[0] and "cfg_scale" not in sent[0]
    assert "temperature" not in sent[1], "an option left unset is not sent"
