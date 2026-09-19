"""Tests for the deployable TTS harness conformance report."""

from __future__ import annotations

import io
import json
import wave
from types import SimpleNamespace

from scripts import tts_conformance
from scripts.tts_conformance import inspect_audio, inspect_generation_policy, inspect_model, validate_manifest


def _json_bytes(value):
    return json.dumps(value).encode("utf-8")


class FakeClient:
    def __init__(self, *, clone=False, audio_status=200, capabilities_status=200, input_limit=None):
        self.clone = clone
        self.audio_status = audio_status
        self.capabilities_status = capabilities_status
        self.input_limit = input_limit
        self.calls = []

    def request(self, path, payload=None):
        self.calls.append((path, payload))
        if path.startswith("/api/tts/capabilities"):
            if self.capabilities_status != 200:
                return self.capabilities_status, _json_bytes({"detail": "unavailable"})
            return 200, _json_bytes({
                "backend": "example", "model": "example/model", "sample_rate": 24000,
                "capabilities": {"speed_control": False, "instructions": False, "voice_clone": self.clone},
            })
        if path.startswith("/v1/audio/voices"):
            voices = [] if self.clone else [{"id": "speaker", "name": "Speaker"}]
            return 200, _json_bytes({"voices": voices})
        if self.input_limit is not None and len(payload["input"]) > self.input_limit:
            return 400, _json_bytes({"detail": "Input too long"})
        if payload.get("speed", 1) != 1 or payload.get("instructions") or payload["voice"] == "__invalid_voice__":
            return 400, _json_bytes({"detail": "Unsupported control"})
        if self.audio_status != 200:
            return self.audio_status, _json_bytes({"detail": {"code": "generation_limit_reached"}})
        return 200, _wav()


def _wav():
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(b"\x01\x00" * 2400)
    return output.getvalue()


def _statuses(result):
    return {check["name"]: check["status"] for check in result["checks"]}


def test_manifest_rejects_wrong_version_identity_and_duplicate_voices():
    models, errors = validate_manifest({
        "schema_version": 99,
        "provider": "other",
        "models": [{
            "id": "example/model", "provider": "other", "sample_rate": True,
            "capabilities": {}, "voices": [{"id": "speaker"}, {"id": "speaker"}],
        }],
    }, "example")

    assert "example/model" in models
    assert "unsupported schema_version" in errors
    assert "provider identity mismatch" in errors
    assert "example/model: invalid sample_rate" in errors
    assert "example/model: duplicate voice" in errors


def test_metadata_rejection_and_audio_checks_have_evidence():
    client = FakeClient()
    result = inspect_model(client, "example/model", probe_rejections=True, synthesize=True)

    statuses = _statuses(result)
    assert all(statuses[name] == "pass" for name in (
        "capabilities", "voices", "unsupported_speed", "unsupported_instructions",
        "unknown_voice", "audio",
    ))
    assert statuses["live_reader"] == "skip"
    assert statuses["generation_limit"] == "skip"
    assert result["audio"]["sample_rate"] == 24000
    assert result["audio"]["duration_s"] == 0.1
    assert any("cache=false" in path for path, _payload in client.calls)
    assert all(
        payload["input"] == "x"
        for path, payload in client.calls
        if payload is not None and path == "/v1/audio/speech?cache=false" and payload.get("speed") == 1.2
    )


def test_clone_without_reference_is_skipped_not_passed():
    result = inspect_model(FakeClient(clone=True), "example/model", synthesize=True)
    assert _statuses(result)["audio"] == "skip"
    assert _statuses(result)["unknown_voice"] == "skip"


def test_generation_failure_is_reported_by_code():
    result = inspect_model(FakeClient(audio_status=502), "example/model", synthesize=True)
    audio = next(check for check in result["checks"] if check["name"] == "audio")
    assert audio["status"] == "fail"
    assert "generation_limit_reached" in audio["detail"]


def test_input_limit_probe_is_opt_in_and_uses_declared_boundary():
    manifest = {"max_input_chars": 8, "sample_rate": 24000,
                "capabilities": {"speed_control": False, "instructions": False, "voice_clone": False},
                "voices": [{"id": "speaker"}]}
    client = FakeClient(input_limit=8)

    metadata = inspect_model(client, "example/model", manifest=manifest)
    assert _statuses(metadata)["input_limit_enforcement"] == "skip"
    assert all(payload is None for _path, payload in client.calls)

    checked = inspect_model(client, "example/model", manifest=manifest, probe_rejections=True)
    assert _statuses(checked)["input_limit_enforcement"] == "pass"
    assert any(payload and payload["input"] == "x" * 9 for _path, payload in client.calls)


def test_generation_policy_is_configuration_evidence_not_forced_limit():
    status, detail = inspect_generation_policy({"generation": {
        "max_new_tokens_ceiling": 1200,
        "max_segment_units": 400.0,
        "effective_segment_units": 400.0,
    }})
    assert status == "pass"
    assert "ceiling=1200" in detail
    assert inspect_generation_policy({"generation": {"max_new_tokens_ceiling": 0}})[0] == "fail"
    report = inspect_model(FakeClient(), "example/model", generation_policy=detail)
    check = next(check for check in report["checks"] if check["name"] == "generation_limit")
    assert check["status"] == "skip"
    assert "Forced-limit" in check["detail"]


def test_metadata_only_does_not_post_synthesis():
    client = FakeClient()
    result = inspect_model(client, "example/model")
    assert all(payload is None for _path, payload in client.calls)
    assert _statuses(result)["unsupported_speed"] == "skip"
    assert _statuses(result)["audio"] == "skip"


def test_unloaded_model_is_skipped_and_early_failure_has_full_scaffold():
    unloaded = inspect_model(FakeClient(audio_status=409), "example/model", synthesize=True)
    assert _statuses(unloaded)["audio"] == "skip"

    unavailable = inspect_model(FakeClient(capabilities_status=503), "example/model")
    assert _statuses(unavailable)["capabilities"] == "fail"
    assert _statuses(unavailable)["audio"] == "skip"
    assert _statuses(unavailable)["unknown_voice"] == "skip"


def test_run_metadata_summary_is_machine_readable_and_get_only(monkeypatch):
    client = FakeClient()
    monkeypatch.setattr(tts_conformance, "Client", lambda *_args, **_kwargs: client)
    args = SimpleNamespace(
        url="http://core", api_key_env="OS_API_KEY", insecure=False, timeout=5,
        worker_url=None, provider=None, models=["example/model"],
        synthesize=False, voice_library_ref="", probe_rejections=False,
    )

    report = tts_conformance.run(args)

    assert report["schema_version"] == 1
    assert report["summary"]["fail"] == 0
    assert report["summary"]["skip"] > 0
    assert all(payload is None for _path, payload in client.calls)


def test_audio_must_match_sample_rate_and_contain_signal():
    assert inspect_audio(_wav(), 24000)["frames"] == 2400
    try:
        inspect_audio(_wav(), 22050)
    except ValueError as exc:
        assert "invalid audio" in str(exc)
    else:
        raise AssertionError("Mismatched sample rate was accepted")
