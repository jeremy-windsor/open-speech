"""Produce a machine-readable TTS harness conformance report.

Metadata checks do not load models. Rejection probes and inference are opt-in.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from typing import Any

TEST_TEXT = "The quick brown fox jumps over the lazy dog."
SCHEMA_VERSION = 1
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
MAX_LIMIT_PROBE_CHARS = 8192
MODEL_CHECK_NAMES = (
    "capabilities", "voices", "manifest_core_match", "input_limit_enforcement",
    "unsupported_speed", "unsupported_instructions", "unknown_voice", "audio",
    "live_reader", "generation_limit",
)


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class Client:
    def __init__(self, url: str, *, api_key: str = "", insecure: bool = False, timeout: float = 120) -> None:
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        context = ssl._create_unverified_context() if insecure else ssl.create_default_context()
        self.opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}),
            urllib.request.HTTPSHandler(context=context),
            _NoRedirects(),
        )

    def request(self, path: str, payload: dict[str, Any] | None = None) -> tuple[int, bytes]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = None
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            self.url + path,
            data=data,
            headers=headers,
            method="POST" if payload is not None else "GET",
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                body = response.read(MAX_RESPONSE_BYTES + 1)
                if len(body) > MAX_RESPONSE_BYTES:
                    raise ValueError("Conformance response exceeds 32 MiB")
                return response.status, body
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(65536)


def _json(data: bytes) -> dict[str, Any]:
    value = json.loads(data)
    if not isinstance(value, dict):
        raise TypeError("Expected a JSON object")
    return value


def _detail(data: bytes) -> str:
    try:
        detail = _json(data).get("detail", "")
        if isinstance(detail, dict):
            return str(detail.get("code") or detail.get("message") or "")
        if isinstance(detail, str):
            return detail[:160]
    except (ValueError, TypeError, UnicodeDecodeError):
        pass
    return ""


def _check(checks: list[dict[str, Any]], name: str, status: str, detail: str = "") -> None:
    checks.append({"name": name, "status": status, "detail": detail})


def validate_manifest(payload: dict[str, Any], provider: str) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Validate the public worker metadata without loading model code."""
    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    if payload.get("provider") != provider:
        errors.append("provider identity mismatch")
    raw_models = payload.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        return {}, errors + ["models must be a nonempty list"]
    models: dict[str, dict[str, Any]] = {}
    for index, model in enumerate(raw_models):
        if not isinstance(model, dict):
            errors.append(f"models[{index}] is not an object")
            continue
        model_id = model.get("id")
        if not isinstance(model_id, str) or not model_id:
            errors.append(f"models[{index}] has no id")
            continue
        if model_id in models:
            errors.append(f"duplicate model {model_id}")
        models[model_id] = model
        if model.get("provider") != provider:
            errors.append(f"{model_id}: provider mismatch")
        rate = model.get("sample_rate")
        if type(rate) is not int or rate <= 0:
            errors.append(f"{model_id}: invalid sample_rate")
        limit = model.get("max_input_chars")
        if limit is not None and (type(limit) is not int or limit <= 0):
            errors.append(f"{model_id}: invalid max_input_chars")
        if not isinstance(model.get("capabilities"), dict):
            errors.append(f"{model_id}: missing capabilities")
        voices = model.get("voices")
        if not isinstance(voices, list) or any(
            not isinstance(voice, dict) or not isinstance(voice.get("id"), str) or not voice["id"]
            for voice in voices
        ):
            errors.append(f"{model_id}: invalid voices")
        elif len({voice["id"] for voice in voices}) != len(voices):
            errors.append(f"{model_id}: duplicate voice")
    return models, errors


def _speech_payload(model: str, voice: str, **fields: Any) -> dict[str, Any]:
    return {
        "model": model,
        "voice": voice,
        "input": TEST_TEXT,
        "response_format": "wav",
        **fields,
    }


def _rejects(client: Client, payload: dict[str, Any]) -> tuple[bool, str]:
    # A broken provider might accept the invalid request and synthesize.
    # Keep that accidental work short and outside the shared output cache.
    status, data = client.request(
        "/v1/audio/speech?cache=false", {**payload, "input": "x"}
    )
    return status in {400, 422}, f"HTTP {status}" + (f": {_detail(data)}" if status >= 400 else "")


def _limit_rejection(status: int, data: bytes) -> bool:
    detail = data[:4096].decode("utf-8", errors="replace").lower()
    return status in {400, 413, 422} and any(
        marker in detail for marker in ("too long", "max_length", "string_too_long", "input_length")
    )


def _probe_input_limit(client: Client, model: str, voice: str, limit: int) -> tuple[bool, str]:
    # Opt-in only: a broken limit could otherwise start an expensive synthesis.
    overlong = "x" * (limit + 1)
    status, data = client.request(
        "/v1/audio/speech?cache=false",
        _speech_payload(model, voice, input=overlong),
    )
    return _limit_rejection(status, data), f"core HTTP {status}"


def inspect_generation_policy(payload: dict[str, Any]) -> tuple[str, str]:
    generation = payload.get("generation")
    if not isinstance(generation, dict):
        return "skip", "Worker does not advertise generation policy"
    ceiling = generation.get("max_new_tokens_ceiling")
    segment = generation.get("max_segment_units")
    effective = generation.get("effective_segment_units")
    if (
        type(ceiling) is not int or ceiling <= 0
        or type(segment) not in {int, float} or not math.isfinite(segment) or segment <= 0
        or type(effective) not in {int, float} or not math.isfinite(effective)
        or not 0 < effective <= segment
    ):
        return "fail", "Invalid generation ceiling or segment-unit policy"
    return "pass", f"ceiling={ceiling}, effective_segment_units={effective}"


def inspect_audio(data: bytes, expected_rate: int) -> dict[str, Any]:
    with wave.open(io.BytesIO(data), "rb") as audio:
        rate = audio.getframerate()
        channels = audio.getnchannels()
        frames = audio.getnframes()
        pcm = audio.readframes(frames)
    if rate != expected_rate or channels != 1 or frames == 0 or not any(pcm):
        raise ValueError(f"invalid audio: {rate} Hz, {channels} channel(s), {frames} frames, nonzero={any(pcm)}")
    return {"sample_rate": rate, "duration_s": round(frames / rate, 3), "frames": frames}


def inspect_model(
    client: Client,
    model: str,
    *,
    manifest: dict[str, Any] | None = None,
    probe_rejections: bool = False,
    synthesize: bool = False,
    voice_library_ref: str = "",
    generation_policy: str = "",
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    result: dict[str, Any] = {"model": model, "checks": checks}

    def finish(reason: str = "Not applicable for this model") -> dict[str, Any]:
        completed = {check["name"] for check in checks}
        for name in MODEL_CHECK_NAMES:
            if name not in completed:
                _check(checks, name, "skip", reason)
        return result

    query = urllib.parse.quote(model, safe="")
    try:
        status, data = client.request(f"/api/tts/capabilities?model={query}")
        if status != 200:
            _check(checks, "capabilities", "fail", f"HTTP {status}: {_detail(data)}")
            return finish("Not reached: capabilities failed")
        info = _json(data)
        caps = info.get("capabilities")
        rate = info.get("sample_rate")
        if info.get("model") != model or not isinstance(caps, dict) or type(rate) is not int or rate <= 0:
            _check(checks, "capabilities", "fail", "Invalid capabilities response")
            return finish("Not reached: capabilities failed")
        result["provider"] = info.get("backend")
        result["capabilities"] = caps
        result["sample_rate"] = rate
        _check(checks, "capabilities", "pass")

        status, data = client.request(f"/v1/audio/voices?model={query}")
        voices = _json(data).get("voices") if status == 200 else None
        if not isinstance(voices, list) or any(not isinstance(v, dict) or not v.get("id") for v in voices):
            _check(checks, "voices", "fail", f"HTTP {status} or invalid voice list")
            return finish("Not reached: voice catalog failed")
        voice_ids = [v["id"] for v in voices]
        _check(
            checks, "voices",
            "skip" if not voice_ids else "pass" if len(set(voice_ids)) == len(voice_ids) else "fail",
            f"{len(voice_ids)} voices",
        )
        result["voice_count"] = len(voice_ids)
        voice = voice_ids[0] if voice_ids else "conformance_voice"

        if manifest is not None:
            matching = (
                manifest.get("sample_rate") == rate
                and manifest.get("capabilities") == caps
                and [v.get("id") for v in manifest.get("voices", [])] == voice_ids
            )
            _check(checks, "manifest_core_match", "pass" if matching else "fail")
            limit = manifest.get("max_input_chars")
            if type(limit) is int and limit > 0:
                if not probe_rejections:
                    _check(checks, "input_limit_enforcement", "skip", "Use --probe-rejections")
                elif limit + 1 > MAX_LIMIT_PROBE_CHARS:
                    _check(checks, "input_limit_enforcement", "skip", "Advertised limit exceeds safe probe size")
                else:
                    rejected, detail = _probe_input_limit(client, model, voice, limit)
                    _check(checks, "input_limit_enforcement", "pass" if rejected else "fail", detail)

        for name, enabled, fields in (
            ("unsupported_speed", caps.get("speed_control", False), {"speed": 1.2}),
            ("unsupported_instructions", caps.get("instructions", False), {"instructions": "Speak warmly."}),
        ):
            if not probe_rejections:
                _check(checks, name, "skip", "Use --probe-rejections")
            elif enabled:
                _check(checks, name, "skip", "Capability is supported")
            else:
                rejected, detail = _rejects(client, _speech_payload(model, voice, **fields))
                _check(checks, name, "pass" if rejected else "fail", detail)

        if not probe_rejections:
            _check(checks, "unknown_voice", "skip", "Use --probe-rejections")
        elif voice_ids and info.get("backend") != "piper":
            rejected, detail = _rejects(client, _speech_payload(model, "__invalid_voice__"))
            _check(checks, "unknown_voice", "pass" if rejected else "fail", detail)
        else:
            _check(checks, "unknown_voice", "skip", "No selectable voices or single-speaker model")

        if not synthesize:
            _check(checks, "audio", "skip", "Use --synthesize to run inference")
        elif caps.get("voice_clone") and not voice_ids and not voice_library_ref:
            _check(checks, "audio", "skip", "Supply --voice-library-ref for clone inference")
        else:
            payload = _speech_payload(model, voice)
            if voice_library_ref and caps.get("voice_clone"):
                payload["voice_library_ref"] = voice_library_ref
            started = time.perf_counter()
            status, data = client.request("/v1/audio/speech?cache=false", payload)
            if status == 409:
                _check(checks, "audio", "skip", "Model is not loaded (HTTP 409)")
            elif status != 200:
                _check(checks, "audio", "fail", f"HTTP {status}: {_detail(data)}")
            else:
                try:
                    result["audio"] = inspect_audio(data, rate)
                    result["audio"]["elapsed_s"] = round(time.perf_counter() - started, 3)
                    _check(checks, "audio", "pass")
                except (ValueError, EOFError, wave.Error) as exc:
                    _check(checks, "audio", "fail", str(exc))
        _check(checks, "live_reader", "skip", "WebSocket playback and cancellation need a separate run")
        _check(
            checks, "generation_limit", "skip",
            (generation_policy + "; " if generation_policy else "") + "Forced-limit rejection requires a controlled fixture",
        )
    except (OSError, TimeoutError, ValueError, TypeError, UnicodeDecodeError) as exc:
        _check(checks, "transport", "fail", str(exc)[:160])
        return finish("Not reached: transport or response failed")
    return finish()


def run(args: argparse.Namespace) -> dict[str, Any]:
    client = Client(args.url, api_key=os.environ.get(args.api_key_env, ""), insecure=args.insecure, timeout=args.timeout)
    report: dict[str, Any] = {
        "schema_version": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": [],
        "checks": [],
    }
    manifests: dict[str, dict[str, Any]] = {}
    generation_policy = ""
    if args.worker_url:
        worker = Client(args.worker_url, timeout=args.timeout)
        try:
            status, data = worker.request("/v1/manifest")
            if status != 200:
                _check(report["checks"], "worker_manifest", "fail", f"HTTP {status}")
            else:
                manifests, errors = validate_manifest(_json(data), args.provider)
                _check(report["checks"], "worker_manifest", "fail" if errors else "pass", "; ".join(errors))
        except (OSError, ValueError, TypeError) as exc:
            _check(report["checks"], "worker_manifest", "fail", str(exc)[:160])
        try:
            status, data = worker.request("/health")
            if status == 200:
                policy_status, generation_policy = inspect_generation_policy(_json(data))
                _check(report["checks"], "worker_generation_policy", policy_status, generation_policy)
            else:
                _check(report["checks"], "worker_generation_policy", "skip", f"HTTP {status}")
        except (OSError, ValueError, TypeError) as exc:
            _check(report["checks"], "worker_generation_policy", "skip", str(exc)[:160])
    for model in args.models:
        report["models"].append(
            inspect_model(
                client, model, manifest=manifests.get(model),
                probe_rejections=args.probe_rejections,
                synthesize=args.synthesize, voice_library_ref=args.voice_library_ref,
                generation_policy=generation_policy if model in manifests else "",
            )
        )
        if args.worker_url and model not in manifests and model.startswith(args.provider + "/"):
            _check(report["models"][-1]["checks"], "manifest_presence", "fail", "Model absent from worker manifest")
    if args.probe_rejections:
        try:
            rejected, detail = _rejects(client, _speech_payload("__invalid_tts_model__", "invalid"))
            _check(report["checks"], "unknown_model", "pass" if rejected else "fail", detail)
        except (OSError, TimeoutError, ValueError, TypeError) as exc:
            _check(report["checks"], "unknown_model", "fail", str(exc)[:160])
    else:
        _check(report["checks"], "unknown_model", "skip", "Use --probe-rejections")
    _check(report["checks"], "worker_down", "skip", "Requires a controlled worker outage")
    _check(report["checks"], "aborted_stream", "skip", "Requires fault injection")
    _check(report["checks"], "manifest_version_rejection", "skip", "Requires an incompatible worker fixture")
    statuses = [check["status"] for check in report["checks"]]
    statuses += [check["status"] for model in report["models"] for check in model["checks"]]
    report["summary"] = {status: statuses.count(status) for status in ("pass", "fail", "skip")}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://localhost:8100", help="Core harness URL")
    parser.add_argument("--model", action="append", dest="models", required=True, help="Repeat for each model")
    parser.add_argument("--worker-url", help="Optional private worker URL, reachable from this process")
    parser.add_argument("--provider", help="Expected worker provider ID")
    parser.add_argument("--api-key-env", default="OS_API_KEY", help="Environment variable holding the core API key")
    parser.add_argument("--insecure", action="store_true", help="Allow a self-signed core TLS certificate")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--probe-rejections", action="store_true", help="Opt in to invalid synthesis requests")
    parser.add_argument("--synthesize", action="store_true", help="Opt in to inference; may lazily load local models")
    parser.add_argument("--voice-library-ref", default="", help="Existing provider-neutral clone asset")
    args = parser.parse_args()
    if bool(args.worker_url) != bool(args.provider):
        parser.error("--worker-url and --provider must be supplied together")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    report = run(args)
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 1 if report["summary"]["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
