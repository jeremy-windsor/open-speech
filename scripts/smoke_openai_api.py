"""End-to-end smoke test for a running Open Speech server.

Stdlib only, so it runs from any Python 3.9+ (Windows, Linux, macOS) with no pip install.
It checks the OpenAI-compatible surface the way a third-party app would use it:

  1. GET  /health                      -> 200
  2. GET  /v1/models                   -> returns a non-empty OpenAI-style model list
  3. POST /v1/audio/speech             -> real WAV (mono PCM16, non-silent)
  4. POST /v1/audio/transcriptions     -> transcribes step 3's WAV back to roughly the same words
  5. POST /v1/audio/transcriptions     -> text / srt / vtt / verbose_json formats all return 200
  6. OpenAI model names                -> tts-1 and whisper-1 work, as an unmodified OpenAI app sends them

Usage (PowerShell or bash):
  python scripts/smoke_openai_api.py --url https://localhost:8100 --insecure
  python scripts/smoke_openai_api.py --url http://localhost:8100 --stt-model Systran/faster-whisper-base.en

The first run can take minutes while the server downloads model weights. Exit code 0 = all passed.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import ssl
import struct
import sys
import time
import urllib.error
import urllib.request
import uuid
import wave

TEXT = "The quick brown fox jumps over the lazy dog near the river bank."


def request(args, method, path, *, body=None, headers=None):
    url = args.url.rstrip("/") + path
    req = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    key = os.environ.get(args.api_key_env, "")
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    ctx = ssl._create_unverified_context() if args.insecure else None
    try:
        with urllib.request.urlopen(req, timeout=args.timeout, context=ctx) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def multipart(fields, file_bytes, filename="smoke.wav"):
    boundary = uuid.uuid4().hex
    out = io.BytesIO()
    for name, value in fields.items():
        out.write(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    out.write(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n".encode()
    )
    out.write(file_bytes)
    out.write(f"\r\n--{boundary}--\r\n".encode())
    return out.getvalue(), {"Content-Type": f"multipart/form-data; boundary={boundary}"}


def words(text):
    return re.findall(r"[a-z']+", text.lower())


def check_health(args, state):
    status, body = request(args, "GET", "/health")
    assert status == 200, f"HTTP {status}: {body[:200]!r}"
    return body.decode()[:120]


def check_models(args, state):
    # /v1/models lists loaded models plus the server's default STT and TTS ids, not every model.
    status, body = request(args, "GET", "/v1/models")
    assert status == 200, f"HTTP {status}: {body[:200]!r}"
    ids = [m.get("id") for m in json.loads(body).get("data", [])]
    assert ids, "empty model list"
    return ", ".join(ids)


def check_speech(args, state):
    payload = {"model": args.tts_model, "input": TEXT, "voice": args.voice, "response_format": "wav"}
    t0 = time.monotonic()
    status, body = request(
        args, "POST", "/v1/audio/speech",
        body=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
    )
    elapsed = time.monotonic() - t0
    assert status == 200, f"HTTP {status}: {body[:300]!r}"
    with wave.open(io.BytesIO(body)) as wav:
        channels, width, rate, frames = wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
        raw = wav.readframes(frames)
    assert channels == 1 and width == 2, f"expected mono PCM16, got {channels}ch {width * 8}-bit"
    duration = frames / rate
    assert duration > 0.5, f"audio too short ({duration:.2f}s)"
    samples = struct.unpack(f"<{len(raw) // 2}h", raw)
    rms = math.sqrt(sum(s * s for s in samples) / len(samples))
    assert rms > 100, f"audio looks silent (RMS {rms:.0f})"
    state["wav"] = body
    return f"{duration:.2f}s @ {rate} Hz, RMS {rms:.0f}, {elapsed:.1f}s"


def check_transcribe(args, state):
    assert "wav" in state, "skipped: no WAV from the speech step"
    body, headers = multipart({"model": args.stt_model, "response_format": "json"}, state["wav"])
    t0 = time.monotonic()
    status, resp = request(args, "POST", "/v1/audio/transcriptions", body=body, headers=headers)
    elapsed = time.monotonic() - t0
    assert status == 200, f"HTTP {status}: {resp[:300]!r}"
    text = json.loads(resp).get("text", "")
    expected, heard = words(TEXT), set(words(text))
    hit = sum(1 for w in expected if w in heard) / len(expected)
    assert hit >= 0.7, f"only {hit:.0%} of words matched: {text!r}"
    return f"{hit:.0%} word match, {elapsed:.1f}s: {text.strip()!r}"


def check_formats(args, state):
    assert "wav" in state, "skipped: no WAV from the speech step"
    bad = []
    for fmt in ("text", "srt", "vtt", "verbose_json"):
        body, headers = multipart({"model": args.stt_model, "response_format": fmt}, state["wav"])
        status, resp = request(args, "POST", "/v1/audio/transcriptions", body=body, headers=headers)
        if status != 200 or not resp.strip():
            bad.append(f"{fmt}=HTTP {status}")
    assert not bad, ", ".join(bad)
    return "text, srt, vtt, verbose_json OK"


def check_openai_names(args, state):
    payload = {"model": "tts-1", "input": TEXT, "voice": "alloy", "response_format": "wav"}
    status, wav = request(
        args, "POST", "/v1/audio/speech",
        body=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
    )
    assert status == 200, f"tts-1: HTTP {status}: {wav[:300]!r}"
    body, headers = multipart({"model": "whisper-1", "response_format": "json"}, wav)
    status, resp = request(args, "POST", "/v1/audio/transcriptions", body=body, headers=headers)
    assert status == 200, f"whisper-1: HTTP {status}: {resp[:300]!r}"
    text = json.loads(resp).get("text", "")
    assert text.strip(), "whisper-1 returned empty text"
    return f"tts-1 and whisper-1 accepted: {text.strip()!r}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="https://localhost:8100")
    p.add_argument("--insecure", action="store_true", help="accept the self-signed TLS cert")
    p.add_argument("--api-key-env", default="OS_API_KEY", help="env var holding the API key, if auth is on")
    p.add_argument("--stt-model", default="Systran/faster-whisper-tiny.en")
    p.add_argument("--tts-model", default="kokoro")
    p.add_argument("--voice", default="af_heart")
    p.add_argument("--timeout", type=float, default=900, help="seconds per request (first load downloads weights)")
    args = p.parse_args()

    checks = [
        ("health", check_health), ("models", check_models), ("speech", check_speech),
        ("transcribe", check_transcribe), ("formats", check_formats),
        ("openai-names", check_openai_names),
    ]
    state, failed = {}, 0
    for name, fn in checks:
        try:
            print(f"PASS {name:<12} {fn(args, state)}", flush=True)
        except Exception as exc:  # noqa: BLE001 - report every failure, keep going
            failed += 1
            print(f"FAIL {name:<12} {exc}", flush=True)
    print(f"\n{len(checks) - failed}/{len(checks)} checks passed against {args.url}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
