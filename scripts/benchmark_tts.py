#!/usr/bin/env python3
"""Record repeatable one-shot or Live Reader TTS harness measurements."""

from __future__ import annotations

import argparse
import base64
import json
import ssl
import subprocess
import time
import urllib.parse
import urllib.request
import wave
from pathlib import Path
from typing import Any


DEFAULT_TEXT = (
    "In the beginning God created the heaven and the earth. "
    "And the earth was without form, and void; and darkness was upon the face of the deep."
)


def _ssl_context(insecure: bool) -> ssl.SSLContext | None:
    return ssl._create_unverified_context() if insecure else None


def _request(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    api_key: str = "",
    insecure: bool = False,
    timeout: float = 900,
) -> tuple[bytes, dict[str, str]]:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(
        request,
        timeout=timeout,
        context=_ssl_context(insecure),
    ) as response:
        return response.read(), dict(response.headers.items())


def _wav_duration(data: bytes) -> tuple[float, int]:
    import io

    with wave.open(io.BytesIO(data), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate(), wav_file.getframerate()


def _gpu_sample() -> dict[str, Any] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
        name, driver, used, total = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
        return {
            "name": name,
            "driver": driver,
            "memory_used_mb": int(used),
            "memory_total_mb": int(total),
        }
    except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError):
        return None


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _load_model(args: argparse.Namespace) -> float:
    started = time.perf_counter()
    _request(
        f"{args.url}/api/models/{urllib.parse.quote(args.model, safe='')}/load",
        method="POST",
        api_key=args.api_key,
        insecure=args.insecure,
    )
    return time.perf_counter() - started


def _oneshot(args: argparse.Namespace) -> tuple[bytes, dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": args.model,
        "voice": args.voice,
        "input": args.text,
        "speed": args.speed,
        "response_format": "wav",
    }
    if args.voice_library_ref:
        payload["voice_library_ref"] = args.voice_library_ref
    if args.clone_transcript:
        payload["clone_transcript"] = args.clone_transcript
    started = time.perf_counter()
    audio, _headers = _request(
        f"{args.url}/v1/audio/speech",
        method="POST",
        payload=payload,
        api_key=args.api_key,
        insecure=args.insecure,
    )
    total = time.perf_counter() - started
    duration, sample_rate = _wav_duration(audio)
    return audio, {
        "time_to_complete_s": total,
        "ttfa_s": None,
        "ttfa_source": None,
        "audio_duration_s": duration,
        "rtf": total / duration if duration else None,
        "sample_rate": sample_rate,
    }


def _live(args: argparse.Namespace) -> tuple[bytes, dict[str, Any]]:
    from websockets.sync.client import connect

    parsed = urllib.parse.urlsplit(args.url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_url = urllib.parse.urlunsplit(
        (scheme, parsed.netloc, "/v1/audio/speech/stream", "", "")
    )
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else None
    ssl_context = _ssl_context(args.insecure) if scheme == "wss" else None
    pcm_parts: list[bytes] = []
    first_delta: float | None = None
    sample_rate = 24000
    with connect(
        ws_url,
        additional_headers=headers,
        ssl=ssl_context,
        open_timeout=30,
        close_timeout=10,
    ) as websocket:
        created = json.loads(websocket.recv(timeout=30))
        if created.get("type") != "session.created":
            raise RuntimeError(f"Expected session.created, got {created.get('type')}")
        websocket.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "model": args.model,
                        "voice": args.voice,
                        "speed": args.speed,
                        "language": args.language,
                        "latency_mode": "natural",
                    },
                }
            )
        )
        updated = json.loads(websocket.recv(timeout=30))
        if updated.get("type") == "error":
            raise RuntimeError(updated.get("error", {}).get("message", "session.update failed"))
        started = time.perf_counter()
        websocket.send(json.dumps({"type": "input_text.append", "text": args.text}))
        websocket.send(json.dumps({"type": "input_text.commit"}))
        while True:
            event = json.loads(websocket.recv(timeout=900))
            event_type = event.get("type")
            if event_type == "response.output_audio.delta":
                if first_delta is None:
                    first_delta = time.perf_counter() - started
                pcm_parts.append(base64.b64decode(event["delta"]))
                websocket.send(
                    json.dumps(
                        {
                            "type": "playback.ack",
                            "response_id": event["response_id"],
                            "sequence": event["sequence"],
                        }
                    )
                )
                sample_rate = int(event.get("sample_rate") or sample_rate)
            elif event_type == "error":
                raise RuntimeError(event.get("error", {}).get("message", "Live Reader failed"))
            elif event_type == "response.done":
                break

    total = time.perf_counter() - started
    pcm = b"".join(pcm_parts)
    output_path = Path(args.output)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    duration = len(pcm) / 2 / sample_rate
    return output_path.read_bytes(), {
        "time_to_complete_s": total,
        "ttfa_s": first_delta,
        "ttfa_source": "live_reader_first_pcm_delta",
        "audio_duration_s": duration,
        "rtf": total / duration if duration else None,
        "sample_rate": sample_rate,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://localhost:8100")
    parser.add_argument("--model", default="kokoro")
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--language")
    parser.add_argument("--mode", choices=("oneshot", "live"), default="oneshot")
    parser.add_argument("--output", default="tts-benchmark.wav")
    parser.add_argument("--results", default="tts-benchmark.json")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--no-load", action="store_true")
    parser.add_argument("--voice-library-ref")
    parser.add_argument("--clone-transcript")
    args = parser.parse_args()
    args.url = args.url.rstrip("/")

    record: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _git_sha(),
        "mode": args.mode,
        "model": args.model,
        "voice": args.voice,
        "speed": args.speed,
        "text_chars": len(args.text),
        "gpu_before": _gpu_sample(),
    }
    try:
        if not args.no_load:
            record["load_seconds"] = _load_model(args)
        audio, metrics = _live(args) if args.mode == "live" else _oneshot(args)
        if args.mode == "oneshot":
            Path(args.output).write_bytes(audio)
        record.update(metrics)
        record["output_bytes"] = len(audio)
        record["error"] = None
    except Exception as exc:
        record["error"] = str(exc)
    record["gpu_after"] = _gpu_sample()
    Path(args.results).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 1 if record.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
