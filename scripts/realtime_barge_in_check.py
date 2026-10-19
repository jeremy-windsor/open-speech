"""Check /v1/realtime barge-in against a running server.

Starts a long spoken reply, then talks over it with server-made speech:
- real words should cancel the reply (status "cancelled", reason "turn_detected")
- an acknowledgement ("Yeah, okay.") should let it finish ("completed")

    python scripts/realtime_barge_in_check.py --url https://localhost:8100 --insecure
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import ssl
import sys
import time
import urllib.request

import websockets

REPLY = (
    "Here is a long answer so there is time to talk over it. "
    "The weather this week starts cool and clear, warms up by Wednesday, "
    "and turns rainy on the weekend, so plan outdoor work for the middle of the week."
)
CASES = (
    ("interrupt", "Wait, stop. What about Tuesday?", "cancelled"),
    ("backchannel", "Yeah, okay.", "completed"),
)
CHUNK_MS = 100
BYTES_PER_MS = 48  # PCM16 mono at 24 kHz


def _ssl_context(url: str, insecure: bool) -> ssl.SSLContext | None:
    if not url.startswith(("https", "wss")):
        return None
    return ssl._create_unverified_context() if insecure else ssl.create_default_context()


def speech_pcm(url: str, text: str, model: str | None, insecure: bool) -> bytes:
    body = {"input": text, "response_format": "pcm"}
    if model:
        body["model"] = model
    request = urllib.request.Request(
        url.rstrip("/") + "/v1/audio/speech",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=_ssl_context(url, insecure)),
    )
    with opener.open(request, timeout=300) as response:
        return response.read()


async def run_case(url: str, overlap: bytes, insecure: bool, model: str | None) -> dict:
    ws_url = url.replace("https://", "wss://").replace("http://", "ws://").rstrip("/") + "/v1/realtime"
    async with websockets.connect(ws_url, ssl=_ssl_context(url, insecure), max_size=None) as ws:
        async def next_event() -> dict:
            return json.loads(await asyncio.wait_for(ws.recv(), timeout=300))

        while (await next_event())["type"] != "session.created":
            pass
        started = time.monotonic()
        response = {"instructions": REPLY, **({"model": model} if model else {})}
        await ws.send(json.dumps({"type": "response.create", "response": response}))

        first_audio_s = None
        while first_audio_s is None:
            event = await next_event()
            if event["type"] in ("response.audio.delta", "response.output_audio.delta"):
                first_audio_s = time.monotonic() - started
            elif event["type"] in ("error", "response.done"):
                return {"error": event}

        async def talk() -> None:
            chunk = CHUNK_MS * BYTES_PER_MS
            silence = b"\x00" * (1500 * BYTES_PER_MS)
            for i in range(0, len(overlap + silence), chunk):
                piece = (overlap + silence)[i:i + chunk]
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(piece).decode("ascii"),
                }))
                await asyncio.sleep(CHUNK_MS / 1000)

        talker = asyncio.create_task(talk())
        heard = None
        try:
            while True:
                event = await next_event()
                if event["type"] == "conversation.item.input_audio_transcription.completed":
                    heard = event["transcript"]
                if event["type"] == "response.done":
                    done = event["response"]
                    return {
                        "first_audio_s": round(first_audio_s, 2),
                        "status": done["status"],
                        "reason": done.get("status_details", {}).get("reason"),
                        "heard": heard,
                        "total_s": round(time.monotonic() - started, 2),
                    }
        finally:
            talker.cancel()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="https://localhost:8100")
    parser.add_argument("--insecure", action="store_true", help="accept a self-signed certificate")
    parser.add_argument("--model", help="TTS model for the reply and the overlap (default: server's TTS_MODEL)")
    args = parser.parse_args()

    failures = 0
    for name, text, expected in CASES:
        overlap = speech_pcm(args.url, text, args.model, args.insecure)
        result = asyncio.run(run_case(args.url, overlap, args.insecure, args.model))
        ok = result.get("status") == expected
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'} {name}: expected {expected}, got {json.dumps(result)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
