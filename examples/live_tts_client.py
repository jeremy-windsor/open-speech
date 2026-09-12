"""Stream stdin text to Live Reader and save returned PCM as a WAV file.

The requester consumes the returned audio. Open Speech performs synthesis only;
it never opens an audio device. Browser clients can play the same PCM events as
they arrive instead of writing them to a file.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import ssl
import sys
import wave
from pathlib import Path

from websockets.asyncio.client import connect


def websocket_url(url: str) -> str:
    url = url.rstrip("/")
    if url.startswith("https://"):
        url = "wss://" + url.removeprefix("https://")
    elif url.startswith("http://"):
        url = "ws://" + url.removeprefix("http://")
    return url + "/v1/audio/speech/stream"


def tls_context(url: str, insecure: bool) -> ssl.SSLContext | None:
    if not insecure or not url.startswith("https://"):
        return None
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


async def run(args: argparse.Namespace) -> None:
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else None
    ready = asyncio.Event()
    input_complete = asyncio.Event()
    output_complete = asyncio.Event()
    sample_rate = 24000
    pcm_chunks: list[bytes] = []

    async with connect(
        websocket_url(args.url),
        additional_headers=headers,
        ssl=tls_context(args.url, args.insecure),
    ) as websocket:

        async def send_input() -> None:
            await ready.wait()
            while line := await asyncio.to_thread(sys.stdin.readline):
                await websocket.send(json.dumps({"type": "input_text.append", "text": line}))
            await websocket.send(json.dumps({"type": "input_text.commit"}))
            input_complete.set()

        sender = asyncio.create_task(send_input())
        try:
            async for raw_event in websocket:
                event = json.loads(raw_event)
                event_type = event.get("type")
                if event_type == "session.created":
                    sample_rate = event["session"]["audio"]["sample_rate"]
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "session.update",
                                "session": {
                                    "model": args.model,
                                    "voice": args.voice,
                                    "speed": args.speed,
                                    "latency_mode": args.latency,
                                },
                            }
                        )
                    )
                elif event_type == "session.updated":
                    ready.set()
                elif event_type == "response.output_audio.delta":
                    pcm_chunks.append(base64.b64decode(event["delta"]))
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "playback.ack",
                                "response_id": event["response_id"],
                                "sequence": event["sequence"],
                            }
                        )
                    )
                elif event_type == "response.created":
                    print(f"Reading: {event['response']['text']}", file=sys.stderr)
                elif event_type == "input_text.done" and input_complete.is_set():
                    output_complete.set()
                    break
                elif event_type == "error":
                    raise RuntimeError(event.get("error", {}).get("message", "Live TTS failed"))
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)

    if not output_complete.is_set():
        raise RuntimeError("Connection closed before synthesis completed")
    output_path = Path(args.output)
    with wave.open(str(output_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"".join(pcm_chunks))
    print(f"Saved {output_path} ({sample_rate} Hz PCM16 mono)", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="https://localhost:8100")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="kokoro")
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--latency", choices=("natural", "instant_word"), default="natural")
    parser.add_argument("--output", default="live-reader.wav")
    parser.add_argument(
        "--insecure", action="store_true", help="Allow a self-signed HTTPS certificate"
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
