#!/usr/bin/env python3
"""Manual WebSocket streaming test client.

Usage:
    python tests/test_ws_client.py [--url wss://localhost:8100] [--file test.wav]
    python tests/test_ws_client.py --generate-tone  # Generate synthetic speech-like audio

This sends audio to the streaming endpoint and prints all responses.
Useful for debugging the Live tab streaming bug.
"""

import argparse
import asyncio
import json
import struct
import sys
import time

import numpy as np


def generate_tone_pcm16(duration_s: float = 3.0, sample_rate: int = 16000) -> bytes:
    """Generate a tone that triggers VAD (speech-like frequency mix)."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s))
    # Mix of speech-range frequencies (100-1000 Hz) with slight noise
    tone = (
        np.sin(2 * np.pi * 200 * t) * 0.3
        + np.sin(2 * np.pi * 400 * t) * 0.2
        + np.sin(2 * np.pi * 800 * t) * 0.1
        + np.random.randn(len(t)) * 0.05
    )
    return (tone * 16000).astype(np.int16).tobytes()


def load_wav_as_pcm16(path: str) -> tuple[bytes, int]:
    """Load WAV file and return (pcm16_bytes, sample_rate)."""
    with open(path, "rb") as f:
        data = f.read()

    assert data[:4] == b"RIFF", "Not a WAV file"
    assert data[8:12] == b"WAVE"

    sample_rate = struct.unpack("<I", data[24:28])[0]
    # Find data chunk
    pos = 12
    while pos < len(data) - 8:
        chunk_id = data[pos : pos + 4]
        chunk_size = struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        if chunk_id == b"data":
            pcm = data[pos + 8 : pos + 8 + chunk_size]
            return pcm, sample_rate
        pos += 8 + chunk_size

    raise ValueError("No data chunk found in WAV")


async def run_stream_test(url: str, pcm_data: bytes, sample_rate: int, chunk_duration_s: float = 2.0):
    """Send audio to WebSocket and print all responses."""
    try:
        import websockets
    except ImportError:
        print("pip install websockets")
        sys.exit(1)

    import ssl
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    ws_url = f"{url}/v1/audio/stream?sample_rate={sample_rate}&interim_results=true"
    print(f"Connecting to {ws_url}...")

    async with websockets.connect(ws_url, ssl=ssl_ctx) as ws:
        print("Connected!")

        # Start receiver task
        async def receiver():
            async for msg in ws:
                event = json.loads(msg)
                ts = time.strftime("%H:%M:%S")
                etype = event.get("type", "?")
                if etype == "transcript":
                    final = "FINAL" if event.get("is_final") else "interim"
                    speech = " [SPEECH_FINAL]" if event.get("speech_final") else ""
                    print(f"  [{ts}] {final}{speech}: {event.get('text', '')}")
                elif etype == "error":
                    print(f"  [{ts}] ERROR: {event.get('message', '')}")
                else:
                    print(f"  [{ts}] {etype}: {json.dumps(event)}")

        recv_task = asyncio.create_task(receiver())

        # Send audio in chunks
        chunk_bytes = int(sample_rate * chunk_duration_s) * 2
        total = len(pcm_data)
        sent = 0
        chunk_num = 0

        print(f"Sending {total} bytes ({total/2/sample_rate:.1f}s) in {chunk_duration_s}s chunks...")

        while sent < total:
            chunk = pcm_data[sent : sent + chunk_bytes]
            await ws.send(chunk)
            sent += len(chunk)
            chunk_num += 1
            print(f"  Sent chunk {chunk_num}: {len(chunk)} bytes ({len(chunk)/2/sample_rate:.2f}s)")
            # Simulate real-time by waiting
            await asyncio.sleep(chunk_duration_s * 0.9)

        print("All audio sent. Sending stop...")
        await ws.send(json.dumps({"type": "stop"}))

        # Wait a bit for final responses
        await asyncio.sleep(3)
        recv_task.cancel()

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="WebSocket streaming test client")
    parser.add_argument("--url", default="wss://localhost:8100", help="Server base URL")
    parser.add_argument("--file", help="WAV file to stream")
    parser.add_argument("--generate-tone", action="store_true", help="Generate synthetic audio")
    parser.add_argument("--rate", type=int, default=16000, help="Sample rate for generated audio")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration for generated audio")
    args = parser.parse_args()

    if args.file:
        pcm_data, sample_rate = load_wav_as_pcm16(args.file)
        print(f"Loaded {args.file}: {len(pcm_data)} bytes, {sample_rate} Hz")
    elif args.generate_tone:
        sample_rate = args.rate
        pcm_data = generate_tone_pcm16(args.duration, sample_rate)
        print(f"Generated {args.duration}s tone at {sample_rate} Hz")
    else:
        print("Specify --file or --generate-tone")
        sys.exit(1)

    asyncio.run(run_stream_test(args.url, pcm_data, sample_rate))


if __name__ == "__main__":
    main()
