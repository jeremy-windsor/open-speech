"""Headless local microphone listener for Open Speech streaming STT."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import ssl
import sys
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import websockets


DEFAULT_URL = "ws://localhost:8100/v1/audio/stream?sample_rate=16000"


def _load_sounddevice():
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:
        raise SystemExit(
            "sounddevice could not load. Install example dependencies with "
            "python3 -m pip install -e '.[examples]' and install PortAudio "
            "(for example: apt install libportaudio2, brew install portaudio, "
            "or your OS equivalent)."
        ) from exc
    return sd


def _build_url(url: str, sample_rate: int, model: str | None, vad: bool | None) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["sample_rate"] = str(sample_rate)
    if model:
        query["model"] = model
    if vad is not None:
        query["vad"] = "true" if vad else "false"
    return urlunsplit(parts._replace(query=urlencode(query)))


def _connect(url: str, headers: dict[str, str], insecure: bool):
    kwargs = {}
    if insecure and urlsplit(url).scheme == "wss":
        kwargs["ssl"] = ssl._create_unverified_context()
    if not headers:
        return websockets.connect(url, **kwargs)
    params = inspect.signature(websockets.connect).parameters
    if "additional_headers" in params:
        return websockets.connect(url, additional_headers=headers, **kwargs)
    return websockets.connect(url, extra_headers=headers, **kwargs)


def _enqueue_frame(queue: asyncio.Queue[bytes], frame: bytes) -> None:
    if queue.full():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
    try:
        queue.put_nowait(frame)
    except asyncio.QueueFull:
        pass


def _print_event(event: dict) -> None:
    event_type = event.get("type")
    if event_type == "transcript":
        text = str(event.get("text", "")).strip()
        if text:
            kind = "final" if event.get("speech_final") or event.get("is_final") else "interim"
            print(f"[transcript:{kind}] {text}", flush=True)
    elif event_type == "vad":
        print(f"[vad] {event.get('state')}", flush=True)
    elif event_type == "session.begin":
        print(f"[session] begin {event.get('session_id', '')}", flush=True)
    elif event_type == "session.end":
        print(f"[session] end reason={event.get('reason', '')}", flush=True)
    elif event_type == "error":
        print(f"[error] {event.get('message', event)}", file=sys.stderr, flush=True)
    else:
        print(json.dumps(event, sort_keys=True), flush=True)


async def _send_audio(ws, queue: asyncio.Queue[bytes]) -> None:
    while True:
        frame = await queue.get()
        await ws.send(frame)


async def _receive_events(ws) -> None:
    async for message in ws:
        if isinstance(message, bytes):
            continue
        try:
            event = json.loads(message)
        except json.JSONDecodeError:
            print(message, flush=True)
            continue
        _print_event(event)
        if event.get("type") == "session.end":
            return


async def listen(args: argparse.Namespace) -> None:
    sd = _load_sounddevice()
    if args.list_devices:
        print(sd.query_devices())
        return

    ws_url = _build_url(args.url, args.sample_rate, args.model, args.vad)
    headers = {}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    blocksize = max(1, int(args.sample_rate * args.block_ms / 1000))
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=args.queue_size)
    loop = asyncio.get_running_loop()

    def callback(indata, frames, time, status) -> None:
        del frames, time
        if status:
            loop.call_soon_threadsafe(
                lambda text=str(status): print(f"[audio] {text}", file=sys.stderr, flush=True)
            )
        loop.call_soon_threadsafe(_enqueue_frame, queue, bytes(indata))

    async with _connect(ws_url, headers, args.insecure) as ws:
        print(f"[connect] {ws_url}", flush=True)
        with sd.RawInputStream(
            samplerate=args.sample_rate,
            blocksize=blocksize,
            device=args.device,
            channels=1,
            dtype="int16",
            callback=callback,
        ):
            print("[listen] microphone active; press Ctrl+C to stop", flush=True)
            send_task = asyncio.create_task(_send_audio(ws, queue))
            recv_task = asyncio.create_task(_receive_events(ws))
            done, pending = await asyncio.wait(
                {send_task, recv_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                exc = task.exception()
                if exc is not None:
                    raise exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help=f"WebSocket URL (default: {DEFAULT_URL})")
    parser.add_argument("--model", help="Optional STT model id query parameter")
    parser.add_argument(
        "--vad",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override server VAD setting for this stream",
    )
    parser.add_argument("--api-key", default=os.environ.get("OS_API_KEY"), help="Bearer API key")
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS verification for wss:// self-signed local testing",
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Microphone sample rate")
    parser.add_argument("--block-ms", type=int, default=100, help="Microphone frame size in ms")
    parser.add_argument("--device", help="sounddevice input device id or name")
    parser.add_argument("--queue-size", type=int, default=20, help="Buffered mic frames before dropping")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    return parser.parse_args()


def main() -> None:
    try:
        asyncio.run(listen(parse_args()))
    except KeyboardInterrupt:
        print("\n[listen] stopped", flush=True)


if __name__ == "__main__":
    main()
