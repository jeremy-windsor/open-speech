"""Talk to Open Speech's AI assistant through /v1/realtime, with your microphone and speaker.

Speak, pause, and Open Speech transcribes your words, sends them to its AI server (OS_ASSISTANT_URL, or the
connection saved in Settings > AI assistant), says a short phrase such as "Hmm." while the AI works, and then
speaks the reply. The script prints your words, the phrases, the AI's thinking (dimmed) and the reply as they
arrive: the reply on one [ai] line that grows a sentence at a time, and a [cut off: CODE] line when an error or the
token limit stopped it after some of it was spoken. Talk over a reply to stop it.

Use headphones. The script has no echo cancellation: through speakers the microphone hears the reply, the server
takes that for you talking, and the reply interrupts itself. Without headphones, run it with --no-interrupt.

    python examples/voice_chat.py --url https://HOST:8100 --insecure

Needs the `examples` extra (sounddevice and PortAudio) and the AI assistant turned on in the server. A server that
refuses the connection (a missing or wrong key, an Origin it doesn't allow, the realtime API turned off) closes it
right after it opens; the script then prints one [refused] line with the close code, the reason and what to do.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import json
import os
import ssl
import sys
import threading
from typing import Any, Callable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosedError

DEFAULT_URL = "wss://localhost:8100/v1/realtime"
REALTIME_PATH = "/v1/realtime"
SAMPLE_RATE = 24000  # a GA session's PCM16, in and out
_PCM = {"type": "audio/pcm", "rate": SAMPLE_RATE}
_SCHEMES = {"https": "wss", "http": "ws", "wss": "wss", "ws": "ws"}
_AUDIO_DELTAS = ("response.output_audio.delta", "response.audio.delta")
_REPLY_DELTAS = ("response.output_audio_transcript.delta", "response.audio_transcript.delta")
# The events that end a reply's text: its [ai] line is finished.
_REPLY_ENDS = ("response.output_audio_transcript.done", "response.audio_transcript.done", "response.done")
_LABELS = {
    "you": "[you]",
    "phrase": "[waiting]",
    "thinking": "[thinking]",
    "reply": "[ai]",
    "error": "[error]",
    "cancelled": "[canceled]",  # the key is the API's status value
    "failed": "[failed]",
    "cut_off": "[cut off]",
    "note": "[note]",
}
_DIM, _RESET = "\x1b[2m", "\x1b[0m"
HEADPHONES_TIP = (
    "Use headphones: through speakers the microphone hears the reply and the reply interrupts itself. "
    "Without headphones, run with --no-interrupt."
)
# A refused socket opens, then the server closes it at once with one of these codes and the reason.
_REFUSALS = {
    4001: "Pass the server's key with --api-key, or set OS_API_KEY in this shell.",
    1008: "The server allows this client's Origin only once it is listed in OS_WS_ALLOWED_ORIGINS.",
    4004: "The realtime API is turned off on the server (OS_REALTIME_ENABLED=false).",
}


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


# ── The connection: URL, TLS and the API key, as in examples/listen.py ────────


def websocket_url(url: str, model: str | None = None) -> str:
    """The realtime socket of the server ``url`` names: http(s) becomes ws(s), a bare server gets /v1/realtime,
    and ``model`` (the voice model, or an OpenAI realtime name) goes in ``?model=``. ValueError for anything else."""
    parts = urlsplit(url.strip())
    scheme = _SCHEMES.get(parts.scheme.lower())
    if scheme is None or not parts.hostname:
        raise ValueError(f"Not a server address: {url!r}; use https://HOST:8100 or wss://HOST:8100{REALTIME_PATH}")
    path = parts.path.rstrip("/") or REALTIME_PATH
    query = parse_qsl(parts.query, keep_blank_values=True)
    if model:
        query = [(key, value) for key, value in query if key != "model"] + [("model", model)]
    return urlunsplit((scheme, parts.netloc, path, urlencode(query), ""))


def tls_context(url: str, insecure: bool) -> ssl.SSLContext | None:
    """With --insecure on wss://, accept the server's self-signed certificate; otherwise verify as usual."""
    if insecure and urlsplit(url).scheme == "wss":
        return ssl._create_unverified_context()
    return None


def auth_headers(api_key: str | None) -> dict[str, str]:
    """The OS_API_KEY, when the server has one, as a bearer header on the handshake."""
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


# ── The events this client sends ──────────────────────────────────────────────


def session_update(*, voice: str | None = None, preset: str | None = None, interrupt: bool = True) -> dict:
    """A GA session.update: 24 kHz PCM16 both ways, server voice detection that replies after each turn
    (create_response), talking over a reply stops it unless ``interrupt`` is false, the voice (a Kokoro blend
    works), and the AI preset (the server's ``voice`` preset when left out)."""
    output: dict[str, Any] = {"format": dict(_PCM)}
    if voice:
        output["voice"] = voice
    session: dict[str, Any] = {
        "type": "realtime",
        "audio": {
            "input": {
                "format": dict(_PCM),
                "turn_detection": {"type": "server_vad", "create_response": True, "interrupt_response": interrupt},
            },
            "output": output,
        },
    }
    if preset:
        session["assistant"] = {"preset": preset}
    return {"type": "session.update", "session": session}


def audio_append(pcm: bytes) -> dict:
    """One microphone block (24 kHz mono PCM16) for the server's input buffer."""
    return {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode("ascii")}


# ── The events this client receives ───────────────────────────────────────────


def decode_audio_delta(event: dict) -> bytes | None:
    """The PCM16 of a reply's (or a phrase's) audio delta, or None for any other event."""
    if event.get("type") not in _AUDIO_DELTAS or not isinstance(event.get("delta"), str):
        return None
    try:
        return base64.b64decode(event["delta"], validate=True)
    except (binascii.Error, ValueError):
        return None


def describe(event: dict) -> tuple[str, str] | None:
    """(kind, text) for an event worth printing, or None. The kinds are the keys of the labels above."""
    kind = event.get("type")
    if kind == "conversation.item.input_audio_transcription.completed":
        text = str(event.get("transcript") or "").strip()
        return ("you", text) if text else None
    if kind == "conversation.item.input_audio_transcription.failed":
        return ("error", str((event.get("error") or {}).get("message") or "transcription failed"))
    if kind == "open_speech.assistant.filler":
        return ("phrase", str(event.get("text") or ""))
    if kind == "open_speech.assistant.thinking":
        return ("thinking", str(event.get("text") or ""))
    if kind in _REPLY_DELTAS:
        return ("reply", str(event.get("delta") or ""))
    if kind == "error":
        error = event.get("error") or {}
        message = str(error.get("message") or "error")
        return ("error", f"{message} ({error['code']})" if error.get("code") else message)
    if kind == "response.done":
        response = event.get("response") or {}
        details = response.get("status_details") or {}
        if response.get("status") == "cancelled":
            return ("cancelled", str(details.get("reason") or ""))
        if response.get("status") == "failed":
            return ("failed", str((details.get("error") or {}).get("code") or ""))
        if response.get("status") == "incomplete":
            # A streamed reply an error or the token limit stopped after some of it was spoken.
            return ("cut_off", str((details.get("error") or {}).get("code") or ""))
    return None


def session_problem(event: dict) -> str | None:
    """Why the server won't answer, from its session.updated, or None when the AI assistant replies."""
    if event.get("type") != "session.updated":
        return None
    session = event.get("session") or {}
    if session.get("assistant") is None:
        return ("The server's AI assistant is off, so nothing answers: set the AI server's URL in Settings > "
                "AI assistant or OS_ASSISTANT_URL (and keep TTS on).")
    turn = ((session.get("audio") or {}).get("input") or {}).get("turn_detection")
    if not turn:
        return "The server's voice detection is off, so it can't tell when you've finished speaking."
    if not turn.get("create_response"):
        return "The server won't reply after each turn (create_response is false)."
    return None


def style(kind: str, text: str, *, color: bool) -> str:
    """One printed line: a label, then the text; the thinking is dimmed on a color terminal. A cut-off reply's
    line is ``[cut off: CODE]``."""
    if kind == "cut_off":
        return f"[cut off: {text}]" if text else _LABELS["cut_off"]
    line = f"{_LABELS.get(kind, '[' + kind + ']')} {text}".rstrip()
    return f"{_DIM}{line}{_RESET}" if color and kind == "thinking" else line


class Terminal:
    """What the script prints, through ``write`` (text, newlines included): one line per event worth showing,
    except a reply, whose one [ai] line grows as its sentences arrive and ends with the reply. Any
    other line ends an [ai] line still open first."""

    def __init__(self, write: Callable[[str], None], *, color: bool = False) -> None:
        self._write = write
        self._color = color
        self._reply: str | None = None  # the response whose [ai] line is open

    def line(self, kind: str, text: str) -> None:
        self.end_reply()
        self._write(style(kind, text, color=self._color) + "\n")

    def reply(self, response_id: str, text: str) -> None:
        if not text:
            return
        if self._reply is not None and self._reply == response_id:
            self._write(text)
            return
        self.end_reply()
        self._write(f"{_LABELS['reply']} {text.lstrip()}")
        self._reply = response_id

    def end_reply(self) -> None:
        if self._reply is not None:
            self._write("\n")
            self._reply = None


class PcmBuffer:
    """The speaker's queue of PCM16 bytes. The audio thread reads it, padding an underrun with silence; a
    canceled reply clears it, so what was left of it isn't played."""

    def __init__(self) -> None:
        self._data = bytearray()
        self._lock = threading.Lock()

    def write(self, pcm: bytes) -> None:
        with self._lock:
            self._data.extend(pcm)

    def read(self, nbytes: int) -> bytes:
        with self._lock:
            chunk = bytes(self._data[:nbytes])
            del self._data[:nbytes]
        return chunk + b"\x00" * (nbytes - len(chunk))

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


def handle_event(event: dict, player: PcmBuffer, terminal: Terminal) -> None:
    """Play an audio delta, print what's worth printing (a reply's text on its one [ai] line), and flush the
    speaker when a reply is canceled."""
    pcm = decode_audio_delta(event)
    if pcm:
        player.write(pcm)
        return
    problem = session_problem(event)
    if problem:
        terminal.line("note", problem)
    if event.get("type") in _REPLY_ENDS:
        terminal.end_reply()
    shown = describe(event)
    if shown is None:
        return
    kind, text = shown
    if kind == "reply":
        terminal.reply(str(event.get("response_id") or ""), text)
        return
    if kind == "cancelled":
        player.clear()
    terminal.line(kind, text)


# ── Running it ────────────────────────────────────────────────────────────────


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


async def _send_microphone(ws, queue: asyncio.Queue[bytes]) -> None:
    while True:
        frame = await queue.get()
        await ws.send(json.dumps(audio_append(frame)))


async def _receive(ws, player: PcmBuffer, terminal: Terminal) -> None:
    async for message in ws:
        if isinstance(message, bytes):
            continue
        try:
            event = json.loads(message)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            handle_event(event, player, terminal)


async def chat(args: argparse.Namespace) -> None:
    sd = _load_sounddevice()
    if args.list_devices:
        print(sd.query_devices())
        return

    url = websocket_url(args.url, args.model)
    color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

    def write(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    terminal = Terminal(write, color=color)
    player = PcmBuffer()
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=args.queue_size)
    loop = asyncio.get_running_loop()
    blocksize = max(1, SAMPLE_RATE * args.block_ms // 1000)

    def on_microphone(indata, frames, time, status) -> None:
        del frames, time
        if status:
            loop.call_soon_threadsafe(terminal.line, "audio", str(status))
        loop.call_soon_threadsafe(_enqueue_frame, queue, bytes(indata))

    def on_speaker(outdata, frames, time, status) -> None:
        del frames, time, status
        outdata[:] = player.read(len(outdata))

    if args.interrupt:
        terminal.line("note", HEADPHONES_TIP)
    async with connect(url, additional_headers=auth_headers(args.api_key) or None,
                       ssl=tls_context(url, args.insecure)) as ws:
        terminal.line("connect", url)
        await ws.send(json.dumps(session_update(voice=args.voice, preset=args.preset, interrupt=args.interrupt)))
        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=blocksize, device=args.device, channels=1,
                               dtype="int16", callback=on_microphone), \
                sd.RawOutputStream(samplerate=SAMPLE_RATE, device=args.output_device, channels=1, dtype="int16",
                                   callback=on_speaker):
            terminal.line("talk", "speak, then pause; press Ctrl+C to stop")
            send_task = asyncio.create_task(_send_microphone(ws, queue))
            receive_task = asyncio.create_task(_receive(ws, player, terminal))
            done, pending = await asyncio.wait({send_task, receive_task}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                exc = task.exception()
                if exc is not None:
                    raise exc


def closed_message(error: ConnectionClosedError) -> str:
    """One line for a connection the server closed with an error, printed in place of a traceback: a refusal
    (4001, 1008 or 4004) with its reason and what to do, another close with its code and reason, or a drop."""
    close = error.rcvd
    if close is None:
        return "[closed] the connection to the server dropped, with no close code"
    said = f"close code {close.code}" + (f": {close.reason}" if close.reason else "")
    if close.code in _REFUSALS:
        return f"[refused] {said}. {_REFUSALS[close.code]}"
    return f"[closed] {said}"


def _device(value: str) -> int | str:
    """A sounddevice device: its number from --list-devices, or part of its name."""
    return int(value) if value.strip().isdigit() else value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Talk to Open Speech's AI assistant over /v1/realtime. " + HEADPHONES_TIP,
    )
    parser.add_argument("--url", default=DEFAULT_URL,
                        help=f"The server (https://HOST:8100) or its realtime socket (default: {DEFAULT_URL})")
    parser.add_argument("--preset", help="AI preset id (default: the server's voice preset)")
    parser.add_argument("--voice", help="Voice of the reply, such as af_heart or a Kokoro blend "
                                        "af_bella(2)+af_sky(1) (default: the server's)")
    parser.add_argument("--model", help="Voice model, sent as ?model= (default: the server's TTS_MODEL)")
    parser.add_argument("--no-interrupt", dest="interrupt", action="store_false",
                        help="Don't stop a reply when the microphone hears speech: for speakers without headphones")
    parser.add_argument("--api-key", default=os.environ.get("OS_API_KEY"),
                        help="Bearer API key (default: the OS_API_KEY variable of this shell)")
    parser.add_argument("--insecure", action="store_true",
                        help="Accept a self-signed certificate on wss:// (local testing)")
    parser.add_argument("--device", type=_device, help="Microphone: a number from --list-devices, or a name")
    parser.add_argument("--output-device", type=_device, help="Speaker: a number from --list-devices, or a name")
    parser.add_argument("--block-ms", type=int, default=100, help="Microphone block size in ms")
    parser.add_argument("--queue-size", type=int, default=20, help="Buffered microphone blocks before dropping")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    return parser.parse_args(argv)


def main() -> None:
    try:
        asyncio.run(chat(parse_args()))
    except KeyboardInterrupt:
        print("\n[talk] stopped", flush=True)
    except ConnectionClosedError as error:
        # Printed to stderr, with exit status 1.
        raise SystemExit(closed_message(error)) from None


if __name__ == "__main__":
    main()
