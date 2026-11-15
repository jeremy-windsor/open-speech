"""examples/voice_chat.py: talk to the AI assistant over /v1/realtime.

What it must do (docs/AI-ASSISTANT.md, "Talking to it"): the URL, TLS and --insecure handling of
examples/listen.py; a GA session (no subprotocol) whose session.update sets 24 kHz PCM16 input, server VAD with create_response, the voice and an
optional --preset; --no-interrupt for speakers; the transcript, the phrases, the thinking (dimmed) and the reply
printed as they arrive; the speaker flushed when a reply is cancelled.

Only the pure parts run here. No microphone, speaker or network: sounddevice is never imported and no socket is
opened. The session.update body goes through the server's own parser (src.realtime.session), so the example sends
what /v1/realtime takes.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import ssl
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "voice_chat.py"


def _load():
    spec = importlib.util.spec_from_file_location("voice_chat_example", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def chat():
    return _load()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def test_it_imports_without_an_audio_library(monkeypatch):
    """sounddevice (the `examples` extra) is loaded only when it runs, so the pure parts work without it."""
    monkeypatch.setitem(sys.modules, "sounddevice", None)  # any import of it now raises ImportError
    module = _load()
    assert module.SAMPLE_RATE == 24000


# ── The URL, TLS and the API key (as examples/listen.py) ──────────────────────


@pytest.mark.parametrize(
    ("given", "model", "expected"),
    [
        ("https://speech.lan:8100", None, "wss://speech.lan:8100/v1/realtime"),
        ("http://localhost:8100/", None, "ws://localhost:8100/v1/realtime"),
        ("wss://localhost:8100/v1/realtime", None, "wss://localhost:8100/v1/realtime"),
        ("wss://localhost:8100", "gpt-realtime", "wss://localhost:8100/v1/realtime?model=gpt-realtime"),
        ("ws://192.0.2.10:8100/v1/realtime?ticket=abc", "kokoro",
         "ws://192.0.2.10:8100/v1/realtime?ticket=abc&model=kokoro"),
    ],
)
def test_the_url_is_the_realtime_socket_of_the_server_given(chat, given, model, expected):
    assert chat.websocket_url(given, model) == expected


def test_the_default_url_is_the_local_realtime_socket(chat):
    assert chat.DEFAULT_URL == "wss://localhost:8100/v1/realtime"
    assert chat.websocket_url(chat.DEFAULT_URL) == chat.DEFAULT_URL


@pytest.mark.parametrize("bad", ["speech.lan:8100", "ftp://speech.lan/v1/realtime", "wss://"])
def test_a_url_that_is_not_a_web_address_is_refused(chat, bad):
    with pytest.raises(ValueError):
        chat.websocket_url(bad)


def test_insecure_accepts_the_self_signed_certificate_only_for_wss(chat):
    context = chat.tls_context("wss://localhost:8100/v1/realtime", True)
    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_NONE and context.check_hostname is False
    assert chat.tls_context("wss://localhost:8100/v1/realtime", False) is None  # verified by default
    assert chat.tls_context("ws://localhost:8100/v1/realtime", True) is None  # no TLS at all


def test_the_api_key_goes_in_a_bearer_header(chat):
    assert chat.auth_headers("my-secret-key") == {"Authorization": "Bearer my-secret-key"}
    assert chat.auth_headers("") == {} and chat.auth_headers(None) == {}


# ── session.update ────────────────────────────────────────────────────────────


def test_session_update_asks_for_24_khz_pcm_server_vad_and_a_reply_after_each_turn(chat):
    body = json.loads(json.dumps(chat.session_update(voice="af_bella(2)+af_sky(1)", preset="voice")))
    assert body["type"] == "session.update"
    session = body["session"]
    assert session["type"] == "realtime"
    assert session["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert session["audio"]["input"]["turn_detection"] == {
        "type": "server_vad", "create_response": True, "interrupt_response": True,
    }
    assert session["audio"]["output"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert session["audio"]["output"]["voice"] == "af_bella(2)+af_sky(1)"
    assert session["assistant"] == {"preset": "voice"}


def test_session_update_leaves_out_the_voice_and_preset_it_was_not_given(chat):
    session = chat.session_update()["session"]
    assert "voice" not in session["audio"]["output"]
    assert "assistant" not in session  # the server's voice preset
    assert session["audio"]["input"]["turn_detection"]["interrupt_response"] is True


def test_no_interrupt_turns_off_talking_over_the_reply(chat):
    turn = chat.session_update(interrupt=False)["session"]["audio"]["input"]["turn_detection"]
    assert turn["interrupt_response"] is False and turn["create_response"] is True


def test_the_server_takes_the_session_update_as_a_ga_session_with_the_assistant(chat):
    from src.realtime.session import SessionConfig

    config = SessionConfig(assistant_on=True)
    config.update_from(chat.session_update(voice="af_heart", preset="voice", interrupt=False))
    assert config.ga is True
    assert (config.input_audio_format, config.output_audio_format) == ("pcm16", "pcm16")
    assert config.voice == "af_heart"
    assert config.turn_detection.create_response is True
    assert config.turn_detection.interrupt_response is False
    assert config.assistant_dict() == {"preset": "voice"}


# ── Audio in and out ──────────────────────────────────────────────────────────


def test_mic_audio_goes_in_an_append_event(chat):
    pcm = bytes(range(16))
    assert chat.audio_append(pcm) == {"type": "input_audio_buffer.append", "audio": _b64(pcm)}


@pytest.mark.parametrize("event_type", ["response.output_audio.delta", "response.audio.delta"])
def test_an_audio_delta_decodes_to_its_pcm(chat, event_type):
    assert chat.decode_audio_delta({"type": event_type, "delta": _b64(b"\x01\x02\x03\x04")}) == b"\x01\x02\x03\x04"


@pytest.mark.parametrize(
    "event",
    [
        {"type": "response.output_audio_transcript.delta", "delta": "Hello."},
        {"type": "response.output_audio.delta", "delta": "not base64!"},
        {"type": "response.output_audio.delta"},
    ],
)
def test_anything_else_is_not_audio(chat, event):
    assert chat.decode_audio_delta(event) is None


def test_the_speaker_buffer_plays_in_order_pads_with_silence_and_flushes(chat):
    player = chat.PcmBuffer()
    player.write(b"\x01\x02")
    player.write(b"\x03\x04")
    assert len(player) == 4
    assert player.read(2) == b"\x01\x02"
    assert player.read(6) == b"\x03\x04\x00\x00\x00\x00"  # underrun: silence, not an error
    assert len(player) == 0
    player.write(b"\x05\x06\x07\x08")
    player.clear()
    assert len(player) == 0 and player.read(2) == b"\x00\x00"


# ── What is printed ───────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        ({"type": "conversation.item.input_audio_transcription.completed", "transcript": "What's the time?"},
         ("you", "What's the time?")),
        ({"type": "open_speech.assistant.filler", "text": "Hmm."}, ("phrase", "Hmm.")),
        ({"type": "open_speech.assistant.thinking", "text": "The user asks the time."},
         ("thinking", "The user asks the time.")),
        ({"type": "response.output_audio_transcript.delta", "delta": "It's noon."}, ("reply", "It's noon.")),
        ({"type": "response.audio_transcript.delta", "delta": "It's noon."}, ("reply", "It's noon.")),
        ({"type": "error", "error": {"type": "server_error", "code": "assistant_unreachable",
                                      "message": "The AI server can't be reached"}},
         ("error", "The AI server can't be reached (assistant_unreachable)")),
        ({"type": "response.done", "response": {"status": "cancelled",
                                                "status_details": {"reason": "turn_detected"}}},
         ("cancelled", "turn_detected")),
        ({"type": "response.done", "response": {"status": "failed",
                                                "status_details": {"error": {"code": "assistant_timeout"}}}},
         ("failed", "assistant_timeout")),
    ],
)
def test_each_event_to_show_is_described(chat, event, expected):
    assert chat.describe(event) == expected


@pytest.mark.parametrize(
    "event",
    [
        {"type": "conversation.item.input_audio_transcription.completed", "transcript": "  "},
        {"type": "response.done", "response": {"status": "completed"}},
        {"type": "response.output_audio.delta", "delta": _b64(b"\x00\x00")},
        {"type": "input_audio_buffer.speech_started", "audio_start_ms": 0},
    ],
)
def test_other_events_print_nothing(chat, event):
    assert chat.describe(event) is None


def test_the_thinking_is_dimmed_only_on_a_colour_terminal(chat):
    dimmed = chat.style("thinking", "Let me see.", color=True)
    assert dimmed.startswith("\x1b[2m") and dimmed.endswith("\x1b[0m") and "Let me see." in dimmed
    plain = chat.style("thinking", "Let me see.", color=False)
    assert "\x1b" not in plain and "Let me see." in plain
    reply = chat.style("reply", "It's noon.", color=True)
    assert "\x1b[2m" not in reply and "It's noon." in reply


def test_a_session_without_the_assistant_says_so(chat):
    off = {"type": "session.updated", "session": {"assistant": None, "audio": {"input": {
        "turn_detection": {"type": "server_vad", "create_response": False}}}}}
    problem = chat.session_problem(off)
    assert problem and "AI assistant" in problem
    on = {"type": "session.updated", "session": {"assistant": {"preset": "voice"}, "audio": {"input": {
        "turn_detection": {"type": "server_vad", "create_response": True}}}}}
    assert chat.session_problem(on) is None
    no_vad = {"type": "session.updated", "session": {"assistant": {"preset": "voice"}, "audio": {"input": {
        "turn_detection": None}}}}
    assert "voice detection" in chat.session_problem(no_vad)
    assert chat.session_problem({"type": "session.created", "session": off["session"]}) is None


def _terminal(chat):
    """A Terminal that keeps what it writes; ``lines()`` is the screen so far, an open last line included."""
    written: list[str] = []
    terminal = chat.Terminal(written.append)
    return terminal, written, lambda: "".join(written).splitlines()


def test_a_conversation_prints_in_order_and_a_cancelled_reply_flushes_the_speaker(chat):
    terminal, _written, lines = _terminal(chat)
    player = chat.PcmBuffer()
    script = [
        {"type": "conversation.item.input_audio_transcription.completed", "transcript": "Tell me a story."},
        {"type": "open_speech.assistant.filler", "text": "Hmm."},
        {"type": "response.output_audio.delta", "delta": _b64(b"\x10\x00" * 4)},
        {"type": "open_speech.assistant.thinking", "text": "A short one."},
        {"type": "response.output_audio_transcript.delta", "delta": "Once upon a time."},
        {"type": "response.output_audio.delta", "delta": _b64(b"\x20\x00" * 4)},
    ]
    for event in script:
        chat.handle_event(event, player, terminal)
    assert len(player) == 16
    assert [next(word for word in ("Tell me", "Hmm.", "A short one.", "Once upon") if word in line)
            for line in lines()] == ["Tell me", "Hmm.", "A short one.", "Once upon"]

    chat.handle_event({"type": "response.done", "response": {"status": "cancelled",
                                                             "status_details": {"reason": "turn_detected"}}},
                      player, terminal)
    assert len(player) == 0  # what was left of the reply isn't played
    assert "turn_detected" in lines()[-1]
    assert lines()[-2] == "[ai] Once upon a time.", "the reply's line ended before the next one"


# With streaming a reply's text comes a
# sentence at a time, and an error or the token limit after some of it was spoken ends the response `incomplete`,
# with the code in status_details.error.code. The example prints one [ai] line per reply, the sentences joined as
# they arrive, and a [cut off: <code>] line for an incomplete one.


def _incomplete(code):
    return {"type": "response.done", "response": {
        "id": "resp_1", "status": "incomplete",
        "status_details": {"type": "incomplete", "error": {"type": "server_error", "code": code}}}}


def test_an_incomplete_reply_is_described_by_its_code(chat):
    assert chat.describe(_incomplete("assistant_reply_cut_off")) == ("cut_off", "assistant_reply_cut_off")
    assert chat.style("cut_off", "assistant_reply_cut_off", color=False) == "[cut off: assistant_reply_cut_off]"
    assert chat.style("cut_off", "", color=False) == "[cut off]"


def test_a_streamed_reply_prints_one_ai_line_that_grows_as_its_sentences_arrive_then_says_it_was_cut_off(chat):
    terminal, written, lines = _terminal(chat)
    player = chat.PcmBuffer()

    def send(event):
        chat.handle_event(event, player, terminal)

    def delta(text):
        return {"type": "response.output_audio_transcript.delta", "response_id": "resp_1", "delta": text}

    def audio():
        return {"type": "response.output_audio.delta", "response_id": "resp_1", "delta": _b64(b"\x10\x00" * 4)}

    send({"type": "conversation.item.input_audio_transcription.completed", "transcript": "Tell me about the Moon."})
    send({"type": "open_speech.assistant.filler", "response_id": "resp_1", "text": "Hmm."})
    send(delta("The Moon is about 384,000 km away."))
    send(audio())
    assert "".join(written).endswith("[ai] The Moon is about 384,000 km away."), "shown as it arrives, line open"
    send(delta(" It has no air."))
    send(audio())
    send({"type": "response.output_audio.done", "response_id": "resp_1"})
    send({"type": "response.output_audio_transcript.done", "response_id": "resp_1",
          "transcript": "The Moon is about 384,000 km away. It has no air."})
    send(_incomplete("assistant_reply_cut_off"))
    assert lines() == [
        "[you] Tell me about the Moon.",
        "[waiting] Hmm.",
        "[ai] The Moon is about 384,000 km away. It has no air.",
        "[cut off: assistant_reply_cut_off]",
    ]
    assert "".join(written).endswith("\n"), "every line ended"
    assert len(player) == 16, "what was said was played; nothing is flushed"

    # The next reply starts a line of its own, and a whole one prints no cut-off line.
    send({"type": "conversation.item.input_audio_transcription.completed", "transcript": "Thanks."})
    send({"type": "response.output_audio_transcript.delta", "response_id": "resp_2", "delta": "You're welcome."})
    send({"type": "response.done", "response": {"id": "resp_2", "status": "completed"}})
    assert lines()[-2:] == ["[you] Thanks.", "[ai] You're welcome."]
    assert "".join(written).endswith("\n")


# ── The command line ──────────────────────────────────────────────────────────


def test_the_command_line_defaults(chat, monkeypatch):
    monkeypatch.setenv("OS_API_KEY", "from-the-shell")
    args = chat.parse_args([])
    assert args.url == chat.DEFAULT_URL
    assert args.interrupt is True and args.insecure is False
    assert args.preset is None and args.voice is None and args.model is None
    assert args.api_key == "from-the-shell"


def test_the_command_line_options(chat, monkeypatch):
    monkeypatch.delenv("OS_API_KEY", raising=False)
    args = chat.parse_args(["--url", "https://speech.lan:8100", "--preset", "short-answers", "--voice", "af_heart",
                            "--no-interrupt", "--insecure", "--api-key", "k"])
    assert (args.url, args.preset, args.voice) == ("https://speech.lan:8100", "short-answers", "af_heart")
    assert args.interrupt is False and args.insecure is True and args.api_key == "k"


def test_the_help_gives_the_headphones_advice(chat, capsys):
    with pytest.raises(SystemExit):
        chat.parse_args(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "headphones" in text.lower() and "--no-interrupt" in text and "--preset" in text


# ── US spelling ──────────────────────────────
# US spelling covers everything a person reads: the script's printed lines, its
# docstrings and its comments. The API's status value `cancelled` (OpenAI's spelling) stays as the server sends it.


def test_a_canceled_reply_prints_a_us_spelled_label(chat):
    assert chat.describe({"type": "response.done", "response": {
        "status": "cancelled", "status_details": {"reason": "turn_detected"}}}) == ("cancelled", "turn_detected")
    assert chat.style("cancelled", "turn_detected", color=False) == "[canceled] turn_detected"


def test_the_example_reads_in_us_spelling():
    """Every string literal, docstring and comment of the script; a whole literal that is a protocol value (such as
    "cancelled") and a word joined into a name in code are allowed, as in tests/test_ui_spelling.py."""
    import io
    import tokenize

    from tests.test_ui_spelling import BRITISH, _joined

    found = []
    for token in tokenize.generate_tokens(io.StringIO(EXAMPLE.read_text(encoding="utf-8")).readline):
        if token.type not in (tokenize.STRING, tokenize.COMMENT):
            continue
        found += [f"line {token.start[0]}: {m.group(0)!r}" for m in BRITISH.finditer(token.string)
                  if not _joined(token.string, m)]
    assert found == [], "British spellings in examples/voice_chat.py:\n" + "\n".join(found)


# ── A refused socket ─────────────────────────────
# A refused /v1/realtime socket now opens, then closes at once with 4001, 1008 or 4004 and the reason;
# the handshake used to fail with HTTP 403. The script says so in one line, with the code and the reason and what to
# do, in place of a traceback.


def _closed_by_the_server(code: int, reason: str):
    from websockets.exceptions import ConnectionClosedError
    from websockets.frames import Close

    return ConnectionClosedError(Close(code, reason), Close(code, reason), True)


def _run_main(chat, monkeypatch, error) -> SystemExit:
    import argparse

    async def closed(args):
        print(f"[connect] {chat.DEFAULT_URL}", flush=True)
        raise error

    monkeypatch.setattr(chat, "parse_args", lambda argv=None: argparse.Namespace())
    monkeypatch.setattr(chat, "chat", closed)
    with pytest.raises(SystemExit) as stopped:
        chat.main()
    return stopped.value


@pytest.mark.parametrize(("code", "hint"), [
    (4001, "--api-key"),
    (1008, "OS_WS_ALLOWED_ORIGINS"),
    (4004, "OS_REALTIME_ENABLED"),
])
def test_a_refused_socket_ends_with_its_close_code_and_reason_not_a_traceback(chat, monkeypatch, code, hint):
    from src import middleware

    reason = {
        middleware.WS_CLOSE_KEY: middleware.WS_KEY_REFUSED,
        middleware.WS_CLOSE_ORIGIN: middleware.WS_ORIGIN_REFUSED,
        middleware.WS_CLOSE_FEATURE_OFF: "Realtime API is disabled",  # src/routes/realtime.py
    }[code]
    stopped = _run_main(chat, monkeypatch, _closed_by_the_server(code, reason))
    message = stopped.code
    assert isinstance(message, str), message  # SystemExit prints it to stderr and exits with status 1
    assert message.startswith("[refused]") and "\n" not in message
    assert f"close code {code}" in message and reason in message and hint in message


def test_another_close_gives_its_code_and_reason_and_a_drop_says_so(chat, monkeypatch):
    message = _run_main(chat, monkeypatch, _closed_by_the_server(4008, "Session idle timeout")).code
    assert message == "[closed] close code 4008: Session idle timeout"

    from websockets.exceptions import ConnectionClosedError

    message = _run_main(chat, monkeypatch, ConnectionClosedError(None, None, None)).code
    assert message.startswith("[closed]") and "no close code" in message


# ── examples/listen.py: a refused socket ─────
# /v1/audio/stream refuses the same way: it opens, then closes at once with 4001 or
# 1008 (a foreign Origin, or a sample_rate it can't take) and the reason. listen.py ended that in a traceback
# (ConnectionClosedError: received 4001 ...); it now prints one line, as voice_chat.py does.

LISTEN = ROOT / "examples" / "listen.py"


@pytest.fixture(scope="module")
def listen():
    spec = importlib.util.spec_from_file_location("listen_example", LISTEN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_listen_main(listen, monkeypatch, error) -> SystemExit:
    import argparse

    async def closed(args):
        print(f"[connect] {listen.DEFAULT_URL}", flush=True)
        raise error

    monkeypatch.setattr(listen, "parse_args", lambda: argparse.Namespace())
    monkeypatch.setattr(listen, "listen", closed)
    with pytest.raises(SystemExit) as stopped:
        listen.main()
    return stopped.value


def _stream_refusals():
    from src import middleware
    from src.streaming import MAX_SAMPLE_RATE, MIN_SAMPLE_RATE

    return [
        (middleware.WS_CLOSE_KEY, middleware.WS_KEY_REFUSED, "--api-key"),
        (middleware.WS_CLOSE_ORIGIN, middleware.WS_ORIGIN_REFUSED, "OS_WS_ALLOWED_ORIGINS"),
        (middleware.WS_CLOSE_ORIGIN, f"Invalid sample_rate: must be {MIN_SAMPLE_RATE}-{MAX_SAMPLE_RATE}",
         "--sample-rate"),
        (middleware.WS_CLOSE_FEATURE_OFF, "Live TTS is disabled", "turned off"),
    ]


@pytest.mark.parametrize("which", range(4), ids=["key", "origin", "sample_rate", "off"])
def test_listen_ends_a_refused_socket_with_one_line_not_a_traceback(listen, monkeypatch, which):
    code, reason, hint = _stream_refusals()[which]
    stopped = _run_listen_main(listen, monkeypatch, _closed_by_the_server(code, reason))
    message = stopped.code
    assert isinstance(message, str), message  # SystemExit prints it to stderr and exits with status 1
    assert message.startswith("[refused]") and "\n" not in message
    assert f"close code {code}: {reason}" in message and hint in message


def test_listen_gives_another_close_its_code_and_reason_and_says_when_it_dropped(listen, monkeypatch):
    message = _run_listen_main(listen, monkeypatch, _closed_by_the_server(1011, "Internal error")).code
    assert message == "[closed] close code 1011: Internal error"

    from websockets.exceptions import ConnectionClosedError

    message = _run_listen_main(listen, monkeypatch, ConnectionClosedError(None, None, None)).code
    assert message.startswith("[closed]") and "no close code" in message


def test_local_listener_guide_quotes_the_refused_lines_listen_prints(listen, monkeypatch):
    """docs/LOCAL-LISTENER.md told the reader a wrong key ends the listener with
    `ConnectionClosedError: received 4001 (private use) ...`, a traceback listen.py no longer prints. Each
    `[refused] ...` line the guide quotes must be (the start of) a line listen.py prints for a refusal of
    /v1/audio/stream, and the guide must quote the wrong-key one and the sample_rate one."""
    import re

    guide = " ".join((ROOT / "docs" / "LOCAL-LISTENER.md").read_text(encoding="utf-8").split())
    assert "ConnectionClosedError" not in guide, "docs/LOCAL-LISTENER.md still promises a ConnectionClosedError"
    # A lockout still fails the handshake, which listen.py doesn't catch: that sentence stays.
    assert "HTTP 429" in guide

    printed = {
        reason: _run_listen_main(listen, monkeypatch, _closed_by_the_server(code, reason)).code
        for code, reason, _hint in _stream_refusals()
    }
    quoted = re.findall(r"`(\[refused\][^`]*)`", guide)
    assert quoted, "docs/LOCAL-LISTENER.md quotes no [refused] line"
    unmatched = [line for line in quoted if not any(message.startswith(line) for message in printed.values())]
    assert not unmatched, f"docs/LOCAL-LISTENER.md quotes lines listen.py never prints: {unmatched}; it prints {printed}"

    from src import middleware

    key_line = printed[middleware.WS_KEY_REFUSED]
    assert any(key_line.startswith(line) and "close code 4001" in line for line in quoted), (
        f"docs/LOCAL-LISTENER.md doesn't quote the wrong-key line {key_line!r}"
    )
    sample_rate_line = next(message for reason, message in printed.items() if "sample_rate" in reason)
    assert any(sample_rate_line.startswith(line) and "Invalid sample_rate" in line for line in quoted), (
        f"docs/LOCAL-LISTENER.md doesn't quote the sample_rate line {sample_rate_line!r}"
    )
    assert "--sample-rate" in guide and "OS_WS_ALLOWED_ORIGINS" in guide


def _unreadable_query_reasons() -> list[str]:
    """The 1008 reasons /v1/audio/stream gives for a query value other than sample_rate that it can't read, taken
    from the route itself (src/routes/streaming.py `_query_values`)."""
    from src.routes.streaming import _query_values

    reasons = []
    for name, bad in (("interim_results", "maybe"), ("endpointing", "300ms"), ("vad", "maybe")):
        raw = {"sample_rate": None, "interim_results": None, "endpointing": None, "vad": None, name: bad}
        _values, reason = _query_values(**raw)
        assert reason is not None and reason.startswith(f"Invalid {name}"), (name, reason)
        reasons.append(reason)
    return reasons


@pytest.mark.parametrize("which", range(3), ids=["interim_results", "endpointing", "vad"])
def test_listen_points_an_unreadable_query_value_at_the_url_not_at_the_origin_list(listen, monkeypatch, which):
    """/v1/audio/stream also closes with 1008 for an unreadable `interim_results`,
    `endpointing` or `vad`, but listen.py gave every 1008 whose reason isn't about sample_rate the
    Origin hint. listen.py sends no Origin header, so that hint was never the cause: the line must point at the
    query value in --url that the reason names."""
    reason = _unreadable_query_reasons()[which]
    message = _run_listen_main(listen, monkeypatch, _closed_by_the_server(1008, reason)).code
    assert isinstance(message, str) and message.startswith("[refused]") and "\n" not in message
    assert f"close code 1008: {reason}" in message
    assert "OS_WS_ALLOWED_ORIGINS" not in message, f"listen.py blames the Origin for {reason!r}: {message}"
    assert "--url" in message and "--sample-rate" not in message, message


def test_local_listener_guide_names_each_cause_of_a_1008():
    """docs/LOCAL-LISTENER.md said "`1008` with another reason means the
    client's Origin isn't listed in `OS_WS_ALLOWED_ORIGINS`", but an unreadable endpointing, vad or interim_results
    closes 1008 too. The guide must say the Origin case is the `Origin not allowed` reason, and name the query ones."""
    guide = " ".join((ROOT / "docs" / "LOCAL-LISTENER.md").read_text(encoding="utf-8").split())
    assert "`1008` with another reason" not in guide, "the guide still says any other 1008 is the Origin"
    assert "Origin not allowed" in guide and "OS_WS_ALLOWED_ORIGINS" in guide
    for name in ("endpointing", "vad", "interim_results"):
        assert f"`{name}`" in guide or f"Invalid {name}" in guide, f"the guide doesn't name the {name} refusal"
