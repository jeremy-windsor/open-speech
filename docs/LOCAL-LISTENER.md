# Local Always-On Listener Smoke Test

This is a focused local-machine path for validating streaming STT, Silero VAD,
and a headless microphone client.

## Install

```bash
python3 -m pip install -e ".[dev,examples]"
```

If `sounddevice` cannot load, install PortAudio for your OS as well. Examples:
`apt install libportaudio2`, `brew install portaudio`, or the equivalent package
from your platform package manager.

## Start the server

The easiest local path is plain HTTP/WebSocket:

```bash
OS_SSL_ENABLED=false python3 -m src.main
```

That matches the listener default, `ws://localhost:8100/v1/audio/stream`.

If you keep Open Speech TLS enabled, use `wss://` and either trust the generated
self-signed cert in your OS/browser trust store or pass `--insecure` for explicit
headless local testing. TLS verification is not disabled by default.

## Prewarm the STT model

```bash
curl -sS -X POST http://localhost:8100/api/ps/Systran/faster-whisper-base
```

The first run can take a while because faster-whisper and Silero VAD model files
may need to download into the local cache.

## Run the headless listener

```bash
python3 examples/listen.py
```

Useful variants:

```bash
python3 examples/listen.py --model Systran/faster-whisper-base
python3 examples/listen.py --no-vad
python3 examples/listen.py --api-key "$OS_API_KEY"
python3 examples/listen.py --url "wss://localhost:8100/v1/audio/stream?sample_rate=16000" --insecure
python3 examples/listen.py --list-devices
python3 examples/listen.py --device 2
```

## Browser microphone caveat

Browser microphone APIs require a secure context. `localhost` is usually allowed
over HTTP, but remote hosts need HTTPS. With Open Speech self-signed HTTPS, open
`https://localhost:8100/web` first and accept the certificate before testing the
web UI microphone.

## Expected behavior

Pass:

- The listener prints `session.begin`.
- Speaking into the mic emits `[vad] speech_start`, transcript lines, then
  `[vad] speech_end`.
- Silence should not continuously produce transcripts.

Fail:

- Speech only produces `session.begin` and `session.end`.
- VAD never reports `speech_start` while speaking.
- The server logs model download/load errors, or the client cannot open a local
  input device.
