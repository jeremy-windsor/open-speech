# Local Always-On Listener Smoke Test

This is a focused local-machine path for validating streaming STT, Silero VAD,
and a headless microphone client.

## Install

Open Speech supports Python 3.12 only. With [uv](https://docs.astral.sh/uv/) installed:

```bash
uv sync --frozen --extra cpu --extra dev --extra examples
```

If `sounddevice` cannot load, install PortAudio for your OS as well. Examples:
`apt install libportaudio2`, `brew install portaudio`, or the equivalent package
from your platform package manager.

## Start the harness

The easiest local path is plain HTTP/WebSocket. The default data and cert paths
(`/home/openspeech/...`, `/var/lib/open-speech/...`) are meant for Docker, so create
the `.env` file shown in the README's
[Installation (from source)](../README.md#installation-from-source) section
(it includes `OS_SSL_ENABLED=false` and local data paths), then run:

```bash
uv run --frozen --env-file .env python -m src.main
```

If you keep Open Speech TLS enabled instead, use `https://` with `curl -k` in the prewarm command below.

## Prewarm the STT model

```bash
curl -sS -X POST http://localhost:8100/api/ps/Systran/faster-whisper-base
```

The first run can take a while because faster-whisper and Silero VAD model files
may need to download into the local cache.

## Run the headless listener

The listener defaults to `wss://localhost:8100/v1/audio/stream?sample_rate=16000`.
With TLS off (the setup above), pass the plain `ws://` URL:

```bash
uv run --frozen python examples/listen.py --url "ws://localhost:8100/v1/audio/stream?sample_rate=16000"
```

With TLS on, keep the default `wss://` URL and either trust the generated self-signed
cert in your OS/browser trust store or pass `--insecure` for explicit headless local
testing. TLS verification is not disabled by default:

```bash
uv run --frozen python examples/listen.py --insecure
```

Useful variants (add the `--url` or `--insecure` option that matches your setup):

```bash
uv run --frozen python examples/listen.py --model Systran/faster-whisper-base
uv run --frozen python examples/listen.py --no-vad
uv run --frozen python examples/listen.py --api-key "$OS_API_KEY"
uv run --frozen python examples/listen.py --list-devices
uv run --frozen python examples/listen.py --device 2
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
- The harness logs model download/load errors, or the client cannot open a local
  input device.
