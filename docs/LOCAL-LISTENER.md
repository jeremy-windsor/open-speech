# Local Always-On Listener Smoke Test

This is a focused local-machine path for validating streaming STT, Silero VAD,
and a headless microphone client.

## Install

Open Speech supports Python 3.12 only. With [uv](https://docs.astral.sh/uv/) installed (for a CUDA install use
`--extra cuda --no-install-package onnxruntime` in place of `--extra cpu`, and `uv run --no-sync` in place of
`uv run --frozen`, as the README's [Installation (from source)](../README.md#installation-from-source) explains; keep
`--extra all`: uv's sync is exact and removes what the extras it is given don't need, so leaving it out uninstalls
Kokoro, Piper and Pocket-TTS):

```bash
uv sync --frozen --extra cpu --extra all --extra dev --extra examples
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

`uv run --env-file` expands a `$` outside single quotes (`$$` is no escape there), drops double quotes, and stops
reading `.env` at a value with a blank or a backslash, so single-quote such a value, as in
`OS_API_KEY='Xk9$mP2q'`. An unquoted apostrophe also stops it there (two are dropped), and single quotes can't
hold one: write it as `\'`, as in `OS_API_KEY=Xk9\'mP2q` (see the README's source install section).

If you keep Open Speech TLS enabled instead, use `https://` with `curl -k` in the prewarm command below.

## Prewarm the STT model

```bash
curl -sS -X POST http://localhost:8100/api/ps/Systran/faster-whisper-base
```

The first run can take a while because faster-whisper and Silero VAD model files
may need to download into the local cache. This call unloads any other loaded STT model
first. With the default `OS_PIN_DEFAULT_MODELS=true`, the default `STT_MODEL`
(`Systran/faster-whisper-base` for a source install) then stays loaded: idle (`OS_MODEL_TTL`)
and LRU eviction never unload it. If you set `OS_API_KEY`, add
`-H "Authorization: Bearer $OS_API_KEY"` to the command. Only the server reads `.env`
(through `--env-file`), so export `OS_API_KEY` in this shell too, or put the key itself in
the command.

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

`--api-key` defaults to the `OS_API_KEY` variable of the shell (the listener does not read
`.env`) and is sent as an `Authorization: Bearer` header on the WebSocket handshake.

To listen from another machine, address the server by IP address, by a dotless LAN name
(`lan-box`), by the server machine's own host name, or by a name under `.local`, `.lan`,
`.home.arpa`, or `.internal`. Any other dotted host name must be listed in
`OS_ALLOWED_HOSTS` on the server, or the handshake is refused with HTTP 400.

## Browser microphone caveat

Browser microphone APIs require a secure context. `localhost` is usually allowed
over HTTP, but remote hosts need HTTPS. With Open Speech self-signed HTTPS, open
`https://localhost:8100/web` first and accept the certificate before testing the
web UI microphone.

## Expected behavior

Pass:

- The listener prints `[session] begin <session id>`.
- Speaking into the mic emits `[vad] speech_start`, transcript lines
  (`[transcript:interim]` or `[transcript:final]`), then `[vad] speech_end`.
- Silence should not continuously produce transcripts.

Fail:

- Speech only produces `[session] begin` and `[session] end`.
- VAD never reports `speech_start` while speaking.
- An `[error]` line arrives before `[session] begin`: the stream did not start. (An `[error]`
  right after it about an unknown `language` query parameter is not fatal: the stream
  continues with auto-detect.)
- The listener prints `[connect]`, then stops with
  `[refused] close code 4001: Invalid or missing API key` and what to do: the server has
  `OS_API_KEY` set and the key is missing or wrong (pass it with `--api-key`, or set `OS_API_KEY`
  in this shell). A refused socket opens and is then closed with its code and reason (README,
  [Refused WebSockets](../README.md#refused-websockets)); before, the handshake failed.
  A `1008` close names its cause in the reason.
  `[refused] close code 1008: Invalid sample_rate` means the server doesn't take that rate: pass
  another `--sample-rate` (8000-192000). `Invalid endpointing`, `Invalid vad` or
  `Invalid interim_results` means that query value in `--url` can't be read: fix it (`vad` also
  has `--vad` and `--no-vad`). `Origin not allowed` means the client's Origin isn't listed in
  `OS_WS_ALLOWED_ORIGINS` (`listen.py` itself sends no Origin).
  `server rejected WebSocket connection: HTTP 429` means the address is locked out for a minute
  after too many wrong keys, or went over `OS_RATE_LIMIT`.
- The harness logs model download/load errors, or the client cannot open a local
  input device.
