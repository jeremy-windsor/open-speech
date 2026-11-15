# Troubleshooting

- [The server log](#the-server-log)
- [`400 Host not allowed`](#400-host-not-allowed), [An app refuses the certificate](#an-app-refuses-the-certificate)
- [Nothing transcribed?](#nothing-transcribed)
- [A voice worker is offline or can't run](#a-voice-worker-is-offline-or-cant-run),
  [`503 insufficient_gpu_memory`](#503-insufficient_gpu_memory)
- [The AI assistant](#the-ai-assistant)

## The server log

When something doesn't work, set `OS_LOG_LEVEL=DEBUG` in `.env`, restart with the same `-f` files and profiles
that started the server, and watch `docker compose logs -f open-speech`. That restart is `docker compose up -d`
for the plain CPU launch only; for the `docker-compose.cpu.yml` launch it is
`docker compose -f docker-compose.cpu.yml up -d`, and on a GPU host it is
`docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d`, plus
`-f docker-compose.voice-models.yml --profile qwen3` (your workers' profiles) when you use them. A bare
`docker compose up -d` recreates any other launch from `docker-compose.yml` alone, unless `COMPOSE_FILE` in `.env`
names the files ([Docker Compose](CONFIGURATION.md#docker-compose)): a GPU server as the CPU image
without the voice workers, and a `docker-compose.cpu.yml` server with Wyoming on `0.0.0.0`, where it has no
authentication, the Turbo STT model and a two-model cap. DEBUG lines can
hold what was said (transcript text), so set it back to `INFO` afterwards. No level logs the API key or a WebSocket ticket from Open Speech's own code, and uvicorn's
request lines show the values of `api_key`, `ticket`, `exp` and `sig` in a query string as `***` at every
level. A reverse proxy's logs and browser history keep a `?api_key=`, so prefer the `Authorization`
header or a ticket to the deprecated `?api_key=`. At `DEBUG`, httpx, httpcore, urllib3, websockets and other
chatty libraries stay at `INFO`.

## `400 Host not allowed`

Open Speech answers requests addressed to an IP address, `localhost`, or a LAN name such as `lan-box` or
`lan-box.local`. Any other name, such as `speech.example.com`, gets `400 Host not allowed` until you list it in
`OS_ALLOWED_HOSTS` ([Host names](SECURITY.md#host-names)). Behind a reverse proxy, see
[Reverse proxies](SECURITY.md#reverse-proxies).

## An app refuses the certificate

Open Speech serves HTTPS with a self-signed certificate by default, which browsers ask you to accept once and most
other apps refuse. Turn HTTPS off on a trusted network, put a reverse proxy with a real certificate in front, or
give Open Speech a trusted certificate ([HTTPS](API.md#https), [TLS certificates](SECURITY.md#tls-certificates)).
The microphone in the web app needs HTTPS, or `localhost`.

## Nothing transcribed?

When a `/v1/audio/stream` or `/v1/realtime` session ends, the server logs one INFO line, such as
`Streaming session … ended (transcriptions=0, errors=0, audio=6.9 s, peak=silent, vad_max=0.00, speech=0)` or
`Realtime session … ended (audio=6.9 s, peak=-23.1 dBFS, vad_max=0.97, speech=2)`:

- `audio=0.0 s`: no audio arrived.
- `peak=silent`: the client sent only zeros (a muted or wrong microphone).
- A peak with `speech=0`: the VAD heard no speech. `vad_max` stayed below `STT_VAD_THRESHOLD`, or below
  `turn_detection.threshold` on `/v1/realtime`.
- `speech` above 0 with no transcript: the speech was transcribed to nothing.
- `vad_max=none`: the VAD scored no audio (it was off, or less than one 32 ms window arrived).

The line holds no audio and no text.

## A voice worker is offline or can't run

A worker whose profile isn't turned on reads "Worker offline" in Models, and its models can't be loaded; turn it on
as in [GPU voice workers](CONFIGURATION.md#gpu-voice-workers). A worker whose torch build has no code for the GPU
(for example an RTX 50-series card, `sm_120`) can never load a model: its `/health` reports `status: "failed"` with
`failure.code` `cuda_arch_unsupported` and a message naming the architecture, and Compose shows it unhealthy. Open
Speech then answers its models' loads with `503 provider_unavailable` and the worker's reason, and the Models view
reads "Worker can't run" (see [Voice provider deployment](TTS-BACKENDS.md#voice-provider-deployment)).

## `503 insufficient_gpu_memory`

The model doesn't fit in the GPU's free memory, even after the idle models that aren't pinned were unloaded. The
message says what to unload, and `detail.gpu` gives the figures in MiB ([GPU memory guard](MODELS.md#gpu-memory-guard)).
A worker that runs out of GPU memory while it speaks answers HTTP `507`, frees its CUDA cache and stays healthy; Open
Speech reports it as the same `503`.

## The AI assistant

[AI-ASSISTANT.md](AI-ASSISTANT.md#expected-behavior) lists what each spoken error means and what to
check. Settings › AI assistant › Connection's **Check** asks the AI server for its models without generating
anything, and **Test** makes one real call.

**Is the wait the AI or the voice?** Each voice turn's [timing line](AI-ASSISTANT.md#what-to-expect-for-speed) splits it:
a large `ai_first_answer_ms` is the AI thinking (lower the effort or the thinking budget), a large gap from
there to `first_audio_ms` is the first sentence being long or the voice model being slow, and `silence_ms`
plus `stt_ms` is Open Speech hearing you.

**The voice only starts when the AI has finished, though `streamed=1`.** Something between Open Speech and
the AI server holds the stream back and sends it all at the end: the timing line shows `ai_first_token_ms`
close to `ai_done_ms`. A reverse proxy such as nginx, or a front end such as Open WebUI, must pass server-sent
events through unbuffered; for nginx, `proxy_buffering off;` in the location that proxies `/v1/`. A server
that can't stream at all answers in one piece (`streamed=0`) and still works; with `OS_ASSISTANT_STREAM=false`
Open Speech doesn't ask it to.
