# Cloning models

A cloning model has no built-in voices. It speaks in the voice of a short **reference recording** you give it,
from that recording alone: nothing is trained. Open Speech has four, each in an isolated GPU worker:
Chatterbox (`chatterbox/regular`, `chatterbox/turbo`), Qwen3 Base (`qwen3/0.6b-base`) and Fish Audio S2 (`fish/s2-pro`). Their workers list no voices for them
(`voices: []` in the worker's manifest).

The in-process models (Kokoro, Piper, Pocket-TTS) and `qwen3/0.6b-custom-voice` have voices of their own and
don't clone.

## The models

| Model | Languages | Reference recording | Transcript | Live Reader | Text per request |
|---|---|---|---|---|---|
| `chatterbox/regular` | `en` | Any length the library takes | Not used | Yes | 350 characters (`CHATTERBOX_MAX_INPUT_CHARS`) |
| `chatterbox/turbo` | `en` | Longer than 5 seconds | Not used | Yes | 350 characters (`CHATTERBOX_MAX_INPUT_CHARS`) |
| `qwen3/0.6b-base` | `zh`, `en`, `ja`, `ko`, `de`, `fr`, `ru`, `pt`, `es`, `it` | Any length the library takes | Required, word for word | No | 4,096 characters |
| `fish/s2-pro` | `en`, `zh`, `ja`, `ko`, `es`, `pt`, `ar`, `ru`, `fr`, `de` | Up to 30 seconds | Required, word for word | Yes | 1,500 characters (`FISH_MAX_INPUT_CHARS`) |

- **Languages** are the worker's own list. A `language` hint outside it gets `400` before the worker is asked. A
  regional tag counts as its language (`pt-BR` is `pt`). Qwen3 Base also takes its own names (`English`), and
  with no hint it detects the language (`Auto`).
- **The recording**: the voice library keeps a PCM WAV of up to `OS_VOICE_LIBRARY_MAX_SECONDS` (60 by default).
  Chatterbox Turbo refuses a reference of 5 seconds or less, and Fish Audio S2 one over 30 seconds
  (`reference_audio_too_short`, `reference_audio_too_long`, both `400`).
- **The transcript**: Qwen3 Base and Fish Audio S2 need the exact words spoken in the recording, and
  answer `400` without one. Chatterbox doesn't read it.
- **Fish Audio S2** runs S2 Pro quantized to 4 bits (NF4) so it fits in about 3.3 GB of GPU memory. Its first load
  fetches about 5 GB of weights and compiles for a few minutes; only English has been tried. It has a fixed speed
  and takes no `instructions` (both `400`).
- **Live Reader**: every model but Qwen3 Base can read through the Live Reader and Speak › Live.
- **Text per request**: the most a model takes at once. `/v1/audio/speech` refuses more with
  `400 Input too long for MODEL. Max: N characters`; the Reader and the Live Reader cut longer text into pieces.
  The limits for Chatterbox and Fish Audio S2 are the worker settings named, read when the worker starts.
- **Other controls**: none of them takes `instructions` (how to speak) or a `speed` other than 1.0 (`400`).
  Chatterbox Turbo reads sound tags such as `[laugh]` or `[clear throat]` in the text
  as sounds.

## Giving it a voice

Two ways, both built on a recording in the voice library.

### A recording

In the web app, **Voices › Clone a voice** records or uploads a clip, checks its quality, has you check the
transcript word for word, lets you test it and saves it as a recording (**Voices › Recordings**). In Speak,
choose the cloning model, then the recording in the voice picker, or in **Reference recording**.

Through the API, store the recording once, with its transcript:

```bash
curl -k -X POST https://localhost:8100/api/voices/library \
  -F 'name=narrator' \
  -F 'transcript=These are the exact words spoken in the reference clip.' \
  -F 'audio=@reference.wav;type=audio/wav'
```

Then name it in `voice_library_ref`. Its stored transcript goes with it as `clone_transcript`:

```bash
curl -k https://localhost:8100/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model": "fish/s2-pro", "voice": "reference", "voice_library_ref": "narrator",
       "input": "This is a cloning test.", "response_format": "wav"}' \
  -o clone.wav
```

`voice` names nothing for these models: any value will do. The Live Reader takes the same field in its
`session.update` (`{"model": "chatterbox/turbo", "voice": "reference", "voice_library_ref": "narrator"}`). A
request can also send the clip itself, as `reference_audio` (base64) with `clone_transcript`, or as a file to
`POST /v1/audio/speech/clone` (README, [`POST /v1/audio/speech/clone`](../README.md#post-v1audiospeechclone)).

### A named voice with a version for the model

A named voice is who speaks; its **version** for one exact model holds that model's voice, and for a cloning
model the recording ([Named voices and reading presets](VOICE-IDENTITIES.md)). In Speak, choose the cloning model
and the recording, then **Save named voice…**: a new name makes a named voice, and an existing one's name adds a
version for this model. Through the API, add the version with
`POST /api/voices/identities/{id}/realizations` and `{"model": "chatterbox/turbo", "voice": "reference",
"reference_audio_id": "narrator"}`.

The request then gives only `"voice": "voice:<id>"` with the model. The version brings the recording and its
transcript, and a request that sends its own reference or transcript with it gets `400`. A named voice without
a version for that model gets `400` too. A Kokoro version doesn't make a Chatterbox one: each model's version
is added on its own.

### Where a cloned voice speaks

- **Speak**, `/v1/audio/speech` and `/v1/audio/speech/clone`: a recording or a named voice.
- **The Live Reader** (`/v1/audio/speech/stream`): a recording (`voice_library_ref`) or a named voice; not on
  Qwen3 Base.
- **The Reader**, **Ask AI**'s spoken answer and **Studio** conversations: a named voice (a Studio preset may also
  hold a recording).
- **Talk** (`/v1/realtime`) can't speak in a cloned voice yet: a named voice whose version holds a recording is
  refused there.

## What the voice pickers show

For a cloning model, every voice picker (Speak's, the Reader's, Talk's and Ask AI's) starts with a note,
"*Model* has no built-in voices." It lists no voices of the model's own. After any suggestion and your favorites
come the named voices that have a version for the model, and last, under "Named voices with no *Model* version",
the ones that don't. Those can't be chosen; each says "No *Model* version".

Talk can't speak in a cloned voice yet, so in Talk's picker a named voice whose version for the model is a recording
shows but is disabled, with "Can't speak in Talk yet". If Speak uses such a voice, Talk's "Same as Speak" row is
disabled the same way and Talk answers in the server's default voice instead. A cloned voice that Talk remembers
from an earlier visit is set aside quietly, and Talk follows Speak again.

Speak's picker, the one that offers recordings, adds "It speaks in the voice of a recording." (and "No recordings
yet." when you have none) and a **Clone a voice…** link to Voices › Clone a voice, and lists your recordings before
the named voices.

## Starting the workers

The workers are optional and off by default. They need an NVIDIA GPU and the GPU Compose file, and each one has a
Compose profile of its own, named after it: `chatterbox`, `qwen3` and `fish`. Layering
`docker-compose.voice-models.yml` alone starts and builds none of them; name a worker's profile to turn it on, here
Qwen3:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  -f docker-compose.voice-models.yml --profile qwen3 up -d --build
```

Use `--profile chatterbox` or `--profile fish` for the others. For two, repeat the flag
(`--profile qwen3 --profile chatterbox`); `--profile "*"` turns on all three. `COMPOSE_PROFILES=qwen3,chatterbox`, in the shell or in
`.env`, does the same as the flags and also covers later commands, so `down` then stops those workers too. To stop
one worker and keep the rest, name its service: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml -f
docker-compose.voice-models.yml stop qwen3`. Only Qwen3 Custom Voice (`qwen3/0.6b-custom-voice`) has built-in
voices: the four models above clone from a recording, so to audition them, turn on one worker at a time, and stop
the one before (`stop qwen3`) when you turn on the next: a `--profile` flag replaces `COMPOSE_PROFILES` rather than
adding to it, and a worker container an earlier `up` started keeps running until it is stopped, since `up` leaves
alone a worker whose profile isn't named. A host that ran the file before the profiles has all three running; stop
them once before the first `up` that names a profile (this stops the core too, and that `up` starts it again):

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  -f docker-compose.voice-models.yml --profile "*" down
```

`docker-compose.voice-models.yml` adds the three workers (`qwen3`, `chatterbox` and `fish`, on the
private Compose network only) and sets the core's `TTS_EXTERNAL_PROVIDERS` to all three addresses, whichever are turned on.
Each address is a network alias only that worker holds (`qwen3.voice-workers.internal.` and so on), so a worker
that is off can't be stood in for by a device on the LAN with the same name; see
[Voice provider deployment](TTS-BACKENDS.md#voice-provider-deployment).
It also sets `QWEN3_ENABLE_BASE=true`, without which the Qwen3 worker hides `qwen3/0.6b-base`. A running worker
holds no model on the GPU: in the web app's **Models** view, **Download** fetches a model's files and **Load** loads
it (an explicit load unloads the other TTS models first), and **Test clone** in Voices › Recordings loads its model
by itself. A worker that isn't running (its profile isn't turned on, or it was stopped) reads "Worker offline"
there; the core serves without it. The workers' settings (`CHATTERBOX_*`,
`QWEN3_*`, `FISH_*`, `HF_TOKEN`) are in the README under
[Voice provider settings](../README.md#voice-provider-settings), and how they're built and what they do on a GPU
they can't use in [TTS Backends](TTS-BACKENDS.md#voice-provider-deployment).
