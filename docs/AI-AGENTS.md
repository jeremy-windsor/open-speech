# Connect an AI Agent: Open Speech as Its Ears and Mouth

Your agent, a chat app, a voice bot or a home automation, does the thinking. Open Speech writes down what
people say and speaks what the agent answers. This guide gives one section per job, each with the smallest
request that works. The README's [API Reference](../README.md#api-reference) has every field and error.

None of this makes Open Speech call an AI: leave its AI assistant off (the default, while no AI server URL
is set). To have Open Speech send speech or text to your AI server itself and speak the reply, see
[AI-ASSISTANT.md](AI-ASSISTANT.md).

| Job | Endpoint | Section |
|---|---|---|
| Transcribe a recording or a file | `POST /v1/audio/transcriptions` | [1](#1-transcribe-a-file) |
| Turn a live microphone into text | `WS /v1/audio/stream` | [2](#2-stream-the-microphone-to-text) |
| Speak a whole reply | `POST /v1/audio/speech` | [3](#3-speak-a-reply) |
| Speak a reply while the agent is still writing it | `WS /v1/audio/speech/stream` | [4](#4-speak-a-streamed-reply-token-by-token) |
| Listen and speak on one connection, with barge-in | `WS /v1/realtime` | [5](#5-full-duplex-with-v1realtime) |
| Be Home Assistant's speech to text and text to speech | Wyoming, TCP `10400` | [6](#6-wyoming-home-assistant) |

A chat app whose replies arrive token by token usually pairs 2 (or 1) with 4. A voice bot that should be
interruptible fits 5.

## Before you start

- **The address.** `https://HOST:8100`, and `wss://HOST:8100` for WebSockets. Open Speech serves HTTPS with
  a self-signed certificate by default: pass `-k` to curl, `--insecure` to the examples, or
  `verify=False` to httpx, or turn HTTPS off on a trusted network (`OS_SSL_ENABLED=false`, then `http://` and
  `ws://`). An IP address, `localhost` or a LAN name works as `HOST`; a public DNS name must be listed in
  `OS_ALLOWED_HOSTS` (README, [Host names](../README.md#host-names)).
- **The API key.** With `OS_API_KEY` set on the server, send `Authorization: Bearer KEY` on every request and
  every WebSocket handshake. A browser can't set headers on a WebSocket: it asks `POST /api/auth/ws-ticket`
  (with the key) for a one-time ticket and connects with `?ticket=TICKET` within 60 seconds.
- **OpenAI SDKs** work unchanged with the base URL `https://HOST:8100/v1`. `whisper-1` means the server's
  `STT_MODEL`, and `tts-1` its `TTS_MODEL` (Kokoro by default); OpenAI's voice names map to Kokoro voices
  (README, [Use it from an OpenAI app](../README.md#use-it-from-an-openai-app)).
- **The first request** for a model loads it, and downloads it the first time, which can take a while.
  `TTS_MODEL` loads at startup (`TTS_PRELOAD_DEFAULT`); `STT_MODEL` loads on its first request.
- **Audio formats.** Speech comes back as signed 16-bit little-endian mono PCM at 24 kHz (`pcm`), or encoded.
  Microphone audio goes in as the same PCM16, at the rate you say.

## 1. Transcribe a file

One recording in, its text out: OpenAI's transcription API.

```bash
curl -sk https://HOST:8100/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "response_format=json"
```

The answer is `{"text": "..."}`. Add `-F "language=en"` to skip language detection, and
`-F "model=deepdml/faster-whisper-large-v3-turbo-ct2"` for a model other than `STT_MODEL`. With the OpenAI
Python SDK:

```python
import httpx
from openai import OpenAI

client = OpenAI(base_url="https://HOST:8100/v1", api_key="not-needed",  # your OS_API_KEY if one is set
                http_client=httpx.Client(verify=False))                # accept the self-signed certificate
with open("audio.wav", "rb") as f:
    print(client.audio.transcriptions.create(model="whisper-1", file=f).text)
```

A file may be up to `OS_MAX_UPLOAD_MB` (100 MB) and 4 hours long. `response_format` also takes `text`, `srt`,
`vtt` and `verbose_json`. `POST /v1/audio/translations` gives English text instead, and
`POST /v1/audio/transcriptions/batch` queues up to 20 files as a job.

Only some models translate. Large-v3-turbo (the Compose `STT_MODEL`) and English-only models (`*.en`, distil) answer
`400` with code `translation_unsupported`; send `model=Systran/faster-whisper-medium` (or `-large-v3`), or have the
server's administrator set `STT_TRANSLATE_MODEL=Systran/faster-whisper-medium` so that `whisper-1` translates.

## 2. Stream the microphone to text

For a live microphone: send audio as it is recorded, and get the words back while the person speaks.

1. Connect to `wss://HOST:8100/v1/audio/stream?sample_rate=16000` (any rate from 8,000 to 192,000 Hz).
2. Send the audio as **binary** frames of PCM16 little-endian mono at that rate; 100 ms per frame is typical.
3. Read the JSON events. The one to act on is a `transcript` with `speech_final: true`: its `text` is the
   whole utterance, once the speaker paused.
4. Send `{"type": "stop"}` to end. The server finishes the utterance and answers `session.end`.

```javascript
const ws = new WebSocket("wss://HOST:8100/v1/audio/stream?vad=true");
ws.onmessage = (e) => console.log(JSON.parse(e.data));
ws.onopen = () => ws.send(audioChunkArrayBuffer);
```

From a terminal, with a microphone (`uv sync --frozen --extra cpu --extra all --extra dev --extra examples`,
`--extra cuda` in place of `--extra cpu` for a CUDA install, and PortAudio; see [LOCAL-LISTENER.md](LOCAL-LISTENER.md)):

```bash
uv run --frozen python examples/listen.py --url "wss://HOST:8100/v1/audio/stream?sample_rate=16000" --insecure
```

What comes back, for one utterance:

```json
{"type": "session.begin", "session_id": "...", "model": "Systran/faster-whisper-base", "sample_rate": 16000, "internal_sample_rate": 16000, "vad_enabled": true, "language": null}
{"type": "vad", "state": "speech_start"}
{"type": "transcript", "is_final": false, "speech_final": false, "text": "what's the weather", "start": 0.42, "end": 1.3, "confidence": 0.9}
{"type": "transcript", "is_final": true, "speech_final": true, "text": "What's the weather tomorrow?", "start": 0.42, "end": 2.1, "confidence": 0.95}
{"type": "vad", "state": "speech_end"}
```

- `is_final: false` is interim text that can still change; show it, don't act on it. `is_final: true` with
  `speech_final: false` holds the words confirmed so far.
- An utterance ends after `OS_STREAM_ENDPOINTING_MS` (300 ms) of silence, or `?endpointing=MS` for this
  stream, and after 30 seconds at most. A longer pause suits a person who stops to think.
- `?language=en` fixes the language; `?model=` picks another model.
- If nothing comes back, the server's log line for the session's end says whether audio arrived and how it
  scored (README, [Nothing transcribed?](../README.md#nothing-transcribed)).

## 3. Speak a reply

A whole reply in, its audio out: OpenAI's speech API.

```bash
curl -sk https://HOST:8100/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro","input":"Hello world","voice":"af_heart"}' \
  -o output.mp3
```

For the first sound as soon as possible, stream raw PCM: add `?stream=true` and `"response_format": "pcm"`,
and play the 24 kHz PCM16 as it arrives.

```bash
curl -sk "https://HOST:8100/v1/audio/speech?stream=true" \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro","input":"Hello world","voice":"af_heart","response_format":"pcm"}' \
  -o output.pcm
```

With the OpenAI Python SDK (the client from section 1):

```python
with client.audio.speech.with_streaming_response.create(model="tts-1", input="Hello world",
                                                          voice="alloy") as response:
    response.stream_to_file("output.mp3")
```

- `input` is up to `TTS_MAX_INPUT_LENGTH` (4,096) characters; longer gets `400`. A longer reply, or one still
  being written, goes to section 4.
- `GET /v1/audio/voices?model=kokoro` lists the voices. Kokoro blends work in `voice`:
  `af_bella(2)+af_sky(1)`.
- `speed` (0.25 to 4) and `response_format` (`mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`, `m4a`) are optional.

## 4. Speak a streamed reply token by token

The Live Reader speaks text as it arrives, so the first sentence plays while the agent is still writing the
rest. Markdown in the reply is cleaned for speech: links are read by their label, and a code block becomes
"Code block skipped."

1. Connect to `wss://HOST:8100/v1/audio/speech/stream`. The server sends `session.created`.
2. Send the voice and the latency mode, and wait for `session.updated` (it reports the model's
   `sample_rate`):

   ```json
   {"type": "session.update", "session": {"model": "kokoro", "voice": "af_heart", "latency_mode": "natural"}}
   ```

3. Send each token or chunk of the reply as it comes:

   ```json
   {"type": "input_text.append", "text": "Tomorrow will be "}
   {"type": "input_text.append", "text": "sunny, with a light breeze.\n"}
   ```

4. At the end of the reply, `{"type": "input_text.commit"}` releases what is still held back.
5. Play each `response.output_audio.delta` (base64 PCM16 mono at its `sample_rate`, in `delta`), and after
   playing it send `{"type": "playback.ack", "response_id": "...", "sequence": N}` with its `response_id`
   and `sequence`. The server stops sending once about 15 seconds are unacknowledged
   (`TTS_LIVE_MAX_UNACKED_SECONDS`), and closes a session that stops acknowledging.
6. `input_text.done` means everything up to the last commit has been synthesized and sent.

From a terminal, text from stdin to a WAV file:

```bash
printf 'This text is streamed to the remote voice.\n' | \
  python examples/live_tts_client.py --url https://HOST:8100 --insecure
```

**Latency modes** (`latency_mode`):

| Mode | Speaks | Use |
|---|---|---|
| `natural` (default) | Whole sentences and paragraphs; prose with no sentence end yet goes after 1.5 s without new text (`TTS_LIVE_SENTENCE_IDLE_MS`) | The best-sounding reading |
| `responsive` | Also clauses, at commas, colons and semicolons, after 250 ms without new text (`TTS_LIVE_SEGMENT_IDLE_MS`) | An AI reply, sooner |
| `instant_word` | Each word as soon as it is complete | The lowest latency; choppier |

- **Interrupting.** When the user talks over the reply, send `{"type": "response.cancel"}` and drop the audio
  you hold. The server answers `response.cancelled`; then send the next reply.
- **Between replies**, end each with a line break, as above. A commit doesn't start a new line, so a next
  reply that starts with `1. ` would otherwise have its "1." spoken as a word.
- **One session at a time** by default (`TTS_LIVE_MAX_CONNECTIONS=1`): a second gets an `error` with code
  `session_busy` and is closed with `1013`. A session with no client event for `TTS_LIVE_IDLE_TIMEOUT_S`
  (300 s) is closed; `{"type": "session.keepalive"}` keeps it open.

The protocol, with pausing and `input_text.context`, is in the README's
[Live Reader](../README.md#live-reader-ws-v1audiospeechstream).

## 5. Full duplex with /v1/realtime

One WebSocket in OpenAI's Realtime protocol: the microphone streams in, server voice detection ends each
turn and transcribes it, your agent decides the answer, and Open Speech speaks it. Talking over the answer
stops it.

1. Connect to `wss://HOST:8100/v1/realtime` with no subprotocol (a GA session) and wait for
   `session.created`.
2. Set the formats, turn detection and voice. `create_response: false` says that your agent answers; it is
   what the session reports anyway while Open Speech's AI assistant is off.

   ```json
   {"type": "session.update", "session": {"type": "realtime",
     "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000},
                         "turn_detection": {"type": "server_vad", "create_response": false}},
               "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "af_heart"}}}}
   ```

3. Stream the microphone as base64 PCM16 mono at 24 kHz:

   ```json
   {"type": "input_audio_buffer.append", "audio": "<base64 PCM16>"}
   ```

4. When the person pauses (`silence_duration_ms`, 500 ms by default), you get
   `input_audio_buffer.speech_stopped`, then `conversation.item.input_audio_transcription.completed` with the
   words in `transcript`.
5. Your agent answers, and you send the whole answer to be spoken word for word:

   ```json
   {"type": "response.create", "response": {"instructions": "It will be sunny tomorrow."}}
   ```

6. Play `response.output_audio.delta` (base64 in `delta`) until `response.done`. The text comes in
   `response.output_audio_transcript.delta` and `.done`.

- **Barge-in.** When the person talks over the answer (`turn_detection.interrupt_response`, on by default),
  real words cancel it: `response.done` has `status: "cancelled"` and `status_details.reason:
  "turn_detected"`. Stop your playback at once; their new words arrive as the next transcript. "Mm-hmm" lets
  the answer go on. While barge-in is on, answer audio comes at playback speed, at most
  `OS_REALTIME_AUDIO_LEAD_MS` (1 s) ahead. Echo cancellation is up to your client (browsers do it); without
  it the server can hear its own voice and stop itself, so use headphones or `interrupt_response: false`.
- **Your own stop.** `{"type": "response.cancel"}` cancels the answer being spoken.
- **Errors** name the client event that caused them in `error.event_id`, when you gave it an `event_id`.
- **Idle.** A session with no event for `OS_REALTIME_IDLE_TIMEOUT_S` (120 s, counted from the end of the last
  answer) is closed with `4008`; a streaming microphone keeps it open.
- **Limits.** The answer text is up to `TTS_MAX_INPUT_LENGTH` (4,096) characters; there is no function calling
  and no text-only output. To speak a reply while it is still being written, use section 4 for the answer.

The current OpenAI Python SDK connects as a GA session, as above. `client-js`'s `RealtimeSession` offers the
`realtime` subprotocol, which keeps the older (beta) event names, such as `response.audio.delta`. Models,
voices, the legacy sessions and every limit are in the README's [Realtime Audio](../README.md#realtime-audio).

## 6. Wyoming (Home Assistant)

Open Speech speaks the [Wyoming protocol](https://github.com/rhasspy/wyoming) on TCP port `10400`, for Home
Assistant's voice pipelines: speech to text with `STT_MODEL` and text to speech with `TTS_MODEL`.

The Docker image and the base Compose file turn it on (publish port `10400` with `docker run`). A source
install needs:

```bash
OS_WYOMING_ENABLED=true
OS_WYOMING_HOST=0.0.0.0
```

In Home Assistant, add the Wyoming integration with Open Speech's address and port, or in YAML:

```yaml
wyoming:
  - host: "YOUR_OPEN_SPEECH_IP"
    port: 10400
```

The Wyoming port has **no authentication**: `OS_API_KEY`, the Host and Origin checks and rate limits don't
apply to it, so keep it on a network you trust. Speech comes back as 16 kHz PCM. Limits are in the README's
[Wyoming](../README.md#wyoming).

## Let Open Speech ask the AI instead

If your agent is an OpenAI-compatible chat server, Open Speech can call it for you, with waiting phrases
while it thinks, spoken errors and editable presets: `/v1/realtime` answers each turn itself, `POST
/api/assistant/ask` sends one text, transcript or document and returns the reply (spoken, if you ask), and the
web app's Talk view and Ask AI use them. Set the AI server's URL (for example `http://ai-server.lan:8080/v1`) in
Settings › AI assistant; see [AI-ASSISTANT.md](AI-ASSISTANT.md).
