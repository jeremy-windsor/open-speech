# @open-speech/client

Minimal TypeScript client for Open Speech.

## Install (local for now)

```bash
npm install ../client-js
```

## Usage

```ts
import { OpenSpeechClient } from "@open-speech/client";

// baseUrl defaults to "https://localhost:8100". Use "http://..." if the server runs with OS_SSL_ENABLED=false.
const client = new OpenSpeechClient({ baseUrl: "https://localhost:8100", apiKey: "" });

const tx = await client.transcribe(await (await fetch("/sample.wav")).arrayBuffer());
console.log(tx.text);

const plainText = await client.transcribe(await (await fetch("/sample.wav")).arrayBuffer(), {
  response_format: "text",
});
console.log(plainText);

const speech = await client.speak("Hello from Open Speech", { voice: "af_heart", response_format: "mp3" });
const url = URL.createObjectURL(speech);
new Audio(url).play();
```

### Realtime session

```ts
const rt = client.realtimeSession();
rt.onTranscript((ev) => console.log("transcript", ev));
rt.onAudio((ev) => console.log("audio", ev));
rt.onVad((ev) => console.log("vad", ev));
await rt.ready;

// Send PCM16 audio chunks (24kHz)
await rt.sendAudio(pcmChunkArrayBuffer);
await rt.commit();
await rt.createResponse("Say this back", "alloy");
```

### Live incremental speech

```ts
const live = client.liveSpeechSession({
  model: "kokoro",
  voice: "af_heart",
  latency_mode: "natural",
});
live.onEvent(async (event) => {
  if (event.type === "response.output_audio.delta") {
    // Decode and play event.delta as PCM16LE mono at event.sample_rate.
    // Acknowledge only after local playback or consumption completes.
    await live.acknowledge(event.response_id, event.sequence);
  }
});
await live.ready;
await live.append("Text from a user, HTTP feed, or AI output stream. ");
await live.commit();
```

HTTP calls send `apiKey` as `Authorization: Bearer KEY`. Browser WebSocket APIs cannot attach that
header, so with `apiKey` set the realtime, Live Speech, and streaming-transcription helpers first
request a one-time ticket from `POST /api/auth/ws-ticket` and connect with `?ticket=...`; the key itself
never appears in a WebSocket URL. A ticket works once and expires after 60 seconds, so every
connection, reconnects included, requests a new one. With `apiKey` set, `RealtimeSession` and
`LiveSpeechSession` therefore open their socket only after the ticket arrives. When the ticket request
fails, `ready` rejects with `WebSocket ticket request failed (<status>)`, or with the network error.
Calling `close()` before the ticket arrives rejects `ready` and opens no socket. `streamTranscribe`
throws at once when the server refuses the ticket with a 4xx status (a wrong key gives 401, and an
address locked out after too many wrong keys gives 429), and retries network and server (5xx) errors
within its reconnect attempts. Without `apiKey`, the helpers connect directly.

By default only the server's own pages can call it from a browser; earlier versions allowed any
origin. A page served from another origin must be listed on the server in `OS_CORS_ORIGINS`
(HTTP calls, the ticket request included) and in `OS_WS_ALLOWED_ORIGINS` (WebSockets). Otherwise
the server sends that page no CORS headers and refuses its writes with 403, so `transcribe()`,
`speak()`, and the ticket request fail in the browser as network errors, and a WebSocket from that
page is opened and closed at once with `1008` ([Refused sockets](#refused-sockets)). A client that
sends no `Origin` header is not affected.

### Refused sockets

The server refuses a WebSocket by opening it and closing it at once, before any event, with a close
code and a reason (the server's README, "Refused WebSockets"):

| Close code | When |
|---|---|
| `4001` | `Invalid or missing API key`: no key, a wrong one, or a used or expired ticket |
| `1008` | The page's Origin isn't in `OS_WS_ALLOWED_ORIGINS`; on `/v1/audio/stream` also a query value the server can't take, which the reason names (`Invalid sample_rate: ...`, `Invalid endpointing: ...`, `Invalid vad: ...`) |
| `4004` | The feature is turned off: the Live Reader (`TTS_ENABLED`, `TTS_LIVE_ENABLED`) or the realtime API (`OS_REALTIME_ENABLED`) |

The SDK reports these three as `StreamRefusedError`, which it exports, with `code` and `reason`,
like the Python client's. It never reconnects after one, since the same socket gets the same answer:

- `streamTranscribe` throws it from the loop.
- `RealtimeSession`: the refused socket opens first, so `ready` resolves; the refusal then rejects
  `closed` and every later send. `ready` rejects with it only when the close comes before the open.
- `LiveSpeechSession`: `ready` rejects with it.

```ts
import { StreamRefusedError } from "@open-speech/client";

const rt = client.realtimeSession();
rt.closed.catch((error) => {
  if (error instanceof StreamRefusedError) console.error(error.code, error.reason);
});
```

`RealtimeSession.closed` resolves on any other close, or after `close()`, and rejects with
`ready`'s error when the socket never opened. Other closes keep their behavior: `streamTranscribe`
ends at a normal close and reconnects after a dropped one, within its reconnect attempts.

Servers before this change failed the handshake instead, which `streamTranscribe` reported as
`WebSocket failed to open` after its retries; a refusal from a current server used to end the loop
with no event and no error.

### Streaming transcription

```ts
const media = await navigator.mediaDevices.getUserMedia({ audio: true });
const stop = new AbortController();
stopButton.onclick = () => stop.abort();
try {
  for await (const event of client.streamTranscribe(media, { signal: stop.signal })) {
    console.log(event);
  }
} catch (error) {
  if (!stop.signal.aborted) throw error;
} finally {
  media.getTracks().forEach((track) => track.stop()); // turns the microphone off
}
```

The second argument is `{ reconnectAttempts, signal }` (or just the number of reconnect attempts;
the default is 2). Aborting the signal stops sending audio, closes the `AudioContext` and the
WebSocket at once, even while the loop waits for the next event, and the loop then throws the
signal's reason (an `AbortError` unless you passed `abort()` a reason of your own). An abort during
the ticket request or a reconnect delay opens no socket, and one during the connection attempt closes
the socket it was opening. Leaving the loop (`break`, `return`, or an exception in its body)
releases them too, but code outside the loop cannot end it while it waits, so use the signal for a
Stop button. `streamTranscribe` never stops the tracks of the `MediaStream` you pass in; stop them
yourself to turn the microphone off.
