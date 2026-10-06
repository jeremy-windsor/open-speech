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

The `apiKey` option is applied to HTTP calls only. Browser WebSocket APIs cannot attach an
`Authorization` header, and this client's realtime, Live Speech, and streaming-transcription helpers do
not add credentials to the WebSocket URL, so they fail when the server sets `OS_API_KEY`. Use those
helpers on a trusted-LAN, no-API-key deployment, like the web UI. The server still accepts a deprecated
`?api_key=KEY` query parameter on WebSocket URLs for clients that cannot send headers; it logs a warning
and can expose the key in logs, so prefer a non-browser client that sends `Authorization: Bearer KEY`.

### Streaming transcription

```ts
const media = await navigator.mediaDevices.getUserMedia({ audio: true });
for await (const event of client.streamTranscribe(media)) {
  console.log(event);
}
```
