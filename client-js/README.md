# @open-speech/client

Minimal TypeScript client for Open Speech.

## Install (local for now)

```bash
npm install ../client-js
```

## Usage

```ts
import { OpenSpeechClient } from "@open-speech/client";

const client = new OpenSpeechClient({ baseUrl: "http://localhost:8100", apiKey: "" });

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
not add query-string credentials. Use those helpers only on the same trusted-LAN, no-API-key deployment
as the web UI. A non-browser client that can set WebSocket headers may use Bearer auth when `OS_API_KEY`
is set.

### Streaming transcription

```ts
const media = await navigator.mediaDevices.getUserMedia({ audio: true });
for await (const event of client.streamTranscribe(media)) {
  console.log(event);
}
```
