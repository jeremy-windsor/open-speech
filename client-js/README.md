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

// Send PCM16 audio chunks (24kHz)
rt.sendAudio(pcmChunkArrayBuffer);
rt.commit();
rt.createResponse("Say this back", "alloy");
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

Browser WebSocket APIs cannot attach an `Authorization` header. Use this helper on the same trusted-LAN,
no-API-key deployment as the web UI; non-browser clients can use the bearer header when `OS_API_KEY` is set.

### Streaming transcription

```ts
const media = await navigator.mediaDevices.getUserMedia({ audio: true });
for await (const event of client.streamTranscribe(media)) {
  console.log(event);
}
```
