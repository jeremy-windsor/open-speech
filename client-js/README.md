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

### Streaming transcription

```ts
const media = await navigator.mediaDevices.getUserMedia({ audio: true });
for await (const event of client.streamTranscribe(media)) {
  console.log(event);
}
```
