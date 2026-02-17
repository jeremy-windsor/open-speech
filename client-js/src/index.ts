export type TranscriptionResult = { text: string; [k: string]: unknown };
export type TranscriptionEvent = { type: string; [k: string]: unknown };

type ClientOptions = {
  baseUrl?: string;
  apiKey?: string;
  secure?: boolean;
};

type RealtimeCallback = (event: any) => void;

function toWsUrl(baseUrl: string, path: string): string {
  if (baseUrl.startsWith("https://")) return `wss://${baseUrl.slice(8)}${path}`;
  if (baseUrl.startsWith("http://")) return `ws://${baseUrl.slice(7)}${path}`;
  return `${baseUrl}${path}`;
}

function f32ToPcm16(input: Float32Array): ArrayBuffer {
  const out = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    out[i] = s < 0 ? s * 32768 : s * 32767;
  }
  return out.buffer;
}

export class OpenSpeechClient {
  baseUrl: string;
  apiKey?: string;
  secure: boolean;

  constructor({ baseUrl = "http://localhost:8100", apiKey, secure = true }: ClientOptions = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.apiKey = apiKey;
    this.secure = secure;
  }

  private headers(contentType?: string): HeadersInit {
    const h: Record<string, string> = {};
    if (this.apiKey) h.Authorization = `Bearer ${this.apiKey}`;
    if (contentType) h["Content-Type"] = contentType;
    return h;
  }

  async transcribe(audio: Blob | ArrayBuffer, options: { model?: string; response_format?: string } = {}): Promise<TranscriptionResult> {
    const form = new FormData();
    const blob = audio instanceof Blob ? audio : new Blob([audio], { type: "audio/wav" });
    form.append("file", blob, "audio.wav");
    if (options.model) form.append("model", options.model);
    if (options.response_format) form.append("response_format", options.response_format);

    const r = await fetch(`${this.baseUrl}/v1/audio/transcriptions`, {
      method: "POST",
      headers: this.headers(),
      body: form,
    });
    if (!r.ok) throw new Error(`Transcribe failed (${r.status})`);
    return await r.json();
  }

  async speak(text: string, options: { voice?: string; model?: string; speed?: number; response_format?: string } = {}): Promise<Blob> {
    const r = await fetch(`${this.baseUrl}/v1/audio/speech`, {
      method: "POST",
      headers: this.headers("application/json"),
      body: JSON.stringify({
        model: options.model ?? "kokoro",
        input: text,
        voice: options.voice ?? "alloy",
        speed: options.speed ?? 1.0,
        response_format: options.response_format ?? "mp3",
      }),
    });
    if (!r.ok) throw new Error(`Speak failed (${r.status})`);
    return await r.blob();
  }

  async *streamTranscribe(mediaStream: MediaStream, reconnectAttempts = 2): AsyncIterableIterator<TranscriptionEvent> {
    const model = "";
    const url = `${toWsUrl(this.baseUrl, "/v1/audio/stream")}?sample_rate=16000&vad=true${model ? `&model=${encodeURIComponent(model)}` : ""}`;

    let attempts = 0;
    while (attempts <= reconnectAttempts) {
      const ws = new WebSocket(url);
      const events: TranscriptionEvent[] = [];
      let closed = false;
      let err: any = null;

      ws.onmessage = (e) => {
        try {
          events.push(JSON.parse(String(e.data)));
        } catch {
          events.push({ type: "error", message: "Invalid JSON from server" });
        }
      };
      ws.onerror = (e) => {
        err = e;
      };
      ws.onclose = () => {
        closed = true;
      };

      await new Promise<void>((resolve, reject) => {
        ws.onopen = () => resolve();
        ws.onerror = () => reject(new Error("WebSocket failed to open"));
      });

      const audioCtx = new AudioContext({ sampleRate: 16000 });
      const source = audioCtx.createMediaStreamSource(mediaStream);
      const analyser = audioCtx.createAnalyser();
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      source.connect(analyser);
      analyser.connect(processor);
      processor.connect(audioCtx.destination);
      processor.onaudioprocess = (ev) => {
        if (ws.readyState !== WebSocket.OPEN) return;
        const pcm = f32ToPcm16(ev.inputBuffer.getChannelData(0));
        ws.send(pcm);
      };

      while (ws.readyState === WebSocket.OPEN || !closed || events.length > 0) {
        while (events.length > 0) {
          const ev = events.shift()!;
          yield ev;
        }
        if (closed) break;
        await new Promise((r) => setTimeout(r, 20));
      }

      processor.disconnect();
      analyser.disconnect();
      source.disconnect();
      audioCtx.close();

      if (!err) return;
      attempts++;
      if (attempts > reconnectAttempts) throw new Error("streamTranscribe reconnect limit reached");
      await new Promise((r) => setTimeout(r, 250 * attempts));
    }
  }

  realtimeSession(): RealtimeSession {
    return new RealtimeSession(this);
  }
}

export class RealtimeSession {
  private client: OpenSpeechClient;
  private ws: WebSocket;
  private transcriptCbs: RealtimeCallback[] = [];
  private audioCbs: RealtimeCallback[] = [];
  private vadCbs: RealtimeCallback[] = [];

  constructor(client: OpenSpeechClient) {
    this.client = client;
    this.ws = new WebSocket(toWsUrl(client.baseUrl, "/v1/realtime"), ["realtime"]);
    this.ws.onmessage = (e) => this.dispatch(JSON.parse(String(e.data)));
  }

  private dispatch(event: any) {
    const t = event?.type || "";
    if (t.includes("transcription") || t === "conversation.item.created") this.transcriptCbs.forEach((cb) => cb(event));
    if (t.startsWith("response.audio")) this.audioCbs.forEach((cb) => cb(event));
    if (t.includes("speech_")) this.vadCbs.forEach((cb) => cb(event));
  }

  sendAudio(chunk: ArrayBuffer) {
    const bytes = new Uint8Array(chunk);
    const audio = typeof Buffer !== "undefined"
      ? Buffer.from(bytes).toString("base64")
      : btoa(String.fromCharCode(...bytes));
    this.ws.send(JSON.stringify({ type: "input_audio_buffer.append", audio }));
  }

  commit() {
    this.ws.send(JSON.stringify({ type: "input_audio_buffer.commit" }));
  }

  createResponse(text: string, voice = "alloy") {
    this.ws.send(JSON.stringify({ type: "response.create", response: { instructions: text, voice } }));
  }

  onTranscript(cb: RealtimeCallback) { this.transcriptCbs.push(cb); }
  onAudio(cb: RealtimeCallback) { this.audioCbs.push(cb); }
  onVad(cb: RealtimeCallback) { this.vadCbs.push(cb); }
  close() { this.ws.close(); }
}
