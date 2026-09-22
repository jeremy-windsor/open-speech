export type TranscriptionResult = { text: string; [k: string]: unknown };
export type TranscriptionEvent = { type: string; [k: string]: unknown };
export type LiveSpeechEvent = { type: string; [k: string]: unknown };

export type TranscriptionOptions = {
  model?: string;
  response_format?: "json" | "verbose_json" | "text" | "srt" | "vtt";
};

export type LiveSpeechSessionOptions = {
  model?: string;
  voice?: string;
  speed?: number;
  language?: string;
  latency_mode?: "natural" | "responsive" | "instant_word";
};

type ClientOptions = {
  baseUrl?: string;
  apiKey?: string;
  secure?: boolean;
};

type RealtimeCallback = (event: any) => void;
type NodeBufferLike = {
  from(input: Uint8Array): { toString(encoding: "base64"): string };
};

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

  async transcribe(
    audio: Blob | ArrayBuffer,
    options: TranscriptionOptions & { response_format: "text" | "srt" | "vtt" },
  ): Promise<string>;
  async transcribe(
    audio: Blob | ArrayBuffer,
    options?: TranscriptionOptions & { response_format?: "json" | "verbose_json" },
  ): Promise<TranscriptionResult>;
  async transcribe(
    audio: Blob | ArrayBuffer,
    options: TranscriptionOptions,
  ): Promise<TranscriptionResult | string>;
  async transcribe(
    audio: Blob | ArrayBuffer,
    options: TranscriptionOptions = {},
  ): Promise<TranscriptionResult | string> {
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
    const responseFormat = options.response_format;
    const contentType = (r.headers.get("content-type") || "").toLowerCase();
    const requestedText = responseFormat === "text" || responseFormat === "srt" || responseFormat === "vtt";
    const receivedText = contentType !== "" && !contentType.includes("json");
    return requestedText || receivedText ? await r.text() : await r.json();
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
      ws.onclose = () => {
        closed = true;
      };

      try {
        await new Promise<void>((resolve, reject) => {
          ws.onopen = () => resolve();
          ws.onerror = (event) => {
            err = event;
            reject(new Error("WebSocket failed to open"));
          };
        });
      } catch (openError) {
        attempts++;
        if (attempts > reconnectAttempts) throw openError;
        await new Promise((resolve) => setTimeout(resolve, 250 * attempts));
        continue;
      }

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

  liveSpeechSession(options: LiveSpeechSessionOptions = {}): LiveSpeechSession {
    return new LiveSpeechSession(this, options);
  }
}

export class LiveSpeechSession {
  private ws: WebSocket;
  private callbacks: Array<(event: LiveSpeechEvent) => void> = [];
  private resolveReady!: () => void;
  private rejectReady!: (error: Error) => void;
  private configured = false;
  readonly ready: Promise<void>;

  constructor(client: OpenSpeechClient, options: LiveSpeechSessionOptions = {}) {
    this.ready = new Promise<void>((resolve, reject) => {
      this.resolveReady = resolve;
      this.rejectReady = reject;
    });
    void this.ready.catch(() => {});
    this.ws = new WebSocket(toWsUrl(client.baseUrl, "/v1/audio/speech/stream"));
    this.ws.onmessage = (message) => {
      let event: LiveSpeechEvent;
      try {
        event = JSON.parse(String(message.data));
      } catch {
        event = { type: "error", error: { code: "invalid_json", message: "Invalid JSON from server" } };
      }
      if (event.type === "session.created") {
        this.ws.send(JSON.stringify({ type: "session.update", session: options }));
      } else if (event.type === "session.updated") {
        this.configured = true;
        this.resolveReady();
      } else if (event.type === "error" && !this.configured) {
        const error = event.error as { message?: string } | undefined;
        this.rejectReady(new Error(error?.message || "Live speech session failed to open"));
        this.ws.close(1008, "Session configuration failed");
      }
      this.callbacks.forEach((callback) => callback(event));
    };
    this.ws.onerror = () => this.rejectReady(new Error("Live speech WebSocket failed"));
    this.ws.onclose = (event) => {
      if (!this.configured || event.code !== 1000) {
        this.rejectReady(new Error(event.reason || `Live speech WebSocket closed (${event.code})`));
      }
    };
  }

  private async send(event: Record<string, unknown>): Promise<void> {
    await this.ready;
    if (this.ws.readyState !== WebSocket.OPEN) throw new Error("Live speech session is closed");
    this.ws.send(JSON.stringify(event));
  }

  append(text: string): Promise<void> {
    if (!text) return Promise.resolve();
    return this.send({ type: "input_text.append", text });
  }

  commit(): Promise<void> { return this.send({ type: "input_text.commit" }); }
  cancel(): Promise<void> { return this.send({ type: "response.cancel" }); }
  pause(): Promise<void> { return this.send({ type: "playback.pause" }); }
  resume(): Promise<void> { return this.send({ type: "playback.resume" }); }
  keepalive(): Promise<void> { return this.send({ type: "session.keepalive" }); }
  acknowledge(responseId: string, sequence: number): Promise<void> {
    return this.send({ type: "playback.ack", response_id: responseId, sequence });
  }

  onEvent(callback: (event: LiveSpeechEvent) => void): () => void {
    this.callbacks.push(callback);
    return () => {
      this.callbacks = this.callbacks.filter((item) => item !== callback);
    };
  }

  close(): void { this.ws.close(1000, "Client closed"); }
}

export class RealtimeSession {
  private client: OpenSpeechClient;
  private ws: WebSocket;
  private transcriptCbs: RealtimeCallback[] = [];
  private audioCbs: RealtimeCallback[] = [];
  private vadCbs: RealtimeCallback[] = [];
  private resolveReady!: () => void;
  private rejectReady!: (error: Error) => void;
  private opened = false;
  readonly ready: Promise<void>;

  constructor(client: OpenSpeechClient) {
    this.client = client;
    this.ready = new Promise<void>((resolve, reject) => {
      this.resolveReady = resolve;
      this.rejectReady = reject;
    });
    void this.ready.catch(() => {});
    this.ws = new WebSocket(toWsUrl(client.baseUrl, "/v1/realtime"), ["realtime"]);
    this.ws.onopen = () => {
      this.opened = true;
      this.resolveReady();
    };
    this.ws.onmessage = (e) => this.dispatch(JSON.parse(String(e.data)));
    this.ws.onerror = () => {
      if (!this.opened) this.rejectReady(new Error("Realtime WebSocket failed to open"));
    };
    this.ws.onclose = (event) => {
      if (!this.opened) {
        this.rejectReady(
          new Error(event.reason || `Realtime WebSocket closed before opening (${event.code})`),
        );
      }
    };
  }

  private dispatch(event: any) {
    const t = event?.type || "";
    if (t.includes("transcription") || t === "conversation.item.created") this.transcriptCbs.forEach((cb) => cb(event));
    if (t.startsWith("response.audio")) this.audioCbs.forEach((cb) => cb(event));
    if (t.includes("speech_")) this.vadCbs.forEach((cb) => cb(event));
  }

  private async send(event: Record<string, unknown>): Promise<void> {
    await this.ready;
    if (this.ws.readyState !== WebSocket.OPEN) throw new Error("Realtime session is closed");
    this.ws.send(JSON.stringify(event));
  }

  async sendAudio(chunk: ArrayBuffer): Promise<void> {
    const bytes = new Uint8Array(chunk);
    const nodeBuffer = (globalThis as typeof globalThis & { Buffer?: NodeBufferLike }).Buffer;
    const audio = nodeBuffer
      ? nodeBuffer.from(bytes).toString("base64")
      : btoa(String.fromCharCode(...bytes));
    await this.send({ type: "input_audio_buffer.append", audio });
  }

  async commit(): Promise<void> {
    await this.send({ type: "input_audio_buffer.commit" });
  }

  async createResponse(text: string, voice = "alloy"): Promise<void> {
    await this.send({ type: "response.create", response: { instructions: text, voice } });
  }

  onTranscript(cb: RealtimeCallback) { this.transcriptCbs.push(cb); }
  onAudio(cb: RealtimeCallback) { this.audioCbs.push(cb); }
  onVad(cb: RealtimeCallback) { this.vadCbs.push(cb); }
  close() { this.ws.close(); }
}
