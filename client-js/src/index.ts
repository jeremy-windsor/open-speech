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

export type StreamTranscribeOptions = {
  reconnectAttempts?: number;
  // Aborting stops capture and closes the socket at once, even while the stream waits for an
  // event; the iteration then throws signal.reason (an AbortError unless abort() was given one).
  signal?: AbortSignal;
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

// The close codes with which the server refuses a socket: it accepts it and closes it at once,
// before any event, so the code and reason reach the client (README, "Refused WebSockets").
// 4001: a missing or wrong API key or ticket; 1008: an Origin not allowed (on /v1/audio/stream
// also a sample_rate it can't take); 4004: the feature is turned off. A reconnect gets the same
// answer, so none of them is retried.
export const REFUSAL_CLOSE_CODES: ReadonlySet<number> = new Set([1008, 4001, 4004]);

// The server refused the socket: it opened it and closed it with `code` and `reason`. The same
// as the Python client's StreamRefusedError (src/client).
export class StreamRefusedError extends Error {
  readonly code: number;
  readonly reason: string;

  constructor(code: number, reason: string) {
    super(`The server refused the stream: close code ${code}${reason ? `, ${reason}` : ""}`);
    this.name = "StreamRefusedError";
    this.code = code;
    this.reason = reason;
  }
}

function refusal(event: { code: number; reason?: string } | null | undefined): StreamRefusedError | null {
  return event && REFUSAL_CLOSE_CODES.has(event.code) ? new StreamRefusedError(event.code, event.reason || "") : null;
}

class TicketError extends Error {
  status: number;

  constructor(status: number) {
    super(`WebSocket ticket request failed (${status})`);
    this.status = status;
  }
}

// Browsers cannot set an Authorization header on a WebSocket. With an apiKey the URL carries a
// one-time ticket from POST /api/auth/ws-ticket instead; a ticket works once, within 60 s, so every
// connection (reconnects included) needs a new one. Without a key the URL is returned at once.
function webSocketUrl(client: OpenSpeechClient, path: string, signal?: AbortSignal): string | Promise<string> {
  const url = toWsUrl(client.baseUrl, path);
  if (!client.apiKey) return url;
  return (async () => {
    const r = await fetch(`${client.baseUrl}/api/auth/ws-ticket`, {
      method: "POST",
      headers: { Authorization: `Bearer ${client.apiKey}` },
      signal,
    });
    if (!r.ok) throw new TicketError(r.status);
    const { ticket } = (await r.json()) as { ticket: string };
    return `${url}${url.includes("?") ? "&" : "?"}ticket=${encodeURIComponent(ticket)}`;
  })();
}

// Settles like `promise`, or rejects with the signal's reason as soon as it aborts.
function abortable<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return promise;
  return new Promise<T>((resolve, reject) => {
    const onAbort = () => reject(signal?.reason);
    if (signal?.aborted) return onAbort();
    signal?.addEventListener("abort", onAbort, { once: true });
    promise.then(resolve, reject).finally(() => signal?.removeEventListener("abort", onAbort));
  });
}

function delay(ms: number, signal?: AbortSignal): Promise<void> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const elapsed = new Promise<void>((resolve) => {
    timer = setTimeout(resolve, ms);
  });
  return abortable(elapsed, signal).finally(() => clearTimeout(timer));
}

function toBase64(bytes: Uint8Array): string {
  const nodeBuffer = (globalThis as typeof globalThis & { Buffer?: NodeBufferLike }).Buffer;
  if (nodeBuffer) return nodeBuffer.from(bytes).toString("base64");
  // Spreading a whole multi-second chunk into one call exceeds the engine's argument limit.
  let binary = "";
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  }
  return btoa(binary);
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

  constructor({ baseUrl = "https://localhost:8100", apiKey, secure = true }: ClientOptions = {}) {
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

  async *streamTranscribe(
    mediaStream: MediaStream,
    options: number | StreamTranscribeOptions = 2,
  ): AsyncIterableIterator<TranscriptionEvent> {
    const settings: StreamTranscribeOptions = typeof options === "number" ? { reconnectAttempts: options } : options;
    const { reconnectAttempts = 2, signal } = settings;
    const model = "";
    const path = `/v1/audio/stream?sample_rate=16000&vad=true${model ? `&model=${encodeURIComponent(model)}` : ""}`;

    let attempts = 0;
    while (attempts <= reconnectAttempts) {
      const events: TranscriptionEvent[] = [];
      let closed = false;
      let refused: StreamRefusedError | null = null;
      let err: any = null;
      let ws: WebSocket;

      try {
        const url = await abortable(Promise.resolve(webSocketUrl(this, path, signal)), signal);
        if (signal?.aborted) throw signal.reason;
        ws = new WebSocket(url);
        ws.onmessage = (e) => {
          try {
            events.push(JSON.parse(String(e.data)));
          } catch {
            events.push({ type: "error", message: "Invalid JSON from server" });
          }
        };
        await new Promise<void>((resolve, reject) => {
          const onAbort = () => {
            ws.close();
            reject(signal?.reason);
          };
          signal?.addEventListener("abort", onAbort, { once: true });
          ws.onopen = () => {
            signal?.removeEventListener("abort", onAbort);
            resolve();
          };
          ws.onerror = (event) => {
            signal?.removeEventListener("abort", onAbort);
            err = event;
            reject(new Error("WebSocket failed to open"));
          };
          // A refusal opens the socket, then closes it with 4001, 1008 or 4004; one that closes
          // before the open event is the same refusal. (A no-op once the open has resolved.)
          ws.onclose = (event) => {
            closed = true;
            refused = refusal(event);
            if (refused) {
              signal?.removeEventListener("abort", onAbort);
              reject(refused);
            }
          };
        });
      } catch (openError) {
        if (signal?.aborted) throw signal.reason;
        if (openError instanceof StreamRefusedError) throw openError;
        // A refused ticket (a wrong key gives 401) is not transient, and every retry would count
        // against the server's failed-auth limit.
        if (openError instanceof TicketError && openError.status < 500) throw openError;
        attempts++;
        if (attempts > reconnectAttempts) throw openError;
        await delay(250 * attempts, signal);
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
      // Without this the microphone keeps streaming to the server over the open socket. It runs
      // when the generator finishes (a consumer's `break` calls return()) and the moment the
      // signal aborts: a Stop button needs that, since return() waits for the next event.
      let released = false;
      const release = () => {
        if (released) return;
        released = true;
        processor.onaudioprocess = null;
        processor.disconnect();
        analyser.disconnect();
        source.disconnect();
        audioCtx.close();
        ws.close();
      };
      signal?.addEventListener("abort", release, { once: true });

      try {
        while (ws.readyState === WebSocket.OPEN || !closed || events.length > 0) {
          if (signal?.aborted) throw signal.reason;
          while (events.length > 0) {
            const ev = events.shift()!;
            yield ev;
            if (signal?.aborted) throw signal.reason;
          }
          if (closed) break;
          await delay(20, signal);
        }
      } finally {
        signal?.removeEventListener("abort", release);
        release();
      }

      if (refused) throw refused;
      if (!err) return;
      attempts++;
      if (attempts > reconnectAttempts) throw new Error("streamTranscribe reconnect limit reached");
      await delay(250 * attempts, signal);
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
  private ws: WebSocket | null = null;
  private callbacks: Array<(event: LiveSpeechEvent) => void> = [];
  private resolveReady!: () => void;
  private rejectReady!: (error: Error) => void;
  private configured = false;
  private closed = false;
  readonly ready: Promise<void>;

  constructor(client: OpenSpeechClient, options: LiveSpeechSessionOptions = {}) {
    this.ready = new Promise<void>((resolve, reject) => {
      this.resolveReady = resolve;
      this.rejectReady = reject;
    });
    void this.ready.catch(() => {});
    const url = webSocketUrl(client, "/v1/audio/speech/stream");
    if (typeof url === "string") this.connect(url, options);
    // The catch also covers a WebSocket constructor that throws once the ticket has arrived.
    else void url.then((ticketUrl) => this.connect(ticketUrl, options)).catch((error: Error) => this.rejectReady(error));
  }

  private connect(url: string, options: LiveSpeechSessionOptions): void {
    if (this.closed) return;
    const ws = new WebSocket(url);
    this.ws = ws;
    ws.onmessage = (message) => {
      let event: LiveSpeechEvent;
      try {
        event = JSON.parse(String(message.data));
      } catch {
        event = { type: "error", error: { code: "invalid_json", message: "Invalid JSON from server" } };
      }
      let refused = false;
      if (event.type === "session.created") {
        ws.send(JSON.stringify({ type: "session.update", session: options }));
      } else if (event.type === "session.updated") {
        this.configured = true;
        this.resolveReady();
      } else if (event.type === "error" && !this.configured) {
        const error = event.error as { message?: string } | undefined;
        this.rejectReady(new Error(error?.message || "Live speech session failed to open"));
        refused = true;
      }
      try {
        this.callbacks.forEach((callback) => callback(event));
      } finally {
        // The server keeps a socket whose session.update it refused open, holding its Live slot,
        // so close it. Browsers only let a client close with 1000 or 3000-4999 (others throw).
        if (refused) ws.close(1000, "Session configuration failed");
      }
    };
    ws.onerror = () => this.rejectReady(new Error("Live speech WebSocket failed"));
    ws.onclose = (event) => {
      if (!this.configured || event.code !== 1000) {
        this.rejectReady(refusal(event) ?? new Error(event.reason || `Live speech WebSocket closed (${event.code})`));
      }
    };
  }

  private async send(event: Record<string, unknown>): Promise<void> {
    await this.ready;
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) throw new Error("Live speech session is closed");
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

  close(): void {
    this.closed = true;
    // Still waiting for its ticket: no socket will be opened.
    if (!this.ws) this.rejectReady(new Error("Live speech session is closed"));
    this.ws?.close(1000, "Client closed");
  }
}

export class RealtimeSession {
  private client: OpenSpeechClient;
  private ws: WebSocket | null = null;
  private transcriptCbs: RealtimeCallback[] = [];
  private audioCbs: RealtimeCallback[] = [];
  private vadCbs: RealtimeCallback[] = [];
  private resolveReady!: () => void;
  private rejectReady!: (error: Error) => void;
  private resolveClosed!: () => void;
  private rejectClosed!: (error: Error) => void;
  private refused: StreamRefusedError | null = null;
  private opened = false;
  private closeCalled = false;
  readonly ready: Promise<void>;
  // Settles when the session ends. It rejects with StreamRefusedError when the server refused
  // the socket (it opens, so `ready` resolves, then closes at once with 4001, 1008 or 4004), and
  // with ready's error when the socket never opened; any other close, or close(), resolves it.
  readonly closed: Promise<void>;

  constructor(client: OpenSpeechClient) {
    this.client = client;
    this.ready = new Promise<void>((resolve, reject) => {
      this.resolveReady = resolve;
      this.rejectReady = (error: Error) => {
        reject(error);
        if (!this.opened) this.rejectClosed(error);
      };
    });
    this.closed = new Promise<void>((resolve, reject) => {
      this.resolveClosed = resolve;
      this.rejectClosed = reject;
    });
    void this.ready.catch(() => {});
    void this.closed.catch(() => {});
    const url = webSocketUrl(client, "/v1/realtime");
    if (typeof url === "string") this.connect(url);
    // The catch also covers a WebSocket constructor that throws once the ticket has arrived.
    else void url.then((ticketUrl) => this.connect(ticketUrl)).catch((error: Error) => this.rejectReady(error));
  }

  private connect(url: string): void {
    if (this.closeCalled) return;
    const ws = new WebSocket(url, ["realtime"]);
    this.ws = ws;
    ws.onopen = () => {
      this.opened = true;
      this.resolveReady();
    };
    ws.onmessage = (e) => this.dispatch(JSON.parse(String(e.data)));
    ws.onerror = () => {
      if (!this.opened) this.rejectReady(new Error("Realtime WebSocket failed to open"));
    };
    ws.onclose = (event) => {
      // A refusal (4001, 1008 or 4004) usually comes right after the open, so `ready` has already
      // resolved: it is reported through `closed` and the sends. Nothing reconnects.
      this.refused = refusal(event);
      if (this.refused) this.rejectClosed(this.refused);
      else if (this.opened || this.closeCalled) this.resolveClosed();
      if (!this.opened) {
        this.rejectReady(
          this.refused ?? new Error(event.reason || `Realtime WebSocket closed before opening (${event.code})`),
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
    if (this.refused) throw this.refused;
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) throw new Error("Realtime session is closed");
    this.ws.send(JSON.stringify(event));
  }

  async sendAudio(chunk: ArrayBuffer): Promise<void> {
    await this.send({ type: "input_audio_buffer.append", audio: toBase64(new Uint8Array(chunk)) });
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
  close() {
    this.closeCalled = true;
    // Still waiting for its ticket: no socket will be opened.
    if (!this.ws) {
      this.resolveClosed();
      this.rejectReady(new Error("Realtime session is closed"));
    }
    this.ws?.close();
  }
}
