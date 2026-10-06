import { afterEach, beforeEach, describe, expect, test } from "bun:test";

import { OpenSpeechClient } from "../src/index.ts";

type SocketHandler = ((event: any) => void) | null;

class FakeWebSocket {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSING = 2;
  static readonly CLOSED = 3;
  static instances: FakeWebSocket[] = [];

  readonly url: string;
  readonly protocols?: string | string[];
  readyState = FakeWebSocket.CONNECTING;
  sent: unknown[] = [];
  onopen: SocketHandler = null;
  onmessage: SocketHandler = null;
  onerror: SocketHandler = null;
  onclose: SocketHandler = null;

  constructor(url: string, protocols?: string | string[]) {
    this.url = url;
    this.protocols = protocols;
    FakeWebSocket.instances.push(this);
  }

  open(): void {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.({ type: "open" });
  }

  fail(): void {
    this.onerror?.({ type: "error" });
  }

  message(data: string): void {
    this.onmessage?.({ data });
  }

  send(data: unknown): void {
    if (this.readyState !== FakeWebSocket.OPEN) {
      throw new Error("send called before WebSocket opened");
    }
    this.sent.push(data);
  }

  close(code = 1000, reason = ""): void {
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.({ code, reason });
  }
}

class FakeAudioNode {
  connect(): FakeAudioNode { return this; }
  disconnect(): void {}
}

class FakeScriptProcessor extends FakeAudioNode {
  onaudioprocess: ((event: any) => void) | null = null;
}

class FakeAudioContext {
  static instances: FakeAudioContext[] = [];

  readonly destination = new FakeAudioNode();
  readonly processor = new FakeScriptProcessor();
  closed = false;

  constructor(_options?: AudioContextOptions) {
    FakeAudioContext.instances.push(this);
  }

  createMediaStreamSource(_stream: MediaStream): FakeAudioNode { return new FakeAudioNode(); }
  createAnalyser(): FakeAudioNode { return new FakeAudioNode(); }
  createScriptProcessor(): FakeScriptProcessor { return this.processor; }
  close(): Promise<void> {
    this.closed = true;
    return Promise.resolve();
  }
}

const originalWebSocket = globalThis.WebSocket;
const originalAudioContext = globalThis.AudioContext;
const originalFetch = globalThis.fetch;

async function waitFor(predicate: () => boolean, timeoutMs = 1500): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (!predicate()) {
    if (Date.now() >= deadline) throw new Error("Timed out waiting for test condition");
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}

beforeEach(() => {
  FakeWebSocket.instances = [];
  FakeAudioContext.instances = [];
  globalThis.WebSocket = FakeWebSocket as unknown as typeof WebSocket;
  globalThis.AudioContext = FakeAudioContext as unknown as typeof AudioContext;
});

afterEach(() => {
  globalThis.WebSocket = originalWebSocket;
  globalThis.AudioContext = originalAudioContext;
  globalThis.fetch = originalFetch;
});

describe("transcribe response formats", () => {
  test("returns text formats as strings and uses the response content type as a fallback", async () => {
    const responses = [
      new Response("plain transcript", { headers: { "content-type": "application/json" } }),
      new Response("backend raw text", { headers: { "content-type": "text/plain; charset=utf-8" } }),
      Response.json({ text: "structured transcript" }),
    ];
    globalThis.fetch = (async () => responses.shift()!) as typeof fetch;
    const client = new OpenSpeechClient({ baseUrl: "http://example.test" });

    await expect(
      client.transcribe(new ArrayBuffer(0), { response_format: "text" }),
    ).resolves.toBe("plain transcript");
    await expect(client.transcribe(new ArrayBuffer(0))).resolves.toBe("backend raw text");
    await expect(client.transcribe(new ArrayBuffer(0))).resolves.toEqual({
      text: "structured transcript",
    });
  });
});

describe("RealtimeSession readiness", () => {
  test("queues sends until the WebSocket opens", async () => {
    const session = new OpenSpeechClient({ baseUrl: "http://example.test" }).realtimeSession();
    const socket = FakeWebSocket.instances[0];

    const pending = [
      session.sendAudio(new Uint8Array([1, 2]).buffer),
      session.commit(),
      session.createResponse("Reply", "alloy"),
    ];

    await Promise.resolve();
    expect(socket.sent).toHaveLength(0);

    socket.open();
    await session.ready;
    await Promise.all(pending);

    expect(socket.sent.map((item) => JSON.parse(String(item)).type)).toEqual([
      "input_audio_buffer.append",
      "input_audio_buffer.commit",
      "response.create",
    ]);
    session.close();
  });

  test("rejects readiness on an early close and rejects sends after close", async () => {
    const errored = new OpenSpeechClient({ baseUrl: "http://example.test" }).realtimeSession();
    FakeWebSocket.instances[0].fail();
    await expect(errored.ready).rejects.toThrow("Realtime WebSocket failed to open");
    errored.close();

    const earlyClose = new OpenSpeechClient({ baseUrl: "http://example.test" }).realtimeSession();
    FakeWebSocket.instances[1].close(1006, "network failed");
    await expect(earlyClose.ready).rejects.toThrow("network failed");

    const opened = new OpenSpeechClient({ baseUrl: "http://example.test" }).realtimeSession();
    const socket = FakeWebSocket.instances[2];
    socket.open();
    await opened.ready;
    socket.close();
    await expect(opened.commit()).rejects.toThrow("Realtime session is closed");
  });
});

describe("streamTranscribe reconnects", () => {
  test("retains its error handler after open and reconnects after a failed socket", async () => {
    const client = new OpenSpeechClient({ baseUrl: "http://example.test" });
    const events = client.streamTranscribe({} as MediaStream, 1);
    const firstEvent = events.next();

    await waitFor(() => FakeWebSocket.instances.length === 1);
    const firstSocket = FakeWebSocket.instances[0];
    firstSocket.open();
    await waitFor(() => FakeAudioContext.instances.length === 1);

    firstSocket.fail();
    firstSocket.close(1006, "connection lost");

    await waitFor(() => FakeWebSocket.instances.length === 2);
    const secondSocket = FakeWebSocket.instances[1];
    secondSocket.open();
    await waitFor(() => FakeAudioContext.instances.length === 2);
    secondSocket.message(JSON.stringify({ type: "transcript", text: "reconnected" }));

    await expect(firstEvent).resolves.toEqual({
      value: { type: "transcript", text: "reconnected" },
      done: false,
    });

    const completion = events.next();
    secondSocket.close(1000, "done");
    await expect(completion).resolves.toEqual({ value: undefined, done: true });
    expect(FakeAudioContext.instances.every((context) => context.closed)).toBe(true);
  });
});
