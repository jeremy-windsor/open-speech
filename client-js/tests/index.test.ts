// The SDK's own tests. They run on node:test, the runner of the Node.js CI sets up (Node 22.18 or
// newer runs this TypeScript as is); tests/test_client_sdk.py runs this file.
import assert from "node:assert/strict";
import { afterEach, beforeEach, describe, test } from "node:test";

import * as sdk from "../src/index.ts";

const { OpenSpeechClient } = sdk;

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

  // The client's close(): as in browsers (WHATWG WebSockets, close() step 1), a code other than
  // 1000 or 3000-4999 throws and leaves the socket open.
  close(code = 1000, reason = ""): void {
    if (code !== 1000 && !(code >= 3000 && code <= 4999)) {
      throw new DOMException("invalid code", "InvalidAccessError");
    }
    this.closedByServer(code, reason);
  }

  // The server or the network ending the connection, with any code.
  closedByServer(code: number, reason = ""): void {
    if (this.readyState === FakeWebSocket.CLOSED) return;
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

// Settles like `promise`, or rejects with "still pending" after ms, so a client that never
// reports fails the test instead of hanging it.
function within<T>(promise: Promise<T>, ms = 1000): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const late = new Promise<never>((_resolve, reject) => {
    timer = setTimeout(() => reject(new Error("still pending")), ms);
  });
  return Promise.race([promise, late]).finally(() => clearTimeout(timer));
}

async function drain(events: AsyncIterableIterator<unknown>): Promise<unknown[]> {
  const seen: unknown[] = [];
  for await (const event of events) seen.push(event);
  return seen;
}

// The server's refusals (README, "Refused WebSockets"): it accepts the socket, then closes it at
// once with one of these codes and the reason, before any event.
const REFUSALS: Array<[number, string]> = [
  [4001, "Invalid or missing API key"],
  [1008, "Origin not allowed. List it in OS_WS_ALLOWED_ORIGINS to allow it."],
  [4004, "Realtime API is disabled"],
];

function refusedError(code: number, reason: string) {
  return {
    name: "StreamRefusedError",
    code,
    reason,
    message: `The server refused the stream: close code ${code}, ${reason}`,
  };
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

    assert.equal(
      await client.transcribe(new ArrayBuffer(0), { response_format: "text" }),
      "plain transcript",
    );
    assert.equal(await client.transcribe(new ArrayBuffer(0)), "backend raw text");
    assert.deepEqual(await client.transcribe(new ArrayBuffer(0)), {
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
    assert.equal(socket.sent.length, 0);

    socket.open();
    await session.ready;
    await Promise.all(pending);

    assert.deepEqual(socket.sent.map((item) => JSON.parse(String(item)).type), [
      "input_audio_buffer.append",
      "input_audio_buffer.commit",
      "response.create",
    ]);
    session.close();
  });

  test("rejects readiness on an early close and rejects sends after close", async () => {
    const errored = new OpenSpeechClient({ baseUrl: "http://example.test" }).realtimeSession();
    FakeWebSocket.instances[0].fail();
    await assert.rejects(errored.ready, { message: /Realtime WebSocket failed to open/ });
    errored.close();

    const earlyClose = new OpenSpeechClient({ baseUrl: "http://example.test" }).realtimeSession();
    FakeWebSocket.instances[1].closedByServer(1006, "network failed");
    await assert.rejects(earlyClose.ready, { message: /network failed/ });

    const opened = new OpenSpeechClient({ baseUrl: "http://example.test" }).realtimeSession();
    const socket = FakeWebSocket.instances[2];
    socket.open();
    await opened.ready;
    socket.close();
    await assert.rejects(opened.commit(), { message: /Realtime session is closed/ });
  });
});

describe("LiveSpeechSession readiness", () => {
  test("closes the socket of a refused session.update and still reports the error", async () => {
    const session = new OpenSpeechClient({ baseUrl: "http://example.test" }).liveSpeechSession({ voice: "no_such_voice" });
    const seen: string[] = [];
    session.onEvent((event) => seen.push(event.type));
    const socket = FakeWebSocket.instances[0];
    socket.open();
    socket.message(JSON.stringify({ type: "session.created", session: {} }));
    assert.equal(JSON.parse(String(socket.sent[0])).type, "session.update");

    // The server keeps the socket (and its only Live slot) open after refusing the update.
    socket.message(JSON.stringify({ type: "error", error: { code: "invalid_event", message: "Named voice is unavailable" } }));
    await assert.rejects(session.ready, { message: /Named voice is unavailable/ });
    assert.deepEqual(seen, ["session.created", "error"]);
    assert.equal(socket.readyState, FakeWebSocket.CLOSED);
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
    firstSocket.closedByServer(1006, "connection lost");

    await waitFor(() => FakeWebSocket.instances.length === 2);
    const secondSocket = FakeWebSocket.instances[1];
    secondSocket.open();
    await waitFor(() => FakeAudioContext.instances.length === 2);
    secondSocket.message(JSON.stringify({ type: "transcript", text: "reconnected" }));

    assert.deepEqual(await firstEvent, {
      value: { type: "transcript", text: "reconnected" },
      done: false,
    });

    const completion = events.next();
    secondSocket.close(1000, "done");
    assert.deepEqual(await completion, { value: undefined, done: true });
    assert.equal(FakeAudioContext.instances.every((context) => context.closed), true);
  });
});

// The server
// accepts a socket it refuses and closes it at once with 4001, 1008 or 4004 and the reason. The
// SDK took only a failed open as a refusal, so a wrong key ended streamTranscribe with no event
// and no error. A refusal now throws StreamRefusedError (code, reason), like the Python client's,
// and is never retried; other closes keep their behaviour.
describe("refused WebSockets (close 4001, 1008 or 4004)", () => {
  test("StreamRefusedError is exported, is an Error, and carries the code and reason", () => {
    const error = new sdk.StreamRefusedError(4001, "Invalid or missing API key");
    assert.ok(error instanceof Error);
    assert.deepEqual(
      { name: error.name, code: error.code, reason: error.reason, message: error.message },
      refusedError(4001, "Invalid or missing API key"),
    );
    assert.equal(new sdk.StreamRefusedError(4004, "").message, "The server refused the stream: close code 4004");
  });

  for (const [code, reason] of REFUSALS) {
    test(`streamTranscribe throws on a ${code} close after the socket opens, and does not reconnect`, async () => {
      const client = new OpenSpeechClient({ baseUrl: "http://example.test" });
      const done = within(drain(client.streamTranscribe({} as MediaStream, 2)));
      await waitFor(() => FakeWebSocket.instances.length === 1);
      const socket = FakeWebSocket.instances[0];
      socket.open();
      await waitFor(() => FakeAudioContext.instances.length === 1);
      socket.closedByServer(code, reason);

      await assert.rejects(done, (error: any) => {
        assert.ok(error instanceof sdk.StreamRefusedError);
        assert.deepEqual(
          { name: error.name, code: error.code, reason: error.reason, message: error.message },
          refusedError(code, reason),
        );
        return true;
      });
      await new Promise((resolve) => setTimeout(resolve, 300));  // longer than the first retry's delay
      assert.equal(FakeWebSocket.instances.length, 1, "a refusal is not retried");
      assert.equal(FakeAudioContext.instances[0].closed, true, "capture is released");
    });
  }

  test("streamTranscribe throws on a refusal close that comes before the open event", async () => {
    const client = new OpenSpeechClient({ baseUrl: "http://example.test" });
    const done = within(drain(client.streamTranscribe({} as MediaStream, 2)));
    await waitFor(() => FakeWebSocket.instances.length === 1);
    FakeWebSocket.instances[0].closedByServer(4001, "Invalid or missing API key");

    await assert.rejects(done, refusedError(4001, "Invalid or missing API key"));
    await new Promise((resolve) => setTimeout(resolve, 300));
    assert.equal(FakeWebSocket.instances.length, 1);
    assert.equal(FakeAudioContext.instances.length, 0);
  });

  test("streamTranscribe still ends quietly on a normal close", async () => {
    const client = new OpenSpeechClient({ baseUrl: "http://example.test" });
    const done = within(drain(client.streamTranscribe({} as MediaStream, 2)));
    await waitFor(() => FakeWebSocket.instances.length === 1);
    const socket = FakeWebSocket.instances[0];
    socket.open();
    await waitFor(() => FakeAudioContext.instances.length === 1);
    socket.message(JSON.stringify({ type: "transcript", text: "hello" }));
    socket.closedByServer(1000, "done");
    assert.deepEqual(await done, [{ type: "transcript", text: "hello" }]);
    assert.equal(FakeWebSocket.instances.length, 1);
  });

  for (const [code, reason] of REFUSALS) {
    test(`RealtimeSession reports a ${code} close after ready through closed and its sends`, async () => {
      const session = new OpenSpeechClient({ baseUrl: "http://example.test" }).realtimeSession();
      const socket = FakeWebSocket.instances[0];
      socket.open();
      await session.ready;  // a refused socket opens first
      socket.closedByServer(code, reason);

      await assert.rejects(within(session.closed), (error: any) => {
        assert.ok(error instanceof sdk.StreamRefusedError);
        assert.deepEqual(
          { name: error.name, code: error.code, reason: error.reason, message: error.message },
          refusedError(code, reason),
        );
        return true;
      });
      await assert.rejects(within(session.commit()), refusedError(code, reason));
      await new Promise((resolve) => setTimeout(resolve, 50));
      assert.equal(FakeWebSocket.instances.length, 1, "a refusal is not retried");
    });
  }

  test("RealtimeSession rejects ready with StreamRefusedError on a refusal close before it opens", async () => {
    const session = new OpenSpeechClient({ baseUrl: "http://example.test" }).realtimeSession();
    FakeWebSocket.instances[0].closedByServer(4001, "Invalid or missing API key");
    await assert.rejects(within(session.ready), refusedError(4001, "Invalid or missing API key"));
    await assert.rejects(within(session.closed), refusedError(4001, "Invalid or missing API key"));
  });

  test("RealtimeSession keeps its other closes: closed resolves, and sends say the session is closed", async () => {
    const dropped = new OpenSpeechClient({ baseUrl: "http://example.test" }).realtimeSession();
    const socket = FakeWebSocket.instances[0];
    socket.open();
    await dropped.ready;
    socket.closedByServer(1006, "connection lost");
    assert.equal(await within(dropped.closed), undefined);
    await assert.rejects(dropped.commit(), { message: /Realtime session is closed/ });

    const ended = new OpenSpeechClient({ baseUrl: "http://example.test" }).realtimeSession();
    FakeWebSocket.instances[1].open();
    await ended.ready;
    ended.close();
    assert.equal(await within(ended.closed), undefined);

    const early = new OpenSpeechClient({ baseUrl: "http://example.test" }).realtimeSession();
    FakeWebSocket.instances[2].closedByServer(1006, "network failed");
    await assert.rejects(within(early.ready), { message: /network failed/ });
    await assert.rejects(within(early.closed), { message: /network failed/ });
  });

  test("LiveSpeechSession rejects ready with StreamRefusedError on a refusal close", async () => {
    const session = new OpenSpeechClient({ baseUrl: "http://example.test" }).liveSpeechSession();
    const socket = FakeWebSocket.instances[0];
    socket.open();
    socket.closedByServer(4004, "Live TTS is disabled");
    await assert.rejects(within(session.ready), refusedError(4004, "Live TTS is disabled"));
  });
});
