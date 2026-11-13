// Fakes for the browser APIs the UI talks to: fetch routes, WebSocket, media elements and Media Session,
// Web Audio, media devices. They run on the fake clock that env.js installs (global setTimeout/Date).
import { FakeEventTarget, FakeEvent, FakeMediaElement } from './dom.js';

export { FakeMediaElement };

/* ───────────────────────────── fetch ───────────────────────────── */

const NETWORK_ERROR = Symbol('network-error');

function headersObject(h) {
  const out = {};
  if (!h) return out;
  if (typeof h.forEach === 'function' && !Array.isArray(h)) {
    h.forEach((value, name) => { out[String(name).toLowerCase()] = String(value); });
    return out;
  }
  for (const [k, v] of Array.isArray(h) ? h : Object.entries(h)) out[String(k).toLowerCase()] = String(v);
  return out;
}

/** A JSON response. */
export function json(status, body, headers = {}) {
  const empty = status === 204 || status === 304;
  return new Response(empty ? null : (body === undefined ? '' : JSON.stringify(body)), {
    status, statusText: STATUS_TEXT[status] || '', headers: { 'content-type': 'application/json', ...headers },
  });
}

/** A text response (text/plain unless headers say otherwise). */
export function text(status, body, headers = {}) {
  return new Response(status === 204 ? null : String(body ?? ''), {
    status, statusText: STATUS_TEXT[status] || '', headers: { 'content-type': 'text/plain; charset=utf-8', ...headers },
  });
}

/** A binary response. */
export function bytes(status, data, type = 'application/octet-stream', headers = {}) {
  const body = data instanceof Uint8Array || data instanceof ArrayBuffer || data instanceof Blob ? data : new Uint8Array(data || [1, 2, 3]);
  return new Response(body, { status, statusText: STATUS_TEXT[status] || '', headers: { 'content-type': type, ...headers } });
}

/** A rejected fetch (TypeError: Failed to fetch). */
export function networkError() {
  return NETWORK_ERROR;
}

/** A response that arrives after `ms` on the fake clock. */
export function delay(ms, response) {
  return new Promise((resolve) => setTimeout(() => resolve(typeof response === 'function' ? response() : response), ms));
}

const NULL_BODY = new Set([101, 103, 204, 205, 304]);

/**
 * Tie a Response's body to the request's signal, as browsers do: an abort after the headers arrived errors
 * the body stream, so reading it rejects with an AbortError (the streaming Stop).
 */
function abortableBody(response, signal) {
  if (!signal || !response.body || NULL_BODY.has(response.status)) return response;
  const reader = response.body.getReader();
  let finished = false;
  const stream = new ReadableStream({
    start(controller) {
      const onAbort = () => {
        if (finished) return;
        finished = true;
        try { controller.error(new DOMException('The operation was aborted.', 'AbortError')); } catch { /* closed */ }
        reader.cancel().catch(() => {});
      };
      if (signal.aborted) onAbort();
      else signal.addEventListener('abort', onAbort, { once: true });
    },
    async pull(controller) {
      try {
        const { done, value } = await reader.read();
        if (finished) return;
        if (done) {
          finished = true;
          controller.close();
        } else {
          controller.enqueue(value);
        }
      } catch (error) {
        if (!finished) {
          finished = true;
          controller.error(error);
        }
      }
    },
    cancel(reason) {
      finished = true;
      return reader.cancel(reason);
    },
  });
  return new Response(stream, { status: response.status, statusText: response.statusText, headers: response.headers });
}

const STATUS_TEXT = {
  200: 'OK', 201: 'Created', 202: 'Accepted', 204: 'No Content', 304: 'Not Modified', 400: 'Bad Request', 401: 'Unauthorized',
  403: 'Forbidden', 404: 'Not Found', 408: 'Request Timeout', 409: 'Conflict', 413: 'Content Too Large', 422: 'Unprocessable Content',
  429: 'Too Many Requests', 500: 'Internal Server Error', 502: 'Bad Gateway', 503: 'Service Unavailable', 504: 'Gateway Timeout',
};

function compile(pattern) {
  const space = pattern.indexOf(' ');
  const method = space > 0 ? pattern.slice(0, space).toUpperCase() : '*';
  const path = space > 0 ? pattern.slice(space + 1).trim() : pattern.trim();
  const names = [];
  const src = path.split('/').map((seg) => {
    if (seg === '*') return '.*';
    if (seg.startsWith(':')) { names.push(seg.slice(1)); return '([^/]+)'; }
    return seg.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/\\\*/g, '.*');
  }).join('/');
  return { method, re: new RegExp(`^${src}$`), names, pattern };
}

/**
 * A fake fetch driven by routes: {'GET /api/models': (req) => json(200, {...}), 'POST /v1/audio/speech': …}.
 * Patterns use :param and *; the query string is not part of the match (req.query has it). A handler may
 * return a Response, a promise of one (delay()), networkError(), or a plain object (sent as JSON 200).
 * Unmatched routes answer the 404 envelope and are pushed to fetch.unmatched. Every call is recorded in
 * fetch.calls as {method, url, path, query, headers, body, json, signal}. As in a browser, aborting the
 * request's signal after the headers arrived errors the body: reading it rejects with an AbortError.
 */
export function createFetch(routes = {}, { origin = 'http://127.0.0.1:8100' } = {}) {
  let compiled = Object.entries(routes).map(([pattern, handler]) => ({ ...compile(pattern), handler }));
  const calls = [];
  const unmatched = [];
  async function fakeFetch(input, init = {}) {
    const url = new URL(typeof input === 'string' ? input : input.url, globalThis.location?.href || origin);
    const method = String(init.method || 'GET').toUpperCase();
    const headers = headersObject(init.headers);
    let parsed;
    if (typeof init.body === 'string') {
      try { parsed = JSON.parse(init.body); } catch { parsed = undefined; }
    }
    const call = {
      method, url: url.href, path: url.pathname, query: url.searchParams, search: url.search, headers, body: init.body,
      json: parsed, signal: init.signal || null, keepalive: !!init.keepalive, at: Date.now(),
    };
    calls.push(call);
    if (init.signal?.aborted) throw new DOMException('The operation was aborted.', 'AbortError');
    const route = compiled.find((r) => (r.method === '*' || r.method === method) && r.re.test(url.pathname));
    if (!route) {
      unmatched.push(call);
      return json(404, { error: { message: `Not found: ${method} ${url.pathname}`, code: 'not_found' } });
    }
    const m = route.re.exec(url.pathname);
    call.params = Object.fromEntries(route.names.map((n, i) => [n, decodeURIComponent(m[i + 1])]));
    let result = typeof route.handler === 'function' ? route.handler(call) : route.handler;
    if (init.signal) {
      result = Promise.race([
        Promise.resolve(result),
        new Promise((_, reject) => {
          if (init.signal.aborted) reject(new DOMException('The operation was aborted.', 'AbortError'));
          init.signal.addEventListener('abort', () => reject(new DOMException('The operation was aborted.', 'AbortError')), { once: true });
        }),
      ]);
    }
    const resolved = await result;
    if (resolved === NETWORK_ERROR) throw new TypeError('Failed to fetch');
    if (resolved instanceof Response) return abortableBody(resolved, init.signal);
    if (resolved === undefined || resolved === null) return text(204, '');
    return abortableBody(json(200, resolved), init.signal);
  }
  fakeFetch.calls = calls;
  fakeFetch.unmatched = unmatched;
  fakeFetch.route = (pattern, handler) => { compiled = [{ ...compile(pattern), handler }, ...compiled.filter((r) => r.pattern !== pattern)]; };
  fakeFetch.reset = () => { calls.length = 0; unmatched.length = 0; };
  fakeFetch.callsTo = (pattern) => {
    const c = compile(pattern);
    return calls.filter((call) => (c.method === '*' || c.method === call.method) && c.re.test(call.path));
  };
  return fakeFetch;
}

/* ───────────────────────────── WebSocket ───────────────────────────── */

export class FakeSocket extends FakeEventTarget {
  constructor(url, protocols) {
    super();
    this.url = String(url);
    this.protocols = protocols;
    this.readyState = FakeSocket.CONNECTING;
    this.sent = [];
    this.binaryType = 'blob';
    this.bufferedAmount = 0;
    this.closeCode = null;
    this.closeReason = '';
    this.onopen = null;
    this.onmessage = null;
    this.onerror = null;
    this.onclose = null;
    FakeSocket.instances.push(this);
    if (FakeSocket.onCreate) FakeSocket.onCreate(this);
  }
  get query() { return new URL(this.url).searchParams; }
  send(data) {
    if (this.readyState !== FakeSocket.OPEN) throw new DOMException('WebSocket is not open', 'InvalidStateError');
    this.sent.push(data);
  }
  /** The JSON messages the client sent. */
  get sentJson() {
    return this.sent.filter((d) => typeof d === 'string').map((d) => { try { return JSON.parse(d); } catch { return d; } });
  }
  /** Client side: like a browser, only 1000 or 3000–4999 may be passed (else InvalidAccessError), and the
   *  reason is at most 123 bytes (else SyntaxError). */
  close(code = 1000, reason = '') {
    if (code !== undefined && code !== 1000 && !(code >= 3000 && code <= 4999)) {
      throw new DOMException(`Failed to execute 'close' on 'WebSocket': The close code must be either 1000, or between 3000 and 4999. ${code} is neither.`, 'InvalidAccessError');
    }
    if (new TextEncoder().encode(String(reason)).length > 123) {
      throw new DOMException("Failed to execute 'close' on 'WebSocket': The close reason must not be greater than 123 UTF-8 bytes.", 'SyntaxError');
    }
    if (this.readyState >= FakeSocket.CLOSING) return;
    this.readyState = FakeSocket.CLOSING;
    this.closeCode = code;
    this.closeReason = reason;
    queueMicrotask(() => this._closed(code, reason, true));
  }
  _closed(code, reason, wasClean) {
    if (this.readyState === FakeSocket.CLOSED) return;
    this.readyState = FakeSocket.CLOSED;
    const event = new FakeEvent('close');
    event.code = code;
    event.reason = reason;
    event.wasClean = wasClean;
    this.dispatchEvent(event);
  }
  /** Server side: accept the connection. */
  open() {
    this.readyState = FakeSocket.OPEN;
    this.dispatchEvent(new FakeEvent('open'));
  }
  /** Server side: send a JSON object (or a string, ArrayBuffer or Blob as is). */
  serverSend(data) {
    const event = new FakeEvent('message');
    event.data = data instanceof ArrayBuffer || ArrayBuffer.isView(data) || typeof data === 'string' || data instanceof Blob ? data : JSON.stringify(data);
    this.dispatchEvent(event);
  }
  /** Server side: close with a code and reason. */
  serverClose(code = 1000, reason = '') {
    this.closeCode = code;
    this.closeReason = reason;
    this._closed(code, reason, code !== 1006);
  }
  /** Network failure: an error event, then close 1006. */
  error() {
    this.dispatchEvent(new FakeEvent('error'));
    this._closed(1006, '', false);
  }
}
FakeSocket.CONNECTING = 0; FakeSocket.OPEN = 1; FakeSocket.CLOSING = 2; FakeSocket.CLOSED = 3;
FakeSocket.instances = [];
FakeSocket.onCreate = null;

/* ───────────────────────────── Media Session ───────────────────────────── */

export class FakeMediaMetadata {
  constructor(init = {}) {
    this.title = init.title ?? '';
    this.artist = init.artist ?? '';
    this.album = init.album ?? '';
    this.artwork = init.artwork ?? [];
  }
}

export class FakeMediaSession {
  constructor() {
    this.metadata = null;
    this.playbackState = 'none';
    this.handlers = {};
    this.positionStates = [];
  }
  setActionHandler(action, handler) {
    if (handler === null || handler === undefined) delete this.handlers[action];
    else this.handlers[action] = handler;
  }
  setPositionState(state) {
    if (state && (!(state.duration >= 0) || state.position > state.duration || state.playbackRate === 0 || state.position < 0)) {
      throw new TypeError('Invalid position state');
    }
    this.positionStates.push(state ? { ...state } : null);
  }
  /** Test helper: run a lock-screen action. */
  action(name, details = {}) {
    const fn = this.handlers[name];
    if (!fn) throw new Error(`No Media Session handler for ${name}`);
    return fn({ action: name, ...details });
  }
}

/**
 * Safari 17+'s navigator.audioSession (W3C Audio Session draft): a page-wide `type`. Like Safari, an unknown
 * type is ignored. `history` records every type that was set, in order.
 */
export class FakeAudioSession extends FakeEventTarget {
  constructor(type = 'auto') {
    super();
    this.__type = type;
    this.history = [];
    this.state = 'inactive';
  }
  get type() { return this.__type; }
  set type(value) {
    if (!FakeAudioSession.TYPES.includes(value)) return;
    this.__type = value;
    this.history.push(value);
  }
}
FakeAudioSession.TYPES = ['auto', 'playback', 'transient', 'transient-solo', 'ambient', 'play-and-record'];

/* ───────────────────────────── Web Audio ───────────────────────────── */

class FakeAudioNode {
  constructor(ctx, kind) {
    this.context = ctx;
    this.kind = kind;
    this.connections = [];
    this.disconnected = false;
  }
  connect(target) { this.connections.push(target); return target; }
  disconnect() { this.connections = []; this.disconnected = true; }
}

class FakeAudioParam {
  constructor(value) { this.value = value; this.defaultValue = value; }
  setValueAtTime(v) { this.value = v; return this; }
  linearRampToValueAtTime(v) { this.value = v; return this; }
  exponentialRampToValueAtTime(v) { this.value = v; return this; }
  setTargetAtTime(v) { this.value = v; return this; }
  cancelScheduledValues() { return this; }
}

export class FakeAudioBuffer {
  constructor({ numberOfChannels = 1, length = 0, sampleRate = 48000, channels = null } = {}) {
    this.numberOfChannels = numberOfChannels;
    this.length = length;
    this.sampleRate = sampleRate;
    this.duration = length / sampleRate;
    this._channels = channels || Array.from({ length: numberOfChannels }, () => new Float32Array(length));
  }
  getChannelData(c) { return this._channels[c]; }
  copyToChannel(src, c, offset = 0) { this._channels[c].set(src, offset); }
  copyFromChannel(dest, c, offset = 0) { dest.set(this._channels[c].subarray(offset, offset + dest.length)); }
}

export class FakeAudioWorkletNode extends FakeAudioNode {
  constructor(ctx, name, options = {}) {
    super(ctx, 'worklet');
    if (!ctx.__modules.length) throw new DOMException('No AudioWorklet module was added', 'InvalidStateError');
    this.name = name;
    this.options = options;
    this.port = { onmessage: null, sent: [], postMessage(msg) { this.sent.push(msg); } };
    ctx.__worklets.push(this);
  }
  /** Test helper: the processor posts a frame. */
  __emit(samples, extra = {}) {
    const data = { samples: samples instanceof Float32Array ? samples : new Float32Array(samples), ...extra };
    if (typeof this.port.onmessage === 'function') this.port.onmessage({ data });
  }
}

export class FakeAudioContext extends FakeEventTarget {
  constructor(options = {}) {
    super();
    const defaults = FakeAudioContext.defaults;
    this.sampleRate = options.sampleRate || defaults.sampleRate;
    this.state = defaults.state;
    this.__createdAt = Date.now();
    this.__modules = [];
    this.__worklets = [];
    this.__processors = [];
    this.__sources = [];
    this.__decodeResult = defaults.decodeResult;
    this.destination = new FakeAudioNode(this, 'destination');
    this.baseLatency = 0.01;
    this.outputLatency = 0.02;
    const ctx = this;
    this.audioWorklet = defaults.worklet === false ? undefined : {
      addModule(url) {
        ctx.__modules.push(String(url));
        if (defaults.worklet === 'reject') return Promise.reject(new DOMException('Refused to load the module', 'AbortError'));
        return Promise.resolve();
      },
    };
    FakeAudioContext.instances.push(this);
  }
  get currentTime() { return (Date.now() - this.__createdAt) / 1000; }
  resume() { if (this.state !== 'closed') this.state = 'running'; this.dispatchEvent(new FakeEvent('statechange')); return Promise.resolve(); }
  suspend() { if (this.state !== 'closed') this.state = 'suspended'; this.dispatchEvent(new FakeEvent('statechange')); return Promise.resolve(); }
  close() { this.state = 'closed'; this.dispatchEvent(new FakeEvent('statechange')); return Promise.resolve(); }
  createBuffer(numberOfChannels, length, sampleRate) { return new FakeAudioBuffer({ numberOfChannels, length, sampleRate }); }
  createBufferSource() {
    const ctx = this;
    const node = new FakeAudioNode(this, 'buffer-source');
    node.buffer = null;
    node.playbackRate = new FakeAudioParam(1);
    node.onended = null;
    node.startedAt = null;
    node.stopped = false;
    const target = new FakeEventTarget();
    node.addEventListener = target.addEventListener.bind(target);
    node.removeEventListener = target.removeEventListener.bind(target);
    const fireEnded = () => {
      if (node.ended) return;
      node.ended = true;
      const event = new FakeEvent('ended');
      if (typeof node.onended === 'function') node.onended(event);
      target.dispatchEvent(event);
    };
    node.start = (when = 0) => {
      node.startedAt = Math.max(when, ctx.currentTime);
      const durationMs = node.buffer ? node.buffer.duration * 1000 / (node.playbackRate.value || 1) : 0;
      const waitMs = Math.max(0, (node.startedAt - ctx.currentTime) * 1000) + durationMs;
      node.__timer = setTimeout(fireEnded, waitMs);
    };
    node.stop = () => {
      node.stopped = true;
      clearTimeout(node.__timer);
      fireEnded();
    };
    this.__sources.push(node);
    return node;
  }
  createMediaStreamSource(stream) {
    const node = new FakeAudioNode(this, 'media-stream-source');
    node.mediaStream = stream;
    return node;
  }
  createScriptProcessor(bufferSize = 4096, inputs = 1, outputs = 1) {
    const node = new FakeAudioNode(this, 'script-processor');
    node.bufferSize = bufferSize;
    node.numberOfInputs = inputs;
    node.numberOfOutputs = outputs;
    node.onaudioprocess = null;
    node.__process = (samples) => {
      const data = samples instanceof Float32Array ? samples : new Float32Array(samples);
      if (typeof node.onaudioprocess === 'function') node.onaudioprocess({ inputBuffer: { getChannelData: () => data }, outputBuffer: { getChannelData: () => new Float32Array(data.length) } });
    };
    this.__processors.push(node);
    return node;
  }
  createAnalyser() {
    const node = new FakeAudioNode(this, 'analyser');
    node.fftSize = 2048;
    node.frequencyBinCount = 1024;
    node.smoothingTimeConstant = 0.8;
    node.getByteTimeDomainData = (arr) => arr.fill(128);
    node.getFloatTimeDomainData = (arr) => arr.fill(0);
    node.getByteFrequencyData = (arr) => arr.fill(0);
    return node;
  }
  createGain() {
    const node = new FakeAudioNode(this, 'gain');
    node.gain = new FakeAudioParam(1);
    return node;
  }
  createMediaElementSource() {
    throw new Error('createMediaElementSource is forbidden: long-form audio never goes through an AudioContext');
  }
  decodeAudioData(data, ok, fail) {
    const result = typeof this.__decodeResult === 'function' ? this.__decodeResult(data) : this.__decodeResult;
    const promise = result instanceof Error ? Promise.reject(result) : Promise.resolve(result || new FakeAudioBuffer({ length: 4800, sampleRate: this.sampleRate }));
    promise.then((b) => ok && ok(b), (e) => fail && fail(e));
    return promise;
  }
}
FakeAudioContext.instances = [];
FakeAudioContext.defaults = { sampleRate: 48000, state: 'running', worklet: true, decodeResult: null };

export class FakeOfflineAudioContext {
  constructor(channels = 1, length = 1, sampleRate = 48000) {
    if (typeof channels === 'object') ({ numberOfChannels: channels = 1, length = 1, sampleRate = 48000 } = channels);
    this.numberOfChannels = channels;
    this.length = length;
    this.sampleRate = sampleRate;
    this.__decodeResult = FakeOfflineAudioContext.decodeResult;
    FakeOfflineAudioContext.instances.push(this);
  }
  decodeAudioData(data) {
    const result = typeof this.__decodeResult === 'function' ? this.__decodeResult(data) : this.__decodeResult;
    return result instanceof Error ? Promise.reject(result) : Promise.resolve(result || new FakeAudioBuffer({ length: 4800, sampleRate: this.sampleRate }));
  }
}
FakeOfflineAudioContext.instances = [];
FakeOfflineAudioContext.decodeResult = null;

/* ───────────────────────────── MediaSource ───────────────────────────── */

export class FakeMediaSource extends FakeEventTarget {
  constructor() {
    super();
    this.readyState = 'closed';
    this.sourceBuffers = [];
    FakeMediaSource.instances.push(this);
  }
  static isTypeSupported(type) { return FakeMediaSource.supported.includes(String(type).split(';')[0].trim()); }
  addSourceBuffer(type) {
    const sb = new FakeEventTarget();
    sb.type = type;
    sb.updating = false;
    sb.appended = [];
    sb.mode = 'sequence';
    sb.appendBuffer = (buf) => {
      sb.appended.push(buf);
      sb.updating = true;
      queueMicrotask(() => { sb.updating = false; sb.dispatchEvent(new FakeEvent('updateend')); });
    };
    sb.abort = () => {};
    this.sourceBuffers.push(sb);
    return sb;
  }
  endOfStream() { this.readyState = 'ended'; this.dispatchEvent(new FakeEvent('sourceended')); }
  /** Test helper: the element attached the source. */
  __open() { this.readyState = 'open'; this.dispatchEvent(new FakeEvent('sourceopen')); }
}
FakeMediaSource.instances = [];
FakeMediaSource.supported = ['audio/mpeg'];

/* ───────────────────────────── media devices ───────────────────────────── */

export class FakeMediaStreamTrack extends FakeEventTarget {
  constructor(kind = 'audio', settings = {}) {
    super();
    this.kind = kind;
    this.readyState = 'live';
    this.enabled = true;
    this.__settings = settings;
    this.label = settings.label || 'Fake microphone';
  }
  stop() { this.readyState = 'ended'; }
  getSettings() { return { ...this.__settings }; }
}

export class FakeMediaStream {
  constructor(tracks = []) { this.__tracks = tracks; this.id = `stream-${Math.random().toString(36).slice(2, 8)}`; }
  getTracks() { return this.__tracks.slice(); }
  getAudioTracks() { return this.__tracks.filter((t) => t.kind === 'audio'); }
  get active() { return this.__tracks.some((t) => t.readyState === 'live'); }
}

/**
 * navigator.mediaDevices. policy: 'grant' | 'NotAllowedError' | 'NotFoundError'; asking for an exact
 * deviceId that isn't in `devices` fails with OverconstrainedError.
 */
export class FakeMediaDevices extends FakeEventTarget {
  constructor({ devices, policy = 'grant' } = {}) {
    super();
    this.devices = devices || [
      { deviceId: 'mic-a', kind: 'audioinput', label: 'Built-in Microphone', groupId: 'g1' },
      { deviceId: 'mic-b', kind: 'audioinput', label: 'USB Microphone', groupId: 'g2' },
      { deviceId: 'out-a', kind: 'audiooutput', label: 'Speakers', groupId: 'g1' },
    ];
    this.policy = policy;
    this.calls = [];
    this.streams = [];
    this.ondevicechange = null;
  }
  async enumerateDevices() { return this.devices.map((d) => ({ ...d })); }
  async getUserMedia(constraints) {
    this.calls.push(constraints);
    if (this.policy === 'NotAllowedError') throw new DOMException('Permission denied', 'NotAllowedError');
    if (this.policy === 'NotFoundError') throw new DOMException('Requested device not found', 'NotFoundError');
    const audio = constraints?.audio;
    const exact = audio && typeof audio === 'object' && audio.deviceId && typeof audio.deviceId === 'object' ? audio.deviceId.exact : null;
    if (exact && !this.devices.some((d) => d.deviceId === exact && d.kind === 'audioinput')) {
      const error = new DOMException('Constraints could not be satisfied', 'OverconstrainedError');
      error.constraint = 'deviceId';
      throw error;
    }
    const device = exact ? this.devices.find((d) => d.deviceId === exact) : this.devices.find((d) => d.kind === 'audioinput');
    const stream = new FakeMediaStream([new FakeMediaStreamTrack('audio', { deviceId: device?.deviceId || 'default', label: device?.label })]);
    this.streams.push(stream);
    return stream;
  }
  getSupportedConstraints() { return { deviceId: true, echoCancellation: true, noiseSuppression: true }; }
  /** Test helper: devices changed. */
  fireDeviceChange() {
    const event = new FakeEvent('devicechange');
    this.dispatchEvent(event);
  }
}

/* ───────────────────────────── observers ───────────────────────────── */

export class FakeResizeObserver {
  constructor(callback) {
    this.callback = callback;
    this.targets = new Set();
    FakeResizeObserver.instances.push(this);
  }
  observe(el) { this.targets.add(el); }
  unobserve(el) { this.targets.delete(el); }
  disconnect() { this.targets.clear(); }
  /** Test helper: report sizes for the observed elements. */
  __trigger(entries) {
    const list = entries || [...this.targets].map((target) => ({ target, contentRect: target.getBoundingClientRect(), borderBoxSize: [{ blockSize: target.getBoundingClientRect().height, inlineSize: target.getBoundingClientRect().width }] }));
    this.callback(list, this);
  }
}
FakeResizeObserver.instances = [];

export class FakeIntersectionObserver {
  constructor(callback, options = {}) {
    this.callback = callback;
    this.options = options;
    this.targets = new Set();
    FakeIntersectionObserver.instances.push(this);
  }
  observe(el) { this.targets.add(el); }
  unobserve(el) { this.targets.delete(el); }
  disconnect() { this.targets.clear(); }
  takeRecords() { return []; }
  __trigger(entries) { this.callback(entries, this); }
}
FakeIntersectionObserver.instances = [];

export class FakeMutationObserver {
  constructor(callback) { this.callback = callback; FakeMutationObserver.instances.push(this); }
  observe() {}
  disconnect() {}
  takeRecords() { return []; }
}
FakeMutationObserver.instances = [];
