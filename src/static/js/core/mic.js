/**
 * Microphone capture: support and permission checks, the device list (shared by Transcribe and Voices),
 * and one capture at a time through an AudioWorklet (ScriptProcessor only as a fallback).
 * @module core/mic
 */
import { pauseAll, holdCaptureAudioSession } from './audio.js';
import { KEYS, read, write, remove } from './storage.js';
import { setRecording } from './status.js';
import { toast } from './toast.js';
import { FRAME_SIZE, PROCESSOR_NAME } from './mic-worklet.js';
import { rms as rmsOf, peak as peakOf, toDb } from './pcm.js';

export const WORKLET_URL = '/static/js/core/mic-worklet.js';
export const DEFAULT_DEVICE_LABEL = 'Default microphone';
// Talk streams its microphone to the AI assistant.
const OWNER_LABEL = { transcribe: 'Transcribe', voices: 'Voices', talk: 'Talk' };

/** Thrown when another view holds the microphone. */
export class MicBusyError extends Error {
  constructor(message) {
    super(message);
    this.name = 'MicBusyError';
  }
}

/** Thrown when capture can't start: reason 'insecure' | 'unsupported' | 'denied' | 'no-device' | 'failed'. */
export class MicError extends Error {
  constructor(message, reason, cause) {
    super(message);
    this.name = 'MicError';
    this.reason = reason;
    this.cause = cause;
  }
}

let session = null;          // the open MicSession
let reservedBy = null;       // owner while open() is starting or a session is open
let starting = null;         // settles when the open() that is starting now has finished, either way
let deviceListeners = new Set();
let deviceBound = null;

function mediaDevices() {
  try {
    return (typeof navigator !== 'undefined' && navigator.mediaDevices) || null;
  } catch {
    return null;
  }
}

/**
 * Whether this page can capture audio: getUserMedia needs a secure context (HTTPS or localhost).
 * @returns {{ok: true} | {ok: false, reason: 'insecure'|'unsupported'}}
 */
export function support() {
  if (globalThis.isSecureContext === false) return { ok: false, reason: 'insecure' };
  const md = mediaDevices();
  if (!md || typeof md.getUserMedia !== 'function') return { ok: false, reason: 'unsupported' };
  return { ok: true };
}

/** @returns {Promise<'granted'|'denied'|'prompt'|'unknown'>} from the Permissions API, when it answers */
export async function permission() {
  try {
    const status = await navigator.permissions.query({ name: 'microphone' });
    return ['granted', 'denied', 'prompt'].includes(status?.state) ? status.state : 'unknown';
  } catch {
    return 'unknown';
  }
}

/** @returns {string} the saved deviceId ('' for the default device) */
export function savedDevice() {
  return read(KEYS.micDevice, '') || '';
}

/** Persist a device choice: the opaque deviceId only ('' clears it). */
export function saveDevice(id) {
  if (id) write(KEYS.micDevice, String(id));
  else remove(KEYS.micDevice);
}

/**
 * The audio inputs for a device select: 'Default microphone' first (id ''), then each input with its label
 * or 'Microphone N'. Inputs the browser hides (no deviceId before permission) are left out. A saved device
 * that has vanished is cleared. [] when listing is unsupported or there are no inputs, so the select can
 * be disabled.
 * @returns {Promise<{id: string, label: string}[]>}
 */
export async function listDevices() {
  const md = mediaDevices();
  if (!md || typeof md.enumerateDevices !== 'function') return [];
  let inputs;
  try {
    inputs = (await md.enumerateDevices()).filter((d) => d.kind === 'audioinput');
  } catch {
    return [];
  }
  if (!inputs.length) return [];
  const rows = inputs
    .map((d, i) => ({ id: d.deviceId || '', label: d.label || `Microphone ${i + 1}` }))
    .filter((d) => d.id && d.id !== 'default');
  const saved = savedDevice();
  const listed = inputs.some((d) => d.deviceId);
  if (saved && listed && !rows.some((d) => d.id === saved)) saveDevice('');
  return [{ id: '', label: DEFAULT_DEVICE_LABEL }, ...rows];
}

function fireDeviceChange() {
  for (const cb of [...deviceListeners]) {
    try { cb(); } catch (error) { if (globalThis.console) globalThis.console.error(error); }
  }
}

/**
 * Call `cb` when the inputs change ('devicechange'), and after a permission grant reveals their labels.
 * @returns {() => void} off
 */
export function onDeviceChange(cb) {
  deviceListeners.add(cb);
  const md = mediaDevices();
  if (md && !deviceBound) {
    const handler = () => fireDeviceChange();
    if (typeof md.addEventListener === 'function') {
      md.addEventListener('devicechange', handler);
      deviceBound = () => md.removeEventListener('devicechange', handler);
    } else if ('ondevicechange' in md) {
      md.ondevicechange = handler;
      deviceBound = () => { md.ondevicechange = null; };
    }
  }
  return () => {
    deviceListeners.delete(cb);
    if (!deviceListeners.size && deviceBound) {
      deviceBound();
      deviceBound = null;
    }
  };
}

/** @returns {'transcribe'|'voices'|'talk'|null} who holds the microphone */
export function owner() {
  return reservedBy;
}

function micErrorFor(error) {
  const name = error?.name || '';
  if (name === 'NotAllowedError' || name === 'SecurityError') {
    return new MicError("Microphone access is blocked. Allow it in your browser's site settings, then try again.", 'denied', error);
  }
  if (name === 'NotFoundError' || name === 'OverconstrainedError') return new MicError('No microphone found.', 'no-device', error);
  return new MicError(error?.message || "The microphone couldn't start.", 'failed', error);
}

async function getStream(deviceId, constraints) {
  const md = mediaDevices();
  if (deviceId) {
    try {
      return { stream: await md.getUserMedia({ audio: { ...constraints, deviceId: { exact: deviceId } } }), fellBack: false };
    } catch (error) {
      if (error?.name === 'NotAllowedError' || error?.name === 'SecurityError') throw micErrorFor(error);
      saveDevice('');
      try {
        const stream = await md.getUserMedia({ audio: { ...constraints } });
        return { stream, fellBack: true };
      } catch (fallbackError) {
        throw micErrorFor(fallbackError);
      }
    }
  }
  try {
    return { stream: await md.getUserMedia({ audio: { ...constraints } }), fellBack: false };
  } catch (error) {
    throw micErrorFor(error);
  }
}

function busyError(who) {
  const holder = reservedBy || (session && session.owner) || who;
  return new MicBusyError(holder === who ? 'A recording is already running' : `Stop the ${OWNER_LABEL[holder]} recording first`);
}

// Microtask turns an open() waiting behind another gives that one's caller to stop a session it no longer
// wants (the views do it synchronously right after their await).
const CALLER_TURNS = 4;

/**
 * Open the microphone for one owner. Pauses all playback first; while the capture is open nothing plays
 * (on iPhone a live capture would move playback to the earpiece). A selected device that fails falls back
 * to the default device with a toast.
 *
 * An open() made while another one is still starting (the browser's permission prompt is up) waits for it
 * instead of failing: usually its view gave up meanwhile (Cancel, then Record again, or the other view's
 * Record), and that view stops the session it gets at once. Only if the earlier caller keeps its capture does
 * this one fail with MicBusyError. Its AudioContext is still made inside the gesture, before the wait.
 *
 * MicSession: {owner, stream, sampleRate, onFrame(cb(Float32Array)) -> off, level() -> {rms, peak, db},
 *   stop()}. Frames are 2048 samples from the AudioWorklet, or 4096 from the ScriptProcessor fallback.
 * @param {{owner: 'transcribe'|'voices'|'talk', deviceId?: string, echoCancellation?: boolean,
 *   noiseSuppression?: boolean}} options
 * @returns {Promise<object>} the MicSession
 * @throws {MicBusyError} when another owner holds the microphone
 * @throws {MicError} insecure, unsupported, denied, no-device or failed
 */
export async function open({ owner: who, deviceId, echoCancellation = true, noiseSuppression = true } = {}) {
  if (!OWNER_LABEL[who]) throw new TypeError(`Unknown microphone owner: ${who}`);
  if (session || (reservedBy && !starting)) throw busyError(who);
  const ok = support();
  if (!ok.ok) {
    throw new MicError(ok.reason === 'insecure'
      ? 'The microphone needs a secure connection. Open this page over HTTPS (Open Speech can make a self-signed certificate).'
      : "This browser can't capture audio.", ok.reason);
  }
  const before = starting;
  let finished = () => {};
  const mine = new Promise((resolve) => { finished = resolve; });
  starting = mine;
  // Reserved before the first await, so a double start never opens two captures.
  let reserved = !before;
  if (reserved) reservedBy = who;
  pauseAll();
  // Safari: the page's audio session becomes 'play-and-record' for the capture and gets its earlier type
  // ('playback' once something played) back when it stops (core/audio.js owns the type).
  const releaseAudioSession = holdCaptureAudioSession();
  const Ctor = globalThis.AudioContext || globalThis.webkitAudioContext;
  let ctx = null;
  let stream = null;
  try {
    // Created inside the gesture, before any await, so iOS lets it run.
    ctx = Ctor ? new Ctor() : null;
    if (!ctx) throw new MicError("This browser can't capture audio.", 'unsupported');
    if (before) {
      await before;
      for (let i = 0; i < CALLER_TURNS; i += 1) await null;
      if (session || reservedBy) throw busyError(who);
      reservedBy = who;
      reserved = true;
      pauseAll();   // anything started while this one waited
    }
    const chosen = deviceId === undefined ? savedDevice() : deviceId;
    const got = await getStream(chosen, { echoCancellation: !!echoCancellation, noiseSuppression: !!noiseSuppression });
    stream = got.stream;
    if (got.fellBack) toast('Selected microphone unavailable; using default microphone');
    if (ctx.state === 'suspended' && typeof ctx.resume === 'function') await ctx.resume().catch(() => {});
    const built = await buildGraph(ctx, stream);
    const s = makeSession(who, ctx, stream, built, releaseAudioSession);
    session = s;
    setRecording({ owner: who, startedAt: Date.now() });
    fireDeviceChange();   // labels are visible after a grant
    return s;
  } catch (error) {
    if (stream) for (const track of stream.getTracks()) track.stop();
    if (ctx && typeof ctx.close === 'function') ctx.close().catch(() => {});
    if (reserved && reservedBy === who && !session) reservedBy = null;
    releaseAudioSession();
    throw error;
  } finally {
    if (starting === mine) starting = null;
    finished();
  }
}

async function buildGraph(ctx, stream) {
  const source = ctx.createMediaStreamSource(stream);
  const sink = ctx.createGain();
  sink.gain.value = 0;
  sink.connect(ctx.destination);
  const subscribers = new Set();
  let lastFrame = null;
  const deliver = (frame) => {
    lastFrame = frame;
    for (const cb of [...subscribers]) {
      try { cb(frame); } catch (error) { if (globalThis.console) globalThis.console.error(error); }
    }
  };
  let node = null;
  let mode = 'worklet';
  if (ctx.audioWorklet && typeof ctx.audioWorklet.addModule === 'function' && typeof globalThis.AudioWorkletNode === 'function') {
    try {
      await ctx.audioWorklet.addModule('/static/js/core/mic-worklet.js');
      node = new globalThis.AudioWorkletNode(ctx, PROCESSOR_NAME, { numberOfInputs: 1, numberOfOutputs: 1, channelCount: 1 });
      node.port.onmessage = (event) => {
        const samples = event?.data?.samples;
        if (samples) deliver(samples instanceof Float32Array ? samples : new Float32Array(samples));
      };
    } catch {
      node = null;
    }
  }
  if (!node) {
    mode = 'script-processor';
    node = ctx.createScriptProcessor(4096, 1, 1);
    node.onaudioprocess = (event) => deliver(new Float32Array(event.inputBuffer.getChannelData(0)));
  }
  source.connect(node);
  node.connect(sink);
  return { source, node, sink, subscribers, mode, last: () => lastFrame };
}

function makeSession(who, ctx, stream, graph, releaseAudioSession = () => {}) {
  let stopped = false;
  const onPageHide = () => s.stop();
  const s = {
    owner: who,
    stream,
    sampleRate: Math.round(ctx.sampleRate || 48000),
    mode: graph.mode,
    frameSize: graph.mode === 'worklet' ? FRAME_SIZE : 4096,
    onFrame(cb) {
      graph.subscribers.add(cb);
      return () => graph.subscribers.delete(cb);
    },
    level() {
      const frame = graph.last();
      const r = rmsOf(frame);
      return { rms: r, peak: peakOf(frame), db: toDb(r) };
    },
    stop() {
      if (stopped) return;
      stopped = true;
      graph.subscribers.clear();
      if (graph.node.port) graph.node.port.onmessage = null;
      if ('onaudioprocess' in graph.node) graph.node.onaudioprocess = null;
      for (const n of [graph.source, graph.node, graph.sink]) {
        try { n.disconnect(); } catch { /* already disconnected */ }
      }
      for (const track of stream.getTracks()) track.stop();
      if (typeof ctx.close === 'function') ctx.close().catch(() => {});
      window.removeEventListener('pagehide', onPageHide);
      window.removeEventListener('beforeunload', onPageHide);
      if (session === s) session = null;
      if (reservedBy === who) reservedBy = null;
      releaseAudioSession();
      setRecording(null);
    },
    get active() { return !stopped; },
  };
  window.addEventListener('pagehide', onPageHide);
  window.addEventListener('beforeunload', onPageHide);
  return s;
}

/** The open MicSession, or null. */
export function currentSession() {
  return session;
}

/** Test hook. */
export function __resetForTests() {
  session = null;
  reservedBy = null;
  starting = null;
  deviceListeners = new Set();
  if (deviceBound) {
    try { deviceBound(); } catch { /* old env */ }
  }
  deviceBound = null;
}
