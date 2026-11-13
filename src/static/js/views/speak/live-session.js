/**
 * Live (read as you type) session engine (inventory LR-02…LR-23, BUG-20).
 *
 * One engine per Live slot runs at most one session at a time against /v1/audio/speech/stream:
 * - start() runs synchronously inside the click: it snapshots the voice settings, creates the AudioContext
 *   and calls resume() inside the gesture (iOS), claims the app's audio session (kind 'live'), then awaits
 *   models.ensureReady (with its switch confirm) and api.openSocket (a fresh ticket per connection). The
 *   resume() is not awaited: it runs alongside the model check, so Live adds no wait before first audio.
 * - session.created -> one session.update with the snapshot; session.updated -> configured: the text from the
 *   start offset (cursor, or 0 for Read all) goes out with a commit (after input_text.context, the text before
 *   it on its line, when the cursor is mid-line), and appended text follows in 40 ms
 *   batches. Text travels in bounded chunks, one in flight at a time (createTextFlow, pure). An edit never
 *   stops Live (LR-10): text already spoken (or before the start offset) stays said and Live reads on; an edit
 *   to text the server has but hasn't spoken cancels what the server holds (response.cancel, the protocol's
 *   only way to drop queued text) and sends the text again as it now reads, from the end of what was spoken
 *   (after input_text.context, the text before it on its line, when that is mid-line),
 *   while the sentence playing plays on (a cancel waits until all of its audio is here); text still in the
 *   40 ms batch (or an input method's composition, read once it ends) can change freely.
 * - Audio deltas (base64 PCM16) are scheduled back to back on the AudioContext; playback.ack goes out per
 *   chunk only once that chunk has played locally, in FIFO order. A generation counter drops stale audio. A
 *   chunk the browser can't schedule counts as played, so the acks behind it keep flowing.
 * - The Now line follows the playback: it shows the text whose audio is playing (BUG-20; an in-editor
 *   highlight of the spoken span is deferred). A response that fails before its audio shows the reason at once.
 * - Pause/resume, the hidden page, keepalive, Stop, errors and close codes (plain copy per close code). Nothing
 *   resumes while the microphone records, and a context the system suspends (an iOS
 *   interruption, a resume the browser refused) counts as the listener's pause. The voice latency
 *   leaves out responses whose first audio waited on a pause.
 * - An admission refusal right after this engine closed its own session (the server still shutting it down)
 *   is retried a few times, each with a fresh ticket, before it is shown.
 * - Leaving the page closes the socket (GS-20): on beforeunload, and on a pagehide that discards the page (iOS
 *   Safari never fires beforeunload), which also releases the session. A page back from the back/forward
 *   cache whose socket the browser closed releases the session on pageshow.
 * Injected for tests: AudioContextCtor, openSocket, ensureReady, clock. No DOM or browser API at import time.
 * @module views/speak/live-session
 */
import { app } from '../../core/store.js';
import { openSocket as apiOpenSocket, explain, explainClose, closeAction } from '../../core/api.js';
import { ensureReady as modelsEnsureReady, markUsed } from '../../core/models.js';
import * as audio from '../../core/audio.js';
import { toast } from '../../core/toast.js';
import { sliceCodePoints } from '../../core/format.js';

export const LIVE_PATH = '/v1/audio/speech/stream';
export const HOME_ROUTE = 'speak/live';
export const MAX_CHUNK = 2048;
export const DEFAULT_MAX_BUFFER = 8192;
export const BATCH_MS = 40;
export const KEEPALIVE_MS = 30000;
export const SCHEDULE_LEAD_S = 0.025;
export const LATENCY_WINDOW = 5;
/** Speaking turns back into Listening only after this much quiet, so the status doesn't flap per sentence. */
export const LISTEN_DELAY_MS = 700;
/**
 * The waits before connecting again after an admission refusal (session_busy before session.created) that
 * came while the server was still shutting a session down: its reason is 'draining', or this engine's own
 * session ended less than OWN_CLOSE_WINDOW_MS earlier (the server had not finished with it yet). Together
 * they cover the server's default wait for a session's synthesis worker (tts_live_shutdown_timeout_s, 5 s).
 */
export const ADMISSION_RETRY_MS = Object.freeze([250, 500, 1000, 1500, 2000]);
export const OWN_CLOSE_WINDOW_MS = 3000;
/** After this many audio chunks in a row that the browser could not schedule, the session stops. */
export const MAX_AUDIO_FAILURES = 3;
/** A change of the AudioContext's state is read this long after it happened (it settles first). */
export const CTX_SETTLE_MS = 250;
/** A resume() the browser holds back (outside a gesture) is given up on after this long. */
export const CTX_RESUME_WAIT_MS = 1500;
/**
 * A resume() asked for inside the listener's click (Start, the toggle) is never held back, only slow at times
 * (an output device waking up, Bluetooth headphones): it is waited on this long before a still-suspended
 * context counts as a pause, so a slow device never turns a fresh Start into "Paused".
 */
export const CTX_GESTURE_RESUME_WAIT_MS = 10000;

const OPEN = 1;
const CLOSING = 2;

/** The latency modes (latency_mode) with their labels and hints. */
export const MODES = Object.freeze([
  Object.freeze({ value: 'natural', label: 'Book', hint: 'Waits for full sentences: the smoothest for reading.' }),
  Object.freeze({ value: 'responsive', label: 'Responsive', hint: 'Speaks short phrases as you type.' }),
  Object.freeze({ value: 'instant_word', label: 'Each word', hint: 'Speaks every completed word: fastest, choppier.' }),
]);
export const DEFAULT_MODE = 'natural';

/** @returns {object} the MODES entry for a value (the default for anything unknown) */
export function modeFor(value) {
  return MODES.find((m) => m.value === value) || MODES[0];
}

/** Status words. */
export const STATUS = Object.freeze({
  stopped: 'Stopped',
  preparing: 'Preparing…',
  connecting: 'Connecting…',
  listening: 'Listening',
  speaking: 'Speaking',
  paused: 'Paused',
  background: 'Paused in background',
  busy: 'Busy',
  error: 'Error',
  disconnected: 'Disconnected',
});

/** The pill tone for each phase (the status text always carries the meaning). */
export const TONES = Object.freeze({
  stopped: 'idle',
  preparing: 'busy',
  connecting: 'busy',
  listening: 'ok',
  speaking: 'ok',
  paused: 'idle',
  background: 'idle',
  // A refused admission is a failure the banner beside it reports: the error style, as in the old UI
  // (inventory LR-07: Busy, Error and Disconnected share it), so the pill and the banner never disagree.
  busy: 'error',
  error: 'error',
  disconnected: 'error',
});

export const COPY = Object.freeze({
  idle: 'Start, then type or paste. Live reads from the cursor as you write.',
  /** By the controls once the text changes while Live is stopped. */
  startPrompt: 'Press Start to hear this live.',
  preparing: 'Preparing the selected voice…',
  connecting: 'Connecting to the voice engine…',
  listening: 'Listening for text…',
  reading: 'Reading…',
  paused: 'Paused. Resume to keep reading.',
  background: 'Paused while this page is in the background.',
  unavailable: "This model needs a reference recording and isn't available in Live yet",
  gated: 'Choose another model in Voice and settings to use Live.',
  noVoice: 'Choose a voice or a reference recording first',
  noModel: 'Choose a speech model first',
  recording: 'Stop recording to listen.',
  /** The mini-player's status while a Resume waits for the recording to end (the words clips use). */
  recordingStatus: 'Stop recording to listen',
  noAudio: "This browser can't play Live audio.",
  audioPart: "This browser couldn't play part of the Live audio.",
  invalid: 'Live received invalid data from the server.',
  draining: 'The last Live session is still finishing. Start again in a moment.',
  title: 'Live stopped',
});

/** Errors after which the server closes the socket itself (src/live_tts/server.py, close 4008). */
const SERVER_CLOSES = new Set(['idle_timeout', 'pause_timeout', 'playback_stalled']);
const FLOW_COPY = Object.freeze({
  idle_timeout: 'Live stopped after a long time without new text. Start again to continue.',
  pause_timeout: 'Live stopped after being paused for too long. Start again to continue.',
  playback_stalled: 'Live stopped because this browser stopped playing the audio. Start again to continue.',
});

/* ───────────────────────────── pure helpers ───────────────────────────── */

/**
 * PURE: decode base64 little-endian PCM16 into float samples in [-1, 1).
 * @param {string} base64
 * @returns {Float32Array}
 */
export function decodePcm16(base64) {
  const binary = atob(String(base64 || ''));
  const n = binary.length >> 1;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 1) {
    let v = binary.charCodeAt(2 * i) | (binary.charCodeAt(2 * i + 1) << 8);
    if (v >= 0x8000) v -= 0x10000;
    out[i] = v / 32768;
  }
  return out;
}

/** PURE: the median of the finite numbers in `values`, or null. */
export function median(values) {
  const v = (values || []).filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  if (!v.length) return null;
  const mid = v.length >> 1;
  return v.length % 2 ? v[mid] : (v[mid - 1] + v[mid]) / 2;
}

/** PURE: "Voice latency ~180 ms" ('' when unknown). */
export function latencyLabel(ms) {
  if (!Number.isFinite(ms)) return '';
  return ms < 1000 ? `Voice latency ~${Math.round(ms)} ms` : `Voice latency ~${(ms / 1000).toFixed(1)} s`;
}

/** PURE: the status after a stop: "Stopped", or "Stopped: another player started" for a reason. */
export function stoppedStatus(reason = 'Stopped') {
  const text = String(reason || '').trim();
  if (!text || text === 'Stopped') return { status: STATUS.stopped, reason: '' };
  return { status: `${STATUS.stopped}: ${lowerFirst(text)}`, reason: text };
}

function lowerFirst(text) {
  // "Another player started" -> "another player started"; an acronym ("API key changed") stays as written.
  return /^[A-Z][a-z]/.test(text) ? text[0].toLowerCase() + text.slice(1) : text;
}

/**
 * PURE: what the status live region says. It changes only on a phase change, never per response: while
 * connected, Speaking and Listening read the same ("Listening"), because the speech itself is what a
 * listener hears, and a flip per sentence would talk over it (announce only on state changes).
 */
export function spokenStatus(phase, status) {
  return phase === 'speaking' ? STATUS.listening : String(status || STATUS.stopped);
}

/**
 * PURE: the session config Live sends, from Speak's voice config and the chosen mode (inventory LR-04):
 * {model, voice, speed, latency_mode, instructions? (only when non-empty), voice_library_ref?}.
 * @returns {{config: object, caps: object}}
 */
export function snapshotConfig(raw, mode) {
  const src = raw && typeof raw === 'object' ? raw : {};
  const caps = src.caps && typeof src.caps === 'object' ? src.caps : {};
  const config = {
    model: String(src.model || ''),
    voice: String(src.voice || ''),
    speed: Number(src.speed) > 0 ? Number(src.speed) : 1,
    latency_mode: modeFor(mode).value,
  };
  const instructions = typeof src.instructions === 'string' ? src.instructions.trim() : '';
  if (instructions) config.instructions = instructions;
  if (src.voice_library_ref) config.voice_library_ref = String(src.voice_library_ref);
  return { config: Object.freeze(config), caps };
}

/**
 * PURE: why Start can't run for this snapshot ('' when it can), in the order Start checks (LR-02).
 * @param {{config: object, caps: object}} snap
 * @param {{recording?: boolean}} [context]
 */
export function refusalFor({ config, caps }, { recording = false } = {}) {
  if (caps && caps.live_reader === false) return COPY.unavailable;
  if (!config.model) return COPY.noModel;
  if (!config.voice) return COPY.noVoice;
  if (recording) return COPY.recording;
  return '';
}

/**
 * PURE flow control for the outbound text (inventory LR-09):
 * - queue(text, commit) adds text (and an input_text.commit) to the outbound queue;
 * - drain() returns the events to send now: commits only while nothing is in flight, and at most one
 *   input_text.append in flight, of at most min(2048, max_buffer_chars - buffered_chars) code points;
 * - accepted() shifts the FIFO in-flight list and takes buffered_chars and generation from the server;
 * - status() takes buffered_chars and generation from session.status;
 * - overflow() puts the rejected chunk back at the front and treats the buffer as full until a status;
 * - cancelled() starts the next generation and drops everything queued or in flight.
 * @param {{maxBufferChars?: number}} [options]
 */
export function createTextFlow({ maxBufferChars = DEFAULT_MAX_BUFFER } = {}) {
  const st = { outbound: [], inflight: [], maxBufferChars, serverBufferedChars: 0, generation: 0 };
  return {
    state: st,
    setLimits(max) {
      st.maxBufferChars = Number.isInteger(max) && max > 0 ? max : DEFAULT_MAX_BUFFER;
    },
    queue(text, commit = false) {
      if (text) st.outbound.push({ type: 'text', text: String(text) });
      if (commit) st.outbound.push({ type: 'commit' });
    },
    drain() {
      const events = [];
      while (st.outbound.length && !st.inflight.length) {
        const next = st.outbound[0];
        if (next.type === 'commit') {
          events.push({ type: 'input_text.commit' });
          st.outbound.shift();
          continue;
        }
        const room = Math.max(0, st.maxBufferChars - st.serverBufferedChars);
        if (!room) break;
        const chunk = sliceCodePoints(next.text, Math.min(MAX_CHUNK, room));
        if (!chunk) break;
        events.push({ type: 'input_text.append', text: chunk });
        st.inflight.push(chunk);
        next.text = next.text.slice(chunk.length);
        if (!next.text) st.outbound.shift();
        break;
      }
      return events;
    },
    accepted(event = {}) {
      st.serverBufferedChars = Number(event.buffered_chars) || 0;
      st.inflight.shift();
      if (Number.isInteger(event.generation)) st.generation = event.generation;
    },
    status(event = {}) {
      st.serverBufferedChars = Number(event.buffered_chars) || 0;
      if (Number.isInteger(event.generation)) st.generation = event.generation;
    },
    overflow() {
      if (!st.inflight.length) return false;
      const rejected = st.inflight.shift();
      st.outbound.unshift({ type: 'text', text: rejected });
      st.serverBufferedChars = st.maxBufferChars;
      return true;
    },
    cancelled(event = {}) {
      st.generation = Number.isInteger(event.generation) ? event.generation + 1 : st.generation + 1;
      st.inflight.length = 0;
      st.outbound.length = 0;
    },
    get generation() { return st.generation; },
    get busy() { return st.inflight.length > 0 || st.outbound.length > 0; },
  };
}

const isHighSurrogate = (c) => c >= 0xd800 && c <= 0xdbff;
const isLowSurrogate = (c) => c >= 0xdc00 && c <= 0xdfff;

/**
 * PURE: where `after` differs from `before` (an edit): `before`'s [start, oldEnd) became `after`'s
 * [start, newEnd). The common start and end are left out, and a surrogate pair is never split.
 * @returns {{start: number, oldEnd: number, newEnd: number}}
 */
export function changedRange(before, after) {
  const a = String(before ?? '');
  const b = String(after ?? '');
  const max = Math.min(a.length, b.length);
  let start = 0;
  while (start < max && a.charCodeAt(start) === b.charCodeAt(start)) start += 1;
  if (start > 0 && isHighSurrogate(a.charCodeAt(start - 1))) start -= 1;
  let oldEnd = a.length;
  let newEnd = b.length;
  while (oldEnd > start && newEnd > start && a.charCodeAt(oldEnd - 1) === b.charCodeAt(newEnd - 1)) {
    oldEnd -= 1;
    newEnd -= 1;
  }
  if (oldEnd < a.length && isLowSurrogate(a.charCodeAt(oldEnd))) {
    oldEnd += 1;
    newEnd += 1;
  }
  return { start, oldEnd, newEnd };
}

/**
 * PURE: what `input_text.context` says before text sent again from `from`:
 * the text before `from` on its line, from just after the last line break ('' when a line starts
 * at `from`). Only the line's state matters to the server, so a very long line keeps its last `max` code points
 * (within one append's limit of 4 UTF-8 bytes per buffered character), never half a surrogate pair.
 * @param {string} value the editor text
 * @param {number} from where the text sent again starts
 * @param {number} [max] code points at most
 */
export function lineContext(value, from, max = DEFAULT_MAX_BUFFER) {
  const text = String(value ?? '');
  const end = Math.max(0, Math.min(Number(from) || 0, text.length));
  if (!end) return '';
  const lineStart = text.lastIndexOf('\n', end - 1) + 1;
  let at = end;
  let count = 0;
  while (at > lineStart && count < max) {
    const pair = at - 2 >= lineStart && isLowSurrogate(text.charCodeAt(at - 1)) && isHighSurrogate(text.charCodeAt(at - 2));
    at -= pair ? 2 : 1;
    count += 1;
  }
  return text.slice(at, end);
}

/* ───────────────────────────── the engine ───────────────────────────── */

/** The state while no session runs: the last session's outcome (status, reason, error) or a refusal. */
function idleState({ phase = 'stopped', status = '', reason = '', refusal = '', error = null } = {}) {
  const text = status || STATUS[phase] || STATUS.stopped;
  return Object.freeze({
    active: false,
    phase,
    status: text,
    spoken: spokenStatus(phase, text),
    reason,
    tone: TONES[phase] || 'idle',
    paused: false,
    background: false,
    now: Object.freeze({ kind: 'hint', text: COPY.idle }),
    latencyMs: null,
    refusal,
    error,
    mode: '',
  });
}

function defaultClock() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function' ? performance.now() : Date.now();
}

function pageHidden() {
  return typeof document !== 'undefined' && !!document.hidden;
}

function logError(error) {
  if (globalThis.console) globalThis.console.error(error);
}

/**
 * Create the Live engine for one slot.
 * @param {object} options
 * @param {HTMLTextAreaElement} options.editor the shared #sp-input (value, selectionStart, input events)
 * @param {() => {model: string, voice: string, speed: number, instructions?: string,
 *   voice_library_ref?: string, caps?: object}} options.getVoiceConfig read once per start (the snapshot)
 * @param {() => string} [options.getMode] the chosen latency_mode
 * @param {() => string} [options.inputStopReason] '' while the editor's changes are Live's to read; otherwise
 *   the reason to stop with when the editor changes (Speak shows Generate: the text is meant for a take)
 * @param {(state: object) => void} [options.onChange] every state change (see getState())
 * @param {Function} [options.AudioContextCtor] default: window.AudioContext (or webkitAudioContext)
 * @param {(path: string) => Promise<WebSocket>} [options.openSocket] default: api.openSocket
 * @param {(model: string, options: object) => Promise<boolean>} [options.ensureReady] default: models.ensureReady
 * @param {() => number} [options.clock] milliseconds, for the latency readout and the admission retry
 * @returns {{start: (o?: {readAll?: boolean}) => boolean, stop: (reason?: string) => void,
 *   togglePause: () => void, isActive: () => boolean, getState: () => object, check: () => string,
 *   clearError: () => void, clearRefusal: () => void, refuse: (message: string) => void, destroy: () => void}}
 */
export function createLiveEngine({
  editor = null,
  getVoiceConfig = () => ({}),
  getMode = () => DEFAULT_MODE,
  inputStopReason = () => '',
  onChange = () => {},
  AudioContextCtor = null,
  openSocket = apiOpenSocket,
  ensureReady = modelsEnsureReady,
  clock = defaultClock,
} = {}) {
  let s = null;            // the running session
  let last = idleState();  // what shows when no session runs
  let destroyed = false;
  let lastSignature = '';
  let lastPlayer = '';
  let lastClosedAt = -Infinity;   // when a session the server had admitted last ended

  const current = (sess) => s === sess && !sess.stopping && !sess.released;

  function getState() {
    return s ? stateOf(s) : last;
  }

  /** Tell the UI (and the mini-player) about a change; identical states are not repeated. */
  function publish() {
    const state = getState();
    if (s && s.handle) {
      const player = `${state.phase}|${state.paused}|${state.background}|${state.refusal}`;
      if (player !== lastPlayer) {
        lastPlayer = player;
        s.handle.notify();
      }
    }
    const signature = JSON.stringify(state);
    if (signature === lastSignature) return;
    lastSignature = signature;
    try { onChange(state); } catch (error) { logError(error); }
  }

  /* ── state ── */

  function phaseOf(sess) {
    if (sess.fatal) return sess.fatal.phase;
    if (!sess.configured) return sess.connecting ? 'connecting' : 'preparing';
    if (sess.paused) return 'paused';
    if (sess.background) return 'background';
    return sess.speaking ? 'speaking' : 'listening';
  }

  function nowOf(sess, phase) {
    if (sess.fatal) return { kind: 'reason', text: sess.fatal.message };
    if (sess.nowText) return { kind: 'spoken', text: sess.nowText };
    if (sess.reasonText) return { kind: 'reason', text: sess.reasonText };
    const hint = {
      preparing: COPY.preparing, connecting: COPY.connecting, paused: COPY.paused, background: COPY.background, speaking: COPY.reading,
    };
    return { kind: 'hint', text: hint[phase] || COPY.listening };
  }

  function stateOf(sess) {
    const phase = phaseOf(sess);
    return Object.freeze({
      active: true,
      phase,
      status: STATUS[phase],
      spoken: spokenStatus(phase, STATUS[phase]),
      reason: '',
      tone: TONES[phase],
      paused: sess.paused,
      background: sess.background,
      now: Object.freeze(nowOf(sess, phase)),
      latencyMs: sess.latencyMs,
      refusal: sess.refusal,
      error: null,
      mode: sess.config.latency_mode,
    });
  }

  /* ── start ── */

  function snapshot() {
    let raw = {};
    try { raw = getVoiceConfig() || {}; } catch (error) { logError(error); raw = {}; }
    return snapshotConfig(raw, getMode());
  }

  /** Why Start can't run with the current settings ('' when it can). Reads getVoiceConfig() once. */
  function check() {
    return refusalFor(snapshot(), { recording: !!app.get().recording });
  }

  function refuse(message) {
    last = idleState({ ...plain(last), refusal: String(message || ''), error: null });
    publish();
  }

  function plain(state) {
    const { phase, status, reason, refusal, error } = state;
    return { phase, status, reason, refusal, error };
  }

  /**
   * Start a session (call it synchronously inside the click). Returns false when it refused or one runs.
   * @param {{readAll?: boolean}} [options] readAll reads from the start of the text, else from the cursor
   */
  function start({ readAll = false } = {}) {
    if (s || destroyed) return false;                     // LR-02: already running -> no-op
    const snap = snapshot();                              // LR-04: before any await
    const refusal = refusalFor(snap, { recording: !!app.get().recording });
    if (refusal) {
      refuse(refusal);
      return false;
    }
    const Ctor = AudioContextCtor || globalThis.AudioContext || globalThis.webkitAudioContext;
    let ctx = null;
    try {
      if (typeof Ctor === 'function') ctx = new Ctor();   // LR-05: inside the gesture
    } catch {
      ctx = null;
    }
    if (!ctx) {
      refuse(COPY.noAudio);
      return false;
    }
    const value = String(editor?.value ?? '');
    const caret = Number.isInteger(editor?.selectionStart) ? editor.selectionStart : value.length;
    const startOffset = readAll ? 0 : Math.max(0, Math.min(caret, value.length));   // LR-03
    const sess = {
      config: snap.config,
      ctx,
      ws: null,
      opened: false,          // the current socket fired 'open' (the server accepted it)
      handle: null,
      flow: createTextFlow(),
      connecting: false,
      created: false,
      configured: false,
      updateSent: false,
      textAccepted: false,
      stopping: false,
      released: false,
      paused: false,
      background: false,
      ctxSuspended: false,
      ctxPending: 0,          // suspend()/resume() calls of this engine that have not settled yet
      ctxTimer: null,
      serverPaused: false,
      refusal: '',            // why a Resume did not happen (the microphone records)
      fatal: null,
      lastValue: value,
      committed: startOffset, // the editor text before this offset is the server's (sent, queued or skipped)
      initialText: value.slice(startOffset),
      // This generation's text as sent (genText), the editor offset it reads from (genBase: kept so that
      // genBase + a position after the spoken part is that text's editor offset), how much of it is spoken
      // (spokenPos: the end of the latest response whose audio started), and where its commit sits (commitAt:
      // the editor offset after the initial text, or null).
      genBase: startOffset,
      genText: '',
      genCursor: { cp: 0, pos: 0 },
      spokenPos: 0,
      commitAt: null,
      // LR-10: an edit to text the server has but hasn't spoken: {resumeAt (editor offset), keepId (the
      // response playing, which plays on), sent (the response.cancel went out)}.
      edit: null,
      batch: '',
      batchTimer: null,
      keepalive: null,
      retries: 0,
      retryTimer: null,
      admissionRetry: false,
      scheduled: [],
      nextStartTime: 0,
      audioFailures: 0,
      audioWarned: false,
      responses: new Map(),
      activeResponse: null,
      speaking: false,
      listenTimer: null,
      nowText: '',
      nowResponse: '',
      reasonText: '',
      latencies: [],
      latencyMs: null,
      offs: [],
    };
    // resume() inside the gesture is what iOS needs; it is not awaited (no wait before the model check).
    setCtx(sess, false, { gesture: true });
    s = sess;
    last = idleState();
    lastPlayer = '';
    claimAudio(sess);
    listen(sess);
    publish();
    run(sess);
    return true;
  }

  function claimAudio(sess) {
    sess.handle = audio.claim({
      id: 'live',
      owner: 'live',
      kind: 'live',
      title: 'Live',
      subtitle: 'Reads text as you type',
      homeRoute: HOME_ROUTE,
      ephemeral: false,
      controls: {
        play: () => { if (current(sess) && (sess.paused || sess.background)) resume(sess); },
        pause: () => { if (current(sess) && !sess.paused) pause(sess); },
        toggle: () => { if (current(sess)) togglePause(); },
        // Another player starting stops Live (the Reader renderer waits while Live holds its slot).
        stop: (reason) => { if (s === sess) stop(reason || 'Stopped'); },
      },
      getState: () => {
        const phase = s === sess ? phaseOf(sess) : 'stopped';
        return {
          playing: s === sess && !sess.paused && !sess.background && !sess.fatal,
          waiting: phase === 'preparing' || phase === 'connecting',
          position: 0,
          duration: 0,
          rate: 1,
          canSeek: false,
          statusText: s === sess && sess.refusal ? COPY.recordingStatus : STATUS[phase],
        };
      },
    });
  }

  function listen(sess) {
    // An input method's composition (CJK, dictation) is read once it ends, not keystroke by keystroke: its
    // interim text is replaced as the listener types, and each change to sent text would cost a cancel.
    const onInput = (event) => { if (!(event && event.isComposing)) textChanged(sess); };
    const onCompositionEnd = () => textChanged(sess);
    const onVisibility = () => visibilityChanged(sess);
    const onCtxState = () => checkCtxSoon(sess);
    const onUnload = () => {
      // GS-20: the socket closes with the page.
      quietClose(sess.ws, 'Page closed');
    };
    // iOS Safari never fires beforeunload; pagehide is the event it does fire (core/mic.js binds both too).
    // A pagehide whose page is discarded (persisted: false) closes the socket and lets the session go
    // (AudioContext, audio session). A page that may come back (the back/forward cache) keeps it: the hidden
    // page already paused it (LR-16), and pageshow looks at the socket when the page returns, so a socket
    // the browser closed meanwhile ends in "Stopped: left the page", never a stale "Listening".
    const onPageHide = (event) => {
      if (!event || event.persisted !== false) return;
      quietClose(sess.ws, 'Page closed');
      if (s === sess) stop('Left the page');
    };
    const onPageShow = (event) => {
      if (!event || event.persisted !== true || s !== sess || !sess.ws) return;
      if (sess.ws.readyState >= CLOSING) stop('Left the page');
    };
    if (editor && typeof editor.addEventListener === 'function') {
      editor.addEventListener('input', onInput);
      editor.addEventListener('compositionend', onCompositionEnd);
      sess.offs.push(() => {
        editor.removeEventListener('input', onInput);
        editor.removeEventListener('compositionend', onCompositionEnd);
      });
    }
    if (sess.ctx && typeof sess.ctx.addEventListener === 'function') {
      sess.ctx.addEventListener('statechange', onCtxState);
      sess.offs.push(() => sess.ctx.removeEventListener('statechange', onCtxState));
    }
    // A Resume refused while the microphone recorded stops saying so once the recording ends (it never
    // resumes by itself: nothing ever auto-resumes).
    sess.offs.push(app.select((st) => st.recording, (recording) => {
      if (recording || !sess.refusal || !current(sess)) return;
      sess.refusal = '';
      publish();
    }));
    if (typeof document !== 'undefined' && typeof document.addEventListener === 'function') {
      document.addEventListener('visibilitychange', onVisibility);
      sess.offs.push(() => document.removeEventListener('visibilitychange', onVisibility));
    }
    if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
      window.addEventListener('beforeunload', onUnload);
      window.addEventListener('pagehide', onPageHide);
      window.addEventListener('pageshow', onPageShow);
      sess.offs.push(() => {
        window.removeEventListener('beforeunload', onUnload);
        window.removeEventListener('pagehide', onPageHide);
        window.removeEventListener('pageshow', onPageShow);
      });
    }
  }

  async function run(sess) {
    let ok = false;
    try {
      // LR-06: the model first (a cancelled switch confirm resolves false), then the socket.
      ok = await ensureReady(sess.config.model, { kind: 'tts' });
    } catch (error) {
      failStart(sess, error);
      return;
    }
    if (!current(sess)) return;
    if (!ok) {
      release(sess, { phase: 'stopped' });
      return;
    }
    sess.connecting = true;
    publish();
    await connect(sess);
  }

  /** Open the socket (a fresh ticket each time) and listen to it. */
  async function connect(sess) {
    let ws;
    try {
      ws = await openSocket(LIVE_PATH);
    } catch (error) {
      failStart(sess, error);
      return;
    }
    if (!current(sess)) {
      quietClose(ws, 'Stopped by user');
      return;
    }
    sess.ws = ws;
    sess.opened = false;
    sess.nextStartTime = safeTime(sess.ctx);
    // The server accepts a socket it refuses and closes it with 4001, 1008 or 4004 and the reason, which say why.
    // A socket that never fires 'open' and closes with 1006 (no server, or an HTTP
    // refusal of the handshake) "didn't open", it never "dropped" (api.explainClose's {opened}).
    ws.addEventListener('open', () => { if (sess.ws === ws) sess.opened = true; });
    ws.addEventListener('message', (event) => { if (sess.ws === ws) onMessage(sess, event); });
    ws.addEventListener('close', (event) => onClose(sess, ws, event));
    ws.addEventListener('error', () => {});   // a close (1006) always follows
    clearInterval(sess.keepalive);
    sess.keepalive = setInterval(() => send(sess, { type: 'session.keepalive' }), KEEPALIVE_MS);   // LR-17
    publish();
  }

  function failStart(sess, error) {
    if (!current(sess)) return;
    const info = explain(error, 'live');
    if (!info.title) {
      release(sess, { phase: 'stopped' });   // aborted: nothing to say
      return;
    }
    release(sess, {
      phase: 'error',
      error: {
        title: info.title, message: info.message, detail: info.detailText,
        action: info.action === 'open-settings' ? 'open-settings' : 'retry',
      },
    });
  }

  function quietClose(ws, reason) {
    if (!ws) return;
    try {
      if (ws.readyState < CLOSING) ws.close(1000, reason);
    } catch { /* closed */ }
  }

  function safeTime(ctx) {
    const t = Number(ctx && ctx.currentTime);
    return Number.isFinite(t) ? t : 0;
  }

  /* ── socket ── */

  function send(sess, event) {
    if (!sess.ws || sess.ws.readyState !== OPEN) return false;
    try {
      sess.ws.send(JSON.stringify(event));
      return true;
    } catch {
      return false;
    }
  }

  function pump(sess) {
    // During an edit nothing more goes out: what the server holds is cancelled, and sent again as it reads.
    if (!current(sess) || !sess.configured || sess.edit || !sess.ws || sess.ws.readyState !== OPEN) return;
    for (const event of sess.flow.drain()) send(sess, event);
  }

  function invalidData() {
    toast(COPY.invalid, { kind: 'error', id: 'live-invalid' });   // LR-22
  }

  function onMessage(sess, event) {
    if (s !== sess || sess.released) return;
    const raw = event && event.data;
    if (typeof raw !== 'string') {
      invalidData();
      return;
    }
    let data;
    try {
      data = JSON.parse(raw);
    } catch {
      invalidData();
      return;
    }
    if (!data || typeof data !== 'object' || typeof data.type !== 'string') {
      invalidData();
      return;
    }
    try {
      handle(sess, data);
      // An edit waiting for the rest of the sentence playing goes on once that has arrived (or failed).
      if (current(sess)) continueEdit(sess);
    } catch (error) {
      logError(error);
      invalidData();
    }
  }

  function handle(sess, data) {
    switch (data.type) {
      case 'session.created': {
        sess.created = true;
        sess.flow.setLimits(data.session?.limits?.max_buffer_chars);   // LR-08 (default 8192)
        // LR-23: settings go out exactly once, before any text; the server refuses a change once text
        // has been accepted (session_busy), so a repeated session.created never sends them again.
        if (!sess.updateSent && !sess.textAccepted) {
          sess.updateSent = send(sess, { type: 'session.update', session: sess.config });
        }
        return;
      }
      case 'session.updated': {
        if (sess.configured || sess.fatal) return;
        sess.configured = true;
        try { markUsed(sess.config.model); } catch { /* bookkeeping only */ }
        if (pageHidden() && !sess.paused) sess.background = true;   // LR-08: hidden -> pause at once
        syncPause(sess);
        // The text from the start offset, as it reads now (typing before this point changed it freely). The
        // server's segmenter is fresh, so a start in the middle of a line first says what comes before it on that
        // line (input_text.context, before the first append; nothing at a line start), and a leading "12.", "3)"
        // or ">" there is read as the line's continuation (as resend() does).
        const context = lineContext(sess.lastValue, sess.committed, sess.flow.state.maxBufferChars);
        if (context) send(sess, { type: 'input_text.context', text: context });
        resetGeneration(sess, sess.committed);
        if (sess.initialText) {
          queueText(sess, sess.initialText, true);
          sess.committed += sess.initialText.length;
          sess.commitAt = sess.committed;
        }
        sess.initialText = '';
        pump(sess);
        publish();
        return;
      }
      case 'input_text.accepted':
        sess.textAccepted = true;
        sess.flow.accepted(data);
        pump(sess);
        return;
      case 'session.status':
        sess.flow.status(data);
        pump(sess);
        return;
      case 'response.created': {
        const r = data.response || {};
        if (sess.edit && sess.edit.sent) return;   // the cancel for an edit is on its way: this goes with it
        if (Number.isInteger(r.generation) && r.generation !== sess.flow.generation) return;
        const id = String(r.id || '');
        sess.responses.set(id, {
          text: String(r.text || ''), createdAt: clock(), measured: false, shown: false, scheduled: false,
          // The server holds a response's audio while playback is paused: that wait is not voice latency.
          noLatency: sess.paused || sess.background,
          // Where its text ends in this generation's text: once its audio starts, the text before is spoken.
          end: genPos(sess, r.source_end),
        });
        sess.activeResponse = id;
        sess.reasonText = '';
        // LR-12 / BUG-20: the Now line shows this text at once when nothing else is playing; otherwise when
        // its own audio starts (settle()).
        if (!sess.scheduled.length) showResponse(sess, id);
        updateSpeaking(sess);
        publish();
        return;
      }
      case 'response.output_audio.delta':
        schedule(sess, data);
        return;
      case 'response.done': {
        const id = String(data.response?.id || '');
        if (sess.activeResponse === id) sess.activeResponse = null;
        if (!sess.scheduled.some((entry) => entry.responseId === id)) sess.responses.delete(id);
        updateSpeaking(sess);
        publish();
        return;
      }
      case 'response.cancelled': {
        // LR-13: the next generation; everything queued, in flight or batched is dropped.
        const edit = sess.edit;
        sess.edit = null;
        sess.flow.cancelled(data);
        clearTimeout(sess.batchTimer);
        sess.batchTimer = null;
        sess.batch = '';
        sess.activeResponse = null;
        sess.responses.clear();
        if (edit) {
          // LR-10: the cancel for an edit. The audio of the sentence playing was kept when it went out; the
          // text from the resume point goes out again, as it reads now.
          if (!edit.sent) keepOnly(sess, edit.keepId);
          resend(sess, edit.resumeAt);
        } else {
          // Not asked for by an edit: Live goes on from what is typed next (the batched text too).
          sess.committed = sess.lastValue.length;
          resetGeneration(sess, sess.committed);
          sess.commitAt = null;
          stopSources(sess);
          sess.scheduled = [];
          sess.nowText = '';
          sess.nowResponse = '';
        }
        updateSpeaking(sess);
        publish();
        return;
      }
      case 'error':
        onError(sess, data.error && typeof data.error === 'object' ? data.error : {});
        return;
      default:
        // input_text.committed / .done / .skipped, response.output_audio.done, playback.paused/.resumed,
        // session.keepalive.ack, and anything newer: nothing to do.
        return;
    }
  }

  function onError(sess, err) {
    const code = String(err.code || '');
    const message = String(err.message || 'Live error');
    if (code === 'buffer_overflow' && sess.flow.overflow()) return;   // requeued; waits for session.status
    const admission = code === 'session_busy' && !sess.created;     // the server then closes with 1013
    const draining = admission && String(err.reason || '') === 'draining';
    if (admission && retryAdmission(sess, draining)) return;
    const serverCloses = admission || SERVER_CLOSES.has(code);
    if (sess.configured && !serverCloses) {
      // LR-14 / BUG-20: one response failed (tts_error), or a refused mid-session change (a non-closing
      // session_busy): the session reads on, and the reason shows until the next response.
      dropFailedResponse(sess, String(err.response_id || ''));
      sess.reasonText = message;
      toast(message, { kind: 'warning', id: 'live-error' });
      publish();
      return;
    }
    const friendly = FLOW_COPY[code] || '';
    sess.fatal = {
      phase: admission ? 'busy' : 'error',
      message: friendly || message,
      detail: [friendly ? message : '', err.reason ? String(err.reason) : ''].filter(Boolean).join('\n'),
      code,
      draining,
    };
    publish();
    if (!serverCloses && sess.ws && sess.ws.readyState < CLOSING) {
      // An error before session.updated rejected the only session.update: the server keeps that socket
      // open and unconfigured, so it is closed here and the session released with the reason (LR-14).
      try { sess.ws.close(1000, 'Session settings rejected'); } catch { /* closing */ }
    }
  }

  /**
   * The server refused admission while it was still shutting a session down (its reason says 'draining', or
   * this engine's own session ended moments ago and the server had not finished with it): connect again
   * after the server's 1013 close, a few times, instead of reporting another Live session at once.
   */
  function retryAdmission(sess, draining) {
    const recent = clock() - lastClosedAt < OWN_CLOSE_WINDOW_MS;
    if (!(draining || recent) || sess.retries >= ADMISSION_RETRY_MS.length) return false;
    sess.admissionRetry = true;
    return true;
  }

  /**
   * A response that failed (tts_error) is over: unless some of its audio is still playing, the Now line and
   * the pill drop it at once, so the reason shows instead of the text that will not be spoken.
   */
  function dropFailedResponse(sess, id) {
    if (!id) return;
    if (sess.activeResponse === id) sess.activeResponse = null;
    if (sess.scheduled.some((entry) => entry.responseId === id)) return;
    sess.responses.delete(id);
    if (sess.nowResponse === id) {
      sess.nowText = '';
      sess.nowResponse = '';
    }
    if (!sess.activeResponse && !sess.scheduled.length) {
      clearTimeout(sess.listenTimer);
      sess.listenTimer = null;
      sess.speaking = false;
    }
  }

  function onClose(sess, ws, event) {
    if (s !== sess || sess.released || sess.ws !== ws) return;
    const code = Number(event && event.code) || 1006;
    const reason = String((event && event.reason) || '');
    if (sess.admissionRetry) {
      // Still "Connecting…": the next attempt gets a fresh ticket.
      sess.admissionRetry = false;
      sess.ws = null;
      clearInterval(sess.keepalive);
      sess.keepalive = null;
      const delay = ADMISSION_RETRY_MS[sess.retries];
      sess.retries += 1;
      sess.retryTimer = setTimeout(() => {
        sess.retryTimer = null;
        if (current(sess)) connect(sess);
      }, delay);
      return;
    }
    if (sess.fatal) {
      // The status and reason the server gave before closing stay (LR-14); an admission refusal (Busy)
      // reads as the plain copy for close code 1013 (or says the last session is still finishing), with the server's words in
      // Details.
      const busy = sess.fatal.phase === 'busy';
      let message = sess.fatal.message;
      if (busy) message = sess.fatal.draining ? COPY.draining : explainClose(1013, reason, 'live').message;
      release(sess, {
        phase: sess.fatal.phase,
        error: {
          title: COPY.title,
          message,
          detail: [busy ? sess.fatal.message : '', sess.fatal.detail].filter(Boolean).join('\n'),
          action: 'retry',
        },
      });
      return;
    }
    const opened = !!sess.opened;
    const copy = explainClose(code, reason, 'live', { opened });
    // A socket that never opened was refused or never reached the server: an error, not a disconnection.
    const phase = code === 1013 ? 'busy' : ([4004, 4001, 1008].includes(code) || !opened ? 'error' : 'disconnected');
    // A refused key or ticket points to Settings; Live turned off or a refused origin won't change on a retry
    // (api.closeAction, shared with Transcribe and Talk).
    const action = closeAction(code, reason);
    release(sess, {
      phase,
      error: { title: COPY.title, message: copy.message, detail: copy.detail ? `${copy.detail} (code ${code})` : `Code ${code}`, action },
    });
  }

  /* ── audio ── */

  function schedule(sess, ev) {
    if (!current(sess)) return;
    if (Number.isInteger(ev.generation) && ev.generation !== sess.flow.generation) return;   // stale (LR-11/13)
    if (sess.edit && sess.edit.sent) return;   // audio of what an edit's cancel drops (LR-10)
    const samples = decodePcm16(ev.delta);   // a malformed delta is the server's: "invalid data" (LR-22)
    const id = String(ev.response_id || '');
    const resp = sess.responses.get(id);
    if (resp && !resp.measured) {
      resp.measured = true;
      if (!resp.noLatency) recordLatency(sess, clock() - resp.createdAt);
    }
    let text = '';
    if (resp && !resp.scheduled) {
      resp.scheduled = true;
      text = resp.text || String(ev.text || '');
    } else if (!resp && typeof ev.text === 'string') {
      text = ev.text;
    }
    const entry = {
      responseId: id, sequence: ev.sequence, generation: ev.generation, text, shown: false, ended: false, source: null,
      end: resp ? resp.end : null, endAt: null,
    };
    const wasIdle = sess.scheduled.length === 0;
    sess.scheduled.push(entry);
    if (wasIdle) headStarted(sess);
    if (samples.length) {
      try {
        play(sess, entry, samples, ev.sample_rate);
        sess.audioFailures = 0;
      } catch (error) {
        // The browser couldn't schedule this chunk (an unsupported sample rate, a closed context): it still
        // counts as played, so its ack goes out in order and the server's flow control moves on.
        entry.ended = true;
        if (audioFailed(sess, error)) return;
      }
    } else {
      entry.ended = true;
    }
    if (wasIdle && text) {
      entry.shown = true;
      setNow(sess, text, id);
    }
    updateSpeaking(sess);
    if (entry.ended) settle(sess);
    else publish();
  }

  /** Schedule one chunk right after the previous one (LR-11). Throws, leaving nothing behind, if it can't. */
  function play(sess, entry, samples, sampleRate) {
    const ctx = sess.ctx;
    const rate = Number(sampleRate) > 0 ? Number(sampleRate) : 24000;
    const buffer = ctx.createBuffer(1, samples.length, rate);
    buffer.getChannelData(0).set(samples);
    const source = ctx.createBufferSource();
    source.buffer = buffer;
    source.connect(ctx.destination);
    const startAt = Math.max(sess.nextStartTime, safeTime(ctx) + SCHEDULE_LEAD_S);
    source.onended = () => {
      entry.ended = true;
      try { source.disconnect(); } catch { /* gone */ }
      settle(sess);
    };
    try {
      source.start(startAt);
    } catch (error) {
      source.onended = null;
      try { source.disconnect(); } catch { /* gone */ }
      throw error;
    }
    sess.nextStartTime = startAt + buffer.duration;
    entry.endAt = sess.nextStartTime;
    entry.source = source;
  }

  /**
   * A chunk the browser couldn't play: say so once (a local problem, not the server's), and stop the session
   * when it keeps happening, rather than reading on in silence. Returns true when the session was released.
   */
  function audioFailed(sess, error) {
    logError(error);
    sess.audioFailures += 1;
    if (sess.audioFailures >= MAX_AUDIO_FAILURES) {
      release(sess, {
        phase: 'error',
        error: {
          title: COPY.title, message: COPY.noAudio, detail: String((error && error.message) || error || ''), action: 'retry',
        },
      });
      return true;
    }
    if (!sess.audioWarned) {
      sess.audioWarned = true;
      toast(COPY.audioPart, { kind: 'warning', id: 'live-audio' });
    }
    return false;
  }

  /** Acks the chunks that finished playing, in FIFO order, and moves the Now line with the playback. */
  function settle(sess) {
    if (!current(sess)) return;
    while (sess.scheduled.length && sess.scheduled[0].ended) {
      const done = sess.scheduled.shift();
      const stale = Number.isInteger(done.generation) && done.generation !== sess.flow.generation;
      if (!stale && Number.isInteger(done.sequence)) {
        // LR-11: the server's flow control counts on this; it goes out only after local playback.
        send(sess, { type: 'playback.ack', response_id: done.responseId, sequence: done.sequence });
      }
      if (!sess.scheduled.some((e) => e.responseId === done.responseId) && sess.activeResponse !== done.responseId) {
        sess.responses.delete(done.responseId);
      }
    }
    headStarted(sess);
    const head = sess.scheduled[0];
    if (head && head.text && !head.shown) {
      head.shown = true;
      setNow(sess, head.text, head.responseId);
    }
    updateSpeaking(sess);
    publish();
  }

  /**
   * LR-10: the head of the playback queue plays (or starts within SCHEDULE_LEAD_S): the text of its response,
   * and everything before it, counts as spoken. An edit there takes nothing back.
   */
  function headStarted(sess) {
    const head = sess.scheduled[0];
    if (!head || !Number.isInteger(head.end)) return;
    if (Number.isInteger(head.generation) && head.generation !== sess.flow.generation) return;
    if (head.end > sess.spokenPos) sess.spokenPos = head.end;
  }

  function showResponse(sess, id) {
    const resp = sess.responses.get(id);
    if (resp && resp.text && !resp.shown) {
      resp.shown = true;
      setNow(sess, resp.text, id);
    }
  }

  function setNow(sess, text, responseId = '') {
    sess.nowText = String(text || '');
    sess.nowResponse = String(responseId || '');
    sess.reasonText = '';
  }

  function updateSpeaking(sess) {
    const busy = !!sess.activeResponse || sess.scheduled.length > 0;
    if (busy) {
      clearTimeout(sess.listenTimer);
      sess.listenTimer = null;
      sess.speaking = true;
      return;
    }
    if (!sess.speaking || sess.listenTimer) return;
    sess.listenTimer = setTimeout(() => {
      sess.listenTimer = null;
      if (!current(sess) || sess.activeResponse || sess.scheduled.length) return;
      sess.speaking = false;
      sess.nowText = '';
      sess.nowResponse = '';
      publish();
    }, LISTEN_DELAY_MS);
  }

  function recordLatency(sess, ms) {
    if (!Number.isFinite(ms) || ms < 0) return;
    sess.latencies.push(ms);
    if (sess.latencies.length > LATENCY_WINDOW) sess.latencies.shift();
    sess.latencyMs = median(sess.latencies);
  }

  function stopSources(sess, entries = sess.scheduled) {
    for (const entry of entries) {
      if (!entry.source) continue;
      try { entry.source.onended = null; } catch { /* read-only */ }
      try { entry.source.stop(); } catch { /* not started or already stopped */ }
      try { entry.source.disconnect(); } catch { /* gone */ }
    }
  }

  /* ── the AudioContext, pause, background ── */

  /**
   * Suspend or resume the AudioContext for the engine; the state changes this causes are not a system pause.
   * `gesture`: the resume was asked for inside the listener's click, so it is only ever slow, never refused.
   */
  function setCtx(sess, suspend, { gesture = false } = {}) {
    sess.ctxSuspended = suspend;
    sess.ctxPending += 1;
    let settled = false;
    const done = () => {
      if (settled) return;
      settled = true;
      sess.ctxPending = Math.max(0, sess.ctxPending - 1);
      checkCtxSoon(sess);
    };
    try {
      const p = suspend ? sess.ctx.suspend() : sess.ctx.resume();
      if (p && typeof p.then === 'function') p.then(done, done);
      else done();
    } catch {
      done();
    }
    // A resume the browser holds back (outside a gesture) may never settle: look at the state after a while.
    if (!suspend) setTimeout(done, gesture ? CTX_GESTURE_RESUME_WAIT_MS : CTX_RESUME_WAIT_MS);
  }

  function checkCtxSoon(sess) {
    if (sess.released || sess.ctxTimer) return;
    sess.ctxTimer = setTimeout(() => {
      sess.ctxTimer = null;
      reconcileCtx(sess);
    }, CTX_SETTLE_MS);
  }

  /**
   * The system (an iOS call or Siri, another app taking the audio) or the browser (a resume it
   * refused outside a gesture) stopped the AudioContext without Live asking. That counts as the listener's
   * pause: the status says Paused, the server holds the audio, and the next Resume tap (a gesture) brings
   * the sound back. Live never resumes by itself.
   */
  function reconcileCtx(sess) {
    if (!current(sess) || sess.ctxPending > 0 || sess.paused || sess.background) return;
    const state = String((sess.ctx && sess.ctx.state) || '');
    if (state !== 'suspended' && state !== 'interrupted') return;
    sess.paused = true;
    syncPause(sess);
    publish();
  }

  function syncPause(sess, { gesture = false } = {}) {
    const want = sess.paused || sess.background;
    if (want) {
      // The server holds a response's audio while playback is paused: that wait is not voice latency.
      for (const resp of sess.responses.values()) if (!resp.measured) resp.noLatency = true;
    }
    if (want !== sess.ctxSuspended) setCtx(sess, want, { gesture });
    if (sess.configured && want !== sess.serverPaused && send(sess, { type: want ? 'playback.pause' : 'playback.resume' })) {
      sess.serverPaused = want;
    }
  }

  function pause(sess) {
    sess.paused = true;
    syncPause(sess);
    publish();
  }

  /**
   * Resume (the toggle, the mini-player, the lock screen). Nothing plays while the microphone records.
   * `gesture`: called inside the listener's click (the toggles); the lock screen's Play may not be one.
   */
  function resume(sess, { gesture = false } = {}) {
    if (app.get().recording) {
      // The Resume waits; the status says why until the recording ends (no auto-resume then).
      sess.refusal = COPY.recording;
      publish();
      return false;
    }
    // A user's Resume (or Play on the lock screen) also ends a background pause.
    sess.refusal = '';
    sess.paused = false;
    sess.background = false;
    syncPause(sess, { gesture });
    publish();
    return true;
  }

  /** LR-15: Pause or Resume the running session (synchronously: iOS resumes audio only inside a gesture). */
  function togglePause() {
    const sess = s;
    if (!sess || !current(sess)) return;
    if (sess.paused) resume(sess, { gesture: true });
    else pause(sess);
  }

  function visibilityChanged(sess) {
    if (!current(sess)) return;
    if (pageHidden()) {
      if (sess.paused || sess.background) return;
      sess.background = true;                             // LR-16
      syncPause(sess);
      publish();
    } else if (sess.background) {
      sess.background = false;
      syncPause(sess);
      publish();
    }
  }

  /* ── typing ── */

  function textChanged(sess) {
    if (!current(sess)) return;
    const why = String(inputStopReason() || '');
    if (why) {
      // Speak shows Generate: what the listener types now is meant for a take (REV-31).
      stop(why);
      return;
    }
    const value = String(editor?.value ?? '');
    const before = sess.lastValue;
    if (value === before) return;
    sess.lastValue = value;
    // LR-10: an edit never stops Live. Offsets behind the edit stay; those after it move with the text. One
    // inside the replaced text goes to its end (the commit follows the initial text) or its start (Live reads
    // on from there).
    const { start, oldEnd, newEnd } = changedRange(before, value);
    const delta = newEnd - oldEnd;
    const toEnd = (x) => (x <= start ? x : x >= oldEnd ? x + delta : newEnd);
    const toStart = (x) => (x <= start ? x : x >= oldEnd ? x + delta : start);
    if (sess.commitAt !== null) sess.commitAt = toEnd(sess.commitAt);
    if (!sess.configured) {
      // Nothing was sent yet: the start offset follows the text, and the rest joins the initial text.
      sess.committed = toStart(sess.committed);
      sess.initialText = value.slice(sess.committed);
      return;
    }
    if (sess.edit) {
      // The cancel for an earlier edit is on its way: the text from the resume point goes out as it reads then.
      sess.edit.resumeAt = toStart(sess.edit.resumeAt);
      return;
    }
    if (start >= sess.committed) {
      queueBatch(sess);                                   // not sent yet: it may still change freely
      return;
    }
    const spokenEnd = sess.genBase + sess.spokenPos;
    if (oldEnd <= spokenEnd && start < spokenEnd) {
      // Only text already spoken (or before the start offset) changed. It can't be unsaid: Live reads on, and
      // what it has yet to say keeps its place in the text.
      sess.genBase += delta;
      sess.committed += delta;
      queueBatch(sess);
      return;
    }
    // Text the server has but hasn't spoken changed: what it holds is cancelled and sent again as it now
    // reads, from the end of what was spoken (or from the edit, when the edit begins in spoken text).
    beginEdit(sess, Math.min(start, spokenEnd));
  }

  /** The text after what the server has goes out in one batch, BATCH_MS after the first change. */
  function queueBatch(sess) {
    const pending = sess.lastValue.slice(sess.committed);
    sess.batch = pending;
    if (pending && !sess.batchTimer) sess.batchTimer = setTimeout(() => flushBatch(sess), BATCH_MS);
  }

  function flushBatch(sess) {
    clearTimeout(sess.batchTimer);
    sess.batchTimer = null;
    if (!current(sess) || sess.edit) return;
    const text = sess.batch;
    sess.batch = '';
    if (!text) return;
    queueText(sess, text);
    sess.committed += text.length;
    pump(sess);
  }

  /** A new generation of text, read from the editor offset `base`. */
  function resetGeneration(sess, base) {
    sess.genBase = base;
    sess.genText = '';
    sess.genCursor = { cp: 0, pos: 0 };
    sess.spokenPos = 0;
  }

  function queueText(sess, text, commit = false) {
    if (!text) return;
    sess.flow.queue(text, commit);
    sess.genText += text;
  }

  /**
   * The position in this generation's text (UTF-16) of a server source offset, which counts code points as
   * Python does (src/live_tts/segmenter.py). Offsets only grow within a generation, so it reads on from the last.
   */
  function genPos(sess, codePoints) {
    if (!Number.isInteger(codePoints) || codePoints < 0) return null;
    const text = sess.genText;
    let { cp, pos } = sess.genCursor;
    if (codePoints < cp) {
      cp = 0;
      pos = 0;
    }
    while (cp < codePoints && pos < text.length) {
      const pair = isHighSurrogate(text.charCodeAt(pos)) && pos + 1 < text.length && isLowSurrogate(text.charCodeAt(pos + 1));
      pos += pair ? 2 : 1;
      cp += 1;
    }
    sess.genCursor = { cp, pos };
    return pos;
  }

  /** LR-10: an edit to text the server holds unspoken. The response playing (the head) plays on. */
  function beginEdit(sess, resumeAt) {
    clearTimeout(sess.batchTimer);
    sess.batchTimer = null;
    sess.batch = '';
    const head = sess.scheduled[0];
    sess.edit = { resumeAt, keepId: head ? head.responseId : null, sent: false };
    continueEdit(sess);
  }

  /**
   * Send the edit's response.cancel, unless the response playing is still arriving: a cancel now would cut it
   * off, so it waits for that response's response.done (or failure). Then only that response's audio stays.
   */
  function continueEdit(sess) {
    const edit = sess.edit;
    if (!edit || edit.sent) return;
    if (edit.keepId !== null && sess.activeResponse === edit.keepId) return;
    edit.sent = true;
    keepOnly(sess, edit.keepId);
    send(sess, { type: 'response.cancel' });
    updateSpeaking(sess);
    publish();
  }

  /** Stop the scheduled audio of every response but `keepId`; the next audio follows what is kept. */
  function keepOnly(sess, keepId) {
    const keep = keepId === null ? [] : sess.scheduled.filter((entry) => entry.responseId === keepId);
    stopSources(sess, sess.scheduled.filter((entry) => !keep.includes(entry)));
    sess.scheduled = keep;
    const ends = keep.map((entry) => entry.endAt).filter(Number.isFinite);
    sess.nextStartTime = ends.length ? Math.max(...ends) : safeTime(sess.ctx);
    if (!keep.length) {
      sess.nowText = '';
      sess.nowResponse = '';
    }
  }

  /**
   * After an edit's cancel: send the text from `at` as it reads now, the initial text with its commit again. The
   * server reads it with a fresh segmenter, as a line start; from mid-line, `input_text.context` (the text before
   * it on its line) goes first, so a leading "12.", "3)", ">" or "-" there is read as the line's continuation.
   * It is not spoken and counts in no offset.
   */
  function resend(sess, at) {
    const value = sess.lastValue;
    const from = Math.max(0, Math.min(at, value.length));
    const context = lineContext(value, from, sess.flow.state.maxBufferChars);
    if (context) send(sess, { type: 'input_text.context', text: context });
    resetGeneration(sess, from);
    if (sess.commitAt !== null && sess.commitAt > from) {
      queueText(sess, value.slice(from, sess.commitAt), true);
      queueText(sess, value.slice(sess.commitAt));
    } else {
      sess.commitAt = null;
      queueText(sess, value.slice(from));
    }
    sess.committed = value.length;
    pump(sess);
  }

  /* ── stop and release ── */

  /**
   * LR-18: stop the scheduled audio, cancel the response, close the socket (1000 "Stopped by user") and
   * release. The status reads "Stopped", or "Stopped: <reason>" for any other reason.
   * @param {string} [reason]
   */
  function stop(reason = 'Stopped') {
    const sess = s;
    if (!sess || sess.released) return;
    sess.stopping = true;
    stopSources(sess);
    if (sess.created) send(sess, { type: 'response.cancel' });
    quietClose(sess.ws, 'Stopped by user');
    const { status, reason: why } = stoppedStatus(reason);
    release(sess, { phase: 'stopped', status, reason: why });
  }

  function release(sess, { phase = 'stopped', status = '', reason = '', error = null } = {}) {
    if (sess.released) return;
    sess.released = true;
    sess.stopping = true;
    if (sess.created) lastClosedAt = clock();   // the server may still be shutting this session down
    clearInterval(sess.keepalive);
    clearTimeout(sess.batchTimer);
    clearTimeout(sess.listenTimer);
    clearTimeout(sess.retryTimer);
    clearTimeout(sess.ctxTimer);
    stopSources(sess);
    sess.scheduled = [];
    for (const off of sess.offs) off();
    sess.offs = [];
    quietClose(sess.ws, 'Stopped');
    try {
      if (sess.ctx && sess.ctx.state !== 'closed') {
        const p = sess.ctx.close();
        if (p && typeof p.catch === 'function') p.catch(() => {});
      }
    } catch { /* closed */ }
    if (s === sess) s = null;
    last = idleState({ phase, status: status || STATUS[phase] || STATUS.stopped, reason, error });
    if (sess.handle) sess.handle.release();
    publish();
    if (error && app.get().route.path !== HOME_ROUTE) {
      // Away from Speak there is no nearby control: the failure goes in a toast.
      toast(`${error.title}. ${error.message}`, { kind: 'error', id: 'live-stopped' });
    }
  }

  return {
    start,
    stop,
    togglePause,
    isActive: () => !!s,
    getState,
    check,
    /** Dismiss the error banner of the last session. */
    clearError() {
      if (s || !last.error) return;
      last = idleState({ ...plain(last), error: null });
      publish();
    },
    /** Clear a Start refusal (the settings changed). */
    clearRefusal() {
      if (s || !last.refusal) return;
      last = idleState({ ...plain(last), refusal: '' });
      publish();
    },
    refuse,
    destroy() {
      if (s) stop('Stopped');
      destroyed = true;
    },
  };
}
