/**
 * Talk's session engine: one GA /v1/realtime conversation with the AI assistant at a
 * time, spoken or typed.
 *
 * - start() runs inside the click: it makes the AudioContext the answers play on and resumes it, opens the
 *   microphone (core/mic.js, owner 'talk', the browser's echo cancellation on) so its context is made inside the
 *   gesture too, and claims a 'live' audio session (core/audio.js; another player starting stops Talk). Then it
 *   makes the voice model ready (models.ensureReady) and opens the socket with api.openSocket (a fresh ticket
 *   when OS_API_KEY is set). No subprotocol, so the session is GA.
 * - session.created -> one session.update with the choices (lib sessionUpdate); session.updated -> ready. A
 *   refused update before that stops Talk with the server's reason; later, it reports the reason and keeps what
 *   the server last accepted. A session with no turn detection (the server couldn't load its VAD) turns the
 *   microphone off: typing still works.
 * - The microphone streams 100 ms appends of 24 kHz PCM16 while the session is ready; frames before that are
 *   dropped. Turning it off mid-turn sends 800 ms of silence, so the server's VAD ends the turn.
 * - A spoken turn: speech_started / speech_stopped / the transcript (your words), then the answer: waiting
 *   phrases (open_speech.assistant.filler), the thinking (open_speech.assistant.thinking), the answer's text
 *   (response.output_audio_transcript.delta, before its audio), its audio (base64 PCM16, played back to back on
 *   the AudioContext, as views/speak/live-session.js does), response.done. Errors keep the session: an AI error
 *   is logged as the preset's spoken sentence.
 * - A typed question: a running answer is cancelled first (response.cancel and a flush); then
 *   conversation.item.create, and response.create only once conversation.item.added names the item. A refusal
 *   (an error with its event_id) is reported and the text is kept. Typing while Talk is stopped starts it with the
 *   microphone off.
 * - Keeping the conversation: the server ends a session after OS_REALTIME_IDLE_TIMEOUT_S without a client event,
 *   so once nothing was sent for 30 s an empty session.update goes out. A socket that drops anyway (or the idle
 *   timeout's close 4008) is opened again quietly, up to three times, and the log says "Reconnected: the
 *   conversation started over." (the server keeps a conversation per connection). Closes a retry can't fix
 *   (a refused key, Realtime turned off, too many sessions, a first socket that never opened) stop Talk.
 * - Interrupt by talking off: once your turn is committed (input_audio_buffer.committed), a typed question is sent
 *   or an answer starts, the microphone's audio is dropped, not sent (`micHeld`; the microphone stays open), through
 *   the thinking, the phrases and the whole spoken answer. It is sent again once the answer is done and its sound
 *   has played, plus HOLD_TAIL_MS; at once after Stop reply, a refused question, an answer that failed with nothing
 *   left to play, Interrupt turned on, or a new connection. A turn being spoken as it starts ends with silence.
 * - Pause (the mini-player, the lock screen, the microphone of another view) suspends the AudioContext, so the
 *   answer's sound waits; Resume (a click) brings it back. Talk's own microphone opening never pauses it.
 * - Stop, another player, leaving the page: the socket closes, the microphone and the AudioContext are released,
 *   every timer and listener goes.
 * - Delete chat: while a conversation runs it is newConversation(); while Talk is stopped clearLog()
 *   empties the chat on screen and restoreLog() is its Undo.
 * - A cut-off answer: a streamed reply that an error or the token limit stopped after some of it was
 *   spoken ends `incomplete` (response.done, the code in status_details.error.code). What was said
 *   stays, and a note right under it says so in plain words (logic cutOffNote); it is announced politely once that
 *   answer's sound has played (or was stopped), so it never talks over the answer.
 * Injected for tests: AudioContextCtor, openSocket, openMic, ensureReady, clock, announce. No DOM or browser API at
 * import time.
 * @module views/talk/session
 */
import { app } from '../../core/store.js';
import { openSocket as apiOpenSocket, explain, explainClose, closeAction } from '../../core/api.js';
import { ensureReady as modelsEnsureReady } from '../../core/models.js';
import * as audio from '../../core/audio.js';
import { open as micOpen } from '../../core/mic.js';
import { announce as a11yAnnounce } from '../../core/a11y.js';
import { decodePcm16 } from '../speak/live-session.js';
import {
  REALTIME_PATH, AUDIO_RATE, TEXT_MAX, SEND_SAMPLES, KEEPALIVE_MS, KEEPALIVE_CHECK_MS, RECONNECT_MS, MAX_LOG,
  END_TURN_SILENCE_MS, HOLD_TAIL_MS, COPY, createResampler, appendEvent, sessionUpdate, keepalive, typedQuestion, micProblem,
  errorEntry, cutOffNote, statusOf, recordingRefusal,
} from './logic.js';

export const HOME_ROUTE = 'talk';
/** Audio is scheduled this far ahead of the context's clock. */
export const SCHEDULE_LEAD_S = 0.025;

const OPEN = 1;
const CLOSING = 2;
/** Closes a reconnect can't fix: a refused key or origin, Realtime turned off, too many sessions. */
const NO_RETRY = new Set([4001, 1008, 4004, 1013]);

/**
 * PURE (but for api.explainClose's key check): what a close means for Talk.
 * @param {number} code
 * @param {string} reason
 * @param {{opened?: boolean}} [options]
 * @returns {{message: string, detail: string, action: string, retryable: boolean}}
 */
export function closeInfo(code, reason = '', { opened = true } = {}) {
  const c = Number(code) || 1006;
  const r = String(reason || '');
  const detail = r ? `${r} (code ${c})` : `Code ${c}`;
  if (c === 4004) return { message: COPY.disabled, detail, action: '', retryable: false };
  if (c === 1013) return { message: COPY.busy, detail, action: 'retry', retryable: false };
  const copy = explainClose(c, r, 'talk', { opened });
  const action = closeAction(c, r);   // shared with Live and Transcribe
  return { message: copy.message, detail, action, retryable: !NO_RETRY.has(c) };
}

function logError(error) {
  if (globalThis.console) globalThis.console.error(error);
}

function safeTime(ctx) {
  const t = Number(ctx && ctx.currentTime);
  return Number.isFinite(t) ? t : 0;
}

function quietClose(ws, reason) {
  if (!ws) return;
  try {
    if (ws.readyState < CLOSING) ws.close(1000, reason);
  } catch { /* closed */ }
}

/**
 * @param {object} [options]
 * @param {() => {model?: string, voice?: string, preset?: string, assistant?: string, interrupt?: boolean, presetInfo?: object}} [options.getChoices]
 *   the voice, the AI preset (and its view, for the error sentences) and the interrupt switch, read at each start
 *   and update
 * @param {(state: object) => void} [options.onChange] every state change (see getState())
 * @param {Function} [options.AudioContextCtor] default: window.AudioContext (or webkitAudioContext)
 * @param {(path: string) => Promise<WebSocket>} [options.openSocket] default: api.openSocket
 * @param {(options: object) => Promise<object>} [options.openMic] default: core/mic.js open()
 * @param {(model: string, options: object) => Promise<boolean>} [options.ensureReady] default: models.ensureReady
 * @param {() => number} [options.clock] milliseconds
 * @param {(text: string) => void} [options.announce] a polite announcement (default: core/a11y.js announce)
 * @param {() => (Promise<unknown>|null)} [options.choicesReady] a choice the view is still checking (a remembered
 *   voice): start() waits for this promise before the model check and the connection, then reads the choices
 *   again; null when nothing is being checked
 */
export function createTalkEngine({
  getChoices = () => ({}),
  choicesReady = () => null,
  onChange = () => {},
  AudioContextCtor = null,
  openSocket = apiOpenSocket,
  openMic = micOpen,
  ensureReady = modelsEnsureReady,
  clock = () => Date.now(),
  announce = a11yAnnounce,
} = {}) {
  let s = null;                 // the running session
  let log = [];                 // the conversation log, kept across sessions (frozen entries)
  let entrySeq = 0;
  let sessions = 0;             // sessions started (the second one on adds a note to an old log)
  let last = { phase: 'stopped', stopReason: '', error: null, refusal: '', micProblem: null };
  let typedSent = 0;
  let destroyed = false;
  let lastSignature = '';
  let lastPlayer = '';

  const current = (sess) => s === sess && !sess.released;

  /* ── state ── */

  function choices() {
    let c = {};
    try { c = getChoices() || {}; } catch (error) { logError(error); c = {}; }
    return {
      model: String(c.model || ''), voice: String(c.voice || ''), preset: String(c.preset || 'voice'),
      assistant: c.assistant === null ? null : String(c.assistant || ''), interrupt: c.interrupt !== false, presetInfo: c.presetInfo || null,
      textOnly: c.textOnly === true,
    };
  }

  function plainChoices(c) {
    return c ? Object.freeze({ model: c.model, voice: c.voice, preset: c.preset, assistant: c.assistant, interrupt: c.interrupt, textOnly: c.textOnly }) : null;
  }

  function replyOf(sess) {
    if (sess.reply) return !sess.reply.answered ? 'thinking' : sess.choices.textOnly ? 'writing' : 'speaking';
    return sess.sources.length ? 'speaking' : 'idle';
  }

  function stateOf(sess) {
    const base = {
      active: true,
      phase: sess.phase,
      mic: sess.vadUnavailable ? 'unavailable' : sess.micState,
      micHeld: sess.held && !!sess.mic,
      reply: replyOf(sess),
      hearing: sess.hearing,
      transcribing: sess.transcribing,
      interim: sess.interim,
      paused: sess.paused,
      refusal: sess.refusal,
      stopReason: '',
      error: null,
      micProblem: sess.micProblem,
      typed: Object.freeze({ busy: !!(sess.typedPending || sess.typedQueue !== null), error: sess.typedError, sent: typedSent }),
      choiceError: sess.choiceError,
      accepted: plainChoices(sess.accepted),
      log,
    };
    return Object.freeze({ ...base, status: Object.freeze(statusOf(base)) });
  }

  function idleState() {
    const base = {
      active: false, phase: 'stopped', mic: 'off', micHeld: false, reply: 'idle', hearing: false, transcribing: false, interim: '',
      paused: false, refusal: last.refusal || '', stopReason: last.stopReason || '', error: last.error || null,
      micProblem: last.micProblem || null, typed: Object.freeze({ busy: false, error: '', sent: typedSent }),
      choiceError: '', accepted: null, log,
    };
    return Object.freeze({ ...base, status: Object.freeze(statusOf(base)) });
  }

  function getState() {
    return s ? stateOf(s) : idleState();
  }

  function publish() {
    const state = getState();
    if (s && s.handle) {
      const player = `${state.phase}|${state.paused}|${state.reply}|${state.mic}|${state.refusal}`;
      if (player !== lastPlayer) {
        lastPlayer = player;
        try { s.handle.notify(); } catch (error) { logError(error); }
      }
    }
    const signature = JSON.stringify([state, log.length, entrySeq]);
    if (signature === lastSignature) return;
    lastSignature = signature;
    try { onChange(state); } catch (error) { logError(error); }
  }

  /* ── the log ── */

  /** Add an entry at the end of the log, or right after the entry `extra.after` names. */
  function addEntry(kind, text, extra = {}) {
    entrySeq += 1;
    const entry = Object.freeze({ id: `e${entrySeq}`, kind, text: String(text || ''), detail: String(extra.detail || ''), state: extra.state || '', typed: !!extra.typed });
    const at = extra.after ? log.findIndex((e) => e.id === extra.after) : -1;
    const next = at >= 0 ? [...log.slice(0, at + 1), entry, ...log.slice(at + 1)] : [...log, entry];
    log = Object.freeze(next.length > MAX_LOG ? next.slice(next.length - MAX_LOG) : next);
    return entry.id;
  }

  function patchEntry(id, patch) {
    if (!id) return;
    let changed = false;
    const next = log.map((e) => {
      if (e.id !== id) return e;
      changed = true;
      return Object.freeze({ ...e, ...patch });
    });
    if (changed) log = Object.freeze(next);
  }

  /* ── start and stop ── */

  function refuse(message) {
    last = { ...last, refusal: String(message || ''), error: null };
    publish();
    return false;
  }

  /**
   * Start a conversation (call it synchronously inside the click). Returns false when it refused or one runs.
   * @param {{mic?: boolean}} [options] mic false: typing only (the microphone can be turned on later)
   */
  function start({ mic: withMic = true } = {}) {
    if (s || destroyed) return false;
    const recording = recordingRefusal(app.get().recording);
    if (recording) return refuse(recording);
    const Ctor = AudioContextCtor || globalThis.AudioContext || globalThis.webkitAudioContext;
    let ctx = null;
    try {
      if (typeof Ctor === 'function') ctx = new Ctor();   // inside the gesture (iOS)
    } catch {
      ctx = null;
    }
    if (!ctx) return refuse(COPY.noAudio);
    const sess = {
      choices: choices(), accepted: null, ctx, ws: null, opened: false, created: false, configured: false,
      everConfigured: false, phase: 'preparing', attempts: 0, retryTimer: null, reconnected: false,
      updates: [], updateSeq: 0, vadUnavailable: false, keepalive: null, lastSentAt: clock(),
      micState: 'off', mic: null, micWanted: false, micOpening: 0, micOff: null, micToken: 0, resampler: null,
      pending: [], pendingLen: 0, micProblem: null, held: false, holdTimer: null,
      handle: null, paused: false, ourSuspend: false, ctxRan: false, refusal: '', fatal: null,
      reply: null, cancelled: new Set(), aiEntries: new Map(), responseErrors: new Set(),
      sources: [], nextStartTime: 0, cutOffs: [],
      hearing: false, transcribing: false, interim: '',
      typedSeq: 0, typedPending: null, typedQueue: null, typedError: '', choiceError: '',
      offs: [], released: false,
    };
    try {
      const p = ctx.resume();
      if (p && typeof p.catch === 'function') p.catch(() => {});
    } catch { /* resumed later by a click */ }
    s = sess;
    last = { phase: 'stopped', stopReason: '', error: null, refusal: '', micProblem: null };
    lastPlayer = '';
    sessions += 1;
    if (sessions > 1 && log.length) addEntry('note', COPY.newSession);
    // The microphone first: its open() pauses whatever plays, then the claim displaces it.
    if (withMic) openMicrophone(sess);
    claimAudio(sess);
    listen(sess);
    publish();
    run(sess);
    return true;
  }

  function claimAudio(sess) {
    sess.handle = audio.claim({
      id: 'talk',
      owner: 'talk',
      kind: 'live',
      title: 'Talk',
      subtitle: 'Talking with the AI assistant',
      homeRoute: HOME_ROUTE,
      ephemeral: false,
      controls: {
        play: () => { if (current(sess)) resume(); },
        pause: () => { if (current(sess) && !sess.micOpening) pause(sess); },
        toggle: () => { if (!current(sess)) return; if (sess.paused) resume(); else pause(sess); },
        // Another player starting stops Talk.
        stop: (reason) => { if (s === sess) stop(reason || 'Stopped'); },
      },
      getState: () => {
        const st = s === sess ? stateOf(sess) : idleState();
        return {
          playing: s === sess && st.phase === 'ready' && !sess.paused,
          waiting: st.phase === 'preparing' || st.phase === 'connecting' || st.phase === 'reconnecting',
          position: 0,
          duration: 0,
          rate: 1,
          canSeek: false,
          statusText: sess.refusal ? COPY.recordingStatus : st.status.label,
        };
      },
    });
  }

  function listen(sess) {
    const onPageHide = (event) => {
      if (event && event.persisted === true) return;   // kept in the back/forward cache: pageshow looks again
      quietClose(sess.ws, 'Page closed');
      if (s === sess) stop('Left the page');
    };
    const onUnload = () => quietClose(sess.ws, 'Page closed');
    const onPageShow = (event) => {
      if (!event || event.persisted !== true || s !== sess) return;
      if (!sess.ws || sess.ws.readyState >= CLOSING) stop('Left the page');
    };
    const onCtxState = () => {
      if (!current(sess)) return;
      const state = String((sess.ctx && sess.ctx.state) || '');
      if (state === 'running') {
        sess.ctxRan = true;
        return;
      }
      // The system (an iOS call, another app) or the browser stopped the sound without Talk asking: a pause.
      if ((state === 'suspended' || state === 'interrupted') && sess.ctxRan && !sess.ourSuspend && !sess.paused) {
        sess.paused = true;
        publish();
      }
    };
    if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') {
      window.addEventListener('pagehide', onPageHide);
      window.addEventListener('beforeunload', onUnload);
      window.addEventListener('pageshow', onPageShow);
      sess.offs.push(() => {
        window.removeEventListener('pagehide', onPageHide);
        window.removeEventListener('beforeunload', onUnload);
        window.removeEventListener('pageshow', onPageShow);
      });
    }
    if (sess.ctx && typeof sess.ctx.addEventListener === 'function') {
      if (sess.ctx.state === 'running') sess.ctxRan = true;   // its resume() in the click may have settled already
      sess.ctx.addEventListener('statechange', onCtxState);
      sess.offs.push(() => sess.ctx.removeEventListener('statechange', onCtxState));
    }
    // A Resume refused while another view recorded stops saying so once that recording ends.
    sess.offs.push(app.select((st) => st.recording, (recording) => {
      if (!current(sess) || !sess.refusal || recordingRefusal(recording)) return;
      sess.refusal = '';
      publish();
    }));
  }

  async function run(sess) {
    let ok = true;
    // A choice the view is still checking (a remembered voice that may be gone): wait, then read the choices again,
    // so the first session.update never carries a voice the check is about to drop.
    let waiting = null;
    try { waiting = choicesReady(); } catch (error) { logError(error); }
    if (waiting && typeof waiting.then === 'function') {
      try { await waiting; } catch { /* the check failed: the choices stand */ }
      if (!current(sess)) return;
      sess.choices = choices();
      publish();
    }
    try {
      if (sess.choices.model) ok = await ensureReady(sess.choices.model, { kind: 'tts' });
    } catch (error) {
      failStart(sess, error);
      return;
    }
    if (!current(sess)) return;
    if (!ok) {
      release(sess);
      return;
    }
    await connect(sess);
  }

  function failStart(sess, error) {
    if (!current(sess)) return;
    const info = explain(error, 'talk');
    if (!info.title) {
      release(sess);   // aborted: nothing to say
      return;
    }
    release(sess, {
      error: { title: COPY.title, message: info.message, detail: info.detailText || '', action: info.action === 'open-settings' ? 'open-settings' : 'retry' },
    });
  }

  /** Open a socket (a fresh ticket each time) and listen to it. */
  async function connect(sess) {
    sess.phase = sess.everConfigured ? 'reconnecting' : 'connecting';
    publish();
    let ws;
    try {
      ws = await openSocket(REALTIME_PATH);
    } catch (error) {
      if (!current(sess)) return;
      if (sess.everConfigured && retry(sess)) return;
      failStart(sess, error);
      return;
    }
    if (!current(sess)) {
      quietClose(ws, 'Stopped');
      return;
    }
    sess.ws = ws;
    sess.opened = false;
    sess.created = false;
    sess.configured = false;
    sess.updates = [];
    // A refused socket opens, then closes with 4001, 1008 or 4004 and the reason (src/routes/realtime.py):
    // closeInfo says why, and NO_RETRY keeps a reconnect from trying again. A socket
    // that never fires 'open' and closes with 1006 (no server, or an HTTP refusal of the handshake) "didn't open".
    ws.addEventListener('open', () => { if (sess.ws === ws) sess.opened = true; });
    ws.addEventListener('message', (event) => { if (sess.ws === ws && current(sess)) onMessage(sess, event); });
    ws.addEventListener('close', (event) => onClose(sess, ws, event));
    ws.addEventListener('error', () => {});   // a close (1006) always follows
    if (!sess.keepalive) {
      sess.keepalive = setInterval(() => {
        if (!current(sess) || !sess.configured) return;
        // The server answers it with session.updated like any update, in order: it holds a place in the queue.
        if (clock() - sess.lastSentAt >= KEEPALIVE_MS && send(sess, keepalive())) sess.updates.push({ eventId: '', choices: null });
      }, KEEPALIVE_CHECK_MS);
    }
    publish();
  }

  /** Schedule the next reconnect; false once the tries are used up. */
  function retry(sess) {
    if (sess.attempts >= RECONNECT_MS.length) return false;
    const wait = RECONNECT_MS[sess.attempts];
    sess.attempts += 1;
    sess.phase = 'reconnecting';
    clearTimeout(sess.retryTimer);
    sess.retryTimer = setTimeout(() => {
      sess.retryTimer = null;
      if (current(sess)) connect(sess);
    }, wait);
    publish();
    return true;
  }

  function onClose(sess, ws, event) {
    if (s !== sess || sess.released || sess.ws !== ws) return;
    const code = Number(event && event.code) || 1006;
    const reason = String((event && event.reason) || '');
    const opened = !!sess.opened;
    sess.ws = null;
    resetConversation(sess);
    if (sess.fatal) {
      // Talk closed it after the server refused the session (its settings, or an error before it was ready).
      release(sess, { error: sess.fatal });
      return;
    }
    const info = closeInfo(code, reason, { opened });
    if (sess.everConfigured && info.retryable) {
      sess.reconnected = true;
      if (retry(sess)) return;
      release(sess, { error: { title: COPY.title, message: COPY.dropped, detail: info.detail, action: 'retry' } });
      return;
    }
    release(sess, { error: { title: COPY.title, message: info.message, detail: info.detail, action: info.action } });
  }

  /** What a connection held: the answer, its sound, the turn being spoken, a question waiting for its answer. */
  function resetConversation(sess) {
    sess.created = false;
    sess.configured = false;
    sess.updates = [];
    sess.cutOffs = [];   // the chat they were about goes, or the conversation started over
    if (sess.reply) {
      sess.cancelled.add(sess.reply.id);
      sess.reply = null;
    }
    flush(sess);
    sess.hearing = false;
    sess.transcribing = false;
    sess.interim = '';
    sess.typedPending = null;
    sess.pending = [];
    sess.pendingLen = 0;
    unhold(sess);
  }

  /* ── the microphone held while it answers (Interrupt by talking off) ── */

  function hold(sess) {
    clearTimeout(sess.holdTimer);
    sess.holdTimer = null;
    if (sess.held || sess.choices.interrupt) return;
    if (sess.hearing && sess.configured) {
      // A turn being spoken ends with silence, so the server's voice detection isn't left mid-speech.
      send(sess, appendEvent(new Float32Array(Math.round((AUDIO_RATE * END_TURN_SILENCE_MS) / 1000))));
      sess.hearing = false;
    }
    sess.held = true;
    sess.pending = [];
    sess.pendingLen = 0;
  }

  function unhold(sess) {
    clearTimeout(sess.holdTimer);
    sess.holdTimer = null;
    if (!sess.held) return;
    sess.held = false;
    sess.pending = [];
    sess.pendingLen = 0;
  }

  /** Listen again once nothing is answering or playing: after HOLD_TAIL_MS, or `now`. */
  function settleHold(sess, { now = false } = {}) {
    if (!sess.held || sess.reply || sess.sources.length || sess.typedPending) return;
    if (now) {
      unhold(sess);
      return;
    }
    if (sess.holdTimer) return;
    sess.holdTimer = setTimeout(() => {
      sess.holdTimer = null;
      if (!current(sess) || sess.reply || sess.sources.length || sess.typedPending) return;
      unhold(sess);
      publish();
    }, HOLD_TAIL_MS);
  }

  /* ── the socket ── */

  function send(sess, event) {
    if (!sess.ws || sess.ws.readyState !== OPEN) return false;
    try {
      sess.ws.send(JSON.stringify(event));
      sess.lastSentAt = clock();
      return true;
    } catch {
      return false;
    }
  }

  function sendUpdate(sess) {
    sess.updateSeq += 1;
    const eventId = `evt_update_${sess.updateSeq}`;
    const snapshot = sess.choices;
    if (send(sess, sessionUpdate(snapshot, { turnDetection: !sess.vadUnavailable, eventId }))) {
      sess.updates.push({ eventId, choices: snapshot });
    }
  }

  function onMessage(sess, event) {
    let data;
    try {
      data = JSON.parse(typeof event.data === 'string' ? event.data : '');
    } catch {
      return;   // not JSON: nothing Talk reads
    }
    if (!data || typeof data !== 'object') return;
    try {
      handle(sess, data);
    } catch (error) {
      logError(error);
    }
    if (current(sess)) publish();
  }

  function handle(sess, data) {
    switch (data.type) {
      case 'session.created':
        sess.created = true;
        sendUpdate(sess);
        return;
      case 'session.updated':
        onUpdated(sess, data.session || {});
        return;
      case 'input_audio_buffer.speech_started':
        sess.hearing = true;
        sess.interim = '';
        return;
      case 'input_audio_buffer.speech_stopped':
        if (sess.hearing) sess.transcribing = true;
        sess.hearing = false;
        return;
      case 'conversation.item.input_audio_transcription.delta':
        sess.interim += String(data.delta || '');
        return;
      case 'conversation.item.input_audio_transcription.completed': {
        sess.transcribing = false;
        sess.interim = '';
        const words = String(data.transcript || '').trim();
        if (words) addEntry('you', words);
        // A turn with words gets its response.created next; one that gets no answer listens again.
        settleHold(sess, { now: !words });
        return;
      }
      case 'conversation.item.input_audio_transcription.failed': {
        sess.transcribing = false;
        sess.interim = '';
        const err = data.error && typeof data.error === 'object' ? data.error : {};
        addEntry('error', COPY.transcriptionFailed, { detail: err.message });
        settleHold(sess, { now: true });
        return;
      }
      case 'conversation.item.added':
        onItemAdded(sess, data.item || {});
        return;
      case 'input_audio_buffer.committed':
        hold(sess);
        return;
      case 'response.created': {
        const id = String((data.response && data.response.id) || '');
        sess.reply = { id, answered: false };
        hold(sess);
        return;
      }
      case 'open_speech.assistant.filler':
        if (!live(sess, data.response_id)) return;
        addEntry('filler', data.text);
        return;
      case 'open_speech.assistant.thinking':
        if (!live(sess, data.response_id)) return;
        addEntry('thinking', data.text);
        return;
      case 'response.output_audio_transcript.delta':
      case 'response.audio_transcript.delta':
      case 'response.output_text.delta':
      case 'response.text.delta': {
        const id = String(data.response_id || '');
        if (!live(sess, id)) return;
        const entry = sess.aiEntries.get(id);
        if (entry) {
          const cur = log.find((e) => e.id === entry);
          patchEntry(entry, { text: `${cur ? cur.text : ''}${String(data.delta || '')}` });
        } else {
          sess.aiEntries.set(id, addEntry('ai', String(data.delta || '')));
        }
        if (sess.reply && sess.reply.id === id) sess.reply.answered = true;
        return;
      }
      case 'response.output_audio_transcript.done':
      case 'response.audio_transcript.done':
      case 'response.output_text.done':
      case 'response.text.done': {
        const id = String(data.response_id || '');
        const entry = sess.aiEntries.get(id);
        const text = typeof data.transcript === 'string' ? data.transcript : data.text;
        if (entry && typeof text === 'string' && !sess.cancelled.has(id)) patchEntry(entry, { text });
        return;
      }
      case 'response.output_audio.delta':
      case 'response.audio.delta':
        if (!live(sess, data.response_id)) return;
        schedule(sess, data);
        return;
      case 'response.done':
        onResponseDone(sess, data.response || {});
        return;
      case 'error':
        onError(sess, data.error && typeof data.error === 'object' ? data.error : {});
        return;
      default:
        // input_audio_buffer.cleared, conversation.item.done, response.output_item.*,
        // response.content_part.*, response.output_audio.done, conversation.item.truncated: nothing to do.
        return;
    }
  }

  /** Whether a response's events still count (not cancelled by Talk or the server). */
  function live(sess, responseId) {
    return !sess.cancelled.has(String(responseId || ''));
  }

  function onUpdated(sess, session) {
    const answered = sess.updates.shift();
    if (answered && answered.choices) sess.accepted = answered.choices;
    const input = session.audio && session.audio.input;
    if (input && 'turn_detection' in input) {
      sess.vadUnavailable = input.turn_detection === null;
      if (sess.vadUnavailable) noVoiceDetection(sess);
    }
    if (!sess.configured) {
      sess.configured = true;
      sess.everConfigured = true;
      sess.attempts = 0;
      sess.phase = 'ready';
      if (sess.reconnected) addEntry('note', COPY.reconnected);
      sess.reconnected = false;
      if (sess.typedQueue !== null) {
        const text = sess.typedQueue;
        sess.typedQueue = null;
        sendTyped(sess, text);
      }
    }
  }

  function noVoiceDetection(sess) {
    if (sess.mic || sess.micState !== 'off') closeMicrophone(sess, { endTurn: false });
    sess.micWanted = false;
    sess.micProblem = { title: COPY.micTitle, message: COPY.vad };
  }

  function onItemAdded(sess, item) {
    const pending = sess.typedPending;
    if (!pending || item.id !== pending.itemId) return;   // a spoken turn's item, or one Talk doesn't wait for
    sess.typedPending = null;
    sess.typedError = '';
    typedSent += 1;
    addEntry('you', pending.text, { typed: true });
    send(sess, { type: 'response.create' });
  }

  function onResponseDone(sess, response) {
    const id = String(response.id || '');
    const status = String(response.status || '');
    const failed = status === 'failed' || sess.responseErrors.has(id);
    if (status === 'cancelled') {
      const reason = String((response.status_details && response.status_details.reason) || '');
      sess.cancelled.add(id);
      flush(sess, id);
      const entry = sess.aiEntries.get(id);
      if (entry) patchEntry(entry, { state: reason === 'turn_detected' ? 'interrupted' : 'stopped' });
    } else if (status === 'failed' && !sess.responseErrors.has(id)) {
      const code = response.status_details && response.status_details.error && response.status_details.error.code;
      addEntry('error', COPY.serverError, { detail: code ? `Code ${code}` : '' });
    } else if (status === 'incomplete' && live(sess, id) && !sess.responseErrors.has(id)) {
      // What was said stays; a note right under it says it was cut off, announced once it has played.
      const code = response.status_details && response.status_details.error && response.status_details.error.code;
      const text = cutOffNote(code, sess.choices.presetInfo);
      addEntry('note', text, { state: 'cut-off', after: sess.aiEntries.get(id) });
      sess.cutOffs.push({ responseId: id, text });
      announceCutOffs(sess);
    }
    sess.responseErrors.delete(id);
    sess.aiEntries.delete(id);
    if (sess.reply && sess.reply.id === id) sess.reply = null;
    // A failed answer with nothing (more) to say listens again at once; any other once its sound has played.
    settleHold(sess, { now: failed });
    // Old cancelled ids are dropped once they can no longer arrive.
    if (sess.cancelled.size > 16) sess.cancelled = new Set([...sess.cancelled].slice(-8));
  }

  function onError(sess, err) {
    const code = String(err.code || '');
    const message = String(err.message || '');
    const eventId = String(err.event_id || '');
    if (code === 'vad_unavailable') {
      sess.vadUnavailable = true;
      noVoiceDetection(sess);
      return;
    }
    if (eventId && sess.typedPending && eventId === sess.typedPending.eventId) {
      // The question was refused; its text stays in the box and nothing asks the AI.
      sess.typedPending = null;
      sess.typedError = message || COPY.serverError;
      settleHold(sess, { now: true });
      return;
    }
    const update = eventId ? sess.updates.findIndex((u) => u.eventId === eventId) : -1;
    if (update >= 0) {
      sess.updates.splice(update, 1);
      if (!sess.configured) {
        sess.fatal = { title: COPY.title, message: COPY.settingsRefused, detail: message, action: 'retry' };
        quietClose(sess.ws, 'Session settings rejected');
        return;
      }
      sess.choiceError = message || COPY.settingsRefused;
      return;
    }
    if (code === 'idle_timeout') return;                 // the close that follows reconnects quietly
    if (code === 'response_cancel_not_active') return;  // the answer ended before the cancel
    if (code === 'transcription_error') return;         // the transcription.failed event said it
    if (!sess.configured && !sess.everConfigured) {
      sess.fatal = { title: COPY.title, message: message || COPY.serverError, detail: code ? `Code ${code}` : '', action: 'retry' };
      quietClose(sess.ws, 'Session error');
      return;
    }
    if (sess.reply) sess.responseErrors.add(sess.reply.id);
    const preset = sess.choices.presetInfo;
    const entry = errorEntry(err, preset);
    addEntry('error', entry.text, { detail: entry.detail });
    settleHold(sess, { now: true });
  }

  /* ── typed questions ── */

  /**
   * Ask a typed question (call it inside the click or the Enter key). While Talk is stopped it starts a session
   * with the microphone off and asks once it is ready. Returns false for blank or too long text.
   * @param {string} text
   */
  function ask(text) {
    const words = String(text || '');
    if (!words.trim() || words.length > TEXT_MAX || destroyed) return false;
    if (!s) {
      if (!start({ mic: false })) return false;
      s.typedQueue = words;
      publish();
      return true;
    }
    const sess = s;
    sess.typedError = '';
    if (!sess.configured) {
      sess.typedQueue = words;
      publish();
      return true;
    }
    sendTyped(sess, words);
    publish();
    return true;
  }

  function sendTyped(sess, words) {
    // A running answer is cancelled first (the server answers its response.done before it
    // reads the next event, so the question can follow at once).
    if (sess.reply) cancelRunning(sess);
    hold(sess);
    sess.typedSeq += 1;
    const q = typedQuestion(words, sess.typedSeq);
    if (send(sess, q.event)) sess.typedPending = { ...q, text: words };
    else settleHold(sess, { now: true });
  }

  function cancelRunning(sess) {
    const id = sess.reply ? sess.reply.id : '';
    send(sess, { type: 'response.cancel' });
    if (id) sess.cancelled.add(id);
    flush(sess);
    sess.reply = null;
  }

  /** Stop the answer the AI is giving (or still thinking about). */
  function cancelReply() {
    const sess = s;
    if (!sess || !current(sess) || (!sess.reply && !sess.sources.length)) return;
    if (sess.reply) cancelRunning(sess);
    else flush(sess);
    unhold(sess);
    publish();
  }

  /* ── the microphone ── */

  function openMicrophone(sess) {
    if (sess.vadUnavailable) return false;
    sess.micWanted = true;
    sess.micProblem = null;
    sess.micState = 'starting';
    const token = (sess.micToken += 1);
    // core/mic.js pauses all playback as it opens: that pause is not the listener's, so Talk ignores it.
    sess.micOpening += 1;
    let opening;
    try {
      opening = Promise.resolve(openMic({ owner: 'talk', echoCancellation: true, noiseSuppression: true }));
    } catch (error) {
      opening = Promise.reject(error);
    }
    opening.then((m) => {
      sess.micOpening = Math.max(0, sess.micOpening - 1);
      if (!current(sess) || !sess.micWanted || token !== sess.micToken || sess.vadUnavailable) {
        m.stop();
        if (current(sess) && token === sess.micToken) {
          sess.micState = 'off';
          publish();
        }
        return;
      }
      sess.mic = m;
      sess.resampler = createResampler(m.sampleRate, AUDIO_RATE);
      sess.micOff = m.onFrame((frame) => onFrame(sess, frame));
      sess.micState = 'on';
      publish();
    }, (error) => {
      sess.micOpening = Math.max(0, sess.micOpening - 1);
      if (!current(sess) || token !== sess.micToken) return;
      sess.micWanted = false;
      sess.micState = 'off';
      sess.micProblem = micProblem(error);
      publish();
    });
    return true;
  }

  function closeMicrophone(sess, { endTurn = true } = {}) {
    sess.micWanted = false;
    sess.micToken += 1;   // an open() still on its way stops its capture when it arrives
    if (endTurn && sess.hearing && sess.configured) {
      // The turn being spoken ends with silence, so the server's VAD commits it (and the AI answers it).
      send(sess, appendEvent(new Float32Array(Math.round((AUDIO_RATE * END_TURN_SILENCE_MS) / 1000))));
      sess.hearing = false;
    }
    if (sess.micOff) {
      try { sess.micOff(); } catch { /* gone */ }
      sess.micOff = null;
    }
    if (sess.mic) {
      try { sess.mic.stop(); } catch (error) { logError(error); }
      sess.mic = null;
    }
    sess.resampler = null;
    sess.pending = [];
    sess.pendingLen = 0;
    sess.micState = 'off';
  }

  /**
   * Turn the microphone on (inside the click: its AudioContext must start in the gesture) or off.
   * @param {boolean} on
   * @returns {boolean} false when it can't (no session, or the server has no voice detection)
   */
  function setMic(on) {
    const sess = s;
    if (!sess || !current(sess)) return false;
    if (on) {
      if (sess.vadUnavailable) return false;
      if (sess.micState === 'on' || sess.micState === 'starting') return true;
      openMicrophone(sess);
      publish();
      return true;
    }
    closeMicrophone(sess);
    publish();
    return true;
  }

  function onFrame(sess, frame) {
    if (!current(sess) || !sess.resampler) return;
    if (!sess.configured || !sess.ws || sess.ws.readyState !== OPEN || sess.vadUnavailable || sess.held) {
      sess.pending = [];
      sess.pendingLen = 0;
      return;
    }
    const samples = sess.resampler(frame);
    if (!samples.length) return;
    sess.pending.push(samples);
    sess.pendingLen += samples.length;
    while (sess.pendingLen >= SEND_SAMPLES) {
      const chunk = new Float32Array(SEND_SAMPLES);
      let filled = 0;
      while (filled < SEND_SAMPLES) {
        const head = sess.pending[0];
        const take = Math.min(head.length, SEND_SAMPLES - filled);
        chunk.set(head.subarray(0, take), filled);
        filled += take;
        if (take === head.length) sess.pending.shift();
        else sess.pending[0] = head.subarray(take);
      }
      sess.pendingLen -= SEND_SAMPLES;
      send(sess, appendEvent(chunk));
    }
  }

  /* ── the answers' sound ── */

  function schedule(sess, ev) {
    const id = String(ev.response_id || '');
    let samples;
    try {
      samples = decodePcm16(ev.delta);
    } catch (error) {
      logError(error);
      return;
    }
    if (!samples.length) return;
    const ctx = sess.ctx;
    try {
      const buffer = ctx.createBuffer(1, samples.length, AUDIO_RATE);
      buffer.getChannelData(0).set(samples);
      const source = ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(ctx.destination);
      const startAt = Math.max(sess.nextStartTime, safeTime(ctx) + SCHEDULE_LEAD_S);
      const entry = { source, responseId: id };
      source.onended = () => {
        try { source.disconnect(); } catch { /* gone */ }
        sess.sources = sess.sources.filter((x) => x !== entry);
        announceCutOffs(sess);
        settleHold(sess);
        if (current(sess)) publish();
      };
      source.start(startAt);
      sess.nextStartTime = startAt + buffer.duration;
      sess.sources.push(entry);
    } catch (error) {
      logError(error);   // a chunk the browser can't schedule is skipped
    }
  }

  /** Stop the sound: every chunk, or one response's. */
  function flush(sess, responseId = '') {
    const keep = [];
    for (const entry of sess.sources) {
      if (responseId && entry.responseId !== responseId) {
        keep.push(entry);
        continue;
      }
      try { entry.source.onended = null; } catch { /* read-only */ }
      try { entry.source.stop(); } catch { /* not started or stopped */ }
      try { entry.source.disconnect(); } catch { /* gone */ }
    }
    sess.sources = keep;
    if (!keep.length) sess.nextStartTime = safeTime(sess.ctx);
    announceCutOffs(sess);   // a sound stopped (Stop reply) no longer holds its note back
  }

  /**
   * Announce, politely, the note of each cut-off answer whose sound has played: never over the
   * answer, which the listener is hearing. A stopped session or a new conversation drops what still waits.
   */
  function announceCutOffs(sess) {
    if (!sess.cutOffs.length || !current(sess)) return;
    const waiting = [];
    for (const cut of sess.cutOffs) {
      if (sess.sources.some((x) => x.responseId === cut.responseId)) {
        waiting.push(cut);
        continue;
      }
      try { announce(cut.text); } catch (error) { logError(error); }
    }
    sess.cutOffs = waiting;
  }

  function setCtx(sess, suspend) {
    sess.ourSuspend = suspend;
    try {
      const p = suspend ? sess.ctx.suspend() : sess.ctx.resume();
      if (p && typeof p.catch === 'function') p.catch(() => {});
    } catch { /* closed */ }
  }

  function pause(sess) {
    if (sess.paused) return;
    sess.paused = true;
    setCtx(sess, true);
    publish();
  }

  /** Resume the answers' sound (a click: iOS resumes audio only inside a gesture). */
  function resume() {
    const sess = s;
    if (!sess || !current(sess)) return false;
    const refusal = recordingRefusal(app.get().recording);
    if (refusal) {
      sess.refusal = refusal;
      publish();
      return false;
    }
    sess.refusal = '';
    sess.paused = false;
    setCtx(sess, false);
    publish();
    return true;
  }

  /* ── stop ── */

  /**
   * Stop the conversation: the socket closes (the server forgets it), the microphone and the sound are released.
   * @param {string} [reason] "Stopped", or why ("Another player started", "Left the page")
   */
  function stop(reason = 'Stopped') {
    const sess = s;
    if (!sess || sess.released) return;
    quietClose(sess.ws, 'Stopped by user');
    release(sess, { stopReason: reason === 'Stopped' ? '' : String(reason || '') });
  }

  function release(sess, { error = null, stopReason = '' } = {}) {
    if (sess.released) return;
    sess.released = true;
    clearInterval(sess.keepalive);
    clearTimeout(sess.retryTimer);
    unhold(sess);
    sess.keepalive = null;
    sess.retryTimer = null;
    closeMicrophone(sess, { endTurn: false });
    flush(sess);
    for (const off of sess.offs) {
      try { off(); } catch { /* gone */ }
    }
    sess.offs = [];
    const ws = sess.ws;
    sess.ws = null;
    quietClose(ws, 'Stopped');
    try {
      if (sess.ctx && sess.ctx.state !== 'closed') {
        const p = sess.ctx.close();
        if (p && typeof p.catch === 'function') p.catch(() => {});
      }
    } catch { /* closed */ }
    if (s === sess) s = null;
    last = { phase: 'stopped', stopReason, error, refusal: '', micProblem: error ? null : sess.micProblem };
    if (sess.handle) {
      try { sess.handle.release(); } catch (e) { logError(e); }
    }
    publish();
  }

  /** New conversation: the server forgets the old one with its socket; the log is emptied; the microphone stays. */
  function newConversation() {
    const sess = s;
    if (!sess || !current(sess)) return;
    const ws = sess.ws;
    sess.ws = null;   // its close is not a drop
    quietClose(ws, 'New conversation');
    resetConversation(sess);
    clearTimeout(sess.retryTimer);
    sess.retryTimer = null;
    sess.attempts = 0;
    sess.reconnected = false;
    sess.everConfigured = false;
    sess.typedQueue = null;
    sess.typedError = '';
    sess.choiceError = '';
    log = Object.freeze([]);
    publish();
    connect(sess);
  }

  /**
   * Delete chat with Talk stopped: the chat on screen empties. Nothing is sent or
   * stored. While a conversation runs, Delete chat is newConversation() instead, and this does nothing.
   * @returns {ReadonlyArray<object>|null} the entries deleted (for restoreLog, the Undo), or null when nothing was
   */
  function clearLog() {
    if (s || !log.length) return null;
    const removed = log;
    log = Object.freeze([]);
    publish();
    return removed;
  }

  /**
   * Undo Delete chat: the entries come back while Talk is still stopped and the chat still empty.
   * @param {ReadonlyArray<object>} entries what clearLog() returned
   * @returns {boolean}
   */
  function restoreLog(entries) {
    if (s || destroyed || log.length || !Array.isArray(entries) || !entries.length) return false;
    log = Object.freeze(entries.slice(-MAX_LOG));
    publish();
    return true;
  }

  /** The choices changed (assistant, preset, voice, interrupt switch): tell a ready session. */
  function update() {
    const sess = s;
    if (!sess || !current(sess)) return;
    sess.choices = choices();
    sess.choiceError = '';
    // Interrupt by talking turned on mid-answer listens at once; turned off, it holds from the next turn.
    if (sess.choices.interrupt) unhold(sess);
    if (sess.configured) sendUpdate(sess);
    publish();
  }

  return {
    start,
    stop,
    setMic,
    ask,
    cancelReply,
    newConversation,
    clearLog,
    restoreLog,
    update,
    resume,
    getState,
    isActive: () => !!s,
    /** Dismiss a typed question's refusal (the text changed). */
    clearTyped() {
      if (s && s.typedError) {
        s.typedError = '';
        publish();
      }
    },
    /** Dismiss the last session's stop banner. */
    clearError() {
      if (s || !last.error) return;
      last = { ...last, error: null };
      publish();
    },
    /** Dismiss the microphone banner. */
    clearMicProblem() {
      if (s && s.micProblem && !s.vadUnavailable) s.micProblem = null;
      else if (!s) last = { ...last, micProblem: null };
      publish();
    },
    destroy() {
      if (s) stop('Stopped');
      destroyed = true;
    },
  };
}
