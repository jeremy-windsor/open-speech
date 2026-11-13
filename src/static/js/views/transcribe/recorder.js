/**
 * Live transcription from the microphone: a state machine over an injected microphone, model
 * check and socket (the real core modules by default), so the Node harness drives it unchanged.
 *
 *   idle → requesting (permission) → starting (model check, socket, the server's session.begin) → recording
 *     → stopping → done
 *   a failure → error (with a banner); cancel → idle
 *
 * - start() is refused while a session is active, and the phase changes before the first await, so a
 *   double click never opens two microphones or two sockets (BUG-08).
 * - The microphone opens first, inside the click (iOS lets the AudioContext run), and frames captured while
 *   the model check and the socket are still on their way are kept (up to 20 s) and sent on open, so the
 *   first words are never lost. From the open on, frames go straight out.
 * - Recording starts when the server begins the session (session.begin: the model and the VAD are loaded),
 *   not when the socket opens. A failure before it (the model or VAD didn't load, a stream refused right
 *   after it was accepted) is a start that failed: no "Recording started", and the transcript on screen is
 *   never touched. The recording's clock starts at the first sound the server gets (the kept audio
 *   included), so the timer, the 30 s discard confirm, "Recorded m:ss" and the segment times agree, and a
 *   model download before it never counts.
 * - PCM16 frames go out at the real AudioContext rate: ?sample_rate=<rate>&interim_results=true&model=<id>,
 *   plus a fresh one-time ticket from api.openSocket (never the key).
 * - Server events are read, not ignored (BUG-02): session.begin, transcript, vad, error (fatal before
 *   session.begin; after it a notice, while the stream goes on: a failure that persists repeats on every
 *   chunk, so each different notice is reported once per session) and session.end, which ends a graceful
 *   stop. A session.end nobody asked for (no Stop) is the server ending the session, after a failure: the close
 *   that follows decides (1011 is a disconnect; 1000, or none within the grace, a quiet end).
 * - stop() is graceful (TR-10): capture stops, {type:'stop'} goes out, and the socket stays up to 4 s for
 *   the last segments; one timer per socket. cancel() closes at once.
 * - Events from an earlier session's socket are ignored (TR-11).
 * - Nothing heard: NOTHING_HEARD_MS after the server began the session with no
 *   speech yet (no vad speech_start, no words), state.nothingHeard says why: 'silence' when no sound came in at
 *   all (no frame, or every frame under one 16-bit step: the server gets zeros), else 'no-speech'. Speech clears
 *   it; a session that ends with nothing heard keeps it, so the view can say so after Stop.
 * @module views/transcribe/recorder
 */
import * as micModule from '../../core/mic.js';
import * as modelsModule from '../../core/models.js';
import * as apiModule from '../../core/api.js';
import { floatToInt16, rms, peak, toDb } from '../../core/pcm.js';

export const STOP_GRACE_MS = 4000;
export const BUFFER_SECONDS = 20;
export const STREAM_PATH = '/v1/audio/stream';
/** How long a recording may hear no speech before the view says so. */
export const NOTHING_HEARD_MS = 4000;
/** A frame whose loudest sample is under one 16-bit step reaches the server as zeros: no sound at all. */
export const SILENT_PEAK = 1 / 32768;
/** The phases in which a session holds the microphone or the socket. */
export const ACTIVE = Object.freeze(['requesting', 'starting', 'recording', 'stopping']);

const OPEN = 1;
const CLOSING = 2;

const BACKGROUND = 'Recording stopped when the app went to the background.';

/**
 * The title of a start that failed, whatever stopped it (the server's error before session.begin, a socket
 * closed or refused before it): api.explain()'s own title for a file or a start request that failed, so one
 * failure reads one way ("Couldn't transcribe").
 */
export const START_FAILED_TITLE = apiModule.BANNER_TITLES.transcribe;

/**
 * PURE: the title of a server error while the recording goes on (a notice), saying what happened: the
 * server's message says why. A failed chunk ("Transcription failed: …") lost that part of the recording, and
 * the recording goes on; anything else is the server's own note.
 * @param {string} message
 * @returns {string}
 */
export function noticeTitle(message) {
  return /transcription failed/i.test(String(message || ''))
    ? "Part of the recording wasn't transcribed"
    : 'The server reported a problem';
}

/**
 * PURE: what makes two server notices the same: the words, not the numbers in them. A failure that persists is
 * sent again with every chunk, often with numbers or addresses that change ("CUDA out of memory. Tried to
 * allocate 20.00 MiB", "illegal memory access at 0x7f3a…").
 * @param {string} message
 * @returns {string}
 */
export function noticeKey(message) {
  return String(message || '').toLowerCase()
    .replace(/\b0x[0-9a-f]+\b/g, '#')
    .replace(/\d+(?:[.,]\d+)*/g, '#')
    .replace(/\s+/g, ' ')
    .trim();
}

/**
 * @param {object} [deps]
 * @param {object} [deps.mic] core/mic.js
 * @param {object} [deps.models] core/models.js
 * @param {object} [deps.api] core/api.js (openSocket, explain, explainClose, closeAction)
 * @param {Record<string, (detail: *) => void>} [deps.on] listeners: state, started, level, begin, transcript,
 *   vad, notice (a non-fatal server error), end ({reason: 'stopped'|'cancelled'|'error', duration, begun});
 *   started fires when the server begins the session (the state is 'recording' from then on)
 * @returns {{start: Function, stop: Function, cancel: Function, readonly state: object, isActive: () => boolean}}
 */
export function createRecorder({ mic = micModule, models = modelsModule, api = apiModule, on = {} } = {}) {
  let state = Object.freeze({
    phase: 'idle', model: '', startedAt: 0, stoppedAt: 0, sampleRate: 0, sessionId: '', vadEnabled: false,
    vad: '', error: null, nothingHeard: '',
  });
  let run = null;   // the session in progress: its token, microphone, socket, timers and buffer
  let seq = 0;

  const emit = (type, detail) => {
    const fn = on[type];
    if (typeof fn !== 'function') return;
    try { fn(detail); } catch (error) { if (globalThis.console) globalThis.console.error(error); }
  };

  function set(patch) {
    state = Object.freeze({ ...state, ...patch });
    emit('state', state);
  }

  const isActive = () => ACTIVE.includes(state.phase);
  const current = (my) => run === my;

  function stopCapture(my) {
    if (my.offFrame) { my.offFrame(); my.offFrame = null; }
    for (const off of my.offs.splice(0)) {
      try { off(); } catch { /* gone */ }
    }
    if (my.session) {
      try { my.session.stop(); } catch { /* already stopped */ }
      my.session = null;
    }
  }

  function closeSocket(my) {
    const ws = my.ws;
    if (ws && ws.readyState < CLOSING) {
      try { ws.close(1000); } catch { /* closing */ }
    }
  }

  function teardown(my) {
    clearTimeout(my.stopTimer);
    my.stopTimer = null;
    clearTimeout(my.quietTimer);
    my.quietTimer = null;
    stopCapture(my);
    closeSocket(my);
    my.buffer = [];
  }

  // Seconds of audio the server got: from the first sound sent (the kept audio included) to the stop.
  function duration(my) {
    const end = state.stoppedAt || Date.now();
    return my.firstSampleAt ? Math.max(0, (end - my.firstSampleAt) / 1000) : 0;
  }

  function fail(my, error) {
    if (!current(my)) return;
    const seconds = duration(my);
    teardown(my);
    run = null;
    set({ phase: error ? 'error' : 'idle', error: error || null, vad: '', nothingHeard: '', stoppedAt: state.stoppedAt || Date.now() });
    emit('end', { reason: 'error', duration: seconds, error, begun: my.begun });
  }

  function finish(my) {
    if (!current(my)) return;
    const seconds = duration(my);
    // The model just worked: Record again within 20 s opens the socket with no status check first.
    if (my.begun && state.model && typeof models.markUsed === 'function') models.markUsed(state.model);
    teardown(my);
    run = null;
    set({ phase: 'done', vad: '', error: my.notice || state.error || null, stoppedAt: state.stoppedAt || Date.now() });
    emit('end', { reason: 'stopped', duration: seconds, begun: my.begun });
  }

  function onFrame(my, frame) {
    if (!current(my) || !frame) return;
    const loudest = peak(frame);
    emit('level', { db: toDb(rms(frame)), peak: loudest });
    if (loudest > my.loudest) my.loudest = loudest;
    // "No sound at all" was said, and sound comes in now: the question is the level.
    if (state.nothingHeard === 'silence' && loudest >= SILENT_PEAK) set({ nothingHeard: 'no-speech' });
    const pcm = floatToInt16(frame);
    const ws = my.ws;
    if (my.opened && ws && ws.readyState === OPEN) {
      try { ws.send(pcm.buffer); } catch { /* the close event reports it */ }
      return;
    }
    // Before the socket is open: keep the audio (bounded) and send it on open.
    my.buffer.push(pcm);
    my.buffered += pcm.length;
    const cap = BUFFER_SECONDS * (my.sampleRate || 48000);
    while (my.buffered > cap && my.buffer.length > 1) my.buffered -= my.buffer.shift().length;
  }

  function onOpen(my) {
    if (!current(my) || my.opened || state.phase !== 'starting') return;
    my.opened = true;
    // The server hears the kept audio first, so the recording's clock starts where that audio started (at most
    // BUFFER_SECONDS back: a model download before the socket is never counted).
    my.firstSampleAt = Date.now() - Math.round((my.buffered / (my.sampleRate || 48000)) * 1000);
    const ws = my.ws;
    for (const chunk of my.buffer.splice(0)) {
      try { ws.send(chunk.buffer); } catch { break; }
    }
    my.buffered = 0;
  }

  /**
   * The server began the session: the model and the VAD are loaded and it reads the audio. Recording starts
   * here, not at the socket's open: a failure before this point never recorded anything.
   */
  function begin(my, msg = {}) {
    if (my.begun) return;
    my.begun = true;
    // The model answered: the next Record within 20 s opens the socket with no status check.
    if (state.model && typeof models.markUsed === 'function') models.markUsed(state.model);
    const starting = state.phase === 'starting';
    set({
      sessionId: String(msg.session_id || ''),
      vadEnabled: msg.vad_enabled === true,
      model: msg.model || state.model,
      ...(starting ? { phase: 'recording', startedAt: my.firstSampleAt || Date.now() } : {}),
    });
    if (starting) {
      // The server reads the audio from here on: a few seconds with no speech is worth saying.
      clearTimeout(my.quietTimer);
      my.quietTimer = setTimeout(() => checkHeard(my), NOTHING_HEARD_MS);
      emit('started', state);
    }
    emit('begin', msg);
  }

  /**
   * NOTHING_HEARD_MS into the recording: no speech yet. 'silence' when no sound came in at all (no
   * frame, or none with a sample of at least one 16-bit step), else 'no-speech' (a level the server's VAD never
   * called speech: too quiet, the wrong microphone, or nobody talking yet).
   */
  function checkHeard(my) {
    my.quietTimer = null;
    if (!current(my) || state.phase !== 'recording' || my.heard) return;
    set({ nothingHeard: my.loudest >= SILENT_PEAK ? 'no-speech' : 'silence' });
  }

  /** The server heard speech (a vad speech_start, or words): nothing more to say about the microphone. */
  function heard(my) {
    my.heard = true;
    clearTimeout(my.quietTimer);
    my.quietTimer = null;
    if (state.nothingHeard) set({ nothingHeard: '' });
  }

  function onMessage(my, event) {
    if (!current(my)) return;   // a stale socket from an earlier session (TR-11)
    if (typeof event.data !== 'string') return;
    let msg;
    try { msg = JSON.parse(event.data); } catch { return; }
    if (!msg || typeof msg !== 'object') return;
    switch (msg.type) {
      case 'session.begin':
        begin(my, msg);
        break;
      case 'transcript':
        begin(my);   // a server that skipped session.begin has begun all the same
        if (String(msg.text || '').trim()) heard(my);
        emit('transcript', msg);
        break;
      case 'vad': {
        begin(my, { vad_enabled: true });
        if (msg.state === 'speech_start') heard(my);
        const vad = msg.state === 'speech_start' ? 'speech' : 'silence';
        if (vad !== state.vad) set({ vad });
        emit('vad', msg);
        break;
      }
      case 'error': {
        const message = String(msg.message || 'The server reported an error.');
        // Before session.begin (the model or VAD didn't load) the stream never started: end it, and check the
        // model's status before the next try instead of trusting the inventory (the fast path goes off).
        if (!my.begun) {
          if (state.model && typeof models.markFailed === 'function') models.markFailed(state.model);
          fail(my, { kind: 'stream', title: START_FAILED_TITLE, message, detail: '' });
          break;
        }
        // After it the stream goes on (a notice). A failure that persists comes again with every chunk, up to
        // ten a second: each different notice is reported once per session, never as a stream of alerts.
        const key = noticeKey(message);
        if (my.notices.has(key)) break;
        my.notices.add(key);
        const error = { kind: 'notice', title: noticeTitle(message), message, detail: '' };
        set({ error });
        emit('notice', error);
        break;
      }
      case 'session.end':
        if (state.phase === 'stopping' && !my.serverEnded) finish(my);
        else if (state.phase === 'recording') serverEnded(my);
        break;
      default:
        break;
    }
  }

  /**
   * session.end while still recording: the user never pressed Stop, so the server ended the session. It does that
   * after a failure, then closes 1011. Capture stops (the server reads no more), and the
   * close decides: 1000, or no close within STOP_GRACE_MS, ends it as a Stop would; any other code is a disconnect.
   */
  function serverEnded(my) {
    my.serverEnded = true;
    stopCapture(my);
    set({ phase: 'stopping', stoppedAt: Date.now(), vad: '' });
    clearTimeout(my.stopTimer);
    my.stopTimer = setTimeout(() => finish(my), STOP_GRACE_MS);
  }

  function onClose(my, event) {
    if (!current(my)) return;
    const code = event?.code;
    if (state.phase === 'stopping' && !(my.serverEnded && code !== 1000)) {
      finish(my);
      return;
    }
    // The server accepts a socket it refuses (a wrong key or ticket, an Origin not allowed, a bad sample rate), then
    // closes it with the code and reason, which say why. An HTTP refusal of the
    // handshake, or no server, reaches the browser as 1006 before `open`: it "didn't open"; it never "dropped"
    // (api.explainClose's `opened`). `action`: what helps (api.closeAction): no Retry for an address not allowed.
    const copy = api.explainClose(code, event?.reason || '', 'transcribe', { opened: my.opened });
    fail(my, {
      kind: 'close',
      code,
      action: typeof api.closeAction === 'function' ? api.closeAction(code, event?.reason || '') : 'retry',
      // Before session.begin it is a start that failed, not a disconnect: a socket that never opened, or one
      // refused right after it was accepted (1013: too many at once).
      title: my.begun ? 'The microphone stream disconnected' : START_FAILED_TITLE,
      message: copy.message,
      detail: copy.detail || '',
    });
  }

  // The capture can end under us: a track that ends (the device was unplugged) or an iOS Home Screen app
  // that lost the microphone in the background.
  function watchCapture(my) {
    const session = my.session;
    let wasHidden = typeof document !== 'undefined' && !!document.hidden;
    const lost = () => {
      if (!current(my) || !(state.phase === 'recording' || state.phase === 'starting')) return;
      const message = wasHidden || (typeof document !== 'undefined' && document.hidden)
        ? BACKGROUND
        : 'The microphone stopped. Check that it is still connected, then record again.';
      my.notice = { kind: 'mic', reason: message === BACKGROUND ? 'background' : 'ended', title: 'Recording stopped', message, detail: '' };
      if (state.phase === 'recording') {
        set({ error: my.notice });
        stop();
      } else {
        fail(my, my.notice);
      }
    };
    const tracks = session?.stream && typeof session.stream.getTracks === 'function' ? session.stream.getTracks() : [];
    for (const track of tracks) {
      if (typeof track.addEventListener !== 'function') continue;
      track.addEventListener('ended', lost);
      my.offs.push(() => track.removeEventListener('ended', lost));
    }
    if (typeof document !== 'undefined' && typeof document.addEventListener === 'function') {
      const onVisibility = () => {
        if (document.hidden) {
          wasHidden = true;
          return;
        }
        const dead = session && (session.active === false || tracks.some((t) => t.readyState === 'ended'));
        if (dead) lost();
        else wasHidden = false;
      };
      document.addEventListener('visibilitychange', onVisibility);
      my.offs.push(() => document.removeEventListener('visibilitychange', onVisibility));
    }
  }

  function classify(error) {
    if (error && error.name === 'MicBusyError') return { kind: 'busy', title: 'Microphone problem', message: error.message, detail: '' };
    if (error && error.name === 'MicError') return { kind: 'mic', reason: error.reason, title: 'Microphone problem', message: error.message, detail: '' };
    const ex = api.explain(error, 'transcribe');
    if (!ex.title) return null;   // aborted: nothing to say
    return { kind: 'start', title: ex.title, message: ex.message, detail: ex.detailText, action: ex.action, retryable: ex.retryable };
  }

  /**
   * Start a session. Refused (false) while one is active.
   * @param {{model: string, deviceId?: string, echoCancellation?: boolean, noiseSuppression?: boolean}} options
   * @returns {Promise<boolean>} true once the socket is opening (recording starts at the server's session.begin)
   */
  async function start({ model, deviceId, echoCancellation = true, noiseSuppression = true } = {}) {
    if (isActive()) return false;
    const my = {
      token: (seq += 1), session: null, ws: null, offFrame: null, offs: [], buffer: [], buffered: 0, opened: false, begun: false,
      stopTimer: null, serverEnded: false, sampleRate: 0, firstSampleAt: 0, notice: null, notices: new Set(),
      quietTimer: null, heard: false, loudest: 0,
    };
    run = my;
    set({ phase: 'requesting', model: String(model || ''), error: null, startedAt: 0, stoppedAt: 0, sampleRate: 0, sessionId: '', vadEnabled: false, vad: '', nothingHeard: '' });
    try {
      const session = await mic.open({ owner: 'transcribe', deviceId, echoCancellation, noiseSuppression });
      if (!current(my)) {
        session.stop();
        return false;
      }
      my.session = session;
      my.sampleRate = Math.round(session.sampleRate || 48000);
      my.offFrame = session.onFrame((frame) => onFrame(my, frame));
      watchCapture(my);
      set({ phase: 'starting', sampleRate: my.sampleRate });
      const ready = await models.ensureReady(state.model, { kind: 'stt' });
      if (!current(my)) return false;
      if (ready === false) {
        fail(my, null);
        return false;
      }
      const query = new URLSearchParams({ sample_rate: String(my.sampleRate), interim_results: 'true' });
      if (state.model) query.set('model', state.model);
      const ws = await api.openSocket(STREAM_PATH, query);
      if (!current(my)) {
        try { ws.close(1000); } catch { /* closing */ }
        return false;
      }
      my.ws = ws;
      try { ws.binaryType = 'arraybuffer'; } catch { /* read-only in some fakes */ }
      ws.addEventListener('open', () => onOpen(my));
      ws.addEventListener('message', (event) => onMessage(my, event));
      ws.addEventListener('close', (event) => onClose(my, event));
      if (ws.readyState === OPEN) onOpen(my);
      return true;
    } catch (error) {
      if (current(my)) fail(my, classify(error));
      return false;
    }
  }

  /**
   * Stop gracefully: capture stops at once, {type:'stop'} goes out and the socket stays up to 4 s for the
   * last segments (session.end ends it sooner). Before the socket is open, stopping is cancelling.
   * @returns {boolean} whether a session was stopping
   */
  function stop() {
    const my = run;
    if (!my) return false;
    if (state.phase === 'requesting' || state.phase === 'starting') return cancel();
    if (state.phase !== 'recording') return false;
    stopCapture(my);
    set({ phase: 'stopping', stoppedAt: Date.now(), vad: '' });
    const ws = my.ws;
    if (ws && ws.readyState === OPEN) {
      try { ws.send(JSON.stringify({ type: 'stop' })); } catch { /* the close event finishes */ }
      clearTimeout(my.stopTimer);
      my.stopTimer = setTimeout(() => finish(my), STOP_GRACE_MS);
    } else {
      finish(my);
    }
    return true;
  }

  /** Close at once, without waiting for the last segments (the caller discards the session's text). */
  function cancel() {
    const my = run;
    if (!my) return false;
    const seconds = duration(my);
    teardown(my);
    run = null;
    set({ phase: 'idle', error: null, vad: '', nothingHeard: '', stoppedAt: Date.now() });
    emit('end', { reason: 'cancelled', duration: seconds, begun: my.begun });
    return true;
  }

  /** Forget a shown error (the banner was dismissed). */
  function clearError() {
    if (state.error) set({ error: null });
  }

  return {
    start,
    stop,
    cancel,
    clearError,
    isActive,
    get state() { return state; },
  };
}
