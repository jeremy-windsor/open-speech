/**
 * The app-wide audio coordinator: one audible session at a time, mirrored into app.player, and
 * the only module that touches navigator.mediaSession and navigator.audioSession (the page's audio session
 * type, where Safari has it: 'playback' once a non-ephemeral session is claimed, 'play-and-record' while
 * core/mic.js captures).
 *
 * Session object (claim):
 *   {id, owner, kind: 'long'|'clip'|'live', title, subtitle, artwork, homeRoute, ephemeral,
 *    controls: {play, pause, toggle, stop, seekBy?, seekTo?, prev?, next?},
 *    getState: () => ({playing, waiting, position, duration, rate, canSeek, statusText}),
 *    mediaMetadata?: () => ({title, artist, album, artwork}) | null}
 * @module core/audio
 */
import { app } from './store.js';

let active = null;      // the current session's entry: {session, handle, released}
let resumable = null;   // a displaced 'long' session: {session, handle, released}
let msOwner = null;     // the entry whose handlers and metadata are on the lock screen (active or resumable)
let sharedEl = null;
let primeToken = 0;
let silentUrl = '';
let captureHolds = 0;   // open microphone captures (holdCaptureAudioSession)
let typeBeforeCapture = '';

const ACTIONS = ['play', 'pause', 'stop', 'seekbackward', 'seekforward', 'seekto', 'previoustrack', 'nexttrack'];

/** While the microphone is open nothing plays on the shared element: playClip() throws this. */
export class CaptureActiveError extends Error {
  constructor(message = 'Stop recording to listen.') {
    super(message);
    this.name = 'CaptureActiveError';
  }
}

function capturing() {
  return !!app.get().recording;
}

function prefsSkip() {
  const prefs = app.get().prefs || {};
  return { back: Number(prefs.skipBack) || 15, forward: Number(prefs.skipForward) || 30 };
}

function mediaSession() {
  try {
    return (typeof navigator !== 'undefined' && navigator.mediaSession) || null;
  } catch {
    return null;
  }
}

function safely(fn) {
  try { fn(); } catch { /* unsupported action or invalid state */ }
}

/* ── the page's audio session type (Safari 17+ navigator.audioSession) ──
 * iOS treats Web Audio as ambient sound: with the ringer switch on silent, Live (an AudioContext) is mute
 * while the Reader's media elements play. Declaring the page a 'playback' page fixes that; a capture needs
 * 'play-and-record'. The type is page-wide, so this module is its one owner: claim() asks for 'playback',
 * mic.open() holds 'play-and-record' while it captures and gives the earlier type back when it stops.
 * Where the API is missing nothing happens. */

function pageAudioSession() {
  try {
    const s = typeof navigator !== 'undefined' ? navigator.audioSession : null;
    return s && typeof s === 'object' && 'type' in s ? s : null;
  } catch {
    return null;
  }
}

function setPageAudioType(type) {
  const s = pageAudioSession();
  if (!s) return;
  safely(() => { if (s.type !== type) s.type = type; });
}

function preferPlayback() {
  if (!pageAudioSession()) return;
  // During a capture the type stays 'play-and-record'; playback is what the page goes back to afterwards.
  if (captureHolds > 0) typeBeforeCapture = 'playback';
  else setPageAudioType('playback');
}

/**
 * The page's audio session type, where the browser has one (Safari 17+ navigator.audioSession), else ''.
 * @returns {string} 'auto' | 'playback' | 'play-and-record' | … | ''
 */
export function audioSessionType() {
  const s = pageAudioSession();
  if (!s) return '';
  try { return String(s.type || ''); } catch { return ''; }
}

/**
 * For core/mic.js: switch the page's audio session to 'play-and-record' for a capture (where supported) and
 * return the function that gives the earlier type back. Call it before the capture's AudioContext and
 * getUserMedia; call the returned function once the capture has stopped (it runs only once).
 * @returns {() => void}
 */
export function holdCaptureAudioSession() {
  if (!pageAudioSession()) return () => {};
  if (captureHolds === 0) typeBeforeCapture = audioSessionType() || 'auto';
  captureHolds += 1;
  setPageAudioType('play-and-record');
  let done = false;
  return () => {
    if (done) return;
    done = true;
    captureHolds = Math.max(0, captureHolds - 1);
    if (captureHolds === 0) {
      setPageAudioType(typeBeforeCapture || 'auto');
      typeBeforeCapture = '';
    }
  };
}

function stateOf(session) {
  let s = {};
  try { s = session.getState ? session.getState() || {} : {}; } catch { s = {}; }
  return {
    playing: !!s.playing,
    waiting: !!s.waiting,
    position: Number.isFinite(s.position) ? s.position : 0,
    duration: Number.isFinite(s.duration) ? s.duration : 0,
    rate: Number.isFinite(s.rate) && s.rate > 0 ? s.rate : 1,
    canSeek: !!s.canSeek,
    statusText: s.statusText ? String(s.statusText) : '',
  };
}

/** The public, serialisable view of a session (what app.player and app.resumable hold). */
function publicState(session) {
  const controls = session.controls || {};
  return Object.freeze({
    id: session.id,
    owner: session.owner,
    kind: session.kind,
    title: session.title || '',
    subtitle: session.subtitle || '',
    artwork: typeof session.artwork === 'string' ? session.artwork : '',
    homeRoute: session.homeRoute || '',
    ephemeral: !!session.ephemeral,
    has: Object.freeze({
      seekBy: typeof controls.seekBy === 'function',
      seekTo: typeof controls.seekTo === 'function',
      prev: typeof controls.prev === 'function',
      next: typeof controls.next === 'function',
    }),
    ...stateOf(session),
  });
}

function metadataFor(session) {
  let md = null;
  if (typeof session.mediaMetadata === 'function') {
    try { md = session.mediaMetadata(); } catch { md = null; }
  }
  if (!md) {
    md = { title: session.title || '', artist: session.subtitle || '', album: '', artwork: session.artwork || '' };
  }
  let artwork = md.artwork || [];
  if (typeof artwork === 'string') artwork = artwork ? [{ src: artwork, sizes: '512x512', type: 'image/png' }] : [];
  const init = { title: md.title || '', artist: md.artist || '', album: md.album || '', artwork };
  return typeof MediaMetadata === 'function' ? new MediaMetadata(init) : init;
}

function installMediaSession(entry) {
  const ms = mediaSession();
  const { session } = entry;
  if (!ms || session.ephemeral) return;
  msOwner = entry;
  const c = session.controls || {};
  safely(() => { ms.metadata = metadataFor(session); });
  const handlers = {
    play: () => c.play && c.play(),
    pause: () => c.pause && c.pause(),
    stop: () => closeCurrent(),
    seekbackward: c.seekBy ? (d) => c.seekBy(-(Number(d?.seekOffset) || prefsSkip().back)) : null,
    seekforward: c.seekBy ? (d) => c.seekBy(Number(d?.seekOffset) || prefsSkip().forward) : null,
    seekto: c.seekTo ? (d) => { if (Number.isFinite(d?.seekTime)) c.seekTo(d.seekTime, d); } : null,
    previoustrack: c.prev && c.next ? () => c.prev() : null,
    nexttrack: c.prev && c.next ? () => c.next() : null,
  };
  for (const action of ACTIONS) safely(() => ms.setActionHandler(action, handlers[action]));
  mirrorMediaSession(entry);
}

function clearMediaSession() {
  msOwner = null;
  const ms = mediaSession();
  if (!ms) return;
  for (const action of ACTIONS) safely(() => ms.setActionHandler(action, null));
  safely(() => { ms.metadata = null; });
  safely(() => { ms.playbackState = 'none'; });
}

function mirrorMediaSession(entry) {
  const ms = mediaSession();
  if (!ms || entry.session.ephemeral) return;
  const s = stateOf(entry.session);
  safely(() => { ms.playbackState = s.playing ? 'playing' : 'paused'; });
  if (s.duration > 0 && s.position <= s.duration && s.rate > 0 && typeof ms.setPositionState === 'function') {
    safely(() => ms.setPositionState({ duration: s.duration, position: Math.max(0, s.position), playbackRate: s.rate }));
  }
}

function publish() {
  app.set({
    player: active ? publicState(active.session) : null,
    resumable: resumable ? publicState(resumable.session) : null,
  });
}

function makeEntry(session) {
  const entry = { session, released: false, handle: null };
  entry.handle = Object.freeze({
    notify: () => notify(entry),
    update: (meta) => update(entry, meta),
    release: () => release(entry, 'released'),
    get active() { return !entry.released && active === entry; },
  });
  return entry;
}

function notify(entry) {
  if (entry.released) return;
  if (active === entry) {
    mirrorMediaSession(entry);
    publish();
  } else if (resumable === entry) {
    // The resumable Reader re-offered on the lock screen keeps its state and position there current.
    if (msOwner === entry) mirrorMediaSession(entry);
    publish();
  }
}

function update(entry, meta = {}) {
  if (entry.released) return;
  for (const key of ['title', 'subtitle', 'artwork', 'homeRoute', 'mediaMetadata']) {
    if (key in meta) entry.session[key] = meta[key];
  }
  if ((active === entry || msOwner === entry) && !entry.session.ephemeral) {
    const ms = mediaSession();
    if (ms) safely(() => { ms.metadata = metadataFor(entry.session); });
  }
  notify(entry);
}

function release(entry, reason = 'released') {
  if (entry.released) return;
  entry.released = true;
  if (typeof entry.onRelease === 'function') safely(() => entry.onRelease(reason));
  if (active === entry) {
    active = null;
    clearMediaSession();
    // The displaced Reader is offered again, paused, on the lock screen and in the mini-player.
    if (resumable && !resumable.released) installMediaSession(resumable);
  } else if (msOwner === entry) {
    // The resumable Reader that was re-offered on the lock screen is closed (Stop there, or the mini-player's
    // Close): its handlers, metadata and 'paused' state go with it (RE-50: 'none' on stop).
    clearMediaSession();
  }
  if (resumable === entry) resumable = null;
  publish();
}

function displace(entry) {
  const { session } = entry;
  const c = session.controls || {};
  if (session.kind === 'live') {
    safely(() => c.stop && c.stop('Another player started'));
    release(entry, 'displaced');
  } else if (session.kind === 'long') {
    safely(() => c.pause && c.pause());
    if (resumable && resumable !== entry && !resumable.released) {
      const old = resumable;
      resumable = null;
      release(old, 'displaced');
    }
    resumable = entry;
  } else {
    safely(() => c.pause && c.pause());
    release(entry, 'displaced');
  }
}

/**
 * Make `session` the current audible session. The previous one is displaced: Live is
 * stopped, the Reader ('long') is paused and kept in app.resumable, a clip is paused and released.
 * Claiming the id that is already current returns its handle; claiming the resumable id brings it back.
 * @param {object} session see the module comment
 * @returns {{notify: () => void, update: (meta: object) => void, release: () => void, active: boolean}}
 */
export function claim(session) {
  if (!session || !session.id || !session.controls) throw new TypeError('claim() needs a session with an id and controls');
  if (active && !active.released && active.session.id === session.id) {
    if (active.session !== session) Object.assign(active.session, session);
    installMediaSession(active);
    publish();
    return active.handle;
  }
  let entry;
  if (resumable && !resumable.released && resumable.session.id === session.id) {
    entry = resumable;
    resumable = null;
    if (entry.session !== session) Object.assign(entry.session, session);
  } else {
    entry = makeEntry(session);
  }
  if (active && !active.released) displace(active);
  active = entry;
  if (session.ephemeral) clearMediaSession();
  else {
    installMediaSession(entry);
    preferPlayback();
  }
  publish();
  return entry.handle;
}

/** @returns {object|null} the current session (with its controls), or null */
export function current() { return active ? active.session : null; }

/** @returns {object|null} the displaced Reader session the mini-player can return to, or null */
export function resumableSession() { return resumable ? resumable.session : null; }

/**
 * The shared <audio id="app-audio" preload="auto"> inside #app-audio-host (created when missing), with
 * preservesPitch set.
 * @returns {HTMLAudioElement}
 */
export function element() {
  if (sharedEl && sharedEl.isConnected) return sharedEl;
  let el = document.getElementById('app-audio');
  if (!el) {
    el = document.createElement('audio');
    el.id = 'app-audio';
    el.setAttribute('preload', 'auto');
    (document.getElementById('app-audio-host') || document.body).appendChild(el);
  }
  el.preservesPitch = true;
  el.webkitPreservesPitch = true;
  sharedEl = el;
  return el;
}

/** @returns {boolean} whether the shared element can play this MIME type */
export function canPlay(mime) {
  try {
    return element().canPlayType(String(mime)) !== '';
  } catch {
    return false;
  }
}

function silentWav() {
  if (silentUrl) return silentUrl;
  const rate = 8000;
  const samples = 800;   // 0.1 s
  const bytes = new Uint8Array(44 + samples);
  const view = new DataView(bytes.buffer);
  const text = (offset, s) => { for (let i = 0; i < s.length; i += 1) bytes[offset + i] = s.charCodeAt(i); };
  text(0, 'RIFF'); view.setUint32(4, 36 + samples, true); text(8, 'WAVE'); text(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, rate, true); view.setUint32(28, rate, true); view.setUint16(32, 1, true);
  view.setUint16(34, 8, true); text(36, 'data'); view.setUint32(40, samples, true);
  bytes.fill(128, 44);
  let binary = '';
  for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
  silentUrl = `data:audio/wav;base64,${btoa(binary)}`;
  return silentUrl;
}

/**
 * MUST be called synchronously inside a click whose result plays on the shared element only after an
 * await (Generate, Preview, Test clone): plays 0.1 s of silence on #app-audio and pauses it, so iOS allows
 * the later, non-gesture play(). Never call it for audio that is already loaded (it replaces the source).
 */
export function prime() {
  // Nothing plays while the microphone is open: the silent clip would be heard during the capture,
  // and it would end a clip the capture only paused (it stays in the mini-player for later). The result is
  // refused anyway until the capture stops (playClip throws CaptureActiveError), so there is nothing to unlock.
  if (capturing()) return;
  const el = element();
  // The silent clip replaces the shared element's source, so a clip session playing on it could only play
  // silence afterwards: end it now, with its element listeners (its onEnded does not run).
  if (active && !active.released && active.clipSrc !== undefined) {
    safely(() => el.pause());
    release(active, 'replaced');
  }
  const token = (primeToken += 1);
  const src = silentWav();
  try {
    el.muted = false;
    el.src = src;
    const played = el.play();
    const settle = () => {
      if (token === primeToken && el.src === src) safely(() => el.pause());
    };
    if (played && typeof played.then === 'function') played.then(settle, () => {});
    else settle();
  } catch {
    // the result will still try to play; the mini-player offers Play if the browser refuses
  }
}

const TIMEUPDATE_MS = 250;

/**
 * Play a one-shot clip (a take, a render, a mix, a clone test or a preview) on the shared element.
 * Playing the same id with the same src again resumes it. Nothing plays while the microphone is open:
 * it rejects with CaptureActiveError ("Stop recording to listen."), and the session's play()
 * refuses the same way.
 * @param {{id: string, owner: string, title: string, subtitle?: string, homeRoute?: string, src: string,
 *   artwork?: string, ephemeral?: boolean, onEnded?: () => void,
 *   onRelease?: (reason: 'ended'|'displaced'|'closed'|'replaced'|'released'|'error') => void}} clip
 *   onEnded runs only when the clip played to its end; onRelease runs however the session ended (after
 *   onEnded), so an owner can revoke its object URL or reset a "playing" indicator in one place
 * @returns {Promise<object>} the session handle; a refused autoplay leaves the session paused (statusText
 *   'Tap play to start listening') rather than throwing. A media error after the start (a network drop or a
 *   server restart while a long render streams) keeps the session with "Couldn't play"; the next Play (the
 *   owner's toggle, the mini-player, the lock screen, or playClip with the same id and src) loads the source
 *   again and resumes where it stopped, because play() alone never reloads an errored element.
 */
export async function playClip({ id, owner, title, subtitle = '', homeRoute = '', src, artwork, ephemeral = false, onEnded, onRelease } = {}) {
  if (!id || !src) throw new TypeError('playClip() needs an id and a src');
  if (capturing()) throw new CaptureActiveError();
  const el = element();
  if (active && !active.released && active.session.id === id) {
    if (active.clipSrc === src) {
      await Promise.resolve(active.session.controls.play()).catch(() => {});
      return active.handle;
    }
    // The same id with new audio (the next voice preview, a second clone test) is a new session: release
    // the old one first, so its element listeners and its onEnded go with it.
    safely(() => el.pause());
    release(active, 'replaced');
  }
  let waiting = false;
  let statusText = '';
  let entryRef = null;
  let failed = false;     // the media failed after the start: the next play() loads the source again
  let resumeAt = 0;       // where a reloaded source continues
  const reload = () => {
    failed = false;
    resumeAt = el.currentTime || 0;
    waiting = true;       // 'Preparing audio…' until it plays again (or "Couldn't play" if it fails again)
    primeToken += 1;
    el.src = src;
    // Before the metadata the browser keeps this as the start position; loadedmetadata checks it.
    if (resumeAt > 0) el.currentTime = resumeAt;
  };
  const session = {
    id, owner, kind: 'clip', title, subtitle, homeRoute, artwork: artwork || '', ephemeral: !!ephemeral,
    controls: {
      play: () => {
        if (capturing()) {
          statusText = 'Stop recording to listen';
          if (entryRef) notify(entryRef);
          return Promise.resolve();
        }
        statusText = '';
        if (failed || el.error) reload();
        const p = el.play();
        if (p && p.catch) p.catch(() => {});
        return p;
      },
      pause: () => el.pause(),
      toggle: () => (el.paused ? session.controls.play() : el.pause()),
      stop: () => { el.pause(); if (entryRef) release(entryRef, 'closed'); },
      seekBy: (s) => { el.currentTime = Math.max(0, Math.min((el.duration || 0), el.currentTime + Number(s || 0))); },
      seekTo: (s) => { el.currentTime = Math.max(0, Number(s) || 0); },
    },
    getState: () => ({
      playing: !el.paused && !el.ended,
      waiting,
      position: el.currentTime || 0,
      duration: Number.isFinite(el.duration) ? el.duration : 0,
      rate: el.playbackRate || 1,
      canSeek: Number.isFinite(el.duration) && el.duration > 0,
      statusText: statusText || (waiting ? 'Preparing audio…' : ''),
    }),
  };
  const handle = claim(session);
  const entry = active;
  entryRef = entry;
  entry.clipSrc = src;
  let lastTick = 0;
  const listeners = {
    play: () => handle.notify(),
    pause: () => handle.notify(),
    ratechange: () => handle.notify(),
    seeked: () => handle.notify(),
    // The length is known before anything plays (a refused autoplay waits paused), so the mini-player and
    // an inline transport can show it and seek at once.
    loadedmetadata: () => {
      if (resumeAt > 0) {
        const at = resumeAt;
        resumeAt = 0;
        if (Math.abs((el.currentTime || 0) - at) > 0.25) el.currentTime = at;
      }
      handle.notify();
    },
    durationchange: () => handle.notify(),
    waiting: () => {
      // After a media error nothing more arrives: stop at once, so the mini-player offers Play (which reloads)
      // instead of 'Preparing audio…' forever.
      if (failed) { safely(() => el.pause()); waiting = false; handle.notify(); return; }
      waiting = true;
      handle.notify();
    },
    playing: () => { waiting = false; handle.notify(); },
    timeupdate: () => {
      const now = Date.now();
      if (now - lastTick >= TIMEUPDATE_MS) { lastTick = now; handle.notify(); }
    },
    ended: () => {
      if (typeof onEnded === 'function') safely(onEnded);
      release(entry, 'ended');
    },
    error: () => {
      // The buffered audio may still play out (paused stays false until it stalls); a stalled element stops now.
      failed = true;
      statusText = "Couldn't play";
      if (waiting || el.readyState < 3) safely(() => el.pause());
      waiting = false;
      handle.notify();
    },
  };
  for (const [type, fn] of Object.entries(listeners)) el.addEventListener(type, fn);
  entry.onRelease = (reason) => {
    for (const [type, fn] of Object.entries(listeners)) el.removeEventListener(type, fn);
    if (typeof onRelease === 'function') safely(() => onRelease(reason));
  };
  primeToken += 1;   // a pending prime() must not pause the real clip
  el.playbackRate = 1;
  el.preservesPitch = true;
  el.src = src;
  try {
    await el.play();
  } catch (error) {
    if (error && error.name === 'NotAllowedError') {
      statusText = 'Tap play to start listening';
      handle.notify();
    } else if (!(error && error.name === 'AbortError')) {
      release(entry, 'error');
      throw error;
    }
  }
  return handle;
}

/** Pause whatever plays (before the microphone opens). */
export function pauseAll() {
  if (active && !active.released) safely(() => active.session.controls.pause && active.session.controls.pause());
  if (sharedEl && !sharedEl.paused) safely(() => sharedEl.pause());
  publish();
}

/**
 * A player this module does not run started (the clone draft's native <audio controls>): the current session
 * is displaced as claim() displaces it. Live is stopped and released, the Reader is paused and
 * kept resumable (offered again, paused, in the mini-player and on the lock screen), a clip is paused and
 * released. Nothing becomes current.
 */
export function otherPlayerStarted() {
  const entry = active && !active.released ? active : null;
  if (entry) displace(entry);
  if (entry && active === entry) {
    // The displaced Reader: displace() kept it in `resumable`, and it is no longer the current session.
    active = null;
    clearMediaSession();
    if (resumable && !resumable.released) installMediaSession(resumable);
  }
  if (sharedEl && !sharedEl.paused) safely(() => sharedEl.pause());
  publish();
}

/**
 * The mini-player's Close and the lock screen's Stop: controls.stop() and release. With no current
 * session, it closes the resumable Reader instead.
 */
export function closeCurrent() {
  const entry = active && !active.released ? active : (resumable && !resumable.released ? resumable : null);
  if (!entry) return;
  safely(() => entry.session.controls.stop && entry.session.controls.stop());
  release(entry, 'closed');
}

/**
 * PURE: what the mini-player shows for an app state.
 * @param {object} state app.get()
 * @param {{immersive?: boolean}} [options]
 * @returns {{visible: boolean, source: 'player'|'resumable'|'', session: object|null}}
 */
export function miniPlayerState(state, { immersive = false } = {}) {
  const s = state || {};
  const session = s.player || (!s.player ? s.resumable : null) || null;
  const source = s.player ? 'player' : (s.resumable ? 'resumable' : '');
  const path = s.route?.path || '';
  const visible = !!session && !session.ephemeral && !immersive && path !== session.homeRoute;
  return { visible, source: visible ? source : '', session: visible ? session : null };
}

/** Test hook. */
export function __resetForTests() {
  active = null;
  resumable = null;
  msOwner = null;
  sharedEl = null;
  primeToken = 0;
  captureHolds = 0;
  typeBeforeCapture = '';
}
