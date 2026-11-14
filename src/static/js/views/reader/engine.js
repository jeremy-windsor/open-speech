/**
 * The Reader playback engine: a singleton that survives navigation. It ports the
 * old reader.js player with the same behaviour, and adds the app-wide audio session, the chapter timeline,
 * the sleep timer and the wake lock.
 *
 * - Two <audio> elements (#rd-audio-a, #rd-audio-b) inside #app-audio-host: the active one plays a part, the
 *   idle one preloads the next for gapless playback. Never routed through an AudioContext (RE-02).
 * - The silent keep-alive loop while a start waits for its first part, and the iOS unlock dance (RE-25, RE-26).
 * - Render and poll: POST /render, then GET /parts with backoff (500 ms doubling to 4 s), 10 failures in a row
 *   stop playback, only network errors, 5xx, 408 and 429 retry, and the render is asked again when our window
 *   vanished for 3 s (RE-07, RE-08, RE-10).
 * - Recovery from expired signed URLs and dropped connections (RE-21).
 * - Tracking runs on requestAnimationFrame while the page is visible and on the active element's timeupdate
 *   events (about 4 Hz) always, because rAF stops on a locked phone. The sleep timer, progress saves, chapter
 *   changes and the lock-screen metadata all hang off tracking (REV-34).
 * - audio.claim({id: 'reader:<docId>', kind: 'long', …}) on play: the mini-player and the lock screen drive the
 *   engine through the session's controls; its stop is unload().
 *
 * API: getEngine() -> {load, unload, play, pause, toggle, playFrom, jumpTo, skipBlock, skipSentence,
 *   seekBy, seekChapterMs, setSpeed, setVoice, setSleep, unlock, saveProgress, state, on}
 * plus: step(delta) (prev/next per prefs.stepUnit), prevTrack()/nextTrack() (prefs.trackUnit), doc,
 *   voiceLabel(choice), unload({forget}) (a deleted document), and the events 'progress' and 'voice-needed'.
 * @module views/reader/engine
 */
import { get as apiGet, post as apiPost, put as apiPut } from '../../core/api.js';
import { app, on as onAppEvent } from '../../core/store.js';
import { KEYS, read, readJson, write, writeJson } from '../../core/storage.js';
import * as prefs from '../../core/prefs.js';
import * as models from '../../core/models.js';
import * as audio from '../../core/audio.js';
import { listNamed } from '../../core/voices.js';
import { toast } from '../../core/toast.js';
import { navigate } from '../../core/router.js';
import { safeUrl } from '../../core/dom.js';
import { artworkDataUrl } from '../../core/tiles.js';
import {
  COPY, MAX_FAILURES, LOOKAHEAD_POLL_MS, RERENDER_AFTER_MS, RERENDER_MAX_MS, PREPARING_DELAY_MS, SLOW_PREPARING_MS, MODELS_WAIT_MS,
  SAVE_EVERY_MS, REQUEST_TIMEOUT_MS, NOTIFY_MS, clamp, locate, isRetryable, backoffMs, parkSeek, clearParkedSeek,
  takeParkedSeek, positionOf, resolveVoice, resumeOffset, clampSpeed, storedSpeed, documentSpeed,
  chapterMap, chapterRange, chapterStart, charsOf, trackPosition, sentenceStartMs, sentenceStep, voiceLabel,
  silentWavUrl, isRecordingChoice,
} from './play-logic.js';
import * as timeline from './timeline.js';
import * as sleepTimer from './sleep-timer.js';
import * as lockScreen from './media-session.js';

const API = '/v1/reader/documents';
const ELEMENT_IDS = ['rd-audio-a', 'rd-audio-b'];
const SPEED_SAVE_MS = 1000;   // a run of speed changes saves once
const docUrl = (id, suffix = '') => `${API}/${encodeURIComponent(id)}${suffix}`;
const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const voiceKey = (v) => (v && v.model && v.voice ? `${v.model}|${v.voice}` : '');

function docHidden() {
  try {
    return typeof document !== 'undefined' && !!document.hidden;
  } catch {
    return false;
  }
}

function report(error) {
  if (globalThis.console) globalThis.console.error(error);
}

let instance = null;

/** @returns {object} the Reader engine (created on first use; importing this module has no side effects) */
export function getEngine() {
  if (!instance) instance = createEngine();
  return instance;
}

/** Test hook: stop everything, drop listeners and the elements' handlers, forget the singleton. */
export function __resetForTests() {
  if (instance) instance.__destroy();
  instance = null;
}

function createEngine() {
  const r = {
    doc: null,
    chapterAt: [],
    cursor: 0,
    sentence: -1,
    chapter: 0,
    playing: false,       // the listener wants audio (also true while waiting for a part)
    waiting: false,       // a start or skip is waiting for its first part
    token: 0,             // bumped on every start, skip and stop so stale async work bails out
    lookahead: 0,
    parts: [],            // the latest planned window from the server
    renderKey: '',        // doc, voice and from_block of the last render request
    active: null,         // {el, part} now playing (or paused)
    preloaded: null,      // {el, part} loaded into the idle element
    keepAlive: null,      // the element looping silence while waiting (iOS keeps the audio session)
    resume: null,         // saved progress (on open, or where playback stopped), used once by the next play
    voice: null,          // {model, voice} once resolved
    speed: 1,
    recovering: false,
    audioFailures: 0,
    status: { text: '', kind: '' },   // kind: '' | 'preparing' | 'retry' | 'error'
    preparing: '',        // the waiting text, put back once a retry settles
    loadingText: '',      // "Loading Kokoro…" while a start waits for its model
    waitText: '',         // a one-off waiting text ("Voice changed. Re-rendering from here.")
    lastSaved: '',
    lastSaveAt: 0,
    raf: 0,
    preparingTimer: null,
    slowTimer: null,
    speedTimer: null,
    stateFrame: 0,
    stateTimer: null,
    els: null,
    handle: null,         // the audio session handle while claimed
    session: null,
    lastNotify: 0,
    known: new Map(),     // voice key -> Map(first_block -> ready part): the timeline's rendered parts
    reportedMs: 0,        // the smoothed chapter duration reported to the lock screen
    reportedKnown: -1,
    reportedChapter: -1,
    sleep: sleepTimer.OFF,
    wakeLock: null,
    wakeWanted: false,
    wakeToken: 0,         // one per wake-lock change of mind: a late answer to an older request is released
    artwork: { id: null, list: [] },
    names: new Map(),     // named voice id -> name
    namesLoading: null,
    namesFailedAt: 0,
    playError: null,      // the "Couldn't play" toast
    listeners: new Map(),
    elementOffs: [],
    globalOffs: [],
    bound: false,
    destroyed: false,
    switching: null,      // the element whose src is being replaced right now
  };

  /* ───────────── events ───────────── */

  function on(type, cb) {
    if (typeof cb !== 'function') throw new TypeError('on() needs a callback');
    const list = r.listeners.get(type) || [];
    list.push(cb);
    r.listeners.set(type, list);
    return () => {
      r.listeners.set(type, (r.listeners.get(type) || []).filter((fn) => fn !== cb));
    };
  }

  function emit(type, detail) {
    const list = r.listeners.get(type) || [];
    for (const cb of list.slice()) {
      try { cb(detail); } catch (error) { report(error); }
    }
    return list.length;
  }

  function flushState() {
    if (r.stateFrame && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(r.stateFrame);
    if (r.stateTimer !== null) clearTimeout(r.stateTimer);
    r.stateFrame = 0;
    r.stateTimer = null;
    if (!r.destroyed) emit('state', snapshot());
  }

  /** 'state' at most once per frame (a timer stands in while the page is hidden, where frames don't run). */
  function emitState() {
    if (r.destroyed || r.stateFrame || r.stateTimer !== null) return;
    if (!docHidden() && typeof requestAnimationFrame === 'function') r.stateFrame = requestAnimationFrame(flushState);
    else r.stateTimer = setTimeout(flushState, NOTIFY_MS);
  }

  /* ───────────── elements ───────────── */

  function elements() {
    if (r.els && r.els.every((el) => el.isConnected !== false)) return r.els;
    for (const off of r.elementOffs.splice(0)) off();
    const host = document.getElementById('app-audio-host') || document.body;
    r.els = ELEMENT_IDS.map((id) => {
      let el = document.getElementById(id);
      if (!el) {
        el = document.createElement('audio');
        el.id = id;
        el.setAttribute('preload', 'auto');
        host.appendChild(el);
      }
      el.preload = 'auto';
      el.preservesPitch = true;
      el.webkitPreservesPitch = true;
      bindElement(el);
      return el;
    });
    return r.els;
  }

  function otherEl(el) {
    const [a, b] = elements();
    return el === a ? b : a;
  }

  function bindElement(el) {
    const listen = (type, fn) => {
      el.addEventListener(type, fn);
      r.elementOffs.push(() => el.removeEventListener(type, fn));
    };
    listen('ended', handleEnded);
    listen('error', handleAudioError);
    listen('pause', handlePause);
    listen('play', handlePlay);
    listen('playing', handlePlaying);
    // Apply a seek made before this source's metadata arrived (setSrc clears it for the next source).
    listen('loadedmetadata', () => {
      const seconds = takeParkedSeek(el);
      if (seconds != null) {
        try { el.currentTime = seconds; } catch { /* not seekable yet */ }
      }
    });
    for (const type of ['loadedmetadata', 'durationchange', 'ratechange', 'seeked']) {
      listen(type, () => {
        if (el !== r.active?.el) return;
        notify(true);
        emitState();
      });
    }
    listen('timeupdate', () => { if (el === r.active?.el) track(); });
  }

  function holdsPart(el) {
    return !!el.dataset.part && !!el.getAttribute('src') && !String(el.getAttribute('src')).startsWith('data:');
  }

  function applyRate(el) {
    el.defaultPlaybackRate = r.speed;
    el.playbackRate = r.speed;
    el.preservesPitch = true;
    el.webkitPreservesPitch = true;
  }

  function applyVolume(volume) {
    if (!r.els) return;
    for (const el of r.els) {
      try { el.volume = volume; } catch { /* read-only on iOS */ }
    }
  }

  // Every src change goes through here, so a seek parked for the old source can't land on the new one (RE-16).
  function setSrc(el, url, partId = null) {
    clearParkedSeek(el);
    // A currentTime set before metadata becomes the browser's start position, which survives a src change.
    if (el.readyState === 0) {
      try { el.currentTime = 0; } catch { /* nothing loaded */ }
    }
    el.loop = false;
    if (partId) el.dataset.part = partId;
    else delete el.dataset.part;
    // A new source stops the element; a pause event from that (some engines fire one) is ours, not the system's.
    r.switching = el;
    try {
      el.src = url;
    } finally {
      r.switching = null;
    }
  }

  function seekEl(el, seconds) {
    const to = Math.max(0, seconds || 0);
    clearParkedSeek(el);
    if (el.readyState === 0) {
      if (to > 0) parkSeek(el, to);   // applied by the element's loadedmetadata listener (RE-14)
      return;
    }
    try { el.currentTime = to; } catch { /* not seekable */ }
  }

  /* ───────────── iOS: unlock and keep-alive ───────────── */

  /** True once both elements have played inside a gesture. */
  function allUnlocked() {
    return elements().every((el) => el.dataset.unlocked);
  }

  // iOS only lets an element start later (after a network wait, or from 'ended' on a locked phone) if it was
  // played once inside a user gesture. Each element is flagged when such a play succeeds; until both are,
  // every gesture in the Reader tries again. The silent clip plays unmuted, because iOS lets a muted element
  // play without a gesture, so a muted success would prove nothing. Only an element that already holds a part
  // is muted, so the unlock doesn't blip its audio, and its position is put back (RE-26).
  function unlock() {
    for (const el of elements()) {
      if (el.dataset.unlocked || el.dataset.unlocking) continue;
      if (!el.paused) {
        el.dataset.unlocked = '1';   // already playing (a part, or the keep-alive)
        continue;
      }
      if (el === r.active?.el) continue;   // play() starts it inside this same gesture
      if (el === r.keepAlive) {
        playKeepAlive(el);
        continue;
      }
      const holding = holdsPart(el);
      const at = el.currentTime;
      if (!holding) setSrc(el, silentWavUrl());
      el.muted = holding;
      el.dataset.unlocking = '1';
      let attempt = null;
      try { attempt = el.play(); } catch { attempt = null; }
      if (!attempt || typeof attempt.then !== 'function') {
        delete el.dataset.unlocking;
        el.muted = false;
        continue;
      }
      attempt.then(() => {
        el.dataset.unlocked = '1';
        if (el === r.active?.el || el === r.keepAlive) return;   // it became the player meanwhile
        el.pause();
        if (holding) seekEl(el, at);
      }).catch(() => {}).finally(() => {
        delete el.dataset.unlocking;
        el.muted = false;
      });
    }
  }

  // While a start waits for its first part, loop silence on the idle element so iOS keeps the audio session
  // (and the lock-screen controls) alive across the gap (RE-25).
  function startKeepAlive() {
    if (r.keepAlive) return;
    const el = r.active ? otherEl(r.active.el) : elements()[1];
    const url = silentWavUrl();
    if (el.getAttribute('src') !== url) setSrc(el, url);
    el.loop = true;
    el.muted = false;
    r.keepAlive = el;
    delete el.dataset.keepAlivePlaying;
    playKeepAlive(el);
  }

  function playKeepAlive(el) {
    let attempt = null;
    try { attempt = el.play(); } catch { attempt = null; }
    Promise.resolve(attempt).then(() => {
      if (!attempt) return;
      el.dataset.unlocked = '1';
      if (r.keepAlive === el) el.dataset.keepAlivePlaying = '1';   // later pauses come from the system
    }).catch(() => {});
  }

  function stopKeepAlive() {
    const el = r.keepAlive;
    if (!el) return;
    r.keepAlive = null;
    delete el.dataset.keepAlivePlaying;
    el.loop = false;
    el.pause();
  }

  /* ───────────── voice ───────────── */

  function usableModels() {
    try {
      return models.steadyTtsModels(app.get().models.list);
    } catch {
      return [];
    }
  }

  function modelIds() { return usableModels().map((m) => m.id); }

  // Speak's voice (app.speakVoice), never a raw reference recording: Reader renders send only {model, voice},
  // so a recording name would not be a voice the server can read with (REV-09). Speak publishes it with its
  // kind, at boot from its saved setup or default (main.js, views/speak/speak-voice.js) and from its own view
  // once that runs, so a page that opens straight into the Reader has it too (RE-04).
  function speakVoice() {
    const s = app.get().speakVoice;
    if (!s || !s.model) return null;
    const voice = s.voice && !isRecordingChoice(s) ? s.voice : '';
    return { model: s.model, voice };
  }

  function storedVoice() {
    const value = readJson(KEYS.readerVoice, null, (v) => !!v && typeof v === 'object');
    return value && typeof value.model === 'string' && typeof value.voice === 'string' && value.model && value.voice
      ? { model: value.model, voice: value.voice }
      : null;
  }

  // The voice to render with, checked against the models the app has loaded; null until one is known. The
  // document's saved voice and the last Reader voice are kept once resolved. Speak's is followed as it changes
  // (the chip refreshes with Speak, RE-63) until a playback starts with it: readyVoice() keeps it then, so every
  // part of that playback renders with the same voice.
  function currentVoice() {
    const ids = modelIds();
    if (r.voice && ids.includes(r.voice.model)) return r.voice;
    const own = resolveVoice({ progress: r.doc?.progress, stored: storedVoice(), speak: null, modelIds: ids });
    if (own) {
      r.voice = own;
      return own;
    }
    return resolveVoice({ progress: null, stored: null, speak: speakVoice(), modelIds: ids }) || r.voice;
  }

  // What the chip shows (and progress records) before the models load: the document's saved voice, else the
  // last Reader voice, else Speak's.
  function candidateVoice() {
    const progress = r.doc?.progress;
    return currentVoice()
      || (progress?.model && progress.voice ? { model: progress.model, voice: progress.voice } : null)
      || storedVoice()
      || speakVoice();
  }

  // Named voice names for the chip and the lock screen: one background request, never before first audio
  // matters (the label falls back to "Named voice"); a failure waits a minute before trying again.
  function loadNames() {
    if (r.namesLoading) return r.namesLoading;
    if (r.namesFailedAt && Date.now() - r.namesFailedAt < 60000) return Promise.resolve();
    r.namesLoading = listNamed({ check: false }).then((list) => {
      r.names = new Map(list.filter((v) => v && v.id).map((v) => [String(v.id), String(v.name || v.id)]));
      r.namesFailedAt = 0;
      const voice = candidateVoice();
      emit('voice', voice ? { ...voice } : null);
      emitState();
    }).catch(() => {
      r.namesLoading = null;
      r.namesFailedAt = Date.now();
    });
    return r.namesLoading;
  }

  function labelFor(choice) {
    if (choice?.voice && String(choice.voice).startsWith('voice:') && !r.names.has(String(choice.voice).slice(6))) loadNames();
    return voiceLabel(choice, r.names);
  }

  // A deep link may arrive before the model list: give it a moment, then trust the saved voice (RE-04).
  async function readyVoice(token) {
    const until = Date.now() + MODELS_WAIT_MS;
    while (!usableModels().length && !app.get().models.fetchedAt && Date.now() < until) {
      await wait(100);
      if (token !== r.token) return null;
    }
    const voice = currentVoice();
    if (voice?.model && voice.voice) {
      r.voice = voice;   // this playback keeps it (Speak's voice too), so every part renders with the same voice
      return voice;
    }
    if (usableModels().length || app.get().models.fetchedAt) return null;
    const progress = r.doc?.progress;
    r.voice = progress?.model && progress.voice ? { model: progress.model, voice: progress.voice } : storedVoice();
    return r.voice;
  }

  function needVoice() {
    const noModel = !usableModels().length;
    const detail = { reason: noModel ? 'no-model' : 'no-voice', message: noModel ? COPY.noModel : COPY.noVoice };
    // The player opens the voice sheet (or toasts when it isn't on screen); without one, toast here.
    if (!emit('voice-needed', detail)) {
      const action = noModel ? { label: COPY.openModels, run: () => navigate('models/tts') } : null;
      toast(detail.message, { kind: 'error', id: 'reader-voice', action });
    }
  }

  // Load the voice's model before asking for audio (RE-06): a status job ("Loading Piper…", "Unloads Kokoro"),
  // no confirm dialog, and the fast path makes no request when the model is known to be loaded.
  // False when this start was replaced or the load failed (playback then stopped with the reason).
  async function loadVoiceModel(model, token) {
    const key = `model:${model}`;
    const off = app.select((s) => (s.jobs || []).find((job) => job.key === key)?.label || '', (label) => {
      if (token !== r.token) return;
      r.loadingText = label;
      if (label) showPreparing(label);
    }, { immediate: true });
    try {
      await models.ensureReady(model, { kind: 'tts', confirmSwitch: false });
    } catch (error) {
      if (token !== r.token) return false;
      // No status answer: the render request reports any real problem (the old behaviour).
      const soft = error && (error.kind === 'network' || error.kind === 'timeout' || error.status === 404);
      if (!soft) {
        fail(error?.message || String(error));
        return false;
      }
    } finally {
      off();
      const hadLoading = !!r.loadingText;
      r.loadingText = '';
      if (hadLoading && token === r.token && r.waiting) showPreparing(r.waitText || COPY.preparing);
    }
    return token === r.token;
  }

  /* ───────────── status ───────────── */

  function setStatus(text, kind = '') {
    const next = { text: text || '', kind: text ? kind : '' };
    if (next.text === r.status.text && next.kind === r.status.kind) return;
    r.status = next;
    emit('status', { ...next });
    notify(true);
    emitState();
  }

  // The waiting text: the play button's busy ring and the status line; a "Reconnecting…" or error line stays
  // until it settles.
  function showPreparing(text) {
    r.preparing = text;
    if (text && r.status.kind !== 'retry' && r.status.kind !== 'error') setStatus(text, 'preparing');
    else if (!text && r.status.kind === 'preparing') setStatus('');
    emitState();
  }

  // A retry succeeded: put back whatever the wait was showing.
  function settleStatus() {
    if (r.status.kind === 'retry') setStatus(r.preparing, r.preparing ? 'preparing' : '');
  }

  function dismissPlayError() {
    if (r.playError) {
      const t = r.playError;
      r.playError = null;
      t.dismiss();
    }
  }

  function setPlaying(on) {
    const changed = r.playing !== on;
    sleepTick();   // the time so far counts with the old state, the next stretch with the new one
    r.playing = on;
    if (on) startTicker();
    else stopTicker();
    sleepTick();
    if (changed) updateWakeLock();
    notify(true);
    emitState();
  }

  function setWaiting(on) {
    sleepTick();
    r.waiting = on;
    sleepTick();
    clearTimeout(r.preparingTimer);
    clearTimeout(r.slowTimer);
    r.preparingTimer = null;
    r.slowTimer = null;
    if (on) {
      startKeepAlive();
      r.preparingTimer = setTimeout(() => showPreparing(r.loadingText || r.waitText || COPY.preparing), PREPARING_DELAY_MS);
      r.slowTimer = setTimeout(() => showPreparing(r.loadingText || COPY.slow), SLOW_PREPARING_MS);
    } else {
      stopKeepAlive();
      r.waitText = '';
      showPreparing('');
    }
    emitState();
  }

  /* ───────────── the audio session ───────────── */

  function sessionMeta() {
    const md = currentMetadata();
    return { ...lockScreen.sessionText(md), artwork: md.artwork.length ? md.artwork[Math.min(1, md.artwork.length - 1)].src : '' };
  }

  function currentMetadata() {
    const doc = r.doc;
    const art = doc && r.artwork.id === doc.id ? r.artwork.list : [];
    return lockScreen.metadata({ doc, chapterIndex: r.chapter, voiceLabel: labelFor(candidateVoice()), artwork: art });
  }

  // The artwork is drawn on a canvas: when the page is idle after a document loads, or at the latest when its
  // first part starts, never on the way to first audio.
  function scheduleArtwork() {
    const doc = r.doc;
    const run = () => {
      if (r.destroyed || r.doc !== doc) return;
      ensureArtwork();
      updateMetadata();
    };
    if (typeof requestIdleCallback === 'function') requestIdleCallback(run, { timeout: 3000 });
    else setTimeout(run, 1500);
  }

  function ensureArtwork() {
    const doc = r.doc;
    if (!doc || r.artwork.id === doc.id) return;
    const urls = {};
    for (const size of lockScreen.ARTWORK_SIZES) urls[size] = artworkDataUrl(doc.id, doc.title, size);
    r.artwork = { id: doc.id, list: lockScreen.artworkList(urls) };
  }

  function sessionObject() {
    const doc = r.doc;
    const id = `reader:${doc.id}`;
    if (r.session && r.session.id === id) return r.session;
    const meta = sessionMeta();
    r.session = {
      id,
      owner: 'reader',
      kind: 'long',
      title: meta.title,
      subtitle: meta.subtitle,
      artwork: meta.artwork,
      homeRoute: `reader/${encodeURIComponent(doc.id)}`,
      ephemeral: false,
      controls: {
        play: () => engine.play(),
        pause: () => engine.pause(),
        toggle: () => engine.toggle(),
        stop: () => { engine.unload(); },
        seekBy: (seconds) => engine.seekBy(seconds),
        seekTo: (seconds) => seekToSession(seconds),
        prev: () => engine.prevTrack(),
        next: () => engine.nextTrack(),
      },
      getState: () => sessionState(),
      mediaMetadata: () => currentMetadata(),
    };
    return r.session;
  }

  function claim() {
    if (!r.doc) return;
    const session = sessionObject();
    if (r.handle && r.handle.active) return;
    r.handle = audio.claim(session);
  }

  function releaseSession() {
    const handle = r.handle;
    r.handle = null;
    r.session = null;
    r.reportedMs = 0;
    r.reportedKnown = -1;
    if (!handle) return;
    // Releasing the Reader also clears the lock screen it owns, also after a clip that displaced it ended and
    // core/audio.js offered it there again (RE-50).
    handle.release();
  }

  function notify(force) {
    if (!r.handle) return;
    const now = Date.now();
    if (!force && now - r.lastNotify < NOTIFY_MS) return;
    r.lastNotify = now;
    r.handle.notify();
  }

  function updateMetadata() {
    if (!r.handle || !r.doc) return;
    const meta = sessionMeta();
    r.handle.update({ title: meta.title, subtitle: meta.subtitle, artwork: meta.artwork });
  }

  // What the mini-player and the lock screen show (the audio session's getState()): the chapter-relative position.
  function sessionState() {
    const t = currentTimeline();
    const active = r.active;
    let pos = null;
    if (lockScreen.POSITION_SCOPE === 'part') {
      const el = active?.el;
      const duration = el && Number.isFinite(el.duration) && el.duration > 0 ? el.duration : (active?.part?.duration_ms || 0) / 1000;
      pos = lockScreen.positionState({ scope: 'part', partPosition: el ? positionOf(el) : 0, partDuration: duration, rate: r.speed });
    } else if (t) {
      const knownCount = knownFor(candidateVoice()).size;
      const partReady = knownCount !== r.reportedKnown || r.chapter !== r.reportedChapter;
      r.reportedMs = lockScreen.smoothDuration(r.reportedMs, t.durationMs, { partReady });
      r.reportedKnown = knownCount;
      r.reportedChapter = r.chapter;
      pos = lockScreen.positionState({ durationMs: r.reportedMs, elapsedMs: t.elapsedMs, rate: r.speed });
    }
    return {
      playing: r.playing,
      waiting: r.waiting,
      position: pos ? pos.position : 0,
      duration: pos ? pos.duration : 0,
      rate: r.speed,
      canSeek: !!pos,
      statusText: r.status.text,
    };
  }

  function seekToSession(seconds) {
    if (!Number.isFinite(Number(seconds))) return;
    if (lockScreen.POSITION_SCOPE === 'part') {
      if (r.active) {
        seekEl(r.active.el, Number(seconds));
        syncFromAudio();
        notify(true);
      }
      return;
    }
    engine.seekChapterMs(Number(seconds) * 1000);
  }

  /* ───────────── the timeline ───────────── */

  function knownFor(voice) {
    const key = voiceKey(voice);
    if (!r.known.has(key)) r.known.set(key, new Map());
    return r.known.get(key);
  }

  function currentTimeline() {
    const doc = r.doc;
    if (!doc?.blocks.length) return null;
    const voice = candidateVoice();
    const known = knownFor(voice);
    const perChar = timeline.msPerChar(known, doc.blocks);
    const [start, end] = chapterRange(doc, r.chapter);
    const active = r.active;
    const t = timeline.chapterTimeline({
      blocks: doc.blocks,
      start,
      end,
      known,
      perChar,
      active: active ? active.part : null,
      activeMs: active ? positionOf(active.el) * 1000 : 0,
      cursor: r.cursor,
      offsetMs: active ? 0 : resumeOffset(r.resume, voice, r.cursor),
      finished: isFinished(),   // "Finished" shows the end of the last chapter, not the start of its last paragraph
    });
    return { ...t, start, end, perChar };
  }

  /* ───────────── server ───────────── */

  function renderKey(fromBlock, voice) { return `${r.doc?.id}|${voice?.model}|${voice?.voice}|${fromBlock}`; }

  function setParts(parts, voice) {
    r.parts = Array.isArray(parts) ? parts : [];
    timeline.remember(knownFor(voice), r.parts);
    emitState();
  }

  // Both return false when a newer start or stop made the response stale, so it never overwrites r.parts (RE-07).
  async function requestRender(fromBlock, voice, token) {
    const data = await apiPost(docUrl(r.doc.id, '/render'), { from_block: fromBlock, model: voice.model, voice: voice.voice },
      { accept: 'json', timeoutMs: REQUEST_TIMEOUT_MS });
    if (token !== r.token) return false;
    setParts(data?.parts, voice);
    r.renderKey = renderKey(fromBlock, voice);
    return true;
  }

  async function refreshParts(fromBlock, voice, token) {
    const qs = new URLSearchParams({ from_block: String(fromBlock), model: voice.model, voice: voice.voice });
    const data = await apiGet(docUrl(r.doc.id, `/parts?${qs}`), { accept: 'json', timeoutMs: REQUEST_TIMEOUT_MS });
    if (token !== r.token) return false;
    setParts(data?.parts, voice);
    return true;
  }

  /* ───────────── playback ───────────── */

  function haltAudio() {
    // The keep-alive keeps looping, and an in-flight unlock pauses itself.
    if (r.els) {
      for (const el of r.els) {
        if (el !== r.keepAlive && !el.dataset.unlocking) el.pause();
      }
    }
    r.active = null;
    r.preloaded = null;
    r.lookahead += 1;
    stopTicker();
  }

  async function startAt(index, offsetMs = 0, { reason = 'jump', resume = false } = {}) {
    const doc = r.doc;
    if (!doc?.blocks.length) return;
    if (app.get().recording) {
      setStatus(COPY.recording);
      return;
    }
    const at = clamp(index, 0, doc.blocks.length - 1);
    const token = ++r.token;
    haltAudio();
    setCursor(at, -1, reason);
    // Claiming displaces any other player (a Speak take pauses, Live stops) and installs the lock-screen
    // controls at once, so they work while the first part renders (RE-03).
    claim();
    setPlaying(true);
    setWaiting(true);
    setStatus('');
    if (r.waitText) showPreparing(r.waitText);
    dismissPlayError();
    const voice = await readyVoice(token);
    if (token !== r.token) return;
    if (!voice?.model || !voice.voice) {
      setWaiting(false);
      setPlaying(false);
      releaseSession();
      needVoice();
      return;
    }
    emit('voice', { ...voice });
    if (!await loadVoiceModel(voice.model, token)) return;
    if (token !== r.token) return;
    // The saved offset is used once, by the first start from the saved paragraph (cleared when audio starts).
    const offset = resume ? resumeOffset(r.resume, voice, at) : offsetMs;
    let failures = 0;
    let rendered = false;
    let lastRender = 0;
    let rerenderWait = RERENDER_AFTER_MS;
    for (;;) {
      try {
        const seen = rendered ? locate(at, r.parts) : null;
        // The single server job can be replaced (another tab, another voice): our window then vanishes, or
        // the server keeps answering our part as 'pending', in no job (its job's first part is 'rendering'
        // at once). Ask again after 3 s (RE-08); asking for a job that is still ours changes nothing. Each
        // later ask waits twice as long (6, 12, then every 24 s): two listeners waiting at once would otherwise
        // take the job from each other every 3 s, and a part slower than that would never finish for either.
        const again = rendered && (!seen || seen.part.status === 'pending') && Date.now() - lastRender > rerenderWait;
        if (!rendered || again) {
          if (again) rerenderWait = Math.min(rerenderWait * 2, RERENDER_MAX_MS);
          lastRender = Date.now();
          await requestRender(at, voice, token);
          rendered = true;
        } else {
          await refreshParts(at, voice, token);
        }
        failures = 0;
        if (token === r.token) settleStatus();
      } catch (error) {
        if (token !== r.token) return;
        failures += 1;
        if (!isRetryable(error) || failures >= MAX_FAILURES) {
          fail(error?.status ? error.message : COPY.unreachable);
          return;
        }
        setStatus(COPY.reconnecting, 'retry');
      }
      if (token !== r.token) return;
      const found = rendered ? locate(at, r.parts) : null;   // r.parts may still be an older window
      if (found?.part.status === 'error') {
        fail(found.part.error || COPY.partFailed);
        return;
      }
      if (found?.part.status === 'ready' && found.part.url) {
        try {
          await beginPart(found.part, found.offsetMs + offset, token);
        } catch (error) {
          if (token === r.token) fail(error?.message || String(error));
        }
        return;
      }
      await wait(backoffMs(failures));
      if (token !== r.token) return;
    }
  }

  async function beginPart(part, offsetMs, token, el = null) {
    const url = safeUrl(part.url, { media: true });
    if (!url) throw new Error(COPY.audioFailed);
    const target = el || (r.active ? otherEl(r.active.el) : elements()[0]);
    const previous = r.active?.el;
    if (target === r.keepAlive) stopKeepAlive();
    // Same part but a fresh signed URL (expired-link recovery) loads again. Also reload after a failed load:
    // a retry within the same second gets a byte-identical link (RE-12).
    if (target.dataset.part !== part.id || target.getAttribute('src') !== url || target.error) {
      setSrc(target, url, part.id);
      if (offsetMs > 0) seekEl(target, offsetMs / 1000);
    } else if (target !== r.preloaded?.el || offsetMs > 0) {
      seekEl(target, offsetMs / 1000);
    }
    target.muted = false;
    applyRate(target);
    r.active = { el: target, part };
    r.preloaded = null;
    timeline.remember(knownFor(currentVoice()), [part]);
    if (previous && previous !== target) previous.pause();
    try {
      await target.play();
    } catch (error) {
      if (token !== r.token || r.active?.el !== target) return;
      if (error?.name === 'NotAllowedError') {
        setWaiting(false);
        setPlaying(false);
        setStatus(COPY.tapPlay);
        return;
      }
      // A failed load also fires 'error', and handleAudioError retries it.
      if (error?.name === 'AbortError' || target.error) return;
      throw error;
    }
    target.dataset.unlocked = '1';
    if (token !== r.token || r.active?.el !== target) return;
    stopKeepAlive();
    onPartStarted(part, token);
  }

  // When a part starts: clear the one-shot resume and the retry status, refresh the lock screen, sync the
  // cursor, save, re-point the server job at the playing part, then look ahead (RE-17).
  function onPartStarted(part, token) {
    setWaiting(false);
    setPlaying(true);
    r.resume = null;
    if (r.status.kind === 'retry') setStatus('');
    dismissPlayError();
    claim();
    ensureArtwork();
    updateMetadata();
    notify(true);
    syncFromAudio();
    saveProgress();
    const voice = currentVoice();
    const repoint = voice && r.renderKey !== renderKey(part.first_block, voice)
      ? requestRender(part.first_block, voice, token).catch(() => {})
      : Promise.resolve();
    repoint.finally(() => {
      if (token === r.token) watchNext();
    });
  }

  // Poll (gently) until the part after the active one is ready, then load it into the idle element. Stops
  // while paused; play() starts it again (RE-18).
  async function watchNext() {
    const loop = ++r.lookahead;
    const active = r.active;
    const voice = currentVoice();
    const token = r.token;
    if (!active || !voice || !r.doc) return;
    const nextBlock = active.part.last_block + 1;
    if (nextBlock >= r.doc.blocks.length) return;
    let failures = 0;
    while (loop === r.lookahead && r.active === active) {
      const next = r.parts.find((part) => part.first_block === nextBlock);
      if (next?.status === 'error') return;
      if (next?.status === 'ready' && next.url) {
        preload(next);
        return;
      }
      await wait(Math.max(LOOKAHEAD_POLL_MS, backoffMs(failures)));
      if (loop !== r.lookahead || r.active !== active || !r.playing) return;
      try {
        await refreshParts(active.part.first_block, voice, token);
        failures = 0;
      } catch {
        failures = Math.min(failures + 1, MAX_FAILURES);
      }
    }
  }

  function preload(part) {
    const active = r.active;
    if (!active) return;
    const url = safeUrl(part.url, { media: true });
    if (!url) return;
    const el = otherEl(active.el);
    if (el === r.keepAlive) stopKeepAlive();
    if (el.dataset.part !== part.id || el.getAttribute('src') !== url) setSrc(el, url, part.id);
    else seekEl(el, 0);
    el.preload = 'auto';
    applyRate(el);
    r.preloaded = { el, part };
  }

  function handleEnded(event) {
    const active = r.active;
    if (!active || event.target !== active.el || !r.playing) return;
    sleepTick();
    const nextBlock = active.part.last_block + 1;
    if (nextBlock >= r.doc.blocks.length) {
      finishDocument();
      return;
    }
    // End of chapter (or of N chapters): stop before the part that starts the next one.
    if (sleepTimer.stopsBefore(r.sleep, nextBlock)) {
      stopForSleep(nextBlock);
      return;
    }
    const preloaded = r.preloaded;
    if (preloaded && preloaded.part.first_block === nextBlock) {
      const token = r.token;
      beginPart(preloaded.part, 0, token, preloaded.el).catch((error) => {
        if (token === r.token) fail(error?.message || String(error));
      });
      return;
    }
    const next = r.parts.find((part) => part.first_block === nextBlock);
    if (next?.status === 'error') {
      // The finished paragraph was heard to its end: Play (and the saved position) retry the one that failed.
      haltAudio();
      setCursor(nextBlock, -1, 'playback');
      fail(next.error || COPY.nextFailed);
      return;
    }
    startAt(nextBlock, 0, { reason: 'playback' });
  }

  // A failed load on the active element is most often an expired signed URL or a dropped connection: refetch
  // the parts and pick up where it stopped, backing off, and give up after MAX_FAILURES in a row (the count
  // resets once audio is actually playing) (RE-20, RE-21).
  async function handleAudioError(event) {
    const el = event.target;
    if (!holdsPart(el)) return;
    if (r.preloaded?.el === el) {
      r.preloaded = null;   // handleEnded falls back to a fresh start for that part
      delete el.dataset.part;
      return;
    }
    const active = r.active;
    if (!active || active.el !== el || r.recovering) return;
    r.recovering = true;
    const token = r.token;
    const stale = () => token !== r.token || r.active !== active;
    const resumeAt = positionOf(el) * 1000;
    try {
      let fresh = null;
      for (;;) {
        r.audioFailures += 1;
        if (r.audioFailures >= MAX_FAILURES) {
          fail(COPY.audioFailed);
          return;
        }
        if (r.audioFailures > 1) {
          setStatus(COPY.reconnecting, 'retry');
          await wait(backoffMs(r.audioFailures - 1));
          if (stale()) return;
        }
        try {
          await refreshParts(active.part.first_block, currentVoice(), token);
        } catch (error) {
          if (stale()) return;
          if (!isRetryable(error)) throw error;
          continue;
        }
        if (stale()) return;
        fresh = r.parts.find((part) => part.first_block === active.part.first_block && part.status === 'ready' && part.url);
        break;
      }
      r.recovering = false;   // the retried source may fail too, and that error has to be handled
      if (!fresh) {
        r.resume = progressPayload();   // where it stopped, for this start or the next play
        if (r.playing) startAt(r.cursor, 0, { reason: 'playback', resume: true });
        else {
          r.token += 1;
          haltAudio();
        }
        return;
      }
      if (r.playing) {
        await beginPart(fresh, resumeAt, token, el);
      } else {
        const url = safeUrl(fresh.url, { media: true });
        setSrc(el, url, fresh.id);
        seekEl(el, resumeAt / 1000);
        r.active = { el, part: fresh };
      }
    } catch (error) {
      if (token === r.token) fail(error?.message || String(error));
    } finally {
      r.recovering = false;
    }
  }

  function handlePause(event) {
    const el = event.target;
    if (el === r.switching) return;
    sleepTick();
    // The system paused the keep-alive while a part renders (AirPods out, a call): the listener's pause.
    // Pause events queued before the keep-alive started (haltAudio pausing the same element) are ignored.
    if (el === r.keepAlive && el.paused && el.dataset.keepAlivePlaying && r.waiting && r.playing) {
      engine.pause();
      return;
    }
    if (el !== r.active?.el || el.ended || r.waiting || !r.playing) return;
    // Paused by the system (headphones removed, a call, another app): never auto-resumed (RE-27).
    setPlaying(false);
    saveProgress();
  }

  function handlePlay(event) {
    if (event.target === r.active?.el && !r.playing) setPlaying(true);
  }

  // The failure counter and "Reconnecting…" clear once audio really plays (RE-22).
  function handlePlaying(event) {
    if (event.target !== r.active?.el) return;
    r.audioFailures = 0;
    if (r.status.kind === 'retry') setStatus('');
  }

  // Finish: the cursor on the last block, saved past it (the library reads that as 100%), the lock screen
  // cleared (RE-38).
  function finishDocument() {
    const doc = r.doc;
    r.token += 1;
    haltAudio();
    setCursor(doc.blocks.length - 1, -1, 'playback');
    r.resume = { ...progressPayload(), block_index: doc.blocks.length };
    setPlaying(false);
    setWaiting(false);
    setStatus(COPY.finished);
    saveProgress();
    releaseSession();
    emit('finished', { docId: doc.id });
    emitState();
  }

  function stopForSleep(nextBlock) {
    r.token += 1;
    haltAudio();
    setCursor(nextBlock, -1, 'playback');
    r.resume = null;
    r.sleep = sleepTimer.OFF;
    applyVolume(1);
    setWaiting(false);
    setPlaying(false);
    setStatus(COPY.sleepEnded);
    saveProgress();
    emitState();
  }

  function fail(message) {
    // Like a stop: the next play starts where this one stopped (RE-23).
    if (r.active) r.resume = progressPayload();
    r.token += 1;
    r.audioFailures = 0;
    haltAudio();
    setWaiting(false);
    setPlaying(false);
    setStatus(`Couldn't play: ${message}`, 'error');
    r.playError = toast(`Couldn't play: ${message}`, { kind: 'error', id: 'reader-play', action: { label: 'Retry', run: () => engine.play() } });
    emit('error', { message });
  }

  // Stop where it is; the next play starts there, as after a reload (RE-24).
  function stop() {
    if (r.active) r.resume = progressPayload();
    r.token += 1;
    r.audioFailures = 0;
    r.renderKey = '';
    haltAudio();
    setWaiting(false);
    setPlaying(false);
    setStatus('');
  }

  /* ───────────── tracking (RE-51, REV-34) ───────────── */

  function startTicker() {
    if (!r.raf && !docHidden() && typeof requestAnimationFrame === 'function') r.raf = requestAnimationFrame(tickFrame);
  }

  function stopTicker() {
    if (r.raf && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(r.raf);
    r.raf = 0;
  }

  function tickFrame() {
    r.raf = 0;
    if (!r.playing) return;
    track();
    if (!docHidden()) startTicker();
  }

  // One tracking step: the sentence, the sleep timer, the 10 s progress save and the lock screen (4 Hz).
  // Called from the frame ticker and from the active element's timeupdate events.
  function track() {
    const active = r.active;
    if (active && !active.el.paused) syncFromAudio();
    sleepTick();
    if (r.playing && !r.waiting && r.active && Date.now() - r.lastSaveAt >= SAVE_EVERY_MS) saveProgress();
    notify(false);
    emitState();
  }

  function syncFromAudio() {
    const active = r.active;
    if (!active || !r.doc) return;
    const ms = positionOf(active.el) * 1000;
    const el = active.el;
    const durationMs = Number.isFinite(el.duration) && el.duration > 0 ? el.duration * 1000 : 0;
    const { block, sentence } = trackPosition(active.part, ms, { blocks: r.doc.blocks, durationMs });
    if (block !== r.cursor || sentence !== r.sentence) setCursor(block, sentence, 'playback');
  }

  function setCursor(index, sentence = -1, reason = 'playback') {
    const doc = r.doc;
    if (!doc?.blocks.length) return;
    const at = clamp(index, 0, doc.blocks.length - 1);
    const moved = at !== r.cursor || sentence !== r.sentence;
    r.cursor = at;
    r.sentence = sentence;
    const chapter = r.chapterAt[at] ?? 0;
    const chapterChanged = chapter !== r.chapter;
    r.chapter = chapter;
    if (chapterChanged) {
      if (reason === 'jump' && r.sleep.mode === 'chapters') {
        r.sleep = Object.freeze({ ...r.sleep, boundaryBlock: chapterStart(doc, chapter + r.sleep.chapters) });
      }
      updateMetadata();
    }
    if (moved || reason !== 'playback') emit('cursor', { block: at, sentence, chapter, reason });
    emitState();
  }

  /* ───────────── sleep timer ───────────── */

  function sleepTick() {
    if (r.sleep.mode !== 'minutes') return;
    const fade = prefs.get().sleepFade !== false;
    r.sleep = sleepTimer.tick(r.sleep, { playing: r.playing && !r.waiting, now: Date.now(), fade });
    const volume = sleepTimer.volumeFor(r.sleep, { fade });
    applyVolume(volume === null ? 1 : volume);
    if (sleepTimer.expired(r.sleep)) {
      r.sleep = sleepTimer.OFF;
      applyVolume(1);
      engine.pause();
      setStatus(COPY.sleepEnded);
    }
  }

  function sleepRemainingMs(t) {
    if (r.sleep.mode === 'minutes') return r.sleep.remainingMs;
    if (r.sleep.mode !== 'chapters' || !t) return 0;
    let ms = Math.max(0, t.durationMs - t.elapsedMs);
    const boundary = Math.min(r.doc.blocks.length, r.sleep.boundaryBlock);
    for (let b = t.end; b < boundary; b += 1) ms += charsOf(r.doc.blocks[b]) * t.perChar;
    return ms / (r.speed || 1);
  }

  /* ───────────── wake lock ───────────── */

  function readingVisible() {
    const route = app.get().route || {};
    return !!r.doc && route.view === 'reader' && route.id === r.doc.id;
  }

  function releaseLock(lock) {
    Promise.resolve(lock && lock.release && lock.release()).catch(() => {});
  }

  /**
   * Hold a screen wake lock while the reading view plays. Each change of mind takes a token: a request
   * answered after a newer change (Play, Pause, Play quicker than the first answer) is released at once, so only
   * the latest wanted request is ever kept. A lock the system releases (the page hidden, battery saver) clears the
   * state, so the next visible and playing moment asks again; nothing re-asks from the release itself.
   */
  function updateWakeLock() {
    const want = !!(r.playing && prefs.get().wakeLock && !docHidden() && readingVisible());
    if (want === r.wakeWanted) return;
    r.wakeWanted = want;
    const token = (r.wakeToken += 1);
    if (want) {
      let request = null;
      try {
        request = typeof navigator !== 'undefined' && navigator.wakeLock && typeof navigator.wakeLock.request === 'function'
          ? navigator.wakeLock.request('screen')
          : null;
      } catch {
        request = null;
      }
      if (!request) return;
      Promise.resolve(request).then((lock) => {
        if (!r.wakeWanted || r.destroyed || token !== r.wakeToken) {
          releaseLock(lock);
          return;
        }
        r.wakeLock = lock;
        if (lock && typeof lock.addEventListener === 'function') {
          lock.addEventListener('release', () => {
            if (r.wakeLock !== lock) return;   // released here: the state already moved on
            r.wakeLock = null;
            r.wakeWanted = false;
          }, { once: true });
        }
      }).catch(() => { /* unsupported or refused: fail silently */ });
    } else if (r.wakeLock) {
      const lock = r.wakeLock;
      r.wakeLock = null;
      releaseLock(lock);
    }
  }

  /* ───────────── saved position (RE-35, RE-36) ───────────── */

  function isFinished() {
    return !!r.doc && !r.active && (r.resume?.block_index ?? -1) >= r.doc.blocks.length;
  }

  function progressPayload() {
    syncFromAudio();
    let offset = 0;
    const active = r.active;
    const voice = candidateVoice();
    const here = active && locate(r.cursor, [active.part]);
    if (here) {
      const marks = active.part.sentences || [];
      const sentenceStart = marks.find((mark) => mark.block === r.cursor && mark.index === r.sentence)?.start_ms;
      offset = Math.max(0, Math.round((sentenceStart ?? positionOf(active.el) * 1000) - here.offsetMs));
    } else if (!active) {
      offset = resumeOffset(r.resume, voice, r.cursor);   // not started yet: keep the saved offset
    }
    return {
      block_index: isFinished() ? r.doc.blocks.length : r.cursor,
      offset_ms: offset,
      model: voice?.model || null,
      voice: voice?.voice || null,
      speed: r.speed,
    };
  }

  function saveProgress({ keepalive = false } = {}) {
    const doc = r.doc;
    if (!doc?.blocks.length) return Promise.resolve();
    r.lastSaveAt = Date.now();
    const payload = progressPayload();
    const body = JSON.stringify(payload);
    if (body === r.lastSaved) return Promise.resolve();
    r.lastSaved = body;
    return apiPut(docUrl(doc.id, '/progress'), payload, { keepalive, accept: 'json', timeoutMs: REQUEST_TIMEOUT_MS })
      .then((data) => {
        if (data?.progress) emit('progress', { docId: doc.id, progress: data.progress });
      })
      .catch((error) => {
        if (isRetryable(error) && r.lastSaved === body) r.lastSaved = '';   // try again with the next save
      });
  }

  /* ───────────── page lifecycle ───────────── */

  function bindGlobal() {
    if (r.bound) return;
    r.bound = true;
    const onVisibility = () => {
      if (docHidden()) {
        saveProgress({ keepalive: true });
        stopTicker();
      } else {
        if (r.playing) startTicker();
        flushState();
      }
      updateWakeLock();
    };
    const onPageHide = () => { saveProgress({ keepalive: true }); };
    if (typeof document !== 'undefined') {
      document.addEventListener('visibilitychange', onVisibility);
      r.globalOffs.push(() => document.removeEventListener('visibilitychange', onVisibility));
    }
    if (typeof window !== 'undefined' && window.addEventListener) {
      window.addEventListener('pagehide', onPageHide);
      r.globalOffs.push(() => window.removeEventListener('pagehide', onPageHide));
    }
    r.globalOffs.push(app.select((s) => s.route, () => updateWakeLock()));
    r.globalOffs.push(app.select((s) => s.prefs?.wakeLock, () => updateWakeLock()));
    r.globalOffs.push(onAppEvent('voices:changed', (detail) => {
      if (!detail || detail.what === 'named') {
        r.namesLoading = null;
        if (r.names.size || String(candidateVoice()?.voice || '').startsWith('voice:')) loadNames();
      }
    }));
  }

  /* ───────────── the public API ───────────── */

  function blockCount() { return r.doc?.blocks?.length || 0; }

  function elapsedInBlockMs() {
    const active = r.active;
    const here = active && locate(r.cursor, [active.part]);
    if (!here) return 0;
    return Math.max(0, (positionOf(active.el) * 1000 - here.offsetMs) / (active.el.playbackRate || 1));
  }

  function snapshot() {
    const doc = r.doc;
    const t = currentTimeline();
    const voice = candidateVoice();
    const chapters = doc?.chapters || [];
    const zero = { elapsedMs: 0, durationMs: 0, bufferedMs: 0, estimated: false, lastBlockMs: 0 };
    const tl = t ? {
      elapsedMs: t.elapsedMs,
      durationMs: t.durationMs,
      bufferedMs: t.bufferedMs,
      estimated: t.estimated,
      lastBlockMs: timeline.timeOfBlock(t.segments, Math.max(t.start, t.end - 1)),
    } : zero;
    return Object.freeze({
      docId: doc?.id || '',
      cursor: r.cursor,
      sentence: r.sentence,
      chapter: r.chapter,
      playing: r.playing,
      waiting: r.waiting,
      busy: r.waiting && !!r.preparing,
      speed: r.speed,
      voice: voice ? { model: voice.model, voice: voice.voice } : null,
      voiceLabel: labelFor(voice),
      status: { ...r.status },
      finished: isFinished(),
      timeline: tl,
      sleep: { mode: r.sleep.mode, remainingMs: sleepRemainingMs(t), chapters: r.sleep.chapters, minutes: r.sleep.minutes },
      title: String(doc?.title || ''),
      author: String(doc?.author || ''),
      chapterTitle: doc ? lockScreen.chapterTitle(doc, r.chapter) : '',
      chapterCount: chapters.length,
      blockCount: blockCount(),
      remainingMs: t ? timeline.documentRemainingMs({ blocks: doc.blocks, chapterEnd: t.end, perChar: t.perChar, timeline: t }) : 0,
      canSeek: !!t && t.durationMs > 0,
    });
  }

  const engine = {
    /**
     * Load a document (GET /v1/reader/documents/{id}): the cursor goes to progress.block_index (or 0), the speed
     * to progress.speed (else the global default). Does not play. The same id again is a no-op; a new id
     * unloads the old document first.
     */
    load(doc) {
      if (!doc || !doc.id) throw new TypeError('load() needs a document with an id');
      if (r.doc && r.doc.id === doc.id) return;
      if (r.doc) engine.unload();
      bindGlobal();
      elements();
      const blocks = Array.isArray(doc.blocks) ? doc.blocks : [];
      const chapters = Array.isArray(doc.chapters) ? doc.chapters : [];
      r.doc = { ...doc, blocks, chapters };
      r.chapterAt = chapterMap(blocks.length, chapters);
      r.voice = null;   // resolved when playback starts: the models may still be loading
      r.resume = doc.progress || null;
      r.parts = [];
      r.renderKey = '';
      r.known = new Map();
      r.status = { text: '', kind: '' };
      r.sleep = sleepTimer.OFF;
      r.speed = documentSpeed(doc.progress, storedSpeed(read(KEYS.readerSpeed, null)));
      for (const el of elements()) applyRate(el);
      r.cursor = blocks.length ? clamp(doc.progress?.block_index ?? 0, 0, blocks.length - 1) : 0;
      r.sentence = -1;
      r.chapter = r.chapterAt[r.cursor] ?? 0;
      // Opening and leaving without playing leaves the saved position as it was (RE-33).
      r.lastSaved = blocks.length ? JSON.stringify(progressPayload()) : '';
      emit('cursor', { block: r.cursor, sentence: -1, chapter: r.chapter, reason: 'open' });
      emit('status', { ...r.status });
      if (String(candidateVoice()?.voice || '').startsWith('voice:')) loadNames();
      scheduleArtwork();
      flushState();
    },

    /**
     * Stop, save the position and release the audio session (the mini-player's Close player, the lock
     * screen's stop, opening another document). The document stays loaded, so Play resumes where it stopped;
     * {forget: true} also drops it (the document was deleted) and skips the save.
     * @returns {Promise<void>} the progress save
     */
    unload({ forget = false } = {}) {
      if (!r.doc) return Promise.resolve();
      const saving = forget ? Promise.resolve() : saveProgress();
      stop();
      releaseSession();
      r.sleep = sleepTimer.OFF;
      applyVolume(1);
      dismissPlayError();
      clearTimeout(r.speedTimer);
      if (forget) {
        r.doc = null;
        r.chapterAt = [];
        r.cursor = 0;
        r.sentence = -1;
        r.chapter = 0;
        r.resume = null;
        r.parts = [];
        r.lastSaved = '';
      }
      updateWakeLock();
      flushState();
      return saving;
    },

    /** Play from the cursor (unlocking the elements inside this gesture), or resume the paused part (RE-30). */
    play() {
      if (!r.doc?.blocks.length || r.playing) return;
      unlock();
      if (app.get().recording) {
        setStatus(COPY.recording);
        return;
      }
      setStatus('');
      dismissPlayError();
      const active = r.active;
      if (active && holdsPart(active.el)) {
        claim();
        setPlaying(true);
        applyRate(active.el);
        Promise.resolve(active.el.play()).then(() => {
          active.el.dataset.unlocked = '1';
          if (r.active !== active || !r.playing) return;
          updateMetadata();
          notify(true);
          if (!r.preloaded) watchNext();
        }).catch((error) => {
          if (r.active !== active) return;
          if (error?.name === 'NotAllowedError') {
            setPlaying(false);
            setStatus(COPY.tapPlay);
          } else if (error?.name !== 'AbortError' && !active.el.error) {
            startAt(r.cursor, 0, { reason: 'playback' });
          }
        });
        return;
      }
      startAt(r.cursor, 0, { resume: true, reason: 'jump' });
    },

    /** Pause; while waiting it cancels the start. Always saves (RE-31). */
    pause() {
      if (!r.playing) return;
      const active = r.active;
      if (r.waiting || !active) {
        r.token += 1;
        haltAudio();
        setWaiting(false);
        setPlaying(false);
      } else {
        setWaiting(false);
        setPlaying(false);
        active.el.pause();
      }
      saveProgress();
    },

    toggle() {
      if (r.playing) engine.pause();
      else engine.play();
    },

    /** Unlock, move to the block and play (a tap on a paragraph, a chapter pick). */
    playFrom(block) {
      unlock();
      engine.jumpTo(block);
      if (!r.playing) engine.play();
    },

    /**
     * Move playback (or the paused position) to the start of a block (RE-41): inside the active part it seeks;
     * the preloaded part starts; paused, the cursor moves and the position is saved; playing, it starts there.
     */
    jumpTo(block, { offsetMs = 0, reason = 'jump' } = {}) {
      const doc = r.doc;
      if (!doc?.blocks.length) return;
      const index = clamp(Number(block) || 0, 0, doc.blocks.length - 1);
      r.resume = null;   // a finished (100%) position no longer applies (RE-40)
      const active = r.active;
      if (active && !r.waiting) {
        const here = locate(index, [active.part]);
        if (here) {
          seekEl(active.el, (here.offsetMs + offsetMs) / 1000);
          setCursor(index, -1, reason);
          notify(true);
          if (!r.playing) saveProgress();
          emitState();
          return;
        }
        if (r.playing && r.preloaded?.part.first_block === index) {
          const token = r.token;
          beginPart(r.preloaded.part, offsetMs, token, r.preloaded.el).catch((error) => {
            if (token === r.token) fail(error?.message || String(error));
          });
          setCursor(index, -1, reason);
          return;
        }
      }
      if (!r.playing) {
        r.token += 1;
        haltAudio();
        setCursor(index, -1, reason);
        if (offsetMs > 0) {
          const voice = candidateVoice();
          r.resume = { block_index: index, offset_ms: Math.round(offsetMs), model: voice?.model || null, voice: voice?.voice || null };
        }
        saveProgress();
        emitState();
        return;
      }
      startAt(index, offsetMs, { reason });
    },

    /** Previous or next paragraph; forward at the last paragraph does nothing (RE-42). */
    skipBlock(delta) {
      if (!blockCount()) return;
      const target = clamp(r.cursor + delta, 0, blockCount() - 1);
      if (target === r.cursor && delta > 0) return;
      engine.jumpTo(target);
    },

    /** Previous or next sentence inside the playing part; elsewhere it falls back to the paragraph. */
    skipSentence(delta) {
      const active = r.active;
      if (active && !r.waiting) {
        const mark = sentenceStep(active.part, r.cursor, r.sentence, delta);
        if (mark) {
          r.resume = null;
          seekEl(active.el, (mark.start_ms || 0) / 1000);
          setCursor(mark.block, mark.index, 'jump');
          notify(true);
          if (!r.playing) saveProgress();
          emitState();
          return;
        }
      }
      engine.skipBlock(delta);
    },

    /** The prev/next buttons: a paragraph or a sentence, per prefs.stepUnit. */
    step(delta) {
      if (prefs.get().stepUnit === 'sentence') engine.skipSentence(delta);
      else engine.skipBlock(delta);
    },

    /**
     * Skip by seconds from where the element plays (a parked seek counts): before 0 goes to the previous block,
     * past the part's end to the next; ignored while waiting (RE-43). With prefs.snapToSentence it lands on the
     * start of the sentence there (never backwards for a forward skip).
     */
    seekBy(seconds) {
      const active = r.active;
      if (!active || r.waiting) return;
      const el = active.el;
      const from = positionOf(el);
      let to = from + (Number(seconds) || 0);
      if (to < 0) {
        engine.jumpTo(active.part.first_block - 1);
        return;
      }
      const duration = Number.isFinite(el.duration) && el.duration > 0 ? el.duration : (active.part.duration_ms || 0) / 1000;
      if (duration && to >= duration) {
        if (active.part.last_block + 1 < blockCount()) engine.jumpTo(active.part.last_block + 1);
        return;
      }
      if (prefs.get().snapToSentence !== false) {
        const snapped = sentenceStartMs(active.part, to * 1000) / 1000;
        if (!(seconds > 0 && snapped <= from)) to = snapped;
      }
      r.resume = null;
      seekEl(el, to);
      syncFromAudio();
      notify(true);
      emitState();
    },

    /**
     * Seek to a position on the chapter timeline (the scrubber, the lock screen's seekto): inside a ready part
     * at the offset there, elsewhere at the block's start.
     * @param {number} ms
     * @param {{step?: number, atLeastMs?: number}} [options] step: 1 or -1 when the scrubber was stepped (an arrow,
     *   PageUp/PageDown, an assistive increment) rather than dragged: a forward step that would land back on the
     *   start of the unrendered block it is in moves on to the next block start past atLeastMs (the first value the
     *   scrubber shows as a move), so it always moves (REV-13, timeline.stepSeekMs)
     */
    seekChapterMs(ms, { step = 0, atLeastMs = 0 } = {}) {
      const t = currentTimeline();
      if (!t) return;
      const to = step ? timeline.stepSeekMs(t.segments, t.elapsedMs, ms, step, { atLeastMs }) : ms;
      const target = timeline.seekTarget(t.segments, to, { blocks: r.doc.blocks });
      if (!target) return;
      const active = r.active;
      if (active && !r.waiting && target.part && target.part.id === active.part.id) {
        r.resume = null;
        seekEl(active.el, target.partOffsetMs / 1000);
        syncFromAudio();
        notify(true);
        if (!r.playing) saveProgress();
        emitState();
        return;
      }
      engine.jumpTo(target.block, { offsetMs: target.part ? target.blockOffsetMs : 0 });
    },

    /** Lock-screen previous track, per prefs.trackUnit. */
    prevTrack() {
      const target = trackTargetFor('prev');
      if (target !== null) engine.jumpTo(target);
    },

    /** Lock-screen next track, per prefs.trackUnit. */
    nextTrack() {
      const target = trackTargetFor('next');
      if (target !== null) engine.jumpTo(target);
    },

    /**
     * 0.5–3.5×, rounded to 0.05, on both elements with pitch preserved; saved to the document's progress and
     * to KEYS.readerSpeed as the default for new documents.
     * @returns {number} the speed applied
     */
    setSpeed(rate) {
      const speed = clampSpeed(rate);
      if (speed === null) return r.speed;
      r.speed = speed;
      write(KEYS.readerSpeed, String(speed));
      for (const el of elements()) applyRate(el);
      notify(true);
      emitState();
      clearTimeout(r.speedTimer);
      r.speedTimer = setTimeout(() => { r.speedTimer = null; saveProgress(); }, SPEED_SAVE_MS);
      return speed;
    },

    /**
     * Use a voice (RE-66): both a model and a voice are needed; it is remembered as the Reader voice. A change
     * drops the one-shot resume and re-renders the current paragraph when playing, else halts.
     * @returns {boolean} whether it changed
     * @throws {Error} 'Choose a model and a voice'
     */
    setVoice({ model, voice } = {}) {
      if (!model || !voice) throw new Error(COPY.chooseBoth);
      const current = r.voice || currentVoice();
      const changed = model !== current?.model || voice !== current?.voice;
      r.voice = { model: String(model), voice: String(voice) };
      writeJson(KEYS.readerVoice, r.voice);
      if (String(voice).startsWith('voice:')) loadNames();
      emit('voice', { ...r.voice });
      updateMetadata();
      emitState();
      if (!changed) return false;
      r.resume = null;
      if (!r.doc) return true;
      if (r.playing) {
        r.waitText = COPY.voiceChanged;
        startAt(r.cursor, 0, { reason: 'playback' });
      } else {
        r.token += 1;
        haltAudio();
        saveProgress();
      }
      return true;
    },

    /**
     * Set the sleep timer: null (off), {minutes} or {chapters: n} (the end of the current chapter and n - 1
     * more).
     */
    setSleep(spec) {
      const doc = r.doc;
      const chapters = Number(spec?.chapters);
      const boundaryBlock = doc && Number.isFinite(chapters) && chapters > 0 ? chapterStart(doc, r.chapter + Math.round(chapters)) : -1;
      r.sleep = sleepTimer.start(spec || null, { boundaryBlock });
      applyVolume(1);
      if (r.status.text === COPY.sleepEnded) setStatus('');
      sleepTick();
      emitState();
      return { ...r.sleep };
    },

    unlock,
    saveProgress,
    on,

    /** @returns {object} a frozen snapshot (the engine state fields plus title, chapterTitle, voiceLabel, remainingMs…) */
    get state() { return snapshot(); },
    /** @returns {object|null} the loaded document (read-only: do not mutate) */
    get doc() { return r.doc; },
    /** The chip name of a voice choice ("Heart", "Bella + Sky", a named voice's name). */
    voiceLabel: (choice) => labelFor(choice),
    /** True once both elements were unlocked by a gesture (the player stops listening for gestures then). */
    allUnlocked,

    __destroy() {
      r.destroyed = true;
      r.token += 1;
      clearTimeout(r.preparingTimer);
      clearTimeout(r.slowTimer);
      clearTimeout(r.speedTimer);
      if (r.stateTimer !== null) clearTimeout(r.stateTimer);
      if (r.stateFrame && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(r.stateFrame);
      stopTicker();
      for (const off of r.elementOffs.splice(0)) off();
      for (const off of r.globalOffs.splice(0)) {
        try { off(); } catch { /* the store was reset */ }
      }
      r.wakeToken += 1;
      if (r.wakeLock) releaseLock(r.wakeLock);
      r.wakeLock = null;
      r.listeners.clear();
    },
  };

  function trackTargetFor(direction) {
    if (!r.doc?.blocks.length) return null;
    const unit = prefs.get().trackUnit === 'chapter' ? 'chapter' : 'paragraph';
    const chapters = r.doc.chapters.length ? r.doc.chapters : [null];
    const starts = chapters.map((_, i) => chapterStart(r.doc, i));
    const t = currentTimeline();
    return lockScreen.trackTarget(unit, direction, {
      cursor: r.cursor,
      blockCount: blockCount(),
      elapsedInBlockMs: elapsedInBlockMs(),
      chapterIndex: r.chapter,
      chapterStarts: starts,
      elapsedInChapterMs: t ? t.elapsedMs / (r.speed || 1) : 0,
    });
  }

  return engine;
}

