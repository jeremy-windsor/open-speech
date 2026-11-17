/**
 * Connection health, background model refreshes, status jobs and the status chip's model.
 * @module core/status
 */
import { get } from './api.js';
import { app } from './store.js';
import { announce } from './a11y.js';
import { refresh as refreshModels, modelLabel, deviceLabel, classifyKind, providerOf, PROVIDER_DISPLAY, sttLabel } from './models.js';

const HEALTH_EVERY_MS = 30000;
const MODELS_EVERY_MS = 60000;
const BACKOFF_MS = [2000, 4000, 8000, 16000, 30000];
const JOB_ANNOUNCE_DONE_MS = 2000;

let running = false;
let healthTimer = null;
let modelsTimer = null;
let failures = 0;
let inflight = null;
let unbind = [];
let jobs = new Map();   // key -> {key, label, startedAt, count}

function setConnection(patch) {
  app.set({ connection: { ...app.get().connection, ...patch } });
}

function hidden() {
  return typeof document !== 'undefined' && !!document.hidden;
}

function nextHealthDelay() {
  return failures ? BACKOFF_MS[Math.min(failures - 1, BACKOFF_MS.length - 1)] : HEALTH_EVERY_MS;
}

function scheduleHealth() {
  clearTimeout(healthTimer);
  healthTimer = null;
  if (!running || hidden()) return;
  healthTimer = setTimeout(() => {
    healthTimer = null;
    refreshHealth().catch(() => {});   // re-arms the timer when it settles
  }, nextHealthDelay());
}

function scheduleModels() {
  clearTimeout(modelsTimer);
  modelsTimer = null;
  if (!running || hidden()) return;
  modelsTimer = setTimeout(async () => {
    modelsTimer = null;
    await refreshModels({ silent: true }).catch(() => {});
    scheduleModels();
  }, MODELS_EVERY_MS);
}

function checkNow() {
  if (!running) return;
  refreshHealth().catch(() => {});
}

function onVisibility() {
  if (hidden()) {
    clearTimeout(healthTimer);
    clearTimeout(modelsTimer);
    healthTimer = null;
    modelsTimer = null;
    return;
  }
  checkNow();
  const { fetchedAt } = app.get().models;
  if (Date.now() - fetchedAt >= MODELS_EVERY_MS) refreshModels({ silent: true }).catch(() => {}).finally(scheduleModels);
  else scheduleModels();
}

/**
 * Start (or restart) background status work; never awaited by boot. GET /health every 30 s while the page
 * is visible (backoff 2, 4, 8, 16, 30 s while offline; immediate re-check on 'online' and on becoming
 * visible) and models.refresh() now and every 60 s while visible. Polling pauses while document.hidden.
 */
export function start() {
  stop();
  running = true;
  failures = 0;
  const onOnline = () => checkNow();
  window.addEventListener('online', onOnline);
  document.addEventListener('visibilitychange', onVisibility);
  unbind = [
    () => window.removeEventListener('online', onOnline),
    () => document.removeEventListener('visibilitychange', onVisibility),
  ];
  refreshHealth().catch(() => {});
  refreshModels({ silent: true }).catch(() => {}).finally(scheduleModels);
}

/** Stop polling and remove the listeners. */
export function stop() {
  running = false;
  clearTimeout(healthTimer);
  clearTimeout(modelsTimer);
  healthTimer = null;
  modelsTimer = null;
  for (const off of unbind) off();
  unbind = [];
}

/**
 * One immediate GET /health (deduplicated while in flight). Updates app.connection: version, revision,
 * modelsLoaded, checkedAt and the state ('offline' on failure, with the retry delay). While polling runs, the
 * next poll is re-armed from this result however the check was made (the timer, a request that failed with a
 * network error, the banner's Retry now, 'online'), so the banner's "Retrying in N s" is the real schedule.
 * @returns {Promise<object|null>} the health payload, or null when unreachable
 */
export function refreshHealth() {
  if (inflight) return inflight;
  inflight = (async () => {
    try {
      const data = await get('/health', { timeoutMs: 5000, accept: 'json' });
      failures = 0;
      const current = app.get().connection;
      const state = current.state === 'offline' || current.state === 'connecting' ? 'online' : current.state;
      setConnection({
        state,
        version: String(data?.version || ''),
        revision: String(data?.revision || ''),
        modelsLoaded: Number(data?.models_loaded || 0),
        checkedAt: Date.now(),
        error: state === 'online' ? '' : current.error,
        retryInMs: state === 'online' ? 0 : current.retryInMs,
        retryAt: state === 'online' ? 0 : current.retryAt,
      });
      return data;
    } catch (error) {
      if (error?.kind === 'locked' || error?.kind === 'auth') {
        // The server answered: a lockout covers even /health. api.request already set the state.
        failures = 0;
        setConnection({ checkedAt: Date.now() });
        return null;
      }
      failures += 1;
      const retryInMs = running ? nextHealthDelay() : 0;
      setConnection({
        state: 'offline',
        checkedAt: Date.now(),
        error: error?.message || "Can't reach the server.",
        retryInMs,
        retryAt: retryInMs ? Date.now() + retryInMs : 0,
      });
      return null;
    } finally {
      inflight = null;
    }
  })();
  const rearm = () => { if (running) scheduleHealth(); };
  inflight.then(rearm, rearm);
  return inflight;
}

function publishJobs() {
  app.set({ jobs: [...jobs.values()].map(({ key, label, detail, startedAt }) => ({ key, label, detail, startedAt })) });
}

function stripEllipsis(text) {
  return String(text || '').replace(/(?:…|\.\.\.)\s*$/, '').trim();
}

/**
 * Show a busy job in the status chip ("Loading Kokoro…" plus an aria-hidden elapsed counter). Jobs are
 * refcounted per key: a second caller with the same key shares the job (its label and detail win); the
 * job ends with its last caller. The first begin announces politely (the label, then the detail); the last
 * end announces "<label> done" only when the job ran for 2 s or more.
 * @param {string} key
 * @param {string} label short, for the chip ("Loading Kokoro…")
 * @param {{detail?: string}} [options] a second line for the chip's detail and the status sheet
 *   ("Unloads Piper")
 * @returns {() => void} end (idempotent)
 */
export function beginJob(key, label, { detail = '' } = {}) {
  const k = String(key);
  let job = jobs.get(k);
  if (job) {
    job.count += 1;
    job.label = String(label);
    job.detail = String(detail || '');
  } else {
    job = { key: k, label: String(label), detail: String(detail || ''), startedAt: Date.now(), count: 1 };
    jobs.set(k, job);
    announce(job.detail ? `${job.label} ${job.detail}.` : job.label);
  }
  publishJobs();
  let ended = false;
  return () => {
    if (ended) return;
    ended = true;
    const current = jobs.get(k);
    if (!current || current !== job) return;
    current.count -= 1;
    if (current.count > 0) return;
    jobs.delete(k);
    publishJobs();
    if (Date.now() - current.startedAt >= JOB_ANNOUNCE_DONE_MS) announce(`${stripEllipsis(current.label)} done`);
  };
}

/**
 * Mirror a microphone capture into app.recording (the chip's top row).
 * @param {{owner: 'transcribe'|'voices', startedAt?: number}|null} rec
 */
export function setRecording(rec) {
  app.set({ recording: rec ? { owner: rec.owner, startedAt: rec.startedAt || Date.now() } : null });
}

function chipName(model) {
  const provider = model.provider || providerOf(model.id);
  if (classifyKind(model) === 'stt') return sttLabel(model);
  return PROVIDER_DISPLAY[provider] || modelLabel(model.id);
}

function fit(text, max = 22) {
  return text.length <= max ? text : `${text.slice(0, max - 1)}…`;
}

function spokenModel(model) {
  const device = deviceLabel(model.device);
  return `${chipName(model)}${device ? ` on ${device}` : ''}`;
}

/**
 * PURE: the status chip for an app state (the first matching row wins). Labels and accessible
 * names never contain a ticking counter: elapsed time comes from `elapsedFrom` and is rendered aria-hidden.
 * @param {object} state app.get()
 * @returns {{tone: 'rec'|'error'|'warn'|'busy'|'ok'|'idle', label: string, elapsedFrom: number,
 *   detail: string, ariaLabel: string, row: string}}
 */
export function chipModel(state) {
  const s = state || {};
  const conn = s.connection || {};
  const jobsList = s.jobs || [];
  const list = (s.models && s.models.list) || [];
  const loaded = list.filter((m) => m && m.state === 'loaded');
  const loadedTts = loaded.filter((m) => classifyKind(m) === 'tts');
  const loadedStt = loaded.filter((m) => classifyKind(m) === 'stt');
  const loadedSentence = loaded.length ? `Loaded: ${[...loadedTts, ...loadedStt].map(spokenModel).join(', ')}.` : 'No model loaded.';
  // WCAG 2.5.3 (label in name): the name starts with the visible label unless the sentence already holds it,
  // so "Kokoro · GPU" is said first and a voice command naming what is on screen finds the chip.
  const words = (text) => String(text).toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ').trim();
  const make = (row, tone, label, ariaState, detail = '', elapsedFrom = 0) => {
    const sentence = `Status: ${ariaState}. Open status.`;
    const lead = words(sentence).includes(words(label)) ? '' : `${label}. `;
    return { row, tone, label, elapsedFrom, detail, ariaLabel: `${lead}${sentence}` };
  };

  if (s.recording) {
    return make('recording', 'rec', 'Recording', 'recording', '', s.recording.startedAt || 0);
  }
  if (conn.state === 'offline') return make('offline', 'error', 'Offline', 'offline', conn.error || "Can't reach the server.");
  if (conn.state === 'unauthorized') return make('unauthorized', 'warn', 'API key needed', 'API key needed', conn.error || '');
  if (conn.state === 'locked') return make('locked', 'error', 'Locked out', 'locked out', conn.error || '');
  if (jobsList.length) {
    const job = jobsList[0];
    const more = jobsList.length > 1 ? ` +${jobsList.length - 1}` : '';
    const detail = [job.detail, ...jobsList.slice(1).map((j) => j.label)].filter(Boolean).join(' · ');
    return make('job', 'busy', `${fit(job.label, 22 - more.length)}${more}`, stripEllipsis(job.label) + (more ? `, and ${jobsList.length - 1} more` : ''), detail, job.startedAt || 0);
  }
  // The voice model Speak (and "Same as Speak") uses comes first when it is loaded; "+1" counts the other ones.
  const wanted = s.speakVoice && s.speakVoice.model;
  const tts = [...loadedTts.filter((m) => m.id === wanted), ...loadedTts.filter((m) => m.id !== wanted)];
  const primary = tts[0] || loadedStt[0];
  if (primary) {
    const device = deviceLabel(primary.device);
    const more = tts.length > 1 ? ` +${tts.length - 1}` : '';
    const label = `${fit(chipName(primary), 22 - more.length - (device ? device.length + 3 : 0))}${more}${device ? ` · ${device}` : ''}`;
    const others = [...tts.slice(1), ...(tts.length ? loadedStt : loadedStt.slice(1))];
    const detail = others.length ? `Also loaded: ${others.map(spokenModel).join(', ')}` : '';
    return make('loaded', 'ok', label, `connected. ${loadedSentence.replace(/\.$/, '')}`, detail);
  }
  if (conn.state === 'online') return make('idle', 'idle', 'No model loaded', 'connected. No model loaded', '');
  return make('connecting', 'idle', 'Connecting…', 'connecting', '');
}

/** Test hook. */
export function __resetForTests() {
  stop();
  failures = 0;
  inflight = null;
  jobs = new Map();
}
