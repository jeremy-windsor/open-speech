/**
 * The 3-step clone wizard (#voices/clone): 1 Capture (record with a start guard, an optional
 * 3-2-1 countdown, a level meter and auto-stop at the server's cap, or upload; quality checks), 2 Verify
 * words (name, transcript, a suggested draft from STT, confirm or skip), 3 Test & save (an optional
 * /v1/audio/speech/clone test that always sends `voice`, then save with an overwrite confirm).
 *
 * The pure helpers (sanitizeVoiceName, saveGate, cloneTestForm, saveForm, draftMeta, tooLongMessage,
 * pickSttModel) are exported for the tests; createWizard() wires the partial's #vo-wizard.
 * @module views/voices/clone
 */
import { html, render } from '../../core/dom.js';
import { icon } from '../../core/icons.js';
import * as api from '../../core/api.js';
import * as mic from '../../core/mic.js';
import * as audio from '../../core/audio.js';
import * as models from '../../core/models.js';
import { app } from '../../core/store.js';
import { announce } from '../../core/a11y.js';
import { confirm } from '../../core/dialogs.js';
import { toast } from '../../core/toast.js';
import { clock, bytes } from '../../core/format.js';
import { concat, encodeWav, decodeToMono, rms, peak, toDb } from '../../core/pcm.js';
import { createLevelMeter } from '../../components/meter.js';
import { analyzeClip, summaryText } from './quality.js';
import { WIZARD_TEST_SENTENCE, showBanner, clearBanner, actionError, modelName, cloneModels } from './shared.js';

export { cloneModels };

/** The reference paragraph read aloud while recording (today's Voice Lab script). */
export const SCRIPT = 'Today I am recording a clear reference for Open Speech. My voice should sound natural, steady, and easy to understand. I will speak at my normal pace, with a short pause between sentences. This sample includes numbers like seven and forty-two, plus questions, statements, and a little expression.';

export const COUNTDOWN_FROM = 3;
export const UNVERIFIED_HINT = 'Unverified draft. Check every word, then confirm.';

/** A capture whose microphone went away under it (the same copy as Transcribe). */
export const MIC_ENDED = 'The microphone stopped. Check that it is still connected, then record again.';
export const MIC_BACKGROUND = 'Recording stopped when the app went to the background.';

/**
 * The name a recording is saved under (today's sanitizeVoiceName): trimmed, lower case, spaces and hyphens
 * become "_", anything else outside [a-z0-9_] is dropped, at most 64 characters.
 */
export function sanitizeVoiceName(name) {
  return String(name || '')
    .trim()
    .toLowerCase()
    .replace(/[ -]/g, '_')
    .replace(/[^a-z0-9_]/g, '')
    .slice(0, 64);
}

/**
 * The save gate (VL-08): a draft, a name that sanitizes to something, and either "Save without a
 * transcript" or a non-empty transcript that is confirmed.
 * @returns {{ok: boolean, reasons: string[]}}
 */
export function saveGate({ draft, name, transcript, confirmed, skipped }) {
  const reasons = [];
  if (!draft) reasons.push('Record or upload a clip.');
  if (!sanitizeVoiceName(name)) reasons.push('Enter a voice name with at least one letter or number.');
  if (!skipped && !String(transcript || '').trim()) reasons.push('Type the transcript, or choose Save without a transcript.');
  else if (!skipped && !confirmed) reasons.push('Confirm that the transcript matches the recording, or choose Save without a transcript.');
  return { ok: reasons.length === 0, reasons };
}

/**
 * The multipart body of the pre-save test, POST /v1/audio/speech/clone (REV-05): `voice` is always sent
 * (the server's default "Ryan" may be refused by a clone model), `speed` never is (the server's 1.0 passes
 * models without speed control), the transcript only when it isn't skipped. The sentence speaks of the new,
 * unsaved recording.
 */
export function cloneTestForm({ model, name, blob, transcript = '', skipped = false }) {
  const voice = sanitizeVoiceName(name) || 'reference';
  const form = new FormData();
  form.append('input', WIZARD_TEST_SENTENCE);
  form.append('model', model);
  form.append('voice', voice);
  form.append('reference_audio', blob, `${voice}.wav`);
  const words = String(transcript || '').trim();
  if (!skipped && words) form.append('transcript', words);
  form.append('response_format', 'wav');
  return form;
}

/** The multipart body of POST /api/voices/library (VL-09): name, audio as <name>.wav, transcript unless skipped. */
export function saveForm({ name, blob, transcript = '', skipped = false }) {
  const safe = sanitizeVoiceName(name);
  const form = new FormData();
  form.append('name', safe);
  form.append('audio', blob, `${safe}.wav`);
  const words = String(transcript || '').trim();
  if (!skipped && words) form.append('transcript', words);
  return form;
}

/** "8.4 s · 48 kHz · mono · 790 KB" */
export function draftMeta(draft) {
  if (!draft) return '';
  const rate = draft.sampleRate ? `${Math.round(draft.sampleRate / 100) / 10} kHz` : '';
  return [`${(Math.round(draft.durationS * 10) / 10).toFixed(1)} s`, rate, 'mono', bytes(draft.size)].filter(Boolean).join(' · ');
}

/** VL-05: the copy for a clip longer than the server accepts. */
export function tooLongMessage(durationS, maxSeconds) {
  return `Reference audio is too long (${Number(durationS).toFixed(2)}s). Max: ${maxSeconds}s`;
}

/**
 * The sample rate a WAV file declares in its fmt chunk, or 0 when the bytes are not a RIFF/WAVE header with
 * one (MP3, M4A, FLAC, OGG and WebM are decoded at the standard rate instead).
 * @param {ArrayBuffer} buffer the first bytes of the file (4 KB is plenty)
 */
export function wavSampleRate(buffer) {
  if (!buffer || buffer.byteLength < 28) return 0;
  const view = new DataView(buffer);
  const tag = (offset) => String.fromCharCode(view.getUint8(offset), view.getUint8(offset + 1), view.getUint8(offset + 2), view.getUint8(offset + 3));
  if (tag(0) !== 'RIFF' || tag(8) !== 'WAVE') return 0;
  let offset = 12;
  while (offset + 8 <= buffer.byteLength) {
    const id = tag(offset);
    const size = view.getUint32(offset + 4, true);
    if (id === 'fmt ') return offset + 16 <= buffer.byteLength ? view.getUint32(offset + 12, true) : 0;
    offset += 8 + size + (size % 2);
  }
  return 0;
}

/** Decoding at the file's own rate is used only for rates every browser's OfflineAudioContext accepts. */
export const OWN_RATE_RANGE = Object.freeze({ min: 8000, max: 96000 });

/**
 * An OfflineAudioContext constructor fixed at `rate`, for pcm.decodeToMono({AudioContextCtor}): decoding at
 * the file's own rate keeps a low-rate clip low, so the sample-rate check tells the truth (the default
 * decode resamples everything to 48 kHz). null when the rate is out of range or there is no offline context.
 * @param {number} rate
 */
export function offlineAt(rate) {
  const Offline = globalThis.OfflineAudioContext || globalThis.webkitOfflineAudioContext;
  if (!Offline || !(rate >= OWN_RATE_RANGE.min && rate <= OWN_RATE_RANGE.max)) return null;
  return class OfflineAtRate extends Offline {
    constructor(channels, length) {
      super(channels, length, rate);
    }
  };
}

/** The STT model that suggests a transcript: a loaded one, else the configured default, else the first. */
export function pickSttModel(state) {
  const list = models.sttModels(state?.list || []);
  const loaded = list.find((m) => m.state === 'loaded');
  if (loaded) return loaded.id;
  const configured = state?.defaults?.stt || '';
  if (configured) return configured;
  return list[0]?.id || '';
}

/** A synthetic, speech-like clip for the screenshot hook (no microphone in headless Edge). */
export function demoSamples(seconds = 6.4, rate = 24000) {
  const out = new Float32Array(Math.round(seconds * rate));
  let seed = 7;
  const noise = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647 - 0.5; };
  for (let i = 0; i < out.length; i += 1) {
    const t = i / rate;
    const syllable = Math.max(0, Math.sin(2 * Math.PI * 3.1 * t)) ** 1.5 * (t % 2.1 < 1.8 ? 1 : 0.05);
    const f0 = 140 + 18 * Math.sin(2 * Math.PI * 0.4 * t);
    let v = 0;
    for (let h = 1; h <= 6; h += 1) v += Math.sin(2 * Math.PI * f0 * h * t) / (h * 1.4);
    out[i] = 0.16 * syllable * v + 0.0012 * noise();
  }
  return out;
}

/** The wizard's clone test plays with this home route (the Recordings cards use voices/recordings). */
export const WIZARD_HOME = 'voices/clone';

/**
 * The state of the wizard's own clone test in app.player: 'none', 'playing' or 'paused' (a store selector,
 * so the Test button re-renders only on these transitions).
 */
export function wizardTestState(player) {
  if (!player || player.owner !== 'voices' || player.id !== 'clone-test' || player.homeRoute !== WIZARD_HOME) return 'none';
  return player.playing ? 'playing' : 'paused';
}

/**
 * Wire the wizard inside `root` (#vo-wizard).
 * @param {HTMLElement} root
 * @param {{recordingConfig: () => Promise<{maxSeconds: number}>, recordings: () => object[],
 *   changed: (what: string) => void,
 *   afterSave: (name: string) => Promise<void>, openModels: () => void}} ctx
 */
export function createWizard(root, ctx) {
  const $ = (id) => root.querySelector(`#${id}`);
  const el = {
    steps: root.querySelector('.vo-steps'),
    title: $('vo-wizard-title'),
    record: $('vo-record'),
    timer: $('vo-timer'),
    timerText: $('vo-timer-text'),
    timerMax: $('vo-timer-max'),
    recDot: $('vo-rec-dot'),
    countdownNum: $('vo-countdown-num'),
    meter: $('vo-meter'),
    captureHint: $('vo-capture-hint'),
    mic: $('vo-mic'),
    countdown: $('vo-countdown'),
    micHelp: $('vo-mic-help'),
    captureError: $('vo-capture-error'),
    file: $('vo-file'),
    dropzone: $('vo-dropzone'),
    draft: $('vo-draft'),
    draftAudio: $('vo-draft-audio'),
    draftNote: $('vo-draft-note'),
    draftMeta: $('vo-draft-meta'),
    quality: $('vo-quality'),
    name: $('vo-name'),
    namePreview: $('vo-name-preview'),
    nameError: $('vo-name-error'),
    transcript: $('vo-transcript'),
    transcriptStatus: $('vo-transcript-status'),
    suggest: $('vo-suggest'),
    suggestError: $('vo-suggest-error'),
    confirm: $('vo-confirm'),
    skip: $('vo-skip'),
    confirmError: $('vo-confirm-error'),
    summary: $('vo-summary'),
    testModel: $('vo-test-model'),
    testModelHint: $('vo-test-model-hint'),
    test: $('vo-test'),
    testNote: $('vo-test-note'),
    testError: $('vo-test-error'),
    save: $('vo-save'),
    saveHint: $('vo-save-hint'),
    saveError: $('vo-save-error'),
    status: $('vo-status'),
    next1: $('vo-next-1'),
  };
  const meter = el.meter ? createLevelMeter(el.meter) : null;

  const st = {
    step: 1,
    phase: 'idle',          // 'idle' | 'starting' | 'countdown' | 'recording' | 'decoding'
    captureSeq: 0,
    session: null,
    offFrame: null,
    offWatch: [],           // the capture's track-ended and visibility listeners (watchCapture)
    chunks: [],
    collected: 0,
    sampleRate: 0,
    countdownLeft: 0,
    countdownTimer: null,
    tickTimer: null,
    maxSeconds: null,
    configLoad: null,       // the library-config request in flight
    configSeq: 0,
    draft: null,
    unverified: false,
    suggesting: false,
    testing: false,
    saving: false,
    testModel: '',
    devicesLoaded: false,
    shown: false,
  };
  const offs = [];

  /* ───────────── status and helpers ───────────── */

  function status(text, { announceIt = false } = {}) {
    if (el.status) el.status.textContent = text || '';
    if (announceIt && text) announce(text);
  }

  function setButton(button, { label, busy = false, disabled = false, iconName = '' } = {}) {
    if (!button) return;
    if (label !== undefined) {
      // A busy button shows the component's spinner instead of its icon.
      render(button, html`${iconName && !busy ? icon(iconName) : ''}<span class="btn__label">${label}</span>`);
    }
    if (busy) button.setAttribute('aria-busy', 'true');
    else button.removeAttribute('aria-busy');
    if (disabled || busy) button.setAttribute('aria-disabled', 'true');
    else button.removeAttribute('aria-disabled');
  }

  /**
   * Ask the server for the recording cap again (GET /api/voices/library-config; every time the wizard opens,
   * as the old Voice Lab did): the 60 s fallback after a failed request (offline, or no API key yet) must not
   * stick for the whole session. The newest answer wins.
   */
  function loadMaxSeconds() {
    const seq = (st.configSeq += 1);
    // Promise.resolve().then(): st.configLoad is set before any of this runs, even for a synchronous answer.
    const load = Promise.resolve()
      .then(() => ctx.recordingConfig())
      .then((config) => Number(config?.maxSeconds) || 0, () => 60)
      .then((value) => {
        if (seq !== st.configSeq) return;
        st.maxSeconds = value;
        st.configLoad = null;
        renderCapture();
      });
    st.configLoad = load;
    return load;
  }

  /** The recording cap in seconds (0: no limit), waiting for a request in flight. */
  async function maxSeconds() {
    if (st.maxSeconds === null && !st.configLoad) loadMaxSeconds();
    while (st.configLoad) {
      const load = st.configLoad;
      await load.catch(() => {});
      if (st.configLoad === load) st.configLoad = null;   // never wait on the same request twice
    }
    return st.maxSeconds ?? 60;
  }

  function maxLabel() {
    if (st.maxSeconds === null) return '';
    return st.maxSeconds > 0 ? `${st.maxSeconds}-second maximum.` : 'No duration limit.';
  }

  /* ───────────── devices ───────────── */

  async function refreshDevices() {
    if (!el.mic) return;
    const devices = await mic.listDevices();
    const saved = mic.savedDevice();
    const current = el.mic.value || saved;
    const rows = devices.length ? devices : [{ id: '', label: mic.DEFAULT_DEVICE_LABEL }];
    render(el.mic, html`${rows.map((d) => html`<option value="${d.id}">${d.label}</option>`)}`);
    el.mic.value = rows.some((d) => d.id === current) ? current : '';
    el.mic.disabled = st.phase !== 'idle' || !mic.support().ok;
    st.devicesLoaded = true;
  }

  /* ───────────── capture ───────────── */

  function renderCapture() {
    const support = mic.support();
    const phase = st.phase;
    if (el.micHelp) {
      if (!support.ok) {
        const text = support.reason === 'insecure'
          ? 'Recording needs a secure connection (HTTPS). You can still upload a clip.'
          : "This browser can't record audio here. Upload a clip instead.";
        render(el.micHelp, html`${icon('info', { className: 'ico banner__icon' })}<div class="banner__body"><p class="banner__title">Microphone unavailable</p><p class="banner__msg">${text}</p></div>`);
        el.micHelp.hidden = false;
      } else {
        el.micHelp.hidden = true;
      }
    }
    const labels = { idle: st.draft ? 'Record again' : 'Record', starting: 'Starting…', countdown: 'Cancel', recording: 'Stop', decoding: 'Record' };
    const icons = { idle: 'record', starting: '', countdown: 'x', recording: 'stop', decoding: 'record' };
    setButton(el.record, {
      label: labels[phase],
      iconName: icons[phase],
      busy: phase === 'starting',
      disabled: !support.ok || phase === 'decoding',
    });
    if (el.record) {
      el.record.classList.toggle('is-recording', phase === 'recording' || phase === 'countdown');
      // One primary action: Record until there is a clip, then Next (Record again steps back).
      const hasDraft = !!st.draft && phase === 'idle';
      el.record.classList.toggle('btn--primary', !hasDraft);
      el.record.classList.toggle('btn--secondary', hasDraft);
      if (el.next1) {
        el.next1.classList.toggle('btn--primary', hasDraft);
        el.next1.classList.toggle('btn--secondary', !hasDraft);
      }
    }
    const live = phase === 'recording';
    if (el.recDot) el.recDot.hidden = !live;
    if (el.countdownNum) {
      el.countdownNum.hidden = phase !== 'countdown';
      el.countdownNum.textContent = phase === 'countdown' ? String(st.countdownLeft) : '';
    }
    if (el.timer) el.timer.hidden = phase === 'countdown';
    updateTimer();
    if (el.timerMax) el.timerMax.textContent = st.maxSeconds > 0 ? ` / ${clock(st.maxSeconds)}` : '';
    if (el.captureHint) {
      const max = maxLabel();
      el.captureHint.textContent = phase === 'countdown'
        ? 'Get ready. Recording starts when the count reaches zero.'
        : `${max ? `${max} ` : ''}Speak at your normal pace and aim for a Good level.`;
    }
    if (el.mic) el.mic.disabled = phase !== 'idle' || !support.ok;
    if (el.countdown) el.countdown.disabled = phase !== 'idle';
    if (el.draft) el.draft.hidden = !st.draft || phase === 'recording' || phase === 'countdown' || phase === 'starting';
    if (el.dropzone) {
      el.dropzone.classList.toggle('is-busy', phase === 'decoding');
      el.dropzone.setAttribute('aria-busy', phase === 'decoding' ? 'true' : 'false');
    }
    renderSteps();
  }

  function updateTimer() {
    if (!el.timerText) return;
    // While recording: the time so far; otherwise the draft's length (0:00 before there is one).
    const seconds = st.phase === 'recording'
      ? (st.sampleRate ? st.collected / st.sampleRate : 0)
      : (st.phase === 'idle' && st.draft ? st.draft.durationS : 0);
    el.timerText.textContent = clock(seconds);
  }

  function clearTimers() {
    clearInterval(st.countdownTimer);
    clearInterval(st.tickTimer);
    st.countdownTimer = null;
    st.tickTimer = null;
  }

  async function startCapture() {
    if (st.phase !== 'idle') return;   // start guard (BUG-08): one capture at a time, a second press is ignored
    clearBanner(el.captureError);
    const seq = (st.captureSeq += 1);
    st.phase = 'starting';
    renderCapture();
    status('Starting the microphone…');
    let session;
    try {
      // mic.open() pauses every player first and refuses while Transcribe holds the microphone (VL-02).
      const prefs = app.get().prefs || {};
      session = await mic.open({
        owner: 'voices',
        // Until the device list has loaded, the select only holds "Default microphone": undefined lets
        // mic.open() use the saved device instead of forcing the default one.
        deviceId: el.mic && st.devicesLoaded ? el.mic.value : undefined,
        echoCancellation: prefs.echoCancellation !== false,
        noiseSuppression: prefs.noiseSuppression !== false,
      });
    } catch (error) {
      if (seq !== st.captureSeq) return;
      st.phase = 'idle';
      renderCapture();
      const message = error?.message || "The microphone couldn't start.";
      showBanner(el.captureError, { title: 'Microphone problem', message });
      status('');   // the alert banner says it; the status line would repeat it
      return;
    }
    if (seq !== st.captureSeq) {   // cancelled while starting (an upload, or the view was left)
      session.stop();
      return;
    }
    await maxSeconds();
    if (seq !== st.captureSeq) {
      session.stop();
      return;
    }
    st.session = session;
    st.sampleRate = session.sampleRate;
    st.chunks = [];
    st.collected = 0;
    st.offFrame = session.onFrame((frame) => onFrame(seq, frame));
    watchCapture(seq, session);
    if (seq !== st.captureSeq) return;   // the microphone was already gone: watchCapture said so
    refreshDevices().catch(() => {});   // labels show after a grant
    if (el.countdown && el.countdown.checked) {
      st.phase = 'countdown';
      st.countdownLeft = COUNTDOWN_FROM;
      renderCapture();
      status(`Recording starts in ${COUNTDOWN_FROM} seconds.`);
      st.countdownTimer = setInterval(() => {
        if (seq !== st.captureSeq) return;
        st.countdownLeft -= 1;
        if (st.countdownLeft <= 0) {
          clearInterval(st.countdownTimer);
          st.countdownTimer = null;
          beginRecording(seq);
        } else {
          renderCapture();
        }
      }, 1000);
    } else {
      beginRecording(seq);
    }
  }

  /**
   * The capture can end under us (VL-01): a track that ends (the microphone was unplugged, a phone call or
   * another app took it) or an iPhone Home Screen app that lost it in the background. Frames just stop, so
   * without this the wizard would stay on Stop with a frozen timer and app.recording set, and every player in
   * the app would refuse to play. Stop at once, keep what was recorded, and say why (the same
   * copy and checks as Transcribe's recorder). stopCapture() removes the listeners.
   */
  function watchCapture(seq, session) {
    const doc = typeof document !== 'undefined' ? document : null;
    let wasHidden = !!(doc && doc.hidden);
    const tracks = session && session.stream && typeof session.stream.getTracks === 'function' ? session.stream.getTracks() : [];
    const dead = () => session.active === false || tracks.some((t) => t.readyState === 'ended');
    const lost = () => {
      if (seq !== st.captureSeq || st.session !== session) return;
      const message = wasHidden || (doc && doc.hidden) ? MIC_BACKGROUND : MIC_ENDED;
      const before = st.draft;
      stopCapture({ keep: true, reason: 'lost' });
      if (st.draft === before) status('');   // nothing was kept: the banner says it all
      showBanner(el.captureError, { title: 'Recording stopped', message });
    };
    for (const track of tracks) {
      if (typeof track.addEventListener !== 'function') continue;
      track.addEventListener('ended', lost);
      st.offWatch.push(() => track.removeEventListener('ended', lost));
    }
    if (doc && typeof doc.addEventListener === 'function') {
      const onVisibility = () => {
        if (doc.hidden) {
          wasHidden = true;
          return;
        }
        if (dead()) lost();
        else wasHidden = false;
      };
      doc.addEventListener('visibilitychange', onVisibility);
      st.offWatch.push(() => doc.removeEventListener('visibilitychange', onVisibility));
    }
    if (dead()) lost();   // it ended while the cap was still loading
  }

  function beginRecording(seq) {
    if (seq !== st.captureSeq || !st.session) return;
    st.phase = 'recording';
    st.chunks = [];
    st.collected = 0;
    renderCapture();
    status(`Recording… ${maxLabel() || 'No duration limit.'}`);
    st.tickTimer = setInterval(updateTimer, 250);
  }

  function onFrame(seq, frame) {
    if (seq !== st.captureSeq || !frame) return;
    if (meter) {
      const r = rms(frame);
      meter.set({ db: toDb(r), peak: peak(frame) });
    }
    if (st.phase !== 'recording') return;
    st.chunks.push(new Float32Array(frame));
    st.collected += frame.length;
    const cap = st.maxSeconds > 0 ? Math.floor(st.maxSeconds * st.sampleRate) : 0;
    if (cap && st.collected >= cap) {
      stopCapture({ keep: true, reason: 'cap' });
    }
  }

  /**
   * Stop the capture. keep: a recording becomes the draft, trimmed to the server's cap before validation
   * (VL-03, VL-05); otherwise it is discarded.
   */
  function stopCapture({ keep = true, reason = '' } = {}) {
    const wasRecording = st.phase === 'recording';
    const wasActive = st.phase === 'starting' || st.phase === 'countdown' || wasRecording;
    st.captureSeq += 1;
    clearTimers();
    if (st.offFrame) st.offFrame();
    st.offFrame = null;
    for (const off of st.offWatch.splice(0)) off();
    const session = st.session;
    st.session = null;
    if (session) session.stop();
    if (meter) meter.reset();
    const rate = st.sampleRate;
    const chunks = st.chunks;
    st.chunks = [];
    st.phase = 'idle';
    if (keep && wasRecording) {
      const cap = st.maxSeconds > 0 ? Math.floor(st.maxSeconds * rate) : 0;
      const samples = concat(chunks, { maxSamples: cap });
      st.collected = 0;
      if (samples.length && rate) {
        try {
          setDraft({ samples, sampleRate: rate, source: 'record' });
          const secs = (samples.length / rate).toFixed(1);
          const said = {
            cap: `Recording stopped at the ${st.maxSeconds}-second maximum. ${secs} s recorded. Check the quality below.`,
            lost: `${secs} s recorded before the microphone stopped. Check the quality below.`,
          };
          status(said[reason] || `Recording stopped. ${secs} s recorded. Check the quality below.`, { announceIt: false });
        } catch (error) {
          showBanner(el.captureError, { title: "Couldn't use the recording", message: error.message });
          status('');
        }
      } else {
        status('Nothing was recorded. Check the microphone and try again.');
      }
    } else {
      st.collected = 0;
      // 'lost': the microphone went away before recording began; watchCapture's banner says so.
      if (wasActive) status(reason === 'lost' ? '' : (reason || 'Recording canceled.'));
    }
    renderCapture();
  }

  /* ───────────── draft ───────────── */

  function setDraft({ samples, sampleRate, source }) {
    const durationS = samples.length / sampleRate;
    if (st.maxSeconds > 0 && durationS > st.maxSeconds) throw new Error(tooLongMessage(durationS, st.maxSeconds));
    const blob = encodeWav(samples, sampleRate);
    const quality = analyzeClip(samples, sampleRate);
    if (st.draft && st.draft.url) URL.revokeObjectURL(st.draft.url);
    const url = URL.createObjectURL(blob);
    st.draft = { blob, url, sampleRate, durationS, source, size: blob.size, quality };
    // A new draft resets the transcript verification (VL-05, VL-08): an upload clears the words, a recording
    // keeps the script that was read aloud.
    if (el.confirm) el.confirm.checked = false;
    if (el.skip) el.skip.checked = false;
    st.unverified = false;
    if (el.transcript) {
      el.transcript.value = source === 'upload' ? '' : SCRIPT;
      el.transcript.classList.remove('is-unverified');
    }
    setTranscriptStatus(source === 'upload'
      ? 'Type the exact words spoken in the clip, or ask for a suggested draft.'
      : 'This is the script you read. Fix any words you said differently, then confirm.');
    clearBanner(el.suggestError);
    clearBanner(el.testError);
    clearBanner(el.saveError);
    if (el.draftAudio) {
      el.draftAudio.pause && el.draftAudio.pause();
      el.draftAudio.src = url;
    }
    renderDraft();
    renderVerify();
    renderSave();
  }

  /** "Stop recording to listen." under the draft player while the microphone records. */
  function renderDraftNote() {
    const muted = !!app.get().recording;
    if (el.draftNote) el.draftNote.hidden = !muted;
    if (el.draftAudio) {
      if (muted) el.draftAudio.setAttribute('aria-describedby', 'vo-draft-note');
      else el.draftAudio.removeAttribute('aria-describedby');
    }
  }

  function renderDraft() {
    const d = st.draft;
    renderDraftNote();
    if (!d) {
      if (el.draft) el.draft.hidden = true;
      return;
    }
    if (el.draftMeta) el.draftMeta.textContent = draftMeta(d);
    if (el.quality) {
      const q = d.quality;
      render(el.quality, html`<p class="vo-quality__summary">${summaryText(q)}</p><ul class="vo-quality__list" aria-label="Quality checks">${q.checks.map((c) => html`<li class="vo-quality__item" data-status="${c.status}"><span class="pill ${c.status === 'ok' ? 'pill--ok' : 'pill--warn'}"><span class="pill__dot" aria-hidden="true"></span><span class="pill__label">${c.status === 'ok' ? 'Good' : 'Check'}</span></span><span class="vo-quality__name">${c.label}</span><span class="vo-quality__value tabular">${c.value}</span><span class="vo-quality__text">${c.text}</span></li>`)}</ul>`);
    }
    renderCapture();
  }

  async function handleFile(file) {
    if (!file) return;
    if (st.phase === 'starting' || st.phase === 'countdown' || st.phase === 'recording') stopCapture({ keep: false, reason: 'Recording stopped for the upload.' });
    if (st.phase === 'decoding') return;
    clearBanner(el.captureError);
    const seq = (st.captureSeq += 1);
    st.phase = 'decoding';
    renderCapture();
    status('Preparing audio…');
    try {
      await maxSeconds();
      // A WAV decodes at the rate it declares (honest quality checks, no needless upsampling); any other
      // format at the standard rate.
      let Ctor = null;
      try {
        Ctor = file && file.size && typeof file.slice === 'function' ? offlineAt(wavSampleRate(await file.slice(0, 4096).arrayBuffer())) : null;
      } catch {
        Ctor = null;
      }
      let decoded;
      try {
        decoded = await decodeToMono(file, Ctor ? { AudioContextCtor: Ctor } : {});
      } catch (error) {
        if (!Ctor) throw error;
        decoded = await decodeToMono(file);   // that rate was refused by this browser: the standard one
      }
      if (seq !== st.captureSeq) return;
      setDraft({ samples: decoded.samples, sampleRate: decoded.sampleRate, source: 'upload' });
      status(`Uploaded ${(decoded.samples.length / decoded.sampleRate).toFixed(1)} s. Check the quality below, then type its exact words.`);
    } catch (error) {
      if (seq !== st.captureSeq) return;
      const message = error?.message || "That file couldn't be used.";
      showBanner(el.captureError, { title: "Couldn't use that file", message });
      status('');   // the alert banner says it
    } finally {
      if (seq === st.captureSeq) {
        st.phase = 'idle';
        renderCapture();
      }
      if (el.file) el.file.value = '';
    }
  }

  /* ───────────── steps ───────────── */

  function gate() {
    return saveGate({
      draft: st.draft,
      name: el.name ? el.name.value : '',
      transcript: el.transcript ? el.transcript.value : '',
      confirmed: !!(el.confirm && el.confirm.checked),
      skipped: !!(el.skip && el.skip.checked),
    });
  }

  function reachable(step) {
    if (step <= 1) return true;
    if (!st.draft) return false;
    if (step === 2) return true;
    return gate().ok;
  }

  function renderSteps() {
    if (!el.steps) return;
    for (const button of el.steps.querySelectorAll('[data-step]')) {
      const n = Number(button.getAttribute('data-step'));
      if (n === st.step) button.setAttribute('aria-current', 'step');
      else button.removeAttribute('aria-current');
      const done = n < st.step || (n === 1 && !!st.draft && st.step !== 1);
      const li = button.closest('li');
      if (li) li.classList.toggle('is-done', done);
      // Not colour alone: a done step shows a check for its number, and its name says "done".
      const said = button.querySelector('.vo-steps__done');
      if (said) said.textContent = done ? ', done' : '';
      if (reachable(n) && st.phase === 'idle') button.removeAttribute('aria-disabled');
      else button.setAttribute('aria-disabled', 'true');
    }
    for (const n of [1, 2, 3]) {
      const section = root.querySelector(`#vo-step-${n}`);
      if (section) section.hidden = n !== st.step;
    }
    if (el.next1) setButton(el.next1, { disabled: !st.draft || st.phase !== 'idle' });
  }

  function goStep(n, { focus = true } = {}) {
    if (!reachable(n)) return false;
    if (st.phase !== 'idle') return false;
    st.step = n;
    renderSteps();
    if (n === 2) renderVerify();
    if (n === 3) renderSave();
    status('');   // focus moves to the step heading, which says where the listener is
    if (focus) {
      const heading = root.querySelector(`#vo-step-${n}-title`);
      if (heading) heading.focus({ preventScroll: false });
    }
    return true;
  }

  /* ───────────── step 2: verify words ───────────── */

  function setTranscriptStatus(text) {
    if (el.transcriptStatus) el.transcriptStatus.textContent = text;
  }

  function renderVerify() {
    const raw = el.name ? el.name.value : '';
    const safe = sanitizeVoiceName(raw);
    const exists = safe && (ctx.recordings() || []).some((r) => r && r.name === safe);
    if (el.namePreview) el.namePreview.textContent = `Saves as: ${safe || '—'}${exists ? ' · replaces the recording with this name' : ''}`;
    const skipped = !!(el.skip && el.skip.checked);
    if (el.transcript) el.transcript.disabled = skipped;
    if (el.confirm) el.confirm.disabled = skipped;
    if (el.suggest) setButton(el.suggest, { busy: st.suggesting, disabled: skipped || !st.draft || st.suggesting });
    renderSteps();
    renderSave();
  }

  function setFieldError(input, errorEl, text) {
    if (!errorEl || !input) return;
    errorEl.hidden = !text;
    render(errorEl, text ? html`${icon('alert')}<span>${text}</span>` : html``);
    if (text) input.setAttribute('aria-invalid', 'true');
    else input.removeAttribute('aria-invalid');
  }

  function nextFromVerify() {
    const name = el.name ? el.name.value : '';
    let first = null;
    if (!sanitizeVoiceName(name)) {
      setFieldError(el.name, el.nameError, 'Enter a name with at least one letter or number.');
      first = first || el.name;
    }
    const g = gate();
    const skipped = !!(el.skip && el.skip.checked);
    if (!skipped && !(el.confirm && el.confirm.checked && String(el.transcript.value || '').trim())) {
      setFieldError(el.confirm, el.confirmError, String(el.transcript?.value || '').trim()
        ? 'Confirm that the transcript matches the recording, or choose Save without a transcript.'
        : 'Type the transcript, or choose Save without a transcript.');
      first = first || (String(el.transcript?.value || '').trim() ? el.confirm : el.transcript);
    }
    if (!g.ok || first) {
      if (first) first.focus();
      status('');   // the field errors say what is missing, focus is on the first
      return;
    }
    goStep(3);
  }

  async function suggest() {
    if (!st.draft || st.suggesting) return;
    clearBanner(el.suggestError);
    const model = pickSttModel(app.get().models);
    if (!model) {
      showBanner(el.suggestError, { title: "Couldn't suggest a transcript", message: 'No speech-to-text model is available. Open Models to download one.', actions: [{ label: 'Open Models', action: 'open-models-stt' }] });
      return;
    }
    // The words are this clip's: Record and upload stay usable while it runs, and an answer that comes
    // after another clip became the draft leaves that clip's words and the status line alone.
    const draft = st.draft;
    st.suggesting = true;
    renderVerify();
    status('Listening to the clip for a draft transcript…');
    try {
      const ok = await models.ensureReady(model, { kind: 'stt' });
      if (st.draft !== draft) return;
      if (!ok) {
        status('Suggestion canceled.');
        return;
      }
      const form = new FormData();
      form.append('file', draft.blob, 'voice-reference.wav');
      form.append('model', model);
      form.append('response_format', 'json');
      const result = await api.request('/v1/audio/transcriptions', { method: 'POST', body: form, accept: 'json' });
      models.markUsed(model);
      if (st.draft !== draft) return;
      if (el.transcript) {
        el.transcript.value = String(result?.text || '').trim();
        el.transcript.classList.add('is-unverified');
      }
      if (el.confirm) el.confirm.checked = false;
      st.unverified = true;
      setTranscriptStatus(UNVERIFIED_HINT);
      setFieldError(el.confirm, el.confirmError, '');
      status('A draft transcript is in. Check every word, then confirm.', { announceIt: false });
    } catch (error) {
      if (error && error.kind === 'aborted') return;
      models.markFailed(model);
      if (st.draft !== draft) return;   // the failure was the previous clip's; setDraft() cleared its banner
      const e = actionError(error, "Couldn't suggest a transcript");
      showBanner(el.suggestError, e);
      status('');
    } finally {
      st.suggesting = false;
      renderVerify();
    }
  }

  /* ───────────── step 3: test and save ───────────── */

  function renderTestModels() {
    if (!el.testModel) return;
    const list = cloneModels(app.get().models.list);
    // The listener's pick wins (a models refresh re-renders this select every minute); the remembered one
    // covers the select having been empty while no clone model was available.
    const keep = el.testModel.value || st.testModel;
    if (!list.length) {
      render(el.testModel, html`<option value="">No clone model available</option>`);
      el.testModel.value = '';
      el.testModel.disabled = true;
      if (el.testModelHint) render(el.testModelHint, html`No model that can clone a voice is running. You can still save the recording. <a href="#models/tts">Open Models</a>`);
    } else {
      render(el.testModel, html`${list.map((m) => html`<option value="${m.id}">${modelName(m.id)}</option>`)}`);
      el.testModel.value = list.some((m) => m.id === keep) ? keep : list[0].id;
      el.testModel.disabled = false;
      if (el.testModelHint) el.testModelHint.textContent = 'Speaks one short sentence in the cloned voice.';
    }
    if (el.testModel.value) st.testModel = el.testModel.value;
  }

  function renderSave() {
    const g = gate();
    if (el.save) setButton(el.save, { busy: st.saving, disabled: !g.ok || st.saving });
    if (el.saveHint) {
      el.saveHint.textContent = g.ok ? '' : g.reasons.join(' ');
      el.saveHint.hidden = g.ok;
    }
    if (el.test) {
      // Generating: busy. Its clip playing: Stop test. Paused (autoplay refused, or paused from the lock
      // screen): Play test. Otherwise: Test before saving. While the microphone records (another view),
      // nothing may play: unavailable, and #vo-test-note says why.
      const playback = wizardTestState(app.get().player);
      const live = playback !== 'none';
      const muted = !!app.get().recording && !st.testing && playback !== 'playing';
      const label = st.testing ? 'Testing…' : ({ playing: 'Stop test', paused: 'Play test', none: 'Test before saving' })[playback];
      setButton(el.test, {
        label,
        iconName: playback === 'playing' ? 'stop' : 'play',
        busy: st.testing,
        disabled: !st.draft || muted || (!live && !cloneModels(app.get().models.list).length),
      });
      if (muted) el.test.setAttribute('aria-describedby', 'vo-test-note');
      else el.test.removeAttribute('aria-describedby');
      if (el.testNote) el.testNote.hidden = !muted;
    }
    if (el.summary) {
      const d = st.draft;
      const skipped = !!(el.skip && el.skip.checked);
      const words = el.transcript ? String(el.transcript.value || '').trim() : '';
      const transcriptText = skipped ? 'None (saved without a transcript)' : (el.confirm && el.confirm.checked ? 'Confirmed' : (words ? 'Not confirmed yet' : 'Missing'));
      render(el.summary, html`<div class="vo-summary__row"><dt>Name</dt><dd class="mono">${sanitizeVoiceName(el.name ? el.name.value : '') || '—'}</dd></div><div class="vo-summary__row"><dt>Clip</dt><dd class="tabular">${d ? draftMeta(d) : 'None yet'}</dd></div><div class="vo-summary__row"><dt>Transcript</dt><dd>${transcriptText}</dd></div>`);
    }
  }

  /** Stop the wizard's test clip when it plays, play it (inside the click) when it is paused. */
  function toggleTestPlayback() {
    const s = audio.current();
    if (!s || s.id !== 'clone-test' || s.homeRoute !== WIZARD_HOME || !s.controls) return false;
    const playing = typeof s.getState === 'function' && !!s.getState().playing;
    if (playing) s.controls.stop();
    else s.controls.play();
    return true;
  }

  async function test() {
    if (st.testing) return;
    if (wizardTestState(app.get().player) !== 'none' && toggleTestPlayback()) return;
    if (!st.draft) return;
    if (app.get().recording) {   // checked before prime(), the model load and the synthesis
      renderSave();
      return;
    }
    clearBanner(el.testError);
    const modelId = el.testModel ? el.testModel.value : '';
    const list = cloneModels(app.get().models.list);
    if (!modelId) {
      showBanner(el.testError, { title: "Couldn't test the clone", message: 'No clone model is available. Start or download one in Models.', actions: [{ label: 'Open Models', action: 'open-models' }] });
      return;
    }
    const model = list.find((m) => m.id === modelId);
    if (!model) {
      showBanner(el.testError, { title: "Couldn't test the clone", message: `Clone model ${modelId} is unavailable` });
      return;
    }
    const skipped = !!(el.skip && el.skip.checked);
    const words = el.transcript ? String(el.transcript.value || '').trim() : '';
    if (model.capabilities?.clone_transcript_required && (skipped || !words)) {
      showBanner(el.testError, { title: "Couldn't test the clone", message: `${modelName(modelId)} requires an exact reference transcript` });
      return;
    }
    // Synchronously inside the click, once the test will run (a refused click leaves what plays alone): the
    // result plays after a network wait (iOS).
    audio.prime();
    const label = modelName(modelId);
    // The test is this clip's: one that finishes after another clip became the draft is dropped, never
    // played or announced as the new clip's test.
    const draft = st.draft;
    st.testing = true;
    renderSave();
    status(`Generating a clone test with ${label}…`);
    let url = '';
    try {
      const ok = await models.ensureReady(modelId, { kind: 'tts' });
      if (st.draft !== draft) return;
      if (!ok) {
        status('Clone test canceled.');
        return;
      }
      const blob = await api.request('/v1/audio/speech/clone', {
        method: 'POST',
        body: cloneTestForm({ model: modelId, name: el.name ? el.name.value : '', blob: draft.blob, transcript: words, skipped }),
        accept: 'blob',
      });
      models.markUsed(modelId);
      if (st.draft !== draft) return;
      url = URL.createObjectURL(blob);
      const playUrl = url;
      st.testing = false;   // generated: from here the button follows the clip (Stop test / Play test)
      renderSave();
      await audio.playClip({
        id: 'clone-test', owner: 'voices', title: 'Clone test', subtitle: `${sanitizeVoiceName(el.name?.value) || 'New voice'} · ${label}`,
        homeRoute: WIZARD_HOME, src: playUrl,
        onRelease: () => {
          URL.revokeObjectURL(playUrl);
          renderSave();
        },
      });
      url = '';
      if (st.draft !== draft) return;
      status(`Clone test complete with ${label}.`);
    } catch (error) {
      if (url) URL.revokeObjectURL(url);
      if (error && error.kind === 'aborted') return;
      if (error && error.name !== 'CaptureActiveError') models.markFailed(modelId);
      if (st.draft !== draft) return;   // the failure was the previous clip's; setDraft() cleared its banner
      const e = actionError(error, 'Clone test failed');
      showBanner(el.testError, e);
      status('');
    } finally {
      st.testing = false;
      renderSave();
    }
  }

  async function save() {
    const g = gate();
    if (!g.ok || st.saving) {
      status(g.reasons[0] || '');
      return;
    }
    clearBanner(el.saveError);
    const name = sanitizeVoiceName(el.name.value);
    const skipped = !!(el.skip && el.skip.checked);
    const transcript = el.transcript ? el.transcript.value : '';
    st.saving = true;
    renderSave();
    try {
      let exists = false;
      try {
        await api.get(`/api/voices/library/${encodeURIComponent(name)}`, { accept: 'json' });
        exists = true;
      } catch (error) {
        if (!(error && error.status === 404)) throw error;
      }
      if (exists) {
        const ok = await confirm({
          title: `Replace the existing recording “${name}”?`,
          message: 'Its audio and transcript will be overwritten.',
          confirmLabel: 'Replace',
          danger: true,
        });
        if (!ok) {
          status('Not saved. Choose another name to keep both.');
          return;
        }
      }
      status(`Saving ${name}…`);
      await api.request('/api/voices/library', { method: 'POST', body: saveForm({ name, blob: st.draft.blob, transcript, skipped }), accept: 'json' });
      toast(`Saved recording ${name}`, { kind: 'success' });
      status(`Saved ${name}.`);
      ctx.changed('recordings');
      reset();
      await ctx.afterSave(name);
    } catch (error) {
      if (error && error.kind === 'aborted') return;
      const e = actionError(error, "Couldn't save the recording");
      showBanner(el.saveError, e);
      status('');
    } finally {
      st.saving = false;
      renderSave();
    }
  }

  /** Back to an empty wizard (after a save): no draft, the script, step 1. */
  function reset() {
    if (st.phase !== 'idle') stopCapture({ keep: false });
    if (st.draft && st.draft.url) URL.revokeObjectURL(st.draft.url);
    st.draft = null;
    st.unverified = false;
    if (el.draftAudio) {
      el.draftAudio.pause && el.draftAudio.pause();
      el.draftAudio.removeAttribute('src');
    }
    if (el.name) el.name.value = '';
    if (el.transcript) {
      el.transcript.value = SCRIPT;
      el.transcript.classList.remove('is-unverified');
      el.transcript.disabled = false;
    }
    if (el.confirm) el.confirm.checked = false;
    if (el.skip) el.skip.checked = false;
    setFieldError(el.name, el.nameError, '');
    setFieldError(el.confirm, el.confirmError, '');
    for (const host of [el.captureError, el.suggestError, el.testError, el.saveError]) clearBanner(host);
    setTranscriptStatus('Read this paragraph while recording, or replace it with the exact words of an uploaded clip.');
    st.step = 1;
    renderDraft();
    renderCapture();
    renderVerify();
  }

  /* ───────────── events ───────────── */

  const onNameInput = () => {
    setFieldError(el.name, el.nameError, '');
    renderVerify();
  };
  const onTranscriptInput = () => {
    // Editing the words un-confirms them (VL-08).
    if (el.confirm && el.confirm.checked) el.confirm.checked = false;
    setFieldError(el.confirm, el.confirmError, '');
    renderVerify();
  };
  const onConfirmChange = () => {
    if (el.confirm.checked) {
      st.unverified = false;
      if (el.transcript) el.transcript.classList.remove('is-unverified');
      setTranscriptStatus('Transcript confirmed.');
    }
    setFieldError(el.confirm, el.confirmError, '');
    renderVerify();
  };
  const onSkipChange = () => {
    setFieldError(el.confirm, el.confirmError, '');
    renderVerify();
  };
  const onMicChange = () => mic.saveDevice(el.mic.value);
  const onFileChange = () => {
    const file = el.file.files && el.file.files[0];
    handleFile(file).catch(() => {});
  };
  const onDragOver = (event) => {
    event.preventDefault();
    el.dropzone.classList.add('is-dragover');
  };
  const onDragLeave = () => el.dropzone.classList.remove('is-dragover');
  const onDrop = (event) => {
    event.preventDefault();
    el.dropzone.classList.remove('is-dragover');
    const file = event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0];
    handleFile(file).catch(() => {});
  };
  const onDraftPlay = () => {
    // The draft player is a plain <audio controls>: it follows the app's one-player rule by hand. Another
    // player starting stops Live (a paused Live session would keep the server's Live slot), pauses the
    // Reader and keeps it resumable, and pauses and releases a clip.
    if (app.get().recording) {
      el.draftAudio.pause();
      return;
    }
    audio.otherPlayerStarted();
  };
  const onTestModelChange = () => {
    st.testModel = el.testModel.value;
    clearBanner(el.testError);   // a failure was about the model that was picked before
  };

  const listen = (node, type, fn) => {
    if (!node) return;
    node.addEventListener(type, fn);
    offs.push(() => node.removeEventListener(type, fn));
  };
  listen(el.name, 'input', onNameInput);
  listen(el.transcript, 'input', onTranscriptInput);
  listen(el.confirm, 'change', onConfirmChange);
  listen(el.skip, 'change', onSkipChange);
  listen(el.mic, 'change', onMicChange);
  listen(el.testModel, 'change', onTestModelChange);
  listen(el.file, 'change', onFileChange);
  listen(el.dropzone, 'dragover', onDragOver);
  listen(el.dropzone, 'dragleave', onDragLeave);
  listen(el.dropzone, 'drop', onDrop);
  listen(el.draftAudio, 'play', onDraftPlay);
  offs.push(app.select((s) => s.player && s.player.playing, (playing) => {
    if (playing && el.draftAudio && !el.draftAudio.paused) el.draftAudio.pause();
  }));
  // While the microphone records (another view), Test before saving and the draft player can't play: say so.
  offs.push(app.select((s) => !!s.recording, () => {
    renderSave();
    renderDraftNote();
  }));
  offs.push(app.select((s) => s.models.list, () => {
    renderTestModels();
    renderSave();
  }));
  // The Test button follows the test clip: Stop test while it plays, Play test while it is paused.
  offs.push(app.select((s) => wizardTestState(s.player), () => renderSave()));
  offs.push(mic.onDeviceChange(() => { if (st.shown) refreshDevices().catch(() => {}); }));

  /** Click actions inside the wizard (bound by the view with dom.actions). */
  const actions = {
    record: () => {
      if (st.phase === 'idle') return startCapture();
      if (st.phase === 'countdown') return stopCapture({ keep: false, reason: 'Recording canceled.' });
      if (st.phase === 'recording') return stopCapture({ keep: true });
      return undefined;   // starting or decoding: the press is ignored (start guard)
    },
    step: (button) => goStep(Number(button.getAttribute('data-step'))),
    'next-1': () => goStep(2),
    'next-2': () => nextFromVerify(),
    'back-1': () => goStep(1),
    'back-2': () => goStep(2),
    suggest: () => suggest(),
    test: () => test(),
    save: () => save(),
    'open-models': () => ctx.openModels('tts'),
    'open-models-stt': () => ctx.openModels('stt'),
  };

  // Initial render.
  if (el.transcript && !el.transcript.value) el.transcript.value = SCRIPT;
  renderTestModels();
  renderCapture();
  renderVerify();

  return {
    actions,
    /** The route showed the wizard. */
    show() {
      st.shown = true;
      renderCapture();
      loadMaxSeconds();   // every time: a cap that failed to load (offline, no key yet) is asked for again
      if (!st.devicesLoaded) refreshDevices().catch(() => {});
      renderTestModels();
      renderVerify();
      renderDraftNote();
    },
    /** Whether the wizard is on screen (a save that finishes after the listener left must not pull them back). */
    isShown: () => st.shown,
    /** The wizard was left (another sub or view): a capture stops and keeps what was recorded. */
    hide() {
      st.shown = false;
      if (st.phase === 'recording') stopCapture({ keep: true });
      else if (st.phase === 'starting' || st.phase === 'countdown') stopCapture({ keep: false, reason: 'Recording canceled.' });
      if (el.draftAudio && !el.draftAudio.paused) el.draftAudio.pause();
    },
    /** The step heading to focus when the wizard opens from a button. */
    heading() {
      return el.title;
    },
    /** Screenshot states: a synthetic clip stands in for the microphone. */
    applyShot(shot) {
      if (shot === 'clone-recording') {
        st.phase = 'recording';
        st.sampleRate = 24000;
        st.collected = 24000 * 7;
        st.maxSeconds = st.maxSeconds ?? 60;
        renderCapture();
        if (meter) meter.set({ db: -22, peak: 0.4 });
        status(`Recording… ${maxLabel()}`);
        return true;
      }
      if (shot === 'clone-draft' || shot === 'clone-verify' || shot === 'clone-save' || shot === 'clone-error') {
        if (st.maxSeconds === null) st.maxSeconds = 60;
        setDraft({ samples: demoSamples(), sampleRate: 24000, source: 'record' });
        status('Recording stopped. 6.4 s recorded. Check the quality below.');
        if (shot !== 'clone-draft') {
          if (el.name) el.name.value = 'Alex reference';
          renderVerify();
          goStep(2, { focus: false });
        }
        if (shot === 'clone-save' || shot === 'clone-error') {
          if (el.confirm) el.confirm.checked = true;
          onConfirmChange();
          goStep(3, { focus: false });
        }
        if (shot === 'clone-error') {
          // A failed test as the server reports it: the banner shows the reason, Details the code.
          const error = new api.ApiError("Voice 'Ryan' is not available for this model", {
            status: 400, code: 'voice_not_found', kind: 'http',
            detail: JSON.stringify({ error: { message: "Voice 'Ryan' is not available for this model", code: 'voice_not_found' } }),
          });
          const e = actionError(error, 'Clone test failed');
          showBanner(el.testError, e);
          status('');
        }
        return true;
      }
      return false;
    },
    /** Test hook: the internal state. */
    state: () => st,
    destroy() {
      if (st.phase !== 'idle') stopCapture({ keep: false });
      if (meter) meter.destroy();
      for (const off of offs.splice(0)) off();
      if (st.draft && st.draft.url) URL.revokeObjectURL(st.draft.url);
    },
  };
}
