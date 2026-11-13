/**
 * Transcribe: record from the microphone or upload a file, and get a timed transcript to
 * copy or export. The pure parts live beside it: recorder.js (the session state machine), upload.js,
 * transcript.js (the reducer), export.js, log.js (the role=log rendering), model-select.js and ui.js (copy).
 *
 * The view follows the view-host contract (shell/view-host.js): init() binds the partial once; show() is never awaited (it renders at
 * once and loads the device list in the background); hide() stops the view-only timers. A recording or an
 * upload keeps running while another view shows (the status chip says "Recording").
 * @module views/transcribe/index
 */
import { html, render, actions } from '../../core/dom.js';
import { app, on as onBus } from '../../core/store.js';
import * as mic from '../../core/mic.js';
import * as audio from '../../core/audio.js';
import * as prefs from '../../core/prefs.js';
import { setRecording } from '../../core/status.js';
import { announce } from '../../core/a11y.js';
import { toast } from '../../core/toast.js';
import { confirm } from '../../core/dialogs.js';
import { copyText } from '../../core/clipboard.js';
import { downloadBlob } from '../../core/download.js';
import { register } from '../../core/shortcuts.js';
import { navigate } from '../../core/router.js';
import { clock } from '../../core/format.js';
import { reducedMotion } from '../../core/theme.js';
import { encodeWav } from '../../core/pcm.js';
import { bindMenuButton } from '../../components/menu.js';
import { createLevelMeter } from '../../components/meter.js';
import { bindAskButton, openAskAi, openSample, SHOTS as ASK_SHOTS } from '../../components/ask-ai.js';
import { charCount } from '../../components/ask-ai-logic.js';
import { openConnectionSettings } from '../../shell/status-chip.js';
import { createRecorder } from './recorder.js';
import { createUploader } from './upload.js';
import * as tx from './transcript.js';
import { createLog } from './log.js';
import { bindModelSelect } from './model-select.js';
import { exportFile, FORMATS } from './export.js';
import {
  captureView, micHelp, vadLabel, metaLine, bannerHtml, bannerKey, errorActions, toastLine, CONFIRM_CANCEL_AFTER_S,
  nothingHeard,
} from './ui.js';
import { SHOTS, recordingShot, doneShot, uploadingShot, errorShot } from './shots.js';

/** The audio session id of the uploaded file (timestamps play it from there, REV-39). */
export const FILE_SESSION = 'transcribe:file';

const ACTIVE_PHASES = new Set(['requesting', 'starting', 'recording', 'stopping']);

/**
 * Build the controller for a mounted partial.
 * @param {HTMLElement} root the view's section, with partials/transcribe.html inside
 * @param {{recorder?: object, uploader?: object}} [deps] test seams; the real modules by default
 * @returns {{show: Function, hide: Function, destroy: Function, state: () => object}}
 */
export function createTranscribeView(root, deps = {}) {
  const $ = (id) => root.querySelector(`#${id}`);
  const el = {
    layout: $('tr-layout'), nomodel: $('tr-nomodel'), help: $('tr-mic-help'), record: $('tr-record'), phase: $('tr-phase'),
    dot: $('tr-rec-dot'), timer: $('tr-timer'), hint: $('tr-rec-hint'), meter: $('tr-meter'), vad: $('tr-vad'),
    vadLabel: $('tr-vad-label'), heard: $('tr-heard'), recActions: $('tr-rec-actions'), cancel: $('tr-cancel'), micField: $('tr-mic-field'),
    mic: $('tr-mic'), recError: $('tr-rec-error'), drop: $('tr-drop'), file: $('tr-file'), dropTitle: $('tr-drop-title'),
    dropHint: $('tr-drop-hint'), dropProgress: $('tr-drop-progress'), dropPhase: $('tr-drop-phase'),
    uploadStatus: $('tr-upload-status'), uploadElapsed: $('tr-upload-elapsed'),
    uploadCancel: $('tr-upload-cancel'), uploadError: $('tr-upload-error'), model: $('tr-model'), modelHint: $('tr-model-hint'),
    modelField: $('tr-model-field'),
    ask: $('tr-ask'), copy: $('tr-copy'), exportBtn: $('tr-export'), exportMenu: $('tr-export-menu'), clear: $('tr-clear'),
    showInterim: $('tr-show-interim'), showTs: $('tr-show-ts'), empty: $('tr-empty'), emptyTitle: $('tr-empty-title'),
    emptyText: $('tr-empty-text'), log: $('tr-log'), interim: $('tr-interim'), foot: $('tr-foot'), play: $('tr-play'),
    meta: $('tr-meta'), panelTitle: $('tr-transcript-title'),
  };

  const s = {
    transcript: tx.createState(),
    snapshot: null,         // the transcript before the current microphone session (Cancel restores it)
    rec: null,              // the recorder state shown (the shot state while a shot is on)
    recError: null,         // the error banner in the capture card
    helpError: null,        // a microphone failure that has help copy (denied, no device, background)
    helpDismissed: '',
    heardDismissed: '',     // the nothing-heard reason Dismiss hid for this session
    devices: null,
    permission: 'unknown',
    modelId: '',
    modelOptions: [],
    modelsLoaded: false,
    upload: null,           // the uploader state shown
    lastFile: null,
    fileUrl: '',
    shown: false,
    started: false,         // the first show() ran
    shot: '',
    shotRecording: false,
    recordFocus: false,     // Record had focus when it was disabled (it gets it back)
    uploadFocus: false,     // the file input had focus when its file went busy (Cancel holds it meanwhile)
    playing: false,
  };
  let ticker = null;
  let devToken = 0;
  const offs = [];
  /** What each banner shows (bannerKey), '' while hidden: a banner is written only when that changes. */
  const shownBanner = { rec: '', upload: '', help: '', heard: '' };
  let modelSelect = null;   // bindModelSelect, bound once the controls below are wired

  const meter = createLevelMeter(el.meter);
  meter.reset();
  const log = createLog({ log: el.log, interim: el.interim });

  // A notice (a server error while the stream goes on) arrives as the state's error: onRecorderState shows it.
  const recorder = deps.recorder || createRecorder({
    on: {
      state: (st) => onRecorderState(st),
      started: (st) => onStarted(st),
      level: (level) => { if (s.shown && !s.shot) meter.set(level); },
      transcript: (message) => dispatch({ type: 'transcript', message }),
      end: (detail) => onEnded(detail),
    },
  });
  const uploader = deps.uploader || createUploader({ onChange: (st) => { if (!s.shot) s.upload = st; renderAll(); } });
  s.rec = recorder.state;
  s.upload = uploader.state;

  /* ───────────── transcript ───────────── */

  function dispatch(action) {
    const before = s.transcript;
    s.transcript = tx.reduce(s.transcript, action);
    const t = s.transcript;
    if (action.type === 'file-result') {
      // Announced as one summary, never read out whole (REV-24). Away from the view a toast says it instead.
      const summary = t.segments.length ? tx.summary(t) : 'No speech found in this file.';
      log.replace(t.segments, { file: true, summary: s.shown ? summary : '' });
      renderTranscript();
      revealTranscript();
      return;
    }
    if (action.type === 'session-start' || action.type === 'restore' || action.type === 'clear') {
      log.replace(t.segments, { file: t.source === 'file' && !!s.fileUrl });
    } else if (t !== before) {
      const follow = nearTail();
      log.sync(t.segments);
      renderTranscript();
      if (follow) followTail();
      return;
    }
    renderTranscript();
  }

  // A finished file on a phone: its transcript is below the capture controls, so bring it into view (the
  // summary is announced; focus stays where it was).
  function revealTranscript() {
    const panel = $('tr-panel');
    if (!s.shown || !panel || typeof panel.scrollIntoView !== 'function') return;
    // Where the words start decides, not the panel's head: a head at the bottom edge still hides them all
    // behind the bottom bar. (No speech: the empty state says so instead.)
    const words = el.log.firstElementChild || el.empty;
    if (!words || typeof words.getBoundingClientRect !== 'function') return;
    const top = words.getBoundingClientRect().top;
    const h = globalThis.innerHeight || 0;
    if (h && top > h - 160) panel.scrollIntoView({ block: 'start', behavior: reducedMotion() ? 'auto' : 'smooth' });
  }

  // Keep the newest words in view while the listener is reading along at the bottom.
  function nearTail() {
    if (!recorder.isActive() || typeof el.interim.getBoundingClientRect !== 'function') return false;
    const r = el.interim.getBoundingClientRect();
    const h = globalThis.innerHeight || 0;
    return r.bottom > 0 && r.top < h + 160;
  }
  function followTail() {
    if (typeof el.interim.scrollIntoView === 'function') el.interim.scrollIntoView({ block: 'nearest' });
  }

  function showInterim() { return prefs.get().showInterim !== false; }

  function renderTranscript() {
    const t = s.transcript;
    const rec = s.rec || {};
    const active = ACTIVE_PHASES.has(rec.phase);
    const pend = tx.pendingLine(t, { showInterim: showInterim() });
    log.pending({ ...pend, listening: rec.phase === 'recording' && t.source === 'mic' });
    const content = tx.hasContent(t);
    const hasPending = !!(pend.confirmed || pend.interim);
    el.empty.hidden = content || hasPending || active || (s.shot === 'recording');
    // A recording that ended with nothing heard says so where its words would be.
    const unheard = rec.phase === 'done' && t.source === 'mic' && !content ? nothingHeard(rec) : null;
    if (t.source === 'file' && !t.segments.length) {
      el.emptyTitle.textContent = 'No speech found.';
      el.emptyText.textContent = t.fileName ? `There were no words to transcribe in ${t.fileName}.` : 'There were no words to transcribe in this file.';
    } else if (unheard) {
      el.emptyTitle.textContent = unheard.title;
      el.emptyText.textContent = unheard.message;
    } else {
      el.emptyTitle.textContent = 'Your transcript appears here.';
      el.emptyText.textContent = 'Record, or upload a file. Timestamps, copy and export follow.';
    }
    // Copy and Export only once a segment exists (BUG-17: nothing to save is never saved as "—"); Ask AI too.
    el.copy.disabled = !content;
    el.exportBtn.disabled = !content;
    if (el.ask) el.ask.disabled = !content;
    el.clear.disabled = active || !(content || s.fileUrl || t.source === 'file');
    const meta = active ? '' : metaLine(t);
    el.meta.textContent = meta;
    const playable = t.source === 'file' && !!s.fileUrl && t.segments.length > 0;
    el.play.hidden = !playable;
    el.foot.hidden = !(meta || playable);
    renderPlay();
  }

  /* ───────────── recorder ───────────── */

  function onRecorderState(st) {
    if (s.shot === 'recording') exitShot();
    const prevError = s.rec ? s.rec.error : null;
    s.rec = st;
    if (st.error && st.error !== prevError) {
      if (st.error.kind === 'mic' && micHelp({ error: st.error })) {
        s.helpError = st.error;
        s.helpDismissed = '';
      } else {
        s.recError = st.error;
      }
      // Each different error once (the recorder reports a repeated notice once per session), with what happened.
      tellHidden(toastLine(st.error), 'error');
    }
    // 'starting' comes only once the microphone opened: access is granted, whatever the permission said before.
    if ((st.phase === 'starting' || st.phase === 'recording') && s.permission !== 'granted') {
      s.permission = 'granted';
      if (s.helpError && s.helpError.reason !== 'background') s.helpError = null;
    }
    renderAll();
  }

  // The server began the session (the recorder's 'started'): only now does a recording replace the transcript
  // on screen, so a stream that fails before it (the model or VAD didn't load, too many streams) leaves a
  // file's transcript and its audio as they were.
  function onStarted(st) {
    s.snapshot = s.transcript;
    dispatch({ type: 'session-start', model: st.model });
    // The chip counts from the same moment as the timer (the first sound the server got), not from when the
    // microphone opened before a model download.
    if (st.startedAt && mic.owner() === 'transcribe') setRecording({ owner: 'transcribe', startedAt: st.startedAt });
    announce('Recording started');
  }

  function onEnded({ reason, duration } = {}) {
    meter.reset();
    const snapshot = s.snapshot;
    s.snapshot = null;
    if (reason === 'cancelled') {
      if (snapshot) dispatch({ type: 'restore', state: snapshot });
      announce(snapshot ? 'Recording discarded' : 'Recording canceled');
    } else if (snapshot) {
      const before = snapshot.source === 'mic' ? tx.exportSegments(snapshot).length : 0;
      const added = tx.exportSegments(s.transcript).length > before;
      // A session that brought back no words gives back what it replaced: a file's transcript with its audio,
      // however the session ended (it failed, the microphone went, the server ended it, Stop), and after a
      // failure the recording before it too.
      if (!added && (reason === 'error' || snapshot.source === 'file')) {
        dispatch({ type: 'restore', state: snapshot });
      } else {
        if (snapshot.source === 'file') revokeFile();
        dispatch({ type: 'session-end', duration });
      }
      if (reason === 'stopped') {
        // Only the server ends a session while another view shows (Stop lives here): a toast says so there,
        // unless a lost microphone ended it (that was told already).
        if (s.shown) announce('Recording stopped');
        else if (!(s.rec && s.rec.error && s.rec.error.kind === 'mic')) tellHidden('Recording stopped', 'info');
      }
    }
    renderAll();
  }

  function blockedBy() {
    const help = currentHelp();
    if (uploadBusy()) return 'upload';
    if (s.shot === 'nomodel' || (s.modelsLoaded && !s.modelId)) return 'model';
    if (!s.modelId) return 'models-loading';
    if (help && help.blocking) return 'mic';
    return '';
  }

  /**
   * A result or a failure that lands while another view shows has no control nearby, so it gets a toast
   * with a way back; while Transcribe shows, the page itself says it.
   */
  function tellHidden(message, kind) {
    if (s.shown) return;
    toast(message, { kind, id: 'tr-background', action: { label: 'Show', run: () => navigate('transcribe') } });
  }

  function jobLabel() {
    const id = (s.rec && s.rec.model) || s.modelId;
    const job = (app.get().jobs || []).find((j) => j.key === `model:${id}`);
    return job ? job.label : '';
  }

  /* Keyboard focus never falls to <body> when the control that had it is disabled or goes away. */
  function activeEl() {
    const doc = root.ownerDocument;
    return doc ? doc.activeElement : null;
  }
  function focusLost(node = activeEl()) {
    const doc = root.ownerDocument;
    return !node || !doc || node === doc.body || node === doc.documentElement;
  }

  function renderCapture() {
    const rec = s.rec || {};
    const v = captureView(rec, { jobLabel: jobLabel(), blockedBy: blockedBy() });
    el.layout.setAttribute('data-phase', rec.phase || 'idle');
    const btn = el.record;
    if (v.disabled && !btn.disabled && activeEl() === btn) s.recordFocus = true;
    btn.setAttribute('aria-label', v.buttonLabel);
    btn.removeAttribute('aria-pressed');
    btn.disabled = v.disabled;
    if (v.busy) btn.setAttribute('aria-busy', 'true');
    else btn.removeAttribute('aria-busy');
    btn.classList.toggle('is-recording', v.recording || rec.phase === 'stopping');
    if (!v.disabled && s.recordFocus) {
      s.recordFocus = false;
      if (s.shown && focusLost()) btn.focus({ preventScroll: true });
    }
    if (el.phase.textContent !== v.phase) el.phase.textContent = v.phase;
    el.dot.hidden = !v.recording;
    el.timer.hidden = !v.showTimer;
    if (el.hint.textContent !== v.hint) el.hint.textContent = v.hint;
    const cancelHadFocus = activeEl() === el.cancel;
    el.recActions.hidden = !(v.active && rec.phase !== 'stopping');
    // Cancel goes away with the session it cancelled: its focus moves to Record.
    if (cancelHadFocus && el.recActions.hidden && s.shown) {
      if (btn.disabled) s.recordFocus = true;
      else btn.focus({ preventScroll: true });
    }
    el.micField.hidden = v.active;
    const showVad = v.recording && rec.vadEnabled === true;
    el.vad.hidden = !showVad;
    if (showVad) {
      el.vadLabel.textContent = vadLabel(rec.vad);
      el.vad.classList.toggle('pill--ok', rec.vad === 'speech');
      el.vad.classList.toggle('pill--idle', rec.vad !== 'speech');
    }
    // "Default microphone" is always a valid choice: the select is off only while recording, or when the
    // listing says there is no input (or this page can't capture at all).
    el.mic.disabled = v.active || (Array.isArray(s.devices) && !s.devices.length) || !mic.support().ok;
    tick();
  }

  function currentHelp() {
    const help = micHelp({ support: mic.support(), permission: s.permission, devices: s.devices, error: s.helpError });
    return help;
  }

  /* Where keyboard focus goes when the banner holding it goes away (never to <body>). */
  function focusRecord() {
    if (!s.shown) return;
    if (el.record.disabled) s.recordFocus = true;   // it gets focus back once it is on again
    else el.record.focus({ preventScroll: true });
  }
  function focusUpload() {
    if (!s.shown) return;
    if (!el.file.disabled) {
      el.file.focus({ preventScroll: true });
    } else if (uploadBusy()) {
      s.uploadFocus = true;   // Cancel holds it while the file is busy; the input gets it back after
      el.uploadCancel.focus({ preventScroll: true });
    }
  }

  /**
   * Show a banner, writing it only when what it says changes (bannerKey). Everything else that renders (a
   * models poll every 60 s, or every 2 s while Models works; each VAD change; a phase change) leaves it alone:
   * the focused button stays, an open Details stays open, and a role=alert is not read out again. When the
   * banner holding keyboard focus changes or goes, focus moves to the same command in the new banner, else to
   * `fallback`.
   */
  function setBanner(node, slot, error, options, fallback) {
    const key = bannerKey(error, options);
    if (shownBanner[slot] !== key) {
      const at = activeEl();
      const had = !!at && at !== node && node.contains(at);
      const cmd = had && at.getAttribute ? at.getAttribute('data-action') : null;
      render(node, error ? bannerHtml(error, options) : html``);
      shownBanner[slot] = key;
      if (had && s.shown) {
        const same = cmd && error ? Array.from(node.querySelectorAll('[data-action]')).find((b) => b.getAttribute('data-action') === cmd) : null;
        if (same) same.focus({ preventScroll: true });
        else fallback();
      }
    }
    node.hidden = !error;
  }

  function renderHelp() {
    const help = currentHelp();
    // While audio is coming in (the microphone is open from 'starting' on), "no microphone" or "blocked" is
    // plainly not the case: say nothing then.
    const capturing = !!s.rec && (s.rec.phase === 'starting' || s.rec.phase === 'recording');
    const show = !!help && !(capturing && !help.blocking) && !(s.helpDismissed && s.helpDismissed === help.reason && !help.blocking);
    setBanner(el.help, 'help', show ? { title: 'Microphone problem', message: help.message } : null, {
      tone: 'warning', dismissAction: show && help.blocking ? '' : 'dismiss-help',
    }, focusRecord);
    if (show) el.help.setAttribute('data-reason', help.reason);
    else el.help.removeAttribute('data-reason');
  }

  /**
   * A recording that hears nothing for a few seconds says so, so a silent or too quiet microphone
   * never looks like a broken feature. The banner sits under the level meter it is about, never above Stop (the
   * button never moves under the pointer), and Dismiss hides it for the session.
   */
  function renderHeard() {
    const rec = s.rec || {};
    const live = !s.shot && (rec.phase === 'recording' || rec.phase === 'stopping') ? nothingHeard(rec) : null;
    const quiet = live && s.heardDismissed !== live.reason ? live : null;
    setBanner(el.heard, 'heard', quiet ? { title: quiet.title, message: quiet.message } : null, {
      tone: 'warning', dismissAction: 'dismiss-heard',
    }, focusRecord);
    if (quiet) el.heard.setAttribute('data-reason', quiet.reason);
    else el.heard.removeAttribute('data-reason');
  }

  function renderErrors() {
    const rec = s.recError;
    setBanner(el.recError, 'rec', rec, {
      actions: rec ? errorActions(rec, 'record-retry') : [], dismissAction: 'dismiss-rec-error',
    }, focusRecord);
    const up = s.upload && s.upload.error;
    setBanner(el.uploadError, 'upload', up || null, {
      actions: up ? errorActions(up, 'upload-retry') : [], dismissAction: 'dismiss-upload-error',
    }, focusUpload);
  }

  function renderNoModel() {
    const none = s.shot === 'nomodel' || (s.modelsLoaded && !s.modelOptions.length);
    el.nomodel.hidden = !none;
    el.layout.classList.toggle('is-nomodel', none);
    // With nothing to choose, the select would only repeat the card above: it steps aside until a model comes.
    el.modelField.hidden = none;
  }

  /* ───────────── upload ───────────── */

  function uploadBusy() {
    return !!s.upload && (s.upload.phase === 'preparing' || s.upload.phase === 'uploading');
  }

  // The phase shown (the recorder's, or a screenshot state's) decides, so a picture matches the real thing.
  function canUpload() {
    return !uploadBusy() && !ACTIVE_PHASES.has((s.rec || {}).phase) && !!s.modelId && s.shot !== 'nomodel';
  }

  function renderUpload() {
    const up = s.upload || {};
    const busy = uploadBusy();
    const disabled = !busy && !canUpload();
    // A file chosen from the keyboard: the input is off while the file is busy, so its focus goes to Cancel,
    // and comes back to the input when the file is done or cancelled. The model select goes off too (a file
    // dropped while it had focus).
    if (busy && (activeEl() === el.file || activeEl() === el.model)) s.uploadFocus = true;
    if (busy) el.drop.setAttribute('aria-busy', 'true');
    else el.drop.removeAttribute('aria-busy');
    el.drop.classList.toggle('is-busy', busy);
    el.drop.classList.toggle('is-disabled', disabled);
    el.file.disabled = busy || disabled;
    el.dropTitle.textContent = busy ? up.fileName : 'Upload audio';
    el.dropHint.hidden = busy;
    let phase = '';
    if (busy) {
      const job = up.phase === 'preparing' ? jobLabel() : '';
      phase = up.phase === 'preparing' ? (job || 'Preparing…') : 'Transcribing…';
    }
    el.dropProgress.hidden = !busy;
    if (el.dropPhase.textContent !== phase && phase) el.dropPhase.textContent = phase;
    // The live status changes with the phase only (the elapsed time beside it is aria-hidden).
    const status = !busy ? '' : `${up.phase === 'preparing' ? 'Preparing' : 'Transcribing'} ${up.fileName}…`;
    if (el.uploadStatus.textContent !== status) el.uploadStatus.textContent = status;
    el.uploadCancel.hidden = !busy;
    if (s.uploadFocus) {
      const at = activeEl();
      const ours = at === el.file || at === el.model || at === el.uploadCancel || focusLost(at);
      if (busy) {
        if (ours && at !== el.uploadCancel && s.shown) el.uploadCancel.focus({ preventScroll: true });
      } else {
        s.uploadFocus = false;
        if (ours && s.shown && !el.file.disabled) el.file.focus({ preventScroll: true });
      }
    }
    tick();
  }

  // The model select is off while a recording or a file runs: picking another model then would change nothing
  // for it (on desktop the select stays in view beside a running session). After renderUpload, which has moved
  // a keyboard focus on it to the file's Cancel.
  function renderModelLock() {
    const lock = ACTIVE_PHASES.has((s.rec || {}).phase) || uploadBusy();
    if (lock && !el.model.disabled && activeEl() === el.model) s.recordFocus = true;   // Record takes it back
    if (modelSelect) modelSelect.setLocked(lock);
  }

  async function startUpload(file) {
    if (!file) return;
    if (s.shot) exitShot();
    if (uploadBusy()) {
      toast('One file at a time: cancel the current one first.', { id: 'tr-one-file' });
      return;
    }
    if (recorder.isActive()) {
      toast('Stop the recording first.', { id: 'tr-one-file' });
      return;
    }
    if (!canUpload()) return;
    if (s.transcript.source === 'mic' && tx.hasContent(s.transcript)) {
      const ok = await confirm({
        title: 'Replace the transcript?',
        message: "The file's transcript replaces what you recorded. Copy or export it first to keep it.",
        confirmLabel: 'Replace',
      });
      if (!ok) return;
    }
    s.lastFile = file;
    const result = await uploader.start(file, { model: s.modelId });
    if (el.file) el.file.value = '';
    if (!result) {
      const failed = s.upload && s.upload.error;
      if (failed) tellHidden(`Couldn't transcribe ${file.name || 'the file'}`, 'error');
      else if (s.upload && s.upload.phase === 'idle') announce('Transcription canceled');
      return;
    }
    stopFilePlayback();
    revokeFile();
    try { s.fileUrl = URL.createObjectURL(file); } catch { s.fileUrl = ''; }
    dispatch({ type: 'file-result', ...result });
    if (result.segments.length) tellHidden(`Transcript ready: ${result.fileName}`, 'success');
    else tellHidden(`No speech found in ${result.fileName}`, 'info');
  }

  /* ───────────── playback of the uploaded file ───────────── */

  function revokeFile() {
    if (!s.fileUrl) return;
    stopFilePlayback();
    try { URL.revokeObjectURL(s.fileUrl); } catch { /* gone */ }
    s.fileUrl = '';
  }

  function stopFilePlayback() {
    const cur = audio.current();
    if (cur && cur.id === FILE_SESSION) audio.closeCurrent();
  }

  function playFrom(start) {
    if (!s.fileUrl) return Promise.resolve();
    const t = s.transcript;
    const p = audio.playClip({
      id: FILE_SESSION, owner: 'transcribe', title: t.fileName || 'Uploaded audio', subtitle: 'Transcribe',
      homeRoute: 'transcribe', src: s.fileUrl,
    });
    // playClip set the source synchronously: seek before the audio loads, so it starts there.
    const cur = audio.current();
    if (cur && cur.id === FILE_SESSION && cur.controls.seekTo) cur.controls.seekTo(Math.max(0, Number(start) || 0));
    return p.then(() => undefined, (error) => {
      toast(error && error.name === 'CaptureActiveError' ? error.message : "Couldn't play the file.", { kind: 'error', id: 'tr-play' });
    });
  }

  function renderPlay() {
    const player = app.get().player;
    const ours = !!player && player.id === FILE_SESSION;
    const playing = ours && player.playing;
    if (playing !== s.playing || el.play.getAttribute('aria-label') === null) {
      s.playing = playing;
      el.play.setAttribute('aria-label', playing ? 'Pause' : 'Play the file');
      el.play.setAttribute('data-tooltip', playing ? 'Pause' : 'Play the file');
      el.play.classList.toggle('is-playing', playing);
    }
    log.setCurrent(ours ? tx.segmentAt(s.transcript, player.position) : -1);
  }

  /* ───────────── record, stop, cancel ───────────── */

  function toggleRecord() {
    if (s.shot) exitShot();
    if (recorder.state.phase === 'recording') {
      recorder.stop();
      return;
    }
    startSession();
  }

  /** Start a recording; never stops one (Retry goes through here). */
  function startSession() {
    if (s.shot) exitShot();
    if (recorder.isActive() || el.record.disabled) return;
    s.recError = null;
    if (s.helpError && s.helpError.reason !== 'denied') s.helpError = null;
    s.helpDismissed = '';
    s.heardDismissed = '';
    renderErrors();
    // start() moves to 'requesting' before its first await, so a second click is refused (BUG-08).
    recorder.start({
      model: s.modelId,
      // Until the device list is in, the select shows only "Default microphone": leave the choice to the
      // saved device then (undefined), instead of forcing the default.
      deviceId: Array.isArray(s.devices) ? el.mic.value : undefined,
      echoCancellation: prefs.get().echoCancellation !== false,
      noiseSuppression: prefs.get().noiseSuppression !== false,
    });
  }

  async function cancelRecording() {
    if (s.shot === 'recording') {
      exitShot();
      return;
    }
    const rec = recorder.state;
    if (!recorder.isActive() || rec.phase === 'stopping') return;
    const seconds = rec.startedAt ? (Date.now() - rec.startedAt) / 1000 : 0;
    if (rec.phase === 'recording' && seconds > CONFIRM_CANCEL_AFTER_S) {
      const ok = await confirm({
        title: 'Discard this recording?',
        message: "Its transcript isn't kept.",
        confirmLabel: 'Discard',
        danger: true,
      });
      if (!ok) return;
    }
    recorder.cancel();
  }

  /* ───────────── devices ───────────── */

  async function refreshDevices() {
    const token = (devToken += 1);
    if (!mic.support().ok) {
      s.devices = null;
      renderAll();
      return;
    }
    const rows = await mic.listDevices();
    if (token !== devToken) return;
    s.devices = rows;
    const saved = mic.savedDevice();
    if (rows.length) {
      render(el.mic, html`${rows.map((r) => html`<option value="${r.id}">${r.label}</option>`)}`);
      el.mic.value = rows.some((r) => r.id === saved) ? saved : '';
    } else {
      render(el.mic, html`<option value="">Default microphone</option>`);
    }
    renderAll();
  }

  /* ───────────── timers (view-only) ───────────── */

  function tick() {
    const rec = s.rec || {};
    if (rec.phase === 'recording' || rec.phase === 'stopping') {
      const end = rec.phase === 'stopping' && rec.stoppedAt ? rec.stoppedAt : Date.now();
      const text = clock(Math.max(0, (end - (rec.startedAt || end)) / 1000));
      if (el.timer.textContent !== text) el.timer.textContent = text;
    }
    if (uploadBusy()) {
      const text = clock(Math.max(0, (Date.now() - (s.upload.startedAt || Date.now())) / 1000));
      if (el.uploadElapsed.textContent !== text) el.uploadElapsed.textContent = text;
    }
    const need = s.shown && ((rec.phase === 'recording') || uploadBusy());
    if (need && ticker === null) ticker = setInterval(tick, 1000);
    if (!need && ticker !== null) {
      clearInterval(ticker);
      ticker = null;
    }
  }

  function renderAll() {
    renderNoModel();
    renderHelp();
    renderCapture();
    renderHeard();
    renderUpload();
    renderModelLock();
    renderErrors();
    renderTranscript();
  }

  /* ───────────── screenshot hook ───────────── */

  function applyShot(shot) {
    if (!SHOTS.includes(shot) || s.shot === shot) return;
    if (recorder.isActive() || uploader.busy) return;   // a real session or upload is never covered up
    exitShot();
    s.shot = shot;
    const now = Date.now();
    if (shot === 'recording') {
      const demo = recordingShot(now);
      s.rec = demo.rec;
      s.transcript = Object.freeze({ ...tx.createState(), source: 'mic', model: demo.rec.model, segments: Object.freeze(demo.transcript.segments.map((x) => ({ ...x, source: 'mic' }))), confirmed: { ...demo.transcript.confirmed, source: 'mic' }, interim: demo.transcript.interim });
      log.replace(s.transcript.segments, { file: false });
      meter.set(demo.level);
      setRecording({ owner: 'transcribe', startedAt: demo.rec.startedAt });
      s.shotRecording = true;
    } else if (shot === 'done') {
      const demo = doneShot();
      revokeFile();
      try {
        const rate = 8000;
        s.fileUrl = URL.createObjectURL(encodeWav(new Float32Array(Math.ceil(demo.duration * rate)), rate));
      } catch {
        s.fileUrl = '';
      }
      dispatch({ type: 'file-result', ...demo });
    } else if (shot === 'uploading') {
      s.upload = Object.freeze(uploadingShot(now));
    } else if (shot === 'error') {
      const demo = errorShot();
      s.recError = demo.rec;
      s.upload = Object.freeze({ phase: 'error', fileName: 'interview.m4a', startedAt: 0, error: demo.upload });
    }
    renderAll();
  }

  function exitShot() {
    if (!s.shot) return;
    const was = s.shot;
    s.shot = '';
    if (s.shotRecording) {
      s.shotRecording = false;
      if (app.get().recording && !mic.owner()) setRecording(null);
      meter.reset();
    }
    s.rec = recorder.state;
    s.upload = uploader.state;
    if (was === 'done') revokeFile();
    if (was === 'recording' || was === 'done') dispatch({ type: 'clear' });
    if (was === 'error') s.recError = null;
    renderAll();
  }

  /* ───────────── wiring ───────────── */

  const commands = {
    record: () => toggleRecord(),
    cancel: () => cancelRecording(),
    // Retry leaves with its banner: focus goes to the control it retries first, not to <body>. It only
    // ever starts a session: a recording that is running goes on.
    'record-retry': () => {
      s.recError = null;
      recorder.clearError();
      renderErrors();
      focusRecord();
      startSession();
    },
    'upload-retry': () => {
      if (s.shot) exitShot();
      uploader.clearError();
      el.file.focus({ preventScroll: true });
      return startUpload(s.lastFile);
    },
    'upload-cancel': () => {
      if (s.shot === 'uploading') {
        exitShot();
        return;
      }
      uploader.cancel();
    },
    'dismiss-rec-error': () => {
      s.recError = null;
      recorder.clearError();
      renderErrors();
      focusRecord();
    },
    'dismiss-upload-error': () => {
      if (s.shot === 'error') s.upload = Object.freeze({ ...s.upload, phase: 'idle', error: null });
      else uploader.clearError();
      renderAll();
      el.file.focus({ preventScroll: true });
    },
    'dismiss-heard': () => {
      s.heardDismissed = el.heard.getAttribute('data-reason') || '';
      renderHeard();
      focusRecord();
    },
    'dismiss-help': () => {
      const help = currentHelp();
      s.helpDismissed = help ? help.reason : '';
      renderHelp();
      focusRecord();
    },
    'open-settings': () => openConnectionSettings(),
    copy: async () => {
      const text = tx.fullText(s.transcript);
      if (!text) return;
      if (await copyText(text)) toast('Transcript copied', { kind: 'success', id: 'tr-copy' });
      else toast("Couldn't copy here: select the transcript and copy it instead", { kind: 'error', id: 'tr-copy' });
    },
    clear: async () => {
      const t = s.transcript;
      const ok = await confirm({
        title: 'Clear the transcript?',
        message: t.source === 'file' ? 'The transcript of the file stays in History.' : "A recording's transcript isn't saved anywhere else.",
        confirmLabel: 'Clear',
        danger: true,
      });
      if (!ok) return;
      if (s.shot) exitShot();
      revokeFile();
      dispatch({ type: 'clear' });
      announce('Transcript cleared');
      if (!el.panelTitle.hasAttribute('tabindex')) el.panelTitle.setAttribute('tabindex', '-1');
      el.panelTitle.focus({ preventScroll: true });
    },
    'play-from': (button) => playFrom(Number(button.getAttribute('data-start')) || 0),
    'play-file': () => {
      const cur = audio.current();
      if (cur && cur.id === FILE_SESSION) {
        cur.controls.toggle();
        return undefined;
      }
      return playFrom(0);
    },
  };
  // The banners' buttons (ui.js bannerHtml) name these commands in their own data-action, like the partial's.
  offs.push(actions(root, commands, {
    onError: (error) => toast(error && error.message ? error.message : 'Something went wrong.', { kind: 'error' }),
  }));

  /**
   * Ask AI: the transcript as it reads now (the words, without timestamps), as a
   * `transcript` input. The button shows only while the assistant is on.
   */
  function transcriptSource() {
    const t = s.transcript;
    const text = tx.fullText(t);
    if (!text) return null;
    const label = t.source === 'file' && t.fileName ? `The transcript of ${t.fileName}` : 'The transcript';
    return { id: 'transcript', label, input: { kind: 'transcript', text }, text, chars: charCount(text), title: label };
  }
  const ask = bindAskButton(el.ask, {
    onClick: (button) => {
      const source = transcriptSource();
      if (!source) return;
      openAskAi({ trigger: button, sources: [source] }).catch((error) => toast(error && error.message ? error.message : 'Something went wrong.', { kind: 'error' }));
    },
  });
  offs.push(() => ask.destroy());

  const menu = bindMenuButton(el.exportBtn, el.exportMenu, {
    placement: 'bottom-end',
    onSelect: (item) => {
      const kind = item.getAttribute('data-format');
      if (!FORMATS[kind]) return;
      const t = s.transcript;
      const file = exportFile(kind, { segments: tx.exportSegments(t), model: t.model, duration: t.duration });
      downloadBlob(new Blob([file.content], { type: file.mime }), file.filename);
      toast(`Saved ${file.filename}`, { kind: 'success', id: 'tr-export' });
    },
  });

  // The upload's object URL goes when the page goes. A page kept for Back (the back/forward cache)
  // keeps it, so its timestamps still play when the listener returns.
  const onPageHide = (event) => {
    if (!event || !event.persisted) revokeFile();
  };
  if (typeof window !== 'undefined' && typeof window.addEventListener === 'function') window.addEventListener('pagehide', onPageHide);

  el.file.addEventListener('change', () => {
    const file = el.file.files && el.file.files[0];
    if (file) startUpload(file);
  });
  const onDragOver = (event) => {
    event.preventDefault();
    const ok = canUpload();
    if (event.dataTransfer) event.dataTransfer.dropEffect = ok ? 'copy' : 'none';
    el.drop.classList.toggle('is-dragover', ok);
  };
  const onDragLeave = (event) => {
    if (event.relatedTarget && el.drop.contains(event.relatedTarget)) return;
    el.drop.classList.remove('is-dragover');
  };
  const onDrop = (event) => {
    event.preventDefault();
    el.drop.classList.remove('is-dragover');
    const file = event.dataTransfer && event.dataTransfer.files ? event.dataTransfer.files[0] : null;
    if (file) startUpload(file);
  };
  el.drop.addEventListener('dragover', onDragOver);
  el.drop.addEventListener('dragenter', onDragOver);
  el.drop.addEventListener('dragleave', onDragLeave);
  el.drop.addEventListener('drop', onDrop);

  el.mic.addEventListener('change', () => mic.saveDevice(el.mic.value));
  el.showInterim.addEventListener('change', () => { prefs.set({ showInterim: !!el.showInterim.checked }); renderTranscript(); });
  el.showTs.addEventListener('change', () => { prefs.set({ compactTranscript: !el.showTs.checked }); });

  const applyPrefs = () => {
    const p = prefs.get();
    el.showInterim.checked = p.showInterim !== false;
    el.showTs.checked = p.compactTranscript !== true;
    log.setCompact(p.compactTranscript === true);
  };
  applyPrefs();
  offs.push(app.select((st) => st.prefs, () => { applyPrefs(); renderTranscript(); }));
  offs.push(app.select((st) => st.player, () => renderPlay()));
  offs.push(app.select((st) => st.jobs, () => { renderCapture(); renderUpload(); }));

  modelSelect = bindModelSelect(el.model, {
    hint: el.modelHint,
    onChange: ({ value, options, loaded }) => {
      s.modelId = value;
      s.modelOptions = options;
      s.modelsLoaded = loaded;
      if (s.started) renderAll();
    },
  });

  offs.push(mic.onDeviceChange(() => { refreshDevices(); }));
  offs.push(onBus('auth:changed', () => { if (s.shown) renderAll(); }));

  offs.push(register({
    id: 'transcribe-record', keys: ['r'], scope: 'view:transcribe', singleKey: true,
    description: 'Start or stop recording', handler: () => toggleRecord(),
  }));
  offs.push(register({
    id: 'transcribe-cancel', keys: ['Escape'], scope: 'view:transcribe', allowInEditable: true,
    description: 'Cancel recording (asks first if longer than 30 s)',
    when: () => s.shot === 'recording' || (recorder.isActive() && recorder.state.phase !== 'stopping'),
    handler: () => cancelRecording(),
  }));

  renderAll();

  return {
    show(route) {
      s.shown = true;
      if (!s.started) {
        s.started = true;
        refreshDevices();
        mic.permission().then((p) => { s.permission = p; renderAll(); }, () => {});
      }
      const shot = route && route.shot ? route.shot : '';
      if (SHOTS.includes(shot)) applyShot(shot);
      else if (s.shot) exitShot();   // no shot, or one for the shell: the view shows its real state
      renderAll();
      if (!shot) ask.refresh();   // a screenshot state is display only: it asks the server nothing
      if (ASK_SHOTS.includes(shot)) {
        // The Ask AI sheet as it looks (display only: nothing is sent): over the transcript when there is one.
        const source = transcriptSource();
        openSample(shot, { trigger: el.ask, sources: source ? [source] : undefined }).catch((error) => { if (globalThis.console) globalThis.console.error(error); });
      }
    },
    hide() {
      s.shown = false;
      if (s.shotRecording) exitShot();
      tick();
    },
    destroy() {
      s.shown = false;
      tick();
      for (const off of offs.splice(0)) {
        try { off(); } catch { /* gone */ }
      }
      menu.destroy();
      modelSelect.destroy();
      meter.destroy();
      if (typeof window !== 'undefined' && typeof window.removeEventListener === 'function') window.removeEventListener('pagehide', onPageHide);
      el.drop.removeEventListener('dragover', onDragOver);
      el.drop.removeEventListener('dragenter', onDragOver);
      el.drop.removeEventListener('dragleave', onDragLeave);
      el.drop.removeEventListener('drop', onDrop);
    },
    /** Test hook: the view's state. */
    state: () => s,
  };
}

export default {
  id: 'transcribe',
  title: 'Transcribe',
  partial: '/static/partials/transcribe.html',
  init(root) {
    return createTranscribeView(root);
  },
};
