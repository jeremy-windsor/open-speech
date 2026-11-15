/**
 * Speak: Generate mode, and the host of Live mode (live-ui.js mounts into #sp-live-slot).
 *
 * Live is a streaming test for developers: Speak shows Generate
 * only, with no mode switch, unless Settings › Developer turns on prefs.showLive. Off, a link, the saved last
 * route or Back that asks for #speak/live shows Generate (the address is replaced), and Live never mounts, so its
 * shortcuts are not in the "?" sheet; turning it off unmounts Live, which stops a running session. On, the switch
 * reads Generate | Live (test).
 *
 * The Generate path keeps first audio fast: the click primes the shared audio element and snapshots the settings
 * synchronously, then at most one request goes out before POST /v1/audio/speech (the status check inside
 * models.ensureReady), none on the fast path. Streaming is on by default and never used with effects.
 * Takes play through the app's audio coordinator, so they keep playing when the view is left.
 *
 * Editing from Voices: Edit in Speak on a named voice's version
 * shows the editing bar #sp-editing, "Editing Narrator for Kokoro", with Update voice (the version's PUT, no name
 * prompt) and Cancel (Speak goes back to what it had); changing the model ends it with a note. Edit on a preset
 * applies it and shows "Editing preset “Podcast”" with Update preset (PUT /api/profiles/{id} with Speak's setup),
 * Save as new… and Cancel. New preset on a voice's card chooses that voice and focuses Save preset…, whose name
 * starts as the voice's.
 *
 * A take whose voice was designed from the description can be kept: its Save as voice…, or Save named voice…
 * while Speak designs (the newest designed take for the model), saves it as a designed voice.
 * @module views/speak/index
 */
import { html, render, debounce, setBusy } from '../../core/dom.js';
import { icon } from '../../core/icons.js';
import { app, on as onEvent, emit as emitEvent, takeIntent } from '../../core/store.js';
import * as router from '../../core/router.js';
import * as models from '../../core/models.js';
import * as audio from '../../core/audio.js';
import { request, explain } from '../../core/api.js';
import * as auth from '../../core/auth.js';
import * as perf from '../../core/perf.js';
import * as prefs from '../../core/prefs.js';
import { KEYS, readJson, writeJson } from '../../core/storage.js';
import { register, modLabel } from '../../core/shortcuts.js';
import { toast } from '../../core/toast.js';
import { announce } from '../../core/a11y.js';
import { copyText } from '../../core/clipboard.js';
import { downloadBlob } from '../../core/download.js';
import { codePointLength } from '../../core/format.js';
import { encodeWav } from '../../core/pcm.js';
import * as theme from '../../core/theme.js';
import { listNamed, withModel } from '../../core/voices.js';
import { bindMenuButton } from '../../components/menu.js';
import { bindAskButton, openAskAi } from '../../components/ask-ai.js';
import { createEditor, checkInput } from './editor.js';
import { createInspector, renderBanner, errorBanner, realizationFor } from './inspector.js';
import * as presets from './presets.js';
import { createTakes, receive, rtfText, peaks as wavePeaks, sampleProblem, sampleWav, MAX_SAMPLE_S, SAMPLE_TOO_LONG } from './takes.js';
import { buildSpeechRequest, playbackPlan, downloadHint, takeFilename, mseSupported } from './request.js';
import { toCurl, toPython, toJs } from './code.js';
import { savedSetup, claim as claimSpeakVoice, voiceKind, __resetForTests as resetSpeakVoice } from './speak-voice.js';

/** The sample rate of the take the `take` screenshot state makes locally. */
const SHOT_RATE = 8000;

/** The data a view reloads when it is older than this (GS-12). */
export const STALE_MS = 30000;
export const PERSIST_MS = 300;
/** A pointer click this soon after Generate started is the tail of a double click, not a Stop. */
export const STOP_GUARD_MS = 400;
export const SAMPLE_TEXT = 'Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do.';
const BLEND_SAMPLE = 'af_bella(2)+af_sky(1)';
const CODE = { curl: { fn: toCurl, label: 'curl' }, python: { fn: toPython, label: 'Python' }, js: { fn: toJs, label: 'JavaScript' } };
const USER_KINDS = new Set(['model-pending', 'model', 'voice', 'named', 'blend', 'reference', 'speed', 'options', 'format', 'effects', 'instructions']);
/** The editing bar's hints. */
export const EDIT_VOICE_HINT = 'Change the voice, blend or recording, then Update voice. Presets that use it follow.';
export const EDIT_PRESET_HINT = 'Change the voice, speed, format or effects, then Update preset.';

/** PURE: the editing bar's title: "Editing Narrator for Kokoro", "Editing preset “Podcast”". */
export function editingTitle(editing) {
  if (!editing) return '';
  if (editing.kind === 'voice') return `Editing ${editing.name} for ${models.modelLabel(editing.model)}`;
  return `Editing preset “${editing.name}”`;
}

/* ───────────────────────────── pure helpers (tested) ───────────────────────────── */

/** The stored setup's validation lives with the voice Speak publishes (speak-voice.js); tests import it here. */
export { savedSetup };

/**
 * The setup to restore on a model: the whole stored setup on the model it was made for; on another model
 * only what doesn't depend on it (speed, format, stream, effects, instructions).
 */
export function restoreFor(saved, model) {
  if (!saved) return { model };
  if (saved.model === model) return { ...saved, model };
  const { speed, format, stream, effects, instructions } = saved;
  const out = { model };
  for (const [k, val] of Object.entries({ speed, format, stream, effects, instructions })) if (val !== undefined) out[k] = val;
  return out;
}

/** The status line for a phase. */
export function phaseText(phase, { job = '', ttfaMs = null, rtf = null, cached = false, loadMs = 0 } = {}) {
  const extra = loadMs > 0 ? ` · model loaded in ${(loadMs / 1000).toFixed(1)} s` : '';
  switch (phase) {
    case 'checking': return 'Checking model…';
    case 'loading': return job || 'Loading the model…';
    case 'generating': return 'Generating…';
    case 'streaming': return Number.isFinite(ttfaMs) ? `Streaming · first audio ${Math.round(ttfaMs)} ms` : 'Streaming…';
    case 'done': {
      const parts = ['Done'];
      if (cached) parts.push('cached');
      if (Number.isFinite(rtf)) parts.push(rtfText(rtf));
      return parts.join(' · ') + extra;
    }
    case 'stopped': return 'Stopped';
    default: return '';
  }
}

/**
 * PURE: the audio of the `take` screenshot state, made here: `seconds` of a soft murmur in syllables, with
 * pauses between words, so its waveform looks like speech. Deterministic (a fixed seed).
 * @returns {Float32Array}
 */
export function shotSamples(seconds = 6.4, rate = SHOT_RATE) {
  const n = Math.max(1, Math.round(seconds * rate));
  const out = new Float32Array(n);
  let seed = 7;
  const rand = () => {
    seed = (Math.imul(seed, 1103515245) + 12345) >>> 0;
    return (seed & 0x7fffffff) / 0x7fffffff;
  };
  let at = 0;
  while (at < n) {
    const len = Math.round((0.12 + rand() * 0.14) * rate);
    const amp = 0.08 + rand() * 0.2;
    const pitch = 110 + rand() * 60;
    for (let i = 0; i < len && at + i < n; i += 1) {
      out[at + i] = amp * Math.sin((Math.PI * i) / len) * Math.sin((2 * Math.PI * pitch * (at + i)) / rate);
    }
    at += len + Math.round((rand() < 0.3 ? 0.18 : 0.03) * rate);
  }
  return out;
}

/** Whether Settings shows Speak › Live, the streaming test for developers (prefs.showLive). */
export function liveEnabled() {
  try {
    return prefs.get().showLive === true;
  } catch {
    return false;
  }
}

/* ───────────────────────────── module state ───────────────────────────── */

let takeSeq = 0;

/** Test hook. */
export function __resetForTests() {
  takeSeq = 0;
  presets.__resetForTests();
  resetSpeakVoice();
}

/* ───────────────────────────── the view ───────────────────────────── */

function init(root) {
  const q = (id) => root.querySelector(`#${id}`);
  const el = {
    title: q('sp-title'), mode: q('sp-mode'), more: q('sp-more'), moreMenu: q('sp-more-menu'), notice: q('sp-notice'),
    empty: q('sp-empty'), layout: q('sp-layout'), input: q('sp-input'), counter: q('sp-counter'), upload: q('sp-upload'),
    limitHint: q('sp-limit-hint'), limitText: q('sp-limit-text'), presets: q('sp-presets'), genbar: q('sp-genbar'),
    generate: q('sp-generate'), kbdMod: q('sp-kbd-mod'), genHint: q('sp-gen-hint'), status: q('sp-gen-status'),
    statusText: q('sp-gen-status-text'), elapsed: q('sp-gen-elapsed'), statusExtra: q('sp-gen-status-extra'),
    genError: q('sp-gen-error'), liveSlot: q('sp-live-slot'),
    takesSection: q('sp-takes-section'), takes: q('sp-takes'), takesEmpty: q('sp-takes-empty'), inspector: q('sp-inspector'),
    voice: q('sp-voice'), savePreset: q('sp-save-preset'), saveVoice: q('sp-save-voice'), ask: q('sp-ask'),
    editing: q('sp-editing'), editingTitle: q('sp-editing-title'), editingHint: q('sp-editing-hint'),
    editingUpdate: q('sp-editing-update'), editingNew: q('sp-editing-new'), editingCancel: q('sp-editing-cancel'),
  };
  // editing: what the editing bar edits: {kind: 'voice', id, name, model, before} or {kind:
  // 'preset', id, name, before}; `before` is Speak's setup and applied preset when it began, for Cancel.
  // newPreset: {id, name} after a New preset hand-off, so Save preset… starts with the voice's name.
  // voiceName: the named voice whose editing a model change ended, so Save named voice… starts with its name.
  const ctx = {
    mode: 'generate', shown: false, visible: false, gen: null, lastGen: null, live: null, liveLoading: null,
    setupStarted: false, setupResolve: null, setupDone: null, presetsStale: true, persistOff: false, editing: null,
    newPreset: null, voiceName: '', elapsedTimer: null, lastSpeakVoice: '', redirecting: false,
  };
  ctx.setupDone = new Promise((resolve) => { ctx.setupResolve = resolve; });
  const offs = [];
  // From now on Speak publishes app.speakVoice itself; the boot publisher (main.js) stops.
  claimSpeakVoice();

  /* ── Live ── */

  function liveActive() {
    try {
      return !!(ctx.live && typeof ctx.live.isActive === 'function' && ctx.live.isActive());
    } catch {
      return false;
    }
  }

  function stopLive(reason) {
    try {
      if (liveActive()) ctx.live.stop(reason);
    } catch { /* Live reports its own errors */ }
  }

  function getVoiceConfig() {
    const s = inspector.state();
    const caps = { ...inspector.caps() };
    return {
      model: s.model,
      voice: inspector.voiceSpec(),
      speed: s.speed,
      instructions: caps.instructions ? String(s.instructions || '').trim() : '',
      voice_library_ref: caps.voice_clone ? s.reference : '',
      caps,
    };
  }

  function ensureLive() {
    if (ctx.live) return Promise.resolve(ctx.live);
    if (ctx.liveLoading) return ctx.liveLoading;
    ctx.liveLoading = (async () => {
      try {
        const mod = await import('./live-ui.js');
        const live = await Promise.resolve(mod.mountLive(el.liveSlot, {
          editor: el.input,
          getVoiceConfig,
          onState: (state) => {
            ctx.liveState = state;
            root.setAttribute('data-live-active', String(!!(ctx.live && ctx.live.isActive && ctx.live.isActive())));
          },
        }));
        ctx.live = live || null;
        // Settings hid Live while its controls loaded: they go again.
        if (!liveEnabled()) unmountLive();
        return ctx.live;
      } catch (error) {
        if (globalThis.console) globalThis.console.error(error);
        render(el.liveSlot, html`<div class="empty sp-live-missing">${icon('alert', { className: 'ico empty__icon' })}<h2 class="empty__title">Live isn't available</h2><p class="empty__text">This page couldn't load Live mode. Reload the page, or use Generate.</p></div>`);
        return null;
      } finally {
        ctx.liveLoading = null;
      }
    })();
    return ctx.liveLoading;
  }

  /** Unmount Live: its engine stops a running session, and its shortcuts leave the "?" sheet. */
  function unmountLive() {
    const live = ctx.live;
    ctx.live = null;
    if (live && typeof live.destroy === 'function') {
      try { live.destroy(); } catch (error) { if (globalThis.console) globalThis.console.error(error); }
    }
    render(el.liveSlot, html``);
    root.setAttribute('data-live-active', 'false');
  }

  /**
   * Show or hide Speak › Live as Settings says (prefs.showLive). Off: no mode
   * switch, Live unmounts, and Speak in Live mode shows Generate (the address follows while Speak shows).
   */
  function syncLiveSwitch() {
    const on = liveEnabled();
    el.mode.hidden = !on;
    root.setAttribute('data-show-live', String(on));
    if (!on) {
      if (ctx.live) unmountLive();
      if (ctx.mode === 'live') {
        setMode('generate', { switched: false });
        if (ctx.visible) {
          ctx.redirecting = true;   // the router shows #speak again at once: Speak shows it already
          try {
            router.navigate('speak', { replace: true, focus: 'keep' });
          } finally {
            ctx.redirecting = false;
          }
        }
      }
    }
    editor.refresh();   // the limit hint names Live only while it shows
  }

  /* ── notices ── */

  function hideNotice() {
    el.notice.hidden = true;
    render(el.notice, html``);
  }

  function showNotice(banner) {
    if (!banner) {
      hideNotice();
      return;
    }
    renderBanner(el.notice, { ...banner, onDismiss: () => { hideNotice(); el.input.focus(); } });
  }

  /* ── the parts ── */

  const editor = createEditor(
    { input: el.input, counter: el.counter, hint: el.limitHint, hintText: el.limitText, upload: el.upload },
    {
      getLimit: () => inspector.limit(),
      getSpeed: () => inspector.state().speed,
      enforceLimit: () => ctx.mode === 'generate',
      liveShown: liveEnabled,
      onLimit: () => updateGenerate(),
      onChange: () => { updateGenerate(); syncAsk(); },
      onUpload: () => stopLive('Text changed'),
      onError: (error) => toast(`Couldn't open the file: ${error?.message || error}`, { kind: 'error' }),
    },
  );

  const inspector = createInspector(root, {
    onChange: settingsChanged,
    stopLive,
    onNotice: showNotice,
  });

  const row = presets.createPresetsRow(el.presets, {
    onApply: (id, name) => {
      presets.markPicked();
      applyPresetById(id, name);
    },
    onSave: () => { savePresetFlow(); },
    onBlocked: () => { announce(saveBlockedReason()); },
  });

  const takes = createTakes({ list: el.takes, empty: el.takesEmpty }, {
    onUse: (take) => { useTake(take); },
    onCopy: (take) => { copyCode('curl', take.req); },
    onSaveVoice: (take) => { saveTakeAsVoice(take); },
    onRemove: (take, { focus, next }) => {
      if (take.offPlaying) take.offPlaying();
      if (!focus) return;
      if (next) takes.focusPlay(next);
      else el.input.focus();
    },
    onPlayError: (error, take) => playFailed(error, take),
  });

  /* ── settings ── */

  const persist = debounce(() => {
    if (ctx.persistOff || !inspector.ready()) return;
    writeJson(KEYS.speak, inspector.currentSetup());
  }, PERSIST_MS);

  /**
   * app.speakVoice for the Reader (RE-04, "Same as Speak"): the voice string Speak sends, its label, and what
   * it is (`kind`), so a raw reference recording never reaches a Reader render (REV-09).
   */
  function updateSpeakVoice() {
    const s = inspector.state();
    const kind = voiceKind({ identity: s.identity, blend: s.mix.length >= 2, voice: s.voice, reference: s.reference });
    // A provider voice or blend carries its model's name elsewhere ("Kokoro · Heart").
    const label = kind === 'voice' || kind === 'blend' ? withModel(s.model, inspector.voiceLabel()) : inspector.voiceLabel();
    const next = { model: s.model, voice: inspector.voiceSpec(), label, kind };
    const key = JSON.stringify(next);
    if (key === ctx.lastSpeakVoice) return;
    ctx.lastSpeakVoice = key;
    app.set({ speakVoice: next });
  }

  function settingsChanged(kind) {
    if (USER_KINDS.has(kind)) row.setApplied('');
    // A version is one model's: another model ends its editing, with a note. A preset may change
    // model, so its editing goes on.
    const editing = ctx.editing;
    const model = inspector.state().model;
    if (editing && editing.kind === 'voice' && model && model !== editing.model) stopEditingForModel();
    persist();
    updateSpeakVoice();
    editor.refresh();
    updateGenerate();
    // Live gates Start on the model (live_reader) and shows the voice it will use: let it look again.
    if (ctx.live && typeof ctx.live.refresh === 'function') {
      try { ctx.live.refresh(); } catch { /* Live reports its own errors */ }
    }
  }

  /* ── Generate availability ── */

  function generateBlock() {
    const st = app.get();
    if (!st.models.fetchedAt) {
      if (!st.models.error) return { reason: 'Loading models…' };
      // The shell's connection banner says how to fix a refused key; the hint only names it.
      if (st.connection.state === 'unauthorized') return { reason: 'API key needed.' };
      if (st.connection.state === 'locked') return { reason: 'Too many wrong API keys.' };
      return { reason: "The models didn't load." };
    }
    if (!models.ttsModels(st.models.list).length) return { reason: 'No speech model is available.' };
    if (st.connection.state === 'offline') return { reason: "Offline. Generate works again once the server answers." };
    if (!inspector.ready()) return { reason: inspector.state().error ? "This model's voices didn't load." : 'Loading voices…' };
    if (editor.error() === 'over' && ctx.mode === 'generate') return { reason: '', limit: true };
    return null;
  }

  /** Why there is no voice setup to save yet (the same words as under Generate). */
  function saveBlockedReason() {
    const block = generateBlock();
    return (block && block.reason) || (inspector.state().error ? "This model's voices didn't load." : 'Loading voices…');
  }

  /**
   * "Save preset…" (both) and "Save named voice…" are aria-disabled, described by the Generate hint, while
   * the inspector has no setup to save; a click then says why instead of doing nothing.
   */
  function updateSaves() {
    const blocked = !inspector.ready();
    for (const button of [el.savePreset, el.saveVoice]) {
      if (!button) continue;
      button.setAttribute('aria-disabled', String(blocked));
      if (blocked) button.setAttribute('aria-describedby', 'sp-gen-hint');
      else button.removeAttribute('aria-describedby');
    }
    row.setSaveBlocked(blocked, 'sp-gen-hint');
  }

  function updateGenerate() {
    updateSaves();
    if (ctx.gen) {
      el.generate.setAttribute('aria-disabled', 'false');
      el.genHint.hidden = true;
      el.generate.removeAttribute('aria-describedby');
      return;
    }
    const block = generateBlock();
    el.generate.setAttribute('aria-disabled', String(!!block));
    const hint = block && block.reason ? block.reason : '';
    el.genHint.textContent = hint;
    el.genHint.hidden = !hint;
    el.generate.setAttribute('aria-describedby', block && block.limit ? 'sp-limit-hint' : 'sp-gen-hint');
  }

  function renderEmpty() {
    const st = app.get();
    const none = st.models.fetchedAt > 0 && !models.ttsModels(st.models.list).length;
    el.empty.hidden = !none;
    el.inspector.hidden = none;
  }

  /* ── the Generate path ── */

  function setPhase(gen, phase, info = {}) {
    if (gen) gen.phase = phase;
    if (ctx.lastGen !== gen && gen) return;
    el.statusText.textContent = phaseText(phase, info);
    el.status.setAttribute('data-phase', phase || '');
    if (el.statusExtra) el.statusExtra.textContent = '';
    if (phase !== 'loading') {
      el.elapsed.textContent = '';
      if (ctx.elapsedTimer) {
        clearInterval(ctx.elapsedTimer);
        ctx.elapsedTimer = null;
      }
    }
  }

  /**
   * The real-time factor arrives after "Done" was announced (the waveform decode measures it): it goes
   * into an aria-hidden part of the status line, so the status region says "Done" once per take; the take
   * card carries the RTF for assistive technology.
   */
  function setDoneRtf(gen, rtf) {
    if (ctx.lastGen !== gen || gen.phase !== 'done' || !el.statusExtra) return;
    el.statusExtra.textContent = Number.isFinite(rtf) ? ` · ${rtfText(rtf)}` : '';
  }

  /** Mod+Enter as aria-keyshortcuts names it (Cmd on Apple, Ctrl elsewhere). */
  function generateKeys() {
    return modLabel() === 'Ctrl' ? 'Control+Enter' : 'Meta+Enter';
  }

  function setGenerating(on) {
    const label = el.generate.querySelector('.btn__label');
    if (label) label.textContent = on ? 'Stop' : 'Generate';
    const glyph = el.generate.querySelector('.sp-generate__icon use');
    if (glyph) glyph.setAttribute('href', `/static/icons/sprite.svg#i-${on ? 'stop' : 'play'}`);
    el.generate.classList.toggle('is-generating', on);
    el.generate.setAttribute('aria-keyshortcuts', on ? 'Escape' : generateKeys());
    root.setAttribute('data-generating', String(on));
    updateGenerate();
  }

  function clearGenError() {
    // Retry inside the banner starts the next run: the focus goes to Generate (now Stop), never to <body>.
    if (typeof document !== 'undefined' && el.genError.contains(document.activeElement)) el.generate.focus({ preventScroll: true });
    el.genError.hidden = true;
    render(el.genError, html``);
  }

  function showGenError(error) {
    const banner = errorBanner(error, 'speak', { retry: () => { generate(); } });
    if (error && error.kind !== 'auth' && !banner.actions.length) banner.actions.push({ label: 'Retry', run: () => { generate(); } });
    renderBanner(el.genError, { ...banner, live: false, onDismiss: () => { clearGenError(); el.generate.focus(); } });
  }

  function snapshot(text) {
    const s = inspector.state();
    const setup = inspector.currentSetup();
    const caps = { ...inspector.caps() };
    const req = buildSpeechRequest({
      model: s.model,
      voice: inspector.voiceSpec(),
      input: text,
      speed: s.speed,
      format: s.format,
      effects: setup.effects,
      instructions: caps.instructions ? s.instructions : '',
      reference: caps.voice_clone ? s.reference : '',
      options: setup.options,
      caps,
      // Nothing plays while a recording runs: the take is then fetched whole, to play later.
      stream: s.stream && !app.get().recording,
      mse: mseSupported(),
    });
    const row = app.get().models.list.find((m) => m.id === s.model);
    // A take designed from the description can become a designed voice: what it was designed with.
    const design = inspector.designing()
      ? { description: String(s.instructions || '').trim(), model: s.model, options: { ...setup.options } } : null;
    return {
      req, setup, design, text: req.body.input, model: s.model, format: s.format, speed: req.body.speed,
      voiceLabel: inspector.voiceLabel(), modelLabel: row ? models.shortName(row) : models.modelLabel(s.model),
      device: row ? models.deviceLabel(row.device) : '',
    };
  }

  /**
   * The synchronous part of Generate: checks, audio.prime() inside the click, the settings snapshot.
   * @returns {boolean} whether a generation started
   */
  function generate() {
    if (ctx.gen) return false;
    if (generateBlock()) {
      if (editor.error() === 'over') el.input.focus();
      return false;
    }
    const text = editor.value();
    const problem = checkInput(text, inspector.limit());
    if (problem) {
      editor.showError(problem);
      el.input.focus();
      return false;
    }
    const s = inspector.state();
    const caps = inspector.caps();
    if (inspector.designing()) {
      if (!String(s.instructions || '').trim()) {
        showNotice({
          tone: 'error', title: 'Describe the voice first',
          message: caps.voice_clone ? 'Write a voice description or pick a reference recording, then generate.' : 'Write a voice description, then generate.',
        });
        inspector.focusDescription();
        return false;
      }
    } else if (caps.clone_transcript_required && !s.reference && !s.identity) {
      inspector.referenceError('Choose a reference recording for this model');
      return false;
    } else if (!inspector.voiceSpec()) {
      inspector.setOpen(true);
      showNotice({ tone: 'error', title: 'Choose a voice first', message: 'Pick a voice or a reference recording, then generate.' });
      el.voice.focus();
      return false;
    }
    // iOS: the result plays after an await. A take that will only download (PCM, a format this
    // browser can't play) plays nothing, so the clip playing now is left alone.
    if (playbackPlan(s.format, (mime) => audio.canPlay(mime)) === 'play') audio.prime();
    const snap = snapshot(text);         // nothing the listener changes from here on affects this take
    run(snap).catch((error) => { if (globalThis.console) globalThis.console.error(error); });
    return true;
  }

  function watchLoadJob(gen, model) {
    const key = `model:${model}`;
    const show = (jobs) => {
      const job = (jobs || []).find((j) => j.key === key);
      if (!job || gen.cancelled || ctx.lastGen !== gen) return;
      if (!gen.loadStartedAt) gen.loadStartedAt = job.startedAt || Date.now();
      setPhase(gen, 'loading', { job: job.label });
      const tick = () => { el.elapsed.textContent = ` ${Math.max(0, Math.round((Date.now() - gen.loadStartedAt) / 1000))} s`; };
      tick();
      if (!ctx.elapsedTimer) ctx.elapsedTimer = setInterval(tick, 1000);
    };
    show(app.get().jobs);
    return app.select((st) => st.jobs, show);
  }

  function newTake(snap, { cached }) {
    takeSeq += 1;
    // The device that rendered the take: a model that loaded for it had no device yet at the click.
    const row = app.get().models.list.find((m) => m.id === snap.model);
    const device = (row && models.deviceLabel(row.device)) || snap.device;
    return {
      n: takeSeq, id: `take:${takeSeq}`, text: snap.text, setup: snap.setup, req: snap.req, model: snap.model, design: snap.design || null,
      format: snap.format, voiceLabel: snap.voiceLabel, modelLabel: snap.modelLabel, speed: snap.speed,
      chars: codePointLength(snap.text), device, at: Date.now(), cached, plan: 'play', state: 'pending',
      blob: null, url: '', src: '', streamSrc: '', peaks: null, durationS: 0, ttfaMs: null, renderMs: null, rtf: null, hint: '',
      position: 0,
    };
  }

  /** Bring a new take into view when it landed below the fold (phones), with no jump when it is visible. */
  function revealTake(take) {
    const item = el.takes.querySelector(`[data-take="${take.n}"]`);
    if (!item || typeof item.getBoundingClientRect !== 'function' || typeof item.scrollIntoView !== 'function') return;
    const rect = item.getBoundingClientRect();
    let chrome = 0;
    try { chrome = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--chrome-bottom')) || 0; } catch { chrome = 0; }
    const height = (globalThis.innerHeight || 0) - chrome;
    if (height <= 0 || (rect.top >= 0 && rect.bottom <= height)) return;
    item.scrollIntoView({ block: 'nearest', behavior: theme.reducedMotion() ? 'auto' : 'smooth' });
  }

  function playFailed(error, take) {
    take.hint = error && error.name === 'CaptureActiveError' ? 'Stop recording to listen.' : "Couldn't play this take.";
    takes.paint(take);
  }

  /** The take won't reach its first audio on its own: a later tap on Play is not "time to first audio". */
  function stopWatchingFirstAudio(take) {
    if (!Number.isFinite(take.ttfaMs) && take.offPlaying) take.offPlaying();
  }

  /**
   * Start a take on the shared element. `onRelease` runs when the element lets go of it (another clip took
   * it, it was closed, it ended) or when it could not start at all (a recording runs).
   * @returns {Promise<boolean>} whether the take is on the element (playing, or paused by a refused autoplay)
   */
  function startPlay(take, src, { onRelease = null } = {}) {
    return takes.play(take, {
      src,
      onRelease: (reason) => {
        stopWatchingFirstAudio(take);
        if (onRelease) onRelease(reason);
      },
    }).then(() => {
      const elm = audio.element();
      const cur = audio.current();
      // Autoplay was refused (the first play will be a tap) or the take was displaced already: neither says
      // anything about the time to first audio.
      if (!cur || cur.id !== take.id || elm.paused) stopWatchingFirstAudio(take);
      return true;
    }, (error) => {
      stopWatchingFirstAudio(take);
      if (onRelease) onRelease('error');
      playFailed(error, take);
      return false;
    });
  }

  /** Time to first audio: the shared element's first `playing` for this take. */
  function watchPlaying(take, t0, onFirst) {
    const elm = audio.element();
    const handler = () => {
      const cur = audio.current();
      if (!cur || cur.id !== take.id) return;
      off();
      onFirst(performance.now() - t0);
    };
    const off = () => {
      elm.removeEventListener('playing', handler);
      if (take.offPlaying === off) take.offPlaying = null;
    };
    elm.addEventListener('playing', handler);
    take.offPlaying = off;
    return off;
  }

  async function run(snap) {
    const controller = new AbortController();
    const gen = {
      controller, snap, phase: '', take: null, cancelled: false, loadStartedAt: 0, streaming: false, startedAt: performance.now(),
    };
    if (ctx.lastGen && ctx.lastGen.take && ctx.lastGen.take.offPlaying) ctx.lastGen.take.offPlaying();
    ctx.gen = gen;
    ctx.lastGen = gen;
    clearGenError();
    setGenerating(true);
    const startedAt = performance.now();
    let stopJob = null;
    try {
      // At most one request before the POST: the status check in ensureReady; none on the fast path.
      let settled = false;
      const ready = models.ensureReady(snap.model, { kind: 'tts' });
      ready.then(() => { settled = true; }, () => { settled = true; });
      await null;
      await null;
      if (!settled) {
        setPhase(gen, 'checking');
        stopJob = watchLoadJob(gen, snap.model);
      }
      const ok = await ready;
      if (stopJob) stopJob();
      stopJob = null;
      if (gen.cancelled) {
        setPhase(gen, 'stopped');
        return;
      }
      if (!ok) {
        setPhase(gen, '');   // the listener cancelled the model switch: back to idle, silently
        return;
      }
      const loadMs = gen.loadStartedAt ? performance.now() - startedAt : 0;
      inspector.modelsChanged();
      setPhase(gen, 'generating');
      const t0 = performance.now();
      const res = await request(snap.req.url, {
        method: 'POST', json: snap.req.body, history: true, signal: controller.signal, accept: 'response',
      });
      // The server logged this take (X-History) before it answered, streamed or not: History reloads on its
      // next show. A synchronous flag in History; nothing waits for it.
      emitEvent('history:changed');
      const cached = /hit/i.test((res.headers && res.headers.get('x-cache')) || '');
      const take = newTake(snap, { cached });
      take.plan = playbackPlan(snap.format, (mime) => audio.canPlay(mime));
      gen.take = take;
      takes.add(take);
      revealTake(take);
      let blobReady = false;
      let playStarted = take.plan !== 'play';
      let decoding = false;
      const finish = () => {
        if (!blobReady || !playStarted || decoding) return;
        decoding = true;
        takes.decode(take).then(() => {
          if (take.removed) return;
          if (Number.isFinite(take.renderMs) && take.durationS > 0) take.rtf = take.renderMs / 1000 / take.durationS;
          takes.paint(take);
          if (!cached && take.durationS > 0) perf.record(snap.model, { ttfaMs: take.ttfaMs ?? undefined, renderMs: take.renderMs, durationS: take.durationS });
          if (!ctx.gen) setDoneRtf(gen, take.rtf);
        });
      };
      if (take.plan === 'play') {
        watchPlaying(take, t0, (ms) => {
          take.ttfaMs = ms;
          takes.paint(take);
          if (gen.streaming && ctx.gen === gen) setPhase(gen, 'streaming', { ttfaMs: ms });
          playStarted = true;
          finish();
        });
      }
      const stream = snap.req.stream && take.plan === 'play';
      const blob = await receive(res, {
        stream,
        type: 'audio/mpeg',
        signal: controller.signal,
        onSource: (url, source) => {
          gen.streaming = true;
          take.state = 'streaming';
          take.streamSrc = url;
          takes.paint(take);
          // When the element lets go of the stream (another clip, a closed player) or never takes it (a
          // recording runs), the rest is still read: the take keeps every byte and plays from its button.
          startPlay(take, url, { onRelease: () => source.release() }).then(() => { if (!take.ttfaMs) { playStarted = true; finish(); } });
        },
      });
      take.renderMs = performance.now() - t0;
      take.state = 'done';
      takes.setAudio(take, blob);
      models.markUsed(snap.model);
      if (take.plan === 'play') {
        if (!gen.streaming) startPlay(take, take.src).then(() => { if (!take.ttfaMs) { playStarted = true; finish(); } });
      } else {
        downloadBlob(blob, takeFilename(snap.format, take.at));
        take.hint = downloadHint(snap.format);
        takes.paint(take);
      }
      blobReady = true;
      setPhase(gen, 'done', { cached, loadMs });
      finish();
    } catch (error) {
      if (stopJob) stopJob();
      const aborted = gen.cancelled || (error && (error.kind === 'aborted' || error.name === 'AbortError'));
      if (gen.take && !gen.take.blob) takes.remove(gen.take.n);
      if (aborted) {
        setPhase(gen, 'stopped');
        return;
      }
      models.markFailed(snap.model);
      setPhase(gen, '');
      showGenError(error);
    } finally {
      if (ctx.gen === gen) {
        ctx.gen = null;
        setGenerating(false);
      }
    }
  }

  function stopGeneration() {
    const gen = ctx.gen;
    if (!gen) return;
    gen.cancelled = true;
    gen.controller.abort();
    const cur = audio.current();
    if (gen.take && cur && cur.id === gen.take.id) audio.closeCurrent();
    if (gen.phase === 'checking' || gen.phase === 'loading') {
      // The model keeps loading in the background (the status chip shows it); the take is cancelled.
      ctx.gen = null;
      setPhase(gen, 'stopped');
      setGenerating(false);
    }
  }

  /* ── presets, named voices, intents ── */

  const ttsIds = () => models.ttsModels(app.get().models.list).map((m) => m.id);

  /** @returns {Promise<object|false>} the applied preset (resolved), or false */
  async function applyPresetById(id, name = '', { isDefault = false, quiet = false } = {}) {
    const result = await presets.applyPreset(id, { inspector, ttsIds: ttsIds(), stopLive, name });
    if (result.stale) return false;
    if (!result.ok) {
      const label = result.name || name || (row.find(id) || {}).name || 'this preset';
      const reason = String(result.reason || '').replace(/\.$/, '');
      showNotice({
        tone: 'error',
        title: isDefault ? "The default preset wasn't applied" : `Preset “${label}” can't be used`,
        message: reason ? `${reason.charAt(0).toUpperCase()}${reason.slice(1)}.` : '',
        // The server's message and code, never its JSON envelope; none when it only repeats the reason.
        detail: result.error ? explain(result.error).detailText : '',
        // The Default preset applies on page load, not on a click: polite, never an alert.
        polite: isDefault,
      });
      return false;
    }
    row.setApplied(id);
    // A preset is a whole setup: an edit in progress is over (Edit on a preset starts its own after this).
    endEditing();
    ctx.newPreset = null;
    if (!isDefault && !quiet) presets.announceApplied(result.profile.name || name);
    return result.profile || true;
  }

  async function maybeApplyDefault() {
    const flags = presets.sessionFlags();
    if (flags.defaultTried || flags.picked || !row.loaded() || !row.defaultId() || row.applied()) return;
    flags.defaultTried = true;
    await applyPresetById(row.defaultId(), '', { isDefault: true });
  }

  /** Speak's setup as a preset payload takes it (Save preset…, Update preset). */
  function presetSetupNow() {
    const s = inspector.state();
    const setup = inspector.currentSetup();
    return {
      model: s.model, voice: s.voice, blend: s.mix.length >= 2 ? setup.voice : '', identity: s.identity,
      reference: s.reference, speed: s.speed, format: s.format, effects: setup.effects, instructions: s.instructions,
      options: setup.options,
    };
  }

  /** @returns {Promise<object|null>} the new preset, or null (cancelled, failed or blocked) */
  async function savePresetFlow() {
    if (!inspector.ready()) return null;
    const s = inspector.state();
    // After New preset on a voice's card, the name starts as the voice's while it is chosen.
    const fresh = ctx.newPreset && s.identity && s.identity === ctx.newPreset.id ? ctx.newPreset : null;
    try {
      const created = await presets.savePreset({
        setup: presetSetupNow(),
        value: fresh ? fresh.name : '',
        onSaved: (made) => {
          row.load().then(() => { if (made && made.id) row.setApplied(made.id); });
        },
      });
      if (created && fresh) {
        ctx.newPreset = null;
        showNotice(null);   // the "New preset with …" hint has done its job
      }
      return created;
    } catch (error) {
      showNotice(errorBanner(error, "Couldn't save the preset"));
      return null;
    }
  }

  /**
   * After a designed voice is saved: Speak speaks with it when it is still designing on the take's model and the
   * new voice has a version for that model. On a model that also clones (Breeze), the field held the voice
   * description, which would steer the cloned voice as instructions: it is cleared.
   */
  async function useSavedDesign(identity, model) {
    if (inspector.state().model !== model || !inspector.designing() || !realizationFor(identity, model)) return;
    const caps = inspector.caps();
    const clearDescription = caps.voice_clone === true && !!caps.instructions && !!String(inspector.state().instructions || '').trim();
    const named = await listNamed({ model });
    const p = await inspector.prepared(model);
    p.data.named = named;
    const setup = { ...inspector.currentSetup(), identity: identity.id };
    if (clearDescription) setup.instructions = '';
    const result = inspector.commitPrepared(p, setup);
    if (result.stale || inspector.state().identity !== identity.id) return;
    announce(`Voice: ${identity.name}${clearDescription ? '. The voice description was cleared.' : ''}`);
  }

  /** Save as voice… on a designed take: its audio as the sample, its words, description, model and options. */
  async function saveTakeAsVoice(take) {
    if (!take || !take.design || take.removed) return null;
    if (!take.blob) {
      showNotice({ tone: 'info', title: "The take isn't ready yet", message: 'Wait for it to finish, then save it as a voice.' });
      return null;
    }
    const problem = sampleProblem(take);
    if (problem) {
      showNotice({ tone: 'error', title: "This take can't be saved as a voice", message: `${problem}.` });
      return null;
    }
    let sample;
    try {
      sample = await sampleWav(take);
    } catch {
      showNotice({ tone: 'error', title: "Couldn't save the voice", message: "This browser can't decode this take. Generate it as WAV, then save it." });
      return null;
    }
    if (sample.durationS > MAX_SAMPLE_S) {
      take.durationS = sample.durationS;
      takes.paint(take);
      showNotice({ tone: 'error', title: "This take can't be saved as a voice", message: `${SAMPLE_TOO_LONG}.` });
      return null;
    }
    const { description, model, options } = take.design;
    try {
      return await presets.saveDesignedVoice({
        model, description, options, transcript: take.text, audio: sample.blob,
        onSaved: async (identity) => {
          showNotice(null);
          try {
            await useSavedDesign(identity, model);
          } catch { /* the voice is saved; Speak keeps what it has */ }
        },
      });
    } catch (error) {
      showNotice(errorBanner(error, "Couldn't save the voice"));
      return null;
    }
  }

  async function saveVoiceFlow() {
    if (!inspector.ready()) return;
    const s = inspector.state();
    const model = s.model;
    const setup = inspector.currentSetup();
    if (inspector.designing()) {
      const take = takes.all().find((t) => t.design && t.design.model === model);
      if (take) saveTakeAsVoice(take);
      else showNotice({ tone: 'info', title: 'No take to save yet', message: 'Generate a take first, then save it as a voice.' });
      return;
    }
    const { recipe, reference } = versionNow();
    if (!recipe && !reference) {
      showNotice({ tone: 'error', title: 'Choose a voice first', message: presets.NO_VOICE });
      return;
    }
    const edited = ctx.editing && ctx.editing.kind === 'voice' ? ctx.editing : null;
    const selected = s.identity ? (s.data.named || []).find((n) => n.id === s.identity) : null;
    try {
      await presets.saveNamedVoice({
        model, recipe, reference, designed: selected && selected.design ? selected.id : '',
        current: s.identityName || (edited && edited.name) || ctx.voiceName || '',
        onSaved: async (identity) => {
          const named = await listNamed({ model });
          const p = await inspector.prepared(model);
          p.data.named = named;
          inspector.commitPrepared(p, { ...inspector.currentSetup(), identity: identity.id });
          if (ctx.editing && ctx.editing.kind === 'voice') endEditing();
          ctx.voiceName = '';
          showNotice(null);
        },
      });
    } catch (error) {
      if (error && error.code === presets.DESIGNED_COPY_CODE) {
        showNotice({ tone: 'info', title: "A designed voice can't be copied here", message: presets.DESIGNED_COPY });
        return;
      }
      showNotice(errorBanner(error, "Couldn't save the named voice"));
    }
  }

  /**
   * The version Speak would save: its recipe and reference, or for a designed voice (which speaks through its
   * sample, not a recording) the selected voice's own version for this model.
   */
  function versionNow() {
    const s = inspector.state();
    let recipe = inspector.currentSetup().voice;
    let reference = s.reference;
    if (!recipe && !reference && s.identity) {
      const version = realizationFor((s.data.named || []).find((n) => n.id === s.identity), s.model);
      recipe = String((version && version.voice) || '');
      reference = String((version && version.reference_audio_id) || '');
    }
    return { recipe, reference };
  }

  /* ── the editing bar ── */

  /** What Speak has now, for Cancel: its setup (when it has one) and the applied preset. */
  function editSnapshot() {
    return { setup: inspector.ready() ? inspector.currentSetup() : null, applied: row.applied() };
  }

  function renderEditing() {
    const e = ctx.editing;
    if (!el.editing) return;
    el.editing.hidden = !e;
    if (!e) return;
    el.editingTitle.textContent = editingTitle(e);
    el.editingHint.textContent = e.kind === 'voice' ? EDIT_VOICE_HINT : EDIT_PRESET_HINT;
    const label = el.editingUpdate.querySelector('.btn__label');
    if (label) label.textContent = e.kind === 'voice' ? 'Update voice' : 'Update preset';
    el.editingNew.hidden = e.kind !== 'preset';
  }

  function startEditing(editing) {
    ctx.editing = editing;
    ctx.voiceName = '';
    renderEditing();
    announce(editingTitle(editing));
  }

  /** Whether the keyboard focus is in the editing bar (it goes to Generate when the bar goes). */
  function focusInBar() {
    return !!(el.editing && typeof document !== 'undefined' && el.editing.contains(document.activeElement));
  }

  /**
   * End the editing as it is (nothing restored, nothing saved). Focus that was in the bar (or `refocus`: it was
   * there when the action began) moves to Generate, or the editor in Live, never to <body>.
   */
  function endEditing({ refocus = focusInBar() } = {}) {
    if (!ctx.editing) return;
    ctx.editing = null;
    renderEditing();
    if (refocus) (el.genbar.hidden ? el.input : el.generate).focus({ preventScroll: true });
  }

  /** The model changed while a version was being edited: the edit is over, and the note says why. */
  function stopEditingForModel() {
    const e = ctx.editing;
    if (!e) return;
    endEditing();
    ctx.voiceName = e.name;
    showNotice({ tone: 'info', title: `Stopped editing ${e.name}`, message: `The model changed. Save named voice… adds a version of ${e.name} for this model.` });
  }

  /** Cancel: Speak goes back to what it had before the edit began, its applied preset included. */
  async function cancelEditing() {
    const e = ctx.editing;
    if (!e) return;
    endEditing({ refocus: focusInBar() });
    showNotice(null);
    announce(`Stopped editing ${e.name}`);
    const before = e.before || {};
    if (!before.setup || !before.setup.model) return;
    try {
      const p = await inspector.prepared(before.setup.model);
      const result = inspector.commitPrepared(p, before.setup);
      if (result.stale) return;
      row.setApplied(before.applied || '');
      warningNotice(result.warnings);
    } catch (error) {
      showNotice(errorBanner(error, "Couldn't restore the earlier setup"));
    }
  }

  /** Update voice: PUT this model's version with Speak's voice, no name prompt; its presets follow it. */
  async function updateVoice() {
    const e = ctx.editing;
    const button = el.editingUpdate;
    if (!e || e.kind !== 'voice' || button.getAttribute('aria-busy') === 'true') return;
    if (!inspector.ready()) {
      announce(saveBlockedReason());
      return;
    }
    const s = inspector.state();
    if (s.model !== e.model) {
      stopEditingForModel();
      return;
    }
    const { recipe, reference } = versionNow();
    if (!recipe && !reference) {
      showNotice({ tone: 'error', title: 'Choose a voice first', message: 'Pick a voice, a blend or a reference recording, then Update voice.' });
      return;
    }
    const refocus = focusInBar();
    setBusy(button, true);
    try {
      await presets.updateNamedVoice({ id: e.id, model: e.model, recipe, reference });
    } catch (error) {
      setBusy(button, false);
      showNotice(errorBanner(error, "Couldn't update the named voice"));
      return;
    }
    setBusy(button, false);
    if (ctx.editing === e) endEditing({ refocus });
    showNotice(null);
    toast(presets.updatedVoiceMessage(e.name, models.modelLabel(e.model), presets.followingCount(row.list(), e.id, e.model)), { kind: 'success' });
    // The named voice stays selected, with its updated version.
    try {
      const named = await listNamed({ model: e.model });
      const p = await inspector.prepared(e.model);
      p.data.named = named;
      if (inspector.state().model === e.model) inspector.commitPrepared(p, { ...inspector.currentSetup(), identity: e.id });
    } catch { /* the version is saved; the selection stays as it is */ }
  }

  /** Update preset: PUT /api/profiles/{id} with Speak's setup under the preset's name; a failure keeps the edit. */
  async function updatePresetFlow() {
    const e = ctx.editing;
    const button = el.editingUpdate;
    if (!e || e.kind !== 'preset' || button.getAttribute('aria-busy') === 'true') return;
    if (!inspector.ready()) {
      announce(saveBlockedReason());
      return;
    }
    const stored = row.find(e.id);
    const name = (stored && stored.name) || e.name;
    const refocus = focusInBar();
    setBusy(button, true);
    try {
      await presets.updatePreset(e.id, name, presetSetupNow());
    } catch (error) {
      setBusy(button, false);
      showNotice(errorBanner(error, "Couldn't update the preset"));
      return;
    }
    setBusy(button, false);
    if (ctx.editing === e) endEditing({ refocus });
    showNotice(null);
    toast(`Updated preset “${name}”`, { kind: 'success' });
    await row.load();
    row.setApplied(e.id);   // the stored preset is Speak's setup now
  }

  /** Save as new…: Save preset… with the edited setup; the edit of the stored preset ends once it is saved. */
  async function saveAsNewFlow() {
    const e = ctx.editing;
    if (!e || e.kind !== 'preset') return;
    if (!inspector.ready()) {
      announce(saveBlockedReason());
      return;
    }
    const created = await savePresetFlow();
    if (created && ctx.editing === e) endEditing();
  }

  function warningNotice(warnings, { missingTitle, missingMessage } = {}) {
    const missing = (warnings || []).find((w) => w && w.kind === 'missing-reference');
    if (missing) {
      showNotice({ tone: 'error', title: missingTitle || 'Reference recording missing', message: missingMessage ? missingMessage(missing.name) : `The reference recording ${missing.name} is missing. Choose one before generating.` });
      return true;
    }
    return false;
  }

  async function applyHistory(entry) {
    const h = presets.historySetup(entry, { ttsIds: ttsIds() });
    if (h.error) {
      showNotice({ tone: 'error', title: "Couldn't open this entry in Speak", message: h.error });
      return;
    }
    stopLive('Text changed');
    editor.set(h.text, 'history');
    const model = h.setup.model || inspector.state().model;
    try {
      const p = await inspector.prepared(model);
      const setup = { ...h.setup, model };
      if (setup.reference === undefined) {
        setup.reference = p.data.caps.voice_clone && !p.data.ids.length && h.referenceFallback ? h.referenceFallback : '';
      }
      if (!p.data.caps.voice_clone) setup.reference = '';
      const result = inspector.commitPrepared(p, setup);
      if (result.stale) return;
      row.setApplied('');
      endEditing();
      if (!warningNotice(result.warnings, { missingMessage: (name) => `The reference recording ${name} this entry used is missing. Choose one before generating.` })) showNotice(null);
      announce('The History entry is in the editor with its settings');
    } catch (error) {
      showNotice(errorBanner(error, "Couldn't load this entry's model"));
    }
  }

  /**
   * Edit in Speak (SE-03; the editing bar): select the model, apply the version and select the
   * named voice, then show "Editing Narrator for Kokoro" with Update voice and Cancel.
   */
  async function editNamedVoice({ id, model }) {
    if (!ttsIds().includes(model)) {
      showNotice({ tone: 'error', title: "Couldn't edit the named voice", message: 'This model is unavailable.' });
      return;
    }
    endEditing();
    const before = editSnapshot();
    stopLive('Voice changed');
    try {
      const p = await inspector.prepared(model);
      let identity = p.data.named.find((n) => n.id === id);
      if (!identity) {
        p.data.named = await listNamed({ model });
        identity = p.data.named.find((n) => n.id === id);
      }
      if (!identity) {
        showNotice({ tone: 'error', title: "Couldn't edit the named voice", message: 'This named voice no longer exists.' });
        return;
      }
      const version = realizationFor(identity, model, { available: false });
      if (!version) {
        showNotice({ tone: 'error', title: "Couldn't edit the named voice", message: `${identity.name} has no version for ${models.modelLabel(model)}.` });
        return;
      }
      const usable = version.available !== false && identity.available !== false;
      const setup = {
        ...inspector.currentSetup(), model, identity: usable ? id : '',
        voice: version.voice, reference: version.reference_audio_id || '',
      };
      const result = inspector.commitPrepared(p, setup);
      if (result.stale) return;
      row.setApplied('');
      inspector.setOpen(true);
      startEditing({ kind: 'voice', id, name: identity.name, model, before });
      if (!warningNotice(result.warnings, { missingTitle: 'Reference recording missing', missingMessage: () => 'Choose a replacement recording, then Update voice.' })) {
        showNotice(null);
      }
    } catch (error) {
      showNotice(errorBanner(error, "Couldn't edit the named voice"));
    }
  }

  /** Edit on a preset: apply it, then show "Editing preset “Podcast”" (nothing when it can't be used). */
  async function editPreset({ id }) {
    endEditing();
    const before = editSnapshot();
    const applied = await applyPresetById(id, '', { quiet: true });
    if (!applied) return;
    const stored = row.find(id);
    inspector.setOpen(true);
    startEditing({ kind: 'preset', id, name: (stored && stored.name) || applied.name || 'this preset', before });
  }

  /**
   * New preset on a voice's card: choose that named voice for the model Speak is on when it has a
   * version there, else for its first model that has one, then focus Save preset…, whose name starts as the
   * voice's. A voice no model can use says so and changes nothing.
   */
  async function newPreset({ id }) {
    endEditing();
    const ids = ttsIds();
    const current = inspector.state().model;
    try {
      let identity = null;
      let model = '';
      if (current && ids.includes(current)) {
        const here = await inspector.prepared(current);
        identity = here.data.named.find((n) => n.id === id) || null;
        if (identity && identity.available !== false && realizationFor(identity, current)) model = current;
      }
      if (!model) {
        const catalog = await listNamed({ check: false });
        identity = catalog.find((n) => n.id === id) || identity;
        model = presets.newPresetModel(identity, ids, current);
      }
      if (!identity) {
        showNotice({ tone: 'error', title: "Couldn't start a preset", message: 'This named voice no longer exists.' });
        return;
      }
      const name = String(identity.name || '');
      const cannot = (message) => showNotice({ tone: 'error', title: `Couldn't start a preset with ${name}`, message });
      if (!model) {
        cannot(`${name} has no version for a model that is available.`);
        return;
      }
      const p = await inspector.prepared(model);
      const offered = p.data.named.find((n) => n.id === id);
      const version = offered ? realizationFor(offered, model, { available: false }) : null;
      if (!offered || !version || offered.available === false || version.available === false) {
        const reason = (version && version.reason) || `${name} isn't available for ${models.modelLabel(model)}`;
        cannot(`${String(reason).replace(/\.$/, '')}.`);
        return;
      }
      stopLive('Voice changed');
      const result = inspector.commitPrepared(p, { ...inspector.currentSetup(), model, identity: id });
      if (result.stale) return;
      row.setApplied('');
      ctx.newPreset = { id, name };
      inspector.setOpen(true);
      if (!warningNotice(result.warnings)) {
        showNotice({ tone: 'info', title: `New preset with ${name}`, message: 'Set its speed, format and effects, then Save preset….' });
      }
      // Save preset… takes the focus unless the listener has already moved on (only the view heading or
      // nothing has it when the hand-off lands).
      const active = typeof document !== 'undefined' ? document.activeElement : null;
      if (!active || active === document.body || active === el.title) el.savePreset.focus();
    } catch (error) {
      showNotice(errorBanner(error, "Couldn't start a preset"));
    }
  }

  /**
   * Use in Speak on a named voice's card: its version for `model` (else for the first available model it has one
   * for), with the rest of Speak's setup kept.
   */
  async function useNamedVoice({ id, model: wanted = '' }) {
    endEditing();
    const ids = ttsIds();
    try {
      const catalog = await listNamed({ check: false });
      const identity = catalog.find((n) => n.id === id) || null;
      if (!identity) {
        showNotice({ tone: 'error', title: "Couldn't use the voice", message: 'This named voice no longer exists.' });
        return;
      }
      const name = String(identity.name || '');
      const cannot = (message) => showNotice({ tone: 'error', title: `Couldn't use ${name}`, message });
      const model = wanted || presets.newPresetModel(identity, ids, inspector.state().model);
      if (!model) {
        cannot(`${name} has no version for a model that is available.`);
        return;
      }
      if (!ids.includes(model)) {
        cannot(`${models.modelLabel(model)} is unavailable.`);
        return;
      }
      const p = await inspector.prepared(model);
      const offered = p.data.named.find((n) => n.id === id);
      const version = offered ? realizationFor(offered, model, { available: false }) : null;
      if (!version) {
        cannot(`${name} has no version for ${models.modelLabel(model)}.`);
        return;
      }
      if (offered.available === false || version.available === false) {
        const reason = version.reason || `${name} isn't available for ${models.modelLabel(model)}`;
        cannot(`${String(reason).replace(/\.$/, '')}.`);
        return;
      }
      // The field was the voice description: on a model that clones with instructions it would steer the voice.
      const caps = p.data.caps || {};
      const clearDescription = inspector.designing() && caps.voice_clone === true && !!caps.instructions
        && !!String(inspector.state().instructions || '').trim();
      const setup = { ...inspector.currentSetup(), model, identity: id };
      if (clearDescription) setup.instructions = '';
      stopLive('Voice changed');
      const result = inspector.commitPrepared(p, setup);
      if (result.stale) return;
      row.setApplied('');
      if (!warningNotice(result.warnings)) showNotice(null);
      announce(`Voice: ${name}${clearDescription ? '. The voice description was cleared.' : ''}`);
      const active = typeof document !== 'undefined' ? document.activeElement : null;
      if (!active || active === document.body || active === el.title) el.input.focus();
    } catch (error) {
      showNotice(errorBanner(error, "Couldn't use the voice"));
    }
  }

  async function useRecording({ name, model }) {
    if (!ttsIds().includes(model)) {
      showNotice({ tone: 'error', title: "Couldn't use the recording", message: 'This model is unavailable.' });
      return;
    }
    stopLive('Voice changed');
    try {
      const p = await inspector.prepared(model);
      const result = inspector.commitPrepared(p, { ...inspector.currentSetup(), model, identity: '', reference: name });
      if (result.stale) return;
      row.setApplied('');
      endEditing();
      if (!warningNotice(result.warnings, { missingMessage: (n) => `The recording ${n} is missing. Choose another one.` })) showNotice(null);
      el.input.focus();
    } catch (error) {
      showNotice(errorBanner(error, "Couldn't use the recording"));
    }
  }

  async function useTake(take) {
    stopLive('Voice changed');
    try {
      const p = await inspector.prepared(take.setup.model);
      const result = inspector.commitPrepared(p, { ...take.setup });
      if (result.stale) return;
      row.setApplied('');
      warningNotice(result.warnings);
      announce(`Settings from take ${take.n} applied`);
    } catch (error) {
      showNotice(errorBanner(error, "Couldn't apply the take's settings"));
    }
  }

  async function handleIntent(intent) {
    await ctx.setupDone;
    if (!intent) return;
    ctx.newPreset = null;   // a new hand-off: an earlier New preset's or edit's name no longer applies
    ctx.voiceName = '';
    if (intent.type === 'apply-history') await applyHistory(intent.entry);
    else if (intent.type === 'apply-preset') await applyPresetById(intent.id);
    else if (intent.type === 'edit-named-voice') await editNamedVoice(intent);
    else if (intent.type === 'edit-preset') await editPreset(intent);
    else if (intent.type === 'new-preset') await newPreset(intent);
    else if (intent.type === 'use-named-voice') await useNamedVoice(intent);
    else if (intent.type === 'use-recording') await useRecording(intent);
  }

  /* ── the first setup: the last setup, then the Default preset once per session (SP-34) ── */

  async function initialSetup() {
    ctx.setupStarted = true;
    try {
      const st = app.get();
      const list = models.ttsModels(st.models.list);
      const saved = savedSetup(readJson(KEYS.speak, null));
      const model = models.pickTtsModel(list, { preferred: saved && saved.model, configuredDefault: st.models.defaults.tts });
      const loads = [row.load()];
      if (model) loads.push(inspector.selectModel(model, restoreFor(saved, model)));
      await Promise.all(loads);
      ctx.presetsStale = false;
      if (model) await maybeApplyDefault();
    } finally {
      updateSpeakVoice();
      updateGenerate();
      ctx.setupResolve();
    }
  }

  function maybeStartSetup() {
    if (ctx.setupStarted || !ctx.shown || !app.get().models.fetchedAt) return;
    initialSetup().catch((error) => { if (globalThis.console) globalThis.console.error(error); });
  }

  /* ── copy as code ── */

  async function copyCode(kind, req = null) {
    const spec = CODE[kind];
    if (!spec) return;
    const request_ = req || snapshot(editor.value().trim() || SAMPLE_TEXT).req;
    const code = spec.fn(request_, { origin: globalThis.location ? location.origin : '', hasKey: auth.hasKey() });
    const ok = await copyText(code);
    toast(ok ? `Copied the request as ${spec.label}` : "Couldn't copy to the clipboard", { kind: ok ? 'success' : 'error' });
  }

  /* ── mode ── */

  /**
   * Show Generate or Live. `switched`: the listener changed the mode while Speak was showing (the mode
   * control, Back between the two); only then does Generate stop a running Live session, because the editor
   * is shared and Live would otherwise go on reacting to edits made for a take (REV-31). Coming back to
   * Speak from another view never stops it (show() keeps Live mode then).
   */
  function setMode(mode, { switched = true } = {}) {
    const next = mode === 'live' && liveEnabled() ? 'live' : 'generate';
    const changed = next !== ctx.mode;
    ctx.mode = next;
    root.setAttribute('data-mode', next);
    for (const radio of el.mode.querySelectorAll('input[name="sp-mode"]')) radio.checked = radio.value === next;
    el.genbar.hidden = next !== 'generate';
    el.takesSection.hidden = next !== 'generate';
    el.liveSlot.hidden = next !== 'live';
    if (next === 'generate' && changed && switched) stopLive('Switched to Generate');
    if (next === 'live') ensureLive();
    editor.refresh();
    updateGenerate();
  }

  /* ── screenshot states ── */

  // A screenshot state is display only. A link (or a reload, or Back onto the address) must never write to
  // the server: no speech request, no History row, no model load. The page's own requests carry the stored
  // key, so the server's cross-site guard could not tell them from the listener's.

  /**
   * The `take` state: a finished take of the sample text with the current setup, its audio made here (a WAV
   * of shotSamples()), timed like a typical Kokoro take. Nothing autoplays, nothing is recorded.
   */
  function showShotTake() {
    if (!inspector.ready() || takes.all().some((t) => t.shot)) return;
    const snap = snapshot(SAMPLE_TEXT);
    const samples = shotSamples();
    const take = newTake({ ...snap, format: 'wav' }, { cached: false });
    const durationS = samples.length / SHOT_RATE;
    Object.assign(take, {
      shot: true, plan: 'play', state: 'done', peaks: wavePeaks(samples), durationS,
      ttfaMs: 140, renderMs: Math.round(durationS * 40), rtf: 0.04,
    });
    takes.add(take);
    takes.setAudio(take, encodeWav(samples, SHOT_RATE));
    takes.redraw(take);
    revealTake(take);
    const gen = { phase: '', take, cancelled: false, snap };
    ctx.lastGen = gen;
    setPhase(gen, 'done');
    setDoneRtf(gen, take.rtf);
  }

  async function applyShotState(shot) {
    await ctx.setupDone;
    ctx.persistOff = true;
    if (shot === 'voice-picker') {
      inspector.setOpen(true);
      inspector.openPicker(el.voice).catch(() => {});
    } else if (shot === 'blend') {
      inspector.showBlend(BLEND_SAMPLE);
      const blendEl = q('sp-blend');
      if (blendEl && typeof blendEl.scrollIntoView === 'function' && !globalThis.matchMedia?.('(min-width: 960px)').matches) {
        blendEl.scrollIntoView({ block: 'center' });
      }
    } else if (shot === 'take') {
      editor.set(SAMPLE_TEXT, 'shot');
      showShotTake();
    } else if (shot === 'limit') {
      editor.set(`${SAMPLE_TEXT} `.repeat(Math.ceil((inspector.limit() + 40) / (SAMPLE_TEXT.length + 1))), 'shot');
    } else if (shot === 'editing-voice') {
      // The editing bars: the first named voice with a version for this model, or the default
      // (else the first) preset. Editing saves nothing until Update, so the state writes nothing.
      const s = inspector.state();
      const identity = ((s.data && s.data.named) || []).find((n) => n.available !== false && realizationFor(n, s.model));
      if (identity) await editNamedVoice({ id: identity.id, model: s.model });
    } else if (shot === 'editing-preset') {
      const id = row.defaultId() || (row.list()[0] || {}).id;
      if (id) await editPreset({ id });
    }
  }

  /* ── wiring ── */

  el.generate.addEventListener('click', (event) => {
    if (!ctx.gen) {
      generate();
      return;
    }
    // The second click of a double click (or a second tap right after the first) is not a Stop: the take
    // would be lost. Esc, the keyboard and a later click stop it.
    const pointer = Number(event && event.detail) > 0;
    if (Number(event && event.detail) > 1 || (pointer && performance.now() - ctx.gen.startedAt < STOP_GUARD_MS)) return;
    stopGeneration();
  });
  if (el.kbdMod) el.kbdMod.textContent = modLabel();
  el.generate.setAttribute('aria-keyshortcuts', generateKeys());
  el.mode.addEventListener('change', (event) => {
    const value = event.target && event.target.value;
    if (value !== 'live' && value !== 'generate') return;
    router.navigate(value === 'live' ? 'speak/live' : 'speak', { focus: 'keep' });
  });
  el.saveVoice.addEventListener('click', () => {
    if (!inspector.ready()) announce(saveBlockedReason());
    else saveVoiceFlow();
  });
  el.savePreset.addEventListener('click', () => {
    if (!inspector.ready()) announce(saveBlockedReason());
    else savePresetFlow();
  });
  el.editingUpdate.addEventListener('click', () => {
    if (ctx.editing && ctx.editing.kind === 'voice') updateVoice();
    else updatePresetFlow();
  });
  el.editingNew.addEventListener('click', () => { saveAsNewFlow(); });
  el.editingCancel.addEventListener('click', () => { cancelEditing(); });
  const moreMenu = bindMenuButton(el.more, el.moreMenu, {
    placement: 'bottom-end',
    onSelect: (item) => {
      const act = item.getAttribute('data-more');
      if (act === 'open') el.upload.click();
      else if (CODE[act]) copyCode(act);
      else if (act === 'clear') {
        stopLive('Text changed');
        editor.set('', 'clear');
        editor.showError('');
        el.input.focus();
      }
    },
  });
  offs.push(() => moreMenu.destroy());

  offs.push(register({
    id: 'speak-generate', keys: ['Mod+Enter'], scope: 'view:speak', allowInEditable: true, description: 'Generate speech',
    when: () => ctx.mode === 'generate' && !ctx.gen,
    handler: () => { generate(); },
  }));
  offs.push(register({
    id: 'speak-stop', keys: ['Escape'], scope: 'view:speak', allowInEditable: true, description: 'Stop generating',
    when: () => ctx.mode === 'generate' && !!ctx.gen,
    handler: () => { stopGeneration(); },
  }));

  offs.push(app.select((st) => st.models, () => {
    inspector.modelsChanged();
    renderEmpty();
    updateGenerate();
    maybeStartSetup();
    const s = inspector.state();
    const ids = ttsIds();
    if (ctx.setupStarted && ids.length && (!s.model || !ids.includes(s.model)) && !inspector.loading()) {
      // No model yet (a worker came up) or the selected one went away: pick as on first load (SP-02).
      const next = models.pickTtsModel(models.ttsModels(app.get().models.list), { configuredDefault: app.get().models.defaults.tts });
      if (next) inspector.selectModel(next);
    }
  }));
  offs.push(app.select((st) => st.ops, () => inspector.modelsChanged()));
  offs.push(app.select((st) => st.connection.state, () => updateGenerate()));
  offs.push(app.select((st) => st.player, (player) => takes.sync(player)));
  offs.push(onEvent('voices:changed', (detail) => {
    const what = detail && detail.what;
    if (!what || what === 'presets') {
      ctx.presetsStale = true;
      if (ctx.shown) row.load().then(() => { ctx.presetsStale = false; });
    }
    if (!what || what === 'named' || what === 'recordings') inspector.refreshAssets(what || '');
  }));
  offs.push(theme.onThemeChange(() => takes.redrawAll()));
  const onResize = debounce(() => takes.redrawAll(), 150);
  if (typeof window !== 'undefined' && window.addEventListener) {
    window.addEventListener('resize', onResize);
    offs.push(() => window.removeEventListener('resize', onResize));
  }
  const onPagehide = () => { persist.flush(); editor.flushDraft(); };
  if (typeof window !== 'undefined' && window.addEventListener) {
    window.addEventListener('pagehide', onPagehide);
    offs.push(() => window.removeEventListener('pagehide', onPagehide));
  }

  // Settings › Developer shows or hides Speak › Live, here at once when it changes.
  offs.push(app.select((st) => !!(st.prefs && st.prefs.showLive), () => syncLiveSwitch()));

  /**
   * Ask AI: the editor's text as a `text` input, and Speak's voice speaks the answer
   * (the sheet starts with "Speak it in a voice" on). The button shows only while the assistant is on.
   */
  function syncAsk() {
    if (el.ask) el.ask.disabled = !editor.value().trim();
  }
  const ask = bindAskButton(el.ask, {
    onClick: (button) => {
      const text = editor.value().trim();
      if (!text) return;
      const source = { id: 'text', label: 'The text in the editor', input: { kind: 'text', text }, text, chars: codePointLength(text) };
      openAskAi({ trigger: button, sources: [source], outputs: { speak: true } }).catch((error) => toast(error && error.message ? error.message : 'Something went wrong.', { kind: 'error' }));
    },
  });
  offs.push(() => ask.destroy());

  editor.restoreDraft();
  editor.refresh();
  syncAsk();
  renderEmpty();
  syncLiveSwitch();
  setMode('generate');

  return {
    show(route, nav = {}) {
      // The address the fallback below puts in place: the router shows it again at once, and this call has it.
      if (ctx.redirecting) return undefined;
      const first = !ctx.shown;
      const wasVisible = ctx.visible;
      ctx.shown = true;
      ctx.visible = true;
      const intent = takeIntent('speak');
      if (intent) presets.markPicked();   // a hand-off is the listener's choice: no Default preset over it
      let mode = route && route.sub === 'live' ? 'live' : 'generate';
      let shot = (route && route.shot) || '';
      if (mode === 'live' && !liveEnabled()) {
        // Speak › Live is off (Settings › Developer): a link, the saved last route or Back that asks
        // for it shows Generate, and the address follows (replaced, no extra history entry). A Live screenshot
        // state has nothing to show in Generate.
        mode = 'generate';
        if (shot.startsWith('live-')) shot = '';
        ctx.redirecting = true;
        try {
          router.navigate('speak', { replace: true, focus: (nav && nav.focus) || 'keep' });
        } finally {
          ctx.redirecting = false;
        }
      } else if (mode === 'generate' && !wasVisible && !intent && !shot && liveActive()) {
        // Back to Speak from another view (the nav, g s) while Live speaks in the background: show Live,
        // where it runs, instead of stopping it; the address follows (replaced, no extra history entry).
        mode = 'live';
        router.navigate('speak/live', { replace: true, focus: (nav && nav.focus) || 'heading' });
      }
      setMode(mode, { switched: wasVisible });
      if (!ctx.setupStarted) {
        maybeStartSetup();
      } else if (nav && nav.reason === 'auth') {
        // A new API key: reload what the server refused, keeping the listener's picks (AU-09).
        inspector.invalidate();
        const s = inspector.state();
        if (s.model && (!inspector.ready() || s.error)) inspector.selectModel(s.model);
        row.load().then(() => maybeApplyDefault());
      } else if (ctx.presetsStale || Date.now() - row.loadedAt() > STALE_MS) {
        row.load().then(() => { ctx.presetsStale = false; });
      }
      if (!first) takes.redrawAll();
      if (intent) handleIntent(intent).catch((error) => { if (globalThis.console) globalThis.console.error(error); });
      if (shot) applyShotState(shot).catch(() => {});
      else ask.refresh();   // is the AI assistant on (the Ask AI button)? A screenshot state asks nothing
      syncAsk();
      return undefined;
    },
    hide() {
      ctx.visible = false;
      persist.flush();
      editor.flushDraft();
    },
    /** Test hooks. */
    __ctx: ctx,
    __parts: { editor, inspector, row, takes },
    destroy() {
      for (const off of offs.splice(0)) off();
      editor.destroy();
      inspector.destroy();
      row.destroy();
      takes.destroy();
      if (ctx.live && typeof ctx.live.destroy === 'function') ctx.live.destroy();
    },
  };
}

export default {
  id: 'speak',
  title: 'Speak',
  partial: '/static/partials/speak.html',
  init,
};
