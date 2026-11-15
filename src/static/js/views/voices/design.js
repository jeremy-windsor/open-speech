/**
 * The 3-step design wizard (#voices/design): 1 Describe (a description of the voice, a model whose capabilities
 * declare voice_design, and that model's own options), 2 Listen (POST /api/voices/design speaks an editable sample
 * text in that voice; listen and generate again), 3 Name & save (POST /api/voices/designed makes a designed named
 * voice: the sample, its words as the transcript, and a version for each "Speak with" model that clones it).
 *
 * The seed of the sample being listened to is kept: new sample words speak the same voice again, Generate again
 * makes a new one. Edit description (startEdit) fills the wizard from a designed voice; its step 3 offers Replace
 * (PUT /api/voices/designed/{id}) or Save as new voice.
 *
 * The pure helpers (designModels, designGate, designBody, designForm, defaultSpeakWith) are exported for the
 * tests; createDesignWizard() wires the partial's #vo-design-wizard.
 * @module views/voices/design
 */
import { html, render } from '../../core/dom.js';
import { icon } from '../../core/icons.js';
import * as api from '../../core/api.js';
import * as audio from '../../core/audio.js';
import * as models from '../../core/models.js';
import { saveDesigned, replaceDesigned } from '../../core/voices.js';
import { app } from '../../core/store.js';
import { announce } from '../../core/a11y.js';
import { confirm } from '../../core/dialogs.js';
import { toast } from '../../core/toast.js';
import { mountModelOptions } from '../../components/model-options.js';
import { showBanner, clearBanner, actionError, modelName, cloneModels, idPart } from './shared.js';
import { draftMeta, wavSampleRate } from './clone.js';

/** The sample text the wizard starts with (the server speaks the same one when a request leaves it out). */
export const SAMPLE_TEXT = 'Hello! This is the voice you described. I can read your stories, your notes and your messages aloud.';
export const MAX_DESCRIPTION = 1000;
/** About 20 seconds of speech: a saved sample must stay within MAX_SAMPLE_SECONDS. */
export const MAX_SAMPLE = 300;
export const MAX_SAMPLE_SECONDS = 30;
export const NAME_MAX = 200;
export const NAME_IN_USE = 'A named voice with this name already exists. Choose another name.';
/** States whose weights are on the server: the cloning models a voice speaks with by default (as the API). */
const INSTALLED = new Set(['downloaded', 'loaded']);

/** TTS models that can design a voice from a description (capabilities.voice_design), in the app's model order. */
export function designModels(list) {
  return models.ttsModels(list || []).filter((m) => m && m.capabilities && m.capabilities.voice_design === true);
}

/** "A voice sample can be at most 30 s (this one is 31.2 s). Shorten the sample text, then generate again." */
export function tooLongText(durationS) {
  return `A voice sample can be at most ${MAX_SAMPLE_SECONDS} s (this one is ${(Math.round(durationS * 10) / 10).toFixed(1)} s). Shorten the sample text, then generate again.`;
}

/** The JSON body of POST /api/voices/design: options only when some are set, the seed only when one is kept. */
export function designBody({ model, description, text, options = {}, seed = null }) {
  const body = { model, description: String(description || '').trim(), text: String(text || '').trim() };
  if (options && Object.keys(options).length) body.options = { ...options };
  if (Number.isInteger(seed)) body.seed = seed;
  return body;
}

/**
 * The multipart body of POST /api/voices/designed (with `name` and `speakWith`) or PUT …/{id} (without them).
 * An empty `speakWith` is sent as one empty field: no versions, rather than the server's default.
 */
export function designForm({ sample, name = null, speakWith = null }) {
  const form = new FormData();
  if (name !== null) form.append('name', String(name).trim());
  form.append('model', sample.model);
  form.append('description', sample.description);
  form.append('transcript', sample.text);
  form.append('audio', sample.blob, 'sample.wav');
  if (Number.isInteger(sample.seed)) form.append('seed', String(sample.seed));
  if (sample.options && Object.keys(sample.options).length) form.append('options', JSON.stringify(sample.options));
  if (Array.isArray(speakWith)) {
    if (!speakWith.length) form.append('speak_with', '');
    for (const id of speakWith) form.append('speak_with', id);
  }
  return form;
}

/**
 * The "Speak with" models checked at first, as the API defaults them: the design model when it clones, else the
 * installed cloning models of its provider.
 */
export function defaultSpeakWith(modelId, list) {
  const clones = cloneModels(list || []);
  if (clones.some((m) => m.id === modelId)) return [modelId];
  const provider = models.providerOf(modelId);
  return clones
    .filter((m) => m.id !== modelId && (m.provider || models.providerOf(m.id)) === provider && INSTALLED.has(m.state))
    .map((m) => m.id)
    .sort();
}

/**
 * The save gate: a generated sample within MAX_SAMPLE_SECONDS and, for a new voice, a name not in use.
 * @param {{sample: object|null, name?: string, named?: object[], needsName?: boolean}} input
 * @returns {{ok: boolean, reasons: string[]}}
 */
export function designGate({ sample, name = '', named = [], needsName = true }) {
  const reasons = [];
  if (!sample) reasons.push('Generate a sample first.');
  else if (sample.durationS > MAX_SAMPLE_SECONDS) reasons.push(tooLongText(sample.durationS));
  if (needsName) {
    const wanted = String(name || '').trim();
    if (!wanted) reasons.push('Enter a name for the voice.');
    else if ((named || []).some((n) => String(n?.name || '').toLowerCase() === wanted.toLowerCase())) reasons.push(NAME_IN_USE);
  }
  return { ok: reasons.length === 0, reasons };
}

/** The WAV's length in seconds from its header (16-bit mono, as the server writes it), or 0. */
async function wavInfo(blob) {
  try {
    const head = await blob.slice(0, 4096).arrayBuffer();
    const sampleRate = wavSampleRate(head);
    return { sampleRate, durationS: sampleRate ? Math.max(0, blob.size - 44) / (sampleRate * 2) : 0 };
  } catch {
    return { sampleRate: 0, durationS: 0 };
  }
}

const sameOptions = (a, b) => JSON.stringify(a || {}) === JSON.stringify(b || {});

/**
 * Wire the wizard inside `root` (#vo-design-wizard).
 * @param {HTMLElement} root
 * @param {{named: () => object[], changed: (what: string) => void,
 *   afterSave: (identity: object) => Promise<void>, openModels: () => void}} ctx
 */
export function createDesignWizard(root, ctx) {
  const $ = (id) => root.querySelector(`#${id}`);
  const el = {
    steps: root.querySelector('.vo-steps'),
    title: $('vd-title'),
    status: $('vd-status'),
    description: $('vd-description'),
    descriptionError: $('vd-description-error'),
    model: $('vd-model'),
    modelHint: $('vd-model-hint'),
    options: $('vd-options'),
    next1: $('vd-next-1'),
    text: $('vd-text'),
    textError: $('vd-text-error'),
    generate: $('vd-generate'),
    generateError: $('vd-generate-error'),
    sample: $('vd-sample'),
    audio: $('vd-audio'),
    sampleMeta: $('vd-sample-meta'),
    sampleLong: $('vd-sample-long'),
    sampleNote: $('vd-sample-note'),
    next2: $('vd-next-2'),
    step3Title: $('vd-step-3-title'),
    name: $('vd-name'),
    nameHint: $('vd-name-hint'),
    nameError: $('vd-name-error'),
    speakWith: $('vd-speak-with-list'),
    speakWithHint: $('vd-speak-with-hint'),
    summary: $('vd-summary'),
    replace: $('vd-replace'),
    save: $('vd-save'),
    saveHint: $('vd-save-hint'),
    saveError: $('vd-save-error'),
  };

  const st = {
    step: 1,
    model: '',
    sample: null,        // {blob, url, text, description, model, options, seed, size, sampleRate, durationS}
    seed: null,          // the seed the next Generate sample sends (the sample's, or the edited voice's)
    generating: false,
    generateSeq: 0,
    saving: false,
    shown: false,
    edit: null,          // {id, name, design} while Edit description fills the wizard
    speakWith: null,     // the checked "Speak with" models once the listener changed one, else the defaults
    speakSig: '',
  };
  const offs = [];
  let options = null;    // the model's options (components/model-options.js)
  let optionsModel = '';

  function status(text, { announceIt = false } = {}) {
    if (el.status) el.status.textContent = text || '';
    if (announceIt && text) announce(text);
  }

  function setButton(button, { label, busy = false, disabled = false, iconName = '' } = {}) {
    if (!button) return;
    if (label !== undefined) {
      render(button, html`${iconName && !busy ? icon(iconName) : ''}<span class="btn__label">${label}</span>`);
    }
    if (busy) button.setAttribute('aria-busy', 'true');
    else button.removeAttribute('aria-busy');
    if (disabled || busy) button.setAttribute('aria-disabled', 'true');
    else button.removeAttribute('aria-disabled');
  }

  function setFieldError(input, errorEl, text) {
    if (!errorEl || !input) return;
    errorEl.hidden = !text;
    render(errorEl, text ? html`${icon('alert')}<span>${text}</span>` : html``);
    if (text) input.setAttribute('aria-invalid', 'true');
    else input.removeAttribute('aria-invalid');
  }

  const description = () => String(el.description ? el.description.value : '').trim();
  const sampleText = () => String(el.text ? el.text.value : '').trim();
  const voiceName = () => String(el.name ? el.name.value : '').trim();
  const available = () => designModels(app.get().models.list);
  const optionValues = () => (options ? options.values() : {});

  /* ───────────── steps ───────────── */

  function gate({ needsName = true } = {}) {
    return designGate({ sample: st.sample, name: voiceName(), named: ctx.named() || [], needsName });
  }

  function reachable(step) {
    if (step <= 1) return true;
    if (!description() || !st.model) return false;
    if (step === 2) return true;
    return !!st.sample;
  }

  function renderSteps() {
    if (el.steps) {
      for (const button of el.steps.querySelectorAll('[data-step]')) {
        const n = Number(button.getAttribute('data-step'));
        if (n === st.step) button.setAttribute('aria-current', 'step');
        else button.removeAttribute('aria-current');
        const done = n < st.step;
        const li = button.closest('li');
        if (li) li.classList.toggle('is-done', done);
        const said = button.querySelector('.vo-steps__done');
        if (said) said.textContent = done ? ', done' : '';
        if (reachable(n) && !st.generating) button.removeAttribute('aria-disabled');
        else button.setAttribute('aria-disabled', 'true');
      }
    }
    for (const n of [1, 2, 3]) {
      const section = root.querySelector(`#vd-step-${n}`);
      if (section) section.hidden = n !== st.step;
    }
    if (el.title) el.title.textContent = st.edit ? `Edit ${st.edit.name}` : 'Design a voice';
    if (el.next1) setButton(el.next1, { disabled: !available().length });
    if (el.next2) {
      setButton(el.next2, { disabled: !st.sample || st.generating });
      // One primary action: Generate until there is a sample, then Next.
      el.next2.classList.toggle('btn--primary', !!st.sample);
      el.next2.classList.toggle('btn--secondary', !st.sample);
    }
    if (el.generate) {
      el.generate.classList.toggle('btn--primary', !st.sample);
      el.generate.classList.toggle('btn--secondary', !!st.sample);
    }
  }

  function goStep(n, { focus = true } = {}) {
    if (!reachable(n) || st.generating) return false;
    st.step = n;
    renderAll();
    status('');
    if (focus) {
      const heading = root.querySelector(`#vd-step-${n}-title`);
      if (heading) heading.focus({ preventScroll: false });
    }
    return true;
  }

  /** Next from Describe: say what is missing, with focus on the first field that needs it. */
  function nextFromDescribe() {
    if (!available().length) {
      renderModels();
      return;
    }
    if (!description()) {
      setFieldError(el.description, el.descriptionError, 'Describe the voice you want.');
      if (el.description) el.description.focus();
      status('');
      return;
    }
    goStep(2);
  }

  /* ───────────── step 1: describe ───────────── */

  /** The options of the model chosen: mounted again when the model changes (`values` start them). */
  function mountOptions(values) {
    if (!el.options) return;
    if (options) options.destroy();
    options = null;
    optionsModel = st.model;
    const row = available().find((m) => m.id === st.model);
    if (!row) {
      el.options.hidden = true;
      return;
    }
    options = mountModelOptions(el.options, row.capabilities || {}, { values: values || {}, idPrefix: 'vd-option' });
  }

  function renderModels() {
    if (!el.model) return;
    const list = available();
    const keep = el.model.value || st.model;
    if (!list.length) {
      render(el.model, html`<option value="">No model that can design a voice</option>`);
      el.model.value = '';
      el.model.disabled = true;
      if (el.modelHint) render(el.modelHint, html`No model that can design a voice is running. Breeze TTS 2 can: turn on its worker, then load it. <a href="#models/tts">Open Models</a>`);
    } else {
      render(el.model, html`${list.map((m) => html`<option value="${m.id}">${modelName(m.id)}</option>`)}`);
      el.model.value = list.some((m) => m.id === keep) ? keep : list[0].id;
      el.model.disabled = false;
      if (el.modelHint) el.modelHint.textContent = 'Models that can make a voice from a description.';
    }
    st.model = el.model.value || '';
    if (st.model !== optionsModel || (!options && st.model)) mountOptions(st.edit && st.edit.design.model === st.model ? st.edit.design.options : {});
    renderSteps();
  }

  /* ───────────── step 2: listen ───────────── */

  function renderSample() {
    const s = st.sample;
    if (el.sample) el.sample.hidden = !s;
    if (el.sampleMeta) el.sampleMeta.textContent = s ? `${draftMeta(s)} · ${modelName(s.model)}` : '';
    const long = !!s && s.durationS > MAX_SAMPLE_SECONDS;
    if (el.sampleLong) {
      el.sampleLong.hidden = !long;
      render(el.sampleLong, long ? html`${icon('alert', { className: 'ico ico--sm' })}${tooLongText(s.durationS)}` : html``);
    }
    const muted = !!app.get().recording;
    if (el.sampleNote) el.sampleNote.hidden = !muted;
    if (el.audio) {
      if (muted) el.audio.setAttribute('aria-describedby', 'vd-sample-note');
      else el.audio.removeAttribute('aria-describedby');
    }
    if (el.generate) {
      setButton(el.generate, {
        label: st.generating ? 'Generating…' : (s ? 'Generate again' : 'Generate sample'),
        iconName: s ? 'refresh' : 'wand',
        busy: st.generating,
        disabled: !st.model || !available().length,
      });
    }
  }

  function dropSample() {
    if (!st.sample) return;
    if (el.audio) {
      if (el.audio.pause) el.audio.pause();
      el.audio.removeAttribute('src');
    }
    URL.revokeObjectURL(st.sample.url);
    st.sample = null;
  }

  async function generate() {
    if (st.generating) return;
    clearBanner(el.generateError);
    setFieldError(el.text, el.textError, '');
    const modelId = st.model;
    const words = sampleText();
    const wanted = description();
    if (!modelId || !available().some((m) => m.id === modelId)) {
      showBanner(el.generateError, { title: "Couldn't generate a sample", message: 'No model that can design a voice is available. Start or download one in Models.', actions: [{ label: 'Open Models', action: 'open-models-design' }] });
      return;
    }
    if (!wanted) {
      goStep(1);
      setFieldError(el.description, el.descriptionError, 'Describe the voice you want.');
      return;
    }
    if (!words) {
      setFieldError(el.text, el.textError, 'Type the words the sample speaks.');
      if (el.text) el.text.focus();
      return;
    }
    // Generate again is a new voice; the first sample (and new words) speak the kept seed's voice.
    const seed = st.sample ? null : st.seed;
    const chosen = optionValues();
    const seq = (st.generateSeq += 1);
    const label = modelName(modelId);
    st.generating = true;
    renderAll();
    status(`Generating a sample with ${label}…`);
    try {
      const ok = await models.ensureReady(modelId, { kind: 'tts' });
      if (seq !== st.generateSeq) return;
      if (!ok) {
        status('Sample canceled.');
        return;
      }
      const res = await api.request('/api/voices/design', {
        method: 'POST',
        json: designBody({ model: modelId, description: wanted, text: words, options: chosen, seed }),
        accept: 'response',
      });
      const blob = await res.blob();
      const header = Number.parseInt(res.headers.get('X-Voice-Seed') || '', 10);
      models.markUsed(modelId);
      if (seq !== st.generateSeq) return;
      const info = await wavInfo(blob);
      if (seq !== st.generateSeq) return;
      dropSample();
      if (sampleText() !== words) {
        // Spoken from the words as they were: the saved transcript must match what it says.
        st.generating = false;
        renderAll();
        status('The sample text changed. Generate the sample again.');
        return;
      }
      const usedSeed = Number.isInteger(header) ? header : seed;
      st.seed = Number.isInteger(usedSeed) ? usedSeed : null;
      if (el.audio) el.audio.src = URL.createObjectURL(blob);
      st.sample = {
        blob, url: el.audio ? el.audio.src : '', text: words, description: wanted, model: modelId, options: chosen,
        seed: st.seed, size: blob.size, ...info,
      };
      st.generating = false;
      renderAll();
      status(info.durationS > MAX_SAMPLE_SECONDS
        ? `Sample ready, but it is ${(Math.round(info.durationS * 10) / 10).toFixed(1)} s: a voice sample can be at most ${MAX_SAMPLE_SECONDS} s.`
        : 'Sample ready. Listen to it, generate again, or go on to name it.', { announceIt: true });
      if (el.audio && st.shown && seq === st.generateSeq && !app.get().recording) {
        audio.otherPlayerStarted();
        const playing = el.audio.play && el.audio.play();
        if (playing && playing.catch) playing.catch(() => {});
      }
    } catch (error) {
      if (error && error.kind === 'aborted') return;
      models.markFailed(modelId);
      if (seq !== st.generateSeq) return;
      showBanner(el.generateError, actionError(error, "Couldn't generate a sample"));
      status('');
    } finally {
      if (seq === st.generateSeq) {
        st.generating = false;
        renderAll();
      }
    }
  }

  /* ───────────── step 3: name and save ───────────── */

  function speakChoices() {
    return cloneModels(app.get().models.list);
  }

  function checkedSpeakWith() {
    const choices = speakChoices().map((m) => m.id);
    if (st.speakWith) return choices.filter((id) => st.speakWith.has(id));
    const model = st.sample ? st.sample.model : st.model;
    return defaultSpeakWith(model, app.get().models.list).filter((id) => choices.includes(id));
  }

  /** The "Speak with" checkboxes: re-rendered only when the choices or the defaults change (focus stays put). */
  function renderSpeakWith() {
    if (!el.speakWith) return;
    const choices = speakChoices();
    const checked = checkedSpeakWith();
    const sig = `${choices.map((m) => m.id).join(',')}|${checked.join(',')}`;
    if (sig !== st.speakSig) {
      st.speakSig = sig;
      render(el.speakWith, choices.length
        ? html`${choices.map((m) => html`<label class="check"><input type="checkbox" class="check__input" id="vd-speak-${idPart(m.id)}" data-model="${m.id}"${checked.includes(m.id) ? html` checked` : ''}><span>${modelName(m.id)}</span></label>`)}`
        : html``);
    }
    if (el.speakWithHint) {
      el.speakWithHint.textContent = choices.length
        ? `${st.edit ? 'For Save as new voice: each' : 'Each'} model you check gets a version of the voice that clones its sample.`
        : 'No model that can clone a voice is running, so the voice is saved without versions.';
    }
  }

  function renderSave() {
    const editing = !!st.edit;
    const g = gate();
    const r = gate({ needsName: false });
    if (el.step3Title) el.step3Title.textContent = editing ? `Replace ${st.edit.name}, or save a new voice` : 'Name it, then save';
    if (el.nameHint) {
      el.nameHint.textContent = editing
        ? `Only for Save as new voice. Replace keeps the name and versions of ${st.edit.name}.`
        : 'Speak, presets and the Reader show the voice by this name.';
    }
    if (el.replace) {
      el.replace.hidden = !editing;
      if (editing) setButton(el.replace, { label: `Replace ${st.edit.name}`, iconName: 'check', busy: st.saving === 'replace', disabled: !r.ok || !!st.saving });
    }
    if (el.save) {
      setButton(el.save, { label: editing ? 'Save as new voice' : 'Save voice', iconName: editing ? 'plus' : 'check', busy: st.saving === 'new', disabled: !g.ok || !!st.saving });
      el.save.classList.toggle('btn--primary', !editing);
      el.save.classList.toggle('btn--secondary', editing);
    }
    if (el.saveHint) {
      const reasons = editing ? (r.ok ? [] : r.reasons) : g.reasons;
      el.saveHint.textContent = reasons.join(' ');
      el.saveHint.hidden = !reasons.length;
    }
    renderSpeakWith();
    if (el.summary) {
      const s = st.sample;
      const speak = checkedSpeakWith();
      render(el.summary, html`<div class="vo-summary__row"><dt>Description</dt><dd>${s ? s.description : '—'}</dd></div><div class="vo-summary__row"><dt>Sample</dt><dd class="tabular">${s ? `${draftMeta(s)} · ${modelName(s.model)}` : 'None yet'}</dd></div><div class="vo-summary__row"><dt>Transcript</dt><dd>${s ? s.text : '—'}</dd></div>${editing ? '' : html`<div class="vo-summary__row"><dt>Versions</dt><dd>${speak.length ? speak.map(modelName).join(', ') : 'None'}</dd></div>`}`);
    }
  }

  async function saveNew() {
    if (!gate().ok || st.saving) return;
    clearBanner(el.saveError);
    const name = voiceName();
    const sample = st.sample;
    st.saving = 'new';
    renderSave();
    status(`Saving ${name}…`);
    try {
      const identity = await saveDesigned(designForm({ sample, name, speakWith: checkedSpeakWith() }));
      await finish(identity, `Saved ${identity?.name || name}`);
    } catch (error) {
      if (error && error.kind === 'aborted') return;
      if (error && error.status === 409) {
        setFieldError(el.name, el.nameError, NAME_IN_USE);
        if (el.name) el.name.focus();
        status('');
        return;
      }
      showBanner(el.saveError, actionError(error, "Couldn't save the voice"));
      status('');
    } finally {
      st.saving = false;
      renderSave();
    }
  }

  async function replace() {
    const edit = st.edit;
    if (!edit || st.saving || !gate({ needsName: false }).ok) return;
    const ok = await confirm({
      title: `Replace ${edit.name}?`,
      message: 'Its versions and presets speak with the new sample from now on. The old sample is not kept.',
      confirmLabel: 'Replace',
    });
    if (!ok) return;
    clearBanner(el.saveError);
    const sample = st.sample;
    st.saving = 'replace';
    renderSave();
    status(`Replacing ${edit.name}…`);
    try {
      const identity = await replaceDesigned(edit.id, designForm({ sample }));
      await finish(identity || { id: edit.id, name: edit.name }, `Replaced ${edit.name}`);
    } catch (error) {
      if (error && error.kind === 'aborted') return;
      showBanner(el.saveError, actionError(error, `Couldn't replace ${edit.name}`));
      status('');
    } finally {
      st.saving = false;
      renderSave();
    }
  }

  async function finish(identity, message) {
    toast(message, { kind: 'success' });
    status(`${message}.`);
    ctx.changed('named');
    reset();
    await ctx.afterSave(identity);
  }

  /** Back to an empty wizard: the description and sample text keep their starting values. */
  function reset() {
    st.generateSeq += 1;
    st.generating = false;
    dropSample();
    st.seed = null;
    st.edit = null;
    st.speakWith = null;
    if (el.description) el.description.value = '';
    if (el.text) el.text.value = SAMPLE_TEXT;
    if (el.name) el.name.value = '';
    setFieldError(el.description, el.descriptionError, '');
    setFieldError(el.text, el.textError, '');
    setFieldError(el.name, el.nameError, '');
    for (const host of [el.generateError, el.saveError]) clearBanner(host);
    st.step = 1;
    if (options) options.set({});
    renderAll();
  }

  /** Fill the wizard from a designed named voice (Edit description). */
  function startEdit(identity) {
    const design = identity && identity.design;
    if (!design) return;
    reset();
    st.edit = { id: identity.id, name: identity.name, design };
    if (el.description) el.description.value = String(design.description || '');
    if (el.text) el.text.value = String(design.transcript || SAMPLE_TEXT);
    st.seed = Number.isInteger(design.seed) ? design.seed : null;
    if (el.model) el.model.value = '';
    st.model = design.model || '';
    optionsModel = '';
    renderModels();
    if (st.model !== design.model) st.seed = null;   // another model makes another voice from the same seed
    renderAll();
  }

  function renderAll() {
    renderSteps();
    renderSample();
    renderSave();
  }

  /* ───────────── events ───────────── */

  const listen = (node, type, fn) => {
    if (!node) return;
    node.addEventListener(type, fn);
    offs.push(() => node.removeEventListener(type, fn));
  };
  listen(el.description, 'input', () => {
    setFieldError(el.description, el.descriptionError, '');
    renderSteps();
  });
  listen(el.model, 'change', () => {
    st.model = el.model.value;
    st.seed = null;
    if (st.sample && st.sample.model !== st.model) dropSample();
    mountOptions(st.edit && st.edit.design.model === st.model ? st.edit.design.options : {});
    clearBanner(el.generateError);
    renderAll();
  });
  listen(el.text, 'input', () => {
    setFieldError(el.text, el.textError, '');
    // The saved transcript must be the words the sample speaks: new words need a new sample.
    if (st.sample && sampleText() !== st.sample.text) {
      dropSample();
      status('The sample text changed. Generate the sample again.');
    }
    renderAll();
  });
  listen(el.name, 'input', () => {
    const wanted = voiceName().toLowerCase();
    const taken = !!wanted && (ctx.named() || []).some((n) => String(n?.name || '').toLowerCase() === wanted);
    setFieldError(el.name, el.nameError, taken ? NAME_IN_USE : '');
    renderSave();
  });
  listen(el.speakWith, 'change', (event) => {
    const box = event.target && event.target.closest ? event.target.closest('input[data-model]') : null;
    if (!box) return;
    st.speakWith = new Set([...el.speakWith.querySelectorAll('input[data-model]')].filter((b) => b.checked).map((b) => b.getAttribute('data-model')));
    st.speakSig = '';
    renderSave();
  });
  listen(el.audio, 'play', () => {
    // A plain <audio controls>: it follows the app's one-player rule by hand, as the clone wizard's draft does.
    if (app.get().recording) {
      el.audio.pause();
      return;
    }
    audio.otherPlayerStarted();
  });
  offs.push(app.select((s) => s.player && s.player.playing, (playing) => {
    if (playing && el.audio && !el.audio.paused) el.audio.pause();
  }));
  offs.push(app.select((s) => !!s.recording, () => renderSample()));
  offs.push(app.select((s) => s.models.list, () => {
    renderModels();
    renderAll();
  }));

  const actions = {
    'design-step': (button) => goStep(Number(button.getAttribute('data-step'))),
    'design-next-1': () => nextFromDescribe(),
    'design-next-2': () => goStep(3),
    'design-back-1': () => goStep(1),
    'design-back-2': () => goStep(2),
    'design-generate': () => generate(),
    'design-save': () => saveNew(),
    'design-replace': () => replace(),
    'open-models-design': () => ctx.openModels('tts'),
  };

  if (el.text && !el.text.value) el.text.value = SAMPLE_TEXT;
  if (el.description) el.description.maxLength = MAX_DESCRIPTION;
  if (el.text) el.text.maxLength = MAX_SAMPLE;
  if (el.name) el.name.maxLength = NAME_MAX;
  renderModels();
  renderAll();

  return {
    actions,
    show() {
      st.shown = true;
      renderModels();
      renderAll();
    },
    isShown: () => st.shown,
    hide() {
      st.shown = false;
      // A sample still on its way is dropped: it must not start playing elsewhere in the app.
      if (st.generating) {
        st.generateSeq += 1;
        st.generating = false;
        renderAll();
        status('');
      }
      if (el.audio && !el.audio.paused) el.audio.pause();
    },
    /** Design a voice: a new design (an Edit description left open starts afresh). */
    startNew() {
      if (st.edit) reset();
    },
    startEdit,
    heading() {
      return el.title;
    },
    /** Test hook: the internal state. */
    state: () => st,
    destroy() {
      for (const off of offs.splice(0)) off();
      if (options) options.destroy();
      dropSample();
    },
  };
}
