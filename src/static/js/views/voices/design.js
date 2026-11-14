/**
 * The 3-step design wizard (#voices/design): 1 Describe (a description of the voice and a model whose
 * capabilities declare voice_design), 2 Listen (POST /api/voices/design speaks an editable sample text in that
 * voice; listen and generate again), 3 Name & save (the sample goes to POST /api/voices/library with its text as
 * the transcript, so it clones like any other recording, with any cloning model).
 *
 * The pure helpers (designModels, designGate, designBody) are exported for the tests; createDesignWizard() wires
 * the partial's #vo-design-wizard.
 * @module views/voices/design
 */
import { html, render } from '../../core/dom.js';
import { icon } from '../../core/icons.js';
import * as api from '../../core/api.js';
import * as audio from '../../core/audio.js';
import * as models from '../../core/models.js';
import { app } from '../../core/store.js';
import { announce } from '../../core/a11y.js';
import { confirm } from '../../core/dialogs.js';
import { toast } from '../../core/toast.js';
import { showBanner, clearBanner, actionError, modelName } from './shared.js';
import { sanitizeVoiceName, saveForm, draftMeta, wavSampleRate } from './clone.js';

/** The sample text the wizard starts with (the server speaks the same one when a request leaves it out). */
export const SAMPLE_TEXT = 'Hello! This is the voice you described. I can read your stories, your notes and your messages aloud.';
export const MAX_DESCRIPTION = 1000;
/** About 20 seconds of speech: a saved sample must stay within the 30 s a recording to clone from can be. */
export const MAX_SAMPLE = 300;

/** TTS models that can design a voice from a description (capabilities.voice_design), in the app's model order. */
export function designModels(list) {
  return models.ttsModels(list || []).filter((m) => m && m.capabilities && m.capabilities.voice_design === true);
}

/** The JSON body of POST /api/voices/design. */
export function designBody({ model, description, text }) {
  return { model, description: String(description || '').trim(), text: String(text || '').trim() };
}

/**
 * The save gate: a generated sample and a name that sanitizes to something.
 * @returns {{ok: boolean, reasons: string[]}}
 */
export function designGate({ sample, name }) {
  const reasons = [];
  if (!sample) reasons.push('Generate a sample first.');
  if (!sanitizeVoiceName(name)) reasons.push('Enter a voice name with at least one letter or number.');
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

/**
 * Wire the wizard inside `root` (#vo-design-wizard).
 * @param {HTMLElement} root
 * @param {{recordings: () => object[], changed: (what: string) => void,
 *   afterSave: (name: string) => Promise<void>, openModels: () => void}} ctx
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
    next1: $('vd-next-1'),
    text: $('vd-text'),
    textError: $('vd-text-error'),
    generate: $('vd-generate'),
    generateError: $('vd-generate-error'),
    sample: $('vd-sample'),
    audio: $('vd-audio'),
    sampleMeta: $('vd-sample-meta'),
    sampleNote: $('vd-sample-note'),
    next2: $('vd-next-2'),
    name: $('vd-name'),
    namePreview: $('vd-name-preview'),
    nameError: $('vd-name-error'),
    summary: $('vd-summary'),
    save: $('vd-save'),
    saveHint: $('vd-save-hint'),
    saveError: $('vd-save-error'),
  };

  const st = {
    step: 1,
    model: '',
    sample: null,        // {blob, url, text, description, model, size, sampleRate, durationS}
    generating: false,
    generateSeq: 0,
    saving: false,
    shown: false,
  };
  const offs = [];

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
  const available = () => designModels(app.get().models.list);

  /* ───────────── steps ───────────── */

  function gate() {
    return designGate({ sample: st.sample, name: el.name ? el.name.value : '' });
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
    renderSteps();
  }

  /* ───────────── step 2: listen ───────────── */

  function renderSample() {
    const s = st.sample;
    if (el.sample) el.sample.hidden = !s;
    if (el.sampleMeta) el.sampleMeta.textContent = s ? `${draftMeta(s)} · ${modelName(s.model)}` : '';
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
      const blob = await api.request('/api/voices/design', {
        method: 'POST',
        json: designBody({ model: modelId, description: wanted, text: words }),
        accept: 'blob',
      });
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
      if (el.audio) el.audio.src = URL.createObjectURL(blob);
      st.sample = { blob, url: el.audio ? el.audio.src : '', text: words, description: wanted, model: modelId, size: blob.size, ...info };
      st.generating = false;
      renderAll();
      status('Sample ready. Listen to it, generate again, or go on to name it.', { announceIt: true });
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

  function renderSave() {
    const raw = el.name ? el.name.value : '';
    const safe = sanitizeVoiceName(raw);
    const exists = safe && (ctx.recordings() || []).some((r) => r && r.name === safe);
    if (el.namePreview) el.namePreview.textContent = `Saves as: ${safe || '—'}${exists ? ' · replaces the recording with this name' : ''}`;
    const g = gate();
    if (el.save) setButton(el.save, { busy: st.saving, disabled: !g.ok || st.saving });
    if (el.saveHint) {
      el.saveHint.textContent = g.ok ? '' : g.reasons.join(' ');
      el.saveHint.hidden = g.ok;
    }
    if (el.summary) {
      const s = st.sample;
      render(el.summary, html`<div class="vo-summary__row"><dt>Name</dt><dd class="mono">${safe || '—'}</dd></div><div class="vo-summary__row"><dt>Description</dt><dd>${s ? s.description : '—'}</dd></div><div class="vo-summary__row"><dt>Sample</dt><dd class="tabular">${s ? `${draftMeta(s)} · ${modelName(s.model)}` : 'None yet'}</dd></div><div class="vo-summary__row"><dt>Transcript</dt><dd>${s ? s.text : '—'}</dd></div>`);
    }
  }

  async function save() {
    const g = gate();
    if (!g.ok || st.saving) {
      if (!sanitizeVoiceName(el.name ? el.name.value : '')) {
        setFieldError(el.name, el.nameError, 'Enter a name with at least one letter or number.');
        if (el.name) el.name.focus();
      }
      status(g.reasons[0] || '');
      return;
    }
    clearBanner(el.saveError);
    const name = sanitizeVoiceName(el.name.value);
    const sample = st.sample;
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
      await api.request('/api/voices/library', { method: 'POST', body: saveForm({ name, blob: sample.blob, transcript: sample.text }), accept: 'json' });
      toast(`Saved recording ${name}`, { kind: 'success' });
      status(`Saved ${name}.`);
      ctx.changed('recordings');
      reset();
      await ctx.afterSave(name);
    } catch (error) {
      if (error && error.kind === 'aborted') return;
      showBanner(el.saveError, actionError(error, "Couldn't save the recording"));
      status('');
    } finally {
      st.saving = false;
      renderSave();
    }
  }

  /** Back to an empty wizard (after a save): the description and sample text keep their starting values. */
  function reset() {
    st.generateSeq += 1;
    st.generating = false;
    dropSample();
    if (el.description) el.description.value = '';
    if (el.text) el.text.value = SAMPLE_TEXT;
    if (el.name) el.name.value = '';
    setFieldError(el.description, el.descriptionError, '');
    setFieldError(el.text, el.textError, '');
    setFieldError(el.name, el.nameError, '');
    for (const host of [el.generateError, el.saveError]) clearBanner(host);
    st.step = 1;
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
    setFieldError(el.name, el.nameError, '');
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
    'design-save': () => save(),
    'open-models-design': () => ctx.openModels('tts'),
  };

  if (el.text && !el.text.value) el.text.value = SAMPLE_TEXT;
  if (el.description) el.description.maxLength = MAX_DESCRIPTION;
  if (el.text) el.text.maxLength = MAX_SAMPLE;
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
    heading() {
      return el.title;
    },
    /** Test hook: the internal state. */
    state: () => st,
    destroy() {
      for (const off of offs.splice(0)) off();
      dropSample();
    },
  };
}
