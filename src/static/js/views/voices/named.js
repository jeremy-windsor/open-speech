/**
 * Named voices in Voices' one list (#voices, inventory SE-02…SE-04, SE-07):
 * one card per named voice with a row per version (model badge, voice recipe in mono, a blend bar for Kokoro
 * blends), [Edit in Speak] (the `edit-named-voice` intent: Speak shows its editing bar, with Update voice) and
 * [Remove version]; under them its presets (presets.js) and [New preset] (the `new-preset` intent); "N presets
 * use it", [Rename] and a menu with Rename… (PATCH; a 409 keeps the prompt open with a field error) and Delete….
 * A designed voice (identity.design) also shows a "Designed" badge with its design model, its description,
 * [Play sample], [Edit description] (the design wizard, filled in) and [Use in Speak] (the `use-named-voice`
 * intent); deleting it deletes its sample. Every action disables its button while it runs. list.js renders the
 * cards.
 * @module views/voices/named
 */
import { html, setBusy } from '../../core/dom.js';
import { icon } from '../../core/icons.js';
import * as api from '../../core/api.js';
import * as audio from '../../core/audio.js';
import { confirm, prompt } from '../../core/dialogs.js';
import { toast } from '../../core/toast.js';
import { app, setIntent } from '../../core/store.js';
import { describeVoice, isBlend, parseBlend, blendPercents } from '../../core/voices.js';
import { monogram } from '../../core/tiles.js';
import { idPart, presetCountText, modelName, modelAvailability, matches, cardMenuMarkup, STOP_RECORDING } from './shared.js';
import { presetItemMarkup } from './presets.js';

export const UNAVAILABLE = 'This model is unavailable';
export const LINKED_COPY = 'Linked presets keep their current voice setup. Recordings are kept.';
export const RENAME_CONFLICT = 'A named voice with this name already exists. Choose another name.';
export const NO_VERSIONS = 'No versions. Save this voice from Speak with a model to add one.';
export const NO_MODEL = 'None of its models is available.';
export const DESIGNED_SAMPLE = 'Clones its designed sample';
export const DESIGNED_DELETE = 'Presets that use it keep working: its sample moves to Recordings. A sample nothing uses is deleted with it.';
/** The audio session of a designed voice's sample. */
export const SAMPLE_SESSION = 'designed-sample';

const enc = encodeURIComponent;

/** "1 version", "2 versions". */
function versionCount(n) {
  return n === 1 ? '1 version' : `${n} versions`;
}

/** The meta line of a card: "2 versions · 1 preset uses it". */
export function namedMeta(identity) {
  const count = (identity?.realizations || []).length;
  return `${count ? versionCount(count) : 'No versions'} · ${presetCountText(identity?.preset_count)}`;
}

/**
 * How a version's voice reads: a Kokoro blend gives its parts with percents, a reference recording its name,
 * a plain voice its described label.
 * @returns {{kind: 'blend'|'reference'|'voice', text: string, parts: {voice: string, label: string, percent: number}[]}}
 */
export function describeVersion(realization, design = null) {
  const voice = String(realization?.voice || '');
  const reference = realization?.reference_audio_id || '';
  if (reference && design && reference === design.sample) return { kind: 'designed', text: DESIGNED_SAMPLE, parts: [] };
  if (isBlend(voice)) {
    try {
      const items = parseBlend(voice);
      const percents = blendPercents(items);
      const parts = items.map((item, i) => ({ voice: item.voice, label: describeVoice(item.voice).label, percent: percents[i] }));
      return { kind: 'blend', text: parts.map((p) => `${p.label} ${p.percent}%`).join(' · '), parts };
    } catch {
      return { kind: 'voice', text: voice, parts: [] };
    }
  }
  if (reference) return { kind: 'reference', text: `Recording ${reference}`, parts: [] };
  return { kind: 'voice', text: describeVoice(voice).label || voice, parts: [] };
}

/** A voice recipe that may break after each "+" (long blends wrap between voices, not inside one). */
export function recipeMarkup(recipe) {
  const parts = String(recipe || '').split('+');
  return html`${parts.map((p, i) => html`${i ? html`+<wbr>` : ''}${p}`)}`;
}

/** Whether a named voice matches the search text: its name, a model or a voice in any version. */
export function namedMatches(identity, query) {
  const versions = identity?.realizations || [];
  return matches(query, identity?.name, identity?.design?.description, ...versions.map((r) => r.model), ...versions.map((r) => modelName(r.model)), ...versions.map((r) => r.voice), ...versions.map((r) => r.reference_audio_id));
}

function versionMarkup(identity, realization, modelsState) {
  const key = `${idPart(identity.id)}-${idPart(realization.model)}`;
  const label = modelName(realization.model);
  const desc = describeVersion(realization, identity.design);
  const avail = modelAvailability(realization.model, modelsState);
  const warnId = `vo-ver-${key}-warn`;
  const bar = desc.kind === 'blend' && desc.parts.length > 1
    ? html`<div class="blend-bar vo-blend" role="img" aria-label="${desc.parts.map((p) => `${p.label} ${p.percent} percent`).join(', ')}">${desc.parts.map((p, i) => html`<span class="blend-bar__seg" data-i="${i % 6}" data-w="${p.percent}"></span>`)}</div>
      <p class="vo-legend">${desc.parts.map((p, i) => html`<span class="vo-legend__item"><span class="swatch" data-i="${i % 6}"></span>${p.label} <span class="tabular">${p.percent}%</span></span>`)}</p>`
    : html`<p class="vo-version__desc">${desc.text}</p>`;
  return html`<li class="vo-version" id="vo-ver-${key}">
    <div class="vo-version__main">
      <p class="vo-version__head"><span class="badge">${label}</span>${desc.kind === 'designed' ? '' : html`<span class="mono vo-version__spec">${recipeMarkup(realization.voice)}</span>`}</p>
      ${bar}
      ${avail.ok ? '' : html`<p class="vo-version__warn" id="${warnId}">${icon('alert', { className: 'ico ico--sm' })}${UNAVAILABLE}</p>`}
    </div>
    <div class="vo-version__actions">
      ${desc.kind === 'designed' ? '' : html`<button type="button" class="btn btn--secondary btn--sm" id="vo-ver-${key}-edit" data-action="named-edit" data-id="${identity.id}" data-model="${realization.model}" aria-label="Edit in Speak: ${identity.name} for ${label}"${avail.ok ? '' : html` aria-disabled="true" aria-describedby="${warnId}"`}>${icon('edit')}<span class="btn__label">Edit in Speak</span></button>`}
      <button type="button" class="btn btn--ghost btn--sm" id="vo-ver-${key}-remove" data-action="named-remove" data-id="${identity.id}" data-model="${realization.model}" aria-label="Remove version: ${identity.name} for ${label}"><span class="btn__label">Remove version</span></button>
    </div>
  </li>`;
}

/** A card's More actions menu. */
const CARD_MENU = Object.freeze([
  { action: 'rename', label: 'Rename…', icon: 'edit' },
  { action: 'delete', label: 'Delete…', icon: 'trash', danger: true },
]);

/**
 * Why [New preset] can't start one with this named voice now, or null: it has no version yet, or none of its
 * models is available. {reason, id}: `id` is the element that says so (aria-describedby).
 */
export function newPresetBlock(identity, modelsState) {
  const key = `vo-named-${idPart(identity?.id)}`;
  const versions = identity?.realizations || [];
  if (!versions.length) return { reason: NO_VERSIONS, id: `${key}-noversions` };
  if (!versions.some((r) => modelAvailability(r.model, modelsState).ok)) return { reason: NO_MODEL, id: `${key}-new-hint` };
  return null;
}

/** The model Use in Speak opens a named voice with: its first version whose model is available, or ''. */
export function useModel(identity, modelsState) {
  const versions = (identity?.realizations || []).slice().sort((a, b) => String(a.model).localeCompare(String(b.model)));
  const usable = versions.find((r) => modelAvailability(r.model, modelsState).ok);
  return usable ? usable.model : '';
}

/** A designed voice's part of its card: the badge with its design model, the description and its actions. */
function designMarkup(identity, key, block) {
  const design = identity.design;
  return html`<div class="vo-design">
      <p class="vo-design__head"><span class="badge badge--accent">${icon('wand')}Designed</span><span class="vo-design__model">${modelName(design.model)}</span></p>
      ${design.description ? html`<p class="vo-design__desc">${design.description}</p>` : ''}
      <div class="vo-design__actions">
        <button type="button" class="btn btn--secondary btn--sm" id="vo-${key}-sample" data-action="named-sample" data-id="${identity.id}" aria-label="Play sample: ${identity.name}">${icon('play')}<span class="btn__label">Play sample</span></button>
        <button type="button" class="btn btn--secondary btn--sm" id="vo-${key}-redesign" data-action="named-redesign" data-id="${identity.id}" aria-label="Edit description: ${identity.name}">${icon('wand')}<span class="btn__label">Edit description</span></button>
        <button type="button" class="btn btn--secondary btn--sm" id="vo-${key}-use" data-action="named-use" data-id="${identity.id}" aria-label="Use in Speak: ${identity.name}"${block ? html` aria-disabled="true" aria-describedby="${block.id}"` : ''}><span class="btn__label">Use in Speak</span></button>
      </div>
    </div>`;
}

/**
 * A named voice's card: its versions, then its presets and [New preset].
 * @param {object} identity a named voice
 * @param {{models?: object, presets?: object[], defaultId?: string|null, named?: object[]}} [options] the presets
 *   to show on the card (already filtered and ordered)
 */
export function namedCardMarkup(identity, { models, presets = [], defaultId = null, named = [] } = {}) {
  const key = `named-${idPart(identity.id)}`;
  const versions = (identity.realizations || []).slice().sort((a, b) => String(a.model).localeCompare(String(b.model)));
  const block = newPresetBlock(identity, models);
  return html`<li class="vo-grid__item"><article class="card vo-card vo-card--voice" id="vo-${key}" aria-labelledby="vo-${key}-title">
    <div class="vo-card__head">
      <span class="avatar vo-card__avatar" data-tile="${identity.name}" aria-hidden="true">${monogram(identity.name)}</span>
      <div class="vo-card__heading">
        <h2 class="vo-card__title" id="vo-${key}-title" tabindex="-1">${identity.name}</h2>
        <p class="vo-card__meta">${namedMeta(identity)}</p>
      </div>
      <button type="button" class="btn btn--ghost btn--sm vo-rename" id="vo-${key}-rename" data-action="named-rename" data-id="${identity.id}" aria-label="Rename: ${identity.name}"><span class="btn__label">Rename</span></button>
      ${cardMenuMarkup(key, identity.name, CARD_MENU)}
    </div>
    ${identity.design ? designMarkup(identity, key, block) : ''}
    ${versions.length
    ? html`<p class="section-label vo-card__label" id="vo-${key}-versions">Versions</p><ul class="vo-versions" aria-labelledby="vo-${key}-versions">${versions.map((r) => versionMarkup(identity, r, models))}</ul>`
    : html`<p class="vo-card__empty" id="vo-${key}-noversions">${NO_VERSIONS}</p>`}
    <p class="section-label vo-card__label" id="vo-${key}-presets">Presets</p>
    ${presets.length
    ? html`<ul class="vo-presets" aria-labelledby="vo-${key}-presets">${presets.map((p) => presetItemMarkup(p, { isDefault: p.id === defaultId, named, identity }))}</ul>`
    : html`<p class="vo-card__nopresets">No presets use it yet.</p>`}
    <div class="vo-card__foot">
      <button type="button" class="btn btn--secondary btn--sm" id="vo-${key}-new" data-action="named-new-preset" data-id="${identity.id}" aria-label="New preset with ${identity.name}"${block ? html` aria-disabled="true" aria-describedby="${block.id}"` : ''}>${icon('plus')}<span class="btn__label">New preset</span></button>
      ${block && block.reason === NO_MODEL ? html`<p class="field__hint vo-card__hint" id="${block.id}">${NO_MODEL}</p>` : ''}
    </div>
    <div class="vo-slot vo-card__error" id="vo-${key}-error" hidden></div>
  </article></li>`;
}

/**
 * The named voices' actions in the one list.
 * @param {{lists: object, load: (what: string) => Promise<boolean>, models: () => object,
 *   changed: (what: string) => void, navigate: (path: string, options?: object) => *, announce: (text: string) => void,
 *   listEl: HTMLElement, render: (options?: object) => void, error: (slot: string, error: *, title: string) => void,
 *   clearError: (slot: string) => void}} ctx
 */
export function createNamed(ctx) {
  const { listEl } = ctx;
  const list = () => ctx.lists.named;
  const find = (id) => (list().items || []).find((identity) => identity.id === id) || null;
  let sampleOn = '';        // the named voice whose sample is the current audio session
  let samplePending = '';   // the named voice whose sample audio is loading
  let sampleSeq = 0;        // the newest Play sample press wins

  /** Where focus goes when the focused card is gone: the next card, else the previous, else the list. */
  function fallbackAfter(index) {
    return () => {
      const titles = listEl.querySelectorAll('.vo-card__title');
      if (titles.length) return titles[Math.min(index, titles.length - 1)];
      return listEl.querySelector('.empty__title, h2') || null;
    };
  }

  function cardError(id, error, title) {
    ctx.error(`vo-named-${idPart(id)}-error`, error, title);
  }

  function clearCardError(id) {
    ctx.clearError(`vo-named-${idPart(id)}-error`);
  }

  async function refreshAfterChange(index, restore) {
    await Promise.all([ctx.load('named'), ctx.load('presets')]);
    ctx.render({ fallback: fallbackAfter(index), restore });
  }

  function indexOf(id) {
    return (list().items || []).findIndex((identity) => identity.id === id);
  }

  /** Edit in Speak (SE-03): the version's model must be usable; Speak applies it (intent edit-named-voice). */
  function edit(button) {
    const id = button.getAttribute('data-id');
    const model = button.getAttribute('data-model');
    const identity = find(id);
    if (!identity) return;
    const avail = modelAvailability(model, ctx.models());
    if (!avail.ok) {
      cardError(id, new Error(UNAVAILABLE), "Couldn't open in Speak");
      return;
    }
    setIntent('speak', { type: 'edit-named-voice', id, model });
    ctx.navigate('speak');
  }

  /** Remove version (SE-04): confirm, DELETE …/realizations/{model}, refresh named voices and presets. */
  async function removeVersion(button) {
    const id = button.getAttribute('data-id');
    const model = button.getAttribute('data-model');
    const identity = find(id);
    if (!identity) return;
    const label = modelName(model);
    const ok = await confirm({
      title: `Remove ${identity.name} for ${label}?`,
      message: LINKED_COPY,
      confirmLabel: 'Remove version',
      danger: true,
    });
    if (!ok) return;
    clearCardError(id);
    setBusy(button, true);
    try {
      await api.del(`/api/voices/identities/${enc(id)}/realizations/${enc(model)}`);
    } catch (error) {
      setBusy(button, false);
      if (error?.kind === 'aborted') return;
      cardError(id, error, "Couldn't delete");
      return;
    }
    toast(`Removed ${identity.name} for ${label}`, { kind: 'success' });
    ctx.changed('named');
    ctx.changed('presets');
    await refreshAfterChange(indexOf(id), button.id);
  }

  /** Rename… (SE-04): prompt; unchanged is skipped; a 409 re-opens the prompt with a field error. */
  async function rename(id, trigger) {
    const identity = find(id);
    if (!identity) return;
    let value = identity.name;
    let error = '';
    for (;;) {
      const name = await prompt({
        title: 'Rename named voice', label: 'Name', value, error, maxLength: 200, confirmLabel: 'Rename',
      });
      if (name === null) return;
      if (name === identity.name) return;   // unchanged: nothing is sent
      clearCardError(id);
      if (trigger) setBusy(trigger, true);
      try {
        await api.patch(`/api/voices/identities/${enc(id)}`, { name });
      } catch (err) {
        if (trigger) setBusy(trigger, false);
        if (err?.status === 409) {
          value = name;
          error = RENAME_CONFLICT;
          continue;
        }
        if (err?.kind === 'aborted') return;
        cardError(id, err, "Couldn't rename");
        return;
      }
      if (trigger) setBusy(trigger, false);
      toast(`Renamed ${identity.name} to ${name}`, { kind: 'success' });
      ctx.changed('named');
      await refreshAfterChange(indexOf(id), trigger ? trigger.id : '');
      return;
    }
  }

  /** Delete… (SE-04): confirm with the linked-presets copy, then DELETE the named voice. */
  async function remove(id, trigger) {
    const identity = find(id);
    if (!identity) return;
    const ok = await confirm({
      title: `Delete ${identity.name}?`,
      message: identity.design
        ? `Every version of ${identity.name} is removed. ${DESIGNED_DELETE}`
        : `Every version of ${identity.name} is removed. ${LINKED_COPY}`,
      confirmLabel: 'Delete',
      danger: true,
    });
    if (!ok) return;
    const index = indexOf(id);
    clearCardError(id);
    if (trigger) setBusy(trigger, true);
    let answer = null;
    try {
      answer = await api.del(`/api/voices/identities/${enc(id)}`);
    } catch (error) {
      if (trigger) setBusy(trigger, false);
      if (error?.kind === 'aborted') return;
      cardError(id, error, "Couldn't delete");
      return;
    }
    // A designed sample that presets or other voices still use stays, as a recording named after the voice.
    const moved = answer && typeof answer === 'object' && answer.moved_to ? String(answer.moved_to) : '';
    toast(moved ? `Deleted ${identity.name}. Its sample moved to Recordings as ${moved}` : `Deleted ${identity.name}`, { kind: 'success' });
    ctx.changed('named');
    ctx.changed('presets');
    if (moved) {
      ctx.changed('recordings');
      ctx.load('recordings');
    }
    await refreshAfterChange(index, trigger ? trigger.id : '');
  }

  function onMenu(action, item, button) {
    const id = (button.id.match(/^vo-named-(.+)-more$/) || [])[1];
    const identity = (list().items || []).find((i) => idPart(i.id) === id);
    if (!identity) return;
    const run = action === 'rename' ? rename(identity.id, button) : action === 'delete' ? remove(identity.id, button) : null;
    if (run) run.catch((error) => toast(error?.message || String(error), { kind: 'error' }));
  }

  /** The visible Rename: the same prompt as the menu's Rename…. */
  function renameFrom(button) {
    const id = button.getAttribute('data-id');
    if (!find(id)) return;
    rename(id, button).catch((error) => toast(error?.message || String(error), { kind: 'error' }));
  }

  /**
   * New preset: Speak opens with this voice for the model it is on when that has a version, else
   * its first model that has one, and Save preset… focused (intent new-preset). Unavailable says why.
   */
  function newPreset(button) {
    const id = button.getAttribute('data-id');
    const identity = find(id);
    if (!identity) return;
    const block = newPresetBlock(identity, ctx.models());
    if (block) {
      ctx.announce(block.reason);
      return;
    }
    setIntent('speak', { type: 'new-preset', id });
    ctx.navigate('speak');
  }

  /* ───────────── a designed voice ───────────── */

  function sampleSession() {
    const session = audio.current();
    return session && session.owner === 'voices' && session.id === SAMPLE_SESSION && session.controls ? session : null;
  }

  /** Play sample / Stop on each designed card, from the sample session and the microphone. */
  function patchSamples() {
    const player = app.get().player;
    const playing = !!(player && player.owner === 'voices' && player.id === SAMPLE_SESSION && player.playing);
    const recording = !!app.get().recording;
    for (const button of listEl.querySelectorAll('[data-action="named-sample"]')) {
      const id = button.getAttribute('data-id');
      const identity = find(id);
      const on = sampleOn === id && playing;
      const pending = samplePending === id;
      if (pending) button.setAttribute('aria-busy', 'true');
      else button.removeAttribute('aria-busy');
      if (pending || (recording && !on)) button.setAttribute('aria-disabled', 'true');
      else button.removeAttribute('aria-disabled');
      button.setAttribute('aria-label', `${on ? 'Stop sample' : 'Play sample'}: ${identity ? identity.name : ''}`);
      const label = button.querySelector('.btn__label');
      if (label) label.textContent = on ? 'Stop' : 'Play sample';
      const use = button.querySelector('use');
      if (use) use.setAttribute('href', `/static/icons/sprite.svg#i-${on ? 'stop' : 'play'}`);
    }
  }

  async function playSample(button) {
    const id = button.getAttribute('data-id');
    const identity = find(id);
    const sample = identity?.design?.sample;
    if (!sample || samplePending === id) return;
    const session = sampleSession();
    if (session && sampleOn === id) {
      const playing = typeof session.getState === 'function' && !!session.getState().playing;
      if (playing) session.controls.stop();
      else session.controls.play();
      return;
    }
    if (app.get().recording) {
      ctx.announce(STOP_RECORDING);
      return;
    }
    audio.prime();   // synchronously in the click: the clip plays after a network wait (iOS)
    clearCardError(id);
    const token = (sampleSeq += 1);
    samplePending = id;
    patchSamples();
    let media = null;
    try {
      media = await api.mediaUrl(`/api/voices/library/${enc(sample)}/audio`);
      if (token !== sampleSeq) {
        media.revoke();
        return;
      }
      samplePending = '';
      sampleOn = id;
      const release = media;
      await audio.playClip({
        id: SAMPLE_SESSION, owner: 'voices', title: identity.name, subtitle: 'Designed sample', src: media.url, ephemeral: true,
        onRelease: () => {
          release.revoke();
          if (sampleOn === id) {
            sampleOn = '';
            patchSamples();
          }
        },
      });
      media = null;
      patchSamples();
    } catch (error) {
      if (media) media.revoke();
      if (token !== sampleSeq) return;
      if (samplePending === id) samplePending = '';
      if (sampleOn === id) sampleOn = '';
      patchSamples();
      if (error?.kind === 'aborted') return;
      cardError(id, error, "Couldn't play the sample");
    }
  }

  /** The Voices list is no longer on screen: a sample (no mini-player) stops, one still loading never starts. */
  function stopSample() {
    sampleSeq += 1;
    samplePending = '';
    const session = sampleSession();
    if (sampleOn && session) session.controls.stop();
  }

  /** Use in Speak: Speak opens with this voice for its first available model (intent use-named-voice). */
  function use(button) {
    const id = button.getAttribute('data-id');
    const identity = find(id);
    if (!identity) return;
    const block = newPresetBlock(identity, ctx.models());
    if (block) {
      ctx.announce(block.reason);
      return;
    }
    const model = useModel(identity, ctx.models());
    setIntent('speak', model ? { type: 'use-named-voice', id, model } : { type: 'use-named-voice', id });
    ctx.navigate('speak');
  }

  /** Edit description: the design wizard, filled in from this voice. */
  function redesign(button) {
    const identity = find(button.getAttribute('data-id'));
    if (identity && identity.design) ctx.editDesign(identity);
  }

  const offs = [
    app.select((s) => {
      const p = s.player;
      return `${p && p.owner === 'voices' && p.id === SAMPLE_SESSION ? (p.playing ? 1 : 0) : '-'}|${s.recording ? 1 : 0}`;
    }, () => patchSamples()),
  ];

  return {
    onMenu,
    patchSamples,
    stopSample,
    destroy() {
      for (const off of offs.splice(0)) off();
    },
    actions: {
      'named-sample': (button) => playSample(button),
      'named-redesign': (button) => redesign(button),
      'named-use': (button) => use(button),
      'named-edit': (button) => edit(button),
      'named-remove': (button) => removeVersion(button),
      'named-rename': (button) => renameFrom(button),
      'named-new-preset': (button) => newPreset(button),
      'open-speak': () => ctx.navigate('speak'),
    },
  };
}
