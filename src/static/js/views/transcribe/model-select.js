/**
 * The Transcribe model select (TR-01): the faster-whisper models from the inventory, sorted
 * loaded > downloaded > available > missing, then id (models.sttModels), each with its state in words
 * from models.uiState. The configured default is selected until the listener picks another model.
 * The select follows app.models and app.ops, so a download or load from Models shows here at once
 * (BUG-11: the old list was built once at start-up).
 * @module views/transcribe/model-select
 */
import { html, render } from '../../core/dom.js';
import { app } from '../../core/store.js';
import { sttModels, shortName, uiState } from '../../core/models.js';

/**
 * PURE: the options for a model inventory.
 * @param {object[]} list app.models.list
 * @param {object} [ops] app.ops
 * @returns {{id: string, name: string, state: string, label: string, ready: boolean}[]}
 */
export function sttOptions(list, ops = {}) {
  return sttModels(list).map((m) => {
    const ui = uiState(m, (ops || {})[m.id] || null, { list });
    const name = shortName(m);
    return { id: m.id, name, state: ui.label, label: `${name} — ${ui.label}`, ready: ui.key === 'loaded', key: ui.key };
  });
}

/**
 * PURE: the model to select: the listener's pick when it is still listed, else the configured default,
 * else the first.
 * @param {{id: string}[]} options
 * @param {{picked?: string, configuredDefault?: string}} [choice]
 * @returns {string} '' when there is none
 */
export function pickStt(options, { picked = '', configuredDefault = '' } = {}) {
  const has = (id) => !!id && options.some((o) => o.id === id);
  if (has(picked)) return picked;
  if (has(configuredDefault)) return configuredDefault;
  return options[0]?.id || '';
}

/**
 * PURE: the hint under the select for the chosen model's state.
 * @param {{key: string}|undefined} option
 */
export function modelHint(option) {
  if (!option) return '';
  switch (option.key) {
    case 'not-downloaded': return 'Not downloaded yet. It downloads when you start, which can take a few minutes.';
    case 'downloaded': return 'It loads when you start.';
    case 'downloading': return 'Downloading now. You can start once it is ready.';
    case 'loading': return 'Loading now.';
    case 'error': return "The last operation on this model failed. Starting tries again.";
    default: return '';
  }
}

/**
 * Keep a <select> in step with the inventory. The options are rebuilt only when the set of models changes;
 * otherwise only their text is patched, so an open select is never closed under the pointer.
 * setLocked(true) keeps it off while a recording or a file runs (a pick then would change nothing for it);
 * an inventory refresh meanwhile leaves it off.
 * @param {HTMLSelectElement} select
 * @param {{hint?: HTMLElement, onChange?: (state: {value: string, options: object[], loaded: boolean}) => void}} [options]
 * @returns {{value: () => string, options: () => object[], refresh: () => void, setLocked: (locked: boolean) => void,
 *   destroy: () => void}}
 */
export function bindModelSelect(select, { hint, onChange } = {}) {
  let picked = '';
  let options = [];
  let shownIds = '';
  let locked = false;

  function paint() {
    const { models, ops } = app.get();
    options = sttOptions(models.list, ops);
    const loaded = models.fetchedAt > 0 || models.list.length > 0;
    const value = pickStt(options, { picked, configuredDefault: models.defaults.stt });
    const ids = options.map((o) => o.id).join('\n');
    if (!options.length) {
      const text = loaded ? 'No transcription model' : (models.error ? 'Models didn\'t load' : 'Loading models…');
      if (shownIds !== `empty:${text}`) {
        render(select, html`<option value="">${text}</option>`);
        shownIds = `empty:${text}`;
      }
      select.disabled = true;
    } else {
      if (ids !== shownIds) {
        render(select, html`${options.map((o) => html`<option value="${o.id}">${o.label}</option>`)}`);
        shownIds = ids;
      } else {
        const els = Array.from(select.querySelectorAll('option'));
        options.forEach((o, i) => {
          if (els[i] && els[i].textContent !== o.label) els[i].textContent = o.label;
        });
      }
      select.disabled = locked;
      if (select.value !== value) select.value = value;
    }
    if (hint) {
      const text = modelHint(options.find((o) => o.id === value));
      if (hint.textContent !== text) hint.textContent = text;
      hint.hidden = !text;
    }
    if (typeof onChange === 'function') onChange({ value, options, loaded, error: models.error || '' });
  }

  const onSelect = () => {
    picked = select.value;
    paint();
  };
  select.addEventListener('change', onSelect);
  const off = app.select((s) => [s.models, s.ops], paint, {
    equals: (a, b) => a[0] === b[0] && a[1] === b[1],
  });
  paint();

  return {
    value: () => pickStt(options, { picked, configuredDefault: app.get().models.defaults.stt }),
    options: () => options.slice(),
    refresh: paint,
    setLocked(value) {
      locked = !!value;
      if (options.length && select.disabled !== locked) select.disabled = locked;
    },
    destroy() {
      off();
      select.removeEventListener('change', onSelect);
    },
  };
}
