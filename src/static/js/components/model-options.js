/**
 * A model's own options (capabilities.options, such as Supertonic's steps or Breeze's Guidance and Variation) as
 * sliders in a host element. An option whose default is null starts at "Auto": it sends nothing until it is moved,
 * and its Auto button takes it back. The host is hidden for a model without options.
 * @module components/model-options
 */
import { html, render } from '../core/dom.js';
import { bindRange } from './range.js';

const TYPES = new Set(['integer', 'number']);
let mounts = 0;

const finite = (value) => value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value));

/**
 * PURE: the options a model declares that can be shown as a slider: integer or number, with a min and a max.
 * @param {object} caps a model's capabilities
 * @returns {{name: string, type: string, label: string, min: number, max: number, step: number,
 *   default: number|null, description: string}[]}
 */
export function optionSpecs(caps) {
  const declared = caps && caps.options && typeof caps.options === 'object' ? caps.options : {};
  const out = [];
  for (const [name, spec] of Object.entries(declared)) {
    if (!spec || !TYPES.has(spec.type) || !finite(spec.min) || !finite(spec.max)) continue;
    const min = Number(spec.min);
    const max = Number(spec.max);
    if (!(max > min)) continue;
    const integer = spec.type === 'integer';
    const step = finite(spec.step) && Number(spec.step) > 0 ? Number(spec.step) : (integer ? 1 : (max - min) / 100);
    out.push({
      name,
      type: spec.type,
      label: String(spec.label || name),
      min,
      max,
      step: integer ? Math.max(1, Math.round(step)) : step,
      default: finite(spec.default) ? Number(spec.default) : null,
      description: String(spec.description || ''),
    });
  }
  return out;
}

function decimals(step) {
  const text = String(step);
  return text.includes('.') ? text.split('.')[1].length : 0;
}

/**
 * PURE: `value` on the option's range and steps, or null when it isn't a number.
 * @param {{min: number, max: number, step: number}} spec from optionSpecs
 */
export function snapOption(spec, value) {
  if (!finite(value)) return null;
  const n = Math.min(spec.max, Math.max(spec.min, Number(value)));
  const stepped = spec.min + Math.round((n - spec.min) / spec.step) * spec.step;
  return Number(Math.min(spec.max, stepped).toFixed(decimals(spec.step)));
}

/**
 * Render a model's options into `host` and keep their values.
 * @param {HTMLElement} host
 * @param {object} caps the model's capabilities
 * @param {{values?: object, idPrefix?: string, onChange?: (values: object, name: string) => void}} [options]
 *   values: the starting values (an option left out starts at its default, or Auto)
 * @returns {{values: () => object, set: (values: object) => void, has: () => boolean, destroy: () => void}}
 *   values(): what a request sends ({cfg_scale: 2.5, temperature: 0.9}); an option at Auto is left out
 */
export function mountModelOptions(host, caps, { values: initial = {}, idPrefix = '', onChange = null } = {}) {
  const specs = optionSpecs(caps);
  const prefix = idPrefix || `model-option-${(mounts += 1)}`;
  const state = {};
  const id = (spec) => `${prefix}-${spec.name}`;
  render(host, html`<div class="model-options">${specs.map((spec) => html`
    <div class="field model-option" data-option="${spec.name}">
      <div class="model-option__head">
        <label class="field__label" for="${id(spec)}">${spec.label}</label>
        <span class="model-option__end">
          ${spec.default === null ? html`<button type="button" class="btn btn--ghost btn--sm model-option__auto" data-auto="${spec.name}" aria-label="${`${spec.label}: back to Auto`}" hidden>Auto</button>` : ''}
          <output class="model-option__out tabular" for="${id(spec)}" aria-live="off"></output>
        </span>
      </div>
      <input type="range" class="slider" id="${id(spec)}" min="${spec.min}" max="${spec.max}" step="${spec.step}"${spec.description ? html` aria-describedby="${`${id(spec)}-hint`}"` : ''}>
      ${spec.description ? html`<p class="field__hint" id="${`${id(spec)}-hint`}">${spec.description}</p>` : ''}
    </div>`)}</div>`);
  host.hidden = specs.length === 0;

  const rows = specs.map((spec) => {
    const field = host.querySelector(`[data-option="${spec.name}"]`);
    const input = field.querySelector('input');
    const output = field.querySelector('output');
    const auto = field.querySelector('[data-auto]');
    const text = (value) => (state[spec.name] === null ? 'Auto' : value.toFixed(decimals(spec.step)));
    const range = bindRange(input, { output, format: text });
    return { spec, input, auto, range };
  });

  function show(row) {
    const value = state[row.spec.name];
    // At Auto the thumb rests mid-range; the output and aria-valuetext say Auto.
    row.range.set(value === null ? snapOption(row.spec, (row.spec.min + row.spec.max) / 2) : value);
    if (row.auto) row.auto.hidden = value === null;
  }

  function set(values = {}) {
    for (const row of rows) {
      const given = values ? values[row.spec.name] : undefined;
      state[row.spec.name] = snapOption(row.spec, given) ?? row.spec.default;
      show(row);
    }
  }

  function current() {
    const out = {};
    for (const row of rows) if (state[row.spec.name] !== null) out[row.spec.name] = state[row.spec.name];
    return out;
  }

  const listeners = [];
  const listen = (el, type, fn) => { el.addEventListener(type, fn); listeners.push(() => el.removeEventListener(type, fn)); };
  for (const row of rows) {
    listen(row.input, 'input', () => {
      state[row.spec.name] = snapOption(row.spec, row.input.value);
      show(row);
      if (onChange) onChange(current(), row.spec.name);
    });
    if (row.auto) {
      listen(row.auto, 'click', () => {
        state[row.spec.name] = null;
        show(row);
        row.input.focus();
        if (onChange) onChange(current(), row.spec.name);
      });
    }
  }
  set(initial);

  return {
    values: current,
    set,
    has: () => rows.length > 0,
    destroy() {
      for (const off of listeners.splice(0)) off();
      for (const row of rows) row.range.destroy();
      render(host, html``);
      host.hidden = true;
    },
  };
}
