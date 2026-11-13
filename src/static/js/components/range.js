/**
 * Native range binding: keeps an <output>, aria-valuetext and the --fill custom property
 * (the filled part of the track, 0–100%) in step with an <input type="range">. Every slider uses it, the
 * Reader scrubber included, so keyboards and iOS VoiceOver (swipe up and down) adjust them natively.
 * @module components/range
 */

/**
 * @param {HTMLInputElement} input
 * @param {{output?: HTMLElement|null, format?: (value: number) => string, spoken?: ((value: number) => string)|null,
 *   fill?: boolean}} [options] format: the visible text ('1.25×'); spoken: aria-valuetext ('1.25 times'),
 *   format when omitted
 * @returns {{update: () => void, set: (value: number) => void, destroy: () => void}}
 */
export function bindRange(input, { output = null, format = (value) => String(value), spoken = null, fill = true } = {}) {
  function update() {
    const value = Number(input.value);
    const min = Number(input.min || input.getAttribute('min') || 0);
    const maxAttr = input.max || input.getAttribute('max');
    const max = Number(maxAttr === '' || maxAttr === null || maxAttr === undefined ? 100 : maxAttr);
    const text = format(value);
    if (output && output.textContent !== text) output.textContent = text;
    input.setAttribute('aria-valuetext', (typeof spoken === 'function' ? spoken : format)(value));
    if (fill) {
      const span = max - min;
      const pct = span > 0 ? Math.min(100, Math.max(0, ((value - min) / span) * 100)) : 0;
      input.style.setProperty('--fill', `${Math.round(pct * 100) / 100}%`);
    }
  }
  input.addEventListener('input', update);
  input.addEventListener('change', update);
  update();
  return {
    update,
    set(value) {
      input.value = String(value);
      update();
    },
    destroy() {
      input.removeEventListener('input', update);
      input.removeEventListener('change', update);
    },
  };
}
