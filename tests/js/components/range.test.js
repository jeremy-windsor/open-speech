import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';

let env;
afterEach(() => env?.restore());

const MARKUP = `<label for="sp-speed">Speed</label>
<input type="range" class="slider" id="sp-speed" min="0.5" max="2" step="0.05" value="1.25">
<output id="sp-speed-out" for="sp-speed"></output>`;

async function setup() {
  env = installBrowserEnv({ html: MARKUP });
  const { bindRange } = await importFresh('components/range.js');
  const format = await importFresh('core/format.js');
  return { bindRange, format, input: document.getElementById('sp-speed'), output: document.getElementById('sp-speed-out') };
}

describe('components/range', () => {
  it('keeps the <output>, aria-valuetext (spoken) and --fill in step with the native slider', async () => {
    const { bindRange, format, input, output } = await setup();
    bindRange(input, { output, format: format.speed, spoken: format.spokenSpeed });
    assert.equal(output.textContent, '1.25×');
    assert.equal(input.getAttribute('aria-valuetext'), '1.25 times');
    assert.equal(input.style.getPropertyValue('--fill'), '50%');
    env.input(input, '2');
    assert.equal(output.textContent, '2×');
    assert.equal(input.getAttribute('aria-valuetext'), '2 times');
    assert.equal(input.style.getPropertyValue('--fill'), '100%');
    env.change(input, '0.5');
    assert.equal(input.style.getPropertyValue('--fill'), '0%');
    assert.equal(input.getAttribute('type'), 'range', 'it stays a native range, so VoiceOver can adjust it');
  });

  it('set() and update() cover programmatic changes; spoken defaults to format; fill can be off', async () => {
    const { bindRange, input, output } = await setup();
    const range = bindRange(input, { output, format: (v) => `${v} s`, fill: false });
    assert.equal(input.getAttribute('aria-valuetext'), '1.25 s');
    range.set(1.5);
    assert.equal(output.textContent, '1.5 s');
    input.value = '1';
    range.update();
    assert.equal(output.textContent, '1 s');
    assert.equal(input.style.getPropertyValue('--fill'), '', 'no --fill when fill is false');
    range.destroy();
    env.input(input, '2');
    assert.equal(output.textContent, '1 s', 'destroy() unbinds');
  });

  it('works without an output and clamps --fill to 0–100%', async () => {
    const { bindRange, input } = await setup();
    input.setAttribute('max', '1');
    input.setAttribute('min', '1');
    bindRange(input);
    assert.equal(input.style.getPropertyValue('--fill'), '0%', 'an empty span does not divide by zero');
    assert.equal(input.getAttribute('aria-valuetext'), '1.25');
  });
});
