import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';
import { assertAccessibleNames } from '../harness/a11y.js';

let env;
afterEach(() => env?.restore());

const BREEZE = {
  options: {
    cfg_scale: { type: 'number', label: 'Guidance', min: 1, max: 5, step: 0.5, default: null, description: 'How closely it follows the description. Above 3 can sound harsh.' },
    temperature: { type: 'number', label: 'Variation', min: 0.1, max: 1.3, step: 0.05, default: 0.9, description: 'Lower is steadier, higher is livelier.' },
  },
};
const SUPERTONIC = {
  options: { steps: { type: 'integer', label: 'Quality steps', min: 1, max: 32, step: 1, default: 8, description: 'Fewer is faster, more is cleaner.' } },
};

async function setup() {
  env = installBrowserEnv({ html: '<div id="host"></div>' });
  const mod = await importFresh('components/model-options.js');
  return { mod, host: document.getElementById('host') };
}

const field = (host, name) => host.querySelector(`[data-option="${name}"]`);
const slider = (host, name) => field(host, name).querySelector('input');
const out = (host, name) => field(host, name).querySelector('output');
const auto = (host, name) => field(host, name).querySelector('.model-option__auto');

describe('components/model-options', () => {
  it("renders each option as Speak's slider field: label, output, native range, hint", async () => {
    const { mod, host } = await setup();
    mod.mountModelOptions(host, SUPERTONIC, { idPrefix: 'sp-opt' });
    assert.equal(host.hidden, false);
    const steps = field(host, 'steps');
    assert.ok(steps.classList.contains('field'));
    assert.equal(steps.querySelector('label.field__label').getAttribute('for'), 'sp-opt-steps');
    assert.equal(steps.querySelector('label').textContent, 'Quality steps');
    const input = slider(host, 'steps');
    assert.deepEqual([input.getAttribute('type'), input.className, input.id, input.getAttribute('min'), input.getAttribute('max'), input.getAttribute('step')],
      ['range', 'slider', 'sp-opt-steps', '1', '32', '1']);
    assert.equal(input.getAttribute('aria-describedby'), 'sp-opt-steps-hint');
    assert.equal(steps.querySelector('#sp-opt-steps-hint').textContent, 'Fewer is faster, more is cleaner.');
    assert.equal(out(host, 'steps').textContent, '8');
    assert.equal(out(host, 'steps').getAttribute('for'), 'sp-opt-steps');
    assert.equal(auto(host, 'steps'), null, 'an option with a default has no Auto');
    assertAccessibleNames(host);
  });

  it('values() sends what is set and the defaults; a null default is Auto and sends nothing', async () => {
    const { mod, host } = await setup();
    const changes = [];
    const ctl = mod.mountModelOptions(host, BREEZE, { onChange: (values, name) => changes.push([name, values]) });
    assert.deepEqual(ctl.values(), { temperature: 0.9 }, 'Guidance starts at Auto');
    assert.equal(out(host, 'cfg_scale').textContent, 'Auto');
    assert.equal(slider(host, 'cfg_scale').getAttribute('aria-valuetext'), 'Auto');
    assert.equal(auto(host, 'cfg_scale').hidden, true, 'nothing to go back from');
    assert.equal(out(host, 'temperature').textContent, '0.90');

    env.input(slider(host, 'cfg_scale'), '2.5');
    assert.deepEqual(ctl.values(), { cfg_scale: 2.5, temperature: 0.9 });
    assert.equal(out(host, 'cfg_scale').textContent, '2.5');
    assert.equal(auto(host, 'cfg_scale').hidden, false);
    env.input(slider(host, 'temperature'), '0.55');
    assert.deepEqual(changes.map(([name]) => name), ['cfg_scale', 'temperature']);
    assert.deepEqual(changes.at(-1)[1], { cfg_scale: 2.5, temperature: 0.55 });

    env.click(auto(host, 'cfg_scale'));
    assert.deepEqual(ctl.values(), { temperature: 0.55 }, 'Auto again: Guidance is left out');
    assert.equal(out(host, 'cfg_scale').textContent, 'Auto');
    assert.equal(auto(host, 'cfg_scale').hidden, true);
    assert.equal(document.activeElement, slider(host, 'cfg_scale'), 'focus goes to the slider, not to the body');
    assert.deepEqual(changes.at(-1), ['cfg_scale', { temperature: 0.55 }]);
    assertAccessibleNames(host);
  });

  it('set() and the starting values are held to the range and its steps; unknown or empty ones are defaults', async () => {
    const { mod, host } = await setup();
    const ctl = mod.mountModelOptions(host, BREEZE, { values: { cfg_scale: 9, temperature: 0.62, steps: 4 } });
    assert.deepEqual(ctl.values(), { cfg_scale: 5, temperature: 0.6 });
    ctl.set({ cfg_scale: 1.2 });
    assert.deepEqual(ctl.values(), { cfg_scale: 1, temperature: 0.9 }, 'what set() leaves out goes back to its default');
    ctl.set({ cfg_scale: null, temperature: 'x' });
    assert.deepEqual(ctl.values(), { temperature: 0.9 });
    assert.equal(out(host, 'cfg_scale').textContent, 'Auto');
    const steps = mod.mountModelOptions(document.createElement('div'), SUPERTONIC, { values: { steps: 12.6 } });
    assert.deepEqual(steps.values(), { steps: 13 }, 'an integer option stays whole');
  });

  it('a model without options hides the host; destroy() empties it and unbinds', async () => {
    const { mod, host } = await setup();
    const none = mod.mountModelOptions(host, { voice_clone: true });
    assert.equal(host.hidden, true);
    assert.equal(none.has(), false);
    assert.deepEqual(none.values(), {});
    const ctl = mod.mountModelOptions(host, BREEZE);
    assert.equal(host.hidden, false);
    assert.equal(ctl.has(), true);
    const input = slider(host, 'temperature');
    ctl.destroy();
    assert.equal(host.hidden, true);
    assert.equal(host.querySelector('input'), null);
    env.input(input, '0.3');
    assert.deepEqual(ctl.values(), { temperature: 0.9 }, 'a destroyed control no longer listens');
  });

  it('optionSpecs keeps only integer and number options with a range; ids are unique per mount', async () => {
    const { mod } = await setup();
    const specs = mod.optionSpecs({ options: {
      ...BREEZE.options,
      voice: { type: 'string', min: 0, max: 1 },
      open: { type: 'number', min: 0 },
      flat: { type: 'number', min: 2, max: 2 },
      loose: { type: 'number', min: 0, max: 2 },
    } });
    assert.deepEqual(specs.map((s) => s.name), ['cfg_scale', 'temperature', 'loose']);
    assert.deepEqual(specs[0], { name: 'cfg_scale', type: 'number', label: 'Guidance', min: 1, max: 5, step: 0.5, default: null,
      description: 'How closely it follows the description. Above 3 can sound harsh.' });
    assert.equal(specs[2].step, 0.02, 'a number option without a step gets a hundredth of its range');
    assert.deepEqual(mod.optionSpecs(null), []);
    const a = document.createElement('div');
    const b = document.createElement('div');
    mod.mountModelOptions(a, SUPERTONIC);
    mod.mountModelOptions(b, SUPERTONIC);
    assert.notEqual(a.querySelector('input').id, b.querySelector('input').id);
  });
});
