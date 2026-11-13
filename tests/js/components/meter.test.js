import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';

let env;
afterEach(() => env?.restore());

const MARKUP = '<div class="meter-row"><div class="meter" id="tr-meter" role="meter" aria-label="Input level" aria-valuemin="0" aria-valuemax="100"></div><span class="meter__label"></span></div>';

async function setup(bars = 9) {
  env = installBrowserEnv({ html: MARKUP });
  const mod = await importFresh('components/meter.js');
  const el = document.getElementById('tr-meter');
  const meter = mod.createLevelMeter(el, { bars });
  return { mod, el, meter, label: el.nextElementSibling, lit: () => el.querySelectorAll('.meter__bar.is-on').length };
}

describe('components/meter', () => {
  it('zones: quiet below -40 dBFS, good -40 to -12, loud above -12, clip at peak >= 0.999', async () => {
    const { mod } = await setup();
    assert.equal(mod.zoneFor({ db: -50, peak: 0.01 }), 'quiet');
    assert.equal(mod.zoneFor({ db: -40, peak: 0.1 }), 'good');
    assert.equal(mod.zoneFor({ db: -12, peak: 0.5 }), 'good');
    assert.equal(mod.zoneFor({ db: -11, peak: 0.5 }), 'loud');
    assert.equal(mod.zoneFor({ db: -20, peak: 0.999 }), 'clip');
    assert.equal(mod.levelPercent(-60), 0);
    assert.equal(mod.levelPercent(-30), 50);
    assert.equal(mod.levelPercent(0), 100);
    assert.equal(mod.levelPercent(-Infinity), 0);
  });

  it('shows the level as bars and a visible text label (never colour alone)', async () => {
    const { el, meter, label, lit } = await setup();
    assert.equal(el.querySelectorAll('.meter__bar').length, 9);
    meter.set({ db: -24, peak: 0.2 });
    assert.equal(el.getAttribute('data-zone'), 'good');
    assert.equal(label.textContent, 'Good level');
    assert.equal(el.getAttribute('aria-valuenow'), '60');
    assert.equal(el.getAttribute('aria-valuetext'), 'Good level');
    assert.equal(lit(), 5);
    env.clock.tick(300);
    meter.set({ db: -3, peak: 1 });
    assert.equal(label.textContent, 'Clipping');
    assert.equal(el.getAttribute('data-zone'), 'clip');
  });

  it('bars update at most every 66 ms and the label at most every 250 ms, with a trailing update', async () => {
    const { el, meter, label, lit } = await setup();
    meter.set({ db: -50 });
    assert.equal(label.textContent, 'Too quiet');
    env.clock.tick(10);
    meter.set({ db: -6 });
    assert.equal(lit(), 2, 'bars wait for their 66 ms slot');
    assert.equal(label.textContent, 'Too quiet');
    env.clock.tick(60);
    assert.equal(lit(), 8, 'the trailing update shows the last level');
    assert.equal(el.getAttribute('data-zone'), 'loud');
    assert.equal(label.textContent, 'Too quiet', 'the label waits for its 250 ms slot');
    env.clock.tick(200);
    assert.equal(label.textContent, 'Too loud');
  });

  it('reset() returns to a flat, idle meter and creates its label when missing', async () => {
    env = installBrowserEnv({ html: '<div class="meter" id="m"></div>' });
    const { createLevelMeter } = await importFresh('components/meter.js');
    const el = document.getElementById('m');
    const meter = createLevelMeter(el, { bars: 5 });
    assert.equal(el.getAttribute('role'), 'meter');
    assert.ok(el.nextElementSibling.classList.contains('meter__label'));
    meter.set({ db: -20 });
    meter.reset();
    assert.equal(el.querySelectorAll('.meter__bar.is-on').length, 0);
    assert.equal(el.getAttribute('data-zone'), 'idle');
    assert.equal(el.nextElementSibling.textContent, '');
    meter.destroy();
  });
});
