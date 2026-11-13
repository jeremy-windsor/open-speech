import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';
import { FakePointerEvent } from '../harness/dom.js';

let env;
let tips;
afterEach(() => {
  tips?.destroy();
  env?.restore();
});

const MARKUP = `<main id="main">
  <button type="button" class="icon-btn" id="close" aria-label="Close" data-tooltip="Close">x</button>
  <button type="button" class="icon-btn" id="skip" aria-label="Forward 30 seconds" data-tooltip="Skip ahead">30</button>
  <button type="button" id="plain">Plain</button>
</main>
<div id="app-tooltip" class="tooltip" role="tooltip" hidden></div>`;

async function setup() {
  env = installBrowserEnv({ html: MARKUP, viewport: { width: 400, height: 800 } });
  const mod = await importFresh('components/tooltip.js');
  mod.__resetForTests();
  tips = mod.initTooltips(document);
  return { mod, tip: document.getElementById('app-tooltip') };
}

const over = (el, type = 'pointerover', init = {}) => el.dispatchEvent(new FakePointerEvent(type, { bubbles: true, ...init }));

describe('components/tooltip', () => {
  it('shows at once on keyboard focus and hides on blur', async () => {
    const { tip } = await setup();
    const close = document.getElementById('close');
    close.focus();
    assert.equal(tip.hidden, false);
    assert.equal(tip.textContent, 'Close');
    assert.equal(close.getAttribute('aria-describedby'), null, 'the same text as the aria-label is not described twice');
    document.getElementById('plain').focus();
    assert.equal(tip.hidden, true);
  });

  it('shows after 500 ms of hover, stays while the pointer is over the tooltip, and closes when it leaves both', async () => {
    const { tip } = await setup();
    const skip = document.getElementById('skip');
    over(skip);
    env.clock.tick(400);
    assert.equal(tip.hidden, true, 'not before the hover delay');
    env.clock.tick(100);
    assert.equal(tip.hidden, false);
    assert.equal(skip.getAttribute('aria-describedby'), 'app-tooltip', 'a different text is linked with aria-describedby');
    over(skip, 'pointerout', { relatedTarget: tip });
    env.clock.tick(500);
    assert.equal(tip.hidden, false, 'moving onto the tooltip keeps it (WCAG 1.4.13 hoverable)');
    tip.dispatchEvent(new FakePointerEvent('pointerenter'));
    tip.dispatchEvent(new FakePointerEvent('pointerleave'));
    env.clock.tick(200);
    assert.equal(tip.hidden, true);
    assert.equal(skip.getAttribute('aria-describedby'), null, 'the description goes with it');
  });

  it('Esc dismisses it without moving focus (WCAG 1.4.13), and touch never opens it', async () => {
    const { tip } = await setup();
    const close = document.getElementById('close');
    close.focus();
    assert.equal(tip.hidden, false);
    const down = env.key('Escape');
    assert.equal(tip.hidden, true);
    assert.equal(down.defaultPrevented, true);
    assert.equal(document.activeElement, close);
    over(document.getElementById('skip'), 'pointerover', { pointerType: 'touch' });
    env.clock.tick(1000);
    assert.equal(tip.hidden, true);
  });

  it('placeTooltip keeps it inside the viewport and flips below when there is no room above', async () => {
    const { mod } = await setup();
    const near = { top: 4, bottom: 48, left: 2, right: 46, width: 44, height: 44 };
    const spot = mod.placeTooltip(near, { width: 100, height: 30 }, { placement: 'top', width: 400, height: 800 });
    assert.equal(spot.placement, 'bottom');
    assert.equal(spot.x, 8, 'clamped to the edge');
    const right = mod.placeTooltip({ top: 100, bottom: 144, left: 12, right: 56, width: 44, height: 44 }, { width: 80, height: 30 }, { placement: 'right', width: 400, height: 800 });
    assert.deepEqual([right.placement, right.x, right.y], ['right', 64, 107]);
  });
});
