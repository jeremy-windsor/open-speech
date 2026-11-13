import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';

let env;
afterEach(() => env?.restore());

const GROUP = `<div role="radiogroup" aria-label="Speed" id="g">
  <button type="button" role="radio" aria-checked="false">0.75×</button>
  <button type="button" role="radio" aria-checked="true">1×</button>
  <button type="button" role="radio" aria-checked="false">1.25×</button>
  <button type="button" role="radio" aria-checked="false" disabled>1.5×</button>
  <button type="button" role="radio" aria-checked="false">2×</button>
</div>`;

async function setup(options = {}) {
  env = installBrowserEnv({ html: GROUP });
  const { bindRoving } = await importFresh('components/roving.js');
  const changes = [];
  const group = document.getElementById('g');
  const roving = bindRoving(group, { onChange: (el, i) => changes.push([el.textContent, i]), ...options });
  const items = Array.from(group.querySelectorAll('[role=radio]'));
  return { roving, items, changes, group };
}

const tabStops = (items) => items.filter((el) => el.getAttribute('tabindex') === '0').map((el) => el.textContent);
const checked = (items) => items.filter((el) => el.getAttribute('aria-checked') === 'true').map((el) => el.textContent);

describe('components/roving', () => {
  it('[GS-14] one tab stop; ArrowLeft/Right wrap, Home/End; the target gets focus and is activated', async () => {
    const { items, changes } = await setup();
    assert.deepEqual(tabStops(items), ['1×'], 'the checked item is the only tab stop');
    items[1].focus();
    env.key('ArrowRight');
    assert.equal(document.activeElement, items[2]);
    assert.deepEqual(checked(items), ['1.25×']);
    assert.deepEqual(tabStops(items), ['1.25×']);
    env.key('ArrowRight');
    assert.equal(document.activeElement.textContent, '2×', 'the disabled item is skipped');
    env.key('ArrowRight');
    assert.equal(document.activeElement.textContent, '0.75×', 'it wraps to the first');
    env.key('ArrowLeft');
    assert.equal(document.activeElement.textContent, '2×', 'and back to the last');
    env.key('Home');
    assert.equal(document.activeElement.textContent, '0.75×');
    env.key('End');
    assert.equal(document.activeElement.textContent, '2×');
    assert.deepEqual(checked(items), ['2×']);
    assert.deepEqual(changes.map((c) => c[0]), ['1.25×', '2×', '0.75×', '2×', '0.75×', '2×']);
  });

  it('[GS-14] wrap: false stops at the ends; vertical groups use ArrowUp/ArrowDown', async () => {
    const { items } = await setup({ wrap: false, orientation: 'vertical' });
    items[4].focus();
    env.key('ArrowDown');
    assert.equal(document.activeElement, items[4], 'no wrap at the end');
    env.key('ArrowRight');
    assert.equal(document.activeElement, items[4], 'horizontal keys do nothing in a vertical group');
    env.key('ArrowUp');
    assert.equal(document.activeElement, items[2]);
  });

  it('activateOnMove: false moves focus and the tab stop only; Space or a click activates', async () => {
    const { items, changes } = await setup({ activateOnMove: false });
    items[1].focus();
    env.key('ArrowLeft');
    assert.equal(document.activeElement, items[0]);
    assert.deepEqual(checked(items), ['1×'], 'not activated by moving');
    assert.deepEqual(tabStops(items), ['0.75×']);
    env.key(' ');
    assert.deepEqual(checked(items), ['0.75×'], 'Space clicks the focused button');
    env.click(items[4]);
    assert.deepEqual(checked(items), ['2×']);
    assert.equal(changes.length, 2);
  });

  it('select() sets the state without calling onChange; refresh() repairs the tab stop; destroy() unbinds', async () => {
    const { roving, items, changes, group } = await setup();
    roving.select(2);
    assert.deepEqual(checked(items), ['1.25×']);
    assert.equal(changes.length, 0);
    items.forEach((el) => el.setAttribute('tabindex', '0'));
    roving.refresh();
    assert.deepEqual(tabStops(items), ['1.25×']);
    assert.equal(roving.current(), items[2]);
    roving.destroy();
    items[2].focus();
    env.key('ArrowRight');
    assert.equal(document.activeElement, items[2]);
    assert.equal(group.__listenerCount('keydown'), 0);
  });
});
