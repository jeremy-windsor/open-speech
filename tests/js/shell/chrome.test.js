import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { FakeResizeObserver } from '../harness/fakes.js';
import { shellEnv, load, settle } from './helpers.js';

let env;
afterEach(() => env?.restore());

const rect = (top, height, width = 390) => ({ top, left: 0, width, height, bottom: top + height, right: width });

async function setup({ desktop = false } = {}) {
  env = await shellEnv({ viewport: { width: desktop ? 1440 : 390, height: 844 }, media: { 'min-width: 960px': desktop } });
  const chrome = await load('shell/chrome.js');
  const toast = await load('core/toast.js');
  const root = document.documentElement;
  const vars = () => ({
    top: root.style.getPropertyValue('--chrome-top'),
    base: root.style.getPropertyValue('--chrome-bottom-base'),
    bottom: root.style.getPropertyValue('--chrome-bottom'),
  });
  return { chrome, toast, vars };
}

describe('shell/chrome', () => {
  it('computeMetrics: the top chrome reaches down, the bottom chrome reaches up; hidden chrome counts for nothing', async () => {
    const { chrome } = await setup();
    const m = chrome.computeMetrics({ viewportHeight: 844, tops: [rect(0, 52)], bottoms: [rect(780, 64), rect(708, 64)], toasts: null });
    assert.deepEqual(m, { top: 52, bottomBase: 136, bottom: 136 });
    assert.deepEqual(chrome.computeMetrics({ viewportHeight: 844, tops: [rect(0, 0)], bottoms: [rect(844, 0)] }), { top: 0, bottomBase: 0, bottom: 0 });
  });

  it('[REV-21] toasts are bottom-anchored: on a phone their stack joins --chrome-bottom (never --chrome-top), so a focused element is never hidden behind one', async () => {
    const { chrome, toast, vars } = await setup();
    document.getElementById('app-topbar').__rect = rect(0, 52);
    document.getElementById('app-bottombar').__rect = rect(780, 64);
    chrome.initChrome();
    assert.deepEqual(vars(), { top: '52px', base: '64px', bottom: '64px' });
    toast.toast('Saved.', { kind: 'success' });
    document.getElementById('app-toasts').__rect = rect(700, 72);
    for (const ro of FakeResizeObserver.instances) ro.__trigger();
    await settle(env);
    assert.deepEqual(vars(), { top: '52px', base: '64px', bottom: '144px' }, 'the toast stack raises the scroll padding, not the toasts');
    toast.dismissAll();
    document.getElementById('app-toasts').__rect = rect(844, 0);
    chrome.measure();
    assert.equal(vars().bottom, '64px');
  });

  it('[REV-21] on desktop the toast stack (bottom right, over the content column) joins the scroll padding too, above the docked mini-player', async () => {
    // With the stack left out at 960 px and up, an error toast (it stays until dismissed)
    // hid a control Tab brought in from below completely (WCAG 2.4.11).
    const { chrome, toast, vars } = await setup({ desktop: true });
    const mp = document.getElementById('app-miniplayer');
    mp.hidden = false;
    mp.__rect = rect(844 - 16 - 64, 64, 1100);
    toast.toast("Couldn't save.", { kind: 'error' });
    document.getElementById('app-toasts').__rect = rect(700, 72, 380);
    chrome.initChrome();
    assert.deepEqual(vars(), { top: '0px', base: '80px', bottom: '144px' }, 'the toasts are positioned from the base and counted in the padding');
    toast.dismissAll();
    document.getElementById('app-toasts').__rect = rect(844, 0, 380);
    chrome.measure();
    assert.deepEqual(vars(), { top: '0px', base: '80px', bottom: '80px' }, 'a dismissed stack drops out again');
    mp.hidden = true;
    chrome.measure();
    toast.toast("Couldn't save.", { kind: 'error' });
    document.getElementById('app-toasts').__rect = rect(844 - 16 - 85, 85, 380);
    chrome.measure();
    assert.deepEqual(vars(), { top: '0px', base: '0px', bottom: '101px' }, 'without a mini-player the stack alone sets the padding');
  });

  it('immersive mode: the hidden shell bars drop out and the Reader bars (data-chrome) count instead', async () => {
    const { chrome, vars } = await setup();
    const topbar = document.getElementById('app-topbar');
    const bottombar = document.getElementById('app-bottombar');
    topbar.__rect = rect(0, 0);
    bottombar.__rect = rect(844, 0);
    const readerTop = document.createElement('header');
    readerTop.setAttribute('data-chrome', 'top');
    readerTop.__rect = rect(0, 56);
    const transport = document.createElement('div');
    transport.setAttribute('data-chrome', 'bottom');
    transport.__rect = rect(700, 144);
    document.getElementById('view-reader').append(readerTop, transport);
    document.getElementById('view-reader').hidden = false;
    chrome.initChrome();
    assert.deepEqual(vars(), { top: '56px', base: '144px', bottom: '144px' });
    document.getElementById('view-reader').hidden = true;
    chrome.measure();
    assert.deepEqual(vars(), { top: '0px', base: '0px', bottom: '0px' }, 'chrome inside a hidden view does not count');
  });
});
