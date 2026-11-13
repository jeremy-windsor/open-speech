import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';
import { assertAccessibleNames } from '../harness/a11y.js';

let env;
afterEach(() => env?.restore());

const SHELL = '<main id="main" tabindex="-1"><button id="trigger" type="button">x</button></main><div class="toasts" id="app-toasts"><div id="app-toasts-polite" role="status" aria-live="polite"></div><div id="app-toasts-alert" role="alert" aria-live="assertive"></div></div>';

async function setup() {
  env = installBrowserEnv({ html: SHELL });
  return importFresh('core/toast.js');
}

const polite = () => document.getElementById('app-toasts-polite').querySelectorAll('.toast');
const alerts = () => document.getElementById('app-toasts-alert').querySelectorAll('.toast');

describe('core/toast', () => {
  it('[TE-01] info, success and warning go in the polite region and leave after 6 s', async () => {
    const { toast } = await setup();
    const t = toast('Speech generated', { kind: 'success' });
    toast('Heads up', { kind: 'warning' });
    toast('Note');
    assert.equal(polite().length, 3);
    assert.equal(t.el.className, 'toast toast--success');
    assert.equal(t.el.querySelector('.toast__msg').textContent, 'Speech generated');
    env.clock.tick(5999);
    assert.equal(polite().length, 3);
    env.clock.tick(1);
    assert.equal(polite().length, 0);
  });

  it('[BUG-14] errors go in the role=alert region, persist until dismissed, and have a labelled close button', async () => {
    const { toast } = await setup();
    const t = toast("Couldn't delete: the file is in use", { kind: 'error' });
    assert.equal(alerts().length, 1);
    assert.equal(document.getElementById('app-toasts-alert').getAttribute('role'), 'alert');
    env.clock.tick(60000);
    assert.equal(alerts().length, 1, 'errors persist');
    const close = t.el.querySelector('button[aria-label="Dismiss notification"]');
    assert.ok(close);
    assertAccessibleNames(t.el);
    close.click();
    assert.equal(alerts().length, 0);
  });

  it('the message is text, never HTML', async () => {
    const { toast } = await setup();
    const t = toast('<img src=x onerror=alert(1)>', { kind: 'error' });
    assert.equal(t.el.querySelector('img'), null);
    assert.equal(t.el.querySelector('.toast__msg').textContent, '<img src=x onerror=alert(1)>');
  });

  it('shows at most 3: the oldest non-error goes first', async () => {
    const { toast } = await setup();
    toast('e1', { kind: 'error' });
    toast('i1');
    toast('i2');
    toast('i3');
    const all = [...alerts(), ...polite()].map((t) => t.querySelector('.toast__msg').textContent);
    assert.deepEqual(all.sort(), ['e1', 'i2', 'i3']);
    toast('e2', { kind: 'error' });
    toast('e3', { kind: 'error' });
    toast('e4', { kind: 'error' });
    assert.deepEqual(alerts().map((t) => t.querySelector('.toast__msg').textContent), ['e2', 'e3', 'e4']);
  });

  it('the same id replaces the earlier toast', async () => {
    const { toast } = await setup();
    toast('Loading…', { id: 'job' });
    toast('Loaded', { id: 'job', kind: 'success' });
    assert.deepEqual(polite().map((t) => t.querySelector('.toast__msg').textContent), ['Loaded']);
  });

  it('hover and focus pause the timer', async () => {
    const { toast } = await setup();
    const t = toast('Copied');
    env.clock.tick(3000);
    t.el.dispatchEvent(new MouseEvent('mouseenter'));
    env.clock.tick(10000);
    assert.equal(polite().length, 1, 'paused while hovered');
    t.el.dispatchEvent(new MouseEvent('mouseleave'));
    env.clock.tick(2999);
    assert.equal(polite().length, 1);
    env.clock.tick(1);
    assert.equal(polite().length, 0);

    const u = toast('Saved');
    u.el.querySelector('button').focus();
    env.clock.tick(20000);
    assert.equal(polite().length, 1, 'paused while focused');
  });

  it('an action runs once, then the toast is dismissed; focus returns to where it was', async () => {
    const { toast } = await setup();
    let runs = 0;
    const t = toast("Couldn't play", { kind: 'error', action: { label: 'Retry', run: () => { runs += 1; } } });
    document.getElementById('trigger').focus();
    const retry = [...t.el.querySelectorAll('button')].find((b) => b.textContent === 'Retry');
    retry.focus();
    retry.click();
    assert.equal(runs, 1);
    assert.equal(alerts().length, 0);
    assert.equal(document.activeElement.id, 'trigger');
  });

  it('dismissAll clears every toast', async () => {
    const { toast, dismissAll } = await setup();
    toast('a');
    toast('b', { kind: 'error' });
    dismissAll();
    assert.equal(polite().length + alerts().length, 0);
  });

  it('builds its regions when the shell has none', async () => {
    env = installBrowserEnv();
    const { toast } = await importFresh('core/toast.js');
    toast('x', { kind: 'error' });
    assert.equal(document.getElementById('app-toasts-alert').getAttribute('role'), 'alert');
  });
});
