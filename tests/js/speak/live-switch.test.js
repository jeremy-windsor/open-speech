// Speak › Live is a developer tool: Speak shows Generate only
// unless Settings › Developer › "Show Speak › Live (a streaming test for developers)" (prefs.showLive) is on.
// Off (the default): no mode switch, a link or saved route that asks for Live shows Generate, and Live's
// shortcuts are not in the "?" sheet. On: the switch reads Generate | Live (test), and Live opens with one line
// saying what it is. Nothing in the Live mode itself changes.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { mountSpeak, ready, type, settle, json, CAPS, load } from './helpers.js';

let t;
afterEach(() => t?.env.restore());

const LIVE_ON = Object.freeze({ 'open-speech-prefs': JSON.stringify({ v: 1, showLive: true }) });
const LIVE_ROUTE = Object.freeze({ view: 'speak', sub: 'live', id: '', path: 'speak/live', shot: '' });
const SPEAK_ROUTE = Object.freeze({ view: 'speak', sub: '', id: '', path: 'speak', shot: '' });
const ABOUT = 'Reads text as it arrives, the way an AI app streams its reply to the speech endpoint. Type or paste to try it.';

/** The ids of the shortcuts registered now (the "?" sheet lists exactly these). */
async function shortcutIds() {
  const shortcuts = await load('core/shortcuts.js');
  return shortcuts.list().map((s) => s.id);
}

/** Show Live mode and wait for its controls to mount. */
async function showLive() {
  t.ctl.show(LIVE_ROUTE, { focus: 'keep' });
  await settle(t.env);
  if (t.ctx.live && t.ctx.live.ready) await t.ctx.live.ready;
  await settle(t.env);
}

describe('Speak › Live is off unless Settings turns it on', () => {
  it('by default Speak shows Generate only: no mode switch, no Live controls, and none of Live\'s shortcuts', async () => {
    t = await mountSpeak();
    await ready(t);
    const prefs = await load('core/prefs.js');
    assert.equal(prefs.get().showLive, false, 'off by default');
    assert.equal(t.$('sp-mode').hidden, true, 'no Generate | Live switch');
    assert.equal(t.section.getAttribute('data-show-live'), 'false');
    assert.equal(t.$('sp-genbar').hidden, false);
    assert.equal(t.$('sp-live-slot').hidden, true);
    assert.equal(t.$('sp-live-slot').children.length, 0, 'Live never mounted');
    const ids = await shortcutIds();
    assert.ok(ids.includes('speak-generate'), ids.join(', '));
    assert.ok(!ids.some((id) => id.startsWith('live-')), `no Live rows in the ? sheet: ${ids.join(', ')}`);
  });

  it('a link, the saved last route or Back that asks for #speak/live shows Generate, and the address follows', async () => {
    t = await mountSpeak({ route: { sub: 'live', path: 'speak/live' } });
    await ready(t);
    assert.equal(t.ctx.mode, 'generate');
    assert.equal(t.section.getAttribute('data-mode'), 'generate');
    assert.equal(t.$('sp-genbar').hidden, false, 'the Generate controls');
    assert.equal(t.$('sp-live-slot').hidden, true);
    assert.equal(t.$('sp-live-slot').children.length, 0, 'Live never mounted');
    assert.equal(location.hash, '#speak', 'the address is replaced');
    const prefs = await load('core/prefs.js');
    assert.equal(prefs.get().lastRoute, 'speak', 'and the next visit remembers Generate');
    assert.equal(t.env.sockets.length, 0);
    const ids = await shortcutIds();
    assert.ok(!ids.some((id) => id.startsWith('live-')), ids.join(', '));
    // Back to the old address while Speak shows: Generate again.
    t.ctl.show(LIVE_ROUTE, { focus: 'keep' });
    await settle(t.env);
    assert.equal(t.ctx.mode, 'generate');
    assert.equal(t.$('sp-live-slot').hidden, true);
  });

  it('over the limit, the field error points at the Reader while Live is hidden, and at Live once it is shown', async () => {
    const caps = { 'GET /api/tts/capabilities': () => json(200, { capabilities: CAPS.kokoro.capabilities, max_input_length: 4096, max_input_chars: 10 }) };
    t = await mountSpeak({ routes: caps });
    await ready(t);
    type(t, 'x'.repeat(11));
    assert.equal(t.$('sp-limit-text').textContent, 'Generate accepts up to 10 characters. For longer text, use the Reader.');
    const prefs = await load('core/prefs.js');
    prefs.set({ showLive: true });
    await settle(t.env);
    assert.equal(t.$('sp-limit-text').textContent, 'Generate accepts up to 10 characters. Use Live for longer text.');
  });

  it('turned on: the switch reads Generate | Live (test), and Live opens with one line saying what it is', async () => {
    t = await mountSpeak({ seed: LIVE_ON });
    await ready(t);
    assert.equal(t.$('sp-mode').hidden, false);
    assert.equal(t.section.getAttribute('data-show-live'), 'true');
    const labels = [...t.$('sp-mode').querySelectorAll('.seg__opt')].map((o) => o.textContent.trim());
    assert.deepEqual(labels, ['Generate', 'Live (test)']);
    await showLive();
    assert.equal(t.ctx.mode, 'live');
    assert.equal(t.$('sp-live-slot').hidden, false);
    const about = t.$('spl-about');
    assert.ok(about, 'the line is in the Live controls');
    assert.equal(about.textContent.trim(), ABOUT);
    assert.equal(t.$('spl').firstElementChild.nextElementSibling, about, 'first, right after the hidden heading');
    const ids = await shortcutIds();
    assert.ok(ids.includes('live-start') && ids.includes('live-stop'), ids.join(', '));
  });

  it('turned on in the same page: the switch appears at once; turned off: Live stops and unmounts, Speak shows Generate, and Live\'s shortcuts leave the ? sheet', async () => {
    t = await mountSpeak();
    await ready(t);
    const prefs = await load('core/prefs.js');
    prefs.set({ showLive: true });
    await settle(t.env);
    assert.equal(t.$('sp-mode').hidden, false, 'Settings turned it on: the switch shows');
    await showLive();
    assert.ok((await shortcutIds()).includes('live-start'));
    // A Live session running in the background (a stand-in for the mounted controls, which unmount the same way).
    const real = t.ctx.live;
    const calls = [];
    t.ctx.live = {
      isActive: () => true, stop: (reason) => calls.push(`stop ${reason}`), refresh() {},
      destroy: () => { calls.push('destroy'); real.destroy(); },
    };
    prefs.set({ showLive: false });
    await settle(t.env);
    assert.deepEqual(calls, ['destroy'], 'Live stops (its engine stops the session when it unmounts)');
    assert.equal(t.ctx.live, null);
    assert.equal(t.$('sp-mode').hidden, true);
    assert.equal(t.ctx.mode, 'generate');
    assert.equal(t.$('sp-genbar').hidden, false);
    assert.equal(t.$('sp-live-slot').hidden, true);
    assert.equal(t.$('sp-live-slot').children.length, 0, 'the Live controls are gone');
    assert.equal(location.hash, '#speak', 'the address follows');
    assert.ok(!(await shortcutIds()).some((id) => id.startsWith('live-')), 'no Live rows in the ? sheet');
    // Coming back from another view with no Live session: Generate, as for anyone.
    t.ctl.hide();
    t.ctl.show(SPEAK_ROUTE, { focus: 'heading' });
    await settle(t.env);
    assert.equal(t.ctx.mode, 'generate');
    assert.deepEqual(t.env.errors || [], []);
  });
});
