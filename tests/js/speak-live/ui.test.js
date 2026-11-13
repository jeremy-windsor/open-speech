// Speak: Live controls: mountLive() on the real speak-live partial, a fake editor standing
// in for Speak's #sp-input, and a scripted server. Inventory LR-01, LR-07, LR-12, LR-19, LR-20.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { assertAccessibleNames } from '../harness/a11y.js';
import { liveEnv, load, settle, voiceConfig, server, sent, text, partialRoute, FakeAudioContext } from './helpers.js';

let env;
afterEach(() => {
  env?.live?.destroy();
  env?.restore();
});

const $ = (id) => document.getElementById(id);
const radios = () => Array.from(document.querySelectorAll('input[name="spl-mode"]'));
const srStatus = () => (document.getElementById('app-sr-status') || {}).textContent || '';

/**
 * Mount the Live controls. `config` may be changed later with setConfig().
 * @returns {Promise<{ui, live, states, setConfig}>}
 */
async function mount({ config = voiceConfig(), textValue = '', routes = {}, key = '', seed = {} } = {}) {
  env = await liveEnv({ routes, key, seed });
  const ui = await load('views/speak/live-ui.js');
  ui.__resetForTests();
  env.editor.value = textValue;
  env.editor.selectionStart = textValue.length;
  let cfg = config;
  const states = [];
  const live = ui.mountLive(env.slot, {
    editor: env.editor,
    getVoiceConfig: () => cfg,
    onState: (s) => states.push(s),
  });
  env.live = live;
  assert.equal(await live.ready, true, 'the partial rendered');
  await settle(env);
  return { ui, live, states, setConfig: (c) => { cfg = c; } };
}

async function openServer() {
  await settle(env);
  const srv = server(env.sockets[env.sockets.length - 1]);
  srv.open();
  srv.created();
  srv.updated();
  await settle(env);
  return srv;
}

describe('Live controls: mode and buttons', () => {
  it('[LR-01] the mode is a segmented control (Book, Responsive, Each word) with a hint, sent as latency_mode', async () => {
    const { ui } = await mount();
    const group = $('spl-mode');
    assert.equal(group.getAttribute('role'), 'radiogroup');
    assert.equal($(group.getAttribute('aria-labelledby')).textContent, 'Mode');
    assert.equal(group.getAttribute('aria-describedby'), 'spl-mode-hint');
    assert.deepEqual(radios().map((r) => [r.value, r.closest('label').textContent.trim()]), [
      ['natural', 'Book'], ['responsive', 'Responsive'], ['instant_word', 'Each word'],
    ]);
    assert.equal(radios()[0].checked, true, 'Book by default');
    assert.equal($('spl-mode-hint').textContent, 'Waits for full sentences: the smoothest for reading.');
    env.change(radios()[2], true);
    assert.equal($('spl-mode-hint').textContent, 'Speaks every completed word: fastest, choppier.');
    env.change(radios()[1], true);
    assert.equal($('spl-mode-hint').textContent, 'Speaks short phrases as you type.');
    env.click($('spl-start'));
    const srv = server(env.sockets[0] || (await settle(env), env.sockets[0]));
    await settle(env);
    srv.open();
    srv.created();
    await settle(env);
    assert.equal(sent(srv.ws, 'session.update')[0].session.latency_mode, 'responsive');
    assert.ok(radios().every((r) => r.disabled), 'the mode is locked while a session runs');
    env.live.stop();
    await settle(env);
    assert.ok(radios().every((r) => !r.disabled));

    // The choice is kept when Speak mounts the controls again (same page session).
    env.live.destroy();
    const again = ui.mountLive(env.slot, { editor: env.editor, getVoiceConfig: () => voiceConfig() });
    env.live = again;
    await again.ready;
    assert.equal(radios().find((r) => r.checked).value, 'responsive');
  });

  it('[LR-01] the mode is remembered across page loads (prefs.liveMode inside open-speech-prefs); a bad stored value falls back to Book', async () => {
    await mount();
    env.change(radios()[2], true);
    const stored = JSON.parse(localStorage.getItem('open-speech-prefs'));
    assert.equal(stored.liveMode, 'instant_word', 'saved with the other preferences, no key of its own');
    assert.deepEqual(Object.keys(localStorage.__data).filter((k) => /live/i.test(k)), [], 'no new storage key');
    env.live.destroy();
    env.restore();

    // A new page load: the stored preferences are all that is left.
    await mount({ seed: { 'open-speech-prefs': JSON.stringify(stored) } });
    assert.equal(radios().find((r) => r.checked).value, 'instant_word');
    assert.equal($('spl-mode-hint').textContent, 'Speaks every completed word: fastest, choppier.');
    env.click($('spl-start'));
    const srv = await openServer();
    assert.equal(sent(srv.ws, 'session.update')[0].session.latency_mode, 'instant_word', 'the remembered mode is the one sent');
    env.live.stop();
    await settle(env);
    env.live.destroy();
    env.restore();

    await mount({ seed: { 'open-speech-prefs': JSON.stringify({ ...stored, liveMode: 'warp' }) } });
    assert.equal(radios().find((r) => r.checked).value, 'natural', 'an unknown mode falls back to Book');
  });

  it('[LR-20] while a session runs the toggle pauses, Stop works, the mode and Read all wait; after the release it is the reverse', async () => {
    await mount({ textValue: 'Hello.' });
    const toggle = $('spl-start');
    assert.equal($('spl-start-label').textContent, 'Start');
    assert.equal($('spl-stop').disabled, true);
    assert.equal($('spl-read-all').getAttribute('aria-disabled'), null);
    assert.equal(toggle.getAttribute('aria-keyshortcuts'), 'Control+Enter');
    assert.equal($('spl-kbd').textContent, 'Ctrl↵');
    assert.equal($('spl-kbd-mod').textContent, 'Ctrl');
    assert.equal($('spl-kbd').getAttribute('aria-hidden'), 'true');
    env.click(toggle);
    await openServer();
    assert.equal($('spl-start-label').textContent, 'Pause');
    assert.equal(toggle.querySelector('.spl-toggle__pause').hasAttribute('hidden'), false);
    assert.equal(toggle.querySelector('.spl-toggle__play').hasAttribute('hidden'), true, 'the attribute: an SVG has no hidden property in browsers');
    assert.equal($('spl-kbd').hidden, true);
    assert.equal($('spl-stop').disabled, false);
    assert.equal($('spl-read-all').getAttribute('aria-disabled'), 'true');
    assert.ok(radios().every((r) => r.disabled));
    env.click(toggle);
    assert.equal($('spl-start-label').textContent, 'Resume');
    assert.equal(toggle.querySelector('.spl-toggle__play').hasAttribute('hidden'), false);
    env.click(toggle);
    assert.equal($('spl-start-label').textContent, 'Pause');

    $('spl-stop').focus();
    env.click($('spl-stop'));
    await settle(env);
    assert.equal($('spl-start-label').textContent, 'Start');
    assert.equal($('spl-stop').disabled, true);
    assert.equal(document.activeElement, toggle, 'focus moves to the toggle, never to the page');
    assert.equal($('spl-read-all').getAttribute('aria-disabled'), null);
    assert.ok(radios().every((r) => !r.disabled));
    assert.equal(env.sockets[0].closeReason, 'Stopped by user');
  });

  it('[LR-19] Clear text stops the session, empties the text and lets the counter follow', async () => {
    await mount({ textValue: 'Some text.' });
    let counted = '';
    env.editor.addEventListener('input', () => { counted = `${env.editor.value.length} characters`; });
    env.click($('spl-start'));
    await openServer();
    env.click($('spl-more'));
    assert.equal($('spl-more-menu').hidden, false);
    env.click($('spl-clear'));
    await settle(env);
    assert.equal(env.live.isActive(), false);
    assert.equal(env.editor.value, '');
    assert.equal(counted, '0 characters');
    assert.equal(document.activeElement, env.editor);
    assert.equal($('spl-status-label').textContent, 'Stopped');
  });

  it('[LR-03] Read all from start (the menu) reads the whole text whatever the caret', async () => {
    await mount({ textValue: 'Alpha. Beta.' });
    env.editor.selectionStart = 7;
    env.click($('spl-more'));
    env.click($('spl-read-all'));
    const srv = await openServer();
    assert.deepEqual(sent(srv.ws, 'input_text.append').map((e) => e.text), ['Alpha. Beta.']);
    env.live.stop();
  });
});

describe('Live controls: status and the Now line', () => {
  it('[LR-07] the pill walks Preparing…, Connecting…, Listening, Paused and Stopped, and its live region changes per phase, not per response', async () => {
    const { states } = await mount({ textValue: 'One.' });
    const pill = $('spl-status');
    assert.equal(pill.getAttribute('role'), 'status');
    assert.equal($('spl-now').getAttribute('aria-live'), 'off');
    assert.equal($('spl-status-label').getAttribute('aria-hidden'), 'true');
    env.click($('spl-start'));
    assert.equal($('spl-status-label').textContent, 'Preparing…');
    assert.ok(pill.classList.contains('pill--busy'));
    await settle(env);
    assert.equal($('spl-status-label').textContent, 'Connecting…');
    const srv = server(env.sockets[0]);
    srv.open();
    srv.created();
    srv.updated();
    await settle(env);
    assert.equal($('spl-status-label').textContent, 'Listening');
    assert.ok(pill.classList.contains('pill--ok'));
    assert.equal($('spl-status-text').textContent, 'Listening');
    srv.response('r1', 'One.');
    srv.delta('r1', 1);
    await settle(env);
    assert.equal($('spl-status-label').textContent, 'Speaking', 'the visible label says Speaking');
    assert.equal($('spl-status-text').textContent, 'Listening', 'the live region does not flip per sentence');
    assert.equal($('spl-latency').hidden, false);
    assert.match($('spl-latency').textContent, /^Voice latency ~\d+ ms$/);
    env.click($('spl-start'));
    assert.equal($('spl-status-label').textContent, 'Paused');
    assert.equal($('spl-status-text').textContent, 'Paused');
    assert.ok(pill.classList.contains('pill--idle'));
    env.click($('spl-stop'));
    await settle(env);
    assert.equal($('spl-status-text').textContent, 'Stopped');
    assert.equal($('spl-latency').hidden, true);
    assert.deepEqual([...new Set(states.map((s) => s.status))], ['Preparing…', 'Connecting…', 'Listening', 'Speaking', 'Paused', 'Stopped']);
  });

  it('[LR-12] response.created shows the text being read in the Now line; after the audio it goes back to Listening for text…', async () => {
    await mount({ textValue: 'In a hole.' });
    assert.equal($('spl-now-text').textContent, 'Start, then type or paste. Live reads from the cursor as you write.');
    assert.equal($('spl-now-label').hidden, true);
    env.click($('spl-start'));
    const srv = await openServer();
    assert.equal($('spl-now-text').textContent, 'Listening for text…');
    srv.response('r1', 'In a hole in the ground there lived a hobbit.');
    await settle(env);
    assert.equal($('spl-now').getAttribute('data-kind'), 'spoken', 'the active style');
    assert.equal($('spl-now-label').hidden, false);
    assert.equal($('spl-now-label').textContent, 'Now');
    assert.equal($('spl-now-text').textContent, '“In a hole in the ground there lived a hobbit.”');
    srv.delta('r1', 1);
    srv.done('r1');
    await settle(env, 130);
    assert.equal($('spl-now').getAttribute('data-kind'), 'spoken', 'held briefly after the audio ends');
    await settle(env, 700);
    assert.equal($('spl-now').getAttribute('data-kind'), 'hint');
    assert.equal($('spl-now-text').textContent, 'Listening for text…');
    env.live.stop();
  });

  it('[LR-07] a failed session shows the inline error banner with Details, the pill in its error style and no second Start: Start above it starts a new session', async () => {
    await mount();
    env.click($('spl-start'));
    await settle(env);
    const srv = server(env.sockets[0]);
    srv.open();
    srv.error('session_busy', 'Live Reader is already in use', { reason: 'Another Live session is active' });
    srv.close(1013, 'Live Reader busy');
    await settle(env);
    const banner = $('spl-error');
    assert.equal(banner.hidden, false);
    assert.equal(banner.getAttribute('role'), 'alert');
    assert.equal($('spl-error-title').textContent, 'Live stopped');
    assert.equal($('spl-error-msg').textContent, 'Another Live session is running. Stop it first or try again shortly.');
    assert.equal($('spl-error-details').hidden, false);
    assert.match($('spl-error-detail').textContent, /already in use/);
    assert.equal($('spl-status-label').textContent, 'Busy');
    assert.ok($('spl-status').classList.contains('pill--error'), 'Busy shares the error style of the banner (LR-07, as in the old UI)');
    assert.ok(!$('spl-status').classList.contains('pill--warn'));
    assert.equal($('spl-now').hidden, true, 'no idle hint beside the error banner');
    // The retry is Start, right above the banner (in place of the usual Retry): the banner offers no second Start.
    assert.equal($('spl-error-action').hidden, true, 'no "Start again" under the Start button');
    assert.equal($('spl-error-actions').hidden, true, 'and no empty actions line in the banner');
    assert.equal($('spl-start-label').textContent, 'Start');
    assert.equal($('spl-start').getAttribute('aria-disabled'), null, 'Start is the enabled retry');
    $('spl-start').focus();
    env.click($('spl-start'));
    assert.equal($('spl-error').hidden, true, 'the banner clears for the new attempt');
    assert.equal(env.live.isActive(), true);
    assert.equal(document.activeElement, $('spl-start'));
    await settle(env);
    assert.equal(env.sockets.length, 2);
    env.live.stop();
    await settle(env);

    // A refused ticket points to Settings › Connection; Dismiss hides the banner.
    env.click($('spl-start'));
    await settle(env);
    server(env.sockets[2]).close(4001, 'Invalid or missing API key');
    await settle(env);
    assert.equal($('spl-error-action').hidden, false, 'a refused key gets the one action Start cannot do');
    assert.equal($('spl-error-actions').hidden, false);
    assert.equal($('spl-error-action-label').textContent, 'Open Settings');
    env.click($('spl-error-dismiss'));
    assert.equal($('spl-error').hidden, true);
    env.click($('spl-start'));
    await settle(env);
    server(env.sockets[3]).close(4001, 'Invalid or missing API key');
    await settle(env);
    env.click($('spl-error-action'));
    await settle(env);
    assert.equal(env.app.get().route.view, 'settings');
    assert.equal($('spl-error').hidden, true);

    // Live turned off on the server: no action that could not help.
    env.app.set({ route: { view: 'speak', sub: 'live', id: '', path: 'speak/live', shot: '' } });
    env.click($('spl-start'));
    await settle(env);
    server(env.sockets[4]).close(4004, 'Live TTS is disabled');
    await settle(env);
    assert.equal($('spl-error-msg').textContent, 'Live speech is turned off on this server (TTS_ENABLED or TTS_LIVE_ENABLED).');
    assert.equal($('spl-error-action').hidden, true);
  });

  it('[LR-07] a failure while away from Speak goes in an error toast (no control nearby)', async () => {
    await mount();
    env.click($('spl-start'));
    const srv = await openServer();
    env.app.set({ route: { view: 'reader', sub: '', id: '', path: 'reader', shot: '' } });
    assert.equal(env.live.isActive(), true, 'leaving Speak does not stop Live');
    srv.close(1006, '');
    await settle(env);
    const msgs = Array.from(document.querySelectorAll('.toast__msg')).map((el) => el.textContent);
    assert.ok(msgs.includes('Live stopped. The connection to the server dropped.'), msgs.join(' | '));
  });
});

describe('Live controls: gating, keys and the shared editor', () => {
  it('[LR-02] a model without Live support disables Start with the reason; a missing voice is refused and announced', async () => {
    const { live, setConfig } = await mount({ config: voiceConfig({ caps: { live_reader: false } }) });
    const toggle = $('spl-start');
    assert.equal(toggle.getAttribute('aria-disabled'), 'true');
    assert.equal(toggle.getAttribute('aria-describedby'), 'spl-start-hint');
    assert.equal($('spl-start-hint').hidden, false);
    assert.equal($('spl-start-hint').getAttribute('data-tone'), 'info');
    assert.equal($('spl-start-hint-text').textContent, "This model needs a reference recording and isn't available in Live yet");
    assert.equal($('spl-read-all').getAttribute('aria-disabled'), 'true');
    assert.equal($('spl-now-text').textContent, 'Choose another model in Voice and settings to use Live.');
    env.click(toggle);
    await settle(env);
    assert.equal(env.sockets.length, 0);
    assert.equal(FakeAudioContext.instances.length, 0);
    assert.equal(srStatus(), "This model needs a reference recording and isn't available in Live yet");

    // Speak picks another voice: Speak's store update re-reads the gate.
    setConfig(voiceConfig({ voice: '' }));
    env.app.set({ speakVoice: { model: 'kokoro', voice: '', label: '' } });
    await settle(env);
    assert.equal(toggle.getAttribute('aria-disabled'), null);
    assert.equal($('spl-start-hint').hidden, true);
    env.click(toggle);
    await settle(env, 600);
    assert.equal($('spl-start-hint').hidden, false);
    assert.equal($('spl-start-hint').getAttribute('data-tone'), 'alert');
    assert.equal($('spl-start-hint-text').textContent, 'Choose a voice or a reference recording first');
    assert.equal(srStatus(), 'Choose a voice or a reference recording first');
    setConfig(voiceConfig());
    live.refresh();
    assert.equal($('spl-start-hint').hidden, true, 'refresh() clears a stale refusal');
  });

  // Text typed while Live is stopped seemed to do nothing until Start was pressed. Typing before Start stays allowed (it is how a listener prepares or pastes text to read from the
  // caret), but the view now says the text waits for Start, once to a screen reader, and emphasises Start.
  it('[LR-20] typing while Live is stopped shows "Press Start to hear this live." by the controls, emphasises Start and says it once; Start takes both away', async () => {
    await mount();
    const PROMPT = 'Press Start to hear this live.';
    const toggle = $('spl-start');
    const hint = $('spl-start-hint');
    let region = document.getElementById('app-sr-status');
    if (!region) {
      region = document.createElement('div');
      region.id = 'app-sr-status';
      region.setAttribute('role', 'status');
      document.body.appendChild(region);
    }
    let accessor = null;
    for (let p = Object.getPrototypeOf(region); p && !accessor; p = Object.getPrototypeOf(p)) accessor = Object.getOwnPropertyDescriptor(p, 'textContent');
    const said = [];
    Object.defineProperty(region, 'textContent', {
      configurable: true,
      get() { return accessor.get.call(this); },
      set(value) { said.push(String(value)); accessor.set.call(this, value); },
    });
    const prompted = () => said.filter((s) => s === PROMPT).length;
    const shows = () => !hint.hidden && $('spl-start-hint-text').textContent === PROMPT;
    assert.equal(hint.hidden, true, 'nothing typed yet: no prompt');
    assert.equal(toggle.hasAttribute('data-emphasis'), false);

    // In Generate mode the shared editor's text is meant for a take: no prompt.
    env.app.set({ route: { view: 'speak', sub: '', id: '', path: 'speak', shot: '' } });
    env.slot.hidden = true;
    env.input(env.editor, 'A take.');
    await settle(env, 600);
    assert.equal(prompted(), 0);
    env.slot.hidden = false;
    env.app.set({ route: { view: 'speak', sub: 'live', id: '', path: 'speak/live', shot: '' } });
    await settle(env);
    assert.equal(shows(), false);

    for (const value of ['H', 'He', 'Hel', 'Hello']) {
      env.input(env.editor, value);
      await settle(env, 600);
      assert.equal(shows(), true, value);
      assert.equal(hint.getAttribute('data-tone'), 'info');
      assert.equal(toggle.getAttribute('data-emphasis'), 'true', 'Start is emphasised');
      assert.equal(toggle.getAttribute('aria-describedby'), 'spl-start-hint', 'Start is described by the prompt');
    }
    assert.equal(prompted(), 1, `said once, not per keystroke: ${JSON.stringify(said)}`);
    env.input(env.editor, '');
    await settle(env);
    assert.equal(hint.hidden, true, 'no text: nothing to start');
    assert.equal(toggle.hasAttribute('data-emphasis'), false);
    env.input(env.editor, 'Hello');
    await settle(env, 600);
    assert.equal(shows(), true);
    assert.equal(prompted(), 1, 'still once while Live stays stopped');

    env.click(toggle);
    await openServer();
    assert.equal(hint.hidden, true, 'started: the prompt goes away');
    assert.equal(toggle.hasAttribute('data-emphasis'), false);
    env.input(env.editor, 'Hello there');
    await settle(env, 600);
    assert.equal(hint.hidden, true, 'no prompt while Live runs');
    env.click($('spl-stop'));
    await settle(env);
    assert.equal(hint.hidden, true, 'stopped, with nothing typed since');
    env.input(env.editor, 'Hello there again');
    await settle(env, 600);
    assert.equal(shows(), true, 'typed while stopped again');
    assert.equal(prompted(), 2, 'said once more after the new stop');
    delete region.textContent;
    assert.deepEqual(env.errors, []);
  });

  it('[LR-02] a refusal or the model\'s Live gate wins over the prompt to press Start', async () => {
    const { setConfig } = await mount({ config: voiceConfig({ caps: { live_reader: false } }) });
    env.input(env.editor, 'Some text.');
    await settle(env, 600);
    assert.equal($('spl-start-hint-text').textContent, "This model needs a reference recording and isn't available in Live yet");
    assert.equal($('spl-start').hasAttribute('data-emphasis'), false, 'a Start that can\'t run is not emphasised');
    setConfig(voiceConfig({ voice: '' }));
    env.app.set({ speakVoice: { model: 'kokoro', voice: '', label: '' } });
    await settle(env);
    env.click($('spl-start'));
    await settle(env);
    assert.equal($('spl-start-hint-text').textContent, 'Choose a voice or a reference recording first');
    assert.equal($('spl-start').hasAttribute('data-emphasis'), false);
    setConfig(voiceConfig());
    env.live.refresh();
    env.input(env.editor, 'Some text. More.');
    await settle(env);
    assert.equal($('spl-start-hint-text').textContent, 'Press Start to hear this live.', 'once Start can run, the prompt shows');
    assert.equal($('spl-start').getAttribute('data-emphasis'), 'true');
  });

  it('[LR-20] Mod+Enter starts and Esc stops while the Live slot shows; neither fires in Generate mode', async () => {
    await mount({ textValue: 'Typing here.' });
    env.editor.focus();
    env.key('Enter', { ctrlKey: true }, env.editor);
    assert.equal(env.live.isActive(), true, 'Mod+Enter in the editor starts Live');
    await settle(env);
    env.key('Escape', {}, env.editor);
    assert.equal(env.live.isActive(), false, 'Esc stops it');

    env.app.set({ route: { view: 'speak', sub: '', id: '', path: 'speak', shot: '' } });
    env.slot.hidden = true;
    env.key('Enter', { ctrlKey: true }, env.editor);
    assert.equal(env.live.isActive(), false, 'Generate mode owns Mod+Enter');
  });

  it('[LR-20] switching Speak to Generate stops Live ("Switched to Generate"); leaving Speak keeps it running for the mini-player', async () => {
    await mount({ textValue: 'Text.' });
    env.click($('spl-start'));
    await openServer();
    env.app.set({ route: { view: 'history', sub: '', id: '', path: 'history', shot: '' } });
    assert.equal(env.live.isActive(), true);
    const player = env.app.get().player;
    assert.equal(player.kind, 'live');
    assert.equal(player.homeRoute, 'speak/live');
    assert.equal(player.statusText, 'Listening');
    env.app.set({ route: { view: 'speak', sub: 'live', id: '', path: 'speak/live', shot: '' } });
    assert.equal(env.live.isActive(), true);
    env.app.set({ route: { view: 'speak', sub: '', id: '', path: 'speak', shot: '' } });
    assert.equal(env.live.isActive(), false);
    assert.equal($('spl-status-label').textContent, 'Stopped: switched to Generate');
  });

  it('[LR-07] onState reports every change to Speak, and the contract methods drive the same session', async () => {
    const { states } = await mount({ textValue: 'Contract.' });
    assert.equal(env.live.start(), true);
    assert.equal(env.live.isActive(), true);
    await openServer();
    env.live.togglePause();
    assert.equal(states.at(-1).status, 'Paused');
    env.live.stop('Voice changed');
    assert.equal(states.at(-1).status, 'Stopped: voice changed');
    assert.equal(states.at(-1).active, false);
    assert.equal($('spl-status-label').textContent, 'Stopped: voice changed');
  });

  it('[LR-01] every control built from the partial has a name and a type, and the slot has no h1', async () => {
    await mount();
    assertAccessibleNames(env.slot);
    assert.equal(env.slot.querySelectorAll('h1').length, 0);
    assert.equal(env.slot.querySelector('h2').textContent, 'Live reading');
    assert.equal($('spl-more').getAttribute('aria-haspopup'), 'menu');
    assert.equal($('spl-more').getAttribute('aria-controls'), 'spl-more-menu');
  });

  it('[LR-07] a partial that fails to load shows an error with Retry; the engine still works', async () => {
    let fail = true;
    const real = partialRoute();
    env = await liveEnv({
      routes: { 'GET /static/partials/:name': (req) => (fail ? text(503, 'busy') : real(req)) },
    });
    const ui = await load('views/speak/live-ui.js');
    ui.__resetForTests();
    const live = ui.mountLive(env.slot, { editor: env.editor, getVoiceConfig: () => voiceConfig() });
    env.live = live;
    assert.equal(await live.ready, false);
    await settle(env);
    const banner = env.slot.querySelector('.banner--error');
    assert.ok(banner, 'error banner');
    assert.equal(banner.getAttribute('role'), 'alert');
    assert.match(banner.textContent, /Live controls didn't load/);
    assert.equal(live.start(), true, 'start() works without the markup');
    live.stop();
    fail = false;
    env.click(env.slot.querySelector('[data-spl-retry]'));
    await settle(env);
    assert.equal(env.slot.querySelector('.spl-load-error'), null);
    assert.ok($('spl-start'), 'the controls are there');
    assert.equal(document.activeElement, $('spl-start'), 'focus lands on Start, not on the page');
    assert.equal($('spl-status-label').textContent, 'Stopped');
  });

  it('[LR-07] the screenshot hook paints the Live states without starting anything', async () => {
    await mount();
    env.app.set({ route: { view: 'speak', sub: 'live', id: '', path: 'speak/live', shot: 'live-speaking' } });
    await settle(env);
    assert.equal($('spl-status-label').textContent, 'Speaking');
    assert.equal($('spl-now-text').textContent, '“In a hole in the ground there lived a hobbit.”');
    assert.equal($('spl-latency').textContent, 'Voice latency ~182 ms');
    // The route changes before Speak shows the slot: the scroll waits for the controls to have a layout box,
    // looking again on a timer (a headless capture runs timers but may draw no animation frame).
    await settle(env, 100);
    assert.deepEqual($('spl').__scrolledIntoView, [], 'nothing scrolls while the controls have no box');
    $('spl-more').__rect = { x: 300, y: 600, width: 44, height: 44 };
    await settle(env, 20);
    assert.deepEqual($('spl').__scrolledIntoView.at(-1), { block: 'center' }, 'then the controls are scrolled into view');
    assert.equal(env.sockets.length, 0);
    assert.equal(FakeAudioContext.instances.length, 0);
    env.app.set({ route: { view: 'speak', sub: 'live', id: '', path: 'speak/live', shot: 'live-recording' } });
    await settle(env);
    assert.equal($('spl-status-label').textContent, 'Paused');
    assert.equal($('spl-start-label').textContent, 'Resume');
    assert.equal($('spl-start-hint').hidden, false, 'a Resume refused while the microphone records says why');
    assert.equal($('spl-start-hint-text').textContent, 'Stop recording to listen.');
    env.app.set({ route: { view: 'speak', sub: 'live', id: '', path: 'speak/live', shot: 'live-error' } });
    await settle(env);
    assert.equal($('spl-error').hidden, false);
    assert.equal($('spl-status-label').textContent, 'Busy');
    assert.ok($('spl-status').classList.contains('pill--error'), 'the pill and the banner share the error style');
    assert.equal($('spl-error-action').hidden, true, 'the retry is Start, above the banner');
    env.app.set({ route: { view: 'speak', sub: 'live', id: '', path: 'speak/live', shot: 'live-unavailable' } });
    await settle(env);
    assert.equal($('spl-start').getAttribute('aria-disabled'), 'true');
    env.app.set({ speakVoice: { model: 'kokoro', voice: 'af_heart', label: 'Heart' } });
    await settle(env);
    assert.equal($('spl-start').getAttribute('aria-disabled'), 'true', 'Speak settling its voice keeps the shot state');
    // Text typed while Live is stopped (display only: nothing is announced).
    const said = srStatus();
    env.app.set({ route: { view: 'speak', sub: 'live', id: '', path: 'speak/live', shot: 'live-prompt' } });
    await settle(env, 600);
    assert.equal($('spl-start-hint').hidden, false);
    assert.equal($('spl-start-hint-text').textContent, 'Press Start to hear this live.');
    assert.equal($('spl-start').getAttribute('data-emphasis'), 'true');
    assert.equal(srStatus(), said, 'a screenshot state says nothing');
    env.app.set({ route: { view: 'speak', sub: 'live', id: '', path: 'speak/live', shot: '' } });
    await settle(env);
    assert.equal($('spl-status-label').textContent, 'Stopped');
    assert.equal($('spl-error').hidden, true);
    assert.equal($('spl-start').getAttribute('aria-disabled'), null);
    assert.equal($('spl-start').hasAttribute('data-emphasis'), false, 'nothing was typed');
  });

  it('[LR-07] the live-menu screenshot state opens the menu under its button once the button has a box, never in the corner of the screen', async () => {
    await mount();
    const { placeMenu } = await load('components/menu.js');
    const menuEl = $('spl-more-menu');
    env.app.set({ route: { view: 'speak', sub: 'live', id: '', path: 'speak/live', shot: 'live-menu' } });
    await settle(env, 200);
    assert.equal(menuEl.hidden, true, 'it waits: a menu placed from an empty box would open at the top left');
    assert.equal(env.raf.pending, 0, 'on timers, not animation frames (a headless capture may draw none)');
    const rect = { x: 300, y: 120, width: 44, height: 44 };
    $('spl-more').__rect = rect;
    await settle(env, 20);
    assert.equal(menuEl.hidden, false);
    assert.equal(document.activeElement, $('spl-read-all'), 'the first item has focus');
    const size = { width: 220, height: 200 };   // the menu's size before it has one (menu.js)
    const spot = placeMenu($('spl-more').getBoundingClientRect(), size, { placement: 'bottom-end' });
    const corner = placeMenu({ left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0 }, size, { placement: 'bottom-end' });
    assert.equal(menuEl.style.getPropertyValue('--menu-y'), `${spot.y}px`);
    assert.equal(menuEl.style.getPropertyValue('--menu-x'), `${spot.x}px`);
    assert.ok(spot.y >= rect.y + rect.height && spot.y !== corner.y, `under the button (y ${spot.y})`);
    // The menu places itself before its first style: the view just opens it and adds no inline style.
    assert.equal(menuEl.style.getPropertyValue('transition-property'), '');
    await settle(env, 1000);
    assert.equal(menuEl.hidden, false, 'still open for the capture');
    assert.equal(env.sockets.length, 0);

    // A Live state whose controls never get a box (Speak shows Generate) gives up quietly.
    env.app.set({ route: { view: 'speak', sub: 'live', id: '', path: 'speak/live', shot: 'live-speaking' } });
    $('spl-more').__rect = null;
    const scrolled = $('spl').__scrolledIntoView.length;
    await settle(env, 5000);
    assert.equal($('spl').__scrolledIntoView.length, scrolled);
    assert.deepEqual(env.errors, []);
  });
});

describe('Live controls: coming back to Speak, the microphone, the contract', () => {
  it('[LR-20] coming back to Speak through the nav keeps Live running; switching to Generate inside Speak, or typing while Generate shows, stops it', async () => {
    await mount({ textValue: 'Text.' });
    env.click($('spl-start'));
    const srv = await openServer();
    // Live keeps reading on another view (the mini-player shows it)…
    env.app.set({ route: { view: 'history', sub: '', id: '', path: 'history', shot: '' } });
    assert.equal(env.live.isActive(), true);
    // …and the rail's or bottom bar's Speak link (#speak, g s) brings the listener back: nothing was switched,
    // so Live goes on (Speak shows Live mode for it and replaces the address with #speak/live).
    env.app.set({ route: { view: 'speak', sub: '', id: '', path: 'speak', shot: '' } });
    assert.equal(env.live.isActive(), true, 'the nav link is not a switch to Generate');
    env.app.set({ route: { view: 'speak', sub: 'live', id: '', path: 'speak/live', shot: '' } });
    await settle(env);
    assert.equal(env.live.isActive(), true);
    assert.equal(srv.ws.readyState, 1);
    assert.equal($('spl-status-label').textContent, 'Listening');
    assert.equal(sent(srv.ws, 'response.cancel').length, 0);

    // Typing while Speak shows Generate (entered with a hand-off) is meant for a take, not for Live.
    env.app.set({ route: { view: 'history', sub: '', id: '', path: 'history', shot: '' } });
    env.app.set({ route: { view: 'speak', sub: '', id: '', path: 'speak', shot: '' } });
    assert.equal(env.live.isActive(), true);
    env.input(env.editor, 'Text. For a take');
    await settle(env, 50);
    assert.equal(env.live.isActive(), false);
    assert.equal($('spl-status-label').textContent, 'Stopped: switched to Generate');
    assert.deepEqual(sent(srv.ws, 'input_text.append'), [], 'the take\'s text never reached Live (it read from the cursor, at the end)');

    // The switch made inside Speak (the mode control, Back from #speak/live) stops it at once.
    env.app.set({ route: { view: 'speak', sub: 'live', id: '', path: 'speak/live', shot: '' } });
    env.click($('spl-start'));
    await openServer();
    env.app.set({ route: { view: 'speak', sub: '', id: '', path: 'speak', shot: '' } });
    assert.equal(env.live.isActive(), false);
    assert.equal($('spl-status-label').textContent, 'Stopped: switched to Generate');
  });

  it('[LR-15] Resume waits while the microphone records: the toggle stays on Resume and the hint says why until the recording ends', async () => {
    await mount({ textValue: 'Hello.' });
    const audio = await load('core/audio.js');
    env.click($('spl-start'));
    const srv = await openServer();
    const ctx = FakeAudioContext.instances[0];
    env.app.set({ recording: { owner: 'transcribe', startedAt: Date.now() } });
    audio.pauseAll();                              // mic.open() pauses every player first
    await settle(env);
    assert.equal($('spl-start-label').textContent, 'Resume');
    assert.equal($('spl-start-hint').hidden, true);
    const before = sent(srv.ws).length;
    env.click($('spl-start'));                     // Resume while the microphone records
    await settle(env);
    assert.equal($('spl-start-label').textContent, 'Resume', 'still paused');
    assert.equal(ctx.state, 'suspended');
    assert.equal(sent(srv.ws).length, before, 'no playback.resume');
    assert.equal($('spl-start-hint').hidden, false);
    assert.equal($('spl-start-hint').getAttribute('data-tone'), 'alert');
    assert.equal($('spl-start-hint-text').textContent, 'Stop recording to listen.');
    assert.equal($('spl-start').getAttribute('aria-describedby'), 'spl-start-hint');
    assert.equal(srStatus(), 'Stop recording to listen.');

    env.app.set({ recording: null });              // the recording ends: the hint goes, nothing plays by itself
    await settle(env);
    assert.equal($('spl-start-hint').hidden, true);
    assert.equal($('spl-start-label').textContent, 'Resume');
    assert.equal(ctx.state, 'suspended');
    env.click($('spl-start'));
    await settle(env);
    assert.equal($('spl-start-label').textContent, 'Pause');
    assert.equal(ctx.state, 'running');
    assert.equal(sent(srv.ws).at(-1).type, 'playback.resume');
    env.live.stop();
  });

  it('[LR-07] mountLive keeps the contract Speak relies on: start, stop, togglePause, isActive and destroy (plus refresh, getState and ready)', async () => {
    const { ui } = await mount({ textValue: 'Contract.' });
    const live = env.live;
    for (const name of ['start', 'stop', 'togglePause', 'isActive', 'destroy', 'refresh', 'getState']) {
      assert.equal(typeof live[name], 'function', name);
    }
    assert.equal(typeof live.ready.then, 'function');
    assert.equal(ui.PARTIAL, '/static/partials/speak-live.html');
    assert.equal(live.start({ readAll: true }), true);
    assert.equal(live.start(), false, 'one session at a time');
    assert.equal(live.getState().phase, 'preparing');
    live.destroy();
    assert.equal(live.isActive(), false, 'destroy() stops the session');
    assert.equal(env.app.get().player, null, 'and releases the audio session');
    live.destroy();                                // twice is harmless
  });
});
