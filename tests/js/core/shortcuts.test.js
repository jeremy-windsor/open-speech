import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';

let env;
afterEach(() => env?.restore());

const MARKUP = '<main id="main"><textarea id="sp-input" aria-label="Text to speak"></textarea><input id="field" aria-label="f"><button id="btn" type="button">b</button><p id="p" tabindex="-1">text</p></main><dialog id="app-shortcuts-sheet"><h2>Shortcuts</h2><button type="button" id="close">x</button></dialog>';

async function setup(options = {}) {
  env = installBrowserEnv({ html: MARKUP, ...options });
  const shortcuts = await importFresh('core/shortcuts.js');
  const { app } = await import('../../../src/static/js/core/store.js');
  const prefs = await import('../../../src/static/js/core/prefs.js');
  const dialogs = await import('../../../src/static/js/core/dialogs.js');
  prefs.load();
  return { shortcuts, app, prefs, dialogs };
}

const at = (id) => document.getElementById(id);

describe('core/shortcuts', () => {
  it('[REV-11] Mod+Enter with allowInEditable fires inside the Speak editor; a single character never does', async () => {
    const { shortcuts, app } = await setup();
    app.set({ route: { view: 'speak', sub: '', id: '', path: 'speak', shot: '' } });
    const fired = [];
    shortcuts.register({ id: 'generate', keys: ['Mod+Enter'], scope: 'view:speak', allowInEditable: true, description: 'Generate', handler: () => fired.push('generate') });
    shortcuts.register({ id: 'help', keys: ['?'], description: 'Shortcuts', handler: () => fired.push('help') });
    at('sp-input').focus();
    const ev = env.key('Enter', { ctrlKey: true });
    assert.equal(ev.defaultPrevented, true);
    env.key('?', { shiftKey: true });
    assert.deepEqual(fired, ['generate']);
    at('btn').focus();
    env.key('?', { shiftKey: true });
    assert.deepEqual(fired, ['generate', 'help']);
  });

  it('[REV-11] allowInEditable is refused for single characters', async () => {
    const { shortcuts } = await setup();
    assert.throws(() => shortcuts.register({ keys: ['k'], allowInEditable: true, handler() {} }), /allowInEditable/);
    assert.throws(() => shortcuts.register({ keys: ['Shift+ArrowLeft'], allowInEditable: true, handler() {} }), /allowInEditable/);
    assert.doesNotThrow(() => shortcuts.register({ keys: ['Escape'], allowInEditable: true, handler() {} }));
    assert.doesNotThrow(() => shortcuts.register({ keys: ['Alt+ArrowUp'], allowInEditable: true, handler() {} }));
  });

  it('Mod is Cmd on Apple platforms and Ctrl elsewhere', async () => {
    const { shortcuts } = await setup({ platform: 'MacIntel' });
    let n = 0;
    shortcuts.register({ keys: ['Mod+Enter'], allowInEditable: true, handler: () => { n += 1; } });
    at('sp-input').focus();
    env.key('Enter', { ctrlKey: true });
    assert.equal(n, 0);
    env.key('Enter', { metaKey: true });
    assert.equal(n, 1);
    assert.equal(shortcuts.modLabel(), '⌘');
  });

  it('an element handler that called preventDefault wins', async () => {
    const { shortcuts } = await setup();
    let n = 0;
    shortcuts.register({ keys: ['k'], scope: 'global', handler: () => { n += 1; } });
    at('p').addEventListener('keydown', (e) => e.preventDefault());
    at('p').focus();
    env.key('k');
    assert.equal(n, 0);
    at('btn').focus();
    env.key('k');
    assert.equal(n, 1);
  });

  it('nothing fires while a modal sheet is open, except that sheet\'s own shortcuts', async () => {
    const { shortcuts, dialogs } = await setup();
    const fired = [];
    shortcuts.register({ keys: ['k'], handler: () => fired.push('global') });
    shortcuts.register({ keys: ['j'], scope: 'dialog:app-shortcuts-sheet', handler: () => fired.push('sheet') });
    dialogs.openSheet(at('app-shortcuts-sheet'));
    env.key('k');
    env.key('j');
    dialogs.closeSheet(at('app-shortcuts-sheet'));
    env.key('k');
    env.key('j');
    assert.deepEqual(fired, ['sheet', 'global']);
  });

  it('single-key shortcuts obey prefs.shortcuts (WCAG 2.1.4); chords and non-character keys keep working', async () => {
    const { shortcuts, prefs } = await setup();
    const fired = [];
    shortcuts.register({ keys: ['Space', 'k'], scope: 'global', handler: () => fired.push('toggle') });
    shortcuts.register({ keys: ['Shift+ArrowRight'], handler: () => fired.push('next-sentence') });
    prefs.set({ shortcuts: false });
    at('p').focus();
    env.key('k');
    env.key(' ');
    env.key('ArrowRight', { shiftKey: true });
    assert.deepEqual(fired, ['next-sentence']);
    prefs.set({ shortcuts: true });
    env.key('k');
    assert.deepEqual(fired, ['next-sentence', 'toggle']);
    assert.equal(shortcuts.list().find((s) => s.keys.includes('k')).singleKey, true, 'character keys are single-key shortcuts');
  });

  it('Space on a focused button activates the button, not the shortcut', async () => {
    const { shortcuts } = await setup();
    const fired = [];
    shortcuts.register({ keys: ['Space'], handler: () => fired.push('play') });
    at('btn').addEventListener('click', () => fired.push('button'));
    at('btn').focus();
    env.key(' ');
    assert.deepEqual(fired, ['button']);
    at('p').focus();
    env.key(' ');
    assert.deepEqual(fired, ['button', 'play']);
  });

  it('Space and Enter on a focused native media player (<audio controls>) belong to that player, not the shortcut', async () => {
    const { shortcuts } = await setup();
    const fired = [];
    shortcuts.register({ keys: ['Space'], scope: 'global', handler: () => fired.push('space') });
    shortcuts.register({ keys: ['Enter'], scope: 'global', handler: () => fired.push('enter') });
    const clip = env.mount('<audio id="clip" controls aria-label="Your clip"></audio>', at('main'));
    clip.focus();
    assert.equal(document.activeElement, clip);
    // Keys pressed in the browser's own controls (its shadow root) reach the page with the <audio> as their target.
    const space = env.key(' ');
    const enter = env.key('Enter');
    assert.deepEqual(fired, []);
    assert.equal(space.defaultPrevented, false, "the player's own Space handling is not cancelled");
    assert.equal(enter.defaultPrevented, false);
    at('p').focus();
    env.key(' ');
    assert.deepEqual(fired, ['space']);
  });

  it('[ and ] typed with AltGr (Windows reports Ctrl+Alt with AltGraph) fire; a real Ctrl+Alt, Ctrl, Alt or Meta chord does not', async () => {
    const { shortcuts } = await setup();
    const fired = [];
    shortcuts.register({ keys: ['['], handler: () => fired.push('[') });
    shortcuts.register({ keys: [']'], handler: () => fired.push(']') });
    shortcuts.register({ keys: ['Space'], handler: () => fired.push('space') });
    at('p').focus();
    // German, French, Nordic, Spanish layouts: AltGr+8 / AltGr+9 type [ and ].
    const slower = env.key('[', { ctrlKey: true, altKey: true, modifierAltGraph: true });
    env.key(']', { ctrlKey: true, altKey: true, modifierAltGraph: true });
    assert.deepEqual(fired, ['[', ']']);
    assert.equal(slower.defaultPrevented, true);
    env.key('[', { ctrlKey: true, altKey: true });   // Ctrl+Alt+[ on a US layout is a chord
    env.key('[', { ctrlKey: true });
    env.key('[', { altKey: true });
    env.key('[', { metaKey: true });
    env.key('[', { metaKey: true, modifierAltGraph: true });
    env.key(' ', { ctrlKey: true, altKey: true, modifierAltGraph: true });   // AltGr types no space
    assert.deepEqual(fired, ['[', ']'], 'chords never fire a single-key shortcut');
  });

  it('[ and ] typed with Option on Apple platforms fire; Cmd and Ctrl chords do not', async () => {
    const { shortcuts } = await setup({ platform: 'MacIntel' });
    const fired = [];
    shortcuts.register({ keys: ['['], handler: () => fired.push('[') });
    shortcuts.register({ keys: [']'], handler: () => fired.push(']') });
    shortcuts.register({ keys: ['Space'], handler: () => fired.push('space') });
    at('p').focus();
    // German Mac layout: Option+5 / Option+6 type [ and ].
    env.key('[', { altKey: true });
    env.key(']', { altKey: true });
    assert.deepEqual(fired, ['[', ']']);
    env.key('[', { metaKey: true });
    env.key('[', { ctrlKey: true });
    env.key('[', { metaKey: true, altKey: true });
    env.key(' ', { altKey: true });
    assert.deepEqual(fired, ['[', ']'], 'chords never fire a single-key shortcut');
  });

  it("a pending 'g' sequence wins over a single-key shortcut on its second key", async () => {
    const { shortcuts, app } = await setup();
    app.set({ route: { view: 'transcribe', sub: '', id: '', path: 'transcribe', shot: '' } });
    const fired = [];
    shortcuts.register({ keys: ['g r'], description: 'Go to Reader', handler: () => fired.push('go-reader') });
    shortcuts.register({ keys: ['r'], scope: 'view:transcribe', description: 'Record', handler: () => fired.push('record') });
    at('p').focus();
    env.key('g');
    env.key('r');
    assert.deepEqual(fired, ['go-reader']);
    env.key('r');
    assert.deepEqual(fired, ['go-reader', 'record']);
    env.key('g');
    env.clock.tick(1001);
    env.key('r');
    assert.deepEqual(fired, ['go-reader', 'record', 'record'], 'the sequence expires after 1 s');
  });

  it('view: shortcuts fire only while that view is shown; the more specific scope wins', async () => {
    const { shortcuts, app } = await setup();
    const fired = [];
    shortcuts.register({ keys: ['/'], scope: 'view:history', handler: () => fired.push('history-search') });
    shortcuts.register({ keys: ['k'], scope: 'player', handler: () => fired.push('mini-player') });
    shortcuts.register({ keys: ['k'], scope: 'view:reader', handler: () => fired.push('reader') });
    at('p').focus();
    env.key('/');
    env.key('k');
    app.set({ route: { view: 'history', sub: '', id: '', path: 'history', shot: '' }, player: { id: 'take:1' } });
    env.key('/');
    env.key('k');
    app.set({ route: { view: 'reader', sub: '', id: 'd', path: 'reader/d', shot: '' } });
    env.key('k');
    assert.deepEqual(fired, ['history-search', 'mini-player', 'reader']);
  });

  it('when() gates a shortcut; off() removes it; list() describes them for the "?" sheet', async () => {
    const { shortcuts } = await setup();
    let enabled = false;
    let n = 0;
    const off = shortcuts.register({ id: 'x', keys: ['['], description: 'Speed down', when: () => enabled, handler: () => { n += 1; } });
    at('p').focus();
    env.key('[');
    enabled = true;
    env.key('[');
    assert.equal(n, 1);
    assert.deepEqual(shortcuts.list(), [{ id: 'x', keys: ['['], description: 'Speed down', scope: 'global', singleKey: true }]);
    off();
    env.key('[');
    assert.equal(n, 1);
    assert.deepEqual(shortcuts.list(), []);
  });
});
