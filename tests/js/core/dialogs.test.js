import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';
import { assertAccessibleNames } from '../harness/a11y.js';

let env;
afterEach(() => env?.restore());

const SHEET = `<main id="main"><button id="open" type="button">Chapters</button></main>
<dialog class="sheet" id="rd-chapters-sheet" aria-labelledby="rd-ch-title">
  <div class="sheet__head"><h2 id="rd-ch-title">Chapters</h2><button type="button" class="icon-btn" data-close aria-label="Close">x</button></div>
  <div class="sheet__body"><ol><li><button type="button" id="c1">One</button></li><li><button type="button" id="c2" data-autofocus>Two</button></li></ol></div>
</dialog>`;

async function setup(html = SHEET) {
  env = installBrowserEnv({ html });
  return importFresh('core/dialogs.js');
}

describe('core/dialogs.openSheet', () => {
  it('[RE-68] focuses its first choice ([data-autofocus]) on open and returns focus to the trigger', async () => {
    const d = await setup();
    const trigger = document.getElementById('open');
    trigger.focus();
    const dialog = document.getElementById('rd-chapters-sheet');
    d.openSheet(dialog);
    assert.equal(dialog.open, true);
    assert.ok(dialog.classList.contains('sheet--open'));
    assert.equal(document.activeElement.id, 'c2');
    assert.equal(d.topSheet(), dialog);
    d.closeSheet(dialog);
    assert.equal(dialog.open, false);
    assert.equal(document.activeElement, trigger);
    assert.equal(d.isModalOpen(), false);
  });

  it('[RE-68] closes on a backdrop click and on [data-close] buttons', async () => {
    const d = await setup();
    const dialog = document.getElementById('rd-chapters-sheet');
    const closed = [];
    d.openSheet(dialog, { onClose: (v) => closed.push(v) });
    dialog.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    assert.equal(dialog.open, false);
    d.openSheet(dialog, { onClose: (v) => closed.push(v) });
    env.click(dialog.querySelector('[data-close]'));
    assert.equal(dialog.open, false);
    assert.deepEqual(closed, ['backdrop', '']);
    await env.flush();
    assert.deepEqual(closed, ['backdrop', ''], 'the native close event that follows does not close twice');
  });

  it('Esc (the cancel event) closes and returns focus', async () => {
    const d = await setup();
    const trigger = document.getElementById('open');
    const dialog = document.getElementById('rd-chapters-sheet');
    d.openSheet(dialog, { trigger });
    env.key('Escape');
    assert.equal(dialog.open, false);
    assert.equal(document.activeElement, trigger);
  });

  it('focuses the heading when there is no [data-autofocus], or initialFocus when given', async () => {
    const d = await setup(SHEET.replace(' data-autofocus', ''));
    const dialog = document.getElementById('rd-chapters-sheet');
    d.openSheet(dialog);
    assert.equal(document.activeElement.id, 'rd-ch-title');
    d.closeSheet(dialog);
    d.openSheet(dialog, { initialFocus: '#c1' });
    assert.equal(document.activeElement.id, 'c1');
  });
});

describe('core/dialogs.confirm', () => {
  it('[AX-11] a destructive confirm focuses Cancel and resolves false on Esc', async () => {
    const d = await setup('<main id="main"><button id="del" type="button">Delete</button></main>');
    const del = document.getElementById('del');
    del.focus();
    const answer = d.confirm({ title: 'Delete files?', message: 'Delete downloaded files for Kokoro?', confirmLabel: 'Delete files', danger: true });
    await env.flush();
    const dialog = document.getElementById('app-confirm');
    assert.equal(dialog.open, true);
    assert.equal(document.activeElement.id, 'app-confirm-cancel');
    assert.equal(document.getElementById('app-confirm-title').textContent, 'Delete files?');
    assert.equal(document.getElementById('app-confirm-msg').textContent, 'Delete downloaded files for Kokoro?');
    assert.ok(document.getElementById('app-confirm-ok').classList.contains('btn--danger'));
    assertAccessibleNames(dialog);
    env.key('Escape');
    assert.equal(await answer, false);
    assert.equal(document.activeElement, del);
  });

  it('[AX-11] confirming resolves true; a plain confirm focuses the confirm button', async () => {
    const d = await setup('<main id="main"></main>');
    const answer = d.confirm({ title: 'Switch speech model?', message: 'Loading Piper will unload Kokoro. Continue?', confirmLabel: 'Load' });
    await env.flush();
    assert.equal(document.activeElement.id, 'app-confirm-ok');
    assert.equal(document.activeElement.textContent, 'Load');
    env.click(document.getElementById('app-confirm-ok'));
    assert.equal(await answer, true);
  });

  it('queues a second confirm until the first is answered', async () => {
    const d = await setup('<main id="main"></main>');
    const first = d.confirm({ title: 'One?' });
    const second = d.confirm({ title: 'Two?' });
    await env.flush();
    assert.equal(document.getElementById('app-confirm-title').textContent, 'One?');
    env.click(document.getElementById('app-confirm-ok'));
    assert.equal(await first, true);
    await env.flush();
    assert.equal(document.getElementById('app-confirm-title').textContent, 'Two?');
    env.click(document.getElementById('app-confirm-cancel'));
    assert.equal(await second, false);
  });

  it("reuses the shell's #app-confirm when it has the expected ids", async () => {
    const shell = `<dialog class="sheet sheet--alert" id="app-confirm" aria-labelledby="app-confirm-title" aria-describedby="app-confirm-msg"><h2 id="app-confirm-title" data-shell>t</h2><p id="app-confirm-msg"></p><button type="button" id="app-confirm-cancel">Cancel</button><button type="button" id="app-confirm-ok">OK</button></dialog>`;
    const d = await setup(shell);
    const answer = d.confirm({ title: 'Remove?' });
    await env.flush();
    assert.ok(document.getElementById('app-confirm-title').hasAttribute('data-shell'));
    env.click(document.getElementById('app-confirm-ok'));
    assert.equal(await answer, true);
  });
});

describe('core/dialogs.prompt', () => {
  it('[BUG-15] asks in a dialog (not window.prompt), focuses the field and returns the trimmed value', async () => {
    const d = await setup('<main id="main"></main>');
    assert.throws(() => window.prompt('x'), /not allowed/);
    const answer = d.prompt({ title: 'Save preset', label: 'Preset name', value: 'Audiobook', hint: 'Shown in Speak and Studio.' });
    await env.flush();
    const input = document.getElementById('app-prompt-input');
    assert.equal(document.activeElement, input);
    assert.equal(input.value, 'Audiobook');
    assert.equal(input.getAttribute('maxlength'), '200');
    assert.equal(input.getAttribute('aria-describedby'), 'app-prompt-hint');
    assertAccessibleNames(document.getElementById('app-prompt'));
    env.input(input, '  Night reading  ');
    env.key('Enter');
    assert.equal(await answer, 'Night reading');
  });

  it('[BUG-15] a validate error shows in the field (aria-invalid, aria-describedby) and keeps the dialog open', async () => {
    const d = await setup('<main id="main"></main>');
    const taken = ['Narrator'];
    const answer = d.prompt({ title: 'Name this voice', label: 'Voice name', validate: (v) => (taken.includes(v) ? 'That name is taken.' : null) });
    await env.flush();
    const input = document.getElementById('app-prompt-input');
    const error = document.getElementById('app-prompt-error');
    env.key('Enter');
    assert.equal(error.hidden, false);
    assert.equal(error.textContent, 'This field is required.');
    env.input(input, 'Narrator');
    env.click(document.getElementById('app-prompt-ok'));
    assert.equal(input.getAttribute('aria-invalid'), 'true');
    assert.equal(error.textContent, 'That name is taken.');
    assert.match(input.getAttribute('aria-describedby'), /app-prompt-error/);
    assert.equal(document.getElementById('app-prompt').open, true);
    env.input(input, 'Storyteller');
    assert.equal(input.hasAttribute('aria-invalid'), false, 'typing clears the error');
    env.click(document.getElementById('app-prompt-ok'));
    assert.equal(await answer, 'Storyteller');
  });

  it('cancel and Esc resolve null; a server conflict reopens with the error shown', async () => {
    const d = await setup('<main id="main"></main>');
    const first = d.prompt({ title: 'Rename', label: 'Name', value: 'A' });
    await env.flush();
    env.click(document.getElementById('app-prompt-cancel'));
    assert.equal(await first, null);
    const second = d.prompt({ title: 'Rename', label: 'Name', value: 'B', error: 'A voice named B already exists.' });
    await env.flush();
    assert.equal(document.getElementById('app-prompt-error').textContent, 'A voice named B already exists.');
    assert.equal(document.getElementById('app-prompt-input').getAttribute('aria-invalid'), 'true');
    env.key('Escape');
    assert.equal(await second, null);
  });

  it('a backdrop click does not close a prompt with edited input', async () => {
    const d = await setup('<main id="main"></main>');
    const answer = d.prompt({ title: 'Name', label: 'Name', value: 'x' });
    await env.flush();
    const dialog = document.getElementById('app-prompt');
    env.input(document.getElementById('app-prompt-input'), 'changed');
    dialog.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    assert.equal(dialog.open, true);
    env.click(document.getElementById('app-prompt-cancel'));
    assert.equal(await answer, null);
  });
});

describe('core/dialogs: what stays reachable while a sheet is open (foundation repair)', () => {
  const REGIONS = `<div class="toasts" id="app-toasts"><div id="app-toasts-polite" class="toasts__region" role="status" aria-live="polite"></div><div id="app-toasts-alert" class="toasts__region" role="alert" aria-live="assertive"></div></div>
<div id="app-sr-status" class="visually-hidden" role="status" aria-live="polite"></div><div id="app-sr-alert" class="visually-hidden" role="alert"></div>
<div id="app-tooltip" class="tooltip" role="tooltip" hidden></div>
<dialog class="sheet" id="second" aria-labelledby="second-title"><h2 id="second-title">Second</h2><button type="button" id="inner">Inner</button></dialog>`;
  const byId = (id) => document.getElementById(id);

  it('the announcement regions, the toast stack and the tooltip move into the top modal sheet (a modal makes the rest inert) and back after', async () => {
    const d = await setup(SHEET + REGIONS);
    const { announce } = await import('../../../src/static/js/core/a11y.js');
    const { toast } = await import('../../../src/static/js/core/toast.js');
    const sheet = byId('rd-chapters-sheet');
    const second = byId('second');
    d.openSheet(sheet);
    for (const id of d.LIVE_HOSTED) assert.equal(byId(id).parentNode, sheet, `#${id} is inside the open sheet`);
    announce('Bella added to favorites');
    env.raf.flush();
    assert.equal(sheet.querySelector('#app-sr-status').textContent, 'Bella added to favorites', 'announced from inside the modal');
    const shown = toast("Couldn't save the preset", { kind: 'error', action: { label: 'Retry', run: () => {} } });
    assert.ok(sheet.contains(shown.el), 'a toast shown now is inside the sheet: above the backdrop, clickable and announced');
    d.openSheet(second);
    for (const id of d.LIVE_HOSTED) assert.equal(byId(id).parentNode, second, 'a sheet opened on top takes them');
    d.closeSheet(second);
    for (const id of d.LIVE_HOSTED) assert.equal(byId(id).parentNode, sheet, 'back to the sheet underneath');
    d.closeSheet(sheet);
    for (const id of d.LIVE_HOSTED) assert.equal(byId(id).parentNode, document.body, 'back in <body> once no sheet is open');
    assert.ok(document.body.contains(shown.el), 'the toast is still there');
  });

  it('regions created while a sheet is open are created inside it', async () => {
    const d = await setup(SHEET);
    const { announce } = await import('../../../src/static/js/core/a11y.js');
    const sheet = byId('rd-chapters-sheet');
    d.openSheet(sheet);
    announce('Saved', { assertive: true });
    env.raf.flush();
    assert.equal(byId('app-sr-alert').parentNode, sheet);
    assert.equal(d.liveHost(), sheet);
    d.closeSheet(sheet);
    assert.equal(byId('app-sr-alert').parentNode, document.body);
  });

  it('a press that starts inside the sheet and ends on the backdrop (a text selection, a slider drag) does not close it', async () => {
    const d = await setup();
    const dialog = byId('rd-chapters-sheet');
    dialog.__rect = { left: 0, top: 400, right: 390, bottom: 844, width: 390, height: 444, x: 0, y: 400 };
    d.openSheet(dialog);
    byId('c1').dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, clientX: 50, clientY: 500 }));
    dialog.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, clientX: 50, clientY: 100 }));
    assert.equal(dialog.open, true, 'the press began inside the sheet');
    dialog.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, clientX: 50, clientY: 100 }));
    dialog.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true, clientX: 50, clientY: 100 }));
    assert.equal(dialog.open, false, 'a press on the backdrop closes it');
  });
});
