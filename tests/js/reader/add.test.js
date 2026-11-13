// The Add sheet: File | Paste, validation, the upload and paste requests, busy state, duplicates,
// errors and the library reload.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { readerEnv, $, textOf, settle, json } from './helpers.js';
import { delay } from '../harness/fakes.js';
import { assertAccessibleNames } from '../harness/a11y.js';

let ctx;
afterEach(() => ctx?.env.restore());

async function openAdd() {
  ctx.env.click($('rd-add'));
  await settle(ctx.env);
  assert.equal($('rd-add-dialog').open, true);
}

function chooseFile(name, content = 'Chapter one.\n\nIt begins.', type = 'text/plain') {
  const input = $('rd-file');
  input.files = [new File([content], name, { type })];
  input.dispatchEvent(new Event('change', { bubbles: true }));
}

function source(value) {
  ctx.env.click(document.querySelector(`input[name="rd-source"][value="${value}"]`));
}

const posts = () => ctx.env.fetch.callsTo('POST /v1/reader/documents');

describe('views/reader/add', () => {
  it('[RL-11] the sheet offers only File and Paste; the file input accepts the Reader types and is reached through its label', async () => {
    ctx = await readerEnv();
    await openAdd();
    const radios = Array.from(document.querySelectorAll('input[name="rd-source"]'));
    assert.deepEqual(radios.map((r) => r.value), ['file', 'paste']);
    assert.equal(radios[0].checked, true);
    assert.equal(document.activeElement, radios[0], 'focus starts on the chosen source');
    assert.equal($('rd-file').getAttribute('accept'), '.epub,.txt,.md,.markdown,.html,.htm');
    assert.ok($('rd-file').classList.contains('visually-hidden'));
    assert.equal($('rd-dropzone').getAttribute('for'), 'rd-file');
    assert.equal($('rd-add-paste-panel').hidden, true);
    source('paste');
    assert.equal($('rd-add-paste-panel').hidden, false);
    assert.equal($('rd-add-file-panel').hidden, true);
    assert.equal($('rd-paste-title').getAttribute('maxlength'), '200');
    assertAccessibleNames($('rd-add-dialog'));
  });

  it('[RL-11] a file of another type shows a field error; an accepted one shows its name and size; drag and drop works', async () => {
    ctx = await readerEnv();
    await openAdd();
    chooseFile('scan.pdf');
    assert.equal($('rd-file-error').hidden, false);
    assert.equal(textOf($('rd-file-error')), 'Choose an EPUB, TXT, Markdown or HTML file');
    assert.equal($('rd-file').getAttribute('aria-invalid'), 'true');
    assert.equal($('rd-dropzone').classList.contains('has-file'), false);
    chooseFile('alice.epub', 'x'.repeat(2048), 'application/epub+zip');
    assert.equal($('rd-file-error').hidden, true);
    assert.equal($('rd-file').hasAttribute('aria-invalid'), false);
    assert.equal($('rd-dropzone').classList.contains('has-file'), true);
    assert.equal(textOf($('rd-file-title')), 'alice.epub');
    assert.equal(textOf($('rd-file-hint')), '2 KB · Choose another file');
    const drop = new Event('drop', { bubbles: true, cancelable: true });
    drop.dataTransfer = { files: [new File(['# Notes'], 'notes.md')] };
    $('rd-dropzone').dispatchEvent(drop);
    assert.equal(drop.defaultPrevented, true, 'the browser does not open the dropped file');
    assert.equal(textOf($('rd-file-title')), 'notes.md');
  });

  it('[RL-12] a file goes up as multipart "file"; the button says "Uploading…" and the form is aria-busy meanwhile', async () => {
    ctx = await readerEnv({ routes: { 'POST /v1/reader/documents': () => delay(300, json(201, { id: 'new', title: 'Alice' })) } });
    await openAdd();
    ctx.env.click($('rd-add-submit'));
    await settle(ctx.env);
    assert.equal(textOf($('rd-file-error')), 'Choose a file first');
    assert.equal(document.activeElement, $('rd-file'), 'focus goes to the file input (its label shows the ring)');
    assert.equal(posts().length, 0);
    chooseFile('alice.txt');
    ctx.env.click($('rd-add-submit'));
    await settle(ctx.env, 10);
    assert.equal($('rd-add-submit').getAttribute('aria-busy'), 'true');
    assert.equal(textOf($('rd-add-submit')), 'Uploading…');
    assert.equal($('rd-add-form').getAttribute('aria-busy'), 'true');
    const [call] = posts();
    assert.ok(call.body instanceof FormData);
    assert.equal(call.body.get('file').name, 'alice.txt');
    const gets = ctx.env.fetch.callsTo('GET /v1/reader/documents').length;
    await settle(ctx.env, 300);
    assert.equal($('rd-add-dialog').open, false, 'success closes the sheet');
    assert.match(textOf($('app-toasts-polite')), /Added “Alice”/);
    assert.equal(ctx.env.fetch.callsTo('GET /v1/reader/documents').length, gets + 1, 'the library reloads');
    assert.equal(textOf($('rd-file-title')), 'Drop a book or text file', 'the form is reset');
    assert.equal($('rd-add-form').hasAttribute('aria-busy'), false);
    assert.equal(textOf($('rd-add-submit')), 'Add to library');
  });

  it('[RL-12] pasted text goes as JSON {type, text, title?}; empty text is refused with focus; a duplicate says it is already there', async () => {
    ctx = await readerEnv({ routes: { 'POST /v1/reader/documents': (req) => json(req.json.title ? 201 : 200, { id: 'x', title: req.json.title || 'Notes for Tuesday' }) } });
    await openAdd();
    source('paste');
    ctx.env.click($('rd-add-submit'));
    await settle(ctx.env);
    assert.equal(textOf($('rd-paste-error')), 'Paste some text first');
    assert.equal($('rd-paste-text').getAttribute('aria-invalid'), 'true');
    assert.equal(document.activeElement, $('rd-paste-text'));
    ctx.env.input($('rd-paste-text'), '  Buy milk.  ');
    assert.equal($('rd-paste-error').hidden, true, 'typing clears the error');
    ctx.env.click($('rd-add-submit'));
    await settle(ctx.env);
    assert.deepEqual(posts()[0].json, { type: 'text', text: 'Buy milk.' });
    assert.match(textOf($('app-toasts-polite')), /Already in your library: “Notes for Tuesday”/);
    await openAdd();
    source('paste');
    ctx.env.input($('rd-paste-title'), 'Groceries');
    ctx.env.input($('rd-paste-text'), 'Eggs.');
    ctx.env.click($('rd-add-submit'));
    await settle(ctx.env);
    assert.deepEqual(posts()[1].json, { type: 'text', text: 'Eggs.', title: 'Groceries' });
    assert.match(textOf($('app-toasts-polite')), /Added “Groceries”/);
  });

  it('[RL-12] a refused import\'s Details read as the server\'s message and code, never the JSON envelope; none when they would only repeat the message', async () => {
    let answer = json(422, { error: { message: 'Unsupported file type: .pdf', code: 'unsupported_format' } });
    ctx = await readerEnv({ routes: { 'POST /v1/reader/documents': () => answer } });
    await openAdd();
    source('paste');
    ctx.env.input($('rd-paste-text'), 'Some text.');
    ctx.env.click($('rd-add-submit'));
    await settle(ctx.env);
    const banner = $('rd-add-error');
    assert.equal(textOf(banner.querySelector('.banner__msg')), 'Unsupported file type: .pdf');
    assert.equal(banner.querySelector('details pre').textContent, 'Unsupported file type: .pdf (unsupported_format)');
    assert.doesNotMatch(banner.textContent, /[{}"]/, 'no JSON anywhere in the banner');
    answer = json(413, { error: { message: 'Text is larger than 20 MB', code: 'http_error' } });
    ctx.env.click($('rd-add-submit'));
    await settle(ctx.env);
    assert.equal(textOf(banner.querySelector('.banner__msg')), 'Text is larger than 20 MB');
    // (Booleans, not nodes: a failing assert would inspect the whole fake DOM through its getters.)
    assert.equal(!!banner.querySelector('details'), false, 'the generic code adds nothing: no Details');
  });

  it('[RL-12] a refused import shows an inline banner in the sheet, which stays open with the text', async () => {
    ctx = await readerEnv({ routes: { 'POST /v1/reader/documents': () => json(413, { detail: 'Text is larger than 20 MB' }) } });
    await openAdd();
    source('paste');
    ctx.env.input($('rd-paste-text'), 'Very long text.');
    ctx.env.click($('rd-add-submit'));
    await settle(ctx.env);
    const banner = $('rd-add-error');
    assert.equal(banner.hidden, false);
    assert.equal(banner.getAttribute('role'), 'alert');
    assert.equal(textOf(banner.querySelector('.banner__title')), "Couldn't add the document");
    assert.equal(textOf(banner.querySelector('.banner__msg')), 'Text is larger than 20 MB');
    assert.equal($('rd-add-dialog').open, true);
    assert.equal($('rd-paste-text').value, 'Very long text.');
  });

  it('a click on the backdrop never throws away pasted text', async () => {
    ctx = await readerEnv();
    await openAdd();
    const dialog = $('rd-add-dialog');
    source('paste');
    ctx.env.input($('rd-paste-text'), 'Keep me.');
    dialog.dispatchEvent(new Event('pointerdown', { bubbles: true }));
    dialog.click();
    await settle(ctx.env);
    assert.equal(dialog.open, true);
  });
});
