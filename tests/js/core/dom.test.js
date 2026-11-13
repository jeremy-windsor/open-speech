import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';

let env;
afterEach(() => env?.restore());

async function setup(options = {}) {
  env = installBrowserEnv({ html: '<main id="main"></main><div id="app-toasts"><div id="app-toasts-polite" role="status"></div><div id="app-toasts-alert" role="alert"></div></div>', ...options });
  return importFresh('core/dom.js');
}

describe('core/dom: safe markup', () => {
  it('html`` escapes every value, quotes included, and renders null/undefined/false as nothing', async () => {
    const { html, raw } = await setup();
    const name = '<img src=x onerror="alert(1)">\'&';
    const out = html`<p title="${name}">${name}${null}${undefined}${false}${0}${true}${raw('<b>ok</b>')}</p>`;
    assert.equal(out.toString(), '<p title="&lt;img src=x onerror=&quot;alert(1)&quot;&gt;&#39;&amp;">&lt;img src=x onerror=&quot;alert(1)&quot;&gt;&#39;&amp;0true<b>ok</b></p>');
  });

  it('html`` joins arrays and nests SafeHtml without double escaping', async () => {
    const { html } = await setup();
    const items = ['a<', 'b'].map((x) => html`<li>${x}</li>`);
    assert.equal(html`<ul>${items}</ul>`.toString(), '<ul><li>a&lt;</li><li>b</li></ul>');
    assert.equal(html`${['x', html`<i>y</i>`, null]}`.toString(), 'x<i>y</i>');
  });

  it('render() accepts SafeHtml only: a plain string throws TypeError', async () => {
    const { html, render, byId } = await setup();
    const main = byId('main');
    render(main, html`<p class="x">${'<script>'}</p>`);
    assert.equal(main.querySelector('p').textContent, '<script>');
    assert.equal(main.querySelector('script'), null);
    render(main, [html`<i>1</i>`, html`<i>2</i>`]);
    assert.equal(main.querySelectorAll('i').length, 2);
    assert.throws(() => render(main, '<b>x</b>'), TypeError);
    assert.throws(() => render(main, ['<b>x</b>']), TypeError);
  });

  it('escapeHtml covers & < > " \'', async () => {
    const { escapeHtml } = await setup();
    assert.equal(escapeHtml(`&<>"'`), '&amp;&lt;&gt;&quot;&#39;');
    assert.equal(escapeHtml(null), '');
  });
});

describe('core/dom: events', () => {
  it('[GS-04] buttons in built markup work through delegation (data-action), never inline handlers', async () => {
    const { html, render, actions, byId } = await setup();
    const main = byId('main');
    const seen = [];
    actions(main, { load: (el) => seen.push(`load:${el.dataset.model}`), unload: (el) => seen.push(`unload:${el.dataset.model}`) });
    render(main, html`<ul>${['kokoro', 'piper/en_US-amy-medium'].map((m) => html`<li><button type="button" data-action="load" data-model="${m}">Load</button><button type="button" data-action="unload" data-model="${m}">Unload</button></li>`)}</ul>`);
    const buttons = main.querySelectorAll('button');
    buttons[0].click();
    buttons[3].click();
    assert.deepEqual(seen, ['load:kokoro', 'unload:piper/en_US-amy-medium']);
    assert.ok(!main.innerHTML.match(/\son[a-z]+=/), 'no on*= attributes');
  });

  it('[BUG-13] a click on the SVG icon inside a button still finds the button (closest, not event.target.dataset)', async () => {
    const { html, render, actions, byId } = await setup();
    const main = byId('main');
    let got = null;
    actions(main, { remove: (el) => { got = el.dataset.name; } });
    render(main, html`<button type="button" data-action="remove" data-name="demo" aria-label="Delete demo"><svg class="ico" aria-hidden="true"><use href="/static/icons/sprite.svg#i-trash"></use></svg></button>`);
    main.querySelector('use').dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    assert.equal(got, 'demo');
  });

  it('actions ignore disabled and aria-disabled controls and unknown or inherited names', async () => {
    const { html, render, actions, byId } = await setup();
    const main = byId('main');
    let calls = 0;
    actions(main, { go: () => { calls += 1; } });
    render(main, html`<button type="button" data-action="go" disabled>a</button><button type="button" data-action="go" aria-disabled="true">b</button><button type="button" data-action="constructor">c</button><button type="button" data-action="toString">d</button>`);
    for (const b of main.querySelectorAll('button')) b.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    assert.equal(calls, 0);
  });

  it('[TE-08] a failing async handler is reported (default: an error toast); an aborted request stays silent', async () => {
    const { html, render, actions, byId } = await setup();
    const { ApiError } = await import('../../../src/static/js/core/api.js');
    const main = byId('main');
    actions(main, {
      fail: async () => { throw new Error('Delete failed on the server'); },
      sync: () => { throw new Error('sync boom'); },
      abort: async () => { throw new ApiError('Canceled', { kind: 'aborted', code: 'aborted' }); },
    });
    render(main, html`<button type="button" data-action="fail">x</button><button type="button" data-action="sync">y</button><button type="button" data-action="abort">z</button>`);
    for (const b of main.querySelectorAll('button')) b.click();
    await env.flush();
    const toasts = byId('app-toasts-alert').querySelectorAll('.toast__msg').map((p) => p.textContent);
    assert.deepEqual(toasts, ['sync boom', 'Delete failed on the server']);
    assert.deepEqual(env.errors, [], 'no unhandled errors');
  });

  it('[TE-08] actions() takes a custom onError (for an inline banner next to the control)', async () => {
    const { html, render, actions, byId } = await setup();
    const main = byId('main');
    const errors = [];
    actions(main, { go: async () => { throw new Error('nope'); } }, { onError: (e) => errors.push(e.message) });
    render(main, html`<button type="button" data-action="go">go</button>`);
    main.querySelector('button').click();
    await env.flush();
    assert.deepEqual(errors, ['nope']);
  });

  it('on() delegates with closest and stops matching outside the root', async () => {
    const { on, byId } = await setup({ html: '<div id="outer" class="hit"><div id="root"><span id="s"><b id="b">x</b></span></div></div><div id="toasts"></div>' });
    const root = byId('root');
    const got = [];
    const off = on(root, 'click', 'span, .hit', (e, el) => got.push(el.id));
    byId('b').click();
    byId('outer').click();
    off();
    byId('b').click();
    assert.deepEqual(got, ['s']);
  });
});

describe('core/dom: helpers', () => {
  it('setBusy sets aria-busy and disabled, swaps the label and restores it', async () => {
    const { setBusy, byId } = await setup({ html: '<button id="g" type="button" class="btn"><span class="btn__label">Generate</span></button>' });
    const b = byId('g');
    setBusy(b, true, { label: 'Generating…' });
    assert.equal(b.getAttribute('aria-busy'), 'true');
    assert.equal(b.disabled, true);
    assert.equal(b.querySelector('.btn__label').textContent, 'Generating…');
    setBusy(b, false);
    assert.equal(b.hasAttribute('aria-busy'), false);
    assert.equal(b.disabled, false);
    assert.equal(b.querySelector('.btn__label').textContent, 'Generate');
  });

  it('setHidden is a no-op when unchanged', async () => {
    const { setHidden, byId } = await setup({ html: '<p id="p"></p>' });
    const p = byId('p');
    setHidden(p, true);
    assert.equal(p.hidden, true);
    setHidden(p, false);
    assert.equal(p.hasAttribute('hidden'), false);
  });

  it('debounce and throttle run on the clock', async () => {
    const { debounce, throttle } = await setup();
    const seen = [];
    const d = debounce((v) => seen.push(`d${v}`), 500);
    d(1); d(2);
    env.clock.tick(499);
    assert.deepEqual(seen, []);
    env.clock.tick(1);
    assert.deepEqual(seen, ['d2']);
    d(3);
    d.flush();
    assert.deepEqual(seen, ['d2', 'd3']);
    const t = throttle((v) => seen.push(`t${v}`), 250);
    t(1); t(2); t(3);
    assert.deepEqual(seen.slice(2), ['t1']);
    env.clock.tick(250);
    assert.deepEqual(seen.slice(2), ['t1', 't3']);
  });

  it('isEditable covers fields, selects, contenteditable and widget roles, not buttons', async () => {
    const { isEditable, byId } = await setup({ html: '<input id="i"><input id="c" type="checkbox"><input id="r" type="range"><input id="sb" type="submit"><textarea id="t"></textarea><select id="s"></select><div id="ce" contenteditable="true"></div><div id="cb" role="combobox"></div><div id="sl" role="slider"></div><button id="b" type="button">b</button><p id="p"></p>' });
    for (const id of ['i', 'c', 'r', 't', 's', 'ce', 'cb', 'sl']) assert.equal(isEditable(byId(id)), true, id);
    for (const id of ['sb', 'b', 'p']) assert.equal(isEditable(byId(id)), false, id);
    assert.equal(isEditable(null), false);
  });

  it('focusFirst skips hidden and disabled controls and makes headings focusable', async () => {
    const { focusFirst, byId } = await setup({ html: '<section id="v"><button type="button" hidden>a</button><div hidden><button type="button">b</button></div><button type="button" disabled>c</button><h2 id="h">Title</h2><button type="button" id="d">d</button></section>' });
    const first = focusFirst(byId('v'));
    assert.equal(first.id, 'h');
    assert.equal(byId('h').getAttribute('tabindex'), '-1');
    assert.equal(document.activeElement.id, 'h');
    assert.equal(focusFirst(byId('v'), 'button').id, 'd');
  });

  it('nextFrame resolves on the next animation frame', async () => {
    const { nextFrame } = await setup();
    let done = false;
    nextFrame().then(() => { done = true; });
    await env.flush();
    assert.equal(done, false);
    env.raf.flush();
    await env.flush();
    assert.equal(done, true);
  });
});

describe('core/dom: setBusy focus and safeUrl (foundation repair)', () => {
  it('setBusy gives focus back to a control that had it when it went busy, and restores its earlier disabled state', async () => {
    const dom = await setup({ html: '<main id="main" tabindex="-1"><button type="button" id="save"><span class="btn__label">Save</span></button><button type="button" id="off" disabled>Off</button></main>' });
    const save = document.getElementById('save');
    save.focus();
    dom.setBusy(save, true, { label: 'Saving…' });
    assert.equal(save.disabled, true);
    save.blur();   // the browser's focus fixup: a focused control that becomes disabled loses focus to <body>
    assert.equal(document.activeElement, document.body);
    dom.setBusy(save, false);
    assert.equal(document.activeElement, save, 'keyboard focus is back on the button');
    assert.equal(save.querySelector('.btn__label').textContent, 'Save');
    const off = document.getElementById('off');
    dom.setBusy(off, true);
    dom.setBusy(off, false);
    assert.equal(off.disabled, true, 'a control that was disabled before stays disabled');
    save.focus();
    dom.setBusy(save, true);
    save.blur();
    document.getElementById('main').focus();   // the listener moved on meanwhile
    dom.setBusy(save, false);
    assert.equal(document.activeElement.id, 'main', 'focus is not pulled back');
  });

  it('safeUrl lets http(s), blob and relative URLs through, data:audio/image only when asked, and nothing else', async () => {
    const dom = await setup();
    assert.equal(dom.safeUrl('/api/conversations/c1/audio'), '/api/conversations/c1/audio');
    assert.equal(dom.safeUrl('https://example.com/a.mp3'), 'https://example.com/a.mp3');
    assert.equal(dom.safeUrl('blob:http://127.0.0.1:8100/x'), 'blob:http://127.0.0.1:8100/x');
    assert.equal(dom.safeUrl('javascript:alert(1)'), '');
    assert.equal(dom.safeUrl(' JaVaScRiPt:alert(1)'), '');
    assert.equal(dom.safeUrl('data:text/html,<b>x</b>'), '');
    assert.equal(dom.safeUrl('data:audio/wav;base64,AAAA'), '', 'media only on request');
    assert.equal(dom.safeUrl('data:audio/wav;base64,AAAA', { media: true }), 'data:audio/wav;base64,AAAA');
    assert.equal(dom.safeUrl('data:image/png;base64,AAAA', { image: true }), 'data:image/png;base64,AAAA');
    assert.equal(dom.safeUrl('data:image/svg+xml,<svg/>', { image: true }), '', 'no SVG data URLs');
    assert.equal(dom.safeUrl(''), '');
    assert.equal(dom.safeUrl(null), '');
  });
});
