import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';
import { FakePointerEvent, FakeMouseEvent } from '../harness/dom.js';

let env;
afterEach(() => env?.restore());

const MENU = `<button type="button" class="btn" id="sort">Sort</button>
<div class="menu" id="sort-menu" role="menu" aria-label="Sort" hidden>
  <button type="button" class="menu__item" role="menuitemradio" aria-checked="false" data-v="recent">Recent</button>
  <button type="button" class="menu__item" role="menuitemradio" aria-checked="true" data-v="title">Title</button>
  <button type="button" class="menu__item" role="menuitemradio" aria-checked="false" data-v="progress">Progress</button>
  <div role="separator" class="menu__sep"></div>
  <button type="button" class="menu__item menu__item--danger" role="menuitem" data-v="delete">Delete</button>
</div>
<button type="button" id="other">Other</button>`;

async function setup(html = MENU) {
  env = installBrowserEnv({ html, viewport: { width: 400, height: 800 } });
  const mod = await importFresh('components/menu.js');
  mod.__resetForTests();
  const picked = [];
  const button = document.getElementById('sort');
  const menu = document.getElementById('sort-menu');
  const ctl = mod.bindMenuButton(button, menu, { onSelect: (item) => picked.push(item.getAttribute('data-v')) });
  return { mod, ctl, button, menu, picked };
}

describe('components/menu', () => {
  it('[AX-10] Esc closes the menu and returns focus to its trigger', async () => {
    const { button, menu } = await setup();
    button.focus();
    env.key('Enter');
    assert.equal(menu.hidden, false);
    assert.equal(button.getAttribute('aria-expanded'), 'true');
    assert.equal(document.activeElement.textContent, 'Title', 'it opens on the checked item');
    env.key('Escape');
    assert.equal(menu.hidden, true);
    assert.equal(button.getAttribute('aria-expanded'), 'false');
    assert.equal(document.activeElement, button);
  });

  it('APG keys: arrows wrap, Home/End, type-ahead; Enter activates, updates menuitemradio and closes', async () => {
    const { button, menu, picked } = await setup();
    assert.equal(button.getAttribute('aria-haspopup'), 'menu');
    assert.equal(button.getAttribute('aria-controls'), 'sort-menu');
    button.focus();
    env.key('ArrowUp');
    assert.equal(document.activeElement.textContent, 'Delete', 'ArrowUp opens on the last item');
    env.key('ArrowDown');
    assert.equal(document.activeElement.textContent, 'Recent', 'it wraps');
    env.key('End');
    assert.equal(document.activeElement.textContent, 'Delete');
    env.key('Home');
    assert.equal(document.activeElement.textContent, 'Recent');
    env.key('p');
    assert.equal(document.activeElement.textContent, 'Progress');
    env.key('Enter');
    assert.deepEqual(picked, ['progress']);
    assert.equal(menu.hidden, true);
    assert.equal(document.activeElement, button);
    const checked = [...menu.querySelectorAll('[aria-checked="true"]')].map((el) => el.getAttribute('data-v'));
    assert.deepEqual(checked, ['progress']);
  });

  it('Tab closes the menu and moves on from its button (the browser\'s own Tab runs from there); Shift+Tab stops on the button; a click on an item selects it', async () => {
    const { ctl, button, menu, picked } = await setup();
    env.click(button);
    assert.equal(ctl.isOpen(), true);
    const tab = env.key('Tab');
    assert.equal(menu.hidden, true);
    assert.equal(tab.defaultPrevented, false, 'the browser still moves focus on…');
    assert.equal(document.activeElement, button, '…from the button, wherever the menu element sits (WCAG 2.4.3)');
    env.click(button);
    env.key('ArrowDown');
    const back = env.key('Tab', { shiftKey: true });
    assert.equal(back.defaultPrevented, true, 'Shift+Tab stops on the button: the stop before a menu written after it');
    assert.equal(document.activeElement, button);
    assert.equal(menu.hidden, true);
    env.click(button);
    env.click(menu.querySelector('[data-v="recent"]'));
    assert.deepEqual(picked, ['recent']);
    assert.equal(menu.hidden, true);
  });

  it('a click outside closes the menu without pressing what was under the pointer', async () => {
    const { button, menu, picked } = await setup();
    let otherClicks = 0;
    const other = document.getElementById('other');
    other.addEventListener('click', () => { otherClicks += 1; });
    env.click(button);
    other.dispatchEvent(new FakePointerEvent('pointerdown', { bubbles: true }));
    assert.equal(menu.hidden, true);
    other.dispatchEvent(new FakeMouseEvent('click', { bubbles: true, cancelable: true }));
    assert.equal(otherClicks, 0, 'the click that follows the closing pointerdown is swallowed');
    other.dispatchEvent(new FakeMouseEvent('click', { bubbles: true, cancelable: true }));
    assert.equal(otherClicks, 1, 'only that one click');
    assert.deepEqual(picked, []);
  });

  it('[AX-10] inside a sheet, Esc closes only the menu, not the sheet', async () => {
    const html = `<dialog class="sheet" id="d" aria-labelledby="t"><h2 id="t">Voice</h2>${MENU}</dialog>`;
    env = installBrowserEnv({ html });
    const mod = await importFresh('components/menu.js');
    const dialogs = await importFresh('core/dialogs.js');
    const dialog = document.getElementById('d');
    dialogs.openSheet(dialog);
    const button = document.getElementById('sort');
    mod.bindMenuButton(button, document.getElementById('sort-menu'));
    env.click(button);
    env.key('Escape');
    assert.equal(document.getElementById('sort-menu').hidden, true);
    assert.equal(dialog.open, true, 'the sheet stays open');
    assert.equal(document.activeElement, button);
    env.key('Escape');
    assert.equal(dialog.open, false, 'the next Esc closes the sheet');
  });

  it('[AX-10] a menu written inside a .card (a size container) opens from the view section, placed at its button, and goes back into the card when it closes', async () => {
    const html = `<main><section class="view" id="view-voices" data-view="voices">
      <ul class="vo-grid" id="grid"><li class="card" id="card1"><h3>Narrator</h3>
        <button type="button" class="icon-btn" id="more" aria-label="More actions for Narrator">…</button>
        <div class="menu" id="card1-menu" role="menu" aria-label="Narrator" hidden>
          <button type="button" class="menu__item" role="menuitem" data-v="rename">Rename…</button>
          <button type="button" class="menu__item" role="menuitem" data-v="delete">Delete…</button>
        </div>
        <button type="button" class="btn" id="edit">Edit in Speak</button></li></ul>
      <p id="view-end">End</p></section></main>`;
    env = installBrowserEnv({ html, viewport: { width: 1024, height: 768 } });
    const mod = await importFresh('components/menu.js');
    const section = document.getElementById('view-voices');
    const card = document.getElementById('card1');
    const button = document.getElementById('more');
    const menu = document.getElementById('card1-menu');
    card.style.setProperty('container-type', 'inline-size');   // computed style in the fake browser
    button.__rect = { left: 600, top: 300, width: 44, height: 44 };
    menu.__rect = { width: 180, height: 96 };
    const picked = [];
    const ctl = mod.bindMenuButton(button, menu, { placement: 'bottom-end', onSelect: (item) => picked.push(item.getAttribute('data-v')) });
    const inPlace = () => menu.parentElement === card && menu.previousElementSibling === button;
    assert.ok(inPlace());

    button.focus();
    env.key('Enter');
    assert.equal(ctl.isOpen(), true);
    assert.equal(menu.parentElement, section, 'open: the menu lives in the view section (its fixed box is placed against the viewport)…');
    assert.equal(section.lastElementChild, menu, '…at its end');
    assert.equal(card.contains(menu), false);
    assert.equal(card.childNodes.some((n) => n.nodeType === 8), true, 'a placeholder keeps its place in the card');
    assert.equal(menu.style.getPropertyValue('--menu-x'), `${644 - 180}px`, 'bottom-end: its end edge on the button\'s');
    assert.equal(menu.style.getPropertyValue('--menu-y'), `${344 + 4}px`);
    assert.equal(document.activeElement.getAttribute('data-v'), 'rename');
    assert.equal(button.getAttribute('aria-controls'), 'card1-menu');

    env.key('Escape');
    assert.equal(menu.hidden, true);
    assert.ok(inPlace(), 'closed: back in the card, right after its button');
    assert.equal(card.childNodes.some((n) => n.nodeType === 8), false, 'no placeholder left behind');
    assert.equal(document.activeElement, button, 'Esc returns focus to the button');

    env.key('Enter');
    env.key('Tab');
    assert.ok(inPlace());
    assert.equal(document.activeElement, button, 'Tab goes on from the button (into the card), not from the end of the section');

    env.click(button);
    env.click(menu.querySelector('[data-v="delete"]'));
    assert.deepEqual(picked, ['delete'], 'onSelect runs for a re-hosted menu');
    assert.ok(inPlace());

    env.click(button);
    document.getElementById('view-end').dispatchEvent(new FakePointerEvent('pointerdown', { bubbles: true }));
    assert.equal(menu.hidden, true, 'a click outside closes it…');
    assert.ok(inPlace(), '…back in its place');
  });

  it('[AX-10] a menu needs no move when nothing traps it, and a menu in a sheet stays in that sheet (Esc closes only the menu)', async () => {
    const html = `<section class="view" id="view-speak" data-view="speak"><div class="sp-head" id="head">
      <button type="button" class="icon-btn" id="more" aria-label="More actions">…</button>
      <div class="menu" id="head-menu" role="menu" aria-label="Speak actions" hidden><button type="button" class="menu__item" role="menuitem">Clear text</button></div></div></section>
      <dialog class="sheet" id="sheet" aria-labelledby="sheet-t"><h2 id="sheet-t">Voice</h2><div class="sheet__body"><div class="card" id="sheet-card">
        <button type="button" class="btn" id="lang">Language</button>
        <div class="menu" id="lang-menu" role="menu" aria-label="Language" hidden><button type="button" class="menu__item" role="menuitemradio" aria-checked="true">English</button></div>
      </div></div></dialog>`;
    env = installBrowserEnv({ html });
    const mod = await importFresh('components/menu.js');
    const dialogs = await importFresh('core/dialogs.js');
    const head = document.getElementById('head');
    const headMenu = document.getElementById('head-menu');
    mod.bindMenuButton(document.getElementById('more'), headMenu);
    env.click(document.getElementById('more'));
    assert.equal(headMenu.hidden, false);
    assert.equal(headMenu.parentElement, head, 'no containing-block trap: the menu stays where it was written');
    env.key('Escape');

    const sheet = document.getElementById('sheet');
    const sheetCard = document.getElementById('sheet-card');
    sheetCard.style.setProperty('container-type', 'inline-size');
    dialogs.openSheet(sheet);
    const lang = document.getElementById('lang');
    const langMenu = document.getElementById('lang-menu');
    mod.bindMenuButton(lang, langMenu);
    env.click(lang);
    assert.equal(langMenu.parentElement, sheet, 'inside a modal sheet the host is the sheet (the rest of the page is inert)');
    env.key('Escape');
    assert.equal(langMenu.hidden, true);
    assert.equal(sheet.open, true, 'the sheet stays open');
    assert.equal(langMenu.parentElement, sheetCard, 'and the menu is back in its card');
    assert.equal(document.activeElement, lang);
  });

  it('a re-hosted menu whose card was re-rendered while it was open is removed when it closes (no stale copy, no duplicate id)', async () => {
    const html = `<section class="view" id="view-models" data-view="models"><div class="card" id="c"><button type="button" class="icon-btn" id="b" aria-label="More">…</button>
      <div class="menu" id="m" role="menu" aria-label="Model" hidden><button type="button" class="menu__item" role="menuitem">Delete files…</button></div></div></section>`;
    env = installBrowserEnv({ html });
    const mod = await importFresh('components/menu.js');
    const card = document.getElementById('c');
    card.style.setProperty('contain', 'layout paint');
    const menu = document.getElementById('m');
    const ctl = mod.bindMenuButton(document.getElementById('b'), menu);
    env.click(document.getElementById('b'));
    assert.equal(menu.parentElement, document.getElementById('view-models'));
    card.replaceChildren();   // the view re-rendered the card
    ctl.destroy();
    assert.equal(menu.isConnected, false);
    assert.equal(document.querySelectorAll('#m').length, 0);
  });

  it('trapsFixed names every style that makes an element the containing block of a fixed menu', async () => {
    const { mod } = await setup();
    const style = (props) => ({ getPropertyValue: (name) => props[name] ?? '' });
    const traps = [
      { 'container-type': 'inline-size' }, { 'container-type': 'size' }, { contain: 'layout' }, { contain: 'paint' },
      { contain: 'strict' }, { contain: 'content' }, { 'content-visibility': 'auto' }, { transform: 'matrix(1, 0, 0, 1, 0, 0)' },
      { translate: '0px 4px' }, { scale: '0.98' }, { rotate: '45deg' }, { perspective: '800px' }, { filter: 'blur(2px)' },
      { 'backdrop-filter': 'blur(16px)' }, { '-webkit-backdrop-filter': 'blur(16px)' }, { 'will-change': 'transform' },
    ];
    for (const props of traps) assert.equal(mod.trapsFixed(style(props)), true, JSON.stringify(props));
    const free = [
      {}, { 'container-type': 'normal' }, { contain: 'none' }, { contain: 'style' }, { contain: 'size' }, { 'content-visibility': 'visible' },
      { transform: 'none' }, { filter: 'none' }, { 'backdrop-filter': 'none' }, { 'will-change': 'auto' }, { 'will-change': 'opacity' }, { opacity: '0.5' },
    ];
    for (const props of free) assert.equal(mod.trapsFixed(style(props)), false, JSON.stringify(props));
    assert.equal(mod.trapsFixed(null), false);
  });

  it('the menu is placed before it is shown, so its first style already has its position; a later open starts from its last size', async () => {
    const { ctl, button, menu } = await setup();
    button.__rect = { left: 40, top: 100, width: 120, height: 44 };
    menu.__rect = { width: 240, height: 180 };
    const atShow = [];
    const toggle = menu.toggleAttribute.bind(menu);
    menu.toggleAttribute = (name, force) => {
      if (name === 'hidden' && force === false) atShow.push(['--menu-x', '--menu-y', '--menu-max-h', '--menu-w'].map((p) => menu.style.getPropertyValue(p)));
      return toggle(name, force);
    };
    ctl.open();
    assert.deepEqual(atShow[0], ['40px', '148px', `${800 - 144 - 4 - 8}px`, '120px'], 'top, left and the height limit are set while it is still hidden');
    ctl.close();
    button.__rect = { left: 40, top: 600, width: 120, height: 44 };
    ctl.open();
    assert.deepEqual(atShow[1].slice(0, 2), ['40px', `${600 - 4 - 180}px`], 'the guess uses the size measured last time (here it opens above: no room below)');
    assert.equal(menu.getAttribute('data-placement'), 'top');
  });

  it('placeMenu opens below, or above when there is no room below', async () => {
    const { mod } = await setup();
    const low = mod.placeMenu({ top: 700, bottom: 744, left: 20, right: 120, width: 100, height: 44 }, { width: 200, height: 240 }, { width: 400, height: 800 });
    assert.equal(low.up, true);
    assert.equal(low.y, 700 - 4 - 240);
    const high = mod.placeMenu({ top: 100, bottom: 144, left: 300, right: 390, width: 90, height: 44 }, { width: 200, height: 240 }, { width: 400, height: 800 });
    assert.equal(high.up, false);
    assert.equal(high.y, 148);
    assert.equal(high.x, 192, 'kept inside the viewport');
  });
});
