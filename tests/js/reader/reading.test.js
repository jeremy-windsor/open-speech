// The reading surface: highlight, follow mode and "Back to playing", play from a paragraph (click,
// Enter, Space), paging during playback, and the aside's meta.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { readerEnv, makeDoc, progressAt, $, textOf, settle, go, load } from './helpers.js';

let ctx;
let scrolls;
afterEach(() => {
  delete globalThis.getSelection;
  ctx?.env.restore();
});

const para = (i) => document.querySelector(`#rd-text [data-block="${i}"]`);
const current = () => document.querySelector('#rd-text .rd-p.is-current');

async function openAt(options = {}) {
  ctx = await readerEnv({ hash: '#reader/alice', ...options });
  scrolls = [];
  window.scrollBy = (opts) => { scrolls.push(opts); };
  return ctx;
}

describe('views/reader/reading', () => {
  it('[RE-52] the current paragraph is .is-current with tabindex 0; sentence spans are built for it alone, by code point', async () => {
    const docs = [makeDoc({
      progress: progressAt(2),
      paragraph: (c, p) => (p === 2 ? 'Hi 😀 there. The 🐇 runs. Last one.' : `Plain paragraph ${p}. Second sentence.`),
    })];
    await openAt({ docs });
    assert.equal(current().getAttribute('data-block'), '2');
    assert.equal(current().getAttribute('tabindex'), '0');
    assert.equal(para(1).getAttribute('tabindex'), '-1');
    assert.equal(document.querySelectorAll('#rd-text [tabindex="0"]').length, 1, 'one tab stop in the text');
    assert.equal(document.querySelectorAll('#rd-text .rd-s').length, 0, 'no sentence spans until a sentence is spoken');
    const untouched = para(3).innerHTML;
    ctx.engine.advance(2, 1);
    const now = document.querySelector('.rd-s.is-now');
    assert.equal(textOf(now), 'The 🐇 runs.', 'the emoji before it does not shift the slice');
    assert.equal(now.closest('[data-block]').getAttribute('data-block'), '2');
    assert.equal(textOf(para(2)), 'Hi 😀 there. The 🐇 runs. Last one.', 'the text is unchanged by the spans');
    ctx.engine.advance(2, 2);
    assert.equal(document.querySelectorAll('.rd-s.is-now').length, 1);
    assert.equal(textOf(document.querySelector('.rd-s.is-now')), 'Last one.');
    ctx.engine.advance(3, 0);
    assert.equal(para(2).classList.contains('is-current'), false);
    assert.equal(para(2).getAttribute('tabindex'), '-1');
    assert.equal(para(2).querySelector('.rd-s.is-now'), null);
    assert.equal(current().getAttribute('data-block'), '3');
    assert.notEqual(untouched, para(3).innerHTML, 'block 3 got its spans only when it became current');
    assert.equal(document.querySelectorAll('#rd-text [data-split]').length, 2, 'spans exist only for paragraphs that were current');
  });

  it('[RE-53] while following, playback keeps the spoken paragraph in view; scrolling away stops it and shows Back to playing', async () => {
    await openAt();
    ctx.engine.play();
    await settle(ctx.env);
    const ret = $('rd-return');
    assert.equal(ret.hidden, true);
    para(13).__rect = { top: 2000, height: 120 };
    scrolls.length = 0;
    ctx.engine.advance(13);
    assert.equal(scrolls.length, 1, 'the next paragraph is scrolled into the band');
    window.dispatchEvent(new Event('wheel'));
    assert.equal(ret.hidden, false, 'the listener scrolled away: the pill shows');
    assert.equal(ctx.ctl.reading.follow, false);
    para(14).__rect = { top: 2400, height: 120 };
    scrolls.length = 0;
    ctx.engine.advance(14);
    assert.equal(scrolls.length, 0, 'not following: the page stays where the listener put it');
    ctx.env.click(ret);
    assert.equal(ctx.ctl.reading.follow, true);
    assert.equal(scrolls.length, 1, 'Back to playing scrolls to the spoken paragraph');
    assert.equal(ret.hidden, true);
    assert.equal(document.activeElement, para(14), 'focus lands on the paragraph being read');
  });

  it('[RE-53] arrow and page keys outside the controls stop following too; scrolling the aside or a sheet does not', async () => {
    await openAt();
    ctx.engine.play();
    await settle(ctx.env);
    const side = $('rd-side');
    side.dispatchEvent(new Event('touchmove', { bubbles: true }));
    window.dispatchEvent(Object.assign(new Event('touchmove'), {}));
    assert.equal(ctx.ctl.reading.follow, false, 'a touchmove on the page stops following');
    $('rd-return').click();
    assert.equal(ctx.ctl.reading.follow, true);
    const inSide = new Event('wheel', { bubbles: true });
    side.dispatchEvent(inSide);
    window.dispatchEvent(Object.assign(inSide, {}));
    ctx.env.key('PageDown', {}, document.body);
    assert.equal(ctx.ctl.reading.follow, false, 'PageDown stops following');
    $('rd-return').click();
    ctx.env.key('ArrowDown', {}, $('rd-chapter-list'));
    ctx.env.key('ArrowDown', {}, document.body);
    assert.equal(ctx.ctl.reading.follow, false);
    $('rd-return').click();
    ctx.env.key('PageDown', {}, document.querySelector('.rd-topbar button'));
    assert.equal(ctx.ctl.reading.follow, true, 'keys on the top bar belong to it');
  });

  it('[RE-54] a far jump (more than 3 screens) or reduced motion scrolls instantly; a near one smoothly', async () => {
    await openAt();
    ctx.engine.play();
    para(13).__rect = { top: 1000, height: 100 };
    scrolls.length = 0;
    ctx.engine.advance(13);
    assert.equal(scrolls[0].behavior, 'smooth');
    para(14).__rect = { top: 844 * 4, height: 100 };
    scrolls.length = 0;
    ctx.engine.playFrom(14);
    assert.equal(scrolls[0].behavior, 'auto', 'far: instant');
    document.documentElement.setAttribute('data-motion', 'reduce');
    para(13).__rect = { top: 1000, height: 100 };
    scrolls.length = 0;
    ctx.engine.playFrom(13);
    assert.equal(scrolls[0].behavior, 'auto', 'reduced motion: instant');
  });

  it('[RE-56] a click or Enter on a paragraph plays from it; a click while text is selected does not', async () => {
    await openAt();
    ctx.env.click(para(5));
    assert.deepEqual(ctx.engine.calls.at(-1), ['playFrom', 5]);
    ctx.engine.advance(5, 1);
    ctx.env.click(document.querySelector('[data-block="5"] .rd-s'));
    assert.deepEqual(ctx.engine.calls.at(-1), ['playFrom', 5], 'a click on a sentence plays its paragraph');
    globalThis.getSelection = () => ({ toString: () => 'selected words' });
    const before = ctx.engine.calls.length;
    ctx.env.click(para(7));
    assert.equal(ctx.engine.calls.length, before, 'selecting text never starts playback');
    delete globalThis.getSelection;
    para(9).focus();
    const enter = ctx.env.key('Enter');
    assert.equal(enter.defaultPrevented, true);
    assert.deepEqual(ctx.engine.calls.at(-1), ['playFrom', 9]);
  });

  it('[RE-56] Space on the current paragraph toggles play; Space on any other paragraph plays from it', async () => {
    await openAt();
    current().focus();
    ctx.env.key(' ');
    assert.deepEqual(ctx.engine.calls.slice(-2), [['unlock'], ['toggle']]);
    para(4).focus();
    ctx.env.key(' ');
    assert.deepEqual(ctx.engine.calls.at(-1), ['playFrom', 4]);
  });

  it('[REV-17] Space on the focused current paragraph toggles once (no restart) and the player shortcut does not fire too', async () => {
    await openAt();
    const audio = await load('core/audio.js');
    let sessionToggles = 0;
    audio.claim({
      id: 'reader:alice', owner: 'reader', kind: 'long', title: 'Chapter 1', homeRoute: 'reader/alice',
      controls: { play() {}, pause() {}, toggle() { sessionToggles += 1; }, stop() {} },
      getState: () => ({ playing: true, waiting: false, position: 0, duration: 10, rate: 1, canSeek: false, statusText: '' }),
    });
    await settle(ctx.env);
    current().focus();
    const down = ctx.env.key(' ');
    assert.equal(down.defaultPrevented, true);
    assert.equal(ctx.engine.calls.filter((c) => c[0] === 'toggle').length, 1);
    assert.equal(ctx.engine.calls.some((c) => c[0] === 'playFrom'), false, 'the paragraph is not restarted');
    assert.equal(sessionToggles, 0, "the player's global Space shortcut stays out of it");
    ctx.env.key(' ', {}, document.body);
    assert.equal(sessionToggles, 1, 'Space elsewhere is still the player shortcut');
  });

  it('[RE-62] the aside meta "<author> · N%" agrees with the library, and a finished document reads 100% until playback moves', async () => {
    const docs = [makeDoc({ progress: progressAt(15) })];   // 15 of 15 blocks: finished
    await openAt({ docs });
    assert.equal(textOf($('rd-side-meta')), 'Lewis Carroll · 100%');
    assert.equal(textOf($('rd-side-title')), "Alice's Adventures in Wonderland");
    assert.equal(current().getAttribute('data-block'), '14', 'the cursor is on the last paragraph');
    ctx.engine.playFrom(3);
    assert.match(textOf($('rd-side-meta')), /^Lewis Carroll · 20% · .+ left$/);
  });

  it('[RE-55] playback turns the page while following; a page link turns it by hand, stops following and focuses the opposite link', async () => {
    const docs = [makeDoc({ chapters: 3, perChapter: 800, progress: progressAt(10) })];
    await openAt({ docs });
    ctx.engine.play();
    await settle(ctx.env);
    assert.equal(document.querySelectorAll('#rd-text [data-block]').length, 800, 'one page (one chapter here) is in the DOM');
    const next = document.querySelector('.rd-page-link--next');
    assert.equal(textOf(next), 'Next: Chapter 2');
    ctx.engine.advance(800);
    assert.ok(para(800), 'playback past the page turned it');
    assert.equal(para(10), null);
    assert.equal(textOf(document.querySelector('.rd-page-link--prev')), 'Previous: Chapter 1');
    ctx.env.click(document.querySelector('.rd-page-link--next'));
    assert.ok(para(1600), 'the next page by hand');
    assert.equal(ctx.ctl.reading.follow, false, 'browsing stops following');
    assert.equal(document.activeElement, document.querySelector('.rd-page-link--prev'), 'focus goes to the link back');
    ctx.engine.advance(801);
    assert.ok(para(1600), 'not following: playback does not pull the page back');
    assert.equal($('rd-return').hidden, false);
    ctx.env.click($('rd-return'));
    assert.ok(para(801), 'Back to playing returns to the page being read');
    assert.equal(current().getAttribute('data-block'), '801');
  });

  it('with auto-scroll off nothing scrolls during playback and Back to playing stays available', async () => {
    await openAt();
    const prefs = await load('core/prefs.js');
    prefs.setReading({ autoScroll: false });
    ctx.engine.play();
    await settle(ctx.env);
    assert.equal($('rd-return').hidden, false);
    para(13).__rect = { top: 2000, height: 100 };
    scrolls.length = 0;
    ctx.engine.advance(13);
    assert.equal(scrolls.length, 0);
    ctx.env.click($('rd-return'));
    assert.equal(scrolls.length, 1);
  });

  it('[REV-25] the reading view has exactly one document title heading', async () => {
    await openAt();
    assert.equal(document.querySelectorAll('#rd-doc-title').length, 1);
    assert.equal(document.querySelectorAll('#view-reader [data-view-heading]').length, 2, 'the library h1 and the reading h1');
    assert.equal(textOf($('rd-top-chapter')), 'Chapter 3', 'the saved position (block 12) is in chapter 3');
    ctx.engine.advance(6);
    assert.equal(textOf($('rd-top-chapter')), 'Chapter 2', 'the chapter line follows playback');
  });

  it('after Close player (the engine keeps the document) a tap on a paragraph plays from it at once', async () => {
    await openAt();
    ctx.engine.play();
    await ctx.engine.unload();   // the mini-player's Close player: stop and save, the document stays loaded
    assert.equal(ctx.engine.state.docId, 'alice');
    assert.equal(ctx.engine.state.playing, false);
    const loads = ctx.engine.calls.filter((c) => c[0] === 'load').length;
    ctx.env.click(para(4));
    assert.deepEqual(ctx.engine.calls.at(-1), ['playFrom', 4]);
    assert.equal(ctx.engine.calls.filter((c) => c[0] === 'load').length, loads, 'no second load');
  });

  it('when the engine no longer holds the document on screen, a tap loads it again before playing', async () => {
    await openAt();
    await ctx.engine.unload({ forget: true });
    assert.equal(ctx.engine.state.docId, '');
    ctx.env.click(para(4));
    assert.deepEqual(ctx.engine.calls.slice(-2), [['load', 'alice'], ['playFrom', 4]]);
  });

  it('[RE-53] following playback costs one engine snapshot per sentence and none per frame (the real getter rebuilds it)', async () => {
    await openAt();
    ctx.engine.play();
    await settle(ctx.env);
    const snapshot = ctx.engine.state;
    ctx.engine.resetStateReads();
    for (let frame = 0; frame < 10; frame += 1) ctx.engine.emit('state', snapshot);   // ~10 frames while playing
    assert.equal(ctx.engine.stateReads, 0, "'state' handlers use the snapshot they are given");
    para(13).__rect = { top: 1000, height: 100 };
    ctx.engine.resetStateReads();
    ctx.engine.advance(13, 1);
    assert.ok(ctx.engine.stateReads <= 1, `a cursor event reads the engine state at most once (read ${ctx.engine.stateReads})`);
    assert.equal(document.querySelector('.rd-s.is-now').closest('[data-block]').getAttribute('data-block'), '13');
    assert.match(textOf($('rd-side-meta')), /^Lewis Carroll · 87% · .+ left$/, 'the meta follows from that one snapshot');
    ctx.engine.resetStateReads();
    await go(ctx, 'reader');   // the mini-player plays the book; the Reader view is elsewhere
    ctx.engine.resetStateReads();
    for (let frame = 0; frame < 10; frame += 1) ctx.engine.emit('state', snapshot);
    assert.equal(ctx.engine.stateReads, 0);
  });

  it('a document without text says so', async () => {
    const empty = makeDoc({ id: 'blank', chapters: 0 });
    await openAt({ hash: '#reader/blank', docs: [empty] });
    assert.equal(textOf($('rd-text')), 'This document has no readable text.');
    await go(ctx, 'reader');
  });
});
