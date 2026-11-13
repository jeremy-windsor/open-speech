// The reading appearance sheet: every control applies at once to #rd-reading and persists in
// open-speech-reading; the spoken sentence is marked by tint and underline by default (REV-35).
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { readerEnv, $, textOf, settle, go, load } from './helpers.js';
import { assertAccessibleNames } from '../harness/a11y.js';

let ctx;
afterEach(() => ctx?.env.restore());

const reading = () => $('rd-reading');
const stored = () => JSON.parse(localStorage.getItem('open-speech-reading') || '{}');
const radio = (name, value) => document.querySelector(`input[name="${name}"][value="${value}"]`);

async function openSheet() {
  ctx.env.click($('rd-appearance-btn'));
  await settle(ctx.env);
  assert.equal($('rd-appearance-sheet').open, true);
}

describe('views/reader/appearance', () => {
  it('[REV-35] by default the spoken sentence gets a tint and an underline ("Both"), never colour alone', async () => {
    ctx = await readerEnv({ hash: '#reader/alice' });
    const prefs = await load('core/prefs.js');
    assert.equal(prefs.READING_DEFAULTS.highlight, 'both');
    assert.equal(reading().getAttribute('data-rd-hl'), 'both');
    await openSheet();
    assert.equal(radio('rd-hl', 'both').checked, true);
    assert.deepEqual(Array.from(document.querySelectorAll('input[name="rd-hl"]')).map((r) => r.value), ['tint', 'underline', 'both', 'off']);
    ctx.env.click(radio('rd-hl', 'tint'));
    assert.equal(reading().getAttribute('data-rd-hl'), 'tint');
    assert.equal(stored().highlight, 'tint');
  });

  it('every control applies at once to the reading surface and is kept in open-speech-reading', async () => {
    ctx = await readerEnv({ hash: '#reader/alice' });
    await openSheet();
    ctx.env.click(radio('rd-face', 'legible'));
    ctx.env.click(radio('rd-leading', 'relaxed'));
    ctx.env.click(radio('rd-measure', 'narrow'));
    ctx.env.click(radio('rd-paper', 'sepia'));
    ctx.env.click($('rd-justify'));
    ctx.env.click($('rd-autoscroll'));
    const el = reading();
    assert.equal(el.getAttribute('data-rd-face'), 'legible');
    assert.equal(el.getAttribute('data-rd-leading'), 'relaxed');
    assert.equal(el.getAttribute('data-rd-measure'), 'narrow');
    assert.equal(el.getAttribute('data-rd-paper'), 'sepia');
    assert.equal(el.getAttribute('data-rd-justify'), 'true');
    assert.deepEqual(
      { face: stored().face, leading: stored().leading, measure: stored().measure, paper: stored().paper, justify: stored().justify, autoScroll: stored().autoScroll },
      { face: 'legible', leading: 'relaxed', measure: 'narrow', paper: 'sepia', justify: true, autoScroll: false },
    );
    assertAccessibleNames($('rd-appearance-sheet'));
    // Reload: the saved appearance comes back.
    ctx.env.restore();
    ctx = await readerEnv({ hash: '#reader/alice', seed: { 'open-speech-reading': JSON.stringify({ v: 1, paper: 'night', face: 'sans', size: 5 }) } });
    assert.equal(reading().getAttribute('data-rd-paper'), 'night');
    assert.equal(reading().getAttribute('data-rd-face'), 'sans');
    assert.equal(reading().getAttribute('data-rd-size'), '5');
  });

  it('text size: 19 px by default on a phone, steps of the reading size scale, and the ends are aria-disabled', async () => {
    ctx = await readerEnv({ hash: '#reader/alice' });
    await openSheet();
    assert.equal(reading().hasAttribute('data-rd-size'), false, 'the responsive default until the listener picks one');
    assert.equal(textOf($('rd-size-value')), '19 px');
    ctx.env.click($('rd-size-inc'));
    assert.equal(reading().getAttribute('data-rd-size'), '3');
    assert.equal(textOf($('rd-size-value')), '20 px');
    for (let i = 0; i < 6; i += 1) ctx.env.click($('rd-size-inc'));
    assert.equal(textOf($('rd-size-value')), '28 px');
    assert.equal($('rd-size-inc').getAttribute('aria-disabled'), 'true');
    assert.equal(document.activeElement === $('rd-size-inc') || !$('rd-size-inc').disabled, true, 'aria-disabled keeps focus on the button');
    for (let i = 0; i < 9; i += 1) ctx.env.click($('rd-size-dec'));
    assert.equal(textOf($('rd-size-value')), '15 px');
    assert.equal(reading().getAttribute('data-rd-size'), '0');
    assert.equal($('rd-size-dec').getAttribute('aria-disabled'), 'true');
    assert.equal($('rd-size-inc').getAttribute('aria-disabled'), 'false');
    assert.equal($('rd-size-value').getAttribute('aria-live'), 'polite', 'the new size is announced');
  });

  it('text size defaults to 20 px from 960 px', async () => {
    ctx = await readerEnv({ hash: '#reader/alice', viewport: { width: 1440, height: 900 }, media: { 'min-width: 960px': true } });
    await openSheet();
    assert.equal(textOf($('rd-size-value')), '20 px');
  });

  it('Reset restores the defaults', async () => {
    ctx = await readerEnv({ hash: '#reader/alice' });
    await openSheet();
    ctx.env.click(radio('rd-paper', 'dusk'));
    ctx.env.click($('rd-size-inc'));
    ctx.env.click(radio('rd-hl', 'off'));
    ctx.env.click($('rd-appearance-reset'));
    const el = reading();
    assert.equal(el.getAttribute('data-rd-paper'), 'auto');
    assert.equal(el.hasAttribute('data-rd-size'), false);
    assert.equal(el.getAttribute('data-rd-hl'), 'both');
    assert.equal(radio('rd-paper', 'auto').checked, true);
    assert.equal($('rd-autoscroll').checked, true);
    assert.equal(stored().paper, 'auto');
  });

  it('a change made elsewhere (Settings) applies live; leaving the view closes the sheet', async () => {
    ctx = await readerEnv({ hash: '#reader/alice' });
    const prefs = await load('core/prefs.js');
    prefs.setReading({ paper: 'paper', measure: 'wide' });
    assert.equal(reading().getAttribute('data-rd-paper'), 'paper');
    assert.equal(reading().getAttribute('data-rd-measure'), 'wide');
    await openSheet();
    assert.equal(radio('rd-paper', 'paper').checked, true);
    await go(ctx, 'speak', 50);
    assert.equal($('rd-appearance-sheet').open, false);
  });

  it('the player opens the sheet through mountPlayer\'s onOpenAppearance', async () => {
    ctx = await readerEnv({ hash: '#reader/alice' });
    ctx.mounted[0].options.onOpenAppearance($('rd-player-slot'));
    await settle(ctx.env);
    assert.equal($('rd-appearance-sheet').open, true);
  });
});
