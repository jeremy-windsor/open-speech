import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';

let env;
afterEach(() => env?.restore());

async function setup() {
  env = installBrowserEnv({ html: '<span id="t" class="tile"></span>' });
  return importFresh('core/tiles.js');
}

describe('core/tiles', () => {
  it('[TH-04] white monogram text keeps 4.5:1 at both gradient stops for every hue; the second stop is hue + 40°', async () => {
    const t = await setup();
    for (let hue = 0; hue < 360; hue += 1) {
      const l1 = t.tileLightness(hue, 52, 46);
      const l2 = t.tileLightness((hue + 40) % 360, 58, 34);
      assert.ok(t.contrastWithWhite(t.hslToRgb(hue, 52, l1)) >= 4.5, `hue ${hue} stop 1`);
      assert.ok(t.contrastWithWhite(t.hslToRgb((hue + 40) % 360, 58, l2)) >= 4.5, `hue ${hue} stop 2`);
    }
    for (const seed of ['doc-1', 'a9c3f0e1', 'The Hobbit', '']) {
      const c = t.tileColors(seed);
      assert.equal(c.hue2, (c.hue + 40) % 360);
      assert.equal(c.hue, t.hueFor(seed));
    }
  });

  it('[RL-09] monogram: first letters of the first two significant words (a leading small word kept), digits, one word -> two letters, empty -> ?', async () => {
    const t = await setup();
    assert.equal(t.monogram('The Hobbit'), 'TH');
    assert.equal(t.monogram('Alice in Wonderland'), 'AW');
    assert.equal(t.monogram('A Tale of Two Cities'), 'AT');
    assert.equal(t.monogram('1984'), '19');
    assert.equal(t.monogram('war and peace'), 'WP');
    assert.equal(t.monogram('Dune'), 'DU');
    assert.equal(t.monogram('  — !! '), '?');
    assert.equal(t.monogram(''), '?');
    assert.equal(t.monogram('élan vital'), 'ÉV');
  });

  it("[RL-09] monogram: an apostrophe inside a word keeps it one word, so a possessive or a contraction never gives its tail letter", async () => {
    const t = await setup();
    assert.equal(t.monogram("Alice's Adventures in Wonderland"), 'AA', 'not "AS"');
    assert.equal(t.monogram('Don’t Panic'), 'DP', 'a typographic apostrophe too, not "DT"');
    assert.equal(t.monogram("The Hitchhiker's Guide to the Galaxy"), 'TH');
    assert.equal(t.monogram("O'Brien"), 'OB', 'one word: its first two letters, never the apostrophe');
    assert.equal(t.monogram("L'Été"), 'LÉ');
    assert.equal(t.monogram("'Tis the Season"), 'TS', 'a leading quote is not part of a word');
    assert.equal(t.monogram("Rock 'n' Roll"), 'RN');
    assert.equal(t.monogram("Can't"), 'CA');
  });

  it('hueFor is stable and in range', async () => {
    const t = await setup();
    assert.equal(t.hueFor('abc'), t.hueFor('abc'));
    for (const s of ['x', 'longer text', '😀']) assert.ok(t.hueFor(s) >= 0 && t.hueFor(s) < 360);
  });

  it('applyTile sets only the three custom properties', async () => {
    const t = await setup();
    const el = document.getElementById('t');
    const c = t.applyTile(el, 'doc-1');
    assert.equal(el.style.getPropertyValue('--tile-hue'), String(c.hue));
    assert.equal(el.style.getPropertyValue('--tile-l1'), `${c.light}%`);
    assert.equal(el.style.getPropertyValue('--tile-l2'), `${c.dark}%`);
    assert.equal(el.style.length, 3);
    assert.equal(el.getAttribute('style'), null, 'no style attribute in markup');
  });

  it('artworkDataUrl draws the tile and monogram on a canvas (a PNG data URL), or "" without a canvas', async () => {
    const t = await setup();
    const url = t.artworkDataUrl('doc-1', 'The Hobbit', 256);
    assert.match(url, /^data:image\/png;base64,/);
    const original = document.createElement;
    document.createElement = () => { throw new Error('no canvas'); };
    try {
      assert.equal(t.artworkDataUrl('doc-1', 'x', 96), '');
    } finally {
      document.createElement = original;
    }
  });
});
