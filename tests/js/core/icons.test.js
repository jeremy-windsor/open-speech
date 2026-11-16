import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { installBrowserEnv, importFresh, REPO_ROOT } from '../harness/env.js';

let env;
afterEach(() => env?.restore());

describe('core/icons', () => {
  it('ICONS mirrors the sprite symbol set (unique, no i- prefix)', async () => {
    env = installBrowserEnv();
    const { ICONS } = await importFresh('core/icons.js');
    assert.equal(new Set(ICONS).size, ICONS.length);
    // 'talk' is the Talk view's nav icon.
    for (const name of ['logo', 'book', 'play', 'pause', 'skip-back', 'skip-fwd', 'type', 'record', 'star-fill', 'dots', 'eye-off', 'monitor', 'keyboard', 'alert', 'info', 'help', 'grip', 'wand', 'layers', 'talk']) {
      assert.ok(ICONS.includes(name), name);
    }
    assert.ok(ICONS.every((n) => !n.startsWith('i-')));
    assert.equal(ICONS.length, 73);
  });

  it('icon() is decorative by default and references the sprite', async () => {
    env = installBrowserEnv();
    const { icon } = await importFresh('core/icons.js');
    assert.equal(icon('play').toString(), '<svg class="ico" aria-hidden="true" focusable="false"><use href="/static/icons/sprite.svg#i-play"></use></svg>');
    const labelled = icon('alert', { className: 'ico ico--lg', label: 'Error "x"' }).toString();
    assert.match(labelled, /role="img" aria-label="Error &quot;x&quot;"/);
    assert.doesNotMatch(labelled, /aria-hidden/);
    assert.throws(() => icon('nope'), TypeError);
  });

  it('every ICONS entry exists in the sprite', async () => {
    env = installBrowserEnv();
    const { ICONS } = await importFresh('core/icons.js');
    const sprite = resolve(REPO_ROOT, 'src/static/icons/sprite.svg');
    // The sprite is part of the app, so a missing one fails (it used to skip).
    assert.ok(existsSync(sprite), `the icon sprite is missing: ${sprite}`);
    const ids = [...readFileSync(sprite, 'utf8').matchAll(/<symbol[^>]*\sid="i-([^"]+)"/g)].map((m) => m[1]);
    assert.deepEqual([...ids].sort(), [...ICONS].sort());
  });
});
