// The harness's own accessibility helpers: every view's tests rely on them for markup built at runtime.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv } from '../harness/env.js';
import { accessibleName, assertAccessibleNames, liveRegions, focusables } from '../harness/a11y.js';

let env;
afterEach(() => env?.restore());

describe('harness a11y helpers', () => {
  it('accessibleName follows aria-labelledby, aria-label, label[for], a wrapping label, content, then title', () => {
    env = installBrowserEnv({
      html: `<span id="lab">From labelledby</span>
        <input id="a" aria-labelledby="lab" aria-label="no">
        <input id="b" aria-label="From aria-label">
        <label for="c">From label for</label><input id="c">
        <label>Wrapped <input id="d"></label>
        <button id="e" type="button"><svg aria-hidden="true"></svg>Text</button>
        <input id="f" title="From title">
        <input id="g" placeholder="Never a name">`,
    });
    const names = Object.fromEntries([...'abcdefg'].map((id) => [id, accessibleName(document.getElementById(id))]));
    assert.deepEqual(names, { a: 'From labelledby', b: 'From aria-label', c: 'From label for', d: 'Wrapped', e: 'Text', f: 'From title', g: '' });
  });

  it('resolves label[for] and aria-labelledby inside a detached subtree a view is still building', () => {
    env = installBrowserEnv();
    const box = document.createElement('div');
    box.innerHTML = '<label for="x-name">Name</label><input id="x-name"><span id="x-lab">Speed</span><input type="range" aria-labelledby="x-lab">';
    assert.equal(accessibleName(box.querySelector('#x-name')), 'Name');
    assert.equal(accessibleName(box.querySelector('[type="range"]')), 'Speed');
    assertAccessibleNames(box);
  });

  it('assertAccessibleNames reports unnamed controls, buttons without a type and icon-only buttons without aria-label', () => {
    env = installBrowserEnv();
    const box = document.createElement('div');
    box.innerHTML = '<input id="q" placeholder="Search"><button><svg aria-hidden="true"></svg></button><button type="button" title="Close"><svg aria-hidden="true"></svg></button><div aria-hidden="true"><input></div><input type="hidden">';
    assert.throws(() => assertAccessibleNames(box), (error) => {
      assert.deepEqual(error.actual, [
        '<input#q> has no accessible name',
        '<button> has no accessible name',
        '<button> has no explicit type',
        '<button> is icon-only without aria-label',
        '<button> is icon-only without aria-label',
      ], 'aria-hidden and hidden inputs are skipped; a title alone does not label an icon button');
      return true;
    });
  });

  it('liveRegions lists implicit and explicit live regions; focusables skips disabled, hidden and inert content', () => {
    env = installBrowserEnv({
      html: `<p id="s" role="status"></p><div id="l" role="log" aria-live="polite"></div><p id="o" aria-live="off"></p>
        <button id="b1" type="button">One</button><button id="b2" type="button" disabled>Two</button>
        <div hidden><button id="b3" type="button">Three</button></div><a id="a1" href="#x">Link</a><span id="t" tabindex="0">T</span>`,
    });
    assert.deepEqual(liveRegions(document.body).map((r) => [r.el.id, r.role, r.live]), [['s', 'status', 'polite'], ['l', 'log', 'polite'], ['o', '', 'off']]);
    assert.deepEqual(focusables(document.body).map((el) => el.id), ['b1', 'a1', 't']);
  });
});
