import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';

let env;
afterEach(() => env?.restore());

async function setup(options = {}) {
  env = installBrowserEnv({ html: '<button id="copy" type="button">Copy</button>', ...options });
  return importFresh('core/clipboard.js');
}

describe('core/clipboard.copyText', () => {
  it('uses the Clipboard API in a secure context', async () => {
    const { copyText } = await setup();
    assert.equal(await copyText('hello'), true);
    assert.deepEqual(env.clipboard.written, ['hello']);
    assert.deepEqual(document.__execCommand.calls, []);
  });

  it('over plain HTTP (no Clipboard API) it falls back to a hidden textarea and returns focus to the button', async () => {
    const { copyText } = await setup({ secure: false, clipboard: 'absent' });
    const button = document.getElementById('copy');
    button.focus();
    assert.equal(await copyText('transcript text'), true);
    assert.deepEqual(document.__execCommand.calls, ['copy']);
    assert.deepEqual(env.clipboard.written, ['transcript text']);
    assert.equal(document.activeElement, button);
    assert.equal(document.querySelector('textarea'), null, 'the helper textarea is removed');
  });

  it('a refused Clipboard API falls back too; a failed fallback returns false', async () => {
    const { copyText } = await setup({ clipboard: 'reject' });
    assert.equal(await copyText('x'), true);
    assert.deepEqual(document.__execCommand.calls, ['copy']);
    env.execCommandResult = false;
    assert.equal(await copyText('y'), false);
  });

  it('nothing to copy does nothing', async () => {
    const { copyText } = await setup();
    assert.equal(await copyText(''), false);
    assert.deepEqual(env.clipboard.written, []);
  });
});
