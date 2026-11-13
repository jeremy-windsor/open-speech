// History › Ask AI: an entry's text goes to the shared Ask AI sheet
// (components/ask-ai.js): a transcript by its history id (the server reads the full text), a speech entry's text
// as typed text (POST /api/assistant/ask). The rows' menus offer it only while the assistant is on.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { historyEnv, historyServer, speech, transcript, json, settle } from './helpers.js';

let t;
afterEach(() => t?.env?.restore());

const PRESETS = [
  { id: 'voice', name: 'Voice', label: 'Answer', inputs: ['text', 'transcript'], output: 'speech', chunk_chars: 12000 },
  { id: 'book', name: 'Book', label: 'Rewrite for listening', inputs: ['document'], output: 'document', chunk_chars: 12000 },
];

function assistant({ enabled = true, asks = [] } = {}) {
  return {
    'GET /api/assistant': () => json(200, { enabled, host: enabled ? '127.0.0.1:9' : null, model: 'qwen3', server_settings: {} }),
    'GET /api/assistant/presets': () => json(200, { presets: PRESETS }),
    'POST /api/assistant/ask': (req) => {
      asks.push(req.json);
      return json(200, { reply: 'A plan.', thinking: null, finish_reason: 'stop', elapsed_ms: 400, preset: 'voice', input: { kind: 'transcript', chars: 40 } });
    },
  };
}

const menuItems = (row) => [...row.querySelectorAll('[data-menu-cmd]')].map((item) => item.getAttribute('data-menu-cmd'));

async function pickFromMenu(row, cmd) {
  t.env.click(row.querySelector('[data-menu-btn]'));
  await settle(t.env);
  t.env.click(row.querySelector(`[data-menu-cmd="${cmd}"]`));
  await settle(t.env);
  await settle(t.env);
}

describe('history: Ask AI', () => {
  it('each row\'s menu offers Ask AI while the assistant is on; never while it is off', async () => {
    t = await historyEnv({ server: historyServer([speech(1), transcript(2)]), routes: assistant() });
    assert.deepEqual(menuItems(t.rows()[0]), ['ask-ai', 'copy-text', 'delete-entry']);
    assert.deepEqual(menuItems(t.rows()[1]), ['ask-ai', 'delete-entry']);
    t.env.restore();
    t = await historyEnv({ server: historyServer([speech(1)]), routes: assistant({ enabled: false }) });
    assert.deepEqual(menuItems(t.rows()[0]), ['copy-text', 'delete-entry']);
  });

  it('a transcript goes by its history id; a speech entry\'s text as typed text', async () => {
    const asks = [];
    t = await historyEnv({ server: historyServer([speech(1), transcript(2)]), routes: assistant({ asks }) });
    await pickFromMenu(t.rows()[1], 'ask-ai');
    const dialog = document.getElementById('app-ask-ai');
    assert.equal(dialog.open, true);
    assert.match(document.getElementById('ask-sources').textContent, /The transcript of meeting-2\.m4a/);
    t.env.change(document.getElementById('ask-out-speak'), false);
    t.env.click(document.getElementById('ask-send'));
    await settle(t.env);
    await settle(t.env);
    assert.deepEqual(asks[0].input, { kind: 'transcript', history_id: 's2' });
    t.env.click(document.getElementById('ask-cancel'));
    await settle(t.env);
    assert.equal(dialog.open, false);
    await pickFromMenu(t.rows()[0], 'ask-ai');
    t.env.click(document.getElementById('ask-send'));
    await settle(t.env);
    await settle(t.env);
    assert.equal(asks[1].input.kind, 'text');
    assert.equal(asks[1].input.text, speech(1).full_text);
  });
});
