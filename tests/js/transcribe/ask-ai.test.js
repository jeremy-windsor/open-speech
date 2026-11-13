// Transcribe › Ask AI: the transcript goes to the shared Ask AI sheet
// (components/ask-ai.js) as a `transcript` input of POST /api/assistant/ask. A fake server answers.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { mountView, settle, chooseFile, audioFile, verbose, json } from './helpers.js';

let t;
afterEach(() => t?.env?.restore());

const PRESETS = [
  { id: 'voice', name: 'Voice', label: 'Answer', inputs: ['text', 'transcript'], output: 'speech', chunk_chars: 12000 },
  { id: 'book', name: 'Book', label: 'Rewrite for listening', inputs: ['document'], output: 'document', chunk_chars: 12000 },
];

function assistantRoutes({ enabled = true, asks = [] } = {}) {
  return {
    'GET /api/assistant': () => json(200, { enabled, host: enabled ? '127.0.0.1:9' : null, model: 'qwen3', server_settings: {} }),
    'GET /api/assistant/presets': () => json(200, { presets: PRESETS }),
    'GET /api/models/:id/status': () => json(200, { state: 'loaded' }),
    'POST /v1/audio/transcriptions': () => json(200, verbose()),
    'POST /api/assistant/ask': (req) => {
      asks.push(req.json);
      return json(200, { reply: 'Two greetings.', thinking: null, finish_reason: 'stop', elapsed_ms: 900, preset: 'voice', input: { kind: 'transcript', chars: 28 } });
    },
  };
}

async function transcribeFile() {
  chooseFile(t, audioFile('meeting.m4a', 4096, 'audio/mp4'));
  await settle(t.env);
  await settle(t.env);
}

describe('transcribe: Ask AI', () => {
  it('the button shows only while the assistant is on, and waits for a transcript like Copy', async () => {
    t = await mountView({ routes: assistantRoutes() });
    await settle(t.env);
    const ask = t.$('tr-ask');
    assert.equal(ask.hidden, false, 'on: shown');
    assert.equal(ask.disabled, true, 'no transcript yet');
    assert.equal(ask.getAttribute('aria-controls'), 'app-ask-ai');
    await transcribeFile();
    assert.equal(ask.disabled, false);
    assert.equal(t.$('tr-copy').disabled, false);
  });

  it('off: the button stays hidden', async () => {
    t = await mountView({ routes: assistantRoutes({ enabled: false }) });
    await settle(t.env);
    assert.equal(t.$('tr-ask').hidden, true);
  });

  it('sends the transcript\'s words (no timestamps) as a transcript input, and shows the answer', async () => {
    const asks = [];
    t = await mountView({ routes: assistantRoutes({ asks }) });
    await settle(t.env);
    await transcribeFile();
    t.env.click(t.$('tr-ask'));
    await settle(t.env);
    await settle(t.env);
    const dialog = t.$('app-ask-ai');
    assert.equal(dialog.open, true);
    assert.match(t.$('ask-sources').textContent, /The transcript of meeting\.m4a/);
    assert.match(t.$('ask-sources').textContent, /Hello there\. General Kenobi\./);
    t.env.change(t.$('ask-out-speak'), false);
    t.env.click(t.$('ask-send'));
    await settle(t.env);
    await settle(t.env);
    assert.equal(asks.length, 1);
    assert.equal(asks[0].input.kind, 'transcript');
    assert.equal(asks[0].input.text, 'Hello there. General Kenobi.');
    assert.equal(asks[0].preset, 'voice');
    assert.equal(t.$('ask-reply').textContent, 'Two greetings.');
  });

  it('the screenshot state ?shot=ask-ai opens the sheet with a sample transcript and sends nothing', async () => {
    const asks = [];
    t = await mountView({ routes: assistantRoutes({ asks }) });
    t.ctl.show({ view: 'transcribe', sub: '', id: '', path: 'transcribe', shot: 'ask-ai' }, { focus: 'none' });
    await settle(t.env);
    await settle(t.env);
    assert.equal(t.$('app-ask-ai').open, true);
    assert.equal(asks.length, 0);
  });
});
