// Speak › Ask AI: the editor's text goes to the shared Ask AI sheet
// (components/ask-ai.js) as a `text` input of POST /api/assistant/ask, and the answer is spoken in
// Speak's voice ("Ask AI, then speak the answer"). A fake server answers.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { mountSpeak, ready, settle, type, json } from './helpers.js';

let t;
afterEach(() => t?.env?.restore());

const PRESETS = [
  { id: 'voice', name: 'Voice', label: 'Answer', inputs: ['text', 'transcript'], output: 'speech', chunk_chars: 12000 },
  { id: 'book', name: 'Book', label: 'Rewrite for listening', inputs: ['document'], output: 'document', chunk_chars: 12000 },
  { id: 'translate', name: 'Translate', label: 'Translate', inputs: ['text'], output: 'text', chunk_chars: 12000 },
];

function assistant({ enabled = true, asks = [] } = {}) {
  return {
    'GET /api/assistant': () => json(200, { enabled, host: enabled ? '127.0.0.1:9' : null, model: 'qwen3', server_settings: {} }),
    'GET /api/assistant/presets': () => json(200, { presets: PRESETS }),
    'POST /api/assistant/ask': (req) => {
      asks.push(req.json);
      return json(200, { reply: 'Bonjour.', thinking: null, finish_reason: 'stop', elapsed_ms: 500, preset: req.json.preset, input: { kind: 'text', chars: 5 },
        audio: { format: 'mp3', content_type: 'audio/mpeg', base64: 'SUQzBAAAAAAA', bytes: 9 } });
    },
  };
}

describe('speak: Ask AI', () => {
  it('the button shows only while the assistant is on, and waits for text in the editor', async () => {
    t = await mountSpeak({ routes: assistant() });
    await ready(t);
    const ask = t.$('sp-ask');
    assert.equal(ask.hidden, false);
    type(t, '');
    assert.equal(ask.disabled, true, 'nothing to send');
    type(t, 'Hello.');
    assert.equal(ask.disabled, false);
    t.env.restore();
    t = await mountSpeak({ routes: assistant({ enabled: false }) });
    await ready(t);
    assert.equal(t.$('sp-ask').hidden, true);
  });

  it('sends the editor\'s text and speaks the answer in Speak\'s voice, whatever the preset\'s first output', async () => {
    const asks = [];
    t = await mountSpeak({ routes: assistant({ asks }) });
    await ready(t);
    type(t, 'Hello.');
    t.env.click(t.$('sp-ask'));
    await settle(t.env);
    await settle(t.env);
    assert.equal(t.$('app-ask-ai').open, true);
    assert.match(t.$('ask-sources').textContent, /The text in the editor/);
    t.env.change(t.$('ask-preset'), 'translate');
    assert.equal(t.$('ask-out-speak').checked, true, 'Speak starts with "speak it"');
    const voice = t.app.get().speakVoice;
    t.env.click(t.$('ask-send'));
    await settle(t.env);
    await settle(t.env);
    assert.deepEqual(asks[0].input, { kind: 'text', text: 'Hello.' });
    assert.equal(asks[0].preset, 'translate');
    assert.equal(asks[0].output.speech.model, voice.model);
    assert.equal(asks[0].output.speech.voice, voice.voice);
    assert.equal(t.$('ask-reply').textContent, 'Bonjour.');
  });
});
