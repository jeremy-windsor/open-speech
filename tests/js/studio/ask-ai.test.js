// Studio › Ask AI: a conversation's script goes to the shared Ask AI sheet
// (components/ask-ai.js) as a `text` input of POST /api/assistant/ask, a "Speaker: words" line per
// turn. The conversation's menu offers it only while the assistant is on.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';
import { studioEnv, studioServer, teaParty, json } from './helpers.js';

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
      return json(200, { reply: 'A rude tea party.', thinking: null, finish_reason: 'stop', elapsed_ms: 400, preset: 'voice', input: { kind: 'text', chars: 90 } });
    },
  };
}

const SCRIPT = 'Hatter: Your hair wants cutting.\nAlice: You should learn not to make personal remarks.\nIt’s very rude.\nMarch Hare: Then you should say what you mean.';

describe('studio: Ask AI about a script', () => {
  it('scriptText: a "Speaker: words" line per turn, in order; turns without words are left out', async () => {
    installBrowserEnv();
    const { scriptText } = await importFresh('views/studio/conversations.js');
    assert.equal(scriptText(teaParty()), SCRIPT);
    assert.equal(scriptText({ turns: [{ speaker: 'A', text: '  ' }, { speaker: '', text: 'Hi' }] }), 'Speaker: Hi');
    assert.equal(scriptText(null), '');
  });

  it('the conversation\'s menu offers it while the assistant is on, and sends the script as text', async () => {
    const asks = [];
    t = await studioEnv({ path: 'studio/conversations/conv-1', server: studioServer({ conversations: [teaParty()] }), routes: assistant({ asks }) });
    const more = t.$('st-conv-more');
    assert.deepEqual(t.menuItems(more).map((item) => item.textContent.trim()), ['Ask AI about the script…', 'Delete conversation…']);
    await t.menu(more, 'Ask AI about the script…');
    assert.equal(document.getElementById('app-ask-ai').open, true);
    assert.match(document.getElementById('ask-sources').textContent, /The script of The Mad Tea-Party/);
    t.env.change(document.getElementById('ask-out-speak'), false);
    await t.click(document.getElementById('ask-send'));
    assert.deepEqual(asks[0], { input: { kind: 'text', text: SCRIPT }, preset: 'voice' });
  });

  it('off: no Ask AI in the menu', async () => {
    t = await studioEnv({ path: 'studio/conversations/conv-1', server: studioServer({ conversations: [teaParty()] }), routes: assistant({ enabled: false }) });
    assert.deepEqual(t.menuItems(t.$('st-conv-more')).map((item) => item.textContent.trim()), ['Delete conversation…']);
  });

  // A turn is left out only when its own text is empty or only spaces, judged on the turn's
  // text, never on the formatted line, so a turn whose words end with a colon is part of the script.
  it('scriptText keeps a turn whose words end with a colon; only a turn without words is left out', async () => {
    installBrowserEnv();
    const { scriptText } = await importFresh('views/studio/conversations.js');
    const conv = {
      turns: [
        { speaker: 'Host', text: 'Welcome back. Today we cover three things:' },
        { speaker: 'Guest', text: 'Pricing, the roadmap, and hiring.' },
        { speaker: 'Host', text: 'Here is the plan:' },
        { speaker: 'Guest', text: 'Ship the Reader first.' },
        { speaker: 'Host', text: '   ' },
        { speaker: 'Guest', text: '' },
        { speaker: 'Host', text: ' \n\t ' },
        { speaker: 'Guest' },
      ],
    };
    assert.equal(scriptText(conv), [
      'Host: Welcome back. Today we cover three things:',
      'Guest: Pricing, the roadmap, and hiring.',
      'Host: Here is the plan:',
      'Guest: Ship the Reader first.',
    ].join('\n'), 'four turns have words, in their order');
    assert.equal(scriptText({ turns: [{ speaker: 'Narrator', text: 'Listen:' }, { speaker: 'Narrator', text: ':' }] }), 'Narrator: Listen:\nNarrator: :');
  });

  it('a script whose every turn ends with a colon is offered, and is sent whole', async () => {
    const asks = [];
    const turns = [
      { id: 't-a', speaker: 'Host', text: 'Today we cover three things:', profile_id: 'p1', duration_ms: 1200 },
      { id: 't-b', speaker: 'Guest', text: 'Here is the plan:', profile_id: 'p2', duration_ms: 1200 },
    ];
    t = await studioEnv({ path: 'studio/conversations/conv-1', server: studioServer({ conversations: [teaParty({ turns })] }), routes: assistant({ asks }) });
    const more = t.$('st-conv-more');
    assert.deepEqual(t.menuItems(more).map((item) => item.textContent.trim()), ['Ask AI about the script…', 'Delete conversation…']);
    await t.menu(more, 'Ask AI about the script…');
    t.env.change(document.getElementById('ask-out-speak'), false);
    await t.click(document.getElementById('ask-send'));
    assert.deepEqual(asks[0], { input: { kind: 'text', text: 'Host: Today we cover three things:\nGuest: Here is the plan:' }, preset: 'voice' });
  });
});
