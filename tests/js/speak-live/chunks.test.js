// Speak: Live text flow control (inventory LR-09): bounded code-point chunks, one in flight,
// FIFO acknowledgements, commits after the last chunk, buffer_overflow requeues. The pure createTextFlow
// first, then the same rules through the engine against a scripted server.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { liveEnv, load, settle, voiceConfig, server, sent } from './helpers.js';

let env;
afterEach(() => env?.restore());

async function flowModule() {
  env = await liveEnv();
  return load('views/speak/live-session.js');
}

const codePoints = (s) => [...s].length;

describe('Live: createTextFlow (pure)', () => {
  it('[LR-09] sends chunks of at most min(2048, buffer room) code points, one in flight, FIFO on input_text.accepted', async () => {
    const live = await flowModule();
    const flow = live.createTextFlow({ maxBufferChars: 8192 });
    flow.queue('a'.repeat(5000), true);
    let out = flow.drain();
    assert.deepEqual(out.map((e) => [e.type, e.text.length]), [['input_text.append', 2048]]);
    assert.deepEqual(flow.drain(), [], 'nothing more while a chunk is in flight');
    flow.accepted({ buffered_chars: 2048 });
    out = flow.drain();
    assert.equal(out[0].text.length, 2048);
    flow.accepted({ buffered_chars: 4096 });
    out = flow.drain();
    assert.equal(out[0].text.length, 904, 'the rest');
    assert.deepEqual(flow.drain(), [], 'the commit waits for the last chunk');
    flow.accepted({ buffered_chars: 5000 });
    assert.deepEqual(flow.drain(), [{ type: 'input_text.commit' }]);
    assert.equal(flow.busy, false);
  });

  it('[LR-09] the buffer room caps a chunk, a full buffer waits for session.status, and generation follows the server', async () => {
    const live = await flowModule();
    const flow = live.createTextFlow({ maxBufferChars: 3000 });
    flow.queue('b'.repeat(2500));
    flow.status({ buffered_chars: 2000, generation: 2 });
    assert.equal(flow.generation, 2);
    assert.equal(flow.drain()[0].text.length, 1000, 'room = 3000 - 2000');
    flow.accepted({ buffered_chars: 3000, generation: 2 });
    assert.deepEqual(flow.drain(), [], 'no room');
    flow.status({ buffered_chars: 100 });
    assert.equal(flow.drain()[0].text.length, 1500);
  });

  it('[LR-09] chunks count code points like the server and never split a surrogate pair', async () => {
    const live = await flowModule();
    const flow = live.createTextFlow({ maxBufferChars: 8192 });
    const text = '😀'.repeat(2050);   // 4100 UTF-16 units, 2050 code points
    flow.queue(text);
    const first = flow.drain()[0].text;
    assert.equal(codePoints(first), 2048);
    assert.equal(first.length, 4096, 'whole emoji only');
    flow.accepted({ buffered_chars: 2048 });
    assert.equal(flow.drain()[0].text, '😀😀');
  });

  it('[LR-09] buffer_overflow puts the rejected chunk back in front and waits for a status; a cancel drops everything', async () => {
    const live = await flowModule();
    const flow = live.createTextFlow({ maxBufferChars: 8192 });
    flow.queue('first ');
    flow.queue('second', true);
    assert.equal(flow.drain()[0].text, 'first ');
    assert.equal(flow.overflow(), true);
    assert.deepEqual(flow.drain(), [], 'the buffer counts as full until session.status');
    flow.status({ buffered_chars: 0 });
    assert.equal(flow.drain()[0].text, 'first ', 'the same chunk again, in order');
    flow.accepted({ buffered_chars: 6 });
    assert.equal(flow.drain()[0].text, 'second');
    assert.equal(flow.overflow(), true);
    flow.cancelled({ generation: 0 });
    assert.equal(flow.generation, 1);
    assert.equal(flow.busy, false, 'queue and in-flight list emptied');
    assert.equal(flow.overflow(), false, 'nothing in flight to requeue');
  });
});

describe('Live: flow control through the engine', () => {
  it('[LR-09] long text goes out in bounded chunks, each after the previous was accepted, then one commit', async () => {
    env = await liveEnv();
    const live = await load('views/speak/live-session.js');
    env.editor.value = 'c'.repeat(4500);
    env.editor.selectionStart = 0;
    const engine = live.createLiveEngine({ editor: env.editor, getVoiceConfig: () => voiceConfig() });
    engine.start();
    await settle(env);
    const srv = server(env.sockets[0]);
    srv.open();
    srv.created({ max_buffer_chars: 8192 });
    srv.updated();
    await settle(env);
    const appends = () => sent(srv.ws, 'input_text.append').map((e) => e.text.length);
    assert.deepEqual(appends(), [2048]);
    srv.accepted(2048);
    await settle(env);
    assert.deepEqual(appends(), [2048, 2048]);
    srv.accepted(4096);
    await settle(env);
    assert.deepEqual(appends(), [2048, 2048, 404]);
    assert.equal(sent(srv.ws, 'input_text.commit').length, 0);
    srv.accepted(4500);
    await settle(env);
    assert.equal(sent(srv.ws, 'input_text.commit').length, 1);

    // buffer_overflow: the rejected chunk is sent again after session.status, before newer text.
    env.input(env.editor, `${'c'.repeat(4500)} later`);
    await settle(env, 40);
    assert.equal(sent(srv.ws, 'input_text.append').at(-1).text, ' later');
    srv.error('buffer_overflow', 'Live TTS input buffer is full; wait for session.status before retrying');
    await settle(env);
    assert.equal(engine.isActive(), true, 'not an error for the listener');
    assert.equal(sent(srv.ws, 'input_text.append').length, 4, 'waits');
    srv.status(0);
    await settle(env);
    assert.equal(sent(srv.ws, 'input_text.append').length, 5);
    assert.equal(sent(srv.ws, 'input_text.append').at(-1).text, ' later');
    engine.stop();
  });
});

describe('Live: helpers', () => {
  it('[LR-09] PCM16 decoding, the median and the stop status are exact', async () => {
    const live = await flowModule();
    const buf = Buffer.alloc(6);
    buf.writeInt16LE(-32768, 0);
    buf.writeInt16LE(0, 2);
    buf.writeInt16LE(16384, 4);
    assert.deepEqual(Array.from(live.decodePcm16(buf.toString('base64'))), [-1, 0, 0.5]);
    assert.equal(live.median([5, 1, 3]), 3);
    assert.equal(live.median([4, 1, 3, 2]), 2.5);
    assert.equal(live.median([]), null);
    assert.deepEqual(live.stoppedStatus('Another player started'), { status: 'Stopped: another player started', reason: 'Another player started' });
    assert.deepEqual(live.stoppedStatus('Stopped'), { status: 'Stopped', reason: '' });
    assert.equal(live.stoppedStatus('API key changed').status, 'Stopped: API key changed');
    assert.equal(live.modeFor('nope').value, 'natural');
  });
});
