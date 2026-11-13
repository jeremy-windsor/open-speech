import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';

let env;
afterEach(() => env?.restore());

async function setup() {
  env = installBrowserEnv();
  return importFresh('core/store.js');
}

describe('core/store', () => {
  it('get() returns a frozen shallow copy and set() merges top-level keys', async () => {
    const { createStore } = await setup();
    const s = createStore({ a: 1, b: { x: 1 } });
    const first = s.get();
    assert.ok(Object.isFrozen(first));
    s.set({ a: 2 });
    assert.equal(s.get().a, 2);
    assert.equal(s.get().b, first.b, 'untouched keys keep their identity');
    assert.equal(first.a, 1, 'an old snapshot does not change');
    s.set((state) => ({ a: state.a + 1 }));
    assert.equal(s.get().a, 3);
  });

  it('select() notifies only when the selected value changes', async () => {
    const { createStore } = await setup();
    const s = createStore({ a: 1, b: 1 });
    const seen = [];
    const off = s.select((st) => st.a, (value, prev) => seen.push([value, prev]));
    s.set({ b: 2 });
    s.set({ a: 1 });
    s.set({ a: 5 });
    assert.deepEqual(seen, [[5, 1]]);
    off();
    s.set({ a: 6 });
    assert.equal(seen.length, 1);
  });

  it('select() supports a custom equals and an immediate first call', async () => {
    const { createStore } = await setup();
    const s = createStore({ list: [1] });
    const seen = [];
    s.select((st) => st.list, (v) => seen.push(v.length), { immediate: true, equals: (x, y) => x.length === y.length });
    s.set({ list: [2] });
    s.set({ list: [2, 3] });
    assert.deepEqual(seen, [1, 2]);
  });

  it('a throwing listener does not stop the others', async () => {
    const { createStore } = await setup();
    const s = createStore({ a: 0 });
    let ran = false;
    const original = console.error;
    console.error = () => {};
    try {
      s.select((st) => st.a, () => { throw new Error('boom'); });
      s.select((st) => st.a, () => { ran = true; });
      s.set({ a: 1 });
    } finally {
      console.error = original;
    }
    assert.ok(ran);
  });

  it('the app store starts with its documented shape', async () => {
    const { app } = await setup();
    const st = app.get();
    assert.deepEqual(Object.keys(st).sort(), ['auth', 'connection', 'jobs', 'models', 'ops', 'player', 'prefs', 'reading', 'recording', 'resumable', 'route', 'speakVoice'].sort());
    assert.equal(st.connection.state, 'connecting');
    assert.deepEqual(st.route, { view: '', sub: '', id: '', path: '', shot: '' });
    assert.deepEqual(st.speakVoice, { model: '', voice: '', label: '' });
  });

  it('emit/on deliver details and refuse unknown event types', async () => {
    const { emit, on, EVENTS } = await setup();
    assert.deepEqual([...EVENTS].sort(), ['auth:changed', 'history:changed', 'library:changed', 'route:changed', 'voices:changed']);
    const got = [];
    const off = on('voices:changed', (detail) => got.push(detail));
    emit('voices:changed', { what: 'named' });
    off();
    emit('voices:changed', { what: 'presets' });
    assert.deepEqual(got, [{ what: 'named' }]);
    assert.throws(() => emit('nope', {}), /Unknown event/);
    assert.throws(() => on('nope', () => {}), /Unknown event/);
  });

  it('intents: one pending per view, taken once, and only the frozen list', async () => {
    const { setIntent, takeIntent } = await setup();
    setIntent('speak', { type: 'apply-preset', id: 'p1' });
    setIntent('speak', { type: 'apply-history', entry: { id: 'h1' } });
    assert.deepEqual(takeIntent('speak'), { type: 'apply-history', entry: { id: 'h1' } });
    assert.equal(takeIntent('speak'), null);
    setIntent('reader', { type: 'open-appearance' });
    assert.equal(takeIntent('reader').type, 'open-appearance');
    assert.throws(() => setIntent('speak', { type: 'open-appearance' }), /Unknown intent/);
    assert.throws(() => setIntent('models', { type: 'apply-preset' }), /Unknown intent/);
  });

  it('__resetForTests restores the initial state and drops listeners', async () => {
    const store = await setup();
    store.app.set({ jobs: [{ key: 'x' }] });
    let calls = 0;
    store.on('history:changed', () => { calls += 1; });
    store.__resetForTests();
    assert.deepEqual(store.app.get().jobs, []);
    store.emit('history:changed');
    assert.equal(calls, 0);
  });
});
