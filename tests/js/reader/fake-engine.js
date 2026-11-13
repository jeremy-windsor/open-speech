// A stand-in for the Reader engine (views/reader/engine.js), so the library and reading surface can be tested without
// audio. It records every call, keeps the engine's state shape and emits its events; tests drive playback by
// calling emit() or the helpers below. Where the Reader view depends on it, it behaves like views/reader/engine.js:
// - unload() stops and keeps the document loaded (Close player: Play resumes it); unload({forget: true}) also
//   drops it (a deleted or reset document). The call is recorded as ['unload', docId, {forget}].
// - docId is '' when no document is loaded; 'state' passes the snapshot; 'progress' {docId, progress} follows
//   a saved position once the server answered (tests send it with saved(progress)).
// - `stateReads` counts reads of `state` (the real getter builds a fresh snapshot each time).

const STATE = () => ({
  docId: '',
  cursor: 0,
  sentence: -1,
  chapter: 0,
  playing: false,
  waiting: false,
  speed: 1,
  voice: null,
  status: { text: '', kind: '' },
  finished: false,
  timeline: { elapsedMs: 0, durationMs: 0, bufferedMs: 0, estimated: true },
  sleep: { mode: 'off', remainingMs: 0, chapters: 0 },
});

export function createFakeEngine() {
  const listeners = new Map();
  const calls = [];
  let state = STATE();
  let doc = null;

  function emit(type, detail) {
    for (const cb of [...(listeners.get(type) || [])]) cb(detail);
  }
  function setState(patch) {
    state = { ...state, ...patch };
    emit('state', engine.state);
  }
  function chapterOf(block) {
    const chapters = (doc && doc.chapters) || [];
    let position = 0;
    chapters.forEach((c, i) => { if (block >= c.first_block) position = i; });
    return position;
  }
  function moveTo(block, reason) {
    const count = doc ? doc.blocks.length : 0;
    const index = Math.max(0, Math.min(Math.max(0, count - 1), block));
    state = { ...state, cursor: index, sentence: -1, chapter: chapterOf(index) };
    emit('cursor', { block: index, sentence: -1, chapter: state.chapter, reason });
  }

  let reads = 0;
  const engine = {
    calls,
    get doc() { return doc; },
    get state() {
      reads += 1;
      return { ...state };
    },
    /** How many times `state` was read (each read is a snapshot rebuild in the real engine). */
    get stateReads() { return reads; },
    resetStateReads() { reads = 0; },
    on(type, cb) {
      const list = listeners.get(type) || [];
      list.push(cb);
      listeners.set(type, list);
      return () => listeners.set(type, (listeners.get(type) || []).filter((l) => l !== cb));
    },
    listenerCount(type) { return (listeners.get(type) || []).length; },
    emit,
    load(next) {
      calls.push(['load', next && next.id]);
      if (doc && next && doc.id === next.id) return;
      doc = next;
      const count = next.blocks.length;
      const saved = next.progress ? Number(next.progress.block_index) || 0 : 0;
      const finished = count > 0 && saved >= count;
      state = { ...STATE(), docId: next.id, speed: (next.progress && next.progress.speed) || 1, finished };
      moveTo(finished ? count - 1 : saved, 'open');
    },
    unload({ forget = false } = {}) {
      calls.push(['unload', doc ? doc.id : '', { forget: !!forget }]);
      if (!doc) return Promise.resolve();
      if (forget) {
        doc = null;
        setState({ ...STATE() });
      } else {
        setState({ playing: false, waiting: false });   // stopped, still loaded: Play resumes it
      }
      return Promise.resolve();
    },
    play() { calls.push(['play']); setState({ playing: true, finished: false }); },
    pause() { calls.push(['pause']); setState({ playing: false }); },
    toggle() { calls.push(['toggle']); setState({ playing: !state.playing }); },
    playFrom(block) {
      calls.push(['playFrom', block]);
      moveTo(block, 'jump');
      setState({ playing: true, finished: false });
    },
    jumpTo(block) { calls.push(['jumpTo', block]); moveTo(block, 'jump'); },
    skipBlock(delta) { calls.push(['skipBlock', delta]); moveTo(state.cursor + delta, 'jump'); },
    skipSentence(delta) { calls.push(['skipSentence', delta]); },
    seekBy(seconds) { calls.push(['seekBy', seconds]); },
    seekChapterMs(ms) { calls.push(['seekChapterMs', ms]); },
    setSpeed(rate) { calls.push(['setSpeed', rate]); setState({ speed: rate }); },
    setVoice(voice) { calls.push(['setVoice', voice]); setState({ voice }); },
    setSleep(spec) { calls.push(['setSleep', spec]); },
    unlock() { calls.push(['unlock']); },
    saveProgress(options) { calls.push(['saveProgress', options]); return Promise.resolve(); },
    /** Test helper: the engine saved a position and the server answered (the real engine emits 'progress' then). */
    saved(progress) {
      if (doc) emit('progress', { docId: doc.id, progress });
    },
    /** Test helper: playback reached `block` (sentence `sentence`), as the tracking loop reports it. */
    advance(block, sentence = -1) {
      state = { ...state, cursor: block, sentence, chapter: chapterOf(block) };
      emit('cursor', { block, sentence, chapter: state.chapter, reason: 'playback' });
    },
    /** Test helper: the last paragraph ended. */
    finish() {
      setState({ playing: false, finished: true });
      emit('finished');
    },
  };
  return engine;
}

/** The deps the Reader controller takes in tests: this engine, and a player mount that records its call. */
export function fakeDeps(engine = createFakeEngine()) {
  const mounted = [];
  return {
    engine,
    mounted,
    deps: {
      loadEngine: async () => engine,
      mountPlayer: async (slot, eng, options) => { mounted.push({ slot, engine: eng, options }); },
    },
  };
}
