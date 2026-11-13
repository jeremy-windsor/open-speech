// Transcribe: the view as a whole: its states, accessible names in every state, the latency rule (no
// request before the socket or upload when the model is known to be loaded), view-only timers, the screenshot hook and the pure copy helpers in ui.js.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import {
  mountView, settle, startRecording, begin, said, chooseFile, audioFile, verbose, json, text, load, ROUTE, lastSocket, countWrites, sameNode, noNode,
} from './helpers.js';
import { delay } from '../harness/fakes.js';
import { installBrowserEnv, importFresh } from '../harness/env.js';
import { assertAccessibleNames } from '../harness/a11y.js';

let t;
afterEach(() => t?.env?.restore());

describe('transcribe/ui: copy helpers', () => {
  it('captureView: the capture phases (Starting…, Recording, Finishing…) and why Record is off', async () => {
    const env = installBrowserEnv();
    t = { env };
    const ui = await importFresh('views/transcribe/ui.js');
    assert.deepEqual(
      ['idle', 'requesting', 'starting', 'recording', 'stopping', 'done', 'error'].map((phase) => {
        const v = ui.captureView({ phase });
        return [phase, v.phase, v.buttonLabel, v.busy, v.disabled];
      }),
      [
        ['idle', 'Record', 'Record', false, false],
        ['requesting', 'Starting…', 'Record', true, true],
        ['starting', 'Starting…', 'Record', true, true],
        ['recording', 'Recording', 'Stop recording', false, false],
        ['stopping', 'Finishing…', 'Stop recording', true, true],
        ['done', 'Record', 'Record', false, false],
        ['error', 'Record', 'Record', false, false],
      ],
    );
    assert.equal(ui.captureView({ phase: 'starting' }, { jobLabel: 'Loading whisper base…' }).phase, 'Loading whisper base…');
    assert.equal(ui.captureView({ phase: 'idle' }, { blockedBy: 'upload' }).disabled, true);
    assert.match(ui.captureView({ phase: 'idle' }, { blockedBy: 'upload' }).hint, /Finish or cancel the file first/);
    assert.equal(ui.captureView({ phase: 'recording' }, { blockedBy: 'upload' }).disabled, false, 'Stop is never blocked');
  });

  it('micHelp: insecure and unsupported block Record; a failure, a denial and no device explain themselves', async () => {
    const env = installBrowserEnv();
    t = { env };
    const { micHelp, MIC_HELP } = await importFresh('views/transcribe/ui.js');
    assert.deepEqual(micHelp({ support: { ok: false, reason: 'insecure' } }), { reason: 'insecure', message: MIC_HELP.insecure, blocking: true });
    assert.equal(micHelp({ support: { ok: false, reason: 'unsupported' } }).blocking, true);
    assert.equal(micHelp({ error: { kind: 'mic', reason: 'no-device' } }).message, 'No microphone found.');
    assert.equal(micHelp({ permission: 'denied' }).reason, 'denied');
    assert.equal(micHelp({ devices: [] }).reason, 'no-device');
    assert.equal(micHelp({ devices: [{ id: '', label: 'Default microphone' }], permission: 'granted' }), null);
    assert.equal(micHelp({ error: { kind: 'mic', reason: 'ended' } }), null, 'no help copy: it goes to the error banner');
  });

  it('nothingHeard: present tense while the recording runs, past once it is done, nothing otherwise', async () => {
    const env = installBrowserEnv();
    t = { env };
    const { nothingHeard } = await importFresh('views/transcribe/ui.js');
    assert.deepEqual(nothingHeard({ phase: 'recording', nothingHeard: 'no-speech' }),
      { reason: 'no-speech', title: 'No speech heard yet', message: 'Check the microphone and its level.' });
    assert.equal(nothingHeard({ phase: 'stopping', nothingHeard: 'silence' }).title, 'The microphone sends silence');
    assert.equal(nothingHeard({ phase: 'done', nothingHeard: 'no-speech' }).title, 'No speech was heard.');
    assert.equal(nothingHeard({ phase: 'done', nothingHeard: 'silence' }).title, 'The microphone sent only silence.');
    for (const rec of [null, { phase: 'recording', nothingHeard: '' }, { phase: 'idle', nothingHeard: 'silence' },
      { phase: 'error', nothingHeard: 'no-speech' }, { phase: 'recording', nothingHeard: 'bogus' }]) {
      assert.equal(nothingHeard(rec), null, JSON.stringify(rec));
    }
  });

  it('metaLine, errorActions, toastLine and bannerKey', async () => {
    const env = installBrowserEnv();
    t = { env };
    const ui = await importFresh('views/transcribe/ui.js');
    assert.equal(ui.metaLine({ source: 'mic', duration: 42.7, model: 'Systran/faster-whisper-small' }), 'Recorded 0:42 · whisper small');
    assert.equal(ui.metaLine({ source: 'file', duration: 0, processingMs: null, model: '' }), '');
    assert.deepEqual(ui.errorActions({ code: 4001 }, 'x'), [{ action: 'open-settings', label: 'Open Settings' }]);
    assert.deepEqual(ui.errorActions({ action: 'open-settings' }, 'x'), [{ action: 'open-settings', label: 'Open Settings' }]);
    // "Retry", the word every view uses, never "Try again".
    assert.deepEqual(ui.errorActions({ retryable: true }, 'upload-retry'), [{ action: 'upload-retry', label: 'Retry' }]);
    assert.deepEqual(ui.errorActions({ kind: 'busy' }, 'x'), []);
    assert.deepEqual(ui.errorActions({ kind: 'stream' }, 'record-retry'), [{ action: 'record-retry', label: 'Retry' }]);
    assert.deepEqual(ui.errorActions({ kind: 'close', code: 1013 }, 'record-retry'), [{ action: 'record-retry', label: 'Retry' }]);
    assert.deepEqual(ui.errorActions({ kind: 'notice', retryable: true }, 'record-retry'), [], 'a notice beside a running recording: Dismiss only');
    assert.equal(ui.toastLine({ title: "Part of the recording wasn't transcribed", message: 'Transcription failed: x' }), "Part of the recording wasn't transcribed. Transcription failed: x");
    assert.equal(ui.toastLine({ title: 'Recording stopped', message: 'Recording stopped when the app went to the background.' }), 'Recording stopped when the app went to the background.');
    assert.equal(ui.bannerKey(null), '');
    assert.equal(ui.bannerKey({ title: 'A', message: 'B' }, { actions: [] }), ui.bannerKey({ title: 'A', message: 'B', kind: 'other' }, { actions: [] }), 'what it says, not the object');
    assert.notEqual(ui.bannerKey({ title: 'A', message: 'B' }), ui.bannerKey({ title: 'A', message: 'B', detail: 'more' }), 'new Details are a new banner');
  });

  it('[GS-03] a banner\'s Details are text, never markup (no JS-built on* attribute), and its buttons name their command in data-action (dom.actions)', async () => {
    const env = installBrowserEnv();
    t = { env };
    const ui = await importFresh('views/transcribe/ui.js');
    const { render } = await importFresh('core/dom.js');
    const node = env.document.createElement('div');
    render(node, ui.bannerHtml(
      { title: "Couldn't transcribe", message: 'The server returned an unexpected response (HTTP 502).', detail: '<b onclick="x()">Bad gateway</b>' },
      { actions: [{ action: 'upload-retry', label: 'Retry' }], dismissAction: 'dismiss-upload-error' },
    ));
    assert.deepEqual(node.querySelectorAll('button').map((b) => b.getAttribute('data-action')), ['upload-retry', 'dismiss-upload-error']);
    noNode(node.querySelector('[data-cmd]'), 'no second attribute to dispatch on');
    assert.equal(node.querySelector('details pre').textContent, '<b onclick="x()">Bad gateway</b>');
    noNode(node.querySelector('b'), 'escaped by html``');
    render(node, ui.bannerHtml({ title: 'Microphone problem', message: 'No microphone found.' }, { tone: 'warning', dismissAction: '' }));
    noNode(node.querySelector('details'), 'no Details without a detail');
    noNode(node.querySelector('button'), 'dismissAction "" leaves Dismiss out');
  });
});

describe('transcribe tests: node checks', () => {
  // node:assert inspects a failing assertion's actual and expected (depth 1000, getters on), and a harness node
  // reaches the whole document, window and env: one failing assert.equal(node, …) went past the 2 GB cap. The
  // tests compare nodes with sameNode()/noNode(), whose failures carry a short description and no node.
  it('sameNode() and noNode() pass on the right node and fail with a short description, never holding a node', async () => {
    t = await mountView();
    const rec = t.$('tr-record');
    sameNode(rec, t.$('tr-record'));
    noNode(t.$('tr-no-such-id'));
    noNode(undefined);
    const caught = (fn) => {
      try {
        fn();
      } catch (error) {
        return error;
      }
      return null;
    };
    const wrong = caught(() => sameNode(t.$('tr-copy'), rec, 'focus'));
    assert.ok(wrong instanceof assert.AssertionError, 'a failure is an AssertionError');
    assert.equal(wrong.message, 'focus: got <button#tr-copy data-action="copy"> "Copy", expected <button#tr-record data-action="record">');
    assert.ok(wrong.actual === undefined && wrong.expected === undefined, 'the error holds no node for a reporter to walk');
    const present = caught(() => noNode(rec, 'none'));
    assert.equal(present.message, 'none: expected no node, got <button#tr-record data-action="record">');
    assert.ok(present.actual === undefined && present.expected === undefined);
    assert.equal(caught(() => sameNode(null, rec)).message, 'got null, expected <button#tr-record data-action="record">');
  });
});

describe('transcribe view: states, names and timers', () => {
  it('idle: Record, the microphone select, the idle meter, the dropzone and the empty transcript', async () => {
    t = await mountView();
    assert.equal(t.$('tr-phase').textContent, 'Record');
    assert.equal(t.$('tr-mic-field').hidden, false);
    assert.equal(t.$('tr-meter').getAttribute('data-zone'), 'idle');
    assert.equal(t.$('tr-meter').getAttribute('role'), 'meter');
    assert.equal(t.$('tr-drop-title').textContent, 'Upload audio');
    assert.equal(t.$('tr-empty').hidden, false);
    assert.equal(t.$('tr-foot').hidden, true);
    assertAccessibleNames(t.root);
  });

  it('every control has a name in the recording, error and file states too', async () => {
    t = await mountView({ routes: { 'POST /v1/audio/transcriptions': () => json(200, verbose()) } });
    const ws = await startRecording(t);
    said(ws, 'Hello.', { start: 0, end: 1 });
    ws.serverSend({ type: 'error', message: 'Transcription failed: x' });
    await settle(t.env);
    assertAccessibleNames(t.root);
    ws.serverClose(1013, 'Too many concurrent streams');
    await settle(t.env);
    assertAccessibleNames(t.root);
    chooseFile(t, audioFile());
    await settle(t.env);
    await settle(t.env);
    assertAccessibleNames(t.root);
    for (const button of t.root.querySelectorAll('button')) assert.equal(button.getAttribute('type'), 'button');
  });

  it('latency: with the model loaded, Record opens the socket with no request before it, and an upload posts at once', async () => {
    t = await mountView({ routes: { 'POST /v1/audio/transcriptions': () => json(200, verbose()) } });
    t.env.fetch.reset();
    t.env.click(t.$('tr-record'));
    await settle(t.env);
    assert.ok(lastSocket(t.env));
    assert.deepEqual(t.env.fetch.calls.map((c) => c.path), [], 'the fast path: no status check');
    lastSocket(t.env).open();
    begin(lastSocket(t.env));
    await settle(t.env);
    t.env.click(t.$('tr-record'));
    lastSocket(t.env).serverSend({ type: 'session.end' });
    await settle(t.env);
    t.env.fetch.reset();
    chooseFile(t, audioFile());
    await settle(t.env);
    await settle(t.env);
    assert.deepEqual(t.env.fetch.calls.map((c) => `${c.method} ${c.path}`), ['POST /v1/audio/transcriptions']);
  });

  it('hide() stops the view-only timers; the recording goes on and show() brings the counter back', async () => {
    t = await mountView();
    await startRecording(t);
    const timers = () => t.env.clock._timers.size;
    const withView = timers();
    t.ctl.hide();
    assert.ok(timers() < withView, 'the 1 s counter stopped');
    await settle(t.env, 5000);
    const { app } = await load('core/store.js');
    assert.equal(app.get().recording.owner, 'transcribe', 'still recording while another view shows');
    t.ctl.show(ROUTE, { focus: 'none' });
    await settle(t.env);
    assert.equal(t.$('tr-timer').textContent, '0:05');
  });

  it('stopping shows "Finishing…" for up to 4 s, then done: the transcript, the meta line, Copy and Export', async () => {
    t = await mountView();
    const ws = await startRecording(t);
    said(ws, 'All of it.', { start: 0, end: 2 });
    await settle(t.env, 2000);
    t.env.click(t.$('tr-record'));
    await settle(t.env);
    assert.equal(t.$('tr-phase').textContent, 'Finishing…');
    assert.equal(t.$('tr-meta').textContent, '', 'no meta line until it is done');
    await settle(t.env, 4000);
    assert.equal(t.$('tr-phase').textContent, 'Record');
    assert.equal(t.$('tr-meta').textContent, 'Recorded 0:02 · whisper base');
    assert.equal(t.$('tr-copy').disabled, false);
    assert.equal(t.$('tr-export').disabled, false);
  });

  it('the starting phase names the model load; Cancel during it closes the microphone', async () => {
    t = await mountView({ routes: { 'GET /api/models/:id/status': () => delay(30000, json(200, { state: 'loaded' })) } });
    t.app.set({ models: { ...t.app.get().models, fetchedAt: 0 } });
    t.env.click(t.$('tr-record'));
    await settle(t.env);
    assert.equal(t.$('tr-rec-actions').hidden, false, 'Cancel shows while starting');
    t.env.click(t.$('tr-cancel'));
    await settle(t.env);
    const mic = await load('core/mic.js');
    assert.equal(mic.owner(), null);
    assert.equal(t.$('tr-phase').textContent, 'Record');
    await settle(t.env, 30000);
    assert.equal(t.env.sockets.length, 0);
  });
});

describe('transcribe view: keyboard focus never falls to <body>', () => {
  it('[BUG-09] a file chosen from the keyboard: Cancel holds focus while it is busy, the input gets it back after', async () => {
    t = await mountView({ routes: { 'POST /v1/audio/transcriptions': () => delay(3000, json(200, verbose())) } });
    const input = t.$('tr-file');
    const cancel = t.$('tr-upload-cancel');
    input.focus();
    chooseFile(t, audioFile('talk.mp3'));
    await settle(t.env);
    assert.equal(input.disabled, true);
    sameNode(t.doc.activeElement, cancel, 'the only control left while the file is busy');
    await settle(t.env, 3000);
    await settle(t.env);
    assert.equal(cancel.hidden, true);
    sameNode(t.doc.activeElement, input, 'back where the listener started');
    // Cancelled from the keyboard: the same way back.
    input.focus();
    chooseFile(t, audioFile('again.mp3'));
    await settle(t.env);
    sameNode(t.doc.activeElement, cancel);
    t.env.key('Enter', {}, cancel);
    await settle(t.env);
    assert.equal(t.$('tr-drop').hasAttribute('aria-busy'), false);
    sameNode(t.doc.activeElement, input);
  });

  it('[TR-12] Cancel from the keyboard hands focus to Record as it goes away with the session', async () => {
    t = await mountView();
    await startRecording(t);
    const cancel = t.$('tr-cancel');
    cancel.focus();
    t.env.key('Enter', {}, cancel);
    await settle(t.env);
    assert.equal(t.$('tr-rec-actions').hidden, true);
    sameNode(t.doc.activeElement, t.$('tr-record'));
  });

  it('[TR-11] Retry leaves with its banner: Record takes the focus and keeps it through the restart', async () => {
    t = await mountView();
    const ws = await startRecording(t);
    ws.error();
    await settle(t.env);
    const retry = t.$('tr-rec-error').querySelector('[data-action="record-retry"]');
    retry.focus();
    t.env.key('Enter', {}, retry);
    await settle(t.env);
    assert.equal(t.$('tr-rec-error').hidden, true);
    lastSocket(t.env).open();
    begin(lastSocket(t.env));
    await settle(t.env);
    assert.equal(t.$('tr-record').getAttribute('aria-label'), 'Stop recording');
    sameNode(t.doc.activeElement, t.$('tr-record'));
  });

  it('[TR-02] Retry on a failed file: the file input takes the focus, then Cancel while it is busy', async () => {
    let fail = true;
    t = await mountView({
      routes: { 'POST /v1/audio/transcriptions': () => (fail ? json(503, { error: { message: 'Busy' } }) : delay(2000, json(200, verbose()))) },
    });
    chooseFile(t, audioFile('talk.mp3'));
    await settle(t.env);
    await settle(t.env);
    fail = false;
    const retry = t.$('tr-upload-error').querySelector('[data-action="upload-retry"]');
    retry.focus();
    t.env.key('Enter', {}, retry);
    await settle(t.env);
    sameNode(t.doc.activeElement, t.$('tr-upload-cancel'));
    await settle(t.env, 2000);
    await settle(t.env);
    sameNode(t.doc.activeElement, t.$('tr-file'));
  });
});

describe('transcribe view: a banner is written only when what it says changes', () => {
  // The status poll refreshes the inventory every 60 s, and every 2 s while Models works on a model.
  async function poll(times = 3) {
    for (let i = 0; i < times; i += 1) {
      const m = t.app.get().models;
      t.app.set({ models: { ...m, list: m.list.slice(), fetchedAt: m.fetchedAt + 1 } });
      t.app.set({ ops: { 'Systran/faster-whisper-tiny': { kind: 'download', startedAt: Date.now(), progress: 0.1 } } });
      await settle(t.env, 2000);
    }
  }

  it('[TR-02] a models poll leaves an upload error alone: the same nodes, the keyboard focus and an open Details', async () => {
    t = await mountView({
      routes: { 'POST /v1/audio/transcriptions': () => text(502, '<html><body>Bad gateway</body></html>', { 'content-type': 'text/html' }) },
    });
    chooseFile(t, audioFile('talk.mp3'));
    await settle(t.env);
    await settle(t.env);
    const banner = t.$('tr-upload-error');
    const details = banner.querySelector('details');
    assert.ok(details, 'the proxy page is in Details');
    details.open = true;
    const retry = banner.querySelector('[data-action="upload-retry"]');
    retry.focus();
    const writes = countWrites(banner);
    await poll();
    assert.equal(writes.n, 0, 'never rewritten, so the role=alert is not read out again');
    sameNode(banner.querySelector('[data-action="upload-retry"]'), retry);
    sameNode(t.doc.activeElement, retry, 'focus stays on Retry');
    assert.equal(banner.querySelector('details').open, true, 'Details stays open');
  });

  it('[TR-13] a models poll leaves the microphone help alone: its Dismiss keeps the focus', async () => {
    t = await mountView({ env: { permissions: { microphone: 'denied' } } });
    await settle(t.env);
    const help = t.$('tr-mic-help');
    const dismiss = help.querySelector('[data-action="dismiss-help"]');
    assert.ok(dismiss);
    dismiss.focus();
    const writes = countWrites(help);
    await poll();
    assert.equal(writes.n, 0);
    sameNode(help.querySelector('[data-action="dismiss-help"]'), dismiss);
    sameNode(t.doc.activeElement, dismiss);
  });

  it('[BUG-02] a notice shown during a recording is not read out again at each utterance boundary, poll or phase change', async () => {
    t = await mountView();
    const ws = await startRecording(t);
    ws.serverSend({ type: 'error', message: 'Unknown language xx. The stream continues with auto-detect.' });
    await settle(t.env);
    const banner = t.$('tr-rec-error');
    const dismiss = banner.querySelector('[data-action="dismiss-rec-error"]');
    dismiss.focus();
    const writes = countWrites(banner);
    for (let i = 0; i < 3; i += 1) {
      ws.serverSend({ type: 'vad', state: 'speech_start' });
      await settle(t.env, 1500);
      said(ws, `Utterance ${i}.`, { start: i * 3, end: i * 3 + 1 });
      ws.serverSend({ type: 'vad', state: 'speech_end' });
      await settle(t.env, 1500);
    }
    await poll(2);
    assert.equal(writes.n, 0, 'six VAD changes, three finals and two polls rewrote nothing');
    sameNode(t.doc.activeElement, dismiss);
    // Stop: stopping, then done; the notice stays as it was.
    t.env.click(t.$('tr-record'));
    await settle(t.env);
    ws.serverSend({ type: 'session.end' });
    await settle(t.env);
    assert.equal(writes.n, 0);
    assert.equal(banner.hidden, false);
    assert.match(banner.textContent, /Unknown language xx/);
  });

  it('[TR-11] when the banner holding the focus says something new, the focus moves to the same command in it', async () => {
    t = await mountView();
    const ws = await startRecording(t);
    ws.serverSend({ type: 'error', message: 'Unknown language xx. The stream continues with auto-detect.' });
    await settle(t.env);
    const banner = t.$('tr-rec-error');
    banner.querySelector('[data-action="dismiss-rec-error"]').focus();
    // Then the stream drops: the banner now says that, and offers Retry.
    ws.serverClose(1011, 'Internal error');
    await settle(t.env);
    assert.equal(banner.querySelector('.banner__title').textContent, 'The microphone stream disconnected');
    assert.ok(banner.querySelector('[data-action="record-retry"]'));
    sameNode(t.doc.activeElement, banner.querySelector('[data-action="dismiss-rec-error"]'), 'the new Dismiss, never <body>');
  });
});

describe('transcribe view: away from the view', () => {
  const toastText = (region) => t.$(region).textContent.replace(/\s+/g, ' ').trim();

  it('[TR-02] a file that finishes while another view shows: a success toast with Show, instead of the summary', async () => {
    t = await mountView({ routes: { 'POST /v1/audio/transcriptions': () => delay(5000, json(200, verbose())) } });
    chooseFile(t, audioFile('talk.mp3'));
    await settle(t.env);
    t.ctl.hide();   // the listener went to the Reader
    await settle(t.env, 5000);
    await settle(t.env);
    assert.match(toastText('app-toasts-polite'), /Transcript ready: talk\.mp3/);
    await settle(t.env, 600);
    assert.equal(t.$('app-sr-status').textContent.includes('segments'), false, 'said once, by the toast');
    const show = [...t.$('app-toasts-polite').querySelectorAll('button')].find((b) => b.textContent.trim() === 'Show');
    t.env.click(show);
    await settle(t.env);
    assert.equal(t.env.location.hash, '#transcribe');
  });

  it('[TR-02] a file that fails while another view shows: a persistent error toast; while the view shows, only the banner', async () => {
    t = await mountView({ routes: { 'POST /v1/audio/transcriptions': () => delay(5000, json(500, { error: { message: 'Transcription failed: decoder' } })) } });
    chooseFile(t, audioFile('talk.mp3'));
    await settle(t.env);
    t.ctl.hide();
    await settle(t.env, 5000);
    await settle(t.env);
    assert.match(toastText('app-toasts-alert'), /Couldn't transcribe talk\.mp3/);
    t.ctl.show(ROUTE, { focus: 'none' });
    await settle(t.env);
    assert.match(t.$('tr-upload-error').textContent, /Transcription failed: decoder/);
    // The same failure with the view on screen: the banner, and no toast.
    t.env.restore();
    t = await mountView({ routes: { 'POST /v1/audio/transcriptions': () => json(500, { error: { message: 'Transcription failed: decoder' } }) } });
    chooseFile(t, audioFile('talk.mp3'));
    await settle(t.env);
    await settle(t.env);
    assert.equal(toastText('app-toasts-alert'), '');
    assert.equal(t.$('tr-upload-error').hidden, false);
  });

  it('[TR-11] a recording that drops while another view shows says so in a toast; the transcript so far is kept', async () => {
    t = await mountView();
    const ws = await startRecording(t);
    said(ws, 'Words before the drop.', { start: 0, end: 1 });
    await settle(t.env);
    t.ctl.hide();
    ws.serverClose(1013, 'Too many concurrent streams');
    await settle(t.env);
    assert.match(toastText('app-toasts-alert'), /The microphone stream disconnected/);
    t.ctl.show(ROUTE, { focus: 'none' });
    await settle(t.env);
    assert.match(t.$('tr-rec-error').textContent, /Too many live transcriptions at once/);
    assert.ok(t.$('tr-log').textContent.includes('Words before the drop.'));
  });

  it('[TR-10] a session the server ends with no failure while another view shows: an info toast "Recording stopped"', async () => {
    // A session.end nobody asked for waits for the close (a 1011 is a failure); with no
    // close it ends quietly once Stop's grace has passed.
    const { STOP_GRACE_MS } = await load('views/transcribe/recorder.js');
    t = await mountView();
    const ws = await startRecording(t);
    t.ctl.hide();
    ws.serverSend({ type: 'session.end', transcriptions: 0, errors: 0 });
    await settle(t.env, STOP_GRACE_MS);
    await settle(t.env);
    assert.match(toastText('app-toasts-polite'), /Recording stopped/);
    assert.equal(toastText('app-toasts-alert'), '');
  });

  it('[TR-11] a server failure while another view shows (session.end, then 1011): an alert toast "The microphone stream disconnected", not "Recording stopped"', async () => {
    t = await mountView();
    const ws = await startRecording(t);
    said(ws, 'Words before the failure.', { start: 0, end: 1 });
    await settle(t.env);
    t.ctl.hide();
    ws.serverSend({ type: 'session.end', reason: 'disconnect', transcriptions: 1, errors: 0 });
    ws.serverClose(1011, 'Streaming session failed');
    await settle(t.env);
    assert.match(toastText('app-toasts-alert'), /The microphone stream disconnected/);
    assert.doesNotMatch(toastText('app-toasts-polite'), /Recording stopped/);
    t.ctl.show(ROUTE, { focus: 'none' });
    await settle(t.env);
    assert.match(t.$('tr-rec-error').textContent, /The microphone stream disconnected/);
    assert.match(t.$('tr-rec-error').textContent, /Streaming session failed/);
    assert.ok(t.$('tr-log').textContent.includes('Words before the failure.'));
  });
});

describe('transcribe view: the screenshot hook', () => {
  it('?shot=recording shows a session without opening the microphone or a socket; leaving clears it', async () => {
    t = await mountView({ env: { mediaDevices: { devices: [] } } });   // as in headless Edge: no input at all
    t.ctl.show({ ...ROUTE, shot: 'recording' }, { focus: 'none' });
    await settle(t.env);
    assert.equal(t.$('tr-phase').textContent, 'Recording');
    assert.equal(t.$('tr-record').getAttribute('aria-label'), 'Stop recording');
    assert.equal(t.$('tr-vad-label').textContent, 'Speech');
    assert.equal(t.$('tr-log').querySelectorAll('.tr-seg').length, 2);
    assert.equal(t.$('tr-mic-help').hidden, true, 'no microphone help next to a running recording');
    assert.ok(t.$('tr-drop').classList.contains('is-disabled'), 'as in a real recording, no upload meanwhile');
    assert.equal(t.$('tr-file').disabled, true);
    assert.equal(t.env.mediaDevices.calls.length, 0);
    assert.equal(t.env.sockets.length, 0);
    const { app } = await load('core/store.js');
    assert.ok(app.get().recording, 'the chip shows Recording for the picture');
    t.ctl.hide();
    assert.equal(app.get().recording, null);
    t.ctl.show(ROUTE, { focus: 'none' });
    await settle(t.env);
    assert.equal(t.$('tr-phase').textContent, 'Record');
    assert.equal(t.$('tr-log').children.length, 0);
  });

  it('?shot=done, uploading, error and nomodel render their states; unknown states are ignored', async () => {
    t = await mountView();
    // The plain show() before them asks whether the AI assistant is on (for the Ask AI
    // button); the screenshot states themselves ask nothing.
    const before = t.env.fetch.calls.length;
    t.ctl.show({ ...ROUTE, shot: 'done' }, { focus: 'none' });
    await settle(t.env);
    assert.equal(t.$('tr-log').querySelectorAll('button.tr-ts').length, 8);
    assert.match(t.$('tr-meta').textContent, /^Duration 2:13 · Processed in 4\.1 s · whisper base$/);
    assert.equal(t.$('tr-copy').disabled, false);
    const demoUrl = t.ctl.state().fileUrl;
    t.ctl.show({ ...ROUTE, shot: 'uploading' }, { focus: 'none' });
    await settle(t.env);
    assert.equal(t.$('tr-drop').getAttribute('aria-busy'), 'true');
    assert.equal(t.$('tr-upload-elapsed').textContent, '0:12');
    assert.ok(t.env.revokedUrls.includes(demoUrl), 'the demo transcript and its silent file go with the state');
    assert.equal(t.$('tr-log').children.length, 0);
    t.ctl.show({ ...ROUTE, shot: 'error' }, { focus: 'none' });
    await settle(t.env);
    // The shot shows what the recorder really shows for a server error before session.begin.
    assert.equal(t.$('tr-rec-error').querySelector('.banner__title').textContent, "Couldn't transcribe");
    assert.equal(t.$('tr-rec-error').querySelector('[data-action="record-retry"]').textContent.trim(), 'Retry');
    assert.match(t.$('tr-upload-error').textContent, /unexpected response \(HTTP 502\)/);
    assert.equal(t.$('tr-upload-error').querySelector('[data-action="upload-retry"]').textContent.trim(), 'Retry');
    t.ctl.show({ ...ROUTE, shot: 'nomodel' }, { focus: 'none' });
    await settle(t.env);
    assert.equal(t.$('tr-nomodel').hidden, false);
    assert.equal(t.$('tr-record').disabled, true, 'as with no model: Record is off too');
    assert.equal(t.$('tr-rec-hint').textContent, 'Recording needs a transcription model.');
    assert.equal(t.$('tr-model-field').hidden, true);
    t.ctl.show({ ...ROUTE, shot: 'app-more' }, { focus: 'none' });
    await settle(t.env);
    assert.equal(t.$('tr-nomodel').hidden, true, 'a shell state ends the view state');
    assert.equal(t.$('tr-model-field').hidden, false);
    assert.equal(t.$('tr-record').disabled, false);
    assert.equal(t.env.fetch.calls.length, before, 'no request for any of it');
    assert.deepEqual(t.env.fetch.calls.map((c) => `${c.method} ${c.path}`), ['GET /api/assistant'], 'only the show before them asked');
  });
});
