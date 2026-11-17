// Transcribe: file transcription: the request, response parsing, the busy state with
// elapsed time and Cancel (BUG-09), errors in a banner, and the superseded format select and Recent list.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import {
  mountView, settle, startRecording, said, announced, chooseFile, audioFile, verbose, json, text, load, STT, MODELS, FakeEvent, answerConfirm, noNode,
} from './helpers.js';
import { delay } from '../harness/fakes.js';
import { installBrowserEnv, importFresh } from '../harness/env.js';

let t;
afterEach(() => t?.env?.restore());

const transcribeCalls = () => t.env.fetch.callsTo('POST /v1/audio/transcriptions');
const uploadError = () => (t.$('tr-upload-error').hidden ? '' : t.$('tr-upload-error').textContent.replace(/\s+/g, ' ').trim());

describe('transcribe/upload: parsing', () => {
  it('[TR-03] a JSON content type is parsed as JSON, anything else is the text; the body is read once', async () => {
    const env = installBrowserEnv();
    t = { env };
    const { parseTranscription } = await importFresh('views/transcribe/upload.js');
    const asJson = new Response(JSON.stringify({ text: 'hi', segments: [] }), { headers: { 'content-type': 'application/json; charset=utf-8' } });
    assert.deepEqual(await parseTranscription(asJson), { text: 'hi', segments: [] });
    assert.equal(asJson.bodyUsed, true);
    const asText = new Response('plain words', { headers: { 'content-type': 'text/plain' } });
    assert.deepEqual(await parseTranscription(asText), { text: 'plain words' });
    // What api.request already parsed passes through; a string is the text.
    assert.deepEqual(await parseTranscription({ text: 'x' }), { text: 'x' });
    assert.deepEqual(await parseTranscription('words'), { text: 'words' });
    assert.deepEqual(await parseTranscription(null), { text: '' });
  });

  it('[TR-03] verbose_json segments become {text, start, end}; a text-only answer is one segment', async () => {
    const env = installBrowserEnv();
    t = { env };
    const { toSegments } = await importFresh('views/transcribe/upload.js');
    assert.deepEqual(toSegments(verbose([[0, 1.5, 'One.'], [1.5, 3, '  ']])), [{ text: 'One.', start: 0, end: 1.5 }]);
    assert.deepEqual(toSegments({ text: ' All of it. ', duration: 4.2 }), [{ text: 'All of it.', start: 0, end: 4.2 }]);
    assert.deepEqual(toSegments({ text: '' }), []);
  });
});

describe('transcribe view: uploading a file', () => {
  it('[TR-02] choosing a file sends multipart {file, model, response_format=verbose_json} with X-History', async () => {
    t = await mountView({
      routes: {
        'GET /api/models/:id/status': () => json(200, { state: 'loaded' }),
        'POST /v1/audio/transcriptions': () => json(200, verbose()),
      },
    });
    t.env.change(t.$('tr-model'), 'Systran/faster-whisper-small');
    chooseFile(t, audioFile('meeting.m4a', 4096, 'audio/mp4'));
    await settle(t.env);
    await settle(t.env);
    const [call] = transcribeCalls();
    assert.ok(call.body instanceof FormData);
    assert.equal(call.body.get('file').name, 'meeting.m4a');
    assert.equal(call.body.get('file').size, 4096);
    assert.equal(call.body.get('model'), 'Systran/faster-whisper-small');
    assert.equal(call.body.get('response_format'), 'verbose_json');
    assert.equal(call.headers['x-history'], 'true');
    assert.equal(call.headers['content-type'], undefined, 'the browser sets the multipart boundary');
    assert.equal(t.$('tr-log').querySelectorAll('.tr-seg').length, 2);
    assert.equal(t.$('tr-file').value, '', 'the same file can be chosen again');
  });

  it('[TR-02] the model is made ready first: a model that is not downloaded is downloaded and loaded, then posted', async () => {
    let state = 'available';
    const models = MODELS.map((m) => (m.id === STT ? { ...m, state: 'available' } : m));
    t = await mountView({
      models,
      routes: {
        'GET /api/models/:id/status': () => json(200, { state }),
        'POST /api/models/:id/download': () => { state = 'downloaded'; return json(200, {}); },
        'POST /api/models/:id/load': () => { state = 'loaded'; return json(200, {}); },
        'GET /api/models': () => json(200, { models, default_stt_model: STT }),
        'POST /v1/audio/transcriptions': () => json(200, verbose()),
      },
    });
    chooseFile(t, audioFile());
    await settle(t.env);
    await settle(t.env);
    const order = t.env.fetch.calls.map((c) => `${c.method} ${c.path.split('/').pop()}`);
    const post = order.indexOf('POST transcriptions');
    assert.ok(post > order.indexOf('POST load') && order.indexOf('POST load') > order.indexOf('POST download'), order.join(', '));
  });

  it('[TR-02] a file dropped on the dropzone is transcribed; dragging over it shows the drop style', async () => {
    t = await mountView({ routes: { 'POST /v1/audio/transcriptions': () => json(200, verbose()) } });
    const drop = t.$('tr-drop');
    const over = new FakeEvent('dragover', { bubbles: true, cancelable: true });
    over.dataTransfer = { dropEffect: 'none', files: [] };
    drop.dispatchEvent(over);
    assert.equal(over.defaultPrevented, true, 'the page does not open the file itself');
    assert.equal(over.dataTransfer.dropEffect, 'copy');
    assert.ok(drop.classList.contains('is-dragover'));
    const dropped = new FakeEvent('drop', { bubbles: true, cancelable: true });
    dropped.dataTransfer = { files: [audioFile('dropped.wav', 1024, 'audio/wav')] };
    drop.dispatchEvent(dropped);
    await settle(t.env);
    await settle(t.env);
    assert.equal(drop.classList.contains('is-dragover'), false);
    assert.equal(transcribeCalls()[0].body.get('file').name, 'dropped.wav');
  });

  it('[TR-04] the result: segments, and a meta line with a formatted duration and the processing time', async () => {
    t = await mountView({ routes: { 'POST /v1/audio/transcriptions': () => delay(4100, json(200, verbose([[0, 60, 'First minute.'], [60, 133.4571875, 'The rest.']]))) } });
    chooseFile(t, audioFile());
    await settle(t.env);
    await settle(t.env, 4100);
    await settle(t.env);
    assert.equal(t.$('tr-meta').textContent, 'Duration 2:13 · Processed in 4.1 s · whisper base');
    assert.equal(t.$('tr-foot').hidden, false);
    assert.equal(t.$('tr-interim').textContent, '', 'nothing pending after a file');
    assert.deepEqual(t.$('tr-log').querySelectorAll('.tr-seg__text').map((e) => e.textContent), ['First minute.', 'The rest.']);
  });

  it('[TR-04] a file with no speech says so instead of an empty transcript (or a "—")', async () => {
    t = await mountView({ routes: { 'POST /v1/audio/transcriptions': () => json(200, { text: '', segments: [], duration: 8 }) } });
    chooseFile(t, audioFile('silence.wav'));
    await settle(t.env);
    await settle(t.env);
    assert.equal(t.$('tr-empty').hidden, false);
    assert.match(t.$('tr-empty').textContent, /No speech found\./);
    assert.match(t.$('tr-empty').textContent, /silence\.wav/);
    assert.equal(t.$('tr-copy').disabled, true);
    assert.equal(t.$('tr-log').textContent.includes('—'), false);
    await settle(t.env, 600);
    assert.equal(announced(t), 'No speech found in this file.');
  });

  it('[TR-04] a plain-text answer still becomes a transcript (one segment over the file)', async () => {
    t = await mountView({ routes: { 'POST /v1/audio/transcriptions': () => text(200, 'Just the words.') } });
    chooseFile(t, audioFile());
    await settle(t.env);
    await settle(t.env);
    assert.deepEqual(t.$('tr-log').querySelectorAll('.tr-seg__text').map((e) => e.textContent), ['Just the words.']);
  });

  it('[TR-05] History is the one record: X-History is sent, History is told, and no local Recent list is written', async () => {
    t = await mountView({ routes: { 'POST /v1/audio/transcriptions': () => json(200, verbose()) } });
    const store = await load('core/store.js');
    let changed = 0;
    store.on('history:changed', () => { changed += 1; });
    chooseFile(t, audioFile());
    await settle(t.env);
    await settle(t.env);
    assert.equal(transcribeCalls()[0].headers['x-history'], 'true');
    assert.equal(changed, 1);
    assert.equal(t.env.localStorage.__data['open-speech-stt-history'], undefined);
    noNode(t.root.querySelector('[id*="history"], [id*="recent"]'), 'no Recent list in the view');
  });

  it('[TR-18] verbose_json is always asked for: there is no format select, exports replace it', async () => {
    t = await mountView({ routes: { 'POST /v1/audio/transcriptions': () => json(200, verbose()) } });
    const selects = t.root.querySelectorAll('select').map((s) => s.id);
    assert.deepEqual(selects.sort(), ['tr-language', 'tr-mic', 'tr-model']);
    chooseFile(t, audioFile());
    await settle(t.env);
    await settle(t.env);
    assert.equal(transcribeCalls()[0].body.get('response_format'), 'verbose_json');
    assert.deepEqual(t.$('tr-export-menu').querySelectorAll('[data-format]').map((b) => b.getAttribute('data-format')), ['txt', 'srt', 'vtt', 'json']);
  });
});

describe('transcribe view: language, words to expect and translation', () => {
  it('the language and the words go with an upload and are remembered; Translate sends the file to /v1/audio/translations', async () => {
    t = await mountView({ routes: {
      'POST /v1/audio/transcriptions': () => json(200, verbose()),
      'POST /v1/audio/translations': () => json(200, verbose()),
    } });
    t.env.change(t.$('tr-language'), 'es');
    t.$('tr-prompt').value = ' Strata, Hermes ';   // typed, no change event yet: a file dropped now still sends them
    chooseFile(t, audioFile());
    await settle(t.env);
    await settle(t.env);
    const body = transcribeCalls()[0].body;
    assert.deepEqual([body.get('language'), body.get('prompt')], ['es', 'Strata, Hermes']);
    t.env.change(t.$('tr-translate'), true);
    chooseFile(t, audioFile('second.mp3'));
    await settle(t.env);
    await settle(t.env);
    const translated = t.env.fetch.calls.find((c) => c.method === 'POST' && c.path === '/v1/audio/translations');
    assert.ok(translated, 'translated');
    assert.equal(translated.body.get('language'), null, 'a translation takes no language');
    assert.equal(translated.body.get('prompt'), 'Strata, Hermes');
  });

  it('Translate to English is off for a model that cannot translate, and says why', async () => {
    const { canTranslate } = await load('views/transcribe/index.js');
    assert.deepEqual(['Systran/faster-whisper-large-v3', 'deepdml/faster-whisper-large-v3-turbo-ct2', 'nvidia/parakeet-tdt-0.6b-v2'].map(canTranslate), [true, false, false]);
  });
});

describe('transcribe view: busy, cancel and errors', () => {
  it('[BUG-09] while a file transcribes the dropzone is busy: its name, an aria-hidden elapsed time and Cancel', async () => {
    t = await mountView({ routes: { 'POST /v1/audio/transcriptions': () => delay(60000, json(200, verbose())) } });
    chooseFile(t, audioFile('long-talk.mp3'));
    await settle(t.env);
    const drop = t.$('tr-drop');
    assert.equal(drop.getAttribute('aria-busy'), 'true');
    assert.ok(drop.classList.contains('is-busy'));
    assert.equal(t.$('tr-file').disabled, true, 'one file at a time');
    assert.equal(t.$('tr-drop-title').textContent, 'long-talk.mp3');
    assert.equal(t.$('tr-drop-progress').hidden, false);
    assert.equal(t.$('tr-drop-progress').getAttribute('aria-hidden'), 'true');
    assert.equal(t.$('tr-upload-status').getAttribute('role'), 'status');
    assert.equal(t.$('tr-upload-status').textContent, 'Transcribing long-talk.mp3…');
    assert.equal(t.$('tr-upload-cancel').hidden, false);
    assert.equal(t.$('tr-record').disabled, true);
    assert.match(t.$('tr-rec-hint').textContent, /Finish or cancel the file first\./);
    await settle(t.env, 5000);
    assert.equal(t.$('tr-upload-elapsed').textContent, '0:05');
    assert.equal(t.$('tr-upload-status').textContent, 'Transcribing long-talk.mp3…', 'the live status never ticks');
  });

  it('[BUG-09] a second file while one is busy is refused (no parallel requests)', async () => {
    t = await mountView({ routes: { 'POST /v1/audio/transcriptions': () => delay(60000, json(200, verbose())) } });
    chooseFile(t, audioFile('one.mp3'));
    await settle(t.env);
    const dropped = new FakeEvent('drop', { bubbles: true, cancelable: true });
    dropped.dataTransfer = { files: [audioFile('two.mp3')] };
    t.$('tr-drop').dispatchEvent(dropped);
    await settle(t.env);
    assert.equal(transcribeCalls().length, 1);
    assert.match(t.$('app-toasts-polite').textContent, /One file at a time/);
  });

  it('[BUG-09] Cancel aborts the request and returns to idle; it is announced', async () => {
    t = await mountView({ routes: { 'POST /v1/audio/transcriptions': () => delay(60000, json(200, verbose())) } });
    chooseFile(t, audioFile('one.mp3'));
    await settle(t.env);
    const [call] = transcribeCalls();
    t.env.click(t.$('tr-upload-cancel'));
    await settle(t.env);
    assert.equal(call.signal.aborted, true);
    assert.equal(t.$('tr-drop').hasAttribute('aria-busy'), false);
    assert.equal(t.$('tr-file').disabled, false);
    assert.equal(t.$('tr-drop-title').textContent, 'Upload audio');
    assert.equal(t.$('tr-upload-cancel').hidden, true);
    assert.equal(t.$('tr-upload-error').hidden, true, 'a cancel is not an error');
    await settle(t.env, 600);
    assert.equal(announced(t), 'Transcription canceled');
    await settle(t.env, 60000);
    assert.equal(t.$('tr-log').children.length, 0, 'the late answer is dropped');
  });

  it('[TR-02][TE-04] a server error shows its message in the banner, never the JSON envelope; Details give its code; it can be dismissed', async () => {
    let fail = true;
    t = await mountView({
      routes: {
        'POST /v1/audio/transcriptions': () => (fail
          ? json(400, { error: { message: 'Unsupported audio format: video/x-flv', code: 'invalid_audio' } })
          : json(200, verbose())),
      },
    });
    chooseFile(t, audioFile('clip.flv', 512, 'video/x-flv'));
    await settle(t.env);
    await settle(t.env);
    assert.match(uploadError(), /Couldn't transcribe/);
    assert.equal(t.$('tr-upload-error').querySelector('.banner__msg').textContent, 'Unsupported audio format: video/x-flv');
    assert.equal(uploadError().includes('{'), false, 'no JSON envelope, not even in Details');
    // The server message goes in Details: with a code of its own, Details read
    // "message (code)", the power user's handle for the failure.
    assert.equal(t.$('tr-upload-error').querySelector('details pre').textContent, 'Unsupported audio format: video/x-flv (invalid_audio)');
    assert.equal(t.$('tr-upload-error').getAttribute('role'), 'alert');
    fail = false;
    const retry = t.$('tr-upload-error').querySelector('[data-action="upload-retry"]');
    noNode(retry, 'a 400 is not worth retrying as is');
    t.env.click(t.$('tr-upload-error').querySelector('[data-action="dismiss-upload-error"]'));
    await settle(t.env);
    assert.equal(t.$('tr-upload-error').hidden, true);
  });

  it('[TR-02][TE-04] Details only when they add something: a generic code, or none, leaves the message alone in the banner', async () => {
    const bodies = [
      json(413, { error: { message: 'File too large', code: 'http_error' } }),
      json(400, { detail: 'File too large' }),
      json(500, { error: { message: 'File too large' } }),
    ];
    let i = 0;
    t = await mountView({ routes: { 'POST /v1/audio/transcriptions': () => bodies[i] } });
    for (; i < bodies.length; i += 1) {
      chooseFile(t, audioFile(`clip${i}.wav`));
      await settle(t.env);
      await settle(t.env);
      const banner = t.$('tr-upload-error');
      assert.equal(banner.hidden, false, `${i}`);
      assert.equal(banner.querySelector('.banner__msg').textContent, 'File too large', `${i}`);
      noNode(banner.querySelector('details'), `${i}: Details would only repeat the message`);
      t.env.click(banner.querySelector('[data-action="dismiss-upload-error"]'));
      await settle(t.env);
    }
  });

  it('[TR-02] a proxy\'s HTML error page: plain copy, the raw text only in Details, and Retry sends the file again', async () => {
    let fail = true;
    t = await mountView({
      routes: {
        'POST /v1/audio/transcriptions': () => (fail
          ? text(502, '<html><body>Bad gateway</body></html>', { 'content-type': 'text/html' })
          : json(200, verbose())),
      },
    });
    chooseFile(t, audioFile('talk.mp3'));
    await settle(t.env);
    await settle(t.env);
    assert.equal(t.$('tr-upload-error').querySelector('.banner__title').textContent, "Couldn't transcribe");
    assert.match(uploadError(), /The server returned an unexpected response \(HTTP 502\)\./);
    const details = t.$('tr-upload-error').querySelector('details pre');
    assert.ok(details && details.textContent.includes('<html>'), 'the raw text is shown as text in Details');
    noNode(t.$('tr-upload-error').querySelector('body'), 'never rendered as HTML');
    fail = false;
    const retry = t.$('tr-upload-error').querySelector('[data-action="upload-retry"]');
    assert.equal(retry.textContent.trim(), 'Retry', 'the word every view uses');
    t.env.click(retry);
    await settle(t.env);
    await settle(t.env);
    assert.equal(transcribeCalls().length, 2);
    assert.equal(transcribeCalls()[1].body.get('file').name, 'talk.mp3');
    assert.equal(t.$('tr-upload-error').hidden, true);
    assert.equal(t.$('tr-log').querySelectorAll('.tr-seg').length, 2);
  });

  it('[BUG-09] an empty file is refused before any request', async () => {
    t = await mountView();
    chooseFile(t, new File([], 'empty.wav', { type: 'audio/wav' }));
    await settle(t.env);
    assert.equal(transcribeCalls().length, 0);
    assert.match(uploadError(), /Choose a non-empty audio file/);
  });

  it('[TR-02] a file replacing a recorded transcript asks first; uploads wait while recording', async () => {
    t = await mountView({ routes: { 'POST /v1/audio/transcriptions': () => json(200, verbose()) } });
    const ws = await startRecording(t);
    assert.equal(t.$('tr-file').disabled, true, 'no upload while recording');
    said(ws, 'Words I said.', { start: 0, end: 1 });
    t.env.click(t.$('tr-record'));
    ws.serverSend({ type: 'session.end' });
    await settle(t.env);
    chooseFile(t, audioFile());
    await settle(t.env);
    assert.equal(t.$('app-confirm-title').textContent, 'Replace the transcript?');
    await answerConfirm(t, false);
    assert.equal(transcribeCalls().length, 0);
    assert.ok(t.$('tr-log').textContent.includes('Words I said.'));
    chooseFile(t, audioFile());
    await settle(t.env);
    await answerConfirm(t, true);
    await settle(t.env);
    assert.equal(transcribeCalls().length, 1);
    assert.equal(t.$('tr-log').textContent.includes('Words I said.'), false);
  });
});
