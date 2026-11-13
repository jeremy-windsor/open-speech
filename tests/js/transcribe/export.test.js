// Transcribe: exports and copy: TXT, SRT, VTT and JSON, transcript-<timestamp> file names,
// Copy through the Clipboard API with the execCommand fallback (plain HTTP), and nothing to save before a
// segment exists (BUG-17).
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { mountView, settle, startRecording, said, chooseFile, audioFile, verbose, json, sameNode, noNode } from './helpers.js';
import { installBrowserEnv, importFresh } from '../harness/env.js';

let t;
afterEach(() => t?.env?.restore());

async function pure() {
  const env = installBrowserEnv();
  t = { env };
  return importFresh('views/transcribe/export.js');
}

const SEGMENTS = [
  { text: 'Hello there.', start: 3.12, end: 5.4 },
  { text: 'General Kenobi.', start: 5.4, end: 3725.007 },
];

describe('transcribe/export: formats', () => {
  it('[BUG-17] SRT: numbered cues with a comma before the milliseconds', async () => {
    const ex = await pure();
    assert.equal(ex.toSrt(SEGMENTS), '1\n00:00:03,120 --> 00:00:05,400\nHello there.\n\n2\n00:00:05,400 --> 01:02:05,007\nGeneral Kenobi.\n\n');
  });

  it('[BUG-17] VTT: the WEBVTT header and a period before the milliseconds', async () => {
    const ex = await pure();
    assert.equal(ex.toVtt(SEGMENTS), 'WEBVTT\n\n00:00:03.120 --> 00:00:05.400\nHello there.\n\n00:00:05.400 --> 01:02:05.007\nGeneral Kenobi.\n\n');
    assert.equal(ex.toVtt([]), 'WEBVTT\n\n');
  });

  it('VTT cue text escapes "&" first, then "<" and ">", so a player shows it as written; SRT keeps it raw', async () => {
    // WebVTT cue text (W3C, §4.2.2 and §6.4): a raw "<" starts a tag that hides the rest of the cue, and a raw
    // "&" starts a character reference; the server's formatter escapes them the same way (src/formatters.py).
    const ex = await pure();
    const cue = (text) => ex.toVtt([{ text, start: 0, end: 1 }]).split('\n')[3];
    assert.equal(cue('a < b & c'), 'a &lt; b &amp; c');
    assert.equal(cue('Scores <5 points, if a<b then'), 'Scores &lt;5 points, if a&lt;b then');
    assert.equal(cue('Type &amp; for an ampersand and &lt; for less-than'), 'Type &amp;amp; for an ampersand and &amp;lt; for less-than', 'the words stay words');
    assert.equal(cue('x > y, then a --> b'), 'x &gt; y, then a -&gt; b', 'no "-->" inside a cue, and ">" escaped too');
    assert.equal(ex.toVtt([{ text: 'a < b & c', start: 0, end: 1.5 }]), 'WEBVTT\n\n00:00:00.000 --> 00:00:01.500\na &lt; b &amp; c\n\n');
    // SubRip has no escape syntax: the text stays as written.
    assert.equal(ex.toSrt([{ text: 'a < b & c --> d', start: 0, end: 1 }]), '1\n00:00:00,000 --> 00:00:01,000\na < b & c -> d\n\n');
    assert.equal(ex.toTxt([{ text: 'a < b & c', start: 0, end: 1 }]), 'a < b & c\n');
    assert.equal(JSON.parse(ex.toJson({ segments: [{ text: 'a < b & c', start: 0, end: 1 }] })).text, 'a < b & c');
  });

  it('[TR-17] TXT is the full text, every segment joined with spaces', async () => {
    const ex = await pure();
    assert.equal(ex.toTxt(SEGMENTS), 'Hello there. General Kenobi.\n');
    assert.equal(ex.toTxt([]), '');
  });

  it('[TR-17] JSON: {text, segments: [{start, end, text}], model, duration}', async () => {
    const ex = await pure();
    const data = JSON.parse(ex.toJson({ segments: SEGMENTS, model: 'Systran/faster-whisper-base', duration: 3725.0071 }));
    assert.deepEqual(data, {
      text: 'Hello there. General Kenobi.',
      segments: [{ start: 3.12, end: 5.4, text: 'Hello there.' }, { start: 5.4, end: 3725.007, text: 'General Kenobi.' }],
      model: 'Systran/faster-whisper-base',
      duration: 3725.007,
    });
  });

  it('[BUG-17] cues players accept: empty segments dropped, a missing end fixed, no blank line or "-->" inside a cue', async () => {
    const ex = await pure();
    const fixed = ex.cues([{ text: 'a', start: 2, end: 0 }, { text: '  ', start: 3, end: 4 }, { text: 'b', start: 4, end: 4 }]);
    assert.deepEqual(fixed, [{ text: 'a', start: 2, end: 4 }, { text: 'b', start: 4, end: 4.5 }]);
    assert.equal(ex.toSrt([{ text: 'one\n\n\ntwo --> three', start: 0, end: 1 }]), '1\n00:00:00,000 --> 00:00:01,000\none\ntwo -> three\n\n');
    assert.equal(ex.cueTime(-1, '.'), '00:00:00.000');
  });

  it('[TR-17] file names are transcript-<timestamp>.<ext>; an unknown format is refused', async () => {
    const ex = await pure();
    const when = new Date(2026, 9, 22, 14, 5, 12);
    assert.equal(ex.fileName('srt', when), 'transcript-20261022-140512.srt');
    const file = ex.exportFile('vtt', { segments: SEGMENTS }, when);
    assert.equal(file.filename, 'transcript-20261022-140512.vtt');
    assert.match(file.mime, /^text\/vtt/);
    assert.throws(() => ex.exportFile('docx', { segments: SEGMENTS }), TypeError);
  });
});

async function recordedTranscript() {
  t = await mountView();
  const ws = await startRecording(t);
  said(ws, 'First thing.', { start: 0.5, end: 1.5 });
  said(ws, 'Second thing', { isFinal: true, speechFinal: false, start: 2, end: 3 });
  await settle(t.env);
  return ws;
}

describe('transcribe view: copy and export', () => {
  it('[BUG-17] Copy, Export and Clear are off until a segment exists (nothing is ever saved as "—")', async () => {
    t = await mountView();
    assert.equal(t.$('tr-copy').disabled, true);
    assert.equal(t.$('tr-export').disabled, true);
    assert.equal(t.$('tr-clear').disabled, true);
    const ws = await startRecording(t);
    said(ws, 'hmm', { isFinal: false });
    await settle(t.env);
    assert.equal(t.$('tr-copy').disabled, true, 'interim text is not a transcript yet');
    said(ws, 'Now it is.', { start: 0, end: 1 });
    await settle(t.env);
    assert.equal(t.$('tr-copy').disabled, false);
    assert.equal(t.$('tr-export').disabled, false);
    assert.equal(t.$('tr-clear').disabled, true, 'not while recording');
  });

  it('[TR-16] [TE-09] Copy uses the Clipboard API and confirms with "Transcript copied"; every utterance is in it', async () => {
    await recordedTranscript();
    t.env.click(t.$('tr-copy'));
    await settle(t.env);
    assert.deepEqual(t.env.clipboard.written, ['First thing. Second thing']);
    assert.match(t.$('app-toasts-polite').textContent, /Transcript copied/);
  });

  it('[TR-16] over plain HTTP (no Clipboard API) the hidden-textarea copy works and focus returns to Copy', async () => {
    // Recording needs HTTPS, so the transcript comes from a file here.
    t = await mountView({
      env: { clipboard: 'absent', secure: false },
      routes: { 'POST /v1/audio/transcriptions': () => json(200, verbose([[0, 1, 'Over plain HTTP.']])) },
    });
    chooseFile(t, audioFile());
    await settle(t.env);
    await settle(t.env);
    const copy = t.$('tr-copy');
    // The shell's own textareas (the Ask AI question box) stay; the helper's must not.
    const shellAreas = t.doc.querySelectorAll('textarea');
    copy.focus();
    t.env.click(copy);
    await settle(t.env);
    assert.deepEqual(t.env.clipboard.written, ['Over plain HTTP.']);
    sameNode(t.doc.activeElement, copy);
    const areas = t.doc.querySelectorAll('textarea');
    assert.ok(areas.length === shellAreas.length && areas.every((area, i) => area === shellAreas[i]), 'the helper textarea is gone');
    assert.match(t.$('app-toasts-polite').textContent, /Transcript copied/);
  });

  it('[TR-16] when copying is impossible the error says what to do instead', async () => {
    t = await mountView({ env: { clipboard: 'reject' } });
    t.env.execCommandResult = false;
    const ws = await startRecording(t);
    said(ws, 'Stuck words.', { start: 0, end: 1 });
    await settle(t.env);
    t.env.click(t.$('tr-copy'));
    await settle(t.env);
    assert.match(t.$('app-toasts-alert').textContent, /Couldn't copy here: select the transcript and copy it instead/);
  });

  it('[TR-17] Export › Text saves transcript-<timestamp>.txt with the full text', async () => {
    await recordedTranscript();
    t.env.click(t.$('tr-export'));
    await settle(t.env);
    const menu = t.$('tr-export-menu');
    assert.equal(menu.hidden, false);
    assert.equal(t.$('tr-export').getAttribute('aria-expanded'), 'true');
    const created = t.env.createdUrls.length;
    t.env.click(menu.querySelector('[data-format="txt"]'));
    await settle(t.env);
    assert.equal(menu.hidden, true);
    const url = t.env.createdUrls[created];
    const blob = t.env.objectUrls.get(url);
    assert.equal(await blob.text(), 'First thing. Second thing\n');
    assert.match(blob.type, /^text\/plain/);
    // A success confirmation, like Copy's: a toast that names the file.
    assert.match(t.$('app-toasts-polite').textContent, /Saved transcript-\d{8}-\d{6}\.txt/);
  });

  it('[BUG-17] Export › SRT and VTT carry the segment times; JSON carries the model and the formatted duration', async () => {
    t = await mountView({ routes: { 'POST /v1/audio/transcriptions': () => json(200, verbose([[0, 3.12, 'Hello there.'], [3.12, 5.4, 'General Kenobi.']])) } });
    chooseFile(t, audioFile());
    await settle(t.env);
    await settle(t.env);
    assert.equal(t.$('tr-meta').textContent, 'Duration 0:05 · Processed in 0.0 s · whisper base', 'no raw float such as 5.4s');
    const save = async (format) => {
      t.env.click(t.$('tr-export'));
      await settle(t.env);
      const created = t.env.createdUrls.length;
      t.env.click(t.$('tr-export-menu').querySelector(`[data-format="${format}"]`));
      await settle(t.env);
      return t.env.objectUrls.get(t.env.createdUrls[created]).text();
    };
    assert.match(await save('srt'), /^1\n00:00:00,000 --> 00:00:03,120\nHello there\.\n\n2\n00:00:03,120 --> 00:00:05,400\n/);
    assert.match(await save('vtt'), /^WEBVTT\n\n00:00:00\.000 --> 00:00:03\.120\n/);
    const data = JSON.parse(await save('json'));
    assert.equal(data.model, 'Systran/faster-whisper-base');
    assert.equal(data.duration, 5.4);
    assert.equal(data.segments.length, 2);
  });
});
