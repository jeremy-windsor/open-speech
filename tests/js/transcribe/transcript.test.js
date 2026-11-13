// Transcribe: the transcript reducer, the role=log rendering, the file
// summary announcement (REV-24) and the timestamps that play the upload (REV-39).
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import {
  mountView, settle, startRecording, said, announced, chooseFile, audioFile, verbose, json, load, STT, FakeEvent, spokenText, sameNode,
} from './helpers.js';
import { installBrowserEnv, importFresh, flushMicrotasks } from '../harness/env.js';
import { liveRegions } from '../harness/a11y.js';

let t;
afterEach(() => t?.env?.restore());

async function pure() {
  const env = installBrowserEnv();
  t = { env };
  return importFresh('views/transcribe/transcript.js');
}

const msg = (text, { isFinal = true, speechFinal = false, start = 0, end = 1 } = {}) => ({
  type: 'transcript', message: { type: 'transcript', text, is_final: isFinal, speech_final: speechFinal, start, end },
});

describe('transcribe/transcript: the reducer', () => {
  it('[TR-07] is_final:false is interim; is_final without speech_final replaces the pending utterance; speech_final closes it', async () => {
    const tx = await pure();
    let s = tx.reduce(tx.createState(), { type: 'session-start', model: STT });
    s = tx.reduce(s, msg('hello', { isFinal: false }));
    assert.equal(s.interim, 'hello');
    assert.equal(s.segments.length, 0);
    s = tx.reduce(s, msg('hello there', { start: 0.5, end: 1.6 }));
    assert.deepEqual(s.confirmed, { text: 'hello there', start: 0.5, end: 1.6, source: 'mic' });
    assert.equal(s.interim, '');
    s = tx.reduce(s, msg('hello there friend', { start: 0.5, end: 2 }));
    assert.equal(s.confirmed.text, 'hello there friend', 'the pending utterance is replaced, not added');
    assert.equal(s.segments.length, 0);
    s = tx.reduce(s, msg('Hello there, friend.', { speechFinal: true, start: 0.5, end: 2.2 }));
    assert.equal(s.segments.length, 1);
    assert.deepEqual(s.segments[0], { text: 'Hello there, friend.', start: 0.5, end: 2.2, source: 'mic' });
    assert.equal(s.confirmed, null);
    assert.equal(s.interim, '');
  });

  it('[TR-07] copy and export keep every utterance of the session, joined with spaces (a pending one too)', async () => {
    const tx = await pure();
    let s = tx.reduce(tx.createState(), { type: 'session-start', model: STT });
    s = tx.reduce(s, msg('One.', { speechFinal: true, start: 0, end: 1 }));
    s = tx.reduce(s, msg('Two.', { speechFinal: true, start: 1.5, end: 2.5 }));
    s = tx.reduce(s, msg('Three so far', { start: 3, end: 4 }));
    assert.equal(tx.fullText(s), 'One. Two. Three so far');
    assert.equal(tx.exportSegments(s).length, 3);
    assert.equal(tx.hasContent(s), true);
    // The session ends with the third utterance unfinished: it is kept.
    s = tx.reduce(s, { type: 'session-end', duration: 5 });
    assert.equal(s.segments.length, 3);
    assert.equal(s.duration, 5);
  });

  it('[TR-07] a second recording continues the transcript; its times follow on; a file starts over', async () => {
    const tx = await pure();
    let s = tx.reduce(tx.createState(), { type: 'session-start', model: STT });
    s = tx.reduce(s, msg('First.', { speechFinal: true, start: 1, end: 2 }));
    s = tx.reduce(s, { type: 'session-end', duration: 10 });
    s = tx.reduce(s, { type: 'session-start', model: STT });
    s = tx.reduce(s, msg('Second.', { speechFinal: true, start: 1, end: 2 }));
    assert.deepEqual(s.segments.map((x) => [x.text, x.start]), [['First.', 1], ['Second.', 11]]);
    s = tx.reduce(s, { type: 'file-result', segments: [{ text: 'File.', start: 0, end: 1 }], duration: 1, model: STT, processingMs: 300, fileName: 'a.wav' });
    assert.equal(s.source, 'file');
    assert.deepEqual(s.segments.map((x) => x.text), ['File.']);
    s = tx.reduce(s, { type: 'session-start', model: STT });
    assert.equal(s.segments.length, 0, 'a recording after a file starts a new transcript');
  });

  it('[TR-08] only speech_final utterances become segments: interim and confirmed text never do', async () => {
    const tx = await pure();
    let s = tx.reduce(tx.createState(), { type: 'session-start', model: STT });
    s = tx.reduce(s, msg('maybe', { isFinal: false }));
    s = tx.reduce(s, msg('maybe this', { isFinal: true, speechFinal: false }));
    assert.equal(s.segments.length, 0);
    s = tx.reduce(s, msg('Maybe this.', { speechFinal: true }));
    assert.equal(s.segments.length, 1);
    // An empty speech_final with a pending utterance commits the pending one.
    s = tx.reduce(s, msg('and that', { isFinal: true, speechFinal: false, start: 2, end: 3 }));
    s = tx.reduce(s, msg('', { speechFinal: true, start: 2, end: 3 }));
    assert.deepEqual(s.segments.map((x) => x.text), ['Maybe this.', 'and that']);
  });

  it('[REV-24] the file summary says how many segments and how long, instead of the whole text', async () => {
    const tx = await pure();
    const s = tx.reduce(tx.createState(), {
      type: 'file-result', segments: Array.from({ length: 12 }, (_, i) => ({ text: `s${i}`, start: i * 10, end: i * 10 + 9 })), duration: 133,
    });
    assert.equal(tx.summary(s), 'Transcript ready: 12 segments, 2 minutes 13 seconds');
    assert.equal(tx.segmentAt(s, 45), 4);
    assert.equal(tx.segmentAt(s, 9.5), -1, 'between segments');
  });

  it('[TR-07] the pending line shows the confirmed words, then only the part of the hypothesis beyond them', async () => {
    const tx = await pure();
    let s = tx.reduce(tx.createState(), { type: 'session-start', model: STT });
    s = tx.reduce(s, msg('and then the speak view', { start: 0, end: 1 }));
    s = tx.reduce(s, msg('and then the speak view gets the editor', { isFinal: false }));
    assert.deepEqual(tx.pendingLine(s), { confirmed: 'and then the speak view', interim: 'gets the editor' });
    assert.deepEqual(tx.pendingLine(s, { showInterim: false }), { confirmed: 'and then the speak view', interim: '' });
  });
});

describe('transcribe view: the log', () => {
  it('[TR-08] #tr-log is role=log aria-live=polite with only p.tr-seg; the interim line sits outside it, aria-live=off', async () => {
    t = await mountView();
    const logEl = t.$('tr-log');
    const regions = liveRegions(t.root);
    assert.ok(regions.some((r) => r.el === logEl && r.role === 'log' && r.live === 'polite'));
    assert.equal(t.$('tr-interim').getAttribute('aria-live'), 'off');
    assert.equal(logEl.contains(t.$('tr-interim')), false);
    const ws = await startRecording(t);
    said(ws, 'partial words', { isFinal: false });
    await settle(t.env);
    assert.equal(logEl.children.length, 0, 'interim text never enters the log');
    assert.ok(t.$('tr-interim').textContent.includes('partial words'));
    said(ws, 'First sentence.', { start: 3.2, end: 5 });
    await settle(t.env);
    said(ws, 'Second one', { isFinal: true, speechFinal: false, start: 6, end: 7 });
    await settle(t.env);
    assert.equal(logEl.children.length, 1, 'one append per final');
    assert.ok(t.$('tr-interim').textContent.includes('Second one'), 'the pending utterance shows outside the log');
    said(ws, 'Second one.', { start: 6, end: 7.5 });
    await settle(t.env);
    assert.equal(logEl.children.length, 2);
    for (const child of logEl.children) assert.ok(child.matches('p.tr-seg'));
    assert.equal(logEl.getAttribute('aria-live'), 'polite', 'microphone segments are announced one at a time');
    assert.equal(t.$('tr-interim').textContent.includes('Second one'), false);
  });

  it('[REV-39] microphone times are plain text, never buttons that do nothing (the audio is not kept)', async () => {
    t = await mountView();
    const ws = await startRecording(t);
    said(ws, 'Hello.', { start: 3.2, end: 4 });
    await settle(t.env);
    const seg = t.$('tr-log').querySelector('.tr-seg');
    const ts = seg.querySelector('.tr-ts');
    assert.equal(ts.localName, 'span');
    assert.equal(ts.textContent, '0:03');
    assert.equal(t.$('tr-log').querySelectorAll('button').length, 0);
    assert.equal(t.$('tr-play').hidden, true);
  });

  it('[TR-08] each appended microphone segment is read as its words, not "zero colon zero three" first', async () => {
    t = await mountView();
    const ws = await startRecording(t);
    said(ws, 'Okay so the plan.', { start: 3.2, end: 5 });
    await settle(t.env);
    const seg = t.$('tr-log').lastElementChild;
    // The log announces what is appended: the words. The time stays on screen (and in every export).
    assert.equal(spokenText(seg).trim(), 'Okay so the plan.');
    assert.equal(seg.querySelector('.tr-ts').textContent, '0:03');
    assert.equal(seg.querySelector('.tr-ts').hidden, false);
    const { exportFile } = await load('views/transcribe/export.js');
    const tx = await load('views/transcribe/transcript.js');
    const srt = exportFile('srt', { segments: tx.exportSegments(t.ctl.state().transcript), model: STT, duration: 5 });
    assert.match(srt.content, /00:00:03,200 --> 00:00:05,000\nOkay so the plan\./);
    // A file's times are buttons named "Play from 0:03": they stay in the accessibility tree.
    const { segmentHtml } = await load('views/transcribe/log.js');
    assert.equal(segmentHtml({ text: 'x', start: 3 }, 0, { playable: true }).markup.includes('aria-hidden'), false);
  });

  it('[REV-24] a file transcript goes in while the log is aria-live=off and one summary is announced', async () => {
    t = await mountView({ routes: { 'POST /v1/audio/transcriptions': () => json(200, verbose([[0, 3.12, 'Hello there.'], [3.12, 133.4, 'General Kenobi.']])) } });
    const logEl = t.$('tr-log');
    // Record, in order, every aria-live change and every write of the log's content.
    const events = [];
    const setAttribute = logEl.setAttribute.bind(logEl);
    logEl.setAttribute = (name, value) => { if (name === 'aria-live') events.push(`live:${value}`); return setAttribute(name, value); };
    let proto = Object.getPrototypeOf(logEl);
    while (proto && !Object.getOwnPropertyDescriptor(proto, 'innerHTML')) proto = Object.getPrototypeOf(proto);
    const inner = Object.getOwnPropertyDescriptor(proto, 'innerHTML');
    Object.defineProperty(logEl, 'innerHTML', {
      configurable: true,
      get() { return inner.get.call(this); },
      set(v) { events.push('content'); inner.set.call(this, v); },
    });
    const append = logEl.appendChild.bind(logEl);
    logEl.appendChild = (node) => { events.push('content'); return append(node); };
    chooseFile(t, audioFile('talk.mp3'));
    await settle(t.env);
    await settle(t.env);
    assert.equal(logEl.querySelectorAll('.tr-seg').length, 2);
    const firstContent = events.indexOf('content');
    assert.ok(firstContent > 0, JSON.stringify(events));
    assert.equal(events.slice(0, firstContent).pop(), 'live:off', `inserted while off: ${JSON.stringify(events)}`);
    const lastContent = events.lastIndexOf('content');
    assert.ok(!events.slice(firstContent, lastContent).includes('live:polite'), `every write happens while off: ${JSON.stringify(events)}`);
    await settle(t.env, 600);
    assert.equal(events[events.length - 1], 'live:polite', 'back to polite for the next recording');
    assert.equal(logEl.getAttribute('aria-live'), 'polite');
    assert.equal(announced(t), 'Transcript ready: 2 segments, 2 minutes 13 seconds');
  });

  it('[REV-24] the log stays off until the browser has taken the file in: two frames and a moment, never time alone', async () => {
    t = await mountView({ routes: { 'POST /v1/audio/transcriptions': () => json(200, verbose()) } });
    const logEl = t.$('tr-log');
    const live = () => logEl.getAttribute('aria-live');
    t.env.raf.flush();
    chooseFile(t, audioFile('talk.mp3'));
    await flushMicrotasks(30);   // the answer arrives; no frame has run yet
    assert.equal(logEl.querySelectorAll('.tr-seg').length, 2);
    assert.equal(live(), 'off');
    t.env.raf.flush();   // frame 1: the accessibility tree takes in the insertion while the log is off
    assert.equal(live(), 'off');
    await t.env.clock.tickAsync(2000);   // no frame (a background tab): it stays off, whatever the time
    assert.equal(live(), 'off');
    t.env.raf.flush();   // frame 2
    assert.equal(live(), 'off');
    await t.env.clock.tickAsync(99);
    assert.equal(live(), 'off');
    await t.env.clock.tickAsync(1);
    assert.equal(live(), 'polite', 'ready to announce the next recording');
  });

  it('[REV-24] continuing, cancelling or clearing changes no announced content: no quiet window then', async () => {
    t = await mountView();
    const logEl = t.$('tr-log');
    const ws1 = await startRecording(t);
    said(ws1, 'Kept.', { start: 0, end: 1 });
    t.env.click(t.$('tr-record'));
    ws1.serverSend({ type: 'session.end' });
    await settle(t.env);
    const kept = logEl.querySelector('.tr-seg');
    // A second recording continues the transcript: what is on screen stays as it is (not re-rendered).
    const ws2 = await startRecording(t);
    sameNode(logEl.querySelector('.tr-seg'), kept);
    assert.equal(logEl.getAttribute('aria-live'), 'polite');
    said(ws2, 'Dropped.', { start: 0, end: 1 });
    await settle(t.env);
    assert.equal(logEl.querySelectorAll('.tr-seg').length, 2);
    // Cancel drops that session's segment by removing it: removals are never read out.
    t.env.click(t.$('tr-cancel'));
    await settle(t.env);
    assert.equal(logEl.querySelectorAll('.tr-seg').length, 1);
    sameNode(logEl.querySelector('.tr-seg'), kept);
    assert.equal(logEl.getAttribute('aria-live'), 'polite');
  });

  it('[REV-39] file times are buttons that play the upload from that point through the app audio', async () => {
    t = await mountView({ routes: { 'POST /v1/audio/transcriptions': () => json(200, verbose([[0, 3.12, 'Hello there.'], [3.12, 5.4, 'General Kenobi.']])) } });
    chooseFile(t, audioFile('talk.mp3'));
    await settle(t.env);
    await settle(t.env);
    const buttons = t.$('tr-log').querySelectorAll('button.tr-ts');
    assert.equal(buttons.length, 2);
    assert.equal(buttons[1].getAttribute('aria-label'), 'Play from 0:03');
    assert.equal(buttons[1].textContent, '0:03');
    const url = t.ctl.state().fileUrl;
    assert.ok(url.startsWith('blob:'), 'the upload is kept as an object URL');
    t.env.click(buttons[1]);
    await settle(t.env);
    const audio = await load('core/audio.js');
    const el = audio.element();
    assert.equal(el.src, url);
    assert.equal(el.currentTime, 3.12);
    const { app } = await load('core/store.js');
    assert.equal(app.get().player.id, 'transcribe:file');
    assert.equal(app.get().player.title, 'talk.mp3');
    // The playing segment is marked, and the play button becomes Pause.
    el.__timeupdate(4);
    await settle(t.env, 300);
    el.__timeupdate(4.1);
    await settle(t.env);
    assert.ok(t.$('tr-log').querySelectorAll('.tr-seg')[1].classList.contains('is-current'));
    assert.equal(t.$('tr-play').hidden, false);
    assert.equal(t.$('tr-play').getAttribute('aria-label'), 'Pause');
    t.env.click(t.$('tr-play'));
    await settle(t.env);
    assert.equal(el.paused, true);
    assert.equal(t.$('tr-play').getAttribute('aria-label'), 'Play the file');
  });

  it('[REV-39] Clear releases the upload (the object URL is revoked) after a confirm', async () => {
    t = await mountView({ routes: { 'POST /v1/audio/transcriptions': () => json(200, verbose()) } });
    chooseFile(t, audioFile('talk.mp3'));
    await settle(t.env);
    await settle(t.env);
    const url = t.ctl.state().fileUrl;
    t.env.click(t.$('tr-clear'));
    await settle(t.env);
    assert.equal(t.$('app-confirm-title').textContent, 'Clear the transcript?');
    t.env.click(t.$('app-confirm-ok'));
    await settle(t.env);
    assert.ok(t.env.revokedUrls.includes(url));
    assert.equal(t.$('tr-log').children.length, 0);
    assert.equal(t.$('tr-empty').hidden, false);
    assert.equal(t.$('tr-clear').disabled, true);
    sameNode(t.doc.activeElement, t.$('tr-transcript-title'), 'focus lands on the panel heading, not on <body>');
  });

  it('[TR-04] a finished file brings its words into view when they start under the bottom bar, not when they show', async () => {
    t = await mountView({ routes: { 'POST /v1/audio/transcriptions': () => json(200, verbose()) } });
    const panel = t.$('tr-panel');
    let proto = Object.getPrototypeOf(panel);
    while (proto && !Object.prototype.hasOwnProperty.call(proto, 'getBoundingClientRect')) proto = Object.getPrototypeOf(proto);
    const original = proto.getBoundingClientRect;
    let segTop = 0;
    proto.getBoundingClientRect = function rect() {
      if (this.classList && this.classList.contains('tr-seg')) return { top: segTop, bottom: segTop + 40, left: 0, right: 300, width: 300, height: 40, x: 0, y: segTop };
      return original.call(this);
    };
    try {
      // A 390 x 844 phone: the panel's head sits at the bottom edge and the first words are under the bar.
      segTop = 760;
      chooseFile(t, audioFile('one.mp3'));
      await settle(t.env);
      await settle(t.env);
      assert.deepEqual(panel.__scrolledIntoView, [{ block: 'start', behavior: 'smooth' }]);
      // The words already show: nothing moves.
      segTop = 300;
      chooseFile(t, audioFile('two.mp3'));
      await settle(t.env);
      await settle(t.env);
      assert.equal(panel.__scrolledIntoView.length, 1);
    } finally {
      proto.getBoundingClientRect = original;
    }
  });

  it('[REV-39] the upload goes with the page (pagehide revokes it), but a page kept for Back keeps it playable', async () => {
    t = await mountView({ routes: { 'POST /v1/audio/transcriptions': () => json(200, verbose()) } });
    chooseFile(t, audioFile('talk.mp3'));
    await settle(t.env);
    await settle(t.env);
    const url = t.ctl.state().fileUrl;
    const kept = new FakeEvent('pagehide');
    kept.persisted = true;   // into the back/forward cache
    t.env.window.dispatchEvent(kept);
    assert.equal(t.env.revokedUrls.includes(url), false);
    assert.equal(t.ctl.state().fileUrl, url);
    t.env.window.dispatchEvent(new FakeEvent('pagehide'));
    assert.ok(t.env.revokedUrls.includes(url));
    assert.equal(t.ctl.state().fileUrl, '');
  });

  it('[TR-08] the Interim text and Timestamps switches persist in the preferences', async () => {
    t = await mountView();
    const prefs = await load('core/prefs.js');
    const ws = await startRecording(t);
    said(ws, 'Done.', { start: 1, end: 2 });
    said(ws, 'the next words', { isFinal: false });
    await settle(t.env);
    assert.ok(t.$('tr-interim').textContent.includes('the next words'));
    t.env.change(t.$('tr-show-interim'), false);
    await settle(t.env);
    assert.equal(t.$('tr-interim').textContent.includes('the next words'), false);
    assert.equal(prefs.get().showInterim, false);
    t.env.change(t.$('tr-show-ts'), false);
    await settle(t.env);
    assert.ok(t.$('tr-log').classList.contains('is-compact'));
    assert.equal(prefs.get().compactTranscript, true);
    assert.match(t.env.localStorage.__data['open-speech-prefs'], /"compactTranscript":true/);
  });

  it('[TR-08] while recording with nothing said yet the transcript says it is listening; the empty state hides', async () => {
    t = await mountView();
    assert.equal(t.$('tr-empty').hidden, false);
    assert.match(t.$('tr-empty').textContent, /Your transcript appears here\./);
    await startRecording(t);
    assert.equal(t.$('tr-empty').hidden, true);
    assert.match(t.$('tr-interim').textContent, /Listening…/);
  });
});
