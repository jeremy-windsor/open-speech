// Reader › Ask AI: the open document's selection, current chapter or whole document
// goes to the shared Ask AI sheet (components/ask-ai.js) as a `document` input of POST /api/assistant/ask:
// only its id and the range, never its text. A whole book is a job with progress and Cancel.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { installBrowserEnv, importFresh } from '../harness/env.js';
import { readerEnv, makeDoc, go, settle, json, $, textOf } from './helpers.js';

let ctx;
let selection = null;
afterEach(() => {
  delete globalThis.getSelection;
  selection = null;
  ctx?.env?.restore();
});

const PRESETS = [
  { id: 'voice', name: 'Voice', label: 'Answer', inputs: ['text', 'transcript'], output: 'speech', chunk_chars: 12000 },
  { id: 'book', name: 'Book', label: 'Rewrite for listening', inputs: ['document'], output: 'document', chunk_chars: 12000 },
  { id: 'explain', name: 'Explain', label: 'Explain', inputs: ['document'], output: 'text', chunk_chars: 12000 },
];

function assistantRoutes(state) {
  return {
    'GET /api/assistant': () => json(200, { enabled: state.enabled !== false, host: '127.0.0.1:9', model: 'qwen3', server_settings: {} }),
    'GET /api/assistant/presets': () => json(200, { presets: PRESETS }),
    'POST /api/assistant/ask': (req) => {
      state.asks.push(req.json);
      if (state.answer) return state.answer(req);
      return json(200, { reply: 'It is about a rabbit hole.', thinking: null, finish_reason: 'stop', elapsed_ms: 700, preset: req.json.preset, input: { kind: 'document', chars: 60 } });
    },
    'GET /v1/reader/assistant-jobs/:id': () => json(200, state.job),
    'DELETE /v1/reader/assistant-jobs/:id': (req) => {
      state.deletes.push(req.params.id);
      state.job = { ...state.job, status: 'cancelled' };
      return json(200, state.job);
    },
  };
}

/** A fake Selection over the text nodes of two paragraphs. */
function selectBlocks(first, last, words = 'Paragraph 1 of chapter 1') {
  const a = document.querySelector(`#rd-text [data-block="${first}"]`);
  const b = document.querySelector(`#rd-text [data-block="${last}"]`);
  selection = {
    isCollapsed: false, rangeCount: 1, anchorNode: a.firstChild || a, focusNode: b.firstChild || b,
    toString: () => words,
  };
  globalThis.getSelection = () => selection;
}

async function openDoc(state = {}) {
  const s = { asks: [], deletes: [], job: null, ...state };
  ctx = await readerEnv({ routes: assistantRoutes(s) });
  ctx.state = s;
  await go(ctx, 'reader/alice', 5);
  await settle(ctx.env);
  return ctx;
}

const sourceIds = () => document.querySelectorAll('#ask-sources input[name="ask-source"]').map((r) => r.value);

describe('reader/ask-sources: the pure parts', () => {
  it('readerSources: the selection (whole paragraphs, by block index), this chapter (by chapter index) and the whole document (just its id), with their sizes', async () => {
    installBrowserEnv();
    const { readerSources } = await importFresh('views/reader/ask-sources.js');
    const doc = makeDoc();   // 3 chapters of 5 blocks: a heading and four paragraphs each
    const sources = readerSources(doc, { selection: { first: 6, last: 7, text: 'Paragraph 1' }, chapter: 1 });
    assert.deepEqual(sources.map((s) => s.id), ['selection', 'chapter', 'document']);
    assert.deepEqual(sources[0].input, { kind: 'document', document_id: 'alice', block_start: 6, block_end: 7 });
    assert.equal(sources[0].label, 'The 2 selected paragraphs');
    assert.equal(sources[0].chars, Array.from(doc.blocks[6].text).length + Array.from(doc.blocks[7].text).length);
    assert.deepEqual(sources[1].input, { kind: 'document', document_id: 'alice', chapters: [1, 1] });
    assert.equal(sources[1].label, 'This chapter: Chapter 2');
    assert.deepEqual(sources[2].input, { kind: 'document', document_id: 'alice' });
    assert.equal(sources[2].chapters, 3, 'a whole book of three chapters is a job');
    // One chapter: no "this chapter" (it is the whole document); no selection: none offered.
    const note = makeDoc({ id: 'note', chapters: 1, perChapter: 3 });
    assert.deepEqual(readerSources(note, { chapter: 0 }).map((s) => s.id), ['document']);
    assert.deepEqual(readerSources({ id: 'empty', blocks: [], chapters: [] }), []);
  });

  it('selectionRange: the blocks the selection\'s two ends sit in, inside the text only', async () => {
    const env = installBrowserEnv();
    const { selectionRange } = await importFresh('views/reader/ask-sources.js');
    const box = env.mount('<article id="t"><p data-block="4">One</p><p data-block="5">Two</p></article>');
    const outside = env.mount('<p data-block="9">Elsewhere</p>');
    const [p4, p5] = box.querySelectorAll('p');
    const sel = (anchor, focus, text = 'x') => ({ isCollapsed: false, rangeCount: 1, anchorNode: anchor, focusNode: focus, toString: () => text });
    assert.deepEqual(selectionRange(sel(p5.firstChild, p4.firstChild, 'Two One'), box), { first: 4, last: 5, text: 'Two One' });
    assert.equal(selectionRange(sel(outside.firstChild, outside.firstChild), box), null);
    assert.equal(selectionRange({ isCollapsed: true, rangeCount: 1 }, box), null);
    assert.equal(selectionRange(sel(p4.firstChild, p4.firstChild, '  '), box), null, 'only white space');
    assert.equal(selectionRange(null, box), null);
  });
});

describe('reader: Ask AI', () => {
  it('the button is in the reading view\'s top bar, shown while the assistant is on', async () => {
    await openDoc();
    const ask = $('rd-ask');
    assert.equal(ask.hidden, false);
    assert.equal(ask.getAttribute('aria-label'), 'Ask AI');
    assert.equal(ask.closest('.rd-topbar') !== null, true);
    ctx.env.restore();
    ctx = null;
    await openDoc({ enabled: false });
    assert.equal($('rd-ask').hidden, true);
  });

  it('a selection: its whole paragraphs go by block index, with the document presets (book first)', async () => {
    await openDoc();
    selectBlocks(6, 7);
    ctx.env.click($('rd-ask'));
    await settle(ctx.env);
    await settle(ctx.env);
    assert.equal($('app-ask-ai').open, true);
    // This paragraph (the highlighted one) comes before the selection; nothing has been read yet, so
    // there is no This sentence. The selection the listener just made is still the one chosen.
    assert.deepEqual(sourceIds(), ['paragraph', 'selection', 'chapter', 'document']);
    assert.equal(document.querySelector('#ask-sources input:checked').value, 'selection', 'the selection first');
    assert.deepEqual($('ask-preset').querySelectorAll('option').map((o) => o.value), ['book', 'explain']);
    assert.equal($('ask-preset').value, 'book');
    ctx.env.change($('ask-preset'), 'explain');
    ctx.env.click($('ask-send'));
    await settle(ctx.env);
    await settle(ctx.env);
    assert.deepEqual(ctx.state.asks[0], { input: { kind: 'document', document_id: 'alice', block_start: 6, block_end: 7 }, preset: 'explain' });
    assert.equal($('ask-reply').textContent, 'It is about a rabbit hole.');
  });

  it('no selection: the current chapter (the one on screen) is offered first, by its chapter index', async () => {
    await openDoc();   // progress at block 12: chapter 3 (blocks 10 to 14)
    ctx.env.click($('rd-ask'));
    await settle(ctx.env);
    await settle(ctx.env);
    // This paragraph is listed before it; with nothing read yet, This chapter is the one chosen.
    assert.deepEqual(sourceIds(), ['paragraph', 'chapter', 'document']);
    assert.equal(document.querySelector('#ask-sources input:checked').value, 'chapter');
    assert.match(textOf($('ask-sources')), /This chapter: Chapter 3/);
    ctx.env.change($('ask-preset'), 'explain');
    ctx.env.click($('ask-send'));
    await settle(ctx.env);
    await settle(ctx.env);
    assert.deepEqual(ctx.state.asks[0].input, { kind: 'document', document_id: 'alice', chapters: [2, 2] });
  });

  it('the whole document goes as a job: progress, then Cancel (DELETE); the privacy line says the answers are kept in the Reader', async () => {
    await openDoc({
      answer: () => {
        ctx.state.job = { id: 'job-9', document_id: 'alice', status: 'running', chapters: [{ pieces: 1, pieces_done: 1 }, { pieces: 1, pieces_done: 0 }, { pieces: 1, pieces_done: 0 }] };
        return json(202, { job: ctx.state.job, preset: 'book', input: { kind: 'document', chars: 1200 } });
      },
    });
    ctx.env.click($('rd-ask'));
    await settle(ctx.env);
    await settle(ctx.env);
    const whole = document.querySelectorAll('#ask-sources input[name="ask-source"]').find((r) => r.value === 'document');
    ctx.env.change(whole, true);
    assert.match(textOf($('ask-privacy-text')), /This is sent to 127\.0\.0\.1:9\. Its answers are saved in the Reader/);
    ctx.env.click($('ask-send'));
    await settle(ctx.env);
    await settle(ctx.env);
    assert.deepEqual(ctx.state.asks[0], { input: { kind: 'document', document_id: 'alice' }, preset: 'book', output: { document: {} } });
    assert.equal(textOf($('ask-progress-text')), '1 of 3 pieces');
    ctx.env.click($('ask-cancel'));
    await settle(ctx.env);
    assert.deepEqual(ctx.state.deletes, ['job-9']);
  });

  it('the screenshot states over a document open the sheet with its sources and send nothing', async () => {
    await openDoc();
    await go(ctx, 'reader/alice?shot=ask-ai', 5);
    await settle(ctx.env);
    assert.equal($('app-ask-ai').open, true);
    // The shot marks the current paragraph's first sentence as read, so every source is offered.
    assert.deepEqual(sourceIds(), ['sentence', 'paragraph', 'selection', 'chapter', 'document']);
    assert.equal(ctx.state.asks.length, 0);
  });
});

// The sources, which the server takes: This sentence (the one
// being read, or the last one read) and This paragraph (the highlighted one) before the selection, the chapter and
// the whole document; the default This sentence while something has been read, otherwise This chapter (the
// selection, when the listener made one, stays the one chosen); and the line saying the document's name and chapter
// go with it.
describe('reader: Ask AI about this sentence or this paragraph', () => {
  const checked = () => document.querySelector('#ask-sources input:checked').value;
  const pick = (id) => ctx.env.change(document.querySelectorAll('#ask-sources input[name="ask-source"]').find((r) => r.value === id), true);
  async function press() {
    ctx.env.click($('rd-ask'));
    await settle(ctx.env);
    await settle(ctx.env);
  }
  async function sendIt() {
    ctx.env.click($('ask-send'));
    await settle(ctx.env);
    await settle(ctx.env);
  }

  it('readerSources: this sentence (block and sentence, its text by code point) and this paragraph (blocks [b, b]) come first; each says what goes with it', async () => {
    installBrowserEnv();
    const { readerSources, defaultSource } = await importFresh('views/reader/ask-sources.js');
    const doc = makeDoc();   // 3 chapters of 5 blocks: a heading and four paragraphs each
    const all = readerSources(doc, { selection: { first: 6, last: 7, text: 'Paragraph 1' }, chapter: 1, paragraph: 8, sentence: 1 });
    assert.deepEqual(all.map((s) => s.id), ['sentence', 'paragraph', 'selection', 'chapter', 'document']);
    const [sentence, paragraph] = all;
    assert.deepEqual(sentence.input, { kind: 'document', document_id: 'alice', block: 8, sentence: 1 });
    assert.equal(sentence.label, 'This sentence');
    assert.equal(sentence.text, 'It has a second sentence.');
    assert.equal(sentence.chars, 25);
    assert.equal(sentence.chapters, 1);
    assert.deepEqual(paragraph.input, { kind: 'document', document_id: 'alice', blocks: [8, 8] });
    assert.equal(paragraph.label, 'This paragraph');
    assert.equal(paragraph.chars, Array.from(doc.blocks[8].text).length);
    assert.equal(paragraph.text, doc.blocks[8].text);
    const title = "Alice's Adventures in Wonderland";
    for (const s of all.slice(0, 4)) assert.equal(s.named, `The AI is told it's from “${title}”, chapter “Chapter 2”.`, s.id);
    assert.equal(all[4].named, `The AI is told it's from “${title}”, and the chapter of each part.`);
    assert.equal(defaultSource(all), 'selection', 'a selection the listener made is the one chosen');
    assert.equal(defaultSource(all.filter((s) => s.id !== 'selection')), 'sentence', 'something has been read: this sentence');
    assert.equal(defaultSource(readerSources(doc, { chapter: 1, paragraph: 8 })), 'chapter', 'nothing read yet: this chapter');

    // Nothing read (sentence -1), or a sentence the block doesn't have: no This sentence.
    assert.deepEqual(readerSources(doc, { chapter: 1, paragraph: 8 }).map((s) => s.id), ['paragraph', 'chapter', 'document']);
    assert.deepEqual(readerSources(doc, { chapter: 1, paragraph: 8, sentence: 3 }).map((s) => s.id), ['paragraph', 'chapter', 'document']);
    assert.deepEqual(readerSources(doc, { chapter: 1, paragraph: 99, sentence: 0 }).map((s) => s.id), ['chapter', 'document'], 'a block outside the document');
    // A block without sentence offsets is one sentence (as the server reads it); offsets count code points.
    const odd = makeDoc({ id: 'odd', chapters: 1, perChapter: 2, paragraph: () => '😀 Smile. Then more.' });
    odd.blocks[1].sentences = [[0, 8], [9, 19]];
    const [second] = readerSources(odd, { paragraph: 1, sentence: 1 });
    assert.deepEqual([second.text, second.chars], ['Then more.', 10]);
    const [first] = readerSources(odd, { paragraph: 1, sentence: 0 });
    assert.deepEqual([first.text, first.chars], ['😀 Smile.', 8]);
    odd.blocks[1].sentences = [];
    const whole = readerSources(odd, { paragraph: 1, sentence: 0 })[0];
    assert.deepEqual([whole.id, whole.text, whole.input.sentence], ['sentence', '😀 Smile. Then more.', 0]);
    // One chapter: the document's name only; a chapter named as the document is not named twice.
    const note = makeDoc({ id: 'note', title: 'Notes for Tuesday', chapters: 1, perChapter: 3 });
    assert.equal(readerSources(note, { paragraph: 1, sentence: 0 })[0].named, 'The AI is told it\'s from “Notes for Tuesday”.');
    assert.equal(defaultSource(readerSources(note, { paragraph: 1 })), 'document', 'one chapter: the whole document is this chapter');
    const titled = makeDoc({ id: 'rel', title: 'Release notes', chapters: 2, chapterTitles: ['Release notes', 'Older'] });
    assert.equal(readerSources(titled, { paragraph: 1, sentence: 0, chapter: 0 })[0].named, 'The AI is told it\'s from “Release notes”.');
  });

  it('nothing read yet: This paragraph (the highlighted one) is offered and This chapter is chosen; This paragraph goes as blocks [b, b]', async () => {
    await openDoc();   // progress at block 12: chapter 3, nothing played in this page
    await press();
    assert.deepEqual(sourceIds(), ['paragraph', 'chapter', 'document']);
    assert.equal(checked(), 'chapter');
    pick('paragraph');
    assert.match(textOf($('ask-sources')), /Paragraph 2 of chapter 3\. It has a second sentence/);
    assert.equal(textOf($('ask-named')), "The AI is told it's from “Alice's Adventures in Wonderland”, chapter “Chapter 3”.");
    ctx.env.change($('ask-preset'), 'explain');
    await sendIt();
    assert.deepEqual(ctx.state.asks[0], { input: { kind: 'document', document_id: 'alice', blocks: [12, 12] }, preset: 'explain' });
  });

  it('while playing, and once paused: This sentence is the one read, chosen first, sent by block and sentence with a question', async () => {
    await openDoc();
    ctx.engine.advance(13, 1);   // playback reached block 13's second sentence
    await settle(ctx.env);
    await press();
    assert.deepEqual(sourceIds(), ['sentence', 'paragraph', 'chapter', 'document']);
    assert.equal(checked(), 'sentence');
    assert.match(textOf($('ask-sources')), /This sentence/);
    assert.match(textOf($('ask-sources')), /It has a second sentence\./);
    assert.match(textOf($('ask-sources')), /25 characters/);
    assert.equal(textOf($('ask-named')), "The AI is told it's from “Alice's Adventures in Wonderland”, chapter “Chapter 3”.");
    ctx.env.click([...$('ask-picks').querySelectorAll('button')].find((b) => textOf(b) === 'Explain this'));
    await sendIt();
    assert.deepEqual(ctx.state.asks[0], { input: { kind: 'document', document_id: 'alice', block: 13, sentence: 1 }, preset: 'book', question: 'Explain this', output: { document: {} } });
    ctx.env.click($('ask-cancel'));
    await settle(ctx.env);
    // Paused (the engine keeps its cursor): the last sentence read is still This sentence.
    ctx.engine.emit('state', { ...ctx.engine.state, playing: false });
    await settle(ctx.env);
    await press();
    assert.equal(checked(), 'sentence');
    // A jump to another paragraph (nothing read in it yet): This paragraph is that one, and This chapter is chosen.
    ctx.engine.skipBlock(-1);
    await settle(ctx.env);
    ctx.env.click($('ask-cancel'));
    await settle(ctx.env);
    await press();
    assert.deepEqual(sourceIds(), ['paragraph', 'chapter', 'document']);
    pick('paragraph');
    await sendIt();
    assert.deepEqual(ctx.state.asks[1].input, { kind: 'document', document_id: 'alice', blocks: [12, 12] });
  });

  it('a selection made while a sentence was read: every source, in order, the selection chosen', async () => {
    await openDoc();
    ctx.engine.advance(12, 0);
    await settle(ctx.env);
    selectBlocks(11, 12);
    await press();
    assert.deepEqual(sourceIds(), ['sentence', 'paragraph', 'selection', 'chapter', 'document']);
    assert.equal(checked(), 'selection');
  });

  it('the screenshot states: each source chosen in turn, and a question in the box; nothing is sent', async () => {
    await openDoc();
    for (const [shot, id] of [['ask-ai-sentence', 'sentence'], ['ask-ai-paragraph', 'paragraph'], ['ask-ai-chapter', 'chapter'], ['ask-ai-document', 'document'], ['ask-ai', 'selection']]) {
      await go(ctx, `reader/alice?shot=${shot}`, 5);
      await settle(ctx.env);
      assert.equal($('app-ask-ai').open, true, shot);
      assert.equal(checked(), id, shot);
    }
    await go(ctx, 'reader/alice?shot=ask-ai-question', 5);
    await settle(ctx.env);
    assert.equal(checked(), 'sentence');
    assert.equal($('ask-question').value, 'Explain this to me.');
    assert.equal(ctx.state.asks.length, 0);
  });
});
