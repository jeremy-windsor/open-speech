/**
 * The Reader's "Ask AI" sources: what the shared sheet (components/ask-ai.js) can
 * send from the open document, as POST /api/assistant/ask `document` inputs, in this order:
 * - this sentence, the one being read or the last one read (`block` and `sentence`, the index into the block's
 *   sentence offsets);
 * - this paragraph, the highlighted one (`blocks: [b, b]`);
 * - the selection (the whole paragraphs it touches, `block_start` and `block_end`);
 * - the current chapter (`chapters: [i, i]`);
 * - the whole document (just its id).
 * The document's text stays on the server; only its id and the range go. Every source says what goes with it: the
 * server names the document and its chapter before the text, and `named` says so in the sheet.
 * Pure, except selectionRange(), which reads a Selection object it is given.
 * @module views/reader/ask-sources
 */
import { charCount } from '../../components/ask-ai-logic.js';
import { chapterMap, codePointSlicer } from './text-logic.js';

/** How much of a part's text the summary needs (the sheet shows its first characters). */
const SUMMARY_SOURCE_CHARS = 400;

function blockIndex(node, container) {
  let el = node && node.nodeType === 3 ? node.parentElement : node;
  if (!el || typeof el.closest !== 'function') return null;
  el = el.closest('[data-block]');
  if (!el || (container && typeof container.contains === 'function' && !container.contains(el))) return null;
  const index = Number(el.getAttribute('data-block'));
  return Number.isInteger(index) && index >= 0 ? index : null;
}

/**
 * The blocks a text selection touches inside the document text, or null (none, collapsed, or outside it).
 * @param {Selection|null} selection
 * @param {HTMLElement} container #rd-text
 * @returns {{first: number, last: number, text: string}|null}
 */
export function selectionRange(selection, container) {
  try {
    if (!selection || selection.isCollapsed || !selection.rangeCount) return null;
    const text = String(selection);
    if (!text.trim()) return null;
    const a = blockIndex(selection.anchorNode, container);
    const b = blockIndex(selection.focusNode, container);
    if (a === null && b === null) return null;
    const first = Math.min(a ?? b, b ?? a);
    const last = Math.max(a ?? b, b ?? a);
    return { first, last, text };
  } catch {
    return null;
  }
}

/** PURE: the text and size of blocks [first, last] (headings count: the server sends whole blocks). */
function partOf(blocks, first, last) {
  let chars = 0;
  let text = '';
  for (let i = first; i <= last && i < blocks.length; i += 1) {
    const t = String((blocks[i] && blocks[i].text) || '');
    chars += charCount(t);
    if (text.length < SUMMARY_SOURCE_CHARS && t) text = text ? `${text} ${t}` : t;
  }
  return { chars, text };
}

/** PURE: how many chapters blocks [first, last] reach into. */
function chaptersIn(map, first, last) {
  if (!map.length) return 1;
  const seen = new Set();
  for (let i = first; i <= last && i < map.length; i += 1) seen.add(map[i]);
  return Math.max(1, seen.size);
}

/**
 * PURE: the sentence `s` of a block as the server counts it: its `[start, end]` code-point offsets
 * in `block.sentences`; a block without offsets is one sentence (index 0). null when the block has no such one.
 */
function sentenceOf(block, s) {
  const text = String((block && block.text) || '');
  const spans = block && Array.isArray(block.sentences) && block.sentences.length ? block.sentences : null;
  if (!Number.isInteger(s) || s < 0) return null;
  if (!spans) return s === 0 && text.trim() ? text : null;
  const span = spans[s];
  if (!Array.isArray(span)) return null;
  const start = Math.max(0, Number(span[0]) || 0);
  const end = Math.max(start, Number(span[1]) || 0);
  const piece = codePointSlicer(text)(start, end);
  return piece.trim() ? piece : null;
}

/**
 * PURE: the line saying what goes with a part, as the server builds it: the document's
 * title, and the chapter's when the document has more than one and its title isn't the document's. `position` is
 * the part's one chapter, or -1 for a part over several (each piece names its own).
 */
function namedLine(title, chapters, position) {
  if (position < 0 && chapters.length > 1) return `The AI is told it's from “${title}”, and the chapter of each part.`;
  const c = chapters.length > 1 ? chapters[position] : null;
  const name = c ? String(c.title || '') : '';
  return name && name !== title ? `The AI is told it's from “${title}”, chapter “${name}”.` : `The AI is told it's from “${title}”.`;
}

/**
 * PURE: the source the sheet starts with: the selection the listener just made, else this sentence
 * while something has been read, else this chapter, else the whole document (one chapter: it is this chapter).
 * @param {{id: string}[]} sources readerSources()
 * @returns {string} '' when there is none
 */
export function defaultSource(sources) {
  const ids = (Array.isArray(sources) ? sources : []).map((s) => s.id);
  return ['selection', 'sentence', 'chapter', 'document'].find((id) => ids.includes(id)) || ids[0] || '';
}

/**
 * PURE: the sources for the open document, in the order the sheet lists them: this sentence (when one has been
 * read in the highlighted paragraph), this paragraph (the highlighted one), the selection (when there is one), the
 * current chapter (when the document has more than one), the whole document.
 * @param {{id: string, title?: string, blocks: object[], chapters: object[]}} doc
 * @param {{selection?: {first: number, last: number, text?: string}|null, chapter?: number, paragraph?: number,
 *   sentence?: number}} [options] paragraph: the highlighted block (the engine cursor's `block`), -1 for none;
 *   sentence: the one read in it (the cursor's `sentence`), -1 while none has been
 * @returns {object[]} ask-ai-logic sources
 */
export function readerSources(doc, { selection = null, chapter = -1, paragraph = -1, sentence = -1 } = {}) {
  if (!doc || !doc.id) return [];
  const blocks = Array.isArray(doc.blocks) ? doc.blocks : [];
  const chapters = Array.isArray(doc.chapters) ? doc.chapters : [];
  if (!blocks.length) return [];
  const map = chapterMap(blocks.length, chapters);
  const title = String(doc.title || 'Untitled');
  const named = (first, last) => namedLine(title, chapters, chaptersIn(map, first, last) > 1 ? -1 : (map[first] ?? 0));
  const out = [];
  if (Number.isInteger(paragraph) && paragraph >= 0 && paragraph < blocks.length) {
    const said = sentenceOf(blocks[paragraph], sentence);
    if (said !== null) {
      out.push({
        id: 'sentence',
        label: 'This sentence',
        input: { kind: 'document', document_id: doc.id, block: paragraph, sentence },
        text: said, chars: charCount(said), chapters: 1, title, named: named(paragraph, paragraph),
      });
    }
    const part = partOf(blocks, paragraph, paragraph);
    out.push({
      id: 'paragraph',
      label: 'This paragraph',
      input: { kind: 'document', document_id: doc.id, blocks: [paragraph, paragraph] },
      text: part.text, chars: part.chars, chapters: 1, title, named: named(paragraph, paragraph),
    });
  }
  if (selection && Number.isInteger(selection.first) && Number.isInteger(selection.last)) {
    const first = Math.max(0, Math.min(blocks.length - 1, selection.first));
    const last = Math.max(first, Math.min(blocks.length - 1, selection.last));
    const part = partOf(blocks, first, last);
    const n = last - first + 1;
    out.push({
      id: 'selection',
      label: n === 1 ? 'The selected paragraph' : `The ${n} selected paragraphs`,
      input: { kind: 'document', document_id: doc.id, block_start: first, block_end: last },
      text: part.text, detail: selection.text ? String(selection.text) : part.text, chars: part.chars,
      chapters: chaptersIn(map, first, last), title, named: named(first, last),
    });
  }
  if (chapters.length > 1 && chapter >= 0 && chapter < chapters.length) {
    const c = chapters[chapter];
    const first = Math.max(0, Number(c.first_block) || 0);
    const last = Math.min(blocks.length - 1, first + Math.max(1, Number(c.block_count) || 1) - 1);
    const part = partOf(blocks, first, last);
    const name = String(c.title || '').trim();
    out.push({
      id: 'chapter',
      label: name ? `This chapter: ${name}` : 'This chapter',
      input: { kind: 'document', document_id: doc.id, chapters: [chapter, chapter] },
      text: part.text, chars: part.chars, chapters: 1, title, named: namedLine(title, chapters, chapter),
    });
  }
  const whole = partOf(blocks, 0, blocks.length - 1);
  out.push({
    id: 'document',
    label: `The whole document: ${title}`,
    input: { kind: 'document', document_id: doc.id },
    text: whole.text, chars: Number(doc.char_count) || whole.chars, chapters: Math.max(1, chapters.length), title,
    named: namedLine(title, chapters, chapters.length > 1 ? -1 : 0),
  });
  return out;
}
