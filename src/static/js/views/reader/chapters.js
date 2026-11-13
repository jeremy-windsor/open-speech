/**
 * The Reader's chapter lists: the aside list `#rd-chapter-list` (≥1180 px) and the sheet
 * `#rd-chapters-sheet` (narrower screens, opened from the player's Chapters chip; the top bar's `#rd-chapters-btn`
 * stands in when there is no player).
 *
 * Both lists show every chapter with its number, unless its title already carries one ("CHAPTER I. Down the
 * Rabbit-Hole" is not shown as "1  CHAPTER I. …"; in a mixed list the number's column stays, so titles line up, and
 * when every title is numbered there is no column). The current one is
 * `.is-current` with
 * `aria-current="true"`; earlier ones are `.is-done` and keep full contrast (no opacity dimming, AX-07).
 * Picking a chapter plays from its first block and closes the sheet; opening the sheet scrolls to the current
 * chapter and focuses it (RE-57).
 * @module views/reader/chapters
 */
import { html, render, setHidden } from '../../core/dom.js';
import { openSheet, closeSheet } from '../../core/dialogs.js';

/** A chapter's visible title ("Chapter 3" when the source gave none). */
export function chapterTitle(chapter, position) {
  const title = chapter && String(chapter.title || '').trim();
  return title || `Chapter ${position + 1}`;
}

// "CHAPTER I. Down the Rabbit-Hole", "Chapter 12", "Chapter Twenty-One", "Ch. 3", "Part II", "Book 1",
// "1. Loomings", "IV – The Pool". A roman numeral must be a well-formed one, and a bare one at the start of a title
// must be upper case, so "Down…", "Ivan" or "Mix: …" don't count.
const roman = (m, d, c, l, x, v, i) => `(?=[${m}${d}${c}${l}${x}${v}${i}])${m}{0,4}(?:${c}${m}|${c}${d}|${d}?${c}{0,3})(?:${x}${c}|${x}${l}|${l}?${x}{0,3})(?:${i}${x}|${i}${v}|${v}?${i}{0,3})`;
const WORDS = 'one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|first|second|third|last|final';
const LABELLED = new RegExp(`^(?:chapter|ch\\.?|part|book|section)\\s+(?:\\d+|${roman('m', 'd', 'c', 'l', 'x', 'v', 'i')}|(?:${WORDS})(?:-[a-z]+)?)\\b`, 'i');
const BARE = new RegExp(`^(?:\\d+|${roman('M', 'D', 'C', 'L', 'X', 'V', 'I')})(?:[.:)]|\\s+[-–—])`);

/** True when a title already carries its own number, so the list's ordinal would only repeat it. */
export function hasOwnNumber(title) {
  const text = String(title || '').trim();
  return LABELLED.test(text) || BARE.test(text);
}

function listMarkup(chapters) {
  return chapters.map((chapter, i) => {
    const title = chapterTitle(chapter, i);
    const n = hasOwnNumber(title) ? '' : String(i + 1);
    return html`<li class="rd-ch" data-chapter-item="${i}"><button type="button" class="rd-ch__btn" data-chapter="${i}"><span class="rd-ch__n tabular" aria-hidden="true">${n}</span><span class="rd-ch__t">${title}</span></button></li>`;
  });
}

/**
 * Bind the chapter lists inside the Reader's root.
 * @param {HTMLElement} root
 * @param {{onPick: (position: number, chapter: object) => void}} options
 */
export function createChapters(root, { onPick = () => {} } = {}) {
  const side = root.querySelector('#rd-chapter-list');
  const sideGroup = root.querySelector('#rd-side-chapters');
  const sheet = root.querySelector('#rd-chapters-sheet');
  const sheetList = root.querySelector('#rd-chapters-sheet-list');
  const button = root.querySelector('#rd-chapters-btn');
  const st = { chapters: [], current: -1 };

  function lists() {
    return [side, sheetList].filter(Boolean);
  }

  function mark() {
    for (const list of lists()) {
      for (const item of list.querySelectorAll('[data-chapter-item]')) {
        const index = Number(item.getAttribute('data-chapter-item'));
        item.classList.toggle('is-current', index === st.current);
        item.classList.toggle('is-done', index < st.current);
        const btn = item.querySelector('button');
        if (!btn) continue;
        if (index === st.current) btn.setAttribute('aria-current', 'true');
        else btn.removeAttribute('aria-current');
      }
    }
  }

  function onClick(event) {
    const start = event.target && event.target.nodeType === 3 ? event.target.parentElement : event.target;
    const btn = start && start.closest ? start.closest('[data-chapter]') : null;
    if (!btn) return;
    const position = Number(btn.getAttribute('data-chapter'));
    const chapter = st.chapters[position];
    if (!chapter) return;
    if (sheet && sheet.open) closeSheet(sheet, 'pick');
    onPick(position, chapter);
  }
  for (const list of lists()) list.addEventListener('click', onClick);

  return {
    /** Show a document's chapters (a document with one chapter or none has no list and no button). */
    setDoc(doc) {
      st.chapters = Array.isArray(doc && doc.chapters) ? doc.chapters : [];
      st.current = -1;
      const markup = listMarkup(st.chapters);
      // Every title numbered by its source (most books): no number column at all, so the titles start flush.
      const ownNumbers = st.chapters.length > 0 && st.chapters.every((chapter, i) => hasOwnNumber(chapterTitle(chapter, i)));
      for (const list of lists()) {
        render(list, markup);
        list.classList.toggle('rd-chapters--titled', ownNumbers);
      }
      const many = st.chapters.length > 1;
      setHidden(sideGroup, !many);
      setHidden(button, !many);
      if (!many && sheet && sheet.open) closeSheet(sheet);
    },
    /** Mark the chapter being read (position in doc.chapters). */
    setCurrent(position) {
      if (position === st.current) return;
      st.current = position;
      mark();
    },
    get current() { return st.current; },
    get count() { return st.chapters.length; },
    /** Open the chapters sheet on the current chapter (focused and scrolled into view). */
    open(trigger) {
      if (!sheet || !st.chapters.length) return;
      const current = sheetList && sheetList.querySelector(`[data-chapter="${Math.max(0, st.current)}"]`);
      openSheet(sheet, { trigger: trigger || document.activeElement, initialFocus: current || undefined });
      if (current && typeof current.scrollIntoView === 'function') current.scrollIntoView({ block: 'center' });
    },
    close() {
      if (sheet && sheet.open) closeSheet(sheet);
    },
    destroy() {
      for (const list of lists()) list.removeEventListener('click', onClick);
    },
  };
}
