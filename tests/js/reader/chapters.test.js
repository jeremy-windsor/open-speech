// The chapter lists: the aside (≥1180 px) and the sheet.
import { describe, it, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { readerEnv, makeDoc, progressAt, $, textOf, settle } from './helpers.js';
import { assertAccessibleNames } from '../harness/a11y.js';

let ctx;
afterEach(() => ctx?.env.restore());

const items = (listId) => Array.from(document.querySelectorAll(`#${listId} .rd-ch`));
const titles = ['Down the Rabbit-Hole', 'The Pool of Tears', 'A Caucus-Race and a Long Tale', 'The Rabbit Sends in a Little Bill'];

describe('views/reader/chapters', () => {
  it('[RE-57] both lists mark the current chapter (is-current, aria-current) and earlier ones as done, following playback', async () => {
    ctx = await readerEnv({ hash: '#reader/alice', docs: [makeDoc({ chapters: 4, chapterTitles: titles, progress: progressAt(12) })] });
    for (const list of ['rd-chapter-list', 'rd-chapters-sheet-list']) {
      const li = items(list);
      assert.deepEqual(li.map((el) => textOf(el.querySelector('.rd-ch__t'))), titles);
      assert.deepEqual(li.map((el) => textOf(el.querySelector('.rd-ch__n'))), ['1', '2', '3', '4']);
      assert.equal(li[2].classList.contains('is-current'), true, `${list}: block 12 is in chapter 3`);
      assert.equal(li[2].querySelector('button').getAttribute('aria-current'), 'true');
      assert.deepEqual(li.map((el) => el.classList.contains('is-done')), [true, true, false, false]);
      assert.equal(li[3].querySelector('button').hasAttribute('aria-current'), false);
    }
    ctx.engine.advance(16);
    const li = items('rd-chapter-list');
    assert.equal(li[3].classList.contains('is-current'), true);
    assert.deepEqual(li.map((el) => el.classList.contains('is-done')), [true, true, true, false]);
    assert.equal(document.querySelectorAll('[data-chapter][aria-current="true"]').length, 2, 'one current item per list');
    assertAccessibleNames($('rd-side'));
  });

  it('[RE-57] opening the sheet focuses and scrolls to the current chapter; picking one plays from its first block and closes it', async () => {
    ctx = await readerEnv({ hash: '#reader/alice', docs: [makeDoc({ chapters: 4, chapterTitles: titles, progress: progressAt(12) })] });
    const button = $('rd-chapters-btn');
    assert.equal(button.hidden, false);
    button.focus();
    ctx.env.click(button);
    await settle(ctx.env);
    const sheet = $('rd-chapters-sheet');
    assert.equal(sheet.open, true);
    const currentButton = items('rd-chapters-sheet-list')[2].querySelector('button');
    assert.equal(document.activeElement, currentButton);
    assert.ok(currentButton.__scrolledIntoView.length > 0, 'scrolled into view');
    ctx.env.click(items('rd-chapters-sheet-list')[1].querySelector('button'));
    await settle(ctx.env);
    assert.equal(sheet.open, false);
    assert.deepEqual(ctx.engine.calls.at(-1), ['playFrom', 5], 'chapter 2 starts at block 5');
    assert.equal(document.activeElement, button, 'focus returns to the Chapters button');
    ctx.env.click(items('rd-chapter-list')[3].querySelector('button'));
    assert.deepEqual(ctx.engine.calls.at(-1), ['playFrom', 15], 'the aside list plays too');
  });

  it('[RE-57] a title that carries its own number gets no second one from the list (the column stays, so titles line up)', async () => {
    const own = ['CHAPTER I. Down the Rabbit-Hole', 'Chapter 2', 'The Pool of Tears', 'IV – The Rabbit Sends in a Little Bill'];
    ctx = await readerEnv({ hash: '#reader/alice', docs: [makeDoc({ chapters: 4, chapterTitles: own, progress: progressAt(12) })] });
    for (const list of ['rd-chapter-list', 'rd-chapters-sheet-list']) {
      const li = items(list);
      assert.deepEqual(li.map((el) => textOf(el.querySelector('.rd-ch__t'))), own);
      assert.deepEqual(li.map((el) => textOf(el.querySelector('.rd-ch__n'))), ['', '', '3', ''], list);
      assert.equal(li.every((el) => el.querySelector('.rd-ch__n')), true, 'every row keeps the number column');
      assert.equal($(list).classList.contains('rd-chapters--titled'), false, 'a mixed list keeps the column');
    }
    const { hasOwnNumber } = await import('../../../src/static/js/views/reader/chapters.js');
    for (const title of ['Chapter Twenty-One', 'Ch. 3', 'PART II', 'Book 1: Arrival', '1. Loomings', 'XII) The End', 'Section 4']) {
      assert.equal(hasOwnNumber(title), true, title);
    }
    ctx.env.restore();
    // Every title numbered by the book (the seeded Alice): no number column at all.
    const all = ['CHAPTER I. Down the Rabbit-Hole', 'CHAPTER II. The Pool of Tears', 'CHAPTER III. A Caucus-Race and a Long Tale'];
    ctx = await readerEnv({ hash: '#reader/alice', docs: [makeDoc({ chapters: 3, chapterTitles: all, progress: progressAt(6) })] });
    for (const list of ['rd-chapter-list', 'rd-chapters-sheet-list']) {
      assert.equal($(list).classList.contains('rd-chapters--titled'), true, list);
      assert.deepEqual(items(list).map((el) => textOf(el.querySelector('.rd-ch__t'))), all);
    }
    for (const title of ['Down the Rabbit-Hole', 'Mix: a recipe', 'Chapterhouse', 'Part of the Plan', 'Ivan', '1984', 'A Caucus-Race']) {
      assert.equal(hasOwnNumber(title), false, title);
    }
  });

  it('[RE-57] a document with one chapter has no chapter list and no Chapters button', async () => {
    ctx = await readerEnv({ hash: '#reader/note' });
    assert.equal($('rd-chapters-btn').hidden, true);
    assert.equal($('rd-side-chapters').hidden, true);
  });

  it('the player opens the sheet through mountPlayer\'s onOpenChapters', async () => {
    ctx = await readerEnv({ hash: '#reader/alice' });
    ctx.mounted[0].options.onOpenChapters($('rd-player-slot'));
    await settle(ctx.env);
    assert.equal($('rd-chapters-sheet').open, true);
  });
});
