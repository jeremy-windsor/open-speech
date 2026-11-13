"""Book jobs: send a Reader document to the AI in pieces, and save its answers as a new Reader document.

A job takes a stored document (all of it, some of its chapters, or some of its blocks: any list of blocks) and an AI
preset (``book`` by default) with optional overrides. The book jobs' routes start one for a document or its chapters;
``POST /api/assistant/ask`` starts one for a long input (a document's blocks, or a long text it keeps as a Reader
document first) through the same runner, ``get_runner()``, so every job shares its one slot, its progress, cancel and
retry, and the Reader's job routes. It works out the settings once, with
``src.assistant.presets.effective(..., purpose="book")``, and keeps them in the job, so a later change to the preset
doesn't change it. Then it cuts each chapter into pieces of at most ``chunk_chars`` at block boundaries (a block longer
than that at sentences) and sends each piece on its own, with no history: the system prompt, then the user template
filled with ``{text}``, ``{title}``, ``{author}`` and ``{chapter}``. A chapter's opening headings aren't sent: the
result keeps them as they are.

A long text or transcript that ``POST /api/assistant/ask`` keeps as a Reader document is different: the document is for
listening (the Reader's Markdown import, as pasting it gives), so the job keeps the text too and sends that, as it was
sent: ``cut_text`` cuts it at blank lines (a paragraph longer than ``chunk_chars`` at line breaks, then at sentences)
and parses nothing, and a piece fills only ``{text}``, as a text of one piece does.

A job of ``POST /api/assistant/ask`` also keeps what goes before each piece's rendered template: the listener's
question, if any, in every piece, and, for a document's part, the line naming the document and the piece's chapter
(``prompts.source_line``; the chapter only when the document has more than one and it isn't the document's title, and no
line when the template uses ``{title}`` or ``{chapter}``). A piece never spans two chapters, so each names its own. The
Reader's own book jobs send neither. ``plan_sentence`` is one sentence of a block, as the Reader counts them, for an ask
answered at once.

Each answer is saved as soon as it comes back, keyed by a hash of the settings that reach the AI and the piece's
blocks and text. So a restart loses no finished piece, and starting the same job again (the same document and the
same settings that reach the AI) sends only what's missing. When every piece is done, the answers become a new
Reader document: one chapter per chapter sent, its opening headings and then the answers in paragraphs (split at
blank lines). An error is retried once; a second one stops the job with its code, keeping the finished pieces.

Jobs run one at a time, in one asyncio task on the app's event loop, started with the first job; later jobs wait
as ``queued``. The task stops at shutdown through ``src.assistant.on_close``; a job it stops, or one a crash left
running, fails with the code ``interrupted`` (``src.storage.init_db`` handles the crash case at startup).

The AI is reached only through ``src.assistant``: ``get_client()``, ``presets.effective()`` and
``prompts.messages_for()``. Nothing server-only (the URL, the key) is kept in a job.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import threading
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from src import assistant
from src.assistant import prompts
from src.assistant.errors import ASSISTANT_EMPTY_REPLY, AssistantError
from src.live_tts.segmenter import split_sentences
from src.reader.document import HEADING, PARAGRAPH, Block, Chapter, Document, finalize
from src.reader.store import _lock as _reader_lock
from src.reader.store import AI_RESULT_SOURCE_TYPE, _writing, get_db
from src.storage import INTERRUPTED_MESSAGE

logger = logging.getLogger(__name__)

ACTIVE = ("queued", "running")  # the statuses besides done, failed and cancelled
INTERRUPTED = "interrupted"  # stopped by a shutdown or a crash
INTERNAL_ERROR = "internal_error"  # a fault of this server, logged
NOT_STARTED = "assistant_not_started"  # the shared client is gone (shutting down)
SOURCE_TYPE = AI_RESULT_SOURCE_TYPE  # a result document's source_type (its title is part of its content hash)
RETRY_DELAY_S = 1.0
CANCEL_WAIT_S = 10.0
STOP_WAIT_S = 10.0
TITLE_MAX_CHARS = 200
JOIN = "\n\n"
# The settings that reach the AI: a change to one of them sends every piece again. (The server-only
# OS_ASSISTANT_EXTRA_BODY reaches it too, but isn't part of a job.)
AI_FIELDS = ("model", "system_prompt", "user_template", "thinking", "reasoning_effort", "max_tokens", "temperature", "top_p")
# Fields added after jobs were first saved: in a piece's key only when set, so a piece saved before them keeps its key
# and its answer.
AI_FIELDS_WHEN_SET = ("thinking_budget",)
_PLAN_CACHE = 32
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
# A text sent as it is (cut_text): a line break, then one or more lines of nothing but spaces or tabs. A "\r\n" is one
# line break, never a "\r" and then a "\n".
_NEWLINE = r"(?:\r\n|\r(?!\n)|\n)"
_BLANK_LINES = re.compile(rf"{_NEWLINE}(?:[^\S\r\n]*{_NEWLINE})+")
_LINE_BREAK = re.compile(_NEWLINE)
# The key of that text in a job's settings_json; never shown in the job's settings.
TEXT_FIELD = "text_as_sent"
# An ask job's question and whether it names its document, kept the same way.
QUESTION_FIELD = "question_as_sent"
NAMED_FIELD = "names_its_document"
_SENT_FIELDS = (TEXT_FIELD, QUESTION_FIELD, NAMED_FIELD)  # what a job sends, not its settings


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Cutting a chapter into pieces ────────────────────────────────────────────


@dataclass(frozen=True)
class Piece:
    """Text sent to the AI in one request: whole blocks ``first_block``..``last_block`` of one chapter, or part of
    one block longer than ``chunk_chars``."""

    chapter_index: int
    first_block: int
    last_block: int
    text: str


def _split_block(text: str, chunk_chars: int) -> list[str]:
    """A block longer than ``chunk_chars`` cut into parts of at most that, at sentence ends where it can."""
    parts: list[str] = []
    start = end = None
    for sentence_start, sentence_end in split_sentences(text, chunk_chars):
        if start is None:
            start = sentence_start
        elif sentence_end - start > chunk_chars:
            parts.append(text[start:end].strip())
            start = sentence_start
        end = sentence_end
    if start is not None and text[start:end].strip():
        parts.append(text[start:end].strip())
    return parts


def cut_chapter(chapter_index: int, blocks: Sequence[Mapping[str, Any]], chunk_chars: int) -> tuple[list[str], list[Piece]]:
    """(the chapter's opening headings, kept as they are, and its pieces for the AI).

    ``blocks`` are the chapter's blocks as ``ReaderStore.get_blocks`` gives them (``index``, ``kind``, ``text``).
    A piece is a run of whole blocks joined by blank lines, at most ``chunk_chars`` long; a block longer than that
    is cut with ``split_sentences``. A heading after the chapter's first text is sent with the text around it.
    """
    position = 0
    headings: list[str] = []
    while position < len(blocks) and blocks[position]["kind"] == HEADING:
        headings.append(blocks[position]["text"])
        position += 1
    pieces: list[Piece] = []
    run: list[Mapping[str, Any]] = []
    size = 0

    def flush() -> None:
        nonlocal run, size
        if run:
            pieces.append(Piece(chapter_index, run[0]["index"], run[-1]["index"], JOIN.join(b["text"] for b in run)))
        run, size = [], 0

    for block in blocks[position:]:
        text = block["text"]
        if len(text) > chunk_chars:
            flush()
            pieces.extend(Piece(chapter_index, block["index"], block["index"], part) for part in _split_block(text, chunk_chars))
            continue
        if run and size + len(JOIN) + len(text) > chunk_chars:
            flush()
        size += (len(JOIN) if run else 0) + len(text)
        run.append(block)
    flush()
    return headings, pieces


def _stretches(text: str, start: int, end: int, separator: re.Pattern[str]) -> list[tuple[int, int]]:
    """The ``(start, end)`` of each stretch of ``text[start:end]`` between two ``separator`` matches that has more
    than whitespace."""
    found = []
    position = start
    for match in separator.finditer(text, start, end):
        if text[position:match.start()].strip():
            found.append((position, match.start()))
        position = match.end()
    if text[position:end].strip():
        found.append((position, end))
    return found


def cut_text(text: str, chunk_chars: int) -> list[str]:
    """A text cut into pieces of at most ``chunk_chars`` exactly as it was sent.

    Each piece is a stretch of ``text`` of whole paragraphs (cut at blank lines); a paragraph longer than
    ``chunk_chars`` is cut at its line breaks, and a line longer than that at sentence ends (``split_sentences``).
    Nothing is parsed or cleaned: Markdown, code, markup and line breaks stay as they are, and only the whitespace
    between two pieces is left out.
    """
    pieces: list[str] = []

    def gather(spans: list[tuple[int, int]], deeper: Callable[[int, int], None] | None) -> None:
        run: tuple[int, int] | None = None
        for start, end in spans:
            if run is not None and end - run[0] > chunk_chars:
                pieces.append(text[run[0]:run[1]])
                run = None
            if end - start > chunk_chars and deeper is not None:
                deeper(start, end)
                continue
            run = (start, end) if run is None else (run[0], end)
        if run is not None:
            pieces.append(text[run[0]:run[1]])

    def sentences(start: int, end: int) -> None:
        gather([(start + first, start + last) for first, last in split_sentences(text[start:end], chunk_chars)], None)

    def lines(start: int, end: int) -> None:
        gather(_stretches(text, start, end, _LINE_BREAK), sentences)

    gather(_stretches(text, 0, len(text), _BLANK_LINES), lines)
    return pieces


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def piece_key(document_id: str, piece: Piece, settings: Mapping[str, Any], lead: str = "") -> str:
    """The key a piece's answer is saved under: the document, the piece, the settings that reach the AI and what
    goes before its template (``prompts.lead``: a question and the line naming the document; a piece
    sent without them keeps the key it always had)."""
    payload = {
        "document": document_id,
        "chapter": piece.chapter_index,
        "blocks": [piece.first_block, piece.last_block],
        "text": _digest(piece.text),
        "ai": {name: settings.get(name) for name in AI_FIELDS},
    }
    payload["ai"].update({name: settings.get(name) for name in AI_FIELDS_WHEN_SET if settings.get(name)})
    if lead:
        payload["lead"] = _digest(lead)
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8", "surrogatepass")).hexdigest()


@dataclass(frozen=True)
class _ChapterPlan:
    index: int
    title: str
    headings: tuple[str, ...]
    pieces: tuple[Piece, ...]
    keys: tuple[str, ...]
    source: str = ""  # the line naming the document and this chapter, before each piece, or ""


@dataclass(frozen=True)
class _Plan:
    document_id: str
    title: str
    author: str | None
    chapters: tuple[_ChapterPlan, ...]
    as_sent: bool = False  # a text sent as it is (plan_text): its pieces fill only {text}
    question: str | None = None  # the listener's question, before every piece

    @property
    def keys(self) -> list[str]:
        return [key for chapter in self.chapters for key in chapter.keys]


class ChaptersError(ValueError):
    """A job's ``chapters`` isn't ``[first, last]`` within the document."""


class BlocksError(ValueError):
    """A job's ``blocks`` isn't ``[first, last]`` within the document."""


class SentenceError(ValueError):
    """A ``sentence`` isn't one of its block's sentences."""


def _pair(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple)) and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    )


def chapter_range(chapters: Any, chapter_count: int) -> tuple[int, int]:
    """``[first, last]`` (Reader chapter indexes, from 0) checked against the document, or all for None.
    Raises ChaptersError."""
    if chapters is None:
        return 0, chapter_count - 1
    if not _pair(chapters) or not 0 <= chapters[0] <= chapters[1] < chapter_count:
        raise ChaptersError(f"chapters must be [first, last] with 0 <= first <= last <= {chapter_count - 1}, or null for all")
    return chapters[0], chapters[1]


def block_range(blocks: Any, block_count: int) -> tuple[int, int]:
    """``[first, last]`` (Reader block indexes, from 0, as the document's blocks number them) checked against the
    document. Raises BlocksError."""
    if not _pair(blocks) or not 0 <= blocks[0] <= blocks[1] < block_count:
        raise BlocksError(f"blocks must be [first, last] with 0 <= first <= last <= {block_count - 1}")
    return blocks[0], blocks[1]


def block_count(document: Mapping[str, Any]) -> int:
    """The number of blocks of a document as ``ReaderStore.get_document`` gives it."""
    return sum(chapter["block_count"] for chapter in document["chapters"])


def _source_line(settings: Mapping[str, Any], document: Mapping[str, Any], chapter: Mapping[str, Any]) -> str:
    """The line naming the document and ``chapter``: the chapter only when the document has more than one and it isn't
    the document's title; "" when the template names them itself."""
    title = document["title"]
    named = chapter["title"] if len(document["chapters"]) > 1 and chapter["title"] != title else None
    return prompts.source_line(settings["user_template"], title, named)


def plan_document(store, document_id: str, settings: Mapping[str, Any], *, chapters: tuple[int, int] | None = None,
                  blocks: tuple[int, int] | None = None, document: Mapping[str, Any] | None = None,
                  question: str | None = None, named: bool = False) -> _Plan | None:
    """The pieces a document's part goes to the AI in: all of it, ``chapters`` (``(first, last)``) or ``blocks``
    (``(first, last)``, which may start or end inside a chapter: only those blocks are sent, and a chapter's
    opening headings are those at the start of its part). ``settings`` is ``EffectiveSettings.to_dict()``; its
    ``chunk_chars`` sizes the pieces. ``question`` goes before every piece and, with ``named``, the line naming the
    document and the piece's chapter. None when the document doesn't exist. The ranges are not checked here."""
    if document is None:
        document = store.get_document(document_id, include_blocks=False)
    if document is None:
        return None
    every = document["chapters"]
    if blocks is not None:
        first, last = blocks
        chosen = [c for c in every if c["first_block"] <= last and c["first_block"] + c["block_count"] - 1 >= first]
    else:
        first_chapter, last_chapter = chapters if chapters is not None else (0, len(every) - 1)
        chosen = [chapter for chapter in every if first_chapter <= chapter["index"] <= last_chapter]
        if chosen:
            first = chosen[0]["first_block"]
            last = chosen[-1]["first_block"] + chosen[-1]["block_count"] - 1
    found = store.get_blocks(document_id, first, last) if chosen else []
    by_chapter: dict[int, list[dict]] = {}
    for block in found:
        by_chapter.setdefault(block["chapter_index"], []).append(block)
    planned = []
    for chapter in chosen:
        headings, pieces = cut_chapter(chapter["index"], by_chapter.get(chapter["index"], []), settings["chunk_chars"])
        source = _source_line(settings, document, chapter) if named else ""
        lead = prompts.lead(question, source)
        keys = tuple(piece_key(document_id, piece, settings, lead) for piece in pieces)
        planned.append(_ChapterPlan(chapter["index"], chapter["title"], tuple(headings), tuple(pieces), keys, source))
    return _Plan(document_id, document["title"], document["author"], tuple(planned), question=question)


def plan_sentence(store, document_id: str, settings: Mapping[str, Any], block: int, sentence: int, *,
                  document: Mapping[str, Any] | None = None, question: str | None = None) -> _Plan | None:
    """One sentence of a block, as one piece of its chapter, for an ask answered at once. ``sentence`` counts the
    block's sentences as the Reader reads and highlights them: its ``sentences`` offsets, from 0 (a block without any is
    one sentence, as the renderer reads it). Nothing is held back, and the piece is named as a document's part is. None
    when the document doesn't exist. Raises BlocksError or SentenceError."""
    if document is None:
        document = store.get_document(document_id, include_blocks=False)
    if document is None:
        return None
    count = block_count(document)
    if not 0 <= block < count:
        raise BlocksError(f"block must be a block of the document, from 0 to {count - 1}")
    found = store.get_blocks(document_id, block, block)
    if not found:
        return None  # deleted meanwhile
    row = found[0]
    spans = row["sentences"] or [(0, len(row["text"]))]
    if not 0 <= sentence < len(spans):
        many = f"{len(spans)} sentence" + ("" if len(spans) == 1 else "s")
        raise SentenceError(f"sentence must be one of block {block}'s sentences, from 0: it has {many}")
    start, end = spans[sentence]
    chapter = next(chapter for chapter in document["chapters"] if chapter["index"] == row["chapter_index"])
    piece = Piece(chapter["index"], block, block, row["text"][start:end])
    source = _source_line(settings, document, chapter)
    keys = (piece_key(document_id, piece, settings, prompts.lead(question, source)),)
    part = _ChapterPlan(chapter["index"], chapter["title"], (), (piece,), keys, source)
    return _Plan(document_id, document["title"], document["author"], (part,), question=question)


def plan_text(store, document_id: str, text: str, settings: Mapping[str, Any], *,
              question: str | None = None) -> _Plan | None:
    """The pieces a text goes to the AI in, as it was sent (``cut_text``), for a job whose document keeps it for
    listening: one part titled as the document, with nothing held back, and ``question`` before every piece. A piece's
    ``first_block`` and ``last_block`` are its number, from 0, since the text has no blocks. None when the document
    doesn't exist."""
    document = store.get_document(document_id, include_blocks=False)
    if document is None:
        return None
    pieces = tuple(Piece(0, number, number, part) for number, part in enumerate(cut_text(text, settings["chunk_chars"])))
    keys = tuple(piece_key(document_id, piece, settings, prompts.lead(question)) for piece in pieces)
    part = _ChapterPlan(0, document["title"], (), pieces, keys)
    return _Plan(document_id, document["title"], document["author"], (part,), as_sent=True, question=question)


def answers_document(title: str, author: str | None, chapters: Sequence[tuple[str, Sequence[str], Sequence[str]]]) -> Document:
    """A Reader document of the AI's answers, as a job's result is made: for each ``(chapter title, its opening
    headings, its answers)``, the headings as they are, then each answer in paragraphs (split at blank lines).
    Raises ValueError when nothing in it can be read."""
    made = []
    for chapter_title, headings, answers in chapters:
        blocks = [Block(heading, HEADING) for heading in headings]
        for answer in answers:
            blocks.extend(Block(paragraph, PARAGRAPH) for paragraph in _PARAGRAPH_BREAK.split(answer) if paragraph.strip())
        made.append(Chapter(chapter_title, blocks))
    return finalize(Document(title, SOURCE_TYPE, made, author), title)


# ── studio.db: the job and piece rows, on the Reader's own connection ────────


def _job_row(job_id: str) -> dict | None:
    with _reader_lock:
        row = get_db().execute("SELECT * FROM assistant_jobs WHERE id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


def _set_stopped(job_id: str, status: str, code: str | None = None, message: str | None = None) -> None:
    """End a job that is still queued or running; one that already ended stays as it is."""
    with _writing() as db:
        db.execute(
            "UPDATE assistant_jobs SET status = ?, error = ?, error_message = ?, updated = ? "
            "WHERE id = ? AND status IN ('queued', 'running')",
            (status, code, message, _now(), job_id),
        )


def _saved_pieces(keys: Sequence[str]) -> dict[str, dict]:
    found: dict[str, dict] = {}
    with _reader_lock:
        db = get_db()
        for start in range(0, len(keys), 500):
            batch = list(keys[start:start + 500])
            marks = ",".join("?" * len(batch))
            for row in db.execute(f"SELECT key, text, thinking, finish_reason FROM assistant_pieces WHERE key IN ({marks})", batch):
                found[row["key"]] = dict(row)
    return found


def _save_piece(key: str, job_id: str, piece: Piece, reply) -> bool:
    """Save one answer; False when the job is gone (its document was deleted)."""
    with _writing() as db:
        if db.execute("SELECT 1 FROM assistant_jobs WHERE id = ?", (job_id,)).fetchone() is None:
            return False
        now = _now()
        db.execute(
            "INSERT OR REPLACE INTO assistant_pieces (key, job_id, chapter_index, first_block, last_block, text, "
            "thinking, finish_reason, created) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (key, job_id, piece.chapter_index, piece.first_block, piece.last_block, reply.text, reply.thinking,
             reply.finish_reason, now),
        )
        db.execute("UPDATE assistant_jobs SET updated = ? WHERE id = ?", (now, job_id))
        return True


_SUMMARY_COLUMNS = (
    "id, document_id, preset_id, title, first_chapter, last_chapter, first_block, last_block, status, error, "
    "error_message, result_document_id, created, updated"
)


def _summary(row: Mapping[str, Any]) -> dict:
    return {
        "id": row["id"],
        "document_id": row["document_id"],
        "preset": row["preset_id"],
        "title": row["title"],
        "chapters_requested": None if row["first_chapter"] is None else [row["first_chapter"], row["last_chapter"]],
        "blocks_requested": None if row["first_block"] is None else [row["first_block"], row["last_block"]],
        "status": row["status"],
        "error": row["error"],
        "error_message": row["error_message"],
        "result_document_id": row["result_document_id"],
        "created": row["created"],
        "updated": row["updated"],
    }


class _JobFailed(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class _JobGone(Exception):
    """The job's row is gone: its source document was deleted."""


# ── The runner ───────────────────────────────────────────────────────────────


class AssistantJobs:
    """The book jobs of one app: start, read, list and cancel them, and the one task that runs them.

    ``start``, ``cancel`` and ``stop`` run on the app's event loop; ``view``, ``list`` and ``forget_document`` may
    be called from any thread.
    """

    def __init__(self, get_store: Callable[[], Any]) -> None:
        self._get_store = get_store
        self._pending: deque[tuple[str, str, Any]] = deque()  # (job id, document id, EffectiveSettings)
        self._running: tuple[str, str, asyncio.Task] | None = None
        self._cancel_requests: set[str] = set()
        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._registered = False
        self._plans: OrderedDict[str, _Plan] = OrderedDict()
        self._plans_lock = threading.Lock()

    # Planning

    @property
    def store(self):
        """The Reader store the jobs read their documents from and save their results to."""
        return self._get_store()

    def _plan(self, job: Mapping[str, Any]) -> _Plan:
        """The job's chapters and pieces (documents never change, so it's cached)."""
        with self._plans_lock:
            plan = self._plans.get(job["id"])
            if plan is not None:
                self._plans.move_to_end(job["id"])
                return plan
        chapters = None if job["first_chapter"] is None else (job["first_chapter"], job["last_chapter"])
        blocks = None if job["first_block"] is None else (job["first_block"], job["last_block"])
        settings = json.loads(job["settings_json"])
        text = settings.pop(TEXT_FIELD, None)
        question = settings.pop(QUESTION_FIELD, None)  # an ask job's
        named = bool(settings.pop(NAMED_FIELD, False))
        if text is not None:  # a text kept with its job, sent as it was
            plan = plan_text(self._get_store(), job["document_id"], text, settings, question=question)
        else:
            plan = plan_document(self._get_store(), job["document_id"], settings, chapters=chapters, blocks=blocks,
                                 question=question, named=named)
        if plan is None:
            raise _JobGone(job["id"])
        with self._plans_lock:
            self._plans[job["id"]] = plan
            while len(self._plans) > _PLAN_CACHE:
                self._plans.popitem(last=False)
        return plan

    # Reading (any thread)

    def view(self, job_id: str) -> dict | None:
        """The job with each chapter's status, its text once done and its thinking if shown; None if unknown."""
        job = _job_row(job_id)
        if job is None:
            return None
        try:
            plan = self._plan(job)
        except _JobGone:
            return None
        settings = json.loads(job["settings_json"])
        for name in _SENT_FIELDS:  # what it sends, not a setting
            settings.pop(name, None)
        saved = _saved_pieces(plan.keys)
        status = job["status"]
        chapters = []
        stopped_at = None
        for chapter in plan.chapters:
            answers = [saved[key] for key in chapter.keys if key in saved]
            done = len(answers) == len(chapter.keys)
            if done:
                chapter_status = "done"
            elif stopped_at is None:
                stopped_at = chapter.index
                chapter_status = status
            else:
                chapter_status = "queued" if status in ACTIVE else status
            thinking = [answer["thinking"] for answer in answers if answer["thinking"]] if done else []
            chapters.append({
                "index": chapter.index,
                "title": chapter.title,
                "status": chapter_status,
                "pieces": len(chapter.keys),
                "pieces_done": len(answers),
                "text": JOIN.join(answer["text"] for answer in answers) if done else None,
                "thinking": JOIN.join(thinking) if settings.get("show_thinking") and thinking else None,
                "error": job["error"] if chapter.index == stopped_at and status == "failed" else None,
            })
        return {**_summary(job), "settings": settings, "chapters": chapters}

    def list(self) -> list[dict]:
        """Every job, newest first, without its chapters."""
        with _reader_lock:  # not settings_json, which may hold a long text
            rows = get_db().execute(f"SELECT {_SUMMARY_COLUMNS} FROM assistant_jobs ORDER BY created DESC, rowid DESC").fetchall()
        return [_summary(row) for row in rows]

    # Starting and cancelling (the event loop)

    async def start(self, document_id: str, settings, *, chapters: Any = None, blocks: Any = None,
                    title: str | None = None, text: str | None = None, question: str | None = None,
                    named: bool = False) -> dict | None:
        """Queue a job over the document, its ``chapters`` or its ``blocks`` (each ``[first, last]``; not both), or
        over ``text``, which the document keeps for listening: the job keeps it and sends it as it was.
        ``question`` goes before every piece and, with ``named``, the line naming the document and the piece's
        chapter. None when the document doesn't exist. Raises ChaptersError or BlocksError for a bad range."""
        job_id = await asyncio.to_thread(self._create, document_id, settings, chapters, blocks, title, text, question,
                                         named)
        if job_id is None:
            return None
        self._pending.append((job_id, document_id, settings))
        self._ensure_task()
        return await asyncio.to_thread(self.view, job_id)

    def _create(self, document_id: str, settings, chapters: Any, blocks: Any, title: str | None,
                text: str | None = None, question: str | None = None, named: bool = False) -> str | None:
        if chapters is not None and blocks is not None:
            raise BlocksError("send chapters or blocks, not both")
        if text is not None and (chapters is not None or blocks is not None):
            raise BlocksError("send a text, or a part of the document, not both")
        stored = settings.to_dict()
        for name, value in ((TEXT_FIELD, text), (QUESTION_FIELD, question), (NAMED_FIELD, named or None)):
            if value is not None:
                stored[name] = value
        document = self._get_store().get_document(document_id, include_blocks=False)
        if document is None:
            return None
        first, last = chapter_range(chapters, len(document["chapters"]))
        first_block, last_block = block_range(blocks, block_count(document)) if blocks is not None else (None, None)
        name = f"{document['title']}: {title or settings.name}"[:TITLE_MAX_CHARS]
        job_id = uuid4().hex
        now = _now()
        with _writing() as db:
            if db.execute("SELECT 1 FROM reader_documents WHERE id = ?", (document_id,)).fetchone() is None:
                return None
            db.execute(
                "INSERT INTO assistant_jobs (id, document_id, preset_id, settings_json, first_chapter, last_chapter, "
                "first_block, last_block, title, status, created, updated) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)",
                (job_id, document_id, settings.preset_id, json.dumps(stored),
                 None if chapters is None else first, None if chapters is None else last, first_block, last_block,
                 name, now, now),
            )
        return job_id

    async def cancel(self, job_id: str) -> dict | None:
        """Cancel a queued or running job (its finished pieces stay saved); an ended job stays as it is."""
        queued = next((entry for entry in self._pending if entry[0] == job_id), None)
        running = self._running
        if queued is not None:
            self._pending.remove(queued)
        elif running is not None and running[0] == job_id:
            self._cancel_requests.add(job_id)
            running[2].cancel()
            await asyncio.wait({running[2]}, timeout=CANCEL_WAIT_S)
        # Also a job this runner doesn't hold but the table still calls queued or running.
        await asyncio.to_thread(_set_stopped, job_id, "cancelled")
        return await asyncio.to_thread(self.view, job_id)

    def forget_document(self, document_id: str) -> None:
        """Its document was deleted (with its jobs): stop its running job. Any thread."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._forget, document_id)
        except RuntimeError:  # the loop closed meanwhile
            pass

    def _forget(self, document_id: str) -> None:
        self._pending = deque(entry for entry in self._pending if entry[1] != document_id)
        running = self._running
        if running is not None and running[1] == document_id:
            self._cancel_requests.add(running[0])
            running[2].cancel()

    async def stop(self) -> None:
        """Stop at shutdown: the running job and the queued ones fail with ``interrupted``."""
        self._registered = False
        pending = [entry[0] for entry in self._pending]
        self._pending.clear()
        task, self._task = self._task, None
        if task is not None and not task.done() and task.get_loop() is asyncio.get_running_loop():
            task.cancel()
            await asyncio.wait({task}, timeout=STOP_WAIT_S)
        for job_id in pending:
            await asyncio.to_thread(_set_stopped, job_id, "failed", INTERRUPTED, INTERRUPTED_MESSAGE)

    def _ensure_task(self) -> None:
        loop = asyncio.get_running_loop()
        if self._task is None or self._task.done() or self._task.get_loop() is not loop:
            self._loop = loop
            self._task = loop.create_task(self._run(), name="reader-assistant-jobs")
        if not self._registered:
            assistant.on_close(self.stop)
            self._registered = True

    # Running

    async def _run(self) -> None:
        while self._pending:
            job_id, document_id, settings = self._pending.popleft()
            task = asyncio.create_task(self._run_job(job_id, settings), name=f"reader-assistant-job-{job_id}")
            self._running = (job_id, document_id, task)
            try:
                await asyncio.wait({task})
            except asyncio.CancelledError:  # shutdown
                task.cancel()
                await asyncio.wait({task}, timeout=STOP_WAIT_S)
                await asyncio.to_thread(_set_stopped, job_id, "failed", INTERRUPTED, INTERRUPTED_MESSAGE)
                raise
            finally:
                self._running = None
                self._cancel_requests.discard(job_id)

    async def _run_job(self, job_id: str, settings) -> None:
        finishing: asyncio.Future | None = None
        try:
            job = await asyncio.to_thread(self._begin, job_id)
            if job is None:
                return  # cancelled while queued, or its document was deleted
            plan = await asyncio.to_thread(self._plan, job)
            saved = await asyncio.to_thread(_saved_pieces, plan.keys)
            for chapter in plan.chapters:
                for piece, key in zip(chapter.pieces, chapter.keys):
                    if key in saved:
                        continue
                    # A text sent as it is fills only {text}, as one of one piece does.
                    values = {} if plan.as_sent else {"title": plan.title, "author": plan.author or "", "chapter": chapter.title}
                    # An ask job's question, and the line naming its document and this chapter.
                    messages = prompts.messages_for(settings, [], piece.text, question=plan.question,
                                                    source=chapter.source, **values)
                    reply = await self._ask(messages, settings)
                    if not await asyncio.to_thread(_save_piece, key, job_id, piece, reply):
                        return
            # The result is saved whatever happens now: a cancel waits for it.
            finishing = asyncio.ensure_future(asyncio.to_thread(self._finish, job_id, plan))
            await asyncio.shield(finishing)
        except asyncio.CancelledError:
            if finishing is not None:
                await asyncio.wait({finishing})
                if not finishing.cancelled():
                    finishing.exception()  # seen: the cancel decides what the job says now
            if job_id in self._cancel_requests:
                await asyncio.to_thread(_set_stopped, job_id, "cancelled")
            else:
                await asyncio.to_thread(_set_stopped, job_id, "failed", INTERRUPTED, INTERRUPTED_MESSAGE)
            raise
        except _JobFailed as exc:
            await asyncio.to_thread(_set_stopped, job_id, "failed", exc.code, exc.message)
        except _JobGone:
            pass
        except Exception:
            logger.exception("Book job %s failed", job_id)
            await asyncio.to_thread(_set_stopped, job_id, "failed", INTERNAL_ERROR, "The server failed while running the job")

    def _begin(self, job_id: str) -> dict | None:
        with _writing() as db:
            started = db.execute(
                "UPDATE assistant_jobs SET status = 'running', updated = ? WHERE id = ? AND status = 'queued'",
                (_now(), job_id),
            ).rowcount
        return _job_row(job_id) if started else None

    async def _ask(self, messages: list[dict], settings):
        """One piece's answer, retried once after an error."""
        for attempt in (1, 2):
            client = assistant.get_client()
            if client is None:
                raise _JobFailed(NOT_STARTED, "The AI assistant client isn't running")
            try:
                return await client.reply(messages, settings)
            except AssistantError as exc:
                if attempt == 2:
                    raise _JobFailed(exc.code, exc.message) from None
                logger.warning("A book job's request failed (%s); retrying it once", exc.code)
                await asyncio.sleep(RETRY_DELAY_S)
        raise AssertionError("unreachable")

    def _finish(self, job_id: str, plan: _Plan) -> None:
        """Save the answers as a new Reader document, and the job as done."""
        saved = _saved_pieces(plan.keys)
        job = _job_row(job_id)
        if job is None or job["status"] != "running":
            return
        chapters = [
            (chapter.title, chapter.headings, [saved[key]["text"] for key in chapter.keys]) for chapter in plan.chapters
        ]
        try:
            document = answers_document(job["title"], plan.author, chapters)
        except ValueError:
            raise _JobFailed(ASSISTANT_EMPTY_REPLY, "The AI's answers have no text to read") from None
        result_id, _created = self._get_store().save_document(document)
        with _writing() as db:
            db.execute(
                "UPDATE assistant_jobs SET status = 'done', error = NULL, error_message = NULL, result_document_id = ?, "
                "updated = ? WHERE id = ? AND status = 'running'",
                (result_id, _now(), job_id),
            )


# ── The app's runner, shared with POST /api/assistant/ask ────────────────────

_runner: AssistantJobs | None = None


def set_runner(runner: AssistantJobs | None) -> None:
    """Make ``runner`` the one ``/api/assistant/ask`` queues its long inputs on. The Reader's routes set theirs
    when the app builds them, so every job, a book job or an ask, waits for the same one slot."""
    global _runner
    _runner = runner


def get_runner() -> AssistantJobs | None:
    """The runner the Reader's routes made (the last app built), or None when there is none."""
    return _runner
