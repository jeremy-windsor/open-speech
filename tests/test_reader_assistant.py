"""Book jobs: a Reader document sent to the AI in pieces, its answers saved as a new Reader document.

The rules are in docs/AI-ASSISTANT.md, "Book jobs". The AI is a fake that stands in
for the shared AssistantClient: its reply() answers from a function, records every call, and can be held until the
test lets it go. Nothing contacts a server, and every wait is bounded by WAIT_S.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from contextlib import closing

import pytest
from fastapi.testclient import TestClient

from src import assistant
from src import main as main_module
from src import storage as storage_module
from src.assistant import presets
from src.assistant.client import AssistantReply
from src.assistant.errors import ASSISTANT_HTTP_ERROR, ASSISTANT_TIMEOUT, AssistantError
from src.reader import assistant_jobs
from src.reader import store as reader_store_module
from src.reader.assistant_jobs import Piece
from src.reader.document import HEADING, PARAGRAPH
from tests.test_assistant import DOCUMENTED_DEFAULTS
from tests.test_reader_sources import _epub

WAIT_S = 10.0
API_KEY = "sk-local-secret-0123456789"

TWO_CHAPTERS = "# One\n\nThe first paragraph.\n\nThe second paragraph.\n\n# Two\n\nThe third paragraph."
FIRST_PIECE = "The first paragraph.\n\nThe second paragraph."
THIRD_PIECE = "The third paragraph."


def _reworked(text: str) -> AssistantReply:
    return AssistantReply(text=f"Reworked: {text}", thinking=None, finish_reason="stop")


class FakeAI:
    """Stands in for the shared AssistantClient.

    ``answer(user_message)`` gives the reply, or an exception for reply() to raise. ``hold(when)`` makes each call
    whose user message matches wait (bounded) until ``release()``; ``waiting`` is set once one does.
    """

    host = "ai-server.local:8000"

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict], object]] = []
        self.answer = _reworked
        self.cancelled = 0
        self.closed = False
        self.waiting = threading.Event()
        self._gate = threading.Event()
        self._gate.set()
        self._hold_when = lambda _text: False

    def hold(self, when=lambda _text: True) -> None:
        self._hold_when = when
        self.waiting.clear()
        self._gate.clear()

    def release(self) -> None:
        self._gate.set()

    @property
    def sent(self) -> list[str]:
        """The user message of each call, in order."""
        return [messages[-1]["content"] for messages, _settings in self.calls]

    async def reply(self, messages, settings):
        self.calls.append((messages, settings))
        text = messages[-1]["content"]
        try:
            deadline = time.monotonic() + WAIT_S
            while not self._gate.is_set() and self._hold_when(text):
                self.waiting.set()
                if time.monotonic() > deadline:
                    raise AssertionError("the fake AI was held longer than the test allows")
                await asyncio.sleep(0.005)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        result = self.answer(text)
        if isinstance(result, BaseException):
            raise result
        return result

    async def aclose(self) -> None:
        self.closed = True


class _Server:
    """The configured app with the assistant on (the documented defaults and an API key the AI server would
    need), the fake AI as its shared client, and a studio.db of its own. ``client()`` is a TestClient: inside its
    ``with`` the app has started (lifespan), and leaving it shuts the app down."""

    def __init__(self, monkeypatch, tmp_path) -> None:
        self.fake = FakeAI()
        self.db_path = tmp_path / "data" / "studio.db"
        settings = main_module.settings
        for name, value in DOCUMENTED_DEFAULTS.items():
            monkeypatch.setattr(settings, name, value)
        monkeypatch.setattr(settings, "os_assistant_api_key", API_KEY)
        monkeypatch.setattr(settings, "os_api_key", "")
        monkeypatch.setattr(settings, "os_studio_db_path", str(self.db_path))
        monkeypatch.setattr(storage_module, "_conn", None)
        monkeypatch.setattr(assistant_jobs, "RETRY_DELAY_S", 0.0)
        monkeypatch.setattr(assistant.AssistantClient, "from_settings", classmethod(lambda _cls, _settings, **_kw: self.fake))
        self.app = main_module.create_app()

    def client(self) -> TestClient:
        return TestClient(self.app)

    def rows(self, sql: str, *parameters) -> list[tuple]:
        with closing(sqlite3.connect(self.db_path)) as db:
            return [tuple(row) for row in db.execute(sql, parameters)]


@pytest.fixture
def server(tmp_path, monkeypatch):
    server = _Server(monkeypatch, tmp_path)
    yield server
    assistant.set_client(None)
    assistant._close_callbacks.clear()
    # The connections this test opened on its own studio.db (Windows can't remove an open database).
    if storage_module._conn is not None:
        storage_module._conn.close()
    reader_store_module.close()


@pytest.fixture
def books(server):
    """A started app: (its TestClient, the fake AI)."""
    with server.client() as client:
        yield client, server.fake


def _paste(client: TestClient, text: str = TWO_CHAPTERS, title: str = "Notes") -> str:
    response = client.post("/v1/reader/documents", json={"type": "text", "text": text, "title": title})
    assert response.status_code in (200, 201), response.text
    return response.json()["id"]


def _start(client: TestClient, document_id: str, **body):
    return client.post(f"/v1/reader/documents/{document_id}/assistant", json=body)


def _job(client: TestClient, job_id: str) -> dict:
    response = client.get(f"/v1/reader/assistant-jobs/{job_id}")
    assert response.status_code == 200, response.text
    return response.json()


def _wait(client: TestClient, job_id: str, until=lambda job: job["status"] in ("done", "failed", "cancelled")) -> dict:
    deadline = time.monotonic() + WAIT_S
    while True:
        job = _job(client, job_id)
        if until(job):
            return job
        if time.monotonic() > deadline:
            raise AssertionError(f"the job never got there: {job}")
        time.sleep(0.01)


def _run(client: TestClient, document_id: str, **body) -> dict:
    started = _start(client, document_id, **body)
    assert started.status_code == 202, started.text
    return _wait(client, started.json()["id"])


def _document(client: TestClient, document_id: str) -> dict:
    response = client.get(f"/v1/reader/documents/{document_id}")
    assert response.status_code == 200, response.text
    return response.json()


def _paragraph(n: int) -> str:
    return " ".join(f"Paragraph {n} sentence {i} goes on a while." for i in range(10))


# ── A job from start to finish ───────────────────────────────────────────────


def test_a_two_chapter_document_becomes_a_done_job_and_a_new_reader_document(books):
    client, fake = books
    source = _paste(client)

    started = _start(client, source)
    assert started.status_code == 202, started.text
    assert started.json()["status"] in ("queued", "running")
    assert started.json()["document_id"] == source and started.json()["preset"] == "book"
    job = _wait(client, started.json()["id"])

    assert job["status"] == "done" and job["error"] is None
    # Each chapter's text in pieces of whole blocks; its opening heading is kept as it is, not sent.
    assert fake.sent == [FIRST_PIECE, THIRD_PIECE]
    assert [(c["index"], c["title"], c["status"], c["pieces"], c["pieces_done"], c["text"]) for c in job["chapters"]] == [
        (0, "One", "done", 1, 1, f"Reworked: {FIRST_PIECE}"),
        (1, "Two", "done", 1, 1, f"Reworked: {THIRD_PIECE}"),
    ]
    result = _document(client, job["result_document_id"])
    assert result["title"] == "Notes: Book" and result["source_type"] == "assistant"
    assert [chapter["title"] for chapter in result["chapters"]] == ["One", "Two"]
    assert [(block["chapter_index"], block["kind"], block["text"]) for block in result["blocks"]] == [
        (0, HEADING, "One"),
        (0, PARAGRAPH, "Reworked: The first paragraph."),
        (0, PARAGRAPH, "The second paragraph."),
        (1, HEADING, "Two"),
        (1, PARAGRAPH, "Reworked: The third paragraph."),
    ]
    assert all(block["sentences"] for block in result["blocks"]), "split into sentences, ready to play"
    library = {document["id"] for document in client.get("/v1/reader/documents").json()["documents"]}
    assert library == {source, result["id"]}


def test_a_chapter_over_chunk_chars_is_sent_in_pieces_cut_at_block_boundaries(books):
    client, fake = books
    paragraphs = [_paragraph(n) for n in range(5)]
    source = _paste(client, "# Long\n\n" + "\n\n".join(paragraphs))

    job = _run(client, source, overrides={"chunk_chars": 1000})

    assert job["status"] == "done"
    assert len(fake.sent) == 3 and all(len(text) <= 1000 for text in fake.sent)
    assert [paragraph for text in fake.sent for paragraph in text.split("\n\n")] == paragraphs
    assert (job["chapters"][0]["pieces"], job["chapters"][0]["pieces_done"]) == (3, 3)
    texts = [block["text"] for block in _document(client, job["result_document_id"])["blocks"]]
    assert texts[0] == "Long" and texts[1:] == [
        paragraph for text in fake.sent for paragraph in f"Reworked: {text}".split("\n\n")
    ]


def test_a_block_longer_than_chunk_chars_is_cut_at_sentences():
    long = " ".join(f"Sentence {i} of a very long block." for i in range(80))
    blocks = [
        {"index": 7, "kind": HEADING, "text": "Title"},
        {"index": 8, "kind": PARAGRAPH, "text": "Short."},
        {"index": 9, "kind": PARAGRAPH, "text": long},
        {"index": 10, "kind": PARAGRAPH, "text": "After."},
    ]

    headings, pieces = assistant_jobs.cut_chapter(2, blocks, 1000)

    assert headings == ["Title"]
    assert pieces[0] == Piece(2, 8, 8, "Short.")
    middle = pieces[1:-1]
    assert len(middle) == 3 and all(piece.first_block == piece.last_block == 9 for piece in middle)
    assert all(len(piece.text) <= 1000 and piece.text.endswith("block.") for piece in middle), "cut at sentence ends"
    assert " ".join(piece.text for piece in middle) == long
    assert pieces[-1] == Piece(2, 10, 10, "After.")


def test_only_a_chapters_opening_headings_are_kept_out_of_the_pieces():
    blocks = [
        {"index": 0, "kind": HEADING, "text": "Part One"},
        {"index": 1, "kind": HEADING, "text": "Chapter 1"},
        {"index": 2, "kind": PARAGRAPH, "text": "Before."},
        {"index": 3, "kind": HEADING, "text": "A section"},
        {"index": 4, "kind": PARAGRAPH, "text": "After."},
    ]

    headings, pieces = assistant_jobs.cut_chapter(0, blocks, 1000)

    assert headings == ["Part One", "Chapter 1"]
    assert pieces == [Piece(0, 2, 4, "Before.\n\nA section\n\nAfter.")]
    assert assistant_jobs.cut_chapter(1, blocks[:2], 1000) == (["Part One", "Chapter 1"], [])


def test_chapters_picks_the_chapters_sent(books):
    client, fake = books

    job = _run(client, _paste(client), chapters=[1, 1])

    assert job["status"] == "done" and fake.sent == [THIRD_PIECE]
    assert [(chapter["index"], chapter["title"]) for chapter in job["chapters"]] == [(1, "Two")]
    assert [chapter["title"] for chapter in _document(client, job["result_document_id"])["chapters"]] == ["Two"]


def test_the_presets_prompts_reach_the_ai_with_title_author_and_chapter_filled_in(books):
    client, fake = books
    created = client.post("/api/assistant/presets", json={
        "name": "Summary",
        "system_prompt": "You summarise {braces} for listening.",
        "user_template": 'Summarise "{chapter}" of {title} by {author}:\n\n{text}',
    })
    assert created.status_code == 201, created.text
    book = _epub([("Chapter One", "<p>Hello from {title} the book.</p>")])
    upload = client.post("/v1/reader/documents", files={"file": ("tiny.epub", book, "application/epub+zip")})
    assert upload.status_code == 201, upload.text

    job = _run(client, upload.json()["id"], preset="summary")

    assert job["status"] == "done" and job["preset"] == "summary"
    messages, settings = fake.calls[0]
    assert messages == [
        {"role": "system", "content": "You summarise {braces} for listening."},
        {"role": "user", "content": 'Summarise "Chapter One" of Tiny Book by A. Writer:\n\nHello from {title} the book.'},
    ]
    assert settings.timeout_s == DOCUMENTED_DEFAULTS["os_assistant_job_timeout_s"], "a book job's timeout"
    assert _document(client, job["result_document_id"])["title"] == "Tiny Book: Summary"


def test_a_job_title_names_the_result(books):
    client, _fake = books

    job = _run(client, _paste(client), title="  Retold  ")

    assert job["title"] == "Notes: Retold"
    assert _document(client, job["result_document_id"])["title"] == "Notes: Retold"


def test_overrides_win_over_the_preset_and_a_preset_edit_does_not_change_a_started_job(books):
    client, fake = books
    edited = client.put("/api/assistant/presets/book", json={"system_prompt": "First prompt.", "temperature": 0.2})
    assert edited.status_code == 200, edited.text
    source = _paste(client)
    fake.hold()
    started = _start(client, source, overrides={"temperature": 0.7, "reasoning_effort": "low"})
    assert started.status_code == 202, started.text
    assert fake.waiting.wait(WAIT_S)

    assert client.put("/api/assistant/presets/book", json={"system_prompt": "Second prompt."}).status_code == 200
    fake.release()
    job = _wait(client, started.json()["id"])

    assert job["status"] == "done" and len(fake.calls) == 2
    for messages, settings in fake.calls:
        assert messages[0] == {"role": "system", "content": "First prompt."}
        assert settings.temperature == 0.7 and settings.reasoning_effort == "low"
    assert job["settings"]["system_prompt"] == "First prompt." and job["settings"]["temperature"] == 0.7
    assert _run(client, source)["status"] == "done"
    assert fake.calls[-1][0][0] == {"role": "system", "content": "Second prompt."}, "a new job takes the edited preset"


def test_the_thinking_is_saved_and_shown_but_never_put_into_the_document(books, server):
    client, fake = books
    fake.answer = lambda text: AssistantReply(text=f"Answer: {text}", thinking=f"Pondering {text}", finish_reason="stop")
    source = _paste(client)

    job = _run(client, source)

    assert [chapter["thinking"] for chapter in job["chapters"]] == [f"Pondering {FIRST_PIECE}", f"Pondering {THIRD_PIECE}"]
    assert not any("Pondering" in block["text"] for block in _document(client, job["result_document_id"])["blocks"])
    assert server.rows("SELECT thinking, finish_reason FROM assistant_pieces ORDER BY chapter_index") == [
        (f"Pondering {FIRST_PIECE}", "stop"), (f"Pondering {THIRD_PIECE}", "stop"),
    ]
    hidden = _run(client, source, overrides={"show_thinking": False})
    assert hidden["status"] == "done" and [chapter["thinking"] for chapter in hidden["chapters"]] == [None, None]
    assert len(fake.calls) == 2, "show_thinking doesn't reach the AI, so the saved pieces were used"


def test_a_finished_chapters_text_is_there_while_later_chapters_run(books):
    client, fake = books
    fake.hold(lambda text: text == THIRD_PIECE)
    started = _start(client, _paste(client)).json()
    assert fake.waiting.wait(WAIT_S)

    job = _job(client, started["id"])

    assert job["status"] == "running"
    assert [(chapter["status"], chapter["text"]) for chapter in job["chapters"]] == [
        ("done", f"Reworked: {FIRST_PIECE}"), ("running", None),
    ]
    fake.release()
    assert _wait(client, started["id"])["status"] == "done"


# ── Errors, and starting a job again ─────────────────────────────────────────


def test_an_error_is_retried_once(books):
    client, fake = books
    failures = [AssistantError("The AI server sent no whole reply within 600 s", ASSISTANT_TIMEOUT)]
    fake.answer = lambda text: failures.pop() if failures else _reworked(text)

    job = _run(client, _paste(client))

    assert job["status"] == "done" and job["error"] is None
    assert fake.sent == [FIRST_PIECE, FIRST_PIECE, THIRD_PIECE]


def test_a_second_error_fails_the_job_keeps_the_finished_pieces_and_a_new_run_sends_only_the_rest(books):
    client, fake = books

    def answer(text):
        if text == THIRD_PIECE:
            return AssistantError("The AI server answered HTTP 500: overloaded", ASSISTANT_HTTP_ERROR)
        return _reworked(text)

    fake.answer = answer
    source = _paste(client)

    job = _run(client, source)

    assert job["status"] == "failed" and job["error"] == ASSISTANT_HTTP_ERROR
    assert job["error_message"] == "The AI server answered HTTP 500: overloaded"
    assert job["result_document_id"] is None
    assert fake.sent == [FIRST_PIECE, THIRD_PIECE, THIRD_PIECE], "the failing piece was tried twice"
    assert [(c["status"], c["error"], c["text"]) for c in job["chapters"]] == [
        ("done", None, f"Reworked: {FIRST_PIECE}"), ("failed", ASSISTANT_HTTP_ERROR, None),
    ]
    assert [document["id"] for document in client.get("/v1/reader/documents").json()["documents"]] == [source]

    fake.answer = _reworked
    again = _run(client, source)

    assert again["status"] == "done" and again["id"] != job["id"]
    assert fake.sent[3:] == [THIRD_PIECE], "only the missing piece is sent"
    assert [chapter["text"] for chapter in again["chapters"]] == [f"Reworked: {FIRST_PIECE}", f"Reworked: {THIRD_PIECE}"]


def test_only_a_setting_that_reaches_the_ai_sends_the_pieces_again(books):
    client, fake = books
    source = _paste(client)
    first = _run(client, source)
    assert first["status"] == "done" and len(fake.calls) == 2

    same = _run(client, source, overrides={"timeout_s": 30, "show_thinking": False, "speak_errors": False})
    assert same["status"] == "done" and len(fake.calls) == 2, "nothing that reaches the AI changed"
    assert same["result_document_id"] == first["result_document_id"], "the same text is the same Reader document"

    for override in (
        {"system_prompt": "Other."}, {"user_template": "Retell: {text}"}, {"model": "other-model"}, {"thinking": "off"},
        {"reasoning_effort": "high"}, {"max_tokens": 500}, {"temperature": 0.5}, {"top_p": 0.9},
    ):
        before = len(fake.calls)
        assert _run(client, source, overrides=override)["status"] == "done"
        assert len(fake.calls) == before + 2, override


def test_a_job_run_again_with_another_title_names_its_result_with_that_title(books):
    # The second job sent nothing (its title doesn't reach the AI), built a result
    # of the same content, and the store's content hash, which left the title out, handed it the first job's
    # document, still named "Notes: Summary". Contract: docs/AI-ASSISTANT.md "Book jobs", the result is named
    # "<document title>: <your title, or the preset's name>" and the job's result_document_id names it.
    client, fake = books
    source = _paste(client)
    first = _run(client, source, title="Summary")
    assert first["status"] == "done" and _document(client, first["result_document_id"])["title"] == "Notes: Summary"
    sent = len(fake.calls)

    second = _run(client, source, title="Key points")

    assert second["status"] == "done" and len(fake.calls) == sent, "the title doesn't reach the AI"
    assert second["result_document_id"] != first["result_document_id"]
    assert _document(client, second["result_document_id"])["title"] == "Notes: Key points"
    assert _document(client, first["result_document_id"])["title"] == "Notes: Summary"
    again = _run(client, source, title="Key points")
    assert again["result_document_id"] == second["result_document_id"], "the same job again is the same document"


def test_an_ai_result_saved_before_its_title_counted_is_still_found_under_that_title(books):
    # A result stored with the old content hash (without its title) is the
    # same job's result run again under the same title, not a second copy; under another title it is not.
    client, _fake = books
    store = reader_store_module.ReaderStore()
    chapters = [("One", (), ["An answer."])]
    old_id, created = store.save_document(assistant_jobs.answers_document("Notes: Summary", None, chapters))
    assert created
    payload = json.dumps([["One", [[PARAGRAPH, "An answer."]]]], ensure_ascii=False, separators=(",", ":"))
    legacy = reader_store_module.hashlib.sha256(f"{assistant_jobs.SOURCE_TYPE}\n{payload}".encode()).hexdigest()
    with reader_store_module._writing() as db:
        db.execute("UPDATE reader_documents SET content_hash = ? WHERE id = ?", (legacy, old_id))

    same = store.save_document(assistant_jobs.answers_document("Notes: Summary", None, chapters))
    other = store.save_document(assistant_jobs.answers_document("Notes: Key points", None, chapters))

    assert same == (old_id, False)
    assert other[1] and _document(client, other[0])["title"] == "Notes: Key points"


# ── Cancelling, one job at a time, shutdown and startup ──────────────────────


def test_delete_cancels_a_running_job_and_keeps_its_finished_pieces(books):
    client, fake = books
    source = _paste(client)
    fake.hold(lambda text: text == THIRD_PIECE)
    started = _start(client, source).json()
    assert fake.waiting.wait(WAIT_S)

    cancelled = client.delete(f"/v1/reader/assistant-jobs/{started['id']}")

    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["status"] == "cancelled"
    assert fake.cancelled == 1, "the AI request was aborted"
    job = _job(client, started["id"])
    assert [chapter["status"] for chapter in job["chapters"]] == ["done", "cancelled"]
    assert job["result_document_id"] is None
    again = client.delete(f"/v1/reader/assistant-jobs/{started['id']}")
    assert again.status_code == 200 and again.json()["status"] == "cancelled", "a stopped job stays as it is"

    fake.release()
    assert _run(client, source)["status"] == "done"
    assert fake.sent == [FIRST_PIECE, THIRD_PIECE, THIRD_PIECE], "the finished piece was kept"


def test_jobs_run_one_at_a_time_newest_listed_first_and_delete_cancels_a_queued_one(books):
    client, fake = books
    first_source = _paste(client)
    second_source = _paste(client, "# Other\n\nA different book.", title="Other")
    third_source = _paste(client, "# Third\n\nYet another book.", title="Third")
    fake.hold()
    first = _start(client, first_source).json()
    assert fake.waiting.wait(WAIT_S)
    second = _start(client, second_source).json()
    third = _start(client, third_source).json()

    assert second["status"] == "queued" and _job(client, second["id"])["status"] == "queued"
    assert [chapter["status"] for chapter in _job(client, second["id"])["chapters"]] == ["queued"]
    assert len(fake.calls) == 1, "the second job waits for the first"
    deleted = client.delete(f"/v1/reader/assistant-jobs/{second['id']}")
    assert deleted.status_code == 200 and deleted.json()["status"] == "cancelled"
    fake.release()

    assert _wait(client, first["id"])["status"] == "done"
    assert _wait(client, third["id"])["status"] == "done"
    assert _job(client, second["id"])["status"] == "cancelled"
    assert "A different book." not in fake.sent and "Yet another book." in fake.sent
    listed = client.get("/v1/reader/assistant-jobs")
    assert listed.status_code == 200
    assert [job["id"] for job in listed.json()["jobs"]] == [third["id"], second["id"], first["id"]]
    assert [job["status"] for job in listed.json()["jobs"]] == ["done", "cancelled", "done"]


def test_shutdown_stops_a_running_job_and_its_ai_request(server):
    fake = server.fake
    with server.client() as client:
        source = _paste(client)
        fake.hold(lambda text: text == THIRD_PIECE)
        running = _start(client, source).json()
        assert fake.waiting.wait(WAIT_S)
        queued = _start(client, source, title="Queued").json()

    assert fake.cancelled == 1 and fake.closed, "the job stopped before the client closed"
    assert dict(server.rows("SELECT id, status || ':' || error FROM assistant_jobs")) == {
        running["id"]: "failed:interrupted", queued["id"]: "failed:interrupted",
    }
    assert server.rows("SELECT chapter_index FROM assistant_pieces") == [(0,)], "the finished piece is kept"


def test_jobs_cut_off_by_a_crash_fail_as_interrupted_at_the_next_startup(server):
    fake = server.fake
    with server.client() as client:
        source = _paste(client)
        done = _run(client, source)
    with closing(sqlite3.connect(server.db_path)) as db:
        # As a crash leaves them: one job running, one queued.
        db.execute("UPDATE assistant_jobs SET status = 'running', result_document_id = NULL WHERE id = ?", (done["id"],))
        db.execute(
            "INSERT INTO assistant_jobs (id, document_id, preset_id, settings_json, title, status, created, updated) "
            "SELECT 'queued-job', document_id, preset_id, settings_json, title, 'queued', created, updated "
            "FROM assistant_jobs WHERE id = ?",
            (done["id"],),
        )
        db.commit()

    with server.client() as client:
        for job_id in (done["id"], "queued-job"):
            job = _job(client, job_id)
            assert (job["status"], job["error"]) == ("failed", "interrupted"), job_id
        assert len(fake.calls) == 2, "nothing is sent at startup"
        again = _run(client, source)
        assert again["status"] == "done" and len(fake.calls) == 2, "the saved pieces are used"


# ── The source document, the result, and what a job holds ───────────────────


def test_deleting_the_source_deletes_its_jobs_and_pieces_but_never_the_result(books, server):
    client, fake = books
    source = _paste(client)
    first = _run(client, source)
    fake.answer = lambda text: AssistantReply(text=f"Another take: {text}", thinking=None, finish_reason="stop")
    second = _run(client, source, overrides={"system_prompt": "Another take."})
    assert second["result_document_id"] != first["result_document_id"]
    assert client.delete(f"/v1/reader/documents/{first['result_document_id']}").status_code == 204
    assert _job(client, first["id"])["result_document_id"] is None, "a deleted result is no longer named"

    assert client.delete(f"/v1/reader/documents/{source}").status_code == 204

    assert client.get(f"/v1/reader/assistant-jobs/{second['id']}").status_code == 404
    assert client.get("/v1/reader/assistant-jobs").json()["jobs"] == []
    assert server.rows("SELECT COUNT(*) FROM assistant_pieces") == [(0,)]
    assert _document(client, second["result_document_id"])["source_type"] == "assistant"


def test_deleting_the_source_stops_its_running_job(books):
    client, fake = books
    source = _paste(client)
    fake.hold()
    started = _start(client, source).json()
    assert fake.waiting.wait(WAIT_S)

    assert client.delete(f"/v1/reader/documents/{source}").status_code == 204

    deadline = time.monotonic() + WAIT_S
    while fake.cancelled == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert fake.cancelled == 1, "the AI request was aborted"
    assert client.get(f"/v1/reader/assistant-jobs/{started['id']}").status_code == 404
    assert client.get("/v1/reader/documents").json()["documents"] == []


def test_a_job_never_holds_a_server_only_setting(books, server):
    client, _fake = books

    job = _run(client, _paste(client))

    listed = client.get("/v1/reader/assistant-jobs").text
    stored = json.dumps(server.rows("SELECT * FROM assistant_jobs"))
    for text in (json.dumps(job), listed, stored):
        assert API_KEY not in text and "ai-server.local" not in text
    assert set(job["settings"]) == set(presets.EffectiveSettings().to_dict())


# ── Refusals ─────────────────────────────────────────────────────────────────


def test_every_route_answers_404_while_the_assistant_is_off(server, monkeypatch):
    monkeypatch.setattr(main_module.settings, "os_assistant_url", "")
    with server.client() as client:
        source = _paste(client)
        for method, path in (
            ("POST", f"/v1/reader/documents/{source}/assistant"),
            ("GET", "/v1/reader/assistant-jobs"),
            ("GET", "/v1/reader/assistant-jobs/some-job"),
            ("DELETE", "/v1/reader/assistant-jobs/some-job"),
        ):
            response = client.request(method, path, json={} if method == "POST" else None)
            assert response.status_code == 404, (method, path, response.text)
            assert response.json()["error"]["code"] == "assistant_disabled", (method, path)
    assert server.fake.calls == []


def test_the_assistant_off_answer_is_the_one_the_assistant_routes_give(server, monkeypatch):
    # A URL saved in Settings turns the assistant on without a restart, so the job
    # routes say what /api/assistant/* says, not "set OS_ASSISTANT_URL and restart".
    monkeypatch.setattr(main_module.settings, "os_assistant_url", "")
    with server.client() as client:
        jobs = client.get("/v1/reader/assistant-jobs")
        presets_route = client.get("/api/assistant/presets")

    assert (jobs.status_code, presets_route.status_code) == (404, 404)
    assert jobs.json()["error"] == presets_route.json()["error"]
    assert jobs.json()["error"]["message"] == (
        "The AI assistant is off: set the AI server's URL in Settings or OS_ASSISTANT_URL"
    )


def test_the_job_routes_need_the_api_key_like_the_rest(books):
    from unittest.mock import patch

    from src.config import Settings

    client, fake = books
    source = _paste(client)
    job = _run(client, source)
    secured = Settings(os_api_key="secret123", stt_model="test-model", stt_device="cpu")
    key = {"Authorization": "Bearer secret123"}

    with patch("src.middleware.settings", secured):
        for method, path in (
            ("POST", f"/v1/reader/documents/{source}/assistant"),
            ("GET", "/v1/reader/assistant-jobs"),
            ("GET", f"/v1/reader/assistant-jobs/{job['id']}"),
            ("DELETE", f"/v1/reader/assistant-jobs/{job['id']}"),
        ):
            response = client.request(method, path, json={} if method == "POST" else None)
            assert response.status_code == 401, (method, path)
        assert client.get(f"/v1/reader/assistant-jobs/{job['id']}", headers=key).status_code == 200
    assert len(fake.calls) == 2


def test_a_missing_document_or_job_is_404(books):
    client, fake = books

    assert _start(client, "no-such-document").status_code == 404
    assert client.get("/v1/reader/assistant-jobs/no-such-job").status_code == 404
    assert client.delete("/v1/reader/assistant-jobs/no-such-job").status_code == 404
    assert fake.calls == []


def test_a_job_waits_for_the_client_to_start(books):
    client, fake = books
    source = _paste(client)
    assistant.set_client(None)

    response = _start(client, source)

    assert response.status_code == 503 and response.json()["error"]["code"] == "assistant_not_started"
    assert client.get("/v1/reader/assistant-jobs").json()["jobs"] == [] and fake.calls == []


@pytest.mark.parametrize(
    "body, param",
    [
        ({"preset": "no-such-preset"}, "preset"),
        ({"preset": 5}, "preset"),
        ({"overrides": {"max_tokens": -1}}, "max_tokens"),
        ({"overrides": {"chunk_chars": 10}}, "chunk_chars"),
        ({"overrides": {"name": "Renamed"}}, "name"),
        ({"overrides": {"no_such_field": 1}}, "no_such_field"),
        ({"overrides": "not an object"}, "overrides"),
        ({"overrides": {"user_template": "No placeholder here."}}, "user_template"),
        ({"chapters": [1, 0]}, "chapters"),
        ({"chapters": [0, 2]}, "chapters"),  # the document has chapters 0 and 1
        ({"chapters": [-1, 0]}, "chapters"),
        ({"chapters": [0]}, "chapters"),
        ({"chapters": [True, 1]}, "chapters"),
        ({"chapters": "all"}, "chapters"),
        ({"title": 5}, "title"),
        ({"title": "x" * 201}, "title"),
        ({"chapter": [0, 0]}, "chapter"),
    ],
)
def test_an_invalid_job_request_gets_422_naming_the_field(books, body, param):
    client, fake = books
    source = _paste(client)

    response = _start(client, source, **body)

    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["code"] == "validation_error" and error["param"] == param, error
    assert fake.calls == [] and client.get("/v1/reader/assistant-jobs").json()["jobs"] == []


# ── studio.db ────────────────────────────────────────────────────────────────


def test_an_older_studio_db_gets_the_book_job_tables_and_keeps_its_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(storage_module.settings, "os_studio_db_path", str(tmp_path / "studio.db"))
    monkeypatch.setattr(storage_module, "_conn", None)
    try:
        storage_module.init_db()
        db = storage_module.get_db()
        # As a studio.db from before the book jobs: no job tables, no migration 7, a document in the Reader.
        db.execute("DROP TABLE assistant_pieces")
        db.execute("DROP TABLE assistant_jobs")
        db.execute("DELETE FROM schema_migrations WHERE version = 7")
        db.execute(
            "INSERT INTO reader_documents (id, title, source_type, content_hash, char_count, created_at, updated_at) "
            "VALUES ('kept', 'Kept', 'text', 'hash', 1, 'then', 'then')"
        )

        storage_module.init_db()

        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert {"assistant_jobs", "assistant_pieces"} <= tables
        assert db.execute("SELECT 1 FROM schema_migrations WHERE version = 7").fetchone() is not None
        assert [row[0] for row in db.execute("SELECT id FROM reader_documents")] == ["kept"]
    finally:
        if storage_module._conn is not None:
            storage_module._conn.close()


# The book jobs' table as it was before a job could take a block range.
_JOBS_BEFORE_BLOCK_RANGES = """
CREATE TABLE assistant_jobs (
  id TEXT PRIMARY KEY,
  document_id TEXT NOT NULL REFERENCES reader_documents(id) ON DELETE CASCADE,
  preset_id TEXT NOT NULL,
  settings_json TEXT NOT NULL,
  first_chapter INTEGER,
  last_chapter INTEGER,
  title TEXT NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('queued','running','done','failed','cancelled')),
  error TEXT,
  error_message TEXT,
  result_document_id TEXT REFERENCES reader_documents(id) ON DELETE SET NULL,
  created TEXT NOT NULL,
  updated TEXT NOT NULL
)
"""


def test_an_older_studio_db_gets_a_jobs_block_range_and_keeps_its_jobs(tmp_path, monkeypatch):
    # A job over some blocks keeps them in two columns a studio.db from before lacks.
    monkeypatch.setattr(storage_module.settings, "os_studio_db_path", str(tmp_path / "studio.db"))
    monkeypatch.setattr(storage_module, "_conn", None)
    try:
        storage_module.init_db()
        db = storage_module.get_db()
        db.execute(
            "INSERT INTO reader_documents (id, title, source_type, content_hash, char_count, created_at, updated_at) "
            "VALUES ('doc', 'Doc', 'text', 'hash', 1, 'then', 'then')"
        )
        db.execute("DROP TABLE assistant_pieces")
        db.execute("DROP TABLE assistant_jobs")
        db.execute(_JOBS_BEFORE_BLOCK_RANGES)
        db.execute(
            "INSERT INTO assistant_jobs (id, document_id, preset_id, settings_json, title, status, created, updated) "
            "VALUES ('old-job', 'doc', 'book', '{}', 'Doc: Book', 'done', 'then', 'then')"
        )
        db.execute("DELETE FROM schema_migrations WHERE version = 8")

        storage_module.init_db()
        storage_module.init_db()  # and again: nothing left to do

        columns = [row[1] for row in db.execute("PRAGMA table_info(assistant_jobs)")]
        assert {"first_block", "last_block"} <= set(columns)
        rows = db.execute("SELECT id, status, first_block, last_block FROM assistant_jobs").fetchall()
        assert [tuple(row) for row in rows] == [("old-job", "done", None, None)]
        assert db.execute("SELECT 1 FROM schema_migrations WHERE version = 8").fetchone() is not None
    finally:
        if storage_module._conn is not None:
            storage_module._conn.close()


# ── Send anything to the AI: POST /api/assistant/ask ──
#
# docs/AI-ASSISTANT.md, "Ask AI: send anything to the AI": a document (all of it, some chapters,
# some blocks, or a selection), a transcript from history or a long text goes to the AI. One that fits in one piece
# is answered at once; a longer one becomes a job on the book jobs' runner, generalised from "a document's chapters"
# to "any list of blocks", with the same progress, cancel, retry, restart and one-slot rules, and the Reader's job
# routes. Typed text in, text and speech out are in tests/test_assistant.py.

ASK = "/api/assistant/ask"


def _ask(client: TestClient, **body):
    return client.post(ASK, json=body)


def _from(document_id: str, **part) -> dict:
    return {"kind": "document", "document_id": document_id, **part}


def test_ask_answers_a_chapter_of_one_piece_at_once_with_its_title_and_chapter(books):
    client, fake = books
    source = _paste(client)

    response = _ask(client, input=_from(source, chapters=[1, 1]), overrides={"user_template": "{title}, {chapter}: {text}"})

    assert response.status_code == 200, response.text
    answer = response.json()
    assert (answer["reply"], answer["preset"]) == ("Reworked: Notes, Two: The third paragraph.", "book")
    assert answer["input"] == {"kind": "document", "chars": len(THIRD_PIECE)}
    assert fake.sent == ["Notes, Two: The third paragraph."]
    assert fake.calls[0][0][0] == {"role": "system", "content": presets.BOOK_SYSTEM_PROMPT}
    assert "document_id" not in answer and client.get("/v1/reader/assistant-jobs").json()["jobs"] == []


def test_ask_sends_a_selection_and_saves_the_reply_as_a_reader_document(books):
    client, fake = books
    source = _paste(client)

    response = _ask(client, input=_from(source, block_start=1, block_end=2), output={"document": {}})

    assert response.status_code == 200, response.text
    answer = response.json()
    named = f'From "Notes", chapter "One":\n\n{FIRST_PIECE}'  # the document's name and chapter first
    assert fake.sent == [named] and answer["reply"] == f"Reworked: {named}"
    result = _document(client, answer["document_id"])
    assert (result["title"], result["source_type"]) == ("Notes: Book", "assistant")
    assert [chapter["title"] for chapter in result["chapters"]] == ["One"]
    assert [(block["kind"], block["text"]) for block in result["blocks"]] == [
        (PARAGRAPH, 'Reworked: From "Notes", chapter "One":'), (PARAGRAPH, "The first paragraph."),
        (PARAGRAPH, "The second paragraph."),
    ]
    # The same blocks as a range.
    again = _ask(client, input=_from(source, blocks=[1, 2]))
    assert again.status_code == 200 and fake.sent[-1] == named


def test_ask_saves_the_same_reply_under_another_title_as_its_own_document(books):
    # A document input's reply is saved under its source
    # chapter's title, so the same reply asked again with another title was handed the earlier document and name.
    client, _fake = books
    source = _paste(client)
    ask = {"input": _from(source, chapters=[1, 1])}

    first = _ask(client, **ask, output={"document": {"title": "Summary"}})
    second = _ask(client, **ask, output={"document": {"title": "Key points"}})
    same = _ask(client, **ask, output={"document": {"title": "Key points"}})

    assert all(response.status_code == 200 for response in (first, second, same)), (first.text, second.text)
    assert _document(client, first.json()["document_id"])["title"] == "Notes: Summary"
    assert _document(client, second.json()["document_id"])["title"] == "Notes: Key points"
    assert same.json()["document_id"] == second.json()["document_id"]


def test_ask_saves_a_typed_questions_answer_as_a_document_under_its_title(books):
    client, _fake = books

    titled = _ask(client, input={"kind": "text", "text": "What is a haiku?"}, output={"document": {"title": "Haiku"}})
    untitled = _ask(client, input={"kind": "text", "text": "Name a colour."}, output={"text": True, "document": {}})

    assert (titled.status_code, untitled.status_code) == (200, 200), (titled.text, untitled.text)
    result = _document(client, titled.json()["document_id"])
    assert (result["title"], [block["text"] for block in result["blocks"]]) == ("Haiku", ["Reworked: What is a haiku?"])
    assert _document(client, untitled.json()["document_id"])["title"] == "Voice: Name a colour."


def test_ask_reads_a_transcript_from_history_on_the_server(books):
    from src.history import HistoryManager

    client, fake = books
    entry = HistoryManager().log_stt("whisper", "memo.wav", "Buy milk and call Sam.")

    response = _ask(client, input={"kind": "transcript", "history_id": entry})
    missing = _ask(client, input={"kind": "transcript", "history_id": "no-such-entry"})

    assert response.status_code == 200, response.text
    assert fake.sent == ["Buy milk and call Sam."] and response.json()["input"] == {"kind": "transcript", "chars": 22}
    assert missing.status_code == 404, missing.text
    assert (missing.json()["error"]["code"], missing.json()["error"]["param"]) == ("history_not_found", "input.history_id")
    assert len(fake.calls) == 1


def test_ask_sends_a_document_of_several_pieces_as_a_job_on_the_book_jobs_runner(books):
    client, fake = books
    source = _paste(client)

    started = _ask(client, input=_from(source))

    assert started.status_code == 202, started.text
    body = started.json()
    assert (body["preset"], body["input"]) == ("book", {"kind": "document", "chars": len(FIRST_PIECE) + len(THIRD_PIECE)})
    job = body["job"]
    assert (job["document_id"], job["preset"]) == (source, "book") and job["status"] in ("queued", "running")
    assert (job["chapters_requested"], job["blocks_requested"]) == (None, None)
    done = _wait(client, job["id"])
    assert done["status"] == "done" and fake.sent == [
        f'From "Notes", chapter "One":\n\n{FIRST_PIECE}', f'From "Notes", chapter "Two":\n\n{THIRD_PIECE}',
    ], "each piece names the document and its own chapter"
    assert _document(client, done["result_document_id"])["title"] == "Notes: Book"
    assert [listed["id"] for listed in client.get("/v1/reader/assistant-jobs").json()["jobs"]] == [job["id"]]


def test_ask_sends_a_long_selection_as_a_job_over_just_its_blocks(books):
    client, fake = books
    paragraphs = [_paragraph(n) for n in range(5)]
    source = _paste(client, "# Long\n\n" + "\n\n".join(paragraphs))  # block 0 is the heading, 1 to 5 the paragraphs

    started = _ask(client, input=_from(source, block_start=2, block_end=4), overrides={"chunk_chars": 1000},
                   output={"document": {"title": "Middle"}})

    assert started.status_code == 202, started.text
    job = started.json()["job"]
    assert (job["blocks_requested"], job["chapters_requested"]) == ([2, 4], None)
    done = _wait(client, job["id"])
    assert done["status"] == "done", done
    # One chapter: the document's name alone, and not counted in chunk_chars.
    assert fake.sent == [f'From "Notes":\n\n{paragraphs[1]}\n\n{paragraphs[2]}', f'From "Notes":\n\n{paragraphs[3]}']
    assert [(chapter["title"], chapter["pieces"], chapter["pieces_done"]) for chapter in done["chapters"]] == [("Long", 2, 2)]
    result = _document(client, done["result_document_id"])
    assert result["title"] == "Notes: Middle"
    assert [block["text"] for block in result["blocks"]] == [
        'Reworked: From "Notes":', paragraphs[1], paragraphs[2], 'Reworked: From "Notes":', paragraphs[3],
    ]


def test_ask_keeps_a_long_text_in_the_reader_and_sending_it_again_sends_only_whats_missing(books):
    client, fake = books
    paragraphs = [_paragraph(n) for n in range(3)]
    body = {
        "preset": "voice", "input": {"kind": "transcript", "text": "\n\n".join(paragraphs)},
        "overrides": {"chunk_chars": 1000},
    }

    first = client.post(ASK, json=body)

    assert first.status_code == 202, first.text
    job = _wait(client, first.json()["job"]["id"])
    assert job["status"] == "done" and fake.sent == [f"{paragraphs[0]}\n\n{paragraphs[1]}", paragraphs[2]]
    kept = _document(client, job["document_id"])
    assert kept["title"] == "Transcript" and [block["text"] for block in kept["blocks"]] == paragraphs
    # A job keeps the preset's own prompt and waits on no one: the job timeout.
    assert fake.calls[0][0][0] == {"role": "system", "content": presets.VOICE_SYSTEM_PROMPT}
    assert job["settings"]["timeout_s"] == 600.0

    again = client.post(ASK, json=body)

    assert again.status_code == 202, again.text
    second = _wait(client, again.json()["job"]["id"])
    assert second["status"] == "done" and second["document_id"] == job["document_id"]
    assert len(fake.calls) == 2, "the finished pieces were kept"


def test_ask_jobs_wait_for_the_one_slot_and_cancel_through_the_readers_route(books):
    client, fake = books
    book = _paste(client)
    fake.hold(lambda text: text == FIRST_PIECE)
    running = _start(client, book).json()
    assert fake.waiting.wait(WAIT_S)

    queued = _ask(client, input={"kind": "text", "text": "\n\n".join(_paragraph(n) for n in range(3))},
                  overrides={"chunk_chars": 1000})

    assert queued.status_code == 202, queued.text
    ask_job = queued.json()["job"]
    assert ask_job["status"] == "queued" and _job(client, running["id"])["status"] == "running"
    cancelled = client.delete(f"/v1/reader/assistant-jobs/{ask_job['id']}")
    assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"
    fake.release()
    assert _wait(client, running["id"])["status"] == "done"
    assert fake.sent == [FIRST_PIECE, THIRD_PIECE], "the cancelled job sent nothing"


def test_ask_refuses_a_missing_document_a_range_outside_it_and_a_range_with_nothing_to_send(books):
    client, fake = books
    source = _paste(client)  # chapters 0 and 1; blocks 0 to 4, block 3 the heading "Two"

    missing = _ask(client, input=_from("no-such-document"))

    assert missing.status_code == 404, missing.text
    assert (missing.json()["error"]["code"], missing.json()["error"]["param"]) == ("document_not_found", "input.document_id")
    for part, param in (
        ({"chapters": [0, 2]}, "input.chapters"),
        ({"blocks": [0, 5]}, "input.blocks"),
        ({"block_start": 4, "block_end": 9}, "input.block_end"),
        ({"blocks": [3, 3]}, "input.blocks"),
    ):
        response = _ask(client, input=_from(source, **part))
        assert response.status_code == 422, (part, response.text)
        assert response.json()["error"]["param"] == param, part
    assert fake.calls == [] and client.get("/v1/reader/assistant-jobs").json()["jobs"] == []


# ── A long text or transcript reaches the AI as it was sent ──
#
# Inputs are typed `text` or a `transcript`, and long inputs become a job: the length changes how the input is
# sent, not what is sent. A long text is kept as
# a Reader document (the job points at it, and the same text again reuses it), which reads as
# pasting the text into the Reader does; the job sends the text itself, cut at blank lines, never parsed as Markdown.

CODE = "def area(r):\n    return 3.14159 * r ** 2"
MARKDOWN_QUESTION = (
    "# Release notes\n\n"
    "Explain what this function does:\n\n"
    f"```python\n{CODE}\n```\n\n"
    "See [the docs](https://example.com/area) and the <b>bold</b> part.\n"
    "Line two of the same paragraph.\n\n"
    "2024. The year this paragraph opens with.\n\n"
)
LONG_MARKDOWN = MARKDOWN_QUESTION + "\n\n".join(_paragraph(n) for n in range(3))


def _ask_job(client: TestClient, text: str, kind: str = "text", **body) -> dict:
    started = _ask(client, input={"kind": kind, "text": text}, overrides={"chunk_chars": 1000}, **body)
    assert started.status_code == 202, started.text
    return _wait(client, started.json()["job"]["id"])


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
@pytest.mark.parametrize("kind", ["text", "transcript"])
def test_ask_sends_a_long_text_as_it_was_sent_in_pieces_cut_at_blank_lines(books, kind, newline):
    # The heading, the code, the link's target, the HTML, the line break and the
    # "2024. " all reach the AI, as they do for a text of one piece.
    client, fake = books
    text = LONG_MARKDOWN.replace("\n", newline)
    blank = newline * 2

    job = _ask_job(client, text, kind)

    assert job["status"] == "done", job
    first = MARKDOWN_QUESTION.replace("\n", newline) + _paragraph(0)
    assert fake.sent == [first, _paragraph(1) + blank + _paragraph(2)]
    assert blank.join(fake.sent) == text, "the pieces are the text, cut at a blank line"
    for part in ("# Release notes", f"```python{newline}{CODE.replace(chr(10), newline)}{newline}```",
                 "[the docs](https://example.com/area)", "<b>bold</b>", f"part.{newline}Line two", "2024. The year"):
        assert part in fake.sent[0], part
    assert "Code block skipped." not in "".join(fake.sent)
    at_once = _ask(client, input={"kind": kind, "text": text}, overrides={"chunk_chars": 200000})
    assert at_once.status_code == 200 and fake.sent[-1] == text, "in one piece, the same text as typed"


def test_ask_keeps_a_long_text_as_the_reader_reads_it_pasted_and_shows_one_part_of_pieces(books):
    # Only what the AI gets changes. The Reader document the text is kept as is the
    # one pasting it into the Reader makes, for listening; the job's view and settings don't carry the text.
    from src.reader.sources import parse_pasted

    client, _fake = books

    job = _ask_job(client, LONG_MARKDOWN)

    assert job["status"] == "done", job
    kept = _document(client, job["document_id"])
    pasted = parse_pasted(LONG_MARKDOWN)
    assert (kept["title"], kept["source_type"]) == ("Release notes", "markdown")
    assert [(block["kind"], block["text"]) for block in kept["blocks"]] == [
        (block.kind, block.text) for chapter in pasted.chapters for block in chapter.blocks
    ]
    assert "Code block skipped." in [block["text"] for block in kept["blocks"]]
    assert _paste(client, LONG_MARKDOWN) == job["document_id"], "pasting the same text is the same document"
    assert [(c["index"], c["title"], c["pieces"], c["pieces_done"], c["status"]) for c in job["chapters"]] == [
        (0, "Release notes", 2, 2, "done"),
    ]
    assert job["chapters"][0]["text"] == f"Reworked: {MARKDOWN_QUESTION}{_paragraph(0)}\n\nReworked: {_paragraph(1)}\n\n{_paragraph(2)}"
    assert set(job["settings"]) == set(presets.EffectiveSettings().to_dict())
    summary = {name: value for name, value in job.items() if name != "chapters"}  # the chapters carry the answers
    assert "https://example.com/area" not in json.dumps(summary) + client.get("/v1/reader/assistant-jobs").text
    result = _document(client, job["result_document_id"])
    assert result["title"] == "Release notes: Voice"
    assert [block["text"] for block in result["blocks"]][:2] == ["Reworked: # Release notes", "Explain what this function does:"]


def test_ask_sends_each_long_text_its_own_pieces_when_the_reader_keeps_both_as_one_document(books):
    # Two texts that differ only where the Reader's import drops something (a
    # link's target) are one Reader document, yet each reaches the AI as it was sent: a piece's saved answer is used
    # again only for the same piece of text.
    client, fake = books
    tail = "\n\n" + "\n\n".join(_paragraph(n) for n in range(3))
    one = "See [the docs](https://example.com/one)." + tail
    two = "See [the docs](https://example.com/two)." + tail

    first = _ask_job(client, one)
    second = _ask_job(client, two)

    assert (first["status"], second["status"]) == ("done", "done"), (first, second)
    assert first["document_id"] == second["document_id"]
    head = f"\n\n{_paragraph(0)}\n\n{_paragraph(1)}"
    assert fake.sent == [
        f"See [the docs](https://example.com/one).{head}", _paragraph(2),
        f"See [the docs](https://example.com/two).{head}",
    ], "the last piece is the same text in both, so its answer is used again"
    assert _document(client, second["result_document_id"])["blocks"][0]["text"] == "Reworked: See [the docs](https://example.com/two)."


def test_a_long_text_job_that_failed_sends_only_the_rest_when_it_is_sent_again(books):
    # The text is kept with its job, so a job of it resumes as a book job does.
    client, fake = books
    second_piece = f"{_paragraph(1)}\n\n{_paragraph(2)}"

    def answer(text):
        if text == second_piece:
            return AssistantError("The AI server answered HTTP 500: overloaded", ASSISTANT_HTTP_ERROR)
        return _reworked(text)

    fake.answer = answer
    failed = _ask_job(client, LONG_MARKDOWN)

    assert failed["status"] == "failed" and failed["error"] == ASSISTANT_HTTP_ERROR, failed
    assert [(c["pieces"], c["pieces_done"]) for c in failed["chapters"]] == [(2, 1)]
    fake.answer = _reworked
    again = _ask_job(client, LONG_MARKDOWN)

    assert again["status"] == "done" and again["document_id"] == failed["document_id"], again
    assert fake.sent == [MARKDOWN_QUESTION + _paragraph(0), second_piece, second_piece, second_piece]
    assert again["chapters"][0]["text"] == f"Reworked: {MARKDOWN_QUESTION}{_paragraph(0)}\n\nReworked: {second_piece}"


def test_a_long_texts_pieces_fill_the_template_as_a_short_text_does(books):
    # {title}, {author} and {chapter} are a document's; a text of one piece leaves
    # them empty, and so does each piece of a long one (not the title the Reader's import found in it).
    client, fake = books
    template = {"user_template": "[{title}|{author}|{chapter}] {text}"}  # the book preset takes these

    short = _ask(client, preset="book", input={"kind": "text", "text": "Why?"}, overrides=template)
    started = _ask(client, preset="book", input={"kind": "text", "text": LONG_MARKDOWN},
                   overrides={**template, "chunk_chars": 1000})

    assert short.status_code == 200, short.text
    assert started.status_code == 202, started.text
    assert _wait(client, started.json()["job"]["id"])["status"] == "done"
    assert fake.sent[0] == "[||] Why?"
    assert fake.sent[1:] == [f"[||] {MARKDOWN_QUESTION}{_paragraph(0)}", f"[||] {_paragraph(1)}\n\n{_paragraph(2)}"]


def test_a_text_is_cut_at_blank_lines_then_line_breaks_then_sentences_each_piece_as_it_was_sent():
    # Nothing is parsed or cleaned; only the whitespace between two pieces is left
    # out, and a piece keeps its first line's indentation.
    lines = [f"Line {n}: " + "word " * 30 for n in range(10)]  # 158 characters each
    sentences = " ".join(f"Sentence {n} of a paragraph with no line break." for n in range(40))
    opening = "  An indented first line\nand its second, <i>as typed</i>."
    text = f"\n\n{opening}\n\n\n" + "\n".join(lines) + "\r\n \t\r\n" + sentences + "\n\n```\n{ }\n```\n"

    pieces = assistant_jobs.cut_text(text, 1000)

    assert pieces[:3] == [opening, "\n".join(lines[:6]), "\n".join(lines[6:])]
    middle = pieces[3:-1]
    assert len(middle) == 2 and all(piece.startswith("Sentence") and piece.endswith("break.") for piece in middle)
    assert " ".join(middle) == sentences
    assert pieces[-1] == "```\n{ }\n```\n"
    assert all(len(piece) <= 1000 for piece in pieces)
    position = 0
    for piece in pieces:  # each a stretch of the text in order, with only whitespace between
        start = text.index(piece, position)
        assert not text[position:start].strip(), text[position:start]
        position = start + len(piece)
    assert position == len(text)
    assert assistant_jobs.cut_text("One.\n\nTwo.", 1000) == ["One.\n\nTwo."]
    assert assistant_jobs.cut_text(" \n\n \n", 1000) == []
    # One line break is no blank line, whichever the text uses: a paragraph too long for a piece is cut at its line
    # breaks and never joins the paragraph before it.
    before, first_line, second_line = "B" * 100, "L" * 600, "M" * 600
    for newline in ("\n", "\r\n", "\r"):
        text = f"{before}{newline}{newline}{first_line}{newline}{second_line}"
        assert assistant_jobs.cut_text(text, 1000) == [before, first_line, second_line], repr(newline)


# ── A question, one sentence, and the document's name and chapter ──
#
# A `question` goes before the text (or the rendered template), as "<question>\n\n<text>", in every piece of a job
# too; a `document` input may name `block` and `sentence`, the sentence's index in that block as the Reader's sentence
# offsets count them, answered at once; every `document` input's message starts with `From "<title>", chapter
# "<chapter>":` (the chapter when the document has chapters), unless the preset's template uses {title} or {chapter};
# the answer's `input` counts the question and says the sentence's place, never their text.

SENTENCES = "# One\n\nCall me Ishmael. Some years ago, I went to sea. It was cold.\n\n# Two\n\nThe end."


def test_ask_sends_one_sentence_of_a_block_as_the_reader_counts_its_sentences(books):
    client, fake = books
    source = _paste(client, SENTENCES, title="Moby-Dick")  # blocks: 0 "One", 1 the paragraph, 2 "Two", 3 "The end."
    blocks = _document(client, source)["blocks"]
    spans = blocks[1]["sentences"]  # what the Reader reads by and highlights
    assert [blocks[1]["text"][start:end] for start, end in spans] == [
        "Call me Ishmael.", "Some years ago, I went to sea.", "It was cold.",
    ]

    response = _ask(client, input=_from(source, block=1, sentence=1))

    assert response.status_code == 200, response.text
    assert fake.sent == ['From "Moby-Dick", chapter "One":\n\nSome years ago, I went to sea.']
    assert response.json()["input"] == {"kind": "document", "chars": 30, "block": 1, "sentence": 1}
    assert client.get("/v1/reader/assistant-jobs").json()["jobs"] == [], "a sentence is answered at once"
    # The last sentence, a heading (a block of one sentence) and a block of the next chapter.
    for block, sentence, sent in ((1, 2, 'From "Moby-Dick", chapter "One":\n\nIt was cold.'),
                                  (2, 0, 'From "Moby-Dick", chapter "Two":\n\nTwo'),
                                  (3, 0, 'From "Moby-Dick", chapter "Two":\n\nThe end.')):
        response = _ask(client, input=_from(source, block=block, sentence=sentence))
        assert response.status_code == 200 and fake.sent[-1] == sent, (block, sentence, response.text)
    # With a question, a template of its own (which says the chapter itself), and saved as a document.
    asked = _ask(client, input=_from(source, block=1, sentence=0), question="Who says this?",
                 overrides={"user_template": "[{chapter}] {text}"}, output={"document": {}})
    assert asked.status_code == 200, asked.text
    assert fake.sent[-1] == "Who says this?\n\n[One] Call me Ishmael."
    assert asked.json()["input"] == {"kind": "document", "chars": 16, "question_chars": 14, "block": 1, "sentence": 0}
    saved = _document(client, asked.json()["document_id"])
    assert (saved["title"], [c["title"] for c in saved["chapters"]]) == ("Moby-Dick: Book", ["One"])
    assert [block["text"] for block in saved["blocks"]] == ["Reworked: Who says this?", "[One] Call me Ishmael."]


def test_ask_refuses_a_sentence_outside_its_block_or_a_block_outside_the_document(books):
    client, fake = books
    source = _paste(client, SENTENCES, title="Moby-Dick")

    for part, param, words in (({"block": 1, "sentence": 3}, "input.sentence", "3 sentences"),
                               ({"block": 3, "sentence": 1}, "input.sentence", "1 sentence"),
                               ({"block": 4, "sentence": 0}, "input.block", "0 to 3")):
        response = _ask(client, input=_from(source, **part))
        assert response.status_code == 422, (part, response.text)
        error = response.json()["error"]
        assert (error["code"], error["param"]) == ("validation_error", param), (part, error)
        assert words in error["message"], (part, error["message"])
    missing = _ask(client, input=_from("no-such-document", block=0, sentence=0))
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "document_not_found"
    assert fake.calls == []


def test_ask_names_the_document_and_its_chapter_unless_the_template_does(books):
    client, fake = books
    notes = _paste(client)  # "Notes": chapters "One" and "Two"
    memo = _paste(client, "Just one paragraph here.", title="Memo")  # one chapter: no chapter named
    release = _paste(client, "# Release notes\n\nIntro.\n\n## Fixes\n\nA fix.", title="")  # titled by its heading
    uploaded = client.post("/v1/reader/documents", files={"file": ("trip_plan.txt", b"Pack the tent.\n\nBuy maps.", "text/plain")})
    assert uploaded.status_code == 201, uploaded.text
    assert [c["title"] for c in _document(client, release)["chapters"]] == ["Release notes", "Fixes"]

    for body, sent in (
        ({"input": _from(notes, chapters=[1, 1])}, 'From "Notes", chapter "Two":\n\nThe third paragraph.'),
        ({"input": _from(notes, blocks=[4, 4]), "preset": "voice"}, 'From "Notes", chapter "Two":\n\nThe third paragraph.'),
        ({"input": _from(memo)}, 'From "Memo":\n\nJust one paragraph here.'),
        ({"input": _from(uploaded.json()["id"])}, 'From "trip plan":\n\nPack the tent.\n\nBuy maps.'),  # its file name
        ({"input": _from(release, chapters=[0, 0])}, 'From "Release notes":\n\nIntro.'),  # the chapter is the title
        ({"input": _from(release, chapters=[1, 1])}, 'From "Release notes", chapter "Fixes":\n\nA fix.'),
        # A template of the preset's own goes after the line; one that says the title or the chapter gets none.
        ({"input": _from(notes, chapters=[1, 1]), "overrides": {"user_template": "Rework: {text}"}},
         'From "Notes", chapter "Two":\n\nRework: The third paragraph.'),
        ({"input": _from(notes, chapters=[1, 1]), "overrides": {"user_template": "By {author}: {text}"}},
         'From "Notes", chapter "Two":\n\nBy : The third paragraph.'),
        ({"input": _from(notes, chapters=[1, 1]), "overrides": {"user_template": "({chapter}) {text}"}},
         "(Two) The third paragraph."),
        ({"input": _from(notes, chapters=[1, 1]), "overrides": {"user_template": "{title}: {text}"}},
         "Notes: The third paragraph."),
        # The question first, then the line, then the text.
        ({"input": _from(notes, chapters=[1, 1]), "question": "Explain this to me."},
         'Explain this to me.\n\nFrom "Notes", chapter "Two":\n\nThe third paragraph.'),
    ):
        response = _ask(client, **body)
        assert response.status_code == 200, (body, response.text)
        assert fake.sent[-1] == sent, body
    assert len(fake.calls) == 11


def test_ask_a_part_spanning_chapters_names_each_pieces_own_chapter(books):
    client, fake = books
    source = _paste(client)  # blocks: 0 "One", 1 and 2 in chapter One, 3 "Two", 4 in chapter Two

    started = _ask(client, input=_from(source, block_start=2, block_end=4))

    assert started.status_code == 202, started.text  # a piece for each chapter, so a job
    assert _wait(client, started.json()["job"]["id"])["status"] == "done"
    assert fake.sent == ['From "Notes", chapter "One":\n\nThe second paragraph.',
                         'From "Notes", chapter "Two":\n\nThe third paragraph.']
    # Text in one chapter only (the next chapter's opening heading isn't sent) names that chapter.
    one = _ask(client, input=_from(source, blocks=[2, 3]))
    assert one.status_code == 200 and fake.sent[-1] == 'From "Notes", chapter "One":\n\nThe second paragraph.'


def test_ask_sends_the_question_with_every_piece_of_a_job_and_a_new_question_sends_them_again(books):
    client, fake = books
    source = _paste(client)
    body = {"input": _from(source), "question": "What happens here?"}

    started = _ask(client, **body)

    assert started.status_code == 202, started.text
    chars = len(FIRST_PIECE) + len(THIRD_PIECE)
    assert started.json()["input"] == {"kind": "document", "chars": chars, "question_chars": 18}
    job = _wait(client, started.json()["job"]["id"])
    assert job["status"] == "done", job
    asked = [f'What happens here?\n\nFrom "Notes", chapter "One":\n\n{FIRST_PIECE}',
             f'What happens here?\n\nFrom "Notes", chapter "Two":\n\n{THIRD_PIECE}']
    assert fake.sent == asked
    # The question is what is sent, not a setting, and only the answers show it.
    assert set(job["settings"]) == set(presets.EffectiveSettings().to_dict())
    summary = json.dumps({name: value for name, value in job.items() if name != "chapters"})
    assert "What happens" not in summary + client.get("/v1/reader/assistant-jobs").text

    # The same question again keeps the answers; another question, or none, sends every piece again.
    for again, sent in ((body, []),
                        ({"input": _from(source), "question": "Who is in it?"},
                         [text.replace("What happens here?", "Who is in it?") for text in asked]),
                        ({"input": _from(source)}, [text.split("\n\n", 1)[1] for text in asked])):
        before = len(fake.calls)
        restarted = _ask(client, **again)
        assert restarted.status_code == 202, restarted.text
        assert _wait(client, restarted.json()["job"]["id"])["status"] == "done"
        assert fake.sent[before:] == sent, again
    # The Reader's own book jobs are as they were: no question and no line, so none of these answers fit.
    book = _run(client, source)
    assert book["status"] == "done" and fake.sent[-2:] == [FIRST_PIECE, THIRD_PIECE]


def test_ask_sends_the_question_with_every_piece_of_a_long_text_and_names_no_document(books):
    client, fake = books

    job = _ask_job(client, LONG_MARKDOWN, question="Explain each part.")

    assert job["status"] == "done", job
    assert fake.sent == [f"Explain each part.\n\n{MARKDOWN_QUESTION}{_paragraph(0)}",
                         f"Explain each part.\n\n{_paragraph(1)}\n\n{_paragraph(2)}"]
    assert set(job["settings"]) == set(presets.EffectiveSettings().to_dict())
    again = _ask_job(client, LONG_MARKDOWN)
    assert again["status"] == "done" and len(fake.calls) == 4, "without the question, each piece is sent again"
    assert fake.sent[2:] == [f"{MARKDOWN_QUESTION}{_paragraph(0)}", f"{_paragraph(1)}\n\n{_paragraph(2)}"]


# Preset fields that never change what the AI is sent for a piece, so a piece's saved answer stays valid without them.
NOT_SENT_TO_THE_AI = {
    "show_thinking", "timeout_s", "history_turns", "fillers", "filler_after_ms", "filler_every_ms", "filler_max",
    "speak_errors", "error_sentences", "chunk_chars",
}


def test_a_pieces_saved_answer_is_keyed_on_every_preset_field_the_ai_is_sent():
    """A piece's key holds every preset field that
    reaches the AI request, so re-running a job with only another thinking budget asks the AI again instead of reusing
    the saved answers. A new preset field must be added to one side or the other. A field added later counts only when
    set, so a piece saved before it keeps its key."""
    keyed = set(assistant_jobs.AI_FIELDS) | set(assistant_jobs.AI_FIELDS_WHEN_SET)
    assert keyed == set(presets.TUNABLE_FIELDS) - NOT_SENT_TO_THE_AI
    piece = Piece(chapter_index=0, first_block=0, last_block=0, text="A paragraph.")
    settings = presets.EffectiveSettings().to_dict()
    unbudgeted = assistant_jobs.piece_key("doc", piece, settings)
    old_key_settings = {name: settings[name] for name in assistant_jobs.AI_FIELDS}  # as saved before thinking_budget was a key field
    assert unbudgeted == assistant_jobs.piece_key("doc", piece, old_key_settings), "no budget: the key it always had"
    budgeted = assistant_jobs.piece_key("doc", piece, {**settings, "thinking_budget": 256})
    assert budgeted != unbudgeted
    assert budgeted != assistant_jobs.piece_key("doc", piece, {**settings, "thinking_budget": 512})
