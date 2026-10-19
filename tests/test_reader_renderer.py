from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from src import main as main_module
from src import storage as storage_module
from src.reader.document import HEADING, Block, Chapter, Document
from src.reader.media import MediaSigner
from src.reader.renderer import (
    HEADING_GAP_S,
    PARAGRAPH_GAP_S,
    SAMPLE_RATE,
    SENTENCE_GAP_S,
    PartPlan,
    ReaderRenderer,
    _Job,
)
from src.reader import store as store_module
from src.reader.store import ReaderStore

SENTENCE_SAMPLES = 2400  # 0.1 s of fake speech per sentence


class FakeVoice:
    """Stands in for synthesis: counts calls and can be held or made to fail."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.busy = False
        self.fail = False
        self.hold = threading.Event()
        self.hold.set()
        self.started = threading.Event()

    def synthesize(self, *, text, model, voice, speed, sample_rate):
        assert speed == 1.0 and sample_rate == SAMPLE_RATE
        self.calls.append(text)
        self.started.set()
        self.hold.wait(5)
        if self.fail:
            raise RuntimeError("voice exploded")
        return np.ones(SENTENCE_SAMPLES, dtype=np.float32)


@pytest.fixture
def library(tmp_path):
    main_module.settings.os_studio_db_path = str(tmp_path / "studio.db")
    storage_module._conn = None
    storage_module.init_db()
    return ReaderStore()


@pytest.fixture
def voice():
    return FakeVoice()


@pytest.fixture
def make_renderer(library, voice, tmp_path):
    renderers: list[ReaderRenderer] = []

    def make(**overrides) -> ReaderRenderer:
        options = {
            "store": library,
            "synthesize_fn": voice.synthesize,
            "encode_fn": lambda audio, sample_rate: b"MP3" + len(audio).to_bytes(4, "big"),
            "pronounce_fn": lambda text: text.replace("SQL", "sequel"),
            "busy_fn": lambda: voice.busy,
            "signer": MediaSigner("test-key"),
            "cache_dir": str(tmp_path / "reader"),
            "lookahead_parts": 3,
            "part_max_chars": 60,
        }
        options.update(overrides)
        renderer = ReaderRenderer(**options)
        renderers.append(renderer)
        return renderer

    yield make
    voice.hold.set()
    for renderer in renderers:
        renderer.stop()


def _save(library: ReaderStore, chapters: list[Chapter]) -> str:
    document_id, _created = library.save_document(Document(title="Book", source_type="text", chapters=chapters))
    return document_id


def _book(library: ReaderStore) -> str:
    return _save(library, [
        Chapter("One", [
            Block("One", HEADING),
            Block("First short block."),
            Block("Second short block."),
            Block("Third block, still short."),
        ]),
        Chapter("Two", [Block("Two", HEADING), Block("A block in the next chapter.")]),
    ])


def _wait_ready(renderer: ReaderRenderer, document_id: str, from_block: int, timeout: float = 5.0) -> list[dict]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        parts = renderer.status(document_id, from_block, "kokoro", "af_heart")
        if all(part["status"] in {"ready", "error"} for part in parts):
            return parts
        time.sleep(0.02)
    raise AssertionError(f"parts never finished: {parts}")


def test_plan_starts_with_one_block_and_never_crosses_chapters(library, make_renderer):
    document_id = _book(library)
    renderer = make_renderer(lookahead_parts=4)

    parts = renderer.plan(document_id, 0, "kokoro", "af_heart")

    assert [(part.first_block, part.last_block) for part in parts] == [(0, 0), (1, 2), (3, 3), (4, 5)]
    assert renderer.plan("missing", 0, "kokoro", "af_heart") is None
    with pytest.raises(ValueError, match="from_block"):
        renderer.plan(document_id, 99, "kokoro", "af_heart")


def test_request_renders_parts_with_sentence_timings(library, make_renderer, voice, tmp_path):
    document_id = _save(library, [Chapter("One", [Block("Hello there. SQL is fun."), Block("Next block.")])])
    renderer = make_renderer()

    renderer.request(document_id, 0, "kokoro", "af_heart")
    parts = _wait_ready(renderer, document_id, 0)

    first = parts[0]
    assert first["status"] == "ready"
    assert first["url"].startswith("/v1/reader/media/") and "sig=" in first["url"]
    assert [sentence["index"] for sentence in first["sentences"]] == [0, 1]
    assert first["sentences"][1]["start_ms"] == round((SENTENCE_SAMPLES + SENTENCE_GAP_S * SAMPLE_RATE) * 1000 / SAMPLE_RATE)
    assert first["duration_ms"] == round(
        (2 * SENTENCE_SAMPLES + int(SENTENCE_GAP_S * SAMPLE_RATE) + int(PARAGRAPH_GAP_S * SAMPLE_RATE)) * 1000 / SAMPLE_RATE
    )
    assert "SQL is fun." not in voice.calls and "sequel is fun." in voice.calls
    assert Path(library.get_part(first["id"])["path"]).is_relative_to(tmp_path / "reader" / document_id)


def test_ready_parts_are_reused_without_rendering_again(library, make_renderer, voice):
    document_id = _book(library)
    renderer = make_renderer()
    renderer.request(document_id, 0, "kokoro", "af_heart")
    first = _wait_ready(renderer, document_id, 0)
    calls = len(voice.calls)

    again = renderer.request(document_id, 0, "kokoro", "af_heart")

    assert [part["id"] for part in again] == [part["id"] for part in first]
    assert all(part["status"] == "ready" for part in again)
    time.sleep(0.1)
    assert len(voice.calls) == calls


def test_headings_get_a_longer_pause(library, make_renderer):
    document_id = _save(library, [Chapter("One", [Block("One", HEADING), Block("Body.")])])
    renderer = make_renderer(lookahead_parts=1)
    renderer.request(document_id, 0, "kokoro", "af_heart")
    heading = _wait_ready(renderer, document_id, 0)[0]
    assert heading["duration_ms"] == round((SENTENCE_SAMPLES + int(HEADING_GAP_S * SAMPLE_RATE)) * 1000 / SAMPLE_RATE)


def test_skip_abandons_the_part_being_rendered(library, make_renderer, voice):
    document_id = _save(library, [
        Chapter("One", [Block("Sentence one. Sentence two. Sentence three."), Block("Middle."), Block("Far away block.")]),
    ])
    renderer = make_renderer(lookahead_parts=1)
    voice.hold.clear()
    old = renderer.request(document_id, 0, "kokoro", "af_heart")[0]
    assert voice.started.wait(5)

    renderer.request(document_id, 2, "kokoro", "af_heart")
    voice.hold.set()
    skipped_to = _wait_ready(renderer, document_id, 2)[0]

    assert skipped_to["status"] == "ready"
    assert library.get_part(old["id"]) is None
    assert voice.calls.count("Sentence two.") == 0


def test_rendering_waits_while_a_live_reader_session_is_active(library, make_renderer, voice):
    document_id = _book(library)
    renderer = make_renderer(lookahead_parts=1)
    voice.busy = True

    renderer.request(document_id, 0, "kokoro", "af_heart")
    time.sleep(0.4)
    assert voice.calls == []
    assert renderer.status(document_id, 0, "kokoro", "af_heart")[0]["status"] == "rendering"

    voice.busy = False
    assert _wait_ready(renderer, document_id, 0)[0]["status"] == "ready"


def test_failed_part_reports_error_and_retries_on_next_request(library, make_renderer, voice):
    document_id = _book(library)
    renderer = make_renderer(lookahead_parts=1)
    voice.fail = True
    renderer.request(document_id, 0, "kokoro", "af_heart")
    failed = _wait_ready(renderer, document_id, 0)[0]
    assert failed["status"] == "error" and "voice exploded" in failed["error"]

    voice.fail = False
    renderer.request(document_id, 0, "kokoro", "af_heart")
    assert _wait_ready(renderer, document_id, 0)[0]["status"] == "ready"


def test_cache_limit_prunes_old_parts_but_keeps_the_current_window(library, make_renderer, monkeypatch):
    monkeypatch.setattr(store_module, "RECENTLY_PLAYED_S", 0)
    document_id = _book(library)
    renderer = make_renderer(lookahead_parts=2, cache_max_bytes=10)
    renderer.request(document_id, 0, "kokoro", "af_heart")
    early = _wait_ready(renderer, document_id, 0)
    early_paths = [library.get_part(part["id"])["path"] for part in early]

    renderer.request(document_id, 4, "kokoro", "af_heart")
    late = _wait_ready(renderer, document_id, 4)

    assert all(part["status"] == "ready" for part in late)
    assert all(library.get_part(part["id"]) is None for part in early)
    assert not any(Path(path).exists() for path in early_paths)


def test_part_finished_after_delete_is_discarded(library, make_renderer, voice, tmp_path):
    document_id = _book(library)
    renderer = make_renderer(lookahead_parts=1)
    voice.hold.clear()
    renderer.request(document_id, 1, "kokoro", "af_heart")
    assert voice.started.wait(5)

    assert library.delete_document(document_id) == []
    voice.hold.set()
    renderer.stop()

    assert list((tmp_path / "reader").rglob("*.mp3")) == []


def test_recently_played_parts_survive_pruning(library, make_renderer):
    document_id = _book(library)
    renderer = make_renderer(lookahead_parts=2, cache_max_bytes=10)
    renderer.request(document_id, 0, "kokoro", "af_heart")
    early = _wait_ready(renderer, document_id, 0)
    renderer.request(document_id, 4, "kokoro", "af_heart")
    _wait_ready(renderer, document_id, 4)
    assert all(library.get_part(part["id"]) is not None for part in early)


def test_worker_drops_a_job_whose_document_vanished(library, make_renderer, voice):
    document_id = _book(library)
    renderer = make_renderer(lookahead_parts=1)
    renderer.encode_fn = counting = _CountingEncoder()
    library.delete_document(document_id)
    with renderer._cond:
        renderer._job = _Job(document_id, "kokoro", "af_heart", [PartPlan("gone", 0, 0)])
        renderer._ensure_thread()
        renderer._cond.notify_all()
    time.sleep(0.3)
    assert counting.calls == 0
    assert renderer._job is None
    assert voice.calls == []


def test_startup_sweep_removes_orphans_and_forgets_missing_files(library, make_renderer, tmp_path):
    document_id = _book(library)
    renderer = make_renderer(lookahead_parts=2)
    renderer.request(document_id, 0, "kokoro", "af_heart")
    kept, missing = _wait_ready(renderer, document_id, 0)
    renderer.stop()
    Path(library.get_part(missing["id"])["path"]).unlink()
    orphan = tmp_path / "reader" / document_id / ".half-written.tmp"
    orphan.write_bytes(b"x")
    empty = tmp_path / "reader" / "deleted-document"
    empty.mkdir()

    fresh = make_renderer(lookahead_parts=2)
    fresh.request(document_id, 3, "kokoro", "af_heart")
    _wait_ready(fresh, document_id, 3)

    assert library.get_part(missing["id"]) is None
    assert library.get_part(kept["id"]) is not None
    assert not orphan.exists() and not empty.exists()


class _CountingEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, audio, sample_rate):
        self.calls += 1
        return b"MP3"
