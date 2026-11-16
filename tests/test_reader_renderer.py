from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from src import main as main_module
from src import storage as storage_module
from src.reader import renderer as renderer_module
from src.reader import store as store_module
from src.reader.document import HEADING, Block, Chapter, Document, spoken_opening
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


def test_plan_starts_short_keeps_headings_with_text_and_never_crosses_chapters(library, make_renderer):
    document_id = _book(library)
    renderer = make_renderer(lookahead_parts=4)

    parts = renderer.plan(document_id, 0, "kokoro", "af_heart")
    assert [(part.first_block, part.last_block) for part in parts] == [(0, 1), (2, 3), (4, 5)]

    from_text = renderer.plan(document_id, 1, "kokoro", "af_heart")
    assert [(part.first_block, part.last_block) for part in from_text] == [(1, 1), (2, 3), (4, 5)]

    # A heading at the end of a chapter has nothing to join and stays on its own.
    lonely = _save(library, [Chapter("One", [Block("Text.")]), Chapter("Two", [Block("Two", HEADING)])])
    assert [(p.first_block, p.last_block) for p in renderer.plan(lonely, 0, "kokoro", "af_heart")] == [(0, 0), (1, 1)]
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
    part = _wait_ready(renderer, document_id, 0)[0]
    heading_samples = SENTENCE_SAMPLES + int(HEADING_GAP_S * SAMPLE_RATE)
    assert part["sentences"][1] == {"block": 1, "index": 0, "start_ms": round(heading_samples * 1000 / SAMPLE_RATE)}
    assert part["duration_ms"] == round(
        (heading_samples + SENTENCE_SAMPLES + int(PARAGRAPH_GAP_S * SAMPLE_RATE)) * 1000 / SAMPLE_RATE
    )


@pytest.mark.parametrize(("text", "spoken"), [
    ("MR. BENNET was among the earliest.", "MR. Bennet was among the earliest."),
    ("IT is a truth universally acknowledged.", "IT is a truth universally acknowledged."),
    ("“I HAVE been thinking,” she said.", "“I Have been thinking,” she said."),
    ("MISS BINGLEY’S letter arrived.", "Miss Bingley’s letter arrived."),
    ("ÉMILE ZOLA wrote it.", "Émile Zola wrote it."),
    # Acronyms keep their capitals, and so do lines with no lowercase text after them.
    ("BBC reporters and US citizens.", "BBC reporters and US citizens."),
    ("CHISWICK PRESS: CHARLES WHITTINGHAM AND CO.", "CHISWICK PRESS: CHARLES WHITTINGHAM AND CO."),
    ("THE END. 1813", "THE END. 1813"),
    ("Nothing to change, NASA said.", "Nothing to change, NASA said."),
])
def test_spoken_opening_gives_capitalised_first_words_normal_case(text, spoken):
    assert spoken_opening(text) == spoken


def test_capitalised_opening_is_spoken_in_normal_case_but_stored_as_printed(library, make_renderer, voice):
    document_id = _save(library, [Chapter("One", [
        Block("CHAPTER BENNET", HEADING),
        Block("MR. BENNET was out. BINGLEY came in."),
    ])])
    renderer = make_renderer(lookahead_parts=1)
    renderer.request(document_id, 0, "kokoro", "af_heart")
    _wait_ready(renderer, document_id, 0)

    assert voice.calls == ["CHAPTER BENNET", "MR. Bennet was out.", "BINGLEY came in."]
    assert library.get_blocks(document_id, 1, 1)[0]["text"] == "MR. BENNET was out. BINGLEY came in."


def test_parts_from_an_older_render_version_are_rendered_again(library, make_renderer, voice, monkeypatch):
    document_id = _book(library)
    monkeypatch.setattr(renderer_module, "RENDER_VERSION", renderer_module.RENDER_VERSION - 1)
    renderer = make_renderer(lookahead_parts=1)
    renderer.request(document_id, 0, "kokoro", "af_heart")
    old = _wait_ready(renderer, document_id, 0)[0]
    old_path = Path(library.get_part(old["id"])["path"])
    calls = len(voice.calls)
    renderer.stop()
    monkeypatch.undo()   # an upgrade: the server restarts with a new renderer

    fresh = make_renderer(lookahead_parts=1)
    fresh.request(document_id, 0, "kokoro", "af_heart")
    new = _wait_ready(fresh, document_id, 0)[0]

    assert new["id"] != old["id"] and new["status"] == "ready"
    assert library.get_part(old["id"]) is None and not old_path.exists()
    assert len(voice.calls) > calls


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
    cache = tmp_path / "reader"
    orphan = cache / document_id / f".{'a' * 24}.tmp"
    orphan.write_bytes(b"x")
    empty = cache / ("b" * 32)
    empty.mkdir()
    # Anything the renderer did not write survives, even if the cache folder is shared.
    foreign = [cache / "studio.db", cache / "voices" / "me.wav", cache / document_id / "notes.txt"]
    for path in foreign:
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"keep")

    fresh = make_renderer(lookahead_parts=2)
    fresh.request(document_id, 3, "kokoro", "af_heart")
    _wait_ready(fresh, document_id, 3)

    assert library.get_part(missing["id"]) is None
    assert library.get_part(kept["id"]) is not None
    assert not orphan.exists() and not empty.exists()
    assert all(path.exists() for path in foreign)


def test_worker_survives_a_database_error(library, make_renderer, monkeypatch):
    document_id = _book(library)
    renderer = make_renderer(lookahead_parts=1)
    real_get_part = library.get_part
    failures = {"left": 1}

    def flaky_get_part(part_id):
        if failures["left"] and threading.current_thread().name == "reader-renderer":
            failures["left"] -= 1
            raise sqlite3.OperationalError("database is locked")
        return real_get_part(part_id)

    monkeypatch.setattr(library, "get_part", flaky_get_part)
    with renderer._cond:
        renderer._job = _Job(document_id, "kokoro", "af_heart", renderer.plan(document_id, 0, "kokoro", "af_heart"))
        renderer._ensure_thread()
        renderer._cond.notify_all()
    assert _wait_ready(renderer, document_id, 0)[0]["status"] == "ready"
    assert failures["left"] == 0


def test_worker_survives_a_database_error_during_its_startup_sweep(library, make_renderer, monkeypatch):
    # The player only polls for parts afterwards, and polling never restarts a dead worker.
    document_id = _book(library)
    renderer = make_renderer(lookahead_parts=1)
    real_part_paths = library.part_paths
    failures = {"left": 1}

    def flaky_part_paths():
        if failures["left"] and threading.current_thread().name == "reader-renderer":
            failures["left"] -= 1
            raise sqlite3.OperationalError("database is locked")
        return real_part_paths()

    monkeypatch.setattr(library, "part_paths", flaky_part_paths)
    renderer.request(document_id, 0, "kokoro", "af_heart")

    assert _wait_ready(renderer, document_id, 0)[0]["status"] == "ready"
    assert failures["left"] == 0


def test_a_sentence_longer_than_the_models_input_limit_is_spoken_in_pieces(library, make_renderer, voice):
    # Sentences are split at import for every model; Chatterbox, for one, takes only 350 characters.
    sentence = "Somewhere " * 37 + "tonight."
    document_id = _save(library, [Chapter("One", [Block(sentence)])])
    renderer = make_renderer(max_input_chars_fn=lambda model: 120 if model == "kokoro" else None)

    renderer.request(document_id, 0, "kokoro", "af_heart")
    part = _wait_ready(renderer, document_id, 0)[0]

    assert part["status"] == "ready", part["error"]
    assert len(voice.calls) > 1 and all(len(text) <= 120 for text in voice.calls)
    assert " ".join(voice.calls).split() == sentence.split()
    assert [mark["index"] for mark in part["sentences"]] == [0]   # still one sentence to highlight


def _silent_runs(audio: np.ndarray) -> list[int]:
    """Lengths in samples of the runs of silence between two sounds."""
    loud = np.abs(audio) > 0.01
    edges = np.flatnonzero(np.diff(loud.astype(np.int8)))
    # Each fall (loud to silent) followed by a rise is one run between two sounds.
    falls = [edge + 1 for edge in edges if loud[edge]]
    rises = [edge + 1 for edge in edges if not loud[edge]]
    return [next(rise for rise in rises if rise > fall) - fall for fall in falls if any(rise > fall for rise in rises)]


def test_the_pieces_of_an_over_limit_sentence_are_joined_with_a_pause(library, make_renderer, voice):
    # The pieces were appended bare, and as each piece's synthesis is trimmed to its sound
    # (TTS_TRIM_SILENCE), the words either side of the cut ran together. The fake voice's pieces are all sound, as a
    # trimmed synthesis is. Later sentences' start_ms count the pause.
    first = "The river ran on past the mill and the old stone bridge where we used to sit in summer"
    sentence = f"{first}, and then it turned north into the hills where nobody had walked for many years at all."
    encoded: list[np.ndarray] = []

    def encode(audio, sample_rate):
        encoded.append(np.asarray(audio))
        return b"MP3"

    document_id = _save(library, [Chapter("One", [Block(f"{sentence} A second sentence.")])])
    renderer = make_renderer(
        encode_fn=encode, pronounce_fn=lambda text: text, max_input_chars_fn=lambda model: 120, part_max_chars=450,
    )

    renderer.request(document_id, 0, "kokoro", "af_heart")
    part = _wait_ready(renderer, document_id, 0)[0]

    assert part["status"] == "ready", part["error"]
    assert len(voice.calls) == 3 and voice.calls[0].endswith("summer,"), voice.calls
    gap = int(SENTENCE_GAP_S * SAMPLE_RATE)
    assert _silent_runs(encoded[0]) == [gap, gap], "a pause at the cut and one after the sentence"
    second_start = 2 * SENTENCE_SAMPLES + 2 * gap
    assert [mark["start_ms"] for mark in part["sentences"]] == [0, round(second_start * 1000 / SAMPLE_RATE)]


def _refuses_nothing_to_say(voice: FakeVoice):
    """Synthesis as a GPU worker answers it: a text with no letter or digit is refused (an empty text is a 422 there)."""
    from src.tts.external import ExternalProviderError

    def synthesize(*, text, **kwargs):
        if not any(character.isalnum() for character in text):
            voice.calls.append(text)
            raise ExternalProviderError(
                "TTS provider 'chatterbox' rejected the request: text: String should have at least 1 character",
                code="invalid_request",
                status_code=400,
            )
        return voice.synthesize(text=text, **kwargs)

    return synthesize


def test_a_sentence_the_pronunciation_dictionary_empties_is_skipped_and_the_part_plays(library, make_renderer, voice):
    # The renderer checked the text as written, then sent what the dictionary left,
    # so a sentence it removes ("[Illustration]": "") went to the model as "" and on a GPU worker the whole part failed;
    # one it leaves as punctuation ("Um": "" turns "Um." into ".") went as ".". Like /v1/audio/speech ("Input text is
    # empty") and realtime (a segment with nothing left is skipped), the Reader skips such a sentence: no audio, and its
    # mark and gap stay, as for a sentence with nothing to say.
    document_id = _save(library, [Chapter("One", [
        Block("It was a fine morning."),
        Block("[Illustration]"),
        Block("Um."),
        Block("She walked to the river and sat down."),
    ])])
    renderer = make_renderer(
        synthesize_fn=_refuses_nothing_to_say(voice),
        pronounce_fn=lambda text: text.replace("[Illustration]", "").replace("Um", ""),
        part_max_chars=450,
    )

    renderer.request(document_id, 0, "kokoro", "af_heart")
    parts = _wait_ready(renderer, document_id, 0)

    assert [(part["first_block"], part["last_block"]) for part in parts] == [(0, 0), (1, 3)]
    assert [(part["status"], part["error"]) for part in parts] == [("ready", None)] * 2, parts
    assert voice.calls == ["It was a fine morning.", "She walked to the river and sat down."]
    paragraph = int(PARAGRAPH_GAP_S * SAMPLE_RATE)
    second = parts[1]
    assert [(mark["block"], mark["index"]) for mark in second["sentences"]] == [(1, 0), (2, 0), (3, 0)]
    assert [mark["start_ms"] for mark in second["sentences"]] == [
        round(samples * 1000 / SAMPLE_RATE) for samples in (0, paragraph, 2 * paragraph)
    ]
    assert second["duration_ms"] == round((2 * paragraph + SENTENCE_SAMPLES + paragraph) * 1000 / SAMPLE_RATE)


def test_a_piece_of_an_over_limit_sentence_with_nothing_to_say_is_not_sent(library, make_renderer, voice):
    # A piece cut for the model's limit is checked like a sentence, and "_" is not a letter or digit. The rule cut off
    # as its own piece is not sent, and no pause is added for it.
    first = "The river ran on past the mill and the old stone bridge where we used to sit in summer,"
    encoded: list[np.ndarray] = []

    def encode(audio, sample_rate):
        encoded.append(np.asarray(audio))
        return b"MP3"

    document_id = _save(library, [Chapter("One", [Block(f"{first} {'_' * 60}")])])
    renderer = make_renderer(
        synthesize_fn=_refuses_nothing_to_say(voice), encode_fn=encode, pronounce_fn=lambda text: text,
        max_input_chars_fn=lambda model: 120, part_max_chars=450,
    )

    renderer.request(document_id, 0, "kokoro", "af_heart")
    part = _wait_ready(renderer, document_id, 0)[0]

    assert (part["status"], part["error"]) == ("ready", None)
    assert voice.calls == [first]
    assert len(encoded[0]) == SENTENCE_SAMPLES + int(PARAGRAPH_GAP_S * SAMPLE_RATE), "no pause for the skipped piece"


def test_is_speakable_needs_a_letter_or_digit_and_underscores_are_neither():
    # The check was \w, which also matches "_".
    from src.reader.document import is_speakable

    results = {text: is_speakable(text) for text in ("__________", "_", "***", " . ", "word_", "4", "é", "ラーメン")}
    assert results == {
        "__________": False, "_": False, "***": False, " . ": False, "word_": True, "4": True, "é": True, "ラーメン": True,
    }


def test_a_rule_line_of_underscores_is_never_sent_to_the_model(library, make_renderer, voice):
    # Import keeps such a line as a block (documents in a library hold it), and the
    # renderer sent it to the model as "__________". Its mark and gap stay, as for "* * *".
    document_id = _save(library, [Chapter("One", [
        Block("The first scene ends here."), Block("__________"), Block("The second scene begins."),
    ])])
    renderer = make_renderer(part_max_chars=450)

    renderer.request(document_id, 0, "kokoro", "af_heart")
    parts = _wait_ready(renderer, document_id, 0)

    assert [(part["status"], part["error"]) for part in parts] == [("ready", None)] * 2, parts
    assert voice.calls == ["The first scene ends here.", "The second scene begins."]
    assert [mark["block"] for part in parts for mark in part["sentences"]] == [0, 1, 2]


def _refuses_japanese(voice: FakeVoice, code: str = "language_unsupported"):
    """A worker that doesn't speak Japanese: a sentence with kana gets 400 language_unsupported."""
    from src.tts.external import ExternalProviderError

    def synthesize(*, text, **kwargs):
        if any("぀" <= character <= "ヿ" for character in text):
            voice.calls.append(text)
            raise ExternalProviderError(
                "Japanese is not supported by chatterbox/regular. Supported: en. Try qwen3/0.6b-base.",
                code=code,
                status_code=400,
            )
        return voice.synthesize(text=text, **kwargs)

    return synthesize


def test_a_sentence_the_voice_refuses_for_its_language_is_skipped_and_the_part_plays(library, make_renderer, voice):
    # A sentence refused with language_unsupported is left out
    # of the part's audio, whole (a piece already spoken is dropped too), and the part goes on. Its mark stays with no
    # audio of its own, so later start_ms values count only what was joined.
    long_refused = "She ordered " + "noodles and " * 12 + "ラーメン at the station, and then she sat down by the window."
    document_id = _save(library, [Chapter("One", [
        Block("It was a fine morning. She ordered ラーメン at the station."),
        Block(long_refused),
        Block("She walked to the river."),
    ])])
    renderer = make_renderer(
        synthesize_fn=_refuses_japanese(voice), pronounce_fn=lambda text: text,
        max_input_chars_fn=lambda model: 120, part_max_chars=450,
    )

    renderer.request(document_id, 0, "kokoro", "af_heart")
    parts = _wait_ready(renderer, document_id, 0)

    assert [(part["first_block"], part["last_block"]) for part in parts] == [(0, 0), (1, 2)]
    assert [(part["status"], part["error"]) for part in parts] == [("ready", None)] * 2, parts
    spoken = [text for text in voice.calls if not any("぀" <= c <= "ヿ" for c in text)]
    assert spoken[0] == "It was a fine morning." and spoken[-1] == "She walked to the river.", voice.calls
    assert len(spoken) == 3, "the long sentence's first piece was asked for, then dropped with the refused piece"
    sentence_gap, paragraph = int(SENTENCE_GAP_S * SAMPLE_RATE), int(PARAGRAPH_GAP_S * SAMPLE_RATE)
    first, second = parts
    refused_start = SENTENCE_SAMPLES + sentence_gap
    assert [(mark["block"], mark["index"]) for mark in first["sentences"]] == [(0, 0), (0, 1)]
    assert [mark["start_ms"] for mark in first["sentences"]] == [0, round(refused_start * 1000 / SAMPLE_RATE)]
    assert first["duration_ms"] == round((refused_start + paragraph) * 1000 / SAMPLE_RATE)
    assert [(mark["block"], mark["index"]) for mark in second["sentences"]] == [(1, 0), (2, 0)]
    assert [mark["start_ms"] for mark in second["sentences"]] == [0, round(paragraph * 1000 / SAMPLE_RATE)]
    assert second["duration_ms"] == round((paragraph + SENTENCE_SAMPLES + paragraph) * 1000 / SAMPLE_RATE)


def test_a_part_whose_every_sentence_the_voice_refuses_for_its_language_fails_with_the_reason(
    library, make_renderer, voice,
):
    # A part with nothing to hear but refused sentences would be silent, so it fails with
    # the refusal, which the web app shows ("Couldn't play: ..." naming the languages and the model to use instead).
    document_id = _save(library, [Chapter("One", [Block("ラーメン。 * * * すし。")])])
    renderer = make_renderer(synthesize_fn=_refuses_japanese(voice), pronounce_fn=lambda text: text, part_max_chars=450)

    renderer.request(document_id, 0, "kokoro", "af_heart")
    parts = _wait_ready(renderer, document_id, 0)

    assert [part["status"] for part in parts] == ["error"]
    assert "Japanese is not supported" in parts[0]["error"], parts[0]["error"]


def test_any_other_refusal_still_fails_the_part(library, make_renderer, voice):
    # Only language_unsupported is skipped; every other error keeps failing the part.
    document_id = _save(library, [Chapter("One", [Block("It was a fine morning. She ordered ラーメン at the station.")])])
    renderer = make_renderer(
        synthesize_fn=_refuses_japanese(voice, code="invalid_request"), pronounce_fn=lambda text: text,
        part_max_chars=450,
    )

    renderer.request(document_id, 0, "kokoro", "af_heart")
    parts = _wait_ready(renderer, document_id, 0)

    assert [part["status"] for part in parts] == ["error"]
    assert "Japanese is not supported" in parts[0]["error"], parts[0]["error"]


# "[clear throat]" (the one sound tag with a space) reaches the voice
# model whole. The cuts by a cap, in the sentences stored at import and in a part's pieces for the model's limit, used
# to fall at the space inside it. A sentence cut must not split [clear throat].
_TAG = "[clear throat]"


def _splits_the_tag(texts: list[str]) -> list[str]:
    return [text for text in texts if ("clear" in text or "throat" in text) and _TAG not in text]


@pytest.mark.parametrize("pad", range(14))
def test_the_sentences_stored_at_import_never_split_a_sound_tag(library, pad):
    head = "x" * pad + " " + " ".join(["word"] * ((store_module.SENTENCE_MAX_CHARS - pad) // 5 - 2))
    text = f"{head} {_TAG} and then some more words to end it."
    document_id = _save(library, [Chapter("One", [Block(text)])])
    block = library.get_blocks(document_id, 0, 0)[0]
    pieces = [block["text"][start:end] for start, end in block["sentences"]]
    assert len(pieces) > 1 and not _splits_the_tag(pieces), [piece[-30:] for piece in pieces]
    assert all(end - start <= store_module.SENTENCE_MAX_CHARS for start, end in block["sentences"])


@pytest.mark.parametrize("pad", range(14))
def test_a_parts_pieces_for_the_models_limit_never_split_a_sound_tag(library, make_renderer, voice, pad):
    sentence = "x" * pad + " " + " ".join(["word"] * 20) + f" {_TAG} and then some more words to end it."
    document_id = _save(library, [Chapter("One", [Block(sentence)])])
    renderer = make_renderer(max_input_chars_fn=lambda model: 120 if model == "kokoro" else None)

    renderer.request(document_id, 0, "kokoro", "af_heart")
    part = _wait_ready(renderer, document_id, 0)[0]

    assert part["status"] == "ready", part["error"]
    assert not _splits_the_tag(voice.calls), voice.calls
    assert sum(call.count(_TAG) for call in voice.calls) == 1, voice.calls
    assert all(len(call) <= 120 for call in voice.calls), voice.calls


class _CountingEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, audio, sample_rate):
        self.calls += 1
        return b"MP3"


def test_touching_a_part_reports_whether_it_is_recorded_and_keeps_it_from_pruning(library, make_renderer):
    # The media route touches a part before serving it, so a prune cannot unlink it mid-request.
    document_id = _book(library)
    renderer = make_renderer(lookahead_parts=2)
    renderer.request(document_id, 0, "kokoro", "af_heart")
    played, other = (part["id"] for part in _wait_ready(renderer, document_id, 0))
    store_module.get_db().execute("UPDATE reader_parts SET last_access = 0")

    assert library.touch_part(played) is True
    library.prune_parts(0, keep=set())

    assert library.get_part(played) is not None
    assert library.get_part(other) is None
    assert library.touch_part(other) is False


def test_an_outline_read_before_a_delete_is_not_cached_after_it(library, make_renderer):
    # A /parts poll can read the outline just before DELETE removes the rows and forgets the document.
    document_id = _book(library)
    renderer = make_renderer()
    read_outline = library.outline

    def outline_then_deleted(doc_id: str):
        outline = read_outline(doc_id)
        library.delete_document(doc_id)
        renderer.forget_document(doc_id)
        return outline

    library.outline = outline_then_deleted
    assert renderer.status(document_id, 0, "kokoro", "af_heart") is not None   # the poll that raced the delete
    library.outline = read_outline

    assert renderer.status(document_id, 0, "kokoro", "af_heart") is None


def _settled(renderer: ReaderRenderer, document_id: str, from_block: int, voice_name: str, timeout: float = 5.0) -> list[dict]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        parts = renderer.status(document_id, from_block, "kokoro", voice_name)
        if all(part["status"] in {"ready", "error"} for part in parts):
            return parts
        time.sleep(0.02)
    raise AssertionError(f"parts never finished: {parts}")


def test_errors_are_kept_for_the_current_job_only(library, make_renderer, voice):
    # Each failed request with another voice (a typo, a provider that is down) left its parts'
    # errors behind for the life of the process.
    document_id = _book(library)
    renderer = make_renderer()
    voice.fail = True
    for n in range(50):
        renderer.request(document_id, 0, "kokoro", f"voice_{n}")
        assert all(part["status"] == "error" for part in _settled(renderer, document_id, 0, f"voice_{n}"))

    assert 0 < len(renderer._errors) <= renderer.lookahead_parts


def _wav() -> bytes:
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(b"\x00\x00" * 2400)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("missing", "message"),
    [("named-voice", "Named voice not found"), ("recording", "Voice library entry 'narrator' not found")],
)
def test_a_part_whose_named_voice_or_recording_is_gone_says_so_without_python_quotes(
    library, make_renderer, tmp_path, monkeypatch, missing, message
):
    # Both are KeyErrors, and str(KeyError("x")) is "'x'": the player showed "Couldn't play: 'narrator'".
    from src.voice_identities import VoiceIdentityManager
    from src.voice_library import VoiceLibraryManager

    monkeypatch.setattr(main_module, "voice_library", VoiceLibraryManager(tmp_path / "voices"))
    document_id = _book(library)
    voices = VoiceIdentityManager()
    identity = voices.create("Narrator")
    main_module.voice_library.save("narrator", _wav())
    voices.add_realization(identity["id"], model="kokoro", voice="af_heart", reference_audio_id="narrator")
    if missing == "named-voice":
        voices.delete(identity["id"])
    else:
        main_module.voice_library.delete("narrator")
    renderer = make_renderer(synthesize_fn=main_module._synthesize_array, lookahead_parts=1)

    renderer.request(document_id, 0, "kokoro", identity["voice"])
    part = _settled(renderer, document_id, 0, identity["voice"])[0]

    assert (part["status"], part["error"]) == ("error", message)


def test_parts_are_rendered_again_once_their_named_voice_speaks_otherwise(library, make_renderer, voice):
    # A part id named only the voice the player sent ("voice:<id>"), so parts rendered before the
    # named voice changed version (or recording) kept playing the old one.
    document_id = _book(library)
    speaks_as = {"key": "af_heart"}
    renderer = make_renderer(lookahead_parts=1, voice_key_fn=lambda model, voice_name: speaks_as["key"])
    renderer.request(document_id, 0, "kokoro", "voice:narrator")
    before = _settled(renderer, document_id, 0, "voice:narrator")[0]
    calls = len(voice.calls)
    assert renderer.status(document_id, 0, "kokoro", "voice:narrator")[0]["id"] == before["id"]

    speaks_as["key"] = "af_bella"
    after = renderer.status(document_id, 0, "kokoro", "voice:narrator")[0]

    assert after["id"] != before["id"] and after["status"] == "pending"
    assert library.get_part(before["id"]) is None
    renderer.request(document_id, 0, "kokoro", "voice:narrator")
    assert _settled(renderer, document_id, 0, "voice:narrator")[0]["status"] == "ready"
    assert len(voice.calls) > calls


def test_a_named_voice_that_cannot_be_spoken_is_refused_before_any_part_is_served(library, make_renderer):
    document_id = _book(library)

    def gone(model, voice_name):
        raise renderer_module.ReaderVoiceError(404, "Named voice not found")

    renderer = make_renderer(voice_key_fn=gone)

    with pytest.raises(renderer_module.ReaderVoiceError, match="Named voice not found"):
        renderer.request(document_id, 0, "kokoro", "voice:narrator")
    with pytest.raises(renderer_module.ReaderVoiceError):
        renderer.status(document_id, 0, "kokoro", "voice:narrator")


def test_a_provider_voice_keeps_the_part_ids_it_had():
    # Only a named voice adds what it speaks with, so caches of provider voices stay valid.
    import hashlib

    key = f"doc\n0\n2\nkokoro\naf_heart\n{renderer_module.RENDER_VERSION}"
    assert renderer_module.part_id("doc", 0, 2, "kokoro", "af_heart") == hashlib.sha256(key.encode()).hexdigest()[:24]
    assert renderer_module.part_id("doc", 0, 2, "kokoro", "voice:x", "a") != renderer_module.part_id("doc", 0, 2, "kokoro", "voice:x", "b")


def test_parts_are_rendered_again_once_the_pronunciation_dictionary_changes(
    library, make_renderer, voice, tmp_path, monkeypatch
):
    # Part ids left out TTS_PRONUNCIATION_DICT, so after the file was
    # edited and the server restarted, cached parts kept the old pronunciation and one book mixed both.
    # The renderer gets the app's own wiring: src/main.py's lambdas read src.main.pronunciation_dict.
    import json

    from src.pronunciation.dictionary import PronunciationDictionary

    app_pronounce = main_module.reader_renderer.pronounce_fn
    app_speech_key = main_module.reader_renderer.speech_key_fn
    document_id = _save(library, [Chapter("One", [Block("I like SQL.")])])
    dictionary = tmp_path / "pronunciations.json"
    running: list[ReaderRenderer] = []

    def restart(entries: dict[str, str] | None) -> ReaderRenderer:
        """A server start: the dictionary file is read again and a new renderer runs on the same cache."""
        if running:
            running.pop().stop()
        if entries is None:
            loaded = PronunciationDictionary(None)
        else:
            dictionary.write_text(json.dumps(entries), encoding="utf-8")
            loaded = PronunciationDictionary(str(dictionary))
        monkeypatch.setattr(main_module, "pronunciation_dict", loaded)
        renderer = make_renderer(lookahead_parts=1, pronounce_fn=app_pronounce, speech_key_fn=app_speech_key)
        running.append(renderer)
        return renderer

    def render(renderer: ReaderRenderer) -> dict:
        voice.calls.clear()
        renderer.request(document_id, 0, "kokoro", "af_heart")
        return _wait_ready(renderer, document_id, 0)[0]

    plain = render(restart(None))
    # Without a dictionary a part keeps the id it always had, so existing caches stay valid.
    assert plain["id"] == renderer_module.part_id(document_id, 0, 0, "kokoro", "af_heart")
    assert voice.calls == ["I like SQL."]

    spelled = render(restart({"SQL": "S Q L"}))
    assert voice.calls == ["I like S Q L."], "the part cached before the dictionary existed was served"
    assert spelled["id"] != plain["id"]

    renderer = restart({"SQL": "sequel"})
    stale = renderer.status(document_id, 0, "kokoro", "af_heart")[0]
    assert (stale["status"], stale["id"] != spelled["id"]) == ("pending", True), (
        "the part spoken with the old entry ('S Q L') is still served after the entry changed to 'sequel'"
    )
    assert library.get_part(spelled["id"]) is None   # its row is forgotten, not left to be served
    said = render(renderer)
    assert (said["id"], voice.calls) == (stale["id"], ["I like sequel."])

    # A restart with the same entries keeps the cached part.
    again = render(restart({"SQL": "sequel"}))
    assert (again["id"], voice.calls) == (said["id"], [])


def test_a_window_whose_job_another_listener_took_stays_pending_with_its_ids(library, make_renderer, voice):
    # What the player's RE-08 recovery reads: GET /parts keeps planning the same window, and a part
    # in no job is "pending". The worker finishes the first part it is on when
    # another document's request takes the job, so here the job is taken before our part starts: the
    # other listener asks first, we ask while its part renders, and its player asks again.
    document_id = _book(library)
    other_id = _save(library, [Chapter("Other", [Block("Another book's only block.")])])
    renderer = make_renderer(lookahead_parts=1)
    voice.hold.clear()
    renderer.request(other_id, 0, "kokoro", "af_heart")   # another tab, or the phone
    assert voice.started.wait(5)
    ours = renderer.request(document_id, 0, "kokoro", "af_heart")[0]
    assert ours["status"] == "pending"                    # the worker is still on the other's part

    renderer.request(other_id, 0, "kokoro", "af_heart")   # the other player asks again
    voice.hold.set()
    assert _wait_ready(renderer, other_id, 0)[0]["status"] == "ready"

    again = renderer.status(document_id, 0, "kokoro", "af_heart")[0]
    assert (again["id"], again["first_block"], again["status"]) == (ours["id"], 0, "pending")


class _Gate:
    """synthesize_fn that holds each call until the test lets it go, so a request lands between two
    known sentences. Calls are recorded as (model, voice, text)."""

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.calls: list[tuple[str, str, str]] = []
        self.allowed = 0
        self.closed = False

    def synthesize(self, *, text, model, voice, speed, sample_rate):
        with self.cond:
            self.calls.append((model, voice, text))
            index = len(self.calls)
            self.cond.notify_all()
            if not self.cond.wait_for(lambda: self.allowed >= index or self.closed, 5):
                raise RuntimeError("the test never let this sentence go")
        return np.ones(SENTENCE_SAMPLES, dtype=np.float32)

    def held(self, timeout: float = 5.0) -> bool:
        """Wait until the worker is held in its first call."""
        with self.cond:
            return self.cond.wait_for(lambda: len(self.calls) > 0, timeout)

    def next(self, timeout: float = 5.0) -> tuple[str, str, str] | None:
        """Let the held call return; the worker's next call, or None if it makes none in ``timeout``."""
        with self.cond:
            count = self.allowed = len(self.calls)
            self.cond.notify_all()
            if not self.cond.wait_for(lambda: len(self.calls) > count, timeout):
                return None
            return self.calls[count]

    def close(self) -> None:
        """Let every call return at once from now on."""
        with self.cond:
            self.closed = True
            self.cond.notify_all()


@pytest.fixture
def gate(make_renderer):
    held = _Gate()
    yield held
    held.close()   # torn down before make_renderer stops the renderers


def _ready(renderer: ReaderRenderer, document_id: str, from_block: int, model: str, voice_name: str) -> dict:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        part = renderer.status(document_id, from_block, model, voice_name)[0]
        if part["status"] in {"ready", "error"}:
            return part
        time.sleep(0.02)
    raise AssertionError(f"part never finished: {part}")


@pytest.mark.parametrize(
    ("taken_by", "kept"),
    [("another document", True), ("another voice", True), ("another model", True), ("a skip", False)],
)
def test_a_jobs_first_part_is_finished_when_a_request_for_another_document_model_or_voice_takes_the_job(
    library, make_renderer, gate, taken_by, kept
):
    # The worker dropped the part it was on at the next sentence boundary
    # whenever another request took the single job. Two listeners waiting at once for first parts longer
    # than the player's 24 s re-request wait then took the job from each other every 24 s, and neither
    # part was ever saved. A job's first part is now finished and saved when a request for another
    # document, model or voice takes the job; a skip (same document, model and voice) still drops it.
    document_id = _book(library)
    other_id = _save(library, [Chapter("Other", [Block("Another book's only block.")])])
    renderer = make_renderer(synthesize_fn=gate.synthesize, lookahead_parts=1)
    ours = renderer.request(document_id, 0, "kokoro", "af_heart")[0]
    assert (ours["first_block"], ours["last_block"]) == (0, 1)   # the heading "One", then "First short block."
    assert gate.held() and gate.calls == [("kokoro", "af_heart", "One")]

    request, their_first_sentence = {
        "another document": ((other_id, 0, "kokoro", "af_heart"), ("kokoro", "af_heart", "Another book's only block.")),
        "another voice": ((document_id, 0, "kokoro", "am_adam"), ("kokoro", "am_adam", "One")),
        "another model": ((document_id, 0, "piper", "af_heart"), ("piper", "af_heart", "One")),
        "a skip": ((document_id, 2, "kokoro", "af_heart"), ("kokoro", "af_heart", "Second short block.")),
    }[taken_by]
    theirs = renderer.request(*request)[0]   # another tab or device, a voice or model change, or a skip

    if kept:
        assert gate.next() == ("kokoro", "af_heart", "First short block."), (
            f"after a request for {taken_by} took the job, the worker dropped the first part it was on: {gate.calls}"
        )
    assert gate.next() == their_first_sentence, gate.calls
    # The worker has now saved our part, or dropped it, and moved on to theirs.
    assert renderer.status(document_id, 0, "kokoro", "af_heart")[0]["status"] == ("ready" if kept else "pending")
    assert (library.get_part(ours["id"]) is not None) is kept
    if not kept:
        assert ("kokoro", "af_heart", "First short block.") not in gate.calls

    gate.close()
    assert _ready(renderer, *request)["status"] == "ready"
    assert library.get_part(theirs["id"]) is not None


def test_a_part_after_its_jobs_first_is_still_dropped_when_another_document_takes_the_job(
    library, make_renderer, gate
):
    # Only a job's first part, the one a listener waits on before any
    # audio, is finished for a replaced job. The rest of the window still stops at the next sentence.
    document_id = _book(library)
    other_id = _save(library, [Chapter("Other", [Block("Another book's only block.")])])
    renderer = make_renderer(synthesize_fn=gate.synthesize, lookahead_parts=2)
    first, second = renderer.request(document_id, 0, "kokoro", "af_heart")
    assert (second["first_block"], second["last_block"]) == (2, 3)
    assert gate.held()
    assert gate.next() == ("kokoro", "af_heart", "First short block.")
    assert gate.next() == ("kokoro", "af_heart", "Second short block.")   # the first part is saved

    renderer.request(other_id, 0, "kokoro", "af_heart")
    assert gate.next() == ("kokoro", "af_heart", "Another book's only block."), gate.calls

    assert library.get_part(first["id"]) is not None
    assert library.get_part(second["id"]) is None
    assert ("kokoro", "af_heart", "Third block, still short.") not in gate.calls


def test_a_replaced_jobs_first_part_stops_once_its_document_is_deleted(library, make_renderer, gate, tmp_path):
    # The first part the worker finishes for a replaced job is still
    # abandoned at the next sentence once its document is deleted, and leaves nothing in the cache.
    document_id = _book(library)
    other_id = _save(library, [Chapter("Other", [Block("Another book's only block.")])])
    renderer = make_renderer(synthesize_fn=gate.synthesize, lookahead_parts=1)
    renderer.request(document_id, 0, "kokoro", "af_heart")
    assert gate.held()

    renderer.request(other_id, 0, "kokoro", "af_heart")
    library.delete_document(document_id)   # what DELETE /v1/reader/documents/{id} does
    renderer.forget_document(document_id)
    assert gate.next() == ("kokoro", "af_heart", "Another book's only block."), gate.calls

    gate.close()
    assert _wait_ready(renderer, other_id, 0)[0]["status"] == "ready"
    assert not (tmp_path / "reader" / document_id).exists()


class _ClipRouter:
    """Stands in for src.main.tts_router: every text is the same clip, silence, speech at 0.5, silence."""

    SPEECH = 0.5

    def __init__(self) -> None:
        self.calls: list[str] = []

    def get_backend(self, model_id):
        from types import SimpleNamespace

        return SimpleNamespace(name="kokoro", capabilities={})

    def validate_voice(self, model_id, voice):
        return None

    def max_input_chars_for(self, model_id):
        return None

    def sample_rate_for(self, model_id):
        return SAMPLE_RATE

    def synthesize(self, *, text, model, voice, speed, lang_code=None, **options):
        self.calls.append(text)
        silence = np.zeros(1200, dtype=np.float32)
        yield np.concatenate([silence, np.full(1200, self.SPEECH, dtype=np.float32), silence])


def _measured(audio, sample_rate) -> bytes:
    """An "MP3" that says how its audio was processed: its peak and its length."""
    audio = np.asarray(audio, dtype=np.float32)
    return f"{float(np.max(np.abs(audio))):.2f} {len(audio)}".encode("utf-8")


@pytest.mark.parametrize(
    "turned_off",
    [("tts_trim_silence",), ("tts_normalize_output",), ("tts_trim_silence", "tts_normalize_output")],
    ids=["trim", "normalize", "both"],
)
def test_parts_are_rendered_again_once_trim_or_normalize_is_turned_off(
    library, make_renderer, monkeypatch, turned_off
):
    # Part ids left out TTS_TRIM_SILENCE and TTS_NORMALIZE_OUTPUT, so after
    # either was turned off and the server restarted, cached parts were still served trimmed and normalized,
    # next to new parts that were not. The renderer gets the app's own wiring: main's synthesis (over a fake
    # tts_router) and its speech_key_fn, which read src.main.settings.
    from src.pronunciation.dictionary import PronunciationDictionary

    router = _ClipRouter()
    monkeypatch.setattr(main_module, "tts_router", router)
    monkeypatch.setattr(main_module, "pronunciation_dict", PronunciationDictionary(None))
    monkeypatch.setattr(main_module.settings, "tts_trim_silence", True)   # the defaults
    monkeypatch.setattr(main_module.settings, "tts_normalize_output", True)
    app = main_module.reader_renderer
    assert app.synthesize_fn is main_module._synthesize_array
    document_id = _save(library, [Chapter("One", [Block("First block.")])])
    running: list[ReaderRenderer] = []

    def restart() -> ReaderRenderer:
        """A server start: a new renderer with the app's hooks runs on the same store and cache."""
        if running:
            running.pop().stop()
        renderer = make_renderer(
            lookahead_parts=1,
            synthesize_fn=app.synthesize_fn,
            pronounce_fn=app.pronounce_fn,
            speech_key_fn=app.speech_key_fn,
            encode_fn=_measured,
        )
        running.append(renderer)
        return renderer

    def render(renderer: ReaderRenderer) -> tuple[dict, tuple[float, int]]:
        router.calls.clear()
        renderer.request(document_id, 0, "kokoro", "af_heart")
        part = _wait_ready(renderer, document_id, 0)[0]
        assert part["status"] == "ready", part
        peak, samples = Path(library.get_part(part["id"])["path"]).read_text(encoding="utf-8").split()
        return part, (float(peak), int(samples))

    default, (default_peak, default_samples) = render(restart())
    # On the defaults a part keeps the id it always had, so existing caches stay valid.
    assert default["id"] == renderer_module.part_id(document_id, 0, 0, "kokoro", "af_heart")
    assert router.calls == ["First block."]
    assert default_peak > _ClipRouter.SPEECH   # normalized
    again, _audio = render(restart())
    assert (again["id"], router.calls) == (default["id"], [])

    for setting in turned_off:   # the operator turns it off, then restarts
        monkeypatch.setattr(main_module.settings, setting, False)
    renderer = restart()
    stale = renderer.status(document_id, 0, "kokoro", "af_heart")[0]
    assert (stale["status"], stale["id"] != default["id"]) == ("pending", True), (
        f"the part processed with {', '.join(turned_off)} on is still served after the restart with it off"
    )
    assert library.get_part(default["id"]) is None   # its row is forgotten, not left to be served
    off, (off_peak, off_samples) = render(renderer)
    assert (off["id"], router.calls) == (stale["id"], ["First block."])
    if "tts_normalize_output" in turned_off:
        assert off_peak == _ClipRouter.SPEECH
    if "tts_trim_silence" in turned_off:
        assert off_samples > default_samples

    # Turned back on, the part is spoken as on the defaults again, under its old id.
    for setting in turned_off:
        monkeypatch.setattr(main_module.settings, setting, True)
    back, audio = render(restart())
    assert (back["id"], router.calls, audio) == (default["id"], ["First block."], (default_peak, default_samples))


def test_post_processing_key_is_empty_on_the_defaults_and_names_each_other_choice():
    # What joins the part ids' speech key for TTS_TRIM_SILENCE and
    # TTS_NORMALIZE_OUTPUT. "" on the defaults keeps the part ids of installs that never changed them.
    key = renderer_module.post_processing_key
    assert key(True, True) == ""
    assert len({key(trim, normalize) for trim in (True, False) for normalize in (True, False)}) == 4
