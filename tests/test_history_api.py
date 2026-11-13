from __future__ import annotations

import asyncio
import contextlib
import io
import threading
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.main import app
from src import main as main_module
from src import storage as storage_module


def _reset_db(tmp_path):
    main_module.settings.os_studio_db_path = str(tmp_path / "studio.db")
    main_module.settings.os_history_enabled = True
    main_module.settings.os_history_max_entries = 1000
    storage_module._conn = None
    storage_module.init_db()


def _wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 1600)
    return buf.getvalue()


def test_history_logs_tts_and_stt_via_api(tmp_path):
    _reset_db(tmp_path)
    client = TestClient(app)

    mock_tts = MagicMock()
    mock_tts.synthesize.return_value = iter([np.zeros(1600, dtype=np.float32)])

    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = {"text": "hello there"}

    with patch.object(main_module, "tts_router", mock_tts), patch.object(main_module, "backend_router", mock_stt):
        tts_resp = client.post("/v1/audio/speech", json={
            "model": "kokoro",
            "input": "hello",
            "voice": "af_heart",
            "response_format": "pcm",
        }, headers={"X-History": "true"})
        assert tts_resp.status_code == 200

        files = {"file": ("sample.wav", _wav_bytes(), "audio/wav")}
        stt_resp = client.post("/v1/audio/transcriptions", files=files, data={"model": "mock", "response_format": "json"}, headers={"X-History": "true"})
        assert stt_resp.status_code == 200

    listing = client.get("/api/history")
    assert listing.status_code == 200
    types = [i["type"] for i in listing.json()["items"]]
    assert "tts" in types and "stt" in types


def test_history_filter_delete_clear_prune_and_streamed_metadata_only(tmp_path):
    _reset_db(tmp_path)
    client = TestClient(app)

    hm = main_module.history_manager
    hm.log_tts("kokoro", "af_heart", 1.0, "mp3", "one", None, 10, streamed=False)
    hm.log_stt("mock", "in.wav", "two")

    tts_only = client.get("/api/history?type=tts")
    assert tts_only.status_code == 200
    assert all(i["type"] == "tts" for i in tts_only.json()["items"])

    entry_id = tts_only.json()["items"][0]["id"]
    deleted = client.delete(f"/api/history/{entry_id}")
    assert deleted.status_code == 204

    cleared = client.delete("/api/history")
    assert cleared.status_code == 200
    assert "deleted" in cleared.json()

    main_module.settings.os_history_max_entries = 2
    hm.log_stt("m", "a.wav", "a")
    hm.log_stt("m", "b.wav", "b")
    hm.log_stt("m", "c.wav", "c")
    hm.prune()
    after = hm.list_entries()
    assert after["total"] == 2

    sid = hm.log_tts("kokoro", "af_heart", 1.0, "mp3", "stream", "/tmp/should-not-be-stored.mp3", 111, streamed=True)
    stream_entry = next(i for i in hm.list_entries()["items"] if i["id"] == sid)
    assert stream_entry["streamed"] is True
    assert stream_entry["output_path"] is None
    assert stream_entry["output_bytes"] is None


def _hold_clock(monkeypatch):
    # Logs inside one clock tick share created_at; hold it constant to make that deterministic.
    from src import history as history_module

    monkeypatch.setattr(history_module, "_now_iso", lambda: "2026-01-01T00:00:00+00:00")


def test_history_list_is_newest_first_when_timestamps_tie(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    _hold_clock(monkeypatch)
    hm = main_module.history_manager
    hm.log_stt("m", "a.wav", "older")
    newest = hm.log_stt("m", "b.wav", "newer")

    assert hm.list_entries()["items"][0]["id"] == newest


def test_prune_keeps_newest_entry_when_timestamps_tie(tmp_path, monkeypatch):
    _reset_db(tmp_path)
    _hold_clock(monkeypatch)
    monkeypatch.setattr(main_module.settings, "os_history_max_entries", 2)
    hm = main_module.history_manager
    oldest = hm.log_stt("m", "a.wav", "first")
    second = hm.log_stt("m", "b.wav", "second")
    newest = hm.log_stt("m", "c.wav", "third")

    ids = [item["id"] for item in hm.list_entries()["items"]]
    assert ids == [newest, second]
    assert oldest not in ids


def test_log_tts_leaves_another_threads_transaction_alone(tmp_path):
    # Every thread shares one studio.db connection: a bare commit from log_tts would commit
    # another request's half-written transaction, and that request could not roll it back.
    _reset_db(tmp_path)
    hm = main_module.history_manager
    inserted, release = threading.Event(), threading.Event()
    logged: list[str] = []

    class Abandoned(Exception):
        pass

    def half_written_transaction():
        with contextlib.suppress(Abandoned), storage_module.transaction() as db:
            db.execute(
                "INSERT INTO history_entries (id, type, created_at, streamed) "
                "VALUES ('half-written', 'stt', '2026-01-01T00:00:00+00:00', 0)"
            )
            inserted.set()
            release.wait(timeout=5)
            raise Abandoned  # rolled back

    other = threading.Thread(target=half_written_transaction, daemon=True)
    other.start()
    assert inserted.wait(timeout=5)
    writer = threading.Thread(
        target=lambda: logged.append(hm.log_tts("kokoro", "af_heart", 1.0, "mp3", "mine", None, 10)),
        daemon=True,
    )
    writer.start()
    writer.join(timeout=0.5)  # time for a bare commit to land inside the open transaction
    release.set()
    other.join(timeout=5)
    writer.join(timeout=5)
    assert not other.is_alive() and not writer.is_alive()

    assert [item["id"] for item in hm.list_entries()["items"]] == logged


def test_history_search_matches_the_whole_text_and_file_name(tmp_path):
    # README: History lets users search saved entries; the preview is cut at 180 characters.
    _reset_db(tmp_path)
    client = TestClient(app)
    hm = main_module.history_manager
    long_text = "Chapter one. " + "A quiet filler sentence about the sea. " * 8 + "The Lighthouse keeper's log."
    assert long_text.index("Lighthouse") > 180
    lighthouse = hm.log_tts("kokoro", "af_heart", 1.0, "mp3", long_text, None, 10)
    cafe = hm.log_stt("mock", "notes.wav", "Lunch at the CAFÉ ÉMILE.")
    percent = hm.log_stt("mock", "interview_50%_final.wav", "first take")
    hm.log_stt("mock", "interview_50x_final.wav", "second take")
    hm.log_tts("kokoro", "af_heart", 1.0, "mp3", "Nothing to see.", None, 10)

    def search(query, **params):
        response = client.get("/api/history", params={"search": query, **params})
        assert response.status_code == 200
        return response.json()

    found = search("lighthouse")
    assert [item["id"] for item in found["items"]] == [lighthouse] and found["total"] == 1
    assert [item["id"] for item in search("café émile")["items"]] == [cafe]
    # Wildcards are literal: "50%" must not match "50x".
    assert [item["id"] for item in search("50%")["items"]] == [percent]
    assert search("50_final")["total"] == 0  # a LIKE "_" would match the "%" and the "x"
    assert search("take", type="tts")["total"] == 0
    assert search("take", type="stt")["total"] == 2
    paged = search("e", limit=2, offset=4)
    assert paged["total"] == 5 and len(paged["items"]) == 1
    assert search("   ")["total"] == 5


def test_history_items_carry_their_parsed_meta(tmp_path):
    # Re-generate reads what was recorded (reference voice, instructions, effects) from item.meta.
    import json

    _reset_db(tmp_path)
    hm = main_module.history_manager
    recorded = hm.log_tts("kokoro", "af_heart", 1.0, "mp3", "recorded", None, 10)
    broken = hm.log_tts("kokoro", "af_heart", 1.0, "mp3", "broken", None, 10)
    listed = hm.log_tts("kokoro", "af_heart", 1.0, "mp3", "listed", None, 10)
    empty = hm.log_stt("mock", "in.wav", "empty")
    db = storage_module.get_db()
    meta = {"reference_voice": "narrator", "instructions": "Whisper.", "effects": [{"type": "robot"}]}
    for entry_id, meta_json in ((recorded, json.dumps(meta)), (broken, "{not json"), (listed, "[1, 2]"), (empty, None)):
        db.execute("UPDATE history_entries SET meta_json = ? WHERE id = ?", (meta_json, entry_id))
    db.commit()

    items = {item["id"]: item for item in TestClient(app).get("/api/history").json()["items"]}

    assert items[recorded]["meta"] == meta
    assert all(items[entry_id]["meta"] == {} for entry_id in (broken, listed, empty))
    assert all("meta_json" not in item for item in items.values())


def test_tts_history_records_what_regenerate_needs(tmp_path, monkeypatch):
    # Re-generate restores the reference voice, instructions, effects, input type and
    # language from item.meta, not from whatever the page holds now.
    from src.voice_library import VoiceLibraryManager

    _reset_db(tmp_path)
    library = VoiceLibraryManager(tmp_path / "voices")
    library.save("narrator", _wav_bytes())
    monkeypatch.setattr(main_module, "voice_library", library)
    backend = MagicMock()
    backend.capabilities = {"voice_clone": True, "instructions": True}
    router = MagicMock()
    router.get_backend.return_value = backend
    router.synthesize.side_effect = lambda **_kwargs: iter([np.full(1600, 0.1, dtype=np.float32)])
    client = TestClient(app)
    cloned = {
        "model": "clone-model", "input": "Hello", "voice": "reference", "voice_library_ref": "narrator",
        "instructions": "Whisper.", "language": "en", "response_format": "pcm",
    }
    effects = [{"type": "normalize"}]

    with (
        patch.object(main_module, "tts_router", router),
        patch.object(main_module.settings, "tts_cache_enabled", False),
        patch.object(main_module.settings, "os_effects_enabled", True),
    ):
        for path, body in (
            ("/v1/audio/speech", {**cloned, "effects": effects}),
            ("/v1/audio/speech?stream=true", {
                "model": "clone-model", "input": "<speak>Hi</speak>", "input_type": "ssml",
                "voice": "reference", "response_format": "pcm",
            }),
        ):
            assert client.post(path, json=body, headers={"X-History": "true"}).status_code == 200

    streamed, whole = client.get("/api/history").json()["items"]
    assert whole["meta"] == {
        "reference_voice": "narrator", "instructions": "Whisper.", "effects": effects,
        "input_type": "text", "language": "en",
    }
    assert streamed["meta"] == {
        "reference_voice": None, "instructions": None, "effects": None,
        "input_type": "ssml", "language": None,
    }


@pytest.mark.parametrize(("method", "path", "call"), [
    ("GET", "/api/history?search=hello", "list_entries"),
    ("DELETE", "/api/history/some-entry", "delete_entry"),
    ("DELETE", "/api/history", "clear_all"),
])
async def test_history_routes_do_not_block_the_event_loop(tmp_path, monkeypatch, method, path, call):
    # delete_entry and clear_all wait on studio.db's shared transaction lock, which TTS and STT
    # history writes take from worker threads: waiting for it on the event loop would stall
    # every other request and live stream until the other writer is done.
    import httpx

    _reset_db(tmp_path)
    waiting, release = threading.Event(), threading.Event()
    released: list[bool] = []
    real = getattr(main_module.history_manager, call)

    def blocking(*args, **kwargs):
        waiting.set()
        released.append(release.wait(timeout=5))  # False: the loop could not release it
        return real(*args, **kwargs)

    monkeypatch.setattr(main_module.history_manager, call, blocking)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        slow = asyncio.create_task(client.request(method, path))
        try:
            assert await asyncio.to_thread(waiting.wait, 5)
            other = await asyncio.wait_for(client.get("/health"), timeout=5)
        finally:
            release.set()
        response = await asyncio.wait_for(slow, timeout=10)

    assert other.status_code == 200
    assert released == [True], "the history call blocked the event loop"
    assert response.status_code == (404 if path.endswith("some-entry") else 200)


# Ported from the old web UI's flow tests: the server answers the History
# view relies on for "Load more", search and "Open in Speak".


def test_history_paging_past_the_last_page_is_empty_and_keeps_the_total(tmp_path):
    # History's "Load more" stops when an offset reaches total; a page at or past it is empty.
    _reset_db(tmp_path)
    client = TestClient(app)
    hm = main_module.history_manager
    for n in range(51):
        hm.log_tts("kokoro", "af_heart", 1.0, "mp3", f"Entry number {n}.", None, 10)

    def page(offset):
        response = client.get("/api/history", params={"limit": 50, "offset": offset})
        assert response.status_code == 200
        return response.json()

    first, last, past = page(0), page(50), page(100)
    assert len(first["items"]) == 50 and first["total"] == 51
    assert [item["full_text"] for item in last["items"]] == ["Entry number 0."]   # newest first
    assert past["items"] == [] and past["total"] == 51

    assert client.delete(f"/api/history/{last['items'][0]['id']}").status_code == 204
    assert page(50)["items"] == [] and page(0)["total"] == 50


def test_history_search_lists_a_match_past_the_preview_with_its_whole_text(tmp_path):
    # The listed text_preview stops at 180 characters; search matches the whole text, and full_text
    # carries it (Copy text, Open in Speak). total counts the matches of every type.
    _reset_db(tmp_path)
    client = TestClient(app)
    hm = main_module.history_manager
    text = "Chapter one. " + "A quiet filler sentence about the sea. " * 8 + "The lighthouse keeper's log ends here."
    assert text.index("lighthouse") > 180
    hm.log_tts("kokoro", "af_heart", 1.0, "mp3", text, None, 10)
    hm.log_tts("kokoro", "af_heart", 1.0, "mp3", "Nothing to find here.", None, 10)
    hm.log_stt("Systran/faster-whisper-base", "harbour.wav", "Ships at the lighthouse.")

    everything = client.get("/api/history").json()
    found = client.get("/api/history", params={"search": "lighthouse"}).json()

    assert everything["total"] == 3 and found["total"] == 2
    speech = next(item for item in found["items"] if item["type"] == "tts")
    assert "lighthouse" not in speech["text_preview"] and speech["full_text"] == text
    assert {item["type"] for item in found["items"]} == {"tts", "stt"}


def test_a_speech_entry_records_the_blend_it_was_spoken_with(tmp_path):
    # "Open in Speak" restores the voice an entry was spoken with; a Kokoro blend is kept as its recipe.
    _reset_db(tmp_path)
    client = TestClient(app)
    router = MagicMock()
    router.synthesize.side_effect = lambda *_args, **_kwargs: iter([np.zeros(1600, dtype=np.float32)])
    blend = "af_bella(2)+af_sky(1)"

    with patch.object(main_module, "tts_router", router), patch.object(main_module.settings, "tts_cache_enabled", False):
        spoken = client.post(
            "/v1/audio/speech",
            json={"model": "kokoro", "input": "Blend sentence.", "voice": blend, "response_format": "pcm"},
            headers={"X-History": "true"},
        )

    assert spoken.status_code == 200, spoken.text
    entry = client.get("/api/history").json()["items"][0]
    assert entry["voice"] == blend and entry["model"] == "kokoro" and entry["full_text"] == "Blend sentence."


@pytest.fixture
def dictionary_client(tmp_path, monkeypatch):
    """The app with {"Ave": "Avenue"} as TTS_PRONUNCIATION_DICT and a recording TTS router."""
    import json

    from src.cache.tts_cache import TTSCache
    from src.pronunciation.dictionary import PronunciationDictionary

    _reset_db(tmp_path)
    dictionary = tmp_path / "dictionary.json"
    dictionary.write_text(json.dumps({"Ave": "Avenue"}), encoding="utf-8")
    monkeypatch.setattr(main_module, "pronunciation_dict", PronunciationDictionary(str(dictionary)))
    monkeypatch.setattr(main_module, "tts_cache", TTSCache(str(tmp_path / "tts-cache"), enabled=True))
    router = MagicMock()
    router.synthesize.side_effect = lambda **_kwargs: iter([np.zeros(1600, dtype=np.float32)])
    router.sample_rate_for.return_value = 24000
    router.loaded_models.return_value = []
    monkeypatch.setattr(main_module, "tts_router", router)
    return TestClient(app), router


@pytest.mark.parametrize("path", ["whole", "stream", "cache-hit"])
def test_history_logs_the_clients_text_so_open_in_speak_speaks_it_again(dictionary_client, monkeypatch, path):
    # Open in Speak puts full_text in the editor and Generate sends it back: the logged text must
    # be the client's input, or the dictionary is applied a second time ("5th Avenuenue").
    client, router = dictionary_client
    monkeypatch.setattr(main_module.settings, "tts_cache_enabled", path == "cache-hit")
    query = "?stream=true" if path == "stream" else ""
    body = {"model": "kokoro", "input": "5th Ave", "voice": "af_heart", "response_format": "wav"}
    if path == "cache-hit":
        assert client.post("/v1/audio/speech", json=body).status_code == 200  # fills the cache

    first = client.post("/v1/audio/speech" + query, json=body, headers={"X-History": "true"})
    assert first.status_code == 200, first.text
    assert path != "cache-hit" or first.headers.get("x-cache") == "HIT"
    logged = client.get("/api/history?type=tts").json()["items"][0]["full_text"]
    monkeypatch.setattr(main_module.settings, "tts_cache_enabled", False)
    again = client.post("/v1/audio/speech" + query, json={**body, "input": logged})

    assert again.status_code == 200, again.text
    assert logged == "5th Ave"
    assert router.synthesize.call_args.kwargs["text"] == "5th Avenue"


def test_history_keeps_ssml_markup_as_its_input_type_says(dictionary_client):
    client, _router = dictionary_client
    ssml = '<speak>Hi<break time="1s"/>there</speak>'
    resp = client.post(
        "/v1/audio/speech",
        json={"model": "kokoro", "input": ssml, "voice": "af_heart", "response_format": "wav", "input_type": "ssml"},
        headers={"X-History": "true"},
    )

    assert resp.status_code == 200, resp.text
    entry = client.get("/api/history?type=tts").json()["items"][0]
    assert (entry["full_text"], entry["meta"]["input_type"]) == (ssml, "ssml")


@pytest.mark.parametrize("parameter", ["limit", "offset"])
def test_history_paging_past_sqlites_integer_range_is_a_client_error(tmp_path, parameter):
    # SQLite binds at most 2**63 - 1; a larger value was an OverflowError 500.
    _reset_db(tmp_path)
    client = TestClient(app)

    assert client.get("/api/history", params={parameter: 2**63}).status_code == 422
    assert client.get("/api/history", params={parameter: 2**63 - 1}).status_code == 200
