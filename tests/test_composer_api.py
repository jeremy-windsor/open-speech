from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.io import wavfile

from src import main as main_module
from src import storage as storage_module
from src.main import app


def _reset_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "voices").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "conversations").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "composer").mkdir(parents=True, exist_ok=True)
    main_module.settings.os_studio_db_path = str(tmp_path / "studio.db")
    main_module.settings.os_composer_dir = str(tmp_path / "data" / "composer")
    storage_module._conn = None
    storage_module.init_db()
    main_module.composer_manager.output_dir = Path(main_module.settings.os_composer_dir)
    main_module.composer_manager.output_dir.mkdir(parents=True, exist_ok=True)


def _write_wav(path: Path, amp: float = 0.2):
    sr = 24000
    t = np.arange(sr // 10) / sr
    audio = (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wavfile.write(path, sr, (audio * 32767).astype(np.int16))


def test_render_two_tracks_returns_composition_and_output(tmp_path, monkeypatch):
    _reset_env(tmp_path, monkeypatch)
    a = tmp_path / "data" / "voices" / "a.wav"
    b = tmp_path / "data" / "voices" / "b.wav"
    _write_wav(a)
    _write_wav(b)
    client = TestClient(app)

    resp = client.post("/api/composer/render", json={
        "name": "Episode Intro",
        "format": "wav",
        "sample_rate": 24000,
        "tracks": [
            {"source_path": "data/voices/a.wav", "offset_s": 0.0, "volume": 1.0, "muted": False, "solo": False, "effects": []},
            {"source_path": "data/voices/b.wav", "offset_s": 0.05, "volume": 0.7, "muted": False, "solo": False, "effects": []},
        ],
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["composition_id"]
    output_path = Path(body["output_path"])
    assert output_path.parts[:2] == ("data", "composer")
    assert output_path.name.startswith("render_")
    assert body["duration_ms"] > 0


def test_muted_and_solo_behaviors(tmp_path, monkeypatch):
    _reset_env(tmp_path, monkeypatch)
    c = main_module.composer_manager

    t1 = {"samples": np.ones(100, dtype=np.float32), "offset_s": 0.0, "muted": False, "solo": False}
    t2 = {"samples": np.ones(100, dtype=np.float32), "offset_s": 0.0, "muted": True, "solo": False}
    active = c._active_tracks([t1, t2])
    assert len(active) == 1

    t3 = {"samples": np.ones(100, dtype=np.float32), "offset_s": 0.0, "muted": False, "solo": True}
    active_solo = c._active_tracks([t1, t3])
    assert len(active_solo) == 1
    assert active_solo[0]["solo"] is True


def test_path_outside_safe_dirs_rejected(tmp_path, monkeypatch):
    _reset_env(tmp_path, monkeypatch)
    outside = tmp_path / "evil.wav"
    _write_wav(outside)
    client = TestClient(app)

    resp = client.post("/api/composer/render", json={
        "format": "wav",
        "tracks": [
            {"source_path": str(outside), "offset_s": 0.0, "volume": 1.0, "muted": False, "solo": False, "effects": []},
        ],
    })
    assert resp.status_code in (400, 403)


def test_outside_path_rejection_does_not_reveal_existence(tmp_path, monkeypatch):
    _reset_env(tmp_path, monkeypatch)
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    existing = outside_dir / "present.wav"
    _write_wav(existing)
    client = TestClient(app)

    def render(path):
        return client.post("/api/composer/render", json={"format": "wav", "tracks": [{"source_path": str(path)}]})

    present, absent = render(existing), render(outside_dir / "absent.wav")
    assert present.status_code == 403
    assert (present.status_code, present.json()) == (absent.status_code, absent.json())

    missing_inside = render("data/voices/absent.wav")
    assert missing_inside.status_code == 400
    assert "not found" in missing_inside.json()["error"]["message"]


def test_list_and_delete_renders(tmp_path, monkeypatch):
    _reset_env(tmp_path, monkeypatch)
    src = tmp_path / "data" / "voices" / "base.wav"
    _write_wav(src)
    client = TestClient(app)

    created = client.post("/api/composer/render", json={
        "format": "wav",
        "tracks": [{"source_path": "data/voices/base.wav", "offset_s": 0.0, "volume": 1.0, "muted": False, "solo": False, "effects": []}],
    })
    cid = created.json()["composition_id"]

    listing = client.get("/api/composer/renders")
    assert listing.status_code == 200
    assert any(item["id"] == cid for item in listing.json()["items"])

    deleted = client.delete(f"/api/composer/render/{cid}")
    assert deleted.status_code == 204

    listing2 = client.get("/api/composer/renders")
    assert all(item["id"] != cid for item in listing2.json()["items"])


def test_track_with_an_invalid_effect_parameter_is_a_client_error(tmp_path, monkeypatch):
    _reset_env(tmp_path, monkeypatch)
    _write_wav(tmp_path / "data" / "voices" / "a.wav")
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.post("/api/composer/render", json={
        "format": "wav",
        "tracks": [
            {"source_path": "data/voices/a.wav"},
            {"source_path": "data/voices/a.wav", "effects": [{"type": "reverb", "mix": "heavy"}]},
        ],
    })

    assert resp.status_code == 400
    assert resp.json()["error"]["message"] == "Track 2: reverb mix must be a number from 0 to 1"
    assert client.get("/api/composer/renders").json()["total"] == 0


def test_composer_render_runs_off_the_event_loop(monkeypatch):
    # Decoding, effects, resampling, mixing and ffmpeg must not stall every other request.
    import asyncio

    threads = []

    class RecordingComposer:
        def render(self, **_kwargs):
            try:
                asyncio.get_running_loop()
                threads.append("event loop")
            except RuntimeError:
                threads.append("worker thread")
            return {"composition_id": "c", "output_path": "c.wav", "download_url": "/c", "duration_ms": 0}

    monkeypatch.setattr(main_module, "composer_manager", RecordingComposer())

    resp = TestClient(app).post("/api/composer/render", json={"tracks": [{"source_path": "data/voices/a.wav"}]})

    assert resp.status_code == 200
    assert threads == ["worker thread"]


def test_composition_longer_than_the_render_limit_is_refused_before_any_source_is_decoded(tmp_path, monkeypatch):
    # The mix buffer spans up to the latest track end, and each decoded track is as long as its
    # source, so both are checked from offsets and WAV headers alone.
    from src import composer as composer_module

    _reset_env(tmp_path, monkeypatch)
    _write_wav(tmp_path / "data" / "voices" / "a.wav")  # 0.1 s
    wavfile.write(tmp_path / "data" / "voices" / "long.wav", 8000, np.zeros(12000, dtype=np.int16))  # 1.5 s
    monkeypatch.setattr(composer_module, "MAX_RENDER_SECONDS", 1)
    load = MagicMock(side_effect=AssertionError("decoded a source of a composition over the limit"))
    monkeypatch.setattr(main_module.composer_manager, "_load_audio", load)
    client = TestClient(app, raise_server_exceptions=False)

    def render(source, offset_s, sample_rate=24000):
        return client.post("/api/composer/render", json={
            "format": "wav", "sample_rate": sample_rate,
            "tracks": [{"source_path": "data/voices/a.wav"}, {"source_path": source, "offset_s": offset_s}],
        })

    ends_late, starts_late = render("data/voices/a.wav", 0.95), render("data/voices/a.wav", 1.5)
    long_source = render("data/voices/long.wav", 0.0, sample_rate=48000)

    for resp in (ends_late, starts_late, long_source):
        assert resp.status_code == 400
        assert resp.json()["error"]["message"] == "Composition is too long. Max: 1 seconds"
    load.assert_not_called()


def test_composer_memory_does_not_grow_with_the_track_count(tmp_path, monkeypatch):
    # Each track is decoded, added to the mix and released before the next one is loaded, so
    # 64 tracks of a long source need no more memory than one.
    import tracemalloc

    _reset_env(tmp_path, monkeypatch)
    sr = 24000
    wavfile.write(tmp_path / "data" / "voices" / "long.wav", sr, np.zeros(5 * sr, dtype=np.int16))
    composer = main_module.composer_manager

    def render(n_tracks):
        composer.render(tracks=[{"source_path": "data/voices/long.wav"} for _ in range(n_tracks)], sample_rate=sr)

    def peak_for(n_tracks):
        tracemalloc.start()
        try:
            render(n_tracks)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    render(1)  # leave one-time allocations out of the measurement
    one, many = peak_for(1), peak_for(24)

    assert many < 2 * one, f"peak with 1 track: {one} bytes, with 24 tracks: {many} bytes"


def test_composer_renders_one_composition_at_a_time(tmp_path, monkeypatch):
    # Renders run in worker threads; two at once would hold two mixes and two decoded tracks.
    import threading

    _reset_env(tmp_path, monkeypatch)
    _write_wav(tmp_path / "data" / "voices" / "a.wav")
    composer = main_module.composer_manager
    real_load = composer._load_audio
    first_loading, release = threading.Event(), threading.Event()
    loading, overlaps, results = [], [], []

    def load(path):
        loading.append(path)
        overlaps.append(len(loading))
        first_loading.set()
        release.wait(timeout=10)
        try:
            return real_load(path)
        finally:
            loading.remove(path)

    def render():
        results.append(composer.render(tracks=[{"source_path": "data/voices/a.wav"}]))

    monkeypatch.setattr(composer, "_load_audio", load)
    first, second = threading.Thread(target=render), threading.Thread(target=render)
    first.start()
    try:
        assert first_loading.wait(timeout=5)
        second.start()
        second.join(timeout=0.5)  # the second render waits for the first one
    finally:
        release.set()
        first.join(timeout=10)
        if second.ident is not None:
            second.join(timeout=10)

    assert overlaps == [1, 1]
    assert len(results) == 2


def test_effect_errors_name_the_track_by_its_position_in_the_request(tmp_path, monkeypatch):
    _reset_env(tmp_path, monkeypatch)
    _write_wav(tmp_path / "data" / "voices" / "a.wav")
    client = TestClient(app, raise_server_exceptions=False)

    resp = client.post("/api/composer/render", json={
        "format": "wav",
        "tracks": [
            {"source_path": "data/voices/a.wav", "muted": True},
            {"source_path": "data/voices/a.wav"},
            {"source_path": "data/voices/a.wav", "effects": [{"type": "reverb", "mix": "heavy"}]},
        ],
    })

    assert resp.status_code == 400
    assert resp.json()["error"]["message"] == "Track 3: reverb mix must be a number from 0 to 1"


def test_deleting_a_render_whose_file_is_still_open_succeeds(tmp_path, monkeypatch):
    # A download keeps the file open, and Windows refuses to remove an open file.
    _reset_env(tmp_path, monkeypatch)
    _write_wav(tmp_path / "data" / "voices" / "a.wav")
    client = TestClient(app, raise_server_exceptions=False)
    created = client.post("/api/composer/render", json={"tracks": [{"source_path": "data/voices/a.wav"}]}).json()

    with open(tmp_path / created["output_path"], "rb"):
        deleted = client.delete(f"/api/composer/render/{created['composition_id']}")

    assert deleted.status_code == 204
    assert client.get("/api/composer/renders").json()["total"] == 0


def test_effects_disabled_setting_also_disables_composer_effects(tmp_path, monkeypatch):
    _reset_env(tmp_path, monkeypatch)
    source = tmp_path / "data" / "voices" / "a.wav"
    _write_wav(source, amp=0.5)
    monkeypatch.setattr(main_module.settings, "os_effects_enabled", False)

    resp = TestClient(app).post("/api/composer/render", json={
        "format": "wav", "tracks": [{"source_path": "data/voices/a.wav", "effects": [{"type": "robot"}]}],
    })

    assert resp.status_code == 200
    _, rendered = wavfile.read(resp.json()["output_path"])
    _, original = wavfile.read(source)
    assert np.max(np.abs(rendered.astype(np.int32) - original.astype(np.int32))) <= 1


def test_a_mix_downloads_from_its_download_url_in_its_format(tmp_path, monkeypatch):
    # Ported from the old web UI's flow tests at the swap: the Composer's Download saves the mix the
    # player shows, from download_url, named for the format it is served in.
    _reset_env(tmp_path, monkeypatch)
    _write_wav(tmp_path / "data" / "voices" / "track.wav")
    client = TestClient(app)

    rendered = client.post("/api/composer/render", json={
        "format": "wav", "sample_rate": 24000,
        "tracks": [{"source_path": "data/voices/track.wav", "offset_s": 0.0, "volume": 1.0, "muted": False, "solo": False, "effects": []}],
    })
    assert rendered.status_code == 200, rendered.text
    served = client.get(rendered.json()["download_url"])

    assert served.status_code == 200
    assert served.headers["content-type"].startswith("audio/wav") and served.content[:4] == b"RIFF"


@pytest.mark.parametrize("parameter", ["limit", "offset"])
def test_composer_render_paging_past_sqlites_integer_range_is_a_client_error(tmp_path, monkeypatch, parameter):
    # SQLite binds at most 2**63 - 1; a larger value was an OverflowError 500.
    _reset_env(tmp_path, monkeypatch)
    client = TestClient(app)

    assert client.get("/api/composer/renders", params={parameter: 2**63}).status_code == 422
    assert client.get("/api/composer/renders", params={parameter: 2**63 - 1}).status_code == 200


def test_a_lone_surrogate_in_a_mixs_name_is_refused_before_it_renders(tmp_path, monkeypatch):
    # Half of an emoji's UTF-16 pair (the JSON escape "\ud83d") in the name
    # reached studio.db after the whole mix had rendered, which can't encode it: a 500 and a stray file.
    _reset_env(tmp_path, monkeypatch)
    _write_wav(tmp_path / "data" / "voices" / "a.wav")
    client = TestClient(app, raise_server_exceptions=False)
    body = '{"name": "Intro \\ud83d", "tracks": [{"source_path": "data/voices/a.wav"}]}'

    response = client.post("/api/composer/render", content=body, headers={"content-type": "application/json"})

    assert response.status_code == 422, response.text
    assert "name" in response.text and "surrogate" in response.text
    assert client.get("/api/composer/renders").json()["total"] == 0
    assert list((tmp_path / "data" / "composer").iterdir()) == []
