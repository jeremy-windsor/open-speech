from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from scipy.io import wavfile

from src.main import app
from src import main as main_module
from src import storage as storage_module


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
    assert body["output_path"].startswith("data/composer/render_")
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
