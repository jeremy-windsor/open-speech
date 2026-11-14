"""GET /v1/audio/models as OpenAI-style clients read it (Open WebUI's "openai" TTS engine).

Open WebUI (backend/open_webui/routers/audio/__init__.py, get_available_models) fetches
``{base_url}/audio/models`` and keeps ``data.get("models", [])``; its admin Audio settings list each row as
``<option value={model.id}>``. A row without ``id`` shows as a blank entry. Its voice list
(get_available_voices) is ``{v["id"]: v["name"] for v in data.get("voices", [])}`` from ``/audio/voices``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src import main as main_module
from src.main import app
from src.model_manager import ModelInfo, ModelManager, ModelState
from src.tts.backends.base import TTSLoadedModelInfo, VoiceInfo


def open_webui_tts_models(payload: dict) -> list[str]:
    """What Open WebUI's TTS model list shows: each row's id, from data.get('models', [])."""
    return [model.get("id") for model in payload.get("models", [])]


def open_webui_tts_voices(payload: dict) -> dict[str, str]:
    """Open WebUI's voice map: {id: name} for each row of data.get('voices', [])."""
    return {voice["id"]: voice["name"] for voice in payload.get("voices", [])}


def _router(loaded: list[TTSLoadedModelInfo] | None = None) -> MagicMock:
    router = MagicMock()
    router.loaded_models.return_value = loaded or []
    router.list_backends.return_value = ["kokoro", "piper"]
    return router


_KOKORO_LOADED = TTSLoadedModelInfo(model="kokoro", backend="kokoro", device="cuda", loaded_at=1000.0, last_used_at=1001.0)


def _inventory(*infos: ModelInfo) -> SimpleNamespace:
    return SimpleNamespace(list_all=lambda: list(infos))


def _get(router, inventory, path="/v1/audio/models", tts_model="kokoro"):
    with patch.object(main_module, "tts_router", router), \
         patch.object(main_module, "model_manager", inventory), \
         patch.object(main_module.settings, "tts_model", tts_model):
        return TestClient(app).get(path)


def test_open_webui_lists_every_row_by_a_non_empty_id():
    inventory = _inventory(
        ModelInfo(id="kokoro", type="tts", provider="kokoro", device="cuda", state=ModelState.LOADED, is_default=True),
        ModelInfo(id="piper/en_US-amy-medium", type="tts", provider="piper", state=ModelState.DOWNLOADED),
    )
    response = _get(_router([_KOKORO_LOADED]), inventory)

    assert response.status_code == 200
    assert open_webui_tts_models(response.json()) == ["kokoro", "piper/en_US-amy-medium"]


def test_rows_keep_the_existing_fields_and_add_id_and_name():
    response = _get(_router([_KOKORO_LOADED]), _inventory())

    [row] = response.json()["models"]
    assert row["id"] == row["model"] == "kokoro"
    assert row["name"] == "Kokoro"
    assert (row["backend"], row["device"], row["status"], row["state"]) == ("kokoro", "cuda", "loaded", "loaded")
    assert (row["loaded_at"], row["last_used_at"], row["is_default"]) == (1000.0, 1001.0, True)


def test_lists_usable_models_that_are_not_loaded_with_their_state():
    inventory = _inventory(
        ModelInfo(id="kokoro", type="tts", provider="kokoro", state=ModelState.LOADED, is_default=True),
        ModelInfo(id="piper/en_US-amy-medium", type="tts", provider="piper", state=ModelState.DOWNLOADED,
                  description="US English - Amy voice"),
        ModelInfo(id="pocket-tts", type="tts", provider="pocket-tts", state=ModelState.AVAILABLE),
        ModelInfo(id="Systran/faster-whisper-small", type="stt", provider="faster-whisper", state=ModelState.DOWNLOADED),
        ModelInfo(id="chatterbox/turbo", type="tts", provider="chatterbox", state=ModelState.PROVIDER_UNAVAILABLE,
                  provider_available=False),
        ModelInfo(id="fish/s2-pro", type="tts", provider="fish", state=ModelState.PROVIDER_MISSING,
                  provider_available=False),
    )
    response = _get(_router([_KOKORO_LOADED]), inventory)

    rows = {row["id"]: row for row in response.json()["models"]}
    # No STT model, and no TTS model whose provider is missing or whose worker can't run.
    assert list(rows) == ["kokoro", "piper/en_US-amy-medium", "pocket-tts"]
    assert {i: (r["status"], r["state"]) for i, r in rows.items()} == {
        "kokoro": ("loaded", "loaded"),
        "piper/en_US-amy-medium": ("not_loaded", "downloaded"),
        "pocket-tts": ("not_loaded", "available"),
    }
    assert rows["piper/en_US-amy-medium"]["name"] == "Piper Amy (US, medium)"
    assert rows["piper/en_US-amy-medium"]["description"] == "US English - Amy voice"
    assert rows["pocket-tts"]["name"] == "Pocket TTS"
    assert rows["pocket-tts"]["device"] is None


def test_an_unusable_default_is_still_listed_with_its_state():
    inventory = _inventory(
        ModelInfo(id="chatterbox/turbo", type="tts", provider="chatterbox", state=ModelState.PROVIDER_UNAVAILABLE,
                  is_default=True, provider_available=False),
    )
    response = _get(_router(), inventory, tts_model="chatterbox/turbo")

    [row] = response.json()["models"]
    assert (row["id"], row["status"], row["state"], row["is_default"]) == (
        "chatterbox/turbo", "not_loaded", "provider_unavailable", True,
    )


def test_loaded_filter_lists_only_loaded_models():
    inventory = _inventory(
        ModelInfo(id="kokoro", type="tts", provider="kokoro", state=ModelState.LOADED, is_default=True),
        ModelInfo(id="pocket-tts", type="tts", provider="pocket-tts", state=ModelState.AVAILABLE),
    )
    router = _router([_KOKORO_LOADED])

    assert open_webui_tts_models(_get(router, inventory, "/v1/audio/models?loaded=true").json()) == ["kokoro"]
    assert _get(_router(), inventory, "/v1/audio/models?loaded=true").json() == {"models": []}


def test_the_real_inventory_lists_installed_providers_models(tmp_path):
    # ModelManager.list_all, as /api/models lists it: Kokoro loaded, Pocket TTS downloaded, Piper's voices
    # installed but not downloaded (provider_installed); Qwen3 (an isolated worker that isn't configured) left out.
    router = MagicMock()
    router._backends = {"kokoro": object(), "piper": object(), "pocket-tts": object()}
    router.loaded_models.return_value = [_KOKORO_LOADED]
    router.provider_is_available.return_value = True
    stt = MagicMock(_backends={})
    stt.loaded_models.return_value = []
    stt.list_cached_models.return_value = []
    manager = ModelManager(stt_router=stt, tts_router=router)
    (tmp_path / "models--pocket-tts").mkdir()

    with patch.object(manager, "_hf_cache_roots", return_value=[tmp_path]):
        response = _get(router, manager)

    rows = {row["id"]: row for row in response.json()["models"]}
    assert all(row_id for row_id in open_webui_tts_models(response.json()))
    assert rows["kokoro"]["status"] == "loaded"
    assert rows["pocket-tts"]["state"] == "downloaded"
    assert rows["piper/en_US-lessac-medium"]["state"] == "provider_installed"
    assert not any(row_id.startswith(("qwen3/", "chatterbox/")) for row_id in rows)
    assert not any(row_id.startswith("Systran/") for row_id in rows)


def test_open_webui_reads_voices_by_id_and_name():
    router = _router()
    router.list_voices.return_value = [
        VoiceInfo(id="af_heart", name="Heart", language="en-us", gender="female"),
        VoiceInfo(id="am_adam", name="Adam", language="en-us", gender="male"),
    ]
    response = _get(router, _inventory(), "/v1/audio/voices")

    assert response.status_code == 200
    assert open_webui_tts_voices(response.json()) == {"af_heart": "Heart", "am_adam": "Adam"}


@pytest.mark.parametrize(("model_id", "name"), [
    ("kokoro", "Kokoro"),
    ("pocket-tts", "Pocket TTS"),
    ("chatterbox/turbo", "Chatterbox turbo"),
    ("qwen3/0.6b-base", "Qwen3 TTS 0.6b-base"),
    ("piper/en_US-amy-medium", "Piper Amy (US, medium)"),
    ("piper/en_GB-southern_english_female-low", "Piper Southern English Female (GB, low)"),
    ("piper/en_US-libritts_r-medium", "Piper Libritts R (US, medium)"),
    ("someone/custom-model", "someone/custom-model"),
])
def test_model_display_name(model_id, name):
    from src.model_registry import model_display_name

    assert model_display_name(model_id) == name
