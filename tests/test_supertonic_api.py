"""Supertonic through the OpenAI-compatible endpoints, as the OpenAI SDK and Open WebUI call them (fake model)."""

from __future__ import annotations

import io
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import main as main_module
from src.cache.tts_cache import TTSCache
from src.main import app
from src.model_manager import ModelManager
from src.tts.backends.base import TTSLoadedModelInfo, VoiceInfo
from src.tts.router import TTSRouter
from tests.test_supertonic_backend import fake_package, model_dir  # noqa: F401 (fixtures)


class PlainBackend:
    """A model without options: it would fail on any keyword it doesn't know."""

    name = "plain"
    sample_rate = 24000
    capabilities = {"speed_control": True, "voice_blend": False}

    def load_model(self, model_id): pass
    def unload_model(self, model_id): pass
    def is_model_loaded(self, model_id): return True
    def loaded_models(self): return []
    def list_voices(self): return [VoiceInfo(id="v", name="V")]

    def synthesize(self, text, voice, speed=1.0, lang_code=None):
        yield np.zeros(2400, dtype=np.float32)


@pytest.fixture
def server(fake_package, model_dir):  # noqa: F811
    router = TTSRouter(device="cpu", external_providers="")
    router.register_backend("plain", PlainBackend())
    stt = MagicMock(_backends={})
    stt.loaded_models.return_value = []
    stt.list_cached_models.return_value = []
    manager = ModelManager(stt_router=stt, tts_router=router)
    with (
        patch.object(main_module, "tts_router", router),
        patch.object(main_module, "model_manager", manager),
        patch.object(main_module.settings, "tts_cache_enabled", False),
        patch.object(main_module.settings, "tts_trim_silence", False),
        patch("src.tts.pipeline.encode_with_ffmpeg", side_effect=lambda audio, fmt, sample_rate=24000: f"{fmt}:{sample_rate}".encode()),
    ):
        yield TestClient(app), router, manager


def engine_calls(router):
    return router.get_backend("supertonic")._loaded["engine"].calls


def test_an_openai_sdk_request(server):
    # openai-python's audio.speech.create(model=..., voice=..., input=..., response_format="wav") body.
    client, router, _ = server
    response = client.post("/v1/audio/speech", json={
        "input": "Hello from Supertonic.", "model": "supertonic", "voice": "M1", "response_format": "wav",
    }, headers={"Authorization": "Bearer sk-anything", "User-Agent": "OpenAI/Python 1.99.0"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    with wave.open(io.BytesIO(response.content)) as wav:
        assert wav.getframerate() == 44100 and wav.getnframes() == 4410
    assert engine_calls(router)[-1]["steps"] == 8, "a plain OpenAI request gets the default steps"


def test_an_open_webui_request(server):
    # Open WebUI's openai TTS engine posts {"input", "model", "voice"} and plays the default format.
    client, router, _ = server
    response = client.post("/v1/audio/speech", json={"input": "Open WebUI says hi.", "model": "supertonic", "voice": "F2"})
    assert response.status_code == 200
    assert response.content == f"{main_module.settings.tts_default_format}:44100".encode()
    assert engine_calls(router)[-1]["text"] == "Open WebUI says hi."


@pytest.mark.parametrize("fmt", ["mp3", "opus", "aac", "flac", "m4a"])
def test_every_compressed_format(server, fmt):
    client, _, _ = server
    response = client.post("/v1/audio/speech", json={"input": "Hi.", "model": "supertonic", "voice": "M1", "response_format": fmt})
    assert response.status_code == 200
    assert response.content == f"{fmt}:44100".encode()


def test_pcm_is_24khz(server):
    client, _, _ = server
    response = client.post("/v1/audio/speech", json={"input": "Hi.", "model": "supertonic", "voice": "M1", "response_format": "pcm"})
    assert response.status_code == 200
    assert len(response.content) == 2 * 2400, "0.1 s at 24 kHz, 16-bit"


def test_streaming_sends_each_sentence_as_it_is_made(server):
    client, router, _ = server
    sent = []

    def encoder(chunks, fmt, sample_rate, output_sample_rate=None):
        for chunk in chunks:
            sent.append((len(chunk), fmt, sample_rate))
            yield b"x" * len(chunk)

    with (
        patch("src.services.tts.encode_audio_streaming", encoder),
        client.stream("POST", "/v1/audio/speech?stream=true", json={
            "input": "One sentence. Another sentence.", "model": "supertonic", "voice": "M3", "response_format": "mp3",
        }) as response,
    ):
        body = b"".join(response.iter_bytes())
    assert response.status_code == 200
    assert len(engine_calls(router)) == 2
    assert sent == [(4410, "mp3", 44100), (13230, "mp3", 44100), (4410, "mp3", 44100)], "sentence, pause, sentence"
    assert len(body) == 4410 + 13230 + 4410


def test_steps_reach_supertonic_and_are_checked(server):
    client, router, _ = server
    ok = client.post("/v1/audio/speech", json={"input": "Hi.", "model": "supertonic", "voice": "M1", "response_format": "wav", "steps": 4})
    assert ok.status_code == 200 and engine_calls(router)[-1]["steps"] == 4
    too_many = client.post("/v1/audio/speech", json={"input": "Hi.", "model": "supertonic", "voice": "M1", "steps": 40})
    assert too_many.status_code == 400 and "steps must be from 1 to 32" in too_many.json()["error"]["message"]
    assert client.post("/v1/audio/speech", json={"input": "Hi.", "model": "supertonic", "voice": "M1", "steps": 0}).status_code == 422


def test_a_model_without_steps_ignores_them(server):
    client, _, _ = server
    response = client.post("/v1/audio/speech", json={"input": "Hi.", "model": "plain", "voice": "v", "response_format": "wav", "steps": 4})
    assert response.status_code == 200


def test_a_blend_and_an_unknown_voice(server):
    client, _, _ = server
    ok = client.post("/v1/audio/speech", json={"input": "Mix.", "model": "supertonic", "voice": "M1(0.6)+F2(0.4)", "response_format": "wav"})
    assert ok.status_code == 200
    bad = client.post("/v1/audio/speech", json={"input": "Hi.", "model": "supertonic", "voice": "Kokoro"})
    assert bad.status_code == 400 and "Unknown Supertonic voice" in bad.json()["error"]["message"]


def test_steps_are_part_of_the_cache_key(server, tmp_path):
    client, router, _ = server
    body = {"input": "Cached.", "model": "supertonic", "voice": "M1", "response_format": "wav"}
    with (
        patch.object(main_module, "tts_cache", TTSCache(str(tmp_path / "cache"), enabled=True)),
        patch.object(main_module.settings, "tts_cache_enabled", True),
    ):
        client.post("/v1/audio/speech", json=body)
        assert client.post("/v1/audio/speech", json=body).headers.get("x-cache") == "HIT"
        assert client.post("/v1/audio/speech", json={**body, "steps": 4}).headers.get("x-cache") is None
        assert client.post("/v1/audio/speech", json={**body, "steps": 4}).headers.get("x-cache") == "HIT"
    assert [c["steps"] for c in engine_calls(router)] == [8, 4]


def test_a_speed_outside_the_models_range_is_held_to_it_and_named_in_a_header(server, tmp_path):
    client, router, _ = server
    body = {"input": "Hi.", "model": "supertonic", "voice": "M1", "response_format": "wav"}
    for speed, applied in ((0.25, "0.7"), (4.0, "1.9")):
        response = client.post("/v1/audio/speech", json={**body, "speed": speed})
        assert response.status_code == 200, speed
        assert response.headers["x-speed-applied"] == applied
        assert engine_calls(router)[-1]["speed"] == pytest.approx(float(applied) * 1.05)
    in_range = client.post("/v1/audio/speech", json={**body, "speed": 1.5})
    assert in_range.status_code == 200 and "x-speed-applied" not in in_range.headers
    with client.stream("POST", "/v1/audio/speech?stream=true", json={**body, "speed": 3.0}) as streamed:
        streamed.read()
    assert streamed.status_code == 200 and streamed.headers["x-speed-applied"] == "1.9"
    with (
        patch.object(main_module, "tts_cache", TTSCache(str(tmp_path / "cache"), enabled=True)),
        patch.object(main_module.settings, "tts_cache_enabled", True),
    ):
        client.post("/v1/audio/speech", json={**body, "speed": 1.9})
        hit = client.post("/v1/audio/speech", json={**body, "speed": 2.5})
    assert hit.headers.get("x-cache") == "HIT" and hit.headers["x-speed-applied"] == "1.9", "the applied speed is the key"
    plain = client.post("/v1/audio/speech", json={**body, "model": "plain", "voice": "v", "speed": 4.0})
    assert plain.status_code == 200 and "x-speed-applied" not in plain.headers


def test_reader_studio_live_and_talk_get_the_models_speed_too(server):
    _, router, _ = server
    from src.services.tts import synthesize_array

    synthesize_array(text="Read on.", model="supertonic", voice="F2", speed=2.5, tts_router=router,
                     settings=main_module.settings)
    assert engine_calls(router)[-1]["speed"] == pytest.approx(1.9 * 1.05)
    list(router.synthesize(text="Live reply.", model="supertonic", voice="F2", speed=0.25))
    assert engine_calls(router)[-1]["speed"] == pytest.approx(0.7 * 1.05)


def test_a_tts_speed_setting_outside_the_range_is_held_to_it(server):
    client, router, _ = server
    with patch.object(main_module.settings, "tts_speed", 2.5):
        response = client.post("/v1/audio/speech", json={"input": "Hi.", "model": "supertonic", "voice": "M1", "response_format": "wav"})
    assert response.status_code == 200
    assert engine_calls(router)[-1]["speed"] == pytest.approx(1.9 * 1.05)


@pytest.mark.parametrize("text", ["😀😀", "...", "#", "  —  "])
@pytest.mark.parametrize("stream", [False, True])
def test_text_with_nothing_to_say_is_a_400_like_empty_text(server, text, stream):
    client, _, _ = server
    url = "/v1/audio/speech?stream=true" if stream else "/v1/audio/speech"
    response = client.post(url, json={"input": text, "model": "supertonic", "voice": "M1", "response_format": "wav"})
    assert response.status_code == 400
    assert response.json()["error"]["message"] == "Input text is empty"
    empty = client.post(url, json={"input": "", "model": "supertonic", "voice": "M1"})
    assert empty.status_code == 400 and empty.json()["error"]["message"] == "Input text is empty"
    assert client.post(url, json={"input": "...", "model": "plain", "voice": "v", "response_format": "wav"}).status_code == 200


@pytest.mark.parametrize("steps", [True, "8", 8.5, 8.0])
def test_steps_that_are_not_an_integer_are_a_422(server, steps):
    client, _, _ = server
    response = client.post("/v1/audio/speech", json={"input": "Hi.", "model": "supertonic", "voice": "M1", "steps": steps})
    assert response.status_code == 422


def test_the_cache_key_has_steps_only_where_they_change_the_audio(server, tmp_path):
    client, router, _ = server
    plain = {"input": "Cached.", "model": "plain", "voice": "v", "response_format": "wav"}
    body = {"input": "Cached.", "model": "supertonic", "voice": "M1", "response_format": "wav"}
    with (
        patch.object(main_module, "tts_cache", TTSCache(str(tmp_path / "cache"), enabled=True)),
        patch.object(main_module.settings, "tts_cache_enabled", True),
    ):
        client.post("/v1/audio/speech", json=plain)
        assert client.post("/v1/audio/speech", json={**plain, "steps": 4}).headers.get("x-cache") == "HIT"
        client.post("/v1/audio/speech", json=body)
        assert client.post("/v1/audio/speech", json={**body, "steps": 8}).headers.get("x-cache") == "HIT"
        client.post("/v1/audio/speech", json={**body, "steps": 4})
        assert client.post("/v1/audio/speech", json=body).headers.get("x-cache") == "HIT"
    assert [c["steps"] for c in engine_calls(router)] == [8, 4]


def test_history_keeps_steps_only_for_a_model_that_has_them(server):
    client, _, _ = server
    history = MagicMock()
    with (
        patch.object(main_module, "history_manager", history),
        patch.object(main_module.settings, "os_history_enabled", True),
    ):
        for model, voice in (("plain", "v"), ("supertonic", "M1")):
            client.post("/v1/audio/speech", json={
                "input": "Logged.", "model": model, "voice": voice, "response_format": "wav", "steps": 4,
            }, headers={"x-history": "true"})
    plain_meta, supertonic_meta = (call.kwargs["meta"] for call in history.log_tts.call_args_list)
    assert "steps" not in plain_meta
    assert supertonic_meta["steps"] == 4


def test_openai_voice_names_and_a_cross_model_blend(server):
    client, router, _ = server
    for voice, expected in (("nova", "F2"), ("onyx", "M1")):
        assert client.post("/v1/audio/speech", json={"input": "Hi.", "model": "supertonic", "voice": voice, "response_format": "wav"}).status_code == 200
        styles = router.get_backend("supertonic")._loaded["styles"]
        np.testing.assert_allclose(engine_calls(router)[-1]["style"].ttl, styles[expected][0])
    bad = client.post("/v1/audio/speech", json={"input": "Hi.", "model": "supertonic", "voice": "M1(0.5)+af_heart(0.5)"})
    assert bad.status_code == 400
    message = bad.json()["error"]["message"]
    assert "af_heart" in message and "AF_HEART" not in message and "A blend must use one model's voices" in message


def test_paragraphs_are_read_as_separate_chunks(server):
    client, router, _ = server
    response = client.post("/v1/audio/speech", json={
        "input": "Chapter 1\n\nIt was dark\nThe end", "model": "supertonic", "voice": "M1", "response_format": "wav",
    })
    assert response.status_code == 200
    assert [c["text"] for c in engine_calls(router)] == ["Chapter one", "It was dark", "The end"]


def test_voices_list_with_their_model(server):
    client, _, _ = server
    response = client.get("/v1/audio/voices", params={"model": "supertonic"})
    assert response.status_code == 200
    voices = response.json()["voices"]
    assert [v["id"] for v in voices] == ["M1", "M2", "M3", "M4", "M5", "F1", "F2", "F3", "F4", "F5"]
    assert voices[0] == {"id": "M1", "name": "M1", "language": "en", "gender": "male", "model": "supertonic"}
    # Open WebUI's voice map, {id: name}.
    assert {v["id"]: v["name"] for v in voices}["F2"] == "F2"


def test_listed_in_the_model_lists(server):
    client, router, manager = server
    with patch.object(manager, "_hf_cache_roots", return_value=[]):
        rows = {row["id"]: row for row in client.get("/v1/audio/models").json()["models"]}
    assert rows["supertonic"]["name"] == "Supertonic"
    assert rows["supertonic"]["state"] == "provider_installed"
    assert client.get("/v1/models/supertonic").status_code == 200
    router.load_model("supertonic")
    listed = [m["id"] for m in client.get("/v1/models").json()["data"]]
    assert "supertonic" in listed, "a loaded model is listed"
    caps = client.get("/api/tts/capabilities", params={"model": "supertonic"}).json()
    assert caps["sample_rate"] == 44100 and caps["capabilities"]["options"]["steps"]["default"] == 8


def test_the_gpu_memory_guard_knows_where_it_runs(server, monkeypatch):
    _, router, manager = server
    assert manager.gpu_need_mb("supertonic", "tts") is None, "on the CPU it needs no GPU memory"
    backend = router.get_backend("supertonic")
    monkeypatch.setattr(backend, "_get_device", lambda: "cuda")
    assert manager.gpu_need_mb("supertonic", "tts") == 1000
    router.load_model("supertonic")
    backend._loaded["device"] = "cuda"
    [resident] = [r for r in manager._gpu_residents("kokoro", "tts") if r.id == "supertonic"]
    assert resident.gpu_mb == 1000


def test_registry_entry():
    from src.model_registry import get_known_model, model_display_name

    entry = get_known_model("supertonic")
    assert (entry["type"], entry["provider"], entry["size_mb"], entry["gpu_mb"]) == ("tts", "supertonic", 383, 1000)
    assert model_display_name("supertonic") == "Supertonic"
    assert isinstance(TTSLoadedModelInfo(model="supertonic", backend="supertonic", device="cpu", loaded_at=0), TTSLoadedModelInfo)
