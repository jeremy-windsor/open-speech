"""Tests for TTSRouter register_backend and auto-discovery."""

from __future__ import annotations

import io
import json
import sys
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from src.tts.backends import kokoro as kokoro_module
from src.tts.backends.base import TTSLoadedModelInfo, VoiceInfo
from src.tts.backends.kokoro import KokoroBackend
from src.tts import router as router_module
from src.tts.router import TTSRouter, _discover_backends


class FakeBackend:
    """Minimal TTSBackend implementation for testing."""
    name: str = "fake"
    sample_rate: int = 16000

    @classmethod
    def is_available(cls) -> bool:
        return True

    def __init__(self, device: str = "cpu") -> None:
        self._loaded = False

    def load_model(self, model_id: str) -> None:
        self._loaded = True

    def unload_model(self, model_id: str) -> None:
        self._loaded = False

    def is_model_loaded(self, model_id: str) -> bool:
        return self._loaded

    def loaded_models(self) -> list[TTSLoadedModelInfo]:
        return []

    def synthesize(self, text, voice, speed=1.0, lang_code=None):
        yield np.zeros(100, dtype=np.float32)

    def list_voices(self) -> list[VoiceInfo]:
        return [VoiceInfo(id="fake_voice", name="Fake")]


class ModelRateBackend(FakeBackend):
    def get_sample_rate(self, model_id: str) -> int:
        return 16000 if model_id == "fake/low-rate" else 22050


class ConcurrentBackend(FakeBackend):
    def __init__(self, name: str, barrier: threading.Barrier) -> None:
        super().__init__()
        self.name = name
        self._barrier = barrier

    def supports_model(self, model_id: str) -> bool:
        return model_id == self.name

    def synthesize(self, text, voice, speed=1.0, lang_code=None):
        self._barrier.wait(timeout=2)
        yield np.zeros(10, dtype=np.float32)


class ResidentBackend(FakeBackend):
    """Lists its model while it is loaded, as the in-process backends do."""

    def loaded_models(self) -> list[TTSLoadedModelInfo]:
        if not self._loaded:
            return []
        return [TTSLoadedModelInfo(model=self.name, backend=self.name, device="cpu", loaded_at=time.time())]


class UnavailableBackend(FakeBackend):
    name = "optional"

    @classmethod
    def is_available(cls) -> bool:
        return False


class AvailableBackend(FakeBackend):
    name = "available"


class TestRegisterBackend:
    def test_register_and_use(self):
        router = TTSRouter(device="cpu")
        fake = FakeBackend()
        router.register_backend("fake", fake)

        assert "fake" in router.list_backends()
        backend = router.get_backend("fake")
        assert backend is fake

    def test_unknown_model_does_not_fall_back_to_registered_default(self):
        router = TTSRouter.__new__(TTSRouter)
        router._backends = {}
        router._default_backend = None
        router._device = "cpu"

        fake = FakeBackend()
        router.register_backend("fake", fake)
        with pytest.raises(ValueError, match="Unknown TTS model or backend"):
            router.get_backend("nonexistent")

    def test_unknown_prefixed_model_is_rejected_when_backend_declines_it(self):
        router = TTSRouter.__new__(TTSRouter)
        router._backends = {}
        router._default_backend = None
        router._device = "cpu"

        fake = FakeBackend()
        fake.supports_model = lambda model_id: model_id == "fake/known"
        router.register_backend("fake", fake)

        assert router.get_backend("fake/known") is fake
        with pytest.raises(ValueError, match="fake/unknown"):
            router.get_backend("fake/unknown")

    def test_list_backends(self):
        with patch(
            "src.tts.router._discover_backends",
            return_value={"available": AvailableBackend},
        ):
            router = TTSRouter(device="cpu")

        assert router.list_backends() == ["available"]

    def test_voices_from_registered_backend(self):
        router = TTSRouter(device="cpu")
        fake = FakeBackend()
        router.register_backend("fake", fake)

        voices = router.list_voices("fake")
        assert any(v.id == "fake_voice" for v in voices)

    def test_aggregate_voices(self):
        with patch("src.tts.router._discover_backends", return_value={}):
            router = TTSRouter(device="cpu")
        fake = FakeBackend()
        router.register_backend("fake", fake)

        second = FakeBackend()
        second.list_voices = lambda: [VoiceInfo(id="second_voice", name="Second")]
        router.register_backend("second", second)

        all_voices = router.list_voices()
        ids = [v.id for v in all_voices]
        assert "fake_voice" in ids
        assert "second_voice" in ids

    def test_sample_rate_uses_model_specific_backend_lookup(self):
        with patch("src.tts.router._discover_backends", return_value={}):
            router = TTSRouter(device="cpu")
        router.register_backend("fake", ModelRateBackend())

        assert router.sample_rate_for("fake/low-rate") == 16000
        assert router.sample_rate_for("fake/medium-rate") == 22050

    def test_sample_rate_falls_back_to_backend_attribute(self):
        with patch("src.tts.router._discover_backends", return_value={}):
            router = TTSRouter(device="cpu")
        router.register_backend("fake", FakeBackend())

        assert router.sample_rate_for("fake") == 16000

    def test_input_limit_is_model_specific_and_optional(self):
        with patch("src.tts.router._discover_backends", return_value={}):
            router = TTSRouter(device="cpu")
        fake = FakeBackend()
        router.register_backend("fake", fake)
        assert router.max_input_chars_for("fake") is None

        fake.get_model_manifest = lambda _model_id: type("Manifest", (), {"max_input_chars": 8})()
        assert router.max_input_chars_for("fake") == 8

    def test_different_providers_can_synthesize_concurrently(self):
        with patch("src.tts.router._discover_backends", return_value={}):
            router = TTSRouter(device="cpu")
        barrier = threading.Barrier(2)
        router.register_backend("alpha", ConcurrentBackend("alpha", barrier))
        router.register_backend("beta", ConcurrentBackend("beta", barrier))

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(lambda: list(router.synthesize("hello", "alpha", "voice"))),
                pool.submit(lambda: list(router.synthesize("hello", "beta", "voice"))),
            ]
            for future in futures:
                assert len(future.result(timeout=3)) == 1

    def test_language_hint_is_validated_by_the_backend_without_loading_it(self, monkeypatch):
        # A streamed request has sent its headers by the time the generator would raise, so a
        # language the model cannot speak has to be rejected before synthesis starts.
        loads = []
        monkeypatch.setitem(
            sys.modules,
            "kokoro",
            SimpleNamespace(KPipeline=lambda *args, **kwargs: loads.append(kwargs)),
        )
        router = _kokoro_router()
        router.register_backend("fake", FakeBackend())

        with pytest.raises(ValueError, match="Unsupported language 'ko' for Kokoro"):
            router.validate_language("kokoro", "ko")
        for language in (None, "", "en", "zh-CN", "en-AU"):
            router.validate_language("kokoro", language)
        # A backend without its own check accepts any hint.
        router.validate_language("fake", "ko")
        assert loads == []
        assert not router.is_model_loaded("kokoro")

    def test_piper_and_pocket_tts_refuse_a_language_they_do_not_declare(self, monkeypatch):
        # README.md (POST /v1/audio/speech): a language hint "the
        # model cannot speak ... gets 400 before synthesis". Every shipped Piper voice is en_US or
        # en_GB and Pocket TTS speaks English, as their capabilities declare, yet a German or
        # Korean hint was accepted and read in English. The check reads only what each model
        # declares, so it loads and downloads nothing.
        from src.tts.backends import piper_backend
        from src.tts.backends.pocket_tts_backend import PocketTTSBackend

        class OfflinePiper(piper_backend.PiperBackend):
            def _speaker_map(self, model_id, *, download=False):
                raise AssertionError("a language check must not read Piper's voice files")

        # A stand-in German voice model: Piper checks the selected voice model's own language,
        # which the router passes as the model ID, not the union of every voice's languages.
        monkeypatch.setitem(
            piper_backend.PIPER_MODELS,
            "piper/de_DE-thorsten-medium",
            {"name": "de_DE-thorsten-medium", "lang": "de_DE", "quality": "medium", "sample_rate": 22050},
        )
        with patch("src.tts.router._discover_backends", return_value={}):
            router = TTSRouter(device="cpu", external_providers="")
        router.register_backend("piper", OfflinePiper(device="cpu"))
        router.register_backend("pocket-tts", PocketTTSBackend(device="cpu"))

        with pytest.raises(ValueError, match="Unsupported language 'de' for piper/en_US-lessac-medium"):
            router.validate_language("piper/en_US-lessac-medium", "de")
        with pytest.raises(ValueError, match="Unsupported language 'en' for piper/de_DE-thorsten-medium"):
            router.validate_language("piper/de_DE-thorsten-medium", "en")
        with pytest.raises(ValueError, match="Unsupported language 'ko' for pocket-tts"):
            router.validate_language("pocket-tts", "ko")
        # A regional tag counts by its primary subtag, in any case and with either separator.
        for language in (None, "", "en", "en-GB", "EN_us"):
            router.validate_language("piper/en_US-lessac-medium", language)
            router.validate_language("piper/en_GB-alan-low", language)
            router.validate_language("pocket-tts", language)
        router.validate_language("piper/de_DE-thorsten-medium", "de-AT")
        assert router.loaded_models() == []

    def test_a_worker_model_is_checked_against_its_manifest_languages(self, monkeypatch):
        # docs/TTS-BACKENDS.md: the worker manifest "declares
        # capabilities, voices, language support ... per model", and README.md refuses a hint "the
        # model cannot speak" with 400 before synthesis. One worker serves several models, so the
        # router passes the model ID, as it does for validate_voice.
        # Qwen3 names its languages ("Auto", "English", ...)
        # instead of giving codes, and its worker refuses any other hint with 400
        # language_unsupported (providers/qwen3/app.py, docs/TTS-BACKENDS.md). The core maps those
        # names to codes, so a hint Qwen3 can't speak is refused before the worker is asked (Live
        # Reader's session.update included). "Auto" admits the hint auto, not every hint. A list
        # with a name the core has no code for still leaves every hint to its worker.
        workers = {
            "chatterbox": _ManifestWorker("chatterbox", {
                # The worker's lists (English only on both models).
                "chatterbox/regular": ["en"],
                "chatterbox/turbo": ["en"],
            }),
            "qwen3": _ManifestWorker("qwen3", {
                "qwen3/0.6b-custom-voice": _QWEN3_LANGUAGE_NAMES,
                "qwen3/0.6b-base": _QWEN3_LANGUAGE_NAMES,
            }),
            "community": _ManifestWorker("community", {"community/tts": ["Auto", "English", "Dutch"]}),
        }
        providers = {
            "chatterbox": "http://chatterbox:8210",
            "qwen3": "http://qwen3:8200",
            "community": "http://community:8300",
        }
        with patch("src.tts.router._discover_backends", return_value={}):
            router = TTSRouter(device="cpu", external_providers=json.dumps(providers))
        for provider, worker in workers.items():
            monkeypatch.setattr(router._backends[provider]._opener, "open", worker.open)

        with pytest.raises(ValueError, match="Unsupported language 'de' for chatterbox/regular"):
            router.validate_language("chatterbox/regular", "de")
        for model in ("chatterbox/regular", "chatterbox/turbo"):
            with pytest.raises(ValueError, match=f"Unsupported language 'ja' for {model}"):
                router.validate_language(model, "ja")
        for language in ("en", "en-GB", "EN_us"):
            router.validate_language("chatterbox/regular", language)
        router.validate_language("chatterbox/turbo", "en")
        for model in ("qwen3/0.6b-custom-voice", "qwen3/0.6b-base"):
            # The worker takes its names in any case, auto, codes and regional tags.
            for language in (
                "en", "zh-CN", "EN_us", "pt-BR", "auto", "AUTO", "English", "english", "JAPANESE",
            ):
                router.validate_language(model, language)
            for language in ("nl", "nl-NL", "Dutch", "xx"):
                with pytest.raises(ValueError) as refused:
                    router.validate_language(model, language)
                assert str(refused.value) == (
                    f"Unsupported language '{language}' for {model}. "
                    f"Supported: {', '.join(_QWEN3_LANGUAGE_NAMES)}"
                )
        for language in ("nl", "Dutch", "xx", "auto"):
            router.validate_language("community/tts", language)
        # Only the manifests were read: nothing was loaded or spoken.
        assert [worker.other_requests for worker in workers.values()] == [[], [], []]


class TestBackendAvailability:
    def test_skips_unavailable_backends(self, caplog):
        caplog.set_level("INFO")
        with patch(
            "src.tts.router._discover_backends",
            return_value={"optional": UnavailableBackend, "available": AvailableBackend},
        ):
            router = TTSRouter(device="cpu")

        assert "available" in router.list_backends()
        assert "optional" not in router.list_backends()
        assert any("Skipping TTS backend optional" in rec.message for rec in caplog.records)

    def test_kokoro_reports_unavailable_when_optional_package_is_missing(self):
        with patch.dict("sys.modules", {"kokoro": None}):
            assert KokoroBackend.is_available() is False


class TestAutoDiscovery:
    def test_discovers_kokoro(self):
        """Auto-discovery should find the kokoro backend."""
        discovered = _discover_backends()
        assert "kokoro" in discovered

    def test_discovers_pocket_tts(self):
        """Auto-discovery should find the pocket-tts backend module."""
        discovered = _discover_backends()
        assert "pocket-tts" in discovered


class _Clock:
    def __init__(self, now: float) -> None:
        self.now = now

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return time.monotonic()


def _kokoro_package(*, on_chunk=None, on_load=None):
    """The kokoro package at its import boundary: KPipeline yields (graphemes, phonemes, audio)."""

    class KPipeline:
        def __init__(self, lang_code, repo_id=None, model=True, trf=False, en_callable=None, device=None):
            if on_load is not None:
                on_load()
            self.lang_code = lang_code
            self.model = object() if model is True else model

        def __call__(self, text, voice=None, speed=1.0):
            for index in range(3):
                if on_chunk is not None:
                    on_chunk(index)
                yield text, "", np.zeros(240, dtype=np.float32)

    return SimpleNamespace(KPipeline=KPipeline)


def _kokoro_router() -> TTSRouter:
    with patch("src.tts.router._discover_backends", return_value={}):
        router = TTSRouter(device="cpu", external_providers="")
    router.register_backend("kokoro", KokoroBackend(device="cpu"))
    return router


_WORKER_MANIFEST = {
    "schema_version": 1,
    "provider": "qwen3",
    "models": [
        {
            "id": "qwen3/0.6b-base",
            "provider": "qwen3",
            "sample_rate": 24000,
            "revision": "test-1",
            "capabilities": {"voice_clone": True},
            "voices": [],
        }
    ],
}


class _HeldManifestWorker:
    """An isolated worker's HTTP boundary: /health answers at once, /v1/manifest is held.

    A worker that accepted the connection but has not answered yet, as a busy or wedged one does
    for up to the 3 s metadata timeout on each read, until the test releases it.
    """

    def __init__(self) -> None:
        self.manifest_requested = threading.Event()
        self.release = threading.Event()

    def open(self, request, **_kwargs):
        if request.full_url.endswith("/health"):
            health = {"status": "ok", "provider": "qwen3", "loaded_model": None}
            return io.BytesIO(json.dumps(health).encode())
        if request.full_url.endswith("/v1/manifest"):
            self.manifest_requested.set()
            self.release.wait(timeout=10)
            return io.BytesIO(json.dumps(_WORKER_MANIFEST).encode())
        raise urllib.error.URLError("this worker loads and speaks nothing")


# providers/qwen3/app.py LANGUAGES: the model's own names, which its worker maps hints to.
_QWEN3_LANGUAGE_NAMES = [
    "Auto", "Chinese", "English", "Japanese", "Korean", "German",
    "French", "Russian", "Portuguese", "Spanish", "Italian",
]


class _ManifestWorker:
    """An isolated worker's HTTP boundary that answers /health and /v1/manifest at once.

    Each model declares the given languages. Any other request is recorded and refused: this
    worker loads and speaks nothing.
    """

    def __init__(self, provider: str, languages: dict[str, list[str]]) -> None:
        self.provider = provider
        self.other_requests: list[str] = []
        self.manifest = {
            "schema_version": 1,
            "provider": provider,
            "models": [
                {
                    "id": model_id,
                    "provider": provider,
                    "sample_rate": 24000,
                    "revision": "test-1",
                    "capabilities": {"voice_clone": True, "languages": declared},
                    "voices": [],
                }
                for model_id, declared in languages.items()
            ],
        }

    def open(self, request, **_kwargs):
        if request.full_url.endswith("/health"):
            health = {"status": "ok", "provider": self.provider, "loaded_model": None}
            return io.BytesIO(json.dumps(health).encode())
        if request.full_url.endswith("/v1/manifest"):
            return io.BytesIO(json.dumps(self.manifest).encode())
        self.other_requests.append(request.full_url)
        raise urllib.error.URLError("this worker loads and speaks nothing")


class TestModelStatusReads:
    @pytest.mark.parametrize("busy_with", ["stream", "first_load"])
    def test_status_reads_do_not_wait_for_an_in_process_request(self, monkeypatch, busy_with):
        # router.py: "one provider must not serialize every other provider". A streamed speech
        # request holds Kokoro's synthesis lock while its client reads, and a first load holds it
        # for the load; GET /api/models, /v1/models and STT model status all list loaded models,
        # and GET /v1/audio/voices only reads a catalog.
        release = threading.Event()
        busy = threading.Event()

        def hold_load():
            busy.set()
            release.wait(timeout=5)

        monkeypatch.setitem(
            sys.modules,
            "kokoro",
            _kokoro_package(on_load=hold_load if busy_with == "first_load" else None),
        )
        monkeypatch.setattr(kokoro_module, "_discover_voices_from_package", lambda: None)
        router = _kokoro_router()

        def request():
            if busy_with == "first_load":
                router.load_model("kokoro")
                return
            stream = router.synthesize(text="A long article.", model="kokoro", voice="af_heart")
            next(stream)  # the client is still playing the first chunk
            busy.set()
            release.wait(timeout=5)
            stream.close()

        if busy_with == "stream":
            router.load_model("kokoro")
        holder = threading.Thread(target=request, daemon=True)
        holder.start()
        assert busy.wait(timeout=5)

        outcome: dict[str, object] = {}

        def read_status():
            outcome["loaded"] = [info.model for info in router.loaded_models()]
            outcome["is_loaded"] = router.is_model_loaded("kokoro")
            outcome["voices"] = len(router.list_voices("kokoro"))
            outcome["all_voices"] = len(router.list_voices())

        reader = threading.Thread(target=read_status, daemon=True)
        reader.start()
        reader.join(timeout=2)
        finished = not reader.is_alive()
        release.set()
        holder.join(timeout=5)
        reader.join(timeout=5)

        assert finished, f"model status or voices waited for a Kokoro {busy_with}"
        # A model still loading is listed too, as when these reads waited for its load: the
        # one-TTS-model eviction has to find it.
        voices = len(kokoro_module.ALL_KOKORO_VOICES)
        assert outcome == {
            "loaded": ["kokoro"], "is_loaded": True, "voices": voices, "all_voices": voices,
        }

    @pytest.mark.parametrize(
        "worker_call", ["list_voices", "is_model_loaded", "synthesize", "load_model", "unload_model"]
    )
    def test_a_worker_manifest_read_does_not_hold_up_other_providers(
        self, monkeypatch, worker_call
    ):
        # router.py: "one provider must not serialize every other provider"; docs/TTS-BACKENDS.md:
        # is_model_loaded(), loaded_models() and list_voices() "run alongside" the others. Routing
        # a worker model reads the worker's /v1/manifest over HTTP, which an unreachable or busy
        # worker holds for up to its timeout. The Voices view and GET /v1/audio/voices make that
        # read for worker models, and an in-process model serving Wyoming or the Reader must not
        # wait for it.
        worker = _HeldManifestWorker()
        with patch("src.tts.router._discover_backends", return_value={}):
            router = TTSRouter(device="cpu", external_providers='{"qwen3": "http://qwen3:8200"}')
        monkeypatch.setattr(router._backends["qwen3"]._opener, "open", worker.open)
        router.register_backend("fake", FakeBackend())
        model = "qwen3/0.6b-base"
        worker_calls = {
            "list_voices": lambda: router.list_voices(model),
            "is_model_loaded": lambda: router.is_model_loaded(model),
            "synthesize": lambda: list(router.synthesize(text="Hello.", model=model, voice="clone")),
            "load_model": lambda: router.load_model(model),
            "unload_model": lambda: router.unload_model(model),
        }

        def worker_request():
            try:
                worker_calls[worker_call]()
            except Exception:  # the worker answers no load or speech; only the wait matters
                pass

        holder = threading.Thread(target=worker_request, daemon=True)
        holder.start()
        assert worker.manifest_requested.wait(timeout=5)

        other_calls = {
            "synthesize": lambda: len(
                list(router.synthesize(text="Hello.", model="fake", voice="fake_voice"))
            ),
            "is_model_loaded": lambda: router.is_model_loaded("fake"),
            "list_voices": lambda: [voice.id for voice in router.list_voices("fake")],
            "loaded_models": lambda: router.loaded_models(),
        }
        outcome: dict[str, object] = {}

        def other_provider(name):
            try:
                outcome[name] = other_calls[name]()
            except Exception as exc:
                outcome[name] = repr(exc)

        readers = {
            name: threading.Thread(target=other_provider, args=(name,), daemon=True)
            for name in other_calls
        }
        for reader in readers.values():
            reader.start()
        deadline = time.monotonic() + 2
        for reader in readers.values():
            reader.join(timeout=max(0.0, deadline - time.monotonic()))
        stalled = sorted(name for name, reader in readers.items() if reader.is_alive())
        manifest_read_in_flight = holder.is_alive()
        worker.release.set()
        holder.join(timeout=5)
        for reader in readers.values():
            reader.join(timeout=5)

        assert manifest_read_in_flight, "the worker's manifest read ended before the other calls"
        assert stalled == [], (
            f"the in-process provider's {stalled} waited for the worker's manifest read "
            f"during {worker_call}({model!r})"
        )
        assert outcome == {
            "synthesize": 1, "is_model_loaded": False, "list_voices": ["fake_voice"],
            "loaded_models": [],
        }

    @pytest.mark.parametrize("first_load_by", ["load", "speech"])
    def test_a_tts_load_during_a_first_load_still_leaves_one_tts_model(
        self, monkeypatch, first_load_by
    ):
        # README: "Loading a TTS model always unloads the previously loaded one first, so only
        # one TTS model is loaded at a time." Status reads do not wait for Kokoro's first load,
        # whether a load or a speech request started it, so the eviction must still see it.
        from src.services.tts import load_tts_model

        loading = threading.Event()
        release = threading.Event()

        def hold_load():
            loading.set()
            release.wait(timeout=5)

        monkeypatch.setitem(sys.modules, "kokoro", _kokoro_package(on_load=hold_load))
        router = _kokoro_router()
        router.register_backend("fake", ResidentBackend())
        evicting = threading.Event()
        unload_model = router.unload_model

        def evict(model_id):
            evicting.set()
            unload_model(model_id)

        monkeypatch.setattr(router, "unload_model", evict)

        def first_load():
            if first_load_by == "load":
                router.load_model("kokoro")
            else:
                list(router.synthesize(text="Hello.", model="kokoro", voice="af_heart"))

        first = threading.Thread(target=first_load, daemon=True)
        first.start()
        assert loading.wait(timeout=5)
        second = threading.Thread(
            target=load_tts_model,
            kwargs={
                "settings": SimpleNamespace(tts_enabled=True),
                "tts_router": router,
                "model_id": "fake",
            },
            daemon=True,
        )
        second.start()
        evicting.wait(timeout=2)  # it found Kokoro's load, or it loaded next to it
        release.set()
        first.join(timeout=5)
        second.join(timeout=5)

        assert not first.is_alive() and not second.is_alive()
        assert [info.model for info in router.loaded_models()] == ["fake"]

    def test_in_process_model_counts_as_used_until_its_request_ends(self, monkeypatch):
        # OS_MODEL_TTL is an idle timeout. Kokoro stamps its use when a synthesis starts, so a
        # 400 s request must not look idle from its first second, during it or after it.
        clock = _Clock(900.0)
        seen_during: list[float | None] = []

        def on_chunk(index):
            if index == 1:
                clock.now = 1200.0
                seen_during.append(router.loaded_models()[0].last_used_at)
            if index == 2:
                clock.now = 1400.0

        monkeypatch.setitem(sys.modules, "kokoro", _kokoro_package(on_chunk=on_chunk))
        monkeypatch.setattr(kokoro_module, "time", clock)
        monkeypatch.setattr(router_module, "time", clock)
        router = _kokoro_router()

        router.load_model("kokoro")
        clock.now = 1000.0
        list(router.synthesize(text="A long chapter.", model="kokoro", voice="af_heart"))
        clock.now = 1430.0

        assert seen_during == [1200.0]
        assert router.loaded_models()[0].last_used_at == 1400.0

        # A request that ended before the current load says nothing about the new residency.
        clock.now = 1500.0
        router.unload_model("kokoro")
        clock.now = 1600.0
        router.load_model("kokoro")
        assert router.loaded_models()[0].last_used_at is None
