"""Backend router — maps model IDs to backend implementations."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
from collections.abc import Callable
from typing import Any

from src.backends.base import STTBackend
from src.backends.faster_whisper import FasterWhisperBackend
from src.backends.parakeet import ParakeetBackend, is_parakeet_model
from src.models import LoadedModelInfo

logger = logging.getLogger(__name__)


class BackendRouter:
    """Routes requests to the appropriate STT backend based on model ID."""

    # ModelManager.gpu_admission, which ModelManager sets: before a model loads, it makes room on the GPU or refuses
    # the load (src/gpu_memory.py). None for a router no manager wraps.
    gpu_admission: Callable[[str, str], contextlib.AbstractContextManager | None] | None = None

    def __init__(self) -> None:
        self._backends: dict[str, STTBackend] = {}
        self._lock = asyncio.Lock()

        # Always register faster-whisper
        fw = FasterWhisperBackend()
        self._backends["faster-whisper"] = fw
        self._default_backend = fw
        # NVIDIA Parakeet through onnx-asr: the model IDs in src/backends/parakeet.py's PARAKEET_MODELS.
        self._backends["parakeet"] = ParakeetBackend()

    def get_backend(self, model_id: str) -> STTBackend:
        """Get the backend for a given model ID: Parakeet for its IDs, faster-whisper for every other."""
        if is_parakeet_model(model_id) and "parakeet" in self._backends:
            return self._backends["parakeet"]
        return self._default_backend

    def _admission(self, backend: STTBackend, model_id: str) -> contextlib.AbstractContextManager | None:
        """The GPU memory guard's admission of loading this model, or None: no guard, already loaded, or no GPU."""
        admit = getattr(self, "gpu_admission", None)
        is_loaded = getattr(backend, "is_model_loaded", None)
        if admit is None or (callable(is_loaded) and is_loaded(model_id)):
            return None
        return admit(model_id, "stt")

    def load_model(self, model_id: str) -> None:
        backend = self.get_backend(model_id)
        with self._admission(backend, model_id) or contextlib.nullcontext():
            backend.load_model(model_id)

    def _load_on_first_use(self, backend: STTBackend, model_id: str) -> None:
        """A decode loads its model on first use, inside the backend; when it would load onto the GPU, load it here
        first, through the guard, so the decode finds it loaded."""
        admission = self._admission(backend, model_id)
        if admission is not None:
            with admission:
                backend.load_model(model_id)

    def unload_model(self, model_id: str) -> None:
        backend = self.get_backend(model_id)
        backend.unload_model(model_id)

    def unload_if_idle(self, model_id: str) -> bool:
        """Unload a model unless its backend is serving a request; False when it is busy."""
        backend = self.get_backend(model_id)
        unload_if_idle = getattr(backend, "unload_if_idle", None)
        if callable(unload_if_idle):
            return bool(unload_if_idle(model_id))
        backend.unload_model(model_id)
        return True

    def loaded_models(self) -> list[LoadedModelInfo]:
        result = []
        for backend in self._backends.values():
            result.extend(backend.loaded_models())
        return result

    def is_model_loaded(self, model_id: str) -> bool:
        backend = self.get_backend(model_id)
        return backend.is_model_loaded(model_id)

    def list_cached_models(self) -> list[dict[str, Any]]:
        result = []
        for backend in self._backends.values():
            if hasattr(backend, 'list_cached_models'):
                result.extend(backend.list_cached_models())
        return result

    def delete_cached_model(self, model_id: str) -> bool:
        backend = self.get_backend(model_id)
        if hasattr(backend, 'delete_cached_model'):
            return backend.delete_cached_model(model_id)
        return False

    def download_model(self, model_id: str) -> bool:
        """Fetch a model's files without loading it; False when its backend cannot."""
        backend = self.get_backend(model_id)
        download_model = getattr(backend, "download_model", None)
        if callable(download_model):
            return bool(download_model(model_id))
        return False

    def start_fetch(self, model_id: str) -> concurrent.futures.Future | None:
        """Start or join the fetch of a model's files (a future of their path); None when its backend can't."""
        start_fetch = getattr(self.get_backend(model_id), "start_fetch", None)
        return start_fetch(model_id) if callable(start_fetch) else None

    def is_model_cached(self, model_id: str) -> bool:
        backend = self.get_backend(model_id)
        if hasattr(backend, 'is_model_cached'):
            return backend.is_model_cached(model_id)
        return False

    def transcribe(self, audio: bytes, model: str, **kwargs: Any) -> dict[str, Any]:
        backend = self.get_backend(model)
        self._load_on_first_use(backend, model)
        return backend.transcribe(audio, model, **kwargs)

    def translate(self, audio: bytes, model: str, **kwargs: Any) -> dict[str, Any]:
        backend = self.get_backend(model)
        self._load_on_first_use(backend, model)
        return backend.translate(audio, model, **kwargs)


async def await_fetch(future: concurrent.futures.Future) -> Any:
    """Await a model fetch without holding a thread.

    Several requests can share one fetch, so a request that ends early (its client left) stops waiting without
    canceling the fetch the others wait for.
    """
    waiter = asyncio.wrap_future(future)
    waiter.add_done_callback(lambda done: done.cancelled() or done.exception())  # read, so no "never retrieved" log
    return await asyncio.shield(waiter)


# Global singleton
router = BackendRouter()
