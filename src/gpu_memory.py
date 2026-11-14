"""The GPU memory guard: check the card's free memory before a model loads onto it, and make room first.

Nothing else stops a load from overfilling the card. On Linux CUDA then fails the load, but on Windows (WDDM, and
WSL, which Docker Desktop runs on) the driver pages other memory out instead, so the pinned models' next request takes
seconds. Before each load that would use the GPU (a core model's, or a worker's through its load call), the guard:

1. leaves alone a model that is already loaded, or a CPU-only server (it can't read a GPU: no NVIDIA driver);
2. on a card with a cap on big models (OS_GPU_MAX_BIG_MODELS), unloads the least recently used idle big model that
   isn't pinned while the cap is reached;
3. compares the card's free memory with the model's size (its worker's own figure, else the registry's ``gpu_mb``)
   plus OS_GPU_HEADROOM_MB, and unloads idle models that aren't pinned, least recently used first, until it fits;
4. refuses the load when it still doesn't fit (503 ``insufficient_gpu_memory``), naming the model, what it needs,
   what is free, and which models could be unloaded. The pinned defaults (OS_PIN_DEFAULT_MODELS) and busy models are
   never unloaded.

The free memory comes from NVML (the NVIDIA driver's library, which Docker's GPU support mounts), else nvidia-smi,
else torch once it holds a CUDA context. None of them starts a CUDA context of its own.
"""

from __future__ import annotations

import contextlib
import ctypes
import itertools
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from src.config import settings
from src.tts.external import ExternalProviderError

logger = logging.getLogger(__name__)

MIB = 1024 * 1024
# A model at least this large (MiB) counts against OS_GPU_MAX_BIG_MODELS: a GPU worker's model or a large Whisper.
BIG_MODEL_MB = 2048
# The size (MiB) of a model that loads onto the GPU without a known one: an estimate, rounded up.
DEFAULT_GPU_MB = 2000
# A worker frees an unloaded model's memory when its process ends (Chatterbox restarts 10 s after an unload, Fish
# after 1 s), so the guard waits this long for an unload to show in the free memory before it unloads another.
RELEASE_WAIT_S = 20.0
RELEASE_POLL_S = 0.5
# Card sizes (MiB) for OS_GPU_MAX_BIG_MODELS=auto. An "8 GB" card reports about 7.6-8.0 GiB, a "16 GB" one 15.6-16,
# and a "24 GB" one 23.6-24.
_AUTO_CAPS = ((11 * 1024, 1), (22 * 1024, 2))


@dataclass(frozen=True)
class GpuMemory:
    """The first GPU's free and total memory, in MiB."""

    free_mb: int
    total_mb: int


class _NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


def _nvml_reader() -> Callable[[], GpuMemory] | None:
    """Read the first GPU through NVML; None when the library or a GPU is missing."""
    names = ("nvml.dll",) if os.name == "nt" else ("libnvidia-ml.so.1", "libnvidia-ml.so")
    library = None
    for name in names:
        try:
            library = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if library is None:
        return None
    try:
        if library.nvmlInit_v2() != 0:
            return None
        handle = ctypes.c_void_p()
        if library.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(0), ctypes.byref(handle)) != 0:
            return None
    except AttributeError:  # not the NVML this was written for
        return None

    def read() -> GpuMemory:
        memory = _NvmlMemory()
        status = library.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory))
        if status != 0:
            raise OSError(f"nvmlDeviceGetMemoryInfo returned {status}")
        return GpuMemory(free_mb=int(memory.free // MIB), total_mb=int(memory.total // MIB))

    return read


def _nvidia_smi_reader() -> Callable[[], GpuMemory] | None:
    """Read the first GPU with nvidia-smi; None when it isn't installed."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None

    def read() -> GpuMemory:
        output = subprocess.run(
            [executable, "--id=0", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
        free, total = (int(float(value)) for value in output.strip().splitlines()[0].split(","))
        return GpuMemory(free_mb=free, total_mb=total)  # nvidia-smi gives MiB

    return read


def _torch_read() -> GpuMemory | None:
    """Read the GPU through torch, only when this process already holds its CUDA context (Kokoro on the GPU)."""
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None)
    is_initialized = getattr(cuda, "is_initialized", None)
    if not callable(is_initialized) or not is_initialized():
        return None
    free, total = cuda.mem_get_info(0)
    return GpuMemory(free_mb=int(free // MIB), total_mb=int(total // MIB))


class GpuMemoryProbe:
    """The first GPU's free and total memory, read when asked; None on a server without an NVIDIA GPU."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reader: Callable[[], GpuMemory] | None = None
        self._chosen = False

    def _choose(self) -> None:
        for make in (_nvml_reader, _nvidia_smi_reader):
            try:
                reader = make()
                if reader is not None:
                    reader()  # one read proves it works
                    self._reader = reader
                    return
            except Exception:  # no NVIDIA driver, or no GPU: try the next
                logger.debug("GPU memory source %s is unusable", make.__name__, exc_info=True)

    def read(self) -> GpuMemory | None:
        with self._lock:
            if not self._chosen:
                self._chosen = True
                self._choose()
            reader = self._reader
        try:
            return reader() if reader is not None else _torch_read()
        except Exception:  # a failed read leaves the load unguarded rather than refusing it
            logger.warning("Could not read the GPU's free memory", exc_info=True)
            return None


class InsufficientGpuMemoryError(ExternalProviderError):
    """A load the GPU memory guard refused: 503 ``insufficient_gpu_memory``, with the figures in ``details``."""

    def __init__(self, message: str, *, details: dict) -> None:
        super().__init__(message, code="insufficient_gpu_memory", status_code=503)
        self.details = details


@dataclass(frozen=True)
class GpuResident:
    """A loaded model on the GPU, as the guard sees it."""

    id: str
    type: str  # "stt" or "tts"
    gpu_mb: int
    last_used_at: float
    pinned: bool = False
    # A worker holds one model: loading another of its models replaces this one, so its memory counts as free.
    replaced_by_load: bool = False


def gib(mb: float) -> str:
    return f"{mb / 1024:.1f} GiB"


def max_big_models(total_mb: int) -> int | None:
    """OS_GPU_MAX_BIG_MODELS for a card of this size; None means no cap."""
    raw = str(settings.os_gpu_max_big_models).strip().lower()
    if raw != "auto":
        return int(raw) or None
    for below_mb, cap in _AUTO_CAPS:
        if total_mb < below_mb:
            return cap
    return None


class GpuMemoryGuard:
    """Admits model loads onto the GPU (see the module docstring)."""

    def __init__(
        self,
        probe: GpuMemoryProbe | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._probe = probe or GpuMemoryProbe()
        self._sleep = sleep
        self._clock = clock
        # One admission at a time: two loads checked at once could each find room for one model.
        self._lock = threading.Lock()
        # Loads admitted and still running: token -> (model id, MiB). A running load hasn't taken its memory yet.
        self._reserved: dict[int, tuple[str, int]] = {}
        self._tokens = itertools.count()

    def read(self) -> GpuMemory | None:
        """The card's memory now, or None when the guard is off or there is no GPU to read."""
        if not settings.os_gpu_memory_guard:
            return None
        return self._probe.read()

    def describe(self) -> str:
        """One line for the startup log: the card, its cap on big models, and the headroom."""
        if not settings.os_gpu_memory_guard:
            return "GPU memory guard: off (OS_GPU_MEMORY_GUARD=false)"
        memory = self.read()
        if memory is None:
            return "GPU memory guard: no NVIDIA GPU to read, loads are not checked"
        cap = max_big_models(memory.total_mb)
        return (
            f"GPU memory guard: {gib(memory.total_mb)} card, {gib(memory.free_mb)} free; "
            f"{'no cap on big models' if cap is None else f'at most {cap} big model(s) next to the pinned defaults'}; "
            f"{settings.os_gpu_headroom_mb} MiB headroom"
        )

    def _pending_mb(self, model_id: str) -> int:
        return sum(mb for reserved_id, mb in self._reserved.values() if reserved_id != model_id)

    @contextlib.contextmanager
    def admission(
        self,
        model_id: str,
        need_mb: int,
        *,
        residents: Callable[[], Iterable[GpuResident]],
        unload_idle: Callable[[GpuResident], bool],
        pinned: bool = False,
        releasing: Callable[[], bool] = lambda: False,
    ) -> Iterator[None]:
        """Make room for ``model_id`` on the GPU before its load runs in the with-block, or raise
        InsufficientGpuMemoryError. ``residents`` lists the models on the GPU; ``unload_idle`` unloads one unless it is
        busy and says whether it did; ``pinned`` is whether the model to load is a pinned default; ``releasing`` says
        whether a worker unloaded a model moments ago and may not have given its memory back yet."""
        with self._lock:
            if any(reserved_id == model_id for reserved_id, _ in self._reserved.values()):
                token = None  # the same model's load is already admitted: this one joins it
            else:
                self._make_room(model_id, need_mb, residents, unload_idle, pinned, releasing)
                token = next(self._tokens)
                self._reserved[token] = (model_id, need_mb)
        try:
            yield
        finally:
            if token is not None:
                with self._lock:
                    self._reserved.pop(token, None)

    def _make_room(
        self,
        model_id: str,
        need_mb: int,
        residents: Callable[[], Iterable[GpuResident]],
        unload_idle: Callable[[GpuResident], bool],
        pinned: bool,
        releasing: Callable[[], bool],
    ) -> None:
        memory = self.read()
        if memory is None:
            return
        loaded = [resident for resident in residents() if resident.id != model_id]
        # Least recently used first; the pinned defaults are never unloaded.
        candidates = sorted(
            (resident for resident in loaded if not resident.pinned and not resident.replaced_by_load),
            key=lambda resident: resident.last_used_at,
        )
        credit = sum(resident.gpu_mb for resident in loaded if resident.replaced_by_load)
        target = need_mb + settings.os_gpu_headroom_mb
        busy: list[GpuResident] = []
        unloaded: list[GpuResident] = []

        def free_now() -> int:
            current = self.read()
            return current.free_mb if current is not None else memory.free_mb

        def available() -> int:
            return free_now() + credit - self._pending_mb(model_id)

        def unload(resident: GpuResident) -> None:
            before = available()
            if not unload_idle(resident):
                busy.append(resident)
                return
            unloaded.append(resident)
            logger.info(
                "GPU memory guard: unloaded %s (%s, idle) to make room for %s (%s)",
                resident.id, gib(resident.gpu_mb), model_id, gib(need_mb),
            )
            self._await_release(available, target, resident.gpu_mb, before)

        cap = max_big_models(memory.total_mb)
        if cap is not None and need_mb >= BIG_MODEL_MB and not pinned:
            big = [resident for resident in candidates if resident.gpu_mb >= BIG_MODEL_MB]
            while sum(resident not in unloaded for resident in big) >= cap:
                idle = [resident for resident in big if resident not in unloaded and resident not in busy]
                if not idle:
                    raise self._cap_refusal(model_id, need_mb, memory, cap, busy)
                unload(idle[0])
        left = [resident for resident in candidates if resident not in unloaded and resident not in busy]
        if available() + sum(resident.gpu_mb for resident in left) < target and not releasing():
            # Unloading every idle model still wouldn't make room: unload none of them.
            raise self._refusal(model_id, need_mb, free_now(), memory.total_mb, busy, loaded, unloaded)
        for resident in candidates:
            if available() >= target:
                return
            if resident not in unloaded and resident not in busy:
                unload(resident)
        # An explicit load first unloads the other models of its type; a worker's comes back when its process ends.
        while available() < target and releasing():
            self._sleep(RELEASE_POLL_S)
        if available() < target:
            raise self._refusal(model_id, need_mb, free_now(), memory.total_mb, busy, loaded, unloaded)

    def _await_release(self, available: Callable[[], int], target: int, freed_mb: int, before: int) -> None:
        """Wait for an unload to show in the free memory: enough for the load, most of the unloaded model's size, or
        RELEASE_WAIT_S. A worker gives an unloaded model's memory back when its process ends, seconds later."""
        deadline = self._clock() + RELEASE_WAIT_S
        while True:
            free = available()
            if free >= target or free - before >= freed_mb * 0.8 or self._clock() >= deadline:
                return
            self._sleep(RELEASE_POLL_S)

    @staticmethod
    def _figures(model_id: str, need_mb: int, free_mb: int, total_mb: int, busy: list[GpuResident],
                 unloadable: list[GpuResident]) -> dict:
        return {
            "model": model_id,
            "need_mb": need_mb,
            "headroom_mb": settings.os_gpu_headroom_mb,
            "free_mb": max(free_mb, 0),
            "total_mb": total_mb,
            "unloadable": [
                {"model": resident.id, "gpu_mb": resident.gpu_mb, "busy": resident in busy} for resident in unloadable
            ],
        }

    def _refusal(self, model_id: str, need_mb: int, free_mb: int, total_mb: int, busy: list[GpuResident],
                 loaded: list[GpuResident], unloaded: list[GpuResident]) -> InsufficientGpuMemoryError:
        unloadable = [resident for resident in loaded
                      if not resident.pinned and not resident.replaced_by_load and resident not in unloaded]
        pinned = [resident.id for resident in loaded if resident.pinned]
        message = (
            f"Not enough GPU memory to load {model_id}: it needs about {gib(need_mb)} plus "
            f"{settings.os_gpu_headroom_mb} MiB of headroom, and {gib(max(free_mb, 0))} of the card's "
            f"{gib(total_mb)} is free."
        )
        named = ", ".join(
            f"{resident.id} ({gib(resident.gpu_mb)}{', busy' if resident in busy else ''})" for resident in unloadable
        )
        room_if_unloaded = free_mb + sum(resident.gpu_mb for resident in unloadable)
        if unloadable and room_if_unloaded >= need_mb + settings.os_gpu_headroom_mb:
            message += f" Unload a model you aren't using ({named}), or wait until it is idle, then try again."
        elif unloadable:
            message += (
                f" Even with {named} unloaded it wouldn't fit: close other programs that use the GPU, or choose a "
                "smaller model."
            )
        else:
            kept = f"The pinned defaults ({', '.join(pinned)}) stay loaded. " if pinned else ""
            message += f" {kept}Close other programs that use the GPU, or choose a smaller model."
        return InsufficientGpuMemoryError(
            message, details=self._figures(model_id, need_mb, free_mb, total_mb, busy, unloadable)
        )

    def _cap_refusal(self, model_id: str, need_mb: int, memory: GpuMemory, cap: int,
                     busy: list[GpuResident]) -> InsufficientGpuMemoryError:
        named = ", ".join(f"{resident.id} ({gib(resident.gpu_mb)})" for resident in busy)
        message = (
            f"{model_id} is a big model (about {gib(need_mb)} of GPU memory), and this {gib(memory.total_mb)} card "
            f"holds at most {cap} big model{'s' if cap != 1 else ''} next to the pinned defaults "
            f"(OS_GPU_MAX_BIG_MODELS). {named} {'is' if len(busy) == 1 else 'are'} loaded and busy: unload "
            f"{'it' if len(busy) == 1 else 'one'}, or try again once {'it is' if len(busy) == 1 else 'they are'} idle."
        )
        return InsufficientGpuMemoryError(
            message, details=self._figures(model_id, need_mb, memory.free_mb, memory.total_mb, busy, busy)
        )
