"""The GPU memory guard: before a model loads onto the GPU, check the card's free memory and make room or refuse.

On Windows (WDDM, WSL) an overfilled card pages the other models out instead of failing the load.
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
# Counts against OS_GPU_MAX_BIG_MODELS.
BIG_MODEL_MB = 2048
# A GPU model without a known size.
DEFAULT_GPU_MB = 2000
# How long a worker may take to give an unloaded model's memory back (its process restarts).
RELEASE_WAIT_S = 20.0
RELEASE_POLL_S = 0.5
# OS_GPU_MAX_BIG_MODELS=auto: (card below this many MiB, cap).
_AUTO_CAPS = ((11 * 1024, 1), (22 * 1024, 2))
# What unload_idle returns when another load or unload of the model's type holds its lock.
LOCKED = "locked"
_WAIT = object()


@dataclass(frozen=True)
class GpuMemory:
    """The first GPU's free and total memory, in MiB."""

    free_mb: int
    total_mb: int


class _NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


def _nvml_reader() -> Callable[[], GpuMemory] | None:
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
    except AttributeError:
        return None

    def read() -> GpuMemory:
        memory = _NvmlMemory()
        status = library.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory))
        if status != 0:
            raise OSError(f"nvmlDeviceGetMemoryInfo returned {status}")
        return GpuMemory(free_mb=int(memory.free // MIB), total_mb=int(memory.total // MIB))

    return read


def _nvidia_smi_reader() -> Callable[[], GpuMemory] | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None

    def read() -> GpuMemory:
        output = subprocess.run(
            [executable, "--id=0", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
        free, total = (int(float(value)) for value in output.strip().splitlines()[0].split(","))
        return GpuMemory(free_mb=free, total_mb=total)

    return read


def _torch_read() -> GpuMemory | None:
    """Read through torch only when this process already holds a CUDA context: never start one."""
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None)
    is_initialized = getattr(cuda, "is_initialized", None)
    if not callable(is_initialized) or not is_initialized():
        return None
    free, total = cuda.mem_get_info(0)
    return GpuMemory(free_mb=int(free // MIB), total_mb=int(total // MIB))


class GpuMemoryProbe:
    """The first GPU's memory (NVML, else nvidia-smi, else torch); None without an NVIDIA GPU."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reader: Callable[[], GpuMemory] | None = None
        self._chosen = False

    def _choose(self) -> None:
        for make in (_nvml_reader, _nvidia_smi_reader):
            try:
                reader = make()
                if reader is not None:
                    reader()
                    self._reader = reader
                    return
            except Exception:
                logger.debug("GPU memory source %s is unusable", make.__name__, exc_info=True)

    def read(self) -> GpuMemory | None:
        with self._lock:
            if not self._chosen:
                self._chosen = True
                self._choose()
            reader = self._reader
        try:
            return reader() if reader is not None else _torch_read()
        except Exception:  # unguarded rather than refused
            logger.warning("Could not read the GPU's free memory", exc_info=True)
            return None


class InsufficientGpuMemoryError(ExternalProviderError):
    """A refused load: 503 ``insufficient_gpu_memory``, with the figures in ``details``."""

    def __init__(self, message: str, *, details: dict) -> None:
        super().__init__(message, code="insufficient_gpu_memory", status_code=503)
        self.details = details


@dataclass(frozen=True)
class GpuResident:
    """A loaded model on the GPU."""

    id: str
    type: str  # "stt" or "tts"
    gpu_mb: int
    last_used_at: float
    pinned: bool = False
    # Its worker holds one model, so the load replaces it: its memory counts as free.
    replaced_by_load: bool = False
    # How much more a pinned model may take while it runs.
    growth_mb: int = 0
    # A pinned model this load may unload once no other idle model is left.
    displaceable: bool = False


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
    """Admits model loads onto the GPU."""

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
        self._lock = threading.Lock()
        # Admitted loads still running (token -> model id, MiB): they haven't taken their memory yet.
        self._reserved: dict[int, tuple[str, int]] = {}
        # Models an admission is unloading right now.
        self._unloading: set[str] = set()
        self._tokens = itertools.count()

    def read(self) -> GpuMemory | None:
        """The card's memory now; None when the guard is off or there is no GPU to read."""
        if not settings.os_gpu_memory_guard:
            return None
        return self._probe.read()

    def describe(self) -> str:
        """The startup log line."""
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
        return sum(mb for reserved_id, mb in tuple(self._reserved.values()) if reserved_id != model_id)

    @contextlib.contextmanager
    def admission(
        self,
        model_id: str,
        need_mb: int,
        *,
        residents: Callable[[], Iterable[GpuResident]],
        unload_idle: Callable[[GpuResident], bool | str],
        pinned: bool = False,
        releasing: Callable[[], bool] = lambda: False,
    ) -> Iterator[None]:
        """Make room for ``model_id`` before the with-block loads it, or raise InsufficientGpuMemoryError.

        ``unload_idle`` returns True when it unloaded the model, False when the model is busy, or LOCKED.
        ``releasing`` says whether a worker may still be giving an unloaded model's memory back.
        """
        token = self._admit(model_id, need_mb, residents, unload_idle, pinned, releasing)
        try:
            yield
        finally:
            if token is not None:
                with self._lock:
                    self._reserved.pop(token, None)

    def _admit(
        self,
        model_id: str,
        need_mb: int,
        residents: Callable[[], Iterable[GpuResident]],
        unload_idle: Callable[[GpuResident], bool | str],
        pinned: bool,
        releasing: Callable[[], bool],
    ) -> int | None:
        """Reserve the load's memory and return its token, or None when the same model's load is already admitted.
        The lock is held only to decide: unloads and release waits run without it."""
        refused: dict[str, str] = {}  # model id -> "busy" or LOCKED
        unloaded: list[GpuResident] = []
        while True:
            with self._lock:
                if any(reserved_id == model_id for reserved_id, _ in self._reserved.values()):
                    return None
                memory = self.read()
                step, available, target = (None, None, 0) if memory is None else self._next_step(
                    model_id, need_mb, memory, residents, pinned, releasing, refused, unloaded
                )
                if step is None:
                    token = next(self._tokens)
                    self._reserved[token] = (model_id, need_mb)
                    return token
                if step is not _WAIT:
                    self._unloading.add(step.id)
            if step is _WAIT:
                self._sleep(RELEASE_POLL_S)
                continue
            try:
                before = available()
                result = unload_idle(step)
                if result is True:
                    unloaded.append(step)
                    logger.info(
                        "GPU memory guard: unloaded %s (%s, idle) to make room for %s (%s)",
                        step.id, gib(step.gpu_mb), model_id, gib(need_mb),
                    )
                    self._await_release(available, target, step.gpu_mb, before)
                else:
                    refused[step.id] = LOCKED if result == LOCKED else "busy"
            finally:
                with self._lock:
                    self._unloading.discard(step.id)

    def _next_step(
        self,
        model_id: str,
        need_mb: int,
        memory: GpuMemory,
        residents: Callable[[], Iterable[GpuResident]],
        pinned: bool,
        releasing: Callable[[], bool],
        refused: dict[str, str],
        unloaded: list[GpuResident],
    ) -> tuple[object, Callable[[], int], int]:
        """(None when the load fits, else a resident to unload or _WAIT; the free-memory reader; the target)."""
        gone = {resident.id for resident in unloaded}
        loaded = [resident for resident in residents() if resident.id != model_id and resident.id not in gone]
        credit = sum(resident.gpu_mb for resident in loaded if resident.replaced_by_load)
        growth = 0 if pinned else sum(resident.growth_mb for resident in loaded if resident.pinned)
        target = need_mb + settings.os_gpu_headroom_mb + growth
        candidates = sorted(
            (resident for resident in loaded
             if not resident.pinned and not resident.replaced_by_load and resident.id not in self._unloading),
            key=lambda resident: resident.last_used_at,
        )
        displaceable = [
            resident for resident in loaded
            if resident.pinned and resident.displaceable and resident.id not in self._unloading
        ]
        idle = [resident for resident in candidates + displaceable if resident.id not in refused]
        # Unloading a pinned model also frees the room kept for it to grow.
        freed_growth = sum(resident.growth_mb for resident in displaceable if resident.id not in refused)

        def free_now() -> int:
            current = self.read()
            return current.free_mb if current is not None else memory.free_mb

        def available() -> int:
            return free_now() + credit - self._pending_mb(model_id)

        def refusal() -> InsufficientGpuMemoryError:
            return self._refusal(model_id, need_mb, growth, free_now(), memory.total_mb, refused, loaded)

        cap = max_big_models(memory.total_mb)
        if cap is not None and need_mb >= BIG_MODEL_MB and not pinned:
            big = [resident for resident in candidates if resident.gpu_mb >= BIG_MODEL_MB]
            if len(big) >= cap:
                idle_big = [resident for resident in big if resident.id not in refused]
                if not idle_big:
                    raise self._cap_refusal(model_id, need_mb, growth, memory, cap, refused, big)
                return idle_big[0], available, target
        free = available()
        if free >= target:
            return None, available, target
        if free + sum(resident.gpu_mb for resident in idle) + freed_growth < target and not releasing():
            raise refusal()  # unloading every idle model wouldn't do: unload none
        if idle:
            return idle[0], available, target
        if releasing() or self._unloading:
            return _WAIT, available, target
        raise refusal()

    def _await_release(self, available: Callable[[], int], target: int, freed_mb: int, before: int) -> None:
        """Wait until the load fits, most of the unloaded model's memory shows as free, or RELEASE_WAIT_S passes."""
        deadline = self._clock() + RELEASE_WAIT_S
        while True:
            free = available()
            if free >= target or free - before >= freed_mb * 0.8 or self._clock() >= deadline:
                return
            self._sleep(RELEASE_POLL_S)

    @staticmethod
    def _label(resident: GpuResident, refused: dict[str, str]) -> str:
        reason = {"busy": ", busy", LOCKED: ", another load or unload is running"}.get(refused.get(resident.id), "")
        return f"{resident.id} ({gib(resident.gpu_mb)}{reason})"

    @staticmethod
    def _figures(model_id: str, need_mb: int, growth_mb: int, free_mb: int, total_mb: int,
                 refused: dict[str, str], unloadable: list[GpuResident]) -> dict:
        entries = []
        for resident in unloadable:
            entry = {"model": resident.id, "gpu_mb": resident.gpu_mb, "busy": refused.get(resident.id) == "busy"}
            if refused.get(resident.id) == LOCKED:
                entry["locked"] = True
            entries.append(entry)
        return {
            "model": model_id,
            "need_mb": need_mb,
            "headroom_mb": settings.os_gpu_headroom_mb,
            "growth_mb": growth_mb,
            "free_mb": max(free_mb, 0),
            "total_mb": total_mb,
            "unloadable": entries,
        }

    def _refusal(self, model_id: str, need_mb: int, growth_mb: int, free_mb: int, total_mb: int,
                 refused: dict[str, str], loaded: list[GpuResident]) -> InsufficientGpuMemoryError:
        unloadable = [resident for resident in loaded
                      if (not resident.pinned or resident.displaceable) and not resident.replaced_by_load]
        pinned = [resident.id for resident in loaded if resident.pinned and not resident.displaceable]
        growth = f" and {gib(growth_mb)} for the pinned models to grow into" if growth_mb else ""
        message = (
            f"Not enough GPU memory to load {model_id}: it needs about {gib(need_mb)} plus "
            f"{settings.os_gpu_headroom_mb} MiB of headroom{growth}, and {gib(max(free_mb, 0))} of the card's "
            f"{gib(total_mb)} is free."
        )
        named = ", ".join(self._label(resident, refused) for resident in unloadable)
        room_if_unloaded = free_mb + sum(resident.gpu_mb + resident.growth_mb * resident.displaceable
                                         for resident in unloadable)
        if unloadable and room_if_unloaded >= need_mb + settings.os_gpu_headroom_mb + growth_mb:
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
            message, details=self._figures(model_id, need_mb, growth_mb, free_mb, total_mb, refused, unloadable)
        )

    def _cap_refusal(self, model_id: str, need_mb: int, growth_mb: int, memory: GpuMemory, cap: int,
                     refused: dict[str, str], big: list[GpuResident]) -> InsufficientGpuMemoryError:
        named = ", ".join(self._label(resident, refused) for resident in big)
        one = len(big) == 1
        message = (
            f"{model_id} is a big model (about {gib(need_mb)} of GPU memory), and this {gib(memory.total_mb)} card "
            f"holds at most {cap} big model{'s' if cap != 1 else ''} next to the pinned defaults "
            f"(OS_GPU_MAX_BIG_MODELS). {named} {'is' if one else 'are'} loaded and can't be unloaded now: unload "
            f"{'it' if one else 'one'}, or try again once {'it is' if one else 'they are'} idle."
        )
        return InsufficientGpuMemoryError(
            message,
            details=self._figures(model_id, need_mb, growth_mb, memory.free_mb, memory.total_mb, refused, big),
        )
