"""Convert Parakeet's published fp32 ONNX encoder to fp16 for the GPU, once, into a cache folder.

There is no published fp16 export that onnx-asr loads, and a naive conversion fails: without shape inference the
model doesn't load (type errors), and without fp32 islands the encoded lengths come out wrong or the encoder output
is garbage. This conversion keeps in fp32 every node outside the conformer layers (subsampling, the length maths,
the positional encoding), plus Softmax, Div, Where and LayerNormalization, and casts the encoder output back to
fp32. On a 0.6B TDT encoder it gives the fp32 model's word error rate at half the memory.

The conversion needs up to about 6 GB of RAM for a couple of minutes, so it runs in a child process
(``python -m src.backends.parakeet_fp16 SOURCE TARGET``): when the kernel's OOM killer stops it, only the child
dies, and the server reports what happened instead of being restarted.

The converted files are written to a temporary folder beside their final one, flushed to disk, and renamed into
place last, so a conversion that fails or is stopped half way never leaves a folder a later load trusts. The
folder's name carries the source revision and CONVERTER_VERSION: a new revision or a changed conversion builds a
new folder, and the old ones are removed. A temporary folder names the host and process that made it, so one left
by a process that died (a container stopped or killed mid-conversion) is removed by the next load.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)

# Bump when the conversion changes, so caches made by an older one are rebuilt.
CONVERTER_VERSION = 1
# Beside the Hugging Face snapshot folders of the source repository.
FP16_DIR_NAME = "open-speech-fp16"
TEMP_PREFIX = ".tmp-"
STAGING_SUFFIX = "-source"
# A temporary folder older than this is left over from a conversion that died, whoever made it: a running one takes
# minutes. A folder this host made is removed sooner, as soon as the process named in it is gone.
STALE_TEMP_S = 3600

ENCODER = "encoder-model.onnx"
ENCODER_FP16 = "encoder-model.fp16.onnx"
DECODER = "decoder_joint-model.onnx"
# onnx-asr loads the files of one quantization by name: the decoder stays fp32 under the fp16 name.
DECODER_FP16 = "decoder_joint-model.fp16.onnx"
COPIED_FILES = ("config.json", "vocab.txt", "nemo128.onnx")
# The source files the conversion reads (encoder-model.onnx.data holds the encoder's weights).
SOURCE_FILES = (ENCODER, ENCODER + ".data", DECODER, *COPIED_FILES)
FP16_FILES = (ENCODER_FP16, ENCODER_FP16 + ".data", DECODER_FP16, *COPIED_FILES)

# Ops kept in fp32 inside the conformer layers, on top of onnxconverter-common's default list.
FP32_OPS = ("Softmax", "Div", "Where", "LayerNormalization")
# Nodes outside the conformer layers stay fp32: their names don't start with this.
FP16_NODE_PREFIX = "/layers."

# The conversion's peak memory on the 0.6B encoder: its fp32 weights and their fp16 copy, and the graph.
CONVERSION_RAM_BYTES = 6 * 10**9
# The child's exit code when Python itself ran out of memory (a MemoryError rather than the OOM killer).
CHILD_EXIT_OUT_OF_MEMORY = 3
OUT_OF_MEMORY_MESSAGE = (
    "Parakeet's one-time fp16 conversion for the GPU ran out of memory: it needs about 6 GB of free RAM for a couple "
    "of minutes. Free some memory (or raise the container's memory limit) and load the model again, or set "
    "STT_DEVICE=cpu to run Parakeet on the CPU, which needs no conversion."
)
# The repository root, so the child can import this module however the server was started.
_ROOT = Path(__file__).resolve().parents[2]

# Temporary folders this process is converting into now: never stale, though their PID is this process's.
_active_temps: set[str] = set()
_active_lock = threading.Lock()
_TEMP_NAME = re.compile(r"-(?P<host>[0-9a-f]{8})-(?P<pid>\d+)-[0-9a-f]{8}$")


class ConversionOutOfMemory(RuntimeError):
    """The conversion ran out of memory (the OOM killer stopped its process, or it raised MemoryError)."""


def fp16_folder_name(source_revision: str) -> str:
    return f"{source_revision}-c{CONVERTER_VERSION}"


def is_complete(folder: Path) -> bool:
    """Every converted file is there and none is empty (as a hard stop can leave a file whose data never reached
    the disk)."""
    try:
        return folder.is_dir() and all((folder / name).stat().st_size > 0 for name in FP16_FILES)
    except OSError:
        return False


def cached_fp16_dir(repo_dir: Path, source_revision: str) -> Path | None:
    """The finished fp16 folder for this revision and converter, or None when there is none yet.

    Finding it also removes the temporary folders that dead conversions left beside it.
    """
    folder = repo_dir / FP16_DIR_NAME / fp16_folder_name(source_revision)
    if not is_complete(folder):
        return None
    _remove_stale(folder.parent, keep=folder.name, temps_only=True)
    return folder


def convert_encoder(source: Path, target: Path) -> None:
    """Write ``target`` (an .onnx with its weights in ``target.name + '.data'``), the fp16 copy of ``source``.

    Needs onnx and onnxconverter-common (the cuda extra), and RAM for the fp32 weights and their fp16 copy. This
    runs in the calling process; ``convert_in_child`` runs it in a child process.
    """
    import onnx
    from onnx import TensorProto, helper
    from onnx.external_data_helper import load_external_data_for_model
    from onnxconverter_common import float16

    # The graph without its weights stays under protobuf's 2 GB limit, so shape inference runs in memory; it reads
    # each initializer's type and dims, not its data.
    model = onnx.load(str(source), load_external_data=False)
    model = onnx.shape_inference.infer_shapes(model)
    load_external_data_for_model(model, str(source.parent))

    block_nodes = [node.name for node in model.graph.node if not node.name.startswith(FP16_NODE_PREFIX)]
    with warnings.catch_warnings():
        # One warning per weight below fp16's smallest normal value: thousands of lines saying it is clamped.
        warnings.simplefilter("ignore", UserWarning)
        model = float16.convert_float_to_float16(
            model,
            keep_io_types=True,
            disable_shape_infer=True,
            node_block_list=block_nodes,
            op_block_list=list(float16.DEFAULT_OP_BLOCK_LIST) + list(FP32_OPS),
        )

    # The converter leaves some Cast(to=float) nodes whose output the inferred value_info now types as fp16; make
    # each Cast agree with the type its output is declared with.
    declared = {value.name: value.type.tensor_type.elem_type for value in [*model.graph.value_info, *model.graph.output]}
    for node in model.graph.node:
        if node.op_type != "Cast":
            continue
        to = next(attribute for attribute in node.attribute if attribute.name == "to")
        want = declared.get(node.output[0])
        if want is not None and want != to.i:
            to.i = want
    # An output the converter left fp16 is cast back, so callers get fp32 as from the source model.
    for output in model.graph.output:
        if output.type.tensor_type.elem_type != TensorProto.FLOAT16:
            continue
        renamed = output.name + "_fp16"
        for node in model.graph.node:
            node.output[:] = [renamed if name == output.name else name for name in node.output]
            node.input[:] = [renamed if name == output.name else name for name in node.input]
        model.graph.node.append(
            helper.make_node("Cast", [renamed], [output.name], to=TensorProto.FLOAT, name=f"final_cast_{output.name}")
        )
        output.type.tensor_type.elem_type = TensorProto.FLOAT

    onnx.save_model(
        model,
        str(target),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=target.name + ".data",
        size_threshold=1024,
    )


# --- The child process ---------------------------------------------------------------------------------------------


def _child_command(source: Path, target: Path) -> list[str]:
    return [sys.executable, "-m", "src.backends.parakeet_fp16", str(source), str(target)]


def available_memory_bytes(proc_meminfo: Path = Path("/proc/meminfo"),
                           cgroup_root: Path = Path("/sys/fs/cgroup")) -> int | None:
    """The RAM free for a new process here: the kernel's MemAvailable, or less when a cgroup (a container's memory
    limit) leaves less. None where neither can be read (not Linux)."""
    found: list[int] = []
    try:
        for line in proc_meminfo.read_text(encoding="ascii", errors="replace").splitlines():
            if line.startswith("MemAvailable:"):
                found.append(int(line.split()[1]) * 1024)
                break
    except (OSError, ValueError, IndexError):
        pass
    # cgroup v2, then v1 (where "no limit" is a huge number).
    for limit_name, usage_name in (("memory.max", "memory.current"),
                                   ("memory/memory.limit_in_bytes", "memory/memory.usage_in_bytes")):
        try:
            limit = (cgroup_root / limit_name).read_text(encoding="ascii").strip()
            usage = int((cgroup_root / usage_name).read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            continue
        if limit.isdigit() and int(limit) < 2**60:
            found.append(max(0, int(limit) - usage))
        break
    return min(found) if found else None


def _gb(size: int) -> str:
    return f"{size / 10**9:.1f} GB"


def convert_in_child(source: Path, target: Path) -> None:
    """``convert_encoder(source, target)`` in a child process, so running out of memory stops only the child.

    Raises ConversionOutOfMemory when the child was killed (the OOM killer's SIGKILL) or ran out of memory, and
    RuntimeError, with the end of the child's output, when it failed otherwise.
    """
    free = available_memory_bytes()
    if free is not None and free < CONVERSION_RAM_BYTES:
        logger.warning("Only %s of RAM is free; Parakeet's fp16 conversion needs about %s and may run out",
                       _gb(free), _gb(CONVERSION_RAM_BYTES))
    result = subprocess.run(
        _child_command(Path(source).resolve(), Path(target).resolve()),
        cwd=_ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", check=False,
    )
    code = result.returncode
    if code == 0:
        return
    sigkill = getattr(signal, "SIGKILL", None)
    if code == CHILD_EXIT_OUT_OF_MEMORY or (sigkill is not None and code in (-sigkill, 128 + sigkill)):
        message = OUT_OF_MEMORY_MESSAGE
        if free is not None:
            message += f" ({_gb(free)} was free when it started.)"
        raise ConversionOutOfMemory(message)
    tail = [line for line in (result.stdout or "").splitlines() if line.strip()][-5:]
    detail = (": " + " | ".join(tail)) if tail else ""
    raise RuntimeError(f"Parakeet's fp16 conversion failed (its process exited with code {code}){detail}")


def _prefer_this_process_for_the_oom_killer() -> None:
    """Make the kernel pick this child, not the server, when memory runs out (raising one's own score needs no
    privilege)."""
    try:
        Path("/proc/self/oom_score_adj").write_text("1000", encoding="ascii")
    except OSError:
        pass


def _child_main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m src.backends.parakeet_fp16 SOURCE_ENCODER TARGET_ENCODER", file=sys.stderr)
        return 2
    _prefer_this_process_for_the_oom_killer()
    try:
        convert_encoder(Path(argv[0]), Path(argv[1]))
    except MemoryError:
        print("out of memory", file=sys.stderr)
        return CHILD_EXIT_OUT_OF_MEMORY
    return 0


# --- Temporary folders and the cache folder ------------------------------------------------------------------------


def _host_tag() -> str:
    return hashlib.sha1(socket.gethostname().encode("utf-8", "replace")).hexdigest()[:8]


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) would terminate the process on Windows; ask for its exit code instead.
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # access denied: it exists
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _is_stale_temp(entry: Path, now: float) -> bool:
    name = entry.name.removesuffix(STAGING_SUFFIX)
    with _active_lock:
        if name in _active_temps:
            return False
    try:
        if now - entry.stat().st_mtime >= STALE_TEMP_S:
            return True
    except OSError:
        return False
    match = _TEMP_NAME.search(name)
    if match is None or match["host"] != _host_tag():
        return False  # an older name, or another host's (a container sharing the volume): only its age tells
    pid = int(match["pid"])
    # A restarted container's server often gets the PID its dead predecessor had: a folder named with this
    # process's PID that it isn't converting into now is a predecessor's.
    return pid == os.getpid() or not _pid_alive(pid)


def _remove_stale(fp16_root: Path, keep: str, *, temps_only: bool = False) -> None:
    """Remove the temporary folders dead conversions left and, unless ``temps_only``, the fp16 folders of other
    revisions or converter versions."""
    try:
        entries = list(fp16_root.iterdir())
    except OSError:
        return
    now = time.time()
    for entry in entries:
        if entry.name == keep or not entry.is_dir():
            continue
        if entry.name.startswith(TEMP_PREFIX):
            if not _is_stale_temp(entry, now):
                continue  # perhaps another process converting now
        elif temps_only:
            continue
        shutil.rmtree(entry, ignore_errors=True)
        logger.info("Removed old Parakeet fp16 cache %s", entry)


def _stage_encoder(source_dir: Path, staging: Path) -> Path:
    """Put the fp32 encoder and its weights file in ``staging`` as plain files, and return the encoder's path there.

    The Hugging Face cache keeps each snapshot file as a symbolic link into the repository's ``blobs`` folder, and
    onnx refuses external data whose real path leaves the model's own folder, or that has more than one hard link.
    So both files are copied (about 2.4 GB, once; the folder is removed after the conversion).
    """
    staging.mkdir()
    for name in (ENCODER, ENCODER + ".data"):
        shutil.copyfile(os.path.realpath(source_dir / name), staging / name)
    return staging / ENCODER


def _fsync(path: Path) -> None:
    """Flush a file, or a folder's entries, to the disk; best effort (Windows can't open a folder, and some
    filesystems refuse fsync)."""
    try:
        if path.is_dir():
            if os.name == "nt":
                return
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        else:
            fd = os.open(path, os.O_RDWR if os.name == "nt" else os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError as exc:
        logger.debug("fsync of %s failed: %s", path, exc)
    finally:
        os.close(fd)


def _flush_folder(folder: Path) -> None:
    for entry in sorted(folder.iterdir()):
        _fsync(entry)
    _fsync(folder)


def ensure_fp16(source_dir: Path, repo_dir: Path, source_revision: str, *, convert=convert_in_child) -> Path:
    """Return the fp16 folder for ``source_revision``, converting ``source_dir``'s fp32 files into it first if needed.

    ``repo_dir`` is the source repository's cache folder; the fp16 folders live in its FP16_DIR_NAME subfolder. The
    conversion writes to a temporary folder there and renames it into place only when every file is written and
    flushed. ``convert(source_encoder, target_encoder)`` does the conversion; by default in a child process.
    """
    fp16_root = repo_dir / FP16_DIR_NAME
    final = fp16_root / fp16_folder_name(source_revision)
    if is_complete(final):
        _remove_stale(fp16_root, keep=final.name)
        return final
    missing = [name for name in SOURCE_FILES if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Parakeet source files missing from {source_dir}: {', '.join(missing)}")

    fp16_root.mkdir(parents=True, exist_ok=True)
    # Before converting, so the 2.4 GB staging copy a killed conversion left doesn't stay beside a new one.
    _remove_stale(fp16_root, keep=final.name)
    temp = fp16_root / f"{TEMP_PREFIX}{final.name}-{_host_tag()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    with _active_lock:
        _active_temps.add(temp.name)
    try:
        return _convert_into(source_dir, fp16_root, final, temp, convert)
    finally:
        with _active_lock:
            _active_temps.discard(temp.name)


def _convert_into(source_dir: Path, fp16_root: Path, final: Path, temp: Path, convert) -> Path:
    temp.mkdir()
    started = time.monotonic()
    logger.info("Converting Parakeet's encoder to fp16 (once; about two minutes and up to 6 GB of RAM)")
    try:
        staging = temp.with_name(temp.name + STAGING_SUFFIX)
        try:
            convert(_stage_encoder(source_dir, staging), temp / ENCODER_FP16)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        shutil.copyfile(source_dir / DECODER, temp / DECODER_FP16)
        for name in COPIED_FILES:
            shutil.copyfile(source_dir / name, temp / name)
        if not is_complete(temp):
            raise RuntimeError("Parakeet fp16 conversion did not write every file")
        _flush_folder(temp)
        if final.exists():
            # A folder without every file is a leftover no load trusts (is_complete); replace it.
            if is_complete(final):
                shutil.rmtree(temp, ignore_errors=True)
                return final
            shutil.rmtree(final, ignore_errors=True)
        try:
            os.replace(temp, final)
        except OSError:
            # Another process (a second container on this volume) renamed its finished folder into place first.
            if not is_complete(final):
                raise
            shutil.rmtree(temp, ignore_errors=True)
            return final
        _fsync(fp16_root)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    logger.info("Parakeet fp16 encoder written to %s in %.0f s", final, time.monotonic() - started)
    _remove_stale(fp16_root, keep=final.name)
    return final


if __name__ == "__main__":
    raise SystemExit(_child_main(sys.argv[1:]))
