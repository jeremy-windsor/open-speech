"""Parakeet's fp16 converter (src/backends/parakeet_fp16.py): the conversion on a tiny graph, the child process it runs
in, the cache folder and its temporary folders, and what is left behind when a conversion dies. Nothing is
downloaded."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from src.backends import parakeet_fp16

REV = "rev1"


def _tiny_encoder(folder: Path) -> None:
    """An 'encoder' in the export's shape: a /layers.* MatMul, a Softmax inside the layers, and a node outside the
    layers, with its weights in an external .data file."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    # Large enough for the converter's external data file (tensors over 1 KB).
    w1 = numpy_helper.from_array((rng.standard_normal((64, 64)) / 8).astype(np.float32), "w1")
    w2 = numpy_helper.from_array((rng.standard_normal((64, 64)) / 8).astype(np.float32), "w2")
    nodes = [
        helper.make_node("MatMul", ["audio_signal", "w1"], ["pre"], name="/pre_encode/MatMul"),
        helper.make_node("MatMul", ["pre", "w2"], ["h"], name="/layers.0/MatMul"),
        helper.make_node("Softmax", ["h"], ["s"], axis=-1, name="/layers.0/Softmax"),
        helper.make_node("Relu", ["s"], ["outputs"], name="/layers.0/Relu"),
        helper.make_node("Identity", ["length"], ["encoded_lengths"], name="/Identity"),
    ]
    graph = helper.make_graph(
        nodes, "encoder",
        [helper.make_tensor_value_info("audio_signal", TensorProto.FLOAT, ["b", 64]),
         helper.make_tensor_value_info("length", TensorProto.INT64, ["b"])],
        [helper.make_tensor_value_info("outputs", TensorProto.FLOAT, ["b", 64]),
         helper.make_tensor_value_info("encoded_lengths", TensorProto.INT64, ["b"])],
        [w1, w2],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    folder.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(folder / parakeet_fp16.ENCODER), save_as_external_data=True,
                    all_tensors_to_one_file=True, location=parakeet_fp16.ENCODER + ".data", size_threshold=0)
    for name in (parakeet_fp16.DECODER, *parakeet_fp16.COPIED_FILES):
        (folder / name).write_bytes(b"decoder" if name == parakeet_fp16.DECODER else b"x")


def _counting_convert(calls: list, fail: bool = False):
    def convert(source: Path, target: Path) -> None:
        calls.append(target)
        target.write_bytes(b"graph")
        Path(str(target) + ".data").write_bytes(b"weights")
        if fail:
            raise MemoryError("killed half way")
    return convert


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "snapshot"
    source.mkdir(exist_ok=True)
    for name in parakeet_fp16.SOURCE_FILES:
        (source / name).write_bytes(b"x")
    return source


def _root(repo: Path) -> Path:
    return repo / parakeet_fp16.FP16_DIR_NAME


def _temps(repo: Path) -> list[str]:
    return sorted(p.name for p in _root(repo).iterdir() if p.name.startswith(parakeet_fp16.TEMP_PREFIX))


def _leftover(repo: Path, pid: int, host: str | None = None, revision: str = REV) -> tuple[Path, Path]:
    """What a conversion killed with its server leaves: its temporary folder and the staging copy of the encoder."""
    root = _root(repo)
    root.mkdir(parents=True, exist_ok=True)
    host = parakeet_fp16._host_tag() if host is None else host
    temp = root / f"{parakeet_fp16.TEMP_PREFIX}{parakeet_fp16.fp16_folder_name(revision)}-{host}-{pid}-deadbeef"
    temp.mkdir()
    staging = temp.with_name(temp.name + parakeet_fp16.STAGING_SUFFIX)
    staging.mkdir()
    (staging / (parakeet_fp16.ENCODER + ".data")).write_bytes(b"w" * 1000)  # stands in for 2.4 GB
    return temp, staging


def _age(*paths: Path, seconds: float) -> None:
    old = time.time() - seconds
    for path in paths:
        os.utime(path, (old, old))


def _dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


@pytest.fixture
def live_pid():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    try:
        yield process.pid
    finally:
        process.kill()
        process.wait()


# --- The conversion -----------------------------------------------------------------------------------------------


def test_the_converter_keeps_fp32_islands_and_gives_the_source_output(tmp_path):
    """The default conversion, in its child process, on a tiny graph."""
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxconverter_common")
    import onnxruntime as ort
    from onnx import TensorProto

    source = tmp_path / "snapshot"
    _tiny_encoder(source)

    folder = parakeet_fp16.ensure_fp16(source, tmp_path / "repo", REV)

    assert folder == tmp_path / "repo" / parakeet_fp16.FP16_DIR_NAME / "rev1-c1"
    assert sorted(p.name for p in folder.iterdir()) == sorted(parakeet_fp16.FP16_FILES)
    assert (folder / parakeet_fp16.DECODER_FP16).read_bytes() == b"decoder"
    model = onnx.load(str(folder / parakeet_fp16.ENCODER_FP16))
    types = {init.name: init.data_type for init in model.graph.initializer}
    assert types["w2"] == TensorProto.FLOAT16  # inside the conformer layers
    assert types["w1"] == TensorProto.FLOAT  # outside them: kept fp32
    assert [o.type.tensor_type.elem_type for o in model.graph.output] == [TensorProto.FLOAT, TensorProto.INT64]

    feed = {"audio_signal": np.random.default_rng(1).standard_normal((2, 64)).astype(np.float32),
            "length": np.array([8, 8], dtype=np.int64)}
    want = ort.InferenceSession(str(source / parakeet_fp16.ENCODER), providers=["CPUExecutionProvider"]).run(None, feed)
    got = ort.InferenceSession(str(folder / parakeet_fp16.ENCODER_FP16), providers=["CPUExecutionProvider"]).run(None, feed)
    assert got[0].dtype == np.float32
    np.testing.assert_allclose(got[0], want[0], atol=2e-2)
    np.testing.assert_array_equal(got[1], want[1])


def test_the_converter_reads_a_hugging_face_snapshot_whose_files_link_into_blobs(tmp_path):
    """The hub cache keeps snapshot files as symbolic links into ../../blobs; onnx refuses external data whose real
    path leaves the model's folder, so the encoder is staged as plain files first."""
    pytest.importorskip("onnx")
    pytest.importorskip("onnxconverter_common")
    built = tmp_path / "built"
    _tiny_encoder(built)
    repo = tmp_path / "models--x--y"
    blobs, snapshot = repo / "blobs", repo / "snapshots" / REV
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    try:
        for index, path in enumerate(sorted(built.iterdir())):
            blob = blobs / f"blob{index}"
            path.replace(blob)
            (snapshot / path.name).symlink_to(Path("..", "..", "blobs", blob.name))
    except OSError:
        pytest.skip("this host can't create symbolic links without a privilege")

    folder = parakeet_fp16.ensure_fp16(snapshot, repo, REV)

    assert sorted(p.name for p in folder.iterdir()) == sorted(parakeet_fp16.FP16_FILES)
    assert not _temps(repo)


def test_the_encoder_is_converted_from_plain_files_in_a_staging_folder_that_is_removed(tmp_path):
    source, repo, seen = _source(tmp_path), tmp_path / "repo", []

    def convert(staged: Path, target: Path) -> None:
        seen.append(staged)
        assert staged.parent != source
        for name in (parakeet_fp16.ENCODER, parakeet_fp16.ENCODER + ".data"):
            assert (staged.parent / name).is_file() and not (staged.parent / name).is_symlink()
        _counting_convert([])(staged, target)

    folder = parakeet_fp16.ensure_fp16(source, repo, REV, convert=convert)

    assert seen and seen[0].name == parakeet_fp16.ENCODER
    assert not seen[0].parent.exists()
    assert (source / parakeet_fp16.ENCODER).read_bytes() == b"x"  # the source is untouched
    assert parakeet_fp16.is_complete(folder)


def test_the_converted_cache_is_reused_and_rebuilt_for_a_new_revision(tmp_path):
    source, repo, calls = _source(tmp_path), tmp_path / "repo", []

    first = parakeet_fp16.ensure_fp16(source, repo, REV, convert=_counting_convert(calls))
    again = parakeet_fp16.ensure_fp16(source, repo, REV, convert=_counting_convert(calls))
    assert first == again and len(calls) == 1
    assert parakeet_fp16.cached_fp16_dir(repo, REV) == first

    newer = parakeet_fp16.ensure_fp16(source, repo, "rev2", convert=_counting_convert(calls))
    assert len(calls) == 2 and newer.name == "rev2-c1"
    assert sorted(p.name for p in _root(repo).iterdir()) == ["rev2-c1"]


def test_a_new_converter_version_rebuilds_the_cache(tmp_path, monkeypatch):
    source, repo, calls = _source(tmp_path), tmp_path / "repo", []
    parakeet_fp16.ensure_fp16(source, repo, REV, convert=_counting_convert(calls))
    monkeypatch.setattr(parakeet_fp16, "CONVERTER_VERSION", 2)

    folder = parakeet_fp16.ensure_fp16(source, repo, REV, convert=_counting_convert(calls))

    assert folder.name == "rev1-c2" and len(calls) == 2


def test_a_failed_conversion_leaves_nothing_a_later_load_trusts(tmp_path):
    source, repo, calls = _source(tmp_path), tmp_path / "repo", []

    with pytest.raises(MemoryError):
        parakeet_fp16.ensure_fp16(source, repo, REV, convert=_counting_convert(calls, fail=True))

    assert list(_root(repo).iterdir()) == []
    assert parakeet_fp16.cached_fp16_dir(repo, REV) is None
    parakeet_fp16.ensure_fp16(source, repo, REV, convert=_counting_convert(calls))
    assert parakeet_fp16.cached_fp16_dir(repo, REV) is not None


def test_a_half_written_final_folder_is_not_trusted_and_is_replaced(tmp_path):
    source, repo, calls = _source(tmp_path), tmp_path / "repo", []
    partial = _root(repo) / "rev1-c1"
    partial.mkdir(parents=True)
    (partial / parakeet_fp16.ENCODER_FP16).write_bytes(b"half")

    assert parakeet_fp16.cached_fp16_dir(repo, REV) is None
    folder = parakeet_fp16.ensure_fp16(source, repo, REV, convert=_counting_convert(calls))

    assert folder == partial and len(calls) == 1 and parakeet_fp16.is_complete(folder)


def test_missing_source_files_are_named(tmp_path):
    source = _source(tmp_path)
    (source / "encoder-model.onnx.data").unlink()
    with pytest.raises(FileNotFoundError, match="encoder-model.onnx.data"):
        parakeet_fp16.ensure_fp16(source, tmp_path / "repo", REV, convert=_counting_convert([]))


# --- What a conversion killed with its server leaves ---------------------------------------------------------------


def test_a_dead_conversions_leftovers_are_removed_before_converting_again(tmp_path):
    """The server was killed mid-conversion (OOM, docker stop): the retry right after the restart removes its
    temporary folder and its 2.4 GB staging copy, however young they are."""
    source, repo = _source(tmp_path), tmp_path / "repo"
    temp, staging = _leftover(repo, _dead_pid())
    seen = []

    def convert(staged: Path, target: Path) -> None:
        seen.append(temp.exists() or staging.exists())
        _counting_convert([])(staged, target)

    parakeet_fp16.ensure_fp16(source, repo, REV, convert=convert)

    assert seen == [False]  # gone before the new conversion started
    assert _temps(repo) == []


def test_a_restarted_server_with_the_dead_ones_pid_removes_its_leftovers(tmp_path):
    """In a container the restarted server often gets its predecessor's PID; a folder with this PID that this
    process isn't converting into is a predecessor's."""
    source, repo = _source(tmp_path), tmp_path / "repo"
    _leftover(repo, os.getpid())

    parakeet_fp16.ensure_fp16(source, repo, REV, convert=_counting_convert([]))

    assert _temps(repo) == []


def test_leftovers_are_removed_once_the_fp16_folder_exists(tmp_path):
    """The early return (the fp16 folder exists) and the backend's cache check both remove them."""
    source, repo = _source(tmp_path), tmp_path / "repo"
    parakeet_fp16.ensure_fp16(source, repo, REV, convert=_counting_convert([]))

    _leftover(repo, _dead_pid())
    assert parakeet_fp16.cached_fp16_dir(repo, REV) is not None
    assert _temps(repo) == []

    _leftover(repo, _dead_pid())
    parakeet_fp16.ensure_fp16(source, repo, REV, convert=_counting_convert([]))
    assert _temps(repo) == []


def test_a_running_conversions_folders_are_kept(tmp_path, live_pid):
    """Another process on this host converting now, another host's (a second container on the volume), and an
    older release's: none is removed while young; each is removed once older than any conversion takes."""
    source, repo = _source(tmp_path), tmp_path / "repo"
    running = _leftover(repo, live_pid, revision="rev0")
    other_host = _leftover(repo, _dead_pid(), host="0badf00d", revision="rev0")
    root = _root(repo)
    older_release = root / f"{parakeet_fp16.TEMP_PREFIX}rev0-c1-{_dead_pid()}-deadbeef"
    older_release.mkdir()

    parakeet_fp16.ensure_fp16(source, repo, REV, convert=_counting_convert([]))
    kept = [*running, *other_host, older_release]
    assert all(path.exists() for path in kept)

    _age(*kept, seconds=parakeet_fp16.STALE_TEMP_S + 60)
    parakeet_fp16.cached_fp16_dir(repo, REV)
    assert _temps(repo) == []


def test_this_process_keeps_the_folder_it_is_converting_into(tmp_path):
    """A second load in this process, while the first converts, doesn't remove the first one's folders."""
    source, repo = _source(tmp_path), tmp_path / "repo"
    during = []

    def convert(staged: Path, target: Path) -> None:
        parakeet_fp16.ensure_fp16(source, repo, "rev0", convert=_counting_convert([]))  # cleans up as it goes
        during.append(target.parent.exists() and staged.exists())
        _counting_convert([])(staged, target)

    parakeet_fp16.ensure_fp16(source, repo, REV, convert=convert)

    assert during == [True]
    assert parakeet_fp16.cached_fp16_dir(repo, REV) is not None


def test_pid_liveness(live_pid):
    assert parakeet_fp16._pid_alive(os.getpid())
    assert parakeet_fp16._pid_alive(live_pid)
    assert not parakeet_fp16._pid_alive(_dead_pid())
    assert not parakeet_fp16._pid_alive(0)


# --- The child process --------------------------------------------------------------------------------------------


def _child_running(monkeypatch, code: str) -> None:
    monkeypatch.setattr(parakeet_fp16, "_child_command", lambda source, target: [sys.executable, "-c", code])


def test_the_default_conversion_runs_in_a_child_process(tmp_path, monkeypatch):
    import inspect

    assert inspect.signature(parakeet_fp16.ensure_fp16).parameters["convert"].default is parakeet_fp16.convert_in_child
    command = parakeet_fp16._child_command(Path("a.onnx"), Path("b.onnx"))
    assert command[:3] == [sys.executable, "-m", "src.backends.parakeet_fp16"]


def test_a_conversion_the_oom_killer_stops_kills_only_the_child_and_says_why(tmp_path, monkeypatch):
    if not hasattr(signal, "SIGKILL"):
        pytest.skip("no SIGKILL on this platform")
    source, repo = _source(tmp_path), tmp_path / "repo"
    _child_running(monkeypatch, "import os, signal; os.kill(os.getpid(), signal.SIGKILL)")

    with pytest.raises(parakeet_fp16.ConversionOutOfMemory, match="about 6 GB") as info:
        parakeet_fp16.ensure_fp16(source, repo, REV)

    assert "STT_DEVICE=cpu" in str(info.value)
    assert list(_root(repo).iterdir()) == []  # temporary and staging folders removed


def test_a_child_that_runs_out_of_memory_in_python_gives_the_same_message(tmp_path, monkeypatch):
    source, repo = _source(tmp_path), tmp_path / "repo"
    _child_running(monkeypatch, f"raise SystemExit({parakeet_fp16.CHILD_EXIT_OUT_OF_MEMORY})")

    with pytest.raises(parakeet_fp16.ConversionOutOfMemory, match="STT_DEVICE=cpu"):
        parakeet_fp16.ensure_fp16(source, repo, REV)

    assert list(_root(repo).iterdir()) == []
    assert parakeet_fp16.cached_fp16_dir(repo, REV) is None


def test_the_child_main_turns_a_memory_error_into_its_exit_code(monkeypatch, tmp_path):
    def convert(source, target):
        raise MemoryError

    monkeypatch.setattr(parakeet_fp16, "convert_encoder", convert)
    monkeypatch.setattr(parakeet_fp16, "_prefer_this_process_for_the_oom_killer", lambda: None)
    assert parakeet_fp16._child_main([str(tmp_path / "a"), str(tmp_path / "b")]) == 3
    assert parakeet_fp16._child_main([]) == 2


def test_another_child_failure_is_reported_with_its_output(tmp_path, monkeypatch):
    source, repo = _source(tmp_path), tmp_path / "repo"
    _child_running(monkeypatch, "raise ValueError('the graph is not an encoder')")

    with pytest.raises(RuntimeError, match="the graph is not an encoder") as info:
        parakeet_fp16.ensure_fp16(source, repo, REV)

    assert not isinstance(info.value, parakeet_fp16.ConversionOutOfMemory)
    assert list(_root(repo).iterdir()) == []


def test_free_memory_is_read_from_meminfo_and_the_cgroup_limit(tmp_path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 16000000 kB\nMemAvailable: 8000000 kB\n", encoding="ascii")
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    assert parakeet_fp16.available_memory_bytes(meminfo, cgroup) == 8_000_000 * 1024

    (cgroup / "memory.max").write_text("max\n", encoding="ascii")
    (cgroup / "memory.current").write_text("1000\n", encoding="ascii")
    assert parakeet_fp16.available_memory_bytes(meminfo, cgroup) == 8_000_000 * 1024

    (cgroup / "memory.max").write_text(f"{4 * 10**9}\n", encoding="ascii")
    (cgroup / "memory.current").write_text(f"{10**9}\n", encoding="ascii")
    assert parakeet_fp16.available_memory_bytes(meminfo, cgroup) == 3 * 10**9

    v1 = tmp_path / "v1"
    (v1 / "memory").mkdir(parents=True)
    (v1 / "memory" / "memory.limit_in_bytes").write_text(f"{2 * 10**9}\n", encoding="ascii")
    (v1 / "memory" / "memory.usage_in_bytes").write_text(f"{5 * 10**8}\n", encoding="ascii")
    assert parakeet_fp16.available_memory_bytes(tmp_path / "none", v1) == 15 * 10**8
    assert parakeet_fp16.available_memory_bytes(tmp_path / "none", tmp_path / "none") is None


def test_the_out_of_memory_message_names_the_free_memory_when_known(tmp_path, monkeypatch):
    source, repo = _source(tmp_path), tmp_path / "repo"
    monkeypatch.setattr(parakeet_fp16, "available_memory_bytes", lambda: 2 * 10**9)
    _child_running(monkeypatch, f"raise SystemExit({parakeet_fp16.CHILD_EXIT_OUT_OF_MEMORY})")

    with pytest.raises(parakeet_fp16.ConversionOutOfMemory, match=r"2\.0 GB was free"):
        parakeet_fp16.ensure_fp16(source, repo, REV)


# --- Two processes, and a hard stop -------------------------------------------------------------------------------


def test_a_second_process_whose_rename_loses_the_race_uses_the_first_ones_folder(tmp_path, monkeypatch):
    """Two processes (two containers on one volume) both found no folder; the other renamed its own into place
    between this one's check and its rename."""
    source, repo = _source(tmp_path), tmp_path / "repo"
    final = _root(repo) / parakeet_fp16.fp16_folder_name(REV)
    real_replace = os.replace

    def replace(src, dst):
        if Path(dst) == final:
            final.mkdir()
            for name in parakeet_fp16.FP16_FILES:
                (final / name).write_bytes(b"theirs")
            raise OSError(39, "Directory not empty")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    folder = parakeet_fp16.ensure_fp16(source, repo, REV, convert=_counting_convert([]))

    assert folder == final and parakeet_fp16.is_complete(final)
    assert (final / parakeet_fp16.ENCODER_FP16).read_bytes() == b"theirs"
    assert _temps(repo) == []


def test_a_rename_that_fails_without_a_finished_folder_still_raises(tmp_path, monkeypatch):
    source, repo = _source(tmp_path), tmp_path / "repo"

    def replace(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", replace)
    with pytest.raises(OSError, match="No space"):
        parakeet_fp16.ensure_fp16(source, repo, REV, convert=_counting_convert([]))
    assert list(_root(repo).iterdir()) == []


def test_every_file_is_flushed_to_disk_before_the_rename(tmp_path, monkeypatch):
    source, repo, events = _source(tmp_path), tmp_path / "repo", []
    real_replace = os.replace
    monkeypatch.setattr(parakeet_fp16, "_fsync", lambda path: events.append(("fsync", Path(path).name)))

    def replace(src, dst):
        events.append(("replace", Path(src).name))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace)
    folder = parakeet_fp16.ensure_fp16(source, repo, REV, convert=_counting_convert([]))

    renamed = next(index for index, event in enumerate(events) if event[0] == "replace")
    temp_name = events[renamed][1]
    flushed_before = {name for kind, name in events[:renamed] if kind == "fsync"}
    assert set(parakeet_fp16.FP16_FILES) | {temp_name} <= flushed_before
    assert ("fsync", parakeet_fp16.FP16_DIR_NAME) in events[renamed:]  # the rename itself
    assert parakeet_fp16.is_complete(folder)


def test_an_empty_file_left_by_a_hard_stop_is_not_trusted(tmp_path):
    source, repo = _source(tmp_path), tmp_path / "repo"
    folder = parakeet_fp16.ensure_fp16(source, repo, REV, convert=_counting_convert([]))
    (folder / parakeet_fp16.ENCODER_FP16).write_bytes(b"")

    assert not parakeet_fp16.is_complete(folder)
    assert parakeet_fp16.cached_fp16_dir(repo, REV) is None
    assert parakeet_fp16.ensure_fp16(source, repo, REV, convert=_counting_convert([])) == folder
    assert parakeet_fp16.is_complete(folder)


def test_fsync_works_on_files_and_folders_here(tmp_path):
    (tmp_path / "f").write_bytes(b"x")
    parakeet_fp16._fsync(tmp_path / "f")
    parakeet_fp16._fsync(tmp_path)
    parakeet_fp16._fsync(tmp_path / "missing")
