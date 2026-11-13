"""The Node test runner (tests/ui_support.run_node_test) keeps a bad test from harming the host.

A node test that looped while allocating audio buffers grew to ~50 GB and starved the machine, and a
timed-out run left its child node processes running. The runner now caps a run's memory (Windows) and
stops its whole process tree when the run ends or times out.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from tests.ui_support import require_node, run_node_test


def _write_test(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "probe.test.js"
    path.write_text("import { test } from 'node:test';\n" + body, encoding="utf-8")
    return path


def _running(pid: int) -> bool:
    if os.name == "nt":
        import ctypes

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            k32.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == 259  # STILL_ACTIVE
        finally:
            k32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.skipif(sys.platform != "win32", reason="the memory cap uses a Windows job object")
def test_a_runaway_allocation_fails_the_test_instead_of_starving_the_host(tmp_path):
    require_node()
    probe = _write_test(tmp_path, """
test('allocates without bound', () => {
  const kept = [];
  for (let i = 0; i < 64; i += 1) kept.push(new Float64Array(8 * 1024 * 1024).fill(i)); // 64 x 64 MB
});
""")
    started = time.monotonic()
    with pytest.raises(pytest.fail.Exception) as failed:
        run_node_test(probe, timeout=60, memory_limit_mb=256)
    assert "memory cap" in str(failed.value)
    assert time.monotonic() - started < 30


def test_a_run_within_the_cap_passes(tmp_path):
    require_node()
    probe = _write_test(tmp_path, """
test('allocates a little', () => {
  const kept = [];
  for (let i = 0; i < 4; i += 1) kept.push(new Float64Array(1024 * 1024).fill(i)); // 4 x 8 MB
});
""")
    assert "pass 1" in run_node_test(probe, timeout=60, memory_limit_mb=512)


def test_a_timed_out_run_leaves_no_node_process_behind(tmp_path):
    require_node()
    pid_file = tmp_path / "pid.txt"
    probe = _write_test(tmp_path, f"""
import {{ writeFileSync }} from 'node:fs';
test('hangs', async () => {{
  writeFileSync({str(pid_file)!r}, String(process.pid));
  await new Promise((resolve) => setTimeout(resolve, 120000));
}});
""")
    with pytest.raises(pytest.fail.Exception) as failed:
        run_node_test(probe, timeout=8)
    assert "timed out" in str(failed.value)
    pid = int(pid_file.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 5
    while _running(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _running(pid), "the test file's node process outlived the timed-out run"
