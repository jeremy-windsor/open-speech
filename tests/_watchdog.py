"""A hard watchdog for a test that can hang (there is no pytest-timeout).

During a test, pytest's capture holds both ``sys.stderr`` and fd 2, and it shows what they
got only in a test report. So a hard exit that dumps there shows nothing: the run stops with no summary and no test
named. And ``faulthandler`` has a single timer, which pytest's ``faulthandler_timeout`` (pyproject.toml) already
uses for each test. So this watchdog writes to a copy of the stderr pytest was started with, taken while capture is
suspended, and it times the test with a thread of its own, leaving pytest's timer armed.
"""

from __future__ import annotations

import contextlib
import faulthandler
import os
import threading
from collections.abc import Iterator

import pytest


@contextlib.contextmanager
def hard_watchdog(request: pytest.FixtureRequest, seconds: float) -> Iterator[None]:
    """End the run if the block still runs after ``seconds``: name the test, print every thread's stack, then exit.

    The report goes to the real stderr, where pytest's own faulthandler dump goes. Then the process ends with
    ``os._exit(1)``, as ``faulthandler.dump_traceback_later(exit=True)`` did before. The timer is a Python thread, so
    it needs the GIL: for a hang that holds it, pytest's ``faulthandler_timeout`` timer, still armed, prints the stacks.
    """
    where = request.node.nodeid
    stderr = _uncaptured_stderr(request.config)
    done = threading.Event()

    def watch() -> None:
        if done.wait(seconds):
            return
        try:
            _write(stderr, f"\n+++ Watchdog: {where} is still running after {seconds:g} s. "
                           "The stack of every thread follows, then the run ends (exit 1). +++\n")
            faulthandler.dump_traceback(file=stderr, all_threads=True)
            _write(stderr, f"+++ Watchdog: the run ended in {where}; the tests after it did not run. +++\n")
        finally:
            os._exit(1)

    watcher = threading.Thread(target=watch, name=f"watchdog for {where}", daemon=True)
    watcher.start()
    try:
        yield
    finally:
        done.set()
        watcher.join()
        os.close(stderr)


def _uncaptured_stderr(config: pytest.Config) -> int:
    """A new descriptor for the stderr pytest was started with, never its capture file.

    While capture is suspended, as ``capsys.disabled()`` does it, fd 2 is the real stderr again.
    """
    capture = config.pluginmanager.getplugin("capturemanager")
    if capture is None:  # -p no:capture
        return os.dup(2)
    with capture.global_and_fixture_disabled():
        return os.dup(2)


def _write(fd: int, text: str) -> None:
    data = text.encode("utf-8", "backslashreplace")
    while data:
        data = data[os.write(fd, data):]
