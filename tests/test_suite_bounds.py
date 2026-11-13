"""The Python suite reports a stuck test as a failure instead of hanging the run.

The Node half bounds each test at 30 s (tests/ui_support.run_node_test, --test-timeout=30000).
Starlette's TestClient WebSocket receive() has no timeout of its own, so a server that stops
sending an event would block a `while websocket.receive_json()[...] != ...` loop until the
server's own idle timeout, or forever.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute
from starlette.testclient import TestClient, WebSocketTestSession


async def _silent(websocket):
    await websocket.accept()
    await websocket.receive()  # never sends; returns when the client disconnects


def _bound_module():
    """The module that installed the bounded receive (tests/conftest.py)."""
    module = sys.modules[WebSocketTestSession.receive.__module__]
    assert hasattr(module, "WEBSOCKET_RECEIVE_TIMEOUT_S"), (
        "TestClient WebSocket receive() is not bounded: a missing server event hangs the suite"
    )
    return module


def test_websocket_receive_without_a_message_fails_the_test(monkeypatch):
    bound = _bound_module()
    assert 0 < bound.WEBSOCKET_RECEIVE_TIMEOUT_S <= 30  # the Node tests' bound
    monkeypatch.setattr(bound, "WEBSOCKET_RECEIVE_TIMEOUT_S", 0.5)
    app = Starlette(routes=[WebSocketRoute("/silent", _silent)])

    with TestClient(app).websocket_connect("/silent") as websocket:
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="no message"):
            websocket.receive_json()
        assert time.monotonic() - started < 10


def test_websocket_receive_still_returns_messages(monkeypatch):
    _bound_module()

    async def echo(websocket):
        await websocket.accept()
        await websocket.send_json({"type": "hello"})
        await websocket.receive()

    with TestClient(Starlette(routes=[WebSocketRoute("/echo", echo)])).websocket_connect("/echo") as websocket:
        assert websocket.receive_json() == {"type": "hello"}


def test_a_stuck_test_dumps_its_traceback(pytestconfig):
    # pyproject.toml [tool.pytest.ini_options]: a test still running after this many seconds dumps
    # every thread's stack, so a hang outside a WebSocket receive names its test and line.
    assert 0 < float(pytestconfig.getini("faulthandler_timeout")) <= 300


# Nothing stopped a test from reaching the network. Once, an unstubbed
# KokoroBackend.download_model got an answer from the Hugging Face hub. tests/conftest.py now keeps the hub offline
# and refuses name lookups and connections to any host but this machine.
#
# The recorder is a CPython audit hook (PEP 578). CPython raises the socket events before it calls the resolver or
# the syscall, and a hook that raises aborts the call, so nothing leaves the machine. It is added after the suite's
# own guard, so it sees only what that guard let through, and it records only while one of these tests runs.


class _NetworkAttempt(RuntimeError):
    """The recorder aborted a call that the suite's guard let through."""


_recording: dict = {"on": False, "attempts": []}
_recording_lock = threading.Lock()


def _is_loopback(host) -> bool:
    if host is None:
        return True
    if isinstance(host, (bytes, bytearray)):
        host = bytes(host).decode("ascii", "replace")
    host = str(host).strip().strip("[]").lower()
    if host in ("", "localhost"):
        return True
    try:
        address = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False  # a name that would go to DNS
    return address.is_loopback or address.is_unspecified


def _record_network(event: str, args: tuple) -> None:
    if not _recording["on"] or not event.startswith("socket."):
        return
    if event in ("socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyname_ex", "socket.gethostbyaddr"):
        host = args[0]
    elif event in ("socket.connect", "socket.connect_ex", "socket.sendto"):
        host = args[1][0] if isinstance(args[1], tuple) and args[1] else None
    else:
        return
    if _is_loopback(host):
        return
    with _recording_lock:
        _recording["attempts"].append((event, str(host)))
    raise _NetworkAttempt(f"{event} to {host!r}: the suite let it through (aborted by the recorder)")


sys.addaudithook(_record_network)


@pytest.fixture
def network_attempts():
    with _recording_lock:
        _recording["attempts"] = []
        _recording["on"] = True
    try:
        yield _recording["attempts"]
    finally:
        with _recording_lock:
            _recording["on"] = False


def _run_bounded(call, timeout: float = 60.0) -> BaseException | None:
    """call() in a thread, bounded by `timeout`; what it raised, or None."""
    outcome: dict = {"error": None}

    def target():
        try:
            call()
        except BaseException as exc:  # noqa: BLE001 - reported to the test
            outcome["error"] = exc

    worker = threading.Thread(target=target, daemon=True, name="bounded-call-worker")
    worker.start()
    worker.join(timeout)
    assert not worker.is_alive(), f"the call did not return within {timeout:g} s"
    return outcome["error"]


def _refused_by_the_suite(error, host: str) -> bool:
    # A refusal inside a task group (anyio's happy-eyeballs connect, as httpx.AsyncClient uses it) comes wrapped.
    while isinstance(error, BaseExceptionGroup) and len(error.exceptions) == 1:
        error = error.exceptions[0]
    return type(error).__name__ == "NetworkAccessBlocked" and host in str(error) and "Stub the boundary" in str(error)


def test_the_hugging_face_hub_is_offline_for_every_test():
    from huggingface_hub import constants

    assert constants.HF_HUB_OFFLINE is True, "HF_HUB_OFFLINE=1 must be set before huggingface_hub is imported"


def test_an_unstubbed_kokoro_download_does_not_reach_the_hugging_face_hub(network_attempts, monkeypatch, tmp_path):
    from huggingface_hub import constants

    from src.tts.backends.kokoro import KokoroBackend

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hf-hub"))  # an empty cache, not the real one
    error = _run_bounded(lambda: KokoroBackend(device="cpu").download_model("kokoro"))
    assert network_attempts == [], f"reached the network boundary: {network_attempts}; then {error!r}"
    assert error is not None, "offline, with an empty cache, the download fails"


def test_an_unstubbed_silero_download_does_not_reach_the_network(network_attempts, tmp_path):
    from src.vad import silero

    error = _run_bounded(lambda: silero._download_model(tmp_path / "silero.onnx"))
    assert network_attempts == [], f"reached the network boundary: {network_attempts}; then {error!r}"
    assert _refused_by_the_suite(error, "raw.githubusercontent.com"), repr(error)


def test_a_plain_outbound_connection_is_refused(network_attempts):
    def call():
        with socket.create_connection(("huggingface.co", 443), timeout=5):
            pass

    error = _run_bounded(call)
    assert network_attempts == [], f"reached the network boundary: {network_attempts}; then {error!r}"
    assert _refused_by_the_suite(error, "huggingface.co"), repr(error)


def test_loopback_stays_open(network_attempts):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(4)
        port = listener.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout=5):
            pass
        with socket.create_connection(("localhost", port), timeout=5):
            pass
    assert network_attempts == []


def test_without_the_opt_in_marker_a_lookup_is_refused(network_attempts):
    error = _run_bounded(lambda: socket.getaddrinfo("example.invalid", 443))
    assert network_attempts == [] and _refused_by_the_suite(error, "example.invalid"), (network_attempts, error)


@pytest.mark.allow_network
def test_the_opt_in_marker_lets_an_authorized_integration_test_through(network_attempts):
    # The suite's guard steps aside; the recorder, after it, still aborts the call before the resolver.
    error = _run_bounded(lambda: socket.getaddrinfo("example.invalid", 443))
    assert isinstance(error, _NetworkAttempt) and network_attempts == [("socket.getaddrinfo", "example.invalid")]


# On Windows the guard let asyncio connect to any IP address. asyncio's default loop there,
# the ProactorEventLoop, connects and sends with ConnectEx, WSAConnect and WSASendTo, which raise no audit event, and
# asyncio and anyio skip the lookup for an IP address. So httpx.AsyncClient (the AI assistant's client) could reach
# http://192.168.1.50:8080 from a test. The recorder above is an audit hook too, and just as blind, so these tests use
# a trick: the guard counts 127.0.0.1 as another machine, and each call goes to a listener of the test's
# own on 127.0.0.1. A call the guard sees is refused before it connects; one it misses reaches the listener.
_LOOPS = [pytest.param(asyncio.SelectorEventLoop, id="selector")]  # CI's loop, on Linux
if os.name == "nt":
    _LOOPS.append(pytest.param(asyncio.ProactorEventLoop, id="proactor"))  # asyncio's default loop on Windows


def _suite_guard():
    """tests/conftest.py, the module that installed the network guard."""
    module = _bound_module()
    assert hasattr(module, "_is_this_machine") and hasattr(module, "NetworkAccessBlocked")
    return module


@contextlib.contextmanager
def _loopback_counts_as_another_machine():
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(_suite_guard(), "_is_this_machine", lambda host: host is None)
        yield


class _LoopbackListener:
    """A 127.0.0.1 TCP listener on its own thread that counts connections and answers each with an HTTP 200."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.sock.settimeout(0.1)
        self.port = self.sock.getsockname()[1]
        self.accepted = 0
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True, name="counting-listener")
        self._thread.start()

    def _serve(self):
        while not self._stopping.is_set():
            try:
                conn, _ = self.sock.accept()
            except OSError:  # a timeout, or the socket closed
                continue
            self.accepted += 1
            with conn:
                conn.settimeout(5)
                try:
                    conn.recv(65536)
                    conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
                except OSError:
                    pass

    def close(self):
        self._stopping.set()
        self._thread.join(5)
        self.sock.close()


@pytest.fixture
def loopback_listener():
    listener = _LoopbackListener()
    try:
        yield listener
    finally:
        listener.close()


@pytest.fixture
def loopback_datagrams():
    """A 127.0.0.1 UDP socket; read it with a short timeout to see what arrived."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(0.5)
        yield receiver


def _run_on(loop_class, call, timeout: float = 30.0) -> BaseException | None:
    """call()'s coroutine on a new loop of loop_class, bounded by `timeout`; what it raised, or None.

    The loop is made before the call starts: a ProactorEventLoop's own wake-up socket pair connects on 127.0.0.1.
    """
    loop = loop_class()
    try:
        async def bounded():
            async with asyncio.timeout(timeout):
                await call()

        try:
            loop.run_until_complete(bounded())
        except BaseException as exc:  # noqa: BLE001 - reported to the test
            return exc
        return None
    finally:
        loop.close()


async def _open_and_close(port: int) -> None:
    _reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.close()
    await writer.wait_closed()


async def _get_models(port: int) -> None:
    import httpx

    async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
        response = await client.get(f"http://127.0.0.1:{port}/v1/models")
        assert response.status_code == 200


async def _send_datagram(port: int) -> None:
    loop = asyncio.get_running_loop()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
        sender.setblocking(False)
        await loop.sock_sendto(sender, b"ping", ("127.0.0.1", port))


async def _datagram_endpoint(port: int) -> None:
    transport, _protocol = await asyncio.get_running_loop().create_datagram_endpoint(
        asyncio.DatagramProtocol, remote_addr=("127.0.0.1", port))
    transport.sendto(b"ping")
    await asyncio.sleep(0.05)
    transport.close()


def _received(receiver: socket.socket) -> bytes | None:
    try:
        return receiver.recv(64)
    except TimeoutError:
        return None


@pytest.mark.parametrize("loop_class", _LOOPS)
@pytest.mark.parametrize("make_call", [_open_and_close, _get_models], ids=["asyncio.open_connection", "httpx.AsyncClient"])
def test_an_asyncio_connection_to_an_ip_address_is_refused(loopback_listener, loop_class, make_call):
    async def call():
        with _loopback_counts_as_another_machine():
            await make_call(loopback_listener.port)

    error = _run_on(loop_class, call)

    assert _refused_by_the_suite(error, "127.0.0.1"), (
        f"the guard let it through: {error!r}; the listener took {loopback_listener.accepted} connection(s)")
    assert loopback_listener.accepted == 0


@pytest.mark.parametrize("loop_class", _LOOPS)
@pytest.mark.parametrize("make_call", [_send_datagram, _datagram_endpoint], ids=["sock_sendto", "datagram_endpoint"])
def test_an_asyncio_datagram_to_an_ip_address_is_refused(loopback_datagrams, loop_class, make_call):
    port = loopback_datagrams.getsockname()[1]

    async def call():
        with _loopback_counts_as_another_machine():
            await make_call(port)

    error = _run_on(loop_class, call)

    assert _refused_by_the_suite(error, "127.0.0.1"), f"the guard let it through: {error!r}"
    assert _received(loopback_datagrams) is None


@pytest.mark.parametrize("loop_class", _LOOPS)
def test_asyncio_on_loopback_stays_open(loopback_listener, loopback_datagrams, loop_class):
    port = loopback_datagrams.getsockname()[1]

    async def call():
        await _open_and_close(loopback_listener.port)
        await _get_models(loopback_listener.port)
        await _send_datagram(port)
        await _datagram_endpoint(port)

    assert _run_on(loop_class, call) is None
    assert loopback_listener.accepted == 2
    assert _received(loopback_datagrams) == _received(loopback_datagrams) == b"ping"


@pytest.mark.allow_network
@pytest.mark.parametrize("loop_class", _LOOPS)
def test_the_opt_in_marker_lets_an_asyncio_connection_through(loopback_listener, loop_class):
    async def call():
        with _loopback_counts_as_another_machine():
            await _open_and_close(loopback_listener.port)
            await _get_models(loopback_listener.port)  # answered only once the listener took both connections

    assert _run_on(loop_class, call) is None
    assert loopback_listener.accepted == 2


# A run of the suite deleted the developer's own downloaded Systran/faster-whisper-base.
# tests/test_model_lifecycle_manual.py::test_delete_artifacts_scoped_to_managed_dirs patched the model manager's cache
# roots, but the real FasterWhisperBackend deletes the model from its own cache first: STT_MODEL_DIR, or else
# src.config.hf_hub_cache_dir(), which was ~/.cache/huggingface/hub because tests/conftest.py set no cache variable.
# Now the suite has a home and a Hugging Face cache of its own under its temp root, and that test points the backend
# at a temp folder too. These tests use fake caches: none of them reads or changes the real ones.


def _is_inside(path, root: Path) -> bool:
    return Path(path).resolve().is_relative_to(root.resolve())


def test_every_model_cache_the_suite_can_reach_is_inside_its_temp_root():
    from huggingface_hub import constants

    from src.backends.faster_whisper import FasterWhisperBackend
    from src.config import hf_hub_cache_dir
    from src.model_manager import ModelManager
    from src.tts.backends.piper_backend import _get_cache_dir as piper_cache_dir
    from src.vad import silero

    root = _bound_module()._test_root
    caches = {
        "home (Path.home())": Path.home(),
        "src.config.hf_hub_cache_dir()": hf_hub_cache_dir(),
        "huggingface_hub HF_HOME": constants.HF_HOME,
        "huggingface_hub HF_HUB_CACHE (Kokoro, Piper and pocket-tts downloads)": constants.HF_HUB_CACHE,
        "faster-whisper cache (its downloads and deletes)": FasterWhisperBackend()._get_cache_dir(),
        "Piper cache": piper_cache_dir(),
        "Silero VAD cache": silero.SILERO_CACHE_DIR,
    }
    for index, cache_root in enumerate(ModelManager(stt_router=None, tts_router=None)._hf_cache_roots()):
        caches[f"model delete root {index} (ModelManager._hf_cache_roots)"] = cache_root

    outside = {name: str(path) for name, path in caches.items() if not _is_inside(path, root)}
    assert outside == {}, f"outside the suite's temp root {root}: {outside}"


@pytest.mark.parametrize("developer_setting", ["HF_HUB_CACHE", "STT_MODEL_DIR"])
def test_the_model_delete_test_leaves_a_model_in_the_developers_cache(tmp_path, monkeypatch, developer_setting):
    from huggingface_hub import constants

    from src.config import settings
    from tests import test_model_lifecycle_manual as lifecycle

    # Stands in for the developer's own cache (~/.cache/huggingface/hub, or the folder STT_MODEL_DIR names in their
    # shell). Never the real one.
    developer_cache = tmp_path / "developer-cache"
    weights = developer_cache / "models--Systran--faster-whisper-base" / "snapshots" / "abc" / "model.bin"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"pretend weights")
    if developer_setting == "HF_HUB_CACHE":
        monkeypatch.setenv("HF_HUB_CACHE", str(developer_cache))
        monkeypatch.setattr(constants, "HF_HUB_CACHE", str(developer_cache))
    else:
        monkeypatch.setattr(settings, "stt_model_dir", str(developer_cache))

    lifecycle.test_delete_artifacts_scoped_to_managed_dirs(tmp_path / "lifecycle-test")

    assert weights.is_file(), "the model delete test deleted Systran/faster-whisper-base from the developer's cache"
