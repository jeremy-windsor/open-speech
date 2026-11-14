"""Global test configuration."""

import gc
import ipaddress
import os
import socket
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

# Disable SSL in tests so TestClient works over plain HTTP
os.environ["STT_SSL_ENABLED"] = "false"
# No background warm-up of TTS_MODEL at app startup: on a machine with Kokoro installed every
# lifespan would load it, or download it first. A test that wants the warm-up turns it on.
os.environ["TTS_PRELOAD_DEFAULT"] = "false"
# No test reads this machine's GPU: the GPU memory guard is off, and its own tests turn it on with a fake card.
os.environ["OS_GPU_MEMORY_GUARD"] = "false"

# No test reaches the network. huggingface_hub reads this when it
# is imported, so it is set before anything imports it. It also stops hf_xet's native downloads, which no socket
# hook sees. The guard below refuses every other lookup or connection to a host that isn't this machine.
os.environ["HF_HUB_OFFLINE"] = "1"
if "huggingface_hub" in sys.modules:  # a plugin imported it first
    from huggingface_hub import constants as _hf_constants

    _hf_constants.HF_HUB_OFFLINE = True


class NetworkAccessBlocked(RuntimeError):
    """A test tried to reach a host that isn't this machine.

    It isn't an OSError, so no client takes it for a host that is down and retries or falls back: it fails the test.
    """


_network_guard = {"on": True}  # off only during a test marked allow_network
_LOCAL_NAMES = frozenset({"", "localhost", socket.gethostname().lower()})  # socket.getfqdn() looks up its own name
_LOOKUP_EVENTS = frozenset({"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyname_ex",
                            "socket.gethostbyaddr"})
_ADDRESS_EVENTS = frozenset({"socket.connect", "socket.connect_ex", "socket.sendto", "socket.sendmsg"})


def _is_this_machine(host) -> bool:
    if host is None:
        return True
    if isinstance(host, (bytes, bytearray)):
        host = bytes(host).decode("ascii", "replace")
    host = str(host).strip().strip("[]").lower().rstrip(".")
    if host in _LOCAL_NAMES or host.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False  # a name that would go to DNS
    return address.is_loopback or address.is_unspecified


def _host_of(address):
    return address[0] if isinstance(address, tuple) and address else None  # None: an AF_UNIX path, or no address


def _refuse_unless_this_machine(host, event: str) -> None:
    if _network_guard["on"] and not _is_this_machine(host):
        raise NetworkAccessBlocked(
            f"A test tried to reach {host!r} ({event}); tests may not use the network. Stub the boundary "
            "(httpx.MockTransport, a fake backend), or mark an authorized integration test "
            "@pytest.mark.allow_network."
        )


def _refuse_the_network(event: str, args: tuple) -> None:
    # A CPython audit hook (PEP 578): the socket events come before the resolver or the syscall, and raising here
    # aborts the call, so nothing leaves the machine. It sees calls from every library and thread, except asyncio's
    # connections and datagrams on Windows (below).
    if not _network_guard["on"] or not event.startswith("socket."):
        return
    if event in _LOOKUP_EVENTS:
        host = args[0]
    elif event in _ADDRESS_EVENTS:
        host = _host_of(args[1])
    elif event == "socket.getnameinfo":
        host = _host_of(args[0])
    else:
        return
    _refuse_unless_this_machine(host, event)


sys.addaudithook(_refuse_the_network)

if os.name == "nt":
    # asyncio's default loop on Windows, the ProactorEventLoop, connects and sends with
    # ConnectEx, WSAConnect and WSASendTo, which raise no audit event, and neither asyncio nor anyio (httpx.AsyncClient)
    # looks up an IP address. So the hook never saw a connection to one. The proactor checks the address first.
    from asyncio import windows_events as _windows_events

    _real_proactor_connect = _windows_events.IocpProactor.connect
    _real_proactor_sendto = _windows_events.IocpProactor.sendto

    def _guarded_proactor_connect(self, conn, address):
        _refuse_unless_this_machine(_host_of(address), "asyncio IocpProactor.connect")
        return _real_proactor_connect(self, conn, address)

    def _guarded_proactor_sendto(self, conn, buf, flags=0, addr=None):
        # addr is None on a connected socket, whose connect was checked.
        _refuse_unless_this_machine(_host_of(addr), "asyncio IocpProactor.sendto")
        return _real_proactor_sendto(self, conn, buf, flags, addr)

    _windows_events.IocpProactor.connect = _guarded_proactor_connect
    _windows_events.IocpProactor.sendto = _guarded_proactor_sendto


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "allow_network: an authorized integration test that may reach hosts other than this machine"
    )


@pytest.fixture(autouse=True)
def _network_guard_for_this_test(request):
    if request.node.get_closest_marker("allow_network") is None:
        yield
        return
    _network_guard["on"] = False
    try:
        yield
    finally:
        _network_guard["on"] = True


# Import-time services need writable paths before individual test fixtures run.
# Keep their persistent data separate from the checkout and the running service.
_test_data = TemporaryDirectory(prefix="open-speech-tests-")
_test_root = Path(_test_data.name)
os.environ["OS_VOICE_LIBRARY_PATH"] = str(_test_root / "voices")
os.environ["OS_STUDIO_DB_PATH"] = str(_test_root / "studio.db")
os.environ["OS_CONVERSATIONS_DIR"] = str(_test_root / "conversations")
os.environ["OS_COMPOSER_DIR"] = str(_test_root / "composer")
os.environ["TTS_CACHE_DIR"] = str(_test_root / "tts-cache")
os.environ["READER_CACHE_DIR"] = str(_test_root / "reader")

# A test that deleted Systran/faster-whisper-base through the real FasterWhisperBackend
# deleted it from the developer's own ~/.cache/huggingface/hub, because no cache variable was set here. No test may
# write or delete the developer's model caches, so the session has its own under _test_root: a home (Path.home(), which
# holds ModelManager's fixed delete root and the Silero VAD cache) and a Hugging Face home and hub cache, where
# faster-whisper, Kokoro, Piper and pocket-tts download and delete. STT_MODEL_DIR from the shell would send
# faster-whisper's deletes to its folder, so it goes. huggingface_hub reads these when it is imported, like
# HF_HUB_OFFLINE above. A test that wants a cache of its own still sets one; monkeypatch puts these back afterwards.
_test_home = _test_root / "home"
_test_home.mkdir()
os.environ["HOME"] = os.environ["USERPROFILE"] = str(_test_home)  # Path.home() on POSIX, and on Windows
os.environ["XDG_CACHE_HOME"] = str(_test_home / ".cache")
os.environ["HF_HOME"] = str(_test_home / ".cache" / "huggingface")
os.environ["HF_HUB_CACHE"] = str(_test_home / ".cache" / "huggingface" / "hub")
for _name in ("HUGGINGFACE_HUB_CACHE", "STT_MODEL_DIR"):
    os.environ.pop(_name, None)

# src.config builds its `settings` once, when it is first imported. Build it here, from the environment just set, so
# that a test which clears os.environ (patch.dict(..., clear=True)) before anything else imports src.config can't
# leave the app's default paths (/home/openspeech/...) in it; src.main, imported later by another test, would then
# create its data folders there. Without this, test_env_compat.py run alone created D:\home\openspeech this way.
import src.config  # noqa: E402,F401

if "huggingface_hub" in sys.modules:  # a plugin imported it first
    from huggingface_hub import constants as _hf_constants

    _hf_constants.HF_HOME = os.environ["HF_HOME"]
    _hf_constants.HF_HUB_CACHE = _hf_constants.HUGGINGFACE_HUB_CACHE = os.environ["HF_HUB_CACHE"]

# Starlette's TestClient WebSocket receive() waits without a timeout, so a server that stops sending
# an event would hang the run until its own idle timeout. Bound each receive like the Node tests
# (--test-timeout=30000): the test fails with TimeoutError instead. tests/test_suite_bounds.py.
WEBSOCKET_RECEIVE_TIMEOUT_S = 30.0


def _bounded_websocket_receive(self):
    import anyio

    async def receive():
        timeout = WEBSOCKET_RECEIVE_TIMEOUT_S
        with anyio.move_on_after(timeout):
            return await self._send_rx.receive()
        raise TimeoutError(f"TestClient WebSocket: no message from the server within {timeout} s")

    return self.portal.call(receive)


@pytest.fixture(autouse=True, scope="session")
def _bound_websocket_receive():
    from starlette.testclient import WebSocketTestSession

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(WebSocketTestSession, "receive", _bounded_websocket_receive)
        yield


@pytest.fixture(autouse=True)
def _reset_security_state():
    # Rate-limit buckets, failed-auth lockouts and WebSocket tickets are process-wide;
    # a test must not inherit another test's lockout of the shared test client address.
    middleware = sys.modules.get("src.middleware")
    if middleware is not None:
        middleware.reset_security_state_for_tests()


@pytest.fixture
def create_symlink():
    def create(link, target, *, target_is_directory=False):
        try:
            link.symlink_to(target, target_is_directory=target_is_directory)
        except OSError as exc:
            if getattr(exc, "winerror", None) == 1314:
                pytest.skip("Windows account lacks symbolic-link privilege")
            raise

    return create


def pytest_sessionfinish(session, exitstatus):
    # Windows cannot remove SQLite files while connections still own them.
    # Close import-time stores before TemporaryDirectory's exit finalizer runs.
    storage = sys.modules.get("src.storage")
    if storage is not None and storage._conn is not None:
        storage._conn.close()
        storage._conn = None
    reader_store = sys.modules.get("src.reader.store")
    if reader_store is not None:
        reader_store.close()
    main = sys.modules.get("src.main")
    if main is not None and main.batch_store._conn is not None:
        main.batch_store._conn.close()
        main.batch_store._conn = None
    gc.collect()
    _test_data.cleanup()
