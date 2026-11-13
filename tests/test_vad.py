"""Tests for the VAD module (src/vad/silero.py).

Tests the SileroVAD wrapper, is_speech, get_speech_segments,
and configuration behavior without requiring the real ONNX model.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import socket
import threading
import time
import urllib.parse
import urllib.request
from types import SimpleNamespace

import numpy as np
import onnxruntime
import pytest

from src.vad import silero
from src.vad.silero import (
    SileroVAD,
    Segment,
    VAD_CONTEXT_SIZE,
    VAD_WINDOW_SIZE,
)


# ---------------------------------------------------------------------------
# Mock ONNX session
# ---------------------------------------------------------------------------

class MockOrtSession:
    """Mock ONNX session matching Silero VAD signature."""

    def __init__(self, prob: float = 0.5):
        self.prob = prob
        self.call_count = 0

    def run(self, output_names, inputs):
        self.call_count += 1
        state = inputs["state"]
        return [np.array([[self.prob]], dtype=np.float32), state]


class SequenceSession:
    """Mock that returns a sequence of probabilities."""

    def __init__(self, probs: list[float]):
        self.probs = probs
        self.idx = 0

    def run(self, output_names, inputs):
        prob = self.probs[self.idx % len(self.probs)]
        self.idx += 1
        state = inputs["state"]
        return [np.array([[prob]], dtype=np.float32), state]


class RecordingSession:
    """Mock that records ONNX inputs exactly as the wrapper sends them."""

    def __init__(self, prob: float = 0.5):
        self.prob = prob
        self.inputs: list[dict[str, np.ndarray]] = []

    def run(self, output_names, inputs):
        self.inputs.append({
            "input": inputs["input"].copy(),
            "state": inputs["state"].copy(),
            "sr": inputs["sr"].copy(),
        })
        return [np.array([[self.prob]], dtype=np.float32), inputs["state"]]


# ---------------------------------------------------------------------------
# SileroVAD.__call__
# ---------------------------------------------------------------------------

class TestSileroVADCall:
    def test_returns_probability(self):
        vad = SileroVAD(MockOrtSession(prob=0.9))
        audio = np.zeros(512, dtype=np.float32)
        assert vad(audio) == pytest.approx(0.9)

    def test_empty_audio(self):
        vad = SileroVAD(MockOrtSession(prob=0.9))
        assert vad(np.array([], dtype=np.float32)) == 0.0

    def test_short_audio(self):
        vad = SileroVAD(MockOrtSession(prob=0.9))
        assert vad(np.zeros(100, dtype=np.float32)) == 0.0

    def test_multiple_windows_max(self):
        vad = SileroVAD(SequenceSession([0.1, 0.5, 0.3]))
        audio = np.zeros(1536, dtype=np.float32)  # 3 windows
        assert vad(audio) == pytest.approx(0.5)

    def test_reset(self):
        vad = SileroVAD(MockOrtSession())
        vad._state = np.ones((2, 1, 128), dtype=np.float32)
        vad._context = np.ones(VAD_CONTEXT_SIZE, dtype=np.float32)
        vad.reset()
        assert np.all(vad._state == 0)
        assert np.all(vad._context == 0)

    def test_custom_threshold(self):
        vad = SileroVAD(MockOrtSession(prob=0.3), threshold=0.4)
        assert vad.threshold == 0.4

    def test_sends_silero_v5_context_and_carries_between_windows(self):
        session = RecordingSession(prob=0.9)
        vad = SileroVAD(session)
        audio = np.arange(VAD_WINDOW_SIZE * 2, dtype=np.float32)

        assert vad(audio) == pytest.approx(0.9)

        assert len(session.inputs) == 2
        first = session.inputs[0]["input"][0]
        second = session.inputs[1]["input"][0]
        assert first.shape == (VAD_CONTEXT_SIZE + VAD_WINDOW_SIZE,)
        assert second.shape == (VAD_CONTEXT_SIZE + VAD_WINDOW_SIZE,)
        np.testing.assert_array_equal(first[:VAD_CONTEXT_SIZE], np.zeros(VAD_CONTEXT_SIZE))
        np.testing.assert_array_equal(first[VAD_CONTEXT_SIZE:], audio[:VAD_WINDOW_SIZE])
        np.testing.assert_array_equal(
            second[:VAD_CONTEXT_SIZE],
            audio[VAD_WINDOW_SIZE - VAD_CONTEXT_SIZE:VAD_WINDOW_SIZE],
        )
        np.testing.assert_array_equal(second[VAD_CONTEXT_SIZE:], audio[VAD_WINDOW_SIZE:])
        np.testing.assert_array_equal(vad._context, audio[-VAD_CONTEXT_SIZE:])


# ---------------------------------------------------------------------------
# is_speech
# ---------------------------------------------------------------------------

class TestIsSpeech:
    def test_speech_above_threshold(self):
        vad = SileroVAD(MockOrtSession(prob=0.8), threshold=0.5)
        pcm = np.zeros(512, dtype=np.int16).tobytes()
        assert vad.is_speech(pcm) is True

    def test_silence_below_threshold(self):
        vad = SileroVAD(MockOrtSession(prob=0.2), threshold=0.5)
        pcm = np.zeros(512, dtype=np.int16).tobytes()
        assert vad.is_speech(pcm) is False

    def test_empty_bytes(self):
        vad = SileroVAD(MockOrtSession(prob=0.9))
        assert vad.is_speech(b"") is False

    def test_custom_threshold_override(self):
        vad = SileroVAD(MockOrtSession(prob=0.6), threshold=0.5)
        pcm = np.zeros(512, dtype=np.int16).tobytes()
        assert vad.is_speech(pcm, threshold=0.7) is False
        assert vad.is_speech(pcm, threshold=0.5) is True

    def test_at_exact_threshold(self):
        vad = SileroVAD(MockOrtSession(prob=0.5), threshold=0.5)
        pcm = np.zeros(512, dtype=np.int16).tobytes()
        assert vad.is_speech(pcm) is True

    def test_short_pcm(self):
        """PCM shorter than 512 samples → no windows → not speech."""
        vad = SileroVAD(MockOrtSession(prob=0.9), threshold=0.5)
        pcm = np.zeros(100, dtype=np.int16).tobytes()
        assert vad.is_speech(pcm) is False


# ---------------------------------------------------------------------------
# get_speech_segments
# ---------------------------------------------------------------------------

class TestGetSpeechSegments:
    def test_empty_audio(self):
        vad = SileroVAD(MockOrtSession(prob=0.9))
        assert vad.get_speech_segments(b"") == []

    def test_all_speech(self):
        """All windows above threshold → one segment."""
        vad = SileroVAD(MockOrtSession(prob=0.9), threshold=0.5)
        # 2 seconds at 16kHz = 32000 samples
        pcm = np.zeros(32000, dtype=np.int16).tobytes()
        segments = vad.get_speech_segments(pcm, min_speech_ms=0, silence_ms=800)
        assert len(segments) >= 1
        assert segments[0].start_ms == 0

    def test_all_silence(self):
        """All windows below threshold → no segments."""
        vad = SileroVAD(MockOrtSession(prob=0.1), threshold=0.5)
        pcm = np.zeros(32000, dtype=np.int16).tobytes()
        segments = vad.get_speech_segments(pcm, min_speech_ms=0, silence_ms=100)
        assert segments == []

    def test_speech_then_silence(self):
        """Speech followed by enough silence to close segment."""
        # 10 windows speech, then 30 windows silence
        probs = [0.9] * 10 + [0.1] * 30
        vad = SileroVAD(SequenceSession(probs), threshold=0.5)
        # 40 windows × 512 samples = 20480 samples
        pcm = np.zeros(40 * 512, dtype=np.int16).tobytes()
        segments = vad.get_speech_segments(pcm, min_speech_ms=0, silence_ms=100)
        assert len(segments) == 1
        assert segments[0].start_ms == 0

    def test_min_speech_duration_filters(self):
        """Short speech burst below min_speech_ms is filtered."""
        # 2 windows speech (64ms), then silence
        probs = [0.9] * 2 + [0.1] * 30
        vad = SileroVAD(SequenceSession(probs), threshold=0.5)
        pcm = np.zeros(32 * 512, dtype=np.int16).tobytes()
        segments = vad.get_speech_segments(pcm, min_speech_ms=250, silence_ms=100)
        assert segments == []

    def test_min_speech_ms_is_not_rounded_down_to_whole_windows(self):
        """7 windows are 224 ms, short of a 250 ms minimum."""
        probs = [0.9] * 7 + [0.1] * 40
        vad = SileroVAD(SequenceSession(probs), threshold=0.5)
        pcm = np.zeros(len(probs) * 512, dtype=np.int16).tobytes()
        segments = vad.get_speech_segments(pcm, min_speech_ms=250, silence_ms=100)
        assert segments == []

    def test_silence_ms_is_not_rounded_down_to_whole_windows(self):
        """A 96 ms pause (3 windows) must not end a segment that needs 100 ms of silence."""
        probs = [0.9] * 10 + [0.1] * 3 + [0.9] * 10 + [0.1] * 30
        vad = SileroVAD(SequenceSession(probs), threshold=0.5)
        pcm = np.zeros(len(probs) * 512, dtype=np.int16).tobytes()
        segments = vad.get_speech_segments(pcm, min_speech_ms=0, silence_ms=100)
        assert len(segments) == 1

    def test_segment_dataclass(self):
        seg = Segment(start_ms=100, end_ms=500)
        assert seg.start_ms == 100
        assert seg.end_ms == 500

    def test_two_speech_segments(self):
        """Two speech bursts with silence gap between them."""
        # 10 windows speech, 30 silence, 10 speech, 10 silence
        probs = [0.9] * 10 + [0.1] * 30 + [0.9] * 10 + [0.1] * 30
        vad = SileroVAD(SequenceSession(probs), threshold=0.5)
        pcm = np.zeros(80 * 512, dtype=np.int16).tobytes()
        segments = vad.get_speech_segments(pcm, min_speech_ms=0, silence_ms=100)
        assert len(segments) == 2

    def test_get_speech_segments_uses_silero_v5_context(self):
        session = RecordingSession(prob=0.9)
        vad = SileroVAD(session, threshold=0.5)
        samples = np.arange(VAD_WINDOW_SIZE * 2, dtype=np.int16)

        segments = vad.get_speech_segments(samples.tobytes(), min_speech_ms=0)

        assert segments
        assert len(session.inputs) == 2
        first = session.inputs[0]["input"][0]
        second = session.inputs[1]["input"][0]
        expected = samples.astype(np.float32) / 32768.0
        np.testing.assert_array_equal(first[:VAD_CONTEXT_SIZE], np.zeros(VAD_CONTEXT_SIZE))
        np.testing.assert_array_equal(first[VAD_CONTEXT_SIZE:], expected[:VAD_WINDOW_SIZE])
        np.testing.assert_array_equal(
            second[:VAD_CONTEXT_SIZE],
            expected[VAD_WINDOW_SIZE - VAD_CONTEXT_SIZE:VAD_WINDOW_SIZE],
        )
        np.testing.assert_array_equal(second[VAD_CONTEXT_SIZE:], expected[VAD_WINDOW_SIZE:])


# ---------------------------------------------------------------------------
# Model download (never touches the network)
# ---------------------------------------------------------------------------

class _TrickleResponse:
    """An HTTP response whose server sends one byte per read, slowly."""

    def __init__(self, delay_s: float):
        self.delay_s = delay_s

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read1(self, _n=-1):
        time.sleep(self.delay_s)
        return b"x"


# SILERO_DOWNLOAD_TIMEOUT_S bounds the whole attempt, from the host lookup to the
# last read. It is scaled from 60 s to _TIMEOUT_S; a session gets its answer within _BOUND_S (the deadline plus
# slack for the threads).
_TIMEOUT_S = 1.0
_BOUND_S = 1.5 * _TIMEOUT_S


def _is_loopback(host) -> bool:
    return host in (None, "", "localhost", "127.0.0.1", "::1")


class _StubbedNetwork:
    """socket.getaddrinfo and socket.socket.connect for the download host; loopback goes to the real functions.

    The real urlopen, http.client and ssl run on top, with no network: the host resolves to TEST-NET-1 addresses
    (RFC 5737), whose connects are stubbed. A dropped SYN or a dead resolver waits out its timeout, or until
    release is set, so an attempt the code gave up on ends with the test.
    """

    def __init__(self, host: str):
        self.host = host
        self.addresses: list[tuple[str, int]] = []
        self.resolver_is_dead = False
        self.slow_connect_to: tuple[str, int] | None = None  # None: every SYN is dropped
        self.lookups: list[str] = []
        self.connects: list[tuple[tuple, float | None]] = []  # (address, the socket's timeout)
        self.release = threading.Event()
        self._real_getaddrinfo = socket.getaddrinfo
        self._real_connect = socket.socket.connect

    def getaddrinfo(self, host, port, *args, **kwargs):
        if _is_loopback(host):
            return self._real_getaddrinfo(host, port, *args, **kwargs)
        assert host == self.host, f"unexpected lookup of {host!r}"
        self.lookups.append(host)
        if self.resolver_is_dead:
            self.release.wait(5 * _TIMEOUT_S)  # getaddrinfo takes no timeout: the system resolver gives up
            raise socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", address) for address in self.addresses]

    def connect(self, sock, address):
        host = address[0] if isinstance(address, tuple) and address else None
        if _is_loopback(host):
            return self._real_connect(sock, address)
        timeout = sock.gettimeout()
        self.connects.append((address, timeout))
        wait = 5 * _TIMEOUT_S if timeout is None else timeout
        if self.slow_connect_to is None:
            self.release.wait(wait)  # the SYN is dropped: connect waits out the socket's timeout
            raise TimeoutError("timed out")
        self.release.wait(0.8 * wait)  # the SYN-ACK arrives just inside the timeout
        return self._real_connect(sock, self.slow_connect_to)


async def _one_session() -> tuple[Exception | None, float]:
    """What one session awaiting get_vad_model() gets, and how long it waits for it."""
    started = time.monotonic()
    try:
        await asyncio.wait_for(silero.get_vad_model(), timeout=30)
    except Exception as exc:  # noqa: BLE001 - reported to the test
        return exc, time.monotonic() - started
    return None, time.monotonic() - started


class TestModelDownload:
    @pytest.fixture
    def fake_model(self, monkeypatch, tmp_path):
        content = b"onnx" * 1000
        monkeypatch.setattr(silero, "SILERO_CACHE_DIR", tmp_path)
        monkeypatch.setattr(silero, "SILERO_ONNX_BYTES", len(content))
        monkeypatch.setattr(silero, "SILERO_ONNX_SHA256", hashlib.sha256(content).hexdigest())
        monkeypatch.setattr(silero, "_vad_model", None)
        monkeypatch.setattr(onnxruntime, "InferenceSession", lambda *_a, **_k: MockOrtSession())

        def no_network(*_args, **_kwargs):
            raise AssertionError("tests must not download")

        monkeypatch.setattr(urllib.request, "urlretrieve", no_network)
        monkeypatch.setattr(urllib.request, "urlopen", no_network)
        return content

    async def test_download_is_bounded_and_does_not_hold_the_vad_lock(self, monkeypatch, tmp_path, fake_model):
        """A stalled download inside _vad_lock blocked every later streaming and realtime
        session, with no timeout."""
        calls = []

        def urlopen(url, timeout=None):
            calls.append((url, timeout, silero._vad_lock.locked()))
            return io.BytesIO(fake_model)

        monkeypatch.setattr(urllib.request, "urlopen", urlopen)

        model = await asyncio.wait_for(silero.get_vad_model(), timeout=5)

        assert isinstance(model, SileroVAD)
        assert calls == [(silero.SILERO_ONNX_URL, silero.SILERO_DOWNLOAD_TIMEOUT_S, False)]
        assert 0 < silero.SILERO_DOWNLOAD_TIMEOUT_S <= 300
        assert [path.name for path in tmp_path.iterdir()] == [silero.SILERO_ONNX_FILENAME]
        assert (tmp_path / silero.SILERO_ONNX_FILENAME).read_bytes() == fake_model

    async def test_stalled_download_gives_up_and_leaves_no_file(self, monkeypatch, tmp_path, fake_model):
        monkeypatch.setattr(silero, "SILERO_DOWNLOAD_TIMEOUT_S", 0.2)
        monkeypatch.setattr(urllib.request, "urlopen", lambda _url, timeout=None: _TrickleResponse(0.02))

        started = time.monotonic()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(silero.get_vad_model(), timeout=5)

        assert time.monotonic() - started < 2
        assert list(tmp_path.iterdir()) == []
        assert not silero._vad_lock.locked()
        assert silero._vad_model is None

    async def test_sessions_waiting_on_a_failed_download_share_its_failure(self, monkeypatch, tmp_path, fake_model):
        """Each session queued behind a failed download made its own full attempt, so with
        the network down the Nth concurrent session waited N download timeouts."""
        monkeypatch.setattr(silero, "_download_lock", asyncio.Lock())
        monkeypatch.setattr(silero, "_download_attempts", 0)
        monkeypatch.setattr(silero, "_download_error", None)
        calls = []

        def unreachable(_url, timeout=None):
            calls.append(time.monotonic())
            time.sleep(0.2)  # stands in for the connect timeout
            raise TimeoutError("timed out")

        monkeypatch.setattr(urllib.request, "urlopen", unreachable)

        results = await asyncio.wait_for(
            asyncio.gather(*(silero.get_vad_model() for _ in range(3)), return_exceptions=True), timeout=5,
        )

        assert len(calls) == 1
        assert all(isinstance(result, Exception) for result in results)
        assert all("timed out" in str(result) for result in results)
        assert list(tmp_path.iterdir()) == []

        # The failure is not cached: the next session tries again.
        monkeypatch.setattr(urllib.request, "urlopen", lambda _url, timeout=None: io.BytesIO(fake_model))
        assert isinstance(await asyncio.wait_for(silero.get_vad_model(), timeout=5), SileroVAD)

    async def test_cached_model_is_loaded_without_a_download(self, tmp_path, fake_model):
        (tmp_path / silero.SILERO_ONNX_FILENAME).write_bytes(fake_model)

        assert isinstance(await silero.get_vad_model(), SileroVAD)

    async def test_corrupt_download_is_rejected(self, monkeypatch, tmp_path, fake_model):
        monkeypatch.setattr(urllib.request, "urlopen", lambda _url, timeout=None: io.BytesIO(fake_model[:-1] + b"!"))

        with pytest.raises(RuntimeError, match="SHA256"):
            await silero.get_vad_model()

        assert list(tmp_path.iterdir()) == []

    @pytest.fixture
    def network(self, monkeypatch, tmp_path):
        """No cached model, the download timeout scaled to _TIMEOUT_S, and a stubbed network under the real urlopen.

        Stubbing urlopen would skip the lookup, the connects and the TLS handshake. tests/conftest.py still refuses
        every host but loopback.
        """
        monkeypatch.setattr(silero, "SILERO_CACHE_DIR", tmp_path)
        monkeypatch.setattr(silero, "SILERO_DOWNLOAD_TIMEOUT_S", _TIMEOUT_S)
        monkeypatch.setattr(silero, "_vad_model", None)
        monkeypatch.setattr(silero, "_vad_lock", asyncio.Lock())
        monkeypatch.setattr(silero, "_download_lock", asyncio.Lock())
        monkeypatch.setattr(silero, "_download_attempts", 0)
        monkeypatch.setattr(silero, "_download_error", None)
        # An opener with no proxy from the environment or the Windows registry, built (with its TLS context) here
        monkeypatch.setattr(urllib.request, "_opener", urllib.request.build_opener(urllib.request.ProxyHandler({})))
        net = _StubbedNetwork(urllib.parse.urlsplit(silero.SILERO_ONNX_URL).hostname)
        monkeypatch.setattr(socket, "getaddrinfo", net.getaddrinfo)
        monkeypatch.setattr(socket.socket, "connect", lambda sock, address: net.connect(sock, address))
        threads_before = set(threading.enumerate())
        yield net
        # An attempt the code gave up on ends at its stubbed timeouts once released; leave no thread behind.
        # The event loop's own executor threads ("asyncio_N") are its to stop.
        net.release.set()
        started = [thread for thread in set(threading.enumerate()) - threads_before
                   if not thread.name.startswith("asyncio_")]
        for thread in started:
            thread.join(10)
        assert [thread.name for thread in started if thread.is_alive()] == []

    @pytest.fixture
    def silent_tls_server(self):
        """A loopback TCP server that accepts, reads the ClientHello and never answers it."""
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(4)
        listener.settimeout(0.05)
        state = SimpleNamespace(address=listener.getsockname(), connections=[], first_bytes=b"",
                                hello=threading.Event())
        stop = threading.Event()

        def serve():
            while not stop.is_set():
                try:
                    conn, _peer = listener.accept()
                except TimeoutError:
                    continue
                except OSError:
                    return
                state.connections.append(conn)
                conn.settimeout(0.05)
                while not stop.is_set() and not state.first_bytes:
                    try:
                        state.first_bytes = conn.recv(5) or b"<eof>"
                    except TimeoutError:
                        continue
                    except OSError:
                        break
                state.hello.set()

        thread = threading.Thread(target=serve, daemon=True, name="test-vad-silent-tls-server")
        thread.start()
        yield state
        stop.set()
        for conn in state.connections:
            conn.close()  # the client's handshake, if still waiting, fails now
        listener.close()
        thread.join(5)
        assert not thread.is_alive()

    async def test_overall_bound_covers_the_connect_to_every_address(self, network, tmp_path):
        """socket.create_connection gives every resolved address the full
        timeout, so on a network that drops packets a host with four addresses held the session for four
        timeouts before the deadline, checked only in the read loop, was ever looked at."""
        network.addresses = [(f"192.0.2.{n}", 443) for n in (1, 2, 3, 4)]

        error, seconds = await _one_session()

        assert network.lookups == [network.host]
        assert seconds < _BOUND_S, (
            f"the session waited {seconds:.2f} s against an overall bound of {_TIMEOUT_S:g} s (60 s scaled): "
            f"{len(network.connects)} connects with socket timeouts {[t for _a, t in network.connects]}; {error!r}"
        )
        assert isinstance(error, TimeoutError), repr(error)
        assert list(tmp_path.iterdir()) == []  # the attempt it gave up on holds nothing on disk

    async def test_overall_bound_covers_the_tls_handshake(self, network, silent_tls_server, tmp_path):
        """After a slow connect, the TLS handshake waited its own full timeout."""
        network.addresses = [("192.0.2.10", 443)]
        network.slow_connect_to = silent_tls_server.address

        error, seconds = await _one_session()

        assert len(network.connects) == 1
        assert seconds < _BOUND_S, (
            f"the session waited {seconds:.2f} s against an overall bound of {_TIMEOUT_S:g} s (60 s scaled): the "
            f"connect took 0.8 of its {network.connects[0][1]} s timeout, then the handshake waited; {error!r}"
        )
        assert isinstance(error, TimeoutError), repr(error)
        assert silent_tls_server.hello.wait(5) and silent_tls_server.first_bytes[:1] == b"\x16", (
            f"the client should have sent a TLS handshake record, got {silent_tls_server.first_bytes!r}"
        )
        assert list(tmp_path.iterdir()) == []

    async def test_overall_bound_covers_the_host_lookup(self, network, tmp_path):
        """getaddrinfo takes no timeout, so a dead resolver held the session for
        as long as the system resolver waited."""
        network.resolver_is_dead = True

        error, seconds = await _one_session()

        assert network.lookups == [network.host]
        assert seconds < _BOUND_S, (
            f"the session waited {seconds:.2f} s against an overall bound of {_TIMEOUT_S:g} s (60 s scaled) "
            f"for the host lookup; {error!r}"
        )
        assert isinstance(error, TimeoutError), repr(error)

    async def test_overall_bound_covers_a_read_still_waiting_at_the_deadline(self, monkeypatch, network, tmp_path):
        """The deadline was checked only after a read returned, so the last read
        could run a whole read timeout past it."""
        monkeypatch.setattr(urllib.request, "urlopen", lambda _url, timeout=None: _TrickleResponse(0.9 * _TIMEOUT_S))

        error, seconds = await _one_session()

        assert seconds < _BOUND_S, (
            f"the session waited {seconds:.2f} s against an overall bound of {_TIMEOUT_S:g} s (60 s scaled): the "
            f"second read, started before the deadline, returned {0.8 * _TIMEOUT_S:g} s after it; {error!r}"
        )
        assert isinstance(error, TimeoutError), repr(error)
        assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------

class TestVADConfig:
    def test_default_config_values(self):
        from src.config import settings
        assert settings.stt_vad_enabled is True
        assert settings.stt_vad_threshold == 0.5
        assert settings.stt_vad_min_speech_ms == 250
        assert settings.stt_vad_silence_ms == 800

    def test_disabled_config(self, monkeypatch):
        """STT_VAD_ENABLED=false should be accessible."""
        monkeypatch.setenv("STT_VAD_ENABLED", "false")
        from src.config import Settings
        s = Settings()
        assert s.stt_vad_enabled is False

    def test_custom_threshold(self, monkeypatch):
        monkeypatch.setenv("STT_VAD_THRESHOLD", "0.7")
        from src.config import Settings
        s = Settings()
        assert s.stt_vad_threshold == 0.7

    def test_custom_min_speech_ms(self, monkeypatch):
        monkeypatch.setenv("STT_VAD_MIN_SPEECH_MS", "500")
        from src.config import Settings
        s = Settings()
        assert s.stt_vad_min_speech_ms == 500

    def test_custom_silence_ms(self, monkeypatch):
        monkeypatch.setenv("STT_VAD_SILENCE_MS", "1200")
        from src.config import Settings
        s = Settings()
        assert s.stt_vad_silence_ms == 1200


# ---------------------------------------------------------------------------
# Streaming VAD events
# ---------------------------------------------------------------------------

class TestStreamingVADEvents:
    """Test that streaming sessions include VAD state events."""

    def test_session_accepts_vad_param(self):
        """StreamingSession accepts vad_enabled parameter."""
        from src.streaming import StreamingSession
        # Can't fully construct without a WS, but verify the class signature
        import inspect
        sig = inspect.signature(StreamingSession.__init__)
        assert "vad_enabled" in sig.parameters

    def test_streaming_endpoint_accepts_vad(self):
        """streaming_endpoint accepts vad parameter."""
        from src.streaming import streaming_endpoint
        import inspect
        sig = inspect.signature(streaming_endpoint)
        assert "vad" in sig.parameters
