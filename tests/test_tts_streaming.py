"""Tests for streaming ffmpeg encoder and pipeline streaming."""

import gc
import io
import itertools
import subprocess
import sys
import threading
import types
import warnings
import wave

import numpy as np
import pytest

from src.tts import pipeline
from src.tts.pipeline import (
    StreamingFFmpegEncoder,
    encode_audio,
    encode_audio_streaming,
    encode_wav,
)


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


needs_ffmpeg = pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg not installed")


# Stand-ins for ffmpeg at the Popen boundary. The chatty one logs to stderr for every block
# it reads whatever log options it gets; the failing one exits with an encoder error.
_CHATTY_FFMPEG = r"""
import sys
while True:
    data = sys.stdin.buffer.read(4096)
    if not data:
        break
    sys.stderr.buffer.write(b"[libmp3lame @ 0x55d0] Queue input is backward in time\n" * 32)
    sys.stderr.buffer.flush()
    sys.stdout.buffer.write(data[:256])
    sys.stdout.buffer.flush()
"""

_FAILING_FFMPEG = r"""
import sys
sys.stdin.buffer.read()
sys.stderr.buffer.write(b"Unknown encoder 'libmp3lame'\n")
sys.exit(1)
"""

# A build without the encoder fails while opening the output, before it reads any input.
_EARLY_EXIT_FFMPEG = r"""
import sys
sys.stderr.buffer.write(b"Unknown encoder 'libmp3lame'\n")
sys.exit(1)
"""

# Writes a container header even when its input holds no samples, as ffmpeg's muxers do.
_HEADER_FFMPEG = r"""
import sys
sys.stdout.buffer.write(b"HEADER")
sys.stdout.buffer.write(sys.stdin.buffer.read())
"""

# Echoes each block as it arrives, so the stream yields bytes before its input ends.
_ECHO_FFMPEG = r"""
import os, sys
while True:
    data = os.read(0, 65536)
    if not data:
        break
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()
"""


def _fake_ffmpeg(monkeypatch, script: str) -> list:
    started: list = []

    def popen(cmd, **kwargs):
        assert cmd[0] == "ffmpeg"
        proc = subprocess.Popen([sys.executable, "-c", script], **kwargs)
        started.append(proc)
        return proc

    fake_subprocess = types.ModuleType("subprocess")
    fake_subprocess.__dict__.update(vars(subprocess))
    fake_subprocess.Popen = popen
    monkeypatch.setattr(pipeline, "subprocess", fake_subprocess)
    return started


class TestStreamingFFmpegEncoder:
    @needs_ffmpeg
    def test_mp3_produces_valid_output(self):
        enc = StreamingFFmpegEncoder("mp3", sample_rate=24000)
        try:
            # Write several chunks
            for _ in range(5):
                chunk = np.random.randn(4800).astype(np.float32) * 0.1
                enc.write_chunk(chunk)
            result = enc.finish()
            # MP3 files start with 0xff 0xfb or ID3 tag
            assert len(result) > 0
            assert result[:3] == b"ID3" or result[0] == 0xFF
        finally:
            enc.close()

    @needs_ffmpeg
    def test_flac_produces_valid_output(self):
        enc = StreamingFFmpegEncoder("flac", sample_rate=24000)
        try:
            chunk = np.zeros(24000, dtype=np.float32)
            enc.write_chunk(chunk)
            result = enc.finish()
            assert len(result) > 0
            assert result[:4] == b"fLaC"
        finally:
            enc.close()

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            StreamingFFmpegEncoder("wav")

    def test_does_not_stall_while_ffmpeg_writes_to_stderr(self, monkeypatch):
        # Python subprocess docs: writing stdin while another PIPE goes unread can deadlock
        # once the child fills that pipe (about 4 KiB on Windows, 64 KiB on Linux).
        started = _fake_ffmpeg(monkeypatch, _CHATTY_FFMPEG)
        outcome: dict = {}

        def encode():
            try:
                second = np.full(24000, 0.1, dtype=np.float32)
                outcome["bytes"] = sum(
                    len(part) for part in encode_audio_streaming(iter([second] * 40), fmt="mp3")
                )
            except Exception as exc:  # noqa: BLE001 - reported below
                outcome["error"] = exc

        worker = threading.Thread(target=encode, daemon=True)
        worker.start()
        worker.join(timeout=20)
        stalled = worker.is_alive()
        for proc in started:
            if proc.poll() is None:
                proc.kill()
        worker.join(timeout=10)

        assert not stalled, "encoding stalled on an unread stderr pipe"
        assert "error" not in outcome, outcome.get("error")
        assert outcome["bytes"] > 0

    def test_failure_reports_what_ffmpeg_wrote_to_stderr(self, monkeypatch):
        _fake_ffmpeg(monkeypatch, _FAILING_FFMPEG)

        with pytest.raises(RuntimeError, match="Unknown encoder 'libmp3lame'"):
            list(encode_audio_streaming(iter([np.full(2400, 0.1, dtype=np.float32)]), fmt="mp3"))

    def test_exit_before_reading_input_reports_what_ffmpeg_wrote_to_stderr(self, monkeypatch):
        # The write to stdin is what fails here, not finish(), so it must carry the diagnosis.
        started = _fake_ffmpeg(monkeypatch, _EARLY_EXIT_FFMPEG)
        outcome: dict = {}

        def encode():
            try:
                second = np.full(24000, 0.1, dtype=np.float32)
                list(encode_audio_streaming(iter([second] * 40), fmt="mp3"))
            except Exception as exc:  # noqa: BLE001 - reported below
                outcome["error"] = exc

        worker = threading.Thread(target=encode, daemon=True)
        worker.start()
        worker.join(timeout=20)
        hung = worker.is_alive()
        for proc in started:
            if proc.poll() is None:
                proc.kill()
        worker.join(timeout=10)

        assert not hung, "encoding hung after ffmpeg exited"
        assert isinstance(outcome.get("error"), RuntimeError), repr(outcome.get("error"))
        assert "Unknown encoder 'libmp3lame'" in str(outcome["error"])

    @needs_ffmpeg
    def test_no_chunks_returns_empty(self):
        enc = StreamingFFmpegEncoder("mp3")
        enc.finish()
        # Empty input → empty or minimal output
        enc.close()


class TestEncodeAudioStreamingCompressed:
    @needs_ffmpeg
    def test_streaming_mp3_single_output(self):
        """Streaming MP3 should produce a single valid stream, not per-chunk headers."""
        chunks = [
            np.random.randn(4800).astype(np.float32) * 0.1
            for _ in range(3)
        ]
        result = b"".join(encode_audio_streaming(iter(chunks), fmt="mp3"))
        assert len(result) > 0
        # Verify it's valid MP3 by checking header
        assert result[:3] == b"ID3" or result[0] == 0xFF

    def test_streaming_pcm_per_chunk(self):
        """PCM streaming yields independent chunks (no encoding needed)."""
        chunks = [
            np.zeros(100, dtype=np.float32),
            np.ones(100, dtype=np.float32) * 0.5,
        ]
        results = list(encode_audio_streaming(iter(chunks), fmt="pcm"))
        assert len(results) == 2
        assert len(results[0]) == 200
        assert len(results[1]) == 200

    def test_streaming_wav_per_chunk(self):
        """A WAV stream is one RIFF file, not a file per chunk (see the whole-once test below)."""
        chunks = [np.zeros(100, dtype=np.float32)]
        results = list(encode_audio_streaming(iter(chunks), fmt="wav"))
        assert len(results) == 1
        assert results[0][:4] == b"RIFF"

    @pytest.mark.parametrize(
        "chunks",
        [[], [np.array([], dtype=np.float32)]],
        ids=["no-chunks", "empty-chunk"],
    )
    def test_streaming_wav_without_samples_is_a_header_only_file(self, chunks):
        # A WAV body is a RIFF file even with no samples, as the non-streamed response is.
        results = list(encode_audio_streaming(iter(chunks), fmt="wav"))

        assert results == [encode_wav(np.zeros(0, dtype=np.float32))]
        with wave.open(io.BytesIO(results[0])) as audio:
            assert audio.getnframes() == 0

    def test_streaming_wav_is_sent_whole_once_the_backend_finishes(self):
        # A WAV stream stays
        # buffered, because its one RIFF header must hold the true length. Nothing is sent
        # while the backend is still producing audio; then one complete file, byte-identical
        # to the non-streamed response. Clients that want the first bytes early use pcm.
        sample_rate = 24000
        produced = [
            np.full(2400, 0.1, dtype=np.float32),
            np.array([], dtype=np.float32),
            np.full(2400, -0.2, dtype=np.float32),
            np.full(1200, 0.3, dtype=np.float32),
        ]
        events: list[str] = []

        def backend():
            for index, chunk in enumerate(produced):
                events.append(f"backend yields chunk {index}")
                yield chunk
            events.append("backend finished")

        sent: list[bytes] = []
        for part in encode_audio_streaming(backend(), fmt="wav", sample_rate=sample_rate):
            events.append(f"client receives {len(part)} bytes")
            sent.append(part)

        # What POST /v1/audio/speech without a stream returns for the same audio.
        whole = encode_audio(iter([np.concatenate(produced)]), fmt="wav", sample_rate=sample_rate)
        assert events == [
            *(f"backend yields chunk {index}" for index in range(len(produced))),
            "backend finished",
            f"client receives {len(whole)} bytes",
        ]
        assert sent == [whole]
        with wave.open(io.BytesIO(sent[0])) as audio:
            assert audio.getframerate() == sample_rate
            assert audio.getnframes() == 6000

    @pytest.mark.parametrize(
        "chunks",
        [[], [np.array([], dtype=np.float32)]],
        ids=["no-chunks", "empty-chunk"],
    )
    def test_streaming_compressed_without_samples_is_what_ffmpeg_writes(self, monkeypatch, chunks):
        # The non-streamed response hands ffmpeg a clip with no samples and returns its file,
        # so the stream must not skip ffmpeg and send an empty body.
        started = _fake_ffmpeg(monkeypatch, _HEADER_FFMPEG)

        results = list(encode_audio_streaming(iter(chunks), fmt="mp3"))

        assert len(started) == 1
        assert b"".join(results) == b"HEADER"

    def test_streaming_empty_chunks_skipped(self):
        chunks = [
            np.array([], dtype=np.float32),
            np.zeros(100, dtype=np.float32),
            np.array([], dtype=np.float32),
        ]
        results = list(encode_audio_streaming(iter(chunks), fmt="pcm"))
        assert len(results) == 1


def _open_pipes(proc) -> list[str]:
    return [name for name in ("stdin", "stdout", "stderr") if not getattr(proc, name).closed]


def _in_thread(fn, timeout: float = 30):
    """Run fn in a worker thread, so a hung encoder fails the test instead of the run."""
    outcome: dict = {}

    def target():
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            outcome["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout)
    assert not worker.is_alive(), "the encoder hung"
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


class TestStreamingFFmpegPipes:
    """Every streamed mp3/opus/aac/flac response left ffmpeg's pipes open until the Popen
    happened to be garbage-collected (ResourceWarning: unclosed file)."""

    chunk = np.full(2400, 0.1, dtype=np.float32)

    @pytest.fixture
    def echo_ffmpeg(self, monkeypatch):
        started = _fake_ffmpeg(monkeypatch, _ECHO_FFMPEG)
        yield started
        for proc in started:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
            for name in ("stdin", "stdout", "stderr"):
                try:
                    getattr(proc, name).close()
                except OSError:
                    pass

    def test_a_finished_stream_closes_every_ffmpeg_pipe(self, echo_ffmpeg):
        out = _in_thread(lambda: b"".join(encode_audio_streaming(iter([self.chunk]), fmt="mp3")))

        assert out == pipeline.float32_to_int16(self.chunk).tobytes()
        proc = echo_ffmpeg[0]
        assert proc.returncode == 0
        assert _open_pipes(proc) == []

    def test_a_client_disconnect_mid_stream_closes_every_ffmpeg_pipe(self, echo_ffmpeg):
        # services/tts.py closes the encode generator when the client goes away.
        def disconnect_after_the_first_bytes():
            stream = encode_audio_streaming(itertools.repeat(self.chunk), fmt="mp3")
            first = b""
            while not first:
                first = next(stream)
            stream.close()
            return first

        assert _in_thread(disconnect_after_the_first_bytes)
        proc = echo_ffmpeg[0]
        assert proc.poll() is not None  # ffmpeg was stopped
        assert _open_pipes(proc) == []

    def test_a_finished_stream_leaves_nothing_for_the_finalizer(self, echo_ffmpeg):
        _in_thread(lambda: b"".join(encode_audio_streaming(iter([self.chunk]), fmt="mp3")))
        assert echo_ffmpeg[0].returncode == 0

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            echo_ffmpeg.clear()  # drops the last reference to the finished Popen
            gc.collect()

        assert [str(w.message) for w in caught if issubclass(w.category, ResourceWarning)] == []
