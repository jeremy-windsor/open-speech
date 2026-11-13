"""Unit tests for streaming module — pure functions and testable logic.

Tests the non-WebSocket, non-model-dependent parts of streaming.py:
- resample_pcm16 (resampling)
- StreamingResampler, the shared resampler it streams with (src/audio/resample.py)
- LocalAgreement2 (word stabilization)
- StreamingSession._pcm_to_wav (WAV header generation)
- SileroVAD interface contract
- Session constants and limits
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys
import threading
import time
import wave
import io
from pathlib import Path

import numpy as np
import pytest

from src.audio.resample import StreamingResampler, resample
from src.streaming import (
    LocalAgreement2,
    StreamingSession,
    resample_pcm16,
    INTERNAL_SAMPLE_RATE,
    MAX_UTTERANCE_SECONDS,
    MAX_UTTERANCE_BYTES,
    MIN_SAMPLE_RATE,
    MAX_SAMPLE_RATE,
)
from src.vad.silero import SileroVAD, VAD_CONTEXT_SIZE
from tests._watchdog import hard_watchdog


# ---------------------------------------------------------------------------
# resample_pcm16
# ---------------------------------------------------------------------------

class TestResamplePCM16:
    """Tests for PCM16 resampling function."""

    def test_same_rate_passthrough(self):
        """No resampling when rates match."""
        pcm = np.array([100, -200, 300], dtype=np.int16).tobytes()
        result = resample_pcm16(pcm, 16000, 16000)
        assert result == pcm

    def test_upsample_doubles_length(self):
        """Upsampling 16kHz → 32kHz roughly doubles sample count."""
        samples = np.arange(100, dtype=np.int16)
        pcm = samples.tobytes()
        result = resample_pcm16(pcm, 16000, 32000)
        out_samples = np.frombuffer(result, dtype=np.int16)
        assert len(out_samples) == 200  # exactly 2x

    def test_downsample_halves_length(self):
        """Downsampling 48kHz → 16kHz reduces by 3x."""
        samples = np.arange(300, dtype=np.int16)
        pcm = samples.tobytes()
        result = resample_pcm16(pcm, 48000, 16000)
        out_samples = np.frombuffer(result, dtype=np.int16)
        assert len(out_samples) == 100

    def test_empty_input(self):
        """Empty bytes returns empty bytes."""
        result = resample_pcm16(b"", 16000, 48000)
        assert result == b""

    def test_single_sample(self):
        """Single sample edge case."""
        pcm = np.array([1000], dtype=np.int16).tobytes()
        result = resample_pcm16(pcm, 16000, 32000)
        out = np.frombuffer(result, dtype=np.int16)
        assert len(out) == 2

    def test_output_is_valid_pcm16(self):
        """Output is valid int16 range."""
        samples = np.array([32767, -32768, 0, 16000, -16000], dtype=np.int16)
        result = resample_pcm16(samples.tobytes(), 8000, 16000)
        out = np.frombuffer(result, dtype=np.int16)
        assert out.min() >= -32768
        assert out.max() <= 32767

    def test_non_integer_ratio(self):
        """Non-integer ratio like 44100→16000 produces correct length."""
        samples = np.arange(441, dtype=np.int16)  # 10ms at 44.1kHz
        result = resample_pcm16(samples.tobytes(), 44100, 16000)
        out = np.frombuffer(result, dtype=np.int16)
        expected_len = int(441 * 16000 / 44100)
        assert len(out) == expected_len

    def test_rate_coprime_with_16k_resamples_a_chunk_in_bounded_time(self):
        """An accepted rate like 191999 Hz must not design a multi-million-tap filter per chunk."""
        import scipy.signal  # noqa: F401  - keep the one-time import out of the measurement

        samples = np.zeros(19199, dtype=np.int16)  # 100 ms at 191999 Hz
        started = time.perf_counter()
        result = resample_pcm16(samples.tobytes(), 191999, 16000)
        elapsed = time.perf_counter() - started

        assert abs(len(result) // 2 - 1600) <= 1
        assert elapsed < 0.1  # ~0.75 s with the exact 16000/191999 ratio

    def test_preserves_dc_offset(self):
        """Constant signal stays constant after resampling."""
        samples = np.full(100, 5000, dtype=np.int16)
        result = resample_pcm16(samples.tobytes(), 16000, 48000)
        out = np.frombuffer(result, dtype=np.int16)
        # Polyphase resampling keeps DC nearly constant (allow minor FIR ripple/rounding)
        assert np.allclose(out, 5000, atol=4)


# ---------------------------------------------------------------------------
# StreamingResampler
# ---------------------------------------------------------------------------

def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


class TestStreamingResampler:
    """Chunked resampling must equal resampling the whole stream at once."""

    @pytest.mark.parametrize(
        ("src_rate", "dst_rate"),
        [(24000, 16000), (8000, 16000), (24000, 8000), (44100, 16000), (22050, 24000), (16000, 16000)],
    )
    @pytest.mark.parametrize("chunk", [1, 128, 1000])
    def test_chunked_output_matches_one_shot_resample(self, src_rate, dst_rate, chunk):
        audio = np.random.default_rng(0).standard_normal(src_rate // 4).astype(np.float32)
        resampler = StreamingResampler(src_rate, dst_rate)

        parts = [resampler.process(audio[i:i + chunk]) for i in range(0, len(audio), chunk)]
        streamed = np.concatenate([*parts, resampler.flush()])
        whole = resample(audio, src_rate, dst_rate)

        assert streamed.dtype == whole.dtype == np.float32
        assert len(streamed) == len(whole) == -(-len(audio) * dst_rate // src_rate)
        np.testing.assert_allclose(streamed, whole, atol=1e-5)

    def test_small_chunks_do_not_drift(self):
        """128-sample frames at 24 kHz are 85 1/3 samples at 16 kHz; truncating each
        frame on its own lost 0.4% of the audio and made the session clock drift."""
        resampler = StreamingResampler(24000, 16000)
        frame = np.zeros(128, dtype=np.float32)
        emitted = 0
        for frames in range(1, 5761):  # 30.72 s
            emitted += len(resampler.process(frame))
            # Only the filter's lookahead (10 output samples) may be held back.
            assert 0 <= frames * 128 * 2 / 3 - emitted <= 16

        assert emitted + len(resampler.flush()) == 5760 * 128 * 2 // 3

    @pytest.mark.parametrize(
        ("src_rate", "dst_rate", "tone_hz"),
        [(24000, 16000, 10000), (24000, 8000, 9000), (48000, 16000, 12000)],
    )
    def test_tones_above_the_new_nyquist_are_filtered_out(self, src_rate, dst_rate, tone_hz):
        """Decimating without a low-pass filter folds the tone back into the speech band."""
        t = np.arange(src_rate) / src_rate
        tone = np.sin(2 * np.pi * tone_hz * t).astype(np.float32)
        voice = np.sin(2 * np.pi * 1000 * t).astype(np.float32)

        resampler = StreamingResampler(src_rate, dst_rate)
        streamed = np.concatenate([resampler.process(tone[i:i + 480]) for i in range(0, len(tone), 480)])

        assert _rms(resample(tone, src_rate, dst_rate)[100:-100]) < 0.01 * _rms(tone)  # -40 dB
        assert _rms(streamed[100:-100]) < 0.01 * _rms(tone)
        assert _rms(resample(voice, src_rate, dst_rate)[100:-100]) == pytest.approx(_rms(voice), rel=0.01)


# ---------------------------------------------------------------------------
# LocalAgreement2
# ---------------------------------------------------------------------------

class TestLocalAgreement2:
    """Tests for the word stabilization algorithm."""

    def test_first_input_no_confirmed(self):
        """First transcription has nothing to agree with."""
        la = LocalAgreement2()
        confirmed, pending = la.process("hello world")
        assert confirmed == []
        assert pending == ["hello", "world"]

    def test_agreement_on_prefix(self):
        """Matching prefix words become confirmed."""
        la = LocalAgreement2()
        la.process("hello world")
        confirmed, pending = la.process("hello world foo")
        assert confirmed == ["hello", "world"]
        assert pending == ["foo"]

    def test_disagreement_resets(self):
        """When words diverge, nothing new is confirmed."""
        la = LocalAgreement2()
        la.process("hello world")
        confirmed, pending = la.process("hello earth")
        assert confirmed == ["hello"]
        assert pending == ["earth"]

    def test_complete_disagreement(self):
        """Completely different text = no new confirmation."""
        la = LocalAgreement2()
        la.process("hello world")
        confirmed, pending = la.process("goodbye earth")
        assert confirmed == []
        assert pending == ["goodbye", "earth"]

    def test_case_insensitive(self):
        """Matching is case-insensitive."""
        la = LocalAgreement2()
        la.process("Hello World")
        confirmed, pending = la.process("hello world more")
        assert confirmed == ["hello", "world"]

    def test_flush_returns_remaining(self):
        """Flush emits unconfirmed words."""
        la = LocalAgreement2()
        la.process("the quick brown fox")
        la.process("the quick brown fox jumps")
        # "the quick brown fox" confirmed, "jumps" pending
        remaining = la.flush()
        assert remaining == ["jumps"]

    def test_flush_empty_when_all_confirmed(self):
        """Flush returns empty if all confirmed."""
        la = LocalAgreement2()
        la.process("hello")
        la.process("hello")
        remaining = la.flush()
        assert remaining == []

    def test_reset(self):
        """Reset clears all state."""
        la = LocalAgreement2()
        la.process("hello world")
        la.process("hello world more")
        la.reset()
        confirmed, pending = la.process("new text")
        assert confirmed == []
        assert pending == ["new", "text"]

    def test_empty_input(self):
        """Empty string produces no words."""
        la = LocalAgreement2()
        confirmed, pending = la.process("")
        assert confirmed == []
        assert pending == []

    def test_whitespace_only(self):
        """Whitespace-only input = empty."""
        la = LocalAgreement2()
        confirmed, pending = la.process("   ")
        assert confirmed == []
        assert pending == []

    def test_progressive_agreement(self):
        """Three rounds of progressive agreement."""
        la = LocalAgreement2()
        la.process("the")
        c1, p1 = la.process("the quick")
        assert c1 == ["the"]
        assert p1 == ["quick"]

        c2, p2 = la.process("the quick brown")
        assert c2 == ["quick"]
        assert p2 == ["brown"]

    def test_no_double_confirm(self):
        """Already-confirmed words aren't re-emitted."""
        la = LocalAgreement2()
        la.process("a b c")
        la.process("a b c d")  # confirms a, b, c
        confirmed, pending = la.process("a b c d e")  # confirms d
        assert confirmed == ["d"]
        assert pending == ["e"]

    def test_single_word_stability(self):
        """Single repeated word gets confirmed."""
        la = LocalAgreement2()
        la.process("hello")
        confirmed, pending = la.process("hello")
        assert confirmed == ["hello"]
        assert pending == []

    def test_shorter_input_still_confirms_prefix(self):
        """If second input is shorter but shares prefix, prefix is confirmed."""
        la = LocalAgreement2()
        la.process("hello world foo")
        confirmed, pending = la.process("hello world")
        assert confirmed == ["hello", "world"]
        assert pending == []

    def test_empty_after_nonempty(self):
        """Empty input after non-empty confirms nothing."""
        la = LocalAgreement2()
        la.process("hello world")
        confirmed, pending = la.process("")
        assert confirmed == []
        assert pending == []

    def test_flush_after_empty_input(self):
        """Flush after empty input returns nothing (previous_words cleared)."""
        la = LocalAgreement2()
        la.process("hello world")
        la.process("")
        remaining = la.flush()
        assert remaining == []

    def test_retraction_after_confirmation_is_ignored(self):
        """Once words are confirmed, retraction in next input doesn't undo them.

        This is by design — confirmed words have already been sent to the client
        as is_final=True. The algorithm sacrifices retraction for stability.
        """
        la = LocalAgreement2()
        la.process("a b")
        la.process("a b")  # confirms 'a', 'b'
        assert la.confirmed_words == ["a", "b"]
        # Now ASR retracts 'b' → 'c'. common_len=1, but confirmed=2
        confirmed, pending = la.process("a c")
        assert confirmed == []  # No new confirmations
        assert pending == []    # 'c' is lost — by design
        assert la.confirmed_words == ["a", "b"]  # Stale but intentional

    def test_punctuation_affects_matching(self):
        """Punctuation in ASR output affects word matching."""
        la = LocalAgreement2()
        la.process("hello world")
        confirmed, pending = la.process("hello, world")
        # "hello" != "hello," so only no common prefix
        assert confirmed == []
        assert pending == ["hello,", "world"]

    def test_confirmed_preserves_current_casing(self):
        """Confirmed words use the casing from the current (second) input."""
        la = LocalAgreement2()
        la.process("Hello World")
        confirmed, pending = la.process("hello world more")
        # Case-insensitive match, but stores current casing
        assert confirmed == ["hello", "world"]


# ---------------------------------------------------------------------------
# StreamingSession._pcm_to_wav
# ---------------------------------------------------------------------------

class TestPcmToWav:
    """Tests for WAV header generation."""

    def test_valid_wav_header(self):
        """Generated WAV is parseable by the wave module."""
        pcm = np.zeros(1600, dtype=np.int16).tobytes()
        wav_data = StreamingSession._pcm_to_wav(pcm, 16000)

        with wave.open(io.BytesIO(wav_data), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 1600

    def test_different_sample_rates(self):
        """WAV encodes the correct sample rate."""
        for rate in [8000, 16000, 22050, 44100, 48000]:
            pcm = np.zeros(100, dtype=np.int16).tobytes()
            wav_data = StreamingSession._pcm_to_wav(pcm, rate)
            with wave.open(io.BytesIO(wav_data), "rb") as wf:
                assert wf.getframerate() == rate

    def test_empty_audio(self):
        """Empty PCM produces valid (empty) WAV."""
        wav_data = StreamingSession._pcm_to_wav(b"", 16000)
        with wave.open(io.BytesIO(wav_data), "rb") as wf:
            assert wf.getnframes() == 0

    def test_round_trip_data(self):
        """PCM data survives the WAV round-trip."""
        original = np.array([100, -200, 32767, -32768, 0], dtype=np.int16)
        wav_data = StreamingSession._pcm_to_wav(original.tobytes(), 16000)
        with wave.open(io.BytesIO(wav_data), "rb") as wf:
            frames = wf.readframes(5)
        recovered = np.frombuffer(frames, dtype=np.int16)
        np.testing.assert_array_equal(original, recovered)


# ---------------------------------------------------------------------------
# Constants sanity checks
# ---------------------------------------------------------------------------

class TestConstants:
    """Verify streaming constants are sane."""

    def test_internal_sample_rate(self):
        assert INTERNAL_SAMPLE_RATE == 16000

    def test_max_utterance_bytes_matches_seconds(self):
        expected = MAX_UTTERANCE_SECONDS * INTERNAL_SAMPLE_RATE * 2
        assert MAX_UTTERANCE_BYTES == expected

    def test_max_utterance_reasonable(self):
        assert 10 <= MAX_UTTERANCE_SECONDS <= 120

    def test_sample_rate_range(self):
        assert MIN_SAMPLE_RATE == 8000
        assert MAX_SAMPLE_RATE == 192000


# ---------------------------------------------------------------------------
# SileroVAD interface (mocked ONNX session)
# ---------------------------------------------------------------------------

class MockOrtSession:
    """Mock ONNX session matching real Silero VAD return signature.
    
    Real ort.InferenceSession.run() returns a flat list of output arrays.
    Silero VAD returns [output_prob, new_state] as a 2-element list.
    The SileroVAD wrapper unpacks: out, self._state = session.run(...)
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def run(self, output_names, inputs):
        state = inputs["state"]
        # Real signature: returns [prob_array, state_array] as a flat list
        # prob_array shape is (1, 1) so out[0][0] yields a scalar
        return [np.array([[self.prob]], dtype=np.float32), state]


class TestSileroVAD:
    """Tests for the Silero VAD wrapper (with mock session)."""

    def test_returns_probability(self):
        vad = SileroVAD(MockOrtSession(prob=0.9))
        audio = np.zeros(512, dtype=np.float32)
        prob = vad(audio)
        assert prob == pytest.approx(0.9)

    def test_empty_audio_returns_zero(self):
        vad = SileroVAD(MockOrtSession(prob=0.9))
        prob = vad(np.array([], dtype=np.float32))
        assert prob == 0.0

    def test_short_audio_under_window(self):
        """Audio shorter than 512 samples returns 0."""
        vad = SileroVAD(MockOrtSession(prob=0.9))
        audio = np.zeros(100, dtype=np.float32)
        prob = vad(audio)
        assert prob == 0.0

    def test_multiple_windows_returns_max(self):
        """Multiple 512-sample windows return max probability."""
        # Session returns different probs based on call count
        class IncreasingSession:
            def __init__(self):
                self.call = 0
            def run(self, output_names, inputs):
                self.call += 1
                state = inputs["state"]
                prob = 0.1 * self.call
                return [np.array([[prob]], dtype=np.float32), state]

        vad = SileroVAD(IncreasingSession())
        audio = np.zeros(1536, dtype=np.float32)  # 3 windows
        prob = vad(audio)
        assert prob == pytest.approx(0.3)  # max of 0.1, 0.2, 0.3

    def test_reset_clears_state(self):
        vad = SileroVAD(MockOrtSession())
        vad._state = np.ones((2, 1, 128), dtype=np.float32)
        vad._context = np.ones(VAD_CONTEXT_SIZE, dtype=np.float32)
        vad.reset()
        assert np.all(vad._state == 0)
        assert np.all(vad._context == 0)


# ---------------------------------------------------------------------------
# A large frame must not stall the event loop
# ---------------------------------------------------------------------------

# uvicorn's default ws_max_size, so the largest binary frame a client can send
LARGE_FRAME_BYTES = 16 * 1024 * 1024
# 65 s of 16 kHz audio, about 0.4 s of Silero time on a desktop CPU
MAX_VAD_WINDOWS_PER_LOOP_TURN = 2048


@pytest.fixture
def watchdog(request):
    """End the run if the test hangs (there is no pytest-timeout), naming it and printing every thread's stack."""
    with hard_watchdog(request, 300):
        yield


class _LoopTurns:
    """A task that counts its own turns on the event loop."""

    def __init__(self):
        self.count = 0
        self.stop = False

    async def run(self):
        while not self.stop:
            self.count += 1
            await asyncio.sleep(0)


class _WhereVADRuns:
    """Silence for every window; records whether it ran on the loop thread, and at which turn."""

    def __init__(self, turns: _LoopTurns):
        self.turns = turns
        self.loop_thread = threading.get_ident()
        self.calls: list[tuple[bool, int]] = []

    def __call__(self, _window) -> float:
        self.calls.append((threading.get_ident() == self.loop_thread, self.turns.count))
        return 0.0

    def longest_run_on_the_loop(self) -> int:
        """The most windows scored on the loop thread while no other task got a turn."""
        longest = run = 0
        last_turn = None
        for on_loop, turn in self.calls:
            if not on_loop:
                run, last_turn = 0, None
                continue
            run = run + 1 if turn == last_turn else 1
            last_turn = turn
            longest = max(longest, run)
        return longest


class _SentEvents:
    def __init__(self):
        self.sent = []

    async def send_text(self, text):
        self.sent.append(text)


async def test_large_silent_frame_scores_its_vad_windows_off_the_event_loop(watchdog):
    """One 16 MiB frame is 16k Silero windows (~3 s): all of it ran on the event loop,
    so every request and stream stalled until the frame was done."""
    session = StreamingSession(
        ws=_SentEvents(), model="m", language=None, sample_rate=16000,
        interim_results=True, endpointing_ms=300, vad_enabled=True,
    )
    turns = _LoopTurns()
    vad = _WhereVADRuns(turns)
    session.vad_state = vad
    counter = asyncio.create_task(turns.run())
    await asyncio.sleep(0)

    await asyncio.wait_for(session._handle_audio(bytes(LARGE_FRAME_BYTES)), timeout=240)
    turns.stop = True
    await counter

    whole_chunks = LARGE_FRAME_BYTES // 2 // session.chunk_samples * session.chunk_samples
    assert len(vad.calls) == whole_chunks // 512  # every window was scored
    assert vad.longest_run_on_the_loop() <= MAX_VAD_WINDOWS_PER_LOOP_TURN, (
        f"{vad.longest_run_on_the_loop()} of {len(vad.calls)} VAD windows ran on the event loop "
        "with no other task getting a turn"
    )


# The watchdog above, tests/test_realtime.py's and typing_client's in
# tests/test_live_tts.py called faulthandler.dump_traceback_later(..., exit=True, file=sys.stderr). During a test,
# sys.stderr and fd 2 are pytest's capture file, so a hang printed nothing: the run ended with no summary and no test
# named. They also replaced the one faulthandler timer, the per-test one of faulthandler_timeout. Now all
# three use tests/_watchdog.py. A child pytest, with pytest's default capture on, runs a tiny hung test under it.

_HUNG_CHILD = '''
import sys
import time

import pytest

sys.path.insert(0, {repo!r})
from tests._watchdog import hard_watchdog


@pytest.fixture
def watchdog(request):
    with hard_watchdog(request, {seconds!r}):
        yield


def test_a_quick_one_before():
    pass


def test_hangs_here_under_the_watchdog(watchdog):
    time.sleep(60)


def test_a_quick_one_after():
    pass
'''


def test_a_hung_test_under_the_watchdog_is_named_with_its_stack_on_the_real_stderr(tmp_path):
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "test_hung_child.py").write_text(
        _HUNG_CHILD.format(repo=str(Path(__file__).resolve().parents[1]), seconds=2), encoding="utf-8",
    )
    env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    env.pop("PYTEST_ADDOPTS", None)
    started = time.monotonic()
    child = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--capture=fd",
         "-o", "faulthandler_timeout=1", "test_hung_child.py"],
        cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
    )
    seconds = time.monotonic() - started
    err = child.stderr
    shown = f"exit {child.returncode} after {seconds:.1f} s; stdout {child.stdout!r}; stderr {err!r}"

    assert child.returncode == 1 and seconds < 30, shown  # the run still ends, long before the sleep would
    report = err.find("test_hung_child.py::test_hangs_here_under_the_watchdog")
    assert report >= 0, f"nothing on stderr names the hung test: {shown}"
    assert re.search(r'File ".*test_hung_child\.py", line \d+ in test_hangs_here_under_the_watchdog', err[report:]), (
        f"no stack of the hung test after the line naming it: {shown}"
    )
    # pytest's own per-test timer (faulthandler_timeout in pyproject.toml) is still armed: it printed the stacks first, at 1 s
    assert "Timeout (0:00:01)!" in err[:report], shown
