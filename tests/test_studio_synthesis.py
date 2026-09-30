"""Studio audio must use the requested rate, rather than relabel native PCM."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.services.tts import synthesize_array


@pytest.mark.parametrize("native_rate,output_rate", [
    (48000, 24000), (22050, 24000), (24000, 16000), (24000, 24000),
])
def test_studio_synthesis_preserves_duration_and_pitch(native_rate, output_rate):
    duration_s = 0.1
    samples = (0.3 * np.sin(2 * np.pi * 1000 * np.arange(native_rate // 10) / native_rate)).astype(np.float32)

    class Router:
        def get_backend(self, model):
            return SimpleNamespace(name="test", capabilities={})

        def sample_rate_for(self, model):
            return native_rate

        def synthesize(self, **kwargs):
            # Split inside the waveform to also protect chunk join continuity.
            yield samples[:123]
            yield samples[123:]

    audio = synthesize_array(
        text="Test", model="test", voice="default", speed=1.0,
        sample_rate=output_rate, tts_router=Router(),
        settings=SimpleNamespace(tts_trim_silence=False, tts_normalize_output=False),
    )

    assert len(audio) == round(output_rate * duration_s)
    assert audio.dtype == np.float32
    frequencies = np.fft.rfftfreq(len(audio), 1 / output_rate)
    dominant_frequency = frequencies[np.argmax(np.abs(np.fft.rfft(audio)))]
    assert dominant_frequency == pytest.approx(1000, abs=10)
