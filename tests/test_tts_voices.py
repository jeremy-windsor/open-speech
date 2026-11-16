"""Tests for voice parsing, presets, and blending."""

import pytest

from src.tts.voices import OPENAI_VOICE_MAP, parse_voice_spec, resolve_voice_name


class TestResolveVoiceName:
    def test_openai_aliases(self):
        assert resolve_voice_name("alloy") == "af_heart"
        assert resolve_voice_name("echo") == "am_adam"
        assert resolve_voice_name("nova") == "af_nova"

    def test_passthrough(self):
        assert resolve_voice_name("af_bella") == "af_bella"
        assert resolve_voice_name("custom_voice") == "custom_voice"


class TestParseVoiceSpec:
    def test_single_voice(self):
        spec = parse_voice_spec("af_bella")
        assert len(spec.components) == 1
        assert spec.components[0].voice_id == "af_bella"
        assert spec.components[0].weight == 1.0
        assert not spec.is_blend

    def test_openai_alias(self):
        spec = parse_voice_spec("alloy")
        assert spec.primary_id == "af_heart"

    def test_blend_equal_weights(self):
        spec = parse_voice_spec("af_bella+af_sky")
        assert len(spec.components) == 2
        assert spec.components[0].voice_id == "af_bella"
        assert spec.components[1].voice_id == "af_sky"
        assert spec.is_blend
        weights = spec.normalized_weights()
        assert weights == [0.5, 0.5]

    def test_blend_with_weights(self):
        spec = parse_voice_spec("af_bella(2)+af_sky(1)")
        assert len(spec.components) == 2
        assert spec.components[0].weight == 2.0
        assert spec.components[1].weight == 1.0
        weights = spec.normalized_weights()
        assert abs(weights[0] - 2/3) < 1e-6
        assert abs(weights[1] - 1/3) < 1e-6

    def test_three_voices(self):
        spec = parse_voice_spec("af_bella(3)+af_sky(2)+am_adam(1)")
        assert len(spec.components) == 3
        weights = spec.normalized_weights()
        assert abs(sum(weights) - 1.0) < 1e-6
        assert abs(weights[0] - 0.5) < 1e-6

    def test_invalid_spec(self):
        with pytest.raises(ValueError):
            parse_voice_spec("!!!invalid!!!")

    def test_float_weights(self):
        spec = parse_voice_spec("af_bella(1.5)+af_sky(0.5)")
        assert spec.components[0].weight == 1.5
        assert spec.components[1].weight == 0.5

    def test_a_weight_too_large_for_a_float_is_refused(self):
        # float() of 400 digits is inf, which normalized to [nan, 0.0] and
        # blended a NaN voice (a 500 from torch, or silence) instead of a 400.
        with pytest.raises(ValueError, match="Invalid voice spec component"):
            parse_voice_spec("af_bella(" + "9" * 400 + ")+af_sky")
        with pytest.raises(ValueError, match="Invalid voice spec component"):
            parse_voice_spec("af_bella+af_sky(" + "9" * 309 + ".5)")

    def test_weights_whose_sum_overflows_are_still_normalized(self):
        # Two finite weights near 1e308 summed to inf, so both normalized to 0.0
        # and the blend was an all-zero voice. They are an even mix.
        big = "1" + "0" * 308
        weights = parse_voice_spec(f"af_bella({big})+af_sky({big})").normalized_weights()
        assert weights == [0.5, 0.5]
        weights = parse_voice_spec(f"af_bella({big})+af_sky(1)+am_adam({big})").normalized_weights()
        assert weights[0] == weights[2] == pytest.approx(0.5)
        assert 0.0 <= weights[1] < 1e-300
        assert sum(weights) == pytest.approx(1.0)

    def test_ordinary_weights_normalize_exactly_as_before(self):
        spec = parse_voice_spec("af_bella(2)+af_sky(1)+am_adam(0.5)")
        assert spec.normalized_weights() == [2 / 3.5, 1 / 3.5, 0.5 / 3.5]
        assert parse_voice_spec("af_bella(0)+af_sky(0)").normalized_weights() == [0.5, 0.5]

    def test_all_openai_presets_exist(self):
        # A module-level test below was once inserted above this method,
        # which made it a nested function pytest never collected. It belongs to TestParseVoiceSpec.
        for name in ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]:
            assert name in OPENAI_VOICE_MAP


def test_kokoro_refuses_an_unusable_blend_weight_before_synthesis():
    # validate_voice is what /v1/audio/speech, the Live Reader, named voices and
    # presets call, and a ValueError there is the API's 400.
    from src.tts.backends.kokoro import KokoroBackend

    with pytest.raises(ValueError, match="Invalid voice spec component"):
        KokoroBackend(device="cpu").validate_voice("af_bella(" + "9" * 400 + ")+af_sky")


# The voices OpenAI added for gpt-4o-mini-tts and the GA Realtime API, by perceived gender.
OPENAI_GA_VOICES = {
    "ash": "male", "ballad": "male", "cedar": "male", "verse": "male",
    "coral": "female", "marin": "female", "sage": "female",
}


def test_every_openai_voice_maps_to_a_known_kokoro_voice():
    from src.tts.backends.kokoro import ALL_KOKORO_VOICES

    genders = {voice["id"]: voice["gender"] for voice in ALL_KOKORO_VOICES}

    assert {"alloy", "echo", "fable", "onyx", "nova", "shimmer", *OPENAI_GA_VOICES} <= set(OPENAI_VOICE_MAP)
    assert {name: voice for name, voice in OPENAI_VOICE_MAP.items() if voice not in genders} == {}
    assert {name: genders[OPENAI_VOICE_MAP[name]] for name in OPENAI_GA_VOICES} == OPENAI_GA_VOICES
    # OpenAI voices speak English unless told otherwise, so they map to Kokoro's English ones.
    assert all(OPENAI_VOICE_MAP[name][:2] in {"af", "am", "bf", "bm"} for name in OPENAI_GA_VOICES)


@pytest.mark.parametrize("voice", sorted(OPENAI_GA_VOICES))
def test_kokoro_accepts_the_openai_ga_voices(monkeypatch, voice):
    # GA SDK clients that picked one of these got "Unknown Kokoro voice".
    from src.tts.backends import kokoro as kokoro_module
    from src.tts.router import TTSRouter

    # An unknown voice falls back to scanning the local Hugging Face cache.
    monkeypatch.setattr(kokoro_module, "_discover_voices_from_package", lambda: None)
    monkeypatch.setattr("src.tts.router._discover_backends", lambda *_args, **_kwargs: {})
    router = TTSRouter(device="cpu", external_providers="")
    router.register_backend("kokoro", kokoro_module.KokoroBackend(device="cpu"))

    router.validate_voice("kokoro", voice)
    assert parse_voice_spec(voice).primary_id == OPENAI_VOICE_MAP[voice]
