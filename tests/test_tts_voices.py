"""Tests for voice parsing, presets, and blending."""

import pytest

from src.tts.voices import parse_voice_spec, resolve_voice_name, OPENAI_VOICE_MAP


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

    def test_all_openai_presets_exist(self):
        for name in ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]:
            assert name in OPENAI_VOICE_MAP
