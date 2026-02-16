"""Tests for Kokoro backend — lang detection, voice list, blend verification."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")

from src.tts.backends.kokoro import (
    KokoroBackend,
    lang_code_from_voice_id,
    ALL_KOKORO_VOICES,
    VOICE_PREFIX_TO_LANG,
    LANG_CODE_TO_LANGUAGE,
)
from src.tts.backends.base import VoiceInfo
from src.tts.voices import parse_voice_spec


class TestLangCodeFromVoiceId:
    def test_american_english(self):
        assert lang_code_from_voice_id("af_heart") == "a"
        assert lang_code_from_voice_id("am_adam") == "a"

    def test_british_english(self):
        assert lang_code_from_voice_id("bf_emma") == "b"
        assert lang_code_from_voice_id("bm_george") == "b"

    def test_spanish(self):
        assert lang_code_from_voice_id("ef_dora") == "e"
        assert lang_code_from_voice_id("em_alex") == "e"

    def test_french(self):
        assert lang_code_from_voice_id("ff_siwis") == "f"

    def test_hindi(self):
        assert lang_code_from_voice_id("hf_alpha") == "h"
        assert lang_code_from_voice_id("hm_omega") == "h"

    def test_italian(self):
        assert lang_code_from_voice_id("if_sara") == "i"
        assert lang_code_from_voice_id("im_nicola") == "i"

    def test_japanese(self):
        assert lang_code_from_voice_id("jf_alpha") == "j"
        assert lang_code_from_voice_id("jm_kumo") == "j"

    def test_portuguese(self):
        assert lang_code_from_voice_id("pf_dora") == "p"

    def test_mandarin(self):
        assert lang_code_from_voice_id("zf_xiaobei") == "z"
        assert lang_code_from_voice_id("zm_yunxi") == "z"

    def test_unknown_defaults_to_a(self):
        assert lang_code_from_voice_id("unknown_voice") == "a"
        assert lang_code_from_voice_id("") == "a"
        assert lang_code_from_voice_id("x") == "a"


class TestVoiceList:
    def test_all_voices_complete(self):
        """Static voice list should have voices for all languages."""
        langs = {v["lang"] for v in ALL_KOKORO_VOICES}
        expected = set(VOICE_PREFIX_TO_LANG.values())
        assert langs == expected, f"Missing languages: {expected - langs}"

    def test_all_voices_have_valid_fields(self):
        for v in ALL_KOKORO_VOICES:
            assert v["id"], "Empty voice ID"
            assert v["name"], "Empty voice name"
            assert v["lang"] in VOICE_PREFIX_TO_LANG.values()
            assert v["gender"] in ("male", "female")

    def test_voice_ids_match_lang_prefix(self):
        """Voice IDs should start with their declared language prefix."""
        for v in ALL_KOKORO_VOICES:
            prefix = v["id"][0]
            assert VOICE_PREFIX_TO_LANG.get(prefix) == v["lang"], \
                f"Voice {v['id']} prefix '{prefix}' doesn't match lang '{v['lang']}'"

    def test_list_voices_returns_all(self):
        backend = KokoroBackend(device="cpu")
        voices = backend.list_voices()
        assert len(voices) >= 40  # Should have ~48 voices
        assert all(isinstance(v, VoiceInfo) for v in voices)

    def test_list_voices_covers_all_languages(self):
        backend = KokoroBackend(device="cpu")
        voices = backend.list_voices()
        languages = {v.language for v in voices}
        expected_langs = set(LANG_CODE_TO_LANGUAGE.values())
        assert languages == expected_langs


class TestBlendVoices:
    @needs_torch
    def test_blend_calls_load_voice(self):
        """Verify _blend_voices uses KPipeline.load_voice() correctly."""
        import torch

        backend = KokoroBackend(device="cpu")
        mock_pipeline = MagicMock()
        # load_voice returns a tensor
        mock_pipeline.load_voice.return_value = torch.ones(256)
        backend._pipeline = mock_pipeline

        spec = parse_voice_spec("af_bella(2)+af_sky(1)")
        result = backend._blend_voices(spec)

        # Should have called load_voice for each component
        assert mock_pipeline.load_voice.call_count == 2
        mock_pipeline.load_voice.assert_any_call("af_bella")
        mock_pipeline.load_voice.assert_any_call("af_sky")

        # Result should be a tensor (weighted blend)
        assert isinstance(result, torch.Tensor)
        assert result.shape == torch.Size([256])

    @needs_torch
    def test_blend_weights_applied(self):
        """Verify weights are properly applied in blending."""
        import torch

        backend = KokoroBackend(device="cpu")
        mock_pipeline = MagicMock()

        t1 = torch.ones(10) * 3.0
        t2 = torch.ones(10) * 6.0
        mock_pipeline.load_voice.side_effect = [t1, t2]
        backend._pipeline = mock_pipeline

        # 50/50 blend
        spec = parse_voice_spec("af_bella+af_sky")
        result = backend._blend_voices(spec)

        # Should be average: (3*0.5 + 6*0.5) = 4.5
        expected = 4.5
        assert torch.allclose(result, torch.ones(10) * expected)


class TestLangCodeSwitch:
    def test_ensure_loaded_switches_lang(self):
        """Pipeline should reload when lang code changes."""
        backend = KokoroBackend(device="cpu")

        mock_kpipeline = MagicMock()
        mock_module = MagicMock()
        mock_module.KPipeline = mock_kpipeline

        with patch.dict("sys.modules", {"kokoro": mock_module}):
            backend._ensure_loaded(lang_code="a")
            assert backend._current_lang_code == "a"
            assert mock_kpipeline.call_count == 1

            # Same lang — no reload
            backend._ensure_loaded(lang_code="a")
            assert mock_kpipeline.call_count == 1

            # Different lang — reload
            backend._ensure_loaded(lang_code="j")
            assert backend._current_lang_code == "j"
            assert mock_kpipeline.call_count == 2
            mock_kpipeline.assert_called_with(lang_code="j", device="cpu")
