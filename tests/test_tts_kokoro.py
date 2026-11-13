"""Tests for Kokoro backend — lang detection, voice list, blend verification."""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
from importlib.util import find_spec
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.tts.backends.kokoro import (
    KPIPELINE_MAX_PHONEMES,
    split_for_kpipeline,
    KokoroBackend,
    lang_code_from_voice_id,
    ALL_KOKORO_VOICES,
    VOICE_PREFIX_TO_LANG,
    LANG_CODE_TO_LANGUAGE,
)
from src.tts.backends.base import VoiceInfo
from src.tts.voices import parse_voice_spec

HAS_TORCH = find_spec("torch") is not None
needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


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


class TestVoiceValidation:
    def test_openai_alias_and_will_blend_are_valid(self):
        backend = KokoroBackend(device="cpu")

        backend.validate_voice("alloy")
        backend.validate_voice("am_puck(1)+am_liam(1)+am_onyx(0.5)")

    def test_unknown_voice_is_rejected(self):
        backend = KokoroBackend(device="cpu")

        with patch("src.tts.backends.kokoro._discover_voices_from_package", return_value=None):
            with pytest.raises(ValueError, match="Unknown Kokoro voice"):
                backend.validate_voice("af_typo")

    def test_runtime_discovered_custom_voice_is_valid(self):
        backend = KokoroBackend(device="cpu")
        custom = VoiceInfo(id="af_custom", name="Custom", language="en-us", gender="female")

        with patch("src.tts.backends.kokoro._discover_voices_from_package", return_value=[custom]):
            backend.validate_voice("af_custom")

    def test_runtime_voice_discovery_refreshes_after_ttl(self):
        backend = KokoroBackend(device="cpu")
        custom = VoiceInfo(id="af_custom", name="Custom", language="en-us", gender="female")

        with (
            patch("src.tts.backends.kokoro.time.monotonic", side_effect=[10.0, 20.0, 71.0]),
            patch(
                "src.tts.backends.kokoro._discover_voices_from_package",
                side_effect=[[], [custom]],
            ) as discover,
        ):
            with pytest.raises(ValueError, match="Unknown Kokoro voice"):
                backend.validate_voice("af_custom")
            with pytest.raises(ValueError, match="Unknown Kokoro voice"):
                backend.validate_voice("af_custom")
            backend.validate_voice("af_custom")

        assert discover.call_count == 2


def _cache_repo(repo_id: str, revisions: list[list]):
    return SimpleNamespace(
        repo_id=repo_id,
        revisions=[
            SimpleNamespace(files=[SimpleNamespace(file_path=path) for path in files])
            for files in revisions
        ],
    )


def _scan_cache(*repos):
    return patch("huggingface_hub.scan_cache_dir", return_value=SimpleNamespace(repos=list(repos)))


STATIC_IDS = [voice["id"] for voice in ALL_KOKORO_VOICES]
CACHE_ROOT = "/hf/models--hexgrad--Kokoro-82M/snapshots"


class TestVoiceDiscovery:
    @pytest.mark.parametrize(
        "revisions",
        [
            # A moved main branch leaves the same downloaded subset under two snapshots.
            [STATIC_IDS[:30], STATIC_IDS[:30]],
            # As many voices as the static list, but not the same ones.
            [STATIC_IDS[:-2] + ["pm_alex", "pm_santa"]],
        ],
        ids=["two_revisions_of_a_subset", "same_count_different_voices"],
    )
    def test_voice_list_stays_unique_and_complete_when_discovery_is_partial(self, revisions):
        repo = _cache_repo(
            "hexgrad/Kokoro-82M",
            [
                [PurePosixPath(f"{CACHE_ROOT}/rev{index}/voices/{voice}.pt") for voice in voices]
                for index, voices in enumerate(revisions)
            ],
        )

        with _scan_cache(repo):
            ids = [voice.id for voice in KokoroBackend(device="cpu").list_voices()]

        assert len(ids) == len(set(ids))
        assert set(STATIC_IDS) <= set(ids)

    def test_discovery_ignores_voices_cached_for_other_kokoro_repositories(self):
        other_root = "/hf/models--hexgrad--Kokoro-82M-v1.1-zh/snapshots/rev/voices"
        other = _cache_repo(
            "hexgrad/Kokoro-82M-v1.1-zh",
            [[PurePosixPath(f"{other_root}/zf_{index:03d}.pt") for index in range(1, 104)]],
        )
        own = _cache_repo(
            "hexgrad/Kokoro-82M", [[PurePosixPath(f"{CACHE_ROOT}/rev/voices/af_heart.pt")]]
        )
        backend = KokoroBackend(device="cpu")

        with _scan_cache(other, own):
            ids = {voice.id for voice in backend.list_voices()}
            assert "af_heart" in ids
            assert "zf_001" not in ids
            with pytest.raises(ValueError, match="Unknown Kokoro voice"):
                backend.validate_voice("zf_001")

    def test_discovery_finds_voices_in_windows_cache_paths(self):
        root = r"C:\Users\me\.cache\huggingface\hub\models--hexgrad--Kokoro-82M\snapshots\rev"
        repo = _cache_repo(
            "hexgrad/Kokoro-82M", [[PureWindowsPath(rf"{root}\voices\af_custom.pt")]]
        )
        backend = KokoroBackend(device="cpu")

        with _scan_cache(repo):
            backend.validate_voice("af_custom")


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
    def test_iso_language_is_normalized_for_kpipeline(self):
        """Public API language codes must be adapted to Kokoro's one-letter codes."""
        backend = KokoroBackend(device="cpu")
        mock_kpipeline = MagicMock()
        mock_module = MagicMock()
        mock_module.KPipeline = mock_kpipeline

        with patch.dict("sys.modules", {"kokoro": mock_module}):
            list(backend.synthesize("Bonjour", "ff_siwis", lang_code="fr"))

        mock_kpipeline.assert_called_once_with(
            lang_code="f",
            device="cpu",
            repo_id="hexgrad/Kokoro-82M",
        )

    def test_ensure_loaded_switches_lang(self):
        """A language switch selects that language's pipeline."""
        backend = KokoroBackend(device="cpu")

        mock_kpipeline = MagicMock()
        mock_module = MagicMock()
        mock_module.KPipeline = mock_kpipeline

        with patch.dict("sys.modules", {"kokoro": mock_module}):
            backend._ensure_loaded(lang_code="a")
            assert backend._current_lang_code == "a"
            assert mock_kpipeline.call_count == 1

            # Same lang — no new pipeline
            backend._ensure_loaded(lang_code="a")
            assert mock_kpipeline.call_count == 1

            # Different lang — a new pipeline on the loaded model
            backend._ensure_loaded(lang_code="j")
            assert backend._current_lang_code == "j"
            assert mock_kpipeline.call_count == 2
            mock_kpipeline.assert_called_with(
                lang_code="j",
                repo_id="hexgrad/Kokoro-82M",
                model=mock_kpipeline.return_value.model,
            )


def _kokoro_package():
    """The kokoro 0.9 package at its import boundary.

    KPipeline maps its aliases and then asserts the code is one it knows. model=True loads
    a new copy of the weights; a KModel passed in is used as is.
    """

    class KPipeline:
        LANG_CODES = {"a", "b", "e", "f", "h", "i", "p", "j", "z"}
        ALIASES = {
            "en-us": "a", "en-gb": "b", "es": "e", "fr-fr": "f", "hi": "h",
            "it": "i", "pt-br": "p", "ja": "j", "zh": "z",
        }
        created: list = []
        weight_loads = 0

        def __init__(self, lang_code, repo_id=None, model=True, trf=False, en_callable=None, device=None):
            lang_code = lang_code.lower()
            lang_code = self.ALIASES.get(lang_code, lang_code)
            assert lang_code in self.LANG_CODES, (lang_code, self.LANG_CODES)
            self.lang_code = lang_code
            if model is True:
                KPipeline.weight_loads += 1
                self.model = SimpleNamespace(weights=KPipeline.weight_loads)
            else:
                self.model = model
            KPipeline.created.append(self)

        def g2p(self, text):
            return text, None

        def __call__(self, text, voice=None, speed=1.0):
            yield text, self.lang_code, [0.1] * 240

    return SimpleNamespace(KPipeline=KPipeline)


class TestLanguagePipelines:
    def test_language_switches_reuse_the_loaded_model(self):
        # Switching between af_ and bf_ voices (or any languages) must not reload the weights:
        # each language only needs its own text front end on the one loaded KModel.
        package = _kokoro_package()
        backend = KokoroBackend(device="cpu")

        with patch.dict("sys.modules", {"kokoro": package}):
            for voice in ("af_heart", "bf_emma", "af_heart", "jf_alpha", "bf_emma"):
                list(backend.synthesize("Hello", voice))
                assert backend._pipeline.lang_code == voice[0]

        pipelines = package.KPipeline.created
        assert package.KPipeline.weight_loads == 1
        assert [pipeline.lang_code for pipeline in pipelines] == ["a", "b", "j"]
        assert all(pipeline.model is pipelines[0].model for pipeline in pipelines)

    def test_unload_drops_every_language_pipeline(self):
        package = _kokoro_package()
        backend = KokoroBackend(device="cpu")

        with patch.dict("sys.modules", {"kokoro": package}):
            list(backend.synthesize("Hello", "af_heart"))
            list(backend.synthesize("Hello", "bf_emma"))
            backend.unload_model("kokoro")
            assert backend.loaded_models() == []
            list(backend.synthesize("Hello", "bf_emma"))

        assert package.KPipeline.weight_loads == 2
        assert backend._pipelines == {"b": package.KPipeline.created[-1]}

    @pytest.mark.parametrize(
        ("language", "voice", "expected"),
        [
            ("zh-CN", "zf_xiaobei", "z"),
            ("es-MX", "ef_dora", "e"),
            ("ja-JP", "jf_alpha", "j"),
            ("fr-FR", "ff_siwis", "f"),
            ("pt-PT", "pf_dora", "p"),
            ("hi_IN", "hf_alpha", "h"),
            # English keeps the accent of the voice, as plain "en" does.
            ("en-AU", "af_heart", "a"),
            ("en-IE", "bf_emma", "b"),
            ("EN-GB", "af_heart", "b"),
        ],
    )
    def test_regional_language_tags_reach_kpipeline_as_one_letter_codes(
        self, language, voice, expected
    ):
        package = _kokoro_package()
        backend = KokoroBackend(device="cpu")

        with patch.dict("sys.modules", {"kokoro": package}):
            list(backend.synthesize("Hello", voice, lang_code=language))

        assert backend._current_lang_code == expected
        assert [pipeline.lang_code for pipeline in package.KPipeline.created] == [expected]

    def test_alias_and_one_letter_code_share_one_pipeline(self):
        package = _kokoro_package()
        backend = KokoroBackend(device="cpu")

        with patch.dict("sys.modules", {"kokoro": package}):
            list(backend.synthesize("Bonjour", "ff_siwis", lang_code="fr-FR"))
            list(backend.synthesize("Bonjour", "ff_siwis", lang_code="fr"))
            list(backend.synthesize("Bonjour", "ff_siwis"))

        assert len(package.KPipeline.created) == 1

    @pytest.mark.parametrize("language", ["ko", "de-DE", "xx"])
    def test_unsupported_language_is_a_value_error_naming_the_supported_ones(self, language):
        # TTSSpeechRequest.language documents "ko" as an example hint; a language Kokoro has no
        # pipeline for is a client error (the API maps ValueError to 400), not KPipeline's assert.
        package = _kokoro_package()
        backend = KokoroBackend(device="cpu")

        with patch.dict("sys.modules", {"kokoro": package}):
            with pytest.raises(ValueError) as caught:
                list(backend.synthesize("Hello", "af_heart", lang_code=language))

        message = str(caught.value)
        assert f"'{language}'" in message
        assert "Supported: en, en-gb, en-us, es, fr, hi, it, ja, pt, pt-br, zh" in message
        assert package.KPipeline.created == []

    @pytest.mark.parametrize("language", [None, "", "  "])
    def test_missing_language_hint_uses_the_voice_language(self, language):
        package = _kokoro_package()
        backend = KokoroBackend(device="cpu")

        with patch.dict("sys.modules", {"kokoro": package}):
            list(backend.synthesize("Hola", "ef_dora", lang_code=language))

        assert backend._current_lang_code == "e"


# Kokoro 0.9.4's KPipeline splits non-English text only at newlines and
# ASCII .!?, and cuts each chunk's phonemes at 510 (`ps = ps[:510]`), so a long Mandarin, Japanese or
# Hindi paragraph, or a long Spanish sentence, stopped part-way with 200. The text sent is the text
# spoken (README: `input` up to TTS_MAX_INPUT_LENGTH characters).

def _stub_g2p(text):
    """A deterministic G2P: 3 phonemes per Han or kana character, 1 per letter, a space per space."""
    out = []
    for ch in text:
        if "぀" <= ch <= "ヿ" or "一" <= ch <= "鿿":
            out.append("abc")
        elif ch.isalpha() or "ऀ" <= ch <= "ॿ":
            out.append("a")
        elif ch.isspace():
            out.append(" ")
    return "".join(out).strip(), None


def _kokoro_094_package():
    """kokoro 0.9.4's KPipeline.__call__ at its model boundary, its non-English branch line for line.

    The fake model "speaks" the phonemes it is given; `spoken` keeps them per chunk.
    """

    class KPipeline:
        created: list = []

        def __init__(self, lang_code, repo_id=None, model=True, trf=False, en_callable=None, device=None):
            self.lang_code = lang_code
            self.model = model if model is not True else SimpleNamespace()
            self.g2p = _stub_g2p
            self.calls: list = []
            self.spoken: list[str] = []
            self.truncated: list[int] = []
            KPipeline.created.append(self)

        def __call__(self, text, voice=None, speed=1, split_pattern=r"\n+"):
            self.calls.append(text)
            if isinstance(text, str):
                text = re.split(split_pattern, text.strip()) if split_pattern else [text]
            for graphemes in text:
                if not graphemes.strip():
                    continue
                if self.lang_code in "ab":
                    # English chunks at 510 phonemes on its own (en_tokenize); not modelled here.
                    ps, _ = self.g2p(graphemes)
                    self.spoken.append(ps)
                    yield graphemes, ps, [0.1] * len(ps)
                    continue
                chunk_size = 400
                chunks = []
                sentences = re.split(r"([.!?]+)", graphemes)
                current_chunk = ""
                for i in range(0, len(sentences), 2):
                    sentence = sentences[i]
                    if i + 1 < len(sentences):
                        sentence += sentences[i + 1]
                    if len(current_chunk) + len(sentence) <= chunk_size:
                        current_chunk += sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence
                if current_chunk:
                    chunks.append(current_chunk.strip())
                if not chunks:
                    chunks = [graphemes[i:i + chunk_size] for i in range(0, len(graphemes), chunk_size)]
                for chunk in chunks:
                    if not chunk.strip():
                        continue
                    ps, _ = self.g2p(chunk)
                    if not ps:
                        continue
                    elif len(ps) > 510:
                        self.truncated.append(len(ps))
                        ps = ps[:510]
                    self.spoken.append(ps)
                    yield chunk, ps, [0.1] * len(ps)

    return SimpleNamespace(KPipeline=KPipeline)


# 257 Han characters, Chinese punctuation only: on the real core, zf_xiaobei stopped at character 120.
ZH_PARAGRAPH = (
    "春天来了，小镇上的人们纷纷走出家门。河边的柳树长出了嫩绿的新芽，田野里开满了金黄色的油菜花。"
    "孩子们在草地上放风筝，老人们坐在树下喝茶聊天。早上，市场里非常热闹，卖菜的农民把新鲜的蔬菜摆得整整齐齐。"
    "中午，阳光温暖地照在石板路上，小猫懒洋洋地躺在门口睡觉。下午，学校的钟声响起，学生们背着书包高高兴兴地回家。"
    "傍晚，天边的晚霞像一幅美丽的油画，炊烟从每家每户的屋顶慢慢升起。晚上，月亮挂在天空中，星星一闪一闪地眨着眼睛。"
    "整个小镇渐渐安静下来，人们带着满足的心情进入了甜美的梦乡。最后，我们约定明年春天再一起回到这里看花。"
)
JA_PARAGRAPH = "私たちは今日、公園を散歩しました。天気はとても良くて、風が気持ちよかったです。" * 5
HI_PARAGRAPH = "मैं आज सुबह बाजार गया और मैंने ताज़ी सब्ज़ियाँ खरीदीं। " * 14
ES_SENTENCE = (
    "Cuando llegamos al pueblo, despues de caminar durante horas por los senderos de la montana, "
    "bajo un sol que no daba tregua; con las mochilas pesadas, las botas llenas de polvo y la garganta seca, "
    "encontramos una pequena plaza con una fuente de piedra, unos bancos de madera a la sombra de los platanos, "
    "y una panaderia que todavia tenia las puertas abiertas, aunque ya era tarde; entramos, saludamos al panadero, "
    "que nos miro con curiosidad, y compramos pan recien hecho, queso de la region y unas naranjas dulces, "
    "que comimos sentados junto a la fuente mientras el sol, por fin, empezaba a esconderse detras de las colinas."
)
LONG_TEXTS = [
    ("zf_xiaobei", ZH_PARAGRAPH),
    ("jf_alpha", JA_PARAGRAPH),
    ("hf_alpha", HI_PARAGRAPH),
    ("ef_dora", ES_SENTENCE),
    # No punctuation and no spaces at all: the last resort cuts at a character boundary.
    ("zm_yunxi", "春天来了小镇上的人们纷纷走出家门" * 15),
    # The Reader's 400-character span of a Mandarin paragraph, one 4096-character request in French.
    ("zf_xiaoxiao", ZH_PARAGRAPH[:200] + ZH_PARAGRAPH[:200]),
    ("ff_siwis", ("Il était une fois, dans un pays lointain, une reine qui aimait les livres; " * 55)[:4096]),
]
LONG_TEXT_IDS = ["mandarin", "japanese", "hindi", "spanish", "mandarin-unpunctuated", "mandarin-400", "french-4096"]


def _without_spaces(phonemes: str) -> str:
    return phonemes.replace(" ", "")


class TestNonEnglishLongText:
    @pytest.mark.parametrize(("voice", "text"), LONG_TEXTS, ids=LONG_TEXT_IDS)
    def test_every_phoneme_of_a_long_text_reaches_the_model(self, voice, text):
        package = _kokoro_094_package()
        backend = KokoroBackend(device="cpu")

        with patch.dict("sys.modules", {"kokoro": package}):
            audio = list(backend.synthesize(text, voice))

        pipeline = backend._pipeline
        expected = _without_spaces(_stub_g2p(text)[0])
        assert pipeline.truncated == [], f"{voice}: KPipeline cut chunks of {pipeline.truncated} phonemes to 510"
        assert _without_spaces("".join(pipeline.spoken)) == expected
        assert sum(len(chunk) for chunk in audio) == sum(len(ps) for ps in pipeline.spoken)
        assert all(len(ps) <= 510 for ps in pipeline.spoken)

    @pytest.mark.parametrize(
        "text",
        ["春天来了，小镇上的人们纷纷走出家门。", ZH_PARAGRAPH.replace("。", "。\n"), "Hola. ¿Qué tal?\n\nBien."],
        ids=["short-sentence", "one-sentence-per-line", "short-lines"],
    )
    def test_text_that_fits_is_chunked_as_before(self, text):
        # Text whose lines fit in one KPipeline chunk reaches the model in the same chunks as when the
        # whole string goes to KPipeline, so short replies sound as they did.
        voice = "ef_dora" if text.startswith("Hola") else "zf_xiaobei"
        direct = _kokoro_094_package().KPipeline(lang_code=voice[0])
        list(direct(text, voice=voice))

        package = _kokoro_094_package()
        backend = KokoroBackend(device="cpu")
        with patch.dict("sys.modules", {"kokoro": package}):
            list(backend.synthesize(text, voice))

        assert backend._pipeline.spoken == direct.spoken

    @pytest.mark.parametrize("voice", ["af_heart", "bf_emma"])
    def test_english_text_still_goes_to_kpipeline_whole(self, voice):
        # English has its own 510-phoneme chunking in KPipeline: the backend passes the text unchanged.
        text = "This is a long English paragraph. " * 60 + "\nA second line."
        package = _kokoro_094_package()
        backend = KokoroBackend(device="cpu")

        with patch.dict("sys.modules", {"kokoro": package}):
            list(backend.synthesize(text, voice))

        assert backend._pipeline.calls == [text]


# Mandarin and Japanese are often written with half-width "," and "." straight
# before the next character. By UAX #29 a "." before a Han or kana character ends a sentence (SB11), and kokoro
# 0.9.4 splits at [.!?]+ with no space; the split cut such a line at its middle character instead, mid-word.
HALF_WIDTH_ZH = (
    "春天来了,小镇上的人们纷纷走出家门.河边的柳树长出了嫩绿的新芽,田野里开满了金黄色的油菜花."
    "孩子们在草地上放风筝,老人们坐在树下喝茶聊天.早上,市场里非常热闹,卖菜的农民把新鲜的蔬菜摆得整整齐齐."
    "中午,阳光温暖地照在石板路上,小猫懒洋洋地躺在门口睡觉.下午,学校的钟声响起,学生们背着书包高高兴兴地回家."
    "傍晚,天边的晚霞像一幅美丽的油画,炊烟从每家每户的屋顶慢慢升起.春天来了,小镇上的人们纷纷走出家门."
    "河边的柳树长出了嫩绿的新芽,田野里开满了金黄色的油菜花.孩子们在草地上放风筝,老人们坐在树下喝茶聊天."
    "早上,市场里非常热闹,卖菜的农民把新鲜的蔬菜摆得整整齐齐."
)
HALF_WIDTH_JA = (
    "今朝は早く起きて,窓を開けると,冷たい空気が部屋の中に入ってきました."
    "空はまだ少し暗く,遠くの山の上に,薄い雲がゆっくりと流れていました."
    "朝ご飯を食べてから,近くの公園まで散歩に出かけました."
) * 3


class TestHalfWidthPunctuatedCjk:
    @pytest.mark.parametrize("text", [HALF_WIDTH_ZH, HALF_WIDTH_JA], ids=["mandarin", "japanese"])
    def test_a_long_line_is_cut_only_after_its_sentence_ends(self, text):
        assert len(_stub_g2p(text)[0]) > KPIPELINE_MAX_PHONEMES
        pieces = split_for_kpipeline(text, _stub_g2p)

        assert len(pieces) > 1
        assert "".join(pieces) == text
        assert all(len(_stub_g2p(piece)[0]) <= KPIPELINE_MAX_PHONEMES for piece in pieces)
        assert [piece[-6:] for piece in pieces if not piece.endswith(".")] == []

    def test_a_long_clause_is_cut_after_a_half_width_comma(self):
        # One sentence too long for one piece, with only half-width commas inside: cut at a comma, not mid-word.
        text = ",".join(["小镇上的人们纷纷走出家门"] * 15) + "."
        pieces = split_for_kpipeline(text, _stub_g2p)

        assert len(pieces) > 1
        assert "".join(pieces) == text
        assert all(len(_stub_g2p(piece)[0]) <= KPIPELINE_MAX_PHONEMES for piece in pieces)
        assert all(piece.endswith((",", ".")) for piece in pieces)

    @pytest.mark.parametrize(
        "text",
        [
            "El precio subió un 3.5 por ciento hasta 1,000 euros en la tienda del centro, p.ej.mañana. " * 12,
            "价格上涨了3.5倍,达到1,000元" + "春天来了小镇上的人们纷纷走出家门" * 12,
        ],
        ids=["spanish-numbers", "mandarin-numbers"],
    )
    def test_numbers_and_latin_abbreviations_stay_whole(self, text):
        # A digit or Latin letter after "." or "," is not a cut point: "3.5", "1,000" and "p.ej." stay whole.
        pieces = split_for_kpipeline(text, _stub_g2p)

        assert len(pieces) > 1
        assert all(len(_stub_g2p(piece)[0]) <= KPIPELINE_MAX_PHONEMES for piece in pieces)
        assert not [piece for piece in pieces if piece.endswith(("3.", "1,", "p.", "p.ej."))]


# Kokoro 0.9.4's own pipeline.py is a tracked test fixture, so the source check
# below runs on every checkout with Torch, not only on a host that kept a git-ignored data/providers/ folder.
KOKORO_094_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "kokoro_094"
VENDORED_KPIPELINE = KOKORO_094_FIXTURE / "pipeline.py"
# kokoro/pipeline.py's sha256 in the kokoro-0.9.4 wheel's RECORD (urlsafe base64 CdoyqreB96Fjy_n243nVP...).
KPIPELINE_094_SHA256 = "09da32aab781f7a163cbf9f6e379d53db40957cbbab50bc26a21c8440c0eabd6"
VENDOR_HEADER_END = b"# --- end of vendoring header; kokoro 0.9.4 source, unchanged, follows ---\n"


def test_kokoro_094_kpipeline_source_is_vendored_unchanged():
    # The fixture is in the repository and is 0.9.4's file byte for byte below its
    # header, so the source check compares the hand copy above with the real library everywhere.
    # A Windows checkout with core.autocrlf turns the LF file into CRLF; the wheel's file has LF.
    source = VENDORED_KPIPELINE.read_bytes().replace(b"\r\n", b"\n")
    header, found, upstream = source.partition(VENDOR_HEADER_END)
    assert found, "the vendored pipeline.py lost its provenance header"
    assert b"Apache License, Version 2.0" in header
    assert hashlib.sha256(upstream).hexdigest() == KPIPELINE_094_SHA256
    assert "Apache License" in (KOKORO_094_FIXTURE / "LICENSE").read_text(encoding="utf-8")


@needs_torch
@pytest.mark.parametrize(("voice", "text"), LONG_TEXTS[:4], ids=LONG_TEXT_IDS[:4])
def test_kokoro_094_source_speaks_every_phoneme(voice, text):
    """The same check on kokoro 0.9.4's own KPipeline source; only KModel, misaki and loguru are stubbed."""
    import torch

    spoken: list[str] = []
    warnings: list[str] = []

    class FakeKModel:
        Output = SimpleNamespace
        device = "cpu"

        def __init__(self, repo_id=None, **_kwargs):
            pass

        def to(self, _device):
            return self

        def eval(self):
            return self

        def __call__(self, phonemes, ref_s, speed=1, return_output=False):
            spoken.append(phonemes)
            audio = torch.full((len(phonemes),), 0.1)
            return SimpleNamespace(audio=audio, pred_dur=None) if return_output else audio

    kokoro_pkg = ModuleType("kokoro")
    kokoro_pkg.__path__ = [str(VENDORED_KPIPELINE.parent)]
    model_mod = ModuleType("kokoro.model")
    model_mod.KModel = FakeKModel
    loguru = ModuleType("loguru")
    loguru.logger = SimpleNamespace(
        warning=lambda msg, *a, **k: warnings.append(str(msg)),
        debug=lambda *a, **k: None, error=lambda *a, **k: None, info=lambda *a, **k: None,
    )
    hf = ModuleType("huggingface_hub")
    hf.hf_hub_download = lambda *a, **k: pytest.fail("no download")
    misaki = ModuleType("misaki")
    en = ModuleType("misaki.en")
    en.MToken = type("MToken", (), {})
    espeak = ModuleType("misaki.espeak")
    espeak.EspeakG2P = lambda language: _stub_g2p
    zh = ModuleType("misaki.zh")
    zh.ZHG2P = lambda version=None, en_callable=None: _stub_g2p
    ja = ModuleType("misaki.ja")
    ja.JAG2P = lambda: _stub_g2p
    misaki.en, misaki.espeak, misaki.zh, misaki.ja = en, espeak, zh, ja
    modules = {
        "kokoro": kokoro_pkg, "kokoro.model": model_mod, "loguru": loguru, "huggingface_hub": hf,
        "misaki": misaki, "misaki.en": en, "misaki.espeak": espeak, "misaki.zh": zh, "misaki.ja": ja,
    }

    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location("kokoro.pipeline", VENDORED_KPIPELINE)
        pipeline_mod = importlib.util.module_from_spec(spec)
        sys.modules["kokoro.pipeline"] = pipeline_mod
        spec.loader.exec_module(pipeline_mod)
        pipeline_mod.KPipeline.load_single_voice = lambda self, v: torch.zeros(511, 1, 256)
        kokoro_pkg.KPipeline = pipeline_mod.KPipeline
        backend = KokoroBackend(device="cpu")
        list(backend.synthesize(text, voice))

    assert not [w for w in warnings if w.startswith("Truncating")], warnings
    assert _without_spaces("".join(spoken)) == _without_spaces(_stub_g2p(text)[0])
