"""Isolated CosyVoice 2/3 worker for the Open Speech harness."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import os
import re
import threading
import time
import unicodedata
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import torch
from fastapi import HTTPException, Request  # noqa: F401 - callers catch worker.HTTPException
from fastapi.responses import StreamingResponse
from pydantic import Field

from providers._common import worker as common
from providers._common.worker import LoadRequest, WorkerFailure

PROVIDER = "cosyvoice"
SCHEMA_VERSION = common.SCHEMA_VERSION
SAMPLE_RATE = 24000
MAX_INPUT_CHARS = int(os.environ.get("COSYVOICE_MAX_INPUT_CHARS", "1500"))
MAX_REFERENCE_BYTES = int(os.environ.get("COSYVOICE_MAX_REFERENCE_MB", "100")) * 1024 * 1024
REFERENCE_CACHE_SIZE = int(
    os.environ.get("COSYVOICE_REFERENCE_CACHE_SIZE", str(common.DEFAULT_REFERENCE_CACHE_SIZE))
)
FP16 = common.env_flag("COSYVOICE_FP16", "true")
COSYVOICE3_PROMPT = "You are a helpful assistant."
END_OF_PROMPT = "<|endofprompt|>"
OUT_OF_MEMORY_MESSAGE = "CUDA out of memory during CosyVoice synthesis"
SILENCE_SECONDS = 0.2  # the answer to a text of only punctuation or symbols
_END_OF_OUTPUT = object()

MODEL_IDS = {
    "cosyvoice/2-0.5b": "FunAudioLLM/CosyVoice2-0.5B",
    "cosyvoice/3-0.5b": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
}
MODEL_REVISIONS = {
    "cosyvoice/2-0.5b": "eec1ae6c79877dbd9379285cf8789c9e0879293d",
    "cosyvoice/3-0.5b": "29e01c4e8d000f4bcd70751be16fa94bf3d85a18",
}
# CosyVoice 2 speaks Chinese, English and Korean; Fun-CosyVoice3 adds German, French, Russian,
# Spanish and Italian. Both models list Japanese upstream, but the pinned library reads it only as
# katakana ("for Japanese usage, you must translate it to katakana", example.py at 074ca6dc), and
# the image has no step that makes it: digits and kanji were spoken garbled. So Japanese is refused
# (a ja hint, or kana in a text with no hint) and its speakers are pointed at Qwen3 Base.
MODEL_LANGUAGES = {
    "cosyvoice/2-0.5b": ["zh", "en", "ko"],
    "cosyvoice/3-0.5b": ["zh", "en", "ko", "de", "fr", "ru", "es", "it"],
}
JAPANESE_MODEL = "qwen3/0.6b-base"
# CosyVoice's frontend reads numbers, money, times and dates through wetext's text normalizer,
# whose FST files wetext would fetch from ModelScope's default branch at every load in a new
# container. The image bakes them at a pinned commit (providers/cosyvoice/Dockerfile), and the
# worker points wetext at that copy. Without a normalizer CosyVoice only spells out digit runs
# ("$5.20" becomes "$five.twenty"), so a load that ends up without one fails.
WETEXT_REPO = "pengzhendong/wetext"
WETEXT_DIR = "/opt/wetext"
TEXT_FRONTENDS = frozenset({"ttsfrd", "wetext"})
# That frontend (text_frontend=True, the library's default) knows two kinds of text: text with a
# CJK ideograph goes to wetext's Chinese normalizer, any other text to its English one, and every
# digit run left is spelled out in English. It is right for English and Chinese only: German
# "3 Katzen" became "three Katzen". The other languages go to the model as written
# (text_frontend=False), which reads numbers itself.
TEXT_FRONTEND_LANGUAGES = frozenset({"en", "zh"})
# Without a hint (the web app sends none), Latin-script text is weighed by its words: English
# function words against German, French, Spanish and Italian ones, a word with an accented letter
# counting for the others. English, and a tie, keep the frontend. Words common to English and one
# of the others ("a", "in", "an", "am", "on", "me", "no", "so", "was", "will", "non", "pour") count
# for neither.
_ENGLISH_WORDS = frozenset(
    "the and of to is are were be been have has had do does did not but from by with for at "
    "this that these those it its i you he she we they my your his her our their what which who "
    "when where why how there here than then if or would could should can about into after before "
    "just very".split()
)
_OTHER_LATIN_WORDS = frozenset(
    # German
    "der die das den dem des und ist sind hat habe haben bin ich du er sie wir ihr nicht ein eine "
    "einen einem einer mit auf zu von aus bei nach auch noch oder aber wie wenn dass es kein keine "
    "im vom zum zur sich mir mich dir uns hier "
    # Spanish
    "el la los las un una unos unas y de en tengo tiene tienen con para por que del al lo se su sus "
    "muy pero yo tu hay este esta eso ella ellos cuando donde "
    # Italian
    "il gli le uno e di da che ho ha hanno sono della delle dei degli nel nella alla questo questa "
    "anche ma ti ci si io lui lei noi voi loro "
    # French
    "et est sont elle ils elles je nous vous ce cette ces les des qui pas ne avec dans sur au aux "
    "mon ma mes sa ses leur mais ou du une l qu".split()
)
_LATIN_WORD = re.compile(r"[^\W\d_]+")
# Text with a CJK ideograph is Chinese, as the library's own router tells it
# (cosyvoice/utils/frontend_utils.py contains_chinese), and keeps the frontend however many
# Latin words it holds. Kana is refused before this check.
_CJK_IDEOGRAPH = re.compile(r"[一-鿿]")


def _capabilities(model_id: str) -> dict[str, Any]:
    return {
        "voice_blend": False,
        "voice_design": False,
        "voice_clone": True,
        "reference_audio": True,
        "clone_transcript": True,
        "clone_transcript_required": True,
        "streaming": True,
        "instructions": True,
        "languages": list(MODEL_LANGUAGES[model_id]),
        "speed_control": True,
        "speed_mode": "native_non_streaming",
        "ssml": False,
        "batch": False,
        "cancellation": "between_chunks",
        "live_reader": True,
        "native_tags": True,
        "speakers": [],
    }


MANIFEST = {
    "schema_version": SCHEMA_VERSION,
    "provider": PROVIDER,
    "models": [
        {
            "id": model_id,
            "provider": PROVIDER,
            "sample_rate": SAMPLE_RATE,
            "revision": MODEL_REVISIONS[model_id],
            "max_input_chars": MAX_INPUT_CHARS,
            "capabilities": _capabilities(model_id),
            "voices": [],
        }
        for model_id in MODEL_IDS
    ],
}


class SynthesisRequest(common.SynthesisRequest):
    text: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


def _decode_reference(encoded: str | None) -> bytes:
    audio = common.decode_reference_audio(
        encoded,
        max_bytes=MAX_REFERENCE_BYTES,
        missing_message="reference_audio is required for CosyVoice voice cloning",
    )
    info = common.reference_audio_info(audio)
    if info.duration > 30.0:
        raise WorkerFailure(
            "CosyVoice reference audio cannot exceed 30 seconds",
            code="reference_audio_too_long",
            status_code=400,
        )
    return audio


def _prompt_text(model_id: str, transcript: str) -> str:
    if model_id.endswith("3-0.5b"):
        return f"{COSYVOICE3_PROMPT}{END_OF_PROMPT}{transcript}"
    return transcript


def _instruction_text(model_id: str, instructions: str) -> str:
    clean = instructions.replace(END_OF_PROMPT, "").strip()
    if model_id.endswith("3-0.5b"):
        clean = f"{COSYVOICE3_PROMPT} {clean}".strip()
    return f"{clean}{END_OF_PROMPT}"


def _use_baked_text_normalizer() -> None:
    """Make wetext read its FST files from the image instead of fetching them from ModelScope.

    wetext 0.0.4 calls ``snapshot_download("pengzhendong/wetext")``, imported from modelscope
    into its own module, with no revision whenever a Normalizer is built without file paths, as
    CosyVoice builds both of its own. Both names are redirected, so a wetext that imports the
    function late reads the baked copy too. Any other repository still goes to modelscope.
    """
    for module_name in ("modelscope", "wetext.wetext"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        fetch = getattr(module, "snapshot_download", None)
        if not callable(fetch) or getattr(fetch, "open_speech_baked", False):
            continue
        module.snapshot_download = _baked_snapshot(fetch)


def _baked_snapshot(fetch: Callable[..., Any]) -> Callable[..., Any]:
    def baked(*args: Any, **kwargs: Any) -> Any:
        repo_id = args[0] if args else kwargs.get("model_id", kwargs.get("repo_id"))
        if repo_id == WETEXT_REPO:
            return WETEXT_DIR
        return fetch(*args, **kwargs)

    baked.open_speech_baked = True  # type: ignore[attr-defined]
    return baked


def check_text_normalizer() -> dict[str, str]:
    """Normalize one English and one Chinese sample with the baked files, as CosyVoice would.

    The image build runs this with no network, so missing or unreadable files fail the build
    instead of a later model load.
    """
    _use_baked_text_normalizer()
    from wetext import Normalizer

    normalizer = Normalizer(remove_erhua=False)
    samples = {"en": "Chapter 12 has 25 pages.", "zh": "第3章有25页。"}
    results = {language: normalizer.normalize(text) for language, text in samples.items()}
    for language, result in results.items():
        if any(character in "0123456789" for character in result):
            raise RuntimeError(f"wetext left digits in the {language} sample: {result!r}")
    return results


def _primary_subtag(language: str | None) -> str:
    return (language or "").strip().lower().replace("_", "-").split("-", 1)[0]


def _has_kana(text: str) -> bool:
    """Whether the text has a Hiragana or Katakana letter (not the katakana middle dot)."""
    for character in text:
        code = ord(character)
        if (0x3040 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF or 0xFF66 <= code <= 0xFF9F) and (
            character.isalpha()
        ):
            return True
    return False


def _refuse_unsupported_language(model_id: str, language: str | None, text: str) -> None:
    """Refuse a hint the model doesn't list, and Japanese text (kana) sent with no hint."""
    supported = MODEL_LANGUAGES[model_id]
    primary = _primary_subtag(language)
    listing = f"Supported: {', '.join(supported)}."
    if primary:
        if primary in supported:
            return
        message = f"Unsupported language '{(language or '').strip()}' for {model_id}. {listing}"
    elif _has_kana(text):
        message = f"{model_id} does not speak Japanese, and this text has kana. {listing}"
    else:
        return
    if primary in ("", "ja"):
        message += f" For Japanese, use {JAPANESE_MODEL}."
    raise WorkerFailure(message, code="language_unsupported", status_code=400)


def _reads_as_english_or_chinese(text: str) -> bool:
    """Guess, with no hint, whether the text is English or Chinese.

    Hangul, Cyrillic and kana are not. Text with a CJK ideograph is Chinese, whatever Latin words
    it holds. In other text the Latin-script words are weighed: English function words against
    German, French, Spanish and Italian ones, a word with an accented Latin letter counting for
    the others. English, a tie and text with no such word are kept.
    """
    for character in text:
        code = ord(character)
        if code < 0x80:
            continue
        if 0x1100 <= code <= 0x11FF or 0x3130 <= code <= 0x318F or 0xAC00 <= code <= 0xD7AF:
            return False  # Hangul
        if 0x0400 <= code <= 0x052F:
            return False  # Cyrillic
    if _has_kana(text):
        return False
    if _CJK_IDEOGRAPH.search(text):
        return True  # Chinese: its Latin words ("iPhone SE", "MIT", "café") are not weighed
    english = other = 0
    for word in _LATIN_WORD.findall(text.lower()):
        if word in _ENGLISH_WORDS:
            english += 1
        elif word in _OTHER_LATIN_WORDS or any(
            (0xC0 <= ord(letter) <= 0x24F or 0x1E00 <= ord(letter) <= 0x1EFF) for letter in word
        ):
            other += 1
    return other <= english


def _uses_text_frontend(language: str | None, text: str) -> bool:
    """Whether CosyVoice's own text frontend should read this text (English and Chinese only).

    A language hint decides by its primary subtag (en-GB is en). Without one, the text does
    (_reads_as_english_or_chinese).
    """
    primary = _primary_subtag(language)
    if primary:
        return primary in TEXT_FRONTEND_LANGUAGES
    return _reads_as_english_or_chinese(text)


def _nothing_to_say(text: str) -> bool:
    """Whether the text is only punctuation, symbols and spaces.

    CosyVoice's frontend drops every piece of only punctuation or symbols
    (``is_only_punctuation``, ``^[\\p{P}\\p{S}]*$``), so such a text yields no audio at all.
    """
    return all(character.isspace() or unicodedata.category(character)[0] in "PS" for character in text)


def _each_sentence(
    method: Callable[..., Iterator[Any]], text: str, *args: Any, **options: Any
) -> Iterator[Any]:
    """Synthesize the text one sentence at a time.

    With text_frontend=False CosyVoice skips its sentence split too and would speak a long
    text as one segment, so the text is split here, with the workers' sentence rules.
    """
    for sentence in common.sentence_parts(text) or [text]:
        yield from method(sentence, *args, **options)


def _numpy_audio(output: Any) -> np.ndarray:
    if not isinstance(output, dict) or "tts_speech" not in output:
        raise WorkerFailure("Model returned an invalid result", code="invalid_audio")
    return common.float32_audio(output["tts_speech"])


def _supports_speaker_cache(model: Any) -> bool:
    """Whether the model can register reference conditioning as a zero-shot speaker.

    Every parameter the cache passes is checked, so a library without one falls back to
    the uncached call instead of failing each request.
    """
    frontend = getattr(model, "frontend", None)
    if not callable(getattr(model, "add_zero_shot_spk", None)):
        return False
    if not isinstance(getattr(frontend, "spk2info", None), dict):
        return False
    required = [
        (frontend, "text_normalize", ("split", "text_frontend")),
        (model, "inference_zero_shot", ("zero_shot_spk_id",)),
        (model, "inference_instruct2", ("zero_shot_spk_id",)),
    ]
    for owner, method_name, names in required:
        try:
            parameters = inspect.signature(getattr(owner, method_name)).parameters
        except (AttributeError, TypeError, ValueError):
            return False
        if any(name not in parameters for name in names):
            return False
    return True


class _LlmJobs:
    """Make CosyVoice's background LLM thread report its failure and stop on request.

    CosyVoice2Model.tts runs the LLM's token generation (``llm_job``) on its own thread and
    polls ``llm_end_dict`` until that thread marks the end. An error there, such as CUDA out
    of memory, ends the thread without the mark: a streamed request would wait forever with
    the worker locked, and a whole one would vocode the partial tokens as if complete. The
    thread also outlives a request that ends early and would share the GPU with the next.
    """

    def __init__(self, engine: Any, job: Callable[..., Any], uuid_index: int) -> None:
        self._engine = engine
        self._job = job
        self._uuid_index = uuid_index
        self._changed = threading.Condition()
        self._running = 0
        self._stopping = False
        self._failure: Exception | None = None

    @classmethod
    def install(cls, model: Any) -> _LlmJobs | None:
        """Wrap the loaded model's LLM thread, or return None when its internals differ."""
        engine = getattr(model, "model", None)
        job = getattr(engine, "llm_job", None)
        if not callable(job) or not isinstance(getattr(engine, "llm_end_dict", None), dict):
            return None
        try:
            parameters = list(inspect.signature(job).parameters)
        except (TypeError, ValueError):
            return None
        if "uuid" not in parameters:
            return None
        jobs = cls(engine, job, parameters.index("uuid"))
        engine.llm_job = jobs._run
        llm = getattr(engine, "llm", None)
        for name in ("inference", "inference_bistream"):
            inference = getattr(llm, name, None)
            if callable(inference):
                setattr(llm, name, jobs._stoppable(inference))
        return jobs

    def _run(self, *args: Any, **kwargs: Any) -> None:
        with self._changed:
            self._running += 1
        try:
            self._job(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - any failure must end tts()'s wait
            common.logger.warning("CosyVoice LLM thread failed", exc_info=True)
            with self._changed:
                if self._failure is None:
                    # Without its frames, the failed step's tensors can be freed now.
                    self._failure = exc.with_traceback(None)
            # Set after the failure is recorded, so tts() stops waiting and the request
            # that pulls its output sees the failure.
            uuid = kwargs["uuid"] if "uuid" in kwargs else args[self._uuid_index]
            self._engine.llm_end_dict[uuid] = True
        finally:
            with self._changed:
                self._running -= 1
                self._changed.notify_all()

    def _stoppable(self, inference: Callable[..., Any]) -> Callable[..., Iterator[Any]]:
        def tokens(*args: Any, **kwargs: Any) -> Iterator[Any]:
            generated = inference(*args, **kwargs)
            try:
                for token in generated:
                    if self._stopping:
                        return
                    yield token
            finally:
                close = getattr(generated, "close", None)
                if close is not None:
                    close()

        return tokens

    def raise_failure(self) -> None:
        """Raise the error that ended this request's LLM thread, if it failed."""
        with self._changed:
            failure, self._failure = self._failure, None
        if failure is not None:
            raise failure

    def settle(self) -> None:
        """Stop the LLM threads a finished request left running, and wait until they end."""
        with self._changed:
            self._stopping = True
            try:
                self._changed.wait_for(lambda: self._running == 0)
            finally:
                self._stopping = False
                self._failure = None


def _start_each_stream_from_the_initial_hop(model: Any) -> None:
    """Make every ``tts`` call stream its first chunk after the model's initial hop.

    CosyVoice2Model.tts (inherited by CosyVoice 3) keeps its chunk size on the instance and
    doubles it after each streamed chunk, up to four times the start, and nothing resets it. So
    from the worker's second streamed chunk on, every request and every text segment would
    wait for 100 tokens (4 seconds of speech) before its first audio. The growth within one
    call is the library's design and stays.
    """
    engine = getattr(model, "model", None)
    initial = getattr(engine, "token_hop_len", None)
    tts = getattr(engine, "tts", None)
    if not isinstance(initial, int) or isinstance(initial, bool) or not callable(tts):
        return

    def tts_from_the_initial_hop(*args: Any, **kwargs: Any) -> Any:
        engine.token_hop_len = initial
        return tts(*args, **kwargs)

    engine.tts = tts_from_the_initial_hop


class CosyVoiceRuntime(common.WorkerRuntime):
    display_name = "CosyVoice"

    def __init__(self) -> None:
        super().__init__(reference_cache_size=REFERENCE_CACHE_SIZE)
        self.llm_jobs: _LlmJobs | None = None

    def after_load(self) -> None:
        self.llm_jobs = _LlmJobs.install(self.model)
        _start_each_stream_from_the_initial_hop(self.model)

    def unload(self) -> None:
        self.llm_jobs = None
        super().unload()

    def is_known_model(self, model_id: str) -> bool:
        return model_id in MODEL_IDS

    def build_model(self, model_id: str, capability: tuple[int, int]) -> Any:
        from cosyvoice.cli.cosyvoice import AutoModel
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(
            repo_id=MODEL_IDS[model_id],
            revision=MODEL_REVISIONS[model_id],
        )
        _use_baked_text_normalizer()
        model = AutoModel(model_dir=model_dir, fp16=FP16)
        if int(getattr(model, "sample_rate", 0)) != SAMPLE_RATE:
            raise WorkerFailure(
                f"Model sample rate {getattr(model, 'sample_rate', None)} does not match {SAMPLE_RATE}",
                code="sample_rate_mismatch",
            )
        # CosyVoice logs a missing normalizer only at INFO and carries on without it.
        text_frontend = getattr(getattr(model, "frontend", None), "text_frontend", None)
        if text_frontend not in TEXT_FRONTENDS:
            raise WorkerFailure(
                "CosyVoice's text normalizer did not load (its wetext files are expected in "
                f"{WETEXT_DIR}); without it numbers, money and times are read wrong",
                code="text_frontend_unavailable",
                status_code=503,
            )
        return model

    def _cached_speaker(
        self,
        model: Any,
        *,
        instructed: bool,
        prompt: str,
        reference: bytes,
        reference_path: str,
        text_frontend: bool,
    ) -> str | None:
        """Return a zero-shot speaker ID holding this reference's conditioning.

        Without it, CosyVoice re-extracts the prompt's speech tokens, features, and speaker
        embedding for every segment. The conditioning depends on the prompt text too: the
        transcript for zero-shot synthesis, the instruction for instructed synthesis.
        """
        if not _supports_speaker_cache(model):
            return None
        kind = "instruct" if instructed else f"zero_shot\0text_frontend={text_frontend}"
        key = self.references.key(self.model_id, reference, f"{kind}\0{prompt}")
        speaker_id = self.references.get(key)
        if speaker_id is not None:
            return speaker_id
        speaker_id = f"open-speech-{key[:32]}"
        # inference_zero_shot normalizes its prompt text, with the call's text_frontend, and
        # add_zero_shot_spk does not; inference_instruct2 uses the instruction as given.
        prompt_text = prompt
        if not instructed:
            prompt_text = model.frontend.text_normalize(
                prompt, split=False, text_frontend=text_frontend
            )
        model.add_zero_shot_spk(prompt_text, reference_path, speaker_id)
        for evicted in self.references.put(key, speaker_id):
            model.frontend.spk2info.pop(evicted, None)
        return speaker_id

    def _next_output(self, outputs: Iterator[Any]) -> Any:
        """Pull the model's next output; a failure on its LLM thread fails this step too."""
        try:
            output = next(outputs, _END_OF_OUTPUT)
        except Exception:
            # What went wrong after the LLM thread failed is a consequence; report the cause.
            if self.llm_jobs is not None:
                self.llm_jobs.raise_failure()
            raise
        if self.llm_jobs is not None:
            self.llm_jobs.raise_failure()
        return output

    def generate(self, payload: SynthesisRequest) -> Iterator[np.ndarray]:
        """Yield audio chunks as the model produces them.

        torch.inference_mode is thread-local and the HTTP stream pulls each chunk on
        whichever worker thread is free, so inference mode is entered and left inside
        each step instead of being held across a yield.
        """
        model = self.require_model(payload.model)
        _refuse_unsupported_language(payload.model, payload.language, payload.text)
        transcript = (payload.clone_transcript or "").strip()
        if not transcript:
            raise WorkerFailure(
                "clone_transcript is required for CosyVoice voice cloning",
                code="clone_transcript_required",
                status_code=400,
            )
        reference = _decode_reference(payload.reference_audio)
        instructed = bool(payload.instructions)
        if instructed:
            prompt = _instruction_text(payload.model, payload.instructions or "")
        else:
            prompt = _prompt_text(payload.model, transcript)
        text_frontend = _uses_text_frontend(payload.language, payload.text)
        if text_frontend and _nothing_to_say(payload.text):
            # The frontend would leave nothing for the model, and no audio fails the request
            # (a 502 at the core): a Live Reader, Ask AI or Talk segment such as "..." or a
            # Reader rule line "______" would fail its whole answer or part. Answer with the
            # short pause the model gives such text without the frontend.
            self.last_used_at = time.time()
            yield np.zeros(int(SAMPLE_RATE * SILENCE_SECONDS), dtype=np.float32)
            return
        options: dict[str, Any] = {"stream": payload.speed == 1.0, "speed": payload.speed}
        outputs: Iterator[Any] | None = None
        try:
            method = model.inference_instruct2 if instructed else model.inference_zero_shot
            with common.reference_file(reference) as reference_path:
                with torch.inference_mode():
                    speaker_id = common.guard_out_of_memory(
                        lambda: self._cached_speaker(
                            model,
                            instructed=instructed,
                            prompt=prompt,
                            reference=reference,
                            reference_path=reference_path,
                            text_frontend=text_frontend,
                        ),
                        OUT_OF_MEMORY_MESSAGE,
                    )
                if speaker_id is not None:
                    options = {"zero_shot_spk_id": speaker_id, **options}
                if text_frontend:
                    outputs = iter(method(payload.text, prompt, reference_path, **options))
                else:
                    outputs = _each_sentence(
                        method, payload.text, prompt, reference_path, text_frontend=False, **options
                    )
                produced = False
                while True:
                    with torch.inference_mode():
                        output = common.guard_out_of_memory(
                            lambda: self._next_output(outputs), OUT_OF_MEMORY_MESSAGE
                        )
                        if output is _END_OF_OUTPUT:
                            break
                        audio = _numpy_audio(output)
                    produced = True
                    self.last_used_at = time.time()
                    yield audio
                if not produced:
                    raise WorkerFailure("Model returned no audio", code="invalid_audio")
        except WorkerFailure:
            raise
        except Exception as exc:
            raise WorkerFailure(
                f"CosyVoice synthesis failed: {exc}",
                code="synthesis_failed",
            ) from exc
        finally:
            close = getattr(outputs, "close", None)
            if close is not None:
                with torch.inference_mode():
                    close()
            if self.llm_jobs is not None:
                # The worker stays busy until no LLM thread of this request uses the GPU.
                self.llm_jobs.settle()


_next_chunk = common.next_chunk

runtime = CosyVoiceRuntime()
operation_lock = asyncio.Lock()
app = common.create_app("Open Speech CosyVoice Worker")


@app.get("/health")
async def health() -> dict[str, Any]:
    return common.health_payload(
        PROVIDER,
        runtime,
        busy=operation_lock.locked(),
        details={"dtype": "float16" if FP16 else "float32"},
    )


@app.get("/v1/manifest")
async def manifest() -> dict[str, Any]:
    return MANIFEST


@app.post("/v1/models/load")
async def load_model(payload: LoadRequest) -> dict[str, Any]:
    return await common.load_model(runtime, operation_lock, payload.model)


@app.post("/v1/models/unload")
async def unload_model(payload: LoadRequest) -> dict[str, str]:
    return await common.unload_model(runtime, operation_lock, payload.model)


@app.post("/v1/audio/speech")
async def synthesize(payload: SynthesisRequest, request: Request) -> StreamingResponse:
    if payload.model not in MODEL_IDS:
        raise common.unknown_model_error(payload.model)
    common.ensure_idle(operation_lock)
    await operation_lock.acquire()
    return await common.stream_chunks(
        runtime.generate(payload),
        lock=operation_lock,
        request=request,
        sample_rate=SAMPLE_RATE,
        cancellation="between_chunks",
        headers={"X-Native-Streaming": "true" if payload.speed == 1.0 else "false"},
    )
