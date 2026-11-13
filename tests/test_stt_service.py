"""STT request handling: event-loop offloading, diarization, input validation and formats."""

from __future__ import annotations

import asyncio
import io
import json
import sys
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from starlette.datastructures import Headers, UploadFile

from src.diarization import pyannote_diarizer
from src.diarization.pyannote_diarizer import DiarizationSegment
from src.services import stt as stt_service
from src.utils import audio as audio_module
from tests.test_audio_conversion import _silent_flac


def _wav_bytes(rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * 1600)
    return buf.getvalue()


def _upload(audio: bytes | None = None) -> UploadFile:
    return UploadFile(
        file=io.BytesIO(_wav_bytes() if audio is None else audio),
        filename="a.wav",
        headers=Headers({"content-type": "audio/wav"}),
    )


def _settings(**overrides) -> SimpleNamespace:
    values = dict(
        os_max_upload_mb=10,
        stt_noise_reduce=False,
        stt_normalize=True,
        stt_diarize_enabled=True,
        os_history_enabled=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _on_event_loop_thread() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _backend(result: dict | None = None) -> MagicMock:
    backend = MagicMock()
    backend.transcribe.return_value = {"text": "hi"} if result is None else result
    return backend


async def _transcribe(
    *,
    backend=None,
    response_format="json",
    diarize=False,
    language=None,
    audio=None,
    settings=None,
    headers=None,
    history_manager=None,
    **kwargs,
):
    return await asyncio.wait_for(stt_service.transcribe_request(
        file=_upload(audio), model="m", language=language, prompt=None,
        response_format=response_format, temperature=0.0, diarize=diarize,
        raw_request=SimpleNamespace(headers=headers or {}), settings=settings or _settings(),
        backend_router=backend or _backend(), history_manager=history_manager or MagicMock(),
        **kwargs,
    ), timeout=10)


async def _transcribe_with_diarizer(diarizer_cls):
    return await _transcribe(diarize=True, diarizer_cls=diarizer_cls)


@pytest.mark.asyncio
async def test_upload_conversion_and_preprocessing_run_off_the_event_loop(monkeypatch):
    # ffmpeg (up to 30 s) and denoising would otherwise stall every other request and stream.
    seen: dict[str, bool] = {}

    def fake_convert(audio_bytes, suffix=".ogg"):
        seen["convert_on_loop"] = _on_event_loop_thread()
        return audio_bytes

    def fake_preprocess(wav_bytes, *, noise_reduce, normalize):
        seen["preprocess_on_loop"] = _on_event_loop_thread()
        return wav_bytes

    monkeypatch.setattr(stt_service, "convert_to_wav", fake_convert)
    monkeypatch.setattr(stt_service, "preprocess_stt_audio", fake_preprocess)

    await asyncio.wait_for(
        stt_service.read_and_prepare_upload(file=_upload(), settings=_settings()), timeout=10
    )

    assert seen == {"convert_on_loop": False, "preprocess_on_loop": False}


@pytest.mark.asyncio
async def test_diarization_pipeline_load_runs_off_the_event_loop():
    # PyannoteDiarizer.__init__ runs Pipeline.from_pretrained, a load of seconds or more.
    seen: dict[str, bool] = {}

    class RecordingDiarizer:
        def __init__(self):
            seen["init_on_loop"] = _on_event_loop_thread()

        def diarize(self, wav_bytes):
            return []

    await _transcribe_with_diarizer(RecordingDiarizer)

    assert seen == {"init_on_loop": False}


@pytest.mark.asyncio
async def test_diarization_inference_runtime_error_is_a_server_error():
    # Only setup errors (missing dependency, refused gated download) are a 400; torch
    # reports CUDA OOM and most inference failures as RuntimeError.
    class FailingDiarizer:
        def diarize(self, wav_bytes):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

    with pytest.raises(HTTPException) as exc_info:
        await _transcribe_with_diarizer(FailingDiarizer)

    assert exc_info.value.status_code == 500
    assert "Diarization failed" in exc_info.value.detail


class OutOfMemoryError(RuntimeError):
    """Stands in for torch.cuda.OutOfMemoryError, a RuntimeError subclass."""


@pytest.mark.asyncio
async def test_gpu_out_of_memory_while_loading_the_pipeline_is_a_server_error():
    # Moving the cached pipeline to a GPU shared with Whisper and TTS can run out of
    # memory. That is the server's problem, not a mistake in the client's request.
    class OutOfMemoryOnLoad:
        def __init__(self):
            raise OutOfMemoryError("CUDA out of memory. Tried to allocate 64.00 MiB")

    with pytest.raises(HTTPException) as exc_info:
        await _transcribe_with_diarizer(OutOfMemoryOnLoad)

    assert exc_info.value.status_code == 500
    assert "Diarization failed" in exc_info.value.detail


@pytest.mark.asyncio
async def test_missing_pyannote_is_a_400_that_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyannote", None)
    monkeypatch.setitem(sys.modules, "pyannote.audio", None)

    with pytest.raises(HTTPException) as exc_info:
        await _transcribe(diarize=True)

    assert exc_info.value.status_code == 400
    assert "open-speech[diarize]" in exc_info.value.detail


@pytest.mark.asyncio
async def test_translation_vtt_is_served_as_text_vtt():
    # Transcription serves VTT as text/vtt through src.formatters; translation must match.
    backend = MagicMock()
    backend.translate.return_value = {
        "text": "Hello",
        "segments": [{"start": 0.0, "end": 1.0, "text": " Hello"}],
    }

    response = await asyncio.wait_for(stt_service.translate_request(
        file=_upload(), model="m", prompt=None, response_format="vtt",
        temperature=0.0, settings=_settings(), backend_router=backend,
    ), timeout=10)

    assert backend.translate.call_args.kwargs["response_format"] == "verbose_json"
    assert response.headers["content-type"].startswith("text/vtt")
    assert response.body.decode() == "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello\n"


class _OneSegmentWhisper:
    """Stands in for WhisperModel: one segment, as faster-whisper yields it."""

    def transcribe(self, path, **kwargs):
        segment = SimpleNamespace(
            seek=0, start=0.0, end=1.0, text=" Hello there.", tokens=[50364, 2425, 456, 13],
            temperature=0.0, avg_logprob=-0.2, compression_ratio=1.0, no_speech_prob=0.01,
        )
        return iter([segment]), SimpleNamespace(language="en", duration=1.0)


@pytest.mark.asyncio
async def test_transcription_json_is_the_transcription_object_and_verbose_json_stays_verbose():
    # docs/AI-AGENTS.md section 1 says response_format=json answers
    # {"text": "..."}, as OpenAI's json format does; only verbose_json is the verbose object.
    # The real FasterWhisperBackend builds the result; only the WhisperModel is a fake.
    from src.backends.faster_whisper import FasterWhisperBackend

    backend = FasterWhisperBackend()
    backend._models["m"] = _OneSegmentWhisper()  # loaded, so nothing is fetched
    backend._loaded_at["m"] = 0.0

    plain = json.loads((await _transcribe(backend=backend, response_format="json")).body)
    verbose = json.loads((await _transcribe(backend=backend, response_format="verbose_json")).body)

    assert plain == {"text": "Hello there."}, f"json answered the verbose body, keys {sorted(plain)}"
    assert {key: verbose[key] for key in ("task", "language", "duration", "text")} == {
        "task": "transcribe", "language": "en", "duration": 1.0, "text": "Hello there.",
    }
    assert [segment["text"] for segment in verbose["segments"]] == [" Hello there."]


# ── Diarized responses ──────────────────────────────────────────────────────


def _two_speaker_result() -> dict:
    return {
        "task": "transcribe",
        "language": "en",
        "duration": 3.0,
        "text": "Hello there. Hi.",
        "segments": [
            {"id": 0, "start": 0.0, "end": 2.0, "text": " Hello there."},
            {"id": 1, "start": 2.0, "end": 3.0, "text": " Hi."},
        ],
    }


class _TwoSpeakers:
    def diarize(self, wav_bytes):
        return [
            DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=2.0),
            DiarizationSegment(speaker="SPEAKER_01", start=2.0, end=3.0),
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize(("response_format", "media_type", "body"), [
    ("text", "text/plain", "SPEAKER_00: Hello there.\nSPEAKER_01: Hi."),
    (
        "srt",
        "text/plain",
        "1\n00:00:00,000 --> 00:00:02,000\nSPEAKER_00: Hello there.\n\n"
        "2\n00:00:02,000 --> 00:00:03,000\nSPEAKER_01: Hi.\n",
    ),
    (
        "vtt",
        "text/vtt",
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\n<v SPEAKER_00>Hello there.\n\n"
        "00:00:02.000 --> 00:00:03.000\n<v SPEAKER_01>Hi.\n",
    ),
])
async def test_diarized_response_follows_the_requested_format(response_format, media_type, body):
    # README documents response_format json | verbose_json | text | srt | vtt next to
    # diarize; the body must be in the requested format, with the speakers in it.
    backend = _backend(_two_speaker_result())

    response = await _transcribe(
        backend=backend, response_format=response_format, diarize=True, diarizer_cls=_TwoSpeakers,
    )

    # Speakers are placed by Whisper's timestamps, so the backend is asked for them.
    assert backend.transcribe.call_args.kwargs["response_format"] == "verbose_json"
    assert response.media_type == media_type
    assert response.body.decode() == body


@pytest.mark.asyncio
async def test_diarized_verbose_json_keeps_the_verbose_fields_and_labels_each_segment():
    result = _two_speaker_result()

    response = await _transcribe(
        backend=_backend(result), response_format="verbose_json", diarize=True, diarizer_cls=_TwoSpeakers,
    )

    body = json.loads(response.body)
    assert {key: body[key] for key in ("task", "language", "duration", "text")} == {
        "task": "transcribe", "language": "en", "duration": 3.0, "text": "Hello there. Hi.",
    }
    assert body["segments"] == [
        {**result["segments"][0], "speaker": "SPEAKER_00"},
        {**result["segments"][1], "speaker": "SPEAKER_01"},
    ]


@pytest.mark.asyncio
async def test_diarized_json_gives_each_speaker_the_words_spoken_in_their_turn():
    # Each diarized segment's text is what was said between its start and end. A short
    # answer followed by a pause and fast speech must not hand the fast words to the
    # first speaker because their turn lasted longer.
    result = {
        "task": "transcribe", "language": "en", "duration": 12.0,
        "text": "Yes. one two three four five six seven eight nine ten",
        "segments": [
            {"id": 0, "start": 0.0, "end": 1.0, "text": " Yes."},
            {"id": 1, "start": 10.2, "end": 12.0, "text": " one two three four five six seven eight nine ten"},
        ],
    }

    class PauseThenFastSpeech:
        def diarize(self, wav_bytes):
            return [
                DiarizationSegment(speaker="SPEAKER_00", start=0.0, end=9.9),
                DiarizationSegment(speaker="SPEAKER_01", start=9.9, end=12.0),
            ]

    response = await _transcribe(backend=_backend(result), diarize=True, diarizer_cls=PauseThenFastSpeech)

    assert json.loads(response.body) == {
        "text": result["text"],
        "segments": [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 9.9, "text": "Yes."},
            {
                "speaker": "SPEAKER_01", "start": 9.9, "end": 12.0,
                "text": "one two three four five six seven eight nine ten",
            },
        ],
    }


def _install_pyannote(monkeypatch, from_pretrained) -> None:
    pyannote = ModuleType("pyannote")
    pyannote_audio = ModuleType("pyannote.audio")
    pyannote_audio.Pipeline = SimpleNamespace(from_pretrained=from_pretrained)
    pyannote.audio = pyannote_audio
    monkeypatch.setitem(sys.modules, "pyannote", pyannote)
    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_audio)
    monkeypatch.setattr(pyannote_diarizer, "_pipelines", {})


@pytest.fixture
def gated_pyannote(monkeypatch):
    """pyannote.audio 3.x stand-in whose gated download is refused (from_pretrained -> None)."""
    _install_pyannote(monkeypatch, lambda checkpoint, *args, **kwargs: None)


@pytest.mark.asyncio
async def test_refused_gated_diarization_download_tells_the_operator_to_set_hf_token(gated_pyannote):
    # pyannote/speaker-diarization-3.1 is gated and README documents HF_TOKEN for gated
    # downloads; the error must say what to do, not "'NoneType' object is not callable".
    with pytest.raises(HTTPException) as exc_info:
        await _transcribe(diarize=True)

    assert exc_info.value.status_code == 400
    assert "HF_TOKEN" in exc_info.value.detail
    assert "NoneType" not in exc_info.value.detail


@pytest.mark.asyncio
async def test_refused_gated_segmentation_download_tells_the_operator_to_set_hf_token(monkeypatch):
    # speaker-diarization-3.1 also loads pyannote/segmentation-3.0, gated separately.
    # pyannote.audio 3.x returns None for that refusal too, then calls .eval() on it.
    def from_pretrained(checkpoint, *args, **kwargs):
        raise AttributeError("'NoneType' object has no attribute 'eval'")

    _install_pyannote(monkeypatch, from_pretrained)

    with pytest.raises(HTTPException) as exc_info:
        await _transcribe(diarize=True)

    assert exc_info.value.status_code == 400
    assert "HF_TOKEN" in exc_info.value.detail
    assert "pyannote/segmentation-3.0" in exc_info.value.detail
    assert "NoneType" not in exc_info.value.detail


@pytest.mark.asyncio
async def test_diarized_response_is_rendered_off_the_event_loop(monkeypatch):
    # Matching thousands of Whisper segments to speaker turns and rendering subtitles
    # takes a while for a long meeting; on the loop it would stall every live stream,
    # realtime session and Wyoming client.
    seen: list[tuple[str, bool]] = []

    def recorded(name, func):
        def call(*args, **kwargs):
            seen.append((name, _on_event_loop_thread()))
            return func(*args, **kwargs)
        return call

    monkeypatch.setattr(stt_service, "label_transcript_segments",
                        recorded("label", stt_service.label_transcript_segments))
    monkeypatch.setattr(stt_service, "format_transcription",
                        recorded("format", stt_service.format_transcription))
    attach = recorded("attach", pyannote_diarizer.attach_text_to_speakers)

    for response_format in ("json", "srt"):
        await _transcribe(
            backend=_backend(_two_speaker_result()), response_format=response_format, diarize=True,
            diarizer_cls=_TwoSpeakers, attach_speakers_fn=attach,
        )

    assert {name for name, _ in seen} == {"attach", "label", "format"}
    assert [name for name, on_loop in seen if on_loop] == []


# ── Input validation ────────────────────────────────────────────────────────


def _recording_converter(monkeypatch) -> list[bytes]:
    converted: list[bytes] = []

    def convert(audio_bytes, suffix=".ogg"):
        converted.append(audio_bytes)
        return audio_bytes

    monkeypatch.setattr(stt_service, "convert_to_wav", convert)
    return converted


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["english", "xx", "zz-ZZ"])
async def test_unknown_language_is_a_400_before_the_upload_is_decoded(monkeypatch, language):
    # OpenAI's transcription API takes an ISO-639-1 language and answers an unknown one
    # with a 400; faster-whisper only rejects it, as a 500, after decoding the audio.
    converted = _recording_converter(monkeypatch)
    backend = _backend()

    with pytest.raises(HTTPException) as exc_info:
        await _transcribe(backend=backend, language=language)

    assert exc_info.value.status_code == 400
    assert "ISO-639-1" in exc_info.value.detail
    assert converted == []
    backend.transcribe.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(("language", "expected"), [
    ("en", "en"), ("EN", "en"), ("en-US", "en"), ("pt_BR", "pt"), (" fr ", "fr"), ("", None), (None, None),
    # ISO-639-1 codes Whisper spells differently, and the "auto" many clients send.
    ("nb", "no"), ("nb-NO", "no"), ("nn", "nn"), ("iw", "he"), ("jv", "jw"), ("auto", None),
])
async def test_language_reaches_the_backend_as_a_whisper_code(language, expected):
    backend = _backend()

    await _transcribe(backend=backend, language=language)

    assert backend.transcribe.call_args.kwargs["language"] == expected


def test_language_codes_match_faster_whisper():
    tokenizer = pytest.importorskip("faster_whisper.tokenizer")

    assert stt_service.WHISPER_LANGUAGE_CODES == frozenset(tokenizer._LANGUAGE_CODES)


def test_requests_accept_exactly_the_languages_the_model_registry_lists():
    # One list: a model's advertised languages (Wyoming's info) and request validation.
    from src.model_registry import WHISPER_LANGUAGES

    assert stt_service.WHISPER_LANGUAGE_CODES == frozenset(WHISPER_LANGUAGES)
    assert set(stt_service.WHISPER_LANGUAGE_ALIASES.values()) <= stt_service.WHISPER_LANGUAGE_CODES
    assert all(stt_service.whisper_language_code(code) == code for code in WHISPER_LANGUAGES)


def test_language_validation_does_not_import_faster_whisper(monkeypatch):
    # Request handlers validate on the event loop; importing faster_whisper (and av,
    # ctranslate2) there takes about 0.4 s before any STT model has loaded.
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    monkeypatch.setitem(sys.modules, "faster_whisper.tokenizer", None)

    assert stt_service.whisper_language_code("de-AT") == "de"
    with pytest.raises(ValueError, match="ISO-639-1"):
        stt_service.whisper_language_code("english")


@pytest.mark.asyncio
async def test_upload_longer_than_the_limit_is_a_400_before_reaching_the_backend(monkeypatch):
    # A few KB of FLAC at a legitimate 8 kHz can declare hours of silence, which the
    # backend would decode into one 16 kHz float32 array.
    monkeypatch.setattr(audio_module, "MAX_INPUT_AUDIO_SECONDS", 60)
    backend = _backend()

    with pytest.raises(HTTPException) as exc_info:
        await _transcribe(backend=backend, audio=_silent_flac(10 * 60))

    assert exc_info.value.status_code == 400
    assert "longer than 60 seconds" in exc_info.value.detail
    backend.transcribe.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("rate", [1, 4000, 384000])
async def test_upload_with_implausible_sample_rate_is_a_400_before_decoding(monkeypatch, rate):
    # Decoding resamples to 16 kHz: a few KB declared at 1 Hz would become gigabytes.
    converted = _recording_converter(monkeypatch)
    backend = _backend()

    with pytest.raises(HTTPException) as exc_info:
        await _transcribe(backend=backend, audio=_wav_bytes(rate=rate))

    assert exc_info.value.status_code == 400
    assert f"sample rate {rate} Hz" in exc_info.value.detail
    assert converted == []
    backend.transcribe.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("rate", [8000, 44100, 192000])
async def test_upload_at_a_real_sample_rate_is_decoded(monkeypatch, rate):
    converted = _recording_converter(monkeypatch)

    response = await _transcribe(audio=_wav_bytes(rate=rate))

    assert response.status_code == 200
    assert len(converted) == 1


def _header_only_wav(rate: int = 16000) -> bytes:
    """A WAV header with no frames: not 0 bytes, but it decodes to zero samples."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"")
    return buf.getvalue()


def _ffmpeg(monkeypatch, kind: str) -> None:
    """`missing`: no ffmpeg, the original bytes are decoded by the backend; `present`: a stand-in ffmpeg that
    converts the input to a 16 kHz WAV with as many frames as it has (none for a header-only WAV)."""
    import subprocess
    from pathlib import Path

    def run(command, **kwargs):
        if kind == "missing":
            raise FileNotFoundError("ffmpeg")
        with wave.open(io.BytesIO(Path(command[command.index("-i") + 1]).read_bytes()), "rb") as source:
            frames = source.readframes(source.getnframes())
        with wave.open(command[-1], "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(16000)
            out.writeframes(frames)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(audio_module.subprocess, "run", run)


@pytest.mark.asyncio
@pytest.mark.parametrize("ffmpeg", ["missing", "present"])
@pytest.mark.parametrize("operation", ["transcription", "translation"])
async def test_an_upload_that_decodes_to_zero_samples_is_a_400_like_an_empty_one(monkeypatch, ffmpeg, operation):
    # A header-only WAV is not 0 bytes, so it passed the
    # "Empty audio file" check and went on to the model. It gets the same 400 as a 0-byte upload.
    _ffmpeg(monkeypatch, ffmpeg)
    backend = _backend()
    backend.translate.return_value = {"text": "hi"}

    with pytest.raises(HTTPException) as exc_info:
        if operation == "transcription":
            await _transcribe(backend=backend, audio=_header_only_wav())
        else:
            await asyncio.wait_for(stt_service.translate_request(
                file=_upload(_header_only_wav()), model="m", prompt=None, response_format="json",
                temperature=0.0, settings=_settings(), backend_router=backend,
            ), timeout=10)

    assert (exc_info.value.status_code, exc_info.value.detail) == (400, "Empty audio file")
    backend.transcribe.assert_not_called()
    backend.translate.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("ffmpeg", ["missing", "present"])
async def test_a_short_upload_with_samples_still_reaches_the_model(monkeypatch, ffmpeg):
    # Only zero samples are refused; a tenth of a second of silence is transcribed.
    _ffmpeg(monkeypatch, ffmpeg)
    backend = _backend()

    response = await _transcribe(backend=backend, audio=_wav_bytes())

    assert response.status_code == 200
    backend.transcribe.assert_called_once()


@pytest.mark.asyncio
async def test_history_entry_is_written_off_the_event_loop():
    # The insert prunes old entries, which stats retained audio files on disk.
    seen: list[bool] = []
    history_manager = MagicMock()
    history_manager.log_stt.side_effect = lambda **kwargs: seen.append(_on_event_loop_thread())

    await _transcribe(
        settings=_settings(os_history_enabled=True),
        headers={"x-history": "true"},
        history_manager=history_manager,
    )

    history_manager.log_stt.assert_called_once_with(model="m", input_filename="a.wav", result_text="hi")
    assert seen == [False]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["transcribe", "translate"])
async def test_requests_queued_on_the_inference_lock_leave_the_shared_executor_free(monkeypatch, operation):
    # faster-whisper decodes one file at a time. Requests waiting for its inference lock must not
    # each hold a thread of asyncio's default executor, which every other request shares.
    workers = 4
    queued = workers + 2
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=workers))
    gate, inference_lock, counts_lock = threading.Event(), threading.Lock(), threading.Lock()
    counts = {"arrived": 0, "prepared": 0}

    def decode(**_kwargs):
        with counts_lock:
            counts["arrived"] += 1
        with inference_lock:  # the long decode of the first request holds it
            gate.wait(timeout=30)
        return {"text": "hi"}

    backend = MagicMock()
    backend.transcribe.side_effect = decode
    backend.translate.side_effect = decode
    prepare_audio = stt_service.prepare_audio

    def counting_prepare(*args, **kwargs):
        wav = prepare_audio(*args, **kwargs)
        with counts_lock:
            counts["prepared"] += 1
        return wav

    monkeypatch.setattr(stt_service, "prepare_audio", counting_prepare)

    async def request():
        if operation == "transcribe":
            return await stt_service.transcribe_request(
                file=_upload(), model="m", language=None, prompt=None, response_format="json",
                temperature=0.0, diarize=False, raw_request=SimpleNamespace(headers={}),
                settings=_settings(), backend_router=backend, history_manager=MagicMock(),
            )
        return await stt_service.translate_request(
            file=_upload(), model="m", prompt=None, response_format="json",
            temperature=0.0, settings=_settings(), backend_router=backend,
        )

    def settled() -> bool:
        with counts_lock:
            return counts["arrived"] >= workers or (counts["prepared"] >= queued and counts["arrived"] >= 1)

    waiting = [asyncio.create_task(request()) for _ in range(queued)]
    try:
        for _ in range(1000):
            if settled():
                break
            await asyncio.sleep(0.01)
        assert settled(), f"the requests never reached the backend: {counts}"
        await asyncio.sleep(0.3)  # the last ones dispatch their decode
        try:
            bare = await asyncio.wait_for(asyncio.to_thread(lambda: "ran"), timeout=2)
        except TimeoutError:
            bare = "did not run within 2 s"
    finally:
        gate.set()
        responses = await asyncio.wait_for(asyncio.gather(*waiting, return_exceptions=True), timeout=30)

    assert bare == "ran", f"a bare asyncio.to_thread {bare} while {queued} requests waited to decode"
    assert [getattr(response, "status_code", response) for response in responses] == [200] * queued


# A diarized transcription ran PyannoteDiarizer.diarize in run_in_executor(None),
# where it waits for the module-wide _diarize_lock while another request's pyannote run holds it. Each one queued
# behind a long diarization held a thread of asyncio's default executor, which every other request shares.
@pytest.mark.asyncio
async def test_diarizations_queued_behind_a_long_diarization_leave_the_shared_executor_free():
    workers = 4
    queued = workers + 2
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=workers))
    gate, counts_lock = threading.Event(), threading.Lock()
    counts = {"arrived": 0, "decoded": 0}
    turn = SimpleNamespace(start=0.0, end=0.1)

    def pipeline(_path):  # the long pyannote run of the request that holds _diarize_lock
        gate.wait(timeout=30)
        return SimpleNamespace(itertracks=lambda **_kwargs: [(turn, None, "SPEAKER_00")])

    class GatedDiarizer(pyannote_diarizer.PyannoteDiarizer):
        # The real diarize(), with its module-wide _diarize_lock; only the pipeline is a fake.
        def __init__(self) -> None:
            self._pipeline = pipeline
            self._device = "cpu"

        def diarize(self, wav_bytes):
            with counts_lock:
                counts["arrived"] += 1
            return super().diarize(wav_bytes)

    def decode(**_kwargs):
        with counts_lock:
            counts["decoded"] += 1
        return {"text": "hi", "segments": [{"start": 0.0, "end": 0.1, "text": "hi"}]}

    backend = MagicMock()
    backend.transcribe.side_effect = decode

    async def transcription(diarize: bool):
        response = await stt_service.transcribe_request(
            file=_upload(), model="m", language=None, prompt=None, response_format="json",
            temperature=0.0, diarize=diarize, raw_request=SimpleNamespace(headers={}),
            settings=_settings(), backend_router=backend, history_manager=MagicMock(),
            diarizer_cls=GatedDiarizer,
        )
        return response.status_code

    def settled() -> bool:
        # Either `workers` diarizations wait at the lock, or every request was decoded and one diarizes.
        with counts_lock:
            return counts["arrived"] >= workers or (counts["decoded"] >= queued and counts["arrived"] >= 1)

    diarized = [asyncio.create_task(transcription(True)) for _ in range(queued)]
    try:
        for _ in range(1000):
            if settled():
                break
            await asyncio.sleep(0.01)
        assert settled(), f"the diarizations never reached the lock: {counts}"
        await asyncio.sleep(0.3)  # the last ones dispatch their diarization
        try:
            plain = await asyncio.wait_for(transcription(False), timeout=5)
        except TimeoutError:
            plain = "no answer within 5 s"
        try:
            bare = await asyncio.wait_for(asyncio.to_thread(lambda: "ran"), timeout=2)
        except TimeoutError:
            bare = "did not run within 2 s"
    finally:
        gate.set()
        results = await asyncio.wait_for(asyncio.gather(*diarized, return_exceptions=True), timeout=30)

    assert plain == 200, (
        f"a transcription without diarization: {plain} while {queued} diarized ones waited for one long "
        f"diarization, with {workers} default-executor threads ({counts})"
    )
    assert bare == "ran", f"a bare asyncio.to_thread {bare} while {queued} diarizations waited"
    assert [getattr(result, "status_code", result) for result in results] == [200] * queued


# A Whisper model's first-use download ran in the request's own thread of the STT
# pool (two threads), so two requests for models that aren't downloaded (a client retrying a slow first translation,
# or two uncached models) held both threads for the whole 1.5 or 3 GB download, and every transcription on the loaded
# STT_MODEL waited. The first translation now fetches before taking the STT locks, so
# transcriptions, Talk, Wyoming and other STT loads and unloads go on meanwhile; the pool's comment: "One thread
# decodes and one waits for the lock, so the next decode starts at once". Nothing is downloaded: the fake
# faster_whisper's download_model blocks until released, as a hub download (or the hub's per-file lock) does.
_FIRST_USE_STT = "deepdml/faster-whisper-large-v3-turbo-ct2"
_FIRST_USE_MEDIUM = "Systran/faster-whisper-medium"
_FIRST_USE_LARGE = "Systran/faster-whisper-large-v3"


def _install_blocking_whisper(monkeypatch, downloads: list, release: threading.Event) -> None:
    utils = ModuleType("faster_whisper.utils")

    def download_model(size_or_id, output_dir=None, local_files_only=False, cache_dir=None, **kwargs):
        downloads.append(size_or_id)
        release.wait(20)  # a 1.5 GB hub download
        return f"/cache/{size_or_id}"

    class WhisperModel:
        def __init__(self, model_size_or_path, *args, **kwargs):
            pass

        def transcribe(self, path, **kwargs):
            return iter([_first_use_segment(" hola")]), SimpleNamespace(language="es", duration=1.0)

    utils.download_model = download_model
    module = ModuleType("faster_whisper")
    module.utils, module.WhisperModel = utils, WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", utils)


def _first_use_segment(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        seek=0, start=0.0, end=1.0, text=text, tokens=[], temperature=0.0,
        avg_logprob=-0.1, compression_ratio=1.0, no_speech_prob=0.0,
    )


class _LoadedFirstUseWhisper:
    def transcribe(self, path, **kwargs):
        return iter([_first_use_segment(" hello")]), SimpleNamespace(language="en", duration=1.0)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_uses", "fetched"),
    [
        ([_FIRST_USE_MEDIUM, _FIRST_USE_MEDIUM], [_FIRST_USE_MEDIUM]),  # a client retries a slow first translation
        ([_FIRST_USE_MEDIUM, _FIRST_USE_LARGE], [_FIRST_USE_LARGE, _FIRST_USE_MEDIUM]),  # two uncached models
    ],
    ids=["retried-translation", "two-uncached-models"],
)
async def test_first_use_downloads_hold_no_stt_pool_thread(tmp_path, monkeypatch, first_uses, fetched):
    from src.config import settings
    from src.router import BackendRouter

    downloads: list = []
    release = threading.Event()
    _install_blocking_whisper(monkeypatch, downloads, release)
    monkeypatch.setattr(settings, "stt_model_dir", str(tmp_path))  # an empty cache
    router = BackendRouter()
    backend = router.get_backend(_FIRST_USE_STT)
    backend._models[_FIRST_USE_STT] = _LoadedFirstUseWhisper()
    backend._loaded_at[_FIRST_USE_STT] = 0.0

    def translation(model: str):
        return stt_service.translate_request(
            file=_upload(), model=model, prompt=None, response_format="json",
            temperature=0.0, settings=_settings(), backend_router=router,
        )

    translations = [asyncio.ensure_future(translation(model)) for model in first_uses]
    transcription = None
    try:
        for _ in range(500):
            if len(downloads) >= len(fetched):
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.3)  # every first use reaches its fetch
        downloads_during = sorted(downloads)
        transcription = asyncio.ensure_future(stt_service.transcribe_request(
            file=_upload(), model=_FIRST_USE_STT, language=None, prompt=None, response_format="json",
            temperature=0.0, diarize=False, raw_request=SimpleNamespace(headers={}),
            settings=_settings(), backend_router=router, history_manager=MagicMock(),
        ))
        done, _ = await asyncio.wait({transcription}, timeout=5)
        answered = transcription in done
    finally:
        release.set()
    responses = await asyncio.wait_for(
        asyncio.gather(*translations, *([transcription] if transcription is not None else [])), timeout=15,
    )

    assert answered, (
        f"a transcription on the loaded {_FIRST_USE_STT} waited for the first-use downloads of {first_uses}: both "
        "threads of the STT inference pool were held by the fetches"
    )
    # One download per model, however many requests wait for it.
    assert downloads_during == fetched
    assert [json.loads(response.body)["text"] for response in responses] == ["hola"] * len(first_uses) + ["hello"]
    assert all(backend.is_model_loaded(model) for model in first_uses)


@pytest.mark.asyncio
async def test_a_failed_first_use_download_is_still_the_requests_500(tmp_path, monkeypatch):
    # The fetch moved before the STT pool; its error still answers the request.
    from src.config import settings
    from src.router import BackendRouter

    utils = ModuleType("faster_whisper.utils")

    def download_model(size_or_id, **kwargs):
        raise OSError("hub unreachable")

    utils.download_model = download_model
    module = ModuleType("faster_whisper")
    module.utils, module.WhisperModel = utils, object
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", utils)
    monkeypatch.setattr(settings, "stt_model_dir", str(tmp_path))
    router = BackendRouter()

    with pytest.raises(HTTPException) as caught:
        await asyncio.wait_for(stt_service.translate_request(
            file=_upload(), model=_FIRST_USE_MEDIUM, prompt=None, response_format="json",
            temperature=0.0, settings=_settings(), backend_router=router,
        ), timeout=10)

    assert caught.value.status_code == 500
    assert "hub unreachable" in caught.value.detail
    assert not router.is_model_loaded(_FIRST_USE_MEDIUM)


# A request's model id is stat'ed as a local folder before its first-use fetch
# (FasterWhisperBackend.start_fetch's Path(model_id).is_dir()). await_stt_fetch used to make that stat on the event
# loop, so a slow one (a network drive, or on Windows a UNC id such as \\host\share, whose stat is an SMB connection to
# that host offering the server account's NTLM credentials) held every request. The stat now runs off the loop, and a
# request-chosen id that is a network or device path is refused before anything stats it: 400 invalid_model on HTTP,
# NetworkModelPathError from await_stt_fetch (which Wyoming awaits). A local folder is still a model.
_LOADED_STT = "Systran/faster-whisper-tiny"
_NETWORK_MODEL_IDS = [
    r"\\192.0.2.10\share",  # TEST-NET-1; never contacted
    "//192.0.2.10/share",
    r"\/192.0.2.10/share",
    r"\\?\UNC\192.0.2.10\share",
    r"\\.\pipe\open-speech",
    r"\??\UNC\192.0.2.10\share",
    r" \\192.0.2.10\share",
]


def _stat_recorder(monkeypatch, slow_part: str | None = None, seconds: float = 1.0) -> list:
    """Record every Path.is_dir (its path, and whether it ran on the event loop's thread); a path holding
    ``slow_part`` takes ``seconds``, as a network drive's stat does. A network or device path is never really
    stat'ed here, so the test makes no SMB connection whatever the code does."""
    import pathlib
    import time

    calls: list = []
    real_is_dir = pathlib.Path.is_dir

    def is_dir(self, *args, **kwargs):
        calls.append((str(self), _on_event_loop_thread()))
        text = str(self)
        if "192.0.2.10" in text or "pipe" in text or "nas" in text or text[:2] in ("\\\\", "//", "\\/", "/\\"):
            return False
        if slow_part is not None and slow_part in str(self):
            time.sleep(seconds)
            return False
        return real_is_dir(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "is_dir", is_dir)
    return calls


def _router_with_a_loaded_model(monkeypatch, tmp_path):
    from src.backends.faster_whisper import FasterWhisperBackend
    from src.config import settings
    from src.router import BackendRouter

    monkeypatch.setattr(settings, "stt_model_dir", str(tmp_path))
    monkeypatch.setattr(
        FasterWhisperBackend, "_download", lambda self, model_id: (_ for _ in ()).throw(OSError("no hub here")),
    )
    real_loaded = FasterWhisperBackend.is_model_loaded
    monkeypatch.setattr(
        FasterWhisperBackend, "is_model_loaded",
        lambda self, model_id: model_id == _LOADED_STT or real_loaded(self, model_id),
    )

    def decode(self, audio, model, **kwargs):
        return {"text": "ok"}

    monkeypatch.setattr(FasterWhisperBackend, "transcribe", decode)
    monkeypatch.setattr(FasterWhisperBackend, "translate", decode)
    return BackendRouter()


def _stt_request(kind: str, model: str, router):
    if kind == "transcribe":
        return stt_service.transcribe_request(
            file=_upload(), model=model, language=None, prompt=None, response_format="json", temperature=0.0,
            diarize=False, raw_request=SimpleNamespace(headers={}), settings=_settings(), backend_router=router,
            history_manager=MagicMock(),
        )
    return stt_service.translate_request(
        file=_upload(), model=model, prompt=None, response_format="json", temperature=0.0, settings=_settings(),
        backend_router=router,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["transcribe", "translate"])
async def test_a_first_use_models_folder_check_runs_off_the_event_loop(tmp_path, monkeypatch, kind):
    # A model id on a slow disk or network drive is stat'ed in a thread, so the
    # requests for the loaded model go on meanwhile.
    import time

    slow_model = "slow-drive-models/whisper"
    router = _router_with_a_loaded_model(monkeypatch, tmp_path)
    calls = _stat_recorder(monkeypatch, slow_part="slow-drive-models", seconds=1.0)
    await asyncio.wait_for(_stt_request(kind, _LOADED_STT, router), 10)  # warm up

    slow = asyncio.ensure_future(_stt_request(kind, slow_model, router))
    latencies: list[float] = []
    deadline = time.perf_counter() + 10
    while not slow.done() and time.perf_counter() < deadline:
        started = time.perf_counter()
        await asyncio.wait_for(_stt_request(kind, _LOADED_STT, router), 10)
        latencies.append(time.perf_counter() - started)
    with pytest.raises(HTTPException):  # the fake hub fails its download: the request's 500
        await asyncio.wait_for(slow, 10)

    stats = [on_loop for path, on_loop in calls if "slow-drive-models" in path]
    assert stats, "the model id was never checked as a local folder"
    assert not any(stats), f"[{kind}] the model id {slow_model!r} was stat'ed on the event loop thread"
    assert latencies and max(latencies) < 0.5, (
        f"[{kind}] a request for the loaded model waited {max(latencies or [0]):.2f} s behind the stat"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["transcribe", "translate"])
@pytest.mark.parametrize("model", _NETWORK_MODEL_IDS)
async def test_a_network_or_device_model_path_is_a_400_before_anything_stats_it(
    tmp_path, monkeypatch, kind, model,
):
    # On Windows a stat of \\host\share connects to that host over SMB.
    router = _router_with_a_loaded_model(monkeypatch, tmp_path)
    calls = _stat_recorder(monkeypatch)
    converted = _recording_converter(monkeypatch)

    with pytest.raises(HTTPException) as caught:
        await asyncio.wait_for(_stt_request(kind, model, router), 10)

    assert caught.value.status_code == 400, caught.value.detail
    assert caught.value.detail["code"] == "invalid_model"
    assert "network" in caught.value.detail["message"]
    assert not [path for path, _ in calls if "192.0.2.10" in path or "pipe" in path], calls
    assert converted == [], "the upload was decoded before the model id was refused"


@pytest.mark.asyncio
@pytest.mark.parametrize("model", _NETWORK_MODEL_IDS)
async def test_await_stt_fetch_refuses_a_network_or_device_model_path(monkeypatch, model):
    # Wyoming awaits await_stt_fetch before its decode, so the
    # refusal is here too, and in the backend's own fetch for every other caller (streaming, realtime, legacy load).
    from src.backends.faster_whisper import FasterWhisperBackend

    calls = _stat_recorder(monkeypatch)
    router = MagicMock()
    router.is_model_loaded.return_value = False

    with pytest.raises(ValueError) as caught:
        await asyncio.wait_for(stt_service.await_stt_fetch(router, model), 5)
    assert type(caught.value).__name__ == "NetworkModelPathError"
    assert caught.value.code == "invalid_model"
    router.start_fetch.assert_not_called()

    with pytest.raises(ValueError) as caught:
        FasterWhisperBackend().start_fetch(model)
    assert type(caught.value).__name__ == "NetworkModelPathError"
    assert calls == []


@pytest.mark.asyncio
async def test_a_local_model_folder_and_a_configured_network_model_still_work(tmp_path, monkeypatch):
    # Only a request's own choice is refused. A local folder is a model as it was
    # before, and a share the operator configured as STT_MODEL is the operator's choice.
    from src.backends.faster_whisper import FasterWhisperBackend
    from src.config import settings

    folder = tmp_path / "my-whisper"
    folder.mkdir()
    backend = FasterWhisperBackend()
    future = await asyncio.to_thread(backend.start_fetch, str(folder))
    assert future.result(timeout=1) == str(folder)

    router = MagicMock()
    router.is_model_loaded.return_value = False
    router.start_fetch.return_value = None
    await asyncio.wait_for(stt_service.await_stt_fetch(router, str(folder)), 5)
    router.start_fetch.assert_called_once_with(str(folder))

    configured = r"\\nas\models\whisper"
    monkeypatch.setattr(settings, "stt_model", configured)
    router.start_fetch.reset_mock()
    await asyncio.wait_for(stt_service.await_stt_fetch(router, configured), 5)
    router.start_fetch.assert_called_once_with(configured)


@pytest.mark.parametrize("operation", ["load_legacy_model", "pull_legacy_model"])
def test_the_legacy_load_and_pull_refuse_a_network_model_path_before_unloading(operation):
    # POST /api/ps/{model} and /api/pull/{model} take any id; the load unloaded
    # the model in use before its fetch stat'ed the share.
    from src.services import models as model_service

    router = MagicMock()
    router.loaded_models.return_value = [SimpleNamespace(model=_LOADED_STT)]

    with pytest.raises(HTTPException) as caught:
        getattr(model_service, operation)(model=r"\\192.0.2.10\share", backend_router=router)

    assert caught.value.status_code == 400
    assert caught.value.detail["code"] == "invalid_model"
    router.unload_model.assert_not_called()
    router.load_model.assert_not_called()
    router.download_model.assert_not_called()
