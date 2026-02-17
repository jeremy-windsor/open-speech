"""Wyoming Protocol TCP server for Open Speech.

Runs alongside the FastAPI HTTP server on a separate port (default 10400).
Implements both STT (ASR) and TTS capabilities so Home Assistant can use
Open Speech as a drop-in voice provider.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from wyoming.audio import AudioChunk, AudioStop
from wyoming.asr import Transcribe, Transcript
from wyoming.event import Event
from wyoming.info import (
    AsrModel,
    AsrProgram,
    Attribution,
    Describe,
    Info,
    TtsProgram,
    TtsVoice,
)
from wyoming.server import AsyncEventHandler, AsyncTcpServer
from wyoming.tts import Synthesize

from src.config import settings
from src.wyoming.stt_handler import handle_transcribe
from src.wyoming.tts_handler import handle_synthesize

if TYPE_CHECKING:
    from src.router import STTRouter
    from src.tts.router import TTSRouter

logger = logging.getLogger(__name__)

# Module-level refs set by start_wyoming_server
_stt_router: STTRouter | None = None
_tts_router: TTSRouter | None = None


class OpenSpeechEventHandler(AsyncEventHandler):
    """Handles a single Wyoming TCP connection."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        stt_router: STTRouter,
        tts_router: TTSRouter,
        cli_info: Info,
    ) -> None:
        super().__init__(reader, writer)
        self.stt_router = stt_router
        self.tts_router = tts_router
        self.cli_info = cli_info
        self._audio_chunks: list[bytes] = []
        self._audio_rate: int = 16000
        self._audio_width: int = 2
        self._audio_channels: int = 1
        self._transcribe_model: str | None = None
        self._transcribe_language: str | None = None

    async def handle_event(self, event: Event) -> bool:
        """Process a Wyoming event. Return True to keep connection open."""
        if Describe.is_type(event.type):
            await self.write_event(self.cli_info.event())
            return True

        if Transcribe.is_type(event.type):
            transcribe = Transcribe.from_event(event)
            self._transcribe_model = getattr(transcribe, "name", None)
            self._transcribe_language = getattr(transcribe, "language", None)
            self._audio_chunks = []
            return True

        if AudioChunk.is_type(event.type):
            chunk = AudioChunk.from_event(event)
            self._audio_rate = chunk.rate
            self._audio_width = chunk.width
            self._audio_channels = chunk.channels
            self._audio_chunks.append(chunk.audio)
            return True

        if AudioStop.is_type(event.type):
            if self._audio_chunks:
                text = await handle_transcribe(
                    audio_chunks=self._audio_chunks,
                    rate=self._audio_rate,
                    width=self._audio_width,
                    channels=self._audio_channels,
                    stt_router=self.stt_router,
                    model=self._transcribe_model,
                    language=self._transcribe_language,
                )
                await self.write_event(Transcript(text=text).event())
                self._audio_chunks = []
            return True

        if Synthesize.is_type(event.type):
            synthesize = Synthesize.from_event(event)
            voice_name = None
            if synthesize.voice and synthesize.voice.name:
                voice_name = synthesize.voice.name

            await handle_synthesize(
                text=synthesize.text,
                voice=voice_name,
                tts_router=self.tts_router,
                write_event=self.write_event,
            )
            return True

        logger.debug("Unhandled Wyoming event type: %s", event.type)
        return True


def build_info(stt_router: STTRouter, tts_router: TTSRouter) -> Info:
    """Build Wyoming Info describing our STT + TTS capabilities."""

    attribution = Attribution(name="Open Speech", url="https://github.com/will-assistant/open-speech")

    # ASR (STT) programs
    stt_models = []
    loaded = stt_router.loaded_models()
    loaded_ids = {m.model for m in loaded}
    for m in loaded:
        stt_models.append(AsrModel(
            name=m.model,
            attribution=attribution,
            installed=True,
            description=f"STT model ({m.backend})",
            version=None,
            languages=["en"],
        ))
    if settings.stt_model not in loaded_ids:
        stt_models.append(AsrModel(
            name=settings.stt_model,
            attribution=attribution,
            installed=True,
            description="Default STT model",
            version=None,
            languages=["en"],
        ))

    asr_programs = [
        AsrProgram(
            name="open-speech",
            attribution=attribution,
            installed=True,
            description="Open Speech STT",
            version=None,
            models=stt_models,
        )
    ]

    # TTS programs
    tts_voices = []
    if settings.tts_enabled:
        voices = tts_router.list_voices()
        for v in voices:
            langs = [v.language] if v.language else ["en"]
            tts_voices.append(TtsVoice(
                name=v.id,
                attribution=attribution,
                installed=True,
                description=v.name,
                version=None,
                languages=langs,
            ))
        if not tts_voices:
            tts_voices.append(TtsVoice(
                name=settings.tts_voice,
                attribution=attribution,
                installed=True,
                description="Default voice",
                version=None,
                languages=["en"],
            ))

    tts_programs = [
        TtsProgram(
            name="open-speech",
            attribution=attribution,
            installed=True,
            description="Open Speech TTS",
            version=None,
            voices=tts_voices,
        )
    ] if settings.tts_enabled else []

    return Info(asr=asr_programs, tts=tts_programs)


async def start_wyoming_server(
    host: str,
    port: int,
    stt_router: STTRouter,
    tts_router: TTSRouter,
) -> asyncio.Task:
    """Start the Wyoming TCP server as an asyncio task. Returns the task."""
    info = build_info(stt_router, tts_router)

    server = AsyncTcpServer(host, port)

    def handler_factory(reader, writer):
        return OpenSpeechEventHandler(reader, writer, stt_router, tts_router, info)

    task = asyncio.create_task(_run_server(server, handler_factory), name="wyoming-server")
    logger.info("Wyoming protocol server started on %s:%d", host, port)
    return task


async def _run_server(server: AsyncTcpServer, handler_factory) -> None:
    """Run the server (wraps serve_forever)."""
    try:
        await server.run(handler_factory)
    except asyncio.CancelledError:
        logger.info("Wyoming server shutting down")
    except Exception:
        logger.exception("Wyoming server error")
