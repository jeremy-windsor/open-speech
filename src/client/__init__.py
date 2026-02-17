from __future__ import annotations

import asyncio
from typing import AsyncIterator, Iterator

import httpx


class OpenSpeechClient:
    def __init__(self, base_url: str = "http://localhost:8100", api_key: str | None = None, ssl_verify: bool = True):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.ssl_verify = ssl_verify

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def transcribe(self, audio: bytes, model: str = "") -> dict:
        with httpx.Client(verify=self.ssl_verify, headers=self._headers(), timeout=60) as c:
            files = {"file": ("audio.wav", audio, "audio/wav")}
            data = {"model": model} if model else {}
            r = c.post(f"{self.base_url}/v1/audio/transcriptions", files=files, data=data)
            r.raise_for_status()
            return r.json()

    def speak(self, text: str, voice: str = "alloy", speed: float = 1.0, model: str = "kokoro", response_format: str = "mp3") -> bytes:
        with httpx.Client(verify=self.ssl_verify, headers=self._headers(), timeout=60) as c:
            r = c.post(f"{self.base_url}/v1/audio/speech", json={
                "model": model,
                "input": text,
                "voice": voice,
                "speed": speed,
                "response_format": response_format,
            })
            r.raise_for_status()
            return r.content

    async def async_transcribe(self, audio: bytes, model: str = "") -> dict:
        async with httpx.AsyncClient(verify=self.ssl_verify, headers=self._headers(), timeout=60) as c:
            files = {"file": ("audio.wav", audio, "audio/wav")}
            data = {"model": model} if model else {}
            r = await c.post(f"{self.base_url}/v1/audio/transcriptions", files=files, data=data)
            r.raise_for_status()
            return r.json()

    async def async_speak(self, text: str, voice: str = "alloy", speed: float = 1.0, model: str = "kokoro", response_format: str = "mp3") -> bytes:
        async with httpx.AsyncClient(verify=self.ssl_verify, headers=self._headers(), timeout=60) as c:
            r = await c.post(f"{self.base_url}/v1/audio/speech", json={
                "model": model,
                "input": text,
                "voice": voice,
                "speed": speed,
                "response_format": response_format,
            })
            r.raise_for_status()
            return r.content

    def stream_transcribe(self, audio_stream: Iterator[bytes]):
        # Placeholder interface for future WS protocol streaming support
        for chunk in audio_stream:
            yield {"event": "audio_chunk", "bytes": len(chunk)}

    async def realtime_session(self) -> AsyncIterator[dict]:
        # Placeholder async generator for realtime WS management
        yield {"event": "session.started"}
        await asyncio.sleep(0)
        yield {"event": "session.ended"}
