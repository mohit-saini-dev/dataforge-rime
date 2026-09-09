from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, AsyncGenerator, Callable, Optional, Union

import aiohttp

try:
    from livekit import rtc
    HAS_LIVEKIT = True
except ImportError:
    rtc = None
    HAS_LIVEKIT = False

logger = logging.getLogger("backend.tts.rime")

SyncFenceValidator = Callable[[int], bool]


class TurnInterrupted(Exception):
    pass


class RimeTTSHTTPError(RuntimeError):
    def __init__(self, status: int, body: str = "") -> None:
        self.status = status
        self.body = body
        super().__init__(f"Rime HTTP {status}: {body[:500]}")


class RimeTTSProtocolError(RuntimeError):
    pass


class RimeTTSNetworkError(RuntimeError):
    pass


class PCMFramer:
    """Converts arbitrary s16le byte chunks into exact 10ms (480 bytes) PCM frames."""

    def __init__(self, frame_bytes: int = 480) -> None:
        if frame_bytes <= 0 or frame_bytes % 2 != 0:
            raise ValueError("frame_bytes must be a positive even integer")
        self.frame_bytes = frame_bytes
        self._buffer = bytearray()

    def push(self, data: bytes) -> list[bytes]:
        if not data:
            return []
        self._buffer.extend(data)
        frames: list[bytes] = []
        while len(self._buffer) >= self.frame_bytes:
            frames.append(bytes(self._buffer[: self.frame_bytes]))
            del self._buffer[: self.frame_bytes]
        return frames

    def flush(self) -> list[bytes]:
        """Emits zero-padded final frame for normal completion."""
        if not self._buffer:
            return []

        if len(self._buffer) % 2 != 0:
            self._buffer.clear()
            raise RimeTTSProtocolError("Odd trailing PCM byte count returned by Rime")

        frame = bytes(self._buffer).ljust(self.frame_bytes, b"\x00")
        self._buffer.clear()
        return [frame]


class FencedRimeTTS:
    URL = "https://users.rime.ai/v1/rime-tts"
    MAX_TEXT_CHARS = 500
    FRAME_DURATION_SEC = 0.010

    def __init__(
        self,
        speaker: str = "amber",
        model: str = "mist",
        sample_rate: int = 24000,
        api_key: Optional[str] = None,
        speed_alpha: float = 0.95,
    ) -> None:
        self.speaker = speaker
        self.model = model
        self.sample_rate = sample_rate
        self.api_key = api_key
        self.speed_alpha = speed_alpha

        self.samples_per_channel = int(sample_rate * self.FRAME_DURATION_SEC)
        self.frame_bytes = self.samples_per_channel * 2

        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        self._closed = False

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._closed:
            raise RuntimeError("Rime TTS client is closed")
        if self._session is not None and not self._session.closed:
            return self._session

        async with self._session_lock:
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(
                    total=None,
                    connect=3.0,
                    sock_connect=3.0,
                    sock_read=15.0,  # Tolerant read timeout
                )
                connector = aiohttp.TCPConnector(
                    limit=8,
                    limit_per_host=4,
                    ttl_dns_cache=300,
                    keepalive_timeout=60.0,
                )
                self._session = aiohttp.ClientSession(
                    timeout=timeout,
                    connector=connector,
                )
        return self._session

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r"[`*_#|~<>]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def create_audio_frame(self, data: bytes) -> Any:
        if len(data) != self.frame_bytes:
            raise RimeTTSProtocolError(f"PCM frame length {len(data)} != {self.frame_bytes}")
        if HAS_LIVEKIT and rtc is not None:
            return rtc.AudioFrame(
                data=bytearray(data),
                sample_rate=self.sample_rate,
                num_channels=1,
                samples_per_channel=self.samples_per_channel,
            )
        return data

    async def stream_speech(
        self,
        text: str,
        turn_id: int,
        fence_validator: Optional[SyncFenceValidator] = None,
        framer: Optional[PCMFramer] = None,
        as_audio_frame: bool = False,
    ) -> AsyncGenerator[Union[bytes, Any], None]:
        clean_text = self.clean_text(text)[: self.MAX_TEXT_CHARS]
        if not clean_text:
            return

        if not self.api_key:
            raise RuntimeError("RIME_API_KEY is not configured")

        if fence_validator is not None and not fence_validator(turn_id):
            raise TurnInterrupted()

        local_framer = framer if framer is not None else PCMFramer(self.frame_bytes)

        payload = {
            "text": clean_text,
            "modelId": self.model,
            "speaker": self.speaker,
            "samplingRate": self.sample_rate,
            "speedAlpha": self.speed_alpha,
            "audioFormat": "pcm",
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "audio/L16",
        }

        session = await self._get_session()
        response: Optional[aiohttp.ClientResponse] = None

        try:
            response = await session.post(self.URL, json=payload, headers=headers)
            async with response:
                if response.status != 200:
                    body = await response.text()
                    raise RimeTTSHTTPError(response.status, body)

                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                if content_type not in {"audio/l16", "audio/pcm", "application/octet-stream"}:
                    raise RimeTTSProtocolError(f"Unexpected Rime content type: {content_type!r}")

                async for chunk in response.content.iter_any():
                    if fence_validator is not None and not fence_validator(turn_id):
                        response.close()
                        raise TurnInterrupted()

                    for frame in local_framer.push(chunk):
                        if fence_validator is not None and not fence_validator(turn_id):
                            response.close()
                            raise TurnInterrupted()
                        yield self.create_audio_frame(frame) if as_audio_frame else frame

                if framer is None:
                    for frame in local_framer.flush():
                        if fence_validator is not None and not fence_validator(turn_id):
                            raise TurnInterrupted()
                        yield self.create_audio_frame(frame) if as_audio_frame else frame

        except asyncio.CancelledError:
            if response is not None and not response.closed:
                response.close()
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if response is not None and not response.closed:
                response.close()
            raise RimeTTSNetworkError(f"Rime network failure on epoch {turn_id}: {exc}") from exc
        finally:
            if response is not None and not response.closed:
                response.close()

    async def aclose(self) -> None:
        self._closed = True
        async with self._session_lock:
            if self._session is not None and not self._session.closed:
                await self._session.close()
            self._session = None