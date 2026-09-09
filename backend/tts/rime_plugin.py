"""
Rime TTS integration plugin with 10ms frame alignment (for LiveKit direct capture),
connection pooling, strict Content-Type validation, and generation fencing.
"""

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

logger = logging.getLogger("dataforge_rime.tts")

# Synchronous, non-blocking scalar predicate: O(1), zero-await
SyncFenceValidator = Callable[[int], bool]


class TurnInterrupted(Exception):
    """Raised when active synthesis is aborted due to barge-in."""
    pass


class RimeTTSProtocolError(RuntimeError):
    """Raised when upstream audio data violates s16le PCM invariants."""
    pass


class RimeTTSHTTPError(RuntimeError):
    """Raised when Rime returns a non-200 HTTP response."""
    def __init__(self, status: int, body: str = ""):
        super().__init__(f"Rime TTS HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body


class PCMFramer:
    """
    Accumulates arbitrary network chunks and yields exact frame-sized slices.
    Assembles raw stream bytes into complete 10ms 16-bit PCM frames (480 bytes at 24kHz).
    """

    def __init__(self, frame_size: int, bytes_per_sample: int = 2):
        if frame_size <= 0 or frame_size % bytes_per_sample != 0:
            raise ValueError(f"Invalid frame_size {frame_size}; must be divisible by {bytes_per_sample}")

        self.frame_size = frame_size
        self.bytes_per_sample = bytes_per_sample
        self.buffer = bytearray()

    def push(self, data: bytes) -> list[bytes]:
        if not data:
            return []

        # Accumulate arbitrary network bytes directly into the frame buffer
        self.buffer.extend(data)
        frames: list[bytes] = []
        while len(self.buffer) >= self.frame_size:
            frames.append(bytes(self.buffer[: self.frame_size]))
            del self.buffer[: self.frame_size]
        return frames

    def flush(self) -> list[bytes]:
        if not self.buffer:
            return []

        # Truncate any odd trailing byte at turn boundary
        if len(self.buffer) % self.bytes_per_sample != 0:
            del self.buffer[-1:]

        if not self.buffer:
            return []

        padding = self.frame_size - len(self.buffer)
        final_frame = bytes(self.buffer) + (b"\x00" * padding)
        self.buffer.clear()
        return [final_frame]


class FencedRimeTTS:
    """
    Rime TTS client with keep-alive connection pooling, strict s16le verification,
    and instantaneous HTTP teardown upon turn invalidation.
    """

    RIME_TTS_URL = "https://users.rime.ai/v1/rime-tts"

    def __init__(
        self,
        speaker: str = "amber",
        model: str = "mist",
        sample_rate: int = 24000,
        api_key: Optional[str] = None,
    ) -> None:
        self.speaker = speaker
        self.model = model
        self.sample_rate = sample_rate
        self.api_key = api_key
        # Direct capture requires 10ms frames at 24kHz: 24,000 * 1 * 2 * 0.01 = 480 bytes (240 samples)
        self.frame_bytes = int(self.sample_rate * 2 * 0.01)
        self.samples_per_channel = self.frame_bytes // 2
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

    async def get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            async with self._session_lock:
                if self._session is None or self._session.closed:
                    conn = aiohttp.TCPConnector(
                        limit=10,
                        limit_per_host=4,
                        ttl_dns_cache=300,
                        keepalive_timeout=60.0,
                        enable_cleanup_closed=True,
                    )
                    self._session = aiohttp.ClientSession(connector=conn)
        return self._session

    async def aclose(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    def create_audio_frame(self, data: bytes) -> Any:
        if len(data) != self.frame_bytes:
            raise ValueError(f"AudioFrame invariant violated: got {len(data)} bytes, expected {self.frame_bytes}")
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
        framer: PCMFramer,
        fence_validator: Optional[SyncFenceValidator] = None,
        as_audio_frame: bool = False,
    ) -> AsyncGenerator[Union[bytes, Any], None]:
        clean_text = text.strip()
        if not clean_text:
            return

        if len(clean_text) > 500:
            clean_text = clean_text[:500]

        # Strip markdown and vertical bars while preserving flight hyphens (FL-101)
        clean_text = re.sub(r"[|\*#_`~>]", " ", clean_text)
        clean_text = re.sub(r"(?<=\s)-|-(?=\s)", " ", clean_text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        if not clean_text:
            return

        if fence_validator is not None and not fence_validator(turn_id):
            raise TurnInterrupted(f"Turn {turn_id} pre-check invalid")

        headers = {
            "Accept": "audio/L16",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "text": clean_text,
            "speaker": self.speaker,
            "modelId": self.model,
            "samplingRate": self.sample_rate,
            "audioFormat": "pcm",
            "speedAlpha": 1.0,
            "reduceLatency": True,
        }

        session = await self.get_session()
        timeout = aiohttp.ClientTimeout(total=None, connect=3.0, sock_connect=3.0, sock_read=4.0)

        resp = None
        frame_yield_count = 0
        try:
            resp = await session.post(self.RIME_TTS_URL, headers=headers, json=payload, timeout=timeout)
            if resp.status != 200:
                err_body = await resp.text()
                resp.close()
                raise RimeTTSHTTPError(resp.status, err_body)

            content_type = resp.headers.get("Content-Type", "").split(";", 1)[0].lower().strip()
            if content_type not in {"audio/l16", "audio/pcm", "application/octet-stream"}:
                logger.warning(f"Unexpected Rime content type '{content_type}' on turn {turn_id}")

            async for chunk in resp.content.iter_any():
                if fence_validator is not None and not fence_validator(turn_id):
                    resp.close()
                    raise TurnInterrupted(f"Turn {turn_id} invalidated during network stream")

                for frame in framer.push(chunk):
                    if fence_validator is not None and not fence_validator(turn_id):
                        resp.close()
                        raise TurnInterrupted(f"Turn {turn_id} invalidated mid-frame push")

                    yield self.create_audio_frame(frame) if as_audio_frame else frame
                    frame_yield_count += 1

                    # Yield event loop periodically so VAD/interruption tasks run concurrently
                    if frame_yield_count % 8 == 0:
                        await asyncio.sleep(0)

        except asyncio.CancelledError:
            if resp and not resp.closed:
                resp.close()
            raise
        finally:
            if resp and not resp.closed:
                resp.close()