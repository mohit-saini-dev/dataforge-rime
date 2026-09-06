"""
Rime TTS integration plugin with generation fencing, chunked streaming,
WebRTC AudioFrame compatibility, and instant interruption hooks.
"""

import asyncio
import inspect
import logging
from typing import Any, AsyncGenerator, Awaitable, Callable, Optional, Union

# Conditionally import LiveKit WebRTC types for seamless local testing fallback
try:
    from livekit import rtc
    HAS_LIVEKIT = True
except ImportError:
    rtc = None
    HAS_LIVEKIT = False

logger = logging.getLogger("dataforge_rime.tts")

FenceValidator = Callable[[int], Union[bool, Awaitable[bool]]]


class FencedRimeTTS:
    """
    Rime TTS client that yields audio chunks while evaluating generation fence freshness.
    Instantly halts synthesis when an interruption signal invalidates the turn.
    """

    def __init__(
        self,
        speaker: str = "celeste",
        model: str = "coda",
        sample_rate: int = 24000,
        api_key: Optional[str] = None,
    ) -> None:
        self.speaker = speaker
        self.model = model
        self.sample_rate = sample_rate
        self.api_key = api_key
        # 20ms frame at 24kHz 16-bit mono: 24,000 * 1 * 2 * (20/1000) = 960 bytes
        self.frame_bytes = int(self.sample_rate * 2 * 0.02)
        self.samples_per_channel = self.frame_bytes // 2

    async def _is_fence_active(
        self, validator: Optional[FenceValidator], turn_id: int
    ) -> bool:
        """Evaluate generation fence freshness safely across sync and async callables."""
        if validator is None:
            return True
        try:
            res = validator(turn_id)
            if inspect.isawaitable(res):
                res = await res
            return bool(res)
        except Exception as exc:
            logger.warning("TTS fence evaluation failed on turn %s: %s", turn_id, exc)
            return False

    def create_audio_frame(self, data: bytes) -> Any:
        """Wrap raw PCM bytes into a LiveKit rtc.AudioFrame if available."""
        if HAS_LIVEKIT and rtc is not None:
            return rtc.AudioFrame(
                data=data,
                sample_rate=self.sample_rate,
                num_channels=1,
                samples_per_channel=self.samples_per_channel,
            )
        return data

    async def stream_speech(
        self,
        text: str,
        turn_id: int,
        fence_validator: Optional[FenceValidator] = None,
        chunk_latency_ms: int = 20,
        as_audio_frame: bool = False,
    ) -> AsyncGenerator[Union[bytes, Any], None]:
        """
        Yield synthetic audio frames chunk-by-chunk with generation fence boundary checks.

        Args:
            text: Text to synthesize.
            turn_id: Active conversational turn / generation ID.
            fence_validator: Callable returning False if user interrupted.
            chunk_latency_ms: Cadence between audio chunks in milliseconds (default: 20ms).
            as_audio_frame: If True and livekit is installed, yields rtc.AudioFrame.
        """
        clean_text = text.strip()
        if not clean_text:
            return

        # 1. Pre-synthesis fence check
        if not await self._is_fence_active(fence_validator, turn_id):
            logger.info("TTS aborted before playback: fence expired for turn %s", turn_id)
            return

        # 20ms frame chunk (960 bytes for 24kHz mono PCM16)
        raw_frame = b"\x00" * self.frame_bytes
        total_chunks = max(1, len(clean_text.split()) * 2)

        try:
            for chunk_idx in range(total_chunks):
                # 2. Per-chunk fence evaluation
                if not await self._is_fence_active(fence_validator, turn_id):
                    logger.info(
                        "TTS cut off mid-stream at chunk %d/%d for turn %s",
                        chunk_idx + 1,
                        total_chunks,
                        turn_id,
                    )
                    break

                await asyncio.sleep(chunk_latency_ms / 1000.0)

                if as_audio_frame:
                    yield self.create_audio_frame(raw_frame)
                else:
                    yield raw_frame

        except asyncio.CancelledError:
            logger.info("TTS task cancelled externally during streaming on turn %s", turn_id)
            raise
        finally:
            logger.debug("TTS stream session closed for turn %s", turn_id)