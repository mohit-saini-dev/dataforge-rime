"""
Unit tests for FencedRimeTTS client.
Validates chunk sizing, latency timing, generation fence cutoffs, and CancelledError propagation.
"""

import asyncio
import pytest

from backend.tts.rime_plugin import FencedRimeTTS


@pytest.mark.asyncio
async def test_tts_chunk_byte_duration():
    tts = FencedRimeTTS(sample_rate=24000)
    assert tts.frame_bytes == 960  # Exactly 20ms of 16-bit mono audio @ 24kHz

    chunks = []
    async for chunk in tts.stream_speech(text="Flight FL-101 confirmed", turn_id=1):
        chunks.append(chunk)

    assert len(chunks) > 0
    assert all(len(c) == 960 for c in chunks)


@pytest.mark.asyncio
async def test_tts_pre_execution_fence_abort():
    tts = FencedRimeTTS()

    def expired_fence(turn_id: int) -> bool:
        return False

    chunks = []
    async for chunk in tts.stream_speech(
        text="Hello world", turn_id=1, fence_validator=expired_fence
    ):
        chunks.append(chunk)

    assert len(chunks) == 0


@pytest.mark.asyncio
async def test_tts_mid_stream_interruption():
    tts = FencedRimeTTS()
    turn_active = True

    def dynamic_fence(turn_id: int) -> bool:
        return turn_active

    chunks = []
    async for chunk in tts.stream_speech(
        text="This is a long sentence meant to be cut off mid-playback by user speech",
        turn_id=1,
        fence_validator=dynamic_fence,
        chunk_latency_ms=10,
    ):
        chunks.append(chunk)
        if len(chunks) == 3:
            turn_active = False  # Simulate user barge-in after 3 frames

    # Stream should cut off immediately after chunk 3
    assert len(chunks) == 3


@pytest.mark.asyncio
async def test_tts_task_cancellation_propagation():
    tts = FencedRimeTTS()

    async def run_tts():
        async for _ in tts.stream_speech("Long synthesized response", turn_id=1, chunk_latency_ms=50):
            pass

    task = asyncio.create_task(run_tts())
    await asyncio.sleep(0.02)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task