"""
Deterministic integration tests for VoiceAgentOrchestrator.
Validates barge-in cancellation, task ownership, zero-leak cleanup,
monotonic concurrency, and tool fencing with registry integration.
"""

import asyncio
import inspect
from typing import AsyncGenerator
import pytest

from backend.agent import VoiceAgentOrchestrator
from backend.tools.mock_tools import _db
from backend.tools import registry


class GatedTTS:
    """Mock TTS client using async events to synchronize deterministically without wall-clock sleeps."""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.frame_bytes = int(sample_rate * 2 * 0.02)
        self.first_chunk_emitted = asyncio.Event()
        self.gate = asyncio.Event()
        self.clean_exit = False

    async def stream_speech(
        self,
        text: str,
        turn_id: int,
        fence_validator=None,
        chunk_latency_ms: int = 20,
    ) -> AsyncGenerator[bytes, None]:
        if not text.strip():
            return

        frame = b"\x00" * self.frame_bytes

        if fence_validator:
            val = fence_validator(turn_id)
            is_active = await val if inspect.isawaitable(val) else val
            if not is_active:
                return

        # Emit chunk 1
        yield frame
        self.first_chunk_emitted.set()

        # Deterministically wait for gate release
        await self.gate.wait()

        if fence_validator:
            val = fence_validator(turn_id)
            is_active = await val if inspect.isawaitable(val) else val
            if not is_active:
                return

        # Chunk 2 should only yield if generation is still active
        yield frame
        self.clean_exit = True


@pytest.fixture(autouse=True)
def setup_db():
    if hasattr(_db, "reset"):
        _db.reset()


@pytest.mark.asyncio
async def test_deterministic_barge_in_and_task_finalization():
    """Validates that barge-in halts playback instantly, rejects stale frames, and finishes the task."""
    gated_tts = GatedTTS()
    orchestrator = VoiceAgentOrchestrator(tts_client=gated_tts)

    gen_res = orchestrator.turn_controller.start_generation("Flight status inquiry")
    turn_1 = await gen_res if inspect.isawaitable(gen_res) else gen_res

    received_frames = []

    async def consumer():
        try:
            async for chunk in orchestrator.stream_agent_reply(
                text="Confirming your flight ticket to London Heathrow",
                turn_id=turn_1,
            ):
                received_frames.append(chunk)
        except asyncio.CancelledError:
            pass

    consumer_task = asyncio.create_task(consumer())

    # Wait for first chunk
    await asyncio.wait_for(gated_tts.first_chunk_emitted.wait(), timeout=1.0)
    assert len(received_frames) == 1
    assert orchestrator._current_tts_task is consumer_task

    # Barge-in cancels consumer_task and advances generation
    turn_2 = await orchestrator.handle_user_barge_in("Stop, wait a second")
    assert turn_2 > turn_1

    # Unblock gate
    gated_tts.gate.set()

    # Consumer task must be completed/cancelled
    await asyncio.wait_for(consumer_task, timeout=1.0)

    assert consumer_task.done()
    assert orchestrator._current_tts_task is None
    assert len(received_frames) == 1
    assert not gated_tts.clean_exit


@pytest.mark.asyncio
async def test_concurrent_double_barge_in_deadlock_free():
    """Validates that rapid concurrent barge-ins order IDs monotonically without deadlocking."""
    orchestrator = VoiceAgentOrchestrator()

    async def fire_barge_in(index: int):
        return await orchestrator.handle_user_barge_in(f"Interruption {index}")

    tasks = [fire_barge_in(i) for i in range(5)]
    results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=1.0)

    assert len(set(results)) == 5
    assert results == sorted(results)


@pytest.mark.asyncio
async def test_orchestrator_executes_registered_tool():
    """Validates execution of registered tool through orchestrator."""
    orchestrator = VoiceAgentOrchestrator()
    gen_res = orchestrator.turn_controller.start_generation("Book flight FL-101")
    turn_id = await gen_res if inspect.isawaitable(gen_res) else gen_res

    result = await orchestrator.execute_tool_call(
        tool_name="book_flight",
        arguments={
            "flight_id": "FL-101",
            "passenger_name": "Alice Cooper",
        },
        turn_id=turn_id,
    )

    actual = result.get("result", result)
    assert actual.get("status") == "success"
    booking = actual.get("booking", actual)
    assert booking.get("flight_id") == "FL-101"


@pytest.mark.asyncio
async def test_stale_tool_call_rejected_after_barge_in():
    """Validates that a tool executing when barge-in occurs is rejected by the fence."""
    tool_started = asyncio.Event()
    tool_gate = asyncio.Event()

    async def slow_mock_tool(flight_id: str):
        tool_started.set()
        await tool_gate.wait()
        return {"status": "success", "flight_id": flight_id}

    registry.TOOL_HANDLERS["slow_mock_tool"] = slow_mock_tool
    registry.TOOL_METADATA["slow_mock_tool"] = {
        "max_latency_ms": 3000,
        "idempotent": False,
        "mutates_state": False,
        "required_params": {"flight_id"},
    }

    try:
        orchestrator = VoiceAgentOrchestrator()
        gen_res = orchestrator.turn_controller.start_generation("Trigger slow reservation")
        turn_1 = await gen_res if inspect.isawaitable(gen_res) else gen_res

        exec_task = asyncio.create_task(
            orchestrator.execute_tool_call(
                tool_name="slow_mock_tool",
                arguments={"flight_id": "FL-999"},
                turn_id=turn_1,
            )
        )

        await asyncio.wait_for(tool_started.wait(), timeout=1.0)

        # Invalidate turn_1 via barge-in while tool is running
        await orchestrator.handle_user_barge_in("Cancel that, wrong destination")

        # Let tool complete internally
        tool_gate.set()

        res = await asyncio.wait_for(exec_task, timeout=1.0)

        status_value = str(res.get("status", "")).lower()
        message_value = str(res.get("message", "")).lower()
        error_value = str(res.get("error", "")).lower()
        assert (
            status_value in ["cancelled", "stale", "error"]
            or "cancelled" in message_value
            or "superseded" in message_value
            or "stale" in error_value
        )
    finally:
        registry.TOOL_HANDLERS.pop("slow_mock_tool", None)
        registry.TOOL_METADATA.pop("slow_mock_tool", None)


@pytest.mark.asyncio
async def test_empty_text_no_op_lifecycle():
    """Validates that empty/whitespace text does not leave dangling tasks."""
    orchestrator = VoiceAgentOrchestrator()
    gen_res = orchestrator.turn_controller.start_generation("Silent input check")
    turn_id = await gen_res if inspect.isawaitable(gen_res) else gen_res

    emitted = []
    async for frame in orchestrator.stream_agent_reply("   ", turn_id=turn_id):
        emitted.append(frame)

    assert len(emitted) == 0
    assert orchestrator._current_tts_task is None