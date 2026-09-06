"""
Integration tests for VoiceAgentOrchestrator, TurnController,
and ToolExecutionHarness under high-frequency barge-ins and concurrency.
"""

import asyncio
import inspect
from typing import Any, AsyncGenerator, List, Union
import pytest

from backend.agent import VoiceAgentOrchestrator
from backend.control.turn_controller import TurnController
from backend.tools import registry
from backend.tools.executor import ToolExecutionHarness
from backend.tts.rime_plugin import FencedRimeTTS


class GatedTTS(FencedRimeTTS):
    """
    Deterministic TTS stub that pauses execution after emitting the first chunk
    until a gate event is fired, enabling deterministic barge-in race verification.
    """

    def __init__(self) -> None:
        super().__init__()
        self.first_chunk_emitted = asyncio.Event()
        self.proceed_gate = asyncio.Event()

    async def stream_speech(
        self,
        text: str,
        turn_id: int,
        fence_validator: Any = None,
        chunk_latency_ms: int = 20,
    ) -> AsyncGenerator[Union[bytes, Any], None]:
        if not text or not text.strip():
            return

        if fence_validator and not fence_validator(turn_id):
            return

        words = text.split()
        total_chunks = max(1, len(words))

        for chunk_idx in range(total_chunks):
            if fence_validator and not fence_validator(turn_id):
                break

            yield b"\x00" * 960

            if chunk_idx == 0:
                self.first_chunk_emitted.set()
                await self.proceed_gate.wait()


@pytest.mark.asyncio
async def test_deterministic_barge_in_and_task_finalization():
    gated_tts = GatedTTS()
    controller = TurnController()
    orchestrator = VoiceAgentOrchestrator(turn_controller=controller, tts_client=gated_tts)

    gen_res = controller.start_generation("Tell me a long story")
    turn_1 = await gen_res if inspect.isawaitable(gen_res) else gen_res

    emitted_chunks: List[bytes] = []

    async def consumer():
        async for chunk in orchestrator.stream_agent_reply("One two three four five", turn_id=turn_1):
            emitted_chunks.append(chunk)

    consumer_task = asyncio.create_task(consumer())

    await asyncio.wait_for(gated_tts.first_chunk_emitted.wait(), timeout=1.0)
    assert len(emitted_chunks) == 1
    assert orchestrator._current_tts_task is not None

    new_turn_id = await orchestrator.handle_user_barge_in("Stop talking, listen to me")
    assert new_turn_id > turn_1

    gated_tts.proceed_gate.set()

    try:
        await asyncio.wait_for(consumer_task, timeout=1.0)
    except asyncio.CancelledError:
        pass

    assert len(emitted_chunks) == 1
    assert orchestrator._current_tts_task is None


@pytest.mark.asyncio
async def test_concurrent_double_barge_in_deadlock_free():
    orchestrator = VoiceAgentOrchestrator()

    gen_res = orchestrator.turn_controller.start_generation("Initial turn")
    turn_0 = await gen_res if inspect.isawaitable(gen_res) else gen_res

    tasks = [
        orchestrator.handle_user_barge_in("Barge in 1"),
        orchestrator.handle_user_barge_in("Barge in 2"),
    ]

    results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=1.0)
    assert len(results) == 2
    assert results[0] > turn_0
    assert results[1] > turn_0
    assert results[0] != results[1]
    assert orchestrator.turn_controller.current_generation_id == max(results)


@pytest.mark.asyncio
async def test_orchestrator_executes_registered_tool():
    orchestrator = VoiceAgentOrchestrator()
    gen_res = orchestrator.turn_controller.start_generation("Find flights from JFK to LHR")
    turn_1 = await gen_res if inspect.isawaitable(gen_res) else gen_res

    result = await orchestrator.execute_tool_call(
        tool_name="search_flights",
        arguments={"origin": "JFK", "destination": "LHR"},
        turn_id=turn_1,
    )

    assert result.get("status") == "success"
    assert result.get("query", {}).get("origin") == "JFK"
    assert result.get("query", {}).get("destination") == "LHR"
    assert "_meta" in result
    assert result["_meta"]["turn_id"] == turn_1


@pytest.mark.asyncio
async def test_stale_tool_call_rejected_after_barge_in():
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
        await orchestrator.handle_user_barge_in("Cancel that, wrong destination")
        tool_gate.set()

        try:
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
        except asyncio.CancelledError:
            pass
    finally:
        registry.TOOL_HANDLERS.pop("slow_mock_tool", None)
        registry.TOOL_METADATA.pop("slow_mock_tool", None)


@pytest.mark.asyncio
async def test_empty_text_no_op_lifecycle():
    orchestrator = VoiceAgentOrchestrator()
    gen_res = orchestrator.turn_controller.start_generation("Silent input check")
    turn_id = await gen_res if inspect.isawaitable(gen_res) else gen_res

    emitted = []
    async for frame in orchestrator.stream_agent_reply("   ", turn_id=turn_id):
        emitted.append(frame)

    assert len(emitted) == 0
    assert orchestrator._current_tts_task is None


@pytest.mark.asyncio
async def test_barge_in_when_idle():
    """Validates that barge-in cleanly advances generation without active TTS."""
    orchestrator = VoiceAgentOrchestrator()
    new_turn = await orchestrator.handle_user_barge_in("Spontaneous user talk")
    assert new_turn == 1
    assert orchestrator.turn_controller.current_generation_id == 1
    assert orchestrator._current_tts_task is None


@pytest.mark.asyncio
async def test_concurrent_tts_stream_supersedes_stale():
    """Validates that a subsequent TTS stream displaces any previous active stream."""
    gated_tts = GatedTTS()
    orchestrator = VoiceAgentOrchestrator(tts_client=gated_tts)
    turn_1 = await orchestrator.turn_controller.start_generation("Turn 1")

    # Start first stream
    async def stream_one():
        try:
            async for _ in orchestrator.stream_agent_reply("First speech generation", turn_id=turn_1):
                pass
        except asyncio.CancelledError:
            pass

    task1 = asyncio.create_task(stream_one())
    await asyncio.wait_for(gated_tts.first_chunk_emitted.wait(), timeout=1.0)
    assert orchestrator._current_tts_task is task1

    # Start second stream on same active turn
    task2 = asyncio.create_task(
        orchestrator.stream_agent_reply("Second speech generation", turn_id=turn_1).__anext__()
    )
    await asyncio.sleep(0.02)

    # Task 1 must have been superseded and cancelled
    assert task1.cancelling() or task1.done()
    gated_tts.proceed_gate.set()
    await asyncio.gather(task1, task2, return_exceptions=True)