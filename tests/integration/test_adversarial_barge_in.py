import asyncio
import pytest
from backend.agent import VoiceAgentOrchestrator
from backend.tools.mock_tools import get_mock_db


@pytest.mark.asyncio
async def test_adversarial_tool_mutation_rollback_on_barge_in():
    """Ensure a flight booking cancelled mid-turn rolls back database mutations."""
    orchestrator = VoiceAgentOrchestrator()
    db = get_mock_db()
    db.reset()

    # Initial seats for FL-101 is 12
    initial_flight = next(f for f in db.flights if f["flight_id"] == "FL-101")
    assert initial_flight["available_seats"] == 12

    turn_1 = await orchestrator.turn_controller.start_generation("Book flight FL-101")

    # Start executing tool
    exec_task = asyncio.create_task(
        orchestrator.execute_tool_call(
            tool_name="book_flight",
            arguments={"flight_id": "FL-101", "passenger_name": "Test Passenger"},
            turn_id=turn_1,
        )
    )

    # Allow execution to hit the simulated I/O sleep window
    await asyncio.sleep(0.1)

    # User barges in before tool completes
    await orchestrator.handle_user_barge_in("Actually cancel that!")

    res = await exec_task

    # Tool result must be flagged cancelled
    assert res.get("status") == "cancelled"

    # Verify atomic rollback: seats restored and no phantom booking committed
    flight_after = next(f for f in db.flights if f["flight_id"] == "FL-101")
    assert flight_after["available_seats"] == 12
    assert len(db.bookings) == 0


@pytest.mark.asyncio
async def test_adversarial_rapid_barge_in_storm():
    """Burst multiple barge-in signals concurrently without deadlock or generation corruption."""
    orchestrator = VoiceAgentOrchestrator()
    await orchestrator.turn_controller.start_generation("Initial turn")

    # Fire 8 rapid concurrent interruptions
    async def trigger_barge_in(i: int):
        return await orchestrator.handle_user_barge_in(f"Interruption {i}")

    turns = await asyncio.gather(*(trigger_barge_in(i) for i in range(8)))

    # All generated turn IDs must be unique and properly ordered
    assert len(turns) == 8
    assert len(set(turns)) == 8
    assert orchestrator.turn_controller.current_generation_id == max(turns)


@pytest.mark.asyncio
async def test_adversarial_tts_toctou_pre_yield_suppression():
    """Ensure zero frames leak if turn invalidates right during frame wait."""
    orchestrator = VoiceAgentOrchestrator()
    turn_1 = await orchestrator.turn_controller.start_generation("Start speech")

    frames_received = []

    async def consumer():
        try:
            async for frame in orchestrator.stream_agent_reply(
                text="One two three four five six seven eight nine ten",
                turn_id=turn_1,
                chunk_latency_ms=40,
            ):
                frames_received.append(frame)
        except asyncio.CancelledError:
            pass

    consumer_task = asyncio.create_task(consumer())

    # Wait for initial frame emission
    await asyncio.sleep(0.05)
    assert len(frames_received) >= 1
    count_before_interrupt = len(frames_received)

    # Invalidate turn mid-stream
    await orchestrator.handle_user_barge_in("Stop talking")
    await consumer_task

    # Verify TOCTOU protection: no trailing frames yielded after interruption
    assert len(frames_received) == count_before_interrupt