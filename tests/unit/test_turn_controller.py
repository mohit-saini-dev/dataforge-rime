import asyncio
import pytest

from backend.control.turn_controller import GenState, TurnController


@pytest.fixture
def controller() -> TurnController:
    return TurnController()


@pytest.mark.asyncio
async def test_start_generation_monotonic_ids(controller: TurnController):
    gen1 = await controller.start_generation("Hello")
    gen2 = await controller.start_generation("World")
    assert gen2 > gen1
    assert controller.current_generation_id == gen2


@pytest.mark.asyncio
async def test_validate_generation_active_only(controller: TurnController):
    gen1 = await controller.start_generation("Turn 1")
    assert controller.validate_generation(gen1) is True

    # Starting a new generation invalidates the previous one
    gen2 = await controller.start_generation("Turn 2")
    assert controller.validate_generation(gen1) is False
    assert controller.validate_generation(gen2) is True


@pytest.mark.asyncio
async def test_commit_llm_result_fence_rejects_stale(controller: TurnController):
    gen1 = await controller.start_generation("User 1")
    await controller.start_generation("User 2")

    committed = await controller.commit_llm_result(gen1, "Stale response")
    assert committed is False
    assert len(controller.conversation_history) == 2
    assert all(msg["role"] != "assistant" for msg in controller.conversation_history)


@pytest.mark.asyncio
async def test_invalidate_then_rollback_preserves_context(controller: TurnController):
    gen1 = await controller.start_generation("Book a hotel in Paris")
    await controller.commit_llm_result(gen1, "Checking hotels in Paris...")

    invalidated_gen = await controller.invalidate_current_generation("User said cancel")
    assert invalidated_gen == gen1

    new_gen = await controller.rollback_invalidated_generation(invalidated_gen, "Book flight to Tokyo")
    assert new_gen == 2
    assert controller.current_generation_id == 2

    history = controller.conversation_history
    assert len(history) == 3
    assert history[0]["role"] == "user" and history[0]["content"] == "Book a hotel in Paris"
    assert history[1]["role"] == "system" and "interrupted Generation 1" in history[1]["content"]
    assert history[2]["role"] == "user" and history[2]["content"] == "Book flight to Tokyo"


@pytest.mark.asyncio
async def test_memory_pruning_bounds_retained_generations():
    """Validates that a long-running session automatically prunes stale state."""
    controller = TurnController(max_retained_generations=15)

    for i in range(50):
        await controller.start_generation(f"Message {i}")

    # Retained state must be strictly bounded
    assert len(controller._gen_state) <= 16
    assert len(controller._history_snapshots) <= 16
    assert len(controller._gen_ops) <= 16


@pytest.mark.asyncio
async def test_cancel_generation_ops_lifecycle(controller: TurnController):
    """Validates operation registration and explicit cancellation into DRAINING state."""
    gen1 = await controller.start_generation("Background work")
    event = asyncio.Event()

    async def sample_worker():
        await event.wait()

    task = asyncio.create_task(sample_worker())
    op_id = controller.register_operation(gen1, "worker", task)

    assert controller.validate_operation(op_id) is True

    cancelled_ops = await controller.cancel_generation_ops(gen1)
    assert op_id in cancelled_ops
    assert task.cancelling() or task.done()
    assert controller._gen_state[gen1] == GenState.DRAINING