import asyncio
import pytest
from backend.control.turn_controller import TurnController, GenState

@pytest.fixture
def controller() -> TurnController:
    import backend.control.turn_controller as tc_module
    tc_module._turn_controller = None
    return TurnController()

@pytest.mark.asyncio
async def test_start_generation_monotonic_ids(controller: TurnController):
    gen1 = await controller.start_generation("Hello")
    gen2 = await controller.start_generation("World")
    assert gen1 == 1
    assert gen2 == 2
    assert controller.current_generation_id == 2

@pytest.mark.asyncio
async def test_validate_generation_active_only(controller: TurnController):
    gen1 = await controller.start_generation("Test")
    assert controller.validate_generation(gen1) is True
    assert controller.validate_generation(999) is False

    await controller.start_generation("Interrupt")
    assert controller.validate_generation(gen1) is False
    assert controller.validate_generation(2) is True

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