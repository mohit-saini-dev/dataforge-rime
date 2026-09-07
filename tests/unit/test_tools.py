"""
Unit tests for travel operations mock tools, registry, and execution harness.
Verifies concurrency locks, generation fence boundaries, timeouts, and cancellation semantics.
"""

import asyncio
import pytest

from backend.tools.executor import ToolExecutionHarness
from backend.tools.mock_tools import _db
import backend.tools.registry as registry
from backend.tools.registry import (
    InvalidToolArgumentsError,
    ToolNotFoundError,
    get_available_tool_definitions,
    get_tool_handler,
    validate_tool_arguments,
)


@pytest.fixture(autouse=True)
def reset_mock_db():
    """Ensure mock DB state is clean before every test run."""
    _db.reset()


@pytest.mark.asyncio
async def test_search_flights_success():
    handler = get_tool_handler("search_flights")
    res = await handler(origin="DEL", destination="BOM")
    assert res["status"] == "success"
    assert res["count"] == 2
    assert len(res["flights"]) == 2


@pytest.mark.asyncio
async def test_atomic_seat_deduction_and_overselling():
    """Verify concurrency locks prevent overselling under simultaneous barrier contention."""
    handler = get_tool_handler("book_flight")
    barrier = asyncio.Barrier(6)  # Rendezvous all 6 tasks before acquiring lock

    async def attempt_booking(i: int):
        await barrier.wait()
        return await handler(flight_id="FL-204", passenger_name=f"Passenger-{i}")

    tasks = [asyncio.create_task(attempt_booking(i)) for i in range(6)]
    results = await asyncio.gather(*tasks)

    successes = [r for r in results if r.get("status") == "success"]
    sold_outs = [r for r in results if r.get("error_code") == "SOLD_OUT"]

    # FL-204 has exactly 5 seats
    assert len(successes) == 5
    assert len(sold_outs) == 1

    # Verify no duplicate booking IDs were generated
    booking_ids = [r["booking"]["booking_id"] for r in successes]
    assert len(set(booking_ids)) == 5

    # Verify inventory is strictly 0 and never went negative
    flight = next(f for f in _db.flights if f["flight_id"] == "FL-204")
    assert flight["available_seats"] == 0


@pytest.mark.asyncio
async def test_double_cancellation_and_seat_restoration():
    """Verify cancelling a booking restores inventory once and rejects second cancellation."""
    book_handler = get_tool_handler("book_flight")
    cancel_handler = get_tool_handler("cancel_booking")

    # Initial available seats on FL-101 is 12
    booking = await book_handler(flight_id="FL-101", passenger_name="Alice")
    assert booking["status"] == "success"
    booking_id = booking["booking"]["booking_id"]

    flight = next(f for f in _db.flights if f["flight_id"] == "FL-101")
    assert flight["available_seats"] == 11

    # First cancellation succeeds and restores seat
    cancellation1 = await cancel_handler(booking_id=booking_id)
    assert cancellation1["status"] == "success"
    assert flight["available_seats"] == 12

    # Second cancellation fails idempotently without duplicating seat restoration
    cancellation2 = await cancel_handler(booking_id=booking_id)
    assert cancellation2["status"] == "error"
    assert cancellation2["error_code"] == "ALREADY_CANCELLED"
    assert flight["available_seats"] == 12


@pytest.mark.asyncio
async def test_harness_pre_execution_fence():
    """Harness aborts execution immediately if turn ID is already superseded."""
    harness = ToolExecutionHarness()

    def is_fresh(turn_id: int) -> bool:
        return turn_id != 1

    res = await harness.execute_tool(
        tool_name="search_flights",
        arguments={"origin": "DEL", "destination": "BOM"},
        turn_id=1,
        fence_validator=is_fresh,
    )

    assert res["status"] == "cancelled"
    assert res["reason"] == "pre_execution_fence_failed"


@pytest.mark.asyncio
async def test_harness_post_execution_fence():
    """Harness rejects results deterministically if fence flips mid-execution."""
    harness = ToolExecutionHarness()
    gate = asyncio.Event()
    handler_started = asyncio.Event()

    original_handler = get_tool_handler("search_flights")

    async def slow_handler(**kwargs):
        handler_started.set()
        await gate.wait()
        return await original_handler(**kwargs)

    registry.TOOL_HANDLERS["search_flights"] = slow_handler

    try:
        async def validator(_turn_id: int) -> bool:
            return not gate.is_set()

        exec_task = asyncio.create_task(
            harness.execute_tool(
                tool_name="search_flights",
                arguments={"origin": "DEL", "destination": "BOM"},
                turn_id=1,
                fence_validator=validator,
            )
        )

        await handler_started.wait()
        gate.set()  # Invalidate fence while tool is executing
        res = await exec_task
    finally:
        registry.TOOL_HANDLERS["search_flights"] = original_handler

    assert res["status"] == "cancelled"
    assert res["reason"] == "post_execution_fence_failed"


@pytest.mark.asyncio
async def test_harness_propagates_cancellation():
    """Verify harness re-raises CancelledError for TurnController structured concurrency."""
    harness = ToolExecutionHarness()

    task = asyncio.create_task(
        harness.execute_tool(
            tool_name="search_flights",
            arguments={"origin": "DEL", "destination": "BOM"},
            turn_id=1,
        )
    )

    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_harness_enforces_timeout_budget(monkeypatch):
    """Verify executor catches timeouts when latency budget is exceeded."""
    harness = ToolExecutionHarness()

    # Force 1ms timeout to trigger EXECUTION_TIMEOUT deterministically
    monkeypatch.setitem(
        registry.TOOL_METADATA,
        "search_flights",
        {"max_latency_ms": 1, "required_params": {"origin", "destination"}},
    )

    res = await harness.execute_tool(
        tool_name="search_flights",
        arguments={"origin": "DEL", "destination": "BOM"},
        turn_id=1,
    )
    assert res["status"] == "error"
    assert res["error_code"] == "EXECUTION_TIMEOUT"


def test_registry_argument_validation():
    with pytest.raises(InvalidToolArgumentsError):
        validate_tool_arguments("book_flight", {"flight_id": "FL-101"})

    with pytest.raises(ToolNotFoundError):
        validate_tool_arguments("unknown_tool", {})


def test_tool_schemas_are_strict():
    """Verify all tool definitions enforce additionalProperties: False and IATA regexes."""
    defs = get_available_tool_definitions()
    assert len(defs) == 4

    for tool in defs:
        params = tool["function"]["parameters"]
        assert params.get("additionalProperties") is False, f"{tool['function']['name']} missing strict schema"

    flights_schema = next(d for d in defs if d["function"]["name"] == "search_flights")
    assert flights_schema["function"]["parameters"]["properties"]["origin"]["pattern"] == "^[A-Z]{3}$"