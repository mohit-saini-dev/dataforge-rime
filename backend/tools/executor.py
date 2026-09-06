"""
Async tool execution harness with generation fencing, latency timeouts,
execution latency profiling, cancellation boundaries, and TurnController registration.
"""

import asyncio
import inspect
import logging
import time
from typing import Any, Awaitable, Callable, Dict, Optional, Union

from backend.tools.registry import (
    InvalidToolArgumentsError,
    ToolNotFoundError,
    get_tool_handler,
    get_tool_metadata,
    validate_tool_arguments,
)

logger = logging.getLogger("dataforge_rime.tools.executor")

FenceValidator = Callable[[int], Union[bool, Awaitable[bool]]]


class ToolExecutionHarness:
    """
    Executes travel operation tools wrapped with generation fence verification
    and timeout budgets.
    """

    @staticmethod
    def _err(code: str, message: str, operation: str) -> Dict[str, Any]:
        return {
            "status": "error",
            "error_code": code,
            "message": message,
            "operation": operation,
        }

    @staticmethod
    def _cancelled(operation: str, reason: str, turn_id: int) -> Dict[str, Any]:
        return {
            "status": "cancelled",
            "operation": operation,
            "reason": reason,
            "turn_id": turn_id,
            "message": "Tool execution was cancelled or superseded by turn controller.",
        }

    async def _check_fence(
        self, validator: Optional[FenceValidator], turn_id: int, operation: str
    ) -> bool:
        """Evaluate fence validator safely across sync and async callables."""
        if validator is None:
            return True
        try:
            res = validator(turn_id)
            if inspect.isawaitable(res):
                res = await res
            return bool(res)
        except Exception as exc:
            logger.warning(
                "Fence validator exception on tool '%s' (turn %s): %s",
                operation,
                turn_id,
                exc,
            )
            return False

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        turn_id: int,
        fence_validator: Optional[FenceValidator] = None,
        turn_controller: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Safely execute a tool with generation fencing, timeout enforcement,
        TurnController registration, and latency tracking.
        """
        start_time = time.perf_counter()

        # 1. Pre-execution generation fence check
        is_fresh = await self._check_fence(fence_validator, turn_id, tool_name)
        if not is_fresh:
            logger.info("Tool '%s' cancelled: pre-execution fence failed for turn %s", tool_name, turn_id)
            return self._cancelled(tool_name, "pre_execution_fence_failed", turn_id)

        # 2. Handler & metadata lookup
        try:
            handler = get_tool_handler(tool_name)
            meta = get_tool_metadata(tool_name)
        except ToolNotFoundError as exc:
            return self._err("TOOL_NOT_FOUND", str(exc), tool_name)

        # 3. Parameter validation
        try:
            validate_tool_arguments(tool_name, arguments)
        except InvalidToolArgumentsError as exc:
            return self._err("INVALID_ARGUMENTS", str(exc), tool_name)

        # 4. Latency budget configuration
        timeout_sec = meta.get("max_latency_ms", 1500) / 1000.0

        # 5. Execution within timeout, registration, and cancellation boundary
        current_task = asyncio.current_task()
        op_id = None
        if turn_controller and current_task:
            try:
                op_id = turn_controller.register_operation(
                    gen_id=turn_id,
                    kind=f"tool_{tool_name}",
                    coro_or_task=current_task,
                )
            except Exception as e:
                logger.warning("Failed to register tool '%s' with TurnController: %s", tool_name, e)

        try:
            async with asyncio.timeout(timeout_sec):
                result = await handler(**arguments)
        except asyncio.TimeoutError:
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
            logger.warning("Tool '%s' exceeded timeout of %ss (elapsed: %sms)", tool_name, timeout_sec, elapsed_ms)
            return self._err(
                "EXECUTION_TIMEOUT",
                f"Tool '{tool_name}' exceeded timeout budget of {timeout_sec}s.",
                tool_name,
            )
        except asyncio.CancelledError:
            logger.info("Tool '%s' received task cancellation during execution for turn %s", tool_name, turn_id)
            raise
        except Exception as exc:
            logger.exception("Unhandled failure executing tool '%s' on turn %s", tool_name, turn_id)
            return self._err("EXECUTION_FAILED", f"Unexpected tool failure: {exc}", tool_name)

        # 6. Post-execution fence check
        is_fresh_post = await self._check_fence(fence_validator, turn_id, tool_name)
        if not is_fresh_post:
            logger.info("Tool '%s' cancelled: post-execution fence failed for turn %s", tool_name, turn_id)
            return self._cancelled(tool_name, "post_execution_fence_failed", turn_id)

        # 7. Attach execution metadata for turn binding
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        if not isinstance(result, dict):
            result = {"result": result}

        result["_meta"] = {
            "turn_id": turn_id,
            "tool_name": tool_name,
            "execution_time_ms": elapsed_ms,
        }

        return result