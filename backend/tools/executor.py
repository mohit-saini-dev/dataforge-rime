from __future__ import annotations
import asyncio
import inspect
import logging
import re
import time
from typing import Any, Callable, Dict, Optional

from backend.tools.registry import TOOL_DEFINITIONS, TOOL_METADATA, TOOL_HANDLERS
from backend.tools.mock_tools import get_mock_db

logger = logging.getLogger(__name__)


class ToolExecutionHarness:
    """Production-hardened tool execution harness."""

    def __init__(self, timeout_padding_ms: int = 200):
        self.timeout_padding_ms = timeout_padding_ms

    async def _check_fence(
        self,
        fence_validator: Optional[Callable[[int], Any]],
        turn_id: int,
        tool_name: str,
        stage: str = "check",
    ) -> bool:
        if not fence_validator:
            return True
        try:
            res = fence_validator(turn_id)
            if inspect.isawaitable(res):
                res = await res
            return bool(res)
        except Exception as e:
            logger.error("[FENCE] Error during %s fence check for '%s': %s", stage, tool_name, e)
            return False

    def validate_tool_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        schema_def = next((t for t in TOOL_DEFINITIONS if t.get("name") == tool_name), None)
        if not schema_def:
            return None

        params = schema_def.get("parameters", {})
        properties = params.get("properties", {})
        required = params.get("required", [])

        for req in required:
            if req not in arguments:
                return f"Missing required parameter '{req}'"

        if not params.get("additionalProperties", True):
            for key in arguments:
                if key not in properties:
                    return f"Unexpected property '{key}' not permitted by schema"

        for key, val in arguments.items():
            if key not in properties:
                continue
            prop_spec = properties[key]
            expected_type = prop_spec.get("type")

            if expected_type == "string":
                if not isinstance(val, str):
                    return f"Parameter '{key}' must be a string"
                pattern = prop_spec.get("pattern")
                if pattern and not re.match(pattern, val):
                    return f"Parameter '{key}' does not match pattern '{pattern}'"
            elif expected_type == "integer":
                if not isinstance(val, int) or isinstance(val, bool):
                    return f"Parameter '{key}' must be an integer"
                min_val = prop_spec.get("minimum")
                max_val = prop_spec.get("maximum")
                if min_val is not None and val < min_val:
                    return f"Parameter '{key}' must be >= {min_val}"
                if max_val is not None and val > max_val:
                    return f"Parameter '{key}' must be <= {max_val}"

        return None

    async def _compensate_side_effect(self, tool_name: str, result: Dict[str, Any]) -> None:
        if tool_name == "book_flight" and isinstance(result, dict) and result.get("status") == "success":
            booking = result.get("booking", {})
            booking_id = booking.get("booking_id")
            if booking_id:
                logger.warning("[COMPENSATION] Rolling back flight booking %s due to fence expiry", booking_id)
                db = get_mock_db()
                await db.rollback_booking(booking_id)

    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        turn_id: int,
        fence_validator: Optional[Callable[[int], Any]] = None,
        turn_controller: Optional[Any] = None,
    ) -> Dict[str, Any]:
        meta = TOOL_METADATA.get(tool_name, {})
        max_lat = meta.get("max_latency_ms", 1500)
        timeout_sec = (max_lat + self.timeout_padding_ms) / 1000.0
        start_time = time.perf_counter()

        # 1. Pre-execution fence check
        if not await self._check_fence(fence_validator, turn_id, tool_name, stage="pre"):
            return {
                "status": "cancelled",
                "reason": "pre_execution_fence_failed",
                "error_code": "TURN_SUPERSEDED_PRE_EXECUTION",
                "message": f"Tool '{tool_name}' dropped: Turn {turn_id} superseded before execution started.",
            }

        # 2. Schema check
        val_err = self.validate_tool_arguments(tool_name, arguments)
        if val_err:
            return {
                "status": "error",
                "error_code": "INVALID_ARGUMENTS",
                "message": val_err,
            }

        handler = TOOL_HANDLERS.get(tool_name)
        if not handler:
            return {
                "status": "error",
                "error_code": "TOOL_NOT_FOUND",
                "message": f"No handler registered for '{tool_name}'.",
            }

        # 3. Timed execution
        try:
            async with asyncio.timeout(timeout_sec):
                result = await handler(**arguments)
        except asyncio.TimeoutError:
            return {
                "status": "error",
                "error_code": "EXECUTION_TIMEOUT",
                "message": f"Tool '{tool_name}' timed out after {timeout_sec:.2f}s.",
            }
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return {
                "status": "error",
                "error_code": "EXECUTION_FAILED",
                "message": str(exc),
            }

        # 4. Post-execution fence check with compensation
        if not await self._check_fence(fence_validator, turn_id, tool_name, stage="post"):
            if meta.get("mutates_state", False):
                await self._compensate_side_effect(tool_name, result)
            return {
                "status": "cancelled",
                "reason": "post_execution_fence_failed",
                "error_code": "TURN_SUPERSEDED_POST_EXECUTION",
                "message": f"Tool '{tool_name}' completed, but Turn {turn_id} was superseded. Result discarded.",
            }

        # Attach execution metadata
        duration_ms = (time.perf_counter() - start_time) * 1000
        if isinstance(result, dict):
            result["_meta"] = {
                "turn_id": turn_id,
                "tool_name": tool_name,
                "duration_ms": round(duration_ms, 2),
            }

        return result