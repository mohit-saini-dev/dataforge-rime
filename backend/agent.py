from __future__ import annotations
import asyncio
import inspect
import logging
from typing import Any, AsyncGenerator, Dict, Optional, Set

from backend.control.turn_controller import TurnController
from backend.tools.executor import ToolExecutionHarness
from backend.tts.rime_plugin import FencedRimeTTS

logger = logging.getLogger(__name__)


class VoiceAgentOrchestrator:
    """Production orchestrator combining generation fencing, tools, and TTS streaming."""

    def __init__(
        self,
        turn_controller: Optional[TurnController] = None,
        tool_harness: Optional[ToolExecutionHarness] = None,
        tts_client: Optional[FencedRimeTTS] = None,
    ) -> None:
        self.turn_controller = turn_controller or TurnController()
        self.tool_harness = tool_harness or ToolExecutionHarness()
        self.tts_client = tts_client or FencedRimeTTS()
        self._state_lock = asyncio.Lock()
        self._current_tts_task: Optional[asyncio.Task] = None
        self._draining_tasks: Set[asyncio.Task] = set()

    def _fence_validator(self, turn_id: int) -> bool:
        return self.turn_controller.validate_generation(turn_id)

    async def handle_user_barge_in(self, user_text: str = "[user_interruption]") -> int:
        """Atomically advance generation fence and claim prior TTS task for draining."""
        async with self._state_lock:
            old_turn_id = self.turn_controller.current_generation_id

            if old_turn_id != 0:
                gen_res = self.turn_controller.rollback_invalidated_generation(old_turn_id, user_text)
                new_turn_id = await gen_res if inspect.isawaitable(gen_res) else gen_res
            else:
                gen_res = self.turn_controller.start_generation(user_text)
                new_turn_id = await gen_res if inspect.isawaitable(gen_res) else gen_res

            # Atomically swap and claim the active TTS task while holding lock
            task_to_cancel = self._current_tts_task
            self._current_tts_task = None

        # 1. Abort in-flight operations registered for previous turn
        if old_turn_id != 0:
            try:
                await self.turn_controller.cancel_generation_ops(old_turn_id)
            except Exception as e:
                logger.warning("Error cancelling ops for generation %d: %s", old_turn_id, e)

        # 2. Cancel and drain prior TTS task safely outside lock
        if task_to_cancel and not task_to_cancel.done():
            task_to_cancel.cancel()

            # Bound draining set to prevent task leakage
            if len(self._draining_tasks) > 20:
                self._draining_tasks = {t for t in self._draining_tasks if not t.done()}

            self._draining_tasks.add(task_to_cancel)
            task_to_cancel.add_done_callback(self._draining_tasks.discard)

            try:
                await asyncio.wait_for(asyncio.shield(task_to_cancel), timeout=0.08)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass

        return new_turn_id

    async def execute_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        turn_id: int,
    ) -> Dict[str, Any]:
        """Execute a tool wrapped with pre/post fence validation and state compensation."""
        return await self.tool_harness.execute_tool(
            tool_name=tool_name,
            arguments=arguments,
            turn_id=turn_id,
            fence_validator=self._fence_validator,
            turn_controller=self.turn_controller,
        )

    async def stream_agent_reply(
        self,
        text: str,
        turn_id: int,
        chunk_latency_ms: int = 20,
    ) -> AsyncGenerator[bytes, None]:
        """Stream synthesized audio frames guarded by the generation fence."""
        current_task = asyncio.current_task()
        task_to_displace = None

        async with self._state_lock:
            # Drop obsolete stream before starting
            if not self._fence_validator(turn_id):
                return

            # Cancel and displace any previous active TTS stream
            if self._current_tts_task and self._current_tts_task is not current_task:
                task_to_displace = self._current_tts_task

            if current_task:
                self._current_tts_task = current_task

        if task_to_displace and not task_to_displace.done():
            task_to_displace.cancel()

        try:
            async for frame in self.tts_client.stream_speech(
                text=text,
                turn_id=turn_id,
                fence_validator=self._fence_validator,
                chunk_latency_ms=chunk_latency_ms,
            ):
                yield frame
        finally:
            async with self._state_lock:
                if self._current_tts_task is current_task:
                    self._current_tts_task = None