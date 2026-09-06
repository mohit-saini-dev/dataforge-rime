"""
Agent Orchestrator for dataforge-rime.
Coordinates turn fencing, task ownership, tool execution harnesses,
and interruptible TTS streaming with sub-100ms cancellation guarantees.
"""

import asyncio
import inspect
import logging
from typing import Any, AsyncGenerator, Dict, Optional, Union

from backend.control.turn_controller import TurnController
from backend.tools.executor import ToolExecutionHarness
from backend.tools.registry import (
    ToolNotFoundError,
    get_tool_metadata,
)
from backend.tts.rime_plugin import FencedRimeTTS

logger = logging.getLogger("dataforge_rime.agent")


class VoiceAgentOrchestrator:
    """
    Manages conversational turn lifecycles, ensuring atomic tool actions
    and sub-100ms interruption latency through generation fencing and explicit
    async task lifecycle ownership.
    """

    def __init__(
        self,
        turn_controller: Optional[TurnController] = None,
        tool_harness: Optional[ToolExecutionHarness] = None,
        tts_client: Optional[FencedRimeTTS] = None,
    ) -> None:
        self.turn_controller = turn_controller or TurnController()
        self.tool_harness = tool_harness or ToolExecutionHarness()
        self.tts = tts_client or FencedRimeTTS()
        self._current_tts_task: Optional[asyncio.Task] = None
        self._state_lock = asyncio.Lock()

    def _fence_validator(self, turn_id: int) -> bool:
        """
        Synchronously validates whether the given turn_id is the active generation.
        Safely checks against TurnController state without invoking properties as methods.
        """
        if hasattr(self.turn_controller, "validate_generation"):
            return bool(self.turn_controller.validate_generation(turn_id))
        return getattr(self.turn_controller, "current_generation_id", None) == turn_id

    async def handle_user_barge_in(self, user_text: str = "[user_interruption]") -> int:
        """
        Invoked immediately upon Voice Activity Detection (VAD) trigger.
        Atomically shifts the generation fence, detaches the active TTS task,
        and cleanly cancels the detached task within an 80ms bounded budget.
        """
        async with self._state_lock:
            gen_res = self.turn_controller.start_generation(user_text)
            new_turn_id = await gen_res if inspect.isawaitable(gen_res) else gen_res

            old_tts_task = self._current_tts_task
            self._current_tts_task = None

        if old_tts_task and not old_tts_task.done():
            old_tts_task.cancel()
            try:
                await asyncio.wait_for(
                    asyncio.shield(old_tts_task),
                    timeout=0.08,
                )
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass

        logger.info("Barge-in handled. New active generation turn: %s", new_turn_id)
        return int(new_turn_id)

    async def execute_tool_call(
        self, tool_name: str, arguments: Dict[str, Any], turn_id: int
    ) -> Dict[str, Any]:
        """
        Executes a registered tool within the generation fence boundary using ToolExecutionHarness.
        """
        try:
            get_tool_metadata(tool_name)
        except ToolNotFoundError as err:
            return {"error": str(err)}

        return await self.tool_harness.execute_tool(
            tool_name=tool_name,
            arguments=arguments,
            turn_id=turn_id,
            fence_validator=self._fence_validator,
        )

    async def stream_agent_reply(
        self, text: str, turn_id: int, chunk_latency_ms: int = 20
    ) -> AsyncGenerator[Union[bytes, Any], None]:
        """
        Streams synthesized audio frames under generation fencing.
        Registers running task to enable deterministic cancellation on barge-in.
        """
        if not text or not text.strip():
            return

        current_task = asyncio.current_task()
        async with self._state_lock:
            self._current_tts_task = current_task

        try:
            async for frame in self.tts.stream_speech(
                text=text,
                turn_id=turn_id,
                fence_validator=self._fence_validator,
                chunk_latency_ms=chunk_latency_ms,
            ):
                yield frame
        except asyncio.CancelledError:
            logger.info("stream_agent_reply cancelled for turn %s", turn_id)
            raise
        finally:
            async with self._state_lock:
                if self._current_tts_task is current_task:
                    self._current_tts_task = None