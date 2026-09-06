"""
Agent Orchestrator for dataforge-rime.
Coordinates turn fencing, task ownership, tool execution harnesses,
and interruptible TTS streaming with sub-100ms cancellation guarantees.
"""

import asyncio
import inspect
import logging
from typing import Any, AsyncGenerator, Dict, Optional, Set, Union

from backend.control.turn_controller import TurnController
from backend.tools.executor import ToolExecutionHarness
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
        self._draining_tasks: Set[asyncio.Task] = set()
        self._state_lock = asyncio.Lock()

    def _fence_validator(self, turn_id: int) -> bool:
        """
        Synchronously validates whether the given turn_id is the active generation.
        Directly queries turn_controller without fragile duck-typing or property invocation.
        """
        return self.turn_controller.validate_generation(turn_id)

    async def handle_user_barge_in(self, user_text: str = "[user_interruption]") -> int:
        """
        Invoked immediately upon Voice Activity Detection (VAD) trigger.
        Atomically shifts the generation fence, aborts in-flight tool tasks from
        the previous generation, and cleanly cancels TTS within an 80ms bounded budget.
        """
        async with self._state_lock:
            old_turn_id = self.turn_controller.current_generation_id
            gen_res = self.turn_controller.start_generation(user_text)
            new_turn_id = await gen_res if inspect.isawaitable(gen_res) else gen_res

            old_tts_task = self._current_tts_task
            self._current_tts_task = None

        # 1. Abort any in-flight tool tasks associated with the invalidated generation
        if old_turn_id != 0:
            try:
                await self.turn_controller.cancel_generation_ops(old_turn_id)
            except Exception:
                logger.exception("Error cancelling operations for gen %s", old_turn_id)

        # 2. Cancel and drain the previous generation's TTS task
        if old_tts_task and not old_tts_task.done():
            old_tts_task.cancel()
            self._draining_tasks.add(old_tts_task)
            old_tts_task.add_done_callback(self._draining_tasks.discard)
            try:
                await asyncio.wait_for(old_tts_task, timeout=0.08)
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                logger.warning("TTS task did not finish teardown within 80ms budget; draining in background")
            except Exception:
                logger.exception("Unexpected error during TTS task teardown")

        logger.info("Barge-in handled. Old gen: %s -> New gen: %s", old_turn_id, new_turn_id)
        return int(new_turn_id)

    async def execute_tool_call(
        self, tool_name: str, arguments: Dict[str, Any], turn_id: int
    ) -> Dict[str, Any]:
        """
        Executes a registered tool within the generation fence boundary.
        Delegates completely to ToolExecutionHarness with TurnController tracking.
        """
        return await self.tool_harness.execute_tool(
            tool_name=tool_name,
            arguments=arguments,
            turn_id=turn_id,
            fence_validator=self._fence_validator,
            turn_controller=self.turn_controller,
        )

    async def stream_agent_reply(
        self, text: str, turn_id: int, chunk_latency_ms: int = 20
    ) -> AsyncGenerator[Union[bytes, Any], None]:
        """
        Streams synthesized audio frames under generation fencing.
        Safely registers the active TTS task and rejects stale or duplicate streams.
        """
        if not text or not text.strip():
            return

        # Pre-execution check: turn must be strictly active before initiating stream
        if not self._fence_validator(turn_id):
            logger.warning("Rejecting stream_agent_reply: generation %s is not active", turn_id)
            return

        current_task = asyncio.current_task()
        async with self._state_lock:
            if not self._fence_validator(turn_id):
                return
            # If an existing TTS task is running, cancel it cleanly before registering the new one
            if self._current_tts_task and not self._current_tts_task.done() and self._current_tts_task is not current_task:
                self._current_tts_task.cancel()
            self._current_tts_task = current_task

        try:
            async for frame in self.tts.stream_speech(
                text=text,
                turn_id=turn_id,
                fence_validator=self._fence_validator,
                chunk_latency_ms=chunk_latency_ms,
            ):
                # Final publication fence check right before yielding to external sink
                if not self._fence_validator(turn_id):
                    break
                yield frame
        except asyncio.CancelledError:
            logger.info("stream_agent_reply cancelled for turn %s", turn_id)
            raise
        finally:
            async with self._state_lock:
                if self._current_tts_task is current_task:
                    self._current_tts_task = None