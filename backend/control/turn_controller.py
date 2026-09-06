"""
Monotonic Generation Fence – V2.3 Reference Implementation.
State Machine: CREATED -> ACTIVE -> INVALIDATED -> DRAINING -> TERMINATED
Correctness Guarantee: "Cancellation is best-effort; the Generation Fence is the correctness barrier."
"""

from __future__ import annotations
import asyncio
import uuid
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Awaitable
import logging

logger = logging.getLogger(__name__)


class GenState(Enum):
    CREATED = auto()
    ACTIVE = auto()
    INVALIDATED = auto()
    DRAINING = auto()
    TERMINATED = auto()


@dataclass(slots=True)
class Operation:
    op_id: str
    gen_id: int
    kind: str
    task: asyncio.Task
    created_at: float = field(default_factory=time.monotonic)
    cancelled: bool = False
    result: Any = None
    error: BaseException | None = None


class TurnController:
    def __init__(self, max_retained_generations: int = 50) -> None:
        self._lock = asyncio.Lock()
        self._gen_counter: int = 0
        self._current_gen_id: int = 0
        self._gen_state: Dict[int, GenState] = {}
        self._operations: Dict[str, Operation] = {}
        self._gen_ops: Dict[int, List[str]] = {}
        self.conversation_history: List[Dict[str, Any]] = []
        self._history_snapshots: Dict[int, List[Dict[str, Any]]] = {}
        self._max_retained_generations = max_retained_generations

    @property
    def current_generation_id(self) -> int:
        return self._current_gen_id

    @property
    def is_generation_active(self) -> bool:
        return self._gen_state.get(self._current_gen_id) == GenState.ACTIVE

    def validate_generation(self, gen_id: int) -> bool:
        return gen_id == self._current_gen_id and self._gen_state.get(gen_id) == GenState.ACTIVE

    def validate_operation(self, op_id: str) -> bool:
        op = self._operations.get(op_id)
        if not op:
            return False
        return (
            op.gen_id == self._current_gen_id
            and self._gen_state.get(op.gen_id) == GenState.ACTIVE
            and not op.cancelled
        )

    def _prune_old_generations_locked(self) -> None:
        """Prunes historical state to prevent unbounded memory growth in long-running sessions."""
        if self._gen_counter <= self._max_retained_generations:
            return

        cutoff = self._gen_counter - self._max_retained_generations
        stale_gens = [gid for gid in list(self._gen_state.keys()) if gid < cutoff]

        for gid in stale_gens:
            self._gen_state.pop(gid, None)
            self._history_snapshots.pop(gid, None)
            op_ids = self._gen_ops.pop(gid, [])
            for op_id in op_ids:
                self._operations.pop(op_id, None)

    def _start_generation_locked(self, user_text: str) -> int:
        # Explicitly transition old active generation to INVALIDATED
        if self._current_gen_id != 0 and self._gen_state.get(self._current_gen_id) == GenState.ACTIVE:
            self._gen_state[self._current_gen_id] = GenState.INVALIDATED

        self._gen_counter += 1
        new_gen_id = self._gen_counter
        self._history_snapshots[new_gen_id] = [msg.copy() for msg in self.conversation_history]
        self.conversation_history.append({"role": "user", "content": user_text})
        self._gen_state[new_gen_id] = GenState.ACTIVE
        self._current_gen_id = new_gen_id
        self._gen_ops[new_gen_id] = []

        self._prune_old_generations_locked()

        logger.info(f"[FENCE] Generation {new_gen_id} STARTED | History len={len(self.conversation_history)}")
        return new_gen_id

    async def start_generation(self, user_text: str) -> int:
        async with self._lock:
            return self._start_generation_locked(user_text)

    async def commit_llm_result(self, gen_id: int, assistant_text: str) -> bool:
        async with self._lock:
            if not self.validate_generation(gen_id):
                logger.warning(f"[FENCE] REJECT commit_llm_result: gen={gen_id} current={self._current_gen_id}")
                return False
            self.conversation_history.append({"role": "assistant", "content": assistant_text})
            logger.info(f"[FENCE] Generation {gen_id} COMMITTED assistant msg")
            return True

    def register_operation(self, gen_id: int, kind: str, coro_or_task: Any) -> str:
        """
        Registers an async task/coroutine under generation ownership.
        Rejects stale generations synchronously.
        """
        op_id = f"gen_{gen_id}_op_{uuid.uuid4().hex[:8]}"

        if asyncio.iscoroutine(coro_or_task):
            task = asyncio.create_task(coro_or_task, name=op_id)
        elif isinstance(coro_or_task, asyncio.Task):
            task = coro_or_task
        else:
            raise TypeError("Expected coroutine or asyncio.Task")

        # If the generation is already dead or not active, cancel immediately
        if not self.validate_generation(gen_id):
            task.cancel()
            op = Operation(op_id=op_id, gen_id=gen_id, kind=kind, task=task, cancelled=True)
            self._operations[op_id] = op
            return op_id

        op = Operation(op_id=op_id, gen_id=gen_id, kind=kind, task=task)
        self._operations[op_id] = op
        self._gen_ops.setdefault(gen_id, []).append(op_id)
        logger.debug(f"[FENCE] Registered {kind} op {op_id} for gen {gen_id}")
        return op_id

    async def wait_operation(self, op_id: str) -> Any:
        op = self._operations.get(op_id)
        if not op:
            raise KeyError(f"Operation {op_id} not found")
        try:
            op.result = await op.task
            return op.result
        except asyncio.CancelledError:
            op.cancelled = True
            raise
        except Exception as e:
            op.error = e
            raise

    async def invalidate_current_generation(self, reason: str = "User interruption") -> int:
        async with self._lock:
            invalidated_gen = self._current_gen_id
            if invalidated_gen == 0:
                return 0
            if self._gen_state.get(invalidated_gen) == GenState.INVALIDATED:
                return invalidated_gen
            self._gen_state[invalidated_gen] = GenState.INVALIDATED
            logger.warning(f"[FENCE] Generation {invalidated_gen} INVALIDATED: {reason}")
            return invalidated_gen

    async def rollback_invalidated_generation(self, invalidated_gen_id: int, new_user_text: str) -> int:
        async with self._lock:
            snapshot = self._history_snapshots.get(invalidated_gen_id, [])
            snap_len = len(snapshot)
            original_user_msg = None
            if len(self.conversation_history) > snap_len:
                candidate = self.conversation_history[snap_len]
                if candidate.get("role") == "user":
                    original_user_msg = candidate.copy()

            self.conversation_history = [msg.copy() for msg in snapshot]

            if original_user_msg:
                self.conversation_history.append(original_user_msg)

            self.conversation_history.append({
                "role": "system",
                "content": f"[User interrupted Generation {invalidated_gen_id}. Previous tool execution cancelled.]"
            })

            new_gen_id = self._start_generation_locked(new_user_text)
            self._gen_state[invalidated_gen_id] = GenState.TERMINATED
            return new_gen_id

    async def cancel_generation_ops(self, gen_id: int) -> List[str]:
        """Cancels all active tasks for gen_id and sets state to DRAINING."""
        cancelled_ids = []
        op_ids = self._gen_ops.get(gen_id, [])
        for op_id in op_ids:
            op = self._operations.get(op_id)
            if op and not op.task.done():
                op.task.cancel()
                op.cancelled = True
                cancelled_ids.append(op_id)
        if cancelled_ids or self._gen_state.get(gen_id) == GenState.ACTIVE:
            self._gen_state[gen_id] = GenState.DRAINING
        return cancelled_ids


_turn_controller: TurnController | None = None


def get_turn_controller() -> TurnController:
    global _turn_controller
    if _turn_controller is None:
        _turn_controller = TurnController()
    return _turn_controller