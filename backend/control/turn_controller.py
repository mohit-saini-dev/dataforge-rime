from __future__ import annotations
import asyncio
import logging
from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

class GenState(Enum):
    CREATED = 0
    ACTIVE = 1
    INVALIDATED = 2
    DRAINING = 3
    TERMINATED = 4

@dataclass
class Operation:
    op_id: str
    gen_id: int
    kind: str
    task: asyncio.Task
    cancelled: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class TurnController:
    def __init__(self, max_retained_generations: int = 50):
        self._current_gen_id: int = 0
        self._lock = asyncio.Lock()
        self.conversation_history: List[Dict[str, str]] = []
        self._history_snapshots: Dict[int, List[Dict[str, str]]] = {}
        self._gen_state: Dict[int, GenState] = {}
        self._operations: Dict[str, Operation] = {}
        self._gen_ops: Dict[int, List[str]] = {}
        self.max_retained_generations: int = max_retained_generations
        self._op_counter: int = 0

    @property
    def current_generation_id(self) -> int:
        return self._current_gen_id

    def get_generation_state(self, gen_id: int) -> Optional[GenState]:
        return self._gen_state.get(gen_id)

    def validate_generation(self, gen_id: int) -> bool:
        return gen_id == self._current_gen_id and self._gen_state.get(gen_id) == GenState.ACTIVE

    def validate_operation(self, op_id: str) -> bool:
        op = self._operations.get(op_id)
        if not op:
            return False
        return self.validate_generation(op.gen_id) and not op.cancelled

    async def start_generation(self, user_text: str) -> int:
        async with self._lock:
            return self._start_generation_locked(user_text)

    def _start_generation_locked(self, user_text: str) -> int:
        if self._current_gen_id != 0 and self._gen_state.get(self._current_gen_id) == GenState.ACTIVE:
            self._gen_state[self._current_gen_id] = GenState.INVALIDATED
        
        self._current_gen_id += 1
        new_gen_id = self._current_gen_id
        self._gen_state[new_gen_id] = GenState.ACTIVE
        self.conversation_history.append({"role": "user", "content": user_text})
        self._history_snapshots[new_gen_id] = [msg.copy() for msg in self.conversation_history]
        self._gen_ops[new_gen_id] = []
        self._prune_old_generations_locked()
        return new_gen_id

    async def invalidate_current_generation(self, reason: str = "") -> int:
        async with self._lock:
            curr = self._current_gen_id
            if curr != 0:
                self._gen_state[curr] = GenState.INVALIDATED
            return curr

    def register_operation(self, gen_id: int, kind: str, coro_or_task: Any) -> str:
        if not self.validate_generation(gen_id):
            if isinstance(coro_or_task, asyncio.Task):
                coro_or_task.cancel()
            raise RuntimeError(f"Cannot register operation for inactive generation {gen_id}")

        self._op_counter += 1
        op_id = f"op_{gen_id}_{self._op_counter}"
        
        if isinstance(coro_or_task, asyncio.Task):
            task = coro_or_task
        else:
            task = asyncio.create_task(coro_or_task)

        op = Operation(op_id=op_id, gen_id=gen_id, kind=kind, task=task)
        self._operations[op_id] = op
        self._gen_ops.setdefault(gen_id, []).append(op_id)
        
        def _on_op_done(_):
            ops = self._gen_ops.get(gen_id, [])
            if self._gen_state.get(gen_id) == GenState.DRAINING:
                if all(self._operations[oid].task.done() for oid in ops if oid in self._operations):
                    self._gen_state[gen_id] = GenState.TERMINATED
        task.add_done_callback(_on_op_done)
        return op_id

    async def cancel_generation_ops(self, gen_id: int) -> List[str]:
        async with self._lock:
            op_ids = list(self._gen_ops.get(gen_id, []))
            cancelled_ids = []
            for op_id in op_ids:
                op = self._operations.get(op_id)
                if op and not op.task.done():
                    op.task.cancel()
                    op.cancelled = True
                    cancelled_ids.append(op_id)
            self._gen_state[gen_id] = GenState.DRAINING
            return cancelled_ids

    async def rollback_invalidated_generation(self, invalidated_gen_id: int, new_user_text: str) -> int:
        async with self._lock:
            snapshot = self._history_snapshots.get(invalidated_gen_id, [])
            self.conversation_history = [msg.copy() for msg in snapshot]
            self.conversation_history.append({"role": "system", "content": f"[interrupted Generation {invalidated_gen_id}]"})
            self._gen_state[invalidated_gen_id] = GenState.TERMINATED
            return self._start_generation_locked(new_user_text)

    def _prune_old_generations_locked(self) -> None:
        while len(self._gen_state) > self.max_retained_generations:
            oldest_gen = min(self._gen_state.keys())
            self._gen_state.pop(oldest_gen, None)
            self._history_snapshots.pop(oldest_gen, None)
            op_ids = self._gen_ops.pop(oldest_gen, [])
            for op_id in op_ids:
                self._operations.pop(op_id, None)

    async def commit_llm_result(self, gen_id: int, assistant_response: str) -> bool:
        async with self._lock:
            if not self.validate_generation(gen_id):
                return False
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            return True