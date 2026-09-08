"""
TurnController: Strict 3-clock conversational manager (speech_epoch, audio_epoch, turn_id).
Enforces authoritative spoken-ledger semantics, structured ChatML context generation,
bounded recency windows, and O(1) publication fencing.
"""

from __future__ import annotations
import asyncio
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("backend.turn_controller")


class TurnStatus(Enum):
    IDLE = 0
    ACTIVE = 1
    INTERRUPTED = 2
    COMPLETED = 3


@dataclass
class AssistantSegment:
    segment_id: int
    text: str
    published: bool = False
    timestamp_ns: int = field(default_factory=time.perf_counter_ns)


@dataclass
class ConversationalTurn:
    turn_id: int
    user_prompt: str
    is_redirect: bool = False
    status: TurnStatus = TurnStatus.ACTIVE
    assistant_segments: List[AssistantSegment] = field(default_factory=list)
    interrupted_reason: str = ""
    created_at_ns: int = field(default_factory=time.perf_counter_ns)


class TurnController:
    """
    Authoritative state machine decoupling:
    1. Speech Epoch: Deduplicates VAD vocal episodes.
    2. Audio Epoch: Instantaneous, atomic WebRTC hardware publication fence.
    3. Turn ID: Conversational dialogue lifecycle and LLM context history.
    """

    def __init__(self, max_history_turns: int = 50):
        self._lock = asyncio.Lock()

        # Clock 1: User Speech Episode (VAD)
        self._speech_epoch: int = 0

        # Clock 2: Audio Publication Boundary (Hardware cutoff fence)
        self._audio_epoch: int = 0

        # Clock 3: Conversational Turn Identifier
        self._current_turn_id: int = 0
        self._active_turn_id: int = 0

        self.turns: Dict[int, ConversationalTurn] = {}
        self.max_history_turns: int = max_history_turns

    @property
    def audio_epoch(self) -> int:
        return self._audio_epoch

    @property
    def speech_epoch(self) -> int:
        return self._speech_epoch

    @property
    def active_turn_id(self) -> int:
        return self._active_turn_id

    def validate_audio_epoch(self, epoch: int) -> bool:
        """O(1) scalar check for AudioPump hardware publication."""
        return epoch == self._audio_epoch

    def validate_turn(self, turn_id: int) -> bool:
        """O(1) scalar check to verify if a turn is still active."""
        turn = self.turns.get(turn_id)
        return (
            turn is not None
            and turn.turn_id == self._active_turn_id
            and turn.status == TurnStatus.ACTIVE
        )

    async def begin_speech_episode(self) -> tuple[int, int]:
        """
        Invoked on VAD start.
        Advances speech_epoch and audio_epoch simultaneously.
        Returns: (speech_epoch, audio_epoch)
        """
        async with self._lock:
            self._speech_epoch += 1
            self._audio_epoch += 1
            return self._speech_epoch, self._audio_epoch

    async def cut_audio(self, reason: str = "barge_in") -> int:
        """Immediate hardware fence invalidation without mutating dialogue."""
        async with self._lock:
            self._audio_epoch += 1
            logger.info(f"[TurnController] Audio cut -> Epoch {self._audio_epoch} (Reason: {reason})")
            return self._audio_epoch

    async def start_turn(self, user_prompt: str, is_redirect: bool = False) -> tuple[int, int]:
        """
        Opens a new conversational turn upon receiving a validated substantive STT final.
        Returns: (turn_id, audio_epoch)
        """
        async with self._lock:
            if self._active_turn_id in self.turns:
                prior = self.turns[self._active_turn_id]
                if prior.status == TurnStatus.ACTIVE:
                    prior.status = TurnStatus.INTERRUPTED
                    prior.interrupted_reason = "superseded_by_new_turn"

            self._current_turn_id += 1
            self._active_turn_id = self._current_turn_id
            self._audio_epoch += 1

            turn = ConversationalTurn(
                turn_id=self._active_turn_id,
                user_prompt=user_prompt.strip(),
                is_redirect=is_redirect,
                status=TurnStatus.ACTIVE,
            )
            self.turns[self._active_turn_id] = turn
            self._prune_turns_locked()

            logger.info(
                f"[TurnController] Started Turn {turn.turn_id} (Epoch {self._audio_epoch}, redirect={is_redirect})"
            )
            return turn.turn_id, self._audio_epoch

    async def interrupt_active_turn(self, turn_id: int, published_text: str, reason: str = "vad_speech") -> int:
        """
        Idempotent turn cutoff.
        Records ONLY the text that AudioPump confirmed crossed the WebRTC boundary.
        """
        async with self._lock:
            self._audio_epoch += 1

            turn = self.turns.get(turn_id)
            if not turn or turn.status != TurnStatus.ACTIVE:
                return self._audio_epoch

            turn.status = TurnStatus.INTERRUPTED
            turn.interrupted_reason = reason

            clean_text = published_text.strip()
            if clean_text:
                turn.assistant_segments.append(
                    AssistantSegment(segment_id=0, text=clean_text, published=True)
                )
                logger.info(f"[TurnController] Turn {turn_id} interrupted. Published text preserved: '{clean_text[:50]}...'")

            return self._audio_epoch

    async def complete_turn(self, turn_id: int, epoch: int, full_response: str) -> bool:
        """Commits full response only if the turn and audio epoch remained clean throughout playout."""
        async with self._lock:
            if epoch != self._audio_epoch:
                logger.info(f"[TurnController] Stale audio completion for Epoch {epoch}")
                return False

            turn = self.turns.get(turn_id)
            if not turn or turn.status != TurnStatus.ACTIVE:
                logger.info(f"[TurnController] Cannot complete non-active Turn {turn_id}")
                return False

            turn.status = TurnStatus.COMPLETED
            turn.assistant_segments.clear()
            turn.assistant_segments.append(
                AssistantSegment(segment_id=0, text=full_response.strip(), published=True)
            )
            logger.info(f"[TurnController] Turn {turn_id} successfully completed.")
            return True

    def build_chat_context_messages(self, max_recent_turns: int = 10) -> List[Dict[str, str]]:
        """
        Renders the internal turn state machine into structured ChatML messages.
        Bounded to the most recent turns to fit within LLM context windows.
        """
        messages: List[Dict[str, str]] = []

        sorted_turn_ids = sorted(self.turns.keys())
        if len(sorted_turn_ids) > max_recent_turns:
            sorted_turn_ids = sorted_turn_ids[-max_recent_turns:]

        for turn_id in sorted_turn_ids:
            turn = self.turns[turn_id]
            if not turn.user_prompt:
                continue

            messages.append({"role": "user", "content": turn.user_prompt})

            if turn.status == TurnStatus.COMPLETED and turn.assistant_segments:
                messages.append({"role": "assistant", "content": turn.assistant_segments[0].text})
            elif turn.status == TurnStatus.INTERRUPTED and turn.assistant_segments:
                spoken = turn.assistant_segments[0].text
                messages.append({"role": "assistant", "content": spoken})
                messages.append({
                    "role": "system",
                    "content": (
                        "[Assistant was interrupted by user while speaking above response. "
                        "Acknowledge the user's redirection concisely and do not repeat previous statements.]"
                    ),
                })

        return messages

    def _prune_turns_locked(self) -> None:
        while len(self.turns) > self.max_history_turns:
            oldest_id = min(self.turns.keys())
            if oldest_id == self._active_turn_id:
                break
            self.turns.pop(oldest_id, None)