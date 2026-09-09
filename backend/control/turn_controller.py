from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger("backend.turn_controller")


class TurnStatus(str, Enum):
    ACTIVE = "active"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class SpeechEpisode:
    speech_epoch: int
    started_at: float = field(default_factory=time.monotonic)
    transcript_segments: List[str] = field(default_factory=list)
    latest_interim: str = ""
    endpointed: bool = False
    committed: bool = False

    def append_final(self, text: str) -> None:
        cleaned = " ".join(text.split())
        if not cleaned:
            return
        if self.transcript_segments and self.transcript_segments[-1].lower() == cleaned.lower():
            return
        self.transcript_segments.append(cleaned)

    @property
    def transcript(self) -> str:
        return " ".join(part for part in self.transcript_segments if part).strip()


@dataclass(slots=True)
class PublicationEntry:
    clause_id: int
    text: str
    total_frames: int = 0
    captured_frames: int = 0
    total_samples: int = 0
    captured_samples: int = 0
    complete: bool = False
    invalidated: bool = False

    @property
    def fully_published(self) -> bool:
        return (
            self.complete
            and not self.invalidated
            and self.captured_frames >= self.total_frames
            and self.total_frames > 0
        )

    def was_substantially_published(self, min_ratio: float = 0.75) -> bool:
        if self.fully_published:
            return True
        if self.total_frames > 0:
            return (self.captured_frames / self.total_frames) >= min_ratio
        return False


@dataclass(slots=True)
class PublicationLedger:
    generation_id: int
    entries: OrderedDict[int, PublicationEntry] = field(default_factory=OrderedDict)
    invalidated: bool = False

    def register_clause(self, clause_id: int, text: str) -> None:
        self.entries[clause_id] = PublicationEntry(clause_id=clause_id, text=text)

    def record_generated_frame(self, clause_id: int, samples: int) -> None:
        if clause_id in self.entries:
            entry = self.entries[clause_id]
            entry.total_frames += 1
            entry.total_samples += samples

    def record_captured_frame(self, clause_id: int, samples: int) -> None:
        if clause_id in self.entries:
            entry = self.entries[clause_id]
            entry.captured_frames += 1
            entry.captured_samples += samples

    def mark_clause_complete(self, clause_id: int) -> None:
        if clause_id in self.entries:
            self.entries[clause_id].complete = True

    def invalidate(self) -> None:
        self.invalidated = True
        for entry in self.entries.values():
            if not entry.fully_published:
                entry.invalidated = True

    def published_text(self, min_ratio: float = 0.75) -> str:
        parts = []
        for entry in self.entries.values():
            if entry.was_substantially_published(min_ratio=min_ratio):
                parts.append(entry.text)
        return " ".join(parts).strip()


@dataclass(slots=True)
class AudioGeneration:
    audio_epoch: int
    turn_id: int
    ledger: PublicationLedger
    started_at: float = field(default_factory=time.monotonic)
    invalidated: bool = False


@dataclass(slots=True)
class Turn:
    turn_id: int
    user_prompt: str
    is_redirect: bool
    status: TurnStatus = TurnStatus.ACTIVE
    audio_epoch: int = 0
    assistant_generated_text: str = ""
    publication_ledger: Optional[PublicationLedger] = None
    interrupted_reason: str = ""
    created_at: float = field(default_factory=time.monotonic)
    completed_at: float = 0.0


class TurnController:
    def __init__(self, max_history_turns: int = 50) -> None:
        self._lock = asyncio.Lock()
        self._speech_counter = 0
        self._turn_counter = 0
        self._audio_counter = 0

        self._active_speech_epoch = 0
        self._active_turn_id = 0
        self._active_audio_epoch = 0

        self._speech_episodes: OrderedDict[int, SpeechEpisode] = OrderedDict()
        self._turns: OrderedDict[int, Turn] = OrderedDict()
        self._audio_generations: Dict[int, AudioGeneration] = {}
        self.max_history_turns = max(4, max_history_turns)

    @property
    def speech_epoch(self) -> int:
        return self._active_speech_epoch

    @property
    def active_turn_id(self) -> int:
        return self._active_turn_id

    @property
    def audio_epoch(self) -> int:
        return self._active_audio_epoch

    @property
    def turns(self) -> OrderedDict[int, Turn]:
        return self._turns

    def get_turn(self, turn_id: int) -> Optional[Turn]:
        return self._turns.get(turn_id)

    def validate_speech_epoch(self, epoch: int) -> bool:
        return epoch != 0 and epoch == self._active_speech_epoch

    def validate_audio_epoch(self, epoch: int) -> bool:
        return epoch != 0 and epoch == self._active_audio_epoch

    def validate_turn(self, turn_id: int) -> bool:
        turn = self._turns.get(turn_id)
        return bool(turn and turn_id == self._active_turn_id and turn.status == TurnStatus.ACTIVE)

    async def begin_speech_episode(self) -> SpeechEpisode:
        async with self._lock:
            self._speech_counter += 1
            self._active_speech_epoch = self._speech_counter
            episode = SpeechEpisode(speech_epoch=self._active_speech_epoch)
            self._speech_episodes[episode.speech_epoch] = episode

            while len(self._speech_episodes) > self.max_history_turns * 2:
                self._speech_episodes.popitem(last=False)
            return episode

    async def append_transcript_final(self, speech_epoch: int, text: str) -> bool:
        async with self._lock:
            episode = self._speech_episodes.get(speech_epoch)
            if not episode or episode.committed:
                return False
            episode.append_final(text)
            return True

    async def set_interim_transcript(self, speech_epoch: int, text: str) -> bool:
        async with self._lock:
            episode = self._speech_episodes.get(speech_epoch)
            if not episode or episode.committed:
                return False
            episode.latest_interim = " ".join(text.split())
            return True

    async def endpoint_speech_episode(self, speech_epoch: int) -> Optional[str]:
        async with self._lock:
            episode = self._speech_episodes.get(speech_epoch)
            if not episode or episode.committed:
                return None
            episode.endpointed = True
            transcript = episode.transcript or episode.latest_interim
            episode.committed = True
            return transcript.strip()

    async def start_turn(self, user_prompt: str, is_redirect: bool = False) -> Turn:
        async with self._lock:
            prior = self._turns.get(self._active_turn_id)
            if prior and prior.status == TurnStatus.ACTIVE:
                prior.status = TurnStatus.INTERRUPTED
                prior.interrupted_reason = "superseded_by_new_turn"
                if prior.publication_ledger:
                    prior.publication_ledger.invalidate()
                    prior.assistant_generated_text = prior.publication_ledger.published_text(min_ratio=0.75)

            self._turn_counter += 1
            turn = Turn(
                turn_id=self._turn_counter,
                user_prompt=user_prompt.strip(),
                is_redirect=is_redirect,
            )
            self._turns[turn.turn_id] = turn
            self._active_turn_id = turn.turn_id

            self._audio_counter += 1
            self._active_audio_epoch = self._audio_counter

            ledger = PublicationLedger(generation_id=self._active_audio_epoch)
            turn.audio_epoch = self._active_audio_epoch
            turn.publication_ledger = ledger

            self._audio_generations[self._active_audio_epoch] = AudioGeneration(
                audio_epoch=self._active_audio_epoch,
                turn_id=turn.turn_id,
                ledger=ledger,
            )

            self._prune_locked()
            logger.info(
                f"[TurnController] Started Turn {turn.turn_id} (Epoch {self._active_audio_epoch}, redirect={is_redirect})"
            )
            return turn

    async def interrupt_and_advance_audio(self, turn_id: int, reason: str = "vad_speech") -> tuple[int, str]:
        """
        Atomic operation: freezes ledger, computes spoken text, marks interrupted,
        and advances audio generation epoch without orphaning metrics.
        """
        async with self._lock:
            spoken_text = ""
            turn = self._turns.get(turn_id)
            if turn and turn.status == TurnStatus.ACTIVE:
                turn.status = TurnStatus.INTERRUPTED
                turn.interrupted_reason = reason
                if turn.publication_ledger:
                    turn.publication_ledger.invalidate()
                    spoken_text = turn.publication_ledger.published_text(min_ratio=0.75)
                    turn.assistant_generated_text = spoken_text

            if self._active_audio_epoch:
                old = self._audio_generations.get(self._active_audio_epoch)
                if old:
                    old.invalidated = True
                    old.ledger.invalidate()

            self._audio_counter += 1
            self._active_audio_epoch = self._audio_counter
            return self._active_audio_epoch, spoken_text

    async def complete_turn(self, turn_id: int, full_response: str) -> bool:
        async with self._lock:
            turn = self._turns.get(turn_id)
            if not turn or turn_id != self._active_turn_id or turn.status != TurnStatus.ACTIVE:
                return False

            turn.status = TurnStatus.COMPLETED
            turn.assistant_generated_text = full_response.strip()
            turn.completed_at = time.monotonic()
            logger.info(f"[TurnController] Turn {turn_id} marked COMPLETED.")
            return True

    async def fail_turn(self, turn_id: int, reason: str) -> bool:
        async with self._lock:
            turn = self._turns.get(turn_id)
            if not turn or turn.status != TurnStatus.ACTIVE:
                return False

            turn.status = TurnStatus.FAILED
            turn.interrupted_reason = reason
            if turn.publication_ledger:
                turn.publication_ledger.invalidate()
            return True

    def get_publication_ledger(self, audio_epoch: int) -> Optional[PublicationLedger]:
        gen = self._audio_generations.get(audio_epoch)
        return gen.ledger if gen else None

    def build_chat_context_messages(self, max_recent_turns: int = 10) -> List[Dict[str, str]]:
        result: List[Dict[str, str]] = []
        turn_items = list(self._turns.values())[-max_recent_turns:]

        for turn in turn_items:
            if not turn.user_prompt or turn.status == TurnStatus.FAILED:
                continue

            result.append({"role": "user", "content": turn.user_prompt})

            if turn.status == TurnStatus.COMPLETED and turn.assistant_generated_text:
                result.append({"role": "assistant", "content": turn.assistant_generated_text})
            elif turn.status == TurnStatus.INTERRUPTED:
                spoken = turn.assistant_generated_text
                if not spoken and turn.publication_ledger:
                    spoken = turn.publication_ledger.published_text(min_ratio=0.75)
                if spoken:
                    result.append({"role": "assistant", "content": f"{spoken} [interrupted by user]"})

        return result

    def _prune_locked(self) -> None:
        while len(self._turns) > self.max_history_turns:
            prune_key = None
            for tid, t in self._turns.items():
                if tid != self._active_turn_id and t.status != TurnStatus.ACTIVE:
                    prune_key = tid
                    break
            if prune_key is not None:
                self._turns.pop(prune_key, None)
            else:
                break