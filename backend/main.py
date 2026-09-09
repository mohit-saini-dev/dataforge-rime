from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Optional, Set

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
    stt,
)
from livekit.plugins import deepgram, groq

from backend.config import (
    DEEPGRAM_API_KEY,
    GROQ_API_KEY,
    LIVEKIT_API_KEY,
    LIVEKIT_API_SECRET,
    LIVEKIT_URL,
    RIME_API_KEY,
)
from backend.control.turn_controller import (
    Turn,
    TurnController,
)
from backend.tts.rime_plugin import (
    FencedRimeTTS,
    RimeTTSHTTPError,
    RimeTTSNetworkError,
    TurnInterrupted,
)

if LIVEKIT_URL:
    os.environ["LIVEKIT_URL"] = LIVEKIT_URL
if LIVEKIT_API_KEY:
    os.environ["LIVEKIT_API_KEY"] = LIVEKIT_API_KEY
if LIVEKIT_API_SECRET:
    os.environ["LIVEKIT_API_SECRET"] = LIVEKIT_API_SECRET

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend.main")

SAMPLE_RATE = 24000
NUM_CHANNELS = 1

FRAME_DURATION_MS = 10
FRAME_DURATION_SEC = FRAME_DURATION_MS / 1000.0

SAMPLES_PER_FRAME = 240
FRAME_SIZE_BYTES = 480

SOFTWARE_QUEUE_FRAMES = 8
NATIVE_QUEUE_MS = 50

BARGE_IN_DEBOUNCE_SEC = 0.120

CONTROL_PREFIXES = (
    "wait",
    "wait a second",
    "wait a minute",
    "one second",
    "one moment",
    "hold on",
    "hang on",
    "pause",
    "stop",
    "listen",
    "shut up",
)


@dataclass(slots=True)
class TurnIntent:
    kind: str
    payload: str = ""


def classify_utterance(raw_text: str) -> tuple[str, str]:
    text = " ".join(raw_text.strip().split())
    if not text:
        return "control_only", ""

    lowered = text.casefold()
    if re.fullmatch(r"(?:no\s*)+", lowered):
        return "control_only", ""

    for prefix in sorted(CONTROL_PREFIXES, key=len, reverse=True):
        pattern = rf"^{re.escape(prefix)}(?:[\s,;:!?]+|$)"
        match = re.match(pattern, lowered)
        if match:
            remainder = text[match.end() :].lstrip(" ,;:!?-")
            if not remainder:
                return "control_only", ""
            return "redirect", remainder

    return "query", text


class AuthoritativeAudioPump:
    """
    Real-time 10ms frame scheduler with drift compensation and queue clearance.
    """

    def __init__(self, source: rtc.AudioSource, controller: TurnController) -> None:
        self.source = source
        self.controller = controller
        self.queue: asyncio.Queue[tuple[int, int, bytes]] = asyncio.Queue(
            maxsize=SOFTWARE_QUEUE_FRAMES
        )
        self.active_epoch = 0
        self.running = False
        self.stopped = False
        self.pump_task: Optional[asyncio.Task] = None

    def start(self) -> asyncio.Task:
        if self.running:
            raise RuntimeError("audio pump already running")
        self.running = True
        self.stopped = False
        self.pump_task = asyncio.create_task(self._drain_loop(), name="audio_pump")
        return self.pump_task

    def sync_epoch(self, epoch: int) -> None:
        self.active_epoch = epoch

    async def push_frame(self, epoch: int, clause_id: int, frame_bytes: bytes) -> bool:
        if len(frame_bytes) != FRAME_SIZE_BYTES:
            raise ValueError(f"audio frame must be {FRAME_SIZE_BYTES} bytes")

        if self.stopped or not self.running or epoch != self.active_epoch:
            return False

        ledger = self.controller.get_publication_ledger(epoch)
        if ledger is None:
            return False

        try:
            await self.queue.put((epoch, clause_id, frame_bytes))
            ledger.record_generated_frame(clause_id, SAMPLES_PER_FRAME)
        except asyncio.CancelledError:
            raise

        return epoch == self.active_epoch and not self.stopped

    async def _drain_loop(self) -> None:
        next_deadline: Optional[float] = None

        while self.running:
            try:
                epoch, clause_id, data = await self.queue.get()
            except asyncio.CancelledError:
                break

            try:
                if epoch != self.active_epoch or self.stopped:
                    continue

                now = time.monotonic()
                if next_deadline is None or (now - next_deadline) > (FRAME_DURATION_SEC * 3):
                    next_deadline = now
                else:
                    delay = next_deadline - now
                    if delay > 0:
                        await asyncio.sleep(delay)

                if epoch != self.active_epoch or self.stopped:
                    continue

                frame = rtc.AudioFrame(
                    data=bytearray(data),
                    sample_rate=SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                    samples_per_channel=SAMPLES_PER_FRAME,
                )

                await self.source.capture_frame(frame)

                if epoch != self.active_epoch or self.stopped:
                    if hasattr(self.source, "clear_queue"):
                        self.source.clear_queue()
                    continue

                ledger = self.controller.get_publication_ledger(epoch)
                if ledger is not None:
                    ledger.record_captured_frame(clause_id, SAMPLES_PER_FRAME)

                next_deadline = (next_deadline or time.monotonic()) + FRAME_DURATION_SEC
                drift = time.monotonic() - next_deadline
                if drift > (FRAME_DURATION_SEC * 2):
                    next_deadline = time.monotonic()

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("AudioSource.capture_frame failed")
                if hasattr(self.source, "clear_queue"):
                    try:
                        self.source.clear_queue()
                    except Exception:
                        pass
            finally:
                self.queue.task_done()

    def cut_audio_immediate(self) -> None:
        self.active_epoch = -1
        while True:
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except asyncio.QueueEmpty:
                break

        if hasattr(self.source, "clear_queue"):
            try:
                self.source.clear_queue()
            except Exception:
                pass

    async def wait_until_drained(self, epoch: int, timeout: float = 4.0) -> bool:
        if epoch != self.active_epoch:
            return False
        try:
            await asyncio.wait_for(self.queue.join(), timeout=timeout)
            if epoch != self.active_epoch:
                return False

            if hasattr(self.source, "wait_for_playout"):
                await asyncio.wait_for(self.source.wait_for_playout(), timeout=1.0)
            else:
                await asyncio.sleep(0.04)

            return epoch == self.active_epoch
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return False
        except Exception:
            return False

    async def stop(self) -> None:
        if self.stopped:
            return
        self.stopped = True
        self.running = False
        self.cut_audio_immediate()

        if self.pump_task and not self.pump_task.done():
            self.pump_task.cancel()
            try:
                await self.pump_task
            except asyncio.CancelledError:
                pass


class SessionManager:
    def __init__(self, ctx: JobContext, room: rtc.Room) -> None:
        self.ctx = ctx
        self.room = room
        self.user_identity: Optional[str] = None

        self.controller = TurnController(max_history_turns=50)

        self.tts = FencedRimeTTS(
            speaker="amber",
            model="mist",
            sample_rate=SAMPLE_RATE,
            api_key=RIME_API_KEY,
            speed_alpha=0.95,
        )

        chosen_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        logger.info(f"[Session] Groq LLM initialized with model: {chosen_model}")

        self.llm_client = groq.LLM(
            api_key=GROQ_API_KEY,
            model=chosen_model,
        )

        try:
            self.audio_source = rtc.AudioSource(
                SAMPLE_RATE,
                NUM_CHANNELS,
                queue_size_ms=NATIVE_QUEUE_MS,
            )
            logger.info(f"[Session] AudioSource initialized with queue_size_ms={NATIVE_QUEUE_MS}")
        except TypeError:
            self.audio_source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)

        self.audio_pump = AuthoritativeAudioPump(self.audio_source, self.controller)
        self.audio_track: Optional[rtc.LocalAudioTrack] = None

        self.intent_queue: asyncio.Queue[TurnIntent] = asyncio.Queue()
        self.arbiter_task: Optional[asyncio.Task] = None
        self.session_tasks: Set[asyncio.Task] = set()

        self.current_turn_task: Optional[asyncio.Task] = None
        self.current_turn_event: Optional[asyncio.Event] = None
        self.current_state = "disconnected"

        self.active_speech_epoch = 0
        self._last_barge_in = 0.0
        self._state_sequence = 0
        self.is_closing = False
        self.shutdown_event = asyncio.Event()

    def accepts_participant(self, participant: rtc.RemoteParticipant) -> bool:
        if self.user_identity is None or self.user_identity == participant.identity:
            self.user_identity = participant.identity
            return True
        return False

    def supervise(self, coro, name: str) -> Optional[asyncio.Task]:
        if self.is_closing:
            return None
        task = asyncio.create_task(coro, name=name)
        self.session_tasks.add(task)

        def done_callback(done: asyncio.Task) -> None:
            self.session_tasks.discard(done)
            if done.cancelled():
                return
            try:
                exc = done.exception()
            except asyncio.CancelledError:
                return
            if exc:
                logger.error(f"Task {done.get_name()} failed: {exc}", exc_info=exc)

        task.add_done_callback(done_callback)
        return task

    async def set_state(self, state: str, turn_id: Optional[int] = None) -> None:
        if turn_id is not None and not self.controller.validate_turn(turn_id):
            return

        self._state_sequence += 1
        sequence = self._state_sequence
        self.current_state = state

        try:
            if sequence == self._state_sequence:
                await self.room.local_participant.set_attributes({"lk.agent.state": state})
        except Exception:
            return

    async def start(self) -> None:
        self.audio_track = rtc.LocalAudioTrack.create_audio_track("agent-mic", self.audio_source)
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        await self.room.local_participant.publish_track(self.audio_track, options)

        self.audio_pump.start()
        self.current_state = "listening"
        await self.set_state("listening")
        self.arbiter_task = self.supervise(self._arbiter_worker(), "turn_arbiter")

    async def submit_intent(self, intent: TurnIntent) -> None:
        if not self.is_closing:
            await self.intent_queue.put(intent)

    async def _ensure_speech_episode(self) -> int:
        if self.active_speech_epoch:
            return self.active_speech_epoch
        episode = await self.controller.begin_speech_episode()
        self.active_speech_epoch = episode.speech_epoch
        logger.info(f"[STT Episode] Opened episode {self.active_speech_epoch}")
        return self.active_speech_epoch

    async def _arbiter_worker(self) -> None:
        while not self.is_closing:
            try:
                intent = await self.intent_queue.get()
            except asyncio.CancelledError:
                break

            try:
                if intent.kind == "barge_in":
                    await self._handle_barge_in()
                elif intent.kind == "stt_interim":
                    epoch = await self._ensure_speech_episode()
                    await self.controller.set_interim_transcript(epoch, intent.payload)
                elif intent.kind == "stt_final":
                    epoch = await self._ensure_speech_episode()
                    await self.controller.append_transcript_final(epoch, intent.payload)
                elif intent.kind == "speech_end":
                    await self._handle_speech_end()
            except Exception:
                logger.exception(f"Turn arbiter error on {intent.kind}")
            finally:
                self.intent_queue.task_done()

    async def _handle_barge_in(self) -> None:
        if self.current_state not in ("speaking", "thinking"):
            return

        now = time.monotonic()
        if now - self._last_barge_in < BARGE_IN_DEBOUNCE_SEC:
            return
        self._last_barge_in = now
        await self._interrupt_current_turn("vad_speech")

    async def _cancel_turn_task(self, task: Optional[asyncio.Task]) -> None:
        if not task or task.done():
            return
        task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=0.100)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        except Exception:
            pass

    async def _interrupt_current_turn(self, reason: str) -> None:
        old_turn_id = self.controller.active_turn_id
        old_task = self.current_turn_task
        self.current_turn_task = None

        if self.current_turn_event:
            self.current_turn_event.clear()

        self.audio_pump.cut_audio_immediate()

        new_epoch, spoken = await self.controller.interrupt_and_advance_audio(
            old_turn_id,
            reason=reason,
        )
        if spoken:
            logger.info(f"[Interruption] Preserved spoken text: '{spoken[:60]}...'")

        self.audio_pump.sync_epoch(new_epoch)
        await self._cancel_turn_task(old_task)
        await self.set_state("listening")

    async def _handle_speech_end(self) -> None:
        speech_epoch = self.active_speech_epoch
        if not speech_epoch:
            return

        transcript = await self.controller.endpoint_speech_episode(speech_epoch)
        self.active_speech_epoch = 0
        if not transcript or not transcript.strip():
            return

        kind, payload = classify_utterance(transcript)
        logger.info(f"Speech episode {speech_epoch} => {kind}: '{payload}'")

        if kind == "control_only":
            return

        await self._start_turn(payload, is_redirect=(kind == "redirect"))

    async def _start_turn(self, prompt: str, is_redirect: bool) -> None:
        self.audio_pump.cut_audio_immediate()
        if self.current_turn_event:
            self.current_turn_event.clear()

        old_task = self.current_turn_task
        self.current_turn_task = None
        await self._cancel_turn_task(old_task)

        turn = await self.controller.start_turn(prompt, is_redirect=is_redirect)
        self.audio_pump.sync_epoch(turn.audio_epoch)

        turn_event = asyncio.Event()
        turn_event.set()
        self.current_turn_event = turn_event

        self.current_turn_task = self.supervise(
            self.execute_turn_stream(turn, turn_event),
            f"turn_{turn.turn_id}",
        )

    @staticmethod
    def _extract_text(chunk) -> str:
        delta = getattr(chunk, "delta", None)
        if delta is not None:
            content = getattr(delta, "content", None)
            if content:
                return str(content)

        choices = getattr(chunk, "choices", None)
        if choices:
            choice_delta = getattr(choices[0], "delta", None)
            content = getattr(choice_delta, "content", None) if choice_delta else None
            if content:
                return str(content)
        return ""

    @staticmethod
    def _split_llm_buffer(buffer: str) -> tuple[list[str], str]:
        clauses: list[str] = []
        while True:
            match = re.search(r"[.!?](?:\s+|$)|\n+", buffer)
            if match:
                end = match.end()
                clause = buffer[:end].strip()
                buffer = buffer[end:].lstrip()
                if clause:
                    clauses.append(clause)
                continue

            if len(buffer) >= 380:
                cut = max(
                    buffer.rfind(", ", 0, 420),
                    buffer.rfind("; ", 0, 420),
                    buffer.rfind(" - ", 0, 420),
                    buffer.rfind(" ", 0, 420),
                )
                if cut <= 0:
                    cut = min(420, len(buffer))
                else:
                    if buffer[cut] in ",;":
                        cut += 1
                clauses.append(buffer[:cut].strip())
                buffer = buffer[cut:].lstrip()
                continue
            break
        return clauses, buffer

    async def execute_turn_stream(self, turn: Turn, turn_active: asyncio.Event) -> None:
        turn_id = turn.turn_id
        epoch = turn.audio_epoch
        ledger = turn.publication_ledger

        if ledger is None:
            raise RuntimeError("turn has no publication ledger")

        await self.set_state("thinking", turn_id)

        sentence_queue: asyncio.Queue[Optional[tuple[int, str]]] = asyncio.Queue(maxsize=4)
        full_reply: list[str] = []
        clause_counter = 0

        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(
            role="system",
            content=(
                "You are an articulate, patient, and knowledgeable academic and technical mentor. "
                "Speak in a calm, deliberate, conversational cadence with natural phrasing. "
                "Explain concepts systematically and clearly rather than rushing through shallow summaries. "
                "Never use Markdown, bullets, tables, asterisks, numbered lists, or vertical bars. "
                "If the user interrupted you and changed direction, acknowledge the pivot briefly and answer "
                "the new question directly without restarting previous explanations."
            ),
        )

        for message in self.controller.build_chat_context_messages(max_recent_turns=10):
            chat_ctx.add_message(role=message["role"], content=message["content"])

        if turn.is_redirect:
            await sentence_queue.put((-1, "Understood, focusing on that instead."))

        async def put_sentinel() -> None:
            while True:
                try:
                    await asyncio.wait_for(sentence_queue.put(None), timeout=0.1)
                    break
                except asyncio.TimeoutError:
                    if not turn_active.is_set() or not self.controller.validate_turn(turn_id):
                        break

        async def produce() -> None:
            buffer = ""
            try:
                stream = self.llm_client.chat(chat_ctx=chat_ctx)
                async for chunk in stream:
                    if not turn_active.is_set() or not self.controller.validate_turn(turn_id):
                        return

                    content = self._extract_text(chunk)
                    if not content:
                        continue

                    full_reply.append(content)
                    buffer += content
                    clauses, buffer = self._split_llm_buffer(buffer)

                    for clause in clauses:
                        if clause and any(ch.isalnum() for ch in clause):
                            if not turn_active.is_set() or not self.controller.validate_turn(turn_id):
                                return
                            await sentence_queue.put((0, clause))

                if buffer.strip() and turn_active.is_set() and self.controller.validate_turn(turn_id):
                    await sentence_queue.put((0, buffer.strip()))
            finally:
                await put_sentinel()

        async def consume() -> None:
            nonlocal clause_counter
            try:
                while True:
                    if not turn_active.is_set() or not self.controller.validate_turn(turn_id):
                        return

                    item = await sentence_queue.get()
                    try:
                        if item is None:
                            return

                        _, clause = item
                        clause_counter += 1
                        clause_id = clause_counter

                        ledger.register_clause(clause_id, clause)
                        await self.set_state("speaking", turn_id)

                        self._last_barge_in = time.monotonic()

                        async for frame in self.tts.stream_speech(
                            clause,
                            turn_id=epoch,
                            fence_validator=(
                                lambda e: turn_active.is_set()
                                and self.controller.validate_audio_epoch(e)
                                and self.controller.validate_turn(turn_id)
                            ),
                        ):
                            if not turn_active.is_set() or not self.controller.validate_audio_epoch(epoch):
                                raise TurnInterrupted()

                            accepted = await self.audio_pump.push_frame(epoch, clause_id, frame)
                            if not accepted:
                                raise TurnInterrupted()

                        ledger.mark_clause_complete(clause_id)
                    finally:
                        sentence_queue.task_done()
            except asyncio.CancelledError:
                raise

        producer_task = asyncio.create_task(produce(), name=f"llm_{turn_id}")
        consumer_task = asyncio.create_task(consume(), name=f"tts_{turn_id}")

        try:
            await asyncio.gather(producer_task, consumer_task)

            if not turn_active.is_set() or not self.controller.validate_turn(turn_id):
                return

            if not await self.audio_pump.wait_until_drained(epoch):
                raise RuntimeError("Audio playout drain timeout")

            if not await self.controller.complete_turn(turn_id, "".join(full_reply)):
                return

            await self.set_state("listening")

        except (asyncio.CancelledError, TurnInterrupted):
            producer_task.cancel()
            consumer_task.cancel()
            await asyncio.gather(producer_task, consumer_task, return_exceptions=True)

        except (RimeTTSHTTPError, RimeTTSNetworkError):
            logger.exception(f"Rime TTS failed on turn {turn_id}")
            self.audio_pump.cut_audio_immediate()
            await self.controller.fail_turn(turn_id, "tts_error")
            await self.set_state("listening")

        except Exception:
            logger.exception(f"Turn {turn_id} execution failed")
            self.audio_pump.cut_audio_immediate()
            await self.controller.fail_turn(turn_id, "turn_error")
            await self.set_state("listening")

    async def handle_stt_track(self, track: rtc.Track) -> None:
        try:
            stt_instance = deepgram.STT(
                api_key=DEEPGRAM_API_KEY,
                model="nova-2-general",
                language="en-US",
                interim_results=True,
                punctuate=True,
                smart_format=True,
                endpointing_ms=900,
            )
            logger.info("[STT] Deepgram initialized with endpointing_ms=900")
        except TypeError:
            try:
                stt_instance = deepgram.STT(
                    api_key=DEEPGRAM_API_KEY,
                    model="nova-2-general",
                    language="en-US",
                    interim_results=True,
                )
                logger.info("[STT] Deepgram initialized with interim_results")
            except TypeError:
                stt_instance = deepgram.STT(api_key=DEEPGRAM_API_KEY)

        stream = stt_instance.stream()
        audio_stream = rtc.AudioStream(track)

        async def push_audio() -> None:
            try:
                async for frame_event in audio_stream:
                    frame = getattr(frame_event, "frame", frame_event)
                    stream.push_frame(frame)
            finally:
                try:
                    stream.end_input()
                except Exception:
                    pass

        async def read_events() -> None:
            async for event in stream:
                event_type = getattr(event, "type", None)

                if event_type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                    text = event.alternatives[0].text if event.alternatives else ""
                    if text:
                        await self.submit_intent(TurnIntent("stt_interim", text))

                elif event_type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    text = event.alternatives[0].text if event.alternatives else ""
                    if text:
                        logger.info(f"[STT Final] '{text}'")
                        await self.submit_intent(TurnIntent("stt_final", text))

                elif (
                    event_type == stt.SpeechEventType.END_OF_SPEECH
                    or (hasattr(stt.SpeechEventType, "END_OF_SPEECH") and event_type == stt.SpeechEventType.END_OF_SPEECH)
                ):
                    await self.submit_intent(TurnIntent("speech_end"))

        push_task = asyncio.create_task(push_audio(), name="stt_audio_push")
        read_task = asyncio.create_task(read_events(), name="stt_event_reader")

        try:
            done, pending = await asyncio.wait(
                (push_task, read_task),
                return_when=asyncio.FIRST_EXCEPTION,
            )
            for task in done:
                exc = task.exception()
                if exc:
                    raise exc
            await asyncio.gather(*pending)
        except asyncio.CancelledError:
            raise
        finally:
            for task in (push_task, read_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(push_task, read_task, return_exceptions=True)

            try:
                await audio_stream.aclose()
            except Exception:
                pass
            try:
                await stream.aclose()
            except Exception:
                pass

    async def wait_until_closed(self) -> None:
        await self.shutdown_event.wait()

    async def shutdown(self) -> None:
        if self.is_closing:
            return
        self.is_closing = True
        self.shutdown_event.set()

        self.audio_pump.cut_audio_immediate()
        if self.current_turn_event:
            self.current_turn_event.clear()

        cur_task = self.current_turn_task
        self.current_turn_task = None
        await self._cancel_turn_task(cur_task)

        if self.arbiter_task and not self.arbiter_task.done():
            self.arbiter_task.cancel()
            try:
                await self.arbiter_task
            except asyncio.CancelledError:
                pass

        for task in list(self.session_tasks):
            if not task.done():
                task.cancel()

        if self.session_tasks:
            await asyncio.gather(*list(self.session_tasks), return_exceptions=True)

        await self.audio_pump.stop()
        await self.tts.aclose()


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    room = ctx.room

    session = SessionManager(ctx, room)

    @room.on("participant_speech_started")
    def on_participant_speech_started(p: rtc.RemoteParticipant):
        if session.accepts_participant(p):
            asyncio.create_task(session.submit_intent(TurnIntent("barge_in")))

    subscribed_tracks: Set[str] = set()

    async def attach_track(track: rtc.Track, participant: rtc.RemoteParticipant) -> None:
        if not session.accepts_participant(participant):
            return
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return
        if track.sid in subscribed_tracks:
            return

        subscribed_tracks.add(track.sid)
        logger.info(f"[Audio Ingest] Track {track.sid} attached for {participant.identity}")
        task = session.supervise(session.handle_stt_track(track), f"stt_{track.sid}")
        if task is None:
            subscribed_tracks.discard(track.sid)

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        asyncio.create_task(attach_track(track, participant))

    @room.on("track_unsubscribed")
    def on_track_unsubscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        subscribed_tracks.discard(track.sid)

    @room.on("disconnected")
    def on_room_disconnected(*args):
        asyncio.create_task(session.shutdown())

    try:
        await session.start()

        for remote_participant in room.remote_participants.values():
            for publication in remote_participant.track_publications.values():
                if publication.track is not None:
                    await attach_track(publication.track, remote_participant)

        await session.wait_until_closed()
    except asyncio.CancelledError:
        raise
    finally:
        await session.shutdown()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))