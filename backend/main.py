from dotenv import load_dotenv

load_dotenv()

import asyncio
from dataclasses import dataclass
import logging
import os
import re
import time
from typing import Dict, List, Optional, Set

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
from backend.control.turn_controller import TurnController
from backend.tools.executor import ToolExecutionHarness
from backend.tts.rime_plugin import (
    FencedRimeTTS,
    PCMFramer,
    RimeTTSHTTPError,
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
# Strict LiveKit direct capture (queue_size_ms=0) requires exact 10ms frames (240 samples / 480 bytes)
FRAME_DURATION_MS = 10
SAMPLES_PER_FRAME = int(SAMPLE_RATE * (FRAME_DURATION_MS / 1000.0))  # 240 samples
BYTES_PER_SAMPLE = 2  # 16-bit PCM
FRAME_SIZE_BYTES = SAMPLES_PER_FRAME * NUM_CHANNELS * BYTES_PER_SAMPLE  # 480 bytes

BARGE_IN_DEBOUNCE_SEC = 0.080

CONTROL_STOP_PATTERNS = [
    r"^wait\s+a\s+(minute|second|sec)",
    r"^hold\s+on(\s+a\s+second)?",
    r"^(wait|stop|hold\s+on|hang\s+on|listen|pause|shut\s+up)(\s+(wait|stop|please|now))*",
    r"^no(\s+no)*$",
]


def classify_utterance(raw_text: str) -> tuple[str, str]:
    text = raw_text.strip().lower()
    cleaned = text

    for pat in CONTROL_STOP_PATTERNS:
        match = re.search(pat, cleaned)
        if match:
            cleaned = cleaned[match.end():].lstrip(" ,.?!")
            break

    if not cleaned or len(cleaned) <= 1:
        return ("control_only", "")

    if cleaned != text and len(cleaned) > 2:
        return ("redirect", cleaned)

    return ("query", raw_text)


@dataclass
class TurnIntent:
    kind: str  # "vad_start" | "stt_final"
    payload: str = ""


@dataclass
class AudioSegment:
    segment_id: int
    epoch: int
    text: str
    is_bridge: bool = False
    total_frames: int = 0
    captured_frames: int = 0
    published: bool = False


class AuthoritativeAudioPump:
    """
    Jitter-buffered publication pump (150ms runway) with synchronous purge.
    Authoritatively tracks hardware capture to guarantee accuracy of the spoken ledger.
    """

    def __init__(self, source: rtc.AudioSource, sample_rate: int = SAMPLE_RATE):
        self.source = source
        self.sample_rate = sample_rate
        # 15 frames = 150ms playout runway; purges synchronously without latency penalties
        self.queue: asyncio.Queue[tuple[int, int, bytes]] = asyncio.Queue(maxsize=15)
        self.active_epoch = 1
        self.pump_task: asyncio.Task | None = None
        self.running = False
        self.stopped = False

        self.segments: Dict[int, AudioSegment] = {}
        self._segment_counter = 0

    def start(self) -> asyncio.Task:
        self.running = True
        self.stopped = False
        self.pump_task = asyncio.create_task(self._drain_loop(), name="audio_pump_drain")
        return self.pump_task

    def register_segment(self, epoch: int, text: str, is_bridge: bool = False) -> int:
        self._segment_counter += 1
        seg_id = self._segment_counter
        self.segments[seg_id] = AudioSegment(
            segment_id=seg_id,
            epoch=epoch,
            text=text,
            is_bridge=is_bridge,
        )
        return seg_id

    def cut_audio_immediate(self) -> None:
        """Atomic invalidation, queue clearance, and native WebRTC hardware purge."""
        self.active_epoch = -1

        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except (asyncio.QueueEmpty, ValueError):
                break

        if hasattr(self.source, "clear_queue"):
            try:
                self.source.clear_queue()
            except Exception as e:
                logger.debug(f"[AudioPump] clear_queue: {e}")

    def sync_epoch(self, epoch: int) -> None:
        self.active_epoch = epoch

    def get_authoritative_published_text(self, epoch: int) -> str:
        """
        Returns only the text of non-bridge clauses that physically crossed
        the native capture_frame() boundary before the cutoff.
        """
        published: List[str] = []
        for seg in self.segments.values():
            if seg.epoch == epoch and not seg.is_bridge and seg.captured_frames > 0:
                ratio = seg.captured_frames / max(1, seg.total_frames)
                if ratio >= 0.4:
                    published.append(seg.text)
        return " ".join(published).strip()

    async def push_frame(
        self,
        epoch: int,
        seg_id: int,
        frame: bytes,
        turn_active: asyncio.Event | None = None,
    ) -> bool:
        if self.stopped or not self.running or epoch != self.active_epoch:
            return False

        if turn_active is not None and not turn_active.is_set():
            return False

        if seg_id in self.segments:
            self.segments[seg_id].total_frames += 1

        try:
            await asyncio.wait_for(self.queue.put((epoch, seg_id, frame)), timeout=0.12)
            return True
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return False

    async def _drain_loop(self) -> None:
        while self.running:
            try:
                epoch, seg_id, frame_bytes = await self.queue.get()
            except asyncio.CancelledError:
                break

            # Pre-capture fence
            if epoch != self.active_epoch:
                self.queue.task_done()
                continue

            frame = rtc.AudioFrame(
                data=bytearray(frame_bytes),
                sample_rate=self.sample_rate,
                num_channels=NUM_CHANNELS,
                samples_per_channel=SAMPLES_PER_FRAME,
            )

            try:
                await self.source.capture_frame(frame)

                # Post-capture fence: if generation was invalidated mid-FFI, clear hardware buffer
                if epoch != self.active_epoch:
                    if hasattr(self.source, "clear_queue"):
                        self.source.clear_queue()
                else:
                    if seg_id in self.segments:
                        self.segments[seg_id].captured_frames += 1

            except Exception as e:
                logger.error(f"[AudioPump] capture_frame error: {e}")
            finally:
                self.queue.task_done()

    async def wait_until_drained(self, timeout: float = 3.0) -> bool:
        try:
            await asyncio.wait_for(self.queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            return False

        if hasattr(self.source, "wait_for_playout"):
            try:
                await asyncio.wait_for(self.source.wait_for_playout(), timeout=0.5)
            except Exception:
                pass
        else:
            await asyncio.sleep(0.02)
        return True

    async def stop(self) -> None:
        self.stopped = True
        self.running = False
        self.cut_audio_immediate()
        if self.pump_task and not self.pump_task.done():
            self.pump_task.cancel()
            try:
                await self.pump_task
            except asyncio.CancelledError:
                pass


class UtteranceStabilizer:
    """
    Stabilizes rapid Deepgram final transcripts.
    Debounces multiple revisions within 200ms into a single authoritative turn intent.
    """

    def __init__(self, submit_callback):
        self.submit_callback = submit_callback
        self.buffer: str = ""
        self.timer: asyncio.Task | None = None
        self.lock = asyncio.Lock()

    async def on_final(self, transcript: str):
        async with self.lock:
            self.buffer = transcript
            if self.timer and not self.timer.done():
                self.timer.cancel()
            self.timer = asyncio.create_task(self._emit_delayed())

    async def _emit_delayed(self):
        try:
            await asyncio.sleep(0.20)  # 200ms stabilization window
            async with self.lock:
                if self.buffer:
                    self.submit_callback(TurnIntent(kind="stt_final", payload=self.buffer))
                    self.buffer = ""
        except asyncio.CancelledError:
            pass

    async def cancel(self):
        async with self.lock:
            if self.timer and not self.timer.done():
                self.timer.cancel()
            self.buffer = ""


class SessionManager:
    def __init__(self, ctx: JobContext, room: rtc.Room, participant: rtc.RemoteParticipant | None):
        self.ctx = ctx
        self.room = room
        self.participant = participant
        self.turn_controller = TurnController()
        self.tool_harness = ToolExecutionHarness()
        self.tts = FencedRimeTTS(api_key=RIME_API_KEY)

        self.llm_client = groq.LLM(
            api_key=GROQ_API_KEY,
            model="openai/gpt-oss-20b",
        )

        try:
            self.audio_source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS, queue_size_ms=0)
            logger.info("[Session] LiveKit AudioSource initialized with queue_size_ms=0 (direct capture)")
        except TypeError:
            self.audio_source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)

        self.audio_pump = AuthoritativeAudioPump(self.audio_source, SAMPLE_RATE)
        self.audio_track: rtc.LocalAudioTrack | None = None
        self.audio_publication: rtc.LocalTrackPublication | None = None

        self.current_state = "disconnected"
        self.session_tasks: Set[asyncio.Task] = set()

        self.intent_queue: asyncio.Queue[TurnIntent] = asyncio.Queue()
        self.arbiter_task: asyncio.Task | None = None
        self.stabilizer = UtteranceStabilizer(self.enqueue_intent_nowait)

        self.current_turn_task: asyncio.Task | None = None
        self.current_turn_event: asyncio.Event | None = None
        self.current_turn_id: int = 0
        self.current_epoch: int = 0

        self.last_barge_in_time = 0.0
        self.is_closing = False
        self.shutdown_event = asyncio.Event()

    def supervise_task(self, coro, name: str | None = None) -> asyncio.Task | None:
        if self.is_closing:
            return None

        task = asyncio.create_task(coro, name=name)
        self.session_tasks.add(task)

        def _cleanup(t: asyncio.Task):
            self.session_tasks.discard(t)
            if not t.cancelled() and t.exception():
                logger.error(f"[Session] Task {t.get_name()} failed: {t.exception()}")

        task.add_done_callback(_cleanup)
        return task

    async def set_agent_state_if_current(self, epoch: int, state: str) -> None:
        if not self.turn_controller.validate_audio_epoch(epoch):
            return
        self.current_state = state
        try:
            await self.room.local_participant.set_attributes({"lk.agent.state": state})
        except Exception:
            pass

    async def start(self) -> None:
        self.audio_track = rtc.LocalAudioTrack.create_audio_track("agent-mic", self.audio_source)
        pub_opts = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        self.audio_publication = await self.room.local_participant.publish_track(self.audio_track, pub_opts)

        self.current_state = "listening"
        await self.room.local_participant.set_attributes({"lk.agent.state": "listening"})

        pump_task = self.audio_pump.start()
        self.session_tasks.add(pump_task)
        pump_task.add_done_callback(self.session_tasks.discard)

        self.arbiter_task = self.supervise_task(self._turn_arbiter_worker(), name="turn_arbiter")

    def enqueue_intent_nowait(self, intent: TurnIntent) -> None:
        """Synchronous O(1) intent submission without task creation latency."""
        if not self.is_closing:
            self.intent_queue.put_nowait(intent)

    async def _turn_arbiter_worker(self):
        """Single consumer managing state transitions across VAD and STT."""
        while not self.is_closing:
            try:
                intent = await self.intent_queue.get()
            except asyncio.CancelledError:
                break

            try:
                if intent.kind == "vad_start":
                    await self._arbiter_handle_vad_cut()

                elif intent.kind == "stt_final":
                    await self._arbiter_handle_stt_final(intent.payload)

            except Exception as e:
                logger.error(f"[Arbiter] Error on {intent.kind}: {e}", exc_info=True)
            finally:
                self.intent_queue.task_done()

    async def _arbiter_handle_vad_cut(self):
        if self.current_state not in ("speaking", "thinking"):
            return

        now = time.monotonic()
        if now - self.last_barge_in_time < BARGE_IN_DEBOUNCE_SEC:
            return
        self.last_barge_in_time = now

        # 1. IMMEDIATE HARDWARE STOP
        self.audio_pump.cut_audio_immediate()

        # 2. ADVANCE HARDWARE EPOCH INSTANTLY (Prevents any race with in-flight tasks)
        new_epoch = await self.turn_controller.cut_audio(reason="vad_speech_started")
        self.current_epoch = new_epoch
        self.audio_pump.sync_epoch(new_epoch)

        # 3. Halt producer and consumer tasks
        if self.current_turn_event:
            self.current_turn_event.clear()

        task_to_cancel = self.current_turn_task
        self.current_turn_task = None
        if task_to_cancel and not task_to_cancel.done():
            task_to_cancel.cancel()

        # 4. Commit ledger text that actually crossed WebRTC boundary
        spoken_text = self.audio_pump.get_authoritative_published_text(self.current_epoch - 1)
        await self.turn_controller.interrupt_active_turn(
            turn_id=self.current_turn_id,
            published_text=spoken_text,
            reason="vad_speech_started",
        )

        await self.set_agent_state_if_current(new_epoch, "listening")
        logger.info(f"[Arbiter] VAD Cut: Turn {self.current_turn_id} halted. Preserved: '{spoken_text[:40]}...'")

    async def _arbiter_handle_stt_final(self, raw_transcript: str):
        kind, clean_payload = classify_utterance(raw_transcript)
        logger.info(f"[Arbiter] STT Final -> Kind: {kind}, Clean: '{clean_payload}'")

        if kind == "control_only":
            logger.info("[Arbiter] Standalone control utterance consumed. Maintaining listening state.")
            await self._arbiter_handle_vad_cut()
            return

        is_redirect = (kind == "redirect")

        # Invalidate current generation and audio buffer
        if self.current_turn_event:
            self.current_turn_event.clear()

        task_to_cancel = self.current_turn_task
        self.current_turn_task = None
        if task_to_cancel and not task_to_cancel.done():
            task_to_cancel.cancel()

        self.audio_pump.cut_audio_immediate()

        if is_redirect:
            spoken_text = self.audio_pump.get_authoritative_published_text(self.current_epoch)
            await self.turn_controller.interrupt_active_turn(
                turn_id=self.current_turn_id,
                published_text=spoken_text,
                reason="user_redirected",
            )

        # Canonical TurnController start_turn call
        turn_id, epoch = await self.turn_controller.start_turn(clean_payload, is_redirect=is_redirect)
        self.current_turn_id = turn_id
        self.current_epoch = epoch
        self.audio_pump.sync_epoch(epoch)

        turn_event = asyncio.Event()
        turn_event.set()
        self.current_turn_event = turn_event

        self.current_turn_task = self.supervise_task(
            self.execute_turn_stream(clean_payload, turn_id, epoch, turn_event, is_redirect=is_redirect),
            name=f"turn_{turn_id}",
        )

    async def execute_turn_stream(
        self,
        prompt: str,
        turn_id: int,
        epoch: int,
        turn_active: asyncio.Event,
        is_redirect: bool = False,
    ):
        await self.set_agent_state_if_current(epoch, "thinking")

        sentence_queue: asyncio.Queue[tuple[str, bool] | None] = asyncio.Queue()
        full_reply: List[str] = []
        turn_framer = PCMFramer(FRAME_SIZE_BYTES)

        # Bridge UX chrome marked is_bridge=True so it never enters dialogue history
        if is_redirect:
            await sentence_queue.put(("Got it, focusing on that instead:", True))

        chat_ctx = llm.ChatContext()
        chat_ctx.add_message(
            role="system",
            content=(
                "You are an interactive, real-time voice assistant for travel. "
                "Always respond in natural, direct, spoken English. "
                "Never use Markdown formatting, bolding, bullet points, asterisks, "
                "tables, or vertical bars (|). Speak in concise, fluid sentences. "
                "CRITICAL INSTRUCTION: If your previous turn was interrupted, answer the new request directly. "
                "Never repeat or restart explanations from the beginning."
            ),
        )

        for msg in self.turn_controller.build_chat_context_messages(max_recent_turns=10):
            chat_ctx.add_message(role=msg["role"], content=msg["content"])

        async def llm_producer():
            try:
                stream = self.llm_client.chat(chat_ctx=chat_ctx)
                buffer = ""
                async for chunk in stream:
                    if not turn_active.is_set() or not self.turn_controller.validate_audio_epoch(epoch):
                        return

                    content = ""
                    if hasattr(chunk, "delta") and chunk.delta and hasattr(chunk.delta, "content"):
                        content = chunk.delta.content or ""
                    elif hasattr(chunk, "choices") and chunk.choices:
                        content = chunk.choices[0].delta.content or ""

                    if not content:
                        continue

                    full_reply.append(content)
                    buffer += content

                    if any(p in buffer for p in [". ", "? ", "! ", "\n"]):
                        clause = buffer.strip()
                        buffer = ""
                        if clause and any(c.isalnum() for c in clause):
                            await sentence_queue.put((clause, False))

                remainder = buffer.strip()
                if remainder and any(c.isalnum() for c in remainder) and turn_active.is_set():
                    await sentence_queue.put((remainder, False))
            finally:
                await sentence_queue.put(None)

        async def tts_consumer():
            has_started = False
            try:
                while True:
                    if not turn_active.is_set() or not self.turn_controller.validate_audio_epoch(epoch):
                        return

                    item = await sentence_queue.get()
                    if item is None:
                        break

                    clause, is_bridge = item

                    if not has_started:
                        await self.set_agent_state_if_current(epoch, "speaking")
                        has_started = True

                    seg_id = self.audio_pump.register_segment(epoch, clause, is_bridge=is_bridge)

                    async for frame in self.tts.stream_speech(
                        clause,
                        turn_id=epoch,
                        framer=turn_framer,
                        fence_validator=lambda ep: turn_active.is_set() and self.turn_controller.validate_audio_epoch(ep),
                    ):
                        if not turn_active.is_set() or not self.turn_controller.validate_audio_epoch(epoch):
                            return
                        await self.audio_pump.push_frame(epoch, seg_id, frame, turn_active=turn_active)

                if turn_active.is_set() and self.turn_controller.validate_audio_epoch(epoch):
                    for final_frame in turn_framer.flush():
                        await self.audio_pump.push_frame(epoch, 0, final_frame, turn_active=turn_active)

            except TurnInterrupted:
                logger.info(f"[Session] Turn {turn_id} aborted mid-synthesis")
                return

        try:
            await asyncio.gather(llm_producer(), tts_consumer())

            # Playout barrier: wait for audio to physically drain before committing completed response
            if turn_active.is_set() and self.turn_controller.validate_audio_epoch(epoch):
                drained = await self.audio_pump.wait_until_drained(timeout=3.0)
                if drained and turn_active.is_set() and self.turn_controller.validate_audio_epoch(epoch):
                    await self.turn_controller.complete_turn(turn_id, epoch, "".join(full_reply))
                await self.set_agent_state_if_current(epoch, "listening")

        except asyncio.CancelledError:
            raise
        except RimeTTSHTTPError as e:
            logger.error(f"[Session] Rime API failed during turn {turn_id}: {e}")
            await self.set_agent_state_if_current(epoch, "listening")
        except Exception as e:
            logger.error(f"[Session] Turn {turn_id} error: {e}", exc_info=True)
            await self.set_agent_state_if_current(epoch, "listening")

    async def shutdown(self) -> None:
        if self.is_closing:
            return
        self.is_closing = True
        self.shutdown_event.set()

        logger.info("[Session] Initiating ordered shutdown...")

        # Invalidate audio publication boundary first
        self.audio_pump.cut_audio_immediate()

        if self.arbiter_task and not self.arbiter_task.done():
            self.arbiter_task.cancel()

        await self.stabilizer.cancel()

        if self.current_turn_event:
            self.current_turn_event.clear()

        if self.current_turn_task and not self.current_turn_task.done():
            self.current_turn_task.cancel()

        pending = [t for t in list(self.session_tasks) if not t.done()]
        for t in pending:
            t.cancel()

        if pending:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                pass

        await self.audio_pump.stop()
        await self.tts.aclose()

    async def wait_until_closed(self) -> None:
        await self.shutdown_event.wait()


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    room = ctx.room

    try:
        participant = await asyncio.wait_for(ctx.wait_for_participant(), timeout=2.0)
    except asyncio.TimeoutError:
        participant = next(iter(ctx.room.remote_participants.values()), None)

    session = SessionManager(ctx, room, participant)

    try:
        await session.start()

        # Synchronous, non-blocking O(1) submission upon energy detection
        @room.on("participant_speech_started")
        def on_participant_speech_started(p: rtc.RemoteParticipant):
            if session.participant and p.identity != session.participant.identity:
                return
            session.enqueue_intent_nowait(TurnIntent(kind="vad_start"))

        subscribed_tracks: Set[str] = set()

        @room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.TrackPublication,
            participant: rtc.RemoteParticipant,
        ):
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                if track.sid in subscribed_tracks:
                    return
                subscribed_tracks.add(track.sid)
                logger.info(f"[Room] Subscribed to participant audio track: {track.sid}")

                async def stt_forwarder():
                    stt_instance = deepgram.STT(api_key=DEEPGRAM_API_KEY)
                    stt_stream = stt_instance.stream()
                    audio_stream = rtc.AudioStream(track)

                    async def push_audio():
                        try:
                            async for event in audio_stream:
                                stt_stream.push_frame(event.frame)
                        finally:
                            stt_stream.end_input()

                    async def read_transcripts():
                        async for event in stt_stream:
                            if event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                                raw_text = event.alternatives[0].text.strip() if event.alternatives else ""
                                if raw_text:
                                    await session.stabilizer.on_final(raw_text)

                    try:
                        await asyncio.gather(push_audio(), read_transcripts())
                    except asyncio.CancelledError:
                        pass
                    finally:
                        subscribed_tracks.discard(track.sid)
                        await stt_stream.aclose()

                session.supervise_task(stt_forwarder(), name=f"stt_{track.sid}")

        @room.on("disconnected")
        def on_disconnected():
            asyncio.create_task(session.shutdown())

        await session.wait_until_closed()
    finally:
        await session.shutdown()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="dataforge-agent",
            ws_url=LIVEKIT_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
        )
    )