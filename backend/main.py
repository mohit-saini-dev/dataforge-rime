import asyncio
import logging
from typing import Set
import time

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
)

from backend.config import (
    GROQ_API_KEY,
    LIVEKIT_API_KEY,
    LIVEKIT_API_SECRET,
    LIVEKIT_URL,
    RIME_API_KEY,
)
from backend.control.turn_controller import TurnController
from backend.tools.executor import ToolExecutionHarness
from backend.tts.rime_plugin import FencedRimeTTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend.main")

SAMPLE_RATE = 24000
NUM_CHANNELS = 1
FRAME_DURATION_MS = 20
SAMPLES_PER_FRAME = int(SAMPLE_RATE * (FRAME_DURATION_MS / 1000.0))  # 480 samples
BYTES_PER_SAMPLE = 2  # 16-bit PCM
FRAME_SIZE_BYTES = SAMPLES_PER_FRAME * NUM_CHANNELS * BYTES_PER_SAMPLE  # 960 bytes
FRAME_DURATION_SEC = FRAME_DURATION_MS / 1000.0  # 0.020s

LLM_STREAM_TIMEOUT_SEC = 10.0
SHUTDOWN_TIMEOUT_SEC = 1.5
BARGE_IN_DEBOUNCE_SEC = 0.200  # 200ms VAD flap debounce


class AudioPump:
    """Real-time paced audio pump with monotonic clock discipline and zero-latency barge-in purge."""

    def __init__(self, source: rtc.AudioSource, sample_rate: int = SAMPLE_RATE):
        self.source = source
        self.sample_rate = sample_rate
        # Buffer limited to 2 frames (40ms) to eliminate native WebRTC buffer bloat
        self._queue: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue(maxsize=2)
        self._active_generation = -1
        self._pump_task: asyncio.Task | None = None
        self._running = False
        self._stopped = False
        self._next_frame_time = 0.0
        self._lock = asyncio.Lock()

    def start(self) -> asyncio.Task:
        self._running = True
        self._stopped = False
        self._pump_task = asyncio.create_task(self._drain_loop(), name="audio_pump_drain")
        return self._pump_task

    def set_generation(self, generation_id: int) -> None:
        """Purge stale frames synchronously upon barge-in."""
        self._active_generation = generation_id
        purged = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                purged += 1
            except (asyncio.QueueEmpty, ValueError):
                break

        # Zero-fill frame to instantly overwrite any lingering native WebRTC buffer
        silence_frame = rtc.AudioFrame(
            data=b"\x00" * FRAME_SIZE_BYTES,
            sample_rate=self.sample_rate,
            num_channels=NUM_CHANNELS,
            samples_per_channel=SAMPLES_PER_FRAME,
        )
        try:
            self.source.capture_frame(silence_frame)
        except Exception:
            pass

        if purged > 0:
            logger.debug(f"[AudioPump] Purged {purged} stale frames for Gen {generation_id}")

    async def push_pcm(self, generation_id: int, pcm_data: bytes) -> None:
        """Atomic check and enqueue to prevent boundary frame leaks."""
        async with self._lock:
            if self._stopped or not self._running or generation_id != self._active_generation:
                return

            for offset in range(0, len(pcm_data), FRAME_SIZE_BYTES):
                if self._stopped or not self._running or generation_id != self._active_generation:
                    return

                chunk = pcm_data[offset : offset + FRAME_SIZE_BYTES]
                if len(chunk) < FRAME_SIZE_BYTES:
                    chunk = chunk + b"\x00" * (FRAME_SIZE_BYTES - len(chunk))

                try:
                    self._queue.put_nowait((generation_id, chunk))
                except asyncio.QueueFull:
                    # Drop frame immediately to preserve strict real-time wall clock
                    return

    async def _drain_loop(self) -> None:
        loop = asyncio.get_running_loop()
        self._next_frame_time = loop.time()

        while self._running:
            try:
                gen_id, frame_bytes = await self._queue.get()
            except asyncio.CancelledError:
                break

            if gen_id != self._active_generation:
                self._queue.task_done()
                continue

            frame = rtc.AudioFrame(
                data=frame_bytes,
                sample_rate=self.sample_rate,
                num_channels=NUM_CHANNELS,
                samples_per_channel=SAMPLES_PER_FRAME,
            )

            # Monotonic Wall-Clock Pacing (50 FPS deadline)
            now = loop.time()
            if self._next_frame_time < now - 0.05:  # Anchor reset if drifted >50ms
                self._next_frame_time = now

            if self._next_frame_time > now:
                await asyncio.sleep(self._next_frame_time - now)

            try:
                await self.source.capture_frame(frame)
            except Exception as e:
                logger.error(f"[AudioPump] Frame capture error: {e}")
            finally:
                self._next_frame_time += FRAME_DURATION_SEC
                self._queue.task_done()

    async def stop(self) -> None:
        self._stopped = True
        self._running = False
        if self._pump_task and not self._pump_task.done():
            self._pump_task.cancel()
            try:
                await self._pump_task
            except asyncio.CancelledError:
                pass


class SessionManager:
    """Supervises participant session with strict admission control and task ownership."""

    def __init__(self, ctx: JobContext, room: rtc.Room, participant: rtc.RemoteParticipant | None):
        self.ctx = ctx
        self.room = room
        self.participant = participant
        self.turn_controller = TurnController()
        self.tool_harness = ToolExecutionHarness(self.turn_controller)
        self.tts = FencedRimeTTS(api_key=RIME_API_KEY)
        self.audio_source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
        self.audio_pump = AudioPump(self.audio_source, SAMPLE_RATE)

        self._session_tasks: Set[asyncio.Task] = set()
        self._current_turn_task: asyncio.Task | None = None
        self.last_barge_in_time = 0.0
        self._is_closing = False
        self._shutdown_event = asyncio.Event()

    def supervise_task(self, coro, name: str | None = None) -> asyncio.Task | None:
        """Atomic Admission Barrier: rejects tasks when shutting down."""
        if self._is_closing:
            logger.warning("[Session] Admission rejected: session is shutting down")
            return None

        task = asyncio.create_task(coro, name=name)
        self._session_tasks.add(task)
        task.add_done_callback(self._session_tasks.discard)
        return task

    async def start(self) -> None:
        track = rtc.LocalAudioTrack.create_audio_track("agent-audio", self.audio_source)
        pub_opts = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        await self.room.local_participant.publish_track(track, pub_opts)

        pump_task = self.audio_pump.start()
        self._session_tasks.add(pump_task)
        pump_task.add_done_callback(self._session_tasks.discard)

    async def handle_barge_in(self, reason: str = "user_speech") -> int | None:
        """Debounced barge-in handling with immediate upstream LLM stream cancellation."""
        now = time.monotonic()
        if now - self.last_barge_in_time < BARGE_IN_DEBOUNCE_SEC:
            logger.debug("[Session] Debounced redundant barge-in trigger")
            return None
        self.last_barge_in_time = now

        # 1. Instantly cancel active LLM / TTS turn task to free HTTP sockets
        if self._current_turn_task and not self._current_turn_task.done():
            self._current_turn_task.cancel()
            self._current_turn_task = None

        # 2. Advance monotonic fence and flush audio pump
        gen = self.turn_controller.start_turn()
        self.audio_pump.set_generation(gen)
        logger.info(f"[Session] Barge-in executed ({reason}) -> Advanced to Gen {gen}")
        return gen

    async def run_guarded_user_turn(self, prompt: str, gen_id: int):
        """Runs the turn with explicit reference tracking and hard timeout."""
        try:
            await asyncio.wait_for(
                self.execute_turn_stream(prompt, gen_id),
                timeout=LLM_STREAM_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            logger.error(f"[Session] Turn {gen_id} timed out; dropped")
        except asyncio.CancelledError:
            logger.info(f"[Session] Turn {gen_id} cancelled cleanly via interruption")
            raise

    async def execute_turn_stream(self, prompt: str, gen_id: int):
        # Conversational generation stream logic
        pass

    async def shutdown(self) -> None:
        """Atomic shutdown barrier with hard timeout to eliminate zombie coroutines."""
        if self._is_closing:
            return
        self._is_closing = True
        self._shutdown_event.set()

        logger.info("[Session] Entering atomic teardown barrier...")

        # 1. Stop audio pump immediately
        await self.audio_pump.stop()

        # 2. Cancel current active turn task explicitly
        if self._current_turn_task and not self._current_turn_task.done():
            self._current_turn_task.cancel()

        # 3. Cancel all owned session tasks concurrently
        pending = [t for t in list(self._session_tasks) if not t.done()]
        for t in pending:
            t.cancel()

        if pending:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=SHUTDOWN_TIMEOUT_SEC,
                )
                logger.info(f"[Session] Teardown finished for {len(pending)} tasks")
            except asyncio.TimeoutError:
                logger.warning(f"[Session] Hard shutdown timeout ({SHUTDOWN_TIMEOUT_SEC}s) reached")

    async def wait_until_closed(self) -> None:
        await self._shutdown_event.wait()


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

        @room.on("participant_speech_started")
        def on_participant_speech_started(_: rtc.RemoteParticipant):
            session.supervise_task(session.handle_barge_in("[voice_interruption]"), name="barge_in_handler")

        @room.on("disconnected")
        def on_disconnected():
            logger.info("[Room] Disconnect received; triggering session shutdown")
            asyncio.create_task(session.shutdown())

        await session.wait_until_closed()
    finally:
        # Guarantee teardown on SIGTERM, worker eviction, or unhandled exceptions
        await session.shutdown()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))