import asyncio
import logging
import re
from typing import AsyncGenerator

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    cli,
    llm,
)
from livekit.plugins import groq

from backend.agent import VoiceAgentOrchestrator
from backend.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LIVEKIT_API_KEY,
    LIVEKIT_API_SECRET,
    LIVEKIT_URL,
    LOG_LEVEL,
)
from backend.control.turn_controller import TurnController
from backend.tts.rime_plugin import FencedRimeTTS

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("voice-agent-master")


class AudioPump:
    """
    Decoupled audio clock and hard publication fence.
    Maintains authoritative generation ownership and flushes obsolete frames instantly.
    """

    def __init__(self, source: rtc.AudioSource, sample_rate: int = 24000):
        self.source = source
        self.sample_rate = sample_rate
        self._queue: asyncio.Queue[tuple[int, rtc.AudioFrame]] = asyncio.Queue(maxsize=150)
        self._active_gen: int = -1
        self._lock = asyncio.Lock()
        self._pump_task: asyncio.Task | None = None

    async def start(self) -> None:
        self._pump_task = asyncio.create_task(self._drain_loop(), name="audio_pump_drain")

    async def stop(self) -> None:
        if self._pump_task:
            self._pump_task.cancel()
            try:
                await self._pump_task
            except asyncio.CancelledError:
                pass
        await self.flush(-1)

    async def set_generation(self, gen_id: int) -> None:
        """Atomic generation switch. Drops all stale audio packets instantly."""
        async with self._lock:
            self._active_gen = gen_id
            # Synchronously purge all pending frames belonging to superseded generations
            purged = 0
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                    purged += 1
                except asyncio.QueueEmpty:
                    break
            if purged > 0:
                logger.debug(f"[AudioPump] Purged {purged} stale frames upon transition to Gen {gen_id}")

    async def flush(self, new_gen: int = -1) -> None:
        await self.set_generation(new_gen)

    async def push_frame(self, gen_id: int, pcm_bytes: bytes) -> None:
        """Publishes a raw PCM audio chunk after applying an entry fence check."""
        async with self._lock:
            if gen_id != self._active_gen:
                return

        samples = len(pcm_bytes) // 2
        frame = rtc.AudioFrame(
            data=pcm_bytes,
            sample_rate=self.sample_rate,
            num_channels=1,
            samples_per_channel=samples,
        )

        try:
            self._queue.put_nowait((gen_id, frame))
        except asyncio.QueueFull:
            logger.warning("[AudioPump] Buffer saturation: dropping frame to protect real-time latency")

    async def _drain_loop(self) -> None:
        while True:
            gen_id, frame = await self._queue.get()
            # Authoritative Final Publication Fence (TOCTOU elimination)
            async with self._lock:
                if gen_id != self._active_gen:
                    continue

            try:
                await self.source.capture_frame(frame)
            except Exception as e:
                logger.error(f"[AudioPump] WebRTC frame capture failure: {e}")


class AgentSession:
    """
    Per-participant session encapsulating isolated generation fencing,
    semantic streaming pipelining, and structured task lifetimes.
    """

    CLAUSE_SPLIT_REGEX = re.compile(r"([.?!,;:\n]+)")

    def __init__(self, ctx: JobContext):
        self.ctx = ctx
        self.room = ctx.room

        # Multi-tenant isolation: new instance per room session
        self.turn_controller = TurnController(max_retained_generations=50)
        self.tts_client = FencedRimeTTS(sample_rate=24000)
        self.orchestrator = VoiceAgentOrchestrator(
            turn_controller=self.turn_controller,
            tts_client=self.tts_client,
        )

        self.audio_source = rtc.AudioSource(sample_rate=24000, num_channels=1)
        self.audio_track = rtc.LocalAudioTrack.create_audio_track("agent-audio", self.audio_source)
        self.audio_pump = AudioPump(self.audio_source, sample_rate=24000)

        self.groq_client = groq.LLM(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
        )

        self._session_tasks: set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()

    def supervise_task(self, coro, name: str) -> asyncio.Task:
        task = asyncio.create_task(coro, name=name)
        self._session_tasks.add(task)
        task.add_done_callback(self._session_tasks.discard)
        return task

    async def start(self) -> None:
        await self.audio_pump.start()
        await self.room.local_participant.publish_track(
            self.audio_track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )

        self._bind_room_events()
        logger.info(f"[AgentSession] Room {self.room.name} connected. Generation engine active.")

    def _bind_room_events(self) -> None:
        @self.room.on("participant_speech_started")
        def on_participant_speech_started(participant: rtc.RemoteParticipant):
            """Immediate barge-in: executes generation fence and clears WebRTC audio buffer."""
            logger.info(f"[Barge-In] Interrupt triggered by {participant.identity}")
            self.supervise_task(self.handle_barge_in("[voice_interruption]"), name="barge_in_handler")

        @self.room.on("data_received")
        def on_data_received(data_packet: rtc.DataPacket):
            payload = data_packet.data.decode("utf-8")
            logger.info(f"[Turn] Inbound payload: {payload}")
            self.supervise_task(self.run_user_turn(payload), name="user_turn_lifecycle")

        @self.room.on("disconnected")
        def on_disconnected():
            logger.info("[AgentSession] Room disconnected; initiating teardown")
            self._shutdown_event.set()

    async def handle_barge_in(self, reason: str = "[interruption]") -> int:
        new_gen = await self.orchestrator.handle_user_barge_in(reason)
        # Flush the WebRTC audio pump immediately
        await self.audio_pump.set_generation(new_gen)
        return new_gen

    async def run_user_turn(self, user_prompt: str) -> None:
        """
        Sub-second conversational streaming pipeline:
        Groq Token Stream -> Semantic Clause Chunker -> Fenced TTS -> AudioPump.
        """
        # Register monotonic generation
        gen_id = await self.turn_controller.start_generation(user_prompt)
        await self.audio_pump.set_generation(gen_id)

        chat_context = llm.ChatContext()
        chat_context.append(
            role="system",
            text=(
                "You are an agile, ultra-concise travel voice assistant. "
                "Answer immediately in 1-2 sharp sentences. Never use Markdown or lists."
            ),
        )
        chat_context.append(role="user", text=user_prompt)

        try:
            llm_stream = self.groq_client.chat(chat_ctx=chat_context)
            async for clause in self._clause_streamer(llm_stream, gen_id):
                if not self.turn_controller.is_active_generation(gen_id):
                    logger.debug(f"[Gen {gen_id}] Pipeline interrupted during clause generation")
                    return

                # Stream synthesized clause into audio pump
                await self._synthesize_clause_to_pump(clause, gen_id)

        except asyncio.CancelledError:
            logger.info(f"[Gen {gen_id}] Conversational pipeline cancelled cleanly")
            raise
        except Exception as e:
            logger.exception(f"[Gen {gen_id}] Pipeline execution failed: {e}")

    async def _clause_streamer(
        self, stream: AsyncGenerator, gen_id: int
    ) -> AsyncGenerator[str, None]:
        """Buffers raw LLM tokens and yields complete linguistic clauses."""
        buffer = ""
        async for chunk in stream:
            if not self.turn_controller.is_active_generation(gen_id):
                return

            delta = chunk.choices[0].delta.content or ""
            buffer += delta

            # Check for punctuation boundaries
            parts = self.CLAUSE_SPLIT_REGEX.split(buffer)
            if len(parts) > 2:
                # Complete clause detected: parts[0] + parts[1]
                clause = (parts[0] + parts[1]).strip()
                buffer = "".join(parts[2:])
                if clause:
                    yield clause

        # Yield remainder tokens
        remaining = buffer.strip()
        if remaining and self.turn_controller.is_active_generation(gen_id):
            yield remaining

    async def _synthesize_clause_to_pump(self, clause: str, gen_id: int) -> None:
        """Pipes synthesized PCM frames into the decoupled AudioPump."""
        async for raw_pcm in self.orchestrator.stream_agent_reply(text=clause, turn_id=gen_id):
            if not self.turn_controller.is_active_generation(gen_id):
                break
            await self.audio_pump.push_frame(gen_id, raw_pcm)

    async def shutdown(self) -> None:
        """Teardown session tasks and stop audio pump."""
        await self.handle_barge_in("[session_shutdown]")
        await self.audio_pump.stop()

        # Cancel remaining supervised tasks
        for task in list(self._session_tasks):
            if not task.done():
                task.cancel()

        if self._session_tasks:
            await asyncio.gather(*self._session_tasks, return_exceptions=True)

        logger.info("[AgentSession] Teardown complete.")


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    session = AgentSession(ctx)
    await session.start()

    # Keep worker alive until disconnect signal
    await session._shutdown_event.wait()
    await session.shutdown()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            ws_url=LIVEKIT_URL,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
        )
    )