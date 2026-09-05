# DATAFORGE 2026 × Rime

## Interruptible Travel Operations Voice Agent

A voice-native travel operations agent designed to handle user interruptions safely during asynchronous tool execution.

### Core Engineering Idea

**Monotonic Generation Fence**

When a user interrupts the agent:

1. The current generation is immediately invalidated.
2. In-flight LLM, tool, and TTS work is cancelled on a best-effort basis.
3. Every asynchronous result is checked against the current generation before it can modify state or produce speech.
4. Results from superseded generations are rejected.

### Tech Stack

- Python
- LiveKit Agents
- LiveKit Cloud
- Groq Llama
- Deepgram STT
- Silero VAD
- Rime TTS
- Next.js
- React
- WebRTC

### Key Acceptance Test

With an asynchronous tool delayed by 2000 ms, the user interrupts at 500 ms with a conflicting instruction.

The system must:

- Cut off stale outbound audio in under 150 ms P95
- Reject 100% of stale tool results
- Execute the revised instruction
- Preserve authoritative state without corruption