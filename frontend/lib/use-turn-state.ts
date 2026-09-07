"use client";

import { useLocalParticipant, useRemoteParticipants, useSpeakingParticipants } from "@livekit/components-react";
import { useEffect, useRef, useState } from "react";

export type TurnState = "listening" | "speaking" | "interrupted";

/**
 * How long the "interrupted" badge stays up after a barge-in before falling
 * back to "listening". Mirrors the shape of the debounce window the backend
 * uses for barge-in handling (backend/main.py: BARGE_IN_DEBOUNCE_SEC), just
 * sized for legibility on screen rather than audio timing.
 */
const INTERRUPTED_HOLD_MS = 1500;

/**
 * Turn state is derived entirely from LiveKit's own speaking-activity
 * signals — there's no separate state channel from the agent yet, so this
 * infers the same three states the backend's turn controller cares about:
 *
 *  - "speaking": the agent's remote track is currently active
 *  - "interrupted": the user started talking while the agent was still
 *    speaking (a barge-in) — this is the client-side mirror of
 *    SessionManager.handle_barge_in on the backend
 *  - "listening": the default/idle state — the agent is waiting on the user
 */
export function useTurnState(): TurnState {
  const speakingParticipants = useSpeakingParticipants();
  const { localParticipant } = useLocalParticipant();
  const remoteParticipants = useRemoteParticipants();

  // The agent worker is the (single) remote participant in the room.
  const agent = remoteParticipants[0];

  const agentSpeaking = agent
    ? speakingParticipants.some((p) => p.identity === agent.identity)
    : false;
  const userSpeaking = speakingParticipants.some(
    (p) => p.identity === localParticipant.identity,
  );

  const wasAgentSpeakingRef = useRef(false);
  const [interrupted, setInterrupted] = useState(false);

  // Detect the barge-in edge: user speech starting while the agent's track
  // was already active.
  useEffect(() => {
    if (userSpeaking && wasAgentSpeakingRef.current) {
      setInterrupted(true);
    }
    wasAgentSpeakingRef.current = agentSpeaking;
  }, [agentSpeaking, userSpeaking]);

  // Auto-clear the interrupted badge after a short hold window.
  useEffect(() => {
    if (!interrupted) return;
    const timeout = setTimeout(() => setInterrupted(false), INTERRUPTED_HOLD_MS);
    return () => clearTimeout(timeout);
  }, [interrupted]);

  if (interrupted) return "interrupted";
  if (agentSpeaking) return "speaking";
  return "listening";
}
