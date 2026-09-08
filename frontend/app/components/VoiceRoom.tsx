"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  LiveKitRoom,
  RoomAudioRenderer,
  DisconnectButton,
  useIsSpeaking,
  useLocalParticipant,
  useVoiceAssistant,
} from "@livekit/components-react";
import "@livekit/components-styles";
import { TurnStateBadge } from "./TurnStateBadge";

export type TurnState = "listening" | "thinking" | "speaking" | "interrupted";

const INTERRUPTION_DISPLAY_MS = 1500;
const BARGE_IN_GRACE_MS = 250;

export function useBargeInTurnState(): TurnState {
  const { state: agentState } = useVoiceAssistant();
  const { localParticipant } = useLocalParticipant();
  const isUserSpeaking = useIsSpeaking(localParticipant);

  const [interrupted, setInterrupted] = useState(false);

  const prevAgentSpeaking = useRef(agentState === "speaking");
  const prevUserSpeaking = useRef(isUserSpeaking);
  const agentStoppedAt = useRef<number | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const triggerInterruption = () => {
    setInterrupted(true);
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      setInterrupted(false);
      timerRef.current = null;
    }, INTERRUPTION_DISPLAY_MS);
  };

  useEffect(() => {
    const agentSpeaking = agentState === "speaking";
    const userSpeaking = isUserSpeaking;

    const wasAgentSpeaking = prevAgentSpeaking.current;
    const wasUserSpeaking = prevUserSpeaking.current;

    const agentStoppedSpeaking = wasAgentSpeaking && !agentSpeaking;
    const userStartedSpeaking = !wasUserSpeaking && userSpeaking;
    const agentStartedSpeaking = !wasAgentSpeaking && agentSpeaking;

    // Edge 1: User starts speaking while agent is active
    if (userStartedSpeaking && agentSpeaking) {
      triggerInterruption();
    }

    // Edge 2: Agent cuts off while user is speaking
    if (agentStoppedSpeaking && userSpeaking) {
      agentStoppedAt.current = performance.now();
      triggerInterruption();
    }

    // Edge 3: User starts speaking within the grace window after agent cut off
    if (userStartedSpeaking && !agentSpeaking) {
      const stoppedAt = agentStoppedAt.current;
      if (stoppedAt !== null && performance.now() - stoppedAt <= BARGE_IN_GRACE_MS) {
        triggerInterruption();
      }
    }

    if (agentStartedSpeaking) {
      agentStoppedAt.current = null;
    }

    // Advance refs unconditionally
    prevAgentSpeaking.current = agentSpeaking;
    prevUserSpeaking.current = userSpeaking;
  }, [agentState, isUserSpeaking]);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  if (interrupted) return "interrupted";
  if (agentState === "speaking") return "speaking";
  if (agentState === "thinking") return "thinking";
  return "listening";
}

function RoomContent() {
  const turnState = useBargeInTurnState();
  const { state: rawAgentState, audioTrack } = useVoiceAssistant();

  useEffect(() => {
    console.log("[LiveKit] Current Raw Agent State:", rawAgentState);
    console.log("[LiveKit] Remote Audio Track Publication:", audioTrack);
  }, [rawAgentState, audioTrack]);

  return (
    <div className="flex flex-col items-center justify-center gap-6 p-6">
      <TurnStateBadge state={turnState} />
      <RoomAudioRenderer />
      <DisconnectButton className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white text-sm rounded transition-colors">
        Leave Session
      </DisconnectButton>
    </div>
  );
}

interface VoiceRoomProps {
  roomName?: string;
  participantName?: string;
}

export default function VoiceRoom({
  roomName = "default-room",
  participantName: propParticipantName,
}: VoiceRoomProps) {
  const [token, setToken] = useState<string>("");
  const [error, setError] = useState<string | null>(null);

  // Stable memoized identity prevents component re-render loops
  const participantName = useMemo(
    () => propParticipantName || `user-${Math.floor(Math.random() * 10000)}`,
    [propParticipantName]
  );

  const serverUrl = process.env.NEXT_PUBLIC_LIVEKIT_URL;

  useEffect(() => {
    let isMounted = true;

    async function fetchToken() {
      try {
        const res = await fetch(
          `/api/token?room=${encodeURIComponent(roomName)}&identity=${encodeURIComponent(participantName)}`
        );
        if (!res.ok) {
          throw new Error(`Failed to fetch room token: ${res.statusText}`);
        }
        const data = await res.json();
        if (isMounted) {
          setToken(data.token);
        }
      } catch (err) {
        if (isMounted) {
          setError(err instanceof Error ? err.message : "Unknown token error");
        }
      }
    }

    fetchToken();

    return () => {
      isMounted = false;
    };
  }, [roomName, participantName]);

  if (error) {
    return (
      <div className="p-4 bg-red-900/40 border border-red-500 rounded-lg text-red-200 text-sm">
        {error}
      </div>
    );
  }

  if (!token || !serverUrl) {
    return (
      <div className="flex items-center gap-2 text-zinc-400 text-sm">
        <span className="w-2 h-2 rounded-full bg-zinc-500 animate-pulse" />
        Connecting to session...
      </div>
    );
  }

  return (
    <LiveKitRoom
      token={token}
      serverUrl={serverUrl}
      connect={true}
      audio={true}
      video={false}
      className="flex flex-col items-center justify-center min-h-[300px] w-full"
      onConnected={() => console.log("[LiveKit] Room connected successfully")}
      onDisconnected={(reason) => console.log("[LiveKit] Room disconnected:", reason)}
      onError={(err) => console.error("[LiveKit] Room error:", err)}
    >
      <RoomContent />
    </LiveKitRoom>
  );
}