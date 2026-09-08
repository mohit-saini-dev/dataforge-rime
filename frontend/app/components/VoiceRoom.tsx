"use client";

import { useEffect, useRef, useState } from "react";
import {
  LiveKitRoom,
  RoomAudioRenderer,
  useLocalParticipant,
  useVoiceAssistant,
  useIsSpeaking,
} from "@livekit/components-react";
import "@livekit/components-styles";
import { TurnStateBadge, TurnState } from "./TurnStateBadge";

const LIVEKIT_URL =
  process.env.NEXT_PUBLIC_LIVEKIT_URL ?? "ws://localhost:7880";

export function VoiceRoom() {
  const [token, setToken] = useState<string | null>(null);
  const [connecting, setConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleConnect() {
    setConnecting(true);
    setError(null);
    try {
      const identity = "user-" + Math.random().toString(36).slice(2, 8);
      const res = await fetch(
        `/api/token?room=voice-room&identity=${identity}`
      );
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error ?? `HTTP ${res.status}`);
      }
      const { token: jwt } = await res.json();
      setToken(jwt);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setConnecting(false);
    }
  }

  if (!token) {
    return (
      <div className="flex flex-col items-center justify-center gap-6">
        <h1 className="text-3xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
          Voice Assistant
        </h1>
        {error && (
          <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
        )}
        <button
          onClick={handleConnect}
          disabled={connecting}
          className="px-6 py-3 rounded-full bg-zinc-900 text-white text-sm font-medium hover:bg-zinc-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors dark:bg-zinc-50 dark:text-zinc-900 dark:hover:bg-zinc-200"
        >
          {connecting ? "Connecting…" : "Start Session"}
        </button>
      </div>
    );
  }

  return (
    <LiveKitRoom
      serverUrl={LIVEKIT_URL}
      token={token}
      audio={true}
      video={false}
      className="flex flex-col flex-1 items-center justify-center w-full"
      onDisconnected={() => setToken(null)}
    >
      <RoomAudioRenderer />
      <RoomContent />
    </LiveKitRoom>
  );
}

function RoomContent() {
  const { state: agentState } = useVoiceAssistant();
  const { localParticipant } = useLocalParticipant();
  const isUserSpeaking = useIsSpeaking(localParticipant);

  // Track when agent transitions out of "speaking" while user is talking → interrupted
  const prevAgentState = useRef(agentState);
  const [interrupted, setInterrupted] = useState(false);

  useEffect(() => {
    if (
      prevAgentState.current === "speaking" &&
      agentState !== "speaking" &&
      isUserSpeaking
    ) {
      setInterrupted(true);
      const timer = setTimeout(() => setInterrupted(false), 1500);
      return () => clearTimeout(timer);
    }
    prevAgentState.current = agentState;
  }, [agentState, isUserSpeaking]);

  const turnState: TurnState = interrupted
    ? "interrupted"
    : agentState === "speaking"
    ? "speaking"
    : agentState === "thinking"
    ? "thinking"
    : "listening";

  return (
    <div className="flex flex-col items-center gap-8">
      <h1 className="text-3xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
        Voice Assistant
      </h1>

      <TurnStateBadge state={turnState} />

      <p className="text-sm text-zinc-500 dark:text-zinc-400">
        {isUserSpeaking ? "You are speaking…" : "Mic active — say something"}
      </p>
    </div>
  );
}
