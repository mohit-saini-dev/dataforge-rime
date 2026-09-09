"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  BarVisualizer,
  DisconnectButton,
  LiveKitRoom,
  RoomAudioRenderer,
  TrackToggle,
  useConnectionState,
  useIsSpeaking,
  useLocalParticipant,
  useVoiceAssistant,
} from "@livekit/components-react";
import { ConnectionState, Track } from "livekit-client";
import "@livekit/components-styles";
import { TurnStateBadge, TurnState } from "./TurnStateBadge";
import { TranscriptFeed } from "./TranscriptFeed";

// ---------------------------------------------------------------------------
// Pre-connect: check browser mic permission without triggering a prompt
// ---------------------------------------------------------------------------
async function getMicPermissionState(): Promise<PermissionState | "unsupported"> {
  if (!globalThis.navigator?.permissions) return "unsupported";
  try {
    const status = await navigator.permissions.query({
      name: "microphone" as PermissionName,
    });
    return status.state;
  } catch {
    return "unsupported";
  }
}

// ---------------------------------------------------------------------------
// Root export
// ---------------------------------------------------------------------------
export default function VoiceRoom() {
  const [token, setToken] = useState<string | null>(null);
  const [connecting, setConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const serverUrl = process.env.NEXT_PUBLIC_LIVEKIT_URL;

  async function handleConnect() {
    setConnecting(true);
    setError(null);

    const perm = await getMicPermissionState();
    if (perm === "denied") {
      setError(
        "Microphone access is blocked. Allow it in your browser settings and reload."
      );
      setConnecting(false);
      return;
    }

    try {
      const identity = "user-" + Math.random().toString(36).slice(2, 8);
      const res = await fetch(
        `/api/token?room=voice-room&identity=${identity}`
      );
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error ?? `HTTP ${res.status}`);
      }
      const data = await res.json();
      setToken(data.token);
    } catch (err: any) {
      setError(err?.message ?? "Failed to connect to voice room");
    } finally {
      setConnecting(false);
    }
  }

  // Pre-connect screen
  if (!token) {
    return (
      <div className="flex flex-col items-center justify-center gap-6">
        <h1 className="text-3xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
          Voice Assistant
        </h1>

        {error && <ErrorBanner message={error} />}

        <button
          onClick={handleConnect}
          disabled={connecting}
          className="px-6 py-3 rounded-full bg-zinc-900 text-white text-sm font-medium hover:bg-zinc-700 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
        >
          {connecting ? "Connecting..." : "Start Session"}
        </button>
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
      className="flex flex-col flex-1 items-center justify-center w-full"
      onDisconnected={() => setToken(null)}
      onError={(err) => {
        const msg = err.message.toLowerCase();
        if (
          msg.includes("notallowederror") ||
          msg.includes("permission denied") ||
          msg.includes("permission dismissed")
        ) {
          setError(
            "Microphone permission was denied. Allow access in your browser settings and try again."
          );
          setToken(null);
        }
      }}
    >
      <RoomAudioRenderer />
      <RoomContent onLeave={() => setToken(null)} />
    </LiveKitRoom>
  );
}

// ---------------------------------------------------------------------------
// In-room UI
// ---------------------------------------------------------------------------
function RoomContent({ onLeave }: { onLeave: () => void }) {
  const connectionState = useConnectionState();
  const { state: agentState, audioTrack: agentAudioTrack } = useVoiceAssistant();
  const { localParticipant } = useLocalParticipant();
  const isUserSpeaking = useIsSpeaking(localParticipant);

  // Build a track reference for the local mic BarVisualizer
  const micPub = localParticipant?.getTrackPublication(Track.Source.Microphone);
  const localMicRef = micPub
    ? {
        participant: localParticipant,
        publication: micPub,
        source: Track.Source.Microphone as Track.Source,
      }
    : undefined;

  // Detect interrupted: agent was speaking, user barges in, agent stops
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
    <div className="flex flex-col items-center gap-5 w-full max-w-lg px-6 py-8">
      {/* Connection health banner */}
      {connectionState === ConnectionState.Reconnecting && (
        <div className="w-full px-4 py-2.5 rounded-lg bg-yellow-50 border border-yellow-200 text-yellow-800 text-sm text-center dark:bg-yellow-950 dark:border-yellow-800 dark:text-yellow-200">
          Connection lost — reconnecting...
        </div>
      )}

      <h1 className="text-2xl font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
        Voice Assistant
      </h1>

      {/* Agent audio visualizer */}
      <section className="flex flex-col items-center gap-2 w-full">
        <span className="text-xs font-semibold uppercase tracking-widest text-zinc-400 dark:text-zinc-600">
          Agent
        </span>
        <div className="w-full h-16 flex items-center">
          {agentAudioTrack ? (
            <BarVisualizer
              trackRef={agentAudioTrack}
              barCount={24}
              className="w-full h-full"
              style={{ "--lk-fg": "rgb(59 130 246)" } as React.CSSProperties}
            />
          ) : (
            <FlatBars count={24} className="bg-blue-200 dark:bg-blue-900" />
          )}
        </div>
        <TurnStateBadge state={turnState} />
      </section>

      {/* Rolling transcript */}
      <TranscriptFeed />

      {/* Local mic visualizer */}
      <section className="flex flex-col items-center gap-2 w-full">
        <span className="text-xs font-semibold uppercase tracking-widest text-zinc-400 dark:text-zinc-600">
          You
        </span>
        <div className="w-full h-8 flex items-center">
          {localMicRef ? (
            <BarVisualizer
              trackRef={localMicRef}
              barCount={14}
              className="w-full h-full"
              style={{ "--lk-fg": "rgb(34 197 94)" } as React.CSSProperties}
            />
          ) : (
            <FlatBars count={14} className="bg-green-200 dark:bg-green-900" />
          )}
        </div>
      </section>

      {/* Controls */}
      <div className="flex items-center gap-3 mt-1">
        <TrackToggle
          source={Track.Source.Microphone}
          className="lk-button"
        />
        <DisconnectButton
          onClick={onLeave}
          className="lk-button lk-disconnect-button"
        >
          Leave
        </DisconnectButton>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Static placeholder bars shown before a track is available */
function FlatBars({ count, className }: { count: number; className: string }) {
  return (
    <div className="flex w-full items-end justify-between gap-0.5 h-full px-1">
      {Array.from({ length: count }).map((_, i) => (
        <div
          key={i}
          className={`flex-1 rounded-sm ${className}`}
          style={{ height: "4px" }}
        />
      ))}
    </div>
  );
}

function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-3 px-4 py-3 rounded-lg bg-red-50 border border-red-200 text-red-700 dark:bg-red-950 dark:border-red-800 dark:text-red-200 text-sm max-w-md">
      <span className="shrink-0 mt-0.5 font-bold">!</span>
      <span>{message}</span>
    </div>
  );
}