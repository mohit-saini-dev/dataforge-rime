"use client";

import {
  RoomAudioRenderer,
  useConnectionState,
  useLocalParticipant,
  useRemoteParticipants,
} from "@livekit/components-react";
import { ConnectionState } from "livekit-client";
import { useTurnState } from "@/lib/use-turn-state";
import { StateBadge } from "./StateBadge";
import { MicLevelMeter } from "./MicLevelMeter";

export function RoomContent({ roomName, onLeave }: { roomName: string; onLeave: () => void }) {
  const connectionState = useConnectionState();
  const { localParticipant, isMicrophoneEnabled } = useLocalParticipant();
  const remoteParticipants = useRemoteParticipants();
  const turnState = useTurnState();

  const agentConnected = remoteParticipants.length > 0;
  const connecting = connectionState === ConnectionState.Connecting;

  return (
    <div className="flex w-full max-w-md flex-col items-center gap-8 rounded-2xl border border-white/10 bg-white/[0.03] px-8 py-10">
      <RoomAudioRenderer />

      <div className="flex w-full items-center justify-between text-xs font-mono text-white/40">
        <span>room · {roomName}</span>
        <span>{localParticipant.identity}</span>
      </div>

      <div className="flex flex-col items-center gap-4">
        <StateBadge state={turnState} />
        <p className="text-center text-sm text-white/50">
          {connecting
            ? "Joining the room…"
            : agentConnected
              ? "Connected to the agent."
              : "Waiting for the agent to join…"}
        </p>
      </div>

      <div className="flex flex-col items-center gap-2">
        <MicLevelMeter />
        <span className="text-[11px] font-mono uppercase tracking-wider text-white/30">
          {isMicrophoneEnabled ? "mic live" : "mic muted"}
        </span>
      </div>

      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => localParticipant.setMicrophoneEnabled(!isMicrophoneEnabled)}
          className="rounded-full border border-white/15 px-4 py-2 text-sm text-white/80 transition-colors hover:bg-white/10"
        >
          {isMicrophoneEnabled ? "Mute" : "Unmute"}
        </button>
        <button
          type="button"
          onClick={onLeave}
          className="rounded-full border border-rose-400/30 px-4 py-2 text-sm text-rose-300 transition-colors hover:bg-rose-400/10"
        >
          Leave
        </button>
      </div>
    </div>
  );
}
