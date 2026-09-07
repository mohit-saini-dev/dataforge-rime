"use client";

import { LiveKitRoom } from "@livekit/components-react";
import { useCallback, useState } from "react";
import { RoomContent } from "./RoomContent";

type ConnectionDetails = {
  token: string;
  url: string;
  room: string;
};

const DEFAULT_ROOM =
  process.env.NEXT_PUBLIC_DEFAULT_ROOM_NAME ?? "dataforge-rime-dev";

export function VoiceConsole() {
  const [details, setDetails] = useState<ConnectionDetails | null>(null);
  const [roomName, setRoomName] = useState(DEFAULT_ROOM);
  const [status, setStatus] = useState<"idle" | "connecting" | "error">("idle");
  const [error, setError] = useState<string | null>(null);

  const connect = useCallback(async () => {
    setStatus("connecting");
    setError(null);
    try {
      const res = await fetch(`/api/token?room=${encodeURIComponent(roomName)}`);
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data?.error ?? `Token request failed (${res.status})`);
      }
      setDetails(data);
      setStatus("idle");
    } catch (err) {
      setStatus("error");
      setError(err instanceof Error ? err.message : "Failed to connect");
    }
  }, [roomName]);

  const disconnect = useCallback(() => {
    setDetails(null);
    setStatus("idle");
  }, []);

  if (details) {
    return (
      <LiveKitRoom
        serverUrl={details.url}
        token={details.token}
        // Requesting microphone access and publishing it into the room the
        // moment we connect — this is the "connect the mic track" wiring.
        audio={true}
        video={false}
        connect={true}
        onDisconnected={disconnect}
        className="flex w-full flex-1 items-center justify-center"
      >
        <RoomContent roomName={details.room} onLeave={disconnect} />
      </LiveKitRoom>
    );
  }

  return (
    <div className="flex w-full max-w-md flex-col items-center gap-6 rounded-2xl border border-white/10 bg-white/[0.03] px-8 py-10">
      <div className="flex flex-col items-center gap-1 text-center">
        <h1 className="text-lg font-medium text-white/90">Voice console</h1>
        <p className="text-sm text-white/45">Connect your mic to the agent room.</p>
      </div>

      <label className="flex w-full flex-col gap-1.5 text-left">
        <span className="text-[11px] font-mono uppercase tracking-wider text-white/40">
          Room
        </span>
        <input
          value={roomName}
          onChange={(e) => setRoomName(e.target.value)}
          className="rounded-lg border border-white/15 bg-black/30 px-3 py-2 text-sm text-white/90 outline-none focus:border-sky-400/50"
          placeholder="room name"
        />
      </label>

      <button
        type="button"
        onClick={connect}
        disabled={status === "connecting" || roomName.trim().length === 0}
        className="w-full rounded-full bg-white/90 px-4 py-2.5 text-sm font-medium text-black transition-colors hover:bg-white disabled:cursor-not-allowed disabled:opacity-40"
      >
        {status === "connecting" ? "Connecting…" : "Connect microphone"}
      </button>

      {error && <p className="text-sm text-rose-300">{error}</p>}
    </div>
  );
}
