"use client";

import dynamic from "next/dynamic";

const VoiceRoom = dynamic(() => import("./components/VoiceRoom"), {
  ssr: false,
  loading: () => (
    <div className="flex items-center gap-2 text-zinc-400 text-sm">
      <span className="w-2 h-2 rounded-full bg-zinc-500 animate-pulse" />
      Loading voice session...
    </div>
  ),
});

export default function Home() {
  return (
    <main className="min-h-screen bg-zinc-950 flex flex-col items-center justify-center p-6 text-white">
      <div className="w-full max-w-xl flex flex-col items-center gap-6">
        <h1 className="text-2xl font-semibold tracking-tight">Voice Agent Playground</h1>
        <VoiceRoom />
      </div>
    </main>
  );
}