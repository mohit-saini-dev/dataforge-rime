import dynamic from "next/dynamic";

// WebRTC APIs (navigator.mediaDevices, RTCPeerConnection) fail during server pre-rendering.
// Disabling SSR guarantees execution solely in the browser environment.
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
    <main className="flex min-h-screen flex-col items-center justify-center bg-zinc-50 dark:bg-black p-4">
      <VoiceRoom />
    </main>
  );
}