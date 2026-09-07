import type { TurnState } from "@/lib/use-turn-state";

const COPY: Record<TurnState, string> = {
  listening: "Listening",
  speaking: "Speaking",
  interrupted: "Interrupted",
};

const DOT_CLASS: Record<TurnState, string> = {
  listening: "bg-sky-400 shadow-[0_0_12px_2px_rgba(56,189,248,0.55)]",
  speaking: "bg-amber-400 shadow-[0_0_12px_2px_rgba(251,191,36,0.55)]",
  interrupted: "bg-rose-400 shadow-[0_0_12px_2px_rgba(251,113,133,0.55)]",
};

const RING_CLASS: Record<TurnState, string> = {
  listening: "border-sky-400/30 text-sky-300",
  speaking: "border-amber-400/30 text-amber-300",
  interrupted: "border-rose-400/40 text-rose-300",
};

export function StateBadge({ state }: { state: TurnState }) {
  return (
    <div
      className={`inline-flex items-center gap-2.5 rounded-full border px-4 py-1.5 font-mono text-xs tracking-wide transition-colors duration-200 ${RING_CLASS[state]}`}
      role="status"
      aria-live="polite"
    >
      <span className="relative flex h-2 w-2">
        <span
          className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-60 ${DOT_CLASS[state]} ${
            state === "listening" ? "hidden" : ""
          }`}
        />
        <span className={`relative inline-flex h-2 w-2 rounded-full ${DOT_CLASS[state]}`} />
      </span>
      {COPY[state]}
    </div>
  );
}
