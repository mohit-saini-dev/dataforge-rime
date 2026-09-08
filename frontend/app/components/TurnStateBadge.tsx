"use client";

export type TurnState = "listening" | "speaking" | "thinking" | "interrupted";

const STATE_CONFIG: Record<TurnState, { label: string; dot: string; badge: string }> = {
  listening: {
    label: "Listening",
    dot: "bg-green-500",
    badge: "bg-green-50 text-green-700 border-green-200 dark:bg-green-950 dark:text-green-300 dark:border-green-800",
  },
  speaking: {
    label: "Speaking",
    dot: "bg-blue-500",
    badge: "bg-blue-50 text-blue-700 border-blue-200 dark:bg-blue-950 dark:text-blue-300 dark:border-blue-800",
  },
  thinking: {
    label: "Thinking",
    dot: "bg-yellow-500",
    badge: "bg-yellow-50 text-yellow-700 border-yellow-200 dark:bg-yellow-950 dark:text-yellow-300 dark:border-yellow-800",
  },
  interrupted: {
    label: "Interrupted",
    dot: "bg-red-500",
    badge: "bg-red-50 text-red-700 border-red-200 dark:bg-red-950 dark:text-red-300 dark:border-red-800",
  },
};

export function TurnStateBadge({ state }: { state: TurnState }) {
  const { label, dot, badge } = STATE_CONFIG[state];
  return (
    <div
      className={`inline-flex items-center gap-2.5 px-4 py-2 rounded-full border text-sm font-medium transition-colors duration-200 ${badge}`}
    >
      <span className="relative flex h-2.5 w-2.5 shrink-0">
        <span
          className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-60 ${dot}`}
        />
        <span className={`relative inline-flex h-2.5 w-2.5 rounded-full ${dot}`} />
      </span>
      {label}
    </div>
  );
}
