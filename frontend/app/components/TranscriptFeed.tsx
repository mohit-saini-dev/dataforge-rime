"use client";

import { useEffect, useRef, useState } from "react";
import { useDataChannel } from "@livekit/components-react";

interface TranscriptEntry {
  id: string;
  text: string;
  speaker: "agent" | "user";
  final: boolean;
}

// Switched to custom topic to prevent LiveKit SDK internal interception
const TRANSCRIPTION_TOPIC = "transcript";

export function TranscriptFeed() {
  const [entries, setEntries] = useState<TranscriptEntry[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  const { message } = useDataChannel(TRANSCRIPTION_TOPIC);

  useEffect(() => {
    if (!message?.payload) return;
    try {
      const data = JSON.parse(new TextDecoder().decode(message.payload));
      
      const speaker: "agent" | "user" =
        data.speaker === "user" ? "user" : "agent";

      for (const seg of data.segments ?? []) {
        const text = seg.text?.trim();
        if (!text) continue;

        setEntries((prev) => {
          const idx = prev.findIndex((e) => e.id === seg.id);
          if (idx >= 0) {
            const next = [...prev];
            next[idx] = { ...next[idx], text, final: seg.final ?? true };
            return next;
          }
          return [...prev, { id: seg.id, text, speaker, final: seg.final ?? true }];
        });
      }
    } catch {
      // ignore malformed messages
    }
  }, [message]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [entries]);

  return (
    <div className="w-full rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900 overflow-hidden">
      <div className="px-4 py-2 border-b border-zinc-200 dark:border-zinc-800">
        <p className="text-xs font-semibold uppercase tracking-widest text-zinc-400 dark:text-zinc-600">
          Transcript
        </p>
      </div>

      <div
        ref={scrollRef}
        className="flex flex-col gap-2.5 p-4 h-36 overflow-y-auto scroll-smooth"
      >
        {entries.length === 0 ? (
          <p className="m-auto text-sm italic text-zinc-400 dark:text-zinc-600">
            Conversation will appear here…
          </p>
        ) : (
          entries.map((entry) => (
            <div
              key={entry.id}
              className={`flex gap-2 text-sm leading-relaxed transition-opacity ${
                entry.final ? "opacity-100" : "opacity-50"
              }`}
            >
              <span
                className={`shrink-0 text-xs font-bold uppercase tracking-wide mt-0.5 ${
                  entry.speaker === "agent"
                    ? "text-blue-500 dark:text-blue-400"
                    : "text-green-600 dark:text-green-400"
                }`}
              >
                {entry.speaker === "agent" ? "Agent" : "You"}
              </span>
              <span className="text-zinc-700 dark:text-zinc-300">{entry.text}</span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}