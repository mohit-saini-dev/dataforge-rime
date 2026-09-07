"use client";

import { useLocalParticipant } from "@livekit/components-react";
import { useEffect, useRef, useState } from "react";

const BAR_COUNT = 12;

/**
 * Reads real audio energy off the published microphone track via the Web
 * Audio API. This isn't just decorative — it's a direct visual check that
 * the local mic track is actually attached to the room and carrying audio,
 * not just that publish_track() resolved without throwing.
 */
export function MicLevelMeter() {
  const { microphoneTrack, isMicrophoneEnabled } = useLocalParticipant();
  const [level, setLevel] = useState(0);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const mediaStreamTrack = isMicrophoneEnabled
      ? microphoneTrack?.track?.mediaStreamTrack
      : undefined;
    if (!mediaStreamTrack) return;

    const audioContext = new AudioContext();
    const source = audioContext.createMediaStreamSource(new MediaStream([mediaStreamTrack]));
    const analyser = audioContext.createAnalyser();
    analyser.fftSize = 512;
    analyser.smoothingTimeConstant = 0.6;
    source.connect(analyser);

    const buffer = new Uint8Array(analyser.frequencyBinCount);

    const tick = () => {
      analyser.getByteTimeDomainData(buffer);
      let sumSquares = 0;
      for (let i = 0; i < buffer.length; i++) {
        const normalized = (buffer[i] - 128) / 128;
        sumSquares += normalized * normalized;
      }
      const rms = Math.sqrt(sumSquares / buffer.length);
      setLevel(Math.min(1, rms * 4.5));
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      source.disconnect();
      analyser.disconnect();
      void audioContext.close();
      setLevel(0);
    };
  }, [isMicrophoneEnabled, microphoneTrack]);

  return (
    <div className="flex h-8 items-end gap-[3px]" aria-hidden="true">
      {Array.from({ length: BAR_COUNT }).map((_, i) => {
        const threshold = (i + 1) / BAR_COUNT;
        const active = isMicrophoneEnabled && level >= threshold * 0.92;
        return (
          <span
            key={i}
            className={`w-1 rounded-full transition-colors duration-75 ${
              active ? "bg-emerald-400" : "bg-white/10"
            }`}
            style={{ height: `${8 + i * 2}px` }}
          />
        );
      })}
    </div>
  );
}
