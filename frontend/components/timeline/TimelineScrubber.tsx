"use client";

/**
 * Draggable playback position.
 *
 * Writes only `currentTime` to the store. The prediction index, the brain and
 * the video are all derived from it, so there is exactly one way for time to
 * change and nothing can drift.
 *
 * Prediction rows are drawn as ticks. With a sparse timeline (TRIBE routinely
 * keeps a few rows out of a hundred) this makes it visible that the brain
 * updates at specific moments rather than continuously.
 */

import { useCallback, useRef } from "react";
import { formatTimecode } from "@/lib/temporalMapping";

export interface TimelineScrubberProps {
  currentTime: number;
  duration: number;
  timeline: Float32Array;
  activeIndex: number;
  markers?: number[];
  onSeek: (seconds: number) => void;
}

export function TimelineScrubber({
  currentTime,
  duration,
  timeline,
  activeIndex,
  markers = [],
  onSeek,
}: TimelineScrubberProps) {
  const trackRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef(false);

  const seekFromEvent = useCallback(
    (clientX: number) => {
      const track = trackRef.current;
      if (!track || duration <= 0) return;
      const rect = track.getBoundingClientRect();
      const ratio = Math.min(Math.max((clientX - rect.left) / rect.width, 0), 1);
      onSeek(ratio * duration);
    },
    [duration, onSeek],
  );

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div className="select-none">
      <div className="flex items-baseline justify-between text-[11px] tabular-nums text-slate-500">
        <span>{formatTimecode(0)}</span>
        <span className="font-mono text-base text-slate-200">{formatTimecode(currentTime)}</span>
        <span>{formatTimecode(duration)}</span>
      </div>

      <div
        ref={trackRef}
        role="slider"
        tabIndex={0}
        aria-label="Playback position"
        aria-valuemin={0}
        aria-valuemax={duration}
        aria-valuenow={currentTime}
        className="group relative mt-2 h-8 cursor-pointer touch-none"
        onPointerDown={(event) => {
          draggingRef.current = true;
          // Capture keeps drags tracking outside the element, but it throws for
          // a pointer id the element does not own. Seeking must not depend on
          // it succeeding, or a failed capture silently swallows the click.
          try {
            event.currentTarget.setPointerCapture(event.pointerId);
          } catch {
            /* capture is an enhancement, not a requirement */
          }
          seekFromEvent(event.clientX);
        }}
        onPointerMove={(event) => {
          if (draggingRef.current) seekFromEvent(event.clientX);
        }}
        onPointerUp={(event) => {
          draggingRef.current = false;
          try {
            event.currentTarget.releasePointerCapture(event.pointerId);
          } catch {
            /* nothing to release */
          }
        }}
        onPointerCancel={() => {
          draggingRef.current = false;
        }}
        onKeyDown={(event) => {
          if (event.key === "ArrowRight") onSeek(Math.min(currentTime + 0.5, duration));
          if (event.key === "ArrowLeft") onSeek(Math.max(currentTime - 0.5, 0));
        }}
      >
        <div className="absolute inset-x-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-white/10">
          <div
            className="h-full rounded-full bg-sky-400/80"
            style={{ width: `${progress}%` }}
          />
        </div>

        {/* One tick per prediction row: where the brain actually updates. */}
        {duration > 0 &&
          Array.from(timeline).map((seconds, index) => (
            <span
              key={index}
              className={`absolute top-1/2 h-2.5 w-px -translate-y-1/2 ${
                index === activeIndex ? "bg-sky-300" : "bg-white/25"
              }`}
              style={{ left: `${(seconds / duration) * 100}%` }}
            />
          ))}

        {markers.map((seconds) => (
          <span
            key={`marker-${seconds}`}
            title="Large response change"
            className="absolute top-1/2 h-3.5 w-px -translate-y-1/2 bg-amber-400/70"
            style={{ left: `${(seconds / duration) * 100}%` }}
          />
        ))}

        <span
          className="pointer-events-none absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white shadow-lg ring-2 ring-sky-400/40 transition-transform group-hover:scale-110"
          style={{ left: `${progress}%` }}
        />
      </div>
    </div>
  );
}
