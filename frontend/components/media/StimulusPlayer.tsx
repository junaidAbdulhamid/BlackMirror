"use client";

/**
 * The original stimulus, played beside the brain.
 *
 * The element is intentionally not the source of truth: the store owns time and
 * this follows it. Native controls are hidden so there is exactly one transport
 * and the two views cannot disagree.
 */

import { forwardRef } from "react";
import type { StimulusSummary } from "@/types/neural";

export interface StimulusPlayerProps {
  runId: string;
  stimulus: StimulusSummary;
  src: string;
  onLoadedMetadata?: (durationSeconds: number) => void;
}

export const StimulusPlayer = forwardRef<HTMLVideoElement, StimulusPlayerProps>(
  function StimulusPlayer({ stimulus, src, onLoadedMetadata }, ref) {
    if (!stimulus.playable) {
      return (
        <Placeholder
          title="Stimulus unavailable"
          body={`Phase 1 references media by path and hash rather than copying it, and ${stimulus.filename} is no longer at its recorded location. The neural timeline below still works.`}
        />
      );
    }

    if (stimulus.media_type === "text") {
      return (
        <Placeholder
          title="Text stimulus"
          body={`${stimulus.filename} was synthesised to speech and transcribed to recover word timings. There is no video track to display.`}
        />
      );
    }

    if (stimulus.media_type === "audio") {
      return (
        <div className="flex h-full flex-col items-center justify-center gap-4 rounded-xl border border-white/10 bg-[#0a0c10] p-8">
          <div className="text-[11px] uppercase tracking-[0.14em] text-slate-500">
            Audio stimulus
          </div>
          <div className="font-mono text-sm text-slate-300">{stimulus.filename}</div>
          <video
            ref={ref}
            src={src}
            className="w-full max-w-md"
            onLoadedMetadata={(e) => onLoadedMetadata?.(e.currentTarget.duration)}
          />
        </div>
      );
    }

    return (
      <div className="relative h-full overflow-hidden rounded-xl border border-white/10 bg-black">
        <video
          ref={ref}
          src={src}
          playsInline
          preload="auto"
          className="h-full w-full object-contain"
          onLoadedMetadata={(e) => onLoadedMetadata?.(e.currentTarget.duration)}
        />
      </div>
    );
  },
);

function Placeholder({ title, body }: { title: string; body: string }) {
  return (
    <div className="flex h-full min-h-[240px] items-center justify-center rounded-xl border border-white/10 bg-white/[0.02] p-8">
      <div className="max-w-sm text-center">
        <div className="text-[11px] font-medium uppercase tracking-[0.14em] text-slate-400">
          {title}
        </div>
        <p className="mt-2 text-[12px] leading-relaxed text-slate-500">{body}</p>
      </div>
    </div>
  );
}
