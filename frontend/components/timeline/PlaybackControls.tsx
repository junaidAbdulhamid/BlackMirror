"use client";

/**
 * Play / pause / restart, prediction stepping, and playback rate.
 *
 * Stepping moves to adjacent prediction rows rather than fixed time offsets,
 * because with a sparse timeline "next second" and "next prediction" are not
 * the same thing, and the latter is what changes the brain.
 */

const BUTTON =
  "inline-flex items-center justify-center rounded-md border border-white/10 text-slate-300 transition-colors hover:border-white/25 hover:text-white disabled:opacity-30 disabled:hover:border-white/10";

export interface PlaybackControlsProps {
  isPlaying: boolean;
  onTogglePlay: () => void;
  onRestart: () => void;
  onStep: (delta: number) => void;
  playbackRate: number;
  onRateChange: (rate: number) => void;
  predictionIndex: number;
  timePoints: number;
}

export function PlaybackControls({
  isPlaying,
  onTogglePlay,
  onRestart,
  onStep,
  playbackRate,
  onRateChange,
  predictionIndex,
  timePoints,
}: PlaybackControlsProps) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <button
        type="button"
        onClick={onTogglePlay}
        aria-label={isPlaying ? "Pause" : "Play"}
        className="inline-flex h-9 items-center gap-2 rounded-md bg-sky-500/90 px-4 text-[12px] font-semibold text-white transition-colors hover:bg-sky-400"
      >
        {isPlaying ? "❚❚ Pause" : "▶ Play"}
      </button>

      <button type="button" onClick={onRestart} className={`${BUTTON} h-9 px-3 text-[12px]`}>
        ↺ Restart
      </button>

      <div className="flex items-center gap-1">
        <button
          type="button"
          onClick={() => onStep(-1)}
          disabled={predictionIndex <= 0}
          aria-label="Previous prediction sample"
          className={`${BUTTON} h-9 w-9 text-sm`}
        >
          ◀
        </button>
        <span className="min-w-[76px] text-center font-mono text-[11px] tabular-nums text-slate-400">
          {predictionIndex + 1} / {timePoints}
        </span>
        <button
          type="button"
          onClick={() => onStep(1)}
          disabled={predictionIndex >= timePoints - 1}
          aria-label="Next prediction sample"
          className={`${BUTTON} h-9 w-9 text-sm`}
        >
          ▶
        </button>
      </div>

      <div className="ml-auto flex rounded-md border border-white/10">
        {[0.5, 1, 1.5, 2].map((rate) => (
          <button
            key={rate}
            type="button"
            onClick={() => onRateChange(rate)}
            className={`px-2.5 py-1.5 text-[11px] font-medium transition-colors first:rounded-l-md last:rounded-r-md ${
              playbackRate === rate
                ? "bg-sky-500/15 text-sky-200"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            {rate}×
          </button>
        ))}
      </div>
    </div>
  );
}
