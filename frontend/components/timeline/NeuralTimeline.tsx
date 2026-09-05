"use client";

/**
 * Aggregate predicted response over time.
 *
 * Plots the mean across cortical vertices for each prediction row, giving
 * temporal context that a single 3D frame cannot. Medial-wall vertices are
 * excluded — the model emits values there but they carry no meaning, so
 * including them would bias the average.
 *
 * LANGUAGE MATTERS: peaks here are "large response changes", never "emotional
 * moments". Interpretation belongs to Phase 3, backed by an atlas.
 */

import { useMemo } from "react";
import { formatTimecode } from "@/lib/temporalMapping";

export interface NeuralTimelineProps {
  series: Float32Array;
  timeline: Float32Array;
  currentTime: number;
  duration: number;
  activeIndex: number;
  changeIndices: number[];
  onSeek: (seconds: number) => void;
}

const WIDTH = 1000;
const HEIGHT = 120;
const PADDING = 10;

export function NeuralTimeline({
  series,
  timeline,
  currentTime,
  duration,
  activeIndex,
  changeIndices,
  onSeek,
}: NeuralTimelineProps) {
  const { path, area, points, zeroY } = useMemo(() => {
    if (series.length === 0 || duration <= 0) {
      return { path: "", area: "", points: [] as { x: number; y: number }[], zeroY: HEIGHT / 2 };
    }
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    for (const value of series) {
      if (value < min) min = value;
      if (value > max) max = value;
    }
    // Keep zero visible: the sign of the aggregate response is meaningful.
    min = Math.min(min, 0);
    max = Math.max(max, 0);
    const span = max - min || 1;

    const toY = (value: number) =>
      HEIGHT - PADDING - ((value - min) / span) * (HEIGHT - PADDING * 2);

    const pts = Array.from(series).map((value, index) => ({
      x: ((timeline[index] as number) / duration) * WIDTH,
      y: toY(value),
    }));

    const line = pts
      .map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(2)},${p.y.toFixed(2)}`)
      .join(" ");
    const first = pts[0];
    const last = pts[pts.length - 1];
    const filled =
      first && last
        ? `${line} L${last.x.toFixed(2)},${HEIGHT} L${first.x.toFixed(2)},${HEIGHT} Z`
        : "";

    return { path: line, area: filled, points: pts, zeroY: toY(0) };
  }, [series, timeline, duration]);

  const playheadX = duration > 0 ? (currentTime / duration) * WIDTH : 0;

  return (
    <div>
      <div className="flex items-baseline justify-between">
        <h3 className="text-[11px] font-medium uppercase tracking-[0.14em] text-slate-400">
          Mean Predicted Cortical Response
        </h3>
        <span className="text-[10px] text-slate-500">
          averaged across cortex, medial wall excluded
        </span>
      </div>

      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        preserveAspectRatio="none"
        className="mt-2 h-[120px] w-full cursor-pointer rounded-lg border border-white/10 bg-white/[0.02]"
        onClick={(event) => {
          const rect = event.currentTarget.getBoundingClientRect();
          const ratio = (event.clientX - rect.left) / rect.width;
          onSeek(Math.min(Math.max(ratio, 0), 1) * duration);
        }}
      >
        <line
          x1={0}
          x2={WIDTH}
          y1={zeroY}
          y2={zeroY}
          stroke="rgba(255,255,255,0.14)"
          strokeDasharray="4 4"
        />
        {area && <path d={area} fill="rgba(56,189,248,0.10)" />}
        {path && <path d={path} fill="none" stroke="rgb(56,189,248)" strokeWidth={2} />}

        {changeIndices.map((index) => {
          const point = points[index];
          if (!point) return null;
          return (
            <g key={`change-${index}`}>
              <line
                x1={point.x}
                x2={point.x}
                y1={0}
                y2={HEIGHT}
                stroke="rgba(251,191,36,0.45)"
                strokeWidth={1}
              />
              <circle cx={point.x} cy={point.y} r={3.5} fill="rgb(251,191,36)" />
            </g>
          );
        })}

        {points.map((point, index) => (
          <circle
            key={index}
            cx={point.x}
            cy={point.y}
            r={index === activeIndex ? 4.5 : 2.5}
            fill={index === activeIndex ? "rgb(186,230,253)" : "rgba(56,189,248,0.65)"}
          />
        ))}

        <line
          x1={playheadX}
          x2={playheadX}
          y1={0}
          y2={HEIGHT}
          stroke="white"
          strokeWidth={1.5}
          opacity={0.85}
        />
      </svg>

      {changeIndices.length > 0 && (
        <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-slate-500">
          <span className="text-amber-400/80">● Large response change</span>
          {changeIndices.map((index) => (
            <span key={index} className="font-mono tabular-nums">
              {formatTimecode(timeline[index] as number)}
            </span>
          ))}
          <span className="text-slate-600">
            — descriptive only; no psychological meaning is implied.
          </span>
        </div>
      )}
    </div>
  );
}
