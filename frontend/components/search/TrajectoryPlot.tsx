"use client";

/**
 * Steps 67-68. Best-so-far against evaluations, and against compute.
 *
 * This is the plot that answers the only question worth asking about a search:
 * not what it found, but how much it cost to find it. A curve that flattens
 * early means the budget after that point bought nothing.
 *
 * Individual evaluations are drawn as well as the running best, because a
 * best-so-far line alone hides how much of the search was spent far from it.
 */

import { Label } from "@/components/scoring/Primitives";
import { formatDuration, formatFitness } from "@/lib/search";
import type { TrajectoryPoint } from "@/types/search";

const WIDTH = 560;
const HEIGHT = 160;

export function TrajectoryPlot({
  points,
  xAxis = "evaluations",
}: {
  points: TrajectoryPoint[];
  xAxis?: "evaluations" | "compute";
}) {
  if (points.length === 0) {
    return (
      <p className="text-[12px] leading-relaxed text-[var(--muted)]">
        Nothing evaluated yet. The curve appears once the first candidate returns.
      </p>
    );
  }

  const x = (point: TrajectoryPoint) =>
    xAxis === "evaluations" ? point.evaluation_number : point.wall_seconds;
  const xs = points.map(x);
  const bests = points.map((p) => p.best_fitness);
  const each = points.map((p) => p.fitness ?? p.best_fitness);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...bests, ...each);
  const maxY = Math.max(...bests, ...each);
  const spanX = maxX - minX || 1;
  const spanY = maxY - minY || 1;

  const px = (value: number) => ((value - minX) / spanX) * WIDTH;
  const py = (value: number) => HEIGHT - ((value - minY) / spanY) * HEIGHT;

  const bestPath = points
    .map((point, index) => `${index === 0 ? "M" : "L"}${px(x(point))},${py(point.best_fitness)}`)
    .join(" ");

  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between">
        <Label>
          {xAxis === "evaluations" ? "Best so far by evaluation" : "Best so far by compute"}
        </Label>
        <span className="tv-label">
          {formatFitness(minY, 3)} – {formatFitness(maxY, 3)}
        </span>
      </div>

      <div className="overflow-x-auto">
        <svg
          viewBox={`-4 -8 ${WIDTH + 8} ${HEIGHT + 16}`}
          width="100%"
          height={HEIGHT + 16}
          role="img"
          aria-label="Best objective value so far against search cost"
          className="min-w-[420px]"
        >
          <line
            x1={0} y1={HEIGHT} x2={WIDTH} y2={HEIGHT}
            stroke="var(--line)" strokeWidth={1}
          />
          {points.map((point) => (
            <circle
              key={`dot-${point.evaluation_number}`}
              cx={px(x(point))}
              cy={py(point.fitness ?? point.best_fitness)}
              r={2}
              fill="var(--muted)"
            >
              <title>
                {point.candidate_id}: {formatFitness(point.fitness)}
              </title>
            </circle>
          ))}
          <path d={bestPath} fill="none" stroke="var(--foreground)" strokeWidth={1.5} />
          {points
            .filter((point) => point.is_new_best)
            .map((point) => (
              <circle
                key={`best-${point.evaluation_number}`}
                cx={px(x(point))}
                cy={py(point.best_fitness)}
                r={3.5}
                fill="var(--foreground)"
              >
                <title>
                  new best {point.best_candidate_id} at {formatFitness(point.best_fitness)}
                </title>
              </circle>
            ))}
        </svg>
      </div>

      <div className="tv-label flex justify-between">
        <span>
          {xAxis === "evaluations" ? `${minX} evaluations` : formatDuration(minX)}
        </span>
        <span>
          {xAxis === "evaluations"
            ? `${maxX} evaluations`
            : formatDuration(maxX)}
        </span>
      </div>
    </div>
  );
}
