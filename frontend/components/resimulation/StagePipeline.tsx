"use client";

/**
 * The durable state machine, drawn as what it is.
 *
 * Every completed stage has an artifact on disk with a checksum, so a filled
 * segment is a claim about persisted work, not an animation. A stage that
 * failed is marked at the stage it failed in rather than being left blank,
 * because "stopped here" and "never reached" are different facts.
 */

import { Label } from "@/components/scoring/Primitives";
import { stageLabel } from "@/lib/resimulation";
import { STAGES } from "@/types/resimulation";
import type { ResimulationResult } from "@/types/resimulation";

export function StagePipeline({ state }: { state: ResimulationResult }) {
  const failedStage = state.status === "failed" ? state.failures.at(-1)?.stage : undefined;

  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between">
        <Label>Pipeline</Label>
        <span className="tv-label">
          {stageLabel(state.current_stage)} · attempt {state.attempt_count}/
          {state.request.max_attempts}
        </span>
      </div>

      <div className="flex gap-px">
        {STAGES.map((name, index) => {
          const reached = index <= state.current_stage;
          const failed = failedStage === index;
          return (
            <div
              key={name}
              className="flex-1 border-y border-[var(--line)] first:border-l last:border-r"
              title={
                failed
                  ? `Failed in ${name}`
                  : reached
                    ? `${name} completed and checksummed`
                    : `${name} not reached`
              }
            >
              <div
                className={
                  failed
                    ? "h-1 bg-[var(--muted-strong)]"
                    : reached
                      ? "h-1 bg-[var(--foreground)]"
                      : "h-1 bg-[var(--line)]"
                }
              />
              <div
                className={`px-1 py-1.5 text-center text-[9px] uppercase tracking-[0.1em] ${
                  reached ? "text-[var(--muted-strong)]" : "text-[var(--line-strong)]"
                }`}
              >
                {name}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
