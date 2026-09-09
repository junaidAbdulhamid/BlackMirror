"use client";

/**
 * Whether each approved hypothesis held, kept separate from what was measured.
 *
 * A verdict and an outcome are two different statements. "The number moved
 * toward the declared direction" is a measurement; "the hypothesis passed" is a
 * judgement about a prediction that was written down beforehand. Showing them
 * on the same row, in that order, is what keeps the second from being read as
 * evidence that the intervention caused the first.
 */

import { Label, Notice } from "@/components/scoring/Primitives";
import type { HypothesisOutcome, HypothesisVerdict } from "@/types/resimulation";

const VERDICT_LABEL: Record<HypothesisVerdict, string> = {
  pass: "Held",
  fail: "Did not hold",
  inconclusive: "Undecided",
};

const DIRECTION_LABEL: Record<HypothesisOutcome["expected_direction"], string> = {
  test_for_increase: "tested for an increase",
  test_for_decrease: "tested for a decrease",
  test_for_target_proximity: "tested for proximity to the target",
};

export function HypothesisVerdicts({ outcomes }: { outcomes: HypothesisOutcome[] }) {
  if (!outcomes.length) {
    return (
      <p className="text-[12px] leading-relaxed text-[var(--muted)]">
        No hypothesis has been judged yet. Verdicts are recorded once the run reaches
        evaluation.
      </p>
    );
  }

  return (
    <div className="space-y-4">
      {outcomes.map((outcome) => (
        <div key={outcome.hypothesis_id} className="space-y-1.5">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <Label>{outcome.hypothesis_id}</Label>
            <span
              className={`tv-label ${
                outcome.verdict === "inconclusive"
                  ? "text-[var(--muted)]"
                  : "text-[var(--muted-strong)]"
              }`}
            >
              {VERDICT_LABEL[outcome.verdict]}
            </span>
          </div>
          <p className="text-[11px] leading-relaxed text-[var(--muted-strong)]">
            On {outcome.objective_id}, {DIRECTION_LABEL[outcome.expected_direction]}. Measured:{" "}
            {outcome.measured_outcome}.
          </p>
        </div>
      ))}
      <Notice>
        A hypothesis that held means the objective it named moved as predicted on model-predicted
        responses. It does not establish that the edit caused the movement, and it is not evidence
        about human viewers.
      </Notice>
    </div>
  );
}
