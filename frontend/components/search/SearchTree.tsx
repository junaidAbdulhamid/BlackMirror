"use client";

/**
 * Steps 64, 66, 69-71. The search as a tree, with why each node survived.
 *
 * Generations run down the page and candidates across, which keeps the shape
 * readable without a layout engine. Every node states its standing: eligible,
 * disqualified by a guardrail, failed, or eliminated by the strategy. A node
 * that is absent from the survivors carries the reason it was dropped, because
 * inferring elimination from absence is how a reader ends up guessing.
 */

import { Label } from "@/components/scoring/Primitives";
import {
  byGeneration,
  candidateStanding,
  eliminationFor,
  formatFitness,
} from "@/lib/search";
import type { CandidateResult, Population } from "@/types/search";

export function SearchTree({
  results,
  populations,
  bestCandidateId,
  onSelect,
  selectedId,
}: {
  results: CandidateResult[];
  populations: Population[];
  bestCandidateId: string | null;
  onSelect: (candidateId: string) => void;
  selectedId: string | null;
}) {
  const generations = byGeneration(results);
  if (generations.size === 0) {
    return (
      <p className="text-[12px] leading-relaxed text-[var(--muted)]">
        No candidates yet.
      </p>
    );
  }
  const survivors = new Set(populations.flatMap((item) => item.survivors));

  return (
    <div className="space-y-5">
      {[...generations.entries()]
        .sort((a, b) => a[0] - b[0])
        .map(([generation, members]) => {
          const population = populations.find((item) => item.generation === generation);
          return (
            <div key={generation} className="space-y-2">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <Label>Generation {generation}</Label>
                <span className="tv-label">
                  {members.length} candidate{members.length === 1 ? "" : "s"}
                  {population?.diversity !== null && population?.diversity !== undefined
                    ? ` · diversity ${population.diversity.toFixed(2)}`
                    : ""}
                  {population?.diversity !== null &&
                  population?.diversity !== undefined &&
                  population.diversity < 0.01
                    ? " · collapsed"
                    : ""}
                </span>
              </div>

              <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                {members.map((member) => {
                  const standing = candidateStanding(member);
                  const elimination = eliminationFor(populations, member.candidate_id);
                  const isBest = member.candidate_id === bestCandidateId;
                  const retained = survivors.has(member.candidate_id);
                  return (
                    <button
                      key={member.candidate_id}
                      onClick={() => onSelect(member.candidate_id)}
                      className={`tv-panel px-3 py-2.5 text-left transition-colors ${
                        selectedId === member.candidate_id
                          ? "border-[var(--muted-strong)]"
                          : ""
                      }`}
                    >
                      <div className="flex items-baseline justify-between gap-2">
                        <span className="font-mono text-[11px] text-[var(--foreground)]">
                          {member.candidate_id}
                          {isBest ? " ★" : ""}
                        </span>
                        <span className="tv-num text-[12px] text-[var(--foreground)]">
                          {formatFitness(member.scalar_fitness)}
                        </span>
                      </div>

                      <div className="tv-label mt-1.5 truncate">
                        {Object.entries(member.genome.values)
                          .map(([name, value]) => `${name}=${value}`)
                          .join(", ")}
                      </div>

                      <div
                        className={`mt-1.5 text-[10px] ${
                          standing.eligible
                            ? "text-[var(--muted)]"
                            : "text-[var(--muted-strong)]"
                        }`}
                      >
                        {standing.label}
                        {standing.eligible && retained ? " · retained" : ""}
                        {standing.eligible && !retained && elimination
                          ? ` · ${elimination.detail}`
                          : ""}
                      </div>

                      {!member.feasibility.feasible ? (
                        <div className="mt-1 text-[10px] text-[var(--muted-strong)]">
                          {member.feasibility.violations
                            .map((item) => item.message)
                            .join("; ")}
                        </div>
                      ) : null}
                    </button>
                  );
                })}
              </div>
            </div>
          );
        })}
    </div>
  );
}
