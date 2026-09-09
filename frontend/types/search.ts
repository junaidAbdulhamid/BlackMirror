/**
 * Phase 9 contracts, mirroring `blackmirror.search`.
 *
 * Two distinctions are carried through from the Python side and must not be
 * flattened here. A candidate that is *usable* produced a number; a candidate
 * that is *eligible* is also allowed to win, having satisfied every guardrail.
 * And "best observed" is not "optimal": the search reports the strongest of the
 * variants it happened to evaluate, which is a much smaller claim.
 */

export type SearchStatus =
  | "created"
  | "running"
  | "paused"
  | "completed"
  | "failed"
  | "stopped";

export type StrategyName =
  | "random_search"
  | "grid_search"
  | "local_search"
  | "hill_climbing"
  | "beam_search"
  | "epsilon_greedy"
  | "evolutionary_search";

export type FitnessStatus =
  | "evaluated"
  | "not_measurable"
  | "evaluation_failed"
  | "rejected";

export interface SearchSummary {
  search_id: string;
  experiment_id: string | null;
  status: SearchStatus | null;
  strategy: StrategyName | null;
  stopping_reason: string | null;
  best_candidate_id: string | null;
  metrics: SearchMetrics;
  running: boolean;
  paused: boolean;
}

export interface SearchMetrics {
  best_fitness?: number | null;
  root_fitness?: number | null;
  absolute_improvement?: number | null;
  evaluations?: number;
  evaluations_to_best?: number | null;
  wall_seconds?: number;
  wall_seconds_to_best?: number | null;
  tribe_runs?: number;
  tribe_runs_to_best?: number | null;
  cache_hit_rate?: number | null;
  failure_rate?: number | null;
  /** Whether the lead over the root clears the measured nuisance floor. */
  improvement_is_resolvable?: boolean | null;
  /** Candidates the winner is not measurably ahead of. */
  tied_candidate_ids?: string[];
}

export interface TrajectoryPoint {
  evaluation_number: number;
  generation: number;
  candidate_id: string;
  fitness: number | null;
  best_fitness: number;
  best_candidate_id: string;
  wall_seconds: number;
  tribe_runs: number;
  is_new_best: boolean;
}

export interface BudgetView {
  budget: {
    max_candidates: number | null;
    max_tribe_runs: number | null;
    max_wall_seconds: number | null;
    max_cost: number | null;
    max_generations: number | null;
    cost_per_wall_hour: number;
  };
  ledger: {
    candidates_generated: number;
    candidates_evaluated: number;
    tribe_runs: number;
    cache_hits: number;
    evaluation_failures: number;
    generations_completed: number;
    wall_seconds: number;
  };
}

export interface Elimination {
  candidate_id: string;
  reason: string;
  detail: string;
  rank: number | null;
  of: number | null;
}

export interface Population {
  generation: number;
  candidate_ids: string[];
  survivors: string[];
  eliminated: Elimination[];
  diversity: number | null;
  best_candidate_id: string | null;
  best_fitness: number | null;
}

export interface CandidateResult {
  candidate_id: string;
  status: FitnessStatus;
  scalar_fitness: number | null;
  raw_objectives: Record<string, number | null>;
  candidate_run_id: string | null;
  variant_path: string | null;
  reason: string;
  duration_change_seconds: number;
  compute: { wall_seconds: number; tribe_runs: number; served_from_cache: boolean };
  feasibility: {
    feasible: boolean;
    violations: Array<{ guardrail_id: string; message: string }>;
    unchecked: string[];
  };
  genome: {
    genome_id: string;
    values: Record<string, number>;
    origin: string;
    generation: number;
    parent_genome_ids: string[];
    proposed_by: string;
  };
}

export interface SearchEvent {
  type: string;
  message: string;
  candidate_id: string | null;
  generation: number;
  evaluation_number: number;
  at: string;
}

export interface LineageStep {
  candidate_id: string;
  generation: number;
  origin: string;
  genome: Record<string, number>;
  fitness: number | null;
}

export interface ParetoFront {
  objective_ids: string[];
  non_dominated: string[];
  dominated: string[];
  dominated_by: Record<string, string[]>;
  excluded: string[];
  warnings: string[];
}

export interface SearchReport {
  search_id: string;
  strategy: StrategyName;
  seed: number;
  status: SearchStatus;
  stopping_reason: string | null;
  stopping_detail: string;
  best_observed: {
    candidate_id: string;
    genome: Record<string, number>;
    fitness: number | null;
    raw_objectives: Record<string, number | null>;
    candidate_run_id: string | null;
    variant_path: string | null;
    lineage: string[];
  } | null;
  metrics: SearchMetrics;
  pareto_front: ParetoFront | null;
  populations: Population[];
  lineage: LineageStep[];
  guardrails: string[];
  caveats: string[];
}
