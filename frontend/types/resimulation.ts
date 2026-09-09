/**
 * Phase 8 contracts, mirroring `blackmirror.resimulation.schemas`.
 *
 * `outcome` and `verdict` are separate on purpose and are kept separate here.
 * A measured outcome says which way a declared number moved. A verdict says
 * whether the hypothesis that predicted that movement held. Collapsing them
 * into one badge would turn "the number went up" into "the idea was right".
 */

/** Ordered stages of the durable state machine. The index is the value. */
export const STAGES = [
  "Bound",
  "Inference",
  "Analytics",
  "Content",
  "Scoring",
  "Comparison",
  "Evaluated",
] as const;

export type StageName = (typeof STAGES)[number];

export type RunStatus = "active" | "stop_requested" | "completed" | "failed" | "stopped";

export type StopReason =
  | "completed"
  | "failed_stage"
  | "cancelled"
  | "timeout"
  | "resource_limit"
  | "max_iterations"
  | "max_attempts"
  | "no_measurable_objectives";

export type Outcome = "improved" | "worsened" | "unchanged" | "inconclusive";

export type HypothesisVerdict = "pass" | "fail" | "inconclusive";

export type ExpectedDirection =
  | "test_for_increase"
  | "test_for_decrease"
  | "test_for_target_proximity";

export interface TransformationBinding {
  adapter_id: string;
  adapter_version: string;
  method: string;
  tool: string | null;
  model: string | null;
  parameters: Record<string, string | number | boolean | null>;
  parameters_sha256: string;
  source_path: string;
  source_sha256: string;
  variant_path: string;
  variant_sha256: string;
}

export interface StageArtifact {
  stage: number;
  artifact_id: string;
  artifact_path: string;
  sha256: string;
  artifact_files: Record<string, string>;
  upstream_sha256: Record<string, string>;
  pipeline_version: string;
  created_at: string;
}

export interface FailedAttempt {
  attempt: number;
  stage: number;
  error_type: string;
  message: string;
  occurred_at: string;
}

export interface ObjectiveDelta {
  objective_id: string;
  direction: "maximize" | "minimize" | "target";
  parent_raw_value: number | null;
  candidate_raw_value: number | null;
  raw_delta: number | null;
  parent_score: number | null;
  candidate_score: number | null;
  score_delta: number | null;
  outcome: Outcome;
}

export interface HypothesisOutcome {
  hypothesis_id: string;
  expected_direction: ExpectedDirection;
  verdict: HypothesisVerdict;
  measured_outcome: Outcome;
  objective_id: string;
  note: string;
}

export interface ResimulationRequest {
  resimulation_id: string;
  experiment_id: string;
  optimization_request_key: string;
  parent_run_id: string;
  proposed_variant: {
    proposed_variant_id: string;
    parent_variant_id: string;
    hypothesis_ids: string[];
    target_objective_id: string;
    expected_direction: ExpectedDirection;
  };
  binding: TransformationBinding;
  iteration_index: number;
  max_iterations: number;
  max_attempts: number;
  outcome_tolerance: number;
}

export interface ResimulationResult {
  schema_version: string;
  resimulation_version: string;
  request: ResimulationRequest;
  cache_key: string;
  status: RunStatus;
  current_stage: number;
  attempt_count: number;
  candidate_run_id: string | null;
  artifacts: StageArtifact[];
  failures: FailedAttempt[];
  objective_deltas: ObjectiveDelta[];
  hypothesis_outcomes: HypothesisOutcome[];
  stop_reason: StopReason | null;
  requested_stop_reason: StopReason | null;
  created_at: string;
  updated_at: string;
  interpretation_notice: string;
}

/**
 * A durable state plus the two facts the state file cannot hold.
 *
 * A worker that was killed leaves `active` behind it, because the process that
 * would have recorded the failure is the one that died. `worker_alive` is read
 * from the run's lock at request time, so it is the only thing distinguishing
 * a pass in progress from one that needs resuming.
 */
export interface ResimulationView {
  state: ResimulationResult;
  worker_alive: boolean;
  worker_error: string | null;
}

export interface CreateResimulationBody {
  resimulation_id: string;
  optimization_request_key: string;
  proposed_variant_id: string;
  variant_media_path: string;
  source_media_path?: string | null;
  adapter_id?: string;
  max_attempts?: number;
  outcome_tolerance?: number;
}
