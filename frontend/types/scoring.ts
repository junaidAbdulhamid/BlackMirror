/**
 * Phase 6 scoring contracts.
 *
 * The three-number separation from the backend is preserved here deliberately:
 * `raw_value` is the measurement, `normalized_value` is a presentation choice,
 * and `score` is the only rankable quantity. The UI must show the raw value
 * next to any score, because normalization can make a 0.25% difference look
 * like a landslide.
 */

export type TargetType =
  | "whole_cortex"
  | "roi"
  | "network"
  | "hemisphere"
  | "custom_vertex_set";

export type TemporalScopeType =
  | "full_stimulus"
  | "absolute_time_window"
  | "normalized_time_window"
  | "content_event"
  | "content_event_relative_window";

export type ObjectiveDirection = "maximize" | "minimize" | "target";

export type NormalizationStrategy =
  | "none"
  | "min_max_within_experiment"
  | "z_score_within_experiment"
  | "reference_baseline"
  | "robust_percentile"
  | "target_distance";

export interface ObjectiveTarget {
  type: TargetType;
  region_id?: number | null;
  network_id?: number | null;
  hemisphere?: "left" | "right" | null;
  vertex_indices?: number[];
}

export interface TemporalScope {
  type: TemporalScopeType;
  start_seconds?: number | null;
  end_seconds?: number | null;
  start_fraction?: number | null;
  end_fraction?: number | null;
  event_type?: string | null;
  event_selection?: "first" | "last" | "longest";
  offset_start_seconds?: number | null;
  offset_end_seconds?: number | null;
  skip_events_without_samples?: boolean;
}

export interface ObjectiveNormalization {
  strategy: NormalizationStrategy;
  target_value?: number | null;
  target_tolerance?: number | null;
}

export interface NeuralObjective {
  objective_id: string;
  name: string;
  description?: string | null;
  metric: string;
  target: ObjectiveTarget;
  temporal_scope: TemporalScope;
  direction: ObjectiveDirection;
  normalization: ObjectiveNormalization;
  weight: number;
  parameters?: Record<string, string | number | boolean | null>;
}

export interface ResolvedWindow {
  start_seconds: number;
  end_seconds: number;
  sample_indices: number[];
  sample_count: number;
  source: string;
}

export interface TargetResolution {
  type: TargetType;
  label: string;
  vertex_count: number;
  region_id?: number | null;
  network_id?: number | null;
  atlas_name?: string | null;
}

export interface TemporalContribution {
  times: number[];
  contributions: number[];
  peak_index: number;
  peak_time_seconds: number;
  peak_fraction: number | null;
}

export interface ObjectiveEvaluation {
  objective_id: string;
  variant_id: string;
  /** The measurement, in model units. Always shown beside a score. */
  raw_value: number | null;
  normalized_value: number | null;
  score: number | null;
  window: ResolvedWindow;
  target: TargetResolution;
  statistics: Record<string, number>;
  temporal_contribution: TemporalContribution | null;
  valid: boolean;
  warnings: string[];
}

export interface ObjectiveContribution {
  objective_id: string;
  weight: number;
  score: number;
  contribution: number;
  contribution_fraction: number | null;
}

export interface VariantScore {
  variant_id: string;
  total_score: number | null;
  objective_scores: ObjectiveEvaluation[];
  contributions: ObjectiveContribution[];
  baseline_delta: number | null;
  baseline_relative_delta: number | null;
  rank: number | null;
  valid: boolean;
  warnings: string[];
}

export interface ParetoAnalysis {
  objective_ids: string[];
  non_dominated: string[];
  dominated: string[];
  dominated_by: Record<string, string[]>;
  warnings: string[];
}

export interface SensitivityAnalysis {
  swept_objective_id: string;
  points: { weight: number; winner_variant_id: string; margin: number }[];
  flip_weights: number[];
  stable: boolean;
  note: string | null;
}

export interface ExperimentScoreResult {
  schema_version: string;
  experiment_id: string;
  objectives: NeuralObjective[];
  baseline_variant_id: string | null;
  variant_scores: VariantScore[];
  ranking: string[];
  ranking_margin: number | null;
  pareto: ParetoAnalysis | null;
  sensitivity: SensitivityAnalysis[];
  normalization_detail: Record<string, Record<string, number>>;
  /** The share of composite magnitude each objective actually accounted for. */
  effective_weights: Record<string, number>;
  metadata: {
    scoring_version: string;
    created_at: string;
    objective_set_hash: string;
    objective_definition_hash: string;
    input_artifact_sha256: Record<string, string>;
    atlas_name: string | null;
    metric_descriptions: Record<string, Record<string, string>>;
    interpretation_notice: string;
  };
  warnings: string[];
}

export interface IntervalContext {
  start_seconds: number;
  end_seconds: number;
  visual_description: string | null;
  speech_text: string | null;
  audio_description: string | null;
  on_screen_text: string[];
  objects: string[];
  event_types: string[];
}

export interface ObjectiveComparison {
  objective_id: string;
  objective_name: string;
  metric: string;
  direction: ObjectiveDirection;
  target_label: string;
  leader_variant_id: string | null;
  raw_values: Record<string, number | null>;
  scores: Record<string, number | null>;
  absolute_difference: number | null;
  relative_difference: number | null;
  peak_intervals: Record<string, number>;
  paired_effect: {
    status: "available" | "unavailable";
    method: string;
    value: number | null;
    sample_count: number;
    reason: string | null;
    assumptions: string[];
  };
}

export interface ScoreExplanation {
  experiment_id: string;
  objective_set_hash: string;
  leader_variant_id: string | null;
  runner_up_variant_id: string | null;
  total_difference: number | null;
  ranking_is_close: boolean;
  objectives: ObjectiveComparison[];
  context: Record<string, IntervalContext>;
  statements: string[];
  caveats: string[];
}

export type MetricCatalog = Record<string, Record<string, string>>;
