/**
 * Phase 10 contracts.
 *
 * One distinction runs through all of these and must never be flattened in the
 * interface: a *predicted* score is an estimate from a model trained on past
 * evaluations, and an *actual* score is a measurement from the real pipeline.
 * Every field name here keeps them apart, and the UI labels predictions as
 * estimates wherever they appear.
 */

export type SurrogateState =
  | "uninitialized"
  | "bootstrapping"
  | "training"
  | "ready"
  | "degraded"
  | "disabled";

export type SurrogateModelType =
  | "random_forest"
  | "extra_trees"
  | "gradient_boosting"
  | "gaussian_process";

export type AcquisitionName =
  | "greedy_mean"
  | "upper_confidence_bound"
  | "expected_improvement"
  | "probability_of_improvement";

export interface SurrogateMetrics {
  n: number;
  mae: number | null;
  rmse: number | null;
  r2: number | null;
  spearman: number | null;
  kendall: number | null;
  top_k_recall: number | null;
  top_k: number;
  /** Which folds produced these. "prefix" is the order-aware, honest one. */
  scheme: string;
}

export interface UncertaintyDiagnostics {
  n: number;
  error_correlation: number | null;
  mean_uncertainty: number | null;
  mean_absolute_error: number | null;
  coverage_68: number | null;
  /** "posterior_standard_deviation" and "ensemble_spread" are different objects. */
  kind: string;
}

export interface SurrogateModelMetadata {
  model_type: SurrogateModelType;
  model_version: number;
  training_dataset_hash: string;
  training_dataset_size: number;
  feature_names: string[];
  hyperparameters: Record<string, string | number | boolean | null>;
  training_seed: number;
  supports_uncertainty: boolean;
  uncertainty_kind: string;
  trained_at: string;
}

export interface SurrogateSummary {
  search_id: string;
  training_samples: number;
  target_spread: number | null;
  excluded_records: number;
  state: SurrogateState | null;
  trusted: boolean | null;
  trust_reason: string | null;
  model: SurrogateModelMetadata | null;
  model_versions: number;
}

export interface SurrogateDiagnostics {
  state: SurrogateState;
  trusted: boolean;
  trust_reason: string;
  target_spread: number;
  cross_validation: SurrogateMetrics;
  prefix_validation: SurrogateMetrics;
  uncertainty: UncertaintyDiagnostics;
  model: SurrogateModelMetadata;
}

export interface AcquisitionRound {
  round_number: number;
  /** "bootstrap", "surrogate" or "fallback". */
  mode: string;
  candidate_id: string;
  genome: Record<string, number>;
  predicted_score: number | null;
  predicted_uncertainty: number | null;
  ood_score: number | null;
  acquisition: AcquisitionName | null;
  acquisition_value: number | null;
  incumbent_before: number | null;
  actual_score: number | null;
  prediction_error: number | null;
  pool_size: number;
  surrogate_state: SurrogateState;
  surrogate_trusted: boolean;
  selection_reason: string;
}
