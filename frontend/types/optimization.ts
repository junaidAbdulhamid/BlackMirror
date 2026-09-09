/**
 * Phase 7 contracts.
 *
 * `expected_direction` has no WILL_INCREASE member, and that omission is
 * load-bearing: Phase 7 produces hypotheses, and a type that could express
 * certainty would eventually be used to express it.
 */

export type ReferenceStrategy = "best_score" | "baseline" | "pareto_front" | "manual";
export type OptimizationStrategy = "minimal_edit" | "balanced" | "exploratory";
export type EditCost = "low" | "medium" | "high";
export type RiskLevel = "low" | "medium" | "high";
export type ApprovalState = "pending" | "approved" | "modified" | "rejected";
export type ExpectedDirection =
  | "test_for_increase"
  | "test_for_decrease"
  | "test_for_target_proximity";

export interface OptimizationConstraints {
  frozen_modalities: ("visual" | "audio" | "speech" | "text")[];
  preserve_intervals: [number, number][];
  max_duration_change_seconds: number | null;
  frozen_event_types: string[];
  notes: string[];
  forbid_new_voiceover: boolean;
}

export interface OptimizationRequest {
  experiment_id: string;
  source_variant_id: string;
  objective_set_hash: string;
  baseline_variant_id?: string | null;
  comparison_variant_ids?: string[];
  reference_strategy: ReferenceStrategy;
  strategy: OptimizationStrategy;
  constraints?: OptimizationConstraints;
  max_recommendations: number;
}

export interface ObjectiveGap {
  objective_id: string;
  direction: string;
  source_value: number | null;
  reference_value: number | null;
  reference_variant_id: string | null;
  gap: number | null;
  relative_gap: number | null;
  /** The formula is shown, because "gap" means three things across directions. */
  formula: string;
  note: string | null;
}

export interface OptimizationInterval {
  interval_id: string;
  kind: "weak" | "strong";
  objective_id: string;
  start_seconds: number;
  end_seconds: number;
  sample_count: number;
  contribution_share: number;
  contribution: number;
  reference_contribution_share: number | null;
  reason: string;
}

export interface Evidence {
  evidence_id: string;
  kind: string;
  description: string;
  feature: string | null;
  source_value: number | null;
  reference_value: number | null;
  delta: number | null;
  interval_start_seconds: number | null;
  interval_end_seconds: number | null;
  reference_variant_ids: string[];
  /**
   * Fraction of references differing in the same direction. **null below two
   * references** — a single reference agreeing with itself is not consistency,
   * and rendering 1.0 there would make an anecdote look like a pattern.
   */
  consistency: number | null;
  /** Always present, so a reader can see what consistency was computed from. */
  reference_count: number;
  sample_count: number | null;
  association: "temporal_comparison" | "measured_value";
  note?: string | null;
}

export interface EditInstruction {
  operation: string;
  event_type?: string | null;
  from_time?: number | null;
  to_time?: number | null;
  feature?: string | null;
  target_value?: number | null;
  original?: string | null;
  replacement?: string | null;
  note?: string | null;
}

export interface ContentIntervention {
  intervention_id: string;
  type: string;
  target_interval_start: number;
  target_interval_end: number;
  description: string;
  rationale: string;
  evidence_ids: string[];
  objective_id: string;
  expected_direction: ExpectedDirection;
  edit_instructions: EditInstruction[];
  edit_cost: EditCost;
  modality: string;
}

export interface EvidenceConfidence {
  value: number;
  reference_consistency: number;
  feature_difference_strength: number;
  objective_gap_strength: number;
  evidence_count: number;
  formula: string;
  /** Says in text that this is not a success probability. Rendered verbatim. */
  caveat: string;
}

export interface OptimizationRecommendation {
  recommendation_id: string;
  title: string;
  objective_id: string;
  source_variant_id: string;
  interventions: ContentIntervention[];
  target_interval_start: number;
  target_interval_end: number;
  rationale: string;
  evidence_ids: string[];
  reference_variant_ids: string[];
  evidence_confidence: EvidenceConfidence;
  expected_direction: ExpectedDirection;
  risk: RiskLevel;
  edit_cost: EditCost;
  priority: number;
  is_bundle: boolean;
  hypothesis_id: string | null;
  warnings: string[];
}

export interface OptimizationHypothesis {
  hypothesis_id: string;
  statement: string;
  objective_id: string;
  recommendation_id: string;
  changed_features: string[];
  interval_start_seconds: number;
  interval_end_seconds: number;
  expected_direction: ExpectedDirection;
  outcome: "untested" | "pass" | "fail" | "inconclusive";
}

export interface OptimizationResult {
  schema_version: string;
  request: OptimizationRequest;
  gaps: ObjectiveGap[];
  weak_intervals: OptimizationInterval[];
  strong_intervals: OptimizationInterval[];
  reference_variant_ids: string[];
  evidence: Evidence[];
  recommendations: OptimizationRecommendation[];
  hypotheses: OptimizationHypothesis[];
  traces: { recommendation_id: string; evidence_ids: string[] }[];
  rejected: { recommendation_id: string; reason: string }[];
  metadata: {
    optimization_version: string;
    created_at: string;
    features_considered: number;
    generator: string;
    interpretation_notice: string;
  };
  warnings: string[];
}

export interface RecommendationReview {
  recommendation_id: string;
  state: ApprovalState;
  reviewer?: string | null;
  reason?: string | null;
  modified_interventions?: ContentIntervention[];
}

export interface ProposedVariantSpec {
  proposed_variant_id: string;
  parent_variant_id: string;
  experiment_id: string;
  hypothesis_ids: string[];
  interventions: ContentIntervention[];
  target_objective_id: string;
  expected_direction: ExpectedDirection;
  edit_instructions: EditInstruction[];
  preserve_intervals: [number, number][];
  note: string;
}
