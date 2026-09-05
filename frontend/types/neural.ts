/**
 * Types mirroring the Phase 2 visualization API contracts.
 *
 * These describe *predicted* cortical responses produced by a model, never
 * measurements from a viewer. Field names deliberately match the Python
 * contracts in `blackmirror/api/contracts.py` one-for-one so a mismatch is
 * obvious rather than silently coerced.
 */

export interface HemisphereRange {
  name: string;
  start: number; // inclusive first prediction column
  end: number; // exclusive last prediction column
}

export interface StimulusSummary {
  filename: string;
  media_type: "video" | "audio" | "text";
  duration_seconds: number | null;
  sha256: string;
  playable: boolean;
}

export interface ModelSummary {
  name: string;
  backend: string;
  model_id: string;
  license: string | null;
  device: string;
  dtype: string;
  subject_conditioning: string | null;
  /** True for the mock backend. Such runs MUST be labelled as synthetic. */
  is_synthetic: boolean;
  feature_extractors: Record<string, string>;
}

export interface TemporalSummary {
  n_time_points: number;
  tr_seconds: number;
  /** False means rows were dropped: never compute time as index * TR. */
  timeline_is_contiguous: boolean;
  segments_were_filtered: boolean;
  n_segments_total: number | null;
  n_segments_kept: number | null;
  first_segment_start_seconds: number | null;
  last_segment_end_seconds: number | null;
  covered_seconds: number | null;
  hemodynamic_offset_seconds: number;
  hemodynamic_offset_verified: boolean;
  /** True: a row's segment start IS the stimulus content time. */
  output_is_stimulus_aligned: boolean;
  notes: string[];
}

export interface CorticalSummary {
  surface_space: string;
  vertex_count: number;
  vertices_per_hemisphere: number;
  hemispheres: HemisphereRange[];
  medial_wall_available: boolean;
  mapping_notes: string | null;
}

export interface ResponseScale {
  min: number;
  max: number;
  mean: number;
  std: number;
  p01: number;
  p50: number;
  p99: number;
  /** max(|p01|, |p99|) — symmetric half-range for a zero-centred scale. */
  abs_max: number;
  negative_fraction: number;
  diverging_recommended: boolean;
}

export interface ArrayDescriptor {
  url: string;
  dtype: "float32" | "int32" | "uint8";
  shape: number[];
  byte_length: number;
  description: string;
}

export interface RunSummary {
  run_id: string;
  created_at: string;
  status: string;
  schema_version: string;
  stimulus: StimulusSummary;
  model: ModelSummary;
  temporal: TemporalSummary;
  cortical: CorticalSummary;
  scale: ResponseScale;
  arrays: Record<string, ArrayDescriptor>;
  interpretation_notice: string;
}

export interface RunListItem {
  run_id: string;
  created_at: string;
  stimulus_filename: string;
  media_type: string;
  backend: string;
  is_synthetic: boolean;
  shape: number[];
  status: string;
}

/** Geometry for one hemisphere, ready for a Three.js BufferGeometry. */
export interface HemisphereGeometry {
  name: "left" | "right";
  /** Flat [x,y,z, x,y,z, ...], length = vertexCount * 3. */
  positions: Float32Array;
  /** Flat triangle indices, hemisphere-local. */
  indices: Uint32Array;
  vertexCount: number;
  /** Where this hemisphere starts on the prediction vertex axis. */
  predictionOffset: number;
}

/** Everything needed to render, held as typed arrays rather than JS objects. */
export interface NeuralDataset {
  summary: RunSummary;
  /** [T * V] float32, row-major. The raw Phase 1 matrix, unmodified. */
  predictions: Float32Array;
  timePoints: number;
  vertexCount: number;
  /** Playback clock per row, seconds. */
  timeline: Float32Array;
  /** How much stimulus time each row covers. Row i spans
   *  [timeline[i], timeline[i] + durations[i]). */
  durations: Float32Array;
  hemispheres: HemisphereGeometry[];
  /** 1 = medial wall (no meaningful fMRI signal). */
  medialWall: Uint8Array | null;
}

export type SurfaceKind = "inflated" | "pial";
export type HemisphereView = "both" | "left" | "right";
export type NormalizationMode = "robust_global" | "global" | "per_frame";

export interface AnalyticsRegion {
  region_id: number;
  name: string;
  hemisphere: "left" | "right";
  vertex_count: number;
}

export interface ROIResponseSummary {
  region: AnalyticsRegion;
  mean_response: number;
  peak_response: number;
  peak_timestep: number;
  peak_time_seconds: number;
  std_response: number;
  min_response: number;
  max_response: number;
}

export interface NeuralEventSummary {
  event_id: string;
  timestep: number;
  timestamp_seconds: number;
  event_type: "response_peak" | "large_transition" | "regional_peak";
  score: number;
  affected_region_ids: number[];
  global_response: number;
  change_magnitude: number;
}

export interface NeuralAnalyticsSummary {
  schema_version: string;
  run_id: string;
  atlas: { name: string; version: string; region_count: number };
  validation: { valid: boolean; mapped_fraction: number; medial_wall_vertices: number };
  roi_responses: ROIResponseSummary[];
  events: NeuralEventSummary[];
  metadata: { analytics_version: string; interpretation_notice: string };
}

export interface RegionDifferenceSummary {
  region_id: number;
  name: string;
  hemisphere: string;
  mean_signed_delta: number;
  mean_absolute_delta: number;
  rms_delta: number;
  peak_absolute_delta: number;
  peak_signed_delta: number;
  peak_time_seconds: number;
}

export interface DivergenceEventSummary {
  rank: number;
  aligned_index: number;
  timestamp_seconds: number;
  /** The ranking quantity: equals cortical_rms_difference. */
  score: number;
  /**
   * Raw L2. Grows with how many vertices were jointly finite in that row, so it
   * is NOT comparable between rows and must not be used to order events.
   */
  cortical_l2_difference: number;
  /** L2 / sqrt(finite_vertex_count) — comparable across rows. */
  cortical_rms_difference: number;
  finite_vertex_count: number;
  cortical_cosine_similarity: number | null;
  top_region_ids: number[];
}

export interface PairwiseComparisonSummary {
  candidate_run_id: string;
  comparability: { comparable: boolean; checks: Record<string, boolean> };
  alignment: {
    method: string;
    tolerance_seconds: number;
    reference_samples: number;
    candidate_samples: number;
    aligned_samples: number;
    reference_coverage_fraction: number;
    candidate_coverage_fraction: number;
    first_time_seconds: number;
    last_time_seconds: number;
    interpolation_applied: boolean;
    interpolated_candidate_samples: number;
    extrapolated_samples: number;
  };
  regions: RegionDifferenceSummary[];
  networks: Array<{ network_id: number; name: string; mean_signed_delta: number; mean_absolute_delta: number; rms_delta: number; peak_absolute_delta: number; peak_signed_delta: number; peak_time_seconds: number }>;
  content_status: string;
  content_analysis_version: string | null;
  content_models: Record<string, string>;
  content_configuration: Record<string, number | string | boolean | null>;
  content_features: Array<{ name: string; jointly_available_fraction: number; mean_signed_delta: number | null; mean_absolute_delta: number | null }>;
  events: DivergenceEventSummary[];
  windows: Array<{
    rank: number;
    start_time_seconds: number;
    end_time_seconds: number;
    observed_samples: number;
    mean_l2_difference: number;
    peak_l2_difference: number;
    /** Windows are ranked on this, not on mean_l2_difference. */
    mean_rms_difference: number;
    peak_rms_difference: number;
    peak_time_seconds: number;
  }>;
  arrays_path: string;
  array_keys: Record<string, string>;
}

export interface NeuralComparisonSummary {
  schema_version: string;
  comparison_id: string;
  metadata: {
    comparison_version: string;
    reference_run_id: string;
    candidate_run_ids: string[];
    interpretation_notice: string;
  };
  pairs: PairwiseComparisonSummary[];
}

// ---------------------------------------------------------------------------
// Phase 4 — content intelligence
// ---------------------------------------------------------------------------

/** How one derived content statement was produced. */
export interface ContentProvenance {
  source: string;
  model_id: string | null;
  confidence: number | null;
  notes: string | null;
}

/** How a shot began. Frame-difference detection alone cannot see a fade. */
export type TransitionKind = "cut" | "fade" | "gradual" | "start" | "unknown";

export interface ShotSegment {
  index: number;
  start_time: number;
  end_time: number;
  duration: number;
  keyframe_paths: string[];
  transition_in: TransitionKind;
  /** Which detectors independently found this boundary. Agreement corroborates. */
  detected_by: string[];
}

export interface FeatureCorrelation {
  content_feature: string;
  neural_metric: string;
  method: string;
  coefficient: number;
  p_value: number | null;
  /** Total tests performed in the run (pairs x offsets). */
  comparisons: number;
  lags_tested: number;
  /** p_value x comparisons, capped at 1. Controls ANY false positive. */
  p_value_bonferroni: number | null;
  /** Benjamini-Hochberg FDR. Controls the PROPORTION of false positives. */
  q_value_bh: number | null;
  lag_seconds: number;
  sample_count: number;
  interpretation: string;
}

export interface SceneSegment {
  index: number;
  start_time: number;
  end_time: number;
  duration: number;
  shot_indices: number[];
  description: string | null;
  keyframe_path: string | null;
  grouping_reason: string | null;
}

export interface WordTiming {
  text: string;
  start_time: number;
  end_time: number;
}

export interface TranscriptSegment {
  index: number;
  start_time: number;
  end_time: number;
  text: string;
  /** Set only when diarization had enough separation to be trusted. */
  speaker: string | null;
  words: WordTiming[];
  provenance: ContentProvenance;
}

export interface TextOverlay {
  text: string;
  start_time: number;
  end_time: number;
  confidence: number | null;
  occurrences: number;
}

/** One interval of the stimulus, described across whatever modalities apply. */
export interface ContentEvent {
  event_id: string;
  event_type: string;
  start_time: number;
  end_time: number;
  modalities: string[];
  visual_description: string | null;
  speech_text: string | null;
  on_screen_text: string[];
  audio_description: string | null;
  semantic_description: string | null;
  objects: string[];
  visual_features: Record<string, number>;
  audio_features: Record<string, number>;
  language_features: Record<string, number | boolean | string>;
  shot_indices: number[];
  scene_index: number | null;
}

/**
 * Content that co-occurred with a neural event.
 *
 * TEMPORAL CO-OCCURRENCE ONLY — `interpretation` carries that statement and
 * must be surfaced wherever this is rendered.
 */
export interface NeuralContentAssociation {
  neural_event_id: string;
  neural_event_type: string;
  neural_time: number;
  neural_score: number;
  window_start: number;
  window_end: number;
  context_window_seconds: number;
  content_event_ids: string[];
  visual_context: string[];
  speech_context: string[];
  on_screen_text_context: string[];
  audio_context: string[];
  semantic_context: string[];
  interpretation: string;
}

/** Structured scene content for one keyframe, with explicit abstention. */
export interface VisualAttributes {
  time: number;
  frame_kind: "photographic" | "graphic" | "blank" | "unknown";
  setting: string | null;
  setting_confidence: number | null;
  setting_margin: number | null;
  /** True when no label was confident enough. An honest absence, not a failure. */
  abstained: boolean;
  abstain_reason: string | null;
  objects: string[];
  shot_scale: string | null;
}

/** When a detected object label was on screen. Grouped by label, not instance. */
export interface ObjectAppearance {
  label: string;
  first_seen: number;
  last_seen: number;
  screen_time_seconds: number;
  detection_count: number;
  mean_confidence: number | null;
}

export interface AudioEvent {
  label: string;
  /** Start of the merged span of windows in which the label stayed above threshold. */
  start_time: number;
  end_time: number;
  /**
   * Midpoint of the highest-scoring window. AST's 10.24 s receptive field is
   * fixed, so this is the finest localization its output supports; start/end
   * keep the real uncertainty visible.
   */
  peak_time: number;
  confidence: number;
}

export interface ContentMetrics {
  duration_seconds: number;
  shot_count: number;
  mean_shot_duration: number | null;
  median_shot_duration: number | null;
  cuts_per_minute: number | null;
  scene_count: number;
  speech_fraction: number | null;
  silence_fraction: number | null;
  music_fraction: number | null;
  word_count: number;
  words_per_minute: number | null;
  text_overlay_count: number;
  cta_count: number;
  first_cta_time: number | null;
  mean_motion: number | null;
  mean_audio_energy: number | null;
  mean_brightness: number | null;
  content_event_count: number;
}

export interface ContentAnalysis {
  schema_version: string;
  run_id: string;
  stimulus_filename: string;
  media: {
    duration_seconds: number;
    width: number | null;
    height: number | null;
    fps: number | null;
  };
  modalities_analysed: string[];
  shots: ShotSegment[];
  scenes: SceneSegment[];
  visual_attributes: VisualAttributes[];
  objects: ObjectAppearance[];
  audio_events: AudioEvent[];
  text_overlays: TextOverlay[];
  transcript: TranscriptSegment[];
  calls_to_action: { text: string; start_time: number; modality: string }[];
  events: ContentEvent[];
  associations: NeuralContentAssociation[];
  metrics: ContentMetrics;
  metadata: {
    analysis_version: string;
    models: Record<string, string>;
    stage_seconds: Record<string, number>;
    warnings: string[];
    interpretation_notice: string;
  };
}
