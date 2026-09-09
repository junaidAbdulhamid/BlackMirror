# Optimization Recommendation Schema

Every field a recommendation carries, and what it may and may not assert.

## OptimizationRecommendation

| Field | Meaning |
|---|---|
| `recommendation_id` | Stable id, cited by traces and hypotheses |
| `title` | Short imperative description. Passes the causal-language filter |
| `objective_id` | Which Phase 6 objective this addresses |
| `source_variant_id` | The variant being optimized |
| `interventions` | One for an atomic recommendation, several for a bundle |
| `target_interval_start/end` | The weak interval, in the source's own timeline |
| `rationale` | Why it is a candidate, stated as a comparison |
| `evidence_ids` | **Required.** Every id must resolve or the recommendation is rejected |
| `reference_variant_ids` | Which variants it was compared against |
| `evidence_confidence` | Support for *testing*, not probability of success |
| `expected_direction` | `test_for_increase` / `test_for_decrease` / `test_for_target_proximity` |
| `risk` | Experimental/creative modification risk, not safety |
| `edit_cost` | Implementation effort, used only in ranking |
| `priority` | `confidence × information_value / cost_weight` |
| `is_bundle` | True when attribution will be harder |
| `hypothesis_id` | Links to the statement Phase 8 will pass or fail |

## Intervention taxonomy

Eleven types, deliberately few — hundreds of microtypes would leave Phase 8
unable to group results. Each maps to a modality Phase 4 actually measures,
which is also how constraint checking works.

| Type | Modality | Typical evidence |
|---|---|---|
| `TIMING_EDIT` | visual | event placement differs |
| `TEXT_EDIT` | text | `text_present` differs |
| `SPEECH_EDIT` | speech | `speech_present` / `speech_activity` differ |
| `VISUAL_EDIT` | visual | `motion_structural`, `brightness`, `saturation` |
| `AUDIO_EDIT` | audio | `audio_energy` differs |
| `SHOT_EDIT` | visual | `shot_change`, `luminance_shift` |
| `PACE_EDIT` | visual | `motion` differs |
| `CONTENT_ORDER_EDIT` | visual | event ordering differs |
| `CTA_EDIT` | text | `cta_present` differs, or a CTA event is absent |
| `PRODUCT_VISIBILITY_EDIT` | visual | `object_count` differs |
| `SCENE_STRUCTURE_EDIT` | visual | an event type is absent in the source |

## Expected direction

There is **no `WILL_INCREASE` member**, and that omission is load-bearing. Phase
7 cannot know the outcome, and a schema able to express certainty would
eventually be used to express it.

## Evidence

| Field | Meaning |
|---|---|
| `evidence_id` | Cited by recommendations; must resolve |
| `kind` | `content_feature_delta`, `objective_gap`, `contribution_share`, `content_event_presence`, `feature_consistency` |
| `source_value` / `reference_value` / `delta` | The measured numbers |
| `interval_start/end_seconds` | Where it was measured |
| `consistency` | Fraction of references differing in the same direction |
| `sample_count` | How many samples the interval held |
| `association` | `temporal_comparison` or `measured_value` — the epistemic status, stated |

## Constraints

`frozen_modalities`, `preserve_intervals`, `frozen_event_types`,
`forbid_new_voiceover`, `max_duration_change_seconds`, plus free-text `notes`
recorded for the reviewer but not machine-enforced. A violating recommendation
is **rejected**, and the rejection is stored with its reason.

## Hypothesis

| Field | Meaning |
|---|---|
| `hypothesis_id` | `HYP-<recommendation_id>` |
| `statement` | The recommendation restated as something testable |
| `changed_features` | What the candidate would alter |
| `expected_direction` | The direction to test |
| `outcome` | `untested` here, always. Only Phase 8 may set pass/fail/inconclusive |

## ProposedVariantSpec

The Phase 8 input. Built **only** from approved or modified reviews. Carries
`edit_instructions` (machine-readable), `preserve_intervals` from the source's
strong intervals, the constraints in force, and a `note` stating plainly that no
improvement claim is made.
