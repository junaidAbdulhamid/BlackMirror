# Evaluating the Optimization Agent

How Phase 7's output should eventually be judged. Most of these can be measured
now; the one that matters most cannot be measured until Phase 8 exists.

## Measurable today

| Criterion | Definition | Current status |
|---|---|---|
| **Evidence grounding** | Share of recommendations whose cited ids all resolve | 100% — an unresolved id is a hard rejection |
| **Constraint adherence** | Share respecting frozen modalities and preserved intervals | 100% — violations are rejected, and the rejection is stored |
| **Unsupported-claim rate** | Share containing causal or outcome language | 0% — rejected at validation, asserted by test |
| **Duplication rate** | Share collapsing to the same intervention type over the same interval | Deduplicated structurally before ranking |
| **Diversity** | Distinct intervention types in the returned set | Ranking takes one per type before filling |
| **Actionability** | Share carrying a machine-readable edit instruction | 100% — required by validation |
| **Determinism** | Same inputs, same evidence graph and ranking | Asserted by test |
| **Selection pressure** | Features examined per interval | Reported on every run (18–21) |

## Only measurable after Phase 8

**Did the recommended intervention actually improve the chosen objective?**

That requires building the candidate, re-running TRIBE, rescoring against the
same objective set, and comparing. The pipeline is deliberately arranged so this
is answerable:

- each recommendation carries exactly one hypothesis id;
- atomic recommendations change one thing, so a score change is attributable;
- bundles are labelled as not attributable;
- `expected_direction` states what would count as confirmation;
- `outcome` on the hypothesis is `untested` and only Phase 8 may set it.

Derived metrics once outcomes exist:

- **Hit rate** — share of hypotheses whose objective moved in the expected direction.
- **Effect size** — how far it moved, in raw objective units.
- **Edit efficiency** — objective change per unit of edit cost.
- **Confidence calibration** — whether high `evidence_confidence` recommendations
  pass more often. This is the test of whether the confidence formula means
  anything, and it cannot be run until enough hypotheses have outcomes.

## Known evaluation limits

- With one reference variant, `consistency` is always 1.0 and carries no
  information. It becomes meaningful at three or more comparable variants.
- Only two real runs exist, of different lengths and stimuli, so nothing here
  has been evaluated on a genuine A/B of the same content.
- A hit rate computed over a handful of hypotheses is not evidence of a working
  recommender; the confidence calibration in particular needs tens of outcomes
  before it means anything.
