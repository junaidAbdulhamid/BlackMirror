"""Steps 77-88: the surrogate, the acquisitions, and the optimization loop.

Validated on synthetic functions with known optima, because that is the only
setting where "the surrogate learned something" is falsifiable. On the real
pipeline the optimum is unknown and the label spread is a quarter of the noise
floor, so a real-data success would be unverifiable and a real-data failure
uninformative about the code.

The tests that matter most are the ones asserting refusal: a surrogate that
cannot rank must not be allowed to spend an hour-long evaluation, and an
acquisition that needs a predictive distribution must not silently compute one
from a quantity that is not one.
"""

from __future__ import annotations

import math
import warnings
from random import Random

import numpy as np
import pytest

from blackmirror.materialization.schemas import EditOperation as Op
from blackmirror.scoring.schemas import (
    NeuralObjective,
    ObjectiveDirection,
    ObjectiveNormalization,
    ObjectiveTarget,
    TargetType,
    TemporalScope,
    TemporalScopeType,
)
from blackmirror.search.budget import BudgetLedger, SearchBudget
from blackmirror.search.fitness import CandidateFitness, ComputeCost, FitnessStatus
from blackmirror.search.genome import CandidateGenome
from blackmirror.search.space import ContentSearchSpace, categorical, continuous
from blackmirror.surrogate.acquisition import (
    AcquisitionError,
    AcquisitionName,
    AcquisitionRegistry,
    expected_improvement,
    greedy_mean,
    probability_of_improvement,
    score_candidates,
    select_diverse_batch,
    upper_confidence_bound,
)
from blackmirror.surrogate.encoder import EncodingError, GenomeFeatureEncoder
from blackmirror.surrogate.models import (
    ForestSurrogate,
    GaussianProcessSurrogate,
    SurrogateError,
    default_registry,
)
from blackmirror.surrogate.orchestrator import (
    BayesianOptimizationConfig,
    BayesianOptimizationOrchestrator,
)
from blackmirror.surrogate.pool import CandidatePoolGenerator, PoolConfig
from blackmirror.surrogate.schemas import (
    DatasetIdentity,
    SurrogateDataset,
    SurrogateModelType,
    SurrogatePrediction,
    SurrogateState,
    SurrogateTrainingRecord,
)
from blackmirror.surrogate.trainer import SurrogateTrainer, TrustPolicy, compare_families
from blackmirror.surrogate.validation import (
    cross_validate,
    diagnose_uncertainty,
    ood_scores,
    prefix_validate,
    spearman,
    top_k_recall,
)

# ---------------------------------------------------------------------------
# Fixtures: a two-parameter space and a known objective surface
# ---------------------------------------------------------------------------


def _space() -> ContentSearchSpace:
    return ContentSearchSpace(
        parameters=(
            continuous("gain", Op.AUDIO_GAIN_DB, -10.0, 10.0, resolution=0.1),
            continuous("brightness", Op.BRIGHTNESS, -1.0, 1.0, resolution=0.01),
        )
    )


def _surface(values: dict[str, float]) -> float:
    """A single smooth peak at (3, 0.4); the optimum is 0.0."""
    return -((values["gain"] - 3.0) ** 2) - 10.0 * (values["brightness"] - 0.4) ** 2


def _identity(**overrides: object) -> DatasetIdentity:
    fields: dict[str, object] = {
        "root_media_sha256": "a" * 64,
        "objective_definition_hash": "b" * 64,
        "search_space_version": "1.0",
        "materialization_version": "1.0",
        "scoring_version": "1.0",
        "model_fingerprint": "tribe_v2|test",
    }
    fields.update(overrides)
    return DatasetIdentity(**fields)  # type: ignore[arg-type]


def _objective(direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE) -> NeuralObjective:
    normalization = (
        ObjectiveNormalization(target_value=0.5, target_tolerance=0.1)
        if direction is ObjectiveDirection.TARGET
        else ObjectiveNormalization()
    )
    return NeuralObjective(
        objective_id="o1", name="o1", metric="MEAN_RESPONSE",
        target=ObjectiveTarget(type=TargetType.WHOLE_CORTEX),
        temporal_scope=TemporalScope(type=TemporalScopeType.FULL_STIMULUS),
        direction=direction, normalization=normalization,
    )


def _objective_named(
    objective_id: str, direction: ObjectiveDirection
) -> NeuralObjective:
    return NeuralObjective(
        objective_id=objective_id, name=objective_id, metric="MEAN_RESPONSE",
        target=ObjectiveTarget(type=TargetType.WHOLE_CORTEX),
        temporal_scope=TemporalScope(type=TemporalScopeType.FULL_STIMULUS),
        direction=direction,
    )


def _dataset(n: int, *, seed: int = 0, surface=_surface, flat: bool = False) -> SurrogateDataset:
    rng = Random(seed)
    space = _space()
    dataset = SurrogateDataset(identity=_identity())
    for index in range(n):
        values = space.sample(rng)
        target = 0.5 if flat else surface(values)
        dataset = dataset.with_record(
            SurrogateTrainingRecord(
                candidate_id=f"c{index:03d}",
                genome=values,
                objective_values={"o1": target},
                directional_target=target,
                primary_objective_id="o1",
                sequence=index,
            )
        )
    return dataset


# ---------------------------------------------------------------------------
# Step 81 - encoder
# ---------------------------------------------------------------------------


class TestEncoder:
    def test_encoding_is_deterministic_and_ordered(self) -> None:
        """Column order is the thing that silently corrupts a model."""
        encoder = GenomeFeatureEncoder.from_space(_space())
        genome = {"brightness": 0.4, "gain": 3.0}

        first = encoder.encode(genome)
        second = encoder.encode({"gain": 3.0, "brightness": 0.4})

        assert encoder.feature_names == ("brightness", "gain")
        assert np.array_equal(first, second)

    def test_domain_scaling_maps_the_range_onto_the_unit_interval(self) -> None:
        encoder = GenomeFeatureEncoder.from_space(_space())

        low = encoder.encode({"gain": -10.0, "brightness": -1.0})
        high = encoder.encode({"gain": 10.0, "brightness": 1.0})

        assert np.allclose(low, 0.0)
        assert np.allclose(high, 1.0)

    def test_categorical_becomes_one_hot(self) -> None:
        space = ContentSearchSpace(
            parameters=(categorical("speed", Op.SPEED, (0.9, 1.0, 1.1)),)
        )
        encoder = GenomeFeatureEncoder.from_space(space)

        row = encoder.encode({"speed": 1.0})

        assert encoder.feature_names == ("speed=0.9", "speed=1", "speed=1.1")
        assert list(row) == [0.0, 1.0, 0.0]

    def test_a_missing_parameter_raises_rather_than_defaulting(self) -> None:
        """A zero-filled column is a confident, meaningless prediction."""
        with pytest.raises(EncodingError, match="missing values"):
            GenomeFeatureEncoder.from_space(_space()).encode({"gain": 1.0})

    def test_an_unknown_parameter_says_the_space_probably_changed(self) -> None:
        with pytest.raises(EncodingError, match="search space has probably changed"):
            GenomeFeatureEncoder.from_space(_space()).encode(
                {"gain": 1.0, "brightness": 0.0, "contrast": 1.2}
            )

    def test_the_schema_round_trips_through_json(self) -> None:
        encoder = GenomeFeatureEncoder.from_space(_space())

        restored = GenomeFeatureEncoder.from_json(encoder.to_json())

        assert restored.schema_fingerprint() == encoder.schema_fingerprint()
        assert np.array_equal(
            restored.encode({"gain": 1.0, "brightness": 0.0}),
            encoder.encode({"gain": 1.0, "brightness": 0.0}),
        )

    def test_standard_scaling_needs_more_than_one_genome(self) -> None:
        with pytest.raises(EncodingError, match="at least two genomes"):
            GenomeFeatureEncoder.from_space(_space()).fitted_on([{"gain": 1.0, "brightness": 0.0}])


# ---------------------------------------------------------------------------
# Steps 77-78 - surrogates learn, and uncertainty behaves
# ---------------------------------------------------------------------------


class TestSurrogates:
    def test_a_gaussian_process_learns_a_known_surface(self) -> None:
        """Step 77. If this fails, nothing downstream can be trusted."""
        dataset = _dataset(30, seed=1)
        encoder = GenomeFeatureEncoder.from_space(_space())
        features = encoder.encode_dataset([r.genome for r in dataset.records])

        model = GaussianProcessSurrogate()
        model.fit(features, dataset.targets())
        predicted = model.predict(features)

        assert spearman(dataset.targets(), predicted) > 0.9

    def test_a_forest_learns_a_known_surface(self) -> None:
        dataset = _dataset(40, seed=2)
        encoder = GenomeFeatureEncoder.from_space(_space())
        features = encoder.encode_dataset([r.genome for r in dataset.records])

        model = ForestSurrogate()
        model.fit(features, dataset.targets())

        assert spearman(dataset.targets(), model.predict(features)) > 0.8

    def test_gp_uncertainty_is_low_near_data_and_high_far_from_it(self) -> None:
        """Step 78. The property every uncertainty-aware acquisition relies on."""
        dataset = _dataset(20, seed=3)
        encoder = GenomeFeatureEncoder.from_space(_space())
        genomes = [r.genome for r in dataset.records]
        features = encoder.encode_dataset(genomes)

        model = GaussianProcessSurrogate()
        model.fit(features, dataset.targets())

        _, near = model.predict_with_uncertainty(features)
        far = encoder.encode_dataset(
            [{"gain": 9.9, "brightness": -0.99}, {"gain": -9.9, "brightness": 0.99}]
        )
        _, distant = model.predict_with_uncertainty(far)

        assert near is not None and distant is not None
        assert float(near.mean()) < float(distant.mean())

    def test_a_surrogate_refuses_to_fit_a_single_point(self) -> None:
        model = GaussianProcessSurrogate()

        with pytest.raises(SurrogateError, match="at least two observations"):
            model.fit(np.zeros((1, 2)), np.zeros(1))

    def test_predicting_with_the_wrong_width_is_refused(self) -> None:
        """A stale encoder against a fitted model is a silent corruption."""
        dataset = _dataset(10, seed=4)
        encoder = GenomeFeatureEncoder.from_space(_space())
        model = GaussianProcessSurrogate()
        model.fit(encoder.encode_dataset([r.genome for r in dataset.records]), dataset.targets())

        with pytest.raises(SurrogateError, match="encoder and the model disagree"):
            model.predict(np.zeros((3, 5)))

    def test_the_registry_builds_every_family_it_advertises(self) -> None:
        registry = default_registry()

        for family in registry.available():
            assert registry.create(family).model_type is family

    def test_a_forest_reports_ensemble_spread_not_a_posterior(self) -> None:
        """The label matters: the two are read very differently."""
        assert ForestSurrogate().uncertainty_kind == "ensemble_spread"
        assert GaussianProcessSurrogate().uncertainty_kind == "posterior_standard_deviation"


# ---------------------------------------------------------------------------
# Steps 79-80 - acquisition maths
# ---------------------------------------------------------------------------


class TestAcquisitions:
    def test_ucb_matches_the_worked_example(self) -> None:
        """Step 80: mu 0.5, sigma 0.1, kappa 2 gives exactly 0.7."""
        value = upper_confidence_bound(np.array([0.5]), np.array([0.1]), kappa=2.0)

        assert value[0] == pytest.approx(0.7)

    def test_ei_at_the_incumbent_equals_sigma_times_the_normal_density(self) -> None:
        """Step 79, checked against the closed form rather than a golden value."""
        value = expected_improvement(np.array([0.5]), np.array([0.1]), incumbent=0.5)

        assert value[0] == pytest.approx(0.1 / math.sqrt(2 * math.pi))

    def test_ei_is_zero_for_a_hopeless_candidate_with_no_uncertainty(self) -> None:
        value = expected_improvement(np.array([0.1]), np.array([0.0]), incumbent=0.5)

        assert value[0] == 0.0

    def test_ei_rewards_uncertainty_at_equal_predicted_score(self) -> None:
        """The whole point: an uncertain candidate is worth learning about."""
        values = expected_improvement(
            np.array([0.4, 0.4]), np.array([0.01, 0.20]), incumbent=0.5
        )

        assert values[1] > values[0]

    def test_pi_at_the_incumbent_is_one_half(self) -> None:
        value = probability_of_improvement(np.array([0.5]), np.array([0.1]), incumbent=0.5)

        assert value[0] == pytest.approx(0.5)

    def test_greedy_ignores_uncertainty_entirely(self) -> None:
        values = greedy_mean(np.array([0.4, 0.5]))

        assert list(values) == [0.4, 0.5]

    def test_an_uncertainty_aware_acquisition_refuses_to_guess(self) -> None:
        """Silently substituting zero would make EI a broken greedy."""
        with pytest.raises(AcquisitionError, match="needs a predictive uncertainty"):
            expected_improvement(np.array([0.5]), None, incumbent=0.4)

    def test_ei_needs_an_incumbent(self) -> None:
        with pytest.raises(AcquisitionError, match="no incumbent"):
            expected_improvement(np.array([0.5]), np.array([0.1]), incumbent=None)

    def test_the_registry_knows_which_acquisitions_assume_a_gaussian(self) -> None:
        registry = AcquisitionRegistry()

        assert registry.assumes_gaussian(AcquisitionName.EXPECTED_IMPROVEMENT) is True
        assert registry.assumes_gaussian(AcquisitionName.UPPER_CONFIDENCE_BOUND) is False
        assert registry.requires_uncertainty(AcquisitionName.GREEDY_MEAN) is False


# ---------------------------------------------------------------------------
# Step 86 - diverse batching
# ---------------------------------------------------------------------------


def _prediction(cid: str, score: float, genome: dict[str, float]) -> SurrogatePrediction:
    return SurrogatePrediction(
        candidate_id=cid, genome=genome, predicted_score=score, uncertainty=0.1,
        model_type=SurrogateModelType.GAUSSIAN_PROCESS, model_version=1,
        training_dataset_size=10, training_dataset_hash="h",
    )


class TestBatching:
    def test_a_batch_is_spread_out_rather_than_three_of_the_same_point(self) -> None:
        """Step 86. Near-identical points are the costliest way to learn nothing."""
        space = _space()
        genomes = {
            "a": {"gain": 3.0, "brightness": 0.4},
            "b": {"gain": 3.01, "brightness": 0.4},
            "c": {"gain": -8.0, "brightness": -0.8},
        }
        scores = score_candidates(
            [
                _prediction("a", 0.90, genomes["a"]),
                _prediction("b", 0.89, genomes["b"]),
                _prediction("c", 0.50, genomes["c"]),
            ],
            acquisition=AcquisitionName.GREEDY_MEAN,
            incumbent=0.0,
        )

        from blackmirror.search.genome import genome_distance

        batch = select_diverse_batch(
            scores,
            genomes,
            lambda left, right: genome_distance(
                CandidateGenome(genome_id="l", values=left),
                CandidateGenome(genome_id="r", values=right),
                space,
            ),
            count=2,
            minimum_distance=0.1,
        )

        assert [item.candidate_id for item in batch] == ["a", "c"]

    def test_the_constraint_is_relaxed_rather_than_returning_a_short_batch(self) -> None:
        space = _space()
        genomes = {"a": {"gain": 3.0, "brightness": 0.4}, "b": {"gain": 3.01, "brightness": 0.4}}
        scores = score_candidates(
            [_prediction("a", 0.9, genomes["a"]), _prediction("b", 0.8, genomes["b"])],
            acquisition=AcquisitionName.GREEDY_MEAN,
            incumbent=0.0,
        )

        from blackmirror.search.genome import genome_distance

        batch = select_diverse_batch(
            scores, genomes,
            lambda left, right: genome_distance(
                CandidateGenome(genome_id="l", values=left),
                CandidateGenome(genome_id="r", values=right),
                space,
            ),
            count=2, minimum_distance=0.9,
        )

        assert len(batch) == 2
        assert "relaxed" in batch[1].reason


# ---------------------------------------------------------------------------
# Steps 13-18 - validation and diagnostics
# ---------------------------------------------------------------------------


class TestValidation:
    def _features(self, dataset: SurrogateDataset) -> np.ndarray:
        encoder = GenomeFeatureEncoder.from_space(_space())
        return encoder.encode_dataset([r.genome for r in dataset.records])

    def test_cross_validation_detects_a_learnable_surface(self) -> None:
        dataset = _dataset(30, seed=5)

        metrics = cross_validate(
            GaussianProcessSurrogate, self._features(dataset), dataset.targets()
        )

        assert metrics.spearman is not None and metrics.spearman > 0.7
        assert metrics.mae is not None

    def test_order_aware_validation_predicts_each_point_from_earlier_ones(self) -> None:
        """Step 14. The honest number: random folds leak from the future."""
        dataset = _dataset(25, seed=6)

        metrics = prefix_validate(
            GaussianProcessSurrogate, self._features(dataset), dataset.targets()
        )

        assert metrics.scheme == "prefix"
        assert metrics.n == 22
        assert metrics.spearman is not None

    def test_validation_reports_nothing_rather_than_a_number_from_two_points(self) -> None:
        dataset = _dataset(2, seed=7)

        metrics = cross_validate(
            GaussianProcessSurrogate, self._features(dataset), dataset.targets()
        )

        assert metrics.scheme == "insufficient_data"
        assert metrics.is_informative is False

    def test_rank_correlation_is_none_when_labels_do_not_vary(self) -> None:
        """A flat objective cannot be ranked, and pretending otherwise misleads."""
        assert spearman(np.array([1.0, 1.0, 1.0]), np.array([0.1, 0.2, 0.3])) is None

    def test_top_k_recall_counts_overlap_with_the_true_best(self) -> None:
        actual = np.array([0.1, 0.9, 0.8, 0.2])
        perfect = np.array([0.1, 0.9, 0.8, 0.2])
        inverted = np.array([0.9, 0.1, 0.2, 0.8])

        assert top_k_recall(actual, perfect, 2) == 1.0
        assert top_k_recall(actual, inverted, 2) == 0.0

    def test_uncertainty_diagnostics_detect_a_useful_uncertainty(self) -> None:
        actual = np.array([0.0, 0.0, 0.0, 0.0])
        predicted = np.array([0.1, 0.2, 0.3, 0.4])
        uncertainty = np.array([0.1, 0.2, 0.3, 0.4])

        diagnostics = diagnose_uncertainty(actual, predicted, uncertainty, kind="posterior")

        assert diagnostics.is_useful is True
        assert diagnostics.error_correlation == pytest.approx(1.0)

    def test_ood_distance_grows_with_distance_from_training(self) -> None:
        """Step 18. A property of the data, which model uncertainty can miss."""
        training = np.array([[0.0, 0.0], [0.1, 0.1]])
        near = np.array([[0.05, 0.05]])
        far = np.array([[1.0, 1.0]])

        assert ood_scores(near, training)[0] < ood_scores(far, training)[0]


# ---------------------------------------------------------------------------
# Steps 31, 50, 83-85 - dataset identity and the trust gate
# ---------------------------------------------------------------------------


class TestDatasetAndTrust:
    def test_the_dataset_hash_ignores_order_and_notices_a_changed_label(self) -> None:
        """Step 83."""
        first = _dataset(5, seed=8)
        reordered = SurrogateDataset(
            identity=first.identity, records=tuple(reversed(first.records))
        )
        changed = SurrogateDataset(
            identity=first.identity,
            records=(
                first.records[0].model_copy(update={"directional_target": 99.0}),
                *first.records[1:],
            ),
        )

        assert first.dataset_hash() == reordered.dataset_hash()
        assert first.dataset_hash() != changed.dataset_hash()

    def test_a_changed_objective_makes_the_identity_incompatible(self) -> None:
        """Step 85. Old labels are not comparable under a new objective."""
        original = _identity()
        changed = _identity(objective_definition_hash="c" * 64)

        assert original.is_compatible_with(changed) is False
        assert any("objective" in item for item in original.differences(changed))

    def test_a_changed_search_space_makes_the_identity_incompatible(self) -> None:
        """Step 69."""
        assert _identity().is_compatible_with(_identity(search_space_version="2.0")) is False

    def test_the_gate_refuses_a_surrogate_below_the_sample_minimum(self) -> None:
        """Step 31."""
        trainer = SurrogateTrainer(
            GenomeFeatureEncoder.from_space(_space()), policy=TrustPolicy(min_samples=10)
        )

        trained = trainer.fit(_dataset(6, seed=9))

        assert trained.trusted is False
        assert trained.state is SurrogateState.BOOTSTRAPPING
        assert "bootstrapping" in trained.trust_reason

    def test_the_gate_refuses_a_surrogate_that_cannot_rank(self) -> None:
        """Step 84. The property that stops a bad model spending real budget."""
        trainer = SurrogateTrainer(
            GenomeFeatureEncoder.from_space(_space()),
            policy=TrustPolicy(min_samples=4, min_spearman=0.5),
        )

        trained = trainer.fit(_dataset(12, seed=10, flat=True))

        assert trained.trusted is False
        assert trained.state is SurrogateState.DEGRADED

    def test_the_gate_admits_a_surrogate_that_can_rank(self) -> None:
        trainer = SurrogateTrainer(
            GenomeFeatureEncoder.from_space(_space()),
            policy=TrustPolicy(min_samples=8, min_spearman=0.3),
        )

        trained = trainer.fit(_dataset(30, seed=11))

        assert trained.trusted is True
        assert trained.state is SurrogateState.READY
        assert trained.usable is True

    def test_a_dataset_mixing_parameter_names_is_refused(self) -> None:
        """Step 69: two search spaces cannot share one model."""
        dataset = _dataset(4, seed=12)
        mixed = dataset.with_record(
            SurrogateTrainingRecord(
                candidate_id="odd", genome={"gain": 1.0},
                objective_values={"o1": 0.0}, directional_target=0.0,
                primary_objective_id="o1",
            )
        )
        trainer = SurrogateTrainer(GenomeFeatureEncoder.from_space(_space()))

        with pytest.raises(SurrogateError, match="search space changed"):
            trainer.fit(mixed)

    def test_the_target_spread_is_reported_for_judging_the_labels(self) -> None:
        """The number that says whether the labels are worth learning from."""
        trained = SurrogateTrainer(
            GenomeFeatureEncoder.from_space(_space()), policy=TrustPolicy(min_samples=4)
        ).fit(_dataset(10, seed=13))

        assert trained.target_spread > 0

    def test_families_can_be_compared_on_the_same_data(self) -> None:
        """Step 38, which reports rather than switching automatically."""
        encoder = GenomeFeatureEncoder.from_space(_space())

        scores = compare_families(encoder, _dataset(20, seed=14))

        assert SurrogateModelType.GAUSSIAN_PROCESS in scores
        assert all(metrics.n >= 0 for metrics in scores.values())


# ---------------------------------------------------------------------------
# Steps 26-27 - the candidate pool
# ---------------------------------------------------------------------------


class TestPool:
    def test_it_generates_distinct_valid_candidates(self) -> None:
        space = _space()
        generator = CandidatePoolGenerator(space, config=PoolConfig(size=500))

        pool = generator.generate(Random(0))

        assert len(pool) == 500
        fingerprints = {genome.fingerprint(space) for genome in pool}
        assert len(fingerprints) == 500
        for genome in pool:
            genome.validate_against(space)

    def test_it_never_offers_a_candidate_already_evaluated(self) -> None:
        """Its answer is known; screening it would waste the slot."""
        space = _space()
        generator = CandidatePoolGenerator(space, config=PoolConfig(size=200))
        known = CandidateGenome(genome_id="known", values={"gain": 3.0, "brightness": 0.4})

        pool = generator.generate(Random(1), exclude={known.fingerprint(space)})

        assert known.fingerprint(space) not in {g.fingerprint(space) for g in pool}

    def test_it_clusters_around_the_incumbents_when_given_them(self) -> None:
        space = _space()
        generator = CandidatePoolGenerator(
            space, config=PoolConfig(size=400, local_share=0.5, random_share=0.5)
        )
        incumbent = CandidateGenome(genome_id="best", values={"gain": 3.0, "brightness": 0.4})

        pool = generator.generate(Random(2), incumbents=[incumbent])
        near = sum(
            1 for genome in pool if abs(genome.values["gain"] - 3.0) < 3.0
        )

        assert near > len(pool) * 0.3

    def test_it_is_reproducible_from_a_seed(self) -> None:
        generator = CandidatePoolGenerator(_space(), config=PoolConfig(size=100))

        first = [g.values for g in generator.generate(Random(3))]
        second = [g.values for g in generator.generate(Random(3))]

        assert first == second


# ---------------------------------------------------------------------------
# Steps 29-30, 84, 87-88 - the optimization loop
# ---------------------------------------------------------------------------


class RecordingEvaluator:
    """A cheap stand-in for the real evaluator, on a known surface."""

    def __init__(self, surface=_surface, *, flat: bool = False) -> None:
        self.surface = surface
        self.flat = flat
        self.calls: list[dict[str, float]] = []

    def __call__(self, genome: CandidateGenome) -> CandidateFitness:
        self.calls.append(dict(genome.values))
        value = 0.5 if self.flat else self.surface(genome.values)
        return CandidateFitness(
            candidate_id=genome.genome_id, genome=genome,
            status=FitnessStatus.EVALUATED, scalar_fitness=value,
            primary_objective_id="o1", raw_objectives={"o1": value},
            compute=ComputeCost(wall_seconds=1.0, tribe_runs=1),
        )


def _orchestrator(evaluator, **overrides: object) -> BayesianOptimizationOrchestrator:
    config_fields: dict[str, object] = {
        "bootstrap_evaluations": 8,
        "acquisition": AcquisitionName.EXPECTED_IMPROVEMENT,
        "pool": PoolConfig(size=400),
        "random_seed": 7,
    }
    config_fields.update(overrides)
    return BayesianOptimizationOrchestrator(
        _space(),
        _objective(),
        _identity(),
        evaluator,
        config=BayesianOptimizationConfig(**config_fields),  # type: ignore[arg-type]
        trust=TrustPolicy(min_samples=8, min_spearman=0.2),
        budget=SearchBudget(max_candidates=100, max_generations=100),
    )


class TestNonFinitePredictions:
    """A prediction that is not a number must not choose a candidate.

    The check exists because numpy's floating-point warnings cannot be relied on
    to reveal this. Built against Apple Accelerate, numpy 2.2 raises `divide by
    zero`, `overflow` and `invalid value` for any sufficiently large matmul
    whatever its inputs, so the warnings are noise here and their absence is not
    a guarantee anywhere.
    """

    def test_the_platform_warnings_carry_no_information(self) -> None:
        """Documents why the guard is a value check and not a warning filter."""
        rng = np.random.default_rng(0)
        left, right = rng.random((400, 8)), rng.random(8)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            product = left @ right

        assert np.all(np.isfinite(product))
        assert np.all(np.isfinite(left)) and np.all(np.isfinite(right))
        # On a platform that reports them, the result is still perfectly finite,
        # so the warning says nothing about the numbers. Asserting only the
        # finiteness keeps this test true on platforms that stay quiet.
        _ = {str(item.message) for item in caught}

    def test_a_non_finite_mean_is_refused_rather_than_returned(self) -> None:
        surrogate = GaussianProcessSurrogate()
        features = np.array([[0.1, 0.2], [0.4, 0.7], [0.9, 0.3], [0.6, 0.6]])
        surrogate.fit(features, np.array([0.1, 0.5, 0.2, 0.4]))

        def poisoned(_features, return_std=False):
            mean = np.array([np.nan, 0.0, 0.0, 0.0])
            return (mean, np.zeros(4)) if return_std else mean

        surrogate._model.predict = poisoned  # type: ignore[method-assign]
        with pytest.raises(SurrogateError, match="non-finite"):
            surrogate.predict_with_uncertainty(features)

    def test_the_loop_falls_back_instead_of_crashing(self) -> None:
        """Budget is left; one unusable round should not end the search."""
        orchestrator = _orchestrator(RecordingEvaluator(), bootstrap_evaluations=8)

        def refuse(_candidates):
            raise SurrogateError("posterior mean contained 3 non-finite value(s)")

        orchestrator.run(9)
        orchestrator.trainer.predict = refuse  # type: ignore[method-assign]
        orchestrator.run(11)

        modes = [row.mode for row in orchestrator.rounds]
        assert modes[8] == "surrogate", "the model was choosing before it was poisoned"
        # Every remaining round still happened, each one chosen by the fallback.
        assert modes[9:] == ["fallback"] * 11
        assert orchestrator.stopped_reason == ""
        assert "non-finite" in orchestrator.rounds[9].selection_reason


class TestOptimizationLoop:
    def test_it_bootstraps_before_using_the_model(self) -> None:
        """Step 30. A model fitted to three points should not choose anything."""
        evaluator = RecordingEvaluator()
        orchestrator = _orchestrator(evaluator, bootstrap_evaluations=5)

        orchestrator.run(5)

        assert [row.mode for row in orchestrator.rounds] == ["bootstrap"] * 5
        assert orchestrator.state is SurrogateState.BOOTSTRAPPING

    def test_it_switches_to_the_surrogate_once_it_can_rank(self) -> None:
        evaluator = RecordingEvaluator()
        orchestrator = _orchestrator(evaluator, bootstrap_evaluations=8)

        orchestrator.run(16)
        modes = [row.mode for row in orchestrator.rounds]

        assert modes[:8] == ["bootstrap"] * 8
        assert "surrogate" in modes[8:]

    def test_it_approaches_the_known_optimum(self) -> None:
        """Step 77 end to end: the loop should find the peak at (3, 0.4)."""
        evaluator = RecordingEvaluator()
        orchestrator = _orchestrator(evaluator, bootstrap_evaluations=8)

        orchestrator.run(22)
        best = max(orchestrator.results, key=lambda item: item.scalar_fitness or -1e9)

        assert best.scalar_fitness > -2.0
        assert abs(best.genome.values["gain"] - 3.0) < 3.0

    def test_it_beats_pure_bootstrap_on_the_same_budget(self) -> None:
        """The claim the phase exists to support, on a surface where it is checkable."""
        assisted = _orchestrator(RecordingEvaluator(), bootstrap_evaluations=8)
        assisted.run(20)

        random_only = _orchestrator(RecordingEvaluator(), bootstrap_evaluations=100)
        random_only.run(20)

        best_assisted = max(item.scalar_fitness or -1e9 for item in assisted.results)
        best_random = max(item.scalar_fitness or -1e9 for item in random_only.results)

        assert best_assisted > best_random

    def test_a_flat_objective_falls_back_instead_of_pretending(self) -> None:
        """Step 84 on the loop: an unrankable objective must not drive selection."""
        evaluator = RecordingEvaluator(flat=True)
        orchestrator = _orchestrator(evaluator, bootstrap_evaluations=6)

        orchestrator.run(12)
        modes = [row.mode for row in orchestrator.rounds]

        assert "surrogate" not in modes
        assert "fallback" in modes
        assert orchestrator.trained is not None
        assert orchestrator.trained.trusted is False

    def test_it_records_predicted_against_actual(self) -> None:
        """Step 35. The diagnostic that says whether to believe the model."""
        orchestrator = _orchestrator(RecordingEvaluator(), bootstrap_evaluations=8)

        orchestrator.run(14)
        compared = [row for row in orchestrator.rounds if row.predicted_score is not None]

        assert compared
        for row in compared:
            assert row.actual_score is not None
            assert row.prediction_error == pytest.approx(
                row.actual_score - row.predicted_score
            )

    def test_the_budget_caps_real_evaluations_however_many_are_screened(self) -> None:
        """Step 87. Ten thousand cheap scores cannot buy an eleventh evaluation."""
        evaluator = RecordingEvaluator()
        orchestrator = BayesianOptimizationOrchestrator(
            _space(), _objective(), _identity(), evaluator,
            config=BayesianOptimizationConfig(
                bootstrap_evaluations=3, pool=PoolConfig(size=2000), random_seed=1
            ),
            trust=TrustPolicy(min_samples=3, min_spearman=0.0),
            budget=SearchBudget(max_candidates=5, max_generations=100),
        )

        orchestrator.run(50)

        assert len(evaluator.calls) == 5
        assert "budget exhausted" in orchestrator.stopped_reason

    def test_the_dataset_grows_and_the_model_version_increments(self) -> None:
        """Step 88."""
        orchestrator = _orchestrator(RecordingEvaluator(), bootstrap_evaluations=8)

        orchestrator.run(10)
        first_version = orchestrator.trained.metadata.model_version  # type: ignore[union-attr]
        first_size = orchestrator.dataset.size

        orchestrator.run(2)

        assert orchestrator.dataset.size > first_size
        assert orchestrator.trained.metadata.model_version > first_version  # type: ignore[union-attr]

    def test_only_real_evaluations_become_training_rows(self) -> None:
        """A model trained on its own output learns its own errors."""
        orchestrator = _orchestrator(RecordingEvaluator(), bootstrap_evaluations=4)

        orchestrator.run(10)

        assert orchestrator.dataset.size == len(orchestrator.results)
        assert all(
            record.directional_target
            in {item.scalar_fitness for item in orchestrator.results}
            for record in orchestrator.dataset.records
        )

    def test_a_failed_evaluation_is_kept_out_of_the_training_set(self) -> None:
        class SometimesFails:
            def __init__(self) -> None:
                self.n = 0

            def __call__(self, genome: CandidateGenome) -> CandidateFitness:
                self.n += 1
                if self.n == 2:
                    return CandidateFitness(
                        candidate_id=genome.genome_id, genome=genome,
                        status=FitnessStatus.EVALUATION_FAILED, reason="synthetic failure",
                    )
                value = _surface(genome.values)
                return CandidateFitness(
                    candidate_id=genome.genome_id, genome=genome,
                    status=FitnessStatus.EVALUATED, scalar_fitness=value,
                    primary_objective_id="o1", raw_objectives={"o1": value},
                    compute=ComputeCost(wall_seconds=1.0, tribe_runs=1),
                )

        orchestrator = _orchestrator(SometimesFails(), bootstrap_evaluations=4)
        orchestrator.run(6)

        assert orchestrator.dataset.size == 5
        assert len(orchestrator.results) == 6

    def test_diagnostics_summarise_the_run(self) -> None:
        orchestrator = _orchestrator(RecordingEvaluator(), bootstrap_evaluations=8)
        orchestrator.run(14)

        diagnostics = orchestrator.diagnostics()

        assert diagnostics["rounds"] == 14
        assert diagnostics["bootstrap_rounds"] == 8
        assert diagnostics["training_samples"] == 14
        assert diagnostics["prediction_mae"] is not None

    def test_a_minimize_objective_is_folded_before_training(self) -> None:
        """The surrogate always learns a maximisation problem."""
        class Minimising:
            def __call__(self, genome: CandidateGenome) -> CandidateFitness:
                raw = abs(genome.values["gain"])
                return CandidateFitness(
                    candidate_id=genome.genome_id, genome=genome,
                    status=FitnessStatus.EVALUATED, scalar_fitness=-raw,
                    primary_objective_id="o1", raw_objectives={"o1": raw},
                    compute=ComputeCost(wall_seconds=1.0, tribe_runs=1),
                )

        orchestrator = BayesianOptimizationOrchestrator(
            _space(), _objective(ObjectiveDirection.MINIMIZE), _identity(), Minimising(),
            config=BayesianOptimizationConfig(bootstrap_evaluations=4, pool=PoolConfig(size=200)),
            trust=TrustPolicy(min_samples=4, min_spearman=0.0),
            budget=SearchBudget(max_candidates=20),
        )
        orchestrator.run(8)

        # Folded targets are negative raw values, so higher is better.
        assert all(record.directional_target <= 0 for record in orchestrator.dataset.records)


class TestLedgerIntegration:
    def test_surrogate_predictions_do_not_count_against_the_budget(self) -> None:
        """Step 46. Screening thousands is free; only evaluations are charged."""
        evaluator = RecordingEvaluator()
        orchestrator = _orchestrator(evaluator, bootstrap_evaluations=4)

        orchestrator.run(8)

        assert orchestrator.ledger.tribe_runs == 8
        assert orchestrator.ledger.candidates_evaluated == 8
        assert isinstance(orchestrator.ledger, BudgetLedger)


# ---------------------------------------------------------------------------
# Steps 41-43, 85 - multi-objective surrogates
# ---------------------------------------------------------------------------


class TestMultiObjective:
    def _objectives(self) -> tuple[NeuralObjective, ...]:
        return (
            _objective_named("o1", ObjectiveDirection.MAXIMIZE),
            _objective_named("o2", ObjectiveDirection.MINIMIZE),
        )

    def _multi_dataset(self, n: int, *, seed: int = 0) -> SurrogateDataset:
        rng = Random(seed)
        space = _space()
        dataset = SurrogateDataset(identity=_identity())
        for index in range(n):
            values = space.sample(rng)
            dataset = dataset.with_record(
                SurrogateTrainingRecord(
                    candidate_id=f"m{index:03d}",
                    genome=values,
                    # o1 rewards high gain; o2 is minimised and rewards low
                    # brightness, so the two genuinely conflict.
                    objective_values={"o1": values["gain"], "o2": values["brightness"]},
                    directional_target=values["gain"],
                    primary_objective_id="o1",
                    sequence=index,
                )
            )
        return dataset

    def _surrogate(self, **kwargs: object):
        from blackmirror.surrogate.multi import MultiObjectiveSurrogate

        return MultiObjectiveSurrogate(
            GenomeFeatureEncoder.from_space(_space()),
            self._objectives(),
            policy=TrustPolicy(min_samples=6, min_spearman=0.2),
            **kwargs,  # type: ignore[arg-type]
        )

    def test_it_fits_one_model_per_objective(self) -> None:
        surrogate = self._surrogate()

        trained = surrogate.fit(self._multi_dataset(20, seed=1))

        assert set(trained) == {"o1", "o2"}
        assert surrogate.all_trusted is True

    def test_a_minimize_objective_is_folded_per_model(self) -> None:
        """Each surrogate learns a maximisation problem in its own units."""
        surrogate = self._surrogate()
        dataset = self._multi_dataset(16, seed=2)

        folded = surrogate._dataset_for(dataset, self._objectives()[1])

        for record, original in zip(folded.records, dataset.records, strict=True):
            assert record.directional_target == -original.objective_values["o2"]

    def test_a_record_missing_one_objective_still_trains_the_others(self) -> None:
        """Dropping it everywhere would discard perfectly good measurements."""
        dataset = self._multi_dataset(12, seed=3).with_record(
            SurrogateTrainingRecord(
                candidate_id="partial",
                genome={"gain": 1.0, "brightness": 0.0},
                objective_values={"o1": 1.0},
                directional_target=1.0,
                primary_objective_id="o1",
            )
        )
        surrogate = self._surrogate()

        surrogate.fit(dataset)

        assert surrogate.trained["o1"].metadata.training_dataset_size == 13
        assert surrogate.trained["o2"].metadata.training_dataset_size == 12

    def test_weights_are_drawn_on_the_simplex(self) -> None:
        """Step 43. They sum to one and reach the extremes across rounds."""
        from blackmirror.surrogate.multi import sample_weights

        rng = Random(4)
        draws = [sample_weights(["o1", "o2"], rng, round_number=i) for i in range(200)]

        for draw in draws:
            assert sum(draw.weights.values()) == pytest.approx(1.0)
        assert any(max(draw.weights.values()) > 0.9 for draw in draws)

    def test_opposing_weightings_prefer_different_candidates(self) -> None:
        """If they agreed, the weights would not be doing anything."""
        from blackmirror.surrogate.multi import ScalarizationWeights

        surrogate = self._surrogate()
        surrogate.fit(self._multi_dataset(20, seed=5))
        rng = Random(6)
        space = _space()
        candidates = {f"c{i}": space.sample(rng) for i in range(30)}

        only_o1 = surrogate.scalarized(
            candidates, ScalarizationWeights(weights={"o1": 1.0, "o2": 0.0})
        )
        only_o2 = surrogate.scalarized(
            candidates, ScalarizationWeights(weights={"o1": 0.0, "o2": 1.0})
        )

        best_o1 = max(only_o1, key=lambda item: item.predicted_score).candidate_id
        best_o2 = max(only_o2, key=lambda item: item.predicted_score).candidate_id
        assert best_o1 != best_o2

    def test_scalarized_predictions_carry_the_weights_that_made_them(self) -> None:
        from blackmirror.surrogate.multi import ScalarizationWeights

        surrogate = self._surrogate()
        surrogate.fit(self._multi_dataset(16, seed=7))
        rng = Random(8)
        space = _space()
        candidates = {f"c{i}": space.sample(rng) for i in range(10)}

        scored = surrogate.scalarized(
            candidates, ScalarizationWeights(weights={"o1": 0.7, "o2": 0.3})
        )

        assert scored
        assert "o1 0.70" in scored[0].reason

    def test_the_reported_frontier_comes_from_measurements(self) -> None:
        """A predicted frontier is not a frontier."""
        from blackmirror.surrogate.multi import observed_pareto_front

        measured = [
            CandidateFitness(
                candidate_id="a",
                genome=CandidateGenome(genome_id="a", values={"gain": 1.0}),
                status=FitnessStatus.EVALUATED, scalar_fitness=1.0,
                primary_objective_id="o1", raw_objectives={"o1": 3.0, "o2": 1.0},
            ),
            CandidateFitness(
                candidate_id="b",
                genome=CandidateGenome(genome_id="b", values={"gain": 2.0}),
                status=FitnessStatus.EVALUATED, scalar_fitness=1.0,
                primary_objective_id="o1", raw_objectives={"o1": 1.0, "o2": 5.0},
            ),
        ]

        front = observed_pareto_front(measured, self._objectives())

        # o2 is minimised, so "a" (higher o1, lower o2) dominates "b".
        assert front.non_dominated == ("a",)

    def test_an_unlearnable_objective_degrades_only_itself(self) -> None:
        """One bad objective must not contaminate the others' verdicts."""
        rng = Random(9)
        space = _space()
        dataset = SurrogateDataset(identity=_identity())
        for index in range(14):
            values = space.sample(rng)
            dataset = dataset.with_record(
                SurrogateTrainingRecord(
                    candidate_id=f"x{index:03d}", genome=values,
                    objective_values={"o1": values["gain"], "o2": 0.5},
                    directional_target=values["gain"], primary_objective_id="o1",
                    sequence=index,
                )
            )
        from blackmirror.surrogate.multi import MultiObjectiveSurrogate

        surrogate = MultiObjectiveSurrogate(
            GenomeFeatureEncoder.from_space(space),
            self._objectives(),
            policy=TrustPolicy(min_samples=6, min_spearman=0.3),
        )

        surrogate.fit(dataset)

        assert surrogate.trained["o1"].trusted is True
        assert surrogate.trained["o2"].trusted is False
        assert surrogate.all_trusted is False
        assert surrogate.trusted_objectives == ["o1"]

    def test_it_refuses_to_be_built_with_no_objectives(self) -> None:
        from blackmirror.surrogate.multi import MultiObjectiveSurrogate

        with pytest.raises(ValueError, match="at least one objective"):
            MultiObjectiveSurrogate(GenomeFeatureEncoder.from_space(_space()), ())
