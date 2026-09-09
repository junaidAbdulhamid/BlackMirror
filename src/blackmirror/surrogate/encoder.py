"""Steps 3-6: turning a genome into a fixed numeric vector, reproducibly.

WHAT AN ENCODER IS FOR
    A model needs a fixed-length numeric vector with a stable column order.
    A genome is a mapping from parameter names to values, and its ordering is
    whatever the dictionary happened to have. Encoding makes the two agree, and
    persisting the encoder makes a prediction made today comparable to one made
    after a restart.

WHY COLUMN ORDER IS FROZEN AT CONSTRUCTION
    A model trained with `gain` in column 0 and asked to predict with
    `brightness` in column 0 returns a confident, meaningless number. Nothing
    would raise. So the order is derived from the space once, stored, and
    checked on every encode.

WHY CATEGORICALS ARE ONE-HOT AND NOTHING MORE
    Step 4 asks about embeddings for high-cardinality choices. The search space
    cannot express them: text and structured-intervention parameters are
    rejected at construction because nothing can materialize them, and the
    categorical parameters that do exist hold a handful of numeric choices.
    One-hot is correct here, and an embedding layer would be machinery around a
    feature type the system cannot produce.

WHY SCALING IS FITTED ON TRAINING DATA ONLY
    A Gaussian Process with a shared length scale needs comparably scaled
    inputs. Fitting the scaler to the pool being predicted, rather than to the
    training set, leaks the pool's distribution into the model and quietly
    changes what a stored model means.

    With very few samples a fitted standard deviation is unstable, so the
    default scales by the *declared domain* of each parameter instead. That
    needs no data at all, is exactly reproducible, and maps every parameter to
    [0, 1] whatever its units.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from blackmirror.search.space import ContentSearchSpace, ParameterType
from blackmirror.surrogate.schemas import FEATURE_SCHEMA_VERSION


class EncodingError(ValueError):
    """The genome cannot be encoded against this schema."""


class FeatureColumn(BaseModel):
    """One column of the encoded matrix and where its value comes from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    parameter: str
    #: For a one-hot column, the choice it represents.
    choice: float | None = None
    low: float = 0.0
    high: float = 1.0

    @property
    def is_one_hot(self) -> bool:
        return self.choice is not None


class GenomeFeatureEncoder(BaseModel):
    """Step 3. Deterministic genome to vector, with a versioned schema."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    columns: tuple[FeatureColumn, ...] = Field(min_length=1)
    parameter_names: tuple[str, ...] = Field(min_length=1)
    #: Domain scaling needs no data and is stable at any n; standard scaling is
    #: available for callers with enough samples to estimate a variance.
    scaling: str = "domain"
    means: tuple[float, ...] = ()
    scales: tuple[float, ...] = ()
    feature_schema_version: str = FEATURE_SCHEMA_VERSION

    # --- construction -----------------------------------------------------

    @classmethod
    def from_space(cls, space: ContentSearchSpace) -> GenomeFeatureEncoder:
        """Derive columns from the declared space, in stable name order."""
        columns: list[FeatureColumn] = []
        for parameter in sorted(space.parameters, key=lambda item: item.name):
            if parameter.type in (ParameterType.CONTINUOUS, ParameterType.INTEGER):
                assert parameter.low is not None and parameter.high is not None
                columns.append(
                    FeatureColumn(
                        name=parameter.name,
                        parameter=parameter.name,
                        low=parameter.low,
                        high=parameter.high,
                    )
                )
            elif parameter.type is ParameterType.BOOLEAN:
                # Two choices read as (off, on); one column carrying 0 or 1 is
                # the whole of the information, and one-hot would add a
                # perfectly collinear second column.
                off, on = parameter.choices
                columns.append(
                    FeatureColumn(
                        name=parameter.name, parameter=parameter.name, low=off, high=on
                    )
                )
            else:
                for choice in parameter.choices:
                    columns.append(
                        FeatureColumn(
                            name=f"{parameter.name}={choice:g}",
                            parameter=parameter.name,
                            choice=choice,
                        )
                    )
        if not columns:  # pragma: no cover - the space schema forbids it
            raise EncodingError("a search space with no parameters cannot be encoded")
        return cls(
            columns=tuple(columns),
            parameter_names=tuple(sorted(item.name for item in space.parameters)),
        )

    # --- encoding ---------------------------------------------------------

    @property
    def width(self) -> int:
        return len(self.columns)

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    def encode(self, genome: dict[str, float]) -> np.ndarray:
        """One genome to one row. Raises rather than guessing a missing value."""
        missing = set(self.parameter_names) - set(genome)
        if missing:
            raise EncodingError(
                f"genome is missing values for: {', '.join(sorted(missing))}"
            )
        unknown = set(genome) - set(self.parameter_names)
        if unknown:
            raise EncodingError(
                f"genome carries parameters this schema does not know: "
                f"{', '.join(sorted(unknown))}; the search space has probably "
                f"changed and the surrogate must be retrained"
            )

        row = np.zeros(self.width, dtype=float)
        for index, column in enumerate(self.columns):
            value = genome[column.parameter]
            if column.is_one_hot:
                assert column.choice is not None
                row[index] = 1.0 if value == column.choice else 0.0
            else:
                span = column.high - column.low
                # Domain scaling maps the declared range onto [0, 1], so a knob
                # measured in decibels and one measured in arbitrary units
                # contribute comparably to a distance.
                row[index] = 0.0 if span == 0 else (value - column.low) / span
        return row

    def encode_many(self, genomes: list[dict[str, float]]) -> np.ndarray:
        if not genomes:
            return np.zeros((0, self.width), dtype=float)
        return np.vstack([self.encode(genome) for genome in genomes])

    def fitted_on(self, genomes: list[dict[str, float]]) -> GenomeFeatureEncoder:
        """A copy using standard scaling fitted to these genomes.

        Offered for callers with enough data to estimate a variance. A
        zero-variance column keeps a scale of one rather than dividing by zero,
        which is the correct behaviour for a parameter that never varied.
        """
        matrix = self.encode_many(genomes)
        if matrix.shape[0] < 2:
            raise EncodingError(
                "standard scaling needs at least two genomes; with fewer, domain "
                "scaling is both stable and exactly reproducible"
            )
        means = matrix.mean(axis=0)
        scales = matrix.std(axis=0)
        scales[scales == 0] = 1.0
        return self.model_copy(
            update={
                "scaling": "standard",
                "means": tuple(float(v) for v in means),
                "scales": tuple(float(v) for v in scales),
            }
        )

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        """Apply the stored scaling. A no-op under domain scaling."""
        if self.scaling != "standard":
            return matrix
        if not self.means or not self.scales:  # pragma: no cover - defensive
            raise EncodingError("standard scaling was declared without parameters")
        return (matrix - np.asarray(self.means)) / np.asarray(self.scales)

    def encode_dataset(self, genomes: list[dict[str, float]]) -> np.ndarray:
        return self.transform(self.encode_many(genomes))

    # --- persistence ------------------------------------------------------

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, payload: str) -> GenomeFeatureEncoder:
        return cls.model_validate_json(payload)

    def schema_fingerprint(self) -> str:
        """Identity of the column layout, for detecting a changed space."""
        import hashlib

        encoded = json.dumps(
            {
                "version": self.feature_schema_version,
                "columns": [column.model_dump(mode="json") for column in self.columns],
                "scaling": self.scaling,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode()).hexdigest()

    def describe(self) -> tuple[str, ...]:
        def domain(column: FeatureColumn) -> str:
            if column.is_one_hot:
                assert column.choice is not None
                return f" == {column.choice:g}"
            return f" in [{column.low:g}, {column.high:g}]"

        return tuple(
            f"{column.name} <- {column.parameter}{domain(column)}"
            for column in self.columns
        )


def encoder_for(space: ContentSearchSpace, **kwargs: Any) -> GenomeFeatureEncoder:
    return GenomeFeatureEncoder.from_space(space, **kwargs)


__all__ = ["EncodingError", "FeatureColumn", "GenomeFeatureEncoder", "encoder_for"]
