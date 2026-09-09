"""Deterministic construction of candidate media from a described edit.

The join Phase 8 deliberately left open: Phase 7 describes a variant, Phase 8
measures one that exists, and this builds the file in between. Restricted to
parametric ffmpeg edits, so nothing here can invent footage, speech or text.
"""

from blackmirror.materialization.adapter import (
    MaterializedVariantAdapter,
    binding_from_result,
)
from blackmirror.materialization.ffmpeg import (
    MaterializationError,
    candidate_key,
    materialize,
)
from blackmirror.materialization.schemas import (
    EditOperation,
    EditStep,
    MaterializationPlan,
    MaterializationResult,
)

__all__ = [
    "EditOperation",
    "EditStep",
    "MaterializationError",
    "MaterializationPlan",
    "MaterializationResult",
    "MaterializedVariantAdapter",
    "binding_from_result",
    "candidate_key",
    "materialize",
]
