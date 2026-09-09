"""Bind a machine-materialized variant through Phase 8's adapter contract.

WHY A SEPARATE ADAPTER
    Phase 8 already has `UserSuppliedVariantAdapter`, whose recorded method is
    `user_supplied_media`. Reusing it for a generated file would record a human
    provenance for machine output, which is exactly the kind of small lie that
    makes an artifact store untrustworthy later. This adapter records what
    actually happened: which tool, which version, which parameters.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from blackmirror.materialization.ffmpeg import MaterializationError, ffmpeg_version
from blackmirror.materialization.schemas import MaterializationResult
from blackmirror.resimulation.schemas import TransformationBinding

ADAPTER_ID = "ffmpeg-materialized"
ADAPTER_VERSION = "1.0"


class MaterializedVariantAdapter:
    """Phase 8 `TransformationAdapter` for files this system produced itself."""

    adapter_id = ADAPTER_ID
    adapter_version = ADAPTER_VERSION

    def bind(
        self, source: Path, variant: Path, parameters: dict[str, object]
    ) -> TransformationBinding:
        """Bind two files that already exist, recording the edit parameters.

        Kept signature-compatible with the user-supplied adapter so the registry
        stays uniform. `binding_from_result` is the path Phase 9 actually uses,
        because it carries the plan rather than re-deriving it.
        """
        for label, path in (("source", source), ("variant", variant)):
            if not path.is_file():
                raise MaterializationError(f"{label} media does not exist: {path}")
        normalized = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
        return TransformationBinding(
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            method="deterministic_ffmpeg_edit",
            tool=f"ffmpeg {ffmpeg_version()}",
            parameters=parameters,  # type: ignore[arg-type]
            parameters_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
            source_path=str(source.resolve()),
            source_sha256=_file_hash(source),
            variant_path=str(variant.resolve()),
            variant_sha256=_file_hash(variant),
        )


def binding_from_result(result: MaterializationResult) -> TransformationBinding:
    """Turn a materialization into a Phase 8 binding without re-hashing.

    The hashes were computed while the file was written, so reusing them keeps
    the binding consistent with what was actually produced rather than with
    whatever is at that path now.
    """
    parameters: dict[str, float | int | str | bool | None] = {
        step.operation.value: step.value for step in result.plan.effective_steps
    }
    parameters["candidate_key"] = result.candidate_key
    parameters["duration_change_seconds"] = round(result.duration_change_seconds, 6)
    normalized = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
    return TransformationBinding(
        adapter_id=ADAPTER_ID,
        adapter_version=ADAPTER_VERSION,
        method="deterministic_ffmpeg_edit",
        tool=f"{result.tool} {result.tool_version}",
        parameters=parameters,
        parameters_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
        source_path=str(Path(result.plan.source_path).resolve()),
        source_sha256=result.source_sha256,
        variant_path=str(Path(result.variant_path).resolve()),
        variant_sha256=result.variant_sha256,
    )


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "ADAPTER_ID",
    "ADAPTER_VERSION",
    "MaterializedVariantAdapter",
    "binding_from_result",
]
