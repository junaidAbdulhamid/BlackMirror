"""Step 57: where a surrogate's dataset, models and diagnostics live.

    artifacts/search/<search_id>/surrogate/
        dataset/records.json      real observations only
        encoder/metadata.json     column layout and scaling
        models/model_vNNN.json    metadata per version, newest last
        diagnostics/metrics.json  validation and uncertainty quality
        acquisitions/round_NNN.json   what was predicted and then measured

WHY MODEL METADATA RATHER THAN PICKLED MODELS
    A pickled scikit-learn estimator is tied to the library version that wrote
    it and silently fails to load across upgrades. Everything needed to rebuild
    a model exactly is already recorded: the dataset hash, the encoder schema,
    the hyperparameters and the seed. Refitting from those is fast at these
    sample sizes and cannot go stale, so the artifact stores the recipe rather
    than the object.

    That also makes Step 82's requirement stronger than serialization would:
    the model is not merely reloadable, it is reproducible.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from blackmirror.surrogate.encoder import GenomeFeatureEncoder
from blackmirror.surrogate.orchestrator import RoundRecord
from blackmirror.surrogate.schemas import SurrogateDataset, SurrogateModelMetadata
from blackmirror.surrogate.trainer import TrainedSurrogate
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)


class SurrogateStore:
    """Reads and writes one search's surrogate artifacts."""

    def __init__(self, artifact_root: Path, search_id: str) -> None:
        _validate_id(search_id)
        self.root = Path(artifact_root) / "search" / search_id / "surrogate"
        self.search_id = search_id

    @property
    def exists(self) -> bool:
        return (self.root / "dataset" / "records.json").is_file()

    # --- writing ----------------------------------------------------------

    def write_dataset(self, dataset: SurrogateDataset) -> Path:
        path = self.root / "dataset" / "records.json"
        _write(path, dataset.model_dump_json(indent=2))
        return path

    def write_encoder(self, encoder: GenomeFeatureEncoder) -> Path:
        path = self.root / "encoder" / "metadata.json"
        _write(path, encoder.to_json())
        return path

    def write_model(self, metadata: SurrogateModelMetadata) -> Path:
        path = self.root / "models" / f"model_v{metadata.model_version:03d}.json"
        _write(path, metadata.model_dump_json(indent=2))
        return path

    def write_diagnostics(self, trained: TrainedSurrogate) -> Path:
        path = self.root / "diagnostics" / "metrics.json"
        _write(
            path,
            json.dumps(
                {
                    "state": trained.state.value,
                    "trusted": trained.trusted,
                    "trust_reason": trained.trust_reason,
                    "target_spread": trained.target_spread,
                    "cross_validation": trained.cross_validation.model_dump(mode="json"),
                    "prefix_validation": trained.prefix_validation.model_dump(mode="json"),
                    "uncertainty": trained.uncertainty.model_dump(mode="json"),
                    "model": trained.metadata.model_dump(mode="json"),
                },
                indent=2,
                default=str,
            ),
        )
        return path

    def write_round(self, record: RoundRecord) -> Path:
        path = self.root / "acquisitions" / f"round_{record.round_number:03d}.json"
        _write(path, record.model_dump_json(indent=2))
        return path

    def write_report(self, payload: dict[str, object]) -> Path:
        path = self.root / "report.json"
        _write(path, json.dumps(payload, indent=2, default=str))
        return path

    # --- reading ----------------------------------------------------------

    def read_dataset(self) -> SurrogateDataset | None:
        path = self.root / "dataset" / "records.json"
        if not path.is_file():
            return None
        try:
            return SurrogateDataset.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            logger.warning("surrogate dataset for %s is unreadable", self.search_id)
            return None

    def read_encoder(self) -> GenomeFeatureEncoder | None:
        path = self.root / "encoder" / "metadata.json"
        if not path.is_file():
            return None
        try:
            return GenomeFeatureEncoder.from_json(path.read_text(encoding="utf-8"))
        except ValueError:
            return None

    def read_diagnostics(self) -> dict[str, object] | None:
        return _read_json(self.root / "diagnostics" / "metrics.json")

    def read_report(self) -> dict[str, object] | None:
        return _read_json(self.root / "report.json")

    def read_models(self) -> list[SurrogateModelMetadata]:
        directory = self.root / "models"
        if not directory.is_dir():
            return []
        out: list[SurrogateModelMetadata] = []
        for path in sorted(directory.glob("model_v*.json")):
            try:
                out.append(
                    SurrogateModelMetadata.model_validate_json(
                        path.read_text(encoding="utf-8")
                    )
                )
            except ValueError:
                continue
        return out

    def read_rounds(self) -> list[RoundRecord]:
        directory = self.root / "acquisitions"
        if not directory.is_dir():
            return []
        out: list[RoundRecord] = []
        for path in sorted(directory.glob("round_*.json")):
            try:
                out.append(RoundRecord.model_validate_json(path.read_text(encoding="utf-8")))
            except ValueError:
                continue
        return out

    def latest_model(self) -> SurrogateModelMetadata | None:
        models = self.read_models()
        return models[-1] if models else None


def persist(
    artifact_root: Path,
    search_id: str,
    *,
    dataset: SurrogateDataset,
    encoder: GenomeFeatureEncoder,
    trained: TrainedSurrogate | None,
    rounds: list[RoundRecord],
    report: dict[str, object] | None = None,
) -> SurrogateStore:
    """Write everything a reader or a resumed run would need."""
    store = SurrogateStore(artifact_root, search_id)
    store.write_dataset(dataset)
    store.write_encoder(encoder)
    if trained is not None:
        store.write_model(trained.metadata)
        store.write_diagnostics(trained)
    for record in rounds:
        store.write_round(record)
    if report is not None:
        store.write_report(report)
    return store


def _write(path: Path, payload: str) -> None:
    """Atomic, so a crash mid-write cannot leave half a file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 - renamed below
        mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp",
        dir=path.parent, delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _validate_id(value: str) -> None:
    if value in {".", ".."} or re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", value) is None:
        raise ValueError("invalid search id")


__all__ = ["SurrogateStore", "persist"]
