"""Authorized transformation bindings; no media generation occurs implicitly."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

from blackmirror.resimulation.schemas import TransformationBinding


class TransformationAdapter(Protocol):
    adapter_id: str
    adapter_version: str

    def bind(
        self, source: Path, variant: Path, parameters: dict[str, object]
    ) -> TransformationBinding: ...


class UserSuppliedVariantAdapter:
    adapter_id = "user-supplied"
    adapter_version = "1.0"

    def bind(
        self, source: Path, variant: Path, parameters: dict[str, object]
    ) -> TransformationBinding:
        for label, path in (("source", source), ("variant", variant)):
            if not path.is_file():
                raise FileNotFoundError(f"{label} media does not exist: {path}")
        normalized = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
        return TransformationBinding(
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            method="user_supplied_media",
            parameters=parameters,  # type: ignore[arg-type]
            parameters_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
            source_path=str(source.resolve()),
            source_sha256=_file_hash(source),
            variant_path=str(variant.resolve()),
            variant_sha256=_file_hash(variant),
        )


class AdapterRegistry:
    def __init__(self, adapters: tuple[TransformationAdapter, ...] = ()) -> None:
        supplied = adapters or (UserSuppliedVariantAdapter(),)
        self._adapters = {item.adapter_id: item for item in supplied}
        if len(self._adapters) != len(supplied):
            raise ValueError("transformation adapter ids must be unique")

    def get(self, adapter_id: str) -> TransformationAdapter:
        try:
            return self._adapters[adapter_id]
        except KeyError as exc:
            raise ValueError(f"unregistered transformation adapter: {adapter_id}") from exc


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
