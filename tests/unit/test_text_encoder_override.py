"""Overriding the gated text encoder.

TRIBE's text stream runs on the gated `meta-llama/Llama-3.2-3B`. The override
exists so an operator can point at a repo they can actually read, without
weakening the default or hiding that a substitution happened.
"""

from __future__ import annotations

import pytest

from blackmirror.config.settings import Settings
from blackmirror.schemas.metadata import ModelMetadata
from blackmirror.utils.hashing import build_cache_key

tribe_loader = pytest.importorskip("blackmirror.inference.tribe_loader")
TribeModelLoader = tribe_loader.TribeModelLoader


def _loader(**overrides: object) -> object:
    return TribeModelLoader(Settings(**overrides), device="cpu")  # type: ignore[arg-type]


def test_default_keeps_the_checkpoints_own_encoder() -> None:
    """No override means no config change: the default stays honest."""
    assert _loader()._text_encoder_override() == {}


def test_override_targets_the_text_extractor_only() -> None:
    override = _loader(text_encoder_id="unsloth/Llama-3.2-3B")._text_encoder_override()
    assert override == {"data.text_feature.model_name": "unsloth/Llama-3.2-3B"}


def test_device_overrides_cover_every_frozen_extractor() -> None:
    """The checkpoint ships all four extractors set to 'cuda'."""
    overrides = _loader()._extractor_device_overrides()
    assert overrides == {
        "data.text_feature.device": "cpu",
        "data.audio_feature.device": "cpu",
        # Video/image nest their encoder one level deeper.
        "data.video_feature.image.device": "cpu",
        "data.image_feature.image.device": "cpu",
    }


def test_override_makes_runs_non_comparable() -> None:
    """A different text encoder must not reuse another encoder's cached run."""
    def metadata(text_encoder: str) -> ModelMetadata:
        return ModelMetadata(
            name="TRIBE v2",
            backend="tribe_v2",
            model_id="facebook/tribev2",
            checkpoint="best.ckpt",
            loaded_device="cpu",
            dtype="float32",
            feature_extractors={"text": text_encoder, "audio": "facebook/w2v-bert-2.0"},
        )

    def key(text_encoder: str) -> str:
        return build_cache_key(
            stimulus_sha256="a" * 64,
            model_fingerprint=metadata(text_encoder).fingerprint(),
            preprocessing_fingerprint={"remove_empty_segments": True},
        )

    assert key("meta-llama/Llama-3.2-3B") != key("unsloth/Llama-3.2-3B")
    assert key("unsloth/Llama-3.2-3B") == key("unsloth/Llama-3.2-3B")


def test_setting_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BLACKMIRROR_TEXT_ENCODER_ID", "org/some-mirror")
    assert Settings().text_encoder_id == "org/some-mirror"
