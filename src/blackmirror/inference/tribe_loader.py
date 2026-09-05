"""TRIBE v2 model lifecycle.

Responsibility boundary: this module locates, downloads, configures and caches
the model, and reports its provenance. It performs **no** content preprocessing
and **no** result analysis — those live in ``tribe_backend.py`` and
``postprocessing.py`` respectively.

The loaded instance is held for the lifetime of the loader, so one model serves
many stimuli. That is what makes Phase 5's A/B/N comparison affordable and fair:
every variant is scored by the identical model object.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from blackmirror.config.settings import Settings
from blackmirror.errors import ModelLoadError
from blackmirror.schemas.metadata import ModelMetadata
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_NAME = "TRIBE v2"
MODEL_LICENSE = "CC-BY-NC-4.0"
MODEL_SOURCE = "https://github.com/facebookresearch/tribev2"

#: Fallback only. The real value is read from the checkpoint
#: (``data.neuro.offset``) by :meth:`TribeModelLoader.get_hemodynamic_offset_seconds`.
DEFAULT_HEMODYNAMIC_OFFSET_SECONDS = 5.0

#: How the offset's sign convention was established. Recorded in every run so the
#: claim is auditable rather than folklore.
#:
#: `neuralset.extractors.neuro.FmriExtractor.offset` is documented as "Seconds to
#: shift TRs forward to align delayed BOLD response", and is applied during
#: training as ``start=ta.start - self.offset`` (neuro.py). That places an fMRI
#: sample physically recorded at time T at timeline position T - offset, so a
#: training segment at timeline time t pairs stimulus content at t with BOLD
#: measured at t + offset.
#:
#: Consequence at inference: a prediction row whose segment starts at t is the
#: predicted response TO THE CONTENT AT TIME t. The model already compensated the
#: lag, so segment start time IS stimulus time and must not be shifted again.
HEMODYNAMIC_OFFSET_SOURCE = (
    "neuralset.extractors.neuro.FmriExtractor.offset, read from the checkpoint config "
    "(data.neuro.offset); sign established from its training-time application "
    "`start = ta.start - self.offset`, which aligns stimulus at t with BOLD at t+offset. "
    "Model output is therefore already stimulus-aligned."
)


class TribeModelLoader:
    """Owns the TRIBE v2 model instance and its metadata."""

    def __init__(self, settings: Settings, device: str) -> None:
        self.settings = settings
        self.device = device
        self._model: Any | None = None
        self._metadata: ModelMetadata | None = None
        self._applied_patches: tuple[str, ...] = ()

    # --- Lifecycle -------------------------------------------------------

    @property
    def applied_compat_patches(self) -> tuple[str, ...]:
        return self._applied_patches

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> Any:
        """Load TRIBE v2, or return the already-loaded instance.

        Raises:
            ModelLoadError: dependencies missing, checkpoint unavailable, or the
                upstream loader failed.
        """
        if self._model is not None:
            return self._model

        if self.settings.enable_macos_compat:
            from blackmirror.inference.compat_macos import apply_macos_patches

            self._applied_patches = apply_macos_patches()

        try:
            from tribev2.demo_utils import TribeModel
        except ImportError as exc:
            raise ModelLoadError(
                "tribev2 is not importable. Install it with:\n"
                '  uv pip install "tribev2[plotting] @ '
                'git+https://github.com/facebookresearch/tribev2.git"\n'
                "Note the model is CC-BY-NC-4.0 (non-commercial use only)."
            ) from exc

        cache_folder = Path(self.settings.model_cache_dir)
        cache_folder.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Loading %s from %s (device=%s, cache=%s)",
            MODEL_NAME,
            self.settings.model_id,
            self.device,
            cache_folder,
        )

        config_update: dict[str, Any] = {
            "remove_empty_segments": self.settings.remove_empty_segments,
            **self._extractor_device_overrides(),
            **self._text_encoder_override(),
        }

        try:
            model = TribeModel.from_pretrained(
                self.settings.model_id,
                checkpoint_name=self.settings.checkpoint_name,
                cache_folder=str(cache_folder),
                device=self.device,
                config_update=config_update,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load {MODEL_NAME} from '{self.settings.model_id}' "
                f"(checkpoint '{self.settings.checkpoint_name}', device '{self.device}'): "
                f"{type(exc).__name__}: {exc}\n"
                f"Common causes: no HuggingFace access to the gated feature-extractor "
                f"models, an incompatible torch version (TRIBE pins >=2.5.1,<2.7), or an "
                f"interrupted checkpoint download in {cache_folder}."
            ) from exc

        self._model = model
        logger.info("%s loaded on %s", MODEL_NAME, self.device)
        return model

    def _extractor_device_overrides(self) -> dict[str, str]:
        """Place the frozen feature extractors on the same device as the head.

        ``from_pretrained`` moves only the trained head
        (``model.to(device)``); each frozen extractor carries its own ``device``
        field, and the released checkpoint ships them all set to ``"cuda"``.
        Without this override a CPU run dies inside feature extraction with
        ``AssertionError: Torch not compiled with CUDA enabled``.

        These are ordinary configuration overrides passed through TRIBE's own
        ``config_update`` parameter — not a monkeypatch. Note the video and
        image extractors nest their encoder one level deeper
        (``HuggingFaceVideo.image.device``).
        """
        return {
            "data.text_feature.device": self.device,
            "data.audio_feature.device": self.device,
            "data.video_feature.image.device": self.device,
            "data.image_feature.image.device": self.device,
        }

    def _text_encoder_override(self) -> dict[str, str]:
        """Point the frozen text encoder at an accessible repo, if configured.

        The checkpoint names ``meta-llama/Llama-3.2-3B``, which is gated: without
        an accepted license and a token, any stimulus containing speech fails at
        feature extraction. This override lets an operator supply a repo they can
        actually read while keeping the default honest.
        """
        if not self.settings.text_encoder_id:
            return {}
        logger.warning(
            "Overriding the frozen text encoder with '%s'. It MUST be the same model as "
            "the checkpoint expects (Llama-3.2-3B); a different architecture yields "
            "meaningless predictions. Recorded in the model fingerprint.",
            self.settings.text_encoder_id,
        )
        return {"data.text_feature.model_name": self.settings.text_encoder_id}

    def unload(self) -> None:
        """Drop the model and free device memory."""
        if self._model is None:
            return
        self._model = None
        self._metadata = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("%s unloaded", MODEL_NAME)

    # --- Introspection ---------------------------------------------------

    def get_tr_seconds(self) -> float:
        """Repetition time in seconds, read from the loaded config.

        Never hard-coded: ``Data.TR`` is ``1 / neuro.frequency`` and is a
        property of the checkpoint, not a constant we may assume.
        """
        model = self.load()
        try:
            return float(model.data.TR)
        except (AttributeError, TypeError, ZeroDivisionError) as exc:
            raise ModelLoadError(
                f"Could not read the TR from the loaded TRIBE config: {exc}. "
                f"Temporal alignment cannot be established without it."
            ) from exc

    def get_hemodynamic_offset_seconds(self) -> float:
        """The lag the model was trained to compensate, read from its own config.

        Never hard-coded: it is a property of the checkpoint. See
        :data:`HEMODYNAMIC_OFFSET_SOURCE` for how the sign convention was
        established.
        """
        model = self.load()
        value = _dig(model, "data.neuro.offset")
        if isinstance(value, int | float):
            return float(value)
        logger.warning(
            "Could not read data.neuro.offset from the TRIBE config; falling back to "
            "%.1fs. Verify before relying on content-to-response attribution.",
            DEFAULT_HEMODYNAMIC_OFFSET_SECONDS,
        )
        return DEFAULT_HEMODYNAMIC_OFFSET_SECONDS

    def get_surface_space(self) -> str:
        """Surface template the model predicts onto, read from the config."""
        model = self.load()
        mesh = _dig(model, "data.neuro.projection.mesh")
        if isinstance(mesh, str) and mesh:
            return mesh
        logger.warning(
            "Could not read the surface mesh from the TRIBE config; "
            "the vertex count will be used to identify it."
        )
        return "unknown"

    def get_model_metadata(self) -> ModelMetadata:
        """Provenance for the loaded model, read from its own configuration."""
        if self._metadata is not None:
            return self._metadata

        model = self.load()
        extractors = self._read_feature_extractors(model)

        parameter_count: int | None = None
        torch_module = getattr(model, "_model", None)
        if torch_module is not None:
            try:
                parameter_count = sum(p.numel() for p in torch_module.parameters())
            except (AttributeError, TypeError):
                parameter_count = None

        extra: dict[str, Any] = {}
        for label, dotted in (
            ("surface_mesh", "data.neuro.projection.mesh"),
            ("neuro_frequency_hz", "data.neuro.frequency"),
            ("average_subjects", "average_subjects"),
            ("remove_empty_segments", "remove_empty_segments"),
        ):
            value = _dig(model, dotted)
            if value is not None:
                extra[label] = value
        if self.settings.text_encoder_id:
            extra["text_encoder_override"] = self.settings.text_encoder_id
        extra["hemodynamic_offset_seconds"] = self.get_hemodynamic_offset_seconds()
        extra["output_is_stimulus_aligned"] = True
        extra["applied_compat_patches"] = list(self._applied_patches)

        self._metadata = ModelMetadata(
            name=MODEL_NAME,
            backend="tribe_v2",
            model_id=self.settings.model_id,
            source=MODEL_SOURCE,
            checkpoint=self.settings.checkpoint_name,
            version=_package_version("tribev2"),
            license=MODEL_LICENSE,
            modalities=("video", "audio", "text"),
            feature_extractors=extractors,
            loaded_device=self.device,
            dtype=self.settings.dtype,
            parameter_count=parameter_count,
            subject_conditioning=(
                "average_subjects (TribeModel.from_pretrained forces average_subjects=True; "
                "predictions describe an average subject, not any individual viewer)"
            ),
            is_synthetic=False,
            extra=extra,
        )
        return self._metadata

    @staticmethod
    def _read_feature_extractors(model: Any) -> dict[str, str]:
        """Read the frozen upstream encoder ids from the model config.

        Values are reported only when actually present; nothing is guessed.
        """
        extractors: dict[str, str] = {}
        for modality in ("text", "audio", "video", "image"):
            extractor = _dig(model, f"data.{modality}_feature")
            if extractor is None:
                continue
            # Video/image extractors wrap the real encoder in a sub-config
            # (HuggingFaceVideo.image.model_name), so probe nested paths too.
            # `name` is deliberately absent: neuralset extractors inherit a
            # `name` property that returns the class name, which would shadow
            # the real model id.
            for path in ("model_name", "model_id", "image.model_name",
                         "image.model_id", "checkpoint"):
                value = _dig(extractor, path)
                if isinstance(value, str) and value:
                    extractors[modality] = value
                    break
            else:
                extractors[modality] = type(extractor).__name__
        return extractors


def _dig(obj: Any, dotted: str) -> Any:
    """Follow a dotted attribute path, returning None if any step is missing.

    Defensive by design: TRIBE is a research package whose config layout may
    change between commits. A missing field must degrade to "unknown" in the
    metadata, never crash a completed inference.
    """
    current = obj
    for part in dotted.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _package_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None
