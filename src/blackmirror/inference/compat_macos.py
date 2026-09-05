"""Documented platform compatibility patches for TRIBE v2 on macOS / Apple Silicon.

TRIBE v2 is a research model developed against CUDA Linux. Its transcription
step hard-codes a CUDA-oriented setting that raises on Apple Silicon *before*
any inference happens. Rather than forking TRIBE or editing site-packages, we
patch narrowly at runtime and record every patch in the run's provenance, so a
patched run is never mistaken for a stock upstream run.

Each patch documents: what breaks, why, and the smallest correct fix.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from typing import Any

from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

#: CTranslate2 (which faster-whisper, and therefore WhisperX, runs on) supports
#: only {float32, int8, int8_float32, int16} on Apple Silicon CPU. int8 is the
#: fastest of those and the conventional CPU choice for Whisper.
_CPU_COMPUTE_TYPE = "int8"

#: Identifiers recorded in RunProvenance.applied_compat_patches. Defined once so
#: that re-applying (an already-patched process, e.g. the second stimulus in a
#: batch) reports the SAME list. Two variants whose patch lists differed would
#: look non-comparable purely because of call ordering.
_WHISPERX_PATCH_IDS = [
    "whisperx.compute_type:float16->int8_on_cpu",
    "whisperx.subprocess:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1",
]


def apply_macos_patches() -> tuple[str, ...]:
    """Apply macOS compatibility patches. Returns identifiers of those applied.

    Safe to call on any platform; a no-op off macOS. Idempotent.
    """
    if sys.platform != "darwin":
        return ()
    applied = _patch_whisperx_compute_type()
    if applied:
        logger.warning(
            "Applied %d macOS compatibility patch(es) to TRIBE dependencies: %s. "
            "Recorded in this run's provenance.",
            len(applied),
            ", ".join(applied),
        )
    return tuple(applied)


def _patch_whisperx_compute_type() -> list[str]:
    """Force a CPU-supported compute type for TRIBE's WhisperX transcription.

    Problem
        ``tribev2.eventstransforms.ExtractWordsFromAudio._get_transcript_from_audio``
        builds and runs::

            uvx whisperx <wav> --model large-v3 --device cpu --compute_type float16 ...

        CTranslate2 on Apple Silicon CPU supports only
        {float32, int8, int8_float32, int16}, so the subprocess dies with
        ``ValueError: Requested float16 compute type, but the target device or
        backend do not support efficient float16 computation`` and TRIBE
        re-raises it as ``RuntimeError: whisperx failed``.

        Note this runs in a *separate process* that ``uvx`` provisions in its own
        ephemeral environment — patching an in-process ``whisperx`` import has no
        effect, and the ``whisperx`` package need not be installed here at all.

    Fix
        Wrap the static method so that, for the duration of that call only,
        ``subprocess.run`` rewrites ``--compute_type float16`` to ``int8`` in
        ``uvx whisperx`` command lines running on a non-CUDA device. Nothing
        else about the upstream pipeline is duplicated or altered.

    Scientific impact
        int8 quantization can change transcription slightly versus float16,
        which shifts word-level event timings and therefore the text stream the
        model sees. That is precisely why the patch is recorded in provenance:
        two runs being compared must have been patched identically.
    """
    try:
        from tribev2.eventstransforms import ExtractWordsFromAudio
    except ImportError:
        logger.debug("tribev2 not importable; skipping the WhisperX compute-type patch.")
        return []

    original = ExtractWordsFromAudio.__dict__.get("_get_transcript_from_audio")
    if original is None:
        logger.warning(
            "ExtractWordsFromAudio._get_transcript_from_audio is absent; upstream has "
            "changed and the WhisperX compute-type patch was NOT applied."
        )
        return []

    function = original.__func__ if isinstance(original, staticmethod) else original
    if getattr(function, "_blackmirror_patched", False):
        return list(_WHISPERX_PATCH_IDS)

    def patched(*args: Any, **kwargs: Any) -> Any:
        real_run = subprocess.run

        def run(command: Any, *run_args: Any, **run_kwargs: Any) -> Any:
            rewritten = _rewrite(command)
            if rewritten is not command:
                run_kwargs["env"] = _augment_env(run_kwargs.get("env"))
            return real_run(rewritten, *run_args, **run_kwargs)

        subprocess.run = run  # type: ignore[assignment]
        try:
            return function(*args, **kwargs)
        finally:
            subprocess.run = real_run

    patched._blackmirror_patched = True  # type: ignore[attr-defined]
    ExtractWordsFromAudio._get_transcript_from_audio = staticmethod(patched)
    return list(_WHISPERX_PATCH_IDS)


def _augment_env(env: dict[str, str] | None) -> dict[str, str]:
    """Let the WhisperX subprocess load its pyannote VAD checkpoint.

    Problem
        ``uvx`` provisions its own environment for WhisperX, resolving a recent
        PyTorch. Since PyTorch 2.6 ``torch.load`` defaults to
        ``weights_only=True``, and WhisperX's ``vads/pyannote.py::load_vad_model``
        does not allowlist the globals its checkpoint contains::

            _pickle.UnpicklingError: Weights only load failed ...
            Unsupported global: GLOBAL omegaconf.listconfig.ListConfig

    Fix
        Set PyTorch's documented override, ``TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1``,
        for that subprocess only. It is not set in this process and does not
        affect how the TRIBE checkpoint itself is loaded.

    Security note
        This restores pre-2.6 ``torch.load`` behaviour for the transcription
        subprocess, which unpickles the pyannote VAD checkpoint. That checkpoint
        is fetched from HuggingFace by WhisperX; the setting is scoped to that
        one call, but it is a real relaxation and is recorded in provenance.
    """
    import os

    merged = dict(env if env is not None else os.environ)
    merged["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    return merged


def _rewrite(command: Any) -> Any:
    """Rewrite a ``uvx whisperx`` command to use a CPU-supported compute type.

    Any other command is returned untouched.
    """
    if not isinstance(command, Sequence) or isinstance(command, str | bytes):
        return command
    parts = [str(part) for part in command]
    if "whisperx" not in parts:
        return command

    # Only rewrite when the run is not on CUDA; a CUDA run should keep float16.
    if "--device" in parts:
        position = parts.index("--device") + 1
        device = parts[position] if position < len(parts) else ""
        if device == "cuda":
            return command

    if "--compute_type" not in parts:
        return command
    index = parts.index("--compute_type") + 1
    if index >= len(parts) or parts[index] != "float16":
        return command

    parts[index] = _CPU_COMPUTE_TYPE
    logger.info(
        "WhisperX --compute_type float16 -> %s (float16 is unsupported by CTranslate2 "
        "on Apple Silicon CPU).",
        _CPU_COMPUTE_TYPE,
    )
    return parts
