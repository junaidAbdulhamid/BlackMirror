"""Environment verification and the Phase 1 sanity diagnostic.

Two developer-facing checks that deliberately do not belong in the inference
path: "can this machine run the model?" and "does the output vary over time in a
way that is numerically usable?".

The sanity plot is NOT visualization — Phase 2 owns the interactive cortex. It
answers one question: did the model produce a signal that changes with the
content, or a constant?
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Literal

from blackmirror.config.settings import BackendName, Settings
from blackmirror.schemas.prediction import PredictionResult
from blackmirror.utils.device import probe_devices
from blackmirror.utils.logging import get_logger, redact

logger = get_logger(__name__)

Status = Literal["PASS", "WARN", "FAIL", "SKIP"]

#: TRIBE v2 pins torch>=2.5.1,<2.7 (facebookresearch/tribev2 pyproject.toml).
TORCH_MIN = (2, 5, 1)
TORCH_MAX_EXCLUSIVE = (2, 7, 0)

#: Gated model required for TRIBE's text stream.
GATED_TEXT_MODEL = "meta-llama/Llama-3.2-3B"


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str
    remedy: str | None = None


@dataclass(frozen=True)
class EnvironmentReport:
    checks: tuple[Check, ...]

    @property
    def ready(self) -> bool:
        return all(check.status != "FAIL" for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "detail": c.detail,
                    "remedy": c.remedy,
                }
                for c in self.checks
            ],
        }


def _installed(name: str) -> str | None:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def _parse_version(raw: str) -> tuple[int, ...]:
    """Parse a leading dotted-numeric version, ignoring local/suffix parts."""
    head = raw.split("+")[0]
    parts: list[int] = []
    for chunk in head.split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def run_environment_checks(settings: Settings) -> EnvironmentReport:
    """Diagnose the environment for the configured backend."""
    checks: list[Check] = [_check_python()]
    mock_only = settings.backend is BackendName.MOCK

    checks.append(_check_core_deps())
    checks.extend(_check_torch(mock_only=mock_only))
    checks.append(_check_ffmpeg(mock_only=mock_only))
    checks.extend(_check_tribe(mock_only=mock_only))
    checks.append(_check_hf_access(mock_only=mock_only))
    checks.append(_check_checkpoint(settings, mock_only=mock_only))
    checks.append(_check_writable("Artifacts", settings.artifact_dir))
    checks.append(_check_writable("Model cache", settings.model_cache_dir))
    checks.append(_check_viz())
    return EnvironmentReport(checks=tuple(checks))


def _check_python() -> Check:
    major, minor = sys.version_info[:2]
    detail = f"{sys.version.split()[0]} ({sys.executable})"
    if (major, minor) < (3, 11):
        return Check(
            "Python",
            "FAIL",
            detail,
            "TRIBE v2 requires Python >= 3.11. Install one and recreate the venv: "
            "`brew install python@3.12 && uv venv --python python3.12 .venv`.",
        )
    if (major, minor) >= (3, 13):
        return Check(
            "Python",
            "WARN",
            detail,
            "Python 3.13+ is untested against TRIBE's pinned torch<2.7. Prefer 3.12.",
        )
    return Check("Python", "PASS", detail)


def _check_core_deps() -> Check:
    missing = [
        name
        for name in ("pydantic", "numpy", "pandas", "typer", "rich")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        return Check(
            "Core deps",
            "FAIL",
            f"missing: {', '.join(missing)}",
            "Run `uv pip install -e '.[dev]'` inside the virtualenv.",
        )
    return Check(
        "Core deps", "PASS", f"pydantic {_installed('pydantic')}, numpy {_installed('numpy')}"
    )


def _check_torch(*, mock_only: bool) -> list[Check]:
    raw = _installed("torch")
    if raw is None:
        status: Status = "SKIP" if mock_only else "FAIL"
        return [
            Check(
                "PyTorch",
                status,
                "not installed",
                None
                if mock_only
                else "Installed as a TRIBE dependency: "
                '`uv pip install "tribev2[plotting] @ '
                'git+https://github.com/facebookresearch/tribev2.git"`.',
            ),
            Check("Compute device", "SKIP", "requires PyTorch"),
        ]

    parsed = _parse_version(raw)
    if not (TORCH_MIN <= parsed < TORCH_MAX_EXCLUSIVE):
        torch_check = Check(
            "PyTorch",
            "SKIP" if mock_only else "FAIL",
            f"{raw} (TRIBE v2 pins >=2.5.1,<2.7)",
            "Pin a compatible build: `uv pip install 'torch>=2.6,<2.7'`. "
            "TRIBE v2 was released against that range; newer torch changes "
            "`torch.load` and `weights_only` semantics it depends on.",
        )
    else:
        torch_check = Check("PyTorch", "PASS", raw)

    report = probe_devices()
    if report.cuda_available:
        memory = (
            f", {report.gpu_total_memory_bytes / 1e9:.1f} GB"
            if report.gpu_total_memory_bytes
            else ""
        )
        device_check = Check(
            "Compute device",
            "PASS",
            f"CUDA {report.cuda_version} — {report.gpu_name}{memory}",
        )
    elif report.mps_available:
        device_check = Check(
            "Compute device",
            "WARN",
            "MPS available but unused; TRIBE v2 has no MPS path, so inference runs on CPU",
            "Expect tens of minutes per ~10 s of content. For a fast run use a CUDA "
            "machine or Meta's Colab demo notebook.",
        )
    else:
        device_check = Check(
            "Compute device",
            "WARN",
            "CPU only",
            "CPU inference is functional but very slow for TRIBE v2.",
        )
    return [torch_check, device_check]


def _check_ffmpeg(*, mock_only: bool) -> Check:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        return Check("ffmpeg", "PASS", f"{ffmpeg}")
    if mock_only:
        return Check("ffmpeg", "WARN", "not found; stimulus duration will be unknown",
                     "`brew install ffmpeg`")
    return Check(
        "ffmpeg",
        "FAIL",
        "ffmpeg/ffprobe not on PATH",
        "Required for audio extraction and duration probing: `brew install ffmpeg`.",
    )


def _check_tribe(*, mock_only: bool) -> list[Check]:
    if mock_only:
        return [Check("TRIBE v2", "SKIP", "mock backend selected")]

    checks: list[Check] = []
    tribe_version = _installed("tribev2")
    if importlib.util.find_spec("tribev2") is None:
        checks.append(
            Check(
                "TRIBE v2",
                "FAIL",
                "tribev2 is not importable",
                'Install it: `uv pip install "tribev2[plotting] @ '
                'git+https://github.com/facebookresearch/tribev2.git"`. '
                "Note the model is CC-BY-NC-4.0 (non-commercial).",
            )
        )
    else:
        checks.append(Check("TRIBE v2", "PASS", f"tribev2 {tribe_version or 'installed'}"))

    # ExtractWordsFromAudio does not import whisperx: it shells out to
    # `uvx whisperx ...`, which provisions its own ephemeral environment. So the
    # requirement is the `uvx` launcher on PATH, not an importable package.
    uvx = shutil.which("uvx")
    if uvx is None:
        checks.append(
            Check(
                "WhisperX (uvx)",
                "FAIL",
                "`uvx` not on PATH; TRIBE runs word-level transcription via "
                "`uvx whisperx`",
                "Install uv: `brew install uv`. The whisperx package itself does NOT "
                "need to be installed in this virtualenv.",
            )
        )
    else:
        checks.append(
            Check("WhisperX (uvx)", "PASS", f"{uvx} (whisperx runs in its own uv environment)")
        )

    if sys.platform == "darwin":
        checks.append(
            Check(
                "Apple Silicon compat",
                "WARN",
                "TRIBE invokes `uvx whisperx --compute_type float16`, which CTranslate2 "
                "cannot run on Apple Silicon CPU",
                "BlackMirror rewrites it to int8 at runtime when "
                "BLACKMIRROR_ENABLE_MACOS_COMPAT=true (default). Every applied patch is "
                "recorded in the run's provenance.",
            )
        )
    return checks


def _check_hf_access(*, mock_only: bool) -> Check:
    """TRIBE's text stream needs the gated Llama-3.2-3B. This is a hard gate."""
    if mock_only:
        return Check("HuggingFace access", "SKIP", "mock backend selected")

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or _stored_hf_token()
    )
    if not token:
        return Check(
            "HuggingFace access",
            "FAIL",
            f"no token found; {GATED_TEXT_MODEL} is gated",
            f"1) Accept the license at https://huggingface.co/{GATED_TEXT_MODEL}  "
            "2) Create a read token at https://huggingface.co/settings/tokens  "
            "3) Set HF_TOKEN in .env (or run `huggingface-cli login`).",
        )

    if importlib.util.find_spec("huggingface_hub") is None:
        return Check(
            "HuggingFace access",
            "WARN",
            f"token present ({redact(token)}) but huggingface_hub is not installed, "
            "so gated access could not be confirmed",
            "Installed with tribev2; or `uv pip install huggingface_hub`.",
        )

    try:
        from huggingface_hub import HfApi

        HfApi().model_info(GATED_TEXT_MODEL, token=token)
    except Exception as exc:
        return Check(
            "HuggingFace access",
            "FAIL",
            f"token present ({redact(token)}) but {GATED_TEXT_MODEL} is not accessible "
            f"({type(exc).__name__})",
            f"Accept the license at https://huggingface.co/{GATED_TEXT_MODEL} with the same "
            "account the token belongs to. Approval is not always instant.",
        )
    return Check("HuggingFace access", "PASS", f"{GATED_TEXT_MODEL} accessible ({redact(token)})")


def _stored_hf_token() -> str | None:
    path = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "token"
    try:
        return path.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _check_checkpoint(settings: Settings, *, mock_only: bool) -> Check:
    if mock_only:
        return Check("Checkpoint", "SKIP", "mock backend selected")

    local = Path(settings.model_id)
    if local.exists():
        ckpt = local / settings.checkpoint_name
        if ckpt.exists():
            return Check("Checkpoint", "PASS", f"{ckpt} ({ckpt.stat().st_size / 1e6:.0f} MB)")
        return Check(
            "Checkpoint",
            "FAIL",
            f"{local} exists but {settings.checkpoint_name} is missing",
            f"Place {settings.checkpoint_name} and config.yaml in {local}.",
        )

    hub_cache = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    ) / "hub" / f"models--{settings.model_id.replace('/', '--')}"
    if hub_cache.exists():
        return Check("Checkpoint", "PASS", f"cached at {hub_cache}")
    return Check(
        "Checkpoint",
        "WARN",
        f"{settings.model_id} not in the local HF cache",
        "It downloads on first use (~709 MB). Pre-fetch with "
        "`python scripts/download_model.py`.",
    )


def _check_writable(name: str, path: Path) -> Check:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".blackmirror_write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return Check(name, "FAIL", f"{path} is not writable: {exc}", f"`mkdir -p {path}`")
    free_gb = shutil.disk_usage(path).free / 1e9
    status: Status = "WARN" if free_gb < 20 else "PASS"
    return Check(name, status, f"{path} (writable, {free_gb:.0f} GB free)")


def _check_viz() -> Check:
    missing = [
        name for name in ("matplotlib", "nilearn") if importlib.util.find_spec(name) is None
    ]
    if missing:
        return Check(
            "Diagnostics/mesh",
            "WARN",
            f"missing: {', '.join(missing)} — sanity plots and mesh export unavailable",
            "`uv pip install -e '.[viz]'`",
        )
    return Check(
        "Diagnostics/mesh",
        "PASS",
        f"matplotlib {_installed('matplotlib')}, nilearn {_installed('nilearn')}",
    )


# ---------------------------------------------------------------------------
# Sanity diagnostic
# ---------------------------------------------------------------------------


def write_sanity_plot(result: PredictionResult, settings: Settings) -> Path | None:
    """Plot the global mean predicted response over stimulus time.

    Purpose: confirm the prediction *changes over time and is numerically
    usable*. This is not a scientific result and not a brain visualization — the
    global mean across 20k vertices deliberately discards all spatial structure.

    Returns the plot path, or None if matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.info(
            "matplotlib not installed; skipping the sanity plot "
            "(install with `uv pip install -e '.[viz]'`)"
        )
        return None

    run_dir = result.artifacts.run_dir
    predictions = np.load(run_dir / result.prediction.artifact_path, allow_pickle=False)

    times = np.arange(predictions.shape[0], dtype=float) * result.temporal.tr_seconds
    x_label = f"prediction row index x TR ({result.temporal.tr_seconds:.2f} s)"
    if result.temporal.arrays_artifact_path is not None:
        with np.load(run_dir / result.temporal.arrays_artifact_path) as temporal:
            if "stimulus_time_seconds" in temporal:
                times = temporal["stimulus_time_seconds"]
                x_label = (
                    "approximate stimulus time (s)  "
                    f"[segment start - {result.temporal.hemodynamic_offset_seconds:.0f}s "
                    "hemodynamic offset]"
                )

    with np.errstate(invalid="ignore"):
        global_mean = np.nanmean(predictions, axis=1)
        spatial_std = np.nanstd(predictions, axis=1)

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    top.plot(times, global_mean, linewidth=1.2, color="#2b6cb0")
    top.set_ylabel("global mean")
    top.set_title(
        f"Predicted cortical response — sanity diagnostic\n"
        f"{result.stimulus.filename}  |  {result.model.name}  |  "
        f"{predictions.shape[0]} x {predictions.shape[1]} ({result.cortical.surface_space})",
        fontsize=10,
    )
    top.grid(alpha=0.25)

    bottom.plot(times, spatial_std, linewidth=1.2, color="#805ad5")
    bottom.set_ylabel("spatial std")
    bottom.set_xlabel(x_label)
    bottom.grid(alpha=0.25)

    caption = (
        "Mean and spread ACROSS all cortical vertices at each time point. Spatial structure "
        "is discarded by design; this only verifies the output varies over time."
    )
    if result.model.is_synthetic:
        caption = "SYNTHETIC DATA — not a brain prediction. " + caption
    figure.text(0.01, 0.005, caption, fontsize=7, color="#555555", wrap=True)

    if not result.temporal.timeline_is_contiguous:
        top.text(
            0.99,
            0.95,
            "non-contiguous timeline: event-free segments were dropped",
            transform=top.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="#c53030",
        )

    figure.tight_layout(rect=(0, 0.03, 1, 1))

    output_dir = run_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "global_mean_timecourse.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    logger.info("Wrote sanity diagnostic: %s", path)
    return path
