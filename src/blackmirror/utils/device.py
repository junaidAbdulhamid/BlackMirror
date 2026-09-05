"""Compute device detection and reporting.

Torch is an *optional* import here: the contract layer (schemas, artifact store,
CLI) must work in an environment with no ML stack. Everything degrades to a
report that says "torch not installed" rather than raising at import time.

Design note — no silent GPU->CPU fallback for real models. TRIBE v2 on CPU is
orders of magnitude slower than on an A100; a fallback that turns a 40-second
job into a six-hour one looks like a hang, not a fallback. When
``Settings.allow_cpu_fallback`` is False an unsatisfiable request raises.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from typing import Any

from blackmirror.config.settings import DevicePreference
from blackmirror.errors import DeviceUnavailableError
from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

# Bound at module level: inside DeviceReport the `platform` field name would
# otherwise shadow the `platform` module for later default factories.
_platform_description = platform.platform
_machine_description = platform.machine

#: Devices each backend can actually run on.
#:
#: TRIBE v2 resolves its own device with ``"cuda" if torch.cuda.is_available()
#: else "cpu"`` (tribev2/demo_utils.py::from_pretrained) and its frozen feature
#: extractors do the same. There is no MPS code path, so requesting MPS would
#: place the head on the GPU while the extractors stay on CPU.
BACKEND_SUPPORTED_DEVICES: dict[str, frozenset[str]] = {
    "tribe_v2": frozenset({"cuda", "cpu"}),
    "mock": frozenset({"cuda", "mps", "cpu"}),
}


@dataclass(frozen=True)
class DeviceReport:
    """Everything we know about the compute environment."""

    selected_device: str
    torch_available: bool
    torch_version: str | None = None
    cuda_available: bool = False
    cuda_version: str | None = None
    cudnn_version: str | None = None
    gpu_name: str | None = None
    gpu_total_memory_bytes: int | None = None
    mps_available: bool = False
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=_platform_description)
    machine: str = field(default_factory=_machine_description)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_device": self.selected_device,
            "torch_available": self.torch_available,
            "torch_version": self.torch_version,
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "cudnn_version": self.cudnn_version,
            "gpu_name": self.gpu_name,
            "gpu_total_memory_bytes": self.gpu_total_memory_bytes,
            "mps_available": self.mps_available,
            "python_version": self.python_version,
            "platform": self.platform,
            "machine": self.machine,
            "notes": list(self.notes),
        }


def _import_torch() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def probe_devices() -> DeviceReport:
    """Inspect the machine without committing to a device."""
    torch = _import_torch()
    if torch is None:
        return DeviceReport(
            selected_device="cpu",
            torch_available=False,
            notes=("PyTorch is not installed; only the mock backend can run.",),
        )

    cuda_available = bool(torch.cuda.is_available())
    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

    gpu_name: str | None = None
    gpu_memory: int | None = None
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = int(torch.cuda.get_device_properties(0).total_memory)

    notes: list[str] = []

    return DeviceReport(
        selected_device="cuda" if cuda_available else "cpu",
        torch_available=True,
        torch_version=torch.__version__,
        cuda_available=cuda_available,
        cuda_version=torch.version.cuda,
        cudnn_version=str(torch.backends.cudnn.version()) if cuda_available else None,
        gpu_name=gpu_name,
        gpu_total_memory_bytes=gpu_memory,
        mps_available=mps_available,
        notes=tuple(notes),
    )


def resolve_device(
    preference: DevicePreference,
    *,
    backend: str,
    allow_cpu_fallback: bool = True,
) -> DeviceReport:
    """Pick a device for ``backend``, honouring ``preference``.

    Raises:
        DeviceUnavailableError: the requested device is unsupported by the
            backend, absent from the machine, or CPU fallback is disabled.
    """
    report = probe_devices()
    supported = BACKEND_SUPPORTED_DEVICES.get(backend, frozenset({"cpu"}))

    if preference is DevicePreference.AUTO:
        # Best device the backend *actually* supports, in descending order.
        for candidate in ("cuda", "mps", "cpu"):
            if candidate not in supported:
                continue
            if candidate == "cuda" and not report.cuda_available:
                continue
            if candidate == "mps" and not report.mps_available:
                continue
            selected = candidate
            break
        else:
            selected = "cpu"
        return _finalize(report, selected, backend)

    requested = preference.value

    if requested not in supported:
        raise DeviceUnavailableError(
            f"Backend '{backend}' does not support device '{requested}'. "
            f"Supported: {sorted(supported)}. "
            f"For TRIBE v2 this is a property of the upstream model, not a BlackMirror "
            f"limitation — see docs/tribe_integration.md."
        )

    available = {
        "cuda": report.cuda_available,
        "mps": report.mps_available,
        "cpu": True,
    }[requested]

    if available:
        return _finalize(report, requested, backend)

    if not allow_cpu_fallback:
        raise DeviceUnavailableError(
            f"Device '{requested}' was requested but is not available on this machine "
            f"(platform={report.platform}, torch={report.torch_version}). "
            f"CPU fallback is disabled (BLACKMIRROR_ALLOW_CPU_FALLBACK=false). "
            f"Either provision the device or re-enable fallback and accept that "
            f"CPU inference is far slower."
        )

    logger.warning(
        "Device '%s' unavailable; falling back to CPU. TRIBE v2 on CPU is very slow "
        "(expect tens of minutes for a short clip).",
        requested,
    )
    return _finalize(report, "cpu", backend)


def _finalize(report: DeviceReport, selected: str, backend: str) -> DeviceReport:
    notes = list(report.notes)
    supported = BACKEND_SUPPORTED_DEVICES.get(backend, frozenset({"cpu"}))
    if report.mps_available and selected != "mps" and "mps" not in supported:
        note = (
            f"MPS is available but backend '{backend}' has no MPS code path, "
            f"so it is not used."
        )
        if note not in notes:
            notes.append(note)
    if selected == "cpu" and backend == "tribe_v2":
        notes.append(
            "Running TRIBE v2 on CPU. Feature extraction (V-JEPA2 ViT-g, Llama-3.2-3B) "
            "dominates runtime; budget tens of minutes for a ~10 s stimulus."
        )
    return DeviceReport(
        selected_device=selected,
        torch_available=report.torch_available,
        torch_version=report.torch_version,
        cuda_available=report.cuda_available,
        cuda_version=report.cuda_version,
        cudnn_version=report.cudnn_version,
        gpu_name=report.gpu_name,
        gpu_total_memory_bytes=report.gpu_total_memory_bytes,
        mps_available=report.mps_available,
        python_version=report.python_version,
        platform=report.platform,
        machine=report.machine,
        notes=tuple(notes),
    )
