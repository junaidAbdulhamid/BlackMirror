"""Device selection, including the no-silent-fallback rule."""

from __future__ import annotations

import pytest

from blackmirror.config.settings import DevicePreference
from blackmirror.errors import DeviceUnavailableError
from blackmirror.utils.device import (
    BACKEND_SUPPORTED_DEVICES,
    DeviceReport,
    resolve_device,
)


@pytest.fixture
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "blackmirror.utils.device.probe_devices",
        lambda: DeviceReport(
            selected_device="cpu", torch_available=True, torch_version="2.6.0"
        ),
    )


@pytest.fixture
def apple_silicon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "blackmirror.utils.device.probe_devices",
        lambda: DeviceReport(
            selected_device="cpu",
            torch_available=True,
            torch_version="2.6.0",
            mps_available=True,
        ),
    )


@pytest.fixture
def cuda_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "blackmirror.utils.device.probe_devices",
        lambda: DeviceReport(
            selected_device="cuda",
            torch_available=True,
            torch_version="2.6.0",
            cuda_available=True,
            cuda_version="12.4",
            gpu_name="NVIDIA A100",
            gpu_total_memory_bytes=80 * 10**9,
        ),
    )


def test_tribe_does_not_advertise_mps() -> None:
    """Upstream resolves 'cuda if available else cpu'; there is no MPS path."""
    assert "mps" not in BACKEND_SUPPORTED_DEVICES["tribe_v2"]
    assert "mps" in BACKEND_SUPPORTED_DEVICES["mock"]


def test_auto_prefers_cuda(cuda_machine: None) -> None:
    report = resolve_device(DevicePreference.AUTO, backend="tribe_v2")
    assert report.selected_device == "cuda"
    assert report.gpu_name == "NVIDIA A100"


def test_auto_avoids_mps_for_tribe(apple_silicon: None) -> None:
    """The core reason this rule exists: MPS is present but unusable by TRIBE."""
    report = resolve_device(DevicePreference.AUTO, backend="tribe_v2")
    assert report.selected_device == "cpu"
    assert any("MPS" in note for note in report.notes)


def test_auto_uses_mps_for_a_backend_that_supports_it(apple_silicon: None) -> None:
    report = resolve_device(DevicePreference.AUTO, backend="mock")
    assert report.selected_device == "mps"


def test_requesting_mps_for_tribe_raises(apple_silicon: None) -> None:
    with pytest.raises(DeviceUnavailableError, match="does not support device 'mps'"):
        resolve_device(DevicePreference.MPS, backend="tribe_v2")


def test_unavailable_cuda_falls_back_when_allowed(cpu_only: None) -> None:
    report = resolve_device(
        DevicePreference.CUDA, backend="tribe_v2", allow_cpu_fallback=True
    )
    assert report.selected_device == "cpu"


def test_unavailable_cuda_raises_when_fallback_disabled(cpu_only: None) -> None:
    """A silent GPU->CPU fallback turns a 40s job into hours and looks like a hang."""
    with pytest.raises(DeviceUnavailableError, match="CPU fallback is disabled"):
        resolve_device(DevicePreference.CUDA, backend="tribe_v2", allow_cpu_fallback=False)


def test_cpu_run_carries_a_performance_warning(cpu_only: None) -> None:
    report = resolve_device(DevicePreference.CPU, backend="tribe_v2")
    assert report.selected_device == "cpu"
    assert any("CPU" in note for note in report.notes)
