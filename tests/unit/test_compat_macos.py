"""The macOS compatibility layer rewrites a subprocess command line.

Subtle enough to be worth pinning: it must fix the CPU case, leave CUDA runs
alone, and never touch an unrelated command.
"""

from __future__ import annotations

import sys

import pytest

from blackmirror.inference.compat_macos import (
    _augment_env,
    _rewrite,
    apply_macos_patches,
)

WHISPERX_CPU = [
    "uvx", "whisperx", "/tmp/a.wav",
    "--model", "large-v3",
    "--language", "en",
    "--device", "cpu",
    "--compute_type", "float16",
    "--batch_size", "16",
]


def test_float16_is_rewritten_on_cpu() -> None:
    """float16 is unsupported by CTranslate2 on Apple Silicon CPU."""
    rewritten = _rewrite(list(WHISPERX_CPU))
    assert rewritten[rewritten.index("--compute_type") + 1] == "int8"
    # Nothing else changes.
    assert rewritten[: rewritten.index("--compute_type")] == WHISPERX_CPU[
        : WHISPERX_CPU.index("--compute_type")
    ]


def test_cuda_runs_keep_float16() -> None:
    command = list(WHISPERX_CPU)
    command[command.index("--device") + 1] = "cuda"
    assert _rewrite(command) is command


def test_unrelated_commands_are_untouched() -> None:
    command = ["ffmpeg", "-i", "in.mp4", "--compute_type", "float16"]
    assert _rewrite(command) is command


def test_already_supported_compute_type_is_untouched() -> None:
    command = list(WHISPERX_CPU)
    command[command.index("--compute_type") + 1] = "int8"
    assert _rewrite(command) is command


def test_missing_compute_type_is_untouched() -> None:
    command = ["uvx", "whisperx", "/tmp/a.wav", "--device", "cpu"]
    assert _rewrite(command) is command


def test_trailing_device_flag_does_not_crash() -> None:
    """Malformed input must not raise from inside a patch."""
    assert _rewrite(["uvx", "whisperx", "a.wav", "--device"]) is not None


def test_string_commands_are_untouched() -> None:
    assert _rewrite("uvx whisperx a.wav") == "uvx whisperx a.wav"


def test_env_sets_the_documented_torch_override() -> None:
    env = _augment_env({"PATH": "/usr/bin"})
    assert env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] == "1"
    assert env["PATH"] == "/usr/bin"


def test_patches_are_named_so_provenance_can_record_them() -> None:
    if sys.platform != "darwin":
        assert apply_macos_patches() == ()
        return
    pytest.importorskip("tribev2")
    patches = apply_macos_patches()
    assert "whisperx.compute_type:float16->int8_on_cpu" in patches
    assert "whisperx.subprocess:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1" in patches

    # Re-applying must report the SAME list. Otherwise the second stimulus in a
    # batch would record different provenance and look non-comparable with the
    # first, purely because of call ordering.
    assert apply_macos_patches() == patches
    assert apply_macos_patches() == patches
