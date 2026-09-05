"""Overlapped-speech scoring across an environment boundary.

WHAT IT ADDS
    Diarization assigns exactly one speaker per utterance. When two people talk
    at once the embedding is a mixture, and it is assigned confidently to
    whichever voice dominates: measured, a two-speaker mixture embedded 0.232
    from the dominant speaker and 0.808 from the other, with nothing marking it.
    This scores that risk instead of leaving it invisible.

WHY IT RUNS ELSEWHERE
    `pyannote.audio` and TRIBE cannot share an environment. pyannote 4.x needs
    torch>=2.8 while TRIBE needs <2.7; pyannote 3.x uses `np.NaN` (removed in
    NumPy 2.0, and TRIBE pins numpy==2.2.6) and calls
    `hf_hub_download(use_auth_token=)`, removed in the huggingface_hub that
    transformers 5.x requires. Bridging 3.x needed monkeypatches to two
    libraries and still failed, because Lightning captures its own `torch.load`
    reference at import. So the environments are kept apart and only JSON
    crosses the boundary. See scripts/overlap_worker.py.

WHAT IT REPORTS, AND WHAT IT DOES NOT
    A probability, never a verdict. The model's two-speaker classes respond
    correctly to real overlap but never win a hard argmax on measured material:

        speaker A alone     0.039
        A and B together    0.090      <- ground truth, 2.3x
        speaker B alone     0.022

    So a binary label would report no overlap at all, while the continuous score
    ranks correctly. The threshold below is calibrated on exactly one
    constructed example and is therefore a **flag for review, not a finding** —
    it is reported alongside the raw number so a reader can judge it.

IN:  a 16 kHz mono wav plus utterance spans
OUT: OverlapScore[] — one per span, or [] when the environment is absent
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from blackmirror.utils.logging import get_logger

logger = get_logger(__name__)

MODEL_ID = "pyannote/segmentation-3.0"

#: Interpreter for the isolated environment, relative to the repository root.
DIARIZATION_VENV = Path(".venv-diarization") / "bin" / "python"
WORKER = Path("scripts") / "overlap_worker.py"

#: Above this, an utterance is flagged for review. Calibrated against a single
#: constructed two-speaker mixture (0.090) versus its single-speaker
#: neighbours (0.039, 0.022). One example is not a calibration set, so this
#: marks utterances worth listening to rather than establishing that overlap
#: occurred.
OVERLAP_REVIEW_THRESHOLD = 0.07

#: Scoring loads a model and sweeps the recording; well beyond that is a hang.
TIMEOUT_SECONDS = 900


@dataclass(frozen=True)
class OverlapScore:
    """One utterance's overlap probability."""

    start: float
    end: float
    probability: float | None

    @property
    def flagged(self) -> bool:
        return self.probability is not None and self.probability >= OVERLAP_REVIEW_THRESHOLD


def environment_available(root: Path) -> bool:
    """Whether the isolated diarization environment is installed."""
    return (root / DIARIZATION_VENV).exists() and (root / WORKER).exists()


def score_overlap(
    wav_path: Path, spans: list[tuple[float, float]], *, root: Path
) -> tuple[list[OverlapScore], list[str]]:
    """Score each span. Returns (scores, warnings); never raises."""
    if not spans:
        return [], []
    if not environment_available(root):
        return [], [
            "Overlapped-speech scoring skipped: the isolated diarization "
            "environment is absent. Create it with "
            "`uv venv .venv-diarization && uv pip install --python "
            ".venv-diarization/bin/python pyannote.audio==4.0.7 soundfile`, and "
            "accept the pyannote licence on Hugging Face."
        ]

    interpreter = root / DIARIZATION_VENV
    encoded = ";".join(f"{start},{end}" for start, end in spans)
    try:
        completed = subprocess.run(
            [str(interpreter), str(root / WORKER), "--audio", str(wav_path),
             "--spans", encoded],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [], [f"Overlapped-speech scoring failed to run ({type(exc).__name__})."]

    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().splitlines()
        reason = detail[-1] if detail else f"exit code {completed.returncode}"
        # A gated-repo refusal is the expected failure when the licence has not
        # been accepted, and is worth naming rather than reporting as "failed".
        if "GatedRepo" in reason or "401" in reason:
            reason = (
                f"access to {MODEL_ID} is gated; accept its conditions on "
                f"Hugging Face and run `hf auth login`"
            )
        return [], [f"Overlapped-speech scoring unavailable: {reason}"]

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return [], ["Overlapped-speech scoring returned unreadable output."]

    scores = [
        OverlapScore(
            start=float(entry["start"]),
            end=float(entry["end"]),
            probability=(
                float(entry["overlap_probability"])
                if entry.get("overlap_probability") is not None
                else None
            ),
        )
        for entry in payload.get("spans", [])
    ]

    warnings: list[str] = []
    flagged = [s for s in scores if s.flagged]
    if flagged:
        warnings.append(
            f"{len(flagged)} of {len(scores)} utterance(s) scored at or above "
            f"{OVERLAP_REVIEW_THRESHOLD} for simultaneous speech and may contain more "
            f"than one voice; their speaker label would then name only the dominant "
            f"one. This threshold is calibrated on a single constructed example, so "
            f"treat it as a prompt to listen, not as a finding."
        )
    logger.info("Scored overlap for %d utterance(s); %d flagged", len(scores), len(flagged))
    return scores, warnings


def repository_root() -> Path:
    """Repository root, from this file's location."""
    return Path(__file__).resolve().parents[4]


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    audio = Path(sys.argv[1])
    result, notes = score_overlap(audio, [(0.0, 1.0)], root=repository_root())
    print(result, notes)
