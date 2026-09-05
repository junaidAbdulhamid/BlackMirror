"""Overlapped-speech scoring, run inside the isolated diarization venv.

WHY THIS IS A SEPARATE PROCESS
    pyannote.audio cannot coexist with TRIBE in one environment. Measured:

      pyannote.audio 4.x  needs torch>=2.8      TRIBE needs torch>=2.5.1,<2.7
      pyannote.audio 3.x  uses np.NaN           TRIBE pins numpy==2.2.6 (2.0 removed it)
      pyannote.audio 3.x  calls hf_hub_download(use_auth_token=)
                                                transformers 5.x needs
                                                huggingface_hub>=1.5, which removed it

    Shimming the 3.x path needed one monkeypatch for the hub signature and
    another for torch.load's weights_only default, and the second did not even
    take effect because Lightning captures its own torch.load reference at
    import. Patching three libraries' internals to hold a scientific pipeline
    together is worse than a subprocess, so the dependency sets are kept apart
    and this script is the only thing that crosses the boundary.

WHAT IT REPORTS
    A probability, never a verdict. `pyannote/segmentation-3.0` is a powerset
    model over {silence, s1, s2, s3, s1+s2, s1+s3, s2+s3}; the summed
    probability of the two-speaker classes is the overlap score. Hard argmax
    decoding is NOT used, because on measured material the overlap classes
    respond correctly but never win:

        speaker A alone      mean P(2-speaker) = 0.0395
        A and B together     mean P(2-speaker) = 0.0901   <- 2.3x, and correct
        speaker B alone      mean P(2-speaker) = 0.0215

    So the signal is real and directional, and a hard label would report no
    overlap at all. The number is passed through uncalibrated and the caller
    says so.

IN:  --audio <wav> --spans "start,end;start,end;..."
OUT: JSON on stdout: {"spans": [{"start","end","overlap_probability"}, ...]}
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import soundfile as sf
import torch

MODEL_ID = "pyannote/segmentation-3.0"

#: The model's native window. Shorter audio is zero-padded and the padding is
#: excluded from the scores rather than being scored as silence.
WINDOW_SECONDS = 10.0
SAMPLE_RATE = 16_000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--spans", default="")
    args = parser.parse_args()

    from pyannote.audio import Model
    from pyannote.audio.utils.powerset import Powerset

    model = Model.from_pretrained(MODEL_ID).eval()
    spec = model.specifications
    powerset = Powerset(len(spec.classes), spec.powerset_max_classes)
    multi_speaker = [
        index
        for index, row in enumerate(powerset.mapping.numpy())
        if int(row.sum()) >= 2
    ]

    samples, rate = sf.read(args.audio, dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    if rate != SAMPLE_RATE:
        raise SystemExit(f"expected {SAMPLE_RATE} Hz audio, got {rate}")

    spans = []
    for chunk in filter(None, args.spans.split(";")):
        start, end = (float(v) for v in chunk.split(","))
        spans.append((start, end))
    if not spans:
        spans = [(0.0, len(samples) / rate)]

    # Score the whole recording ONCE, then read spans off the shared time axis.
    # Scoring each span in isolation gives different — and wrong — answers: a
    # 1 s span padded to the model's 10 s window is 9 s of silence, and the
    # speaker slots are reassigned on every call. Measured on the validation
    # clip, per-span scoring inverted the ranking, putting the true overlap
    # region lowest (0.016) and a single-speaker region highest (0.095).
    times, series = _overlap_series(model, powerset, multi_speaker, samples)

    results = []
    for start, end in spans:
        window = (times >= start) & (times < end)
        value = float(series[window].mean()) if window.any() else None
        results.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "overlap_probability": round(value, 5) if value is not None else None,
            }
        )

    json.dump({"model_id": MODEL_ID, "spans": results}, sys.stdout)
    return 0


def _overlap_series(
    model: object, powerset: object, multi_speaker: list[int], samples: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame probability of two speakers at once, across the whole recording.

    Recordings longer than the model's 10 s window are covered by sliding
    windows with 50% overlap; frames covered more than once are averaged.
    """
    width = int(WINDOW_SECONDS * SAMPLE_RATE)
    hop = width // 2
    duration = samples.size / SAMPLE_RATE

    starts = list(range(0, max(1, samples.size - hop), hop)) or [0]
    accumulated: dict[int, list[float]] = {}
    frame_times: dict[int, float] = {}

    for offset in starts:
        chunk = samples[offset : offset + width]
        if chunk.size < SAMPLE_RATE // 4:
            continue
        padded = np.zeros(width, dtype=np.float32)
        padded[: chunk.size] = chunk
        with torch.inference_mode():
            logits = model(torch.from_numpy(padded).reshape(1, 1, -1))  # type: ignore[operator]
        probabilities = torch.softmax(logits, dim=-1).squeeze(0).numpy()

        count = probabilities.shape[0]
        local = np.linspace(0.0, WINDOW_SECONDS, count)
        overlap = probabilities[:, multi_speaker].sum(axis=1)
        real = chunk.size / SAMPLE_RATE
        for index in range(count):
            if local[index] > real:
                break  # zero padding, not audio
            absolute = offset / SAMPLE_RATE + local[index]
            if absolute > duration:
                break
            key = int(round(absolute * 100))
            accumulated.setdefault(key, []).append(float(overlap[index]))
            frame_times[key] = absolute

    if not accumulated:
        return np.zeros(0), np.zeros(0)
    keys = sorted(accumulated)
    times = np.array([frame_times[k] for k in keys], dtype=np.float64)
    series = np.array([float(np.mean(accumulated[k])) for k in keys], dtype=np.float64)
    return times, series


if __name__ == "__main__":
    raise SystemExit(main())
