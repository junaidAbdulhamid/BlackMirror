#!/usr/bin/env python
"""Pre-fetch the TRIBE v2 checkpoint so the first inference does not stall.

    python scripts/download_model.py

Downloads only the TRIBE head (config.yaml + best.ckpt, ~709 MB). The frozen
feature extractors (Llama-3.2-3B, V-JEPA2, W2V-BERT, DINOv2) are fetched lazily
on first use and are far larger.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from blackmirror.config.settings import get_settings
from blackmirror.utils.logging import configure_logging, get_logger

logger = get_logger("scripts.download_model")


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)

    local = Path(settings.model_id)
    if local.exists():
        logger.info("model_id is a local directory (%s); nothing to download.", local)
        return 0

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error(
            "huggingface_hub is not installed. Install TRIBE first:\n"
            '  uv pip install "tribev2[plotting] @ '
            'git+https://github.com/facebookresearch/tribev2.git"'
        )
        return 1

    for filename in ("config.yaml", settings.checkpoint_name):
        logger.info("Fetching %s from %s ...", filename, settings.model_id)
        try:
            path = hf_hub_download(settings.model_id, filename)
        except Exception as exc:
            logger.error("Failed to download %s: %s: %s", filename, type(exc).__name__, exc)
            return 1
        logger.info("  -> %s (%.0f MB)", path, Path(path).stat().st_size / 1e6)

    logger.info(
        "Checkpoint ready. Note TRIBE v2 is CC-BY-NC-4.0 (non-commercial use only)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
