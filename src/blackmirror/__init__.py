"""BlackMirror — in-silico neural content experimentation.

Phase 1 exposes one thing: a standardized cortical response prediction.

    from blackmirror import InferenceService
    result = InferenceService().predict("examples/sample.mp4")

`result` is a `PredictionResult`; see docs/data_contracts.md.
"""

__version__ = "0.1.0"

from blackmirror.config.settings import Settings, get_settings
from blackmirror.inference.service import InferenceService
from blackmirror.schemas.prediction import PredictionResult
from blackmirror.schemas.stimulus import StimulusInput

__all__ = [
    "InferenceService",
    "PredictionResult",
    "Settings",
    "StimulusInput",
    "__version__",
    "get_settings",
]
