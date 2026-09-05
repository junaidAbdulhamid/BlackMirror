"""Read-only visualization API over Phase 1 artifacts.

Importing this package does not import FastAPI; `app` is resolved lazily so the
core library stays usable without web dependencies.
"""

from blackmirror.api.loader import VisualizationLoader, get_loader

__all__ = ["VisualizationLoader", "get_loader"]
