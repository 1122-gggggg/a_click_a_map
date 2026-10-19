"""MapDoctor: diagnostics and regression testing for visual-localization maps."""

from importlib.metadata import PackageNotFoundError, version

from mapdoctor.adapters import (
    ColmapAdapter,
    GlomapAdapter,
    GluemapAdapter,
    get_adapter,
    register_adapter,
)
from mapdoctor.api import load_colmap, load_glomap, load_gluemap

try:
    __version__ = version("normal-map-pipeline")
except PackageNotFoundError:  # Source-tree import before installation.
    __version__ = "0.1.0"

__all__ = [
    "ColmapAdapter",
    "GlomapAdapter",
    "GluemapAdapter",
    "get_adapter",
    "register_adapter",
    "load_colmap",
    "load_glomap",
    "load_gluemap",
    "__version__",
]
