from __future__ import annotations

from pathlib import Path

from mapdoctor.adapters.base import AdapterInspection, MapAdapter
from mapdoctor.io.colmap import detect_colmap_format, load_colmap_format, resolve_sparse_model_path
from mapdoctor.model import MapModel


class GlomapAdapter(MapAdapter):
    """Interface for legacy GLOMAP outputs and COLMAP's global mapper outputs."""

    backend = "glomap"
    display_name = "GLOMAP"

    def inspect(self, path: str | Path) -> AdapterInspection:
        resolved = resolve_sparse_model_path(path)
        return AdapterInspection(
            backend=self.backend,
            resolved_path=resolved,
            model_format=detect_colmap_format(resolved),
            notes=(
                "GLOMAP emits COLMAP sparse reconstruction format.",
                "The adapter stays separate so GLOMAP-specific diagnostics can evolve independently.",
            ),
        )

    def load(self, path: str | Path) -> MapModel:
        inspection = self.inspect(path)
        model = load_colmap_format(inspection.resolved_path, source=self.backend)
        model.metadata.update({"adapter": type(self).__name__, "producer": self.display_name})
        return model
