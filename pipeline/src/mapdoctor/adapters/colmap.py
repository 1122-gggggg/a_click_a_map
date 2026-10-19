from __future__ import annotations

from pathlib import Path

from mapdoctor.adapters.base import AdapterInspection, MapAdapter
from mapdoctor.io.colmap import detect_colmap_format, load_colmap_format, resolve_sparse_model_path
from mapdoctor.model import MapModel


class ColmapAdapter(MapAdapter):
    backend = "colmap"
    display_name = "COLMAP"

    def inspect(self, path: str | Path) -> AdapterInspection:
        resolved = resolve_sparse_model_path(path)
        return AdapterInspection(
            backend=self.backend,
            resolved_path=resolved,
            model_format=detect_colmap_format(resolved),
            notes=("Native COLMAP sparse reconstruction interface.",),
        )

    def load(self, path: str | Path) -> MapModel:
        inspection = self.inspect(path)
        model = load_colmap_format(inspection.resolved_path, source=self.backend)
        model.metadata.update({"adapter": type(self).__name__, "producer": self.display_name})
        return model
