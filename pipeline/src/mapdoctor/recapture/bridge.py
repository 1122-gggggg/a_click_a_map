from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from mapdoctor.model import MapModel

from .types import PoseDirectionCell


def map_producer_from_model(model: MapModel) -> str:
    """Return the normalized producer name carried by the map adapter."""
    explicit = model.metadata.get("producer")
    if isinstance(explicit, str) and explicit.strip():
        token = explicit.strip().lower().replace(" ", "_")
        if token == "gluemap":
            return "gluemap"
        if token == "glomap":
            return "glomap"
        if token == "colmap":
            return "colmap"
    return str(model.source or "unknown").strip().lower() or "unknown"


def enrich_pose_cells_from_model(
    cells: Sequence[PoseDirectionCell],
    model: MapModel,
) -> tuple[PoseDirectionCell, ...]:
    """Attach producer provenance without fabricating pose-local metrics.

    Static map-wide values are deliberately not copied into every pose cell.
    Only producer identity is propagated so GLUEMAP-specific diagnostic
    requirements can be enabled when appropriate.
    """
    producer = map_producer_from_model(model)
    return tuple(
        cell if cell.map_producer not in {"", "unknown"} else replace(cell, map_producer=producer)
        for cell in cells
    )
