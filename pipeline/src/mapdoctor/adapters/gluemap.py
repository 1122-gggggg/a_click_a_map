from __future__ import annotations

from pathlib import Path
from typing import Any

from mapdoctor.adapters.base import AdapterInspection, MapAdapter
from mapdoctor.io.colmap import detect_colmap_format, load_colmap_format, resolve_sparse_model_path
from mapdoctor.model import MapModel


_MARKER_STAGES = {
    "twoview_result.pth": "twoview_inference",
    "star_result.pth": "star_inference",
    "pipeline_timing.pth": "pipeline_timing",
    "database_sift.db": "sift_refinement_support",
}


def _workspace_candidates(resolved_model: Path) -> tuple[Path, ...]:
    candidates = [resolved_model, resolved_model.parent, resolved_model.parent.parent]
    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return tuple(unique)


def _workspace_score(path: Path) -> int:
    score = sum((path / marker).exists() for marker in _MARKER_STAGES)
    score += sum(1 for child in path.glob("coarse*") if child.is_dir())
    return score


def inspect_gluemap_workspace(resolved_model: Path) -> dict[str, Any]:
    """Inspect official GLUEMAP run artifacts without deserializing .pth files."""
    candidates = _workspace_candidates(resolved_model)
    workspace = max(candidates, key=_workspace_score)
    score = _workspace_score(workspace)
    if score == 0:
        return {
            "mode": "sparse-only",
            "workspace": None,
            "detected_artifacts": [],
            "detected_stages": [],
            "coarse_reconstructions": [],
            "notes": [
                "No GLUEMAP run artifacts were found near the sparse model; only final reconstruction provenance is available."
            ],
        }

    artifacts = [marker for marker in _MARKER_STAGES if (workspace / marker).exists()]
    coarse = sorted(child.name for child in workspace.glob("coarse*") if child.is_dir())
    stages = [_MARKER_STAGES[marker] for marker in artifacts]
    if coarse:
        stages.append("global_mapping_coarse_output")
    if "database_sift.db" in artifacts:
        stages.append("refinement_preparation")

    return {
        "mode": "workspace-artifacts",
        "workspace": str(workspace),
        "detected_artifacts": sorted(artifacts),
        "detected_stages": sorted(set(stages)),
        "coarse_reconstructions": coarse,
        "notes": [
            "Artifact detection is read-only: MapDoctor does not torch.load GLUEMAP .pth files.",
            "Stage presence indicates retained run artifacts, not proof that every stage completed successfully.",
        ],
    }


class GluemapAdapter(MapAdapter):
    """Interface for GLUEMAP reconstructions with safe workspace provenance inspection."""

    backend = "gluemap"
    display_name = "GLUEMAP"

    def inspect(self, path: str | Path) -> AdapterInspection:
        resolved = resolve_sparse_model_path(path)
        provenance = inspect_gluemap_workspace(resolved)
        provenance_note = (
            f"GLUEMAP provenance mode: {provenance['mode']}"
            if provenance["mode"] == "sparse-only"
            else "GLUEMAP workspace artifacts detected: "
            + ", ".join(provenance["detected_artifacts"] + provenance["coarse_reconstructions"])
        )
        return AdapterInspection(
            backend=self.backend,
            resolved_path=resolved,
            model_format=detect_colmap_format(resolved),
            notes=(
                "GLUEMAP writes its reconstruction in COLMAP sparse format.",
                provenance_note,
            ),
        )

    def load(self, path: str | Path) -> MapModel:
        inspection = self.inspect(path)
        model = load_colmap_format(inspection.resolved_path, source=self.backend)
        model.metadata.update(
            {
                "adapter": type(self).__name__,
                "producer": self.display_name,
                "map_producer": self.backend,
                "gluemap_provenance": inspect_gluemap_workspace(inspection.resolved_path),
            }
        )
        return model
