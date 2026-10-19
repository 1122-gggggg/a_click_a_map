from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ArticulationImage:
    image_id: int
    image_name: str
    degree: int
    component_size: int
    component_sizes_after_removal: tuple[int, ...]
    separated_from_largest: int
    separated_fraction: float
    route_images_separated: int

    def to_dict(self) -> dict[str, object]:
        output = asdict(self)
        output["component_sizes_after_removal"] = list(
            self.component_sizes_after_removal
        )
        return output


@dataclass(frozen=True)
class BridgeEdge:
    image_id_a: int
    image_name_a: str
    image_id_b: int
    image_name_b: str
    shared_landmarks: int
    component_size: int
    side_sizes: tuple[int, int]
    smaller_side_size: int
    smaller_side_fraction: float
    splits_route: bool
    route_images_on_smaller_side: int

    def to_dict(self) -> dict[str, object]:
        output = asdict(self)
        output["side_sizes"] = list(self.side_sizes)
        return output


@dataclass(frozen=True)
class SpectralConnectivity:
    largest_component_nodes: int
    normalized_laplacian_lambda2: float | None
    solver: str
    solver_status: str
    weighted_degree_p10: float | None
    weighted_degree_median: float | None
    edge_support_p10: float | None
    edge_support_median: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "interpretation": (
                "lambda2 is the second-smallest eigenvalue of the weighted normalized "
                "Laplacian on the largest component. Near-zero values indicate a soft "
                "bottleneck even when no exact bridge or articulation exists."
            ),
        }


@dataclass(frozen=True)
class ThresholdSensitivityPoint:
    minimum_shared_landmarks: int
    edge_count: int
    component_count: int
    largest_component_ratio: float
    isolated_images: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class CovisibilityFragilityReport:
    minimum_shared_landmarks: int
    node_count: int
    edge_count: int
    component_count: int
    component_sizes: tuple[int, ...]
    largest_component_ratio: float
    isolated_images: tuple[dict[str, object], ...]
    articulation_images: tuple[ArticulationImage, ...]
    bridge_edges: tuple[BridgeEdge, ...]
    shared_landmark_backend: str
    shared_landmark_block_rows: int
    minimum_retained_support: int
    track_observations: int
    estimated_pair_expansions: int
    spectral_connectivity: SpectralConnectivity
    threshold_sensitivity: tuple[ThresholdSensitivityPoint, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "minimum_shared_landmarks": self.minimum_shared_landmarks,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "component_count": self.component_count,
            "component_sizes": list(self.component_sizes),
            "largest_component_ratio": self.largest_component_ratio,
            "isolated_images": list(self.isolated_images),
            "articulation_images": [
                image.to_dict() for image in self.articulation_images
            ],
            "bridge_edges": [edge.to_dict() for edge in self.bridge_edges],
            "shared_landmark_backend": self.shared_landmark_backend,
            "shared_landmark_block_rows": self.shared_landmark_block_rows,
            "minimum_retained_support": self.minimum_retained_support,
            "track_observations": self.track_observations,
            "estimated_pair_expansions": self.estimated_pair_expansions,
            "spectral_connectivity": self.spectral_connectivity.to_dict(),
            "threshold_sensitivity": [
                point.to_dict() for point in self.threshold_sensitivity
            ],
        }
