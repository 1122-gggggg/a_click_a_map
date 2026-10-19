from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

from mapdoctor.model import MapModel

from ._graph_types import SpectralConnectivity, ThresholdSensitivityPoint


def _connected_components(
    adjacency: dict[int, set[int]],
) -> tuple[list[set[int]], dict[int, int]]:
    components: list[set[int]] = []
    component_index: dict[int, int] = {}
    seen: set[int] = set()
    for start in adjacency:
        if start in seen:
            continue
        component: set[int] = set()
        queue = deque([start])
        seen.add(start)
        while queue:
            node = queue.popleft()
            component.add(node)
            component_index[node] = len(components)
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components, component_index


def _shared_landmark_counts(
    model: MapModel,
    *,
    minimum_support: int,
) -> tuple[dict[tuple[int, int], int], int, int, int]:
    """Compute exact retained image-pair support with bounded peak memory.

    Let ``B`` be the sparse binary image-landmark incidence matrix. Then
    ``B @ B.T`` contains exact shared-landmark counts. Computing the product
    in row blocks avoids materializing the complete pair matrix at once. Only
    supports needed by the requested threshold-sensitivity profile are retained;
    all reported thresholds remain exact.
    """

    if minimum_support < 1:
        raise ValueError("minimum_support must be >= 1")
    image_ids = sorted(model.images)
    image_index = {image_id: index for index, image_id in enumerate(image_ids)}
    rows: list[int] = []
    columns: list[int] = []
    observations = 0
    estimated_pair_expansions = 0
    for column, point in enumerate(model.points3d.values()):
        track_images = sorted(
            {
                element.image_id
                for element in point.track
                if element.image_id in image_index
            }
        )
        observations += len(track_images)
        estimated_pair_expansions += (
            len(track_images) * (len(track_images) - 1) // 2
        )
        rows.extend(image_index[image_id] for image_id in track_images)
        columns.extend([column] * len(track_images))

    node_count = len(image_ids)
    if not image_ids or not model.points3d or not rows:
        return {}, observations, estimated_pair_expansions, 0
    data = np.ones(len(rows), dtype=np.int32)
    incidence = csr_matrix(
        (data, (np.asarray(rows), np.asarray(columns))),
        shape=(node_count, len(model.points3d)),
        dtype=np.int32,
    )

    # Keep the dense-equivalent block footprint near eight million cells. The
    # actual sparse product is usually much smaller, but this guards corridor
    # maps containing very long tracks and many registered frames.
    block_rows = max(1, min(1024, 8_000_000 // max(node_count, 1)))
    shared: dict[tuple[int, int], int] = {}
    for start in range(0, node_count, block_rows):
        stop = min(node_count, start + block_rows)
        product = (incidence[start:stop] @ incidence.T).tocoo()
        global_rows = product.row.astype(np.int64, copy=False) + start
        supports = product.data.astype(np.int64, copy=False)
        mask = (global_rows < product.col) & (supports >= minimum_support)
        for row, column, support in zip(
            global_rows[mask],
            product.col[mask],
            supports[mask],
            strict=True,
        ):
            shared[(image_ids[int(row)], image_ids[int(column)])] = int(support)
    return shared, observations, estimated_pair_expansions, block_rows


def _graph_at_threshold(
    image_ids: Iterable[int],
    shared: dict[tuple[int, int], int],
    threshold: int,
) -> tuple[dict[int, set[int]], dict[tuple[int, int], int]]:
    adjacency = {image_id: set() for image_id in image_ids}
    edge_support = {
        edge: support for edge, support in shared.items() if support >= threshold
    }
    for image_a, image_b in edge_support:
        adjacency[image_a].add(image_b)
        adjacency[image_b].add(image_a)
    return adjacency, edge_support


def _percentile(values: Iterable[float], q: float) -> float | None:
    array = np.asarray(list(values), dtype=float)
    if len(array) == 0:
        return None
    return float(np.percentile(array, q * 100.0))


def _normalized_lambda2(
    component: set[int],
    edge_support: dict[tuple[int, int], int],
) -> tuple[float | None, str, str]:
    nodes = sorted(component)
    n = len(nodes)
    if n < 2:
        return 0.0, "not_required", "singleton_component"
    node_index = {node: index for index, node in enumerate(nodes)}
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for (image_a, image_b), support in edge_support.items():
        if image_a not in node_index or image_b not in node_index:
            continue
        a = node_index[image_a]
        b = node_index[image_b]
        value = float(support)
        rows.extend((a, b))
        columns.extend((b, a))
        data.extend((value, value))
    adjacency = csr_matrix((data, (rows, columns)), shape=(n, n), dtype=float)
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    if np.any(degree <= 0):
        return None, "not_run", "non_positive_degree_in_component"
    inverse_sqrt = 1.0 / np.sqrt(degree)
    normalized_adjacency = adjacency.multiply(inverse_sqrt[:, None]).multiply(
        inverse_sqrt[None, :]
    )
    laplacian = eye(n, dtype=float, format="csr") - normalized_adjacency
    if n <= 64:
        eigenvalues = np.linalg.eigvalsh(laplacian.toarray())
        solver = "dense_eigvalsh"
    else:
        solver = "arpack_eigsh"
        try:
            eigenvalues = eigsh(
                laplacian,
                k=2,
                which="SM",
                return_eigenvectors=False,
                tol=1e-7,
                maxiter=max(2000, n * 20),
            )
        except ArpackNoConvergence as exc:
            if exc.eigenvalues is None or len(exc.eigenvalues) < 2:
                return None, solver, "not_converged"
            eigenvalues = exc.eigenvalues
            solver = "arpack_eigsh_partial"
    eigenvalues = np.sort(
        np.maximum(np.asarray(eigenvalues, dtype=float), 0.0)
    )
    value = float(eigenvalues[1]) if len(eigenvalues) >= 2 else None
    return value, solver, "ok" if value is not None else "insufficient_eigenvalues"


def _spectral_metrics(
    adjacency: dict[int, set[int]],
    edge_support: dict[tuple[int, int], int],
    components: list[set[int]],
) -> SpectralConnectivity:
    largest = (
        max(
            components,
            key=lambda component: (len(component), -min(component)),
        )
        if components
        else set()
    )
    weighted_degree = {
        node: sum(
            edge_support[tuple(sorted((node, neighbor)))]
            for neighbor in adjacency[node]
        )
        for node in largest
    }
    component_edge_support = [
        support
        for (image_a, image_b), support in edge_support.items()
        if image_a in largest and image_b in largest
    ]
    lambda2, solver, solver_status = _normalized_lambda2(largest, edge_support)
    return SpectralConnectivity(
        largest_component_nodes=len(largest),
        normalized_laplacian_lambda2=lambda2,
        solver=solver,
        solver_status=solver_status,
        weighted_degree_p10=_percentile(weighted_degree.values(), 0.10),
        weighted_degree_median=_percentile(weighted_degree.values(), 0.50),
        edge_support_p10=_percentile(component_edge_support, 0.10),
        edge_support_median=_percentile(component_edge_support, 0.50),
    )


def _sensitivity_thresholds(base_threshold: int) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                max(1, base_threshold // 2),
                base_threshold,
                max(base_threshold + 1, base_threshold * 2),
            }
        )
    )


def _threshold_sensitivity(
    image_ids: Iterable[int],
    shared: dict[tuple[int, int], int],
    base_threshold: int,
) -> tuple[ThresholdSensitivityPoint, ...]:
    image_ids = tuple(image_ids)
    thresholds = _sensitivity_thresholds(base_threshold)
    node_count = len(image_ids)
    rows: list[ThresholdSensitivityPoint] = []
    for threshold in thresholds:
        adjacency, edge_support = _graph_at_threshold(
            image_ids, shared, threshold
        )
        components, _ = _connected_components(adjacency)
        largest = max((len(component) for component in components), default=0)
        rows.append(
            ThresholdSensitivityPoint(
                minimum_shared_landmarks=threshold,
                edge_count=len(edge_support),
                component_count=len(components),
                largest_component_ratio=(
                    largest / node_count if node_count else 0.0
                ),
                isolated_images=sum(
                    not neighbors for neighbors in adjacency.values()
                ),
            )
        )
    return tuple(rows)
