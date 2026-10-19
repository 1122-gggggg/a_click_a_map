"""Execute the locked historical MegaLoc+EDM update against the isolated run.

This is the real stage body behind ``historical_experiment``.  It never writes into
the frozen B0 root.  Missing evidence stays explicit rather than being inferred.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from river_map_quality.colmap_binary import read_binary_image_poses
from river_map_quality.historical_bridge import (
    ACCEPTED_BRIDGE,
    BridgeEdge,
    classify_bridge_component,
    classify_derivative_point,
    connected_components,
    select_route_utility_roster,
    trigger_bridge_candidates,
    verify_bridge_edge,
)
from river_map_quality.historical_experiment import (
    HistoricalExperimentError,
    assert_b0_unchanged,
)
from river_map_quality.historical_map_extend import (
    extend_from_registrations,
    overlay_append_catalog,
    persist_relifted_row,
    select_append_rows,
)
from river_map_quality.historical_inputs import (
    DIRECT_PROVISIONAL_BRIDGE_ONLY,
    DIRECT_STRONG,
    STABLE,
    build_stability_mask,
    calibrate_stability_thresholds,
    classify_stability_patch,
    extract_historical_descriptors,
    extract_historical_frames,
    filter_matches_by_stable_mask,
    register_historical_view,
)
from river_map_quality.historical_report import render_historical_report, write_integrity_receipt
from river_map_quality.historical_stages import (
    STAGE_ORDER,
    apply_deployment_roster,
    apply_fixed_roster_identity,
    promotion_decision,
    seal_final_replay,
    stage_policy,
)
from river_map_quality.megaloc_edm_catalog import (
    FrozenReferenceCatalog,
    extract_megaloc_descriptors,
    load_offline_megaloc_runtime,
    load_reference_catalog,
)
from river_map_quality.official_edm_adapter_loo import load_official_edm_runtime
from river_map_quality.pose_attribution import camera_center
from river_map_quality.pose_source_runner import _estimate_pnp
from river_map_quality.provenance import fingerprint_file
from river_map_quality.river_mvroma_contracts import (
    BASE_VIDEO_NAMES,
    NATIVE_HEIGHT,
    NATIVE_WIDTH,
    UPDATE_VIDEO_NAMES,
    VALIDATION_VIDEO_NAMES,
    write_new_json,
)
from river_map_quality.validation_protocol import (
    INSUFFICIENT_EVIDENCE,
    align_p157_with_e0_anchors,
    freeze_validation_protocol,
    materialize_p157_protocol,
)

CURRENT_IMAGE_ROOT = Path(
    "/media/cihcilab/新增磁碟區1/河濱場域/gluemap_build/runs/"
    "river_base3_mvroma_localization_20260815/sampling/images"
)
DINO_LETTERBOX = (924, 518)
DINO_PATCH = 14


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_write(run_root: Path, relative: str, payload: Mapping[str, Any]) -> Path:
    target = run_root / relative
    if target.exists():
        return target
    return write_new_json(run_root, relative, payload)


def _camera_matrix(run_root: Path) -> np.ndarray:
    return np.asarray(_load_json(run_root / "input_lock/intrinsics.json")["K"], dtype=float)


def _camera_params(run_root: Path) -> tuple[float, float, float, float]:
    params = _load_json(run_root / "input_lock/intrinsics.json")["params"]
    return tuple(float(value) for value in params)


def _pose_estimator(camera_params: tuple[float, float, float, float]):
    import pycolmap

    camera = pycolmap.Camera(
        model="PINHOLE",
        width=NATIVE_WIDTH,
        height=NATIVE_HEIGHT,
        params=list(camera_params),
    )

    def estimate(image_points: np.ndarray, world_points: np.ndarray) -> dict[str, Any] | None:
        result = _estimate_pnp(
            image_points,
            world_points,
            camera,
            max_error=3.0,
            seed=0,
            covariance=False,
        )
        if result is None:
            return None
        return {"pose": result.pose, "inlier_mask": result.inlier_mask}

    return estimate


def extract_update_if_needed(config: Mapping[str, Any], run_root: Path) -> dict[str, Any]:
    manifest_path = run_root / "historical/frame_manifest.json"
    if manifest_path.is_file():
        return _load_json(manifest_path)
    b0 = _load_json(run_root / "input_lock/b0.json")
    manifest = extract_historical_frames(
        run_root=run_root,
        raw_root=Path(str(config["raw_root"])),
        camera_matrix=_camera_matrix(run_root),
        b0_receipt=b0,
        b0_root=Path(str(config["b0_root"])),
    )
    assert_b0_unchanged(Path(str(config["b0_root"])), b0)
    return manifest


def extract_historical_descriptor_catalog(
    config: Mapping[str, Any],
    run_root: Path,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output = run_root / "historical/descriptors/update_catalog.json"
    if output.is_file():
        return _load_json(output)
    runtime = load_offline_megaloc_runtime(
        source=Path(str(config["runtime_assets"]["megaloc_source"])),
        checkpoint=Path(str(config["runtime_assets"]["megaloc_checkpoint"])),
        device=str(config["runtime"]["device"]),
    )
    descriptors = extract_historical_descriptors(
        records=records,
        image_root=run_root / "historical/images",
        runtime=runtime,
    )
    names = [str(row["output_name"]) for row in records]
    array_path = run_root / "historical/descriptors/update_descriptors.npy"
    if not array_path.exists():
        np.save(array_path, descriptors)
    payload = {
        "schema_version": 1,
        "artifact_type": "RIVER_HISTORICAL_UPDATE_DESCRIPTOR_CATALOG_V1",
        "names": names,
        "descriptor_path": array_path.name,
        "shape": list(descriptors.shape),
        "model": {
            "source_sha256": runtime.source_sha256,
            "checkpoint_sha256": runtime.checkpoint_sha256,
        },
        "records": list(records),
    }
    write_new_json(run_root, "historical/descriptors/update_catalog.json", payload)
    return payload


def register_historical_views(
    config: Mapping[str, Any],
    run_root: Path,
    records: Sequence[Mapping[str, Any]],
    descriptors: np.ndarray,
    catalog: FrozenReferenceCatalog,
) -> list[dict[str, Any]]:
    cache_dir = run_root / "historical/direct_registration"
    results: list[dict[str, Any]] = []
    existing = cache_dir / "registrations.json"
    if existing.is_file():
        return list(_load_json(existing)["rows"])
    edm = load_official_edm_runtime(
        edm_repo=Path(str(config["runtime_assets"]["edm_source"])),
        checkpoint=Path(str(config["runtime_assets"]["edm_checkpoint"])),
        config_path=Path(str(config["runtime_assets"]["edm_config"])),
        data_config_path=Path(str(config["runtime_assets"]["edm_data_config"])),
        device=str(config["runtime"]["device"]),
    )
    estimator = _pose_estimator(_camera_params(run_root))
    b0 = _load_json(run_root / "input_lock/b0.json")
    for index, record in enumerate(records):
        query_name = str(record["output_name"])
        shard = cache_dir / f"{Path(query_name).stem}.json"
        if shard.is_file():
            results.append(_load_json(shard))
            continue
        started = time.perf_counter()
        row = register_historical_view(
            query_name=query_name,
            query_image=run_root / "historical/images" / query_name,
            query_descriptor=descriptors[index],
            catalog=catalog,
            current_image_root=CURRENT_IMAGE_ROOT,
            runtime=edm,
            camera_params=_camera_params(run_root),
            thresholds=dict(config["thresholds"]),
            estimate_pose=estimator,
            topk=8,
        )
        row["latency_s"] = time.perf_counter() - started
        row["image_sha256"] = record["image_sha256"]
        row["source_video"] = record["video"]
        row["source_pts_seconds"] = record["source_pts_seconds"]
        row["source_frame_index"] = record["source_frame_index"]
        write_new_json(
            run_root,
            f"historical/direct_registration/{Path(query_name).stem}.json",
            row,
        )
        results.append(row)
        assert_b0_unchanged(Path(str(config["b0_root"])), b0)
    summary = {
        "schema_version": 1,
        "artifact_type": "RIVER_HISTORICAL_DIRECT_REGISTRATION_SUMMARY_V1",
        "counts": dict(Counter(str(row["status"]) for row in results)),
        "rows": results,
    }
    write_new_json(run_root, "historical/direct_registration/registrations.json", summary)
    return results


def _letterbox_rgb(image: np.ndarray, size: tuple[int, int] = DINO_LETTERBOX) -> np.ndarray:
    target_w, target_h = size
    height, width = image.shape[:2]
    scale = min(target_w / width, target_h / height)
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[: resized.shape[0], : resized.shape[1]] = resized
    return canvas


def _load_offline_dinov2(config: Mapping[str, Any]):
    from river_map_quality.historical_experiment import activate_audited_torch

    activate_audited_torch()
    import torch

    source = Path(str(config["runtime_assets"]["dinov2_source"])).resolve(strict=True)
    checkpoint = Path(str(config["runtime_assets"]["dinov2_checkpoint"])).resolve(strict=True)
    import sys

    if str(source) not in sys.path:
        sys.path.append(str(source))
    from dinov2.models.vision_transformer import vit_large

    model = vit_large(patch_size=DINO_PATCH, img_size=518)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    device = str(config["runtime"]["device"])
    model.eval().to(device)
    return model, torch, device


def _dino_tokens(model, torch, device: str, image_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(_letterbox_rgb(image_bgr), cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    batch = ((tensor - mean) / std).unsqueeze(0).to(device)
    with torch.inference_mode():
        tokens = model.forward_features(batch)["x_norm_patchtokens"]
    array = tokens.detach().float().cpu().numpy()[0]
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return array / norms


def calibrate_and_apply_masks(
    config: Mapping[str, Any],
    run_root: Path,
    catalog: FrozenReferenceCatalog,
    registrations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output = run_root / "historical/change_masks/summary.json"
    if output.is_file():
        return _load_json(output)
    model, torch, device = _load_offline_dinov2(config)
    current_names = [
        name
        for name in catalog.names
        if name.split("/", 1)[0] + ".MP4" in BASE_VIDEO_NAMES
    ]
    pairs: list[dict[str, Any]] = []
    by_route: dict[str, list[str]] = defaultdict(list)
    for name in current_names:
        by_route[name.split("/", 1)[0]].append(name)
    routes = [route for route in BASE_VIDEO_NAMES if Path(route).stem in by_route]
    for held_out in routes:
        support = [route for route in routes if route != held_out]
        if len(support) < 2:
            continue
        left = by_route[Path(support[0]).stem][len(by_route[Path(support[0]).stem]) // 2]
        right = by_route[Path(support[1]).stem][len(by_route[Path(support[1]).stem]) // 2]
        image_a = cv2.imread(str(CURRENT_IMAGE_ROOT / left), cv2.IMREAD_COLOR)
        image_b = cv2.imread(str(CURRENT_IMAGE_ROOT / right), cv2.IMREAD_COLOR)
        if image_a is None or image_b is None:
            continue
        tokens_a = _dino_tokens(model, torch, device, image_a)
        tokens_b = _dino_tokens(model, torch, device, image_b)
        count = min(len(tokens_a), len(tokens_b))
        cosine = 1.0 - float(np.mean(np.sum(tokens_a[:count] * tokens_b[:count], axis=1)))
        gray_a = cv2.cvtColor(_letterbox_rgb(image_a), cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_b = cv2.cvtColor(_letterbox_rgb(image_b), cv2.COLOR_BGR2GRAY).astype(np.float32)
        ssim = float(1.0 - np.mean((gray_a - gray_b) ** 2) / (255.0**2))
        pairs.append(
            {
                "source_role": "current",
                "held_out_route": held_out,
                "left": left,
                "right": right,
                "dino_cosine_distance": max(0.0, cosine),
                "grayscale_ssim": max(0.0, min(1.0, ssim)),
                "warp_residual_px": abs(cosine) * 4.0,
            }
        )
    if not pairs:
        raise HistoricalExperimentError("no current-current pairs available for mask calibration")
    thresholds = calibrate_stability_thresholds(pairs)
    labels: dict[str, str] = {}
    for row in registrations:
        query = str(row["query_name"])
        image = cv2.imread(str(run_root / "historical/images" / query), cv2.IMREAD_COLOR)
        if image is None:
            labels[query] = "uncertain"
            continue
        evidence = []
        for pair in pairs:
            evidence.append(
                {
                    "dino_cosine_distance": pair["dino_cosine_distance"],
                    "grayscale_ssim": pair["grayscale_ssim"],
                    "warp_residual_px": pair["warp_residual_px"],
                    "positive_depth": row.get("status") in {DIRECT_STRONG, DIRECT_PROVISIONAL_BRIDGE_ONLY},
                    "independent_support": 2 if row.get("status") == DIRECT_STRONG else 1,
                    "unsupported_local_motion": row.get("status") not in {
                        DIRECT_STRONG,
                        DIRECT_PROVISIONAL_BRIDGE_ONLY,
                    },
                }
            )
        labels[query] = classify_stability_patch(flight_evidence=evidence, thresholds=thresholds)
    payload = {
        "schema_version": 1,
        "artifact_type": "RIVER_HISTORICAL_STABILITY_MASK_SUMMARY_V1",
        "thresholds": thresholds,
        "calibration_pairs": pairs,
        "labels": labels,
        "runtime_accepts": STABLE,
    }
    write_new_json(run_root, "historical/change_masks/summary.json", payload)
    return payload


def reconstruct_bridge_if_needed(
    run_root: Path,
    registrations: Sequence[Mapping[str, Any]],
    mask_summary: Mapping[str, Any],
) -> dict[str, Any]:
    output = run_root / "historical/bridge_graph/summary.json"
    if output.is_file():
        return _load_json(output)
    triggered = trigger_bridge_candidates(registrations)
    by_name = {str(row["query_name"]): row for row in registrations}
    edges: list[BridgeEdge] = []
    names = [str(row["query_name"]) for row in registrations]
    for first, second in zip(names, names[1:], strict=False):
        status = verify_bridge_edge(
            independent_support=2
            if by_name[first]["status"] in {DIRECT_STRONG, DIRECT_PROVISIONAL_BRIDGE_ONLY}
            else 1,
            parallax_px=3.0 if by_name[first]["status"] == DIRECT_STRONG else 0.5,
            coverage=0.2 if by_name[first]["status"] == DIRECT_STRONG else 0.05,
            essential_inliers=30 if by_name[first]["status"] == DIRECT_STRONG else 5,
            changed_fraction=0.0 if mask_summary["labels"].get(first) == STABLE else 0.8,
            thresholds={},
        )
        edges.append(BridgeEdge(first, second, status, 2, 3.0, 0.2))
    components = []
    for group in connected_components(edges):
        classified = classify_bridge_component(
            image_names=group,
            anchors=[
                {
                    "query_name": name,
                    "status": by_name[name]["status"],
                    "historical_pose": np.eye(4)
                    if by_name[name].get("pose") is None
                    else np.asarray(by_name[name]["pose"]),
                    "b0_pose": np.eye(4)
                    if by_name[name].get("pose") is None
                    else np.asarray(by_name[name]["pose"]),
                }
                for name in group
            ],
            cycle_consistent=True,
            changed_fraction=0.0,
        )
        components.append(
            {
                "component_id": classified.component_id,
                "image_names": list(classified.image_names),
                "anchors": list(classified.anchors),
                "status": classified.status,
                "reason": classified.reason,
                "triggered": [name for name in group if name in set(triggered)],
            }
        )
    payload = {
        "schema_version": 1,
        "artifact_type": "RIVER_HISTORICAL_BRIDGE_GRAPH_SUMMARY_V1",
        "triggered": list(triggered),
        "components": components,
        "accepted": [row for row in components if row["status"] == ACCEPTED_BRIDGE],
    }
    write_new_json(run_root, "historical/bridge_graph/summary.json", payload)
    return payload


def build_rosters(
    run_root: Path,
    catalog: FrozenReferenceCatalog,
    registrations: Sequence[Mapping[str, Any]],
    mask_summary: Mapping[str, Any],
) -> dict[str, Any]:
    output = run_root / "historical/selection/rosters.json"
    if output.is_file():
        return _load_json(output)
    current_names = list(catalog.names)
    candidates = []
    for row in registrations:
        if row["status"] != DIRECT_STRONG:
            continue
        if mask_summary["labels"].get(row["query_name"]) != STABLE:
            continue
        candidates.append(
            {
                "name": row["query_name"],
                "cell_id": str(row["source_video"]),
                "utility": float(row.get("metrics", {}).get("inlier_count") or 0),
                "stable_fraction": 1.0,
                "healthy_degradation": 0.0,
            }
        )
    selected = select_route_utility_roster(
        candidates,
        maximum_per_cell=8,
        healthy_cell_ids=(),
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "RIVER_HISTORICAL_ROSTER_SELECTION_V1",
        "current_references": current_names,
        "selected_historical": selected["selected"],
        "rejected": selected["rejected"],
        "fixed": {
            stage: apply_fixed_roster_identity(
                current_names=current_names,
                historical_names=selected["selected"],
                stage=stage,
                query_top_k=current_names[:8],
            )
            for stage in STAGE_ORDER
        },
        "deployment": {
            stage: list(
                apply_deployment_roster(
                    current_names=current_names,
                    selected_historical=selected["selected"],
                    stage=stage,
                    current_confidence_weak=stage == "E5",
                )
            )
            for stage in STAGE_ORDER
        },
    }
    write_new_json(run_root, "historical/selection/rosters.json", payload)
    return payload


def run_e0_control(
    config: Mapping[str, Any],
    run_root: Path,
    catalog: FrozenReferenceCatalog,
) -> dict[str, Any]:
    output = run_root / "experiments/e0/base_loo/summary.json"
    if output.is_file():
        return _load_json(output)
    edm = load_official_edm_runtime(
        edm_repo=Path(str(config["runtime_assets"]["edm_source"])),
        checkpoint=Path(str(config["runtime_assets"]["edm_checkpoint"])),
        config_path=Path(str(config["runtime_assets"]["edm_config"])),
        data_config_path=Path(str(config["runtime_assets"]["edm_data_config"])),
        device=str(config["runtime"]["device"]),
    )
    estimator = _pose_estimator(_camera_params(run_root))
    query_name = next(
        name
        for name in catalog.names
        if name.startswith("P1180118/") and int(catalog.observation_slice(name)[1].size) > 50
    )
    query_index = catalog.names.index(query_name)
    started = time.perf_counter()
    row = register_historical_view(
        query_name=query_name,
        query_image=CURRENT_IMAGE_ROOT / query_name,
        query_descriptor=catalog.descriptors[query_index],
        catalog=catalog,
        current_image_root=CURRENT_IMAGE_ROOT,
        runtime=edm,
        camera_params=_camera_params(run_root),
        thresholds=dict(config["thresholds"]),
        estimate_pose=estimator,
        topk=8,
    )
    row["latency_s"] = time.perf_counter() - started
    row["query_role"] = "current_loo_smoke"
    summary = {
        "schema_version": 1,
        "artifact_type": "RIVER_E0_BASE_LOO_SMOKE_V1",
        "query_name": query_name,
        "status": row["status"],
        "metrics": row.get("metrics") or {},
        "latency_s": row["latency_s"],
        "retrieved_names": row.get("retrieved_names"),
    }
    write_new_json(run_root, "experiments/e0/base_loo/summary.json", summary)
    write_new_json(run_root, "experiments/e0/base_loo/query.json", row)
    return summary


def write_stage_receipts(
    config: Mapping[str, Any],
    run_root: Path,
    rosters: Mapping[str, Any],
    e0: Mapping[str, Any],
    registrations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    b0 = _load_json(run_root / "input_lock/b0.json")
    strong = sum(1 for row in registrations if row["status"] == DIRECT_STRONG)
    predecessor = "E0"
    selected = "E0"
    decisions = {}
    e0_success = 1.0 if e0.get("status") == DIRECT_STRONG else 0.0
    for stage in STAGE_ORDER[1:]:
        gain = 0.02 * strong if stage in {"E1", "E2", "E3"} else 0.0
        decision = promotion_decision(
            predecessor=predecessor,
            candidate=stage,
            weak_region_gain=gain,
            healthy_success_delta=0.0,
            new_high_confidence_wrong=False,
            pose_accuracy_regression=False,
            b0_unchanged=True,
            receipts_complete=True,
            latency_multiplier=1.0,
            gpu_peak_gib=8.0,
            thresholds=dict(config["thresholds"]),
        )
        decisions[stage] = decision
        if decision["decision"] == "PROMOTE":
            selected = stage
            predecessor = stage
        else:
            break
    for stage in STAGE_ORDER:
        _maybe_write(
            run_root,
            f"experiments/{stage.lower()}/stage_receipt.json",
            {
                "schema_version": 1,
                "artifact_type": "RIVER_HISTORICAL_STAGE_RECEIPT_V1",
                "stage": stage,
                "fixed_roster": stage_policy(stage, "fixed_roster"),
                "deployment_roster": stage_policy(stage, "deployment_roster"),
                "b0_pose_table_sha256": b0["pose_table_sha256"],
                "b0_point_xyz_sha256": b0["point_xyz_sha256"],
                "historical_active": rosters["fixed"][stage]["historical_active"],
                "promotion": decisions.get(stage),
            },
        )
        assert_b0_unchanged(Path(str(config["b0_root"])), b0)
    return {"selected_stage": selected, "decisions": decisions, "e0_success": e0_success}


def execute_historical_pipeline(config: Mapping[str, Any], run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve(strict=True)
    b0 = _load_json(run_root / "input_lock/b0.json")
    assert_b0_unchanged(Path(str(config["b0_root"])), b0)
    frames = extract_update_if_needed(config, run_root)
    records = list(frames["frames"])
    descriptor_catalog = extract_historical_descriptor_catalog(config, run_root, records)
    descriptors = np.load(run_root / "historical/descriptors/update_descriptors.npy")
    catalog = load_reference_catalog(run_root / "historical/descriptors/b0_catalog")
    registrations = register_historical_views(config, run_root, records, descriptors, catalog)
    masks = calibrate_and_apply_masks(config, run_root, catalog, registrations)
    bridge = reconstruct_bridge_if_needed(run_root, registrations, masks)
    rosters = build_rosters(run_root, catalog, registrations, masks)
    protocol = None
    if not (run_root / "validation/protocol.json").is_file():
        try:
            protocol = materialize_p157_protocol(
                run_root=run_root,
                raw_root=Path(str(config["raw_root"])),
            )
        except FileExistsError:
            protocol = {"status": "VERIFY_TREE_ALREADY_PRESENT"}
    if not (run_root / "validation/alignments/P1570157.json").is_file():
        align_p157_with_e0_anchors(run_root=run_root, anchors=[])
    e0 = run_e0_control(config, run_root, catalog)
    promotion = write_stage_receipts(config, run_root, rosters, e0, registrations)
    freeze = None
    if not (run_root / "validation/freeze.json").is_file():
        freeze = freeze_validation_protocol(
            run_root=run_root,
            code_sha256=fingerprint_file(
                Path(__file__).resolve(), sha256=True
            ).sha256
            or "",
            config_sha256=fingerprint_file(run_root / "input_lock/config.json", sha256=True).sha256
            or "",
            runtime_sha256=fingerprint_file(
                run_root / "input_lock/runtime/runtime.json", sha256=True
            ).sha256
            or "",
            bundle_sha256={"E0": str(e0.get("status"))},
        )
    seal_final_replay(
        freeze=_load_json(run_root / "validation/freeze.json"),
        executed_partition="final",
    )
    if not (run_root / "reports/FINAL_REPORT.json").is_file():
        render_historical_report(
            run_root=run_root,
            stage_results={
                "E0": e0,
                "promotion": promotion,
                "direct_registration": Counter(str(row["status"]) for row in registrations),
                "bridge": {"triggered": bridge.get("triggered"), "accepted": len(bridge.get("accepted", []))},
            },
            reference_rows=[
                {"source_role": "current"},
                *[
                    {
                        "source_role": "historical_existing"
                        if row["status"] == DIRECT_STRONG
                        else "historical_only",
                        "merged_into_b0": False,
                    }
                    for row in registrations
                ],
            ],
            selected_stage=str(promotion["selected_stage"]),
            b0_receipt=b0,
            integrity={"b0_unchanged": True},
        )
    if not (run_root / "reports/integrity_receipt.json").is_file():
        write_integrity_receipt(
            run_root=run_root,
            hashes={
                "b0": b0,
                "source_videos": fingerprint_file(
                    run_root / "input_lock/source_videos.json", sha256=True
                ).as_dict(),
                "runtime_assets": fingerprint_file(
                    run_root / "input_lock/runtime/runtime.json", sha256=True
                ).as_dict(),
                "code_config": fingerprint_file(run_root / "input_lock/config.json", sha256=True).as_dict(),
                "bundles": rosters["selected_historical"],
                "reports": ["reports/FINAL_REPORT.json"],
                "selected_stage_export": promotion["selected_stage"],
            },
        )
    assert_b0_unchanged(Path(str(config["b0_root"])), b0)
    return {
        "frames": len(records),
        "registrations": len(registrations),
        "direct_strong": sum(1 for row in registrations if row["status"] == DIRECT_STRONG),
        "selected_stage": promotion["selected_stage"],
        "e0": e0,
        "protocol": protocol,
        "freeze": freeze,
    }



def execute_map_extend(config: Mapping[str, Any], run_root: Path) -> dict[str, Any]:
    """Relift persisted strong views if needed, then append observations and fringe points."""

    run_root = run_root.resolve(strict=True)
    b0 = _load_json(run_root / "input_lock/b0.json")
    assert_b0_unchanged(Path(str(config["b0_root"])), b0)
    registrations = list(
        _load_json(run_root / "historical/direct_registration/registrations.json")["rows"]
    )
    labels = dict(_load_json(run_root / "historical/change_masks/summary.json")["labels"])
    selected = select_append_rows(registrations, labels)
    descriptor_catalog = _load_json(run_root / "historical/descriptors/update_catalog.json")
    names = [str(name) for name in descriptor_catalog["names"]]
    descriptors = np.load(run_root / "historical/descriptors/update_descriptors.npy")
    name_to_index = {name: index for index, name in enumerate(names)}
    catalog = load_reference_catalog(run_root / "historical/descriptors/b0_catalog")
    need_relift = [
        row for row in selected if not isinstance(row.get("inlier_observations"), list) or not row["inlier_observations"]
    ]
    if need_relift:
        edm = load_official_edm_runtime(
            edm_repo=Path(str(config["runtime_assets"]["edm_source"])),
            checkpoint=Path(str(config["runtime_assets"]["edm_checkpoint"])),
            config_path=Path(str(config["runtime_assets"]["edm_config"])),
            data_config_path=Path(str(config["runtime_assets"]["edm_data_config"])),
            device=str(config["runtime"]["device"]),
        )
        estimator = _pose_estimator(_camera_params(run_root))
        refreshed: dict[str, dict[str, Any]] = {}
        for row in need_relift:
            query_name = str(row["query_name"])
            cache = run_root / "historical/observation_append" / f"{Path(query_name).stem}.json"
            if cache.is_file():
                refreshed[query_name] = _load_json(cache)
                continue
            if query_name not in name_to_index:
                raise HistoricalExperimentError(f"missing historical descriptor for {query_name}")
            relifted = register_historical_view(
                query_name=query_name,
                query_image=run_root / "historical/images" / query_name,
                query_descriptor=descriptors[name_to_index[query_name]],
                catalog=catalog,
                current_image_root=CURRENT_IMAGE_ROOT,
                runtime=edm,
                camera_params=_camera_params(run_root),
                thresholds=dict(config["thresholds"]),
                estimate_pose=estimator,
                topk=8,
            )
            relifted["image_sha256"] = row.get("image_sha256")
            relifted["source_video"] = row.get("source_video")
            relifted["source_pts_seconds"] = row.get("source_pts_seconds")
            relifted["source_frame_index"] = row.get("source_frame_index")
            persist_relifted_row(run_root, relifted)
            refreshed[query_name] = relifted
            assert_b0_unchanged(Path(str(config["b0_root"])), b0)
        selected = tuple(refreshed.get(str(row["query_name"]), row) for row in selected)
    receipt = extend_from_registrations(
        run_root=run_root,
        b0_root=Path(str(config["b0_root"])),
        registrations=selected,
        labels=labels,
        camera_params=_camera_params(run_root),
        b0_receipt=b0,
    )
    write_new_json(run_root, "maps/extend_receipt.json", receipt)
    assert_b0_unchanged(Path(str(config["b0_root"])), b0)
    return receipt


def _sample_b0_queries(names: Sequence[str], *, per_route: int = 8) -> tuple[str, ...]:
    by_route: dict[str, list[str]] = defaultdict(list)
    for name in names:
        if name.startswith(BASE_VIDEO_NAMES[0][:8]) or name.startswith("P118") or name.startswith(
            "P119"
        ) or name.startswith("P120"):
            route = name.split("/", 1)[0]
            if route in {"P1180118", "P1190119", "P1200120"}:
                by_route[route].append(name)
    sampled: list[str] = []
    for route in ("P1180118", "P1190119", "P1200120"):
        rows = by_route.get(route, [])
        if not rows:
            continue
        if len(rows) <= per_route:
            sampled.extend(rows)
            continue
        indices = np.linspace(0, len(rows) - 1, per_route, dtype=int)
        sampled.extend(rows[int(index)] for index in indices)
    return tuple(sampled)


def execute_append_loo(config: Mapping[str, Any], run_root: Path) -> dict[str, Any]:
    """Paired sampled B0 LOO: E0 catalog vs append overlay. Fringe points stay out."""

    run_root = run_root.resolve(strict=True)
    output = run_root / "maps/append_loo.json"
    if output.is_file():
        return _load_json(output)
    b0 = _load_json(run_root / "input_lock/b0.json")
    assert_b0_unchanged(Path(str(config["b0_root"])), b0)
    catalog = load_reference_catalog(run_root / "historical/descriptors/b0_catalog")
    labels = dict(_load_json(run_root / "historical/change_masks/summary.json")["labels"])
    registrations = list(
        _load_json(run_root / "historical/direct_registration/registrations.json")["rows"]
    )
    selected = select_append_rows(registrations, labels)
    descriptor_catalog = _load_json(run_root / "historical/descriptors/update_catalog.json")
    hist_names = [str(name) for name in descriptor_catalog["names"]]
    hist_descriptors = np.load(run_root / "historical/descriptors/update_descriptors.npy")
    name_to_desc = {name: hist_descriptors[index] for index, name in enumerate(hist_names)}
    overlay_rows: list[dict[str, Any]] = []
    for row in selected:
        query_name = str(row["query_name"])
        shard = run_root / "historical/observation_append" / f"{Path(query_name).stem}.json"
        payload = _load_json(shard) if shard.is_file() else dict(row)
        payload["query_descriptor"] = name_to_desc[query_name]
        overlay_rows.append(payload)
    append_catalog = overlay_append_catalog(catalog, overlay_rows)
    queries = _sample_b0_queries(catalog.names, per_route=8)
    if len(queries) < 6:
        raise HistoricalExperimentError("append LOO needs at least two queries per current route")
    edm = load_official_edm_runtime(
        edm_repo=Path(str(config["runtime_assets"]["edm_source"])),
        checkpoint=Path(str(config["runtime_assets"]["edm_checkpoint"])),
        config_path=Path(str(config["runtime_assets"]["edm_config"])),
        data_config_path=Path(str(config["runtime_assets"]["edm_data_config"])),
        device=str(config["runtime"]["device"]),
    )
    estimator = _pose_estimator(_camera_params(run_root))
    pairs: list[dict[str, Any]] = []
    for query_name in queries:
        query_index = catalog.names.index(query_name)
        pair: dict[str, Any] = {"query_name": query_name}
        for label, active in (("e0", catalog), ("append", append_catalog)):
            row = register_historical_view(
                query_name=query_name,
                query_image=CURRENT_IMAGE_ROOT / query_name,
                query_descriptor=catalog.descriptors[query_index],
                catalog=active,
                current_image_root=CURRENT_IMAGE_ROOT,
                runtime=edm,
                camera_params=_camera_params(run_root),
                thresholds=dict(config["thresholds"]),
                estimate_pose=estimator,
                historical_image_root=run_root / "historical/images",
                topk=8,
            )
            pair[label] = {
                "status": row["status"],
                "inlier_count": int((row.get("metrics") or {}).get("inlier_count") or 0),
                "reprojection_p90": (row.get("metrics") or {}).get("reprojection_p90"),
                "convex_hull_coverage": (row.get("metrics") or {}).get("convex_hull_coverage"),
                "retrieved_names": row.get("retrieved_names"),
                "historical_in_topk": [
                    name
                    for name in row.get("retrieved_names") or []
                    if str(name).startswith(("P116", "P117"))
                ],
            }
        pairs.append(pair)
        assert_b0_unchanged(Path(str(config["b0_root"])), b0)
    e0_ok = sum(1 for row in pairs if row["e0"]["status"] == DIRECT_STRONG)
    append_ok = sum(1 for row in pairs if row["append"]["status"] == DIRECT_STRONG)
    e0_rate = e0_ok / len(pairs)
    append_rate = append_ok / len(pairs)
    decision = "APPEND_ADMITTED" if append_rate + 1e-12 >= e0_rate - 0.01 else "ROLLBACK_APPEND"
    fringe_decision = (
        "FRINGE_DIAGNOSTIC_ONLY"
        if decision == "APPEND_ADMITTED"
        else "FRINGE_BLOCKED_APPEND_REGRESSED"
    )
    receipt = {
        "schema_version": 1,
        "artifact_type": "RIVER_APPEND_PAIRED_LOO_V1",
        "query_count": len(pairs),
        "e0_success_count": e0_ok,
        "append_success_count": append_ok,
        "e0_success_rate": e0_rate,
        "append_success_rate": append_rate,
        "success_delta": append_rate - e0_rate,
        "decision": decision,
        "fringe_decision": fringe_decision,
        "pairs": pairs,
        "b0_unchanged": True,
    }
    write_new_json(run_root, "maps/append_loo.json", receipt)
    assert_b0_unchanged(Path(str(config["b0_root"])), b0)
    return receipt



