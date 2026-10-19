from __future__ import annotations

import json
import hashlib
import os
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .groups import MVGroupConfig, plan_groups
from .tracks import sample_multiview_tracks

RIVER_V3_SPARSE_VIDEOS = (
    "vid_8368de51fcbc5376",
    "vid_d321e99793a92bb6",
    "vid_931611d6ff4f0361",
)
RIVER_V3_WEAK_KEYFRAMES = (
    "vid_349c83c4bf56a785:00001512",
    "vid_600bbf70227311ca:00002388",
)
MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")
TARGETS_PER_SOURCE = 5
MAXIMUM_BRIDGE_HOPS = 4
MINIMUM_TRACK_LENGTH = 5


@dataclass(frozen=True)
class MVRoMaRequest:
    scope: str
    input_model: Path
    images: Path
    selection: Path
    intrinsics: Path
    run_dir: Path
    runtime_python: Path
    runtime_root: Path
    weight: Path
    continuation_receipt: Path | None = None
    coarse_size: tuple[int, int] = (378, 672)
    target_size: tuple[int, int] = (756, 1344)
    certainty_threshold: float = 0.5
    max_samples_per_group: int = 1024
    sparse_video_ids: tuple[str, ...] = RIVER_V3_SPARSE_VIDEOS
    weak_keyframe_ids: tuple[str, ...] = RIVER_V3_WEAK_KEYFRAMES
    timeout_seconds: int = 86_400

    def with_target_size(self, value: tuple[int, int]) -> "MVRoMaRequest":
        return replace(self, target_size=value)

    def validate_resolution(self) -> None:
        for name, size in (("coarse", self.coarse_size), ("target", self.target_size)):
            if any(value <= 0 or value % 14 for value in size):
                raise ValueError(f"MV-RoMa {name} dimensions must be positive and divisible by 14")
        coarse_ratio = self.coarse_size[1] / self.coarse_size[0]
        target_ratio = self.target_size[1] / self.target_size[0]
        native_ratio = 2688 / 1512
        if abs(coarse_ratio - native_ratio) > 0.01 or abs(target_ratio - native_ratio) > 0.01:
            raise ValueError("MV-RoMa resolutions must preserve the River aspect ratio")

    def validate_continuation(self) -> None:
        if self.scope != "full":
            return
        if self.continuation_receipt is None:
            raise ValueError("full MV-RoMa requires a targeted continuation receipt")
        payload = json.loads(self.continuation_receipt.read_text(encoding="utf-8"))
        if not bool((payload.get("mapping_gate") or {}).get("passes")):
            raise ValueError("targeted MV-RoMa continuation receipt did not pass")

    @property
    def worker_request(self) -> Path:
        return self.run_dir / "bootstrap/worker_request.json"

    @property
    def worker_receipt(self) -> Path:
        return self.run_dir / "receipts/worker.json"

    @property
    def shards_dir(self) -> Path:
        return self.run_dir / "mvroma/raw_shards"

    def validate(self) -> None:
        if self.scope not in {"targeted", "full"}:
            raise ValueError("MV-RoMa scope must be targeted or full")
        model = self.input_model.expanduser().resolve(strict=True)
        missing = [name for name in MODEL_FILES if not (model / name).is_file()]
        if missing:
            raise ValueError(f"input model is missing files: {missing}")
        for name, path, directory in (
            ("images", self.images, True),
            ("selection", self.selection, False),
            ("intrinsics", self.intrinsics, False),
            ("runtime root", self.runtime_root, True),
            ("weight", self.weight, False),
        ):
            resolved = path.expanduser().resolve(strict=True)
            if directory != resolved.is_dir():
                raise ValueError(f"MV-RoMa {name} has the wrong file type")
        runtime = self.runtime_python.expanduser().resolve(strict=True)
        if not runtime.is_file() or not os.access(runtime, os.X_OK):
            raise ValueError("MV-RoMa runtime Python must be executable")
        if not (self.runtime_root / "demo.py").is_file():
            raise ValueError("MV-RoMa runtime root is missing demo.py")
        if self.run_dir.resolve().is_relative_to(model):
            raise ValueError("MV-RoMa run must be outside the input model")
        if not 0.0 < self.certainty_threshold < 1.0:
            raise ValueError("MV-RoMa certainty threshold must be in (0,1)")
        if self.max_samples_per_group <= 0 or self.timeout_seconds <= 0:
            raise ValueError("MV-RoMa sample and timeout limits must be positive")
        self.validate_resolution()
        self.validate_continuation()

    def fingerprint(self) -> str:
        self.validate()
        payload = {
            "scope": self.scope,
            "model": {name: _sha256(self.input_model / name) for name in MODEL_FILES},
            "selection": _sha256(self.selection),
            "intrinsics": _sha256(self.intrinsics),
            "weight": _sha256(self.weight),
            "runtime_python": str(self.runtime_python.resolve()),
            "runtime_root": str(self.runtime_root.resolve()),
            "coarse_size": self.coarse_size,
            "target_size": self.target_size,
            "certainty_threshold": self.certainty_threshold,
            "max_samples_per_group": self.max_samples_per_group,
            "track_policy": {
                "targets_per_source": TARGETS_PER_SOURCE,
                "maximum_bridge_hops": MAXIMUM_BRIDGE_HOPS,
                "minimum_track_length": MINIMUM_TRACK_LENGTH,
            },
            "sparse_video_ids": self.sparse_video_ids,
            "weak_keyframe_ids": self.weak_keyframe_ids,
            "continuation": (
                None if self.continuation_receipt is None else _sha256(self.continuation_receipt)
            ),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_worker_command(request: MVRoMaRequest) -> tuple[str, ...]:
    return (
        str(request.runtime_python.expanduser().resolve()),
        "-m",
        "river_mvroma.worker",
        "--request",
        str(request.worker_request),
        "--receipt",
        str(request.worker_receipt),
    )


def run_mvroma(
    request: MVRoMaRequest, *, dry_run: bool = False, resume: bool = False
) -> dict[str, Any]:
    request.validate()
    selection = json.loads(request.selection.read_text(encoding="utf-8"))
    groups = plan_groups(
        selection,
        images_root=request.images,
        config=MVGroupConfig(
            scope=request.scope,
            sparse_video_ids=request.sparse_video_ids,
            weak_keyframe_ids=request.weak_keyframe_ids,
            targets_per_source=TARGETS_PER_SOURCE,
            maximum_bridge_hops=MAXIMUM_BRIDGE_HOPS,
        ),
    )
    fingerprint = request.fingerprint()
    if dry_run:
        return {
            "status": "DRY_RUN_READY",
            "scope": request.scope,
            "groups": len(groups),
            "request_fingerprint": fingerprint,
            "command": list(build_worker_command(request)),
        }
    if request.run_dir.exists():
        if resume:
            final = request.run_dir / "receipts/final.json"
            if final.is_file():
                payload = json.loads(final.read_text(encoding="utf-8"))
                if payload.get("request_fingerprint") == fingerprint:
                    return payload
        raise FileExistsError(request.run_dir)
    (request.run_dir / "bootstrap").mkdir(parents=True)
    (request.run_dir / "receipts").mkdir()
    (request.run_dir / "logs").mkdir()
    (request.run_dir / "locks").mkdir()
    group_manifest = request.run_dir / "bootstrap/groups.json"
    group_manifest.write_text(
        json.dumps([group.as_dict() for group in groups], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    worker_payload = {
        "schema_version": 1,
        "groups": [group.as_dict() for group in groups],
        "output_dir": str(request.shards_dir),
        "weight": str(request.weight.resolve()),
        "coarse_size": list(request.coarse_size),
        "target_size": list(request.target_size),
        "device": "cuda:0",
    }
    request.worker_request.write_text(
        json.dumps(worker_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    from sfm_diagnosis.site_pipeline.adapters import (
        ExclusiveResourceLease,
        probe_python_runtime,
        require_compute_capability,
    )

    runtime = probe_python_runtime(request.runtime_python)
    require_compute_capability(runtime, "12.0")
    (request.run_dir / "runtime_preflight.json").write_text(
        json.dumps(runtime, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    environment = os.environ.copy()
    root = str(request.runtime_root.resolve())
    environment["PYTHONPATH"] = root + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    log_path = request.run_dir / "logs/mvroma.log"
    with ExclusiveResourceLease(request.run_dir / "locks/heavy.lock", "gpu_heavy"):
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit
                build_worker_command(request),
                cwd=request.runtime_root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=request.timeout_seconds,
                check=False,
            )
    if completed.returncode:
        raise RuntimeError(f"MV-RoMa worker failed ({completed.returncode}); see {log_path}")
    if not request.worker_receipt.is_file():
        raise RuntimeError("MV-RoMa worker did not write its receipt")
    worker = json.loads(request.worker_receipt.read_text(encoding="utf-8"))
    if worker.get("status") != "COMPLETED" or int(worker.get("groups") or 0) != len(groups):
        raise RuntimeError("MV-RoMa worker receipt is incomplete")

    observation_tracks: list[dict[str, tuple[float, float]]] = []
    for row in worker.get("shards") or ():
        shard = Path(str(row["path"])).resolve(strict=True)
        with np.load(shard, allow_pickle=False) as arrays:
            source_name = str(arrays["source_name"].item())
            target_names = tuple(str(value) for value in arrays["target_names"].tolist())
            observation_tracks.extend(
                sample_multiview_tracks(
                    source_name=source_name,
                    target_names=target_names,
                    flow=np.asarray(arrays["flow"], dtype=np.float32),
                    certainty_logits=np.asarray(arrays["certainty_logits"], dtype=np.float32),
                    native_size=(1512, 2688),
                    certainty_threshold=request.certainty_threshold,
                    minimum_target_observations=MINIMUM_TRACK_LENGTH - 1,
                    max_samples=request.max_samples_per_group,
                )
            )
    from river_v4_optimizer.detector_free_injection import (
        DetectorFreeInjectionConfig,
        inject_planned_tracks,
        plan_observation_tracks,
    )
    from river_v4_optimizer.runner import atomic_json

    plans, track_plan = plan_observation_tracks(
        model=request.input_model,
        observation_tracks=observation_tracks,
        config=DetectorFreeInjectionConfig(
            required_videos=(),
            minimum_distinct_videos=2,
            minimum_track_length=MINIMUM_TRACK_LENGTH,
            maximum_reprojection_error_px=3.0,
            maximum_reprojection_p90_px=2.0,
            minimum_triangulation_angle_deg=1.0,
            existing_observation_conflict_radius_px=2.0,
        ),
    )
    track_plan.update(
        {
            "groups": len(groups),
            "sampled_observation_tracks": len(observation_tracks),
            "certainty_threshold": request.certainty_threshold,
            "max_samples_per_group": request.max_samples_per_group,
            "minimum_track_length": MINIMUM_TRACK_LENGTH,
        }
    )
    atomic_json(request.run_dir / "receipts/track_plan.json", track_plan)
    if not plans:
        final = {
            "schema_version": 1,
            "status": "COMPLETED",
            "scope": request.scope,
            "request_fingerprint": fingerprint,
            "groups": len(groups),
            "mapping_gate": {"passes": False, "checks": {"approved_tracks": False}},
            "track_plan": track_plan,
            "worker_receipt": str(request.worker_receipt),
            "log": str(log_path),
        }
        atomic_json(request.run_dir / "receipts/final.json", final)
        return final

    injected_model = request.run_dir / "artifacts/injected/model"
    robust_model = request.run_dir / "artifacts/robust/model"
    injection = inject_planned_tracks(request.input_model, injected_model, plans)
    from sfm_diagnosis.site_pipeline.robust_filter import (
        RobustFilterConfig,
        robust_filter_model,
    )

    robust = robust_filter_model(
        injected_model,
        robust_model,
        RobustFilterConfig(
            max_reprojection_error_px=3.0,
            minimum_triangulation_angle_deg=1.5,
            minimum_track_length=3,
            bundle_adjustment_iterations=100,
            bundle_adjustment_threads=8,
        ),
    )
    atomic_json(request.run_dir / "receipts/robust_filter.json", robust)
    mapping_gate = _mapping_gate(
        request.input_model,
        robust_model,
        weak_keyframe_ids=request.weak_keyframe_ids,
        added_point_ids=tuple(int(value) for value in injection["added_point_ids"]),
        worker_groups=int(worker["groups"]),
        expected_groups=len(groups),
    )
    final = {
        "schema_version": 1,
        "status": "COMPLETED",
        "scope": request.scope,
        "request_fingerprint": fingerprint,
        "groups": len(groups),
        "worker_receipt": str(request.worker_receipt),
        "track_plan": track_plan,
        "injection": injection,
        "robust_model": str(robust_model),
        "mapping_gate": mapping_gate,
        "log": str(log_path),
    }
    atomic_json(request.run_dir / "receipts/final.json", final)
    return final


def _mapping_gate(
    baseline_model: Path,
    candidate_model: Path,
    *,
    weak_keyframe_ids: tuple[str, ...],
    added_point_ids: tuple[int, ...],
    worker_groups: int,
    expected_groups: int,
) -> dict[str, Any]:
    import pycolmap
    from river_v4_optimizer.metrics import analyze_model
    from sfm_diagnosis.site_pipeline.loo import (
        aligned_camera_stability,
    )
    from sfm_diagnosis.site_pipeline.map_refinement import (
        _camera_pose_map,
        _load_metrics,
        _topology_summary,
        apply_pose_stability_gate,
        apply_topology_gate,
        evaluate_geometry_gate,
    )

    gate = evaluate_geometry_gate(_load_metrics(baseline_model), _load_metrics(candidate_model))
    gate = apply_pose_stability_gate(
        gate,
        aligned_camera_stability(
            _camera_pose_map(baseline_model), _camera_pose_map(candidate_model)
        ),
    )
    baseline_topology = _topology_summary(analyze_model(baseline_model))
    candidate_topology = _topology_summary(analyze_model(candidate_model))
    gate = apply_topology_gate(gate, baseline_topology, candidate_topology)
    baseline = pycolmap.Reconstruction(str(baseline_model))
    candidate = pycolmap.Reconstruction(str(candidate_model))
    baseline_images = {str(image.name): image for image in baseline.images.values()}
    candidate_images = {str(image.name): image for image in candidate.images.values()}
    weak_counts = {}
    weak_non_regression = True
    for keyframe_id in weak_keyframe_ids:
        video, frame = keyframe_id.split(":", 1)
        name = f"{video}/frame_{frame}.jpg"
        before = int(baseline_images[name].num_points3D) if name in baseline_images else 0
        after = int(candidate_images[name].num_points3D) if name in candidate_images else 0
        weak_counts[name] = {"baseline": before, "candidate": after}
        weak_non_regression = weak_non_regression and after >= before
    surviving_cross_session = 0
    for point_id in added_point_ids:
        if point_id not in candidate.points3D:
            continue
        videos = {
            str(candidate.images[element.image_id].name).split("/", 1)[0]
            for element in candidate.points3D[point_id].track.elements
        }
        if len(videos) >= 2:
            surviving_cross_session += 1
    checks = dict(gate.get("checks") or {})
    checks.update(
        {
            "worker_groups_complete": worker_groups == expected_groups,
            "approved_tracks_survive_cross_session": surviving_cross_session > 0,
            "weak_images_non_regression": weak_non_regression,
        }
    )
    gate["checks"] = checks
    gate["passes"] = all(checks.values())
    gate["weak_image_observations"] = weak_counts
    gate["surviving_added_cross_session_tracks"] = surviving_cross_session
    return gate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve(strict=True).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["MVRoMaRequest", "build_worker_command", "run_mvroma"]
