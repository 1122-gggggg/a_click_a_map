"""Adopted two-rate localization: KLT+PnP fast loop, MegaLoc+EDM reloc off-path."""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from sfm_diagnosis.io import write_json

from .adopted_pipeline import DEFAULT_EDM_DATA_CONFIG, DEFAULT_EDM_MODEL_CONFIG
from .edm_retriangulate import LIFT_DISTANCE_PX


@dataclass(frozen=True)
class DeployConfig:
    map_model: Path
    keyframes: Path
    frames_dir: Path
    output_dir: Path
    localizer_config: Path
    localization_dir: Path
    edm_repo: Path
    edm_checkpoint: Path
    edm_model_config: Path = DEFAULT_EDM_MODEL_CONFIG
    edm_data_config: Path = DEFAULT_EDM_DATA_CONFIG
    reference_depth_dir: Path | None = None
    excluded_session: str | None = None
    gt_video_id: str | None = None
    route: str = "stream"
    tracker: str = "klt"
    track_cap: int = 500
    stream_fps: float = 29.0
    max_frames: int = 0
    reloc_min_points: int = 60
    reloc_period_s: float = 0.3
    megaloc_top_k: int = 2
    lift_distance_px: float = LIFT_DISTANCE_PX
    reloc_matcher: str = "edm"
    sparse_max_keypoints: int = 4096
    sparse_ref_cache: int = 128
    vo: bool = True
    vo_batch_lag: int = 6
    vo_keyframe_stride: int = 8
    vo_min_live: int = 350
    vo_detect_cap: int = 300
    dead_reckon: bool = True
    vo_reentry_fit: bool = True
    topup: bool = False
    vo_refine: bool = False
    vo_window_ba: bool = False
    metric_scale: Path | None = None


TRACKER = "klt"
TRACK_CAP = 500
FLOW_RES = "640x360"
STREAM_FPS = 29.0
MAX_FRAMES = 0
RELOC_MIN_POINTS = 60
RELOC_PERIOD_S = 0.3
MIN_PNP_POINTS = 12
MIN_PNP_INLIERS = 12
SCALE_5090_TO_5060 = 4.0
RESEED_MIN_POINTS = 120
TOPUP = False
VO = True
VO_KEYFRAME_STRIDE = 8
VO_BATCH_LAG = 6
VO_REFINE = False
VO_REFINE_VIEWS = 8
VO_MIN_LIVE = 350
VO_DETECT_CAP = 300
VO_MIN_DISTANCE = 8
VO_MIN_PARALLAX_DEG = 0.5
VO_MAX_REPROJ_PX = 2.0
DEAD_RECKON = True
DEAD_RECKON_MAX_FRAMES = 300
VO_REENTRY_FIT = True
VO_WINDOW_BA = False
VO_BA_WINDOW = 8
VO_BA_MAX_MS = 8.0
SYNTH_HOLE = ""
ROUTE = "stream"
DENSE = Path(".")
OUT = Path(".")
ROUTE_SESSION: dict[str, str | None] = {}
ROUTE_VIDEO_ID: dict[str, str | None] = {}


# --------------------------------------------------------------------------- #
# fast trackers: prev frame 2D -> current frame 2D, same point ordering

# Thread pinning was tried and REJECTED by measurement: `cv2.setNumThreads(1)`
# pushed the KLT stage from 1.85 ms to 10.74 ms on P167, i.e. OpenCV's internal
# parallelism is carrying this workload, not fighting it. `CV_THREADS=0` leaves
# OpenCV's default pool alone; set it only to experiment.
_cv_threads = int(os.environ.get("CV_THREADS", "0"))
if _cv_threads:
    cv2.setNumThreads(_cv_threads)
# --------------------------------------------------------------------------- #
class KLTTracker:
    """Pyramidal Lucas-Kanade with a forward-backward gate. Pure CPU."""

    kind = "cpu"
    name = "klt"

    def __init__(self) -> None:
        self.fb_max_px = float(os.environ.get("KLT_FB_MAX_PX", "1.0"))
        self.win = int(os.environ.get("KLT_WIN", "21"))
        self.levels = int(os.environ.get("KLT_LEVELS", "3"))
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        self._prev_gray: np.ndarray | None = None

    def set_frame(self, bgr: np.ndarray) -> None:
        self._prev_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def track_pair(
        self, prev_bgr: np.ndarray, cur_bgr: np.ndarray, xy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._lk(
            cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY),
            xy,
        )

    def track(self, bgr: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        previous = self._prev_gray
        self._prev_gray = gray
        if previous is None:
            return xy, np.zeros(len(xy), dtype=bool)
        return self._lk(previous, gray, xy)

    # NOTE: pre-built pyramids (cv2.buildOpticalFlowPyramid) would remove three
    # of the four pyramid builds per frame, but OpenCV 5.0's Python binding
    # rejects a pyramid list for `prevImg` ("Expected Ptr<cv::UMat>"), so the
    # gray images are passed directly and OpenCV rebuilds internally.
    def _lk(
        self, previous: np.ndarray, current: np.ndarray, xy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if xy.size == 0:
            return xy, np.zeros(len(xy), dtype=bool)
        p0 = xy.astype(np.float32).reshape(-1, 1, 2)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            previous, current, p0, None, winSize=(self.win, self.win),
            maxLevel=self.levels, criteria=self.criteria,
        )
        p0b, stb, _ = cv2.calcOpticalFlowPyrLK(
            current, previous, p1, None, winSize=(self.win, self.win),
            maxLevel=self.levels, criteria=self.criteria,
        )
        fwd = p1.reshape(-1, 2)
        back = p0b.reshape(-1, 2)
        fb = np.linalg.norm(back - xy, axis=1)
        keep = (
            (st.reshape(-1) == 1) & (stb.reshape(-1) == 1)
            & np.isfinite(fwd).all(axis=1) & (fb <= self.fb_max_px)
        )
        return fwd, keep


class NeuFlowTracker:
    """NeuFlow v2 dense flow, sampled at the tracked pixels. GPU."""

    kind = "gpu"
    name = "neuflow"

    def __init__(self, width: int, height: int) -> None:
        import torch

        sys.path.insert(0, "/home/cihcilab/sfm/runtime/neuflow2")
        from data_utils.frame_utils import InputPadder
        from NeuFlow.neuflow import NeuFlow

        self.torch = torch
        self.w, self.h = width, height
        model = NeuFlow().cuda()
        ckpt = torch.load("/home/cihcilab/sfm/runtime/neuflow2/neuflow_mixed.pth", map_location="cuda")
        model.load_state_dict(ckpt["model"], strict=True)
        self.model = model.eval()
        probe = torch.zeros(1, 3, self.h, self.w, device="cuda")
        self.padder = InputPadder(probe.shape, padding_factor=16)
        padded, _ = self.padder.pad(probe, probe)
        model.init_bhwd(padded.shape[0], padded.shape[-2], padded.shape[-1], "cuda", amp=False)
        self._prev: object | None = None

    def _to_gpu(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if (rgb.shape[1], rgb.shape[0]) != (self.w, self.h):
            rgb = cv2.resize(rgb, (self.w, self.h), interpolation=cv2.INTER_AREA)
        t = self.torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).cuda()
        return self.padder.pad(t, t)[0]

    def set_frame(self, bgr: np.ndarray) -> None:
        self._prev = self._to_gpu(bgr)

    def track_pair(
        self, prev_bgr: np.ndarray, cur_bgr: np.ndarray, xy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._flow(self._to_gpu(prev_bgr), self._to_gpu(cur_bgr), xy)

    def track(self, bgr: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cur = self._to_gpu(bgr)
        prev = self._prev
        self._prev = cur
        if prev is None:
            return xy, np.zeros(len(xy), dtype=bool)
        return self._flow(prev, cur, xy)

    def _flow(self, prev, cur, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if xy.size == 0:
            return xy, np.zeros(len(xy), dtype=bool)
        with self.torch.no_grad():
            flow = self.model(prev, cur)[-1]
        flow = self.padder.unpad(flow)[0].float().cpu().numpy()
        sx, sy = self.w / IMAGE_W, self.h / IMAGE_H
        xs = np.clip((xy[:, 0] * sx).astype(np.int64), 0, self.w - 1)
        ys = np.clip((xy[:, 1] * sy).astype(np.int64), 0, self.h - 1)
        step = np.stack([flow[0][ys, xs] / sx, flow[1][ys, xs] / sy], axis=1)
        fwd = xy + step
        keep = np.isfinite(fwd).all(axis=1)
        return fwd, keep


# --------------------------------------------------------------------------- #
IMAGE_W = 960
IMAGE_H = 540


def _inside(xy: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(xy).all(axis=1)
        & (xy[:, 0] >= 0) & (xy[:, 0] < IMAGE_W)
        & (xy[:, 1] >= 0) & (xy[:, 1] < IMAGE_H)
    )


def _center(cam_from_world: np.ndarray) -> np.ndarray:
    rotation = np.asarray(cam_from_world, dtype=float)[:3, :3]
    translation = np.asarray(cam_from_world, dtype=float)[:3, 3]
    return -rotation.T @ translation


def run_deploy_loop(cfg: DeployConfig) -> dict[str, Any]:
    global TRACKER, TRACK_CAP, STREAM_FPS, MAX_FRAMES, RELOC_MIN_POINTS
    global RELOC_PERIOD_S, TOPUP, VO, VO_KEYFRAME_STRIDE, VO_BATCH_LAG
    global VO_REFINE, VO_MIN_LIVE, VO_DETECT_CAP, DEAD_RECKON, VO_REENTRY_FIT
    global VO_WINDOW_BA, ROUTE, DENSE, OUT

    TRACKER = cfg.tracker
    TRACK_CAP = cfg.track_cap
    STREAM_FPS = cfg.stream_fps
    MAX_FRAMES = cfg.max_frames
    RELOC_MIN_POINTS = cfg.reloc_min_points
    RELOC_PERIOD_S = cfg.reloc_period_s
    TOPUP = cfg.topup
    VO = cfg.vo
    VO_KEYFRAME_STRIDE = cfg.vo_keyframe_stride
    VO_BATCH_LAG = cfg.vo_batch_lag
    VO_REFINE = cfg.vo_refine
    VO_MIN_LIVE = cfg.vo_min_live
    VO_DETECT_CAP = cfg.vo_detect_cap
    DEAD_RECKON = cfg.dead_reckon
    VO_REENTRY_FIT = cfg.vo_reentry_fit
    VO_WINDOW_BA = cfg.vo_window_ba
    ROUTE = cfg.route
    DENSE = Path(cfg.frames_dir)
    OUT = Path(cfg.output_dir)

    from .deployment_localizer import (
        FinalMapEDMProvider,
        _configure_audited_runtime_site_packages,
        _resolve_audited_site_packages,
        scaled_pinhole_parameters,
    )

    _configure_audited_runtime_site_packages(_resolve_audited_site_packages(None))

    import pycolmap
    from river_map_quality.megaloc_edm_catalog import (
        extract_megaloc_descriptors,
        load_offline_megaloc_runtime,
    )
    from sfm_diagnosis.edm_risk.edm_loo import EDMQuery
    from sfm_diagnosis.viewpoint_policy import resolve_megaloc_bank

    metric_sidecar = None
    meters_per_map_unit = None
    if cfg.metric_scale is not None:
        from .metric_scale import (
            add_metric_position_fields,
            load_metric_scale,
            metric_landmarks,
        )

        metric_sidecar = load_metric_scale(Path(cfg.metric_scale), Path(cfg.map_model))
        meters_per_map_unit = float(metric_sidecar["meters_per_map_unit"])

    frames = sorted(DENSE.glob("frame_*.jpg"))
    if not frames:
        raise FileNotFoundError(f"no dense frames under {DENSE}")
    if MAX_FRAMES:
        frames = frames[:MAX_FRAMES]
    frame_interval = 1.0 / STREAM_FPS
    config = json.loads(Path(cfg.localizer_config).read_text(encoding="utf-8"))
    fx, fy, cx, cy = scaled_pinhole_parameters(config["intrinsics"], width=IMAGE_W, height=IMAGE_H)
    query_cam = pycolmap.Camera.create(1, "PINHOLE", fx, IMAGE_W, IMAGE_H)
    query_cam.params = [fx, fy, cx, cy]

    pnp_estimation = pycolmap.AbsolutePoseEstimationOptions()
    pnp_estimation.ransac.max_error = float(os.environ.get("PNP_MAX_ERROR_PX", "4.0"))
    pnp_estimation.ransac.min_num_trials = int(os.environ.get("PNP_MIN_TRIALS", "10"))
    pnp_estimation.ransac.max_num_trials = int(os.environ.get("PNP_MAX_TRIALS", "100"))
    pnp_estimation.ransac.confidence = float(os.environ.get("PNP_CONFIDENCE", "0.999"))
    pnp_refinement = pycolmap.AbsolutePoseRefinementOptions()
    pnp_refinement.max_num_iterations = int(os.environ.get("PNP_REFINE_ITERS", "20"))

    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    manifest_rows = [
        json.dumps(
            {
                "query_id": f"replay_{path.stem}",
                "session_id": f"replay_{ROUTE}",
                "timestamp": index * frame_interval,
                "image_path": str(path),
                "pose_provenance": "NONE",
            }
        )
        for index, path in enumerate(frames)
    ]
    query_manifest = cache / "queries.jsonl"
    query_manifest.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")

    bank_descriptors, bank_names, bank_kind = resolve_megaloc_bank(Path(cfg.localization_dir))
    bundled_names = [str(n) for n in json.loads(Path(bank_names).read_text(encoding="utf-8"))]
    descriptors = np.load(bank_descriptors, allow_pickle=False)
    depth_dir = cfg.reference_depth_dir

    provider = FinalMapEDMProvider(
        map_model=str(Path(cfg.map_model).resolve()),
        keyframes=str(Path(cfg.keyframes).resolve()),
        query_manifest=str(query_manifest),
        cache_dir=str(cache / "provider"),
        edm_config={
            "repo": str(Path(cfg.edm_repo)),
            "checkpoint": str(Path(cfg.edm_checkpoint)),
            "model_config": str(Path(cfg.edm_model_config)),
            "data_config": str(Path(cfg.edm_data_config)),
            "input_size": [640, 640],
            "confidence_threshold": 0.0,
        },
        megaloc_source=str(config["megaloc_source"]),
        megaloc_checkpoint=str(config["megaloc_checkpoint"]),
        intrinsics_calibration=config["intrinsics"],
        top_k=cfg.megaloc_top_k,
        lift_distance_px=cfg.lift_distance_px,
        descriptor_batch_size=1,
        min_reference_occupied_bins=1,
        reference_depth_dir=None if depth_dir is None else str(depth_dir),
        matcher_backend=cfg.reloc_matcher,
        sparse_max_keypoints=cfg.sparse_max_keypoints,
        sparse_ref_cache=cfg.sparse_ref_cache,
    )
    provider._load_geometry()
    provider.apply_reference_bank(bundled_names, descriptors)
    provider._prepared = True
    excluded = frozenset({cfg.excluded_session} if cfg.excluded_session else set())
    index = provider.build_reference_index(excluded_sessions=excluded, strict=bool(excluded))

    map_point_ids = np.asarray(provider._point_ids, dtype=np.int64)
    map_point_xyz = np.asarray(provider._point_xyz, dtype=float)

    def lookup_xyz(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(xyz rows, present mask) for map point ids, via the sorted id table."""
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size == 0:
            return np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=bool)
        slot = np.searchsorted(map_point_ids, ids)
        clipped = np.clip(slot, 0, map_point_ids.size - 1)
        present = map_point_ids[clipped] == ids
        return map_point_xyz[clipped], present

    # The provider matches in the query image's own pixel frame, and the replay
    # feeds it exactly the 960x540 stream frames, so anchor xy needs no rescale.

    # ground truth: frozen GlueMap poses of this session's mapping keyframes
    gt_center: dict[int, np.ndarray] = {}
    gt_pose: dict[int, np.ndarray] = {}
    video_id = cfg.gt_video_id
    if video_id:
        for name, cam_from_world in provider._cam_from_world.items():
            if not str(name).startswith(video_id + "/"):
                continue
            frame_index = int(Path(str(name)).stem.split("_")[-1])
            gt_pose[frame_index] = np.asarray(cam_from_world, dtype=float)
            gt_center[frame_index] = _center(cam_from_world)

    megaloc_runtime = load_offline_megaloc_runtime(
        source=Path(str(config["megaloc_source"])),
        checkpoint=Path(str(config["megaloc_checkpoint"])),
        device="cuda",
    )
    tracker = (
        KLTTracker()
        if TRACKER == "klt"
        else NeuFlowTracker(*(int(v) for v in FLOW_RES.split("x")))
    )

    live_xy = np.zeros((0, 2), dtype=np.float64)
    live_ids = np.zeros((0,), dtype=np.int64)
    live_xyz = np.zeros((0, 3), dtype=np.float64)
    pending: dict | None = None
    last_reloc_frame = -10**9
    reloc_events: list[dict] = []
    rows: list[dict] = []
    stage_ms: dict[str, list[float]] = {
        "decode": [], "track": [], "pnp": [], "refill": [], "vo": [],
        "loop": [], "reseed": [],
    }
    # Deployment preloads the retrieval + EDM weights, so the first stream frame
    # must not pay the lazy model load. Warm it here, outside the timed loop.
    warm_id = f"replay_{frames[0].stem}"
    provider._query_descriptors.update(
        dict(
            zip(
                [warm_id],
                extract_megaloc_descriptors(megaloc_runtime, [frames[0]], batch_size=1),
                strict=True,
            )
        )
    )
    provider.localize(
        EDMQuery(
            query_id=warm_id,
            session_id=f"replay_{ROUTE}",
            timestamp=0.0,
            image_path=str(frames[0]),
            pose_provenance="NONE",
        ),
        index,
    )

    prev_center: np.ndarray | None = None
    pool_ids = np.zeros((0,), dtype=np.int64)
    pool_xyz = np.zeros((0, 3), dtype=float)

    def topup(pose_3x4: np.ndarray, xy: np.ndarray, ids: np.ndarray, xyz: np.ndarray):
        """Re-project the local map into the current frame to refill the live set.

        Off by default: seeding points at their pose-predicted pixel makes the
        next PnP agree with the pose it already had, which locks in drift.
        """
        if not TOPUP:
            return xy, ids, xyz
        if pool_ids.size == 0 or len(ids) >= TRACK_CAP:
            return xy, ids, xyz
        rotation = np.asarray(pose_3x4, dtype=float)[:3, :3]
        translation = np.asarray(pose_3x4, dtype=float)[:3, 3]
        camera_xyz = pool_xyz @ rotation.T + translation
        depth = camera_xyz[:, 2]
        visible = depth > 1e-6
        u = fx * camera_xyz[:, 0] / np.where(visible, depth, 1.0) + cx
        v = fy * camera_xyz[:, 1] / np.where(visible, depth, 1.0) + cy
        visible &= (u >= 4) & (u < IMAGE_W - 4) & (v >= 4) & (v < IMAGE_H - 4)
        if not visible.any():
            return xy, ids, xyz
        fresh = visible & ~np.isin(pool_ids, ids)
        candidates = np.flatnonzero(fresh)
        room = TRACK_CAP - len(ids)
        if candidates.size > room:
            candidates = candidates[np.linspace(0, candidates.size - 1, room).astype(int)]
        add_xy = np.stack([u[candidates], v[candidates]], axis=1)
        return (
            np.concatenate([xy, add_xy]) if len(xy) else add_xy,
            np.concatenate([ids, pool_ids[candidates]]) if len(ids) else pool_ids[candidates],
            np.concatenate([xyz, pool_xyz[candidates]]) if len(xyz) else pool_xyz[candidates],
        )

    intrinsics = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)

    def triangulate_batch(
        seed_xy: np.ndarray,
        seed_pose: np.ndarray,
        cur_xy: np.ndarray,
        cur_pose: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """World points from two known poses. Scale comes from the poses, so a
        VO point inherits the map's scale instead of inventing its own."""

        if seed_xy.shape[0] == 0:
            return np.zeros((0, 3), dtype=float), np.zeros((0,), dtype=bool)
        p_seed = intrinsics @ np.asarray(seed_pose, dtype=float)
        p_cur = intrinsics @ np.asarray(cur_pose, dtype=float)
        homogeneous = cv2.triangulatePoints(
            np.ascontiguousarray(p_seed, dtype=np.float64),
            np.ascontiguousarray(p_cur, dtype=np.float64),
            np.ascontiguousarray(seed_xy.T, dtype=np.float64),
            np.ascontiguousarray(cur_xy.T, dtype=np.float64),
        )
        w = homogeneous[3]
        good = np.abs(w) > 1e-12
        world = np.zeros((seed_xy.shape[0], 3), dtype=float)
        world[good] = (homogeneous[:3, good] / w[good]).T
        for pose, xy in ((seed_pose, seed_xy), (cur_pose, cur_xy)):
            pose = np.asarray(pose, dtype=float)
            camera = world @ pose[:3, :3].T + pose[:3, 3]
            depth = camera[:, 2]
            good &= depth > 1e-3
            safe = np.where(good, depth, 1.0)
            u = fx * camera[:, 0] / safe + cx
            v = fy * camera[:, 1] / safe + cy
            good &= np.hypot(u - xy[:, 0], v - xy[:, 1]) <= VO_MAX_REPROJ_PX
        # parallax: a nearly-zero angle triangulates to noise
        seed_center = _center(seed_pose)
        cur_center = _center(cur_pose)
        ray_a = world - seed_center
        ray_b = world - cur_center
        norm = np.linalg.norm(ray_a, axis=1) * np.linalg.norm(ray_b, axis=1)
        cosine = np.einsum("ij,ij->i", ray_a, ray_b) / np.where(norm > 0, norm, 1.0)
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        good &= angle >= VO_MIN_PARALLAX_DEG
        return world, good

    def detect_features(gray: np.ndarray, occupied: np.ndarray) -> np.ndarray:
        mask = np.full(gray.shape, 255, dtype=np.uint8)
        for point in occupied:
            cv2.circle(mask, (int(point[0]), int(point[1])), VO_MIN_DISTANCE, 0, -1)
        corners = cv2.goodFeaturesToTrack(
            gray, maxCorners=VO_DETECT_CAP, qualityLevel=0.01,
            minDistance=VO_MIN_DISTANCE, mask=mask,
        )
        if corners is None:
            return np.zeros((0, 2), dtype=float)
        return corners.reshape(-1, 2).astype(float)

    def recover_relative(prev_xy, curr_xy, prev_pose, step_len):
        """Calibrated essential-matrix step. Translation magnitude from recent motion."""
        if prev_xy is None or curr_xy is None:
            return None, 0
        prev_xy = np.asarray(prev_xy, dtype=float).reshape(-1, 2)
        curr_xy = np.asarray(curr_xy, dtype=float).reshape(-1, 2)
        if len(prev_xy) < 8 or len(curr_xy) != len(prev_xy):
            return None, 0
        matrix, mask = cv2.findEssentialMat(
            prev_xy, curr_xy, cameraMatrix=intrinsics,
            method=cv2.RANSAC, prob=0.999, threshold=1.0,
        )
        if matrix is None:
            return None, 0
        n_inliers, rotation, translation, _ = cv2.recoverPose(
            matrix, prev_xy, curr_xy, cameraMatrix=intrinsics, mask=mask,
        )
        n_inliers = int(n_inliers)
        if n_inliers < 8:
            return None, n_inliers
        rotation = np.asarray(rotation, dtype=float)
        translation = np.asarray(translation, dtype=float).reshape(3)
        norm = float(np.linalg.norm(translation))
        scaled = np.zeros(3) if norm < 1e-9 else translation / norm * float(step_len)
        prev_r = np.asarray(prev_pose, dtype=float)[:3, :3]
        prev_t = np.asarray(prev_pose, dtype=float)[:3, 3]
        pose = np.concatenate(
            [rotation @ prev_r, (rotation @ prev_t + scaled).reshape(3, 1)], axis=1
        )
        return pose, n_inliers

    def run_window_ba(current_ids, current_xy, current_xyz, current_pose):
        """Huber sliding-window BA. Oldest pose fixed; map points fixed; VO points free."""
        if len(ba_poses) < 3:
            return current_pose, current_xyz
        started = time.perf_counter()
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation

        window = len(ba_poses)
        vo_obs: dict[int, list[tuple[int, np.ndarray]]] = {}
        map_obs: list[tuple[int, np.ndarray, np.ndarray]] = []
        for keyframe, (ids, xy, xyz) in enumerate(zip(ba_ids, ba_xy, ba_xyz)):
            for i, pid in enumerate(ids):
                pid = int(pid)
                if pid < 0:
                    vo_obs.setdefault(pid, []).append((keyframe, xy[i]))
                else:
                    map_obs.append((keyframe, xy[i], xyz[i]))
        vo_ids = [pid for pid, obs in vo_obs.items() if len(obs) >= 2][:80]
        if len(map_obs) > 80:
            map_obs = map_obs[:: max(1, len(map_obs) // 80)][:80]
        n_points = len(vo_ids) + len(map_obs)
        if n_points < 30:
            return current_pose, current_xyz
        vo_xyz0 = np.zeros((len(vo_ids), 3), dtype=float)
        for row, pid in enumerate(vo_ids):
            for ids, xyz in zip(ba_ids, ba_xyz):
                slot = np.flatnonzero(ids == pid)
                if slot.size:
                    vo_xyz0[row] = xyz[slot[0]]
        pose_params = []
        for keyframe in range(1, window):
            pose = ba_poses[keyframe]
            pose_params.append(
                np.concatenate(
                    [Rotation.from_matrix(pose[:3, :3]).as_rotvec(), pose[:3, 3]]
                )
            )
        x0 = np.concatenate(pose_params + ([vo_xyz0.ravel()] if vo_ids else []))

        def unpack(x):
            poses = [ba_poses[0]]
            offset = 0
            for _keyframe in range(1, window):
                rotation = Rotation.from_rotvec(x[offset : offset + 3]).as_matrix()
                translation = x[offset + 3 : offset + 6]
                offset += 6
                poses.append(np.concatenate([rotation, translation.reshape(3, 1)], axis=1))
            points = x[offset:].reshape(-1, 3) if vo_ids else np.zeros((0, 3))
            return poses, points

        def residuals(x):
            poses, points = unpack(x)
            residual = []
            index = {pid: i for i, pid in enumerate(vo_ids)}
            for pid, observations in vo_obs.items():
                if pid not in index:
                    continue
                world = points[index[pid]]
                for keyframe, xy in observations:
                    camera = poses[keyframe][:3, :3] @ world + poses[keyframe][:3, 3]
                    if camera[2] <= 1e-6:
                        residual.extend([10.0, 10.0])
                        continue
                    residual.extend(
                        [
                            fx * camera[0] / camera[2] + cx - xy[0],
                            fy * camera[1] / camera[2] + cy - xy[1],
                        ]
                    )
            for keyframe, xy, world in map_obs:
                camera = poses[keyframe][:3, :3] @ world + poses[keyframe][:3, 3]
                if camera[2] <= 1e-6:
                    residual.extend([10.0, 10.0])
                    continue
                residual.extend(
                    [
                        fx * camera[0] / camera[2] + cx - xy[0],
                        fy * camera[1] / camera[2] + cy - xy[1],
                    ]
                )
            return np.asarray(residual, dtype=float)

        try:
            solution = least_squares(
                residuals, x0, loss="huber", f_scale=2.0,
                max_nfev=6, ftol=1e-3, xtol=1e-3,
            )
            poses, points = unpack(solution.x)
        except Exception:
            return current_pose, current_xyz
        if (time.perf_counter() - started) * 1000.0 > VO_BA_MAX_MS * 8:
            return current_pose, current_xyz
        for i, pose in enumerate(poses):
            ba_poses[i] = pose
        out_xyz = np.asarray(current_xyz, dtype=float).copy()
        index = {pid: i for i, pid in enumerate(vo_ids)}
        for i, pid in enumerate(current_ids):
            if int(pid) in index:
                out_xyz[i] = points[index[int(pid)]]
        vo_stats["window_ba"] = int(vo_stats.get("window_ba", 0)) + 1
        return poses[-1], out_xyz


    vo_xy = np.zeros((0, 2), dtype=float)
    vo_seed_xy = np.zeros((0, 2), dtype=float)
    vo_bid = np.zeros((0,), dtype=np.int64)
    vo_pose_by_bid: dict[int, np.ndarray] = {}
    vo_bid_next = 0
    vo_next_id = -1
    vo_stats = {
        "triangulated": 0, "refined": 0, "keyframes": 0, "vo_only_frames": 0, "window_ba": 0,
    }
    vo_tracks: dict[int, list[tuple[int, np.ndarray]]] = {}
    vo_history: list[np.ndarray] = []
    hole_range: tuple[int, int] | None = None
    if SYNTH_HOLE:
        start_text, length_text = SYNTH_HOLE.split(":")
        hole_range = (int(start_text), int(start_text) + int(length_text))
    last_pose: np.ndarray | None = None
    last_bgr: np.ndarray | None = None
    last_pose_ordinal = -10**9
    last_map_ordinal = -1
    dead_reckon_age = 0
    step_history: list[float] = []
    dead_reckon_stats = {"frames": 0, "longest_run": 0, "current_run": 0, "inliers": []}
    reentry_fits: list[dict] = []
    ba_poses: list[np.ndarray] = []
    ba_ids: list[np.ndarray] = []
    ba_xy: list[np.ndarray] = []
    ba_xyz: list[np.ndarray] = []

    for ordinal, path in enumerate(frames):
        t0 = time.perf_counter()
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"unreadable frame {path}")
        t_decode = time.perf_counter()
        matched_landmarks_m: dict[str, list] | None = (
            {"point_ids": [], "query_xy": [], "optical_z_m": [], "range_m": []}
            if meters_per_map_unit is not None
            else None
        )

        # 1. fast loop: carry the live map-point set and the un-triangulated VO
        #    candidates one frame forward in a single tracker call
        n_live = len(live_xy)
        combined = np.concatenate([live_xy, vo_xy]) if vo_xy.size else live_xy
        fwd_all, keep_all = tracker.track(bgr, combined)
        if combined.size:
            ok_all = keep_all & _inside(fwd_all)
            inside, ok_vo = ok_all[:n_live], ok_all[n_live:]
            dr_prev = combined[ok_all] if DEAD_RECKON else None
            dr_curr = fwd_all[ok_all] if DEAD_RECKON else None
            live_xy = fwd_all[:n_live][inside]
            live_ids = live_ids[inside]
            live_xyz = live_xyz[inside]
            vo_xy = fwd_all[n_live:][ok_vo]
            vo_seed_xy = vo_seed_xy[ok_vo]
            vo_bid = vo_bid[ok_vo]
        else:
            dr_prev = None
            dr_curr = None
        t_track = time.perf_counter()

        # 2. worker handover: chain the anchor through the frames the worker
        #    spent relocalizing, then MERGE. A handover must never shrink a
        #    healthy track -- that was the whole failure mode of replacing it.
        reseed_ms = 0.0
        reseeded = False
        handover_points = 0
        if pending is not None and ordinal >= pending["ready_frame"]:
            r0 = time.perf_counter()
            xy_chain = np.asarray(pending["xy"], dtype=float)
            ids_chain = np.asarray(pending["ids"], dtype=np.int64)
            chain = [*pending["frames"], bgr]
            for step_index in range(len(chain) - 1):
                if xy_chain.size == 0:
                    break
                fwd, keep = tracker.track_pair(chain[step_index], chain[step_index + 1], xy_chain)
                inside = keep & _inside(fwd)
                xy_chain, ids_chain = fwd[inside], ids_chain[inside]
            xyz_chain, present = lookup_xyz(ids_chain)
            xy_chain, ids_chain, xyz_chain = (
                xy_chain[present], ids_chain[present], xyz_chain[present],
            )
            if ids_chain.size >= RESEED_MIN_POINTS:
                # Strong handover: REPLACE. Merging lets an old drifted majority
                # outvote the fresh map-derived 2D, so relocalization stops
                # correcting anything -- measured, not hypothetical.
                live_xy, live_ids, live_xyz = xy_chain, ids_chain, xyz_chain
            elif ids_chain.size:
                fresh = ~np.isin(ids_chain, live_ids)
                xy_chain, ids_chain, xyz_chain = (
                    xy_chain[fresh], ids_chain[fresh], xyz_chain[fresh],
                )
                live_xy = np.concatenate([live_xy, xy_chain]) if live_xy.size else xy_chain
                live_ids = np.concatenate([live_ids, ids_chain]) if live_ids.size else ids_chain
                live_xyz = np.concatenate([live_xyz, xyz_chain]) if live_xyz.size else xyz_chain
            observed = [
                provider._observations[name][1]
                for name in pending["refs"]
                if name in provider._observations
            ]
            if observed:
                pool_ids = np.unique(np.concatenate(observed).astype(np.int64))
                pool_xyz, pool_present = lookup_xyz(pool_ids)
                pool_ids, pool_xyz = pool_ids[pool_present], pool_xyz[pool_present]
            handover_points = int(ids_chain.size)
            reseed_ms = (time.perf_counter() - r0) * 1000.0
            reseeded = True
            pending = None

        status = "NO_POSE"
        inliers = 0
        center = None
        cam_from_world = None
        if len(live_xy) >= MIN_PNP_POINTS:
            answer = pycolmap.estimate_and_refine_absolute_pose(
                live_xy, live_xyz, query_cam, pnp_estimation, pnp_refinement
            )
            if answer is not None:
                mask = np.asarray(answer["inlier_mask"], dtype=bool)
                inliers = int(mask.sum())
                if inliers >= MIN_PNP_INLIERS:
                    pose = answer["cam_from_world"]
                    matrix = np.eye(4)
                    matrix[:3, :4] = pose.matrix()
                    cam_from_world = matrix[:3, :4]
                    center = _center(matrix)
                    map_share = int(np.count_nonzero(live_ids[mask] > 0))
                    if not reseeded:
                        status = "FAST_TRACK" if map_share >= MIN_PNP_INLIERS else "VO_ONLY"
                    else:
                        status = "RELOC_SEED"
                    if meters_per_map_unit is not None and status in ("FAST_TRACK", "RELOC_SEED"):
                        matched_landmarks_m = metric_landmarks(
                            cam_from_world,
                            live_ids[mask],
                            live_xy[mask],
                            live_xyz[mask],
                            meters_per_map_unit=meters_per_map_unit,
                        )
                    # keep only the geometric inliers, then refill from the map
                    live_xy, live_ids, live_xyz = live_xy[mask], live_ids[mask], live_xyz[mask]
                    t_pnp = time.perf_counter()
                    live_xy, live_ids, live_xyz = topup(
                        cam_from_world, live_xy, live_ids, live_xyz
                    )
                    t_refill = time.perf_counter()
        if cam_from_world is None:
            if (
                DEAD_RECKON
                and last_pose is not None
                and last_bgr is not None
                and dead_reckon_age < DEAD_RECKON_MAX_FRAMES
            ):
                prev_xy, curr_xy = dr_prev, dr_curr
                if last_pose_ordinal != ordinal - 1 or prev_xy is None or len(prev_xy) < 8:
                    extra = detect_features(
                        cv2.cvtColor(last_bgr, cv2.COLOR_BGR2GRAY),
                        np.zeros((0, 2), dtype=float),
                    )
                    if extra.size:
                        fwd, keep = tracker.track_pair(last_bgr, bgr, extra)
                        ok = keep & _inside(fwd)
                        prev_xy, curr_xy = extra[ok], fwd[ok]
                step_len = float(np.median(step_history[-5:])) if step_history else 0.0
                pose, n_inl = recover_relative(prev_xy, curr_xy, last_pose, step_len)
                if pose is not None:
                    cam_from_world = pose
                    center = _center(pose)
                    status = "DEAD_RECKON"
                    inliers = n_inl
                    dead_reckon_age += 1
                    dead_reckon_stats["frames"] += 1
                    dead_reckon_stats["current_run"] += 1
                    longest = dead_reckon_stats["longest_run"]
                    if dead_reckon_stats["current_run"] > longest:
                        dead_reckon_stats["longest_run"] = dead_reckon_stats["current_run"]
                    dead_reckon_stats["inliers"].append(n_inl)
            t_pnp = time.perf_counter()
            t_refill = t_pnp
            if status != "DEAD_RECKON":
                dead_reckon_stats["current_run"] = 0
        else:
            dead_reckon_age = 0
            dead_reckon_stats["current_run"] = 0

        # 3. VO keyframes: candidates are seeded every keyframe and triangulated
        #    VO_BATCH_LAG keyframes later, because one keyframe of drone motion
        #    is not enough parallax to triangulate anything (measured: 9 points
        #    survived in 57 keyframes at lag 1). Scale comes from the poses, so
        #    a VO point is in map units. Nothing here writes to the map.
        if VO and cam_from_world is not None and ordinal % VO_KEYFRAME_STRIDE == 0:
            vo_stats["keyframes"] += 1
            mature = vo_bid <= vo_bid_next - VO_BATCH_LAG
            for bid in np.unique(vo_bid[mature]) if mature.any() else ():
                rows_bid = vo_bid == bid
                world, good = triangulate_batch(
                    vo_seed_xy[rows_bid], vo_pose_by_bid[int(bid)],
                    vo_xy[rows_bid], cam_from_world,
                )
                if good.any():
                    add_xy = vo_xy[rows_bid][good]
                    add_xyz = world[good]
                    add_ids = np.arange(
                        vo_next_id, vo_next_id - int(good.sum()), -1, dtype=np.int64
                    )
                    vo_next_id -= int(good.sum())
                    live_xy = np.concatenate([live_xy, add_xy]) if live_xy.size else add_xy
                    live_ids = np.concatenate([live_ids, add_ids]) if live_ids.size else add_ids
                    live_xyz = np.concatenate([live_xyz, add_xyz]) if live_xyz.size else add_xyz
                    vo_stats["triangulated"] += int(good.sum())
                vo_pose_by_bid.pop(int(bid), None)
            keep_vo = ~mature
            vo_xy, vo_seed_xy, vo_bid = vo_xy[keep_vo], vo_seed_xy[keep_vo], vo_bid[keep_vo]
            if len(live_xy) < VO_MIN_LIVE:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                occupied = np.concatenate([live_xy, vo_xy]) if vo_xy.size else live_xy
                seeded = detect_features(gray, occupied)
                if seeded.size:
                    vo_pose_by_bid[vo_bid_next] = np.asarray(cam_from_world, dtype=float).copy()
                    vo_xy = np.concatenate([vo_xy, seeded]) if vo_xy.size else seeded
                    vo_seed_xy = (
                        np.concatenate([vo_seed_xy, seeded.copy()])
                        if vo_seed_xy.size
                        else seeded.copy()
                    )
                    new_bid = np.full(len(seeded), vo_bid_next, dtype=np.int64)
                    vo_bid = np.concatenate([vo_bid, new_bid]) if vo_bid.size else new_bid
                    vo_bid_next += 1
            if VO_REFINE:
                # Structure-only local BA: a point born from a 2-view, ~0.5 deg
                # baseline is noise, and that noise is what the next PnP turns
                # into pose drift. Re-solve each VO point from every keyframe
                # that has seen it, poses fixed (they are the only absolute
                # thing we have). Cheap: one 4x4 eigen-solve per point.
                vo_history.append(np.asarray(cam_from_world, dtype=float).copy())
                is_vo = live_ids < 0
                for row in np.flatnonzero(is_vo):
                    track = vo_tracks.setdefault(int(live_ids[row]), [])
                    track.append((len(vo_history) - 1, live_xy[row].copy()))
                    if len(track) > VO_REFINE_VIEWS:
                        del track[0]
                live_set = set(int(i) for i in live_ids[is_vo])
                for dead in [key for key in vo_tracks if key not in live_set]:
                    del vo_tracks[dead]
                refined = 0
                for row in np.flatnonzero(is_vo):
                    track = vo_tracks[int(live_ids[row])]
                    if len(track) < 3:
                        continue
                    normal = np.zeros((4, 4), dtype=float)
                    for pose_index, xy in track:
                        projection = intrinsics @ vo_history[pose_index]
                        dlt = np.stack(
                            [
                                xy[0] * projection[2] - projection[0],
                                xy[1] * projection[2] - projection[1],
                            ]
                        )
                        normal += dlt.T @ dlt
                    _values, vectors = np.linalg.eigh(normal)
                    solution = vectors[:, 0]
                    if abs(solution[3]) < 1e-12:
                        continue
                    candidate = solution[:3] / solution[3]
                    camera = np.asarray(cam_from_world, dtype=float)
                    in_camera = camera[:3, :3] @ candidate + camera[:3, 3]
                    if in_camera[2] <= 1e-3:
                        continue
                    u = fx * in_camera[0] / in_camera[2] + cx
                    v = fy * in_camera[1] / in_camera[2] + cy
                    if np.hypot(u - live_xy[row, 0], v - live_xy[row, 1]) > VO_MAX_REPROJ_PX:
                        continue
                    live_xyz[row] = candidate
                    refined += 1
                vo_stats["refined"] += refined
                if len(vo_history) > VO_REFINE_VIEWS * 4:
                    # keep the pose history bounded; drop tracks that point into
                    # the trimmed prefix rather than re-indexing them
                    cut = len(vo_history) - VO_REFINE_VIEWS * 2
                    vo_history = vo_history[cut:]
                    for key, track in list(vo_tracks.items()):
                        kept = [(i - cut, xy) for i, xy in track if i - cut >= 0]
                        if kept:
                            vo_tracks[key] = kept
                        else:
                            del vo_tracks[key]
            if VO_WINDOW_BA:
                ba_poses.append(np.asarray(cam_from_world, dtype=float).copy())
                ba_ids.append(np.asarray(live_ids, dtype=np.int64).copy())
                ba_xy.append(np.asarray(live_xy, dtype=float).copy())
                ba_xyz.append(np.asarray(live_xyz, dtype=float).copy())
                while len(ba_poses) > VO_BA_WINDOW:
                    ba_poses.pop(0)
                    ba_ids.pop(0)
                    ba_xy.pop(0)
                    ba_xyz.pop(0)
                cam_from_world, live_xyz = run_window_ba(
                    live_ids, live_xy, live_xyz, cam_from_world
                )
                center = _center(cam_from_world)
            if len(live_xy) > TRACK_CAP:
                # PnP and the tracker are both linear in point count, so the cap
                # is the throughput knob. Map points outrank VO points.
                order = np.argsort(-live_ids, kind="stable")[:TRACK_CAP]
                live_xy, live_ids, live_xyz = (
                    live_xy[order], live_ids[order], live_xyz[order],
                )
        if status == "VO_ONLY":
            vo_stats["vo_only_frames"] += 1
        t_vo = time.perf_counter()
        if (
            VO_REENTRY_FIT
            and reseeded
            and status == "RELOC_SEED"
            and last_map_ordinal >= 0
            and center is not None
        ):
            start = last_map_ordinal + 1
            drifted = None
            drifted_i = -1
            for i in range(ordinal - 1, start - 1, -1):
                if rows[i].get("position"):
                    drifted = np.asarray(rows[i]["position"], dtype=float)
                    drifted_i = i
                    break
            if drifted is not None and drifted_i >= start:
                gap = float(np.linalg.norm(center - drifted))
                span = drifted_i - start + 1
                for i in range(start, drifted_i + 1):
                    pos = rows[i].get("position")
                    if pos is None:
                        continue
                    rows[i]["position_raw"] = list(pos)
                    alpha = (i - start + 1) / span
                    corr = np.asarray(pos, dtype=float) + alpha * (center - drifted)
                    rows[i]["position"] = [float(v) for v in corr]
                    if meters_per_map_unit is not None:
                        add_metric_position_fields(rows[i], meters_per_map_unit)
                    frame_i = int(rows[i]["frame_index"])
                    if frame_i in gt_center:
                        rows[i]["position_error"] = float(
                            np.linalg.norm(corr - gt_center[frame_i])
                        )
                reentry_fits.append(
                    {
                        "start": start,
                        "end": drifted_i,
                        "gap_before": round(gap, 4),
                        "gap_after": 0.0,
                        "n_corrected": drifted_i - start + 1,
                    }
                )
        if status in ("FAST_TRACK", "RELOC_SEED"):
            last_map_ordinal = ordinal

        # --- background relocalizer scheduling -------------------------------
        in_hole = bool(hole_range) and hole_range[0] <= ordinal < hole_range[1]
        if in_hole:
            pending = None
        need_reloc = (
            not in_hole
            and pending is None
            and (
                len(live_ids) < RELOC_MIN_POINTS
                or status == "NO_POSE"
                or (ordinal - last_reloc_frame) * frame_interval >= RELOC_PERIOD_S
            )
        )
        if need_reloc:
            query_id = f"replay_{path.stem}"
            r0 = time.perf_counter()
            retrieval_ms = 0.0
            if query_id not in provider._query_descriptors:
                m0 = time.perf_counter()
                extracted = extract_megaloc_descriptors(megaloc_runtime, [path], batch_size=1)
                retrieval_ms = (time.perf_counter() - m0) * 1000.0
                provider._query_descriptors.update(dict(zip([query_id], extracted, strict=True)))
            result = provider.localize(
                EDMQuery(
                    query_id=query_id,
                    session_id=f"replay_{ROUTE}",
                    timestamp=ordinal * frame_interval,
                    image_path=str(path),
                    pose_provenance="NONE",
                ),
                index,
            )
            reloc_ms_5090 = (time.perf_counter() - r0) * 1000.0
            reloc_ms_5060 = reloc_ms_5090 * SCALE_5090_TO_5060
            anchor = provider.last_track_anchor
            last_reloc_frame = ordinal
            match_ms = float(getattr(result, "runtime_edm", 0.0) or 0.0) * 1000.0
            pnp_ms = float(getattr(result, "runtime_pnp", 0.0) or 0.0) * 1000.0
            sparse_ms = provider.sparse_stage_ms() if hasattr(provider, "sparse_stage_ms") else {}
            event = {
                "frame": ordinal,
                "status": str(result.query_status),
                "ms_5090": round(reloc_ms_5090, 1),
                "ms_5060_est": round(reloc_ms_5060, 1),
                "edm_ms_5090": round(match_ms, 1),
                "match_ms_5090": round(match_ms, 1),
                "pnp_ms_5090": round(pnp_ms, 1),
                "retrieval_ms_5090": round(float(retrieval_ms), 1),
                "query_extract_ms_5090": round(float(sparse_ms.get("query_extract", 0.0)), 1),
                "ref_extract_ms_5090": round(float(sparse_ms.get("ref_extract", 0.0)), 1),
                "matcher_ms_5090": round(float(sparse_ms.get("matcher", 0.0)), 1),
                "anchor_points": 0 if anchor is None else int(anchor.point3d_ids.size),
            }
            if anchor is not None and anchor.point3d_ids.size >= MIN_PNP_POINTS:
                ready = ordinal + int(math.ceil(reloc_ms_5060 / 1000.0 / frame_interval))
                ids = np.asarray(anchor.point3d_ids, dtype=np.int64)
                xy = np.asarray(anchor.xy, dtype=float)
                _xyz, usable = lookup_xyz(ids)
                ids, xy = ids[usable], xy[usable]
                if len(ids) > TRACK_CAP:
                    take = np.linspace(0, len(ids) - 1, TRACK_CAP).astype(int)
                    ids, xy = ids[take], xy[take]
                pending = {
                    "ready_frame": ready,
                    "xy": xy,
                    "ids": ids,
                    "path": path,
                    "refs": tuple(str(r) for r in (anchor.reference_names or ())),
                    # the worker keeps tracking the frames it is busy through, so
                    # its anchor arrives at the current frame, not a stale one
                    "frames": [bgr],
                }
                event["ready_frame"] = ready
                event["scheduled_points"] = int(len(ids))
            reloc_events.append(event)
        t_end = time.perf_counter()

        stage_ms["decode"].append((t_decode - t0) * 1000.0)
        stage_ms["track"].append((t_track - t_decode) * 1000.0 - reseed_ms)
        stage_ms["reseed"].append(reseed_ms)
        stage_ms["pnp"].append((t_pnp - t_track) * 1000.0)
        stage_ms["refill"].append((t_refill - t_pnp) * 1000.0)
        stage_ms["vo"].append((t_vo - t_refill) * 1000.0)
        stage_ms["loop"].append((t_vo - t0) * 1000.0)
        if pending is not None and not reseeded:
            pending["frames"].append(bgr)

        step = None
        if center is not None and prev_center is not None:
            step = float(np.linalg.norm(center - prev_center))
        if center is not None:
            prev_center = center
        if step is not None:
            step_history.append(step)
            if len(step_history) > 15:
                del step_history[:-15]
        if cam_from_world is not None:
            last_pose = np.asarray(cam_from_world, dtype=float).copy()
            last_bgr = bgr
            last_pose_ordinal = ordinal

        frame_index = int(path.stem.split("_")[-1])
        row = {
            "ordinal": ordinal,
            "frame_index": frame_index,
            "status": status,
            "n_live": int(len(live_ids)),
            "handover_points": handover_points,
            "catchup_ms_5090": round(reseed_ms, 2),
            "pnp_inliers": inliers,
            "step": step,
            "loop_ms_5090": round((t_vo - t0) * 1000.0, 2),
            "reloc_ms_5090": round((t_end - t_pnp) * 1000.0, 2) if need_reloc else 0.0,
            "position": None if center is None else [float(v) for v in center],
            "dead_reckon_age": dead_reckon_age,
        }
        if meters_per_map_unit is not None:
            add_metric_position_fields(row, meters_per_map_unit)
            row["matched_landmarks_m"] = matched_landmarks_m
        if frame_index in gt_center and center is not None:
            row["position_error"] = float(np.linalg.norm(center - gt_center[frame_index]))
            truth = gt_pose[frame_index][:3, :3]
            delta = np.asarray(cam_from_world)[:3, :3].T @ truth
            cos = (float(np.trace(delta)) - 1.0) / 2.0
            row["rotation_error_deg"] = float(np.degrees(np.arccos(max(-1.0, min(1.0, cos))))) 
        elif frame_index in gt_center:
            row["position_error"] = None
        rows.append(row)
        if ordinal % 100 == 0:
            print(
                f"[{ordinal}/{len(frames)}] {status} live={len(live_ids)} inl={inliers} "
                f"loop={row['loop_ms_5090']}ms",
                flush=True,
            )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "frames.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )

    gpu_stage = {"track": tracker.kind == "gpu", "reseed": tracker.kind == "gpu",
                 "decode": False, "pnp": False, "refill": False, "vo": False,
                 "loop": False}
    stages = {}
    for key, values in stage_ms.items():
        array = np.asarray(values, dtype=float)
        median = float(np.median(array))
        p90 = float(np.quantile(array, 0.9))
        scale = SCALE_5090_TO_5060 if gpu_stage[key] else 1.0
        stages[key] = {
            "median_ms_5090": round(median, 2),
            "p90_ms_5090": round(p90, 2),
            "kind": "gpu" if gpu_stage[key] else "cpu",
            "median_ms_5060_est": round(median * scale, 2),
        }
    loop_5060 = sum(
        stages[k]["median_ms_5060_est"] for k in ("decode", "track", "pnp", "refill", "vo")
    )
    loop_5060_pessimistic = stages["loop"]["median_ms_5090"] * SCALE_5090_TO_5060
    statuses = Counter(r["status"] for r in rows)
    posed = [r for r in rows if r["position"] is not None]
    errors = np.asarray(
        [r["position_error"] for r in rows if r.get("position_error") is not None], dtype=float
    )
    rotations = np.asarray(
        [r["rotation_error_deg"] for r in rows if r.get("rotation_error_deg") is not None],
        dtype=float,
    )
    steps = np.asarray([r["step"] for r in rows if r["step"] is not None], dtype=float)
    gt_frames = [r for r in rows if r["frame_index"] in gt_center]
    summary = {
        "route": ROUTE,
        "protocol": "deployment_replay_two_rate",
        "tracker": tracker.name,
        "flow_res": FLOW_RES if tracker.name == "neuflow" else None,
        "track_cap": TRACK_CAP,
        "stream_fps": STREAM_FPS,
        "excluded_sessions": sorted(excluded),
        "reference_bank": bank_kind,
        "dense_frames": len(frames),
        "image_size": [IMAGE_W, IMAGE_H],
        "status_counts": dict(statuses),
        "pose_coverage": round(len(posed) / len(rows), 4),
        "stages_ms": stages,
        "loop_ms_5060_est": round(loop_5060, 2),
        "fps_5060_est": round(1000.0 / loop_5060, 2),
        "loop_ms_5060_pessimistic": round(loop_5060_pessimistic, 2),
        "fps_5060_pessimistic": round(1000.0 / loop_5060_pessimistic, 2),
        "fps_5090_measured": round(1000.0 / stages["loop"]["median_ms_5090"], 2),
        "stream_interval_ms": round(1000.0 * frame_interval, 2),
        "realtime_at_stream_5060_est": loop_5060 <= 1000.0 * frame_interval,
        "reloc": {
            "n_jobs": len(reloc_events),
            "statuses": dict(Counter(e["status"] for e in reloc_events)),
            "median_ms_5090": round(float(np.median([e["ms_5090"] for e in reloc_events])), 1)
            if reloc_events
            else None,
            "median_ms_5060_est": round(
                float(np.median([e["ms_5060_est"] for e in reloc_events])), 1
            )
            if reloc_events
            else None,
            "seeded_jobs": sum(1 for e in reloc_events if "ready_frame" in e),
            "duty_cycle_5060_est": round(
                sum(e["ms_5060_est"] for e in reloc_events) / (len(frames) * frame_interval * 1000.0),
                3,
            )
            if reloc_events
            else None,
            "matcher": cfg.reloc_matcher,
            "match_ms_median": round(float(np.median([e.get("match_ms_5090", 0.0) for e in reloc_events])), 1)
            if reloc_events
            else None,
            "pnp_ms_median": round(float(np.median([e.get("pnp_ms_5090", 0.0) for e in reloc_events])), 1)
            if reloc_events
            else None,
            "retrieval_ms_median": round(float(np.median([e.get("retrieval_ms_5090", 0.0) for e in reloc_events])), 1)
            if reloc_events
            else None,
            "matcher_ms_median": round(float(np.median([e.get("matcher_ms_5090", 0.0) for e in reloc_events])), 1)
            if reloc_events
            else None,
            "ref_extract_ms_median": round(float(np.median([e.get("ref_extract_ms_5090", 0.0) for e in reloc_events])), 1)
            if reloc_events
            else None,
            "query_extract_ms_median": round(float(np.median([e.get("query_extract_ms_5090", 0.0) for e in reloc_events])), 1)
            if reloc_events
            else None,
        },
        "gt": {
            "n_gt_frames": len(gt_frames),
            "n_scored": int(errors.size),
            "position_error_p50": round(float(np.median(errors)), 4) if errors.size else None,
            "position_error_p90": round(float(np.quantile(errors, 0.9)), 4) if errors.size else None,
            "position_error_max": round(float(errors.max()), 4) if errors.size else None,
            "rotation_error_deg_p50": round(float(np.median(rotations)), 3)
            if rotations.size
            else None,
            "rotation_error_deg_p90": round(float(np.quantile(rotations, 0.9)), 3)
            if rotations.size
            else None,
        },
        "step_p50": round(float(np.median(steps)), 5) if steps.size else None,
        "vo": {
            **vo_stats,
            "enabled": VO,
            "keyframe_stride": VO_KEYFRAME_STRIDE,
            "window_ba": VO_WINDOW_BA,
        },
        "dead_reckon": {
            "enabled": DEAD_RECKON,
            "frames": dead_reckon_stats["frames"],
            "longest_run": dead_reckon_stats["longest_run"],
            "median_inliers": (
                round(float(np.median(dead_reckon_stats["inliers"])), 1)
                if dead_reckon_stats["inliers"]
                else None
            ),
        },
        "reentry_fits": reentry_fits,
        "synthetic_hole": list(hole_range) if hole_range else None,
        "step_p90": round(float(np.quantile(steps, 0.9)), 5) if steps.size else None,
        "scale_note": (
            "GPU stages x4 (bench/rt5060.json); CPU stages unscaled (same host class). "
            "Pessimistic row scales the whole loop x4."
        ),
        "reloc_matcher": cfg.reloc_matcher,
    }
    if metric_sidecar is not None:
        summary["metric_scale"] = {
            "method": metric_sidecar["method"],
            "status": metric_sidecar["status"],
            "meters_per_map_unit": float(metric_sidecar["meters_per_map_unit"]),
            "model_hashes": metric_sidecar["model_hashes"],
            "request_sha256": metric_sidecar["request_sha256"],
            "sidecar_path": str(Path(cfg.metric_scale).resolve()),
        }
    write_json(OUT / "summary.json", summary)
    (OUT / "reloc_events.json").write_text(json.dumps(reloc_events, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


__all__ = ["DeployConfig", "KLTTracker", "run_deploy_loop"]
