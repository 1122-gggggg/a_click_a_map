"""EXPERIMENTAL bakeoff-only temporal bridge: SEA-RAFT dense flow as KLT drop-in.

Does NOT modify deployment_localizer.py. All switches env-gated:
  P172_TEMPORAL_BACKEND = klt | flow | union   (default klt)
  P172_WEAK_ANCHOR_BIRTH = 0 | 1               (default 0; Arm 3)
  P172_FLOW_SIGMA_MAX = 2.5
  P172_FLOW_ITERS = 4
torch + RAFT are imported lazily inside methods so the base CPU path
(cheng-mapping env, no torch) never pays for them.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SEA_RAFT_ROOT = Path("/home/cihcilab/sfm/runtime/sea_raft")
SEA_RAFT_CKPT = SEA_RAFT_ROOT / "models" / "spring-M" / "model.safetensors"
FLOW_W, FLOW_H = 960, 360

_sea_raft_model: Any = None


def _load_sea_raft() -> Any:
    global _sea_raft_model
    if _sea_raft_model is not None:
        return _sea_raft_model
    import torch

    sys.path.insert(0, str(SEA_RAFT_ROOT))
    sys.path.insert(0, str(SEA_RAFT_ROOT / "core"))
    from types import SimpleNamespace

    from raft import RAFT
    from safetensors.torch import load_file

    args = SimpleNamespace(
        use_var=True, var_min=0, var_max=10, pretrain="resnet34",
        initial_dim=64, block_dims=[64, 128, 256], radius=4, dim=128,
        num_blocks=2, iters=int(os.environ.get("P172_FLOW_ITERS", "4")),
        epsilon=1e-8,
    )
    model = RAFT(args)
    state = load_file(str(SEA_RAFT_CKPT))
    model.load_state_dict(state, strict=False)
    model.to("cuda").eval()
    _sea_raft_model = model
    return model


def _read_rgb(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


SIDEVIEW_WEAK_MIN_INLIERS = 80
SIDEVIEW_WEAK_MAX_P90 = 3.0
SIDEVIEW_WEAK_MIN_OCC30 = 0.040
SIDEVIEW_WEAK_MIN_POS_DEPTH = 0.99
SIDEVIEW_WEAK_MAX_STEP_M = 1.5


def _fix1_sideview_weak(result, prev_position):
    """EXPERIMENTAL Fix1: admit high-inlier sideview poses as WEAK (STRONG untouched).
    Causal: uses only current-frame metrics + previous accepted position."""
    try:
        if result.query_status != "ABSTAINED":
            return result
        if int(result.ransac_inliers or 0) < SIDEVIEW_WEAK_MIN_INLIERS:
            return None
        if float(result.reprojection_p90 or 9.0) > SIDEVIEW_WEAK_MAX_P90:
            return None
        if float(result.occupied_frac_30 or 0.0) < SIDEVIEW_WEAK_MIN_OCC30:
            return None
        metrics_pos = float(getattr(result, "positive_depth_ratio", 1.0) or 1.0)
        if metrics_pos < SIDEVIEW_WEAK_MIN_POS_DEPTH:
            return None
        pos = getattr(result, "estimated_position", None)
        if prev_position is not None and pos is not None:
            step = float(np.linalg.norm(np.asarray(pos) - np.asarray(prev_position)))
            if step > SIDEVIEW_WEAK_MAX_STEP_M:
                return None
        import dataclasses as _dc

        return _dc.replace(result, query_status="POSE_ESTIMATED_WEAK",
                           pose_consistency="ACCEPT_SIDEVIEW_WEAK")
    except Exception:
        return result


TIERB_MIN_INLIERS = 40
TIERB_MAX_P90 = 3.0
TIERB_MAX_STEP_PER_FRAME_M = 0.12
TIERB_MAX_GAP_FRAMES = 3


def _tierb_kinematic_weak(result, prev_position, gap_frames: int):
    """EXPERIMENTAL Tier-B: marginal-support pose admitted as WEAK only when it is
    kinematically continuous with the last accepted pose.

    Substitutes motion continuity for raw inlier mass. Cannot bootstrap: requires a
    recent accepted pose (gap <= TIERB_MAX_GAP_FRAMES). Frozen STRONG gates untouched.
    """
    try:
        if result.query_status != "ABSTAINED":
            return result
        if prev_position is None or gap_frames < 1 or gap_frames > TIERB_MAX_GAP_FRAMES:
            return None
        if int(result.ransac_inliers or 0) < TIERB_MIN_INLIERS:
            return None
        if float(result.reprojection_p90 or 9.0) > TIERB_MAX_P90:
            return None
        pos = getattr(result, "estimated_position", None)
        if pos is None:
            return None
        step = float(np.linalg.norm(np.asarray(pos) - np.asarray(prev_position)))
        if step > TIERB_MAX_STEP_PER_FRAME_M * gap_frames:
            return None
        import dataclasses as _dc

        return _dc.replace(result, query_status="POSE_ESTIMATED_WEAK",
                           pose_consistency="ACCEPT_TIERB_KINEMATIC")
    except Exception:
        return result


def _uncertainty_sigma(info: Any, var_max: float = 10.0) -> Any:
    import torch

    weight = info[:, :2].softmax(dim=1)
    log_b = torch.zeros_like(info[:, 2:])
    log_b[:, 0] = torch.clamp(info[:, 2], min=0, max=var_max)
    log_b[:, 1] = torch.clamp(info[:, 3], min=0, max=0)
    return (log_b * weight).sum(dim=1, keepdim=True).exp()


def flow_track_anchor(anchor_image: Path, anchor_xy: np.ndarray, point3d_ids: np.ndarray,
                      query_path: Path, *, sigma_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Track anchor map-points into query with SEA-RAFT. Returns (fwd_xy, sigma, keep)."""
    import cv2
    import torch

    model = _load_sea_raft()
    prev = _read_rgb(Path(anchor_image))
    curr = _read_rgb(Path(query_path))
    h0, w0 = prev.shape[:2]
    hq, wq = curr.shape[:2]
    prev_r = cv2.resize(prev, (FLOW_W, FLOW_H), interpolation=cv2.INTER_AREA)
    curr_r = cv2.resize(curr, (FLOW_W, FLOW_H), interpolation=cv2.INTER_AREA)
    sx, sy = FLOW_W / w0, FLOW_H / h0
    qx = np.asarray(anchor_xy, dtype=np.float64).reshape(-1, 2) * np.array([sx, sy])

    t1 = torch.from_numpy(prev_r).permute(2, 0, 1).float().unsqueeze(0).to("cuda")
    t2 = torch.from_numpy(curr_r).permute(2, 0, 1).float().unsqueeze(0).to("cuda")
    with torch.no_grad():
        out = model(t1, t2, iters=model.args.iters, test_mode=True)
    flow = out["flow"][-1][0].float().cpu().numpy()  # (2,H,W)
    sigma = _uncertainty_sigma(out["info"][-1]).float().cpu().numpy()[0, 0]  # (H,W)
    del out, t1, t2

    xi = np.clip(qx[:, 0].astype(np.int64), 0, FLOW_W - 1)
    yi = np.clip(qx[:, 1].astype(np.int64), 0, FLOW_H - 1)
    fwd = qx + np.stack([flow[0][yi, xi], flow[1][yi, xi]], axis=1)
    sig = sigma[yi, xi]
    # map back to native query pixels
    fwd_q = fwd * np.array([wq / FLOW_W, hq / FLOW_H])
    keep = (
        np.isfinite(fwd).all(axis=1)
        & (sig <= sigma_max)
        & (fwd_q[:, 0] >= 0.0) & (fwd_q[:, 1] >= 0.0)
        & (fwd_q[:, 0] < float(wq)) & (fwd_q[:, 1] < float(hq))
        & (np.asarray(point3d_ids).reshape(-1) >= 0)
    )
    return fwd_q, sig, keep


NEUFLOW_ROOT = Path("/home/cihcilab/sfm/runtime/neuflow2")
NEUFLOW_CKPT = NEUFLOW_ROOT / "neuflow_mixed.pth"
_neuflow_model: Any = None
_neuflow_shape: tuple = ()


def _load_neuflow() -> Any:
    global _neuflow_model
    if _neuflow_model is not None:
        return _neuflow_model
    import torch

    sys.path.insert(0, str(NEUFLOW_ROOT))
    from NeuFlow.neuflow import NeuFlow

    model = NeuFlow().to("cuda")
    ckpt = torch.load(str(NEUFLOW_CKPT), map_location="cuda")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    _neuflow_model = model
    return model


def neuflow_track_anchor(anchor_image: Path, anchor_xy: np.ndarray, point3d_ids: np.ndarray,
                         query_path: Path, *, fb_check: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Track with NeuFlow v2 (fast loop). Returns (fwd_xy, err, keep); err=fb px or zeros."""
    global _neuflow_shape
    import cv2
    import torch

    model = _load_neuflow()  # inserts NEUFLOW_ROOT on sys.path

    from data_utils.frame_utils import InputPadder

    prev = _read_rgb(Path(anchor_image))
    curr = _read_rgb(Path(query_path))
    h0, w0 = prev.shape[:2]
    hq, wq = curr.shape[:2]
    t1 = torch.from_numpy(prev).permute(2, 0, 1).float().unsqueeze(0).to("cuda")
    t2 = torch.from_numpy(curr).permute(2, 0, 1).float().unsqueeze(0).to("cuda")
    padder = InputPadder(t1.shape, padding_factor=16)
    t1p, t2p = padder.pad(t1, t2)
    key = (t1p.shape[-2], t1p.shape[-1])
    if _neuflow_shape != key:
        model.init_bhwd(t1p.shape[0], key[0], key[1], "cuda", amp=False)
        _neuflow_shape = key
    with torch.no_grad():
        fwd_pad = model(t1p, t2p)[-1]
        fwd = padder.unpad(fwd_pad)[0].float().cpu().numpy()  # (2,h,w) native anchor res
        if fb_check:
            bwd_pad = model(t2p, t1p)[-1]
            bwd = padder.unpad(bwd_pad)[0].float().cpu().numpy()
    del t1p, t2p, fwd_pad
    qx = np.asarray(anchor_xy, dtype=np.float64).reshape(-1, 2)
    xi = np.clip(qx[:, 0].astype(np.int64), 0, w0 - 1)
    yi = np.clip(qx[:, 1].astype(np.int64), 0, h0 - 1)
    fxy = qx + np.stack([fwd[0][yi, xi], fwd[1][yi, xi]], axis=1)
    # to query native pixels
    fwd_q = fxy * np.array([wq / w0, hq / h0])
    if fb_check:
        bxi = np.clip(fxy[:, 0].astype(np.int64), 0, w0 - 1)
        byi = np.clip(fxy[:, 1].astype(np.int64), 0, h0 - 1)
        back = fxy + np.stack([bwd[0][byi, bxi], bwd[1][byi, bxi]], axis=1)
        err = np.linalg.norm(qx - back, axis=1)
    else:
        err = np.zeros(len(qx))
    keep = (
        np.isfinite(fxy).all(axis=1)
        & ((err <= 2.0) if fb_check else np.ones(len(qx), dtype=bool))
        & (fwd_q[:, 0] >= 0.0) & (fwd_q[:, 1] >= 0.0)
        & (fwd_q[:, 0] < float(wq)) & (fwd_q[:, 1] < float(hq))
        & (np.asarray(point3d_ids).reshape(-1) >= 0)
    )
    return fwd_q, err, keep

LOCOTRACK_ROOT = Path("/home/cihcilab/sfm/runtime/locotrack/locotrack_pytorch")
LOCOTRACK_CKPT = {
    "base": Path("/home/cihcilab/sfm/runtime/locotrack/weights/locotrack_base.ckpt"),
    "small": Path("/home/cihcilab/sfm/runtime/locotrack/weights/locotrack_small.ckpt"),
}
_locotrack_model: Any = None


def _load_locotrack() -> Any:
    """LocoTrack (ECCV 2024) point tracker: pure PyTorch, no custom CUDA kernel."""
    global _locotrack_model
    if _locotrack_model is not None:
        return _locotrack_model
    import torch

    size = os.environ.get("P172_LOCOTRACK_SIZE", "base")
    checkpoint = LOCOTRACK_CKPT[size]
    if str(LOCOTRACK_ROOT) not in sys.path:
        sys.path.insert(0, str(LOCOTRACK_ROOT))
    from models.locotrack_model import load_model

    model = load_model(str(checkpoint), model_size=size)
    _locotrack_model = model.to("cuda").eval()
    return _locotrack_model


def _frame_chain(anchor_image: Path, query_path: Path, max_frames: int) -> list[Path]:
    """Frames from the anchor to the query inclusive, when both are route frames.

    A point tracker is only better than pairwise flow if it sees the frames in
    between; a map reference anchor has no chain, so it degrades to a pair.
    """

    anchor_image, query_path = Path(anchor_image), Path(query_path)
    if max_frames < 2:
        raise ValueError("max_frames must be at least 2")
    if anchor_image.parent != query_path.parent:
        return [anchor_image, query_path]
    siblings = sorted(anchor_image.parent.glob("frame_*.jpg"))
    try:
        start, stop = siblings.index(anchor_image), siblings.index(query_path)
    except ValueError:
        return [anchor_image, query_path]
    if stop <= start:
        return [anchor_image, query_path]
    chain = siblings[start : stop + 1]
    if len(chain) <= max_frames:
        return chain
    if max_frames == 2:
        return [chain[0], chain[-1]]
    # keep both endpoints, subsample the interior deterministically
    interior = chain[1:-1]
    step = (len(interior) - 1) / float(max_frames - 2)
    picked = [interior[int(round(i * step))] for i in range(max_frames - 2)]
    return [chain[0], *picked, chain[-1]]


def locotrack_track_anchor(
    anchor_image: Path,
    anchor_xy: np.ndarray,
    point3d_ids: np.ndarray,
    query_path: Path,
    *,
    sigma_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Track anchor map-points into the query with LocoTrack.

    Same contract as ``flow_track_anchor``: (fwd_xy in query pixels, sigma, keep).
    ``sigma`` is ``sigma_scale * (1 - visibility)`` so the frozen ``sigma_max``
    gate keeps its meaning: the default 2.5 with scale 5.0 is LocoTrack's own
    0.5 visibility threshold.
    """

    import cv2
    import torch

    model = _load_locotrack()
    scale = float(os.environ.get("P172_TRACK_SIGMA_SCALE", "5.0"))
    max_frames = int(os.environ.get("P172_TRACK_CHAIN", "6"))
    res_w, res_h = (
        int(value) for value in os.environ.get("P172_TRACK_RES", f"{FLOW_W}x{FLOW_H}").split("x")
    )
    chain = _frame_chain(anchor_image, query_path, max_frames)
    frames = [_read_rgb(path) for path in chain]
    h0, w0 = frames[0].shape[:2]
    hq, wq = frames[-1].shape[:2]
    resized = np.stack(
        [cv2.resize(frame, (res_w, res_h), interpolation=cv2.INTER_AREA) for frame in frames]
    )
    qx = np.asarray(anchor_xy, dtype=np.float64).reshape(-1, 2) * np.array(
        [res_w / w0, res_h / h0]
    )
    video = torch.from_numpy(resized).to("cuda").unsqueeze(0).float() / 255.0 * 2 - 1
    queries = torch.zeros((1, len(qx), 3), dtype=torch.float32, device="cuda")
    queries[0, :, 1] = torch.from_numpy(qx[:, 1]).float().to("cuda")  # y
    queries[0, :, 2] = torch.from_numpy(qx[:, 0]).float().to("cuda")  # x
    with torch.no_grad():
        out = model(
            video,
            queries,
            query_chunk_size=int(os.environ.get("P172_TRACK_CHUNK", "256")),
        )
        tracks = out["tracks"][0, :, -1, :].float().cpu().numpy()
        # LocoTrack's own visibility rule, kept continuous instead of thresholded
        occluded = torch.sigmoid(out["occlusion"][0, :, -1])
        far = torch.sigmoid(out["expected_dist"][0, :, -1])
        visible = ((1.0 - occluded) * (1.0 - far)).float().cpu().numpy()
    del out, video, queries
    fwd_q = tracks * np.array([wq / res_w, hq / res_h])
    sig = scale * (1.0 - visible)
    keep = (
        np.isfinite(fwd_q).all(axis=1)
        & (sig <= sigma_max)
        & (fwd_q[:, 0] >= 0.0) & (fwd_q[:, 1] >= 0.0)
        & (fwd_q[:, 0] < float(wq)) & (fwd_q[:, 1] < float(hq))
        & (np.asarray(point3d_ids).reshape(-1) >= 0)
    )
    return fwd_q, sig, keep


def make_flow_provider():
    """Build the FlowTemporalBridge subclass (import here: needs deployment_localizer)."""
    from sfm_diagnosis.site_pipeline.deployment_localizer import (
        MIN_TEMPORAL_INLIERS,
        TEMPORAL_REFERENCE_NAME,
        FinalMapEDMProvider,
        MapTrackAnchor,
    )

    class FlowTemporalBridge(FinalMapEDMProvider):
        def __init__(self, *args: Any, temporal_backend: str = "klt",
                     weak_anchor_birth: bool = False, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.temporal_backend = temporal_backend
            self.weak_anchor_birth = weak_anchor_birth
            self._megaloc_lifted: tuple = ()
            self.last_flow_runtime = 0.0
            self._prev_position: tuple | None = None
            self._anchor_history: list = []
            self._frame_count: int = 0
            self._last_accept_frame: int = 0

        def _pack_query_result(self, *args: Any, **kwargs: Any):  # stash megaloc lifted for union
            result = super()._pack_query_result(*args, **kwargs)
            try:
                self._megaloc_lifted = tuple(kwargs.get("lifted", ()))
            except Exception:
                self._megaloc_lifted = ()
            if os.environ.get("P172_SIDEVIEW_WEAK", "0") == "1":
                flipped = _fix1_sideview_weak(result, self._prev_position)
                if flipped is not None:
                    result = flipped
            if os.environ.get("P172_TIERB", "0") == "1" and result.query_status == "ABSTAINED":
                gap = self._frame_count - self._last_accept_frame
                flipped = _tierb_kinematic_weak(result, self._prev_position, gap)
                if flipped is not None:
                    result = flipped
            try:
                if result.query_status != "ABSTAINED" and result.estimated_position is not None:
                    self._prev_position = tuple(result.estimated_position)
                    self._last_accept_frame = self._frame_count
            except Exception:
                pass
            return result

        def _flow_lifted(self, query_path: Path, anchor: MapTrackAnchor):
            from river_map_quality.official_edm_adapter import LiftedMatch

            if anchor.xy.size == 0 or anchor.point3d_ids.size == 0:
                return (), 0
            t0 = time.perf_counter()
            sigma_max = float(os.environ.get("P172_FLOW_SIGMA_MAX", "2.5"))
            tracker = os.environ.get("P172_POINT_TRACKER", "none")
            age_min = int(os.environ.get("P172_TRACK_AGE_MIN", "3"))
            args = (
                Path(anchor.image_path), np.asarray(anchor.xy),
                np.asarray(anchor.point3d_ids), Path(query_path),
            )
            try:
                if tracker == "locotrack":
                    fwd, sig, keep = locotrack_track_anchor(*args, sigma_max=sigma_max)
                elif tracker == "hybrid":
                    # measured split (bench/locotrack_transfer.json): SEA-RAFT keeps
                    # 540/486 of 600 at gap 1/2 but only 55/3 at gap 4/8; LocoTrack
                    # keeps 171/32 there. Young anchors stay on the precise tracker,
                    # old anchors switch to the one that still survives.
                    fwd, sig, keep = flow_track_anchor(*args, sigma_max=sigma_max)
                    if int(anchor.age) >= age_min or int(np.count_nonzero(keep)) < MIN_TEMPORAL_INLIERS:
                        loco_fwd, loco_sig, loco_keep = locotrack_track_anchor(
                            *args, sigma_max=sigma_max
                        )
                        fill = loco_keep & ~keep
                        fwd = np.where(fill[:, None], loco_fwd, fwd)
                        sig = np.where(fill, loco_sig, sig)
                        keep = keep | loco_keep
                else:
                    fwd, sig, keep = flow_track_anchor(*args, sigma_max=sigma_max)
            finally:
                self.last_flow_runtime = time.perf_counter() - t0
            ids = np.asarray(anchor.point3d_ids, dtype=np.int64).reshape(-1)
            idx = np.flatnonzero(keep)
            top_n = int(os.environ.get("P172_FLOW_TOPN", "0"))
            if top_n > 0 and idx.size > top_n:
                # keep the lowest-uncertainty transfers; PnP cost is linear in points
                idx = idx[np.argsort(sig[idx], kind="stable")[:top_n]]
            lifted = tuple(
                LiftedMatch(
                    query_xy=(float(fwd[i][0]), float(fwd[i][1])),
                    point3d_id=int(ids[i]),
                    reference_name=TEMPORAL_REFERENCE_NAME,
                    confidence=float(1.0 / (1.0 + sig[i])),
                    lift_distance_px=float(max(sig[i], 1e-6)),
                )
                for i in idx
            )
            return lifted, int(idx.size)

        def _klt_track_anchor(self, query_path: Path, anchor: MapTrackAnchor):
            if self.temporal_backend == "klt":
                return super()._klt_track_anchor(query_path, anchor)
            if self.temporal_backend == "neuflow":
                return self._neuflow_lifted(query_path, anchor)
            return self._flow_lifted(query_path, anchor)

        def _neuflow_lifted(self, query_path: Path, anchor: MapTrackAnchor):
            from river_map_quality.official_edm_adapter import LiftedMatch

            if anchor.xy.size == 0 or anchor.point3d_ids.size == 0:
                return (), 0
            t0 = time.perf_counter()
            fb = os.environ.get("P172_NEOFLOW_FB", "1") == "1"
            try:
                fwd, err, keep = neuflow_track_anchor(
                    Path(anchor.image_path), np.asarray(anchor.xy),
                    np.asarray(anchor.point3d_ids), Path(query_path), fb_check=fb)
            finally:
                self.last_flow_runtime = time.perf_counter() - t0
            ids = np.asarray(anchor.point3d_ids, dtype=np.int64).reshape(-1)
            lifted = tuple(
                LiftedMatch(
                    query_xy=(float(xy[0]), float(xy[1])),
                    point3d_id=int(pid),
                    reference_name="temporal_anchor",
                    confidence=float(1.0 / (1.0 + e)),
                    lift_distance_px=float(max(e, 1e-6)),
                )
                for xy, pid, e in zip(fwd[keep], ids[keep], err[keep], strict=True)
            )
            return lifted, int(keep.sum())

        def _localize_klt(self, query, query_path: Path, started: float, anchor: MapTrackAnchor):
            if self.temporal_backend == "union":
                return self._localize_union(query, query_path, started, anchor)
            result = super()._localize_klt(query, query_path, started, anchor)
            if result is not None and self.temporal_backend in ("flow", "neuflow"):
                self.last_retrieval_source = self.temporal_backend
            return result

        def _localize_union(self, query, query_path: Path, started: float, anchor: MapTrackAnchor):
            import time as _time

            from sfm_diagnosis.site_pipeline.deployment_localizer import MIN_TEMPORAL_INLIERS as _MIN

            depth = max(1, int(os.environ.get("P172_UNION_DEPTH", "1")))
            hist = self._anchor_history
            if int(anchor.point3d_ids.size) >= _MIN and all(
                    str(a.image_path) != str(anchor.image_path) for a in hist):
                hist.append(anchor)
                del hist[:-depth]
            flow_all: list = []
            for anc in list(hist):
                lifted, _n = self._flow_lifted(query_path, anc)
                flow_all.extend(lifted)
            self.last_n_transferred = len(flow_all)
            merged = tuple(self._megaloc_lifted) + tuple(flow_all)
            if len(merged) < 6:
                return None
            pnp_started = _time.perf_counter()
            result, metrics, decision_status = self._solve(query_path, merged, session_groups=False)
            runtime_pnp = _time.perf_counter() - pnp_started
            n_inliers = 0 if result is None else int(np.count_nonzero(result.inlier_mask))
            if n_inliers < _MIN:
                return None
            self.last_retrieval_source = "union"
            refs: list[str] = []
            for anc in list(hist):
                refs.extend(anc.reference_names)
            return self._pack_query_result(
                query=query, started=started, lifted=merged,
                raw_matches=len(merged), result=result, metrics=metrics,
                decision_status=decision_status,
                reference_ids=tuple(dict.fromkeys(refs)),
                viewpoint_pool_fallback=False,
                runtime_edm=self.last_flow_runtime, runtime_pnp=runtime_pnp,
                temporal_consistency=1.0)

        def localize(self, query, index, *, anchor=None):
            self._frame_count += 1
            return super().localize(query, index, anchor=anchor)

        def _anchor_if_admissible(self, query_path, lifted, inlier_mask, metrics,
                                  decision_status, reference_ids):
            try:
                init_n = int(os.environ.get("P172_INIT_BIRTH_N", "0"))
            except ValueError:
                init_n = 0
            in_init = init_n > 0 and self._frame_count <= init_n
            if (self.weak_anchor_birth and decision_status == "ACCEPT") or in_init:
                ti = int(metrics.get("track_inliers") or 0)
                ratio = float(metrics.get("inlier_ratio") or 0.0)
                if ti >= MIN_TEMPORAL_INLIERS and ratio >= 0.10:
                    base = super()._anchor_if_admissible(
                        query_path, lifted, inlier_mask, metrics, decision_status, reference_ids)
                    if base is not None:
                        return base
                    # relaxed birth: same body, lower bar
                    from sfm_diagnosis.site_pipeline.deployment_localizer import TEMPORAL_REFERENCE_NAME as _T
                    map_refs = tuple(n for n in reference_ids if n and n != _T)
                    if not map_refs:
                        return None
                    xy, ids = [], []
                    for match, keep in zip(lifted, inlier_mask, strict=True):
                        if not bool(keep) or match.point3d_id is None or int(match.point3d_id) < 0:
                            continue
                        xy.append((float(match.query_xy[0]), float(match.query_xy[1])))
                        ids.append(int(match.point3d_id))
                    if len(ids) < MIN_TEMPORAL_INLIERS:
                        return None
                    return MapTrackAnchor(image_path=query_path,
                                          xy=np.asarray(xy, dtype=np.float64),
                                          point3d_ids=np.asarray(ids, dtype=np.int64),
                                          reference_names=map_refs, age=0)
            return super()._anchor_if_admissible(
                query_path, lifted, inlier_mask, metrics, decision_status, reference_ids)

    return FlowTemporalBridge
