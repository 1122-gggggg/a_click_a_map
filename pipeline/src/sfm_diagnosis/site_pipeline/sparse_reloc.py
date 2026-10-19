"""Sparse relocalization front-ends sharing the deployment lift path.

Every backend uses the map's own 2D observations as the reference keypoints
and only samples descriptors at those pixels; the query side keeps its own
detector. The downstream path (lift -> dedup -> PnP -> frozen gates) is
identical across front-ends.

All torch / xfeat / lightglue imports are lazy (inside methods) because
``activate_audited_torch`` forbids torch import before the audited runtime
is configured. Module top level imports only stdlib + numpy.
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np

XFEAT_REPO = Path(os.environ.get("XFEAT_REPO", "/home/cihcilab/sfm/runtime/xfeat"))
DEFAULT_MAX_KEYPOINTS = 4096
DEFAULT_REF_CACHE = 128
SUPPORTED_BACKENDS = ("xfeat_lg", "aliked_lg", "xfeat_mnn")


def anchored_reference_keypoints(
    observation_xy: np.ndarray,
    track_lengths: np.ndarray,
    *,
    native_wh: tuple[int, int],
    work_wh: tuple[int, int],
    cap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-``cap`` observations by descending track length, ties by index.

    Returns (work_xy float32 (M,2) scaled into the work resolution, native_xy
    float64 (M,2) left exactly on the observation pixels).
    """
    obs = np.asarray(observation_xy, dtype=np.float64).reshape(-1, 2)
    lengths = np.asarray(track_lengths, dtype=np.int64).reshape(-1)
    n = int(obs.shape[0])
    if n == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float64),
        )
    if len(lengths) != n:
        raise ValueError("observation_xy and track_lengths length mismatch")
    cap = int(cap)
    if cap <= 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float64),
        )
    order = np.lexsort((np.arange(n), -lengths))[: min(cap, n)]
    native_xy = np.asarray(obs[order], dtype=np.float64)
    native_w, native_h = float(native_wh[0]), float(native_wh[1])
    work_w, work_h = float(work_wh[0]), float(work_wh[1])
    work_xy = np.empty_like(native_xy, dtype=np.float64)
    work_xy[:, 0] = native_xy[:, 0] * (work_w / native_w)
    work_xy[:, 1] = native_xy[:, 1] * (work_h / native_h)
    return work_xy.astype(np.float32), native_xy


class _XFeatBackend:
    """XFeat detector + (LighterGlue | mutual-NN) matcher."""

    def __init__(self, *, max_keypoints: int, use_lighterglue: bool) -> None:
        self._max_keypoints = int(max_keypoints)
        self._use_lighterglue = bool(use_lighterglue)
        self._xfeat = None
        self._lighter = None

    def _ensure(self) -> None:
        if self._xfeat is not None and (not self._use_lighterglue or self._lighter is not None):
            return
        import sys

        repo = str(XFEAT_REPO)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        if self._xfeat is None:
            from modules.xfeat import XFeat

            weights = str(XFEAT_REPO / "weights" / "xfeat.pt")
            self._xfeat = XFeat(weights=weights, top_k=self._max_keypoints)
            try:
                self._xfeat = self._xfeat.eval()
            except Exception:
                pass
        if self._use_lighterglue and self._lighter is None:
            from modules.lighterglue import LighterGlue

            weights = str(XFEAT_REPO / "weights" / "xfeat-lighterglue.pt")
            self._lighter = LighterGlue(weights=weights)
            try:
                self._lighter = self._lighter.eval()
            except Exception:
                pass

    def prepare_query(self, image_bgr: np.ndarray) -> dict:
        import cv2
        import torch

        self._ensure()
        assert self._xfeat is not None
        rgb = cv2.cvtColor(np.ascontiguousarray(image_bgr), cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0
        height, width = rgb.shape[:2]
        with torch.inference_mode():
            out = self._xfeat.detectAndCompute(tensor, top_k=self._max_keypoints)[0]
        return {
            "keypoints": out["keypoints"],
            "descriptors": out["descriptors"],
            "image_size": (width, height),
        }

    def prepare_reference(self, image_bgr: np.ndarray, work_xy: np.ndarray) -> dict:
        import cv2
        import torch
        import torch.nn.functional as F

        self._ensure()
        assert self._xfeat is not None
        work = np.asarray(work_xy, dtype=np.float32).reshape(-1, 2)
        height, width = image_bgr.shape[:2]
        if work.shape[0] == 0:
            dev = str(getattr(self._xfeat, "dev", "cuda"))
            return {
                "keypoints": torch.zeros((0, 2), dtype=torch.float32, device=dev),
                "descriptors": torch.zeros((0, 64), dtype=torch.float32, device=dev),
                "image_size": (width, height),
            }
        rgb = cv2.cvtColor(np.ascontiguousarray(image_bgr), cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0
        with torch.inference_mode():
            x, rh, rw = self._xfeat.preprocess_tensor(tensor)
            m1, _, _ = self._xfeat.net(x)
            m1 = F.normalize(m1, dim=1)
            pts = torch.as_tensor(work, dtype=torch.float32, device="cuda") / torch.tensor(
                [rw, rh], dtype=torch.float32, device="cuda"
            )
            feats = self._xfeat.interpolator(
                m1, pts[None], H=x.shape[-2], W=x.shape[-1]
            )
            feats = F.normalize(feats, dim=-1)[0]
            kpts = torch.as_tensor(work, dtype=torch.float32, device="cuda")
        return {
            "keypoints": kpts,
            "descriptors": feats,
            "image_size": (width, height),
        }

    def match(
        self, query_feat: dict, reference_feat: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch

        qk = query_feat["keypoints"]
        qd = query_feat["descriptors"]
        rk = reference_feat["keypoints"]
        rd = reference_feat["descriptors"]
        qw, qh = query_feat["image_size"]
        rw, rh = reference_feat["image_size"]
        if int(qk.shape[0]) == 0 or int(rk.shape[0]) == 0:
            return (
                np.zeros((0, 2), dtype=np.float64),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float64),
            )
        if self._use_lighterglue:
            self._ensure()
            assert self._lighter is not None
            with torch.inference_mode():
                data = {
                    "keypoints0": qk[None],
                    "descriptors0": qd[None],
                    "image_size0": torch.tensor(
                        [qw, qh], dtype=qk.dtype, device=qk.device
                    )[None],
                    "keypoints1": rk[None],
                    "descriptors1": rd[None],
                    "image_size1": torch.tensor(
                        [rw, rh], dtype=rk.dtype, device=rk.device
                    )[None],
                }
                out = self._lighter(data, min_conf=0.1)
                idx = out["matches"][0]
                scores = out["scores"][0]
                if int(idx.shape[0]) == 0:
                    return (
                        np.zeros((0, 2), dtype=np.float64),
                        np.zeros((0,), dtype=np.int64),
                        np.zeros((0,), dtype=np.float64),
                    )
                q_xy = qk[idx[:, 0]].detach().float().cpu().numpy().astype(np.float64)
                r_idx = idx[:, 1].detach().cpu().numpy().astype(np.int64)
                conf = scores.detach().float().cpu().numpy().astype(np.float64)
                return q_xy, r_idx, conf
        # mutual-NN on cosine similarity
        with torch.inference_mode():
            idx0, idx1 = self._xfeat.match(qd, rd, min_cossim=0.82)
            if int(idx0.shape[0]) == 0:
                return (
                    np.zeros((0, 2), dtype=np.float64),
                    np.zeros((0,), dtype=np.int64),
                    np.zeros((0,), dtype=np.float64),
                )
            q_xy = qk[idx0].detach().float().cpu().numpy().astype(np.float64)
            r_idx = idx1.detach().cpu().numpy().astype(np.int64)
            sim = (qd[idx0] * rd[idx1]).sum(dim=-1)
            conf = sim.detach().float().cpu().numpy().astype(np.float64)
            return q_xy, r_idx, conf


class _AlikedBackend:
    """ALIKED detector + LightGlue matcher."""

    def __init__(self, *, max_keypoints: int) -> None:
        self._max_keypoints = int(max_keypoints)
        self._aliked = None
        self._glue = None

    def _ensure(self) -> None:
        if self._aliked is not None and self._glue is not None:
            return
        import torch

        if self._aliked is None:
            from kornia.feature import ALIKED

            self._aliked = ALIKED.from_pretrained(
                "aliked-n16",
                max_num_keypoints=self._max_keypoints,
                detection_threshold=0.0,
            ).eval().cuda()
        if self._glue is None:
            from lightglue import LightGlue

            self._glue = LightGlue(features="aliked").eval().cuda()

    def _to_tensor(self, image_bgr: np.ndarray):
        import torch

        import cv2

        rgb = cv2.cvtColor(np.ascontiguousarray(image_bgr), cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255.0

    def prepare_query(self, image_bgr: np.ndarray) -> dict:
        import torch

        self._ensure()
        assert self._aliked is not None
        tensor = self._to_tensor(image_bgr)
        height, width = image_bgr.shape[:2]
        with torch.inference_mode():
            feats = self._aliked(tensor)[0]
        return {
            "keypoints": feats.keypoints,
            "descriptors": feats.descriptors,
            "image_size": (width, height),
        }

    def prepare_reference(self, image_bgr: np.ndarray, work_xy: np.ndarray) -> dict:
        import torch

        self._ensure()
        assert self._aliked is not None
        work = np.asarray(work_xy, dtype=np.float32).reshape(-1, 2)
        height, width = image_bgr.shape[:2]
        if work.shape[0] == 0:
            return {
                "keypoints": torch.zeros((0, 2), dtype=torch.float32, device="cuda"),
                "descriptors": torch.zeros((0, 128), dtype=torch.float32, device="cuda"),
                "image_size": (width, height),
            }
        tensor = self._to_tensor(image_bgr)
        with torch.inference_mode():
            feature_map, _ = self._aliked.extract_dense_map(tensor)
            _, _, h, w = tensor.shape
            wh = torch.tensor([w - 1, h - 1], dtype=torch.float32, device="cuda")
            px = torch.as_tensor(work, dtype=torch.float32, device="cuda")
            norm = 2.0 * px / wh[None] - 1.0
            descs, _ = self._aliked.desc_head(feature_map, [norm])
            feats = descs[0]
            kpts = torch.as_tensor(work, dtype=torch.float32, device="cuda")
        return {
            "keypoints": kpts,
            "descriptors": feats,
            "image_size": (width, height),
        }

    def match(
        self, query_feat: dict, reference_feat: dict
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import torch

        qk = query_feat["keypoints"]
        qd = query_feat["descriptors"]
        rk = reference_feat["keypoints"]
        rd = reference_feat["descriptors"]
        qw, qh = query_feat["image_size"]
        rw, rh = reference_feat["image_size"]
        if int(qk.shape[0]) == 0 or int(rk.shape[0]) == 0:
            return (
                np.zeros((0, 2), dtype=np.float64),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.float64),
            )
        self._ensure()
        assert self._glue is not None
        with torch.inference_mode():
            data = {
                "image0": {
                    "keypoints": qk[None].float(),
                    "descriptors": qd[None].float(),
                    "image_size": torch.tensor(
                        [[qw, qh]], dtype=torch.float32, device=qk.device
                    ),
                },
                "image1": {
                    "keypoints": rk[None].float(),
                    "descriptors": rd[None].float(),
                    "image_size": torch.tensor(
                        [[rw, rh]], dtype=torch.float32, device=rk.device
                    ),
                },
            }
            out = self._glue(data)
            idx = out["matches"][0]
            scores = out["scores"][0]
            if int(idx.shape[0]) == 0:
                return (
                    np.zeros((0, 2), dtype=np.float64),
                    np.zeros((0,), dtype=np.int64),
                    np.zeros((0,), dtype=np.float64),
                )
            q_xy = qk[idx[:, 0]].detach().float().cpu().numpy().astype(np.float64)
            r_idx = idx[:, 1].detach().cpu().numpy().astype(np.int64)
            conf = scores.detach().float().cpu().numpy().astype(np.float64)
            return q_xy, r_idx, conf


class SparseRelocMatcher:
    """Owns one sparse backend, its reference LRU, and per-call stage timings."""

    def __init__(
        self,
        backend: str,
        *,
        max_keypoints: int = DEFAULT_MAX_KEYPOINTS,
        ref_cache: int = DEFAULT_REF_CACHE,
    ) -> None:
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"unsupported sparse backend: {backend}")
        self.backend = str(backend)
        self.max_keypoints = int(max_keypoints)
        self._cache: OrderedDict[tuple[str, int, int], dict] = OrderedDict()
        self._ref_cache = int(ref_cache)
        if backend == "xfeat_lg":
            self._impl = _XFeatBackend(
                max_keypoints=self.max_keypoints, use_lighterglue=True
            )
        elif backend == "xfeat_mnn":
            self._impl = _XFeatBackend(
                max_keypoints=self.max_keypoints, use_lighterglue=False
            )
        else:
            self._impl = _AlikedBackend(max_keypoints=self.max_keypoints)
        self._ms = {"query_extract": 0.0, "ref_extract": 0.0, "matcher": 0.0}

    def reset_stage_ms(self) -> None:
        self._ms = {"query_extract": 0.0, "ref_extract": 0.0, "matcher": 0.0}

    def stage_ms(self) -> dict[str, float]:
        return dict(self._ms)

    def prepare_query(self, image_bgr: np.ndarray):
        import time

        t0 = time.perf_counter()
        feat = self._impl.prepare_query(image_bgr)
        self._ms["query_extract"] += (time.perf_counter() - t0) * 1000.0
        return feat

    def match_reference(
        self,
        query_feat,
        *,
        name: str,
        image_path,
        observation_xy: np.ndarray,
        track_lengths: np.ndarray,
        native_wh: tuple[int, int],
        work_wh: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        import time

        import cv2

        work_xy, native_xy = anchored_reference_keypoints(
            np.asarray(observation_xy),
            np.asarray(track_lengths),
            native_wh=(int(native_wh[0]), int(native_wh[1])),
            work_wh=(int(work_wh[0]), int(work_wh[1])),
            cap=self.max_keypoints,
        )
        if work_xy.shape[0] == 0:
            return (
                np.zeros((0, 2), dtype=np.float64),
                np.zeros((0, 2), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
            )
        key = (str(name), int(work_wh[0]), int(work_wh[1]))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            ref_feat = cached
        else:
            t0 = time.perf_counter()
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_path)
            if (int(image.shape[1]), int(image.shape[0])) != (
                int(work_wh[0]),
                int(work_wh[1]),
            ):
                image = cv2.resize(
                    image, (int(work_wh[0]), int(work_wh[1])), interpolation=cv2.INTER_AREA
                )
            ref_feat = self._impl.prepare_reference(image, work_xy)
            self._ms["ref_extract"] += (time.perf_counter() - t0) * 1000.0
            self._cache[key] = ref_feat
            while self._ref_cache >= 0 and len(self._cache) > self._ref_cache:
                self._cache.popitem(last=False)
        t0 = time.perf_counter()
        q_xy, r_idx, conf = self._impl.match(query_feat, ref_feat)
        self._ms["matcher"] += (time.perf_counter() - t0) * 1000.0
        if len(r_idx) == 0:
            return (
                np.zeros((0, 2), dtype=np.float64),
                np.zeros((0, 2), dtype=np.float64),
                np.zeros((0,), dtype=np.float64),
            )
        # Guard against stale cache entries if the map changed under us.
        r_idx = np.asarray(r_idx, dtype=np.int64)
        valid = (r_idx >= 0) & (r_idx < native_xy.shape[0])
        if not bool(np.all(valid)):
            q_xy = np.asarray(q_xy)[valid]
            conf = np.asarray(conf)[valid]
            r_idx = r_idx[valid]
            if len(r_idx) == 0:
                return (
                    np.zeros((0, 2), dtype=np.float64),
                    np.zeros((0, 2), dtype=np.float64),
                    np.zeros((0,), dtype=np.float64),
                )
        r_xy = np.asarray(native_xy[r_idx], dtype=np.float64)
        return np.asarray(q_xy, dtype=np.float64), r_xy, np.asarray(conf, dtype=np.float64)


__all__ = [
    "XFEAT_REPO",
    "DEFAULT_MAX_KEYPOINTS",
    "DEFAULT_REF_CACHE",
    "SUPPORTED_BACKENDS",
    "anchored_reference_keypoints",
    "SparseRelocMatcher",
]
