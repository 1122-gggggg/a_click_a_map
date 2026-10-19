from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from pathlib import Path
import hashlib

from sfm_diagnosis.models import CameraIntrinsics, Pose


def load_point_cloud(path: str | Path) -> np.ndarray:
    """Load XYZ from ASCII or binary-little-endian vertex-only PLY."""
    path = Path(path)
    blob = path.read_bytes()
    end = blob.find(b"end_header\n")
    if end < 0 or not blob.startswith(b"ply"):
        raise ValueError("occlusion point cloud must be a PLY with a valid header")
    header = blob[:end].decode("ascii", errors="strict").splitlines()
    count = None
    props: list[tuple[str, str]] = []
    in_vertex_element = False
    for line in header:
        tokens = line.split()
        if tokens[:2] == ["element", "vertex"]:
            count = int(tokens[2])
            in_vertex_element = True
            continue
        if tokens and tokens[0] == "element":
            in_vertex_element = False
            continue
        if in_vertex_element and tokens and tokens[0] == "property":
            if len(tokens) != 3 or tokens[1] == "list":
                raise ValueError("PLY vertex list properties are unsupported")
            props.append((tokens[1], tokens[2]))
    if count is None or len(props) < 3:
        raise ValueError("PLY must contain vertex x/y/z properties")
    names = [name for _, name in props]
    if names[:3] != ["x", "y", "z"]:
        raise ValueError("PLY vertex properties must begin with x y z")
    payload = blob[end + len(b"end_header\n") :]
    fmt = next((x.split()[1] for x in header if x.startswith("format ")), "")
    if fmt == "ascii":
        records = payload.splitlines()
        if len(records) < count:
            raise ValueError("ASCII PLY payload has fewer vertex records than declared")
        return np.asarray(
            [[float(x) for x in line.split()[:3]] for line in records[:count]],
            dtype=float,
        )
    if fmt != "binary_little_endian":
        raise ValueError("PLY must be ASCII or binary_little_endian")
    types = {
        "char": "i1",
        "uchar": "u1",
        "short": "<i2",
        "ushort": "<u2",
        "int": "<i4",
        "uint": "<u4",
        "float": "<f4",
        "double": "<f8",
    }
    unknown = sorted({kind for kind, _ in props if kind not in types})
    if unknown:
        raise ValueError(f"unsupported PLY vertex property types: {unknown}")
    dtype = np.dtype([(name, types[kind]) for kind, name in props])
    if len(payload) < count * dtype.itemsize:
        raise ValueError("binary PLY payload has fewer vertex records than declared")
    data = np.frombuffer(payload, dtype=dtype, count=count)
    if len(data) < count:
        raise ValueError("binary PLY payload has fewer vertex records than declared")
    return np.column_stack([data[names[i]].astype(float) for i in range(3)])


@dataclass(frozen=True)
class OcclusionResult:
    visible: np.ndarray
    occluded_count: int
    uncertain_count: int
    raw_visible_count: int


class PointCloudOcclusionProxy:
    """Conservative point-cloud z-buffer; missing support is always visible."""

    def __init__(
        self,
        source_points,
        intrinsics: CameraIntrinsics,
        *,
        splat_radius_px=1,
        depth_tolerance=0.05,
        angle_tolerance_deg=2.0,
        min_support_count=2,
        max_depth_spread=0.25,
        max_search_radius_px=4,
        source_path=None,
    ):
        source = np.asarray(source_points, dtype=float).reshape(-1, 3)
        finite_source = np.isfinite(source).all(axis=1)
        self.source_input_point_count = int(len(source))
        self.source_nonfinite_dropped = int(np.sum(~finite_source))
        self.points = source[finite_source]
        self.intrinsics = intrinsics
        if (
            int(splat_radius_px) < 0
            or depth_tolerance <= 0
            or angle_tolerance_deg <= 0
            or min_support_count <= 0
            or max_depth_spread <= 0
            or int(max_search_radius_px) < 0
        ):
            raise ValueError("occlusion tolerances/radius/support must be positive")
        self.splat_radius_px = int(splat_radius_px)
        self.depth_tolerance = float(depth_tolerance)
        self.angle_tolerance_deg = float(angle_tolerance_deg)
        self.min_support_count = int(min_support_count)
        self.max_depth_spread = float(max_depth_spread)
        self.max_search_radius_px = int(max_search_radius_px)
        self.source_path = str(source_path) if source_path else None

    def metadata(self):
        out = {
            "mode": "point_cloud_depth_proxy",
            "source": "point_cloud",
            "splat_radius_px": self.splat_radius_px,
            "depth_tolerance": self.depth_tolerance,
            "angle_tolerance_deg": self.angle_tolerance_deg,
            "min_support_count": self.min_support_count,
            "max_depth_spread": self.max_depth_spread,
            "max_search_radius_px": self.max_search_radius_px,
            "source_input_point_count": self.source_input_point_count,
            "source_nonfinite_dropped": self.source_nonfinite_dropped,
            "source_point_count": int(len(self.points)),
        }
        if self.source_path:
            out["source_path"] = self.source_path
            out["source_sha256"] = hashlib.sha256(Path(self.source_path).read_bytes()).hexdigest()
        return out

    def filter(self, pose: Pose, target_points) -> OcclusionResult:
        targets = np.asarray(target_points, dtype=float).reshape(-1, 3)
        raw = len(targets)
        if not raw or not len(self.points):
            return OcclusionResult(np.ones(raw, dtype=bool), 0, raw, raw)
        intr = self.intrinsics
        src = pose.world_to_camera(self.points)
        valid = np.isfinite(src).all(axis=1) & (src[:, 2] > 1e-9)
        src = src[valid]
        su = intr.fx * src[:, 0] / src[:, 2] + intr.cx
        sv = intr.fy * src[:, 1] / src[:, 2] + intr.cy
        in_frame = (su >= 0) & (su < intr.width) & (sv >= 0) & (sv < intr.height)
        su, sv, sd = su[in_frame], sv[in_frame], src[:, 2][in_frame]
        pixel_u = np.rint(su).astype(int)
        pixel_v = np.rint(sv).astype(int)
        rounded_in_frame = (
            (pixel_u >= 0) & (pixel_u < intr.width) & (pixel_v >= 0) & (pixel_v < intr.height)
        )
        pixel_u = pixel_u[rounded_in_frame]
        pixel_v = pixel_v[rounded_in_frame]
        sd = sd[rounded_in_frame]
        pix = pixel_u + intr.width * pixel_v
        size = intr.width * intr.height
        depth = np.full(size, np.inf)
        high = np.full(size, -np.inf)
        counts = np.zeros(size, dtype=int)
        np.minimum.at(depth, pix, sd)
        np.maximum.at(high, pix, sd)
        np.add.at(counts, pix, 1)
        finite_targets = np.isfinite(targets).all(axis=1)
        tc = np.full_like(targets, np.nan)
        tc[finite_targets] = pose.world_to_camera(targets[finite_targets])
        tu = intr.fx * tc[:, 0] / np.maximum(tc[:, 2], 1e-12) + intr.cx
        tv = intr.fy * tc[:, 1] / np.maximum(tc[:, 2], 1e-12) + intr.cy
        visible = np.ones(raw, dtype=bool)
        uncertain = 0
        for i, (u, v, z) in enumerate(zip(tu, tv, tc[:, 2])):
            if not np.isfinite((u, v, z)).all() or z <= 1e-9:
                uncertain += 1
                continue
            bu, bv = int(round(u)), int(round(v))
            if not (0 <= bu < intr.width and 0 <= bv < intr.height):
                uncertain += 1
                continue
            radius = min(
                self.max_search_radius_px,
                max(
                    self.splat_radius_px,
                    int(
                        np.ceil(
                            intr.fx * np.tan(np.radians(self.angle_tolerance_deg)) / max(z, 1e-9)
                        )
                    ),
                ),
            )
            ids = [
                (bu + du) + intr.width * (bv + dv)
                for du in range(-radius, radius + 1)
                for dv in range(-radius, radius + 1)
                if 0 <= bu + du < intr.width and 0 <= bv + dv < intr.height
            ]
            ids = np.asarray(ids)
            valid_ids = ids[counts[ids] >= self.min_support_count]
            if (
                not len(valid_ids)
                or np.max(high[valid_ids]) - np.min(depth[valid_ids]) > self.max_depth_spread
            ):
                uncertain += 1
                continue
            if z > np.min(depth[valid_ids]) + self.depth_tolerance:
                visible[i] = False
        return OcclusionResult(visible, int(np.sum(~visible)), uncertain, raw)
