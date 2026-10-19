"""Optional ActLoc provider seam.

Official interface verified against cvg/ActLoc commit
``8614dc4a9c470736871d24cf92dde8f8e7f45cf9``:
https://github.com/cvg/ActLoc/blob/8614dc4a9c470736871d24cf92dde8f8e7f45cf9/README.md

The official model consumes SfM landmarks and mapping poses in a waypoint-
centric crop and emits a 6x18 LocMap.  The CoRL paper describes this as a
map-only viewpoint prior, not an EDM failure probability:
https://proceedings.mlr.press/v305/li25b.html
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from sfm_diagnosis.actloc import StructuralLocalizabilityProxy
from sfm_diagnosis.heatmap import yaw_pitch_rotation
from sfm_diagnosis.models import MapData, Pose


ACTLOC_X_ANGLES = np.arange(-60.0, 60.0, 20.0)
ACTLOC_Y_ANGLES = np.arange(-180.0, 180.0, 20.0)
OFFICIAL_COMMIT = "8614dc4a9c470736871d24cf92dde8f8e7f45cf9"


class ActLocUnavailableError(RuntimeError):
    pass


class ActLocProvider(Protocol):
    source: str
    fingerprint: str

    def score_grid(self, position: np.ndarray) -> np.ndarray | None: ...

    def score(
        self, position: np.ndarray, yaw_deg: float, pitch_deg: float
    ) -> float | None: ...


def world_yaw_pitch_to_actloc_cell(yaw_deg: float, pitch_deg: float) -> tuple[int, int]:
    yaw = (float(yaw_deg) + 180.0) % 360.0 - 180.0
    # Official x_angle is elevation around its base camera and maps to -world pitch.
    row = int(np.argmin(np.abs(ACTLOC_X_ANGLES - (-float(pitch_deg)))))
    circular = np.abs((ACTLOC_Y_ANGLES - yaw + 180.0) % 360.0 - 180.0)
    column = int(np.argmin(circular))
    return row, column


def rotation_matrices_to_scalar_first_quaternions(
    matrices: np.ndarray,
) -> np.ndarray:
    """Convert matrices to canonical WXYZ without version-specific SciPy kwargs."""

    xyzw = Rotation.from_matrix(np.asarray(matrices, dtype=float)).as_quat()
    xyzw = np.asarray(xyzw, dtype=float).reshape(-1, 4)
    canonical = np.where(xyzw[:, 3:4] < 0.0, -xyzw, xyzw)
    return canonical[:, [3, 0, 1, 2]]


@dataclass
class DisabledActLocProvider:
    reason: str = "disabled"
    source: str = "disabled"
    fingerprint: str = "disabled"

    def score_grid(self, position: np.ndarray) -> None:
        del position
        return None

    def score(self, position: np.ndarray, yaw_deg: float, pitch_deg: float) -> None:
        del position, yaw_deg, pitch_deg
        return None


@dataclass
class FallbackActLocProvider:
    map_data: MapData
    predictor: StructuralLocalizabilityProxy = field(
        default_factory=StructuralLocalizabilityProxy
    )
    source: str = "fallback_structural_proxy"
    fingerprint: str = "fallback_structural_proxy_v1"

    def score_grid(self, position: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                [
                    self.score(position, yaw_deg=float(yaw), pitch_deg=float(-x_angle))
                    for yaw in ACTLOC_Y_ANGLES
                ]
                for x_angle in ACTLOC_X_ANGLES
            ],
            dtype=float,
        )

    def score(self, position: np.ndarray, yaw_deg: float, pitch_deg: float) -> float:
        pose = Pose(
            np.asarray(position, dtype=float),
            yaw_pitch_rotation(float(yaw_deg), float(pitch_deg)),
        )
        return float(self.predictor.predict(self.map_data, pose))


@dataclass
class CachedActLocProvider:
    backend: ActLocProvider
    cache_dir: Path

    @property
    def source(self) -> str:
        return f"cached:{self.backend.source}"

    @property
    def fingerprint(self) -> str:
        return f"cached:{self.backend.fingerprint}"

    def score_grid(self, position: np.ndarray) -> np.ndarray | None:
        point = np.asarray(position, dtype=float).reshape(3)
        path = self._path(point)
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("backend_fingerprint") == self.backend.fingerprint:
                grid = payload.get("score_grid")
                return None if grid is None else np.asarray(grid, dtype=float).reshape(6, 18)
        grid = self.backend.score_grid(point)
        self._write(path, grid)
        return grid

    def score(
        self, position: np.ndarray, yaw_deg: float, pitch_deg: float
    ) -> float | None:
        grid = self.score_grid(position)
        if grid is None:
            return None
        row, column = world_yaw_pitch_to_actloc_cell(yaw_deg, pitch_deg)
        return float(grid[row, column])

    def _path(self, position: np.ndarray) -> Path:
        identity = json.dumps(
            {
                "backend": self.backend.fingerprint,
                "position": position.tolist(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return Path(self.cache_dir) / f"{hashlib.sha256(identity).hexdigest()}.json"

    def _write(self, path: Path, grid: np.ndarray | None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "backend_fingerprint": self.backend.fingerprint,
            "score_grid": None if grid is None else np.asarray(grid, dtype=float).tolist(),
        }
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            temporary = Path(stream.name)
        os.replace(temporary, path)


@dataclass
class OfficialActLocProvider:
    map_data: MapData
    source_root: Path
    checkpoint: Path
    device: str = "cuda"
    point_error_threshold: float = 0.5
    source: str = "official_actloc"
    _model: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.source_root = Path(self.source_root).resolve()
        self.checkpoint = Path(self.checkpoint).resolve()
        required = (
            self.source_root / "models/model.py",
            self.source_root / "actloc_core/torch_utils.py",
            self.checkpoint,
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise ActLocUnavailableError(f"official ActLoc files are missing: {missing}")
        self.fingerprint = (
            f"official:{OFFICIAL_COMMIT}:{_sha256(self.checkpoint)}:"
            f"error<{self.point_error_threshold}"
        )

    def score_grid(self, position: np.ndarray) -> np.ndarray:
        model, torch, create_batch = self._ensure_runtime()
        point = np.asarray(position, dtype=float).reshape(3)
        keep = self.map_data.point_errors < float(self.point_error_threshold)
        local_points = self.map_data.points_xyz[keep] - point[None, :]
        local_colors = self.map_data.point_rgb[keep]
        crop = (
            (local_points[:, 0] >= -4.0)
            & (local_points[:, 0] <= 4.0)
            & (local_points[:, 1] >= -4.0)
            & (local_points[:, 1] <= 4.0)
            & (local_points[:, 2] >= -2.0)
            & (local_points[:, 2] <= 2.0)
        )
        if not np.any(crop):
            raise ActLocUnavailableError("official ActLoc local crop contains no landmarks")
        pc_features = np.column_stack(
            (local_points[crop].astype(np.float32), local_colors[crop].astype(np.float32) / 255.0)
        )
        centers = self.map_data.image_centers - point[None, :]
        quaternions = rotation_matrices_to_scalar_first_quaternions(
            np.transpose(self.map_data.image_R_wc, (0, 2, 1))
        )
        pose_features = np.column_stack((centers, quaternions)).astype(np.float32)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        batch = create_batch(
            pc_features,
            pose_features,
            torch.device(self.device),
            dtype,
        )
        with torch.inference_mode(), torch.autocast(
            device_type=torch.device(self.device).type,
            dtype=dtype,
            enabled=True,
        ):
            logits = model(**batch)
            return torch.softmax(logits, dim=1)[0, 0].float().cpu().numpy()

    def score(self, position: np.ndarray, yaw_deg: float, pitch_deg: float) -> float:
        row, column = world_yaw_pitch_to_actloc_cell(yaw_deg, pitch_deg)
        return float(self.score_grid(position)[row, column])

    def _ensure_runtime(self):
        if self._model is not None:
            import torch
            from actloc_core.torch_utils import create_single_sample_batch_for_inference

            return self._model, torch, create_single_sample_batch_for_inference
        if str(self.source_root) not in sys.path:
            sys.path.insert(0, str(self.source_root))
        try:
            import torch
            from actloc_core.torch_utils import (
                create_single_sample_batch_for_inference,
                load_model,
            )
        except (ImportError, OSError) as exc:
            raise ActLocUnavailableError(f"official ActLoc dependency unavailable: {exc}") from exc
        if not torch.cuda.is_available() and self.device.startswith("cuda"):
            raise ActLocUnavailableError("official ActLoc requires CUDA")
        model = load_model(str(self.checkpoint), torch.device(self.device))
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model.to(dtype=dtype)
        self._model = model
        return model, torch, create_single_sample_batch_for_inference


def build_actloc_provider(
    *,
    mode: str,
    map_data: MapData | None = None,
    source_root: str | Path | None = None,
    checkpoint: str | Path | None = None,
    cache_dir: str | Path | None = None,
    fail_open: bool = True,
) -> ActLocProvider:
    try:
        if mode == "disabled":
            provider: ActLocProvider = DisabledActLocProvider()
        elif mode == "fallback":
            if map_data is None:
                raise ActLocUnavailableError("fallback ActLoc requires map data")
            provider = FallbackActLocProvider(map_data)
        elif mode == "official":
            if map_data is None or source_root is None or checkpoint is None:
                raise ActLocUnavailableError(
                    "official ActLoc requires map data, source_root, and checkpoint"
                )
            provider = OfficialActLocProvider(
                map_data=map_data,
                source_root=Path(source_root),
                checkpoint=Path(checkpoint),
            )
        else:
            raise ValueError("ActLoc mode must be disabled, fallback, or official")
    except ActLocUnavailableError as exc:
        if not fail_open:
            raise
        provider = (
            FallbackActLocProvider(map_data)
            if map_data is not None
            else DisabledActLocProvider(reason=str(exc))
        )
    if cache_dir is not None and provider.source != "disabled":
        return CachedActLocProvider(provider, Path(cache_dir))
    return provider


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
