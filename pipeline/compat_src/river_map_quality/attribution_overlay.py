"""Absolute-coordinate pose-direction glyphs for CloudCompare attribution layers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PoseGlyph:
    query_name: str
    source: str
    center: tuple[float, float, float]
    direction: tuple[float, float, float]
    length: float
    rgb: tuple[int, int, int]


_GLYPH_DTYPE = np.dtype([("xyz", "<f4", (3,)), ("rgb", "u1", (3,))])


def render_pose_glyphs_ply(
    glyphs: Sequence[PoseGlyph],
    *,
    samples_per_glyph: int = 64,
) -> bytes:
    """Render sampled direction lines as a pure-RGB binary PLY point layer."""

    if samples_per_glyph < 2:
        raise ValueError("samples_per_glyph must be at least two")
    vertices = np.empty(len(glyphs) * samples_per_glyph, dtype=_GLYPH_DTYPE)
    interpolation = np.linspace(0.0, 1.0, samples_per_glyph)[:, None]
    offset = 0
    for glyph in glyphs:
        center = np.asarray(glyph.center, dtype=float)
        direction = np.asarray(glyph.direction, dtype=float)
        color = np.asarray(glyph.rgb, dtype=int)
        norm = np.linalg.norm(direction)
        if (
            center.shape != (3,)
            or direction.shape != (3,)
            or not np.isfinite(center).all()
            or not np.isfinite(direction).all()
            or norm <= np.finfo(float).eps
            or not np.isfinite(glyph.length)
            or glyph.length <= 0
        ):
            raise ValueError(f"invalid pose glyph: {glyph.query_name}/{glyph.source}")
        if color.shape != (3,) or np.any(color < 0) or np.any(color > 255):
            raise ValueError("glyph RGB values must be in [0, 255]")
        end = offset + samples_per_glyph
        vertices["xyz"][offset:end] = (
            center + interpolation * (direction / norm) * glyph.length
        ).astype(np.float32)
        vertices["rgb"][offset:end] = color.astype(np.uint8)
        offset = end
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            "comment M0, EDM, and reference-conditioned pose direction glyphs",
            "comment coordinates are the frozen M0 reconstruction frame",
            f"element vertex {len(vertices)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
            "",
        ]
    ).encode("ascii")
    return header + vertices.tobytes(order="C")
