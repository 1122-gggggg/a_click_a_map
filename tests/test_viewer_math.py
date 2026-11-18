"""MapViewer math tests (no display needed: Gdk is stubbed)."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gui"))
import jobs


def _viewer_class():
    src = (Path(__file__).resolve().parents[1] / "gui" / "app.py").read_text(encoding="utf-8")
    start = src.index("class MapViewer:")
    end = src.index("class MapViewerWindow")
    up, down, smooth, left = 0, 1, 2, 3
    ns = {"np": np, "jobs": jobs,
          "Gdk": types.SimpleNamespace(ScrollDirection=types.SimpleNamespace(
              UP=up, DOWN=down, SMOOTH=smooth, LEFT=left)),
          "GdkPixbuf": types.SimpleNamespace(), "Gtk": types.SimpleNamespace(),
          "GLib": types.SimpleNamespace(), "VIEWER_HEIGHT": 300, "Path": Path}
    exec(src[start:end], ns)  # noqa: S102 (test-only, local file)
    ns["SCROLL"] = (up, down, smooth, left)
    return ns["MapViewer"], ns["SCROLL"]


def _viewer_with_outliers():
    cls, _ = _viewer_class()
    viewer = cls.__new__(cls)
    rng = np.random.default_rng(0)
    viewer.points = np.vstack([rng.normal([0, 0, 0], 5, (1000, 3)),
                               np.array([[9e6, 0, 0], [-9e6, 0, 0]])])
    viewer.colors = np.zeros((len(viewer.points), 3), dtype=np.uint8)
    viewer.total = len(viewer.points)
    viewer.center = np.zeros(3)
    viewer.yaw, viewer.pitch, viewer.dist = 0.7, 0.45, 20.0 * 1.7
    viewer.span = np.array([1.8e7, 50.0, 50.0])
    viewer.fit_span = 20.0
    viewer.outlier_ratio = 0.01
    viewer.height = 300
    viewer.drag_button = 0
    viewer.last = (0.0, 0.0)
    viewer.on_change = None
    viewer.box = types.SimpleNamespace(get_allocated_width=lambda: 800)
    calls = []
    viewer.render = lambda: calls.append(1)
    viewer.render_calls = calls
    return viewer


class _Wheel:
    def __init__(self, direction: int, delta_y: float = 0.0) -> None:
        self.direction = direction
        self.delta_y = delta_y


def test_zoom_stays_near_robust_scale_despite_outliers() -> None:
    _, (up, down, smooth, _left) = _viewer_class()
    viewer = _viewer_with_outliers()
    before = viewer.dist
    assert viewer.on_scroll(None, _Wheel(up)) is True
    assert viewer.dist < before
    after_in = viewer.dist
    assert viewer.on_scroll(None, _Wheel(down)) is True
    assert viewer.dist > after_in
    zoomed = viewer.dist
    viewer.on_scroll(None, _Wheel(smooth, 2.0))
    assert viewer.dist > zoomed
    viewer.on_scroll(None, _Wheel(smooth, -2.0))
    assert abs(viewer.dist - zoomed) / zoomed < 0.01
    for _ in range(200):
        viewer.on_scroll(None, _Wheel(up))
    assert viewer.dist >= viewer.fit_span * 0.05 * 0.999
    assert viewer.dist < 1000.0
    assert viewer.render_calls
