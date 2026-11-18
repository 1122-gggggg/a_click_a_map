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
        self.x, self.y = 400.0, 150.0  # screen centre: anchor shift is zero


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


class _FakeEvent:
    def __init__(self, direction: int, x: float, y: float, delta_y: float = 0.0) -> None:
        self.direction = direction
        self.x = x
        self.y = y
        self.delta_y = delta_y


def _viewer_with_points(n: int = 3000, seed: int = 1):
    import sys
    import types
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gui"))
    import jobs
    src = (Path(__file__).resolve().parents[1] / "gui" / "app.py").read_text(encoding="utf-8")
    up, down, smooth, left = 0, 1, 2, 3
    ns = {"np": np, "jobs": jobs,
          "Gdk": types.SimpleNamespace(ScrollDirection=types.SimpleNamespace(UP=up, DOWN=down, SMOOTH=smooth, LEFT=left)),
          "GdkPixbuf": types.SimpleNamespace(), "Gtk": types.SimpleNamespace(), "GLib": types.SimpleNamespace(),
          "VIEWER_HEIGHT": 300, "Path": Path}
    exec(src[src.index("class MapViewer:"):src.index("class MapViewerWindow")], ns)
    viewer = ns["MapViewer"].__new__(ns["MapViewer"])
    rng = np.random.default_rng(seed)
    viewer.points = rng.normal([0, 0, 50], 8, (n, 3))
    viewer.colors = np.zeros((n, 3), dtype=np.uint8)
    viewer.total, viewer.yaw, viewer.pitch, viewer.dist = n, 0.6, 0.3, 40.0
    viewer.span = np.array([30.0, 30.0, 30.0])
    viewer.fit_span, viewer.outlier_ratio, viewer.height = 30.0, 0.0, 300
    viewer.drag_button, viewer.last, viewer.on_change = 0, (0.0, 0.0), None
    viewer.center = np.array([0.0, 0.0, 50.0])
    viewer.box = types.SimpleNamespace(get_allocated_width=lambda: 800)
    viewer.render_calls = []
    viewer.render = lambda: viewer.render_calls.append(1)
    return viewer, (up, down, smooth, left)


def test_wheel_zoom_anchors_cursor() -> None:
    viewer, (up, down, smooth, left) = _viewer_with_points()
    W, H = 800, 300
    sx, sy, _, ok = viewer._project(W, H)[:4]
    inside = ok & (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    idx = int(np.flatnonzero(inside)[len(np.flatnonzero(inside)) // 3])
    ex, ey = float(sx[idx]), float(sy[idx])
    assert viewer.on_scroll(None, _FakeEvent(up, ex, ey)) is True
    assert viewer.on_scroll(None, _FakeEvent(up, ex, ey)) is True
    sx2, sy2, _, ok2 = viewer._project(W, H)[:4]
    assert ok2[idx]
    assert abs(float(sx2[idx]) - ex) <= 1 and abs(float(sy2[idx]) - ey) <= 1
    assert viewer.on_scroll(None, _FakeEvent(left, ex, ey)) is False


class _FakeGesture:
    _ids = iter(range(1000000, 2000000))

    def __init__(self) -> None:
        self._id = next(_FakeGesture._ids)

    def __hash__(self) -> int:
        return self._id


def _viewer_with_gestures(n: int = 1000, seed: int = 3):
    viewer, consts = _viewer_with_points(n, seed)
    viewer._drag_last = {}
    return viewer, consts


def test_drag_never_gets_stuck() -> None:
    viewer, _consts = _viewer_with_gestures()
    gesture = _FakeGesture()
    viewer.on_drag_begin(gesture, 0.0, 0.0)
    x = 0.0
    for _ in range(40):  # several full turns: must keep accumulating
        x += 100.0
        viewer.on_drag_update(gesture, x, 0.0, False)
    assert viewer.yaw > 0.008 * 3900, viewer.yaw
    y = 0.0
    for _ in range(40):  # straight over the top: must wrap, never pin
        y += 100.0
        viewer.on_drag_update(gesture, 0.0, y, False)
    assert -np.pi <= viewer.pitch <= np.pi
    viewer.on_drag_end(gesture, 0.0, 0.0)
    assert id(gesture) not in viewer._drag_last
