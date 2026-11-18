#!/usr/bin/env python3
"""無人機自動建圖巡航系統 - desktop application (GTK 3 via PyGObject; no browser, no network, no pip packages).

    ./gui.sh

操作只有兩條路（二選一），都在左邊分頁：
    建圖：     ① 加入影片 → ② 選相機設定 → ③ 取名＋選輸出位置 → ④ 按「開始建圖」
    巡航定位： ① 選一張已建好的地圖 → ② 選一部巡航影片 → ③ 按「開始巡航定位」
右邊的任務中心顯示目前進度；成果和執行日誌分開。操作條件不足時，開始按鈕上方會顯示原因。
建圖完成後，先在 3D 檢視確認地圖沒問題並按下確認，巡航定位才會放行。
Works with both entry points of this repository: ./run_moge3.sh (MoGe-3 version) and ./run.sh (original).
Needs the distribution's python3 with PyGObject + GTK 3 (Ubuntu desktop has it: python3-gi, gir1.2-gtk-3.0).
"""
from __future__ import annotations

import struct
import subprocess
import sys
import time
import zlib
from pathlib import Path

import numpy as np

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
from gi.repository import Gdk, GdkPixbuf, GLib, Gtk, Pango  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jobs  # noqa: E402

TITLE = "無人機自動建圖巡航系統"
STATUS = {"queued": "排隊中", "running": "執行中", "done": "完成", "failed": "失敗", "stopped": "已停止", "interrupted": "已中斷"}
STATUS_COLOR = {"queued": "#9a6700", "running": "#9a6700", "done": "#1a7f37", "failed": "#cf222e", "stopped": "#cf222e", "interrupted": "#cf222e"}
POSE_COLOR = {"RELOC_SEED": (63, 185, 80), "FAST_TRACK": (58, 160, 255), "VO_ONLY": (210, 153, 34),
              "DEAD_RECKON": (163, 113, 247), "NO_POSE": (248, 81, 73)}
POSE_NAME = {"RELOC_SEED": "重新定位", "FAST_TRACK": "追蹤中", "VO_ONLY": "里程計", "DEAD_RECKON": "推算", "NO_POSE": "無位姿"}
KIND_NAME = {"map": "建圖", "cruise": "巡航定位", "pack": "打包", "export": "交付匯出"}
FINAL = ("done", "failed", "stopped", "interrupted")
CSS = b"""
.section-title { font-size: 13pt; font-weight: bold; }
.task-title { font-size: 16pt; font-weight: bold; }
.step { font-weight: bold; }
.step-done { font-weight: bold; color: @success_color; }
.hint { color: alpha(@theme_fg_color, 0.77); font-size: 10pt; }
.ready { color: @success_color; font-size: 10pt; font-weight: bold; }
.notready { font-size: 10pt; }
.warn { font-size: 10pt; font-weight: bold; color: @error_color; }
.panel { background-color: @theme_base_color; border: 1px solid @borders; border-radius: 8px; }
.panel-padding { padding: 10px 12px; }
.phase { padding: 4px 0; font-weight: bold; }
.readiness { padding: 8px; background-color: @theme_bg_color; border-radius: 5px; }
button.go { font-weight: bold; padding: 8px 18px; }
treeview.view { -GtkTreeView-vertical-separator: 9; }
textview.log, textview.log text { background-color: #0d1117; color: #e6edf3; font-family: monospace; font-size: 10pt; }
"""


def css_label(text: str, *classes: str, wrap: bool = False) -> Gtk.Label:
    label = Gtk.Label(label=text, xalign=0)
    for cls in classes:
        label.get_style_context().add_class(cls)
    if wrap:
        label.set_line_wrap(True)
        label.set_line_wrap_mode(Pango.WrapMode.WORD_CHAR)
        label.set_max_width_chars(46)
    return label


def png_bytes(width: int, height: int, pixels: bytearray) -> bytes:
    raw = b"".join(b"\x00" + bytes(pixels[y * width * 3:(y + 1) * width * 3]) for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def render_trajectory(points: list, width: int, height: int) -> bytes:
    """Top-down (X-Z) trajectory as a PNG, drawn with the standard library only."""
    pixels = bytearray(bytes((13, 17, 23)) * (width * height))

    def put(x: int, y: int, color: tuple) -> None:
        if 0 <= x < width and 0 <= y < height:
            i = (y * width + x) * 3
            pixels[i:i + 3] = bytes(color)

    def line(x0: int, y0: int, x1: int, y1: int, color: tuple) -> None:
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        while True:
            put(x0, y0, color)
            if x0 == x1 and y0 == y1:
                return
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    xs, zs = [p[0][0] for p in points], [p[0][2] for p in points]
    x0, x1, z0, z1 = min(xs), max(xs), min(zs), max(zs)
    margin = 18
    scale = min((width - 2 * margin) / max(x1 - x0, 1e-9), (height - 2 * margin) / max(z1 - z0, 1e-9))
    pts = [(int(margin + (p[0][0] - x0) * scale), int(height - margin - (p[0][2] - z0) * scale), p[1]) for p in points]
    for a, b in zip(pts, pts[1:]):
        line(a[0], a[1], b[0], b[1], (58, 74, 94))
    for x, y, state in pts:
        color = POSE_COLOR.get(state, (136, 136, 136))
        for ox in (-2, -1, 0, 1, 2):
            for oy in (-2, -1, 0, 1, 2):
                if ox * ox + oy * oy <= 5:
                    put(x + ox, y + oy, color)
    return png_bytes(width, height, pixels)


def pixbuf_from_png(data: bytes) -> GdkPixbuf.Pixbuf:
    loader = GdkPixbuf.PixbufLoader()
    loader.write(data)
    loader.close()
    return loader.get_pixbuf()


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


VIEWER_HEIGHT = 300


class MapViewer:
    """Software-rendered RGB point cloud inspector (no GL needed).

    左鍵拖曳旋轉，左鍵點兩下把旋轉中心換到點到的位置，
    右鍵拖曳讓點雲跟著游標平移，滾輪縮放。Renders into a Gtk.Image via numpy.
    """

    BG = (13, 17, 23)
    MARK = (255, 180, 0)

    def __init__(self, *, expand: bool = False) -> None:
        self.points = np.zeros((0, 3))
        self.colors = np.zeros((0, 3), dtype=np.uint8)
        self.total = 0
        self.center = np.zeros(3)
        self.pan_x, self.pan_y = 0.0, 0.0
        self.yaw, self.pitch, self.dist = 0.7, 0.45, 10.0
        self.span = np.ones(3)
        self.fit_span = 1.0
        self.outlier_ratio = 0.0
        self.expand = expand
        self.height = VIEWER_HEIGHT
        self.image = Gtk.Image()
        # The rendered pixbuf must not set the window's minimum size.
        # Only the overlay's main child contributes to its size request.
        viewport = Gtk.Overlay()
        surface = Gtk.DrawingArea()
        surface.set_size_request(320, 200 if expand else VIEWER_HEIGHT)
        viewport.add(surface)
        viewport.add_overlay(self.image)
        viewport.set_overlay_pass_through(self.image, True)
        self.box = Gtk.EventBox()
        self.box.add(viewport)
        self.box.add_events(Gdk.EventMask.BUTTON_PRESS_MASK | Gdk.EventMask.SCROLL_MASK)
        self.box.connect("button-press-event", self.on_press)
        self.box.connect("scroll-event", self.on_scroll)
        self.box.connect("size-allocate", self.on_allocate)
        # GtkGestureDrag grabs the pointer, so a long drag keeps working even
        # when the cursor leaves the view (the old manual handler stalled there).
        self.rotate_gesture = Gtk.GestureDrag.new(self.box)
        self.rotate_gesture.set_button(1)
        self.rotate_gesture.connect("drag-begin", self.on_drag_begin)
        self.rotate_gesture.connect("drag-update", self.on_drag_update, False)
        self.rotate_gesture.connect("drag-end", self.on_drag_end)
        self.pan_gesture = Gtk.GestureDrag.new(self.box)
        self.pan_gesture.set_button(3)
        self.pan_gesture.connect("drag-begin", self.on_drag_begin)
        self.pan_gesture.connect("drag-update", self.on_drag_update, True)
        self.pan_gesture.connect("drag-end", self.on_drag_end)
        self._drag_last: dict = {}
        self.on_change = None

    def on_allocate(self, _widget, allocation) -> None:
        if self.expand:
            self.height = max(allocation.height, 200)
        pixbuf = self.image.get_pixbuf()
        if pixbuf is None or (pixbuf.get_width(), pixbuf.get_height()) != (max(allocation.width, 320), self.height):
            self.render()

    def load(self, path: Path) -> dict:
        xyz, rgb, total = jobs.read_ply_points(path)
        step = max(1, len(xyz) // jobs.VIEWER_MAX_POINTS)
        self.points = xyz[::step]
        self.colors = rgb[::step]
        self.total = total
        lo, hi = xyz.min(axis=0), xyz.max(axis=0)
        self.span = np.maximum(hi - lo, 1e-9)
        # Robust framing: a few runaway triangulations can sit millions of units
        # out; fitting to min/max would shrink the real scene to invisibility.
        qlo, qhi = np.percentile(xyz, [1.0, 99.0], axis=0)
        self.center = jobs.dense_center(xyz, qlo, qhi)
        self.fit_span = float(np.maximum(qhi - qlo, 1e-9).max())
        self.outlier_ratio = float(np.mean((xyz < qlo).any(axis=1) | (xyz > qhi).any(axis=1)))
        self.yaw, self.pitch = 0.7, 0.45
        self.pan_x, self.pan_y = 0.0, 0.0
        self.dist = self.fit_span * 1.7
        self.render()
        return self.info()

    def info(self) -> dict:
        return {"points_total": int(self.total), "points_shown": int(len(self.points)),
                "span": [round(float(v), 3) for v in self.span],
                "center": [round(float(v), 3) for v in self.center],
                "outlier_ratio": round(self.outlier_ratio, 4)}

    def describe(self) -> str:
        s = self.span
        text = (f"顯示 {len(self.points)}／共 {self.total} 點　範圍 {s[0]:.1f}×{s[1]:.1f}×{s[2]:.1f}（地圖座標，非公尺）　"
                f"旋轉中心 ({self.center[0]:.2f}, {self.center[1]:.2f}, {self.center[2]:.2f})")
        if self.outlier_ratio > 0.01:
            text += f"（{self.outlier_ratio:.1%} 遠點已自動縮放到主體，滾輪縮小可看全部）"
        return text

    def _rotation(self):
        cy, sy, cp, sp = np.cos(self.yaw), np.sin(self.yaw), np.cos(self.pitch), np.sin(self.pitch)
        return np.array([[cy, 0.0, sy], [sy * sp, cp, -cy * sp], [-sy * cp, sp, cy * cp]])

    def _screen_origin(self, width: int, height: int) -> tuple[float, float]:
        return width / 2.0 + self.pan_x, height / 2.0 + self.pan_y

    def _project(self, width: int, height: int):
        rot = self._rotation()
        cam = (self.points - self.center) @ rot.T
        depth = cam[:, 2] + self.dist
        focal = height * 1.3
        ok = depth > 1e-6
        sx = np.full(len(self.points), -1)
        sy = np.full(len(self.points), -1)
        x, y = self._screen_origin(width, height)
        sx[ok] = (x + focal * cam[ok, 0] / depth[ok]).astype(int)
        sy[ok] = (y - focal * cam[ok, 1] / depth[ok]).astype(int)
        return sx, sy, depth, ok, rot, focal

    def render(self) -> None:
        width = max(self.box.get_allocated_width(), 320)
        height = max(self.height, 200)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:, :] = self.BG
        if len(self.points):
            sx, sy, depth, ok, _, _ = self._project(width, height)
            inside = ok & (sx >= 0) & (sx < width) & (sy >= 0) & (sy < height)
            if np.any(inside):
                d = depth[inside]
                shade = (0.45 + 0.55 * (1.0 - (d - d.min()) / max(d.max() - d.min(), 1e-9)))[:, None]
                order = np.argsort(d)[::-1]
                flat = sy[inside][order] * width + sx[inside][order]
                canvas.reshape(-1, 3)[flat] = (self.colors[inside][order] * shade[order]).astype(np.uint8)
            x, y = map(int, self._screen_origin(width, height))
            for o in range(-5, 6):
                for xx, yy in ((x + o, y), (x, y + o)):
                    if 0 <= xx < width and 0 <= yy < height:
                        canvas[yy, xx] = self.MARK
        data = GLib.Bytes(canvas.tobytes())
        self.image.set_from_pixbuf(GdkPixbuf.Pixbuf.new_from_bytes(data, GdkPixbuf.Colorspace.RGB, False, 8, width, height, width * 3))

    def reset_view(self) -> None:
        self.yaw, self.pitch = 0.7, 0.45
        self.pan_x, self.pan_y = 0.0, 0.0
        self.dist = float(self.fit_span * 1.7)
        self.render()
        if self.on_change:
            self.on_change()

    def on_press(self, _w, event) -> bool:
        if event.button == 1 and event.type == Gdk.EventType._2BUTTON_PRESS:
            self.recenter(event.x, event.y)
            return True
        return False

    def on_drag_begin(self, gesture, _x: float, _y: float) -> None:
        self._drag_last[id(gesture)] = (0.0, 0.0)

    def on_drag_end(self, gesture, _x: float, _y: float) -> None:
        self._drag_last.pop(id(gesture), None)

    def on_drag_update(self, gesture, offset_x: float, offset_y: float, pan: bool) -> None:
        if not len(self.points):
            return
        last = self._drag_last.get(id(gesture), (0.0, 0.0))
        dx, dy = offset_x - last[0], offset_y - last[1]
        self._drag_last[id(gesture)] = (offset_x, offset_y)
        if not pan:
            self.yaw += dx * 0.008
            # Wrap instead of clamping: orbiting over the top flips past the
            # pole like a trackball instead of getting stuck against a wall.
            self.pitch = (self.pitch + dy * 0.008 + np.pi) % (2.0 * np.pi) - np.pi
        else:
            self.pan_x += dx
            self.pan_y += dy
        self.render()
        if self.on_change:
            self.on_change()

    def on_scroll(self, _w, event) -> bool:
        # CloudCompare-style dolly: zoom toward the cursor, not the screen centre.
        if not len(self.points):
            return False
        old = self.dist
        if event.direction == Gdk.ScrollDirection.UP:
            self.dist /= 1.15
        elif event.direction == Gdk.ScrollDirection.DOWN:
            self.dist *= 1.15
        elif event.direction == Gdk.ScrollDirection.SMOOTH:
            self.dist *= 1.15 ** float(getattr(event, "delta_y", 0.0))
        else:
            return False
        span = float(getattr(self, "fit_span", 0.0) or 0.0) or float(self.span.max())
        self.dist = min(max(self.dist, span * 0.05), span * 30.0)
        if self.dist != old and old > 0:
            width = max(self.box.get_allocated_width(), 320)
            height = max(self.height, 200)
            focal = height * 1.3
            rot = self._rotation()
            shift = (old - self.dist) / focal
            right, up = rot.T[:, 0], -rot.T[:, 1]
            x, y = self._screen_origin(width, height)
            self.center = (self.center + right * (event.x - x) * shift
                           + up * (event.y - y) * shift)
        self.render()
        return True

    def recenter(self, x: float, y: float) -> None:
        if not len(self.points):
            return
        width = max(self.box.get_allocated_width(), 320)
        sx, sy, _, ok, _, _ = self._project(width, max(self.height, 200))
        cand = ok & (sx >= 0) & (sx < width) & (sy >= 0) & (sy < max(self.height, 200))
        if not np.any(cand):
            return
        d2 = (sx[cand] - x) ** 2 + (sy[cand] - y) ** 2
        if float(d2.min()) > 18.0 ** 2:
            return
        idx = np.flatnonzero(cand)[int(d2.argmin())]
        self.center = self.points[idx].copy()
        self.pan_x, self.pan_y = 0.0, 0.0
        self.render()
        if self.on_change:
            self.on_change()

class MapViewerWindow(Gtk.Window):
    """A big standalone 3D inspection window (the in-panel viewer stays as a preview)."""

    def __init__(self) -> None:
        super().__init__(title="3D 點雲檢視")
        self.set_default_size(1100, 800)
        self.viewer = MapViewer(expand=True)
        self.viewer.on_change = self.update_label
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6, margin=10)
        self.add(box)
        hint = css_label("左鍵拖曳旋轉 · 左鍵點兩下設旋轉中心 · 右鍵拖曳移動點雲 · 滾輪以游標為中心縮放",
                         "hint", wrap=True)
        box.pack_start(hint, False, False, 0)
        box.pack_start(self.viewer.box, True, True, 0)
        row = Gtk.Box(spacing=6)
        self.label = css_label("", "hint", wrap=True)
        self.label.set_selectable(True)
        row.pack_start(self.label, True, True, 0)
        self.fullscreen_button = Gtk.Button(label="全螢幕（F11）")
        self.fullscreen_button.connect("clicked", self.toggle_fullscreen)
        row.pack_start(self.fullscreen_button, False, False, 0)
        reset_button = Gtk.Button(label="重設視角")
        reset_button.connect("clicked", lambda _b: self.viewer.reset_view())
        row.pack_start(reset_button, False, False, 0)
        box.pack_start(row, False, False, 0)
        self.connect("window-state-event", self.on_window_state)
        self.connect("key-press-event", self.on_key_press)

    def toggle_fullscreen(self, _button=None) -> None:
        if self.get_window().get_state() & Gdk.WindowState.FULLSCREEN:
            self.unfullscreen()
        else:
            self.fullscreen()

    def on_window_state(self, _window, event) -> bool:
        fullscreen = event.new_window_state & Gdk.WindowState.FULLSCREEN
        self.fullscreen_button.set_label("離開全螢幕（Esc）" if fullscreen else "全螢幕（F11）")
        return False

    def on_key_press(self, _window, event) -> bool:
        if event.keyval == Gdk.KEY_F11:
            self.toggle_fullscreen()
            return True
        if event.keyval == Gdk.KEY_Escape and self.get_window().get_state() & Gdk.WindowState.FULLSCREEN:
            self.unfullscreen()
            return True
        return False

    def update_label(self) -> None:
        self.label.set_text(self.viewer.describe())

    def open_cloud(self, label: str, path: Path) -> None:
        """Load `path` and bring the window forward; raises on unreadable files."""
        self.viewer.load(path)
        self.set_title(f"3D 點雲檢視 — {label}")
        self.update_label()
        self.show_all()
        self.present()

class App(Gtk.Window):
    def __init__(self) -> None:
        super().__init__(title=TITLE)
        self.set_default_size(1240, 820)
        provider = Gtk.CssProvider()
        provider.load_from_data(CSS)
        Gtk.StyleContext.add_provider_for_screen(Gdk.Screen.get_default(), provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        self.engines = jobs.engine_status()
        self.engine = "moge3" if self.engines["moge3"]["ready"] or not self.engines["original"]["ready"] else "original"
        self.radios: dict[str, Gtk.RadioButton] = {}
        self.selected: str | None = None
        self.log_offset = 0
        self.shown: tuple[str, str] | None = None
        self.signature: tuple = ()
        self.plot_points: list = []
        self.plot_width = 0
        self.cruise_maps: dict[str, str] = {}
        self.autoscroll = True
        self._build()
        jobs.mark_interrupted()
        self.connect("delete-event", self.on_close)
        self.on_engine_changed()
        self.tick()
        GLib.timeout_add(1000, self.tick)

    # ------------------------------------------------------------------------------------------ layout
    def _build(self) -> None:
        header = Gtk.HeaderBar(title=TITLE, subtitle="建圖、檢查與影片定位", show_close_button=True)
        self.set_titlebar(header)
        root = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL, wide_handle=True, margin=12)
        self.add(root)

        inputs = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8, margin_end=8)
        inputs.set_size_request(355, -1)
        inputs.pack_start(css_label("建立任務", "section-title"), False, False, 0)
        inputs.pack_start(self._engine_bar(), False, False, 0)
        notebook = Gtk.Notebook()
        for (form, actions), title in ((self._map_page(), "建圖"), (self._cruise_page(), "巡航定位")):
            page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
            page_scroll = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER)
            page_scroll.add(form)
            page.pack_start(page_scroll, True, True, 0)
            actions.set_property("margin", 8)
            page.pack_end(actions, False, False, 0)
            notebook.append_page(page, Gtk.Label(label=title))
        inputs.pack_start(notebook, True, True, 0)
        root.pack1(inputs, False, False)
        root.pack2(self._right(), True, False)
        root.set_position(392)

    def _engine_bar(self) -> Gtk.Widget:
        """A single engine switch for the whole window (both tabs share it)."""
        frame = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=5)
        frame.get_style_context().add_class("panel")
        frame.get_style_context().add_class("panel-padding")
        frame.pack_start(css_label("處理引擎", "step"), False, False, 0)
        ready = [key for key in ("moge3", "original") if self.engines[key]["ready"]]
        if not ready:
            hints = "\n".join(f"{self.engines[k]['label']}：{self.engines[k]['hint']}" for k in self.engines)
            frame.pack_start(css_label("系統尚未安裝完成，請聯絡工程人員。\n" + hints, "warn", wrap=True), False, False, 0)
        elif len(ready) == 1:
            frame.pack_start(css_label(self.engines[ready[0]]["label"], "hint", wrap=True), False, False, 0)
        else:
            choices = Gtk.Box(spacing=12)
            group = None
            for key in ready:
                label = "原始版" if key == "original" else "MoGe-3"
                radio = Gtk.RadioButton.new_with_label_from_widget(group, label)
                radio.set_tooltip_text(self.engines[key]["label"])
                if group is None:
                    group = radio
                radio.set_active(key == self.engine)
                radio.connect("toggled", self.on_engine_toggled, key)
                self.radios[key] = radio
                choices.pack_start(radio, False, False, 0)
            frame.pack_start(choices, False, False, 0)
            self.engine_hint = css_label("", "hint", wrap=True)
            frame.pack_start(self.engine_hint, False, False, 0)
        return frame

    def _step_frame(self, title_label: Gtk.Label, *children: Gtk.Widget) -> Gtk.Box:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=7)
        box.get_style_context().add_class("panel")
        box.get_style_context().add_class("panel-padding")
        box.pack_start(title_label, False, False, 0)
        for child in children:
            box.pack_start(child, False, False, 0)
        return box

    def _map_page(self) -> tuple[Gtk.Widget, Gtk.Widget]:
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8, margin=8)
        # --- step 1: videos ---
        self.map_step1 = css_label("步驟 1／4：加入影片", "step")
        video_list = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.video_store = Gtk.ListStore(str)
        self.video_view = Gtk.TreeView(model=self.video_store, headers_visible=False)
        self.video_view.set_tooltip_text("已加入的影片或資料夾。按住 Ctrl 可多選，再按「移除選取」。")
        self.video_view.append_column(Gtk.TreeViewColumn("影片", Gtk.CellRendererText(ellipsize=Pango.EllipsizeMode.MIDDLE), text=0))
        self.video_view.get_selection().set_mode(Gtk.SelectionMode.MULTIPLE)
        scroll = Gtk.ScrolledWindow(shadow_type=Gtk.ShadowType.IN)
        scroll.set_size_request(-1, 80)
        scroll.add(self.video_view)
        video_list.pack_start(scroll, False, False, 0)
        row1 = Gtk.Box(spacing=6)
        self.add_folder_button = Gtk.Button(label="加入資料夾…")
        self.add_folder_button.set_tooltip_text("選一個放滿影片的資料夾（會自動找出裡面的影片檔）")
        self.add_folder_button.connect("clicked", self.add_folder)
        self.add_files_button = Gtk.Button(label="加入影片檔…")
        self.add_files_button.set_tooltip_text("選一部或多部影片檔（可 Ctrl 多選）")
        self.add_files_button.connect("clicked", self.add_files)
        row1.pack_start(self.add_files_button, False, False, 0)
        row1.pack_start(self.add_folder_button, False, False, 0)
        video_list.pack_start(row1, False, False, 0)
        row2 = Gtk.Box(spacing=6)
        remove_button = Gtk.Button(label="移除選取")
        remove_button.set_tooltip_text("從清單移除反白的項目（不會刪除檔案）")
        remove_button.connect("clicked", self.remove_videos)
        clear_button = Gtk.Button(label="清空全部")
        clear_button.set_tooltip_text("清空整個清單（不會刪除檔案）")
        clear_button.connect("clicked", lambda _b: self.video_store.clear())
        row2.pack_start(remove_button, False, False, 0)
        row2.pack_start(clear_button, False, False, 0)
        selection = self.video_view.get_selection()
        remove_button.set_sensitive(False)
        clear_button.set_sensitive(False)
        selection.connect("changed", lambda selected: remove_button.set_sensitive(selected.count_selected_rows() > 0))
        for signal in ("row-inserted", "row-deleted"):
            self.video_store.connect(signal, lambda model, *_: clear_button.set_sensitive(bool(len(model))))
            self.video_store.connect(signal, self.on_camera_batch_changed)
        video_list.pack_start(row2, False, False, 0)
        self.video_hint = css_label("", "hint", wrap=True)
        video_list.pack_start(self.video_hint, False, False, 0)
        page.pack_start(self._step_frame(self.map_step1, video_list), False, False, 0)

        page.pack_start(self._camera_section(), False, False, 0)

        # --- step 3: settings ---
        self.map_step2 = css_label("步驟 3／4：取名與輸出位置", "step")
        settings = Gtk.Grid(column_spacing=8, row_spacing=6)
        self.name = Gtk.Entry(text="site", hexpand=True, width_chars=12)
        self.name.set_tooltip_text("任務名稱：只能用英文、數字、底線；會變成輸出資料夾的名字")
        self.out = Gtk.Entry(text=str(jobs.REPO / "acam_out"), hexpand=True, width_chars=12)
        self.out.set_tooltip_text("建圖結果要放在哪個資料夾")
        settings.attach(Gtk.Label(label="任務名稱", xalign=0), 0, 0, 1, 1)
        settings.attach(self.name, 1, 0, 2, 1)
        settings.attach(Gtk.Label(label="輸出位置", xalign=0), 0, 1, 1, 1)
        settings.attach(self.out, 1, 1, 1, 1)
        settings.attach(self.chooser_button(self.out, "folder"), 2, 1, 1, 1)
        settings_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        settings_box.pack_start(settings, False, False, 0)
        self.dense_check = Gtk.CheckButton(label="產生稠密點雲")
        self.dense_check.set_tooltip_text("勾選會在建圖後多跑稠密化（需 MVSFormer++ 權重）；不勾只建稀疏地圖，比較快")
        self.dense_check.set_active(True)
        settings_box.pack_start(self.dense_check, False, False, 0)
        page.pack_start(self._step_frame(self.map_step2, settings_box), False, False, 0)

        # --- step 4: go ---
        submit_heading = css_label("步驟 4／4：開始", "step")
        go = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.start_map_button = Gtk.Button(label="開始建圖")
        for style in ("go", "suggested-action"):
            self.start_map_button.get_style_context().add_class(style)
        self.start_map_button.connect("clicked", self.start_map)
        self.map_readiness = css_label("", "notready", "readiness", wrap=True)
        go.pack_start(self.map_readiness, False, False, 0)
        go.pack_start(self.start_map_button, False, False, 0)
        return page, self._step_frame(submit_heading, go)

    def _camera_section(self) -> Gtk.Widget:
        self.calibration_pick = Gtk.ComboBoxText()
        self.calibration_pick.get_cells()[0].set_property("ellipsize", Pango.EllipsizeMode.END)
        self.calibration_pick.connect("changed", self.on_camera_batch_changed)
        row = Gtk.Box(spacing=6)
        row.pack_start(self.calibration_pick, True, True, 0)
        add_profile = Gtk.Button(label="匯入 JSON…")
        add_profile.connect("clicked", self.import_camera)
        row.pack_start(add_profile, False, False, 0)
        self.camera_confirm = Gtk.CheckButton(label="同批影片使用同一相機、鏡頭與錄影模式")
        self.camera_confirm.get_child().set_line_wrap(True)
        self.camera_note = css_label("", "hint", wrap=True)
        return self._step_frame(css_label("步驟 2／4：相機設定", "step"), row, self.camera_note, self.camera_confirm)

    def _cruise_page(self) -> tuple[Gtk.Widget, Gtk.Widget]:
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8, margin=8)
        # --- step 1: map ---
        self.cruise_step1 = css_label("步驟 1／3：選擇已建好的地圖", "step")
        map_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.map_pick = Gtk.ComboBoxText()
        self.map_pick.get_cells()[0].set_property("ellipsize", Pango.EllipsizeMode.END)
        self.map_pick.set_tooltip_text("在這台電腦建好、已完成的建圖任務，直接選就行")
        self.map_pick.connect("changed", self.on_map_pick)
        map_box.pack_start(self.map_pick, False, False, 0)
        row = Gtk.Box(spacing=6)
        self.map_dir = Gtk.Entry(hexpand=True, placeholder_text="選擇或貼上地圖路徑", width_chars=12)
        self.map_dir.set_tooltip_text("也可以手動選：MoGe-3 版選含 localization/bundle 的資料夾；原始版選含 run/ 的資料夾")
        self.map_dir.connect("changed", self.on_map_path_changed)
        row.pack_start(self.map_dir, True, True, 0)
        row.pack_start(self.chooser_button(self.map_dir, "folder"), False, False, 0)
        map_box.pack_start(row, False, False, 0)
        self.map_pick_hint = css_label("", "hint", wrap=True)
        map_box.pack_start(self.map_pick_hint, False, False, 0)
        page.pack_start(self._step_frame(self.cruise_step1, map_box), False, False, 0)

        # --- step 2: query video ---
        self.cruise_step2 = css_label("步驟 2／3：選擇巡航影片", "step")
        query_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        row = Gtk.Box(spacing=6)
        self.query = Gtk.Entry(hexpand=True, placeholder_text="選擇或貼上影片路徑", width_chars=12)
        self.query.set_tooltip_text("想定位哪一部飛行影片，就選哪一部")
        row.pack_start(self.query, True, True, 0)
        row.pack_start(self.chooser_button(self.query, "video"), False, False, 0)
        query_box.pack_start(row, False, False, 0)
        page.pack_start(self._step_frame(self.cruise_step2, query_box), False, False, 0)

        # --- step 3: go ---
        submit_heading = css_label("步驟 3／3：開始", "step")
        go = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.route = Gtk.Entry(placeholder_text="航線名稱（選填）", width_chars=12)
        self.route.set_tooltip_text("留空使用影片檔名")
        go.pack_start(self.route, False, False, 0)
        self.start_cruise_button = Gtk.Button(label="開始巡航定位")
        for style in ("go", "suggested-action"):
            self.start_cruise_button.get_style_context().add_class(style)
        self.start_cruise_button.connect("clicked", self.start_cruise)
        self.cruise_readiness = css_label("", "notready", "readiness", wrap=True)
        go.pack_start(self.cruise_readiness, False, False, 0)
        go.pack_start(self.start_cruise_button, False, False, 0)
        return page, self._step_frame(submit_heading, go)

    def _right(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10, margin_start=8)
        heading = Gtk.Box(spacing=8)
        heading.pack_start(css_label("任務中心", "section-title"), True, True, 0)
        self.task_summary = css_label("尚無任務", "hint")
        heading.pack_end(self.task_summary, False, False, 0)
        box.pack_start(heading, False, False, 0)
        self.job_store = Gtk.ListStore(str, str, str, str, str)  # id, name, kind, status, created
        self.job_view = Gtk.TreeView(model=self.job_store, enable_search=True, search_column=1)
        for index, (title, width) in enumerate((("任務名稱", 132), ("類型", 78), ("狀態", 78), ("建立時間", 105)), start=1):
            renderer = Gtk.CellRendererText(ellipsize=Pango.EllipsizeMode.END)
            column = Gtk.TreeViewColumn(title, renderer, text=index)
            column.set_min_width(width)
            column.set_expand(index == 1)
            self.job_view.append_column(column)
        self.job_view.get_selection().connect("changed", self.on_select)
        scroll = Gtk.ScrolledWindow(shadow_type=Gtk.ShadowType.IN)
        scroll.set_size_request(-1, 132)
        scroll.add(self.job_view)
        box.pack_start(scroll, False, False, 0)

        info = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        info.get_style_context().add_class("panel")
        info.get_style_context().add_class("panel-padding")
        self.title_label = css_label("尚未選擇任務", "task-title")
        self.title_label.set_ellipsize(Pango.EllipsizeMode.END)
        self.status_label = css_label("在左側建立任務，或從上方選擇紀錄。", "hint", wrap=True)
        self.stage_label = css_label("", "hint", wrap=True)
        self.stage_label.set_no_show_all(True)
        self.phase_label = css_label("", "phase", wrap=True)
        self.phase_label.set_no_show_all(True)
        self.stage_bar = Gtk.ProgressBar()
        self.stage_bar.set_no_show_all(True)
        self.path_label = css_label("", "hint")
        self.path_label.set_no_show_all(True)
        self.path_label.set_selectable(True)
        self.path_label.set_ellipsize(Pango.EllipsizeMode.MIDDLE)
        for widget in (self.title_label, self.status_label, self.stage_label, self.phase_label, self.stage_bar, self.path_label):
            info.pack_start(widget, False, False, 0)
        actions = Gtk.Box(spacing=6, margin_top=4)
        self.stop_button = Gtk.Button(label="停止任務", sensitive=False)
        self.stop_button.get_style_context().add_class("destructive-action")
        self.stop_button.connect("clicked", self.confirm_stop)
        self.resume_button = Gtk.Button(label="續跑", sensitive=False)
        self.resume_button.connect("clicked", self.resume_selected)
        self.delete_button = Gtk.Button(label="刪除紀錄", sensitive=False)
        self.delete_button.set_tooltip_text("刪除這筆紀錄與暫存；建圖輸出會保留")
        self.delete_button.connect("clicked", self.confirm_delete)
        self.open_button = Gtk.Button(label="開啟輸出", sensitive=False)
        self.open_button.connect("clicked", self.open_folder)
        for button in (self.stop_button, self.resume_button, self.open_button):
            actions.pack_start(button, False, False, 0)
        actions.pack_end(self.delete_button, False, False, 0)
        info.pack_start(actions, False, False, 0)
        box.pack_start(info, False, False, 0)

        self.detail_tabs = Gtk.Notebook()
        self.results_page = self._result_panel()
        self.log_page = self._log_panel()
        self.detail_tabs.append_page(self.results_page, Gtk.Label(label="成果檢視"))
        self.detail_tabs.append_page(self.log_page, Gtk.Label(label="執行日誌"))
        self.detail_tabs.connect("switch-page", self.on_detail_tab_changed)
        self.detail_tabs.show_all()
        box.pack_start(self.detail_tabs, True, True, 0)
        return box

    def _result_panel(self) -> Gtk.Widget:
        scroll = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8, margin=10)
        scroll.add(box)
        self.transfer_hint = css_label("", "hint", wrap=True)
        self.transfer_hint.set_selectable(True)
        self.transfer_hint.set_no_show_all(True)
        box.pack_start(self.transfer_hint, False, False, 0)
        self.calibration_warn = css_label(
            "相機內參為自動估算（未驗證）。不要只憑點雲外觀判定定位準確；請用獨立影片驗證。",
            "warn", "readiness", wrap=True)
        self.calibration_warn.set_no_show_all(True)
        box.pack_start(self.calibration_warn, False, False, 0)
        self.result_stack = Gtk.Stack()
        self.result_hint = css_label("選擇任務後，這裡會顯示可用成果。", "hint", wrap=True)
        self.result_hint.set_valign(Gtk.Align.START)
        self.result_stack.add_named(self.result_hint, "hint")
        self.file_store = Gtk.ListStore(str, str, str)
        files = Gtk.TreeView(model=self.file_store)
        files.set_tooltip_column(1)
        for index, (title, width) in enumerate((("輸出項目", 155), ("路徑", 90), ("大小", 75))):
            renderer = Gtk.CellRendererText(ellipsize=Pango.EllipsizeMode.MIDDLE)
            column = Gtk.TreeViewColumn(title, renderer, text=index)
            column.set_min_width(width)
            column.set_expand(index == 1)
            column.set_resizable(True)
            files.append_column(column)
        files.connect("row-activated", self.open_result_row)
        map_page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.viewer_frame = Gtk.Frame()
        self.viewer_frame.set_no_show_all(True)
        viewer_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4, margin=6)
        self.viewer = MapViewer()
        self.viewer.on_change = self.update_viewer_label
        viewer_box.pack_start(css_label("3D 地圖檢查", "step"), False, False, 0)
        self.connect_warn = css_label("", "warn", wrap=True)
        self.connect_warn.set_no_show_all(True)
        viewer_box.pack_start(self.connect_warn, False, False, 0)
        viewer_box.pack_start(self.viewer.box, False, False, 0)
        viewer_box.pack_start(css_label("左鍵旋轉／雙擊設中心 · 右鍵平移 · 滾輪縮放", "hint", wrap=True), False, False, 0)
        self.viewer_label = css_label("", "hint", wrap=True)
        viewer_box.pack_start(self.viewer_label, False, False, 0)
        self.approval_label = css_label("", "hint", wrap=True)
        viewer_box.pack_start(self.approval_label, False, False, 0)
        approval_row = Gtk.Box(spacing=6)
        self.approve_button = Gtk.Button(label="✓ 確認地圖可用，可以巡航")
        self.approve_button.get_style_context().add_class("suggested-action")
        self.approve_button.set_tooltip_text("在上方 3D 檢視裡轉一轉、確認地圖沒有明顯問題後再按。按了巡航定位才會放行。")
        self.approve_button.connect("clicked", self.approve_current_map)
        approval_row.pack_start(self.approve_button, False, False, 0)
        self.export_button = Gtk.Button(label="匯出場域 ZIP")
        self.export_button.set_tooltip_text("產生要搬到另一台電腦的壓縮包；解壓後用「＋新增地圖」匯入")
        self.export_button.set_no_show_all(True)
        self.export_button.connect("clicked", self.export_current_map)
        approval_row.pack_start(self.export_button, False, False, 0)
        viewer_box.pack_start(approval_row, False, False, 0)
        view_actions = Gtk.Box(spacing=6)
        big_button = Gtk.Button(label="開啟大視窗")
        big_button.set_tooltip_text("用大視窗看目前這份點雲（跟點兩下檔名一樣）")
        big_button.connect("clicked", lambda _b: self.open_big_viewer())
        view_actions.pack_start(big_button, False, False, 0)
        reset_button = Gtk.Button(label="重設視角")
        reset_button.set_tooltip_text("回到一開始的整體視角")
        reset_button.connect("clicked", lambda _b: self.viewer.reset_view())
        view_actions.pack_start(reset_button, False, False, 0)
        viewer_box.pack_start(view_actions, False, False, 0)
        self.viewer_frame.add(viewer_box)
        viewer_box.show_all()
        map_page.pack_start(self.viewer_frame, False, False, 0)
        map_page.pack_start(css_label("輸出檔案 · 點兩下開啟", "step"), False, False, 0)
        file_scroll = Gtk.ScrolledWindow(shadow_type=Gtk.ShadowType.IN, min_content_height=90)
        file_scroll.add(files)
        map_page.pack_start(file_scroll, True, True, 0)
        self.result_stack.add_named(map_page, "files")
        self.approved_dir = None
        self.approved_engine = ""
        self.viewer_info = {}
        self.viewer_source = ""
        self.viewer_path = None
        self.big_viewer = None
        cruise = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.cruise_summary = css_label("", "step", wrap=True)
        self.coverage_bar = Gtk.ProgressBar(show_text=True)
        self.plot_image = Gtk.Image()
        self.plot_holder = Gtk.ScrolledWindow(shadow_type=Gtk.ShadowType.IN, min_content_height=232,
                                              hscrollbar_policy=Gtk.PolicyType.NEVER)
        self.plot_holder.add(self.plot_image)
        self.plot_holder.connect("size-allocate", lambda _w, _a: self.redraw_plot())
        self.legend = css_label("", "hint", wrap=True)
        cruise.pack_start(self.cruise_summary, False, False, 0)
        cruise.pack_start(self.coverage_bar, False, False, 0)
        cruise.pack_start(self.plot_holder, True, True, 0)
        cruise.pack_start(self.legend, False, False, 0)
        self.result_stack.add_named(cruise, "cruise")
        self.result_stack.set_hhomogeneous(False)
        self.result_stack.set_vhomogeneous(False)
        box.pack_start(self.result_stack, True, True, 0)
        return scroll

    def _log_panel(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8, margin=10)
        log_head = Gtk.Box(spacing=6)
        copy_button = Gtk.Button(label="複製日誌")
        copy_button.set_tooltip_text("複製目前已載入的日誌")
        copy_button.connect("clicked", self.copy_log)
        log_head.pack_start(copy_button, False, False, 0)
        self.autoscroll_check = Gtk.CheckButton(label="自動捲到最新")
        self.autoscroll_check.set_active(True)
        self.autoscroll_check.connect("toggled", self.on_autoscroll_toggled)
        log_head.pack_end(self.autoscroll_check, False, False, 0)
        box.pack_start(log_head, False, False, 0)
        self.log = Gtk.TextView(editable=False, cursor_visible=False, wrap_mode=Gtk.WrapMode.WORD_CHAR)
        self.log.get_style_context().add_class("log")
        self.log.set_left_margin(8)
        self.log.set_right_margin(8)
        self.log.set_top_margin(6)
        log_scroll = Gtk.ScrolledWindow(shadow_type=Gtk.ShadowType.IN, min_content_height=100)
        log_scroll.add(self.log)
        self.log_scroll = log_scroll
        box.pack_start(log_scroll, True, True, 0)
        return box

    def copy_log(self, _button: Gtk.Button) -> None:
        buffer = self.log.get_buffer()
        text = buffer.get_text(*buffer.get_bounds(), False)
        Gtk.Clipboard.get(Gdk.SELECTION_CLIPBOARD).set_text(text, -1)

    # ------------------------------------------------------------------------------------------ inputs
    def chooser_button(self, entry: Gtk.Entry, kind: str) -> Gtk.Button:
        button = Gtk.Button(label="選擇…")
        button.connect("clicked", lambda _b: self.choose(entry, kind))
        return button

    def choose(self, entry: Gtk.Entry, kind: str) -> None:
        path = self.choose_path(kind, entry.get_text().strip())
        if path:
            entry.set_text(path)

    def choose_path(self, kind: str, current: str = "") -> str | None:
        action = Gtk.FileChooserAction.SELECT_FOLDER if kind == "folder" else Gtk.FileChooserAction.OPEN
        dialog = Gtk.FileChooserDialog(title="請選擇", parent=self, action=action)
        dialog.add_buttons("取消", Gtk.ResponseType.CANCEL, "選擇", Gtk.ResponseType.OK)
        flt = Gtk.FileFilter()
        if kind == "video":
            flt.set_name("影片")
            for ext in ("mp4", "mov", "mkv", "avi", "m4v", "mts"):
                flt.add_pattern(f"*.{ext}")
                flt.add_pattern(f"*.{ext.upper()}")
            dialog.add_filter(flt)
        elif kind == "json":
            flt.set_name("JSON")
            flt.add_pattern("*.json")
            flt.add_pattern("*.JSON")
            dialog.add_filter(flt)
        if current and Path(current).expanduser().exists():
            dialog.set_filename(str(Path(current).expanduser()))
        path = dialog.get_filename() if dialog.run() == Gtk.ResponseType.OK else None
        dialog.destroy()
        return path

    def import_camera(self, _button: Gtk.Button) -> None:
        path = self.choose_path("json")
        if path is None:
            return
        try:
            profile = jobs.import_camera_profile(Path(path))
        except (OSError, ValueError) as error:
            self.error(f"無法匯入相機設定：{error}")
            return
        self.refresh_camera_choices(str(profile))
        self.update_readiness()

    def refresh_camera_choices(self, selected: str = "") -> None:
        combo = self.calibration_pick
        combo.remove_all()
        combo.append("", "選擇相機設定…")
        combo.append(str(jobs.BUILTIN_CAMERA_PROFILE), "內建預設 · Parrot ANAFI（1280×720）")
        for path in sorted(jobs.camera_profiles_dir().glob("*.json")):
            combo.append(str(path), path.stem)
        if self.engine == "moge3":
            combo.append("auto", "自動估算（未驗證）")
        if not combo.set_active_id(selected):
            combo.set_active(0)

    def on_camera_batch_changed(self, *_args) -> None:
        self.camera_confirm.set_active(False)
        selected = self.calibration_pick.get_active_id()
        if selected == "auto":
            text = "只供試建；不會驗證內參。影片須已去畸變，正式使用前需驗證定位。"
        elif selected == str(jobs.BUILTIN_CAMERA_PROFILE):
            text = "沿用目前河濱資料的 ANAFI 標定。請確認是相同錄影模式；只支援等比例縮放。"
        elif selected:
            text = "使用所選 JSON 的無畸變 PINHOLE 內參；請確認它對應這批影片的錄影模式。"
        else:
            text = "選擇或匯入標定檔。不同相機、裁切或錄影模式請分批建圖。"
        self.camera_note.set_text(text)

    def add_video_entry(self, path: str) -> None:
        if path and path not in [row[0] for row in self.video_store]:
            self.video_store.append([path])

    def add_folder(self, _button=None) -> None:
        dialog = Gtk.FileChooserDialog(title="選擇影片資料夾", parent=self, action=Gtk.FileChooserAction.SELECT_FOLDER)
        dialog.add_buttons("取消", Gtk.ResponseType.CANCEL, "加入", Gtk.ResponseType.OK)
        if dialog.run() == Gtk.ResponseType.OK:
            self.add_video_entry(dialog.get_filename())
        dialog.destroy()

    def add_files(self, _button=None) -> None:
        dialog = Gtk.FileChooserDialog(title="選擇影片", parent=self, action=Gtk.FileChooserAction.OPEN)
        dialog.add_buttons("取消", Gtk.ResponseType.CANCEL, "加入", Gtk.ResponseType.OK)
        dialog.set_select_multiple(True)
        flt = Gtk.FileFilter()
        flt.set_name("影片")
        for ext in ("mp4", "mov", "mkv", "avi", "m4v", "mts"):
            flt.add_pattern(f"*.{ext}")
            flt.add_pattern(f"*.{ext.upper()}")
        dialog.add_filter(flt)
        if dialog.run() == Gtk.ResponseType.OK:
            for path in dialog.get_filenames():
                self.add_video_entry(path)
        dialog.destroy()

    def remove_videos(self, _button=None) -> None:
        model, paths = self.video_view.get_selection().get_selected_rows()
        if not paths:
            return
        for ref in [Gtk.TreeRowReference.new(model, p) for p in paths]:
            model.remove(model.get_iter(ref.get_path()))

    def on_engine_toggled(self, radio: Gtk.RadioButton, key: str) -> None:
        if radio.get_active() and key != self.engine:
            self.engine = key
            self.on_engine_changed()

    def on_engine_changed(self) -> None:
        for key, radio in self.radios.items():
            if radio.get_active() != (key == self.engine):
                radio.set_active(key == self.engine)
        original = self.engine == "original"
        self.refresh_camera_choices(self.calibration_pick.get_active_id() or "")
        if hasattr(self, "engine_hint"):
            self.engine_hint.set_text("原始版：必須選擇相機標定" if original else "MoGe-3：可選相機標定或自動估算")
        # 兩個版本都能隨意混搭資料夾與單檔；原始版會在開始時自動拼成一個資料夾
        self.add_files_button.set_sensitive(True)
        self.video_hint.set_text("可混搭多個資料夾與影片檔，會自動找出裡面的影片" + ("（同批需同長寬比）" if original else ""))
        self.map_pick_hint.set_text(("原始版：選含 run/ 的資料夾" if original else "MoGe-3 版：選含 localization/bundle 的資料夾")
                                    + "；巡航前需先在建圖結果按「✓ 確認地圖可用」")
        self.refresh_map_choices(force=True)
        self.update_readiness()

    def on_map_pick(self, combo: Gtk.ComboBoxText) -> None:
        path = self.cruise_maps.get(combo.get_active_text() or "")
        if path:
            self.map_dir.set_text(path)

    def on_map_path_changed(self, entry: Gtk.Entry) -> None:
        if self.cruise_maps.get(self.map_pick.get_active_text()) != entry.get_text():
            self.map_pick.set_active(-1)

    def refresh_map_choices(self, force: bool = False) -> None:
        done = [j for j in jobs.list_jobs() if j["kind"] == "map" and j["status"] == "done" and j["engine"] == self.engine]
        mapping = {}
        for j in done:
            ok, _ = jobs.is_map_approved(j["result_dir"])
            mapping[f"{'✓' if ok else '○'} {j['name']} — {j['result_dir']}"] = j["result_dir"]
        if force or mapping != self.cruise_maps:
            self.cruise_maps = mapping
            self.map_pick.remove_all()
            for label in mapping:
                self.map_pick.append_text(label)
            path = self.map_dir.get_text().strip()
            if not path and mapping:
                self.map_pick.set_active(0)
            elif path:
                self.map_pick.set_active(next((index for index, target in enumerate(mapping.values()) if target == path), -1))

    # ------------------------------------------------------------------------------------------ readiness
    def map_request(self) -> dict:
        profile = self.calibration_pick.get_active_id()
        return {"kind": "map", "engine": self.engine, "name": self.name.get_text(),
                "videos": [row[0] for row in self.video_store], "out": self.out.get_text(),
                "intrinsics": profile if profile and profile != "auto" else "",
                "camera_profile": self.calibration_pick.get_active_text() if profile else "",
                "camera_confirmed": self.camera_confirm.get_active(),
                "auto_calibrate": profile == "auto", "dense": self.dense_check.get_active()}

    def cruise_request(self) -> dict:
        return {"kind": "cruise", "engine": self.engine, "map": self.map_dir.get_text(),
                "video": self.query.get_text(), "route": self.route.get_text()}

    def _show_readiness(self, label: Gtk.Label, ready: bool, message: str, button: Gtk.Button) -> None:
        label.set_text(("✓ " if ready else "○ ") + message)
        label.get_style_context().remove_class("ready")
        label.get_style_context().remove_class("notready")
        label.get_style_context().add_class("ready" if ready else "notready")
        button.set_sensitive(ready)

    def update_readiness(self) -> None:
        """Enable the start buttons only when the form is actually submittable."""
        if not hasattr(self, "start_map_button"):
            return
        videos = [row[0] for row in self.video_store]
        self.map_step1.set_text(f"步驟 1／4：加入影片（已選 {len(videos)} 項）")
        self.map_step1.get_style_context().remove_class("step-done")
        self.map_step1.get_style_context().remove_class("step")
        self.map_step1.get_style_context().add_class("step-done" if videos else "step")
        ready, message = jobs.validate_map_request(self.map_request())
        self._show_readiness(self.map_readiness, ready, message, self.start_map_button)
        ready, message = jobs.validate_cruise_request(self.cruise_request())
        self._show_readiness(self.cruise_readiness, ready, message, self.start_cruise_button)

    # ------------------------------------------------------------------------------------------ actions
    def error(self, message: str) -> None:
        dialog = Gtk.MessageDialog(transient_for=self, modal=True, message_type=Gtk.MessageType.ERROR,
                                   buttons=Gtk.ButtonsType.OK, text=message)
        dialog.run()
        dialog.destroy()

    def submit(self, request: dict) -> dict | None:
        try:
            job = jobs.create_job(request)
        except (ValueError, OSError) as error:
            self.error(str(error))
            return None
        self.signature = ()
        self.tick()
        self.select(job["id"])
        return job

    def start_map(self, _button=None) -> None:
        self.submit(self.map_request())

    def start_cruise(self, _button=None) -> None:
        self.submit(self.cruise_request())

    def resume_selected(self, _button=None) -> None:
        if not self.selected:
            return
        try:
            job = jobs.read_job(self.selected)
        except (OSError, ValueError):
            return
        if job.get("status") not in ("failed", "stopped", "interrupted"):
            return
        old_id = job["id"]
        new = self.submit(jobs.resume_request(job))
        if new is None:
            return
        try:
            jobs.delete_job(old_id)
        except (OSError, ValueError) as error:
            self.error(f"續跑已排入，但無法清除舊任務：{error}")
        self.tick()

    def confirm_delete(self, _button=None) -> None:
        if not self.selected:
            return
        try:
            job = jobs.read_job(self.selected)
        except (OSError, ValueError):
            return
        if job.get("status") not in ("done", "failed", "stopped", "interrupted"):
            return
        dialog = Gtk.MessageDialog(transient_for=self, modal=True, message_type=Gtk.MessageType.QUESTION,
                                   buttons=Gtk.ButtonsType.YES_NO,
                                   text=f"確定要刪除「{job.get('name', '')}」這筆任務嗎？\n"
                                        "會清除它的紀錄與暫存檔；建好的地圖檔案會保留，不會動。")
        answer = dialog.run()
        dialog.destroy()
        if answer != Gtk.ResponseType.YES:
            return
        try:
            jobs.delete_job(self.selected)
        except (OSError, ValueError) as error:
            self.error(str(error))
            return
        self.reset_selection(None)
        self.tick()

    def confirm_stop(self, _button=None) -> None:
        if not self.selected:
            return
        try:
            job = jobs.read_job(self.selected)
        except (OSError, ValueError):
            return
        if job.get("status") != "running":
            return
        dialog = Gtk.MessageDialog(transient_for=self, modal=True, message_type=Gtk.MessageType.QUESTION,
                                   buttons=Gtk.ButtonsType.YES_NO,
                                   text=f"確定要停止「{job.get('name', '')}」嗎？")
        dialog.format_secondary_text("已儲存的建圖階段會保留；未儲存的階段可能需要重算。")
        answer = dialog.run()
        dialog.destroy()
        if answer == Gtk.ResponseType.YES:
            jobs.stop_job(self.selected)

    def open_folder(self, _button=None) -> None:
        if self.selected:
            try:
                result = jobs.read_job(self.selected).get("result_dir")
            except (OSError, ValueError):
                return
            if result:
                path = Path(result)
                folder = path.parent if path.is_file() else path
                if folder.is_dir():
                    subprocess.Popen(["xdg-open", str(folder)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def open_result_row(self, view: Gtk.TreeView, path: Gtk.TreePath, _column) -> None:
        row = self.file_store[path]
        target = row[1]
        if not target or not Path(target).exists():
            return
        if target.lower().endswith(".ply"):
            self.view_result_cloud(row[0], Path(target))
            return
        subprocess.Popen(["xdg-open", target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def view_result_cloud(self, label: str, path: Path) -> None:
        """Double-clicked a point cloud row: inspect it in the 3D viewers."""
        try:
            self.viewer_info = self.viewer.load(path)
        except (OSError, ValueError) as error:
            self.viewer_label.set_text(f"3D 檢視載入失敗（{path.name}）：{error}")
            return
        self.viewer_source = f"{label}（{path.name}）"
        self.viewer_path = path
        self.viewer_frame.show()
        self.update_viewer_label()
        self.refresh_approval_ui()
        self.open_big_viewer()

    def open_big_viewer(self) -> None:
        """Show the current cloud in a big standalone window."""
        path = getattr(self, "viewer_path", None)
        if not path or not Path(path).is_file():
            return
        label = getattr(self, "viewer_source", "") or Path(path).name
        if self.big_viewer is None:
            self.big_viewer = MapViewerWindow()
            self.big_viewer.connect("destroy", self.on_big_viewer_closed)
        try:
            self.big_viewer.open_cloud(label, Path(path))
        except (OSError, ValueError) as error:
            self.error(f"大視窗載入失敗：{error}")

    def on_big_viewer_closed(self, _window) -> None:
        self.big_viewer = None

    def on_close(self, *_args) -> bool:
        if jobs.running_count():
            dialog = Gtk.MessageDialog(transient_for=self, modal=True, message_type=Gtk.MessageType.QUESTION,
                                       buttons=Gtk.ButtonsType.YES_NO, text="還有任務在執行，關閉視窗會停止它們。確定要關閉嗎？")
            answer = dialog.run()
            dialog.destroy()
            if answer != Gtk.ResponseType.YES:
                return True
            jobs.stop_all()
        jobs.release_gui_lock()
        Gtk.main_quit()
        return False

    # ------------------------------------------------------------------------------------------ job view
    def select(self, job_id: str) -> None:
        if self.selected != job_id:
            self.reset_selection(job_id)
            self.tick()

    def reset_selection(self, job_id: str | None) -> None:
        self.selected = job_id
        self.log_offset = 0
        self.shown = None
        self.log.get_buffer().set_text("")
        self.show_result(None)
        self.show_detail_tab(self.log_page if job_id else self.results_page)
        if job_id is None:
            self.title_label.set_text("尚未選擇任務")
            self.status_label.set_text("在左側建立任務，或從上方選擇紀錄。")
            for widget in (self.stage_label, self.phase_label, self.stage_bar, self.path_label):
                widget.hide()
            for button in (self.stop_button, self.resume_button, self.open_button, self.delete_button):
                button.set_sensitive(False)

    def show_detail_tab(self, page: Gtk.Widget) -> None:
        self.detail_tabs.set_current_page(self.detail_tabs.page_num(page))

    def on_detail_tab_changed(self, _tabs, page, _index) -> None:
        if page is self.log_page and self.autoscroll:
            GLib.idle_add(self.scroll_log_to_end)

    def on_autoscroll_toggled(self, button: Gtk.CheckButton) -> None:
        self.autoscroll = button.get_active()
        if self.autoscroll:
            GLib.idle_add(self.scroll_log_to_end)

    def scroll_log_to_end(self) -> bool:
        if self.autoscroll and self.log_page.get_mapped():
            adjustment = self.log_scroll.get_vadjustment()
            adjustment.set_value(adjustment.get_upper() - adjustment.get_page_size())
        return False

    def on_select(self, selection: Gtk.TreeSelection) -> None:
        model, tree_iter = selection.get_selected()
        if tree_iter and model[tree_iter][0] != self.selected:
            self.select(model[tree_iter][0])

    def tick(self) -> bool:
        self.update_readiness()
        listing = jobs.list_jobs()
        signature = tuple((j["id"], j["status"]) for j in listing)
        job = next((j for j in listing if j["id"] == self.selected), None)
        if job is None and listing:
            job = next((j for j in listing if j["status"] in ("running", "queued")), listing[0])
            self.reset_selection(job["id"])
        elif job is None and self.selected is not None:
            self.reset_selection(None)
        if signature != self.signature:
            self.signature = signature
            self.job_store.clear()
            running = queued = 0
            for j in listing:
                running += j["status"] == "running"
                queued += j["status"] == "queued"
                self.job_store.append([j["id"], j.get("name", ""), KIND_NAME.get(j["kind"], j["kind"]),
                                       STATUS.get(j["status"], j["status"]),
                                       time.strftime("%m-%d %H:%M", time.localtime(j["created"]))])
            counts = f"{running} 執行中 · {queued} 排隊 · 共 {len(listing)} 筆"
            self.task_summary.set_text(counts if listing else "尚無任務")
            self.refresh_map_choices()
        if job:
            selection = self.job_view.get_selection()
            model, row = selection.get_selected()
            if row is None or model[row][0] != self.selected:
                for candidate in model:
                    if candidate[0] == self.selected:
                        selection.select_iter(candidate.iter)
                        self.job_view.scroll_to_cell(candidate.path)
                        break
            self.update_detail(job)
        return True

    def update_stage(self, job: dict) -> None:
        """Distinguish job state, the current phase and progress within that phase."""
        status, kind = job["status"], job["kind"]
        self.stage_bar.set_visible(status == "running")
        if status != "running" or kind != "map":
            self.stage_label.hide()
        self.phase_label.set_visible(status in ("queued", "running", "done") or kind == "map")
        if status == "queued":
            self.phase_label.set_text("等待開始")
            return
        if status == "done":
            self.phase_label.set_text("任務完成 · 請查看成果")
            return
        if status != "running" and kind != "map":
            return

        info = jobs.parse_progress(jobs.read_log_tail(job["id"]))
        stage = info["stage"]
        if kind == "map":
            current = jobs.STAGES[stage][1] if stage is not None else "準備"
            self.stage_label.set_visible(status == "running" and stage is not None)
            if status != "running":
                self.phase_label.set_text(f"最後記錄階段：{current}")
                return
            original = job["engine"] == "original"
            steps = [(index, name) for index, (_, name) in enumerate(jobs.STAGES)
                     if not original or (index != 5 and (index != 3 or job.get("request", {}).get("dense")))]
            if stage is not None:
                self.stage_label.set_text("  ·  ".join(
                    f"{'✓' if index < stage else '▶' if index == stage else '○'} {name}" for index, name in steps))
            title = f"{current}中"
        else:
            title = {"cruise": "巡航定位中", "pack": "打包中", "export": "匯出場域中"}[kind]

        fraction = None
        details = []
        phase = None
        if kind == "map" and stage == 2 and original:
            engine_text, engine_age = jobs.read_engine_log_tail(job)
            if engine_age is not None and time.time() - engine_age >= job["created"]:
                phase = jobs.parse_mapping_phase(engine_text)
                details.append(f"日誌 {format_duration(engine_age)} 前更新")
        if phase and phase["phase"]:
            title = phase["phase"]
            remaining, elapsed = phase.get("remaining_s"), phase.get("elapsed_s")
            if remaining is not None and remaining >= 1:
                scope = "三角測量" if phase.get("done") else "此輪"
                details.insert(0, f"{scope}約剩 {format_duration(remaining)}")
            elif elapsed is not None:
                details.insert(0, f"此輪已用 {format_duration(elapsed)}")
            done, total = phase.get("done"), phase.get("total")
            if done and total and done < total and remaining:
                fraction = done / total
        elif (kind == "cruise" or (kind == "map" and stage == 2)) and info["total"] and info["done"] < info["total"]:
            title += f" · {info['done']}/{info['total']}"
            if info["remaining"] not in (None, "00:00"):
                details.insert(0, f"目前步驟約剩 {info['remaining']}")
            fraction = info["done"] / info["total"]
        elif kind == "map" and stage == 2 and info["estimate"]:
            details.insert(0, f"起始建圖估時 {info['estimate']}（非剩餘時間）")

        self.phase_label.set_text(" · ".join([title, *details]))
        if fraction is None:
            self.stage_bar.pulse()
        else:
            self.stage_bar.set_fraction(fraction)

    def update_detail(self, job: dict) -> None:
        title = f"{KIND_NAME.get(job['kind'], job['kind'])}：{job.get('name', '')}"
        self.title_label.set_text(title)
        self.title_label.set_tooltip_text(title)
        status = job["status"]
        now = time.time()
        start = job.get("created", now)
        end = job.get("finished") or now
        duration = f"{'耗時' if status in FINAL else '已用'} {format_duration(end - start)}"
        engine = "原始版" if job["engine"] == "original" else "MoGe-3"
        extra = f" · 結束碼 {job['exit_code']}" if job.get("exit_code") not in (0, None) else ""
        extra += f"\n{GLib.markup_escape_text(job['error'])}" if job.get("error") else ""
        state = STATUS.get(status, status)
        color = STATUS_COLOR.get(status)
        state = f'<span foreground="{color}">{state}</span>' if color else GLib.markup_escape_text(state)
        self.status_label.set_markup(f"<b>{state}</b> · {engine} · {duration}{extra}")
        path = job.get("result_dir") or ""
        self.path_label.set_text(f"輸出：{path}")
        self.path_label.set_tooltip_text(path or None)
        self.path_label.set_visible(bool(path))
        self.update_stage(job)
        self.stop_button.set_sensitive(status == "running")
        resumable = status in ("failed", "stopped", "interrupted")
        self.resume_button.set_sensitive(resumable)
        if resumable and job["kind"] == "map" and job["engine"] == "original":
            marks = {"skip": "可能跳過", "reuse": "待校驗", "redo": "預計重算"}
            preview = "；".join(f"{stage}：{marks[action]}" for stage, action in jobs.resume_plan(job))
            self.resume_button.set_tooltip_text(f"以同一設定續跑，實際從引擎認可的檢查點接續。\n{preview}")
        else:
            self.resume_button.set_tooltip_text("用原設定重新執行")
        self.delete_button.set_sensitive(status in FINAL)
        self.open_button.set_sensitive(bool(path) and Path(path).exists())
        chunk, self.log_offset = jobs.read_log(job["id"], self.log_offset)
        if chunk:
            buffer = self.log.get_buffer()
            buffer.insert(buffer.get_end_iter(), chunk.replace("\r", "\n"))
            if buffer.get_line_count() > 4000:
                buffer.delete(buffer.get_start_iter(), buffer.get_iter_at_line(buffer.get_line_count() - 3000))
            if self.autoscroll and self.log_page.get_mapped():
                GLib.idle_add(self.scroll_log_to_end)
        if status in FINAL and self.shown != (job["id"], status):
            self.shown = (job["id"], status)
            self.show_result(job)
            if status == "done":
                self.show_detail_tab(self.results_page)
            elif status == "failed":
                self.show_detail_tab(self.log_page)

    def update_viewer_label(self) -> None:
        text = self.viewer.describe()
        if getattr(self, "viewer_source", ""):
            text = f"【{self.viewer_source}】" + text
        self.viewer_label.set_text(text)

    def load_map_viewer(self, job: dict) -> None:
        """Show the 3D inspection + approval bar for a finished map job."""
        self.approved_dir = job.get("result_dir", "")
        self.approved_engine = job.get("engine", "")
        self.viewer_path = jobs.map_pointcloud(job)
        ply = jobs.map_pointcloud(job) if job["status"] == "done" else None
        if ply is None:
            self.viewer_frame.hide()
            self.connect_warn.hide()
            self.viewer_info = {}
            self.viewer_source = ""
            self.viewer_path = None
        else:
            try:
                self.viewer_info = self.viewer.load(ply)
                self.viewer_source = f"定位用地圖（{ply.name}）"
            except (OSError, ValueError) as error:
                self.viewer_frame.hide()
                self.connect_warn.hide()
                self.viewer_info = {}
                self.viewer_label.set_text(f"3D 檢視載入失敗：{error}")
                return
            self.viewer_frame.show()
            self.update_viewer_label()
        self.refresh_approval_ui()

    def refresh_approval_ui(self) -> None:
        if not self.approved_dir:
            return
        approved, info = jobs.is_map_approved(self.approved_dir)
        if approved:
            when = time.strftime("%m-%d %H:%M", time.localtime(info.get("approved_at", 0)))
            self.approval_label.set_markup(
                f'<span foreground="#1a7f37" weight="bold">✓ 已確認可用（{GLib.markup_escape_text(when)}）</span>'
                "　巡航定位已放行；若改了主意可再按一次重新確認")
            self.approve_button.set_label("↻ 重新確認")
        else:
            self.approval_label.set_text("○ 尚未確認：先在上方 3D 檢視檢查地圖（轉一轉、放大看看有沒有明顯破碎或錯位），沒問題再按右邊確認")
            self.approve_button.set_label("✓ 確認地圖可用，可以巡航")
        self.approve_button.set_sensitive(bool(self.viewer_info) or approved)
        self.export_button.set_visible(
            approved and self.approved_engine == "original"
            and not jobs.delivery_archive(self.approved_dir).is_file()
        )
        conn = jobs.map_connectivity(self.approved_dir)
        if conn is not None and not conn["connected"] and len(conn["components"]) > 1:
            names = "、".join(sorted({v["source"] for v in conn["videos"].values()}))
            self.connect_warn.set_text(
                f"{len(conn['components'])} 個區塊未達跨影片共視門檻（{names}）。"
                "不要把這份成果當成已接通的巡航地圖。")
            self.connect_warn.show()
        else:
            self.connect_warn.hide()

    def approve_current_map(self, _button=None) -> None:
        if not self.approved_dir:
            return
        try:
            info = dict(self.viewer_info)
            info["cloud"] = getattr(self, "viewer_source", "")
            jobs.approve_map(self.approved_dir, info)
        except OSError as error:
            self.error(f"寫入確認紀錄失敗：{error}")
            return
        self.refresh_approval_ui()
        self.refresh_map_choices(force=True)
        self.update_readiness()
        self.export_current_map()

    def export_current_map(self, _button=None) -> None:
        if not self.approved_dir or self.approved_engine != "original":
            return
        base = Path(self.approved_dir)
        archive = jobs.delivery_archive(base)
        if archive.is_file():
            return
        dense = "\n• map/map_dense.ply（稠密點雲）" if (base / "run/products/map/dense/dense.ply").is_file() else ""
        dialog = Gtk.MessageDialog(
            transient_for=self, modal=True, message_type=Gtk.MessageType.QUESTION,
            buttons=Gtk.ButtonsType.YES_NO, text="要匯出給另一台電腦的場域 ZIP 嗎？",
        )
        dialog.format_secondary_text(
            f"{archive}\n\n"
            f"• site_profile.json（場域設定）\n• map/map.ply（RGB 主地圖）{dense}\n"
            "• localization/（定位設定、對齊、參考資料）\n"
            "• model/（COLMAP 模型）\n• keyframes/（關鍵影格 JPEG 與清單）\n\n"
            "只要搬這個 ZIP，不包含原始影片或航線。到另一台電腦後請先解壓縮。"
        )
        answer = dialog.run()
        dialog.destroy()
        if answer == Gtk.ResponseType.YES:
            self.submit({"kind": "export", "engine": "original", "map": str(base)})

    def show_result(self, job: dict | None) -> None:
        self.plot_points = []
        self.transfer_hint.hide()
        self.connect_warn.hide()
        self.calibration_warn.set_visible(bool(job and job["kind"] == "map" and job["engine"] == "moge3"
                                               and jobs.uses_approximate_calibration(job)))
        if job is None or job["kind"] != "map":
            self.approved_dir = None
            self.approved_engine = ""
            self.viewer_path = None
            self.viewer_frame.hide()
        if job is None:
            self.result_hint.set_text("成果會在任務結束後顯示。")
            self.result_stack.set_visible_child_name("hint")
            return
        if job["kind"] == "map":
            rows = jobs.map_result(job)
            self.file_store.clear()
            for row in rows:
                self.file_store.append([row["label"], row["path"], row["size"]])
            if rows:
                self.result_stack.set_visible_child_name("files")
            else:
                hint = ("建圖失敗：請在「執行日誌」查看原因" if job["status"] == "failed"
                        else "任務被停止或中斷：沒有產出檔案" if job["status"] in ("stopped", "interrupted")
                        else "尚無產出檔案")
                self.result_hint.set_text(hint)
                self.result_stack.set_visible_child_name("hint")
            self.load_map_viewer(job)
            return
        if job["kind"] == "export":
            rows = jobs.export_result(job)
            self.file_store.clear()
            for row in rows:
                self.file_store.append([row["label"], row["path"], row["size"]])
            if rows:
                path = Path(rows[0]["path"])
                action = "複製這個 ZIP 並解壓縮" if path.is_file() else "複製整個資料夾"
                self.transfer_hint.set_text(
                    f"搬到另一台電腦：{path}\n"
                    f"{action}，再在定位端按「＋新增地圖」選直接包含 site_profile.json 的資料夾。"
                )
                self.transfer_hint.show()
                self.result_stack.set_visible_child_name("files")
            else:
                hint = {
                    "failed": "匯出失敗：請在「執行日誌」查看原因。",
                    "stopped": "匯出已停止。",
                    "interrupted": "匯出已中斷。",
                    "done": "找不到已匯出的場域交付檔案，可能已經搬移。",
                }.get(job["status"], "正在產生場域 ZIP，完成後會顯示要搬的檔案。")
                self.result_hint.set_text(hint)
                self.result_stack.set_visible_child_name("hint")
            return
        if job["kind"] == "pack":
            rows = jobs.pack_result(job)
            self.file_store.clear()
            for row in rows:
                self.file_store.append([row["label"], row["path"], row["size"]])
            if rows:
                self.result_stack.set_visible_child_name("files")
            else:
                hint = ("打包失敗：請在「執行日誌」查看原因，排除後再續跑" if job["status"] == "failed"
                        else "任務被停止或中斷：沒有產出轉接包" if job["status"] in ("stopped", "interrupted")
                        else "打包中…（轉接包完成後會列在這裡）")
                self.result_hint.set_text(hint)
                self.result_stack.set_visible_child_name("hint")
            return
        data = jobs.cruise_result(job)
        if data is None:
            hint = ("巡航失敗：請在「執行日誌」查看原因" if job["status"] == "failed"
                    else "任務被停止或中斷：沒有定位結果" if job["status"] in ("stopped", "interrupted")
                    else "尚無定位結果")
            self.result_hint.set_text(hint)
            self.result_stack.set_visible_child_name("hint")
            return
        counts = "　".join(f"{POSE_NAME.get(k, k)} {v}" for k, v in data["status_counts"].items())
        coverage = data["coverage"] if data["coverage"] is not None else 0.0
        self.cruise_summary.set_text(f"巡航航跡 · {data['frames']} 影格\n{counts}")
        self.coverage_bar.set_fraction(max(0.0, min(1.0, coverage)))
        self.coverage_bar.set_text(f"覆蓋率 {coverage * 100:.1f}%")
        self.legend.set_markup("　".join(
            f'<span foreground="#{r:02x}{g:02x}{b:02x}">●</span> {POSE_NAME[k]}' for k, (r, g, b) in POSE_COLOR.items())
            + "　　航跡俯視圖（地圖座標 X–Z，非公尺）")
        self.plot_points = data["points"]
        self.plot_width = 0
        self.result_stack.set_visible_child_name("cruise")
        self.redraw_plot()

    def redraw_plot(self) -> None:
        width = max(self.plot_holder.get_allocated_width() - 8, 320)
        if len(self.plot_points) < 2:
            self.plot_image.clear()
            return
        if width == self.plot_width:
            return
        self.plot_width = width
        self.plot_image.set_from_pixbuf(pixbuf_from_png(render_trajectory(self.plot_points, width, 230)))


def main() -> None:
    if not jobs.acquire_gui_lock():
        dialog = Gtk.MessageDialog(modal=True, message_type=Gtk.MessageType.INFO, buttons=Gtk.ButtonsType.OK,
                                   text="程式已經在執行中，請在工作列或桌面上找既有的視窗。")
        dialog.run()
        dialog.destroy()
        return
    window = App()
    window.show_all()
    Gtk.main()


if __name__ == "__main__":
    main()
