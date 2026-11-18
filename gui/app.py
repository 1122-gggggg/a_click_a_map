#!/usr/bin/env python3
"""無人機自動建圖巡航系統 - desktop application (GTK 3 via PyGObject; no browser, no network, no pip packages).

    ./gui.sh

操作只有兩條路（二選一），都在左邊分頁：
    建圖：     ① 加入影片 → ② 取名＋選輸出位置 → ③ 按「開始建圖」
    巡航定位： ① 選一張已建好的地圖 → ② 選一部巡航影片 → ③ 按「開始巡航定位」
右邊顯示任務列表、執行日誌與結果。按鈕在條件沒湊齊時會變灰，
下方會直接告訴你還缺什麼（不用等到按下去才被罵）。
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
KIND_NAME = {"map": "建圖", "cruise": "巡航定位", "pack": "打包"}
FINAL = ("done", "failed", "stopped", "interrupted")
CSS = b"""
* { font-size: 11pt; }
.title { font-size: 19pt; font-weight: bold; }
.sub { color: #6e7781; }
.step { font-weight: bold; }
.step-done { font-weight: bold; color: #1a7f37; }
.hint { color: #6e7781; font-size: 10pt; }
.ready { color: #1a7f37; font-size: 10pt; font-weight: bold; }
.notready { color: #9a6700; font-size: 10pt; }
.warn { color: #cf222e; font-size: 10pt; }
button.go { font-weight: bold; font-size: 13pt; padding: 10px 22px; background-image: none; background-color: #1a7f37; color: #ffffff; }
button.go:disabled { background-color: #94d3a2; color: #f0f0f0; }
textview.log, textview.log text { background-color: #0d1117; color: #e6edf3; font-family: monospace; font-size: 9pt; }
"""


def css_label(text: str, *classes: str, wrap: bool = False) -> Gtk.Label:
    label = Gtk.Label(label=text, xalign=0)
    for cls in classes:
        label.get_style_context().add_class(cls)
    if wrap:
        label.set_line_wrap(True)
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
    右鍵拖曳平移，滾輪縮放。Renders into a Gtk.Image via numpy.
    """

    BG = (13, 17, 23)
    MARK = (255, 180, 0)

    def __init__(self, *, expand: bool = False) -> None:
        self.points = np.zeros((0, 3))
        self.colors = np.zeros((0, 3), dtype=np.uint8)
        self.total = 0
        self.center = np.zeros(3)
        self.yaw, self.pitch, self.dist = 0.7, 0.45, 10.0
        self.span = np.ones(3)
        self.fit_span = 1.0
        self.outlier_ratio = 0.0
        self.expand = expand
        self.height = VIEWER_HEIGHT
        self.image = Gtk.Image()
        if not expand:
            self.image.set_size_request(-1, VIEWER_HEIGHT)
        self.box = Gtk.EventBox()
        self.box.add(self.image)
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

    def on_allocate(self, _widget, _allocation) -> None:
        if self.expand:
            fitted = max(self.box.get_allocated_height(), 200)
            if fitted != self.height:
                self.height = fitted
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

    def _project(self, width: int, height: int):
        rot = self._rotation()
        cam = (self.points - self.center) @ rot.T
        depth = cam[:, 2] + self.dist
        focal = height * 1.3
        ok = depth > 1e-6
        sx = np.full(len(self.points), -1)
        sy = np.full(len(self.points), -1)
        sx[ok] = (width / 2.0 + focal * cam[ok, 0] / depth[ok]).astype(int)
        sy[ok] = (height / 2.0 - focal * cam[ok, 1] / depth[ok]).astype(int)
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
            x, y = width // 2, height // 2
            for o in range(-5, 6):
                for xx, yy in ((x + o, y), (x, y + o)):
                    if 0 <= xx < width and 0 <= yy < height:
                        canvas[yy, xx] = self.MARK
        data = GLib.Bytes(canvas.tobytes())
        self.image.set_from_pixbuf(GdkPixbuf.Pixbuf.new_from_bytes(data, GdkPixbuf.Colorspace.RGB, False, 8, width, height, width * 3))

    def reset_view(self) -> None:
        self.yaw, self.pitch = 0.7, 0.45
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
            rot = self._rotation()
            scale = self.dist / (max(self.height, 200) * 1.3)
            right, up = rot.T[:, 0], -rot.T[:, 1]
            self.center = self.center + right * dx * scale - up * dy * scale
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
            self.center = (self.center + right * float(event.x - width / 2.0) * shift
                           + up * float(event.y - height / 2.0) * shift)
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
        hint = css_label("左鍵拖曳旋轉 · 左鍵點兩下設旋轉中心 · 右鍵拖曳平移 · 滾輪以游標為中心縮放", "hint", wrap=True)
        box.pack_start(hint, False, False, 0)
        box.pack_start(self.viewer.box, True, True, 0)
        row = Gtk.Box(spacing=6)
        self.label = css_label("", "hint", wrap=True)
        self.label.set_selectable(True)
        row.pack_start(self.label, True, True, 0)
        reset_button = Gtk.Button(label="重設視角")
        reset_button.connect("clicked", lambda _b: self.viewer.reset_view())
        row.pack_start(reset_button, False, False, 0)
        box.pack_start(row, False, False, 0)

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
        self.set_default_size(1320, 860)
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
        self.last_status: dict[str, str] = {}
        self.plot_points: list = []
        self.plot_width = 0
        self.plot_span = ""
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
        root = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8, margin=12)
        self.add(root)
        head = Gtk.Box(spacing=14)
        head.pack_start(css_label(TITLE, "title"), False, False, 0)
        head.pack_start(css_label(str(jobs.REPO), "sub"), False, False, 0)
        root.pack_start(head, False, False, 0)
        root.pack_start(self._engine_bar(), False, False, 0)
        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        root.pack_start(paned, True, True, 0)

        notebook = self.notebook = Gtk.Notebook()
        notebook.set_size_request(470, -1)
        notebook.append_page(self._map_page(), Gtk.Label(label="  ① 建圖  "))
        notebook.append_page(self._cruise_page(), Gtk.Label(label="  ② 巡航定位  "))
        paned.pack1(notebook, False, False)
        paned.pack2(self._right(), True, False)
        paned.set_position(480)

    def _engine_bar(self) -> Gtk.Widget:
        """A single engine switch for the whole window (both tabs share it)."""
        frame = Gtk.Frame(label=" 建圖方式（兩頁共用，先選這個） ")
        box = Gtk.Box(spacing=18, margin=8)
        frame.add(box)
        ready = [key for key in ("moge3", "original") if self.engines[key]["ready"]]
        if not ready:
            hints = "\n".join(f"{self.engines[k]['label']}：{self.engines[k]['hint']}" for k in self.engines)
            box.pack_start(css_label("系統尚未安裝完成，請聯絡工程人員。\n" + hints, "warn", wrap=True), False, False, 0)
            return frame
        if len(ready) == 1:
            box.pack_start(css_label(self.engines[ready[0]]["label"], "hint"), False, False, 0)
            return frame
        group = None
        for key in ("moge3", "original"):
            if key not in ready:
                continue
            radio = Gtk.RadioButton.new_with_label_from_widget(group, self.engines[key]["label"])
            radio.set_tooltip_text(self.engines[key]["label"])
            if group is None:
                group = radio
            radio.set_active(key == self.engine)
            radio.connect("toggled", self.on_engine_toggled, key)
            self.radios[key] = radio
            box.pack_start(radio, False, False, 0)
        self.engine_hint = css_label("", "hint")
        box.pack_start(self.engine_hint, False, False, 0)
        return frame

    def _step_frame(self, title_label: Gtk.Label, *children: Gtk.Widget) -> Gtk.Frame:
        frame = Gtk.Frame()
        frame.set_label_widget(title_label)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6, margin=10)
        frame.add(box)
        for child in children:
            box.pack_start(child, False, False, 0)
        return frame

    def _map_page(self) -> Gtk.Widget:
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8, margin=12)
        # --- step 1: videos ---
        self.map_step1 = css_label("步驟 1／3：加入影片", "step")
        video_list = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.video_store = Gtk.ListStore(str)
        self.video_view = Gtk.TreeView(model=self.video_store, headers_visible=False)
        self.video_view.set_tooltip_text("已加入的影片或資料夾。按住 Ctrl 可多選，再按「移除選取」。")
        self.video_view.append_column(Gtk.TreeViewColumn("影片", Gtk.CellRendererText(ellipsize=Pango.EllipsizeMode.MIDDLE), text=0))
        self.video_view.get_selection().set_mode(Gtk.SelectionMode.MULTIPLE)
        scroll = Gtk.ScrolledWindow(shadow_type=Gtk.ShadowType.IN)
        scroll.set_size_request(-1, 130)
        scroll.add(self.video_view)
        video_list.pack_start(scroll, False, False, 0)
        row1 = Gtk.Box(spacing=6)
        self.add_folder_button = Gtk.Button(label="加入資料夾…")
        self.add_folder_button.set_tooltip_text("選一個放滿影片的資料夾（會自動找出裡面的影片檔）")
        self.add_folder_button.connect("clicked", self.add_folder)
        self.add_files_button = Gtk.Button(label="加入影片檔…")
        self.add_files_button.set_tooltip_text("選一部或多部影片檔（可 Ctrl 多選）")
        self.add_files_button.connect("clicked", self.add_files)
        row1.pack_start(self.add_folder_button, False, False, 0)
        row1.pack_start(self.add_files_button, False, False, 0)
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
        video_list.pack_start(row2, False, False, 0)
        self.video_hint = css_label("", "hint")
        video_list.pack_start(self.video_hint, False, False, 0)
        page.pack_start(self._step_frame(self.map_step1, video_list), False, False, 0)

        # --- step 2: settings ---
        self.map_step2 = css_label("步驟 2／3：取名與輸出位置", "step")
        settings = Gtk.Grid(column_spacing=8, row_spacing=6)
        self.name = Gtk.Entry(text="site", hexpand=True)
        self.name.set_tooltip_text("任務名稱：只能用英文、數字、底線；會變成輸出資料夾的名字")
        self.out = Gtk.Entry(text=str(jobs.REPO / "acam_out"), hexpand=True)
        self.out.set_tooltip_text("建圖結果要放在哪個資料夾")
        self.intrinsics = Gtk.Entry(hexpand=True, placeholder_text="留空使用預設")
        self.intrinsics.set_tooltip_text("相機標定檔（JSON，需含 image_width、image_height 與 3x3 K 矩陣，且為已去畸變影像；沒有就留空；格式與範例見操作說明第 4 節）")
        settings.attach(Gtk.Label(label="任務名稱", xalign=0), 0, 0, 1, 1)
        settings.attach(self.name, 1, 0, 2, 1)
        settings.attach(Gtk.Label(label="輸出位置", xalign=0), 0, 1, 1, 1)
        settings.attach(self.out, 1, 1, 1, 1)
        settings.attach(self.chooser_button(self.out, "folder"), 2, 1, 1, 1)
        settings.attach(Gtk.Label(label="相機標定檔", xalign=0), 0, 2, 1, 1)
        settings.attach(self.intrinsics, 1, 2, 1, 1)
        settings.attach(self.chooser_button(self.intrinsics, "json"), 2, 2, 1, 1)
        settings_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        settings_box.pack_start(settings, False, False, 0)
        self.skip_dense = Gtk.CheckButton(label="略過稠密化（比較快，但不產生稠密點雲；只有 MoGe-3 版有效）")
        settings_box.pack_start(self.skip_dense, False, False, 0)
        page.pack_start(self._step_frame(self.map_step2, settings_box), False, False, 0)

        # --- step 3: go ---
        self.map_step3 = css_label("步驟 3／3：開始", "step")
        go = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.start_map_button = Gtk.Button(label="▶ 開始建圖")
        self.start_map_button.get_style_context().add_class("go")
        self.start_map_button.set_tooltip_text("條件都湊齊後才可以按；要花數小時，電腦不要關機")
        self.start_map_button.connect("clicked", self.start_map)
        go.pack_start(self.start_map_button, False, False, 0)
        self.map_readiness = css_label("", "notready")
        go.pack_start(self.map_readiness, False, False, 0)
        page.pack_start(self._step_frame(self.map_step3, go), False, False, 0)
        return page

    def _cruise_page(self) -> Gtk.Widget:
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8, margin=12)
        # --- step 1: map ---
        self.cruise_step1 = css_label("步驟 1／3：選擇已建好的地圖", "step")
        map_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.map_pick = Gtk.ComboBoxText()
        self.map_pick.set_tooltip_text("在這台電腦建好、已完成的建圖任務，直接選就行")
        self.map_pick.connect("changed", self.on_map_pick)
        map_box.pack_start(self.map_pick, False, False, 0)
        row = Gtk.Box(spacing=6)
        self.map_dir = Gtk.Entry(hexpand=True, placeholder_text="或選擇其他地圖資料夾…")
        self.map_dir.set_tooltip_text("也可以手動選：MoGe-3 版選含 localization/bundle 的資料夾；原始版選含 run/ 的資料夾")
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
        self.query = Gtk.Entry(hexpand=True, placeholder_text="點右邊「選擇…」挑一部影片")
        self.query.set_tooltip_text("想定位哪一部飛行影片，就選哪一部")
        row.pack_start(self.query, True, True, 0)
        row.pack_start(self.chooser_button(self.query, "video"), False, False, 0)
        query_box.pack_start(row, False, False, 0)
        page.pack_start(self._step_frame(self.cruise_step2, query_box), False, False, 0)

        # --- step 3: go ---
        self.cruise_step3 = css_label("步驟 3／3：開始", "step")
        go = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.route = Gtk.Entry(placeholder_text="航線名稱（選填，留空就用影片檔名）")
        self.route.set_tooltip_text("給這次巡航取個名字，結果會放在以它命名的資料夾")
        go.pack_start(self.route, False, False, 0)
        self.start_cruise_button = Gtk.Button(label="▶ 開始巡航定位")
        self.start_cruise_button.get_style_context().add_class("go")
        self.start_cruise_button.connect("clicked", self.start_cruise)
        go.pack_start(self.start_cruise_button, False, False, 0)
        self.cruise_readiness = css_label("", "notready")
        go.pack_start(self.cruise_readiness, False, False, 0)
        page.pack_start(self._step_frame(self.cruise_step3, go), False, False, 0)
        return page

    def _right(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6, margin_start=8)
        box.pack_start(css_label("任務列表（點一下看詳情）", "step"), False, False, 0)
        self.job_store = Gtk.ListStore(str, str, str, str, str, str, str)  # id, kind, name, engine, status, time, status colour
        self.job_view = Gtk.TreeView(model=self.job_store)
        for index, (title, width) in enumerate((("類型", 70), ("名稱", 230), ("方式", 80), ("狀態", 90), ("時間", 150)), start=1):
            renderer = Gtk.CellRendererText()
            column = Gtk.TreeViewColumn(title, renderer, text=index)
            if title == "狀態":
                column.add_attribute(renderer, "foreground", 6)
            column.set_min_width(width)
            self.job_view.append_column(column)
        self.job_view.get_selection().connect("changed", self.on_select)
        scroll = Gtk.ScrolledWindow(shadow_type=Gtk.ShadowType.IN)
        scroll.set_size_request(-1, 150)
        scroll.add(self.job_view)
        box.pack_start(scroll, False, False, 0)

        top = Gtk.Box(spacing=8)
        info = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.title_label = Gtk.Label(label="還沒選任務：在左邊開始一個，或點上面的任務列表", xalign=0)
        self.title_label.get_style_context().add_class("step")
        self.status_label = Gtk.Label(xalign=0, use_markup=True)
        self.elapsed_label = css_label("", "hint")
        self.stage_label = Gtk.Label(xalign=0, use_markup=True)
        self.stage_bar = Gtk.ProgressBar(show_text=True)
        self.stage_bar.set_no_show_all(True)
        self.path_label = css_label("", "hint")
        self.path_label.set_selectable(True)
        self.path_label.set_line_wrap(True)
        for widget in (self.title_label, self.status_label, self.elapsed_label, self.stage_label, self.stage_bar, self.path_label):
            info.pack_start(widget, False, False, 0)
        top.pack_start(info, True, True, 0)
        self.stop_button = Gtk.Button(label="■ 停止")
        self.stop_button.set_tooltip_text("中斷這個任務（會先問你確認）")
        self.stop_button.set_sensitive(False)
        self.stop_button.connect("clicked", self.confirm_stop)
        self.resume_button = Gtk.Button(label="↻ 續跑")
        self.resume_button.set_tooltip_text("用同樣的設定再跑一次；建圖會從已完成的階段接續，不會重做")
        self.resume_button.set_sensitive(False)
        self.resume_button.connect("clicked", self.resume_selected)
        self.delete_button = Gtk.Button(label="刪除任務")
        self.delete_button.set_tooltip_text("清除這筆任務的紀錄與暫存（建好的地圖檔案會保留）")
        self.delete_button.set_sensitive(False)
        self.delete_button.connect("clicked", self.confirm_delete)
        self.open_button = Gtk.Button(label="開啟資料夾")
        self.open_button.set_tooltip_text("用檔案總管打開結果資料夾")
        self.open_button.set_sensitive(False)
        self.open_button.connect("clicked", self.open_folder)
        top.pack_end(self.open_button, False, False, 0)
        top.pack_end(self.resume_button, False, False, 0)
        top.pack_end(self.delete_button, False, False, 0)
        top.pack_end(self.stop_button, False, False, 0)
        box.pack_start(top, False, False, 4)

        box.pack_start(css_label("結果（完成後出現在這裡；點雲檔點兩下開大視窗檢視，其他檔名點兩下用外部程式開啟）", "step"), False, False, 0)
        self.result_stack = Gtk.Stack()
        self.result_hint = css_label("任務完成後顯示結果", "hint")
        self.result_stack.add_named(self.result_hint, "hint")
        self.file_store = Gtk.ListStore(str, str, str)
        files = Gtk.TreeView(model=self.file_store)
        files.set_tooltip_text("點兩下開啟檔案或資料夾")
        for index, (title, width) in enumerate((("項目", 190), ("路徑", 420), ("大小", 80))):
            renderer = Gtk.CellRendererText(ellipsize=Pango.EllipsizeMode.MIDDLE)
            column = Gtk.TreeViewColumn(title, renderer, text=index)
            column.set_min_width(width)
            column.set_resizable(True)
            files.append_column(column)
        files.connect("row-activated", self.open_result_row)
        map_page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.viewer_frame = Gtk.Frame(label=" 3D 檢查：左鍵拖曳旋轉 · 左鍵點兩下設旋轉中心 · 右鍵拖曳平移 · 滾輪以游標為中心縮放 ")
        viewer_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4, margin=6)
        self.viewer = MapViewer()
        self.viewer.on_change = self.update_viewer_label
        viewer_box.pack_start(self.viewer.box, False, False, 0)
        self.viewer_label = css_label("", "hint", wrap=True)
        viewer_box.pack_start(self.viewer_label, False, False, 0)
        approval_row = Gtk.Box(spacing=6)
        self.approval_label = css_label("", "hint")
        self.approval_label.set_line_wrap(True)
        approval_row.pack_start(self.approval_label, True, True, 0)
        self.approve_button = Gtk.Button(label="✓ 確認地圖可用，可以巡航")
        self.approve_button.set_tooltip_text("在上方 3D 檢視裡轉一轉、確認地圖沒有明顯問題後再按。按了巡航定位才會放行。")
        self.approve_button.connect("clicked", self.approve_current_map)
        approval_row.pack_start(self.approve_button, False, False, 0)
        self.pack_button = Gtk.Button(label="📦 打包轉接包")
        self.pack_button.set_tooltip_text("把這張地圖包成可搬到另一台電腦的轉接包；只在這台巡航可以不打")
        self.pack_button.connect("clicked", self.pack_current_map)
        approval_row.pack_start(self.pack_button, False, False, 0)
        big_button = Gtk.Button(label="🔍 大視窗")
        big_button.set_tooltip_text("用大視窗看目前這份點雲（跟點兩下檔名一樣）")
        big_button.connect("clicked", lambda _b: self.open_big_viewer())
        approval_row.pack_start(big_button, False, False, 0)
        reset_button = Gtk.Button(label="重設視角")
        reset_button.set_tooltip_text("回到一開始的整體視角")
        reset_button.connect("clicked", lambda _b: self.viewer.reset_view())
        approval_row.pack_start(reset_button, False, False, 0)
        viewer_box.pack_start(approval_row, False, False, 0)
        self.viewer_frame.add(viewer_box)
        map_page.pack_start(self.viewer_frame, False, False, 0)
        file_scroll = Gtk.ScrolledWindow(shadow_type=Gtk.ShadowType.IN)
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
        self.cruise_summary = Gtk.Label(xalign=0)
        self.coverage_bar = Gtk.ProgressBar(show_text=True)
        self.plot_image = Gtk.Image()
        self.plot_holder = Gtk.ScrolledWindow(shadow_type=Gtk.ShadowType.IN)
        self.plot_holder.add(self.plot_image)
        self.plot_holder.connect("size-allocate", lambda _w, _a: self.redraw_plot())
        self.legend = Gtk.Label(xalign=0, use_markup=True)
        cruise.pack_start(self.cruise_summary, False, False, 0)
        cruise.pack_start(self.coverage_bar, False, False, 0)
        cruise.pack_start(self.plot_holder, True, True, 0)
        cruise.pack_start(self.legend, False, False, 0)
        self.result_stack.add_named(cruise, "cruise")
        self.result_stack.set_size_request(-1, 300)
        box.pack_start(self.result_stack, False, False, 0)

        log_head = Gtk.Box(spacing=8)
        log_head.pack_start(css_label("執行日誌（卡住時把這段文字給工程人員看）", "step"), True, True, 0)
        self.autoscroll_check = Gtk.CheckButton(label="自動捲到最新")
        self.autoscroll_check.set_active(True)
        self.autoscroll_check.connect("toggled", lambda b: setattr(self, "autoscroll", b.get_active()))
        log_head.pack_start(self.autoscroll_check, False, False, 0)
        box.pack_start(log_head, False, False, 0)
        self.log = Gtk.TextView(editable=False, cursor_visible=False, wrap_mode=Gtk.WrapMode.WORD_CHAR)
        self.log.get_style_context().add_class("log")
        log_scroll = Gtk.ScrolledWindow(shadow_type=Gtk.ShadowType.IN)
        log_scroll.add(self.log)
        self.log_scroll = log_scroll
        box.pack_start(log_scroll, True, True, 0)
        return box

    # ------------------------------------------------------------------------------------------ inputs
    def chooser_button(self, entry: Gtk.Entry, kind: str) -> Gtk.Button:
        button = Gtk.Button(label="選擇…")
        button.connect("clicked", lambda _b: self.choose(entry, kind))
        return button

    def choose(self, entry: Gtk.Entry, kind: str) -> None:
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
            dialog.add_filter(flt)
        current = entry.get_text().strip()
        if current and Path(current).expanduser().exists():
            dialog.set_filename(str(Path(current).expanduser()))
        if dialog.run() == Gtk.ResponseType.OK:
            entry.set_text(dialog.get_filename())
        dialog.destroy()

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
        if hasattr(self, "engine_hint"):
            self.engine_hint.set_text("原始版：一次一批（同批需同長寬比），輸出轉接包；巡航要另外給查詢影片" if original
                                      else "MoGe-3 版：可一次加入多個資料夾或影片檔")
        # 兩個版本都能隨意混搭資料夾與單檔；原始版會在開始時自動拼成一個資料夾
        self.add_files_button.set_sensitive(True)
        self.video_hint.set_text("可混搭多個資料夾與影片檔，會自動找出裡面的影片" + ("（同批需同長寬比）" if original else ""))
        self.skip_dense.set_sensitive(not original)
        self.map_pick_hint.set_text(("原始版：選含 run/ 的資料夾" if original else "MoGe-3 版：選含 localization/bundle 的資料夾")
                                    + "；巡航前需先在建圖結果按「✓ 確認地圖可用」")
        self.refresh_map_choices(force=True)
        self.update_readiness()

    def on_map_pick(self, combo: Gtk.ComboBoxText) -> None:
        path = self.cruise_maps.get(combo.get_active_text() or "")
        if path:
            self.map_dir.set_text(path)

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
            if mapping and not self.map_dir.get_text().strip():
                first = next(iter(mapping.values()))
                self.map_dir.set_text(first)

    # ------------------------------------------------------------------------------------------ readiness
    def map_request(self) -> dict:
        return {"kind": "map", "engine": self.engine, "name": self.name.get_text(),
                "videos": [row[0] for row in self.video_store], "out": self.out.get_text(),
                "intrinsics": self.intrinsics.get_text(), "skip_dense": self.skip_dense.get_active()}

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
        self.map_step1.set_text(f"步驟 1／3：加入影片（已選 {len(videos)} 項）")
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

    def submit(self, request: dict) -> None:
        try:
            job = jobs.create_job(request)
        except (ValueError, OSError) as error:
            self.error(str(error))
            return
        self.signature = ()
        self.tick()
        self.select(job["id"])

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
        self.submit(jobs.resume_request(job))

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
        self.selected = None
        self.log_offset = 0
        self.shown = None
        self.log.get_buffer().set_text("")
        self.show_result(None)
        self.title_label.set_text("還沒選任務：在左邊開始一個，或點上面的任務列表")
        self.status_label.set_text("")
        self.elapsed_label.set_text("")
        self.stage_label.set_text("")
        self.stage_bar.hide()
        self.path_label.set_text("")
        self.stop_button.set_sensitive(False)
        self.resume_button.set_sensitive(False)
        self.delete_button.set_sensitive(False)
        self.open_button.set_sensitive(False)
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
                                   text=f"確定要停止「{job.get('name', '')}」嗎？\n停掉後這次執行的進度就沒了。")
        answer = dialog.run()
        dialog.destroy()
        if answer == Gtk.ResponseType.YES:
            jobs.stop_job(self.selected)

    def open_folder(self, _button=None) -> None:
        if self.selected:
            try:
                folder = jobs.read_job(self.selected).get("result_dir")
            except (OSError, ValueError):
                return
            if folder and Path(folder).exists():
                subprocess.Popen(["xdg-open", folder], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

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
        self.viewer_frame.show_all()
        self.update_viewer_label()
        self.refresh_approval_ui()
        self.open_big_viewer()

    def auto_show_cloud(self, job: dict) -> None:
        """A map job just finished while we watched: pop open its 3D cloud."""
        ply = jobs.map_pointcloud(job)
        if ply is None:
            return
        label = f"定位用地圖（{ply.name}）"
        if self.big_viewer is None:
            self.big_viewer = MapViewerWindow()
            self.big_viewer.connect("destroy", self.on_big_viewer_closed)
        try:
            self.big_viewer.open_cloud(label, ply)
        except (OSError, ValueError) as error:
            self.error(f"3D 檢視載入失敗（{ply.name}）：{error}")

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
        self.selected = job_id
        self.log_offset = 0
        self.shown = None
        self.log.get_buffer().set_text("")
        for row in self.job_store:
            if row[0] == job_id:
                self.job_view.get_selection().select_iter(row.iter)
        self.show_result(None)
        self.tick()

    def on_select(self, selection: Gtk.TreeSelection) -> None:
        model, tree_iter = selection.get_selected()
        if tree_iter and model[tree_iter][0] != self.selected:
            self.select(model[tree_iter][0])

    def tick(self) -> bool:
        self.update_readiness()
        listing = jobs.list_jobs()
        signature = tuple((j["id"], j["status"]) for j in listing)
        if signature != self.signature:
            self.signature = signature
            self.job_store.clear()
            for j in listing:
                self.job_store.append([j["id"], KIND_NAME.get(j["kind"], j["kind"]), j.get("name", ""), j["engine"],
                                       STATUS.get(j["status"], j["status"]), time.strftime("%m-%d %H:%M:%S", time.localtime(j["created"])),
                                       STATUS_COLOR.get(j["status"], "#24292f")])
            for row in self.job_store:
                if row[0] == self.selected:
                    self.job_view.get_selection().select_iter(row.iter)
            self.refresh_map_choices()
        job = next((j for j in listing if j["id"] == self.selected), None) if self.selected else None
        for j in listing:
            previous = self.last_status.get(j["id"])
            self.last_status[j["id"]] = j["status"]
            if (previous in ("queued", "running") and j["status"] == "done" and j["kind"] == "map"):
                self.auto_show_cloud(j)
        if job:
            self.update_detail(job)
        return True

    def update_stage(self, job: dict) -> None:
        """Show which build step is running plus the in-step counter and ETA."""
        if job["kind"] != "map" or job["status"] == "queued":
            if job["kind"] == "map":
                self.stage_label.set_text("")
                self.stage_bar.hide()
                return
            if job["status"] != "running":
                self.stage_label.set_text("")
                self.stage_bar.hide()
                return
            info = jobs.parse_progress(jobs.read_log_tail(job["id"]))
            self.stage_bar.show()
            if info["percent"] is not None and info["total"]:
                self.stage_bar.set_fraction(info["done"] / info["total"])
                self.stage_bar.set_text(f"{info['done']}/{info['total']}　約剩 {info['remaining']}")
                self.stage_label.set_text(f"定位中：{info['done']}/{info['total']}（約剩 {info['remaining']}）")
            else:
                self.stage_bar.set_fraction(0.0)
                self.stage_bar.set_text("定位中…")
                self.stage_bar.pulse()
                self.stage_label.set_text("定位中…")
            return
        info = jobs.parse_progress(jobs.read_log_tail(job["id"]))
        stage = info["stage"]
        if job["status"] in ("done", "failed", "stopped", "interrupted"):
            if job["status"] == "done":
                marks = " → ".join(f"✓ {name}" for _, name in jobs.STAGES)
                self.stage_label.set_markup(f'<span foreground="#1a7f37">{marks}</span>')
                self.stage_bar.show()
                self.stage_bar.set_fraction(1.0)
                self.stage_bar.set_text("全部完成")
            else:
                where = jobs.STAGES[stage][1] if stage is not None else "開始前"
                self.stage_label.set_text(f"停在「{where}」")
                self.stage_bar.hide()
            return
        parts = []
        for index, (_, name) in enumerate(jobs.STAGES):
            if stage is not None and index < stage:
                parts.append(f"✓ {name}")
            elif stage is not None and index == stage:
                parts.append(f"▶ {name}")
            else:
                parts.append(f"○ {name}")
        self.stage_label.set_text(" → ".join(parts))
        self.stage_bar.show()
        if info["percent"] is not None and info["total"]:
            self.stage_bar.set_fraction(info["done"] / info["total"])
            current = jobs.STAGES[stage][1] if stage is not None else ""
            text = f"{current} {info['done']}/{info['total']}"
            if info["remaining"]:
                text += f"　約剩 {info['remaining']}"
            self.stage_bar.set_text(text)
        else:
            current = jobs.STAGES[stage][1] if stage is not None else "準備中"
            extra = f"（整體約 {info['estimate']}）" if info["estimate"] and (stage or 0) <= 2 else ""
            self.stage_bar.set_fraction(0.0)
            self.stage_bar.set_text(f"{current}進行中…{extra}")
            self.stage_bar.pulse()

    def update_detail(self, job: dict) -> None:
        self.title_label.set_text(f"{KIND_NAME.get(job['kind'], job['kind'])}：{job.get('name', '')}")
        extra = f"　結束碼 {job['exit_code']}" if job.get("exit_code") is not None else ""
        extra += f"　{GLib.markup_escape_text(job['error'])}" if job.get("error") else ""
        self.status_label.set_markup(f'<span foreground="{STATUS_COLOR.get(job["status"], "#24292f")}" weight="bold">'
                                     f'{STATUS.get(job["status"], job["status"])}</span>　方式 {job["engine"]}{extra}')
        start = job.get("created", time.time())
        end = job.get("finished", time.time()) if job["status"] in FINAL else time.time()
        verb = "耗時" if job["status"] in FINAL else "已執行"
        self.elapsed_label.set_text(f"{verb} {format_duration(end - start)}"
                                    + ("　（執行中：看下方日誌確認還活著）" if job["status"] == "running" else ""))
        self.path_label.set_text(job.get("result_dir", ""))
        self.update_stage(job)
        self.stop_button.set_sensitive(job["status"] == "running")
        self.resume_button.set_sensitive(job["status"] in ("failed", "stopped", "interrupted"))
        self.delete_button.set_sensitive(job["status"] in ("done", "failed", "stopped", "interrupted"))
        self.open_button.set_sensitive(Path(job.get("result_dir", "")).exists())
        chunk, self.log_offset = jobs.read_log(job["id"], self.log_offset)
        if chunk:
            buffer = self.log.get_buffer()
            buffer.insert(buffer.get_end_iter(), chunk.replace("\r", "\n"))
            if buffer.get_line_count() > 4000:
                buffer.delete(buffer.get_start_iter(), buffer.get_iter_at_line(buffer.get_line_count() - 3000))
            if self.autoscroll:
                adjustment = self.log_scroll.get_vadjustment()
                GLib.idle_add(lambda: adjustment.set_value(adjustment.get_upper()) and False)
        if job["status"] in FINAL and self.shown != (job["id"], job["status"]):
            self.shown = (job["id"], job["status"])
            self.show_result(job)

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
            self.viewer_info = {}
            self.viewer_source = ""
            self.viewer_path = None
        else:
            try:
                self.viewer_info = self.viewer.load(ply)
                self.viewer_source = f"定位用地圖（{ply.name}）"
            except (OSError, ValueError) as error:
                self.viewer_frame.hide()
                self.viewer_info = {}
                self.viewer_label.set_text(f"3D 檢視載入失敗：{error}")
                return
            self.viewer_frame.show_all()
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
        show_pack = (approved and self.approved_engine == "original"
                     and jobs.map_bundle(self.approved_dir) is None)
        self.pack_button.set_visible(show_pack)

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
        if (self.approved_engine == "original" and jobs.map_bundle(self.approved_dir) is None
                and self.ask_pack_now()):
            self.submit({"kind": "pack", "engine": "original", "map": self.approved_dir})

    def ask_pack_now(self) -> bool:
        rows = jobs.pack_preview(self.approved_dir)
        if not rows:
            self.error("找不到可打包的檔案（地圖輸出不完整），請檢查建圖結果。")
            return False
        lines = "\n".join(f"• {r['label']}：{r['size']}（{r.get('detail', '')}）" for r in rows)
        dialog = Gtk.MessageDialog(transient_for=self, modal=True, message_type=Gtk.MessageType.QUESTION,
                                   buttons=Gtk.ButtonsType.YES_NO,
                                   text="地圖已確認可用。要現在打包轉接包嗎？\n\n" + lines + "\n\n"
                                        "另外會附上：定位程式、安裝腳本、MegaLoc 權重。\n"
                                        "EDM 定位權重不會打包（另一台電腦需自備，放好再定位）。\n"
                                        "只在這台巡航可按否，以後隨時按「📦 打包轉接包」補打。")
        answer = dialog.run()
        dialog.destroy()
        return answer == Gtk.ResponseType.YES

    def pack_current_map(self, _button=None) -> None:
        if not self.approved_dir or self.approved_engine != "original":
            return
        if jobs.map_bundle(self.approved_dir) is not None:
            return
        if self.ask_pack_now():
            self.submit({"kind": "pack", "engine": "original", "map": self.approved_dir})

    def show_result(self, job: dict | None) -> None:
        self.plot_points = []
        if job is None or job["kind"] != "map":
            self.approved_dir = None
            self.approved_engine = ""
            self.viewer_path = None
            self.viewer_frame.hide()
        if job is None:
            self.result_hint.set_text("任務完成後顯示結果")
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
                hint = ("建圖失敗：看上方日誌找原因，修好後再跑一次" if job["status"] == "failed"
                        else "任務被停止或中斷：沒有產出檔案" if job["status"] in ("stopped", "interrupted")
                        else "尚無產出檔案")
                self.result_hint.set_text(hint)
                self.result_stack.set_visible_child_name("hint")
            self.load_map_viewer(job)
            return
        if job["kind"] == "pack":
            rows = jobs.pack_result(job)
            self.file_store.clear()
            for row in rows:
                self.file_store.append([row["label"], row["path"], row["size"]])
            if rows:
                self.result_stack.set_visible_child_name("files")
            else:
                hint = ("打包失敗：看上方日誌找原因，修好後按「↻ 續跑」再打一次" if job["status"] == "failed"
                        else "任務被停止或中斷：沒有產出轉接包" if job["status"] in ("stopped", "interrupted")
                        else "打包中…（轉接包完成後會列在這裡）")
                self.result_hint.set_text(hint)
                self.result_stack.set_visible_child_name("hint")
            return
        data = jobs.cruise_result(job)
        if data is None:
            hint = ("巡航失敗：看上方日誌找原因" if job["status"] == "failed"
                    else "任務被停止或中斷：沒有定位結果" if job["status"] in ("stopped", "interrupted")
                    else "尚無定位結果")
            self.result_hint.set_text(hint)
            self.result_stack.set_visible_child_name("hint")
            return
        counts = "　".join(f"{POSE_NAME.get(k, k)} {v}" for k, v in data["status_counts"].items())
        coverage = data["coverage"] if data["coverage"] is not None else 0.0
        self.cruise_summary.set_text(f"影格數 {data['frames']}　位姿覆蓋率 {coverage * 100:.1f}%\n{counts}")
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
