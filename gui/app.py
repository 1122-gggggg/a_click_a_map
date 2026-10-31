#!/usr/bin/env python3
"""無人機自動建圖巡航系統 - desktop application (GTK 3 via PyGObject; no browser, no network, no pip packages).

    ./gui.sh

建圖: add videos, press 開始建圖, watch the log, open the results.
巡航定位: pick a finished map and a query video; shows coverage, status counts and the flown trajectory.
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
FINAL = ("done", "failed", "stopped", "interrupted")
CSS = b"""
* { font-size: 11pt; }
.title { font-size: 19pt; font-weight: bold; }
.sub { color: #6e7781; }
.step { font-weight: bold; }
.hint { color: #6e7781; font-size: 10pt; }
.warn { color: #cf222e; font-size: 10pt; }
button.go { font-weight: bold; font-size: 13pt; padding: 8px 22px; background-image: none; background-color: #1a7f37; color: #ffffff; }
button.go:disabled { background-color: #94d3a2; }
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


class App(Gtk.Window):
    def __init__(self) -> None:
        super().__init__(title=TITLE)
        self.set_default_size(1320, 860)
        provider = Gtk.CssProvider()
        provider.load_from_data(CSS)
        Gtk.StyleContext.add_provider_for_screen(Gdk.Screen.get_default(), provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        self.engines = jobs.engine_status()
        self.engine = "moge3" if self.engines["moge3"]["ready"] or not self.engines["original"]["ready"] else "original"
        self.selected: str | None = None
        self.log_offset = 0
        self.shown: tuple[str, str] | None = None
        self.signature: tuple = ()
        self.plot_points: list = []
        self.plot_width = 0
        self.cruise_maps: dict[str, str] = {}
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
        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        root.pack_start(paned, True, True, 0)

        notebook = self.notebook = Gtk.Notebook()
        notebook.set_size_request(470, -1)
        notebook.append_page(self._map_page(), Gtk.Label(label="  建圖  "))
        notebook.append_page(self._cruise_page(), Gtk.Label(label="  巡航定位  "))
        paned.pack1(notebook, False, False)
        paned.pack2(self._right(), True, False)
        paned.set_position(480)

    def engine_box(self) -> Gtk.Widget:
        ready = [key for key in ("moge3", "original") if self.engines[key]["ready"]]
        if not ready:
            hints = "\n".join(f"{self.engines[k]['label']}：{self.engines[k]['hint']}" for k in self.engines)
            return css_label("系統尚未安裝完成，請聯絡工程人員。\n" + hints, "warn", wrap=True)
        if len(ready) == 1:  # a single installed engine needs no choice
            return Gtk.Box()
        frame = Gtk.Frame(label=" 建圖方式 ")
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2, margin=8)
        frame.add(box)
        group = None
        for key in ready:
            radio = Gtk.RadioButton.new_with_label_from_widget(group, self.engines[key]["label"])
            group = group or radio
            radio.set_active(key == self.engine)
            radio.connect("toggled", self.on_engine_toggled, key)
            self.radios.setdefault(key, []).append(radio)
            box.pack_start(radio, False, False, 0)
        return frame

    radios: dict = {}

    def _map_page(self) -> Gtk.Widget:
        self.radios = {}
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6, margin=12)
        page.pack_start(self.engine_box(), False, False, 0)
        page.pack_start(css_label("① 加入影片", "step"), False, False, 4)
        self.video_store = Gtk.ListStore(str)
        self.video_view = Gtk.TreeView(model=self.video_store, headers_visible=False)
        self.video_view.append_column(Gtk.TreeViewColumn("影片", Gtk.CellRendererText(ellipsize=Pango.EllipsizeMode.MIDDLE), text=0))
        self.video_view.get_selection().set_mode(Gtk.SelectionMode.MULTIPLE)
        scroll = Gtk.ScrolledWindow(shadow_type=Gtk.ShadowType.IN)
        scroll.set_size_request(-1, 130)
        scroll.add(self.video_view)
        page.pack_start(scroll, False, False, 0)
        buttons = Gtk.Box(spacing=6)
        for text, handler in (("加入資料夾…", self.add_folder), ("加入影片檔…", self.add_files), ("移除選取", self.remove_videos)):
            button = Gtk.Button(label=text)
            button.connect("clicked", handler)
            buttons.pack_start(button, False, False, 0)
        page.pack_start(buttons, False, False, 0)
        self.video_hint = css_label("", "hint")
        page.pack_start(self.video_hint, False, False, 0)

        page.pack_start(css_label("② 設定", "step"), False, False, 4)
        grid = Gtk.Grid(column_spacing=8, row_spacing=6)
        page.pack_start(grid, False, False, 0)
        self.name = Gtk.Entry(text="site", hexpand=True)
        self.out = Gtk.Entry(text=str(jobs.REPO / "acam_out"), hexpand=True)
        self.intrinsics = Gtk.Entry(hexpand=True, placeholder_text="留空使用預設")
        grid.attach(Gtk.Label(label="任務名稱", xalign=0), 0, 0, 1, 1)
        grid.attach(self.name, 1, 0, 2, 1)
        grid.attach(Gtk.Label(label="輸出位置", xalign=0), 0, 1, 1, 1)
        grid.attach(self.out, 1, 1, 1, 1)
        grid.attach(self.chooser_button(self.out, "folder"), 2, 1, 1, 1)
        grid.attach(Gtk.Label(label="相機標定檔", xalign=0), 0, 2, 1, 1)
        grid.attach(self.intrinsics, 1, 2, 1, 1)
        grid.attach(self.chooser_button(self.intrinsics, "json"), 2, 2, 1, 1)
        page.pack_start(css_label("相機標定檔為選填（無畸變 PINHOLE JSON）。沒有的話，MoGe-3 版會自動估計。", "hint", wrap=True), False, False, 0)
        self.skip_dense = Gtk.CheckButton(label="略過稠密化（較快，不產生稠密點雲；僅 MoGe-3 版）")
        page.pack_start(self.skip_dense, False, False, 4)
        page.pack_start(css_label("③ 開始", "step"), False, False, 4)
        self.start_map_button = Gtk.Button(label="開始建圖")
        self.start_map_button.get_style_context().add_class("go")
        self.start_map_button.connect("clicked", self.start_map)
        page.pack_start(self.start_map_button, False, False, 0)
        return page

    def _cruise_page(self) -> Gtk.Widget:
        page = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6, margin=12)
        page.pack_start(self.engine_box(), False, False, 0)
        page.pack_start(css_label("① 選擇已建好的地圖", "step"), False, False, 4)
        self.map_pick = Gtk.ComboBoxText()
        self.map_pick.connect("changed", self.on_map_pick)
        page.pack_start(self.map_pick, False, False, 0)
        row = Gtk.Box(spacing=6)
        self.map_dir = Gtk.Entry(hexpand=True, placeholder_text="或選擇其他地圖資料夾")
        row.pack_start(self.map_dir, True, True, 0)
        row.pack_start(self.chooser_button(self.map_dir, "folder"), False, False, 0)
        page.pack_start(row, False, False, 0)
        page.pack_start(css_label("MoGe-3 版：選含 localization/bundle 的資料夾；原始版：選含 run/ 的資料夾", "hint", wrap=True), False, False, 0)
        page.pack_start(css_label("② 選擇巡航影片", "step"), False, False, 4)
        row = Gtk.Box(spacing=6)
        self.query = Gtk.Entry(hexpand=True)
        row.pack_start(self.query, True, True, 0)
        row.pack_start(self.chooser_button(self.query, "video"), False, False, 0)
        page.pack_start(row, False, False, 0)
        page.pack_start(css_label("③ 航線名稱（選填）", "step"), False, False, 4)
        self.route = Gtk.Entry(placeholder_text="留空則用影片檔名")
        page.pack_start(self.route, False, False, 0)
        page.pack_start(css_label("④ 開始", "step"), False, False, 4)
        self.start_cruise_button = Gtk.Button(label="開始巡航定位")
        self.start_cruise_button.get_style_context().add_class("go")
        self.start_cruise_button.connect("clicked", self.start_cruise)
        page.pack_start(self.start_cruise_button, False, False, 0)
        return page

    def _right(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6, margin_start=8)
        box.pack_start(css_label("任務列表", "step"), False, False, 0)
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
        self.title_label = Gtk.Label(label="尚未選擇任務", xalign=0)
        self.title_label.get_style_context().add_class("step")
        self.status_label = Gtk.Label(xalign=0, use_markup=True)
        self.path_label = css_label("", "hint")
        self.path_label.set_selectable(True)
        self.path_label.set_line_wrap(True)
        for widget in (self.title_label, self.status_label, self.path_label):
            info.pack_start(widget, False, False, 0)
        top.pack_start(info, True, True, 0)
        self.stop_button = Gtk.Button(label="停止")
        self.stop_button.set_sensitive(False)
        self.stop_button.connect("clicked", lambda _b: self.selected and jobs.stop_job(self.selected))
        self.open_button = Gtk.Button(label="開啟資料夾")
        self.open_button.set_sensitive(False)
        self.open_button.connect("clicked", self.open_folder)
        top.pack_end(self.open_button, False, False, 0)
        top.pack_end(self.stop_button, False, False, 0)
        box.pack_start(top, False, False, 4)

        box.pack_start(css_label("結果", "step"), False, False, 0)
        self.result_stack = Gtk.Stack()
        self.result_hint = css_label("任務完成後顯示結果", "hint")
        self.result_stack.add_named(self.result_hint, "hint")
        self.file_store = Gtk.ListStore(str, str, str)
        files = Gtk.TreeView(model=self.file_store)
        for index, (title, width) in enumerate((("項目", 190), ("路徑", 420), ("大小", 80))):
            renderer = Gtk.CellRendererText(ellipsize=Pango.EllipsizeMode.MIDDLE)
            column = Gtk.TreeViewColumn(title, renderer, text=index)
            column.set_min_width(width)
            column.set_resizable(True)
            files.append_column(column)
        file_scroll = Gtk.ScrolledWindow(shadow_type=Gtk.ShadowType.IN)
        file_scroll.add(files)
        self.result_stack.add_named(file_scroll, "files")
        cruise = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.cruise_summary = Gtk.Label(xalign=0)
        self.plot_image = Gtk.Image()
        self.plot_holder = Gtk.ScrolledWindow(shadow_type=Gtk.ShadowType.IN)
        self.plot_holder.add(self.plot_image)
        self.plot_holder.connect("size-allocate", lambda _w, _a: self.redraw_plot())
        self.legend = Gtk.Label(xalign=0, use_markup=True)
        cruise.pack_start(self.cruise_summary, False, False, 0)
        cruise.pack_start(self.plot_holder, True, True, 0)
        cruise.pack_start(self.legend, False, False, 0)
        self.result_stack.add_named(cruise, "cruise")
        self.result_stack.set_size_request(-1, 300)
        box.pack_start(self.result_stack, False, False, 0)

        box.pack_start(css_label("執行日誌", "step"), False, False, 0)
        self.log = Gtk.TextView(editable=False, cursor_visible=False, wrap_mode=Gtk.WrapMode.CHAR)
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
        for ref in [Gtk.TreeRowReference.new(model, p) for p in paths]:
            model.remove(model.get_iter(ref.get_path()))

    def on_engine_toggled(self, radio: Gtk.RadioButton, key: str) -> None:
        if radio.get_active() and key != self.engine:
            self.engine = key
            self.on_engine_changed()

    def on_engine_changed(self) -> None:
        for key, widgets in self.radios.items():
            for radio in widgets:
                if radio.get_active() != (key == self.engine):
                    radio.set_active(key == self.engine)
        original = self.engine == "original"
        self.video_hint.set_text("原始版：只能加入一個影片資料夾" if original else "可加入多個資料夾或影片檔")
        self.skip_dense.set_sensitive(not original)
        self.refresh_map_choices(force=True)

    def on_map_pick(self, combo: Gtk.ComboBoxText) -> None:
        path = self.cruise_maps.get(combo.get_active_text() or "")
        if path:
            self.map_dir.set_text(path)

    def refresh_map_choices(self, force: bool = False) -> None:
        done = [j for j in jobs.list_jobs() if j["kind"] == "map" and j["status"] == "done" and j["engine"] == self.engine]
        mapping = {f"{j['name']} — {j['result_dir']}": j["result_dir"] for j in done}
        if force or mapping != self.cruise_maps:
            self.cruise_maps = mapping
            self.map_pick.remove_all()
            for label in mapping:
                self.map_pick.append_text(label)

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
        self.submit({"kind": "map", "engine": self.engine, "name": self.name.get_text(), "videos": [row[0] for row in self.video_store],
                     "out": self.out.get_text(), "intrinsics": self.intrinsics.get_text(), "skip_dense": self.skip_dense.get_active()})

    def start_cruise(self, _button=None) -> None:
        self.submit({"kind": "cruise", "engine": self.engine, "map": self.map_dir.get_text(), "video": self.query.get_text(),
                     "route": self.route.get_text()})

    def open_folder(self, _button=None) -> None:
        if self.selected:
            folder = jobs.read_job(self.selected).get("result_dir")
            if folder and Path(folder).exists():
                subprocess.Popen(["xdg-open", folder], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def on_close(self, *_args) -> bool:
        if jobs.running_count():
            dialog = Gtk.MessageDialog(transient_for=self, modal=True, message_type=Gtk.MessageType.QUESTION,
                                       buttons=Gtk.ButtonsType.YES_NO, text="還有任務在執行，關閉視窗會停止它們。確定要關閉嗎？")
            answer = dialog.run()
            dialog.destroy()
            if answer != Gtk.ResponseType.YES:
                return True
            jobs.stop_all()
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
        listing = jobs.list_jobs()
        signature = tuple((j["id"], j["status"]) for j in listing)
        if signature != self.signature:
            self.signature = signature
            self.job_store.clear()
            for j in listing:
                self.job_store.append([j["id"], "建圖" if j["kind"] == "map" else "巡航", j.get("name", ""), j["engine"],
                                       STATUS.get(j["status"], j["status"]), time.strftime("%m-%d %H:%M:%S", time.localtime(j["created"])),
                                       STATUS_COLOR.get(j["status"], "#24292f")])
            for row in self.job_store:
                if row[0] == self.selected:
                    self.job_view.get_selection().select_iter(row.iter)
            self.refresh_map_choices()
        job = next((j for j in listing if j["id"] == self.selected), None) if self.selected else None
        if job:
            self.update_detail(job)
        return True

    def update_detail(self, job: dict) -> None:
        self.title_label.set_text(f"{'建圖' if job['kind'] == 'map' else '巡航定位'}：{job.get('name', '')}")
        extra = f"　結束碼 {job['exit_code']}" if job.get("exit_code") is not None else ""
        extra += f"　{GLib.markup_escape_text(job['error'])}" if job.get("error") else ""
        self.status_label.set_markup(f'<span foreground="{STATUS_COLOR.get(job["status"], "#24292f")}" weight="bold">'
                                     f'{STATUS.get(job["status"], job["status"])}</span>　方式 {job["engine"]}{extra}')
        self.path_label.set_text(job.get("result_dir", ""))
        self.stop_button.set_sensitive(job["status"] == "running")
        self.open_button.set_sensitive(Path(job.get("result_dir", "")).exists())
        chunk, self.log_offset = jobs.read_log(job["id"], self.log_offset)
        if chunk:
            adjustment = self.log_scroll.get_vadjustment()
            at_bottom = adjustment.get_value() >= adjustment.get_upper() - adjustment.get_page_size() - 30
            buffer = self.log.get_buffer()
            buffer.insert(buffer.get_end_iter(), chunk.replace("\r", "\n"))
            if buffer.get_line_count() > 4000:
                buffer.delete(buffer.get_start_iter(), buffer.get_iter_at_line(buffer.get_line_count() - 3000))
            if at_bottom:
                GLib.idle_add(lambda: adjustment.set_value(adjustment.get_upper()) and False)
        if job["status"] in FINAL and self.shown != (job["id"], job["status"]):
            self.shown = (job["id"], job["status"])
            self.show_result(job)

    def show_result(self, job: dict | None) -> None:
        self.plot_points = []
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
                self.result_hint.set_text("尚無產出檔案")
                self.result_stack.set_visible_child_name("hint")
            return
        data = jobs.cruise_result(job)
        if data is None:
            self.result_hint.set_text("尚無定位結果")
            self.result_stack.set_visible_child_name("hint")
            return
        counts = "　".join(f"{POSE_NAME.get(k, k)} {v}" for k, v in data["status_counts"].items())
        coverage = f"{data['coverage'] * 100:.1f}%" if data["coverage"] is not None else "—"
        self.cruise_summary.set_text(f"影格數 {data['frames']}　位姿覆蓋率 {coverage}\n{counts}")
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
    window = App()
    window.show_all()
    Gtk.main()


if __name__ == "__main__":
    main()
