"""Job backend of the desktop UI: build engine commands, run them as subprocesses, read results.

No UI code and no network code here. Jobs live in acam_gui/jobs/<id>/ (job.json + log.txt).
Engines:
    original  ./run.sh        (folder of videos in; run.sh map / run.sh localize)
    moge3     ./run_moge3.sh  (videos in; sparse + MVSFormer++ dense + EDM/MegaLoc bundle)
"""
from __future__ import annotations

import json
import numpy as np
import re
import os
import sys
import shlex
import signal
import subprocess
import shutil
import threading
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from acam.calibration import load_calibration  # noqa: E402

BUILTIN_CAMERA_PROFILE = REPO / "configs/river_intrinsics_undistorted.json"
JOBS = REPO / "acam_gui" / "jobs"
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mts"}
LOCK = threading.Lock()
PROCS: dict[str, subprocess.Popen] = {}
VIEWER_MAX_POINTS = 60000


def dense_center(xyz, lo, hi):
    # Mean of the fullest voxel: puts the rotation centre in the densest area.
    # `lo`/`hi` bound the robust range so runaway points cannot win the vote.
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    span = np.maximum(hi - lo, 1e-9)
    cell = np.clip(((np.asarray(xyz) - lo) / span * 24.0).astype(int), 0, 23)
    flat = (cell[:, 0] * 24 + cell[:, 1]) * 24 + cell[:, 2]
    valid = ((xyz >= lo) & (xyz <= hi)).all(axis=1)
    if not bool(np.any(valid)):
        return (lo + hi) / 2.0
    counts = np.bincount(flat[valid], minlength=24 ** 3)
    best = int(np.argmax(counts))
    return xyz[valid][flat[valid] == best].mean(axis=0)


def read_ply_points(path: Path) -> tuple:
    """Read an RGB point cloud PLY (ascii or binary_little_endian) as (xyz, rgb, total)."""
    with open(path, "rb") as stream:
        header = []
        while True:
            line = stream.readline().decode("ascii", errors="replace").strip()
            header.append(line)
            if line == "end_header":
                break
            if len(header) > 200:
                raise ValueError("PLY header too long")
    count, props, current = 0, [], None
    for line in header:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "format":
            pass
        elif parts[0] == "element":
            current = parts[1]
            if current == "vertex":
                count, props = int(parts[2]), []
        elif parts[0] == "property" and current == "vertex":
            props.append((parts[1], parts[2]))
    if count <= 0 or not props:
        raise ValueError("PLY has no vertices")
    names = [name for _, name in props]
    if "x" not in names or "y" not in names or "z" not in names:
        raise ValueError("PLY vertices lack x/y/z")
    rgb_names = ("red", "green", "blue") if "red" in names else (("r", "g", "b") if "r" in names else ())
    kind_to_dtype = {"char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2",
                     "int": "i4", "uint": "u4", "float": "f4", "double": "f8"}
    is_ascii = "format ascii 1.0" in header
    if not is_ascii:
        dtype = [(name, kind_to_dtype[kind]) for kind, name in props]
        with open(path, "rb") as stream:
            offset = sum(len((line + "\n").encode("ascii", errors="replace")) for line in header)
            raw = np.fromfile(stream, dtype=dtype, count=count, offset=offset)
        xyz = np.column_stack([raw["x"].astype(np.float64), raw["y"].astype(np.float64),
                               raw["z"].astype(np.float64)])
        rgb = (np.column_stack([np.asarray(raw[n]).astype(np.uint8) for n in rgb_names])
               if rgb_names else np.full_like(xyz, 160, dtype=np.uint8))
    else:
        rows = np.genfromtxt(str(path), dtype=[(n, kind_to_dtype[k]) for k, n in props],
                             skip_header=len(header), max_rows=count)
        xyz = np.column_stack([rows["x"].astype(np.float64), rows["y"].astype(np.float64),
                               rows["z"].astype(np.float64)])
        rgb = (np.column_stack([np.asarray(rows[n]).astype(np.uint8) for n in rgb_names])
               if rgb_names else np.full_like(xyz, 160, dtype=np.uint8))
    return xyz, rgb, count


GUI_LOCK = JOBS.parent / "gui.lock"
_LOCK_OWNER = 0


def _gui_owner_alive(pid: int) -> bool:
    """Whether `pid` is a live process running this desktop app."""
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
    except OSError:
        return False
    return b"gui/app.py" in cmdline


def acquire_gui_lock(path: Path = GUI_LOCK) -> bool:
    """Single-instance guard for the desktop app: True if we now own the lock.

    A crashed previous run leaves a stale file behind; it is reclaimed when its
    PID is gone (or no longer runs this app).
    """
    global _LOCK_OWNER
    path.parent.mkdir(parents=True, exist_ok=True)
    old_pid = 0
    try:
        old_pid = int(json.loads(path.read_text(encoding="utf-8")).get("pid", 0))
    except (OSError, ValueError, AttributeError):
        old_pid = 0
    if old_pid and _gui_owner_alive(old_pid):
        return False
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
    tmp.replace(path)
    _LOCK_OWNER = os.getpid()
    return True


def release_gui_lock(path: Path = GUI_LOCK) -> None:
    """Give up the lock, but only if we own it."""
    global _LOCK_OWNER
    try:
        owned = _LOCK_OWNER == os.getpid() == int(json.loads(path.read_text(encoding="utf-8")).get("pid", 0))
    except (OSError, ValueError, AttributeError):
        owned = False
    if owned:
        _LOCK_OWNER = 0
        path.unlink(missing_ok=True)

def toolkit() -> Path:
    return Path(os.environ.get("MOGE3_TOOLKIT", REPO / "third_party/moge3"))


def engine_status() -> dict[str, dict]:
    tk = toolkit()
    return {
        "moge3": {
            "label": "MoGe-3 版（稀疏＋稠密點雲＋定位包）",
            "ready": (tk / "envs/gluemap/bin/python").exists() and (tk / "weights").is_dir(),
            "hint": "先執行 moge3/scripts/setup.sh",
        },
        "original": {
            "label": "原始版（一次一個影片資料夾）",
            "ready": (REPO / "run.sh").exists() and (REPO / "third_party").is_dir(),
            "hint": "先執行 ./run.sh setup",
        },
    }


_ENGINE_PYTHON: str | None = None
_ENGINE_PYTHON_CHECKED = False


def _probe_engine_python(python: str) -> bool:
    """True if this interpreter can run the mapping engine (torch + pycolmap + gluemap)."""
    try:
        result = subprocess.run(
            [python, "-c", "import torch, pycolmap, gluemap; assert torch.cuda.is_available()"],
            capture_output=True, timeout=90)
        return result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def find_engine_python() -> str | None:
    """Pick an interpreter for engine commands, cached per GUI process.

    Order: $ACAM_PYTHON (explicit choice wins) → MoGe-3 toolkit env →
    well-known conda-style envs → the GUI's own interpreter. Returns None
    when nothing on this machine can run the engine.
    """
    global _ENGINE_PYTHON, _ENGINE_PYTHON_CHECKED
    if _ENGINE_PYTHON_CHECKED:
        return _ENGINE_PYTHON
    home = Path.home()
    candidates = [
        os.environ.get("ACAM_PYTHON") or "",
        str(toolkit() / "envs/gluemap/bin/python"),
        str(home / "micromamba/envs/gluemap/bin/python"),
        str(home / "miniconda3/envs/gluemap/bin/python"),
        str(home / "anaconda3/envs/gluemap/bin/python"),
        sys.executable,
    ]
    _ENGINE_PYTHON = next(
        (c for c in dict.fromkeys(candidates) if c and Path(c).is_file() and _probe_engine_python(c)), None)
    _ENGINE_PYTHON_CHECKED = True
    return _ENGINE_PYTHON


def reset_engine_python_cache() -> None:
    """Forget the cached probe (tests only)."""
    global _ENGINE_PYTHON, _ENGINE_PYTHON_CHECKED
    _ENGINE_PYTHON, _ENGINE_PYTHON_CHECKED = None, False

STAGES = (("screening", "篩選"), ("preprocessing", "前處理"), ("mapping", "建圖"),
          ("densification", "稠密化"), ("report", "報告"), ("packing", "打包"))
_TQDM = re.compile(r"(\d+)%\|[^\r\n]*? (\d+)/(\d+) \[([0-9:.]+)<([0-9:.]+)")
_STAGE_MARK = re.compile(r"\[([0-4])/4\]")
_DENSE_MARK = re.compile(r"===\s*densification", re.IGNORECASE)
_ETA_HINT = re.compile(r"~([0-9.]+ h)")
_REFINE = re.compile(r"Refinement iteration (\d+)/(\d+)")
_BA_ROUND = re.compile(r"=== Iteration (\d+)/(\d+) ===")
_BA_START = re.compile(r"Starting iterative BA with")
_TRIANGULATED = re.compile(r"triangulate_points produced")
_CERES_DONE = re.compile(r"Ceres Solver Report")
_TRACKS_DONE = re.compile(r"Established (\d+) tracks from (\d+) observations")
_CONCAT = re.compile(r"Concatenating tracks")
_CORR = re.compile(r"Constructing correspondences")
_DICT = re.compile(r"Adding points to dictionary")
_PROF = re.compile(r"\[Profiling\] Iteration")
_RETRI = re.compile(r"Retriangulation and Global bundle adjustment")
_TRI_IMG = re.compile(r"Triangulating image #(\d+)")
_MERGED = re.compile(r"Created merged database with (\d+) images")
_TRI_LINE = re.compile(r"^[IW](\d{4})(\d{2})(\d{2}) (\d{2}):(\d{2}):(\d{2})(?:\.\d+)?[^\n]*?Triangulating image #\d+ \((\d+)\)", re.MULTILINE)
_ROUND_TS = re.compile(r"\[(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}),\d+\][^\n]*?=== Iteration (\d+)/(\d+) ===")
_CERES_TS = re.compile(r"\[(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}),\d+\][^\n]*?Ceres Solver Report")
_PROF_BA = re.compile(r"\[Profiling\] Iteration \d+:.*?ba=([\d.]+)s")


def parse_progress(log_text: str) -> dict:
    """Turn the tail of a job log into a progress summary for the GUI.

    Returns stage checklist state, the last tqdm-style counter (if any),
    and the mapping time estimate printed by the engine. Pure function.
    """
    # [N/4] markers map to stages 0,1,2,4,5; a densification banner means stage 3.
    events = [(m.start(), {0: 0, 1: 1, 2: 2, 3: 4, 4: 5}[int(m.group(1))])
              for m in _STAGE_MARK.finditer(log_text)]
    events += [(m.start(), 3) for m in _DENSE_MARK.finditer(log_text)]
    stage = max(events, key=lambda e: e[0])[1] if events else None
    tqdm = _TQDM.findall(log_text.replace("\r", "\n"))
    percent = done = total = elapsed = remaining = None
    if tqdm:
        percent, done, total, elapsed, remaining = tqdm[-1]
        percent, done, total = int(percent), int(done), int(total)
    estimate = None
    hint = _ETA_HINT.search(log_text)
    if hint:
        estimate = hint.group(1)
    return {"stage": stage, "percent": percent, "done": done, "total": total,
            "elapsed": elapsed, "remaining": remaining, "estimate": estimate}

def engine_log_path(job: dict) -> Path | None:
    """Engine-side mapping.log for a map job (None when not created yet)."""
    try:
        cand = Path(job.get("result_dir", "")).expanduser() / "acam" / "logs" / "mapping.log"
    except (TypeError, ValueError):
        return None
    return cand if cand.is_file() else None


def read_engine_log_tail(job: dict, size: int = 200000) -> tuple[str, float | None]:
    """Last `size` bytes of mapping.log plus log age in seconds (None when missing)."""
    path = engine_log_path(job)
    if path is None:
        return "", None
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - size))
            text = stream.read().decode("utf-8", "replace")
        return text, time.time() - path.stat().st_mtime
    except OSError:
        return "", None


def _log_epoch(year: str, month: str, day: str, hour: str, minute: str, second: str) -> float:
    return time.mktime((int(year), int(month), int(day), int(hour), int(minute), int(second), 0, 0, -1))


def parse_mapping_phase(text: str, now: float | None = None) -> dict:
    """Human-readable sub-stage inside [2/4] mapping, with measured remaining time.

    The job log's tqdm bar freezes at 226/226 100% once track concatenation
    finishes, while Ceres BA runs for minutes with no new tqdm line. This
    reads mapping.log so the GUI can say "BA running" instead of looking stuck,
    and derives remaining time from the log itself:
    triangulation rate comes from glog timestamps on the Triangulating lines,
    BA rounds are predicted from the previous completed round (or the last
    [Profiling] ba= seconds). `now` is injectable for tests; pass None to use
    the wall clock. Extra keys (done/total/rate_per_s/elapsed_s/remaining_s)
    are None when the tail does not support them.
    """
    blank = {"phase": None, "in_ba": False, "done": None, "total": None,
             "rate_per_s": None, "elapsed_s": None, "remaining_s": None, "round": None}
    if not text:
        return blank
    if now is None:
        now = time.time()
    refines = _REFINE.findall(text)
    rounds = _BA_ROUND.findall(text)
    order: list[tuple[int, str]] = []
    for rx, name in ((_DICT, "dict"), (_CORR, "corr"), (_CONCAT, "concat"),
                     (_TRACKS_DONE, "tracks"), (_REFINE, "refine"),
                     (_TRIANGULATED, "tri"), (_BA_START, "ba"),
                     (_BA_ROUND, "round"), (_CERES_DONE, "ceres"), (_PROF, "prof"),
                     (_TRI_IMG, "tri_img"), (_RETRI, "retri")):
        found = list(rx.finditer(text))
        if found:
            order.append((found[-1].start(), name))
    last = max(order)[1] if order else ""
    refine = f"{refines[-1][0]}/{refines[-1][1]}" if refines else None
    rnd = f"{rounds[-1][0]}/{rounds[-1][1]}" if rounds else None
    if last in ("ba", "round", "ceres") and refine:
        phase = f"全域優化 {refine} · BA 第 {rnd} 輪運算中" if rnd else f"全域優化 {refine} · BA 運算中"
        out = dict(blank, phase=phase, in_ba=True, round=rnd)
        starts = [(m.start(), _log_epoch(*m.groups()[:6])) for m in _ROUND_TS.finditer(text)]
        dones = [(m.start(), _log_epoch(*m.groups()[:6])) for m in _CERES_TS.finditer(text)]
        if starts:
            out["elapsed_s"] = max(0.0, now - starts[-1][1])
            prev: float | None = None
            for pos, t0 in reversed(starts[:-1] if len(starts) > 1 else starts):
                nxt = next((t1 for p1, t1 in dones if p1 > pos), None)
                if nxt is not None and nxt > t0:
                    prev = nxt - t0
                    break
            if prev is None:
                prof = _PROF_BA.findall(text)
                prev = float(prof[-1]) if prof else None
            if prev is not None:
                out["remaining_s"] = max(0.0, prev - out["elapsed_s"])
        return out
    if last in ("retri", "tri_img"):
        total = int(_MERGED.findall(text)[-1]) if _MERGED.findall(text) else None
        samples = [(_log_epoch(*m.groups()[:6]), int(m.groups()[6])) for m in _TRI_LINE.finditer(text)]
        out = dict(blank, in_ba=True)
        if samples:
            out["done"] = samples[-1][1] + 1
            out["total"] = total
            if len(samples) > 1 and samples[-1][0] > samples[0][0]:
                out["rate_per_s"] = (samples[-1][1] - samples[0][1]) / (samples[-1][0] - samples[0][0])
            if total and out["rate_per_s"]:
                out["remaining_s"] = max(0.0, (total - out["done"]) / out["rate_per_s"])
        where = f"（看到第 {out['done']}／{total} 張）" if out["done"] and total else ""
        if not where:
            imgs = _TRI_IMG.findall(text)
            where = f"（看到第 {imgs[-1]} 張）" if imgs else ""
        base = f"全域優化 {refine} · 重三角測量＋全域平差" if refine else "重三角測量＋全域平差中"
        return dict(out, phase=base + where)
    if last == "tri":
        return dict(blank, phase=f"全域優化 {refine} · 三角測量" if refine else "三角測量中")
    if last == "refine":
        return dict(blank, phase=f"全域優化 {refine} 開始" if refine else "全域優化中")
    if last == "prof":
        return dict(blank, phase=f"全域優化 {refine} 本輪完成" if refine else "本輪優化完成")
    if last == "tracks":
        m = _TRACKS_DONE.findall(text)[-1]
        return dict(blank, phase=f"軌跡建立完成（{m[0]} 條／{m[1]} 觀測），進全域優化")
    if last == "concat":
        return dict(blank, phase="串接軌跡中（這條 100% 之後還有 BA，別被騙）")
    if last == "corr":
        return dict(blank, phase="建立對應中")
    if last == "dict":
        return dict(blank, phase="建立點字典中")
    return blank


def read_log_tail(job_id: str, size: int = 50000) -> str:
    """Last `size` bytes of a job log (cheap to call every second)."""
    path = job_dir(job_id) / "log.txt"
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - size))
            return stream.read().decode("utf-8", "replace")
    except OSError:
        return ""

def loc_python() -> str:
    return os.environ.get("LOC_PYTHON") or str(toolkit() / "envs/gluemap/bin/python")


def safe_name(text: str, default: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in text).strip("._-") or default


# ------------------------------------------------------------------------------------------------ job files
def job_dir(job_id: str) -> Path:
    if not job_id.isalnum():
        raise ValueError("bad job id")
    return JOBS / job_id


def read_job(job_id: str) -> dict:
    return json.loads((job_dir(job_id) / "job.json").read_text(encoding="utf-8"))


def write_job(job: dict) -> None:
    path = job_dir(job["id"]) / "job.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def list_jobs() -> list[dict]:
    jobs = []
    for path in sorted(JOBS.glob("*/job.json"), reverse=True) if JOBS.exists() else []:
        try:
            jobs.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return jobs


def mark_interrupted() -> None:
    """Jobs that were running when the app last closed cannot be resumed from here."""
    for job in list_jobs():
        if job.get("status") in ("running", "queued"):
            job["status"] = "interrupted"
            write_job(job)


def read_log(job_id: str, offset: int) -> tuple[str, int]:
    path = job_dir(job_id) / "log.txt"
    if not path.exists():
        return "", offset
    data = path.read_bytes()
    return data[offset:].decode("utf-8", "replace"), len(data)


# ------------------------------------------------------------------------------------------------ commands
def stage_selection(videos: list[Path], dest: Path) -> Path:
    """Hard-link the user's hand-picked videos into one folder for the original engine.

    The original entry point only accepts a single video folder, so free
    file/folder picking is staged here (names preserved, collisions suffixed
    with _2, _3, ...). Hard links, never symlinks: the engine refuses those.
    """
    dest.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    for video in videos:
        base = safe_name(video.stem, "video")
        cand = f"{base}{video.suffix}"
        index = 2
        while cand in used:
            cand = f"{base}_{index}{video.suffix}"
            index += 1
        used.add(cand)
        target = dest / cand
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            os.link(video, target)
        except OSError:
            shutil.copy2(video, target)  # cross-device fallback
    return dest

def expand_videos(entries: list[str]) -> list[Path]:
    found: list[Path] = []
    for entry in entries:
        path = Path(entry).expanduser()
        if not path.exists():
            raise ValueError(f"路徑不存在: {entry}")
        if path.is_dir():
            found += sorted(p for p in path.iterdir() if p.suffix.lower() in VIDEO_EXT and p.is_file())
        else:
            found.append(path)
    if not found:
        raise ValueError("沒有找到影片")
    unique: dict[Path, Path] = {}
    for video in found:
        unique.setdefault(video.resolve(), video)
    return list(unique.values())


def map_output_status(out_raw: str, name_raw: str) -> str | None:
    """Why out/name cannot start a fresh map (None when free to use).

    result_dir is out/name with no uniqueness, and the engine resumes from
    existing caches. A same-name rerun with different videos would silently
    reuse the old map, so a fresh task must pick another name (or the old
    folder must be deleted first). Partial failed runs without a report are
    still allowed so the engine can resume them.
    """
    base = Path(out_raw or "").expanduser() / safe_name(name_raw or "site", "site")
    if not base.is_dir():
        return None
    try:
        data = json.loads((base / "map_report.json").read_text(encoding="utf-8"))
        if data.get("images_total"):
            link = "各影片相連" if data.get("connected") else "多段不相連"
            return (f"「{base.name}」已有完成的建圖（{data.get('images_registered', '?')}/"
                    f"{data.get('images_total', '?')} 張，{link}）：換個任務名稱，或刪掉 {base} 再跑")
    except (OSError, ValueError):
        pass
    for marker in ("MAP_REPORT.md", "map/sparse_rgb.ply", "pipeline_receipt.json"):
        if (base / marker).exists():
            return f"「{base.name}」已有建圖輸出（{marker}）：換個任務名稱，或刪掉 {base} 再跑"
    return None


def camera_profiles_dir() -> Path:
    home = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    return home / "a_click_a_map" / "cameras"


def import_camera_profile(source: Path) -> Path:
    """Save a local copy without changing any profile already used by another job."""
    profile = load_calibration(source)
    directory = camera_profiles_dir()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{safe_name(source.stem, 'camera')}.json"
    try:
        with target.open("x", encoding="utf-8") as stream:
            json.dump(profile, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except FileExistsError:
        if load_calibration(target) != profile:
            raise ValueError("已有同名但內容不同的相機設定，請先重新命名來源 JSON")
    return target


def _camera_choice(request: dict) -> Path | None:
    """Return the selected file, or None only for an explicit MoGe-3 opt-in."""
    engine = request.get("engine")
    raw = (request.get("intrinsics") or "").strip()
    if request.get("auto_calibrate") is True:
        if engine != "moge3" or raw:
            raise ValueError("只有 MoGe-3 可單獨選擇自動估算，不可同時指定標定檔")
        path = None
    elif raw:
        path = Path(raw).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"找不到標定檔：{path}")
    else:
        message = "請在「相機設定」選擇或匯入標定；原始版不會自動標定" if engine == "original" else "請選擇相機設定或明確選擇自動估算"
        raise ValueError(message)
    if not request.get("_resume") and request.get("camera_confirmed") is not True:
        raise ValueError("請確認同批影片使用同一相機、鏡頭與錄影模式；否則請分批建圖")
    return path


def uses_approximate_calibration(job: dict) -> bool:
    base = Path(job["result_dir"])
    for receipt in (base / "map/map_receipt.json", base / "pipeline_receipt.json"):
        try:
            data = json.loads(receipt.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict) and isinstance(source := data.get("calibration_source"), str):
            return source.startswith("APPROXIMATE_")
    return False


def validate_map_request(request: dict) -> tuple[bool, str]:
    """Lightweight pre-flight check for the 建圖 form: (ready, reason).

    Cheap (a few stat calls, no decoding); the GUI calls this on every tick to
    enable/disable the start button. `build_map_command` enforces the same rules.
    """
    engine = request.get("engine")
    entries = [e.strip() for e in request.get("videos", []) if e.strip()]
    if not entries:
        return False, "還沒加入影片：先按「加入資料夾」或「加入影片檔」"
    if not (request.get("name") or "").strip():
        return False, "任務名稱不能空白"
    if not (request.get("out") or "").strip():
        return False, "輸出位置不能空白"
    if engine not in ("original", "moge3"):
        return False, "未知的建圖方式"
    taken = map_output_status(request.get("out"), request.get("name"))
    if taken:
        return False, taken
    try:
        count = len(expand_videos(entries))
        path = _camera_choice(request)
    except (OSError, ValueError) as error:
        return False, str(error)
    if engine == "original" and request.get("dense") and mvs_weights() is None:
        return False, "缺 MVSFormer++ 權重：請先安裝權重，或取消「產生稠密點雲」"
    camera = "自動估算（未驗證）" if path is None else request.get("camera_profile") or path.stem
    extra = "＋稠密點雲" if request.get("dense") else ""
    return True, f"就緒：共 {count} 部影片{extra} · {camera}"


def mvs_weights() -> tuple[Path, Path] | None:
    """MVSFormer++ checkpoint + config if the user supplied both files."""
    base = REPO / "third_party/mvsformerpp"
    ckpt = Path(os.environ.get("ACAM_MVS_CHECKPOINT", base / "weights/model_best.pth")).expanduser()
    cfg = Path(os.environ.get("ACAM_MVS_CONFIG", base / "weights/config.json")).expanduser()
    if ckpt.is_file() and cfg.is_file():
        return ckpt, cfg
    return None


APPROVAL_FILE = ".acam_map_approved.json"


def map_pointcloud(job: dict) -> Path | None:
    """The inspectable RGB point cloud of a finished map job, if it exists."""
    base = Path(job.get("result_dir", ""))
    if job["engine"] == "moge3":
        cand = base / "map/sparse_rgb.ply"
    else:
        cand = base / "run/products/map/original_rgb.ply"
    return cand if cand.is_file() else None


def is_map_approved(map_dir: str | Path) -> tuple[bool, dict]:
    """Whether this map folder was confirmed good in the 3D inspection view."""
    path = Path(map_dir).expanduser() / APPROVAL_FILE
    try:
        return True, json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False, {}


def approve_map(map_dir: str | Path, info: dict | None = None) -> dict:
    """Record the operator's map inspection approval inside the map folder."""
    marker = {"approved_at": time.time(), **(info or {})}
    path = Path(map_dir).expanduser() / APPROVAL_FILE
    path.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
    return marker

def validate_cruise_request(request: dict) -> tuple[bool, str]:
    """Lightweight pre-flight check for the 巡航定位 form: (ready, reason)."""
    engine = request.get("engine")
    if engine not in ("original", "moge3"):
        return False, "未知的建圖方式"
    if not request.get("video", "").strip():
        return False, "還沒選擇巡航影片"
    if not Path(request.get("video", "")).expanduser().is_file():
        return False, f"找不到巡航影片：{request.get('video', '')}"
    if not request.get("map", "").strip():
        return False, "還沒選擇已建好的地圖"
    if not Path(request.get("map") or "").expanduser().is_dir():
        return False, f"找不到地圖資料夾：{request.get('map', '')}"
    approved, _ = is_map_approved(Path(request.get("map") or "").expanduser())
    if not approved:
        return False, "這張地圖還沒檢查確認：先在建圖結果打開 3D 檢視，確認沒問題後按「✓ 確認地圖可用」"
    return True, "就緒：按「開始巡航定位」"

def build_map_command(request: dict, job_id: str) -> tuple[list[str], dict]:
    engine = request["engine"]
    entries = [e.strip() for e in request.get("videos", []) if e.strip()]
    if not entries:
        raise ValueError("請加入影片")
    name = safe_name(request.get("name") or "site", "site")
    out = Path(request.get("out") or REPO / "acam_out").expanduser().resolve()
    if not request.get("_resume"):
        taken = map_output_status(str(out), name)
        if taken:
            raise ValueError(taken)
    if engine not in ("original", "moge3"):
        raise ValueError("未知引擎")
    videos = expand_videos(entries)
    path = _camera_choice(request)
    meta = {"name": name, "result_dir": str(out / name)}
    if path is not None:
        try:
            payload = load_calibration(path)
        except (OSError, ValueError) as error:
            raise ValueError(f"標定檔無效：{error}") from error
        snapshot = job_dir(job_id) / "intrinsics.json"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        meta["calibration_file"] = str(snapshot.resolve())
        calibration_args = ["--intrinsics", meta["calibration_file"]]
    else:
        calibration_args = ["--auto_calibrate"]
    if engine == "original":
        try:
            staged = stage_selection(videos, job_dir(job_id) / "selection")
        except OSError as error:
            raise ValueError(f"準備影片失敗（硬碟空間或權限）：{error}")
        cmd = [str(REPO / "run.sh"), "map", str(staged.resolve()), "--out", str(out), "--name", name,
               "--no-pack"]
        if request.get("dense"):
            cmd += ["--dense"]
    else:
        cmd = [str(REPO / "run_moge3.sh"), *[str(v.resolve()) for v in videos], "--output", meta["result_dir"]]
        if not request.get("dense", True):
            cmd.append("--skip-dense")
    return [*cmd, *calibration_args], meta


def map_bundle(map_dir: str | Path) -> Path | None:
    """The packed transfer bundle of a map folder, if the user chose to pack it."""
    bundles = sorted(Path(map_dir).expanduser().glob("*_bundle.tar.zst"))
    return bundles[-1] if bundles else None


def build_pack_command(map_dir: str | Path, job_id: str) -> tuple[list[str], dict]:
    """Pack a finished map run into its transfer bundle (runs after user approval)."""
    base = Path(map_dir).expanduser().resolve()
    run = base / "run"
    if not run.is_dir():
        raise ValueError("找不到這張地圖的 run 資料夾，無法打包")
    name = f"{base.name}_bundle"
    cmd = [str(REPO / "run.sh"), "pack", str(run), "--out", str(base), "--name", name]
    return cmd, {"name": f"{base.name} 打包", "result_dir": str(base)}


def delivery_site(map_dir: str | Path) -> Path:
    base = Path(map_dir).expanduser().resolve()
    site_id = re.sub(r"[^A-Za-z0-9._-]+", "_", base.name)[:64].strip("._-") or "site"
    return base / "場域交付" / site_id


def delivery_archive(map_dir: str | Path) -> Path:
    site = delivery_site(map_dir)
    return site.with_name(f"{site.name}.zip")


def _build_export_command(map_dir: str | Path) -> tuple[list[str], dict]:
    base = Path(map_dir).expanduser().resolve()
    run = base / "run"
    if not run.is_dir() or not is_map_approved(base)[0]:
        raise ValueError("找不到已確認的建圖成果，請先完成 3D 檢查")
    archive = delivery_archive(base)
    if archive.exists():
        raise ValueError(f"交付壓縮包已存在：{archive}")
    cmd = [str(REPO / "run.sh"), "export-site", str(run), "--out", str(archive.parent),
           "--site-id", archive.stem, "--display-name", base.name, "--zip"]
    return cmd, {"name": base.name, "result_dir": str(archive)}


def _dir_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file() and not path.is_symlink())


def _archive_row(path: Path) -> dict:
    return {"label": "搬到另一台電腦的場域 ZIP", "path": str(path), "size": human(path.stat().st_size)}


def export_result(job: dict) -> list[dict]:
    if job["status"] != "done":
        return []
    site = Path(job["result_dir"])
    if site.is_file() and site.suffix == ".zip":
        return [_archive_row(site)]
    profile = site / "site_profile.json"
    folders = [site / name for name in ("map", "localization", "model", "keyframes")]
    if not profile.is_file() or not all(folder.is_dir() for folder in folders):
        return []
    paths = [profile]
    for folder in folders:
        paths.extend(sorted(folder.iterdir()))
    rows = []
    total = 0
    for path in paths:
        size = _dir_size(path) if path.is_dir() else path.stat().st_size
        total += size
        rows.append({"label": path.relative_to(site).as_posix(), "path": str(path), "size": human(size)})
    return [{"label": "搬這整個場域資料夾", "path": str(site), "size": human(total)}, *rows]


def pack_result(job: dict) -> list[dict]:
    bundle = map_bundle(job.get("result_dir", ""))
    if bundle is None:
        return []
    return [{"label": "建圖轉接包（非場域匯入）", "path": str(bundle), "size": human(bundle.stat().st_size)}]


def build_cruise_command(request: dict, job_id: str) -> tuple[list[str] | str, dict]:
    engine = request["engine"]
    video = Path(request.get("video", "")).expanduser()
    if not video.is_file():
        raise ValueError("找不到巡航影片")
    map_dir = Path(request.get("map") or "").expanduser()
    if not map_dir.is_dir():
        raise ValueError("找不到地圖資料夾")
    approved, _ = is_map_approved(map_dir)
    if not approved:
        raise ValueError("這張地圖還沒檢查確認：先在建圖結果打開 3D 檢視，確認沒問題後按「✓ 確認地圖可用」")
    route = safe_name(request.get("route") or video.stem, "route")
    out = job_dir(job_id) / "cruise"
    meta = {"name": route, "result_dir": str(out)}
    if engine == "original":
        run_dir = map_dir / "run" if (map_dir / "run").is_dir() else map_dir
        return [str(REPO / "run.sh"), "localize", str(run_dir.resolve()), str(video.resolve()), "--out", str(out)], meta
    bundle = map_dir / "localization/bundle" if (map_dir / "localization/bundle").is_dir() else map_dir
    if not (bundle / "run_localize.py").is_file():
        raise ValueError("moge3 地圖資料夾需含 localization/bundle（或直接選 bundle）")
    py = shlex.quote(loc_python())
    script = (
        f"cd {shlex.quote(str(bundle.resolve()))} && "
        f"{py} install.py && "
        f"{py} run_localize.py --video {shlex.quote(str(video.resolve()))} --route {shlex.quote(route)} --output {shlex.quote(str(out))}"
    )
    return script, meta

def resume_request(job: dict) -> dict:
    """Rebuild the submit request from a finished job so it reruns with the same settings.

    Map reruns reuse the same out/name, so the engine resumes from finished
    stage caches instead of redoing them. Cruise reruns are quick anyway.
    """
    request = dict(job.get("request", {}))
    request.update(kind=job.get("kind"), engine=job.get("engine"))
    request["_resume"] = True  # same out/name on purpose: continue from stage caches
    if request.get("kind") == "map":
        if snapshot := job.get("calibration_file"):
            request["intrinsics"] = snapshot
        elif "auto_calibrate" not in request:
            # Preserve the original settings of records made before explicit camera choice.
            if job.get("engine") == "original":
                stored = Path(job.get("result_dir") or "") / "acam/intrinsics.json"
                if job.get("result_dir") and stored.is_file():
                    request["intrinsics"] = str(stored)
                elif not request.get("intrinsics"):
                    request["intrinsics"] = str(BUILTIN_CAMERA_PROFILE)
            elif not request.get("intrinsics"):
                request["auto_calibrate"] = True
    return request

def resume_plan(job: dict) -> list[tuple[str, str]]:
    """What a 續跑 will skip vs redo, from the artifacts on disk.

    Returns [(stage, action)] with action in ("skip", "reuse", "redo").
    Mirrors the engine's own resume checks (receipts, validated caches,
    finished model) so the operator knows before pressing resume how much
    compute is saved. Post-star matching/BA have no checkpoint upstream and
    always redo — that is stated, not hidden.
    """
    if job.get("kind") != "map":
        return [("巡航定位", "redo")]
    base = Path(job.get("result_dir", "")).expanduser()
    plan: list[tuple[str, str]] = []
    plan.append(("前處理 keyframes", "skip" if (base / "run/receipts/direct_preprocessing.json").is_file() else "redo"))
    glues = sorted((base / "workspace").glob("work_*/gluemap")) if (base / "workspace").is_dir() else []
    glue = glues[-1] if glues else None
    twoview = glue is not None and (glue / "twoview_result.pth").is_file() and (glue / "twoview_result.pth").stat().st_size > 0
    star = glue is not None and (glue / "star_result.pth").is_file() and (glue / "star_result.pth").stat().st_size > 0
    sift = glue is not None and (glue / "database_sift.db").is_file() and (glue / "database_sift.db").stat().st_size > 0
    plan.append(("雙視角幾何快取", "reuse" if twoview else "redo"))
    plan.append(("特徵/配對快取", "reuse" if star else "redo"))
    plan.append(("SIFT 資料庫", "reuse" if sift else "redo"))
    model = base / "run/artifacts/mapping/edm/model"
    done_model = model.is_dir() and (model / "images.bin").is_file() and (model / "points3D.bin").is_file()
    plan.append(("GlueMap 匹配＋BA", "skip" if done_model else "redo"))
    dense = (base / "run/products/map/dense/dense_receipt.json").is_file() and (base / "run/products/map/dense/dense.ply").is_file()
    plan.append(("稠密點雲", "skip" if dense else "redo"))
    return plan


# ------------------------------------------------------------------------------------------------ running
def _run(job: dict, cmd: list[str] | str) -> None:
    directory = job_dir(job["id"])
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONUNBUFFERED"] = "1"
    if job.get("engine_python"):
        env["ACAM_PYTHON"] = job["engine_python"]
    log = (directory / "log.txt").open("ab", buffering=0)
    log.write(("$ " + (cmd if isinstance(cmd, str) else " ".join(shlex.quote(c) for c in cmd)) + "\n").encode())
    try:
        proc = subprocess.Popen(cmd, shell=isinstance(cmd, str), executable="/bin/bash" if isinstance(cmd, str) else None,
                                stdout=log, stderr=subprocess.STDOUT, env=env, cwd=REPO, start_new_session=True)
    except OSError as error:
        job.update(status="failed", error=str(error), finished=time.time())
        write_job(job)
        return
    with LOCK:
        PROCS[job["id"]] = proc
    job.update(status="running", pid=proc.pid)
    write_job(job)
    code = proc.wait()
    with LOCK:
        PROCS.pop(job["id"], None)
    job = read_job(job["id"])
    if job.get("status") != "stopped":
        job["status"] = "done" if code == 0 else "failed"
    job.update(exit_code=code, finished=time.time())
    write_job(job)


def create_job(request: dict) -> dict:
    kind, engine = request.get("kind"), request.get("engine")
    if kind not in ("map", "cruise", "pack", "export") or engine not in ("original", "moge3"):
        raise ValueError("任務類型或引擎錯誤")
    if kind in ("pack", "export") and engine != "original":
        raise ValueError("此交付格式只支援原始版")
    status = engine_status()[engine]
    if not status["ready"]:
        raise ValueError("此引擎尚未安裝：" + status["hint"])
    job_id = time.strftime("%Y%m%d%H%M%S") + uuid.uuid4().hex[:6]
    job_dir(job_id).mkdir(parents=True)
    engine_python = find_engine_python()
    if not engine_python:
        job_dir(job_id).rmdir()
        raise ValueError("找不到可跑引擎的 Python（需 torch＋CUDA、pycolmap、pygluemap）：先跑 setup，或設 ACAM_PYTHON 再重開程式")
    try:
        if kind == "map":
            cmd, meta = build_map_command(request, job_id)
        elif kind == "cruise":
            cmd, meta = build_cruise_command(request, job_id)
        elif kind == "export":
            cmd, meta = _build_export_command(request["map"])
        else:
            cmd, meta = build_pack_command(request["map"], job_id)
    except (OSError, ValueError):
        shutil.rmtree(job_dir(job_id))
        raise
    job = {"id": job_id, "kind": kind, "engine": engine, "status": "queued", "created": time.time(), **meta,
           "engine_python": engine_python,
           "request": {k: v for k, v in request.items() if k in (
               "videos", "out", "name", "intrinsics", "video", "map", "route", "dense",
               "camera_profile", "camera_confirmed", "auto_calibrate")}}
    write_job(job)
    threading.Thread(target=_run, args=(job, cmd), daemon=True).start()
    return job


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _descendants(pid: int) -> list[int]:
    """All live PIDs in `pid`'s subtree, via /proc (children that escaped the group too)."""
    try:
        children: dict[int, list[int]] = {}
        alive: set[int] = set()
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                parts = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                child, ppid = int(entry.name), int(parts[1])
            except (OSError, ValueError, IndexError):
                continue
            alive.add(child)
            children.setdefault(ppid, []).append(child)
    except OSError:
        return []
    found: list[int] = []
    stack = [pid]
    while stack:
        current = stack.pop()
        for child in children.get(current, []):
            if child in alive:
                found.append(child)
                stack.append(child)
    return found


def _proc_state(pid: int) -> str | None:
    """Process state from /proc, or None if reaped already (Z counts as dead)."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[2]
    except (OSError, IndexError):
        return None


def _kill_tree(pid: int, timeout: float = 10.0) -> None:
    """SIGTERM the group, then every known descendant by PID; SIGKILL stragglers.

    The descendant list is snapshotted *before* killing: once the parent dies,
    orphans are reparented away and a ppid walk can no longer find them.
    """
    me = os.getpid()
    targets = [pid] + [p for p in _descendants(pid) if p != me]
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        pass
    for p in targets[1:]:
        try:
            os.kill(p, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.time() + timeout
    while time.time() < deadline:
        if all(_proc_state(p) in (None, "Z") for p in targets if p != me):
            return
        time.sleep(0.5)
    for p in targets:
        if p == me:
            continue
        try:
            os.kill(-p if p == pid else p, signal.SIGKILL)
        except OSError:
            pass


def stop_job(job_id: str) -> None:
    with LOCK:
        proc = PROCS.get(job_id)
    if proc is None or proc.poll() is not None:
        return
    job = read_job(job_id)
    job["status"] = "stopped"
    write_job(job)
    _kill_tree(proc.pid)


def delete_job(job_id: str) -> None:
    """Delete a finished job's record, logs and staged files (not the map output).

    Only for jobs that are done/failed/stopped/interrupted: running jobs must
    be stopped first. The built map under the output dir is kept; remove it
    with a file manager if the disk space is needed.
    """
    job = read_job(job_id)
    if job.get("status") not in ("done", "failed", "stopped", "interrupted"):
        raise ValueError("執行中的任務請先按「停止」，不能直接刪除")
    shutil.rmtree(job_dir(job_id))


def stop_all() -> int:
    with LOCK:
        ids = list(PROCS)
    for job_id in ids:
        stop_job(job_id)
    return len(ids)


def running_count() -> int:
    with LOCK:
        return sum(1 for p in PROCS.values() if p.poll() is None)


# ------------------------------------------------------------------------------------------------ results
def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} B" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return ""


def map_result(job: dict) -> list[dict]:
    base = Path(job.get("result_dir", ""))
    if job["engine"] == "moge3":
        wanted = [("稀疏點雲 (原始 RGB)", "map/sparse_rgb.ply"), ("稠密點雲 (MVSFormer++)", "dense/dense.ply"),
                  ("COLMAP 地圖", "map/model"), ("獨立定位包（非場域匯入）", "localization/bundle"),
                  ("流程收據", "pipeline_receipt.json")]
    else:
        wanted = [("定位用地圖", "run/artifacts/mapping/edm/model"), ("建圖報告", "MAP_REPORT.md"),
                  ("建圖轉接包（非場域匯入）", f"{job.get('name', '')}_bundle.tar.zst"),
                  ("RGB 點雲", "run/products/map/original_rgb.ply"),
                  ("稠密點雲 (MVSFormer++)", "run/products/map/dense/dense.ply")]
    rows = []
    if job["engine"] == "original":
        archive = delivery_archive(base)
        site = delivery_site(base)
        if archive.is_file():
            rows.append(_archive_row(archive))
        elif (site / "site_profile.json").is_file():
            rows.append({"label": "搬這整個場域資料夾", "path": str(site), "size": ""})
    for label, rel in wanted:
        path = base / rel
        if path.exists():
            size = path.stat().st_size if path.is_file() else sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
            rows.append({"label": label, "path": str(path), "size": human(size)})
    return rows

def map_connectivity(result_dir: str | Path) -> dict | None:
    """Cross-video connectivity from map_report.json (None when not written yet).

    A map can register 226/226 images yet split into disconnected components
    (each video its own coordinate frame). The GUI must surface this before
    the operator approves the map for cruise, otherwise cross-video routes
    silently break. Pure function of the report file.
    """
    try:
        report = json.loads((Path(result_dir).expanduser() / "map_report.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    videos = report.get("videos", {})
    if not videos:
        return None
    return {"connected": bool(report.get("connected", True)),
            "images_registered": report.get("images_registered"),
            "images_total": report.get("images_total"),
            "components": report.get("video_components", []),
            "videos": {vid: {"source": v.get("source", vid), "keyframes": v.get("keyframes"),
                            "registered": v.get("registered")} for vid, v in videos.items()}}


def cruise_result(job: dict) -> dict | None:
    out = Path(job.get("result_dir", ""))
    summaries = sorted(out.rglob("summary.json"), key=lambda p: p.stat().st_mtime)
    if not summaries:
        return None
    folder = summaries[-1].parent
    summary = json.loads(summaries[-1].read_text(encoding="utf-8"))
    points = []
    frames = folder / "frames.jsonl"
    if frames.exists():
        rows = [json.loads(line) for line in frames.read_text(encoding="utf-8").splitlines() if line.strip()]
        step = max(1, len(rows) // 2000)
        points = [(r["position"], r["status"]) for r in rows[::step] if r.get("position")]
    return {"frames": summary.get("dense_frames"), "coverage": summary.get("pose_coverage"),
            "status_counts": summary.get("status_counts", {}), "points": points, "dir": str(folder)}
