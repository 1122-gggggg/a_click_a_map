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
JOBS = REPO / "acam_gui" / "jobs"
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mts"}
LOCK = threading.Lock()
PROCS: dict[str, subprocess.Popen] = {}
VIEWER_MAX_POINTS = 60000


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
          ("report", "報告"), ("packing", "打包"))
_TQDM = re.compile(r"(\d+)%\|[^\r\n]*? (\d+)/(\d+) \[([0-9:.]+)<([0-9:.]+)")
_STAGE_MARK = re.compile(r"\[([0-4])/4\]")
_ETA_HINT = re.compile(r"~([0-9.]+ h)")


def parse_progress(log_text: str) -> dict:
    """Turn the tail of a job log into a progress summary for the GUI.

    Returns stage checklist state, the last tqdm-style counter (if any),
    and the mapping time estimate printed by the engine. Pure function.
    """
    marks = [int(m) for m in _STAGE_MARK.findall(log_text)]
    stage = max(marks) if marks else None
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
    intrinsics = (request.get("intrinsics") or "").strip()
    if intrinsics and not Path(intrinsics).expanduser().is_file():
        return False, f"找不到標定檔：{intrinsics}"
    if engine == "original":
        try:
            count = sum(1 for _ in expand_videos(entries))
        except ValueError as error:
            return False, str(error)
        return True, f"就緒：共 {count} 部影片，按「開始建圖」（同批影片需同長寬比）"
    if engine == "moge3":
        try:
            count = sum(1 for _ in expand_videos(entries))
        except ValueError as error:
            return False, str(error)
        return True, f"就緒：共 {count} 部影片，按「開始建圖」"
    return False, "未知的建圖方式"


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
    intrinsics = (request.get("intrinsics") or "").strip()
    if intrinsics and not Path(intrinsics).expanduser().is_file():
        raise ValueError(f"找不到標定檔: {intrinsics}")
    if engine == "original":
        videos = expand_videos(entries)
        try:
            staged = stage_selection(videos, job_dir(job_id) / "selection")
        except OSError as error:
            raise ValueError(f"準備影片失敗（硬碟空間或權限）：{error}")
        cmd = [str(REPO / "run.sh"), "map", str(staged.resolve()), "--out", str(out), "--name", name,
               "--no-pack"]
        if intrinsics:
            cmd += ["--intrinsics", intrinsics]
        return cmd, {"name": name, "result_dir": str(out / name)}
    if engine == "moge3":
        videos = expand_videos(entries)
        target = out / name
        cmd = [str(REPO / "run_moge3.sh"), *[str(v.resolve()) for v in videos], "--output", str(target)]
        cmd += ["--intrinsics", intrinsics] if intrinsics else ["--auto_calibrate"]
        if request.get("skip_dense"):
            cmd.append("--skip-dense")
        return cmd, {"name": name, "result_dir": str(target)}
    raise ValueError("未知引擎")


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


def pack_result(job: dict) -> list[dict]:
    bundle = map_bundle(job.get("result_dir", ""))
    if bundle is None:
        return []
    return [{"label": "轉接包（搬到另一台電腦）", "path": str(bundle), "size": human(bundle.stat().st_size)}]


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
    return request


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
    if kind not in ("map", "cruise", "pack") or engine not in ("original", "moge3"):
        raise ValueError("任務類型或引擎錯誤")
    if kind == "pack" and engine != "original":
        raise ValueError("只有原始版需要單獨打包")
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
        else:
            cmd, meta = build_pack_command(request["map"], job_id)
    except ValueError:
        job_dir(job_id).rmdir()
        raise
    job = {"id": job_id, "kind": kind, "engine": engine, "status": "queued", "created": time.time(), **meta,
           "engine_python": engine_python,
           "request": {k: v for k, v in request.items() if k in ("videos", "out", "name", "intrinsics", "video", "map", "route", "skip_dense")}}
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


def _kill_tree(pid: int, timeout: float = 10.0) -> None:
    """SIGTERM the group, then any escaped descendants; SIGKILL stragglers."""
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        pass
    deadline = time.time() + timeout
    while time.time() < deadline:
        rest = [p for p in _descendants(pid) if p != os.getpid() and _process_alive(p)]
        if not _process_alive(pid) and not rest:
            return
        for p in rest:
            try:
                os.kill(p, signal.SIGTERM)
            except OSError:
                pass
        time.sleep(0.5)
    for p in [pid, *_descendants(pid)]:
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
                  ("COLMAP 地圖", "map/model"), ("定位包 (搬到另一台電腦)", "localization/bundle"),
                  ("流程收據", "pipeline_receipt.json")]
    else:
        wanted = [("定位用地圖", "run/artifacts/mapping/edm/model"), ("建圖報告", "MAP_REPORT.md"),
                  ("轉接包", f"{job.get('name', '')}_bundle.tar.zst"), ("RGB 點雲", "run/products/map/original_rgb.ply")]
    rows = []
    for label, rel in wanted:
        path = base / rel
        if path.exists():
            size = path.stat().st_size if path.is_file() else sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
            rows.append({"label": label, "path": str(path), "size": human(size)})
    return rows


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
