"""Job backend of the desktop UI: build engine commands, run them as subprocesses, read results.

No UI code and no network code here. Jobs live in acam_gui/jobs/<id>/ (job.json + log.txt).
Engines:
    original  ./run.sh        (folder of videos in; run.sh map / run.sh localize)
    moge3     ./run_moge3.sh  (videos in; sparse + MVSFormer++ dense + EDM/MegaLoc bundle)
"""
from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
JOBS = REPO / "acam_gui" / "jobs"
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mts"}
LOCK = threading.Lock()
PROCS: dict[str, subprocess.Popen] = {}


def toolkit() -> Path:
    return Path(os.environ.get("MOGE3_TOOLKIT", REPO / "third_party/moge3"))


def engine_status() -> dict[str, dict]:
    tk = toolkit()
    return {
        "moge3": {
            "label": "MoGe-3 版（稀疏點雲 + MVSFormer++ 稠密 + 定位包）",
            "ready": (tk / "envs/gluemap/bin/python").exists() and (tk / "weights").is_dir(),
            "hint": "先執行 moge3/scripts/setup.sh",
        },
        "original": {
            "label": "原始版（EDM + KLT，輸入影片資料夾）",
            "ready": (REPO / "run.sh").exists() and (REPO / "third_party").is_dir(),
            "hint": "先執行 ./run.sh setup",
        },
    }


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
        if len(entries) != 1 or not Path(entries[0]).expanduser().is_dir():
            return False, "原始版一次只吃「一個影片資料夾」：請只留一個資料夾"
        return True, "就緒：按「開始建圖」"
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

def build_map_command(request: dict) -> tuple[list[str], dict]:
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
        if len(entries) != 1 or not Path(entries[0]).expanduser().is_dir():
            raise ValueError("原始版一次只吃一個影片資料夾")
        cmd = [str(REPO / "run.sh"), "map", str(Path(entries[0]).expanduser().resolve()), "--out", str(out), "--name", name]
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


# ------------------------------------------------------------------------------------------------ running
def _run(job: dict, cmd: list[str] | str) -> None:
    directory = job_dir(job["id"])
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONUNBUFFERED"] = "1"
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
    if kind not in ("map", "cruise") or engine not in ("original", "moge3"):
        raise ValueError("任務類型或引擎錯誤")
    status = engine_status()[engine]
    if not status["ready"]:
        raise ValueError("此引擎尚未安裝：" + status["hint"])
    job_id = time.strftime("%Y%m%d%H%M%S") + uuid.uuid4().hex[:6]
    job_dir(job_id).mkdir(parents=True)
    try:
        cmd, meta = build_map_command(request) if kind == "map" else build_cruise_command(request, job_id)
    except ValueError:
        job_dir(job_id).rmdir()
        raise
    job = {"id": job_id, "kind": kind, "engine": engine, "status": "queued", "created": time.time(), **meta,
           "request": {k: v for k, v in request.items() if k in ("videos", "out", "name", "intrinsics", "video", "map", "route", "skip_dense")}}
    write_job(job)
    threading.Thread(target=_run, args=(job, cmd), daemon=True).start()
    return job


def stop_job(job_id: str) -> None:
    with LOCK:
        proc = PROCS.get(job_id)
    if proc is None or proc.poll() is not None:
        return
    job = read_job(job_id)
    job["status"] = "stopped"
    write_job(job)
    os.killpg(proc.pid, signal.SIGTERM)


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
