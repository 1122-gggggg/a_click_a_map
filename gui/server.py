#!/usr/bin/env python3
"""無人機自動建圖巡航系統 - local web UI for the two a_click_a_map entry points.

    ./gui.sh [--port 8765] [--host 127.0.0.1]

Standard library only. Two engines:
    original  ./run.sh   (SALAD/MegaLoc GlueMap + EDM, folder of videos in)
    moge3     ./run_moge3.sh (MoGe-3 + MegaLoc GlueMap + MVSFormer++ dense map)
Tasks:  map    videos -> map (runs the engine as a subprocess, streams its log)
        cruise a query video against a finished map (EDM + MegaLoc + KLT), shows the flown trajectory
Jobs live in acam_gui/jobs/<id>/ (job.json + log.txt). The server binds to localhost, requires JSON POSTs and a
local Host header; it has no login, so do not expose it to a network.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parents[1]
JOBS = REPO / "acam_gui" / "jobs"
STATIC = Path(__file__).resolve().parent / "index.html"
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mts"}
TITLE = "無人機自動建圖巡航系統"
LOCK = threading.Lock()
PROCS: dict[str, subprocess.Popen] = {}


def toolkit() -> Path:
    return Path(os.environ.get("MOGE3_TOOLKIT", REPO / "third_party/moge3"))


def engine_status() -> dict:
    tk = toolkit()
    return {
        "original": {
            "label": "原始版（EDM + KLT，輸入影片資料夾）",
            "ready": (REPO / "run.sh").exists() and (REPO / "third_party").is_dir(),
            "hint": "先執行 ./run.sh setup",
        },
        "moge3": {
            "label": "MoGe-3 版（稀疏 + MVSFormer++ 稠密 + 定位包）",
            "ready": (tk / "envs/gluemap/bin/python").exists() and (tk / "weights").is_dir(),
            "hint": "先執行 moge3/scripts/setup.sh",
        },
    }


def loc_python() -> str:
    return os.environ.get("LOC_PYTHON") or str(toolkit() / "envs/gluemap/bin/python")


# ----------------------------------------------------------------------------------------------- jobs
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


def build_map_command(body: dict) -> tuple[list[str], dict]:
    engine = body["engine"]
    entries = [e.strip() for e in body.get("videos", []) if e.strip()]
    if not entries:
        raise ValueError("請輸入影片路徑")
    name = "".join(c if c.isalnum() or c in "._-" else "_" for c in (body.get("name") or "site")).strip("._-") or "site"
    out = Path(body.get("out") or REPO / "acam_out").expanduser().resolve()
    intrinsics = (body.get("intrinsics") or "").strip()
    extra = shlex.split(body.get("extra") or "")
    meta: dict = {"name": name}
    if engine == "original":
        if len(entries) != 1 or not Path(entries[0]).expanduser().is_dir():
            raise ValueError("原始版一次只吃一個影片資料夾")
        cmd = [str(REPO / "run.sh"), "map", str(Path(entries[0]).expanduser().resolve()), "--out", str(out), "--name", name]
        if intrinsics:
            cmd += ["--intrinsics", intrinsics]
        meta["result_dir"] = str(out / name)
    elif engine == "moge3":
        videos = expand_videos(entries)
        target = out / name
        cmd = [str(REPO / "run_moge3.sh"), *[str(v.resolve()) for v in videos], "--output", str(target)]
        cmd += ["--intrinsics", intrinsics] if intrinsics else ["--auto_calibrate"]
        if body.get("skip_dense"):
            cmd.append("--skip-dense")
        meta["result_dir"] = str(target)
    else:
        raise ValueError("未知引擎")
    return cmd + extra, meta


def build_cruise_command(body: dict, job: dict) -> tuple[list[str] | str, dict]:
    engine = body["engine"]
    video = Path(body.get("video", "")).expanduser()
    if not video.is_file():
        raise ValueError("找不到巡航影片")
    map_dir = Path(body.get("map") or "").expanduser()
    if not map_dir.is_dir():
        raise ValueError("找不到地圖目錄")
    route = "".join(c if c.isalnum() or c in "._-" else "_" for c in (body.get("route") or video.stem)) or "route"
    out = job_dir(job["id"]) / "cruise"
    meta = {"result_dir": str(out), "route": route}
    if engine == "original":
        run_dir = map_dir / "run" if (map_dir / "run").is_dir() else map_dir
        return [str(REPO / "run.sh"), "localize", str(run_dir.resolve()), str(video.resolve()), "--out", str(out)], meta
    bundle = map_dir / "localization/bundle" if (map_dir / "localization/bundle").is_dir() else map_dir
    if not (bundle / "run_localize.py").is_file():
        raise ValueError("moge3 地圖目錄需含 localization/bundle（或直接指定 bundle）")
    py = shlex.quote(loc_python())
    script = (f"cd {shlex.quote(str(bundle.resolve()))} && "
              f"( [ -f localizer_config.json ] || {py} install.py ) && "
              f"{py} run_localize.py --video {shlex.quote(str(video.resolve()))} --route {shlex.quote(route)} --output {shlex.quote(str(out))}")
    return script, meta


def run_job(job: dict, cmd: list[str] | str) -> None:
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


def create_job(body: dict) -> dict:
    kind = body.get("kind")
    if kind not in ("map", "cruise"):
        raise ValueError("kind 必須是 map 或 cruise")
    if body.get("engine") not in ("original", "moge3"):
        raise ValueError("未知引擎")
    if not engine_status()[body["engine"]]["ready"]:
        raise ValueError("引擎尚未安裝：" + engine_status()[body["engine"]]["hint"])
    job_id = time.strftime("%Y%m%d%H%M%S") + uuid.uuid4().hex[:6]
    job = {"id": job_id, "kind": kind, "engine": body["engine"], "status": "queued", "created": time.time(),
           "title": body.get("name") or body.get("route") or kind}
    job_dir(job_id).mkdir(parents=True)
    cmd, meta = build_map_command(body) if kind == "map" else build_cruise_command(body, job)
    job.update(meta)
    job["request"] = {k: v for k, v in body.items() if k in ("videos", "out", "name", "intrinsics", "video", "map", "route", "skip_dense")}
    write_job(job)
    threading.Thread(target=run_job, args=(job, cmd), daemon=True).start()
    return job


def stop_job(job_id: str) -> dict:
    job = read_job(job_id)
    with LOCK:
        proc = PROCS.get(job_id)
    if proc is None or proc.poll() is not None:
        return job
    job["status"] = "stopped"
    write_job(job)
    os.killpg(proc.pid, signal.SIGTERM)
    return job


# ----------------------------------------------------------------------------------------------- results
def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return ""


def map_result(job: dict) -> dict:
    base = Path(job.get("result_dir", ""))
    files = []
    if job["engine"] == "moge3":
        wanted = [("稀疏點雲 (原始 RGB)", "map/sparse_rgb.ply"), ("稠密點雲 (MVSFormer++)", "dense/dense.ply"),
                  ("COLMAP 地圖", "map/model"), ("定位包 (搬到另一台電腦)", "localization/bundle"),
                  ("流程收據", "pipeline_receipt.json")]
    else:
        wanted = [("定位用地圖", "run/artifacts/mapping/edm/model"), ("建圖報告", "MAP_REPORT.md"),
                  ("轉接包", f"{job.get('name', '')}_bundle.tar.zst"), ("RGB 點雲", "run/products/map/original_rgb.ply")]
    for label, rel in wanted:
        path = base / rel
        if path.exists():
            size = path.stat().st_size if path.is_file() else sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
            files.append({"label": label, "path": str(path), "size": human(size)})
    return {"files": files, "map_dir": str(base)}


def cruise_result(job: dict) -> dict:
    out = Path(job.get("result_dir", ""))
    summaries = sorted(out.rglob("summary.json"), key=lambda p: p.stat().st_mtime)
    if not summaries:
        return {"ready": False}
    folder = summaries[-1].parent
    summary = json.loads(summaries[-1].read_text(encoding="utf-8"))
    points = []
    frames = folder / "frames.jsonl"
    if frames.exists():
        rows = [json.loads(line) for line in frames.read_text(encoding="utf-8").splitlines() if line.strip()]
        step = max(1, len(rows) // 1500)
        points = [{"i": r["ordinal"], "p": r["position"], "s": r["status"]} for r in rows[::step]]
    return {"ready": True, "summary": {k: summary.get(k) for k in ("route", "dense_frames", "status_counts", "pose_coverage",
                                                                    "fps_5090_measured", "stream_fps")},
            "reloc": summary.get("reloc", {}).get("statuses"), "points": points, "dir": str(folder)}


def browse(path: str) -> dict:
    p = Path(path or str(Path.home())).expanduser().resolve()
    if not p.is_dir():
        p = p.parent
    entries = []
    for child in sorted(p.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower())):
        if child.name.startswith("."):
            continue
        if child.is_dir() or child.suffix.lower() in VIDEO_EXT:
            entries.append({"name": child.name, "dir": child.is_dir(), "path": str(child)})
    return {"path": str(p), "parent": str(p.parent), "entries": entries[:500]}


# ----------------------------------------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    server_version = "AcamGUI/1"

    def log_message(self, *_args) -> None:
        pass

    def local_host(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        return host in ("127.0.0.1", "localhost", "::1")

    def send(self, code: int, payload, ctype: str = "application/json; charset=utf-8") -> None:
        data = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if not self.local_host():
            return self.send(403, {"error": "forbidden host"})
        url = urlparse(self.path)
        query = parse_qs(url.query)
        try:
            if url.path == "/":
                return self.send(200, STATIC.read_bytes(), "text/html; charset=utf-8")
            if url.path == "/api/config":
                return self.send(200, {"title": TITLE, "engines": engine_status(), "repo": str(REPO), "default_out": str(REPO / "acam_out")})
            if url.path == "/api/jobs":
                return self.send(200, list_jobs())
            if url.path == "/api/browse":
                return self.send(200, browse(query.get("path", [""])[0]))
            parts = url.path.strip("/").split("/")
            if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
                return self.send(200, read_job(parts[2]))
            if len(parts) == 4 and parts[:2] == ["api", "jobs"]:
                job = read_job(parts[2])
                if parts[3] == "log":
                    offset = int(query.get("offset", ["0"])[0])
                    data = (job_dir(parts[2]) / "log.txt").read_bytes() if (job_dir(parts[2]) / "log.txt").exists() else b""
                    return self.send(200, {"offset": len(data), "text": data[offset:].decode("utf-8", "replace")})
                if parts[3] == "result":
                    return self.send(200, map_result(job) if job["kind"] == "map" else cruise_result(job))
            self.send(404, {"error": "not found"})
        except (ValueError, OSError) as error:
            self.send(400, {"error": str(error)})

    def do_POST(self) -> None:
        if not self.local_host() or "application/json" not in (self.headers.get("Content-Type") or ""):
            return self.send(403, {"error": "forbidden"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            parts = urlparse(self.path).path.strip("/").split("/")
            if parts == ["api", "jobs"]:
                return self.send(200, create_job(body))
            if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "stop":
                return self.send(200, stop_job(parts[2]))
            self.send(404, {"error": "not found"})
        except (ValueError, KeyError, OSError) as error:
            self.send(400, {"error": str(error)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    JOBS.mkdir(parents=True, exist_ok=True)
    for job in list_jobs():  # a job that was running when the server stopped cannot be resumed
        if job.get("status") in ("running", "queued"):
            job.update(status="interrupted")
            write_job(job)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"{TITLE}: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
