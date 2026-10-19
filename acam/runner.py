"""Run the GlueMap + EDM mapping engine inside a memory-limited scope, retrying after OOM.

Retry policy: the engine's stage caches (retrieval descriptors, two-view results) survive a
kill, but its workspace identity includes the GlueMap config, so lowering
`num_track_per_img` opens a *new* workspace. `prime_workspace` lets the engine create that
workspace, then copies the K-independent caches into it so only the star stage and later
stages are recomputed.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import plan
from .config import GLUEMAP_CHECKPOINTS, REPO, Paths
from .hw import GIB, read_meminfo, scope_available

PROGRESS = re.compile(r"(\d+%\|[^\r\n]*?\[[^\]\r\n]*\])")
SAMPLE_SECONDS = 30
HEARTBEAT_SECONDS = 120


@dataclass(frozen=True)
class MapJob:
    site: str
    corpus: Path
    run: Path
    workspace: Path
    meta: Path          # acam metadata dir: configs, logs, memory trace
    intrinsics: Path


def render_gluemap_config(paths: Paths, tracks: int, out: Path) -> Path:
    config = json.loads((REPO / "configs/gluemap_base.json").read_text(encoding="utf-8"))
    for key, name in GLUEMAP_CHECKPOINTS.items():
        config[key] = str(paths.gluemap_checkpoints / name)
    config["num_track_per_img"] = int(tracks)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return out


def map_argv(paths: Paths, job: MapJob, config: Path, *, preprocess_only: bool = False) -> list[str]:
    argv = [
        str(paths.python), "-m", "sfm_diagnosis.site_pipeline.cli", "map",
        "--site-name", job.site,
        "--corpus", str(job.corpus),
        "--run", str(job.run),
        "--intrinsics", str(job.intrinsics),
        "--gluemap-root", str(paths.gluemap),
        "--gluemap-config", str(config),
        "--workspace-root", str(job.workspace),
        "--megaloc-source", str(paths.megaloc_source),
        "--megaloc-checkpoint", str(paths.megaloc_checkpoint),
        "--edm-root", str(paths.edm),
        "--edm-checkpoint", str(paths.edm_checkpoint),
        "--resume",
    ]
    if preprocess_only:
        argv.append("--preprocess-only")
    return argv


def engine_env(paths: Paths) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = paths.engine_pythonpath() + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("OMP_NUM_THREADS", "4")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _cgroup_dir(pid: int) -> Path | None:
    try:
        line = Path(f"/proc/{pid}/cgroup").read_text().strip().splitlines()[-1]
    except (OSError, IndexError):
        return None
    path = Path("/sys/fs/cgroup") / line.split("::", 1)[-1].lstrip("/")
    return path if (path / "memory.current").exists() else None


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def run_logged(argv: list[str], *, env: dict[str, str], log: Path, trace: Path, label: str,
               use_scope: bool, mem_max: int, swap_max: int) -> tuple[int, str, dict[str, float]]:
    """Run `argv`, append to `log`, sample memory into `trace`; return (rc, log tail, memory stats)."""

    log.parent.mkdir(parents=True, exist_ok=True)
    command = list(argv)
    if use_scope:
        unit = f"acam-{label}-{int(time.time())}"
        command = ["systemd-run", "--user", "--scope", "--quiet", f"--unit={unit}",
                   "-p", f"MemoryMax={mem_max}", "-p", f"MemorySwapMax={swap_max}", *argv]
    stats = {"peak_gib": 0.0, "oom_kills": 0.0}
    with log.open("ab") as stream:
        stream.write(f"\n=== {label}: {' '.join(command[:6])} ...\n".encode())
        stream.flush()
        proc = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        stop = threading.Event()

        def sample() -> None:
            started = last_beat = time.time()
            with trace.open("a") as out:
                while not stop.wait(SAMPLE_SECONDS):
                    group = _cgroup_dir(proc.pid) if use_scope else None
                    current = _read_int(group / "memory.current") if group else None
                    peak = _read_int(group / "memory.peak") if group else None
                    if current is None:
                        current = read_meminfo().total - read_meminfo().available
                        peak = current
                    kills = 0
                    if group and (group / "memory.events").exists():
                        for line in (group / "memory.events").read_text().splitlines():
                            if line.startswith("oom_kill "):
                                kills = int(line.split()[1])
                    stats["peak_gib"] = max(stats["peak_gib"], (peak or current) / GIB)
                    stats["oom_kills"] = max(stats["oom_kills"], kills)
                    mem = read_meminfo()
                    out.write(f"{time.strftime('%H:%M:%S')},{current / GIB:.1f},{(peak or current) / GIB:.1f},"
                              f"{mem.available / GIB:.1f},{(mem.swap_total - mem.swap_free) / GIB:.1f},{kills}\n")
                    out.flush()
                    if time.time() - last_beat >= HEARTBEAT_SECONDS:
                        last_beat = time.time()
                        print(f"  [{label}] {int(time.time() - started) // 60:>4} min  mem {current / GIB:.1f} GiB "
                              f"(peak {stats['peak_gib']:.1f})  {last_progress(log)}", flush=True)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        try:
            rc = proc.wait()
        except KeyboardInterrupt:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait()
            raise
        finally:
            stop.set()
            sampler.join(timeout=5)
    return rc, tail(log), stats


def tail(path: Path, size: int = 200_000) -> str:
    with path.open("rb") as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(stream.tell() - size, 0))
        return stream.read().decode("utf-8", errors="replace")


def last_progress(log: Path) -> str:
    matches = PROGRESS.findall(tail(log, 20_000).replace("\r", "\n"))
    return matches[-1][:70] if matches else ""


def work_dirs(workspace: Path) -> set[Path]:
    return {p for p in workspace.glob("work_*") if p.is_dir()}


def prime_workspace(argv: list[str], env: dict[str, str], workspace: Path, old: set[Path], *, timeout: float = 240.0) -> Path | None:
    """Let the engine create the workspace for a changed config, then stop it."""

    proc = subprocess.Popen(argv, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    deadline = time.time() + timeout
    new: Path | None = None
    try:
        while time.time() < deadline and proc.poll() is None:
            fresh = work_dirs(workspace) - old
            if fresh and (next(iter(fresh)) / "gluemap").is_dir():
                new = next(iter(fresh))
                time.sleep(5)  # let image links materialise
                break
            time.sleep(1)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
    return new


def transplant_caches(old: Path, new: Path) -> list[str]:
    """Copy the caches that do not depend on num_track_per_img."""

    copied: list[str] = []
    src, dst = old / "gluemap", new / "gluemap"
    twoview = src / "twoview_result.pth"
    if twoview.is_file():
        shutil.copy2(twoview, dst / twoview.name)
        copied.append(twoview.name)
    for folder in sorted(src.glob("vid_*")):
        if folder.is_dir():
            shutil.copytree(folder, dst / folder.name, dirs_exist_ok=True)
            copied.append(folder.name)
    return copied


def finished(run: Path) -> bool:
    receipt = run / "products/FINAL_RECEIPT.json"
    if not receipt.is_file():
        return False
    return json.loads(receipt.read_text(encoding="utf-8")).get("artifact_type") == "ADOPTED_MAPPING_FINAL_RECEIPT"


def keyframe_count(run: Path) -> int:
    receipt = json.loads((run / "receipts/direct_preprocessing.json").read_text(encoding="utf-8"))
    return int(receipt["keyframes"])


def preprocess(paths: Paths, job: MapJob, tracks: int) -> int:
    """Hash sources, extract adaptive keyframes; returns the keyframe count. Idempotent."""

    if not (job.run / "receipts/direct_preprocessing.json").is_file():
        config = render_gluemap_config(paths, tracks, job.meta / f"gluemap_config_k{tracks}.json")
        rc, text, _ = run_logged(map_argv(paths, job, config, preprocess_only=True), env=engine_env(paths),
                                 log=job.meta / "logs/mapping.log", trace=job.meta / "logs/memory.csv",
                                 label="preprocess", use_scope=False, mem_max=0, swap_max=0)
        if rc != 0:
            raise RuntimeError(f"preprocessing failed (rc={rc}); see {job.meta / 'logs/mapping.log'}\n{text[-1500:]}")
    return keyframe_count(job.run)


def run_map(paths: Paths, job: MapJob, *, tracks: int, max_attempts: int = 3) -> dict:
    """Run mapping to completion (after `preprocess`), lowering K after OOM kills."""

    env = engine_env(paths)
    meta = job.meta
    log, trace = meta / "logs/mapping.log", meta / "logs/memory.csv"
    mem = read_meminfo()
    mem_max, swap_max = plan.memory_limits(mem.total, mem.swap_free)
    use_scope = scope_available()
    if not use_scope:
        print("  note: systemd-run --user scope unavailable; running without a memory cgroup", flush=True)
    history: list[dict] = []
    config = render_gluemap_config(paths, tracks, meta / f"gluemap_config_k{tracks}.json")
    for attempt in range(1, max_attempts + 1):
        rc, text, stats = run_logged(map_argv(paths, job, config), env=env, log=log, trace=trace,
                                     label=f"map{attempt}", use_scope=use_scope, mem_max=mem_max, swap_max=swap_max)
        history.append({"attempt": attempt, "num_track_per_img": tracks, "returncode": rc,
                        "peak_gib": round(stats["peak_gib"], 1), "oom_kills": stats["oom_kills"]})
        if rc == 0 and finished(job.run):
            return {"num_track_per_img": tracks, "attempts": history}
        oom = plan.is_oom_exit(rc, text) or stats["oom_kills"] > 0
        lower = plan.reduce_tracks(tracks) if oom else None
        if lower is None or attempt == max_attempts:
            raise RuntimeError(
                f"mapping failed (rc={rc}, oom={oom}) after {attempt} attempt(s); see {log}\n{text[-1500:]}"
            )
        print(f"  OOM at num_track_per_img={tracks}; retrying with {lower}", flush=True)
        old = work_dirs(job.workspace)
        config = render_gluemap_config(paths, lower, meta / f"gluemap_config_k{lower}.json")
        new = prime_workspace(map_argv(paths, job, config), env, job.workspace, old)
        if new is not None and old:
            newest = max(old, key=lambda p: (p / "gluemap/twoview_result.pth").stat().st_mtime
                         if (p / "gluemap/twoview_result.pth").exists() else 0)
            history[-1]["transplanted"] = transplant_caches(newest, new)[:3]
        tracks = lower
    raise AssertionError("unreachable")
