"""`acam`: videos in, localizable map (plus a transfer bundle) out."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import __version__, plan, runner
from .config import REPO, Paths, load_paths
from .hw import GIB, free_bytes, gpu, read_meminfo, scope_available
from .report import to_markdown
from .videos import aspect_mismatches, find_videos, probe

DEFAULT_INTRINSICS = REPO / "configs/river_intrinsics_undistorted.json"
COMMANDS = {"map", "doctor", "pack", "localize", "setup"}
MIN_AVAILABLE_GIB = 10


def say(message: str) -> None:
    print(message, flush=True)


# --------------------------------------------------------------------------- doctor
def doctor(paths: Paths) -> list[tuple[str, bool, str]]:
    rows: list[tuple[str, bool, str]] = []
    for name, path in paths.required_files().items():
        rows.append((name, path.exists(), str(path)))
    gpu_info = gpu()
    rows.append(("gpu", gpu_info is not None, json.dumps(gpu_info) if gpu_info else "nvidia-smi found no GPU"))
    mem = read_meminfo()
    rows.append(("ram>=24GiB", mem.total >= 24 * GIB,
                 f"{mem.total / GIB:.1f} GiB total, {mem.available / GIB:.1f} available, swap {mem.swap_total / GIB:.1f}"))
    rows.append(("ffmpeg/ffprobe", bool(subprocess.run("command -v ffprobe ffmpeg", shell=True, capture_output=True).stdout), ""))
    probe_code = ("import torch,pycolmap,sys;"
                  "print(torch.__version__, torch.cuda.is_available(), pycolmap.__version__)")
    result = subprocess.run([str(paths.python), "-c", probe_code], capture_output=True, text=True, env=runner.engine_env(paths))
    ok = result.returncode == 0 and " True " in f" {result.stdout.strip()} "
    rows.append(("engine python (torch+cuda, pycolmap)", ok, result.stdout.strip() or result.stderr.strip()[-200:]))
    if ok:
        gm = subprocess.run([str(paths.python), "-c", "import pygluemap,gluemap;print('ok')"], capture_output=True,
                            text=True, env=runner.engine_env(paths))
        rows.append(("gluemap + pygluemap import", gm.returncode == 0, gm.stdout.strip() or gm.stderr.strip()[-200:]))
    rows.append(("memory cgroup (systemd-run --user)", scope_available(), "optional: protects the host from OOM"))
    return rows


def cmd_doctor(args: argparse.Namespace) -> int:
    rows = doctor(load_paths())
    for name, ok, detail in rows:
        say(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")
    blockers = [name for name, ok, _ in rows if not ok and not name.startswith("memory cgroup")]
    say("ready" if not blockers else f"not ready: {', '.join(blockers)}")
    return 0 if not blockers else 1


# --------------------------------------------------------------------------- map
def safe_name(text: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return name or "site"


def screen_videos(paths: Paths, videos: list[Path], meta: Path) -> list[dict]:
    out = meta / "screening.json"
    cmd = [str(paths.python), "-m", "acam.screen", *[str(v) for v in videos], "--out", str(out),
           "--workers", str(min(4, os.cpu_count() or 1))]
    result = subprocess.run(cmd, env=runner.engine_env(paths), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"video screening failed:\n{result.stderr[-1500:]}")
    return json.loads(out.read_text(encoding="utf-8"))


def stage_corpus(corpus: Path, keep: list[str], sources: dict[str, Path]) -> str:
    """The engine reads a folder and refuses symlinked sources: hard-link the videos that passed
    (free on the same filesystem), or copy when the output lives on another disk."""

    if corpus.exists():
        shutil.rmtree(corpus)
    corpus.mkdir(parents=True)
    mode = "hardlink"
    for name in keep:
        try:
            os.link(sources[name], corpus / name)
        except OSError:
            shutil.copy2(sources[name], corpus / name)
            mode = "copy"
    return mode


def cmd_map(args: argparse.Namespace) -> int:
    paths = load_paths()
    source = args.videos.expanduser().resolve()
    videos = find_videos(source)
    if not videos:
        say(f"no videos in {source}")
        return 2
    out = args.out.expanduser().resolve()
    if out == source or source in out.parents:
        say("--out must not be inside the video folder")
        return 2
    name = safe_name(args.name or source.name)
    base = out / name
    base.mkdir(parents=True, exist_ok=True)
    meta = base / "acam"
    meta.mkdir(parents=True, exist_ok=True)
    intrinsics_src = (args.intrinsics or DEFAULT_INTRINSICS).expanduser().resolve()
    calibration = json.loads(intrinsics_src.read_text(encoding="utf-8"))
    mem = read_meminfo()
    sources = {v.name: v for v in videos}

    say(f"a_click_a_map {__version__}: {len(videos)} video(s) in {source}")
    if not args.skip_checks:
        rows = doctor(paths)
        blockers = [(n, d) for n, ok, d in rows if not ok and not n.startswith("memory cgroup")]
        if blockers:
            for n, d in blockers:
                say(f"  [FAIL] {n}: {d}")
            say("run `./run.sh doctor` and `./run.sh setup` first (or --skip-checks)")
            return 3
        g = gpu()
        if g and g["memory_total_mib"] < 16000:
            say(f"  WARNING: GPU has {g['memory_total_mib'] / 1024:.0f} GiB; the pipeline peaked at ~12 GiB VRAM on 2000 keyframes")
    if mem.available < MIN_AVAILABLE_GIB * GIB and not args.force:
        top = subprocess.run("ps -eo rss,comm --sort=-rss | head -6", shell=True, capture_output=True, text=True).stdout
        say(f"only {mem.available / GIB:.1f} GiB RAM available now (need >= {MIN_AVAILABLE_GIB}); close other programs or use --force.\n"
            f"largest processes (RSS kB):\n{top}")
        return 3

    # ---- [0] screening ---------------------------------------------------------------
    say("[0/4] screening videos (blur, exposure, texture, readability, motion)")
    screening = screen_videos(paths, videos, meta)
    infos = {Path(r["path"]).name: r for r in screening}
    excluded: dict[str, str] = {}
    for row in screening:
        label = Path(row["path"]).name
        say(f"  {row['verdict']:6s} {label}: {row['width']}x{row['height']} {row['fps']:.1f}fps {row['duration']:.0f}s"
            + (f" - {'; '.join(row['reasons'])}" if row["reasons"] else ""))
        if row["verdict"] == "REJECT" and not args.keep_bad:
            excluded[label] = "; ".join(row["reasons"])
    probed = {v.name: probe(v) for v in videos if v.name not in excluded}
    bad = aspect_mismatches(list(probed.values()), calibration)
    if bad:
        say("these videos do not match the calibration's aspect ratio "
            f"({calibration['image_width']}x{calibration['image_height']}): " + ", ".join(v.path.name for v in bad))
        say("one run supports one calibrated camera group; give --intrinsics for that camera or split the folder")
        return 2
    keep = [v.name for v in videos if v.name not in excluded]
    if not keep:
        say("no video passed screening")
        return 2

    # ---- budget for THIS machine -------------------------------------------------------
    budget = plan.plan_budget(
        [plan.VideoBudget(n, infos[n]["duration"], infos[n]["score"] or 0.5) for n in keep], mem.total,
        max_keyframes=args.max_keyframes)
    for dropped_name, why in budget.dropped:
        excluded[dropped_name] = why
    keep = budget.keep
    scale = args.sampling_scale if args.sampling_scale else budget.sampling_scale
    say(f"  machine: {mem.total / GIB:.0f} GiB RAM ({mem.available / GIB:.0f} available) -> keyframe cap "
        f"{plan.keyframe_cap(mem.total, args.max_keyframes)} at the validated setting")
    for note in budget.notes:
        say(f"  budget: {note}")
    say(f"  using {len(keep)} of {len(videos)} video(s), sampling x{scale:.2f}, ~{budget.est_keyframes} keyframes, "
        f"~{plan.eta_hours(budget.est_keyframes):.1f} h on an RTX 5090-class GPU")
    need = 15 * GIB + budget.est_keyframes * 8 * 2**20
    if out.exists() and os.stat(out).st_dev != os.stat(source).st_dev:
        need += sum(sources[k].stat().st_size for k in keep)  # videos get copied next to the run
    if free_bytes(out) < need:
        say(f"not enough free disk in {out}: {free_bytes(out) / GIB:.0f} GiB free, ~{need / GIB:.0f} GiB needed")
        return 3

    job = runner.MapJob(site=safe_name(name), corpus=base / "corpus", run=base / "run", workspace=base / "workspace",
                        meta=meta, intrinsics=meta / "intrinsics.json")
    job.intrinsics.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")

    if runner.finished(job.run):
        say(f"mapping already finished in {job.run}; skipping to report/pack")
        mapping = json.loads((meta / "mapping_summary.json").read_text(encoding="utf-8"))
        excluded.update(json.loads((meta / "excluded.json").read_text(encoding="utf-8")) if (meta / "excluded.json").exists() else {})
    else:
        say("[1/4] preprocessing: hashing sources, extracting adaptive keyframes")
        tracks = budget.tracks
        weak_changed = True
        for round_ in range(1, 4):
            if round_ == 1 or weak_changed:
                mode = stage_corpus(job.corpus, keep, sources)
                if mode == "copy":
                    say("      note: --out is on another filesystem than the videos, so they were copied")
                weak_changed = False
            n_images = runner.preprocess(paths, job, tracks, scale)
            weak = {k: v for k, v in runner.weak_sources(job.run).items() if not args.keep_bad}
            if weak and len(weak) < len(keep):
                for k, v in weak.items():
                    say(f"      excluded {k}: {v}")
                excluded.update(weak)
                keep = [k for k in keep if k not in weak]
                runner.reset_run(job)
                weak_changed = True
                continue
            if weak:
                say("      every video looks unusable after frame analysis; aborting")
                return 2
            retry_scale = None if args.sampling_scale else plan.rescale_after_preprocess(n_images, scale, mem.total, args.max_keyframes)
            if retry_scale is not None and retry_scale >= plan.MIN_SCALE * 0.8 and round_ < 3:
                say(f"      {n_images} keyframes exceed the memory cap; re-extracting at sampling x{retry_scale:.2f}")
                scale = retry_scale
                runner.reset_run(job)
                continue
            break
        if args.tracks == "auto":
            tracks, points, fits = plan.choose_tracks(n_images, mem.total)
        else:
            tracks = int(args.tracks)
            points = n_images * plan.points_per_image(tracks)
            fits = points <= plan.point_budget(mem.total)
        say(f"      {n_images} keyframes -> num_track_per_img={tracks} "
            f"(~{points / 1e6:.1f} M track points, budget {plan.point_budget(mem.total) / 1e6:.1f} M on {mem.total / GIB:.0f} GiB RAM)")
        if not fits:
            say("      WARNING: even the smallest setting exceeds the memory budget; expect OOM (more RAM or fewer videos)")
        (meta / "excluded.json").write_text(json.dumps(excluded, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.dry_run:
            say("dry run: stopping before the GlueMap/EDM run")
            return 0
        say(f"[2/4] mapping: GlueMap poses -> EDM retriangulation -> MegaLoc bank (~{plan.eta_hours(n_images):.1f} h; "
            "memory-guarded, auto-retry on OOM)")
        started = time.time()
        mapping = runner.run_map(paths, job, tracks=tracks, scale=scale, max_attempts=args.max_attempts)
        mapping.update(minutes=round((time.time() - started) / 60, 1), sampling_scale=scale)
        (meta / "mapping_summary.json").write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")

    say("[3/4] report: registration and cross-video connectivity")
    report_path = base / "map_report.json"
    result = subprocess.run([str(paths.python), "-m", "acam.report", str(job.run), "--out", str(report_path)],
                            env=runner.engine_env(paths), capture_output=True, text=True)
    if result.returncode != 0:
        say(result.stderr[-1500:])
        return 4
    report = json.loads(report_path.read_text(encoding="utf-8"))
    markdown = to_markdown(report, name, mapping, screening=infos, excluded=excluded)
    (base / "MAP_REPORT.md").write_text(markdown, encoding="utf-8")
    say(markdown)

    archive = None
    if not args.no_pack:
        say("[4/4] packing the localization/densification bundle")
        from .pack import pack
        freeze = subprocess.run([str(paths.python), "-m", "pip", "freeze"], capture_output=True, text=True).stdout
        archive = pack(job.run, base, f"{name}_bundle", with_megaloc_weights=not args.no_megaloc_weights,
                       mapping_summary=markdown, freeze=freeze)
        say(f"      {archive} ({archive.stat().st_size / GIB:.2f} GiB)")
    for query in args.query or []:
        say(f"[localize] {query.name}")
        run_localize(paths, job.run, query, base / "localize_test")
    say(f"done: map in {job.run / 'artifacts/mapping/edm/model'}" + (f", bundle {archive}" if archive else ""))
    return 0 if report["connected"] and report["images_registered"] == report["images_total"] else 5


def run_localize(paths: Paths, run: Path, video: Path, out: Path) -> int:
    route = safe_name(video.stem)
    result = subprocess.run([str(paths.python), "-m", "acam.localize_run", str(run), "--video", str(video.resolve()),
                             "--route", route, "--out", str(out)], env=runner.engine_env(paths), text=True)
    return result.returncode


def cmd_localize(args: argparse.Namespace) -> int:
    return run_localize(load_paths(), args.run.resolve(), args.video, args.out.resolve())


def cmd_pack(args: argparse.Namespace) -> int:
    from .pack import pack
    paths = load_paths()
    run = args.run.resolve()
    freeze = subprocess.run([str(paths.python), "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    archive = pack(run, args.out.resolve(), args.name, with_megaloc_weights=not args.no_megaloc_weights, freeze=freeze)
    say(str(archive))
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    script = REPO / "scripts/setup.sh"
    return subprocess.run(["bash", str(script), *args.rest]).returncode


# --------------------------------------------------------------------------- argparse
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="acam", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("map", help="videos -> map + bundle (default command)")
    m.add_argument("videos", type=Path, help="folder with the videos (one calibrated camera group)")
    m.add_argument("--out", type=Path, default=Path("acam_out"))
    m.add_argument("--name", help="run name (default: video folder name)")
    m.add_argument("--intrinsics", type=Path, help="undistorted PINHOLE calibration JSON (default: River camera)")
    m.add_argument("--tracks", default="auto", help="num_track_per_img: 'auto' (RAM-based) or an integer 256..1024")
    m.add_argument("--max-attempts", type=int, default=3)
    m.add_argument("--max-keyframes", type=int, help="hard cap on keyframes (default: from this machine's RAM)")
    m.add_argument("--sampling-scale", type=float, help="override keyframe sampling density (default: automatic, 0.5-1.0)")
    m.add_argument("--keep-bad", action="store_true", help="do not exclude videos that fail screening")
    m.add_argument("--force", action="store_true", help="start even if little RAM is available right now")
    m.add_argument("--query", type=Path, action="append", help="query video to localize afterwards (repeatable)")
    m.add_argument("--no-pack", action="store_true")
    m.add_argument("--no-megaloc-weights", action="store_true", help="leave MegaLoc weights out of the bundle")
    m.add_argument("--dry-run", action="store_true", help="preprocess and plan only")
    m.add_argument("--skip-checks", action="store_true")
    m.set_defaults(func=cmd_map)
    d = sub.add_parser("doctor", help="check the environment")
    d.set_defaults(func=cmd_doctor)
    p = sub.add_parser("pack", help="pack a finished run directory")
    p.add_argument("run", type=Path)
    p.add_argument("--out", type=Path, default=Path("."))
    p.add_argument("--name", default="map_bundle")
    p.add_argument("--no-megaloc-weights", action="store_true")
    p.set_defaults(func=cmd_pack)
    q = sub.add_parser("localize", help="EDM+KLT localization of a video against a finished run")
    q.add_argument("run", type=Path)
    q.add_argument("video", type=Path)
    q.add_argument("--out", type=Path, default=Path("localize_out"))
    q.set_defaults(func=cmd_localize)
    s = sub.add_parser("setup", help="clone/patch third-party code and download checkpoints")
    s.add_argument("rest", nargs=argparse.REMAINDER)
    s.set_defaults(func=cmd_setup)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in COMMANDS and not argv[0].startswith("-"):
        argv.insert(0, "map")  # `./run.sh /path/to/videos`
    args = build_parser().parse_args(argv)
    return int(args.func(args))
