"""Localize one query video against a finished run (EDM + KLT two-rate loop). Engine environment.

    python -m acam.localize_run RUN_DIR --video query.MP4 --route NAME --out DIR
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def video_fps(video: Path) -> float:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate", "-of", "csv=p=0", str(video)],
        text=True,
    ).strip()
    num, den = out.split("/")
    return float(num) / float(den)


def localize(run: Path, video: Path, route: str, out: Path) -> dict:
    from sfm_diagnosis.site_pipeline.deploy_loop import DeployConfig, run_deploy_loop

    config = json.loads((run / "products/localization/localizer_config.json").read_text(encoding="utf-8"))
    frames = out / "frames"
    if not any(frames.glob("frame_*.jpg")):
        frames.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video), "-vf", "scale=960:540",
                        "-vsync", "0", "-q:v", "2", "-start_number", "0", str(frames / "frame_%06d.jpg")], check=True)
    result = out / "localize" / route
    run_deploy_loop(DeployConfig(
        map_model=run / "artifacts/mapping/edm/model",
        keyframes=run / "artifacts/keyframes/keyframes.jsonl",
        frames_dir=frames,
        output_dir=result,
        localizer_config=run / "products/localization/localizer_config.json",
        localization_dir=run / "products/localization",
        edm_repo=Path(config["edm_root"]),
        edm_checkpoint=Path(config["edm_checkpoint"]),
        route=route,
        stream_fps=video_fps(video),
        tracker="klt", track_cap=500,
        reloc_min_points=60, reloc_period_s=0.3,
        megaloc_top_k=2, reloc_matcher="edm",
        vo=True, dead_reckon=True,
    ))
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    return {key: summary[key] for key in ("route", "dense_frames", "status_counts", "pose_coverage")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(localize(args.run, args.video, args.route, args.out), ensure_ascii=False))


if __name__ == "__main__":
    main()
