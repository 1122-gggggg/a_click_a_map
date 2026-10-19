#!/usr/bin/env python3
"""EDM + KLT two-rate localization against this package's map (after install.py).

    python3 run_localize.py --video query.MP4 --route myflight --output out/
    python3 run_localize.py --frames dir_of_960x540_frames --fps 29.97 --route myflight --output out/

Frames must be 960x540 JPEGs named frame_000000.jpg ... (what `--video` extracts with
`ffmpeg -vf scale=960:540 -vsync 0 -q:v 2 -start_number 0`). The query camera must be
the same camera/calibration the map was built with (River PINHOLE K, 1280x720 reference,
scaled per resolution; see localization/intrinsics.json).
Settings equal the validated run: KLT fast loop (track_cap 500), MegaLoc top-2 + EDM +
PnP every 0.3 s, lift radius 4 px, VO + dead reckoning on.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
sys.path[:0] = [str(PKG / 'code/normal_map_pipeline_src'), str(PKG / 'code/river_map_quality_src')]


def video_fps(video: Path) -> float:
    out = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=avg_frame_rate', '-of', 'csv=p=0', str(video)], text=True).strip()
    num, den = out.split('/')
    return float(num) / float(den)


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--video', type=Path)
    source.add_argument('--frames', type=Path)
    parser.add_argument('--fps', type=float, help='required with --frames')
    parser.add_argument('--route', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    loc = PKG / 'localization'
    config = json.loads((loc / 'bank/localizer_config.json').read_text(encoding='utf-8'))
    if args.video:
        frames = args.output / 'frames'
        if not any(frames.glob('frame_*.jpg')):
            frames.mkdir(parents=True, exist_ok=True)
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(args.video), '-vf', 'scale=960:540', '-vsync', '0', '-q:v', '2', '-start_number', '0', str(frames / 'frame_%06d.jpg')], check=True)
        fps = video_fps(args.video)
    else:
        if not args.fps:
            parser.error('--fps is required with --frames')
        frames, fps = args.frames, args.fps

    from sfm_diagnosis.site_pipeline.deploy_loop import DeployConfig, run_deploy_loop

    run_deploy_loop(DeployConfig(
        map_model=loc / 'edm_model',
        keyframes=loc / 'keyframes/keyframes.jsonl',
        frames_dir=frames,
        output_dir=args.output / 'localize' / args.route,
        localizer_config=loc / 'bank/localizer_config.json',
        localization_dir=loc / 'bank',
        edm_repo=Path(config['edm_root']),
        edm_checkpoint=Path(config['edm_checkpoint']),
        route=args.route,
        stream_fps=fps,
        tracker='klt', track_cap=500,
        reloc_min_points=60, reloc_period_s=0.3,
        megaloc_top_k=2, reloc_matcher='edm',
        vo=True, dead_reckon=True,
    ))
    summary = json.loads((args.output / 'localize' / args.route / 'summary.json').read_text(encoding='utf-8'))
    print(json.dumps({k: summary[k] for k in ('route', 'dense_frames', 'status_counts', 'pose_coverage')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
