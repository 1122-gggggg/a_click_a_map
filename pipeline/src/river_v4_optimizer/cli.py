from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import analyze_model
from .cleanup import prune_observation_free_registered_images
from .runner import atomic_json, retriangulate_fixed_poses, run_filter_sweep
from .rescue import rescue_connectors
from .selection import rewire_selection


def main() -> None:
    parser = argparse.ArgumentParser(prog="river-v4-optimize")
    commands = parser.add_subparsers(dest="command", required=True)

    sweep = commands.add_parser("sweep")
    sweep.add_argument("--input-model", type=Path, required=True)
    sweep.add_argument("--output-root", type=Path, required=True)

    analyze = commands.add_parser("analyze")
    analyze.add_argument("--model", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)

    retriangulate = commands.add_parser("retriangulate")
    retriangulate.add_argument("--input-model", type=Path, required=True)
    retriangulate.add_argument("--database", type=Path, required=True)
    retriangulate.add_argument("--image-root", type=Path, required=True)
    retriangulate.add_argument("--output-model", type=Path, required=True)
    retriangulate.add_argument("--minimum-angle-deg", type=float, required=True)
    retriangulate.add_argument("--include-two-view-tracks", action="store_true")

    rewire = commands.add_parser("rewire-selection")
    rewire.add_argument("--run", type=Path, required=True)
    rewire.add_argument("--output", type=Path, required=True)
    rewire.add_argument("--remove", action="append", default=[])
    rewire.add_argument("--add", action="append", default=[])

    rescue = commands.add_parser("rescue-connectors")
    rescue.add_argument("--dense-model", type=Path, required=True)
    rescue.add_argument("--base-model", type=Path, required=True)
    rescue.add_argument("--output-model", type=Path, required=True)

    prune = commands.add_parser("prune-observation-free")
    prune.add_argument("--input-model", type=Path, required=True)
    prune.add_argument("--output-model", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "sweep":
        result = run_filter_sweep(args.input_model, args.output_root)
    elif args.command == "analyze":
        result = analyze_model(args.model)
        atomic_json(args.output, result)
    elif args.command == "retriangulate":
        result = retriangulate_fixed_poses(
            input_model=args.input_model,
            database=args.database,
            image_root=args.image_root,
            output_model=args.output_model,
            minimum_angle_deg=args.minimum_angle_deg,
            ignore_two_view_tracks=not args.include_two_view_tracks,
        )
    elif args.command == "rescue-connectors":
        result = rescue_connectors(
            dense_model=args.dense_model,
            base_model=args.base_model,
            output_model=args.output_model,
        )
    elif args.command == "prune-observation-free":
        result = prune_observation_free_registered_images(
            args.input_model,
            args.output_model,
        )
    else:
        run = args.run.resolve(strict=True)
        selection = json.loads((run / "decisions/final_selection.json").read_text())
        keyframes = {
            row["keyframe_id"]: row
            for row in map(
                json.loads,
                (run / "artifacts/keyframes/keyframes.jsonl").read_text().splitlines(),
            )
        }
        geometry = [
            json.loads(line)
            for line in (run / "artifacts/pairs/geometry.jsonl").read_text().splitlines()
            if line
        ]
        result, receipt = rewire_selection(
            selection,
            keyframes,
            geometry,
            remove=set(args.remove),
            add=set(args.add),
        )
        atomic_json(args.output, result)
        atomic_json(args.output.with_suffix(".receipt.json"), receipt)
        result = receipt
    print(json.dumps(result, indent=2, ensure_ascii=False))
