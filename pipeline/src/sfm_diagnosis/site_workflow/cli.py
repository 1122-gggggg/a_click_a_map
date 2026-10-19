from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .config import WorkflowConfig
from .runner import SiteWorkflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="site-sfm-workflow",
        description=(
            "Run reproducible multi-video selection, mapping, map QA, EDM LOO, "
            "and localization-risk diagnosis for any site."
        ),
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--videos", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--from-stage")
    parser.add_argument("--to-stage")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = SiteWorkflow(WorkflowConfig.from_toml(args.config)).run(
        args.videos,
        args.output,
        from_stage=args.from_stage,
        to_stage=args.to_stage,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "run_dir": str(result.run_dir),
                "receipt": str(result.receipt),
                "stages": [
                    {"name": stage.name, "status": stage.status}
                    for stage in result.stages
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
