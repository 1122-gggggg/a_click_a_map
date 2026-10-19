from __future__ import annotations

import argparse
import json
from pathlib import Path

from mapdoctor.adapters import get_adapter

from .audit import audit_cells, audit_source_repository
from .bridge import enrich_pose_cells_from_model
from .compute import compute_metric_bundle
from .io import load_pose_cells, write_plan
from .planner import plan_regions
from .profiles import load_config
from .types import PoseDirectionCell, normalize_localizer


def _add_localizer_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--localizer",
        "--localizer-backend",
        "--backend",
        dest="localizer",
        default="unspecified",
        help="Arbitrary localizer provenance label; it does not select diagnosis logic",
    )


def _add_map_context_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Optional sparse map used only to attach producer provenance",
    )
    parser.add_argument(
        "--map-adapter",
        "--map-backend",
        dest="map_backend",
        default=None,
        help="Map input adapter for --model: built-in name or package.module:AdapterClass",
    )


def _enrich_from_map(
    cells: tuple[PoseDirectionCell, ...],
    args: argparse.Namespace,
) -> tuple[PoseDirectionCell, ...]:
    if args.model is None:
        return cells
    if args.map_backend is None:
        raise ValueError("--map-adapter is required when --model is supplied")
    model = get_adapter(args.map_backend).load(args.model)
    return enrich_pose_cells_from_model(cells, model)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mapdoctor-recapture",
        description="Metric-audited weak-region repair and targeted recapture planner",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    audit_parser = sub.add_parser("audit-metrics")
    audit_parser.add_argument("input")
    _add_localizer_argument(audit_parser)
    _add_map_context_arguments(audit_parser)

    source_parser = sub.add_parser("audit-source")
    source_parser.add_argument("repository")
    _add_localizer_argument(source_parser)

    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("input")
    _add_localizer_argument(plan_parser)
    _add_map_context_arguments(plan_parser)
    plan_parser.add_argument("--config")
    plan_parser.add_argument("--output-dir", default="outputs/recapture_plan")

    compute_parser = sub.add_parser("compute-metrics")
    compute_parser.add_argument("input")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    localizer = normalize_localizer(getattr(args, "localizer", "unspecified"))

    if args.cmd == "audit-source":
        print(json.dumps(audit_source_repository(args.repository, localizer).as_dict(), indent=2))
        return 0
    if args.cmd == "compute-metrics":
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        result = {key: value.as_dict() for key, value in compute_metric_bundle(payload).items()}
        print(json.dumps(result, indent=2))
        return 0

    cells = _enrich_from_map(load_pose_cells(args.input, localizer=localizer), args)
    if args.cmd == "audit-metrics":
        print(json.dumps(audit_cells(cells, localizer).as_dict(), indent=2))
        return 0

    thresholds, capture = load_config(args.config)
    decisions, audits = plan_regions(
        cells,
        localizer,
        thresholds=thresholds,
        capture=capture,
    )
    paths = write_plan(decisions, args.output_dir, audits=audits)
    print(f"JSON: {paths['json']}")
    print(f"HTML: {paths['html']}")
    print(f"CSV: {paths['decisions_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
