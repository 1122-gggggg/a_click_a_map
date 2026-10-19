"""CLI for ``sfm-qa analyze`` / ``check`` / ``check-map`` / ``check-localize`` / ``select-sessions``."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .pipeline import analyze, is_success_status

_ROLES_FALLBACK = (
    "BASE_CORE",
    "BASE_SUPPORT",
    "APPEARANCE_REF",
    "GEOMETRY_REINFORCEMENT",
    "UPDATE_CANDIDATE",
    "NEW_SUBMAP",
    "QUARANTINE",
    "REJECT",
    "VALIDATION_ONLY",
)


def _add_common(parser: argparse.ArgumentParser, *, logs_required: bool) -> None:
    parser.add_argument("model", type=Path, help="Sparse model directory, e.g. sparse/0")
    parser.add_argument(
        "--map-adapter",
        "--backend",
        dest="backend",
        required=True,
        help=(
            "Map input adapter: a built-in name or "
            "package.module:AdapterClass"
        ),
    )
    parser.add_argument(
        "--logs",
        type=Path,
        required=logs_required,
        help="MapDoctor-schema localization CSV (query,success,inliers,...)",
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional MapDoctor JSON settings")
    parser.add_argument("--output", type=Path, default=None, help="Write reports into this directory")
    parser.add_argument("--database", type=Path, default=None, help="Optional COLMAP database for build evidence")
    parser.add_argument("--pairs", type=Path, default=None, help="Optional pair table CSV/JSON/JSONL")
    parser.add_argument(
        "--images-manifest",
        type=Path,
        default=None,
        help="Optional registered-image metadata CSV/JSON",
    )
    parser.add_argument("--images-dir", type=Path, default=None, help="Optional image directory for quality evidence")


def _add_select_sessions(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--videos", type=Path, required=True, help="Directory of input videos")
    parser.add_argument("--output", type=Path, required=True, help="Write Stage 0 selection reports here")
    parser.add_argument("--maps", type=Path, default=None, help="Optional existing maps directory")
    parser.add_argument(
        "--vpr-candidates",
        type=Path,
        default=None,
        help=(
            "Optional JSON session-pair retrieval candidates. Used only to rank "
            "pre-build geometry verification; never treated as a geometric edge."
        ),
    )
    parser.add_argument(
        "--edge-probes",
        type=Path,
        default=None,
        help=(
            "Optional exact-pair geometry probe JSON. Shared-map/VPR records "
            "in this file are ignored."
        ),
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional session_select YAML overlay")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sfm-qa",
        description="Diagnose an SfM map first, then optionally attribute localization failures.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    analyze_p = sub.add_parser("analyze", help="Stage 1 map diagnosis, then Stage 2 if --logs is given")
    _add_common(analyze_p, logs_required=False)
    check_p = sub.add_parser("check", help="Alias for analyze")
    _add_common(check_p, logs_required=False)
    map_p = sub.add_parser("check-map", help="Screen the reconstruction only")
    _add_common(map_p, logs_required=False)
    loc_p = sub.add_parser("check-localize", help="Screen the map and require localization logs")
    _add_common(loc_p, logs_required=True)
    select_p = sub.add_parser(
        "select-sessions",
        help="Stage 0: propose videos, then assign verified multi-session roles",
    )
    _add_select_sessions(select_p)
    return parser


def _role_names() -> tuple[str, ...]:
    try:
        from .session_select import ROLES
    except ImportError:
        return _ROLES_FALLBACK
    if not ROLES:
        return _ROLES_FALLBACK
    return tuple(str(name) for name in ROLES)


def _assignment_role(item: Any) -> str | None:
    if item is None:
        return None
    if isinstance(item, str):
        return item
    if isinstance(item, Mapping):
        for key in ("role", "assigned_role"):
            value = item.get(key)
            if value:
                return str(value)
        return None
    value = getattr(item, "role", None)
    return str(value) if value else None


def _iter_role_values(report: Any) -> list[str]:
    if report is None:
        return []
    if isinstance(report, Mapping):
        counts = report.get("role_counts")
        if isinstance(counts, Mapping):
            expanded: list[str] = []
            for name, count in counts.items():
                try:
                    n = int(count)
                except (TypeError, ValueError):
                    continue
                expanded.extend([str(name)] * max(n, 0))
            if expanded:
                return expanded
        for key in ("roles", "assignments", "rows", "sessions"):
            value = report.get(key)
            if isinstance(value, Mapping):
                found = [role for role in (_assignment_role(item) for item in value.values()) if role]
                if found:
                    return found
            if isinstance(value, (list, tuple)):
                found = [role for role in (_assignment_role(item) for item in value) if role]
                if found:
                    return found
        return []
    if isinstance(report, (list, tuple)):
        return [role for role in (_assignment_role(item) for item in report) if role]
    return []


def _print_select_summary(report: Any, output_dir: Path) -> None:
    print("=== Stage 0A: pre-build proposal ===")
    if isinstance(report, Mapping):
        prebuild = report.get("prebuild") or {}
        if isinstance(prebuild, Mapping):
            print(f"proposal_confidence: {prebuild.get('proposal_confidence', 'NONE')}")
            print(
                "proposed_for_geometry: "
                + ", ".join(str(item) for item in (prebuild.get("proposed_base_sessions") or ()))
            )
            print(
                "reserved_validation: "
                + ", ".join(str(item) for item in (prebuild.get("validation_candidates") or ()))
            )
            print(f"geometry_pair_queue: {len(prebuild.get('verification_pairs') or ())}")
    print("=== Stage 0B: verified role assignment ===")
    counted = Counter(_iter_role_values(report))
    for role in _role_names():
        print(f"{role}: {counted.get(role, 0)}")
    extras = sorted(name for name in counted if name not in set(_role_names()))
    for role in extras:
        print(f"{role}: {counted[role]}")
    if not counted:
        print("(no role assignments in report)")
    print(f"report: {Path(output_dir)}")


def _run_select_sessions(args: argparse.Namespace) -> int:
    from .session_select.run import select_sessions

    report = select_sessions(
        video_dir=args.videos,
        output_dir=args.output,
        config=args.config,
        maps_dir=args.maps,
        vpr_candidates=args.vpr_candidates,
        edge_probes=getattr(args, "edge_probes", None),
    )
    _print_select_summary(report, args.output)
    return 0


def _print_summary(report: dict, output_dir: Path | None, logs_given: bool) -> None:
    mapping = report["map"]
    readiness = mapping["readiness"]
    reconstruction = mapping.get("reconstruction") or {}
    loc = report["localization"]
    print("=== Stage 1: map diagnosis ===")
    print(
        f"map: grade={readiness['grade']} ok={readiness['map_ok']} status={readiness['map_status']}"
    )
    mode = reconstruction.get("diagnostic_mode")
    if mode is not None:
        print(f"diagnostic_mode: {mode}")
    print(
        f"weak_regions: {reconstruction.get('num_weak_regions')} "
        f"weak_images: {reconstruction.get('num_weak_images')}"
    )
    print("=== Stage 2: SfM localization ===")
    if not logs_given or loc is None:
        print("(skipped: no --logs)")
    else:
        print(
            f"localization: ok={loc['localization_ok']} "
            f"rate={loc['strict_success_rate']:.3f} "
            f"status={loc['localization_status']}"
        )
        counts = loc.get("attribution_counts") or {}
        pretty = ", ".join(f"{name}={count}" for name, count in counts.items()) or "(none)"
        print(f"attribution: {pretty}")
    print(f"=== overall_status: {report['overall_status']} ===")
    if output_dir is not None:
        print(f"report: {Path(output_dir) / 'report.json'}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "select-sessions":
        return _run_select_sessions(args)
    logs_path = None if args.command == "check-map" else args.logs
    if args.command == "check-localize" and logs_path is None:
        raise SystemExit("check-localize requires --logs")
    report = analyze(
        args.model,
        backend=args.backend,
        logs_path=logs_path,
        config_path=args.config,
        output_dir=args.output,
        database=args.database,
        pairs=args.pairs,
        images_manifest=args.images_manifest,
        images_dir=args.images_dir,
    )
    _print_summary(report, args.output, logs_given=logs_path is not None)
    return 0 if is_success_status(report["overall_status"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
