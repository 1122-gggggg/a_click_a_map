"""Production stage implementations for :class:`SitePipeline`."""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from sfm_diagnosis.io import load_gluemap, write_json

from .adapters import (
    AdapterRequest,
    CommandAdapter,
    ExclusiveResourceLease,
    probe_python_runtime,
    require_compute_capability,
)
from .domain import PreSfmRole, legacy_session_role, record_dict
from .graph import diagnose_graph
from .mapping_optimization import (
    MappingOptimizationContract,
    optimization_adapter_payload,
    validate_optimizer_result,
)
from .policy import Candidate, assign_roles, diagnostic_select, reinforce_bridges
from .post_sfm import diagnose_post_sfm, model_capabilities
from .preprocessing import adaptive_keyframes, sanitize_frames, split_segments


def build_default_handlers(config) -> dict[str, Callable]:
    return {
        "stage01_sanitization": sanitization_stage,
        "stage02_segment_keyframes": segment_keyframe_stage,
        "stage03_retrieval": _configured("retrieval"),
        "stage04_pair_geometry": _configured("pair_matcher"),
        "stage05_graph_build": graph_build_stage,
        "stage06_pre_sfm_diagnosis": pre_sfm_diagnosis_stage,
        "stage07_diagnostic_selection": diagnostic_selection_stage,
        "stage08_diagnostic_mapping": _configured("diagnostic_mapper"),
        "stage09_post_sfm_diagnosis": post_sfm_diagnosis_stage,
        "stage10_role_assignment": role_assignment_stage,
        "stage11_reinforcement": reinforcement_stage,
        "stage12_final_mapping": final_mapping_stage,
        "stage13_final_diagnosis": final_diagnosis_stage,
        "stage14_localization_validation": _configured("localizer"),
    }


def sanitization_stage(context):
    from .frame_analyzer import analyze_frames

    manifest = _read_json(context.run_dir / "inputs/corpus_manifest.json")
    metadata = _metadata_by_source(context.run_dir / "inputs/metadata.csv")
    rows: list[dict[str, Any]] = []
    mapping_sources = [
        source for source in manifest["sources"] if source.get("evaluation_role") == "MAPPING"
    ]
    mapping_tokens = {
        str(value)
        for source in mapping_sources
        for value in (
            source.get("source_id"),
            source.get("video_id"),
            source.get("path"),
            Path(str(source.get("path") or "")).name,
            Path(str(source.get("path") or "")).stem,
        )
        if value
    }
    adapter = context.config.adapters.get("frame_analyzer") or {}
    configured_evidence = adapter.get("existing_evidence")
    if configured_evidence:
        payload = _read_json_or_jsonl(Path(str(configured_evidence)))
        supplied = list(payload if isinstance(payload, list) else payload.get("frames") or ())
        rows = [
            dict(row, evaluation_role="MAPPING")
            for row in supplied
            if any(
                str(row.get(field) or "") in mapping_tokens
                for field in ("video_id", "video", "source_uri", "source_id", "session")
            )
        ]
    else:
        probe_fps = float(adapter.get("probe_fps", 2.0))
        sources = [
            source
            for source in manifest["sources"]
            if source.get("evaluation_role") == "MAPPING" and source.get("source_kind") != "archive"
        ]
        progress_log = context.run_dir / "logs/stage01_sanitization.progress.jsonl"
        progress_log.parent.mkdir(parents=True, exist_ok=True)
        progress_log.write_text("", encoding="utf-8")
        for source_index, source in enumerate(sources, 1):
            started = time.time()
            source_metadata = metadata.get(str(source["source_id"]), {})
            analyzed = analyze_frames(
                Path(str(source["path"])),
                video_id=str(source.get("video_id") or source["source_id"]),
                session_id=str(source_metadata.get("session_id") or ""),
                fps=probe_fps,
            )
            for row in analyzed:
                row["duplicate_previous"] = bool(row.get("near_duplicate"))
                row["evaluation_role"] = "MAPPING"
            rows.extend(analyzed)
            with progress_log.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "source": source.get("relative_path"),
                            "completed": source_index,
                            "total": len(sources),
                            "sampled_frames": len(analyzed),
                            "runtime_seconds": time.time() - started,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    result = sanitize_frames(rows)
    output = context.expected_outputs[0]
    _write_jsonl(
        output,
        [
            {
                **dict(record.source),
                "frame_id": record.frame_id,
                "status": record.status,
                "warnings": list(record.warnings),
            }
            for record in result.kept
        ]
        + [
            {
                **dict(record.source),
                "frame_id": record.frame_id,
                "status": "INACTIVE_REJECT",
                "rejection_reason": record.reason,
            }
            for record in result.removed
        ],
    )
    return _outcome(
        context,
        {
            "kept": len(result.kept),
            "removed": len(result.removed),
            "candidate_pool": len(result.candidate_pool),
        },
    )


def segment_keyframe_stage(context):
    from .frame_analyzer import extract_frames

    trusted_image_hashes = _trusted_keyframe_image_hashes(context.run_dir)
    rows = [
        row
        for row in _read_jsonl(context.run_dir / "artifacts/sanitization/frames.jsonl")
        if row.get("status") == "CANDIDATE" and row.get("evaluation_role", "MAPPING") == "MAPPING"
    ]
    if not rows:
        _blocked("no sanitized candidate frames are available")
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_video[str(row["video_id"])].append(row)
    segment_rows: list[dict[str, Any]] = []
    keyframe_rows: list[dict[str, Any]] = []
    settings = context.config.thresholds.get("keyframes") or {}
    stage2_progress = context.run_dir / "logs/stage02_keyframes.progress.jsonl"
    stage2_progress.parent.mkdir(parents=True, exist_ok=True)
    stage2_progress.write_text("", encoding="utf-8")
    for video_id, frames in sorted(by_video.items()):
        extraction_started = time.time()
        extraction_requests: list[dict[str, Any]] = []
        pending_hash_rows: list[dict[str, Any]] = []
        source_uris: set[str] = set()
        segments = split_segments(
            frames,
            max_gap=float(settings.get("max_gap_seconds", 1.0)),
            min_segment_frames=int(settings.get("min_segment_frames", 2)),
        )
        for segment in segments:
            segment_rows.append(
                {
                    "segment_id": segment.segment_id,
                    "video_id": segment.video_id,
                    "session_id": segment.session_id,
                    "start_seconds": segment.start_seconds,
                    "end_seconds": segment.end_seconds,
                    "frame_count": len(segment.frames),
                    "boundary_reasons": list(segment.boundary_reasons),
                    "data_status": "CANDIDATE",
                    "pre_sfm_role": "UNDECIDED",
                }
            )
            marked = []
            for index, frame in enumerate(segment.frames):
                source = dict(frame.source)
                source["segment_boundary"] = index in {0, len(segment.frames) - 1}
                marked.append(source)
            selected = adaptive_keyframes(
                marked,
                baseline_fps=float(settings.get("baseline_fps", 1.0)),
                fast_fps=float(settings.get("fast_fps", 2.0)),
                static_fps=float(settings.get("static_fps", 0.5)),
                min_gap=float(settings.get("min_gap_seconds", 0.5)),
                motion_threshold=float(settings.get("motion_threshold", 2.0)),
            )
            for frame in selected:
                image_root = context.run_dir / "artifacts/keyframes/images"
                relative_image = Path(video_id) / f"frame_{frame.frame_index:08d}.jpg"
                image_path = image_root / relative_image
                source_uri = str(frame.source.get("source_uri") or "")
                if not source_uri:
                    _blocked(f"keyframe {frame.frame_id} lacks source_uri")
                source_uris.add(source_uri)
                trusted_hash = trusted_image_hashes.get(str(image_path.resolve()))
                reusable = bool(
                    trusted_hash and image_path.is_file() and _sha256(image_path) == trusted_hash
                )
                if not reusable:
                    extraction_requests.append(
                        {
                            "source_frame_index": frame.frame_index,
                            "source_image_path": frame.source.get("source_image_path"),
                            "output": image_path,
                        }
                    )
                row = {
                    **dict(frame.source),
                    "keyframe_id": frame.frame_id,
                    "frame_id": frame.frame_id,
                    "segment_id": segment.segment_id,
                    "video_id": video_id,
                    "session_id": segment.session_id,
                    "source_pts_seconds": frame.timestamp,
                    "source_frame_index": frame.frame_index,
                    "image_uri": str(image_path),
                    "output_name": relative_image.as_posix(),
                    "status": "CANDIDATE",
                    "warnings": list(frame.warnings),
                }
                keyframe_rows.append(row)
                pending_hash_rows.append(row)
        if len(source_uris) != 1:
            _blocked(f"video {video_id} maps to multiple source URIs")
        if extraction_requests:
            extract_frames(next(iter(source_uris)), extraction_requests)
        for row in pending_hash_rows:
            image_path = Path(str(row["image_uri"]))
            if not image_path.is_file():
                _blocked(f"keyframe extraction did not produce {image_path}")
            row["image_sha256"] = _sha256(image_path)
        with stage2_progress.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "video_id": video_id,
                        "candidate_frames": len(frames),
                        "keyframes": len(pending_hash_rows),
                        "runtime_seconds": time.time() - extraction_started,
                    }
                )
                + "\n"
            )
    _write_jsonl(context.expected_outputs[0], segment_rows)
    _write_jsonl(context.expected_outputs[1], keyframe_rows)
    return _outcome(context, {"segments": len(segment_rows), "keyframes": len(keyframe_rows)})


def _trusted_keyframe_image_hashes(run_dir: Path) -> dict[str, str]:
    """Load per-image hashes only from a self-consistent prior Stage-2 receipt."""

    receipt = run_dir / "receipts/stage02_segment_keyframes.json"
    if not receipt.is_file():
        return {}
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
        referenced = payload.get("referenced_artifacts") or {}
        artifacts = referenced.get("artifacts")
        if not isinstance(artifacts, list) or referenced.get("count") != len(artifacts):
            return {}
        encoded = json.dumps(artifacts, sort_keys=True, separators=(",", ":"), default=str).encode()
        if hashlib.sha256(encoded).hexdigest() != referenced.get("fingerprint"):
            return {}
        trusted: dict[str, str] = {}
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                return {}
            path, digest = artifact.get("path"), artifact.get("sha256")
            if not path or not digest:
                return {}
            trusted[str(Path(str(path)).resolve())] = str(digest)
        return trusted
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {}


def graph_build_stage(context):
    keyframes = _read_jsonl(context.run_dir / "artifacts/keyframes/keyframes.jsonl")
    geometry = _read_jsonl(context.run_dir / "artifacts/pairs/geometry.jsonl")
    index = {
        str(row.get("keyframe_id") or row.get("frame_id") or row.get("image")): row
        for row in keyframes
    }
    verified = [row for row in geometry if row.get("admission") == "VERIFIED"]
    frame_edges: set[tuple[str, str]] = set()
    for row in verified:
        left, right = str(row.get("image_i")), str(row.get("image_j"))
        if left in index and right in index and left != right:
            frame_edges.add(tuple(sorted((left, right))))
    levels = {
        "frame": "keyframe_id",
        "segment": "segment_id",
        "video": "video_id",
        "session": "session_id",
    }
    payload: dict[str, Any] = {
        "schema_version": 2,
        "artifact_type": "PRE_SFM_GRAPH_BUNDLE",
        "excluded_candidate_pairs": len(geometry) - len(verified),
    }
    for level, field in levels.items():
        node_map = {
            identifier: str(row.get(field) or identifier) for identifier, row in index.items()
        }
        nodes = sorted(set(node_map.values()))
        edges = sorted(
            {
                tuple(sorted((node_map[left], node_map[right])))
                for left, right in frame_edges
                if node_map[left] != node_map[right]
            }
        )
        payload[level] = {
            "nodes": nodes,
            "edges": [list(edge) for edge in edges],
            "verified_pair_count": len(verified),
        }
    write_json(context.expected_outputs[0], payload)
    return _outcome(context, {"verified_pairs": len(verified), "frames": len(index)})


def pre_sfm_diagnosis_stage(context):
    graph_bundle = _read_json(context.run_dir / "artifacts/graphs/graph_bundle.json")
    segment = graph_bundle.get("segment") or {"nodes": [], "edges": []}
    diagnostics = diagnose_graph(segment.get("nodes") or (), segment.get("edges") or ())
    payload = {
        "schema_version": 2,
        "artifact_type": "PRE_SFM_GRAPH_DIAGNOSIS",
        "components": [list(component) for component in diagnostics.components],
        "largest_component_ratio": diagnostics.largest_component_ratio,
        "articulation_nodes": list(diagnostics.articulation_nodes),
        "bridges": [
            {
                "edge": list(edge),
                "importance": "HIGH",
                "risk": "HIGH",
                "replacement_paths": diagnostics.replacement_path_counts[edge],
                "cycle_support": diagnostics.cycle_support[edge],
            }
            for edge in diagnostics.bridges
        ],
        "edges": [
            {
                "edge": list(edge),
                "replacement_paths": diagnostics.replacement_path_counts[edge],
                "cycle_support": diagnostics.cycle_support[edge],
            }
            for edge in diagnostics.edges
        ],
    }
    write_json(context.expected_outputs[0], payload)
    return _outcome(
        context, {"components": len(diagnostics.components), "bridges": len(diagnostics.bridges)}
    )


def diagnostic_selection_stage(context):
    segments = _read_jsonl(context.run_dir / "artifacts/keyframes/segments.jsonl")
    keyframes = _read_jsonl(context.run_dir / "artifacts/keyframes/keyframes.jsonl")
    geometry = _read_jsonl(context.run_dir / "artifacts/pairs/geometry.jsonl")
    bundle = _read_json(context.run_dir / "artifacts/graphs/graph_bundle.json")
    diagnosis = _read_json(context.run_dir / "artifacts/diagnosis/pre_sfm.json")
    segment_edges = [tuple(edge) for edge in (bundle.get("segment") or {}).get("edges") or ()]
    neighbors: dict[str, set[str]] = defaultdict(set)
    for left, right in segment_edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    articulations = set(diagnosis.get("articulation_nodes") or ())
    bridge_nodes = {node for row in diagnosis.get("bridges") or () for node in row["edge"]}
    keyframe_count: dict[str, int] = defaultdict(int)
    for row in keyframes:
        keyframe_count[str(row["segment_id"])] += 1
    pair_by_segment: dict[str, list[dict[str, Any]]] = defaultdict(list)
    frame_to_segment = {str(row["keyframe_id"]): str(row["segment_id"]) for row in keyframes}
    for row in geometry:
        if row.get("admission") != "VERIFIED":
            continue
        for image in (str(row.get("image_i")), str(row.get("image_j"))):
            if image in frame_to_segment:
                pair_by_segment[frame_to_segment[image]].append(row)
    candidates: list[Candidate] = []
    required_coverage: set[str] = set()
    for row in segments:
        segment_id = str(row["segment_id"])
        session = str(row.get("session_id") or row.get("video_id") or segment_id)
        required_coverage.add(session)
        evidence = pair_by_segment.get(segment_id, [])
        geometry_score = (
            min(
                1.0,
                median([float(item.get("inliers_F") or 0) for item in evidence]) / 2000,
            )
            if evidence
            else 0.0
        )
        candidates.append(
            Candidate(
                id=segment_id,
                coverage={session},
                connections=neighbors.get(segment_id, set()),
                geometry=geometry_score,
                connectivity=float(len(neighbors.get(segment_id, ()))),
                cost=float(max(keyframe_count.get(segment_id, 1), 1)),
                verified_bridge=segment_id in bridge_nodes,
                articulation=segment_id in articulations,
                pre_sfm_role=(
                    PreSfmRole.BRIDGE_CANDIDATE
                    if segment_id in bridge_nodes or segment_id in articulations
                    else PreSfmRole.CORE_CANDIDATE
                ),
            )
        )
    budget = float(
        context.config.resources.get("diagnostic_keyframe_budget")
        or sum(candidate.cost for candidate in candidates)
    )
    target_cost = float(context.config.resources.get("diagnostic_keyframe_target") or 0)
    selection = diagnostic_select(
        candidates,
        required_coverage,
        budget,
        target_cost=target_cost,
    )
    selected_segments = {row.id for row in selection.selected}
    critical_segments = {
        row.id for row in candidates if row.verified_bridge or row.articulation or row.verified_loop
    }
    missing_critical = sorted(critical_segments - selected_segments)
    if selection.warnings or missing_critical:
        _blocked(
            "diagnostic selection hard constraints failed: "
            f"warnings={list(selection.warnings)}, missing_critical={missing_critical}"
        )
    selected_keyframes = [
        str(row["keyframe_id"]) for row in keyframes if str(row["segment_id"]) in selected_segments
    ]
    admitted_pairs = [
        row
        for row in geometry
        if row.get("admission") == "VERIFIED"
        and str(row.get("image_i")) in selected_keyframes
        and str(row.get("image_j")) in selected_keyframes
    ]
    write_json(
        context.expected_outputs[0],
        {
            "schema_version": 2,
            "artifact_type": "DIAGNOSTIC_SELECTION",
            "selected_segments": sorted(selected_segments),
            "selected_keyframes": sorted(selected_keyframes),
            "admitted_pairs": admitted_pairs,
            "warnings": list(selection.warnings),
            "budget": budget,
            "target_cost": target_cost,
            "cost": selection.cost,
        },
    )
    return _outcome(
        context, {"segments": len(selected_segments), "keyframes": len(selected_keyframes)}
    )


def post_sfm_diagnosis_stage(context):
    configured = context.config.adapters.get("post_sfm")
    if configured:
        return _run_adapter(context, "post_sfm")
    model = context.run_dir / "artifacts/mapping/diagnostic/model"
    outcome = _diagnose_model(context, model, context.expected_outputs[0])
    loo_results = _diagnostic_bridge_loo(context, model)
    from .consistency import pair_rotation_cycles

    keyframes = _read_jsonl(context.run_dir / "artifacts/keyframes/keyframes.jsonl")
    geometry = _read_jsonl(context.run_dir / "artifacts/pairs/geometry.jsonl")
    rotation_cycles = pair_rotation_cycles(keyframes, geometry)
    keyframe_segments = {str(row["keyframe_id"]): str(row["segment_id"]) for row in keyframes}
    inconsistent_segments = {
        keyframe_segments[node]
        for cycle in rotation_cycles
        if cycle["rotation_residual_deg"] > 5.0
        for node in cycle["nodes"]
        if node in keyframe_segments
    }
    segment_path = context.run_dir / "artifacts/diagnosis/post_sfm_segments.jsonl"
    segment_rows = _read_jsonl(segment_path)
    by_target = {str(row["target_id"]): row for row in loo_results if row.get("kind") == "segment"}
    for row in segment_rows:
        result = by_target.get(str(row["segment_id"]))
        if result is None:
            row["alignment_evaluated"] = not (row.get("verified_bridge") or row.get("articulation"))
            continue
        warnings = list(result.get("warnings") or ())
        row.update(
            alignment_evaluated=result.get("status") == "OK",
            alignment_consistent=result.get("status") == "OK" and not warnings,
            unstable=bool(warnings),
            loo_rotation_deg=result.get("rotation_p90_deg"),
            loo_position_normalized=result.get("position_p90_normalized"),
            loo_status=result.get("status"),
        )
        if str(row["segment_id"]) in inconsistent_segments:
            row["alignment_evaluated"] = True
            row["alignment_consistent"] = False
            row["unstable"] = True
            row.setdefault("warnings", []).append("CYCLE_INCONSISTENT")
    _write_jsonl(segment_path, segment_rows)
    payload = _read_json(context.expected_outputs[0])
    payload["bridge_loo"] = loo_results
    payload["pair_rotation_cycles"] = rotation_cycles
    if inconsistent_segments:
        payload.setdefault("warnings", []).append("CYCLE_INCONSISTENT")
    write_json(context.expected_outputs[0], payload)
    return _outcome(
        context,
        {
            **dict(outcome.details),
            "bridge_loo_targets": len(loo_results),
            "bridge_loo_completed": sum(row.get("status") == "OK" for row in loo_results),
        },
    )


def _diagnostic_bridge_loo(context, model):
    from .loo import LOOTarget

    diagnosis = _read_json(context.run_dir / "artifacts/diagnosis/pre_sfm.json")
    bridge_segments = {
        str(node) for row in diagnosis.get("bridges") or () for node in row.get("edge") or ()
    } | {str(node) for node in diagnosis.get("articulation_nodes") or ()}
    if not bridge_segments:
        return []
    selection = _read_json(context.run_dir / "artifacts/selection/diagnostic_selection.json")
    selected_segments = set(selection.get("selected_segments") or ())
    pseudo_roles = [
        {
            "segment_id": segment,
            "post_sfm_role": "BRIDGE" if segment in bridge_segments else "CORE",
            "risk": "HIGH" if segment in bridge_segments else "LOW",
        }
        for segment in sorted(selected_segments)
    ]
    keyframes = _read_jsonl(context.run_dir / "artifacts/keyframes/keyframes.jsonl")
    targets = tuple(
        LOOTarget("segment", segment, "diagnostic_bridge_stability")
        for segment in sorted(bridge_segments & selected_segments)
    )
    return _execute_tiered_loo(context, model, targets, keyframes, pseudo_roles)


def final_diagnosis_stage(context):
    configured = context.config.adapters.get("final_diagnosis")
    if configured:
        _blocked("external final_diagnosis cannot replace the built-in release gates")
    robust_model = context.run_dir / "artifacts/mapping/robust/model"
    robust_receipt = context.run_dir / "receipts/robust_filter.json"
    if robust_model.exists() and not robust_receipt.is_file():
        _blocked("robust model exists without its filter receipt")
    model = (
        robust_model
        if robust_receipt.is_file()
        else context.run_dir / "artifacts/mapping/final/model"
    )
    outcome = _diagnose_model(context, model, context.expected_outputs[0])
    from .consistency import loo_alignment_modes, pair_rotation_cycles
    from .loo import tiered_loo_targets

    keyframes = _read_jsonl(context.run_dir / "artifacts/keyframes/keyframes.jsonl")
    roles = _read_jsonl(context.run_dir / "artifacts/selection/roles.jsonl")
    targets = tiered_loo_targets(keyframes, roles)
    loo_results = _execute_tiered_loo(context, model, targets, keyframes, roles)
    geometry = _read_jsonl(context.run_dir / "artifacts/pairs/geometry.jsonl")
    rotation_cycles = pair_rotation_cycles(keyframes, geometry)
    alignment_modes = loo_alignment_modes(loo_results)
    payload = _read_json(context.expected_outputs[0])
    payload["tiered_loo"] = loo_results
    payload["pair_rotation_cycles"] = rotation_cycles
    payload["submap_alignment_modes"] = alignment_modes
    payload.setdefault("warnings", [])
    if any(
        row.get("warnings") or row.get("status") not in {"OK", "COMPLETED"} for row in loo_results
    ):
        payload["warnings"].append("LOO_STABILITY_WARNING")
    if any(row["rotation_residual_deg"] > 5.0 for row in rotation_cycles):
        payload["warnings"].append("CYCLE_INCONSISTENT")
    if alignment_modes["multimodal_alignment"]:
        payload["warnings"].append("MULTIMODAL_ALIGNMENT")
    write_json(context.expected_outputs[0], payload)
    return _outcome(
        context,
        {
            **dict(outcome.details),
            "loo_targets": len(targets),
            "loo_completed": sum(row.get("status") == "OK" for row in loo_results),
        },
    )


def _execute_tiered_loo(context, reference_model, targets, keyframes, roles):
    from .loo import aligned_camera_stability, loo_warning

    if context.config.resources.get("tiered_loo_enabled", True) is False:
        return [
            {
                "kind": target.kind,
                "target_id": target.target_id,
                "reason": target.reason,
                "status": "NOT_EVALUATED_DISABLED_BY_CONFIG",
                "warnings": ["LOO_NOT_EVALUATED"],
            }
            for target in targets
        ]
    adapter_config = dict(
        context.config.adapters.get("loo_mapper")
        or context.config.adapters.get("diagnostic_mapper")
        or {}
    )
    if not adapter_config:
        return [
            {
                "kind": target.kind,
                "target_id": target.target_id,
                "reason": target.reason,
                "status": "NOT_EVALUATED_ADAPTER_MISSING",
            }
            for target in targets
        ]
    reference = _camera_pose_map(reference_model)
    active_segments = {
        str(row.get("segment_id") or row.get("segment"))
        for row in roles
        if row.get("post_sfm_role") in {"CORE", "BRIDGE"}
    }
    geometry = _read_jsonl(context.run_dir / "artifacts/pairs/geometry.jsonl")
    results = []
    for target in targets:
        selected = [
            row
            for row in keyframes
            if str(row.get("segment_id")) in active_segments
            and not (
                (target.kind == "video" and str(row.get("video_id")) == target.target_id)
                or (target.kind == "segment" and str(row.get("segment_id")) == target.target_id)
            )
        ]
        selected_ids = {str(row["keyframe_id"]) for row in selected}
        pairs = [
            row
            for row in geometry
            if row.get("admission") == "VERIFIED"
            and str(row.get("image_i")) in selected_ids
            and str(row.get("image_j")) in selected_ids
        ]
        safe = hashlib.sha256(f"{target.kind}:{target.target_id}".encode()).hexdigest()[:12]
        root = context.run_dir / "artifacts/diagnosis/loo" / f"{target.kind}_{safe}"
        selection_path = root / "selection.json"
        model_path = root / "model"
        write_json(
            selection_path,
            {
                "schema_version": 2,
                "artifact_type": "LOO_DIAGNOSTIC_SELECTION",
                "selected_keyframes": sorted(selected_ids),
                "admitted_pairs": pairs,
                "excluded": {"kind": target.kind, "id": target.target_id},
            },
        )
        request_payload = {
            "run_root": str(context.run_dir),
            "mode": "diagnostic",
            "keyframes": str(context.run_dir / "artifacts/keyframes/keyframes.jsonl"),
            "selection": str(selection_path),
            "pair_geometry": str(context.run_dir / "artifacts/pairs/geometry.jsonl"),
            "output_model": str(model_path),
        }
        try:
            _run_custom_adapter(context, adapter_config, request_payload, (model_path,))
            metrics = aligned_camera_stability(reference, _camera_pose_map(model_path))
            results.append(
                {
                    "kind": target.kind,
                    "target_id": target.target_id,
                    "reason": target.reason,
                    **metrics,
                    "warnings": list(loo_warning(metrics)),
                }
            )
        except Exception as error:
            results.append(
                {
                    "kind": target.kind,
                    "target_id": target.target_id,
                    "reason": target.reason,
                    "status": "FAILED",
                    "error": str(error),
                    "warnings": ["LOO_RESOLVE_FAILED"],
                }
            )
    return results


def _camera_pose_map(model: Path) -> dict[str, dict[str, Any]]:
    data = load_gluemap(model)
    return {
        name: {"center": center.tolist(), "rotation": rotation.tolist()}
        for name, center, rotation in zip(
            data.image_names, data.image_centers, data.image_R_wc, strict=True
        )
    }


def _diagnosis_keyframes(context, keyframes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selection_path = (
        context.run_dir / "artifacts/selection/diagnostic_selection.json"
        if context.stage_name == "stage09_post_sfm_diagnosis"
        else context.run_dir / "decisions/final_selection.json"
        if context.stage_name == "stage13_final_diagnosis"
        else None
    )
    if selection_path is None or not selection_path.is_file():
        return keyframes
    selection = _read_json(selection_path)
    selected_ids = {str(value) for value in selection.get("selected_keyframes") or ()}
    return [row for row in keyframes if str(row.get("keyframe_id")) in selected_ids]


def _diagnose_model(context, model: Path, output: Path):
    try:
        map_data = load_gluemap(model)
    except (FileNotFoundError, RuntimeError) as error:
        _blocked(str(error))
    keyframes = _diagnosis_keyframes(
        context,
        _read_jsonl(context.run_dir / "artifacts/keyframes/keyframes.jsonl"),
    )
    names = _keyframe_name_index(keyframes)
    registered_names = set(map_data.image_names)
    cameras: list[dict[str, Any]] = []
    image_id_to_segment: dict[int, str] = {}
    for image_id, image_name, center in zip(
        map_data.image_ids, map_data.image_names, map_data.image_centers, strict=True
    ):
        row = names.get(image_name) or names.get(Path(image_name).name) or {}
        segment_id = str(row.get("segment_id") or "unknown")
        image_id_to_segment[int(image_id)] = segment_id
        cameras.append(
            {
                "image_id": str(int(image_id)),
                "image_name": image_name,
                "segment_id": segment_id,
                "registered": True,
                "center": np.asarray(center).tolist(),
            }
        )
    for row in keyframes:
        name = str(row.get("image_uri") or row.get("output_name") or row.get("keyframe_id"))
        if name not in registered_names and Path(name).name not in {
            Path(item).name for item in registered_names
        }:
            cameras.append(
                {
                    "image_id": f"unregistered:{row['keyframe_id']}",
                    "image_name": name,
                    "segment_id": str(row.get("segment_id") or "unknown"),
                    "registered": False,
                    "center": None,
                }
            )
    landmarks = (
        {
            "landmark_id": str(int(point_id)),
            "xyz": np.asarray(xyz).tolist(),
            "observations": [
                {
                    "image_id": str(int(image_id)),
                    "reprojection_error": float(error),
                }
                for image_id in track_ids
                if int(image_id) in image_id_to_segment
            ],
        }
        for point_id, xyz, error, track_ids in zip(
        map_data.point_ids,
        map_data.points_xyz,
        map_data.point_errors,
        map_data.track_image_ids,
        strict=True,
        )
    )
    fim, segment_fim = _pose_observability_fim(map_data, image_id_to_segment)
    diagnosis = diagnose_post_sfm(cameras, landmarks, fim=fim)
    evidence_level = str(
        (context.config.adapters.get("diagnostic_mapper") or {}).get("evidence_level")
        or "coarse_pose"
    )
    capabilities = model_capabilities(
        evidence_level=evidence_level,
        registered_images=len(map_data.image_ids),
        total_images=len(cameras),
        landmarks=len(map_data.point_ids),
        observations=sum(len(image_ids) for image_ids in map_data.track_image_ids),
    )
    diagnosis_warnings = list(diagnosis.warnings)
    if not capabilities["role_assignment_ready"]:
        diagnosis_warnings.append("INSUFFICIENT_MAPPING_EVIDENCE")
    payload = diagnosis.as_dict()
    payload.update(
        {
            "schema_version": 2,
            "artifact_type": "POST_SFM_DIAGNOSIS",
            "model": str(model),
            "model_capabilities": capabilities,
            "warnings": sorted(set(diagnosis_warnings)),
        }
    )
    write_json(output, payload)
    segment_rows = []
    segment_manifest = _read_jsonl(context.run_dir / "artifacts/keyframes/segments.jsonl")
    segment_metadata = {str(row["segment_id"]): row for row in segment_manifest}
    pre_roles = {
        segment_id: str(row.get("pre_sfm_role") or "UNDECIDED")
        for segment_id, row in segment_metadata.items()
    }
    pre_graph = _read_json(context.run_dir / "artifacts/diagnosis/pre_sfm.json")
    articulation = set(pre_graph.get("articulation_nodes") or ())
    bridge_nodes = {node for edge in pre_graph.get("bridges") or () for node in edge["edge"]}
    graph_bundle = _read_json(context.run_dir / "artifacts/graphs/graph_bundle.json")
    connections: dict[str, set[str]] = defaultdict(set)
    for left, right in (graph_bundle.get("segment") or {}).get("edges") or ():
        connections[str(left)].add(str(right))
        connections[str(right)].add(str(left))
    for segment_id, metrics in diagnosis.segment_metrics.items():
        segment_rows.append(
            {
                "segment_id": segment_id,
                "video_id": segment_metadata.get(segment_id, {}).get("video_id"),
                "session_id": segment_metadata.get(segment_id, {}).get("session_id"),
                "pre_sfm_role": pre_roles.get(segment_id, "UNDECIDED"),
                "registered_ratio": metrics.registration_ratio,
                "median_track_length": metrics.median_track_length,
                "geometry": min(1.0, metrics.landmark_count / 1000),
                "verified_bridge": segment_id in bridge_nodes,
                "articulation": segment_id in articulation,
                "connections": sorted(connections.get(segment_id, ())),
                "connectivity": float(len(connections.get(segment_id, ()))),
                "alignment_consistent": True,
                "unstable": False,
                "parallax": diagnosis.triangulation_angle_deg.get("p10") or 0.0,
                "fim_condition": segment_fim.get(segment_id, diagnosis.fim.condition_number),
                "warnings": sorted(set(diagnosis_warnings)),
            }
        )
    _write_jsonl(context.run_dir / "artifacts/diagnosis/post_sfm_segments.jsonl", segment_rows)
    return _outcome(
        context,
        {
            "registered_ratio": diagnosis.registration_ratio,
            "landmarks": len(map_data.point_ids),
            "model_capabilities": capabilities,
        },
    )


def _pose_observability_fim(map_data, image_id_to_segment):
    from sfm_diagnosis.fisher import compute_fisher_metrics, weighted_bearing_fim

    point_indexes_by_image: dict[int, list[int]] = defaultdict(list)
    for point_index, image_ids in enumerate(map_data.track_image_ids):
        for image_id in image_ids:
            point_indexes_by_image[int(image_id)].append(point_index)
    global_fim = np.zeros((6, 6), dtype=float)
    segment_matrices: dict[str, np.ndarray] = defaultdict(lambda: np.zeros((6, 6), dtype=float))
    for image_id, center, rotation_wc in zip(
        map_data.image_ids,
        map_data.image_centers,
        map_data.image_R_wc,
        strict=True,
    ):
        indexes = np.asarray(point_indexes_by_image.get(int(image_id), ()), dtype=int)
        if not len(indexes):
            continue
        world_vectors = map_data.points_xyz[indexes] - np.asarray(center)
        camera_points = (np.asarray(rotation_wc).T @ world_vectors.T).T
        weights = 1.0 / (1.0 + np.square(map_data.point_errors[indexes]))
        matrix = weighted_bearing_fim(camera_points, weights)
        global_fim += matrix
        segment_matrices[str(image_id_to_segment.get(int(image_id), "unknown"))] += matrix
    return global_fim, {
        segment: compute_fisher_metrics(matrix).condition_number
        for segment, matrix in segment_matrices.items()
    }


def role_assignment_stage(context):
    path = context.run_dir / "artifacts/diagnosis/post_sfm_segments.jsonl"
    if not path.is_file():
        payload = _read_json(context.run_dir / "artifacts/diagnosis/post_sfm.json")
        rows = list(payload.get("segments") or ())
    else:
        rows = _read_jsonl(path)
    candidates = [_candidate_from_row(row) for row in rows]
    decisions = assign_roles(candidates)
    output_rows = []
    for row, candidate in zip(rows, candidates, strict=True):
        decision = decisions[candidate.id]
        assignment = decision.assignment(candidate)
        legacy_role = (
            "QUARANTINE"
            if assignment.post_sfm_role is None
            else legacy_session_role(assignment.post_sfm_role)
        )
        output_rows.append(
            {
                **record_dict(assignment),
                "video": row.get("video_id") or row.get("video") or "",
                "segment": candidate.id,
                "base_map": assignment.base_map.value,
                "localization": assignment.localization.value,
                "reason": ";".join(assignment.reasons),
                "legacy_session_role": legacy_role,
                "session_id": row.get("session_id"),
                "video_id": row.get("video_id"),
            }
        )
    _write_jsonl(context.expected_outputs[0], output_rows)
    by_session: dict[str, list[str]] = defaultdict(list)
    for row in output_rows:
        session_id = str(row.get("session_id") or row.get("video_id") or "unknown")
        by_session[session_id].append(str(row["legacy_session_role"]))
    precedence = [
        "BASE_CORE",
        "BASE_SUPPORT",
        "UPDATE_CANDIDATE",
        "APPEARANCE_REF",
        "NEW_SUBMAP",
        "QUARANTINE",
        "REJECT",
    ]
    write_json(
        context.run_dir / "artifacts/selection/legacy_session_roles.json",
        {
            "schema_version": 1,
            "artifact_type": "LEGACY_SESSION_ROLE_PROJECTION",
            "sessions": [
                {
                    "session_id": session_id,
                    "role": next(role for role in precedence if role in set(session_roles)),
                    "segment_roles": session_roles,
                }
                for session_id, session_roles in sorted(by_session.items())
            ],
        },
    )
    return _outcome(context, {"assignments": len(output_rows)})


def reinforcement_stage(context):
    post_path = context.run_dir / "artifacts/diagnosis/post_sfm_segments.jsonl"
    post_rows = _read_jsonl(post_path) if post_path.is_file() else []
    segment_inventory = _read_jsonl(context.run_dir / "artifacts/keyframes/segments.jsonl")
    keyframes = _read_jsonl(context.run_dir / "artifacts/keyframes/keyframes.jsonl")
    geometry = _read_jsonl(context.run_dir / "artifacts/pairs/geometry.jsonl")
    bundle = _read_json(context.run_dir / "artifacts/graphs/graph_bundle.json")
    connections: dict[str, set[str]] = defaultdict(set)
    for left, right in (bundle.get("segment") or {}).get("edges") or ():
        connections[str(left)].add(str(right))
        connections[str(right)].add(str(left))
    post_by_segment = {str(row["segment_id"]): row for row in post_rows}
    role_path = context.run_dir / "artifacts/selection/roles.jsonl"
    role_rows = _read_jsonl(role_path) if role_path.is_file() else []
    candidate_rows = []
    for segment in segment_inventory:
        segment_id = str(segment["segment_id"])
        candidate_rows.append(
            {
                **dict(segment),
                **dict(post_by_segment.get(segment_id) or {}),
                "segment_id": segment_id,
                "connections": sorted(connections.get(segment_id, ())),
                "connectivity": float(len(connections.get(segment_id, ()))),
            }
        )
    candidates = [_candidate_from_row(row) for row in candidate_rows]
    active_ids = {
        str(row.get("segment_id") or row.get("segment"))
        for row in role_rows
        if row.get("post_sfm_role") in {"CORE", "BRIDGE"}
    }
    bridges = [
        row for row in candidates if row.verified_bridge or row.articulation or row.verified_loop
    ]
    pool = [row for row in candidates if row.id not in active_ids and not row.rejected]
    result = reinforce_bridges(bridges, pool, max_rounds=context.config.max_reinforcement_rounds)
    history: list[dict[str, Any]] = []
    added_ids: list[str] = []
    validated_added_ids: list[str] = []
    remaining_pool = list(pool)
    round_index = 0
    while result.added and round_index < context.config.max_reinforcement_rounds:
        round_index += 1
        selected = result.added[0]
        added_ids.append(selected.id)
        remaining_pool = [row for row in remaining_pool if row.id != selected.id]
        try:
            round_payload = _run_reinforcement_round(context, round_index, selected.id)
            history.append(round_payload)
        except Exception as error:
            history.append(
                {
                    "round": round_index,
                    "segment_id": selected.id,
                    "status": "FAILED",
                    "error": str(error),
                }
            )
            break
        role_rows = _read_jsonl(role_path)
        active_ids = {
            str(row.get("segment_id") or row.get("segment"))
            for row in role_rows
            if row.get("post_sfm_role") in {"CORE", "BRIDGE"}
        }
        if selected.id in active_ids:
            validated_added_ids.append(selected.id)
        else:
            history[-1]["status"] = "NOT_ADMITTED"
            history[-1]["reason"] = "reinforcement segment failed final role validation"
            break
        refreshed_rows = _read_jsonl(post_path)
        post_rows = refreshed_rows
        post_by_segment = {str(row["segment_id"]): row for row in refreshed_rows}
        refreshed = [_candidate_from_row(row) for row in refreshed_rows]
        bridges = [
            row for row in refreshed if row.verified_bridge or row.articulation or row.verified_loop
        ]
        result = reinforce_bridges(
            bridges,
            remaining_pool,
            max_rounds=context.config.max_reinforcement_rounds - round_index,
        )
    issues: list[str] = []
    if any(row.get("status") == "FAILED" for row in history):
        issues.append("MISSING_CONNECTIVITY")
    if set(added_ids) != set(validated_added_ids):
        issues.append("MISSING_CONNECTIVITY")
    if result.disposition == "RESHOOT_REQUIRED":
        issues.append("RESHOOT_REQUIRED")
    if any(not row.alignment_consistent or row.unstable for row in bridges):
        issues.append("MISSING_CONNECTIVITY")
    role_rows = _read_jsonl(role_path)
    role_by_segment = {str(row.get("segment_id") or row.get("segment")): row for row in role_rows}
    active_ids = {
        segment_id
        for segment_id, row in role_by_segment.items()
        if row.get("post_sfm_role") in {"CORE", "BRIDGE"}
    }
    active_ids.update(validated_added_ids)
    active_keyframes = {
        str(row["keyframe_id"]) for row in keyframes if str(row.get("segment_id")) in active_ids
    }
    admitted_pairs = [
        row
        for row in geometry
        if row.get("admission") == "VERIFIED"
        and str(row.get("image_i")) in active_keyframes
        and str(row.get("image_j")) in active_keyframes
    ]
    if not active_ids:
        issues.append("NO_ACTIVE_SEGMENTS")
    if not active_keyframes:
        issues.append("NO_ACTIVE_KEYFRAMES")
    if not admitted_pairs:
        issues.append("NO_VERIFIED_FINAL_PAIRS")
    approval_allowed = not issues
    final_selection = {
        "schema_version": 2,
        "artifact_type": "FINAL_MAPPING_SELECTION",
        "active_segments": sorted(active_ids),
        "selected_keyframes": sorted(active_keyframes),
        "admitted_pairs": admitted_pairs,
        "mapping_modes": {
            segment_id: row.get("mapping_mode")
            for segment_id, row in sorted(role_by_segment.items())
            if segment_id in active_ids
        },
    }
    write_json(context.expected_outputs[1], final_selection)
    input_paths = {
        "final_selection": context.expected_outputs[1],
        "roles": role_path,
        "keyframes": context.run_dir / "artifacts/keyframes/keyframes.jsonl",
        "pair_geometry": context.run_dir / "artifacts/pairs/geometry.jsonl",
        "diagnostic_selection": context.run_dir / "artifacts/selection/diagnostic_selection.json",
        "metadata": context.run_dir / "inputs/metadata.csv",
        "corpus_manifest": context.run_dir / "inputs/corpus_manifest.json",
        "post_sfm_diagnosis": context.run_dir / "artifacts/diagnosis/post_sfm.json",
    }
    input_hashes = {name: _sha256(path) for name, path in input_paths.items() if path.is_file()}
    decision = {
        "schema_version": 2,
        "artifact_type": "FINAL_BUILD_DECISION",
        "approval_required": True,
        "approval_allowed": approval_allowed,
        "max_reinforcement_rounds": context.config.max_reinforcement_rounds,
        "reinforcement_rounds": round_index,
        "reinforcement_history": history,
        "attempted_reintroduced_segments": added_ids,
        "reintroduced_segments": validated_added_ids,
        "active_segments": sorted(active_ids),
        "issues": sorted(set(issues)),
        "reasons": list(result.reasons),
        "final_selection": str(context.expected_outputs[1]),
        "input_hashes": input_hashes,
    }
    write_json(context.expected_outputs[0], decision)
    keyframes_by_segment: dict[str, list[str]] = defaultdict(list)
    for row in keyframes:
        keyframes_by_segment[str(row.get("segment_id"))].append(str(row["keyframe_id"]))
    candidate_pool_rows = []
    for row in candidate_rows:
        segment_id = str(row["segment_id"])
        role = role_by_segment.get(segment_id, {})
        if segment_id in active_ids or role.get("post_sfm_role") == "REJECT":
            continue
        candidate_pool_rows.append(
            {
                **row,
                "keyframe_ids": sorted(keyframes_by_segment.get(segment_id, ())),
                "post_sfm_role": role.get("post_sfm_role"),
                "candidate_pool_reason": (
                    "diagnostic_not_selected"
                    if segment_id not in post_by_segment
                    else "inactive_but_retained"
                ),
            }
        )
    _write_jsonl(
        context.expected_outputs[2],
        candidate_pool_rows,
    )
    write_json(
        context.expected_outputs[3],
        {
            "schema_version": 2,
            "artifact_type": "WEAK_REGION_RESHOOT_PLAN",
            "required": bool(issues),
            "issues": sorted(set(issues)),
            "recommendations": (
                ["add alternate bridge", "increase lateral baseline", "add viewpoint diversity"]
                if issues
                else []
            ),
        },
    )
    return _outcome(context, {"approval_allowed": approval_allowed, "issues": issues})


def _run_reinforcement_round(context, round_index: int, segment_id: str) -> dict[str, Any]:
    from .pipeline import StageContext

    mapper_config = dict(context.config.adapters.get("diagnostic_mapper") or {})
    if not mapper_config:
        raise RuntimeError("automatic reinforcement requires diagnostic_mapper")
    base_selection_path = context.run_dir / "artifacts/selection/diagnostic_selection.json"
    selection = _read_json(base_selection_path)
    keyframes = _read_jsonl(context.run_dir / "artifacts/keyframes/keyframes.jsonl")
    geometry = _read_jsonl(context.run_dir / "artifacts/pairs/geometry.jsonl")
    selected_segments = set(selection.get("selected_segments") or ()) | {segment_id}
    selected_keyframes = {
        str(row["keyframe_id"])
        for row in keyframes
        if str(row.get("segment_id")) in selected_segments
    }
    pairs = [
        row
        for row in geometry
        if row.get("admission") == "VERIFIED"
        and str(row.get("image_i")) in selected_keyframes
        and str(row.get("image_j")) in selected_keyframes
    ]
    root = context.run_dir / "artifacts/reinforcement" / f"round_{round_index:02d}"
    selection_path = root / "selection.json"
    model_path = root / "model"
    diagnosis_path = root / "post_sfm.json"
    write_json(
        selection_path,
        {
            **selection,
            "artifact_type": "REINFORCEMENT_DIAGNOSTIC_SELECTION",
            "selected_segments": sorted(selected_segments),
            "selected_keyframes": sorted(selected_keyframes),
            "admitted_pairs": pairs,
            "reinforcement_round": round_index,
            "reintroduced_segment": segment_id,
        },
    )
    _run_custom_adapter(
        context,
        mapper_config,
        {
            "run_root": str(context.run_dir),
            "mode": "diagnostic",
            "keyframes": str(context.run_dir / "artifacts/keyframes/keyframes.jsonl"),
            "selection": str(selection_path),
            "pair_geometry": str(context.run_dir / "artifacts/pairs/geometry.jsonl"),
            "output_model": str(model_path),
        },
        (model_path,),
    )
    diagnosis_context = StageContext(
        context.run_dir,
        f"stage11_reinforcement_round_{round_index}",
        context.config,
        (diagnosis_path,),
        context.fingerprint,
        context.inputs,
    )
    _diagnose_model(diagnosis_context, model_path, diagnosis_path)
    role_context = StageContext(
        context.run_dir,
        f"stage11_role_round_{round_index}",
        context.config,
        (context.run_dir / "artifacts/selection/roles.jsonl",),
        context.fingerprint,
        context.inputs,
    )
    role_assignment_stage(role_context)
    return {
        "round": round_index,
        "segment_id": segment_id,
        "status": "COMPLETED",
        "selection": str(selection_path),
        "model": str(model_path),
        "diagnosis": str(diagnosis_path),
    }


def _candidate_from_row(row: Mapping[str, Any]) -> Candidate:
    fields = Candidate.__dataclass_fields__
    values = {key: row[key] for key in fields if key in row}
    values["id"] = str(row.get("segment_id") or row.get("id"))
    role = row.get("pre_sfm_role")
    if role:
        values["pre_sfm_role"] = PreSfmRole(str(role))
    for key in ("coverage", "connections"):
        if key in values:
            values[key] = set(values[key] or ())
    for key in ("warnings", "hard_failures"):
        if key in values:
            values[key] = tuple(values[key] or ())
    return Candidate(**values)


def _configured(adapter_name: str):
    def handler(context):
        return _run_adapter(context, adapter_name)

    return handler


def final_mapping_stage(context):
    contract = MappingOptimizationContract.from_mapping(context.config.mapping_optimization)
    if not contract.enabled:
        return _run_adapter(context, "final_mapper")
    dense_model = context.run_dir / "artifacts/mapping/optimization/source/model"
    dense_context = replace(context, expected_outputs=(dense_model,))
    mapper_outcome = _run_adapter(dense_context, "final_mapper")
    optimizer_outcome = _run_mapping_optimizer(context, dense_model, contract)
    return _outcome(
        context,
        {
            **dict(optimizer_outcome.details),
            "optimization_enabled": True,
            "dense_model": str(dense_model),
            "baseline_mapper": dict(mapper_outcome.details),
        },
    )


def _run_mapping_optimizer(context, dense_model: Path, contract: MappingOptimizationContract):
    config = dict(context.config.adapters.get("mapping_optimizer") or {})
    if not config:
        _blocked(
            "mapping optimization is enabled but adapter 'mapping_optimizer' is not configured"
        )
    command = config.get("command")
    if not command:
        _blocked("adapter 'mapping_optimizer' requires a command")
    expanded = [
        str(value).replace("{run_dir}", str(context.run_dir)).replace("{stage}", context.stage_name)
        for value in command
    ]
    resource_class = str(config.get("resource_class") or "ba")
    payload = optimization_adapter_payload(
        context.run_dir,
        dense_model,
        context.expected_outputs[0],
        contract,
    )
    payload["recipe"] = dict(context.config.mapping_optimization)
    payload["final_mapper"] = dict(context.config.adapters.get("final_mapper") or {})
    inputs = tuple(
        str(path)
        for path in (
            dense_model,
            context.run_dir / "decisions/final_selection.json",
            context.run_dir / "artifacts/selection/roles.jsonl",
            context.run_dir / "artifacts/keyframes/keyframes.jsonl",
            context.run_dir / "artifacts/pairs/geometry.jsonl",
            context.run_dir / "inputs/corpus_manifest.json",
        )
    )
    request = AdapterRequest(
        "stage12_mapping_optimization",
        payload=payload,
        config=config,
        input_paths=inputs,
        output_dir=str(Path(payload["optimization_root"])),
        resource_class=resource_class,
    )
    adapter = CommandAdapter(
        expanded,
        cwd=(
            str(config["cwd"]).replace("{run_dir}", str(context.run_dir))
            if config.get("cwd")
            else None
        ),
        env={str(key): str(value) for key, value in dict(config.get("env") or {}).items()},
        timeout_seconds=(
            None if config.get("timeout_seconds") is None else float(config["timeout_seconds"])
        ),
    )
    with ExclusiveResourceLease(context.run_dir / "locks/heavy.lock", resource_class):
        receipt = adapter.run(request)
    winner = validate_optimizer_result(receipt.output, contract)
    output_model = context.expected_outputs[0]
    if not output_model.exists() and not output_model.is_symlink():
        _link_or_copy(Path(str(winner["model"])).expanduser().resolve(), output_model)
    required = (
        output_model,
        Path(str(payload["comparison"])),
        Path(str(payload["summary"])),
    )
    absent = [str(path) for path in required if not path.exists()]
    if absent:
        raise RuntimeError(f"mapping optimizer did not produce contracted outputs: {absent}")
    return _outcome(
        context,
        {
            "adapter": "mapping_optimizer",
            "request_fingerprint": receipt.request_fingerprint,
            "command": list(receipt.command),
            "winner": winner,
            "methods_completed": list(receipt.output.get("methods_completed") or ()),
            "comparison": str(payload["comparison"]),
            "summary": str(payload["summary"]),
        },
    )


def _run_adapter(context, adapter_name: str):
    config = dict(context.config.adapters.get(adapter_name) or {})
    if not config:
        _blocked(f"adapter {adapter_name!r} is not configured")
    existing = config.get("existing_artifact") or config.get("existing_model")
    if existing:
        sources = existing if isinstance(existing, list) else [existing]
        if len(sources) != len(context.expected_outputs):
            if len(context.expected_outputs) == 1:
                sources = [existing]
            else:
                _blocked(
                    f"adapter {adapter_name!r} existing artifacts do not match output contract"
                )
        for source, output in zip(sources, context.expected_outputs, strict=True):
            _link_or_copy(Path(str(source)).expanduser().resolve(), output)
        return _outcome(context, {"adapter": adapter_name, "mode": "existing_artifact"})
    command = config.get("command")
    if not command:
        _blocked(f"adapter {adapter_name!r} requires command or existing_artifact")
    expanded = [
        str(value).replace("{run_dir}", str(context.run_dir)).replace("{stage}", context.stage_name)
        for value in command
    ]
    resource_class = str(config.get("resource_class") or "cpu")
    preflight = None
    required_capability = context.config.resources.get("require_compute_capability")
    if resource_class in {"gpu_heavy", "ba", "gluemap", "matcher"}:
        preflight = probe_python_runtime(expanded[0])
        if required_capability:
            require_compute_capability(preflight, str(required_capability))
    request = AdapterRequest(
        context.stage_name,
        payload=_adapter_payload(context, adapter_name),
        config=config,
        input_paths=tuple(str(path) for path in context.inputs),
        output_dir=str(context.expected_outputs[0].parent),
        resource_class=resource_class,
    )
    adapter = CommandAdapter(
        expanded,
        cwd=(
            str(config["cwd"]).replace("{run_dir}", str(context.run_dir))
            if config.get("cwd")
            else None
        ),
        env={str(key): str(value) for key, value in dict(config.get("env") or {}).items()},
        timeout_seconds=(
            None if config.get("timeout_seconds") is None else float(config["timeout_seconds"])
        ),
    )
    with ExclusiveResourceLease(context.run_dir / "locks/heavy.lock", resource_class):
        receipt = adapter.run(request)
    _materialize_receipt_outputs(receipt.output, context.expected_outputs)
    details = {
        "adapter": adapter_name,
        "request_fingerprint": receipt.request_fingerprint,
        "command": list(receipt.command),
        "runtime": dict(receipt.runtime),
        "preflight": preflight,
    }
    for key in (
        "workspace_identity",
        "workspace",
        "pair_count",
        "exact_pair_proof",
        "model_capabilities",
        "intrinsics_seed",
        "images_are_undistorted",
    ):
        if key in receipt.output:
            details[key] = receipt.output[key]
    return _outcome(
        context,
        details,
    )


def _run_custom_adapter(context, config, payload, expected_outputs):
    command = config.get("command")
    if not command:
        raise RuntimeError("LOO mapper requires a command adapter")
    expanded = [
        str(value).replace("{run_dir}", str(context.run_dir)).replace("{stage}", "stage13_loo")
        for value in command
    ]
    resource_class = str(config.get("resource_class") or "gluemap")
    preflight = probe_python_runtime(expanded[0])
    required_capability = context.config.resources.get("require_compute_capability")
    if required_capability:
        require_compute_capability(preflight, str(required_capability))
    request = AdapterRequest(
        "stage13_loo",
        payload=payload,
        config=config,
        input_paths=tuple(str(path) for path in context.inputs),
        output_dir=str(Path(expected_outputs[0]).parent),
        resource_class=resource_class,
    )
    adapter = CommandAdapter(
        expanded,
        cwd=(
            str(config["cwd"]).replace("{run_dir}", str(context.run_dir))
            if config.get("cwd")
            else None
        ),
        env={str(key): str(value) for key, value in dict(config.get("env") or {}).items()},
        timeout_seconds=(
            None if config.get("timeout_seconds") is None else float(config["timeout_seconds"])
        ),
    )
    with ExclusiveResourceLease(context.run_dir / "locks/heavy.lock", resource_class):
        receipt = adapter.run(request)
    _materialize_receipt_outputs(receipt.output, expected_outputs)
    absent = [str(path) for path in expected_outputs if not Path(path).exists()]
    if absent:
        raise RuntimeError(f"LOO mapper did not produce outputs: {absent}")


def _adapter_payload(context, adapter_name: str) -> dict[str, Any]:
    run = context.run_dir
    payload: dict[str, Any] = {
        "expected_outputs": [str(path) for path in context.expected_outputs],
        "run_root": str(run),
        "corpus_manifest": str(run / "inputs/corpus_manifest.json"),
    }
    if adapter_name == "retrieval":
        payload.update(
            keyframes=str(run / "artifacts/keyframes/keyframes.jsonl"),
            images_root=str(run / "artifacts/keyframes/images"),
            output_candidates=str(context.expected_outputs[0]),
        )
    elif adapter_name == "pair_matcher":
        payload.update(
            candidate_pairs=str(run / "artifacts/retrieval/candidates.jsonl"),
            keyframes=str(run / "artifacts/keyframes/keyframes.jsonl"),
            metadata=str(run / "inputs/metadata.csv"),
            output_geometry=str(context.expected_outputs[0]),
            match_artifact_dir=str(run / "artifacts/pairs/matches"),
        )
    elif adapter_name == "diagnostic_mapper":
        payload.update(
            mode="diagnostic",
            keyframes=str(run / "artifacts/keyframes/keyframes.jsonl"),
            selection=str(run / "artifacts/selection/diagnostic_selection.json"),
            pair_geometry=str(run / "artifacts/pairs/geometry.jsonl"),
            output_model=str(context.expected_outputs[0]),
        )
    elif adapter_name == "final_mapper":
        payload.update(
            mode="final",
            keyframes=str(run / "artifacts/keyframes/keyframes.jsonl"),
            selection=str(run / "decisions/final_selection.json"),
            roles=str(run / "artifacts/selection/roles.jsonl"),
            pair_geometry=str(run / "artifacts/pairs/geometry.jsonl"),
            output_model=str(context.expected_outputs[0]),
        )
    elif adapter_name in {"post_sfm", "final_diagnosis"}:
        payload.update(
            model=str(
                run
                / (
                    "artifacts/mapping/final/model"
                    if adapter_name == "final_diagnosis"
                    else "artifacts/mapping/diagnostic/model"
                )
            ),
            keyframes=str(run / "artifacts/keyframes/keyframes.jsonl"),
            output=str(context.expected_outputs[0]),
        )
    elif adapter_name == "localizer":
        payload.update(
            model=str(run / "artifacts/mapping/final/model"),
            corpus=str(run / "inputs/corpus_manifest.json"),
            metadata=str(run / "inputs/metadata.csv"),
            roles=str(run / "artifacts/selection/roles.jsonl"),
            output_validation=str(context.expected_outputs[0]),
            output_references=str(context.expected_outputs[1]),
        )
    return payload


def _materialize_receipt_outputs(payload: Mapping[str, Any], expected: Iterable[Path]) -> None:
    reported = payload.get("outputs")
    if not reported:
        return
    sources = list(reported.values()) if isinstance(reported, Mapping) else list(reported)
    expected_rows = list(expected)
    if len(sources) != len(expected_rows):
        raise RuntimeError("adapter output count does not match stage contract")
    for source, output in zip(sources, expected_rows, strict=True):
        source_path = Path(str(source)).expanduser().resolve()
        if source_path != output.resolve():
            _link_or_copy(source_path, output)


def _link_or_copy(source: Path, output: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        if output.resolve() == source.resolve():
            return
        raise RuntimeError(f"refusing to replace existing stage output {output}")
    try:
        output.symlink_to(source, target_is_directory=source.is_dir())
    except OSError:
        if source.is_dir():
            shutil.copytree(source, output)
        else:
            shutil.copy2(source, output)


def _keyframe_name_index(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        for field in ("image_uri", "output_name", "keyframe_id", "frame_id"):
            value = row.get(field)
            if value:
                result[str(value)] = row
                result[Path(str(value)).name] = row
    return result


def _metadata_by_source(path: Path) -> dict[str, dict[str, str]]:
    import csv

    with path.open(encoding="utf-8") as stream:
        return {str(row["source_id"]): row for row in csv.DictReader(stream)}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _read_json_or_jsonl(path: Path):
    if path.suffix.lower() == ".jsonl":
        return _read_jsonl(path)
    return _read_json(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(dict(row), ensure_ascii=False, default=_json_default) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _outcome(context, details: Mapping[str, Any]):
    from .pipeline import StageOutcome

    return StageOutcome(context.expected_outputs, details)


def _blocked(reason: str):
    from .pipeline import StageBlocked

    raise StageBlocked(reason)


__all__ = [
    "build_default_handlers",
    "diagnostic_selection_stage",
    "final_diagnosis_stage",
    "graph_build_stage",
    "post_sfm_diagnosis_stage",
    "pre_sfm_diagnosis_stage",
    "reinforcement_stage",
    "role_assignment_stage",
    "sanitization_stage",
    "segment_keyframe_stage",
]
