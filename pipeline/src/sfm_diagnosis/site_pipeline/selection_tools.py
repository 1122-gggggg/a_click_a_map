"""Fail-closed final-selection normalization and evidence-backed bridge repair."""

from __future__ import annotations

import hashlib
import json
import shutil
import statistics
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Mapping


def _component_sizes(nodes: set[str], pairs: Iterable[Mapping[str, Any]]) -> list[int]:
    adjacency: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        left, right = str(pair["image_i"]), str(pair["image_j"])
        if left not in nodes or right not in nodes:
            raise RuntimeError("admitted pair references an unselected keyframe")
        if left != right:
            adjacency[left].add(right)
            adjacency[right].add(left)
    remaining = set(nodes)
    sizes: list[int] = []
    while remaining:
        root = remaining.pop()
        queue = deque([root])
        size = 0
        while queue:
            current = queue.popleft()
            size += 1
            for neighbor in adjacency[current] & remaining:
                remaining.remove(neighbor)
                queue.append(neighbor)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def normalize_selection(
    selection: Mapping[str, Any],
    keyframe_to_segment: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remove zero-degree keyframes without changing the admitted pair set."""

    selected = {str(value) for value in selection.get("selected_keyframes") or ()}
    pairs = [dict(value) for value in selection.get("admitted_pairs") or ()]
    if not selected or not pairs:
        raise RuntimeError("final selection must contain keyframes and admitted pairs")
    referenced = {
        str(pair[field]) for pair in pairs for field in ("image_i", "image_j")
    }
    unknown = referenced - selected
    if unknown:
        raise RuntimeError(f"admitted pairs reference unselected keyframes: {sorted(unknown)}")
    removed = sorted(selected - referenced)
    retained = selected - set(removed)
    components = _component_sizes(retained, pairs)
    if components != [len(retained)]:
        raise RuntimeError(f"final pair graph has multiple nontrivial components: {components}")
    missing_metadata = retained - set(keyframe_to_segment)
    if missing_metadata:
        raise RuntimeError(f"keyframe metadata is missing: {sorted(missing_metadata)}")
    retained_segments = {str(keyframe_to_segment[value]) for value in retained}
    original_segments = {str(value) for value in selection.get("active_segments") or ()}
    normalized = dict(selection)
    normalized.update(
        active_segments=sorted(original_segments & retained_segments),
        selected_keyframes=sorted(retained),
        admitted_pairs=pairs,
        mapping_modes={
            str(segment): mode
            for segment, mode in dict(selection.get("mapping_modes") or {}).items()
            if str(segment) in retained_segments
        },
        selection_profile="V3_CORE_DROP_ZERO_DEGREE_KEYFRAMES",
    )
    receipt = {
        "artifact_type": "FINAL_SELECTION_ZERO_DEGREE_NORMALIZATION",
        "selected_before": len(selected),
        "selected_after": len(retained),
        "pairs_preserved": len(pairs),
        "removed_zero_degree_keyframes": removed,
        "removed_empty_segments": sorted(original_segments - retained_segments),
        "component_sizes_after": components,
    }
    return normalized, receipt


def build_enriched_selection(
    selection: Mapping[str, Any],
    keyframes: Iterable[Mapping[str, Any]],
    geometry: Iterable[Mapping[str, Any]],
    *,
    target_segments: set[str] | None = None,
    target_keyframes: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Add only requested frames and all induced VERIFIED pairs."""

    rows = [dict(row) for row in keyframes]
    by_id = {str(row["keyframe_id"]): row for row in rows}
    segments = set(map(str, target_segments or ()))
    explicit = set(map(str, target_keyframes or ()))
    unknown = explicit - set(by_id)
    if unknown:
        raise RuntimeError(f"unknown bridge keyframes: {sorted(unknown)}")
    additions = explicit | {
        identifier
        for identifier, row in by_id.items()
        if str(row.get("segment_id")) in segments
    }
    base = set(map(str, selection.get("selected_keyframes") or ()))
    additions -= base
    if not additions:
        raise RuntimeError("bridge enrichment selected no new keyframes")
    expanded = base | additions
    verified = [
        dict(row)
        for row in geometry
        if row.get("admission") == "VERIFIED"
        and str(row.get("image_i")) in expanded
        and str(row.get("image_j")) in expanded
    ]
    referenced = {
        str(row[field]) for row in verified for field in ("image_i", "image_j")
    }
    zero_degree = sorted(additions - referenced)
    if zero_degree:
        raise RuntimeError(f"bridge enrichment produced zero-degree frames: {zero_degree}")
    components = _component_sizes(expanded, verified)
    if components != [len(expanded)]:
        raise RuntimeError(f"bridge enrichment remains disconnected: {components}")
    added_segments = {str(by_id[value]["segment_id"]) for value in additions}
    bridge_pairs = [
        row
        for row in verified
        if (str(row["image_i"]) in additions) ^ (str(row["image_j"]) in additions)
    ]
    if not bridge_pairs:
        raise RuntimeError("bridge enrichment has no VERIFIED edge to the base selection")
    enriched = dict(selection)
    enriched.update(
        active_segments=sorted(
            set(map(str, selection.get("active_segments") or ())) | added_segments
        ),
        selected_keyframes=sorted(expanded),
        admitted_pairs=verified,
        mapping_modes={
            **{
                str(key): value
                for key, value in dict(selection.get("mapping_modes") or {}).items()
            },
            **{segment: "TRIANGULATE" for segment in added_segments},
        },
        selection_profile="ROBUST_COMPONENT_BRIDGE_ENRICHED",
    )
    receipt = {
        "artifact_type": "ROBUST_COMPONENT_BRIDGE_ENRICHMENT",
        "selected_before": len(base),
        "selected_after": len(expanded),
        "pairs_before": len(selection.get("admitted_pairs") or ()),
        "pairs_after": len(verified),
        "added_keyframes": sorted(additions),
        "added_segments": sorted(added_segments),
        "bridge_pair_count": len(bridge_pairs),
        "bridge_inliers_E_median": _median(bridge_pairs, "inliers_E"),
        "bridge_parallax_p10_deg_median": _median(bridge_pairs, "parallax_p10_deg"),
        "bridge_parallax_p50_deg_median": _median(bridge_pairs, "parallax_p50_deg"),
        "bridge_cheirality_median": _median(bridge_pairs, "cheirality_ratio"),
        "component_sizes_after": components,
    }
    return enriched, receipt


def normalize_run_selection(run_dir: str | Path) -> Path:
    """Normalize a pre-approval run and bind the decision to the new hashes."""

    run = Path(run_dir).expanduser().resolve(strict=True)
    if (run / "approvals/final.json").exists():
        raise RuntimeError("cannot normalize a selection after final approval")
    selection_path = run / "decisions/final_selection.json"
    decision_path = run / "decisions/final_build_decision.json"
    selection_backup = run / "decisions/final_selection.pre_normalization.json"
    decision_backup = run / "decisions/final_build_decision.pre_normalization.json"
    if selection_backup.exists() or decision_backup.exists():
        raise RuntimeError("normalization backup already exists")
    keyframe_rows = _read_jsonl(run / "artifacts/keyframes/keyframes.jsonl")
    keyframe_to_segment = {
        str(row["keyframe_id"]): str(row["segment_id"]) for row in keyframe_rows
    }
    normalized, receipt = normalize_selection(_read_json(selection_path), keyframe_to_segment)
    shutil.copy2(selection_path, selection_backup)
    shutil.copy2(decision_path, decision_backup)
    _write_json_atomic(selection_path, normalized)
    decision = _read_json(decision_path)
    decision.update(
        active_segments=normalized["active_segments"],
        final_selection=str(selection_path),
        input_hashes=_decision_input_hashes(run, selection_path),
        selection_profile=normalized["selection_profile"],
        selection_adjustments=receipt,
        approval_allowed=not bool(decision.get("issues")),
    )
    _write_json_atomic(decision_path, decision)
    receipt.update(
        schema_version=1,
        run=str(run),
        selection_sha256=_sha256(selection_path),
        decision_sha256=_sha256(decision_path),
        selection_backup=str(selection_backup),
        decision_backup=str(decision_backup),
    )
    receipt_path = run / "receipts/final_selection_normalization.json"
    _write_json_atomic(receipt_path, receipt)
    return receipt_path


def enrich_run_selection(
    run_dir: str | Path,
    *,
    target_segments: set[str],
    target_keyframes: set[str],
    attempt_name: str,
) -> Path:
    """Archive invalidated products, enrich selection, and invalidate approval."""

    if not attempt_name or Path(attempt_name).name != attempt_name or attempt_name in {".", ".."}:
        raise ValueError("attempt_name must be one safe path component")
    run = Path(run_dir).expanduser().resolve(strict=True)
    selection_path = run / "decisions/final_selection.json"
    decision_path = run / "decisions/final_build_decision.json"
    enriched, receipt = build_enriched_selection(
        _read_json(selection_path),
        _read_jsonl(run / "artifacts/keyframes/keyframes.jsonl"),
        _read_jsonl(run / "artifacts/pairs/geometry.jsonl"),
        target_segments=target_segments,
        target_keyframes=target_keyframes,
    )
    attempt = _archive_invalidated_attempt(run, attempt_name)
    _write_json_atomic(selection_path, enriched)
    decision = _read_json(decision_path)
    added_segments = receipt["added_segments"]
    decision.update(
        active_segments=enriched["active_segments"],
        final_selection=str(selection_path),
        input_hashes=_decision_input_hashes(run, selection_path),
        selection_profile=enriched["selection_profile"],
        robust_bridge_enrichment=receipt,
        approval_allowed=not bool(decision.get("issues")),
    )
    decision["attempted_reintroduced_segments"] = sorted(
        set(map(str, decision.get("attempted_reintroduced_segments") or ()))
        | set(added_segments)
    )
    decision["reintroduced_segments"] = sorted(
        set(map(str, decision.get("reintroduced_segments") or ())) | set(added_segments)
    )
    decision["reasons"] = list(decision.get("reasons") or ()) + [
        "ROBUST_COVISIBILITY_COMPONENT_REPAIR"
    ]
    _write_json_atomic(decision_path, decision)
    receipt.update(
        schema_version=1,
        run=str(run),
        archived_attempt=str(attempt),
        selection_sha256=_sha256(selection_path),
        decision_sha256=_sha256(decision_path),
    )
    receipt_path = run / "receipts/robust_bridge_enrichment.json"
    _write_json_atomic(receipt_path, receipt)
    return receipt_path


def _archive_invalidated_attempt(run: Path, name: str) -> Path:
    attempt = run / "artifacts/attempts" / name
    if attempt.exists():
        raise RuntimeError(f"attempt archive already exists: {attempt}")
    attempt.mkdir(parents=True)
    for relative in (
        "decisions/final_selection.json",
        "decisions/final_build_decision.json",
        "approvals/final.json",
        "pipeline_receipt.json",
    ):
        source = run / relative
        if source.is_file():
            target = attempt / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    invalidated = (
        "artifacts/mapping/final",
        "artifacts/mapping/robust",
        "artifacts/diagnosis/final_map.json",
        "artifacts/localization/robust",
        "artifacts/localization/dense",
        "artifacts/localization/ensemble",
        "products/base_geometry",
        "products/localization_dense",
        "products/localization_ensemble",
        "products/FINAL_RECEIPT.json",
        "receipts/stage12_final_mapping.json",
        "receipts/stage13_final_diagnosis.json",
        "receipts/stage14_localization_validation.json",
        "receipts/stage15_publish.json",
        "receipts/robust_filter.json",
        "receipts/ensemble_validation.json",
    )
    for relative in invalidated:
        source = run / relative
        if source.exists() or source.is_symlink():
            target = attempt / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
    approval = run / "approvals/final.json"
    if approval.exists():
        approval.unlink()
    return attempt


def _decision_input_hashes(run: Path, selection_path: Path) -> dict[str, str]:
    paths = {
        "final_selection": selection_path,
        "roles": run / "artifacts/selection/roles.jsonl",
        "keyframes": run / "artifacts/keyframes/keyframes.jsonl",
        "pair_geometry": run / "artifacts/pairs/geometry.jsonl",
        "diagnostic_selection": run / "artifacts/selection/diagnostic_selection.json",
        "metadata": run / "inputs/metadata.csv",
        "corpus_manifest": run / "inputs/corpus_manifest.json",
        "post_sfm_diagnosis": run / "artifacts/diagnosis/post_sfm.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"decision input is absent: {missing}")
    return {name: _sha256(path) for name, path in paths.items()}


def _median(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else float(statistics.median(values))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "build_enriched_selection",
    "enrich_run_selection",
    "normalize_run_selection",
    "normalize_selection",
]
