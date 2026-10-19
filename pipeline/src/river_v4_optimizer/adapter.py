from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from .integration import OptimizationRecipe, prunable_loser_models, select_winner
from .metrics import rank_variants
from .rescue import rescue_connectors
from .runner import atomic_json, retriangulate_fixed_poses, run_filter_sweep
from .selection import rewire_selection


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _normalized_filter_variants(recipe: OptimizationRecipe) -> tuple[dict[str, Any], ...]:
    result = []
    for row in recipe.filter_variants:
        normalized = dict(row)
        if "minimum_triangulation_angle_deg" not in normalized:
            normalized["minimum_triangulation_angle_deg"] = normalized.pop("minimum_angle_deg")
        result.append(normalized)
    return tuple(result)


def _mapper_request(
    config: Mapping[str, Any],
    *,
    run_root: Path,
    keyframes: Path,
    selection: Path,
    roles: Path,
    pair_geometry: Path,
    corpus_manifest: Path | None,
    output_model: Path,
) -> dict[str, Any]:
    payload = {
        "run_root": str(run_root),
        "mode": "final",
        "keyframes": str(keyframes),
        "selection": str(selection),
        "roles": str(roles),
        "pair_geometry": str(pair_geometry),
        "output_model": str(output_model),
        "expected_outputs": [str(output_model)],
    }
    input_paths = [str(keyframes), str(selection), str(roles), str(pair_geometry)]
    if corpus_manifest is not None:
        payload["corpus_manifest"] = str(corpus_manifest)
        input_paths.append(str(corpus_manifest))
    return {
        "stage": "stage12_mapping_optimization_rewire",
        "payload": payload,
        "config": dict(config),
        "input_paths": input_paths,
        "output_dir": str(output_model.parent),
        "resource_class": str(config.get("resource_class") or "gluemap"),
    }


def _run_mapper(
    config: Mapping[str, Any],
    *,
    run_root: Path,
    keyframes: Path,
    selection: Path,
    roles: Path,
    pair_geometry: Path,
    corpus_manifest: Path | None,
    output_model: Path,
) -> dict[str, Any]:
    command = [
        str(value)
        .replace("{run_dir}", str(run_root))
        .replace("{stage}", "stage12_mapping_optimization_rewire")
        for value in config.get("command") or ()
    ]
    if not command:
        raise ValueError("connector rewire requires final_mapper.command")
    executable = Path(command[0]).expanduser()
    if not executable.is_absolute() or not executable.is_file():
        raise ValueError("final mapper executable must be an existing absolute path")
    request = _mapper_request(
        config,
        run_root=run_root,
        keyframes=keyframes,
        selection=selection,
        roles=roles,
        pair_geometry=pair_geometry,
        corpus_manifest=corpus_manifest,
        output_model=output_model,
    )
    environment = os.environ.copy()
    environment.update(
        {str(key): str(value) for key, value in dict(config.get("env") or {}).items()}
    )
    completed = subprocess.run(  # nosec B603
        command,
        input=json.dumps(request, ensure_ascii=False),
        text=True,
        capture_output=True,
        cwd=(str(config["cwd"]).replace("{run_dir}", str(run_root)) if config.get("cwd") else None),
        env=environment,
        timeout=(
            None if config.get("timeout_seconds") is None else float(config["timeout_seconds"])
        ),
        check=False,
    )
    if completed.returncode:
        tail = (completed.stderr or completed.stdout)[-2000:]
        raise RuntimeError(f"final mapper failed: {tail}")
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("final mapper emitted no JSON receipt")
    receipt = json.loads(lines[-1])
    if str(receipt.get("status") or "").lower() not in {"ok", "success", "completed"}:
        raise RuntimeError(f"final mapper returned status {receipt.get('status')!r}")
    if not output_model.exists():
        raise RuntimeError("final mapper did not create the rewired dense model")
    return receipt


def _promote_model(source: Path, output: Path) -> None:
    source = source.resolve(strict=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_symlink() and output.resolve(strict=True) == source:
        return
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"refusing to replace existing final model {output}")
    temporary = output.with_name(f".{output.name}.candidate")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(source, target_is_directory=True)
    os.replace(temporary, output)


def _slim_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "name",
        "method",
        "model",
        "source_dense_model",
        "registered_images",
        "points3D",
        "observations",
        "track_length_min",
        "track_length_p50",
        "track_length_p90",
        "reprojection_p90_px",
        "reprojection_p99_px",
        "largest_component_ratio",
        "component_sizes",
        "articulation_count",
        "bridge_count",
        "geometry_gate_pass",
    )
    return {key: row[key] for key in keys if key in row}


def run_integrated_optimization(request: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(request.get("payload") or {})
    recipe_values = dict(payload.get("recipe") or {})
    recipe = OptimizationRecipe.from_mapping(recipe_values)
    if not recipe.enabled:
        raise ValueError("mapping optimization recipe is disabled")
    run_root = Path(str(payload["run_root"])).resolve(strict=True)
    input_model = Path(str(payload["input_model"])).resolve(strict=True)
    output_model = Path(str(payload["output_model"])).resolve()
    candidate_root = Path(str(payload["optimization_root"])).resolve()
    comparison_path = Path(str(payload["comparison"])).resolve()
    summary_path = Path(str(payload["summary"])).resolve()
    selection_path = Path(str(payload["selection"])).resolve(strict=True)
    roles_path = Path(str(payload["roles"])).resolve(strict=True)
    keyframes_path = Path(str(payload["keyframes"])).resolve(strict=True)
    geometry_path = Path(str(payload["pair_geometry"])).resolve(strict=True)
    manifest_value = payload.get("corpus_manifest")
    corpus_manifest = (
        Path(str(manifest_value)).resolve(strict=True) if manifest_value is not None else None
    )
    if not candidate_root.is_relative_to(run_root):
        raise ValueError("optimization_root must be inside run_root")
    candidate_root.mkdir(parents=True, exist_ok=True)

    filters = _normalized_filter_variants(recipe)
    candidates: list[dict[str, Any]] = []
    methods_completed: list[str] = []
    method_notes: list[dict[str, Any]] = []

    baseline_rows = run_filter_sweep(input_model, candidate_root / "filter_sweep", filters)
    for row in baseline_rows:
        row.update(
            name=f"baseline_{row['name']}",
            method="robust_filter_sweep",
            source_dense_model=str(input_model),
        )
        candidates.append(row)
    methods_completed.append("robust_filter_sweep")

    if recipe.connectivity_rescue:
        base = rank_variants(baseline_rows)[0]
        rescue_model = candidate_root / "connectivity_rescue/model"
        try:
            receipt = rescue_connectors(
                dense_model=input_model,
                base_model=Path(str(base["model"])),
                output_model=rescue_model,
            )
        except RuntimeError as error:
            method_notes.append(
                {
                    "method": "connectivity_aware_rescue",
                    "status": "NO_ADMISSIBLE_CONNECTORS",
                    "reason": str(error),
                }
            )
        else:
            receipt.update(
                name="connectivity_rescue",
                method="connectivity_aware_rescue",
                source_dense_model=str(input_model),
            )
            candidates.append(receipt)
        methods_completed.append("connectivity_aware_rescue")

    selection = _read_json(selection_path)
    keyframes = {row["keyframe_id"]: row for row in _read_jsonl(keyframes_path)}
    geometry = _read_jsonl(geometry_path)
    final_mapper = dict(payload.get("final_mapper") or {})
    for mutation in recipe.selection_rewires:
        name = str(mutation["name"])
        root = candidate_root / "selection_rewire" / name
        rewritten, receipt = rewire_selection(
            selection,
            keyframes,
            geometry,
            remove={str(value) for value in mutation["remove"]},
            add={str(value) for value in mutation["add"]},
        )
        rewritten_path = root / "selection.json"
        atomic_json(rewritten_path, rewritten)
        atomic_json(root / "selection_receipt.json", receipt)
        dense_model = root / "dense/model"
        mapper_receipt = _run_mapper(
            final_mapper,
            run_root=run_root,
            keyframes=keyframes_path,
            selection=rewritten_path,
            roles=roles_path,
            pair_geometry=geometry_path,
            corpus_manifest=corpus_manifest,
            output_model=dense_model,
        )
        atomic_json(root / "mapping_receipt.json", mapper_receipt)
        rows = run_filter_sweep(dense_model, root / "filter_sweep", filters)
        for row in rows:
            row.update(
                name=f"rewire_{name}_{row['name']}",
                method="connector_frame_rewire",
                source_dense_model=str(dense_model),
            )
            candidates.append(row)
    if recipe.selection_rewires:
        methods_completed.append("connector_frame_rewire")

    if recipe.retriangulation_variants:
        database = Path(str(recipe_values.get("database") or "")).expanduser().resolve(strict=True)
        image_root = (
            Path(str(recipe_values.get("image_root") or "")).expanduser().resolve(strict=True)
        )
        filter_name = str(recipe_values.get("retriangulation_filter_variant") or filters[0]["name"])
        filter_variant = next((row for row in filters if str(row["name"]) == filter_name), None)
        if filter_variant is None:
            raise ValueError(f"unknown retriangulation_filter_variant {filter_name!r}")
        for variant in recipe.retriangulation_variants:
            name = str(variant["name"])
            root = candidate_root / "retriangulation" / name
            dense_model = root / "dense/model"
            dense_model.parent.mkdir(parents=True, exist_ok=True)
            retriangulate_fixed_poses(
                input_model=input_model,
                database=database,
                image_root=image_root,
                output_model=dense_model,
                minimum_angle_deg=float(variant["minimum_angle_deg"]),
                ignore_two_view_tracks=bool(variant.get("ignore_two_view_tracks", True)),
            )
            rows = run_filter_sweep(dense_model, root / "filter", (filter_variant,))
            for row in rows:
                row.update(
                    name=f"retriangulation_{name}_{row['name']}",
                    method="fixed_pose_retriangulation",
                    source_dense_model=str(dense_model),
                )
                candidates.append(row)
        methods_completed.append("fixed_pose_retriangulation")

    winner = select_winner(candidates)
    ranked = rank_variants(candidates)
    for index, row in enumerate(ranked, 1):
        row["rank"] = index
    atomic_json(
        comparison_path,
        {
            "schema_version": 1,
            "artifact_type": "MAPPING_OPTIMIZATION_COMPARISON",
            "methods_completed": methods_completed,
            "method_notes": method_notes,
            "winner": _slim_candidate(winner),
            "candidates": [_slim_candidate(row) for row in ranked],
        },
    )
    _promote_model(Path(str(winner["model"])), output_model)

    pruned: list[str] = []
    if recipe.cleanup_losers:
        for model in prunable_loser_models(candidate_root, Path(str(winner["model"])), candidates):
            shutil.rmtree(model)
            pruned.append(str(model))
    summary = {
        "schema_version": 1,
        "artifact_type": "MAPPING_OPTIMIZATION_SUMMARY",
        "status": "completed",
        "methods_completed": methods_completed,
        "method_notes": method_notes,
        "winner": _slim_candidate(winner),
        "comparison": str(comparison_path),
        "pruned_loser_models": pruned,
        "cleanup_losers": recipe.cleanup_losers,
    }
    atomic_json(summary_path, summary)
    return {
        **summary,
        "outputs": {
            "model": str(output_model),
            "comparison": str(comparison_path),
            "summary": str(summary_path),
        },
    }


def main() -> int:
    try:
        request = json.loads(sys.stdin.read())
        result = run_integrated_optimization(request)
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "reason": str(error),
                },
                ensure_ascii=False,
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main", "run_integrated_optimization"]
