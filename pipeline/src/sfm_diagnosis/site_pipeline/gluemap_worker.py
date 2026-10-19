"""Small, dependency-free boundary around the pinned GLUEMAP runner.

The worker deliberately does not import GLUEMAP.  Its job is to make the
admitted view graph an exact, auditable input to whichever pinned checkout is
run by a higher-level adapter.  See the upstream workflow and coarse mode:
https://github.com/colmap/gluemap/blob/main/README.md
https://github.com/colmap/gluemap/blob/main/gluemap/controllers/gluemap_impl.py
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
from argparse import Namespace
from contextlib import nullcontext, redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from .intrinsics import calibration_matrix_for_resolution
from .post_sfm import model_capabilities


Pair = tuple[str, str]
IndexPair = tuple[int, int]
COLMAP_MAX_IMAGE_ID = 2_147_483_647


@dataclass(frozen=True)
class MappingInputs:
    selected: tuple[dict[str, Any], ...]
    selected_ids: frozenset[str]
    pose_names: frozenset[str]
    admitted_names: frozenset[Pair] | None


def _canonical(a: str, b: str) -> Pair:
    if a == b:
        raise ValueError(f"self pair is not valid: {a}")
    return (a, b) if a < b else (b, a)


def admitted_pairs_from_jsonl(path: str | Path) -> set[Pair]:
    """Read admitted pair records, accepting the common i/j key spellings."""
    result: set[Pair] = set()
    for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        a = record.get("image_i", record.get("image1", record.get("image_a")))
        b = record.get("image_j", record.get("image2", record.get("image_b")))
        if not isinstance(a, str) or not isinstance(b, str):
            raise ValueError(f"line {line_no}: pair requires image_i/image_j")
        result.add(_canonical(a, b))
    return result


def select_direct_keyframes(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], set[str]]:
    """Select every non-rejected keyframe for direct all-sequence mapping."""

    selected = [dict(row) for row in rows if row.get("status", "CANDIDATE") == "CANDIDATE"]
    if not selected:
        raise ValueError("direct mapping contains no candidate keyframes")
    names = [str(row.get("output_name") or "") for row in selected]
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError("direct keyframe output names must be non-empty and unique")
    pose_only = {
        str(row["output_name"])
        for row in selected
        if row.get("mapping_mode") == "POSE_ONLY"
    }
    return selected, pose_only


def resolve_mapping_inputs(
    *,
    keyframes_path: str | Path,
    pair_source: str,
    selection_path: str | Path | None = None,
    geometry_path: str | Path | None = None,
    mode: str = "final",
) -> MappingInputs:
    """Resolve the small input interface for verified or native mapping."""

    keyframes = _jsonl(Path(keyframes_path))
    if pair_source == "native":
        selected, pose_names = select_direct_keyframes(keyframes)
        return MappingInputs(
            tuple(selected),
            frozenset(str(row["keyframe_id"]) for row in selected),
            frozenset(pose_names),
            None,
        )
    if pair_source != "verified":
        raise ValueError("pair_source must be verified or native")
    selection_file = Path(str(selection_path or ""))
    geometry_file = Path(str(geometry_path or ""))
    if not selection_file.is_file() or not geometry_file.is_file():
        raise ValueError("verified mapping requires selection and pair_geometry files")
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    selected_ids = set(selection.get("selected_keyframes") or ())
    if mode == "final" and not selected_ids:
        active_segments = set(selection.get("active_segments") or ())
        selected_ids = {
            str(row["keyframe_id"])
            for row in keyframes
            if str(row.get("segment_id")) in active_segments
        }
    selected = [row for row in keyframes if str(row.get("keyframe_id")) in selected_ids]
    if not selected:
        raise ValueError("GLUEMAP selection contains no keyframes")
    pairs = [
        row
        for row in _jsonl(geometry_file)
        if row.get("admission") == "VERIFIED"
        and str(row.get("image_i")) in selected_ids
        and str(row.get("image_j")) in selected_ids
    ]
    if not pairs:
        raise ValueError("GLUEMAP selection contains no verified pairs")
    id_to_name = {
        str(row["keyframe_id"]): str(
            row.get("output_name") or Path(str(row["image_uri"])).name
        )
        for row in selected
    }
    admitted = frozenset(
        _canonical(id_to_name[str(row["image_i"])], id_to_name[str(row["image_j"])])
        for row in pairs
    )
    return MappingInputs(tuple(selected), frozenset(selected_ids), frozenset(), admitted)


def _names(dataset: Any) -> list[str]:
    names = getattr(dataset, "images_list", None)
    if names is None:
        names = getattr(dataset, "image_names", None)
    if names is None:
        names = getattr(dataset, "images", None)
        if isinstance(names, Mapping):
            names = list(names)
    if names is None:
        raise TypeError("dataset must expose image_names or images")
    return [Path(str(name)).as_posix() for name in names]


def map_pairs_to_indices(dataset: Any, pairs: Iterable[Pair]) -> set[IndexPair]:
    lookup = {name: i for i, name in enumerate(_names(dataset))}
    out = set()
    for a, b in pairs:
        try:
            out.add(tuple(sorted((lookup[a], lookup[b]))))
        except KeyError as exc:
            raise KeyError(f"admitted image is absent from dataset: {exc.args[0]}") from exc
    return out


def replace_dataset_pairs(
    dataset: Any, pairs: Iterable[Pair] | Iterable[IndexPair]
) -> set[IndexPair]:
    """Replace (never union) the dataset pair list and return its exact set."""
    values = list(pairs)
    indexed = (
        map_pairs_to_indices(dataset, values)
        if values and isinstance(values[0][0], str)
        else set(values)
    )
    dataset.pairs = np.asarray(sorted(indexed), dtype=np.int64).reshape(-1, 2)
    if hasattr(dataset, "sequential_edges"):
        dataset.sequential_edges = sorted(
            restrict_sequential_edges(dataset.sequential_edges, indexed)
        )
    return indexed


def restrict_sequential_edges(
    edges: Iterable[IndexPair], admitted: Iterable[IndexPair]
) -> set[IndexPair]:
    allowed = {tuple(sorted(edge)) for edge in admitted}
    return {tuple(sorted(edge)) for edge in edges if tuple(sorted(edge)) in allowed}


def assert_exact_pair_set(dataset: Any, admitted: Iterable[IndexPair]) -> None:
    actual = {tuple(sorted(pair)) for pair in getattr(dataset, "pairs", ())}
    expected = {tuple(sorted(pair)) for pair in admitted}
    assert actual == expected, f"dataset pair set is not exact: expected {expected}, got {actual}"


def workspace_identity(workspace: str | Path, selection: Any, pairs: Any, config: Any) -> str:
    payload = {
        "workspace": str(workspace),
        "selection": selection,
        "pairs": sorted(map(list, pairs)),
        "config": config,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def admitted_pair_ids_from_database(
    database: str | Path,
    admitted_names: Iterable[Pair],
) -> set[tuple[int, int]]:
    """Resolve admitted image-name pairs through the database's authoritative IDs."""

    path = Path(database)
    if not path.is_file():
        raise FileNotFoundError(path)
    with sqlite3.connect(path) as connection:
        names = {
            str(name): int(image_id)
            for image_id, name in connection.execute("SELECT image_id, name FROM images")
        }
    resolved: set[tuple[int, int]] = set()
    missing: set[str] = set()
    for left, right in admitted_names:
        if left not in names or right not in names:
            missing.update(name for name in (left, right) if name not in names)
            continue
        resolved.add(tuple(sorted((names[left], names[right]))))
    if missing:
        raise RuntimeError(f"admitted image names are absent from SIFT database: {sorted(missing)[:10]}")
    return resolved


def find_completed_refined_model(workspace: str | Path) -> Path | None:
    """Return a fully materialized refined model that is safe to revalidate."""

    root = Path(workspace)
    model = root / "gluemap/gluemap_aba"
    required = [model / name for name in ("cameras.bin", "images.bin", "points3D.bin")]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return None
    if not (root / "gluemap/database_sift.db").is_file():
        return None
    return model


def prepare_gluemap_config(
    mode: str,
    *,
    evidence_level: str | None = None,
    reuse_inference_cache: bool = False,
) -> dict[str, Any]:
    if mode not in {"diagnostic", "final"}:
        raise ValueError("mode must be diagnostic or final")
    level = evidence_level or ("coarse_pose" if mode == "diagnostic" else "refined_geometry")
    if level not in {"coarse_pose", "refined_geometry"}:
        raise ValueError("evidence_level must be coarse_pose or refined_geometry")
    if mode == "final" and level != "refined_geometry":
        raise ValueError("final mapping requires refined_geometry evidence")
    return {
        "coarse_only": level == "coarse_pose",
        "sample_frequency": 1,
        "force_load": bool(reuse_inference_cache),
        "rerun_from": None if reuse_inference_cache else "retrieval",
    }


# Stage markers relative to the GLUEMAP write dir (``<workspace>/gluemap``).
# Mirrors the pinned runner: twoview/star caches (`base_inference.run`),
# Retrieval descriptors (`image_retrieval`), and the SIFT database
# (`gluemap_impl`, skipped when `force_load` finds it).
_TWOVIEW_CACHE = "twoview_result.pth"
_STAR_CACHE = "star_result.pth"
_SIFT_DATABASE = "database_sift.db"
# Artifacts derived from star predictions. Dropped when an upstream stage is
# recomputed so refinement never consumes a stale predecessor. The identity
# hash in the workspace name already pins selection/pairs/config, so this is
# defense in depth, not the primary guard.
_DERIVED_PREFIXES = ("database_sift", "database_merged", "coarse", "gluemap_aba")
_DERIVED_FILES = ("pipeline_timing.pth",)


@dataclass(frozen=True)
class StageResume:
    """Crash-resume decision for one GLUEMAP workspace.

    Fresh workspaces reproduce the historical behavior exactly
    (``force_load=False, rerun_from="retrieval"``). A resumed workspace reuses
    probe-validated caches: every cache is load-tested before reuse, and
    unloadable partial files are deleted so the runner rebuilds them.
    Cross-configuration leakage is impossible by construction — the workspace
    directory embeds the identity hash over selection, pairs, config,
    checkpoints, and intrinsics. Resume reproduces fresh-run outputs on the
    same host; bit-identity across different GPUs/drivers is not claimed.
    """

    force_load: bool
    rerun_from: str | None
    cache: Mapping[str, str]
    dropped: tuple[str, ...]


def _loadable_torch_cache(path: Path) -> bool:
    """Probe-load a torch cache; delete unloadable partial files.

    ``torch.save`` is not atomic, so a crash mid-write leaves a truncated
    file that must be rebuilt, not reused. The probe transiently holds the
    cache in RAM (seconds, versus hours to recompute star).
    """

    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        import torch
    except ImportError:
        # A host without torch cannot run GLUEMAP at all; report the cache
        # as unusable without destroying it.
        return False
    try:
        torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        return False
    return True


def _valid_sift_database(path: Path) -> bool:
    """Check a SIFT database; delete corrupt partial files."""

    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            images = connection.execute("SELECT COUNT(*) FROM images").fetchone()
        if (integrity or [None])[0] != "ok" or (images or [0])[0] <= 0:
            raise ValueError("sift database failed validation")
    except Exception:
        try:
            path.unlink()
        except OSError:
            pass
        return False
    return True


def _drop_derived_artifacts(write_dir: Path) -> tuple[str, ...]:
    """Remove artifacts derived from star predictions ahead of a rebuild."""

    dropped: list[str] = []
    for child in sorted(write_dir.iterdir()):
        name = child.name
        if name in _DERIVED_FILES or name.startswith(_DERIVED_PREFIXES):
            try:
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            except OSError:
                continue
            dropped.append(name)
    return tuple(dropped)


def resolve_stage_resume(write_dir: str | Path) -> StageResume:
    """Decide ``force_load``/``rerun_from`` from validated stage markers.

    Upstream deletion semantics (`base_inference.run`): any set
    ``rerun_from`` deletes the star cache (star triggers are ``None``);
    ``"twoview"`` additionally deletes the twoview cache; only
    ``"retrieval"`` touches descriptor caches. So ``None`` resumes
    everything cached, ``"star"`` redoes star while keeping twoview, and
    ``"twoview"`` redoes both. Post-star stages (coarse/SIFT/refinement)
    always rerun — the runner offers no skip there — except the SIFT
    database, which `force_load` reuses after validation.
    """

    root = Path(write_dir)
    twoview_path = root / _TWOVIEW_CACHE
    star_path = root / _STAR_CACHE
    sift_path = root / _SIFT_DATABASE
    if not (twoview_path.exists() or star_path.exists() or sift_path.exists()):
        return StageResume(
            force_load=False,
            rerun_from="retrieval",
            cache={},
            dropped=(),
        )
    try:
        import torch  # noqa: F401

        torch_available = True
    except ImportError:
        torch_available = False
    cache: dict[str, str] = {}
    for label, found in (
        ("retrieval", any(root.rglob("salad_descriptors.pt")) or any(root.rglob("megaloc_descriptors.pt"))),
        ("sift_database", _valid_sift_database(sift_path)),
    ):
        cache[label] = "present" if found else "missing"
    for label, path in (("twoview", twoview_path), ("star", star_path)):
        if not path.exists():
            cache[label] = "missing"
        elif _loadable_torch_cache(path):
            cache[label] = "validated"
        else:
            cache[label] = "unvalidated-no-torch" if not torch_available else "corrupt-dropped"
    if cache["twoview"] != "validated":
        if star_path.exists():
            try:
                star_path.unlink()
            except OSError:
                pass
            cache["star"] = "stale-dropped"
        dropped = _drop_derived_artifacts(root) if root.is_dir() else ()
        return StageResume(True, "twoview", cache, dropped)
    if cache["star"] != "validated":
        dropped = _drop_derived_artifacts(root) if root.is_dir() else ()
        return StageResume(True, "star", cache, dropped)
    return StageResume(True, None, cache, ())


def force_sift_device(function: Callable[..., Any], sift_device: str) -> Callable[..., Any]:
    """Bind GLUEMAP's native SIFT preparation to an explicit safe device."""

    if sift_device not in {"cpu", "cuda"} and not sift_device.startswith("cuda:"):
        raise ValueError("sift_device must be cpu, cuda, or cuda:N")

    def configured(*args: Any, **kwargs: Any) -> Any:
        kwargs["device"] = sift_device
        return function(*args, **kwargs)

    return configured


def resolve_workspace_root(logical_root: str | Path, configured_root: str | Path | None) -> Path:
    """Resolve an optional physical workspace without changing the logical stage path."""

    if configured_root is None:
        return Path(logical_root).expanduser().resolve()
    candidate = Path(configured_root).expanduser()
    if not candidate.is_absolute():
        raise ValueError("workspace_root must be an absolute path")
    return candidate.resolve()


def assert_run_controlled_path(
    path: str | Path,
    run_root: str | Path,
    *,
    follow_symlinks: bool = True,
) -> None:
    """Require a path inside the run, optionally without following its final symlink."""

    value = Path(path).expanduser()
    root = Path(run_root).expanduser().resolve()
    resolved = value.resolve() if follow_symlinks else value.parent.resolve() / value.name
    if root not in resolved.parents:
        raise ValueError(f"GLUEMAP path escapes run_root: {path}")


def publish_model_symlink(
    output: str | Path,
    model: str | Path,
    *,
    run_root: str | Path,
    workspace_root: str | Path | None = None,
) -> Path:
    """Atomically advance a run-local model symlink while retaining old workspaces."""

    output_path = Path(output).expanduser().absolute()
    model_path = Path(model).expanduser().resolve()
    trusted_root = Path(run_root).expanduser().resolve()
    model_roots = [trusted_root]
    if workspace_root is not None:
        model_roots.append(Path(workspace_root).expanduser().resolve())
    if trusted_root not in output_path.parents or not any(
        root == model_path or root in model_path.parents for root in model_roots
    ):
        raise ValueError("model publication paths must remain inside trusted roots")
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or output_path.is_symlink():
        if not output_path.is_symlink():
            raise RuntimeError(f"refusing to replace non-symlink output {output_path}")
        previous = output_path.resolve()
        if not any(root == previous or root in previous.parents for root in model_roots):
            raise RuntimeError(f"refusing to replace external model symlink {output_path}")
        if previous == model_path:
            return output_path
    temporary = output_path.with_name(f".{output_path.name}.{model_path.parent.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(model_path, target_is_directory=True)
    os.replace(temporary, output_path)
    return output_path


def write_scaled_intrinsics_seed(
    selected: Sequence[Mapping[str, Any]],
    *,
    dimensions_by_video: Mapping[str, tuple[int, int]],
    calibration: Mapping[str, Any],
    output_dir: str | Path,
) -> Path:
    """Write a COLMAP text seed with one undistorted PINHOLE camera per resolution."""

    rows: list[tuple[Mapping[str, Any], tuple[int, int]]] = []
    for row in selected:
        video_id = str(row.get("video_id") or row.get("source_id") or "")
        if video_id not in dimensions_by_video:
            raise ValueError(f"missing calibrated dimensions for selected source: {video_id}")
        rows.append((row, dimensions_by_video[video_id]))
    resolutions = sorted({resolution for _, resolution in rows})
    camera_ids = {resolution: index for index, resolution in enumerate(resolutions, 1)}
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    camera_lines = ["# Camera list", "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    for resolution in resolutions:
        width, height = resolution
        matrix = calibration_matrix_for_resolution(calibration, target_size=resolution)
        camera_lines.append(
            f"{camera_ids[resolution]} PINHOLE {width} {height} "
            f"{float(matrix[0, 0])} {float(matrix[1, 1])} "
            f"{float(matrix[0, 2])} {float(matrix[1, 2])}"
        )
    (root / "cameras.txt").write_text("\n".join(camera_lines) + "\n", encoding="utf-8")
    image_lines = [
        "# Image list",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
    ]
    for image_id, (row, resolution) in enumerate(rows, 1):
        name = Path(str(row.get("output_name") or "")).as_posix()
        if not name or name == "." or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("selected keyframe has an invalid output_name")
        image_lines.extend(
            [f"{image_id} 1 0 0 0 0 0 0 {camera_ids[resolution]} {name}", ""]
        )
    (root / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    (root / "points3D.txt").write_text("# 3D point list\n", encoding="utf-8")
    return root


def run_adapter_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Run the pinned GLUEMAP checkout from one JSON adapter request.

    The worker deliberately uses GLUEMAP's Python orchestration so the exact
    admitted pair set can replace the dataset pairs before two-view/star/global
    mapping. It never relies on ``sample_frequency`` for multi-sequence data.
    """

    config = dict(payload.get("config") or {})
    request_payload = dict(payload.get("payload") or {})
    root = Path(str(config.get("gluemap_root") or "")).expanduser().resolve()
    config_path = Path(str(config.get("config_file") or "")).expanduser().resolve()
    if not (root / "gluemap").is_dir() or not config_path.is_file():
        raise ValueError("GLUEMAP adapter requires valid gluemap_root and config_file")
    mode = str(config.get("mode") or request_payload.get("mode") or "diagnostic")
    if mode not in {"diagnostic", "final"}:
        raise ValueError("GLUEMAP mode must be diagnostic or final")
    pair_source = str(config.get("pair_source") or "verified")
    keyframes_path = Path(str(request_payload.get("keyframes") or ""))
    selection_path = Path(str(request_payload.get("selection") or ""))
    geometry_path = Path(str(request_payload.get("pair_geometry") or ""))
    roles_path = Path(str(request_payload.get("roles") or ""))
    output_model_value = str(request_payload.get("output_model") or "").strip()
    if not output_model_value:
        raise ValueError("GLUEMAP request requires output_model")
    output_model = Path(output_model_value).expanduser()
    if not output_model.is_absolute():
        raise ValueError("GLUEMAP output_model must be an absolute run path")
    run_root_value = str(request_payload.get("run_root") or "").strip()
    if not run_root_value:
        raise ValueError("GLUEMAP request requires trusted run_root")
    run_root = Path(run_root_value).expanduser().resolve()
    if not run_root.is_dir():
        raise ValueError("GLUEMAP run_root is not an initialized directory")
    controlled_paths = [keyframes_path]
    if pair_source == "verified":
        controlled_paths.extend((selection_path, geometry_path))
    if roles_path.is_file():
        controlled_paths.append(roles_path)
    for controlled_path in controlled_paths:
        assert_run_controlled_path(controlled_path, run_root)
    assert_run_controlled_path(output_model, run_root, follow_symlinks=False)
    if not keyframes_path.is_file():
        raise ValueError("GLUEMAP request requires keyframes")
    mapping_inputs = resolve_mapping_inputs(
        keyframes_path=keyframes_path,
        pair_source=pair_source,
        selection_path=selection_path,
        geometry_path=geometry_path,
        mode=mode,
    )
    selected = list(mapping_inputs.selected)
    selected_ids = set(mapping_inputs.selected_ids)
    admitted_names = None if mapping_inputs.admitted_names is None else set(mapping_inputs.admitted_names)
    pose_names = set(mapping_inputs.pose_names)
    id_to_name = {
        str(row["keyframe_id"]): str(
            row.get("output_name") or Path(str(row["image_uri"])).name
        )
        for row in selected
    }

    evidence_level = str(
        config.get("evidence_level")
        or ("coarse_pose" if mode == "diagnostic" else "refined_geometry")
    )
    reuse_inference_cache = bool(config.get("reuse_inference_cache", False))
    execution_profile = prepare_gluemap_config(
        mode,
        evidence_level=evidence_level,
        reuse_inference_cache=reuse_inference_cache,
    )
    sift_device = str(config.get("sift_device") or "cuda")

    configured = json.loads(config_path.read_text(encoding="utf-8"))
    sys.path.insert(0, str(root))
    from gluemap.utils.cli import get_args_parser

    base_config = vars(get_args_parser().parse_args([]))
    base_config.update(configured)
    checkpoint_hashes = _checkpoint_hashes(base_config)
    calibration = config.get("intrinsics_calibration")
    identity = workspace_identity(
        output_model.parent,
        sorted(selected_ids),
        sorted(admitted_names) if admitted_names is not None else ["NATIVE_GLUEMAP_PAIRS"],
        {
            "config": base_config,
            "checkpoints": checkpoint_hashes,
            "intrinsics_calibration": calibration,
            "pair_source": pair_source,
            "pose_names": sorted(pose_names),
        },
    )
    workspace_root = resolve_workspace_root(output_model.parent, config.get("workspace_root"))
    workspace = workspace_root / f"work_{identity[:16]}"
    images_root = workspace / "images"
    approved_roots = [run_root / "artifacts/keyframes/images"]
    _materialize_images(selected, images_root, approved_roots)
    intrinsics_seed = None
    if calibration is not None:
        manifest_path = Path(str(request_payload.get("corpus_manifest") or ""))
        if not manifest_path.is_file():
            raise ValueError("scaled intrinsics require the corpus manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dimensions_by_video: dict[str, tuple[int, int]] = {}
        for source in manifest.get("sources") or ():
            width, height = int(source.get("width") or 0), int(source.get("height") or 0)
            if width <= 0 or height <= 0:
                continue
            for key in (source.get("video_id"), source.get("source_id")):
                if key:
                    dimensions_by_video[str(key)] = (width, height)
        intrinsics_seed = write_scaled_intrinsics_seed(
            selected,
            dimensions_by_video=dimensions_by_video,
            calibration=dict(calibration),
            output_dir=workspace / "intrinsics_seed",
        )
    stage_resume = resolve_stage_resume(workspace / "gluemap")
    gluemap_config = {
        **base_config,
        **execution_profile,
        "images_path": str(images_root),
        "write_path": str(workspace / "gluemap"),
        "temp_path": str(workspace / "tmp"),
        "skip_doppelgangers": bool(config.get("skip_doppelgangers", True)),
        "subfolder_regex": ".*",
        # Crash-resume overrides the execution profile. Fresh workspaces
        # resolve to the historical profile untouched (see StageResume).
        "force_load": stage_resume.force_load,
        "rerun_from": stage_resume.rerun_from,
    }
    if intrinsics_seed is not None:
        gluemap_config["gt_intrinsics_path"] = str(intrinsics_seed)
        gluemap_config["camera_model"] = "PINHOLE"
        gluemap_config["intrinsics_mode"] = "SHARED"
        gluemap_config["refine_intrinsics"] = False
    cached_model = (
        find_completed_refined_model(workspace)
        if pair_source == "verified"
        and mode == "diagnostic"
        and evidence_level == "refined_geometry"
        and reuse_inference_cache
        else None
    )
    if cached_model is not None:
        import pycolmap

        reconstruction = pycolmap.Reconstruction(str(cached_model))
        observations = sum(
            len(point.track.elements) for point in reconstruction.points3D.values()
        )
        capabilities = model_capabilities(
            evidence_level=evidence_level,
            registered_images=reconstruction.num_reg_images(),
            total_images=len(reconstruction.images),
            landmarks=len(reconstruction.points3D),
            observations=observations,
        )
        if not capabilities["role_assignment_ready"]:
            raise RuntimeError("cached refined model lacks landmark/track evidence")
        sift_database = workspace / "gluemap/database_sift.db"
        expected_database_pairs = admitted_pair_ids_from_database(
            sift_database,
            admitted_names,
        )
        pair_proof = {
            "admitted_pairs": len(expected_database_pairs),
            "dataset_pairs_before_and_after_exact": True,
            "refinement_database": validate_colmap_pair_database(
                sift_database,
                expected_database_pairs,
            ),
        }
        publish_model_symlink(
            output_model,
            cached_model,
            run_root=run_root,
            workspace_root=workspace_root,
        )
        return {
            "status": "completed",
            "outputs": [str(output_model)],
            "pose_only_mask": None,
            "resume": asdict(stage_resume),
            "workspace": str(workspace),
            "checkpoint_hashes": checkpoint_hashes,
            "pair_count": len(expected_database_pairs),
            "exact_pair_proof": pair_proof,
            "model_capabilities": capabilities,
            "intrinsics_seed": None if intrinsics_seed is None else str(intrinsics_seed),
            "images_are_undistorted": bool(
                calibration and calibration.get("images_are_undistorted") is True
            ),
            "sift_device": sift_device,
            "runtime": {"cache_reuse": True, "total_pipeline": 0.0},
        }
    args = Namespace(**gluemap_config)
    args.curr_processed = args.write_path
    args.curr_path = args.write_path

    previous_cwd = Path.cwd()
    try:
        os.chdir(root)
        from gluemap.controllers import gluemap_impl
        from gluemap.controllers.image_retrieval import run_preprocessing_pipeline_multi
        from gluemap.datasets.multi_sequence_twoview import MultiSequencePairs
        from gluemap.utils.gpu import init_distributed
        from gluemap.utils.colmap import prepare_sift_database as native_prepare_sift_database

        gluemap_impl.prepare_sift_database = force_sift_device(
            native_prepare_sift_database,
            sift_device,
        )
        run_inference_pipeline = gluemap_impl.run_inference_pipeline

        rank, world_size, device, dtype = init_distributed(args)
        datasets = [
            child.name
            for child in sorted(images_root.iterdir())
            if child.is_dir()
            and (
                not getattr(args, "subfolder_regex", None)
                or re.match(args.subfolder_regex, child.name)
            )
        ]
        if not datasets:
            raise RuntimeError("no GLUEMAP image sequence folders were materialized")
        args.is_multi_sequence = True
        args.is_sequential = True
        run_preprocessing_pipeline_multi(args, world_size, rank, datasets)
        dataset = MultiSequencePairs(args, datasets)
        if admitted_names is None:
            admitted_indices = {
                tuple(sorted((int(pair[0]), int(pair[1])))) for pair in dataset.pairs
            }
            dataset_names = _names(dataset)
            admitted_names = {
                _canonical(dataset_names[left], dataset_names[right])
                for left, right in admitted_indices
            }
        else:
            admitted_indices = replace_dataset_pairs(dataset, admitted_names)
        assert_exact_pair_set(dataset, admitted_indices)
        if pair_source == "verified" and mode == "final" and roles_path.is_file():
            pose_segments = {
                str(row.get("segment_id") or row.get("segment"))
                for row in _jsonl(roles_path)
                if row.get("mapping_mode") == "POSE_ONLY"
            }
            pose_names = {
                id_to_name[str(row["keyframe_id"])]
                for row in selected
                if str(row.get("segment_id")) in pose_segments
            }
        if pose_names:
            from .pose_only import gluemap_pose_only_mask

            mask_context = gluemap_pose_only_mask(pose_names, workspace_root=workspace)
        else:
            mask_context = nullcontext(None)
        with mask_context as pose_only_state:
            prediction_dir, timing = run_inference_pipeline(
                args,
                dataset,
                world_size,
                rank,
                device,
                dtype,
                pairs=dataset.pairs,
            )
        assert_exact_pair_set(dataset, admitted_indices)
    finally:
        os.chdir(previous_cwd)
    if prediction_dir is None:
        raise RuntimeError("GLUEMAP did not produce a rank-0 reconstruction")
    model = Path(args.write_path) / prediction_dir
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(model))
    observations = sum(
        len(point.track.elements) for point in reconstruction.points3D.values()
    )
    capabilities = model_capabilities(
        evidence_level=evidence_level,
        registered_images=reconstruction.num_reg_images(),
        total_images=len(reconstruction.images),
        landmarks=len(reconstruction.points3D),
        observations=observations,
    )
    if evidence_level == "refined_geometry" and not capabilities["role_assignment_ready"]:
        raise RuntimeError("refined_geometry mapping produced no landmark/track evidence")
    pair_proof: dict[str, Any] = {
        "admitted_pairs": len(admitted_indices),
        "dataset_pairs_before_and_after_exact": True,
        "refinement_database": None,
    }
    sift_database = Path(args.write_path) / "database_sift.db"
    if sift_database.is_file():
        expected_database_pairs = admitted_pair_ids_from_database(
            sift_database,
            admitted_names,
        )
        pair_proof["refinement_database"] = validate_colmap_pair_database(
            sift_database,
            expected_database_pairs,
        )
    elif not execution_profile["coarse_only"]:
        raise RuntimeError("Refined GLUEMAP did not produce database_sift.db for pair proof")
    else:
        pair_proof["refinement_database"] = "NOT_APPLICABLE_COARSE_ONLY"
    if pose_names:
        from .pose_only import assert_pose_only_observation_free

        pose_only_state["final_model_check"] = assert_pose_only_observation_free(
            pycolmap.Reconstruction(str(model)), pose_names
        )
    publish_model_symlink(
        output_model,
        model,
        run_root=run_root,
        workspace_root=workspace_root,
    )
    return {
        "status": "completed",
        "outputs": [str(output_model)],
        "workspace_identity": identity,
        "pose_only_mask": pose_only_state,
        "resume": asdict(stage_resume),
        "checkpoint_hashes": checkpoint_hashes,
        "pair_count": len(admitted_indices),
        "pair_source": pair_source,
        "exact_pair_proof": pair_proof,
        "model_capabilities": capabilities,
        "intrinsics_seed": None if intrinsics_seed is None else str(intrinsics_seed),
        "images_are_undistorted": bool(
            calibration and calibration.get("images_are_undistorted") is True
        ),
        "sift_device": sift_device,
        "runtime": timing,
    }


def pose_only_manifest(records: Iterable[Mapping[str, Any]]) -> set[str]:
    return {str(r["image"]) for r in records if r.get("mapping_mode") == "POSE_ONLY"}


def filter_pair_predictions(pairs: Sequence[IndexPair], valid: Sequence[bool]) -> list[IndexPair]:
    if len(pairs) != len(valid):
        raise ValueError("prediction validity mask length must equal pair count")
    return [pair for pair, keep in zip(pairs, valid) if bool(keep)]


def _materialize_images(
    rows: Sequence[Mapping[str, Any]], root: Path, approved_roots: Sequence[Path]
) -> None:
    approved = [path.expanduser().resolve() for path in approved_roots]
    for row in rows:
        source = Path(str(row.get("image_uri") or "")).expanduser().resolve()
        relative = Path(str(row.get("output_name") or source.name))
        if not source.is_file() or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid keyframe image record: {row.get('keyframe_id')}")
        if not any(source == base or base in source.parents for base in approved):
            raise ValueError(f"keyframe image escapes approved roots: {source}")
        expected_sha = row.get("image_sha256")
        if expected_sha and _sha256_file(source) != str(expected_sha):
            raise ValueError(f"keyframe image hash changed: {source}")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            if target.resolve() != source:
                raise RuntimeError(f"keyframe target collision: {target}")
        else:
            target.symlink_to(source)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    hashes = {}
    for key in ("path_feedforward", "path_retrieval", "path_tracker", "path_dg"):
        value = config.get(key)
        if not value:
            continue
        path = Path(str(value)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes[key] = _sha256_file(path)
    return hashes


def validate_colmap_pair_database(
    database: str | Path, admitted_image_ids: set[tuple[int, int]]
) -> dict[str, Any]:
    expected = {tuple(sorted(pair)) for pair in admitted_image_ids}
    path = Path(database)
    if not path.is_file():
        raise FileNotFoundError(path)
    counts = {}
    with sqlite3.connect(path) as connection:
        tables = {
            str(row[0])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        queries = {
            "matches": "SELECT pair_id FROM matches",
            "two_view_geometries": "SELECT pair_id FROM two_view_geometries",
        }
        for table, query in queries.items():
            if table not in tables:
                raise RuntimeError(f"COLMAP database lacks {table}")
            actual = {
                tuple(sorted(divmod(int(row[0]), COLMAP_MAX_IMAGE_ID)))
                for row in connection.execute(query)
            }
            extras = sorted(actual - expected)
            if extras:
                raise RuntimeError(f"COLMAP {table} contains non-admitted pairs: {extras[:10]}")
            counts[table] = len(actual)
    return {"clean": True, "expected_pairs": len(expected), **counts}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def main() -> int:
    output = sys.stdout
    try:
        with redirect_stdout(sys.stderr):
            payload = json.load(sys.stdin)
            result = run_adapter_request(payload)
    except Exception as error:
        output.write(json.dumps({"status": "error", "error": str(error)}) + "\n")
        return 1
    output.write(json.dumps(result, default=str) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
