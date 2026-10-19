"""Publish reference-pixel landmarks from the offline verified River surface."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .geometry_contract import (
    OBS_SOURCE, OBS_DEPTH_AGREE, OBS_PHOTO_AGREE, OBS_MVS_AGREE, OBS_SPARSE_AGREE,
    OBS_CONFLICT, OBS_OCCLUDED, LEDGER_DTYPES, LEDGER_SCHEMA, SCORE_SEMANTICS, validate_mapping_record,
    apply_depth_calibration, validate_depth_calibration, validate_pose, validate_pinhole,
)
from .metric_dense import (
    VALIDATION_VERSION, _load_raw_npz, load_dense_map, model_file_hashes,
    safe_image_name, sha256_file,
)
from .metric_scale import load_metric_scale, request_digest
from .moge_metric_worker import INFERENCE_SETTINGS
from .trusted_landmarks import SCHEMA, _select_ledger_rows, _validate_accuracy_ledger

MIN_SUPPORT = 2
MIN_CONFIDENCE = .3
EVIDENCE_VERSION = "trusted-landmarks-publisher/v3"


def reference_candidates(xyz: np.ndarray, pose: np.ndarray, k_source: np.ndarray,
                         raw: dict[str, Any], per_image_scale: float,
                         native_wh: tuple[int, int], *, calibration: Mapping[str, Any],
                         observed_uv: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """QC a real native observation; interpolate only taps on one depth layer."""
    if not math.isfinite(per_image_scale) or per_image_scale <= 0:
        raise ValueError("invalid per-image scale")
    pose = validate_pose(pose)
    k_source = validate_pinhole(k_source, *native_wh)
    k_work = np.asarray(raw['K_work'], dtype=float)
    transform = np.asarray(raw['source_to_work'], dtype=float)
    if not np.allclose(transform, k_work @ np.linalg.inv(k_source), rtol=1e-6, atol=1e-6):
        raise ValueError("source/work intrinsics and transform disagree")
    camera = np.asarray(xyz, dtype=float) @ pose[:3, :3].T + pose[:3, 3]
    z = camera[:, 2]
    uv_h = camera @ k_source.T
    projected = uv_h[:, :2] / np.where(z > 0, z, np.nan)[:, None]
    uv = projected.copy()
    valid = np.isfinite(camera).all(axis=1) & (z > 0)
    if observed_uv is not None:
        observations = np.asarray(observed_uv, dtype=float)
        if observations.shape != projected.shape:
            raise ValueError('retained native reference pixels are misaligned')
        present = np.isfinite(observations).all(axis=1)
        if np.any(~present & ~np.isnan(observations).all(axis=1)):
            raise ValueError('malformed retained native reference pixels')
        uv[present] = observations[present]
        valid[present] &= np.linalg.norm(projected[present] - uv[present], axis=1) <= 2.
    work_h = np.column_stack((uv, np.ones(len(uv)))) @ transform.T
    work = work_h[:, :2] / work_h[:, 2:3]
    rounded_work = np.rint(work)
    work = np.where(np.abs(work - rounded_work) <= 1e-6, rounded_work, work)
    depth, mask = np.asarray(raw['depth_m']), np.asarray(raw['mask'])
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError('reference depth/mask shape mismatch')
    h, w = depth.shape
    valid &= (np.isfinite(work).all(axis=1) & np.isfinite(uv).all(axis=1)
              & (uv[:, 0] >= 0) & (uv[:, 0] < native_wh[0])
              & (uv[:, 1] >= 0) & (uv[:, 1] < native_wh[1])
              & (work[:, 0] >= 0) & (work[:, 0] <= w - 1)
              & (work[:, 1] >= 0) & (work[:, 1] <= h - 1))
    ids = np.flatnonzero(valid)
    if len(ids):
        samples = work[ids]
        x0, y0 = np.floor(samples).astype(np.int64).T
        x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
        wx, wy = (samples - np.column_stack((x0, y0))).T
        weights = np.column_stack(((1 - wx) * (1 - wy), wx * (1 - wy),
                                   (1 - wx) * wy, wx * wy))
        px, py = np.column_stack((x0, x1, x0, x1)), np.column_stack((y0, y0, y1, y1))
        active = weights > 0
        taps = depth[py, px]
        native_h = np.stack((px, py, np.ones_like(px)), axis=-1) @ np.linalg.inv(transform).T
        native = native_h[..., :2] / native_h[..., 2:3]
        corrected = apply_depth_calibration(taps, native[..., 0], native[..., 1], calibration,
                                            width=native_wh[0], height=native_wh[1]) / per_image_scale
        good_taps = mask[py, px] & np.isfinite(corrected) & (corrected > 0)
        good = np.all(~active | good_taps, axis=1)
        low = np.min(np.where(active & good_taps, corrected, np.inf), axis=1)
        high = np.max(np.where(active & good_taps, corrected, -np.inf), axis=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            good &= np.log(high / low) <= .08
            measured = np.sum(np.where(active & good_taps, corrected, 0.) * weights, axis=1)
            good &= np.abs(np.log(measured / z[ids])) <= .08
        valid[ids] = good
    return uv.astype(np.float32), valid


def _identity_rows(rows: Any, label: str) -> dict[str, dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f'missing {label} image identities')
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('name'), str) or row['name'] in result:
            raise ValueError(f'duplicate or malformed {label} image identity')
        result[row['name']] = row
    return result


def _publication_rows(*, dense: Any, sidecar: Mapping[str, Any], request: Mapping[str, Any],
                      request_json: Path, metric_scale: Path, raw_depth: Path,
                      scales: Mapping[str, Any], rec: Any) -> dict[str, dict[str, Any]]:
    """Bind each view to the original immutable request/raw/native MAPPING capture."""
    metadata = dense.metadata
    for key in ('geometry_run_sha256', 'raw_depth_manifest_sha256', 'keyframes_sha256',
                'request_file_sha256', 'inference_provenance'):
        if not metadata.get(key) or metadata[key] != sidecar.get(key):
            raise ValueError(f'dense/sidecar provenance mismatch: {key}')
    artifacts = metadata.get('independent_artifacts')
    if (not isinstance(artifacts, list)
            or any(not isinstance(artifact, dict) or not isinstance(artifact.get('family'), str)
                   or not isinstance(artifact.get('manifest_sha256'), str) for artifact in artifacts)):
        raise ValueError('dense map lacks exact independent artifact receipts')
    receipts = {artifact['family']: artifact['manifest_sha256'] for artifact in artifacts}
    geometry_run = sidecar['geometry_validation']['geometry_run']
    expected_receipts = {family: geometry_run[key] for family, key in
        (('reciprocal_photo', 'photo_evidence_sha256'), ('fixed_pose_mvs', 'mvs_evidence_sha256'))
        if geometry_run.get(key) is not None}
    if len(receipts) != len(artifacts) or receipts != expected_receipts:
        raise ValueError('dense independent artifacts differ from the signed geometry run')
    if sha256_file(request_json) != sidecar['request_file_sha256']:
        raise ValueError('original request file SHA256 mismatch')
    raw_manifest_path = metric_scale.with_name('raw_depth_manifest.json')
    if sha256_file(raw_manifest_path) != sidecar['raw_depth_manifest_sha256']:
        raise ValueError('immutable raw depth manifest SHA256 mismatch')
    raw_manifest = json.loads(raw_manifest_path.read_text())
    origins = sidecar['inference_provenance']['requests']
    if (raw_manifest.get('schema') != 'moge-immutable-raw-depth/v1'
            or raw_manifest.get('request_sha256') != dense.request_sha256
            or raw_manifest.get('inference_provenance') != origins):
        raise ValueError('immutable raw depth manifest original request mismatch')
    original_rows = {}
    for digest, origin in origins.items():
        path = Path(origin['request_path'])
        if sha256_file(path) != origin['request_file_sha256']:
            raise ValueError('original inference request bytes changed')
        original = json.loads(path.read_text())
        if (original.get('request_sha256') != digest
                or request_digest(original['payload'], original['config']) != digest
                or original['config'] != origin['config']
                or original['config'].get('model_hashes') != dense.model_hashes
                or original['config'].get('keyframes_sha256') != metadata['keyframes_sha256']):
            raise ValueError('original inference request semantic/config mismatch')
        original_rows[digest] = _identity_rows(original['payload'].get('images'), 'original request')
    requested = _identity_rows(request['payload'].get('images'), 'publication request')
    rows = _identity_rows(raw_manifest.get('images'), 'immutable raw manifest')
    images = {image.name: (int(image_id), image) for image_id, image in rec.images.items()}
    unsupported = {entry['name']: entry for entry in metadata.get('unsupported_sources', [])}
    evidence_names = set(metadata['view_names'])
    registered = _identity_rows(metadata.get('registered_capture_identity'), 'frozen registered MAPPING')
    if (set(rows) != set(requested) or set(rows) != set(scales) or set(rows) != set(images)
            or set(rows) != set(registered) or evidence_names & set(unsupported)
            or set(rows) != evidence_names | set(unsupported)):
        raise ValueError('registered MAPPING/request/raw/scale/view identities differ')
    root = raw_depth.resolve(strict=True)
    view_by_name = {name: i for i, name in enumerate(metadata['view_names'])}
    calibrated = {name for name, scale in scales.items() if scale.get('status') == 'SCALE_OK'}
    if (set(metadata.get('per_image_log_scales', {})) != calibrated
            or set(metadata.get('depth_calibration_digests', {})) != calibrated):
        raise ValueError('dense calibration/scale allowlist differs from validated MAPPING images')
    for name, row in rows.items():
        validate_mapping_record(row)
        scale = scales[name]
        validate_mapping_record(scale)
        image_id, image = images[name]
        view = view_by_name.get(name)
        capture = metadata['view_capture_identity'][view] if view is not None else unsupported[name]
        if (capture.get('image_id') != image_id
                or (view is not None and int(dense.observations['view_image_id'][view]) != image_id)
                or any(capture.get(key) != row.get(key) for key in
                       ('name', 'session_id', 'frame_index', 'evaluation_role', 'image_sha256'))):
            raise ValueError('dense supported/unsupported source differs from its registered MAPPING capture')
        for other in (requested[name], scale):
            if (other.get('evaluation_role', 'MAPPING') != 'MAPPING'
                    or any(other.get(key) != row.get(key) for key in
                           ('name', 'session_id', 'frame_index', 'image_sha256', 'width', 'height'))
                    or not np.array_equal(np.asarray(other.get('K_source')), np.asarray(row['K_source']))):
                raise ValueError('MAPPING source pixels/intrinsics/grouping provenance mismatch')
            if Path(other.get('image_path', '')).resolve(strict=True) != Path(row['image_path']).resolve(strict=True):
                raise ValueError('native MAPPING image path differs from its original capture')
        camera = rec.cameras[image.camera_id]
        if str(camera.model).split('.')[-1] != 'PINHOLE':
            raise ValueError('trusted bank requires undistorted PINHOLE references')
        k = validate_pinhole(camera.calibration_matrix(), int(camera.width), int(camera.height))
        if (int(camera.width) != row['width'] or int(camera.height) != row['height']
                or not np.allclose(k, row['K_source'], rtol=0, atol=1e-8)):
            raise ValueError('registered native source/camera provenance mismatch')
        if view is None:
            try:
                validate_pose(image.cam_from_world().matrix())
            except ValueError:
                if (scale.get('status') != 'INVALID_CAMERA_POSE'
                        or scale.get('unsupported_reason') != capture['reason']):
                    raise ValueError('unsupported registered source exclusion/scale provenance mismatch') from None
            else:
                raise ValueError('proper registered camera was relabelled as unsupported')
        else:
            pose = validate_pose(image.cam_from_world().matrix())
            center = -pose[:, :3].T @ pose[:, 3]
            if (not np.allclose(pose, scale.get('cam_from_world'), rtol=0, atol=1e-8)
                    or not np.allclose(center, scale.get('camera_center'), rtol=0, atol=1e-8)
                    or not np.allclose(center, dense.observations['view_camera_center'][view], rtol=0, atol=1e-8)):
                raise ValueError('reference camera/pose provenance mismatch')
        if sha256_file(Path(row['image_path'])) != row['image_sha256']:
            raise ValueError('native MAPPING image SHA256 mismatch')
        raw_path = (root / (safe_image_name(name).as_posix() + '.npz')).resolve(strict=True)
        digest = row.get('raw_depth_sha256')
        origin = row.get('raw_request_sha256')
        if (not raw_path.is_relative_to(root) or Path(row.get('raw_depth_path', '')).resolve() != raw_path
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)
                or origin not in original_rows or sha256_file(raw_path) != digest):
            raise ValueError('immutable raw depth path/SHA256/original request provenance mismatch')
        original = original_rows[origin].get(name)
        if (original is None or original.get('evaluation_role', 'MAPPING') != 'MAPPING'
                or any(original.get(key) != row.get(key) for key in
                       ('name', 'session_id', 'frame_index', 'image_sha256', 'width', 'height'))
                or not np.array_equal(np.asarray(original.get('K_source')), np.asarray(row['K_source']))):
            raise ValueError('original raw request source identity mismatch')
        if Path(original.get('image_path', '')).resolve(strict=True) != Path(row['image_path']).resolve(strict=True):
            raise ValueError('original raw request native pixel source path mismatch')
        if scale['status'] == 'SCALE_OK':
            calibration = scale.get('depth_calibration', {})
            validate_depth_calibration(calibration)
            if calibration.get('digest') != metadata['depth_calibration_digests'].get(name):
                raise ValueError('reference depth calibration digest mismatch')
            frozen_scale = metadata.get('per_image_log_scales', {}).get(name)
            if (isinstance(frozen_scale, bool) or not isinstance(frozen_scale, (int, float))
                    or not math.isfinite(frozen_scale) or isinstance(scale.get('log_scale'), bool)
                    or scale.get('log_scale') != frozen_scale):
                raise ValueError('reference per-image log scale provenance mismatch')
    return rows


def _reference_observations(arrays: Mapping[str, np.ndarray], selected: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate only real source/compatible view observations, retaining native associations."""
    offsets = arrays['observation_offsets']
    owners = np.repeat(np.arange(len(offsets) - 1), np.diff(offsets).astype(np.intp))
    image_ids = arrays['view_image_id'][arrays['observation_view_index']]
    flags = arrays['observation_flags']
    compatible = OBS_SOURCE | OBS_DEPTH_AGREE | OBS_PHOTO_AGREE | OBS_MVS_AGREE | OBS_SPARSE_AGREE
    admitted = np.zeros(len(offsets) - 1, bool)
    admitted[selected] = True
    pairs = np.column_stack((owners, image_ids))
    good = admitted[owners] & ((flags & compatible) != 0) & ((flags & (OBS_CONFLICT | OBS_OCCLUDED)) == 0)
    bad = pairs[(flags & (OBS_CONFLICT | OBS_OCCLUDED)) != 0]
    pairs = np.unique(pairs[good], axis=0)
    if len(bad) and len(pairs):
        pair_dtype = np.dtype([('row', '<i8'), ('image', '<i8')])
        bad_keys = np.ascontiguousarray(bad, dtype=np.int64).view(pair_dtype).reshape(-1)
        keys = np.ascontiguousarray(pairs, dtype=np.int64).view(pair_dtype).reshape(-1)
        pairs = pairs[~np.isin(keys, bad_keys)]
    surface_order = np.argsort(arrays['surface_id'])
    surface_ids = arrays['surface_id'][surface_order]
    reference_ids = arrays['reference_surface_id']
    positions = np.searchsorted(surface_ids, reference_ids)
    if np.any(positions >= len(surface_ids)) or np.any(surface_ids[positions] != reference_ids):
        raise ValueError('retained reference has unknown stable surface identity')
    reference_rows = surface_order[positions]
    retained = admitted[reference_rows]
    reference_pairs = np.column_stack((reference_rows[retained], arrays['reference_image_id'][retained]))
    reference_uv = arrays['reference_uv'][retained]
    pair_dtype = np.dtype([('row', '<i8'), ('image', '<i8')])
    keys = np.ascontiguousarray(pairs, dtype=np.int64).view(pair_dtype).reshape(-1)
    reference_keys = np.ascontiguousarray(reference_pairs, dtype=np.int64).view(pair_dtype).reshape(-1)
    if len(reference_keys) and not np.isin(reference_keys, keys).all():
        raise ValueError('retained reference lacks a real compatible source/view observation')
    source_pairs = np.column_stack((selected, arrays['source_image_id'][selected]))
    source_keys = np.ascontiguousarray(source_pairs, dtype=np.int64).view(pair_dtype).reshape(-1)
    if not np.isin(source_keys, reference_keys).all():
        raise ValueError('selected surface lacks its retained native source pixel association')
    projected_pairs = pairs[~np.isin(keys, reference_keys)]
    combined = np.vstack((reference_pairs, projected_pairs))
    uv = np.vstack((reference_uv, np.full((len(projected_pairs), 2), np.nan, np.float32)))
    # Remove identical associations only; never merge layers or rewrite surface IDs.
    identities = np.column_stack((combined, uv))
    _, indexes = np.unique(identities, axis=0, return_index=True)
    combined, uv = combined[indexes], uv[indexes]
    order = np.lexsort((uv[:, 1], uv[:, 0], combined[:, 0], combined[:, 1]))
    return combined[order, 0].astype(np.intp), combined[order, 1].astype(np.int32), uv[order]


def publish_from_artifacts(*, dense_map: Path, model_dir: Path, raw_depth: Path,
                           image_scales: Path, request_json: Path, out_dir: Path,
                           metric_scale: Path) -> dict[str, Any]:
    import pycolmap

    dense = load_dense_map(dense_map)
    if not dense.metadata or dense.metadata.get('validation_version') != VALIDATION_VERSION:
        raise ValueError(f'dense map must use {VALIDATION_VERSION} verification')
    required = set(LEDGER_DTYPES) | {'xyz', 'confidence', 'support_count', 'normals', 'source_image_id'}
    if (not dense.evidence or not required <= dense.evidence.keys()
            or not dense.observations or not dense.reference_observations):
        raise ValueError('dense map lacks declared real source/evidence/observation identities')
    hashes = model_file_hashes(model_dir)
    if set(hashes) != {'cameras.bin', 'images.bin', 'points3D.bin', 'rigs.bin', 'frames.bin'}:
        raise ValueError('accuracy publication requires all five signed COLMAP model binaries')
    request = json.loads(request_json.read_text())
    digest = request['request_sha256']
    if digest != request_digest(request['payload'], request['config']):
        raise ValueError('metric depth request digest mismatch')
    scale_digest = sha256_file(metric_scale)
    if dense.model_hashes != hashes or dense.request_sha256 != digest:
        raise ValueError('dense map model/request hash mismatch')
    if dense.metric_scale_sha256 != scale_digest:
        raise ValueError('dense map metric scale hash mismatch')
    sidecar = load_metric_scale(metric_scale, model_dir)
    config = request['config']
    if (
        sidecar['checkpoint_id'] != config.get('model_id')
        or sidecar['checkpoint_revision'] != config.get('checkpoint_revision')
        or sidecar['checkpoint_sha256'] != config.get('checkpoint_sha256')
        or sidecar['moge_source_hashes'] != config.get('moge_source_hashes')
        or sidecar['worker_sha256'] != config.get('worker_sha256')
        or any(sidecar.get(key) != config.get(key) for key in INFERENCE_SETTINGS)
        or sidecar['request_sha256'] != digest
        or sidecar['model_hashes'] != hashes
        or config.get('model_hashes') != hashes
        or config.get('keyframes_sha256') != sidecar['keyframes_sha256']
    ):
        raise ValueError('metric sidecar model/checkpoint mismatch')
    scales = _identity_rows([json.loads(line) for line in image_scales.read_text().splitlines()
                             if line.strip()], 'image scales')
    rec = pycolmap.Reconstruction(str(model_dir))
    rows = _publication_rows(dense=dense, sidecar=sidecar, request=request,
        request_json=request_json, metric_scale=metric_scale, raw_depth=raw_depth, scales=scales, rec=rec)
    arrays = {**dense.evidence, **dense.observations, **dense.reference_observations}
    selected = np.flatnonzero((dense.support_count >= MIN_SUPPORT)
                             & (dense.confidence >= MIN_CONFIDENCE))
    if not len(selected):
        raise ValueError('no trusted surface candidates')
    indexes, reference_ids, native_uv = _reference_observations(arrays, selected)
    if not len(indexes):
        raise ValueError('no real compatible reference observations')
    bounds = np.r_[0, np.flatnonzero(np.diff(reference_ids)) + 1, len(reference_ids)]
    kept_rows, kept_images, kept_pixels = [], [], []
    source_pass = np.zeros(len(dense), bool)
    # Group real observation views once; use their original raw paths and
    # hashes. Source QC is required even for publication in another view.
    for ordinal, (start, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        image_id = int(reference_ids[start])
        image = rec.images[image_id]
        name = image.name
        row, scale = rows[name], scales[name]
        if scale['status'] != 'SCALE_OK':
            continue  # Uncalibrated captures remain unknown, never native reference rows.
        camera = rec.cameras[image.camera_id]
        raw = _load_raw_npz(Path(row['raw_depth_path']), row, expected_digest=digest)
        log_scale = scale.get('log_scale')
        if not isinstance(log_scale, (int, float)) or not math.isfinite(log_scale):
            raise ValueError(f'invalid reference scale: {name}')
        uv, valid = reference_candidates(dense.xyz[indexes[start:end]], image.cam_from_world().matrix(),
            np.asarray(camera.calibration_matrix()), raw, math.exp(log_scale),
            (int(camera.width), int(camera.height)), calibration=scale['depth_calibration'],
            observed_uv=native_uv[start:end])
        kept = indexes[start:end][valid]
        if len(kept):
            source_pass[kept[dense.source_image_id[kept] == image_id]] = True
            kept_rows.append(kept)
            kept_images.append(np.full(len(kept), image_id, np.int32))
            kept_pixels.append(uv[valid])
        if ordinal % 100 == 0:
            print(f'[trusted-bank] references={ordinal + 1}/{len(bounds)-1}', flush=True)
    if not kept_rows:
        raise ValueError('no verified points passed strict reference validation')
    indexes, reference_ids, native_uv = map(np.concatenate, (kept_rows, kept_images, kept_pixels))
    good = source_pass[indexes]
    indexes, reference_ids, native_uv = indexes[good], reference_ids[good], native_uv[good]
    if not len(indexes):
        raise ValueError('no verified points passed native calibrated source validation')
    identities = np.empty(len(indexes), dtype=[('image', '<i4'), ('surface', '<i8'),
                                               ('u', '<f4'), ('v', '<f4')])
    identities['image'], identities['surface'] = reference_ids, arrays['surface_id'][indexes]
    identities['u'], identities['v'] = native_uv.T
    _, unique_rows = np.unique(identities, return_index=True)
    unique_rows.sort()
    indexes, reference_ids, native_uv = indexes[unique_rows], reference_ids[unique_rows], native_uv[unique_rows]
    payload = _select_ledger_rows(arrays, indexes, dense.metadata)
    payload.update(reference_xy=native_uv, landmark_id=arrays['surface_id'][indexes],
        reference_surface_id=arrays['surface_id'][indexes], reference_image_id=reference_ids,
        reference_uv=native_uv)
    references, names = {}, []
    bounds = np.r_[0, np.flatnonzero(np.diff(reference_ids)) + 1, len(reference_ids)]
    for start, end in zip(bounds[:-1], bounds[1:]):
        image_id = int(reference_ids[start])
        image = rec.images[image_id]
        name, camera = image.name, rec.cameras[image.camera_id]
        if len(name.encode()) > 80:
            raise ValueError('reference image name exceeds storage limit')
        names.append(np.full(end - start, name, dtype='S80'))
        references[name] = {'start': int(start), 'stop': int(end), 'image_id': image_id,
            'width': int(camera.width), 'height': int(camera.height),
            'K_source': np.asarray(camera.calibration_matrix()).tolist(),
            **{key: rows[name][key] for key in ('image_sha256', 'session_id', 'frame_index', 'evaluation_role')}}
        references[name].update(camera_id=int(image.camera_id),
            cam_from_world=validate_pose(image.cam_from_world().matrix()).tolist(),
            raw_depth_sha256=rows[name]['raw_depth_sha256'], raw_request_sha256=rows[name]['raw_request_sha256'],
            log_scale=scales[name]['log_scale'], depth_calibration_digest=scales[name]['depth_calibration']['digest'])
    payload['reference_name'] = np.concatenate(names)
    count = len(indexes)
    dense_digest = sha256_file(dense_map)
    source_provenance = {}
    unsupported = {entry['name']: entry for entry in dense.metadata.get('unsupported_sources', [])}
    for image_id, image in rec.images.items():
        name = image.name
        row, scale = rows[name], scales[name]
        if name in unsupported:
            source_provenance[name] = {**unsupported[name],
                **{key: row[key] for key in ('raw_depth_sha256', 'raw_request_sha256')},
                'status': 'UNSUPPORTED_GEOMETRY'}
            continue
        source_provenance[name] = {
            **{key: row[key] for key in ('name', 'session_id', 'frame_index', 'evaluation_role',
                'image_sha256', 'width', 'height', 'K_source', 'raw_depth_sha256', 'raw_request_sha256')},
            'image_id': int(image_id), 'camera_id': int(image.camera_id),
            'cam_from_world': validate_pose(image.cam_from_world().matrix()).tolist(), 'status': scale['status'],
        }
        if scale['status'] == 'SCALE_OK':
            source_provenance[name].update(log_scale=scale['log_scale'],
                depth_calibration_digest=scale['depth_calibration']['digest'])
    provenance = {
        'source_sha256': dense_digest, 'dense_map_sha256': dense_digest,
        'metric_scale_sha256': scale_digest, 'request_sha256': digest,
        'view_source_provenance': source_provenance,
        'unsupported_sources': list(unsupported.values()),
        'registered_capture_identity': dense.metadata['registered_capture_identity'],
        'image_scales_sha256': sha256_file(image_scales), 'model_hashes': hashes,
        'evaluation_role': 'MAPPING',
        'geometry_run': sidecar['geometry_validation']['geometry_run'],
        **{key: dense.metadata[key] for key in ('geometry_run_sha256', 'raw_depth_manifest_sha256',
            'keyframes_sha256', 'request_file_sha256', 'inference_provenance',
            'depth_calibration_digests', 'per_image_log_scales')},
    }
    metadata = {**dense.metadata, **provenance, 'published_reference_rows': count,
                'published_surface_count': int(len(np.unique(payload['landmark_id'])))}
    payload['metadata'] = np.array(json.dumps(metadata, allow_nan=False, sort_keys=True))
    manifest = {'schema': SCHEMA, **provenance, 'evidence_version': EVIDENCE_VERSION,
        'validation_version': VALIDATION_VERSION, 'ledger_schema': LEDGER_SCHEMA,
        'score_semantics': SCORE_SEMANTICS, 'validated_geometry_only': True,
        'network_or_depth_inference': False, 'geometry_contract': {
            'coordinates': 'world_map_units', 'pixel_transform': 'native_reference_xy',
            'confidence_meaning': SCORE_SEMANTICS, 'surface_identity': 'stable_offline_surface_id'},
        'min_support': MIN_SUPPORT, 'min_confidence': MIN_CONFIDENCE,
        'reference_names': list(references), 'references': references,
        'num_surface_ids': metadata['published_surface_count']}
    _validate_accuracy_ledger(payload, manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    temporary = out_dir / '.landmarks.tmp.npz'
    np.savez_compressed(temporary, **payload)
    target = out_dir / 'landmarks.npz'
    temporary.replace(target)
    manifest['landmarks'] = {'path': target.name, 'sha256': sha256_file(target), 'count': count}
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print(f'[trusted-bank] complete points={count} surfaces={manifest["num_surface_ids"]}', flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('dense-map', 'model-dir', 'raw-depth', 'image-scales', 'request-json', 'out-dir', 'metric-scale'):
        parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args()
    manifest = publish_from_artifacts(**vars(args))
    print(json.dumps({'points': manifest['landmarks']['count'], 'references': len(manifest['references'])}))


if __name__ == '__main__':
    main()
