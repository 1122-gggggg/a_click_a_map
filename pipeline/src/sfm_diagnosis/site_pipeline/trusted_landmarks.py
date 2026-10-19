"""Hash-bound offline reference landmarks with bounded cached pixel indexes."""
from __future__ import annotations

import hashlib
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA = 'trusted-reference-landmarks/v1'
ACCURACY_VERSION = 'correlation-visibility-ledger/v4'
_OBSERVATION_FIELDS = (
    'observation_view_index', 'observation_residual_view_index', 'observation_flags', 'observation_residual',
    'observation_family_mask', 'observation_evidence_id', 'observation_group_index',
)
_VIEW_FIELDS = ('view_image_id', 'view_camera_center', 'view_session_index', 'view_source_index')
_REFERENCE_FIELDS = ('reference_surface_id', 'reference_image_id', 'reference_uv')


def _select_ledger_rows(arrays: Mapping[str, np.ndarray], rows: np.ndarray,
                        metadata: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Gather aligned geometry and the original variable-length CSR segments."""
    rows = np.asarray(rows, dtype=np.intp)
    result = {name: arrays[name][rows] for name in metadata['ledger_fields']}
    result.update({name: arrays[name] for name in _VIEW_FIELDS})
    offsets = arrays['observation_offsets']
    counts = (offsets[rows + 1] - offsets[rows]).astype(np.intp)
    new_offsets = np.empty(len(rows) + 1, dtype=np.uint64)
    new_offsets[0] = 0
    np.cumsum(counts, dtype=np.uint64, out=new_offsets[1:])
    indexes = (np.repeat(offsets[rows].astype(np.intp), counts)
               + np.arange(int(new_offsets[-1]), dtype=np.intp)
               - np.repeat(new_offsets[:-1].astype(np.intp), counts))
    result['observation_offsets'] = new_offsets
    result.update({name: arrays[name][indexes] for name in _OBSERVATION_FIELDS})
    return result


def _validate_accuracy_ledger(arrays: Mapping[str, np.ndarray],
                              manifest: Mapping[str, Any]) -> None:
    # Legacy banks never import or acquire the newer offline geometry contract.
    from .geometry_contract import (
        LEDGER_DTYPES, LEDGER_SCHEMA, SCORE_SEMANTICS, OBS_SOURCE, OBS_DEPTH_AGREE,
        OBS_PHOTO_AGREE, OBS_MVS_AGREE, OBS_SPARSE_AGREE, OBS_CONFLICT, OBS_OCCLUDED,
        geometry_policy_spec, validate_evidence_ledger, validate_mapping_record, validate_pinhole, validate_pose,
    )
    policy = geometry_policy_spec()
    observation_dtypes = {name: np.dtype(policy['observation_abi'][name]) for name in _OBSERVATION_FIELDS}

    encoded = arrays.get('metadata')
    if encoded is None or encoded.shape != () or encoded.dtype.kind not in 'SU':
        raise ValueError('accuracy bank lacks numeric ledger metadata')
    metadata = json.loads(encoded.item().decode() if encoded.dtype.kind == 'S' else encoded.item())
    if (not isinstance(metadata, dict) or metadata.get('validation_version') != ACCURACY_VERSION
            or manifest.get('validation_version') != ACCURACY_VERSION
            or metadata.get('ledger_schema') != LEDGER_SCHEMA
            or manifest.get('ledger_schema') != LEDGER_SCHEMA
            or metadata.get('score_semantics') != SCORE_SEMANTICS
            or manifest.get('score_semantics') != SCORE_SEMANTICS
            or metadata.get('evaluation_role') != 'MAPPING'
            or manifest.get('evaluation_role') != 'MAPPING'):
        raise ValueError('unsupported accuracy bank provenance/ledger contract')
    for name in ('source_sha256', 'dense_map_sha256', 'metric_scale_sha256', 'request_sha256',
                 'image_scales_sha256', 'geometry_run_sha256', 'raw_depth_manifest_sha256',
                 'keyframes_sha256', 'request_file_sha256'):
        value = metadata.get(name)
        if (not isinstance(value, str) or len(value) != 64
                or any(c not in '0123456789abcdef' for c in value)
                or value != manifest.get(name)):
            raise ValueError(f'accuracy bank provenance digest mismatch: {name}')
    if metadata['source_sha256'] != metadata['dense_map_sha256']:
        raise ValueError('accuracy bank source/dense geometry digest mismatch')
    for name in ('model_hashes', 'inference_provenance', 'depth_calibration_digests',
                 'per_image_log_scales', 'view_source_provenance', 'geometry_run'):
        if (not isinstance(metadata.get(name), dict) or not metadata[name]
                or metadata[name] != manifest.get(name)):
            raise ValueError(f'accuracy bank provenance identity mismatch: {name}')
    if (set(metadata['model_hashes']) != {'cameras.bin', 'images.bin', 'points3D.bin', 'rigs.bin', 'frames.bin'}
            or any(not isinstance(value, str) or len(value) != 64
                   or any(c not in '0123456789abcdef' for c in value)
                   for value in metadata['model_hashes'].values())):
        raise ValueError('accuracy bank model identity lacks SHA256')
    if any(not isinstance(value, str) or len(value) != 64
           or any(c not in '0123456789abcdef' for c in value)
           for value in metadata['depth_calibration_digests'].values()):
        raise ValueError('accuracy bank calibration identity lacks SHA256')
    scales = metadata['per_image_log_scales']
    if (set(scales) != set(metadata['depth_calibration_digests'])
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in scales.values())):
        raise ValueError('accuracy bank per-image scale provenance mismatch')
    provenance = metadata['inference_provenance']
    origins = provenance.get('requests', {})
    if (not isinstance(origins, dict) or not origins
            or provenance.get('schema') != 'moge-original-inference/v1'
            or metadata['request_sha256'] not in origins):
        raise ValueError('accuracy bank lacks original inference request provenance')
    for digest, origin in origins.items():
        if not isinstance(origin, dict):
            raise ValueError('accuracy bank malformed original request provenance')
        file_digest = origin.get('request_file_sha256')
        config = origin.get('config')
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)
                or origin.get('request_sha256') != digest
                or not isinstance(file_digest, str) or len(file_digest) != 64
                or any(c not in '0123456789abcdef' for c in file_digest)
                or not isinstance(config, dict) or config.get('model_hashes') != metadata['model_hashes']
                or config.get('keyframes_sha256') != metadata['keyframes_sha256']):
            raise ValueError('accuracy bank original request digest mismatch')
    if origins[metadata['request_sha256']]['request_file_sha256'] != metadata['request_file_sha256']:
        raise ValueError('accuracy bank original request byte provenance mismatch')
    run = metadata['geometry_run']
    encoded_run = (json.dumps(run, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()
    if (hashlib.sha256(encoded_run).hexdigest() != metadata['geometry_run_sha256']
            or run.get('schema') != 'moge-offline-geometry-run/v1'
            or any(run.get(key) != metadata.get(key) for key in
                   ('model_hashes', 'keyframes_sha256', 'raw_depth_manifest_sha256', 'calibration_contract'))):
        raise ValueError('accuracy bank geometry run model/source/policy binding mismatch')
    artifacts = metadata.get('independent_artifacts')
    if (not isinstance(artifacts, list)
            or any(not isinstance(artifact, dict) or not isinstance(artifact.get('family'), str)
                   or not isinstance(artifact.get('manifest_sha256'), str) for artifact in artifacts)):
        raise ValueError('accuracy bank malformed independent artifact receipts')
    receipts = {artifact['family']: artifact['manifest_sha256'] for artifact in artifacts}
    expected_receipts = {family: run[key] for family, key in
        (('reciprocal_photo', 'photo_evidence_sha256'), ('fixed_pose_mvs', 'mvs_evidence_sha256'))
        if run.get(key) is not None}
    if len(receipts) != len(artifacts) or receipts != expected_receipts:
        raise ValueError('accuracy bank independent artifacts differ from the signed geometry run')
    fields = metadata.get('ledger_fields')
    required_fields = set(LEDGER_DTYPES) | {'xyz', 'confidence', 'support_count', 'normals', 'source_image_id'}
    names = arrays.get('reference_name')
    if names is None or names.ndim != 1 or names.dtype.kind not in 'SU' or not len(names):
        raise ValueError('accuracy bank malformed native reference names')
    n = len(names)
    if not isinstance(fields, dict) or not required_fields <= fields.keys():
        raise ValueError('accuracy bank lacks aligned numeric geometry ledger')
    allowed = (set(fields) | set(_OBSERVATION_FIELDS) | set(_VIEW_FIELDS) | set(_REFERENCE_FIELDS)
               | {'reference_name', 'reference_xy', 'landmark_id', 'metadata', 'observation_offsets'})
    if set(arrays) != allowed:
        raise ValueError('invalid accuracy bank numeric array schema')
    reserved = (set(_OBSERVATION_FIELDS) | set(_VIEW_FIELDS) | set(_REFERENCE_FIELDS)
                | {'reference_name', 'reference_xy', 'landmark_id', 'metadata', 'observation_offsets'})
    if reserved & set(fields):
        raise ValueError('accuracy bank geometry declarations overlap observation/reference fields')
    for name, spec in fields.items():
        value = arrays[name]
        shape = spec.get('shape') if isinstance(spec, dict) else None
        if (not isinstance(shape, list) or any(type(size) is not int or size < 0 for size in shape)
                or value.dtype.kind not in 'biuf' or value.dtype.str != spec.get('dtype')
                or value.shape != (n, *shape)):
            raise ValueError(f'accuracy bank misaligned numeric field: {name}')
    for name, shape, dtype in (('reference_xy', (n, 2), np.dtype('<f4')),
                                ('landmark_id', (n,), np.dtype('<i8')),
                                ('reference_surface_id', (n,), np.dtype('<i8')),
                                ('reference_image_id', (n,), np.dtype('<i4')),
                                ('reference_uv', (n, 2), np.dtype('<f4'))):
        if arrays[name].shape != shape or arrays[name].dtype != dtype:
            raise ValueError(f'accuracy bank misaligned reference field: {name}')
    if (not np.array_equal(arrays['surface_id'], arrays['landmark_id'])
            or not np.array_equal(arrays['reference_surface_id'], arrays['landmark_id'])
            or not np.array_equal(arrays['reference_uv'], arrays['reference_xy'])):
        raise ValueError('accuracy bank surface/reference identity alignment mismatch')
    offsets = arrays.get('observation_offsets')
    if (offsets is None or offsets.dtype != np.dtype(policy['observation_abi']['observation_offsets'])
            or offsets.shape != (n + 1,)
            or offsets[0] != 0 or np.any(offsets[1:] <= offsets[:-1])):
        raise ValueError('accuracy bank malformed point/view observation CSR')
    m = int(offsets[-1])
    for name, dtype in observation_dtypes.items():
        if arrays[name].shape != (m,) or arrays[name].dtype != dtype:
            raise ValueError(f'accuracy bank malformed observation CSR field: {name}')
    # Multiple native reference rows may point to one stable surface. Validate
    # the actual unique geometry and require every repeated ledger/CSR to agree.
    _, first, inverse = np.unique(arrays['surface_id'], return_index=True, return_inverse=True)
    representatives = first[inverse]
    for name in fields:
        if not np.array_equal(arrays[name], arrays[name][representatives]):
            raise ValueError(f'accuracy bank repeated surface ledger mismatch: {name}')
    counts = np.diff(offsets).astype(np.intp)
    if not np.array_equal(counts, counts[representatives]):
        raise ValueError('accuracy bank repeated surface observation count mismatch')
    observation_representatives = (np.repeat(offsets[representatives].astype(np.intp), counts)
        + np.arange(m, dtype=np.intp) - np.repeat(offsets[:-1].astype(np.intp), counts))
    for name in _OBSERVATION_FIELDS:
        if not np.array_equal(arrays[name], arrays[name][observation_representatives]):
            raise ValueError(f'accuracy bank repeated surface observation mismatch: {name}')
    unique = _select_ledger_rows(arrays, first, metadata)
    unique.update({name: arrays[name] for name in _REFERENCE_FIELDS})
    validate_evidence_ledger(unique, len(first), metadata)
    captures = metadata['view_capture_identity']
    view_ids = arrays['view_image_id']
    view_by_id = {int(image_id): i for i, image_id in enumerate(view_ids)}
    cell_width = metadata.get('camera_cell_width')
    if (not isinstance(cell_width, (int, float)) or not math.isfinite(cell_width) or cell_width <= 0):
        raise ValueError('accuracy bank source capture grouping scale missing')
    for view, capture in enumerate(captures):
        session = metadata['view_sessions'][int(arrays['view_session_index'][view])]
        source = json.loads(metadata['view_sources'][int(arrays['view_source_index'][view])])
        expected = {'session': capture['session_id'],
                    'temporal_segment': capture['frame_index'] // policy['dense']['temporal_segment_frames'],
                    'camera_cell': np.floor(arrays['view_camera_center'][view] / cell_width).astype(int).tolist()}
        if session != capture['session_id'] or source != expected:
            raise ValueError('accuracy bank source/session grouping identity mismatch')
    source_provenance = metadata['view_source_provenance']
    registered = metadata.get('registered_capture_identity')
    unsupported = metadata.get('unsupported_sources', [])
    if (not isinstance(registered, list) or registered != manifest.get('registered_capture_identity')
            or unsupported != manifest.get('unsupported_sources')
            or set(source_provenance) != {capture['name'] for capture in registered}):
        raise ValueError('accuracy bank full supported/unsupported MAPPING allowlist mismatch')
    for capture in unsupported:
        entry = source_provenance[capture['name']]
        validate_mapping_record(entry)
        if (any(entry.get(key) != capture.get(key) for key in
                ('name', 'image_id', 'image_sha256', 'session_id', 'frame_index', 'evaluation_role', 'reason'))
                or entry.get('status') != 'UNSUPPORTED_GEOMETRY'
                or {'camera_id', 'width', 'height', 'K_source', 'cam_from_world', 'camera_center',
                    'log_scale', 'depth_calibration_digest'} & entry.keys()):
            raise ValueError('accuracy bank unsupported raw source acquired invented camera/calibrated geometry')
        raw_digest, raw_origin = entry.get('raw_depth_sha256'), entry.get('raw_request_sha256')
        if (not isinstance(raw_digest, str) or len(raw_digest) != 64
                or any(c not in '0123456789abcdef' for c in raw_digest)
                or not isinstance(raw_origin, str) or raw_origin not in origins):
            raise ValueError('accuracy bank unsupported raw source lacks immutable raw/request identity')
    calibrated = set()
    camera_identities = {}
    for view, capture in enumerate(captures):
        name = capture['name']
        entry = source_provenance[name]
        validate_mapping_record(entry)
        if (any(entry.get(key) != capture.get(key) for key in
                ('name', 'image_id', 'image_sha256', 'session_id', 'frame_index', 'evaluation_role'))
                or type(entry.get('camera_id')) is not int or entry['camera_id'] < 0
                or type(entry.get('width')) is not int or type(entry.get('height')) is not int):
            raise ValueError('accuracy bank source/camera capture identity mismatch')
        k = validate_pinhole(np.asarray(entry.get('K_source')), entry['width'], entry['height'])
        pose = validate_pose(np.asarray(entry.get('cam_from_world')))
        center = -pose[:, :3].T @ pose[:, 3]
        if not np.allclose(center, arrays['view_camera_center'][view], rtol=0, atol=1e-8):
            raise ValueError('accuracy bank camera pose/center provenance mismatch')
        camera = (entry['width'], entry['height'], k.tolist())
        if camera_identities.setdefault(entry['camera_id'], camera) != camera:
            raise ValueError('accuracy bank signed camera ID has inconsistent intrinsics')
        raw_digest, raw_origin = entry.get('raw_depth_sha256'), entry.get('raw_request_sha256')
        if (not isinstance(raw_digest, str) or len(raw_digest) != 64
                or any(c not in '0123456789abcdef' for c in raw_digest)
                or not isinstance(raw_origin, str) or raw_origin not in origins):
            raise ValueError('accuracy bank source lacks immutable raw/request identities')
        if not isinstance(entry.get('status'), str) or not entry['status']:
            raise ValueError('accuracy bank source lacks its original calibration status')
        if entry['status'] == 'SCALE_OK':
            calibrated.add(name)
            if (isinstance(entry.get('log_scale'), bool) or entry.get('log_scale') != scales.get(name)
                    or entry.get('depth_calibration_digest') != metadata['depth_calibration_digests'].get(name)):
                raise ValueError('accuracy bank source log scale/correction digest mismatch')
        elif 'log_scale' in entry or 'depth_calibration_digest' in entry:
            raise ValueError('accuracy bank uncalibrated source acquired calibrated geometry')
    if calibrated != set(scales):
        raise ValueError('accuracy bank calibrated source allowlist mismatch')
    references = manifest.get('references')
    if not isinstance(references, dict) or set(references) != set(manifest.get('reference_names', [])):
        raise ValueError('accuracy bank lacks exact native reference identities')
    for name, entry in references.items():
        if not isinstance(entry, dict):
            raise ValueError('accuracy bank malformed native reference identity')
        view = view_by_id.get(entry.get('image_id'))
        if view is None or metadata['view_names'][view] != name:
            raise ValueError('accuracy bank reference/view identity mismatch')
        capture = captures[view]
        if any(entry.get(key) != capture.get(key) for key in
               ('image_sha256', 'session_id', 'frame_index', 'evaluation_role')):
            raise ValueError('accuracy bank reference MAPPING provenance mismatch')
        width, height = entry.get('width'), entry.get('height')
        if type(width) is not int or type(height) is not int:
            raise ValueError('accuracy bank native reference dimensions missing')
        validate_pinhole(np.asarray(entry.get('K_source')), width, height)
        source = source_provenance[name]
        if (source['status'] != 'SCALE_OK' or any(entry.get(key) != source.get(key) for key in
                ('image_id', 'camera_id', 'width', 'height', 'K_source', 'cam_from_world',
                 'raw_depth_sha256', 'raw_request_sha256', 'log_scale', 'depth_calibration_digest'))):
            raise ValueError('accuracy bank reference differs from its signed source camera/calibration')
        start, stop = entry.get('start'), entry.get('stop')
        if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= n:
            raise ValueError('accuracy bank malformed native reference row spans')
        identity = name.encode() if arrays['reference_name'].dtype.kind == 'S' else name
        if (not np.all(arrays['reference_name'][start:stop] == identity)
                or not np.all(arrays['reference_image_id'][start:stop] == int(view_ids[view]))
                or name not in metadata['depth_calibration_digests']):
            raise ValueError('accuracy bank reference names/image IDs/calibration disagree')
        uv = arrays['reference_xy'][start:stop]
        if np.any(uv < 0) or np.any(uv[:, 0] >= width) or np.any(uv[:, 1] >= height):
            raise ValueError('accuracy bank native reference pixels outside camera')
        pose = validate_pose(np.asarray(source['cam_from_world']))
        camera_xyz = arrays['xyz'][start:stop] @ pose[:, :3].T + pose[:, 3]
        z = camera_xyz[:, 2]
        projected = camera_xyz @ np.asarray(source['K_source']).T
        projected = projected[:, :2] / np.where(z > 0, z, np.nan)[:, None]
        if np.any(z <= 0) or not np.isfinite(projected).all() or np.any(np.linalg.norm(projected - uv, axis=1) > 2.):
            raise ValueError('accuracy bank retained native pixel association contradicts signed camera pose')
    if not np.isin(arrays['reference_image_id'], [entry['image_id'] for entry in references.values()]).all():
        raise ValueError('accuracy bank contains undeclared reference identities')
    observation_images = view_ids[arrays['observation_view_index']]
    flags = arrays['observation_flags']
    compatible = OBS_SOURCE | OBS_DEPTH_AGREE | OBS_PHOTO_AGREE | OBS_MVS_AGREE | OBS_SPARSE_AGREE
    owners = np.repeat(np.arange(n), counts)
    source = ((observation_images == arrays['source_image_id'][owners]) & ((flags & OBS_SOURCE) != 0))
    target = observation_images == arrays['reference_image_id'][owners]
    supported = target & ((flags & compatible) != 0)
    contradicted = target & ((flags & (OBS_CONFLICT | OBS_OCCLUDED)) != 0)
    if (np.any(np.bincount(owners[source], minlength=n) == 0)
            or np.any(np.bincount(owners[supported], minlength=n) == 0) or contradicted.any()):
        raise ValueError('accuracy bank reference is not a real compatible retained observation')


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def publish_trusted_landmark_bank(rows: Sequence[Mapping[str, Any]], out_dir: str | Path,
                                  *, source_sha256: str | None = None) -> dict[str, Any]:
    """Small-bank convenience publisher; bulk River publication uses arrays."""
    if not rows or not source_sha256:
        raise ValueError('validated rows and source hash are required')
    if any(r.get('validated_geometry') is not True or r.get('network_or_depth_inference', False)
           for r in rows):
        raise ValueError('all rows must explicitly have validated offline geometry')
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / 'landmarks.npz'
    np.savez_compressed(path, reference_name=np.asarray([r['reference_name'] for r in rows]),
        reference_xy=np.asarray([r['reference_xy'] for r in rows], np.float32),
        xyz=np.asarray([r['xyz'] for r in rows], np.float32),
        confidence=np.asarray([r['confidence'] for r in rows], np.float32),
        support_count=np.asarray([r['support_count'] for r in rows], np.int64),
        landmark_id=np.asarray([r.get('landmark_id', i) for i, r in enumerate(rows)], np.int64))
    manifest = {'schema': SCHEMA, 'landmarks': {'path': path.name, 'sha256': _sha(path),
        'count': len(rows)}, 'source_sha256': source_sha256,
        'model_hashes': dict(rows[0].get('model_hashes', {})),
        'validated_geometry_only': True, 'network_or_depth_inference': False,
        'evidence_version': 'trusted-landmarks-publisher/v2',
        'reference_names': sorted({r['reference_name'] for r in rows})}
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    TrustedLandmarkBank(path)  # Validate before returning a published bank.
    return manifest


class TrustedLandmarkBank:
    """Validate once; query only a cached per-reference KD-tree at runtime."""
    def __init__(self, path: str | Path, *, max_cache: int = 32):
        from scipy.spatial import cKDTree

        self.path = Path(path)
        manifest_path = self.path.with_name('manifest.json')
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get('schema') != SCHEMA or manifest.get('validated_geometry_only') is not True
                or manifest.get('network_or_depth_inference') is not False
                or not manifest.get('evidence_version') or not manifest.get('model_hashes')
                or not manifest.get('source_sha256')):
            raise ValueError('unvalidated trusted landmark manifest')
        declared = manifest.get('landmarks', {})
        if declared.get('path') != self.path.name or declared.get('sha256') != _sha(self.path):
            raise ValueError('trusted landmark path/SHA-256 mismatch')
        accuracy = manifest.get('validation_version') == ACCURACY_VERSION or 'ledger_schema' in manifest
        with np.load(self.path, allow_pickle=False) as payload:
            required = {'reference_name', 'reference_xy', 'xyz', 'confidence', 'support_count'}
            if not required <= set(payload.files) or (
                    not accuracy and set(payload.files) - required - {'landmark_id'}):
                raise ValueError('invalid trusted landmark array schema')
            arrays = {name: payload[name] for name in payload.files} if accuracy else payload
            if accuracy:
                _validate_accuracy_ledger(arrays, manifest)
            self.names = arrays['reference_name']
            self.uv = arrays['reference_xy']
            self.xyz = arrays['xyz']
            self.confidence = arrays['confidence']
            self.support = arrays['support_count']
            self.ids = arrays['landmark_id'] if 'landmark_id' in arrays else np.arange(len(self.names))
        n = len(self.names)
        if (n == 0 or declared.get('count') != n or self.names.ndim != 1
                or self.names.dtype.kind not in 'SU' or self.uv.shape != (n, 2)
                or self.xyz.shape != (n, 3) or self.confidence.shape != (n,)
                or self.support.shape != (n,) or self.ids.shape != (n,)):
            raise ValueError('unaligned trusted landmark shapes/count')
        if (not np.isfinite(self.uv).all() or not np.isfinite(self.xyz).all()
                or not np.isfinite(self.confidence).all() or not np.isfinite(self.support).all()
                or np.any(self.confidence < .3) or np.any(self.confidence > 1)
                or np.any(self.support < 2) or np.any(self.support > 65535)
                or not np.issubdtype(self.support.dtype, np.integer)
                or not np.issubdtype(self.ids.dtype, np.integer) or np.any(self.ids < 0)):
            raise ValueError('invalid trusted landmark coordinates/confidence/support/ids')
        self._rows: dict[str, np.ndarray | slice] = {}
        references = manifest.get('references')
        if references:
            covered = 0
            for name, entry in sorted(references.items(), key=lambda item: item[1]['start']):
                start, stop = entry['start'], entry['stop']
                identity = name.encode() if self.names.dtype.kind == 'S' else name
                if (not isinstance(start, int) or not isinstance(stop, int)
                        or start != covered or not start < stop <= n
                        or not np.all(self.names[start:stop] == identity)):
                    raise ValueError('invalid reference row spans')
                self._rows[name] = slice(start, stop)
                covered = stop
            if covered != n:
                raise ValueError('incomplete reference row spans')
        else:
            # Small legacy test banks are indexed once, never on a hot lookup.
            for name in np.unique(self.names):
                label = name.decode() if isinstance(name, bytes) else str(name)
                self._rows[label] = np.flatnonzero(self.names == name)
        self._tree_type = cKDTree
        self._cache: OrderedDict[str, tuple[Any, np.ndarray]] = OrderedDict()
        self.max_cache = max(1, int(max_cache))
        self.manifest = manifest

    def lookup_evidence(self, reference_name: str, uv: np.ndarray, radius_px: float):
        if not math.isfinite(radius_px) or radius_px <= 0:
            raise ValueError('lookup radius must be finite and positive')
        query = np.asarray(uv, dtype=float).reshape(-1, 2)
        if not np.isfinite(query).all():
            raise ValueError('query pixels must be finite')
        n = len(query)
        ids, dist = np.full(n, -1, np.int64), np.full(n, np.inf)
        xyz, scores = np.full((n, 3), np.nan, np.float32), np.zeros(n, np.float32)
        if reference_name not in self._rows or not n:
            return ids, dist, xyz, scores
        if reference_name in self._cache:
            tree, rows = self._cache.pop(reference_name)
        else:
            selector = self._rows[reference_name]
            rows = (np.arange(selector.start, selector.stop) if isinstance(selector, slice) else selector)
            tree = self._tree_type(self.uv[rows])
        self._cache[reference_name] = (tree, rows)
        while len(self._cache) > self.max_cache:
            self._cache.popitem(last=False)
        distances, indexes = tree.query(query, distance_upper_bound=radius_px)
        good = np.isfinite(distances) & (indexes < len(rows))
        found = rows[indexes[good]]
        ids[good] = self.ids[found]
        dist[good] = distances[good]
        xyz[good] = self.xyz[found]
        scores[good] = self.confidence[found]
        return ids, dist, xyz, scores

    def lookup(self, reference_name: str, uv: np.ndarray, radius_px: float):
        return self.lookup_evidence(reference_name, uv, radius_px)[:3]
