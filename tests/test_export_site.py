"""Site export for the localization machine (needs the engine env: pycolmap, numpy)."""
from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pycolmap = pytest.importorskip("pycolmap")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from acam import export_site as export  # noqa: E402  (repo root may not be on sys.path)


def _mat_to_quat(matrix) -> list:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w, x, y, z = (0.25 / s, (matrix[2, 1] - matrix[1, 2]) * s,
                      (matrix[0, 2] - matrix[2, 0]) * s, (matrix[1, 0] - matrix[0, 1]) * s)
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = 2.0 * math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
        w, x, y, z = ((matrix[2, 1] - matrix[1, 2]) / s, 0.25 * s,
                      (matrix[0, 1] + matrix[1, 0]) / s, (matrix[0, 2] + matrix[2, 0]) / s)
    elif matrix[1, 1] > matrix[2, 2]:
        s = 2.0 * math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
        w, x, y, z = ((matrix[0, 2] - matrix[2, 0]) / s, (matrix[0, 1] + matrix[1, 0]) / s,
                      0.25 * s, (matrix[1, 2] + matrix[2, 1]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
        w, x, y, z = ((matrix[1, 0] - matrix[0, 1]) / s, (matrix[0, 2] + matrix[2, 0]) / s,
                      (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s)
    return [w, x, y, z]


def _synthetic_run(root: Path, n_images: int = 150, n_points: int = 400) -> Path:
    """A gimbal-like synthetic mapping run: cameras ring the scene, roll ~ 0.

    Writes COLMAP binaries directly (stable format, no pycolmap write API).
    """
    import numpy as np

    run = root / "run"
    model_dir = run / "artifacts/mapping/edm/model"
    model_dir.mkdir(parents=True)
    rng = np.random.default_rng(7)
    names = [f"vid_a/frame_{i:08d}.jpg" for i in range(n_images)]
    poses = []
    for i in range(n_images):
        yaw = 2 * math.pi * i / n_images
        t = np.array([20.0 * math.cos(yaw), 8.0, 20.0 * math.sin(yaw)])
        z = -t / np.linalg.norm(t)
        x = np.cross(z, [0.0, -1.0, 0.0])
        x = x / np.linalg.norm(x)
        rotation = np.stack([x, np.cross(z, x), z])
        poses.append((i + 1, _mat_to_quat(rotation), (-rotation @ t).tolist()))
    with (model_dir / "cameras.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", 1) + struct.pack("<iiQQ", 1, 1, 1920, 1080))
        stream.write(struct.pack("<4d", 1500.0, 1500.0, 960.0, 540.0))
    with (model_dir / "images.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", n_images))
        for image_id, quat, tvec in poses:
            stream.write(struct.pack("<i4d3di", image_id, *quat, *tvec, 1))
            stream.write(names[image_id - 1].encode() + b"\x00")
            stream.write(struct.pack("<Q", 0))
    points_xyz = rng.uniform(-5, 5, (n_points, 3))
    points_rgb = rng.integers(0, 255, (n_points, 3))
    with (model_dir / "points3D.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", n_points))
        for i in range(n_points):
            stream.write(struct.pack("<q3d3BdQ", i + 1, *points_xyz[i],
                                    *[int(v) for v in points_rgb[i]], 1.0, 0))
    images = run / "artifacts/keyframes/images/vid_a"
    images.mkdir(parents=True)
    with (run / "artifacts/keyframes/keyframes.jsonl").open("w", encoding="utf-8") as stream:
        for i in range(n_images):
            name = f"frame_{i:08d}.jpg"
            (images / name).write_bytes(b"\xff\xd8" + bytes(64) + b"\xff\xd9")
            stream.write(json.dumps({"video_id": "vid_a", "source_uri": f"vid_a/{name}"}) + "\n")
    bank = run / "products/localization"
    bank.mkdir(parents=True)
    (bank / "megaloc_references.names.json").write_text(json.dumps(names))
    np.save(bank / "megaloc_references.npy",
            rng.standard_normal((n_images, 8448)).astype("float32"))
    with (bank / "reference_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for name in names:
            stream.write(json.dumps({"image_name": name, "observations": 10}) + "\n")
    return run


def test_frame_digest_streams_in_order(tmp_path: Path) -> None:
    import hashlib
    d = tmp_path
    (d / "cameras.bin").write_bytes(b"c")
    (d / "images.bin").write_bytes(b"i")
    (d / "points3D.bin").write_bytes(b"p")
    want = hashlib.sha256(b"c" + b"i" + b"p").hexdigest()
    assert export.frame_digest(d) == want


def test_query_camera_rescales_to_stream() -> None:
    import numpy as np
    K = np.array([[1500.0, 0.0, 960.0], [0.0, 1500.0, 540.0], [0.0, 0.0, 1.0]])
    cam = export.query_camera_from_model_K(K, 1920, 1080)
    assert cam == {"model": "PINHOLE", "width": 1280, "height": 720,
                   "params": [1000.0, 1000.0, 640.0, 360.0]}
    with pytest.raises(ValueError, match="aspect"):
        export.query_camera_from_model_K(K, 1920, 1000)


def test_gravity_recovers_down() -> None:
    import numpy as np
    rotations = []
    for i in range(120):
        yaw = 2 * np.pi * i / 120
        z = -np.array([np.cos(yaw), 0.4, np.sin(yaw)])
        z = z / np.linalg.norm(z)
        x = np.cross(z, [0.0, -1.0, 0.0])
        x = x / np.linalg.norm(x)
        rotations.append(np.stack([x, np.cross(z, x), z]))
    rotation, evidence = export.derive_gravity_alignment(np.asarray(rotations))
    assert evidence["n_poses"] == 120 and evidence["residual_deg"] < 5.0
    assert -rotation[2][1] > 0.99  # anti-gravity points -Y in this synthetic world


def test_export_site_validates_against_localization_repo(tmp_path: Path) -> None:
    wulai = Path("/home/cihcilab/work/wulai-sfm-localization")
    if not (wulai / "定位演算法/deploy_code/sfm_direct_deploy/direct_map.py").exists():
        pytest.skip("localization repo not checked out")
    run = _synthetic_run(tmp_path)
    site = export.export_site(run, tmp_path / "sites", site_id="unittest01",
                              display_name="Unit Test", release_id="unittest01_20261118")
    assert (site / "site_profile.json").is_file()
    sys.path.insert(0, str(wulai / "定位演算法/deploy_code/sfm_direct_deploy"))
    import direct_map
    assets = direct_map.DirectMapAssets.load(site / "localization/direct_bundle.json")
    assert assets.map_revision_id == "unittest01_20261118"
    assert assets.coordinate_frame_id == f"unittest01_20261118_{export.frame_digest(site / 'model')[:12]}"
    assert assets.bank_name == "sideview"
    assert len(assets.ref_names) == 150
    assert assets.reference_depth_dir is None and assets.intersection_cells is None
    assert assets.ref_centers is not None and assets.ref_yaws is not None


def test_export_matches_delivery_tree_with_optional_dense(tmp_path: Path) -> None:
    run = _synthetic_run(tmp_path, n_images=120, n_points=100)
    site = export.export_site(run, tmp_path / "s1", site_id="treecheck",
                              display_name="Tree", release_id="treecheck_20261118")
    import json as _json
    top = {p.name for p in site.iterdir()}
    assert top == {"site_profile.json", "map", "model", "keyframes", "localization"}, top
    assert {p.name for p in (site / "map").iterdir()} == {"map.ply"}
    assert {p.name for p in (site / "model").iterdir()} == {"cameras.bin", "images.bin", "points3D.bin"}
    assert {p.name for p in (site / "keyframes").iterdir()} == {"keyframes.jsonl", "images"}
    loc_names = {p.name for p in (site / "localization").iterdir()}
    assert {"direct_bundle.json", "direct_localizer_profile.json", "reference_poses.json",
            "T_align_gravity.json", "reference_manifest.jsonl",
            "megaloc_references_sideview.npy", "megaloc_references_sideview.names.json"} <= loc_names
    assert "map_dense.ply" not in loc_names and "map_dense.ply" not in {p.name for p in (site / "map").iterdir()}
    # with dense output present, it ships as map/map_dense.ply and is files[]-listed
    dense_src = run / "products/map/dense/dense.ply"
    dense_src.parent.mkdir(parents=True, exist_ok=True)
    dense_src.write_bytes(b"0123456789ABCDEF")
    site2 = export.export_site(run, tmp_path / "s2", site_id="treecheck",
                               display_name="Tree", release_id="treecheck_20261118")
    assert (site2 / "map/map_dense.ply").is_file()
    bundle = _json.loads((site2 / "localization/direct_bundle.json").read_text(encoding="utf-8"))
    assert "../map/map_dense.ply" in [e["path"] for e in bundle["files"]]
