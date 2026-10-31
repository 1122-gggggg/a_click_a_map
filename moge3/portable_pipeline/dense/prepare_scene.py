#!/usr/bin/env python3
"""Convert a COLMAP (pycolmap) sparse model into an official MVSFormer++ / MVSNet scene.

Writes, under --work_root:
  data/<scene>/cams/%08d_cam.txt   MVSNet camera files (exact source extrinsics/intrinsics)
  data/<scene>/images/%08d.jpg     symlinks to the original full-resolution keyframes
  data/<scene>/pair.txt            view selection (top-K sources by sparse-track co-visibility angle score)
  scenes.txt                       scene list for MVSDataset
  mapping_manifest.tsv             MVS index <-> source image_id/name/camera/frame
  products/sparse_edm_all_points.ply  complete source points3D (binary little endian, xyz double, rgb uchar)
  conversion_receipt.json          counts, invariants, verification results

Pose/intrinsics are NOT recomputed or similarity-transformed. Co-visibility scoring follows
colmap2mvsnet.py (theta0/sigma1/sigma2 Gaussian on the triangulation angle per shared point), but is
computed from point tracks in vectorized form instead of all-pairs Python set intersections.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pycolmap
from PIL import Image

INVALID_P3D = np.uint64(18446744073709551615)
DEFAULT_MODEL = None
DEFAULT_IMAGES = None
DEFAULT_WORK = None


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="COLMAP binary model dir (cameras/images/points3D/frames/rigs .bin), PINHOLE cameras")
    ap.add_argument("--image_root", required=True, help="root that image names (vid_x/frame.jpg) are relative to")
    ap.add_argument("--work_root", required=True)
    ap.add_argument("--scene", default="scene")
    ap.add_argument("--expect_views", type=int, default=None, help="optional guard: required number of registered views")
    ap.add_argument("--num_depth", type=int, default=192, help="depth hypotheses written in cam files (official max_d)")
    ap.add_argument("--converter_interval_scale", type=float, default=1.0,
                    help="divisor applied to depth interval in cam files (official colmap2mvsnet default 1). "
                         "Keep 1.0 and pass --interval_scale 1.06 at MVSFormer++ run time (dataset multiplies once).")
    ap.add_argument("--intrinsic_scale", type=float, default=1.0,
                    help="multiplier on source intrinsics written to cam files; 1.0 = exact source intrinsics")
    ap.add_argument("--num_src", type=int, default=10)
    ap.add_argument("--theta0", type=float, default=5.0)
    ap.add_argument("--sigma1", type=float, default=1.0)
    ap.add_argument("--sigma2", type=float, default=10.0)
    ap.add_argument("--zero_baseline_rel", type=float, default=1e-6,
                    help="camera-center distance <= this * scene extent counts as zero baseline (excluded as source)")
    ap.add_argument("--num_verify_images", type=int, default=32)
    return ap.parse_args()


def sha256_file(path, bufsize=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_cam(path, E, K, dmin, dint, dnum, dmax):
    lines = ["extrinsic"]
    for r in E:
        lines.append(" ".join(repr(float(v)) for v in r) + " ")
    lines += ["", "intrinsic"]
    for r in K:
        lines.append(" ".join(repr(float(v)) for v in r) + " ")
    lines += ["", f"{float(dmin)!r} {float(dint)!r} {int(dnum)} {float(dmax)!r}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def parse_cam(path):
    with open(path) as f:
        lines = [ln.rstrip() for ln in f.readlines()]
    assert lines[0] == "extrinsic" and lines[6] == "intrinsic", path
    E = np.array([[float(x) for x in lines[k].split()] for k in range(1, 5)], dtype=np.float64)
    K = np.array([[float(x) for x in lines[k].split()] for k in range(7, 10)], dtype=np.float64)
    d = lines[11].split()
    return E, K, float(d[0]), float(d[1]), int(float(d[2])), float(d[3])


def main():
    a = parse_args()
    t_start = time.time()
    work = Path(a.work_root)
    scene_dir = work / "data" / a.scene
    cams_dir = scene_dir / "cams"
    imgs_dir = scene_dir / "images"
    prod_dir = work / "products"
    for d in (cams_dir, imgs_dir, prod_dir):
        d.mkdir(parents=True, exist_ok=True)
    model_dir = Path(a.model)

    # ------------------------------------------------------------------ load source
    log(f"loading model {model_dir} with pycolmap {pycolmap.__version__}")
    rec = pycolmap.Reconstruction(str(model_dir))
    src_counts = dict(cameras=len(rec.cameras), images_total=len(rec.images), points3D=len(rec.points3D),
                      frames=len(rec.frames), rigs=len(rec.rigs),
                      num_observations_pycolmap=int(rec.compute_num_observations()),
                      mean_reprojection_error_pycolmap=float(rec.compute_mean_reprojection_error()))
    log(f"source counts: {src_counts}")

    cam_info = {}
    for cid, cam in rec.cameras.items():
        mname = str(cam.model).split(".")[-1]
        if mname != "PINHOLE":
            raise SystemExit(f"camera {cid} model {mname} is not PINHOLE")
        fx, fy, cx, cy = [float(v) for v in cam.params]
        cam_info[int(cid)] = dict(model=mname, width=int(cam.width), height=int(cam.height),
                                  params=[fx, fy, cx, cy])

    images = [rec.images[k] for k in sorted(rec.images.keys()) if rec.images[k].has_pose]
    N = len(images)
    if a.expect_views is not None and N != a.expect_views:
        raise SystemExit(f"registered views {N} != expected {a.expect_views}")
    log(f"registered views: {N}")

    names, img_ids, cam_ids, frame_ids = [], [], [], []
    E = np.zeros((N, 4, 4), np.float64)
    K = np.zeros((N, 3, 3), np.float64)
    W = np.zeros(N, np.int64)
    H = np.zeros(N, np.int64)
    for i, im in enumerate(images):
        M = np.asarray(im.cam_from_world().matrix(), dtype=np.float64)  # world -> camera, 3x4
        E[i, :3, :] = M
        E[i, 3, 3] = 1.0
        ci = cam_info[int(im.camera_id)]
        fx, fy, cx, cy = ci["params"]
        s = a.intrinsic_scale
        K[i] = np.array([[fx * s, 0.0, cx * s], [0.0, fy * s, cy * s], [0.0, 0.0, 1.0]])
        W[i], H[i] = ci["width"], ci["height"]
        names.append(im.name)
        img_ids.append(int(im.image_id))
        cam_ids.append(int(im.camera_id))
        frame_ids.append(int(im.frame_id))
    if len(set(names)) != N:
        raise SystemExit("duplicate image names in source model")
    R = E[:, :3, :3]
    t = E[:, :3, 3]
    C = -np.einsum("nji,nj->ni", R, t)  # camera centers in world
    ortho_err = float(np.abs(np.einsum("nij,nkj->nik", R, R) - np.eye(3)).max())
    det_min = float(np.linalg.det(R).min())
    log(f"poses: max |RR^T-I|={ortho_err:.3e} min det={det_min:.6f}")

    # ------------------------------------------------------------------ points3D (streamed once, file order)
    pts = rec.points3D
    n_pts = len(pts)
    pid0 = np.empty(n_pts, np.uint64)
    xyz0 = np.empty((n_pts, 3), np.float64)
    rgb0 = np.empty((n_pts, 3), np.uint8)
    for k, (pid, p) in enumerate(pts.items()):
        pid0[k] = pid
        xyz0[k] = p.xyz
        rgb0[k] = p.color
    order = np.argsort(pid0, kind="stable")
    pid_sorted = pid0[order]
    log(f"points3D read: {n_pts}")

    # ------------------------------------------------------------------ observations per registered image
    rng = np.random.default_rng(0)
    sample_idx = set(np.linspace(0, N - 1, min(a.num_verify_images, N)).round().astype(int).tolist())
    obs_img_parts, obs_row_parts = [], []
    seg = np.zeros(N + 1, np.int64)
    verify_obs = {}
    n_p2d = 0
    for i, im in enumerate(images):
        p2 = im.points2D
        n = len(p2)
        n_p2d += n
        p3 = np.fromiter((q.point3D_id for q in p2), dtype=np.uint64, count=n)
        pos = np.searchsorted(pid_sorted, p3)
        pos_c = np.minimum(pos, n_pts - 1)
        ok = (pos < n_pts) & (pid_sorted[pos_c] == p3) & (p3 != INVALID_P3D)
        rows = order[pos_c[ok]].astype(np.int32)
        obs_row_parts.append(rows)
        obs_img_parts.append(np.full(rows.size, i, np.int32))
        seg[i + 1] = seg[i] + rows.size
        if i in sample_idx:
            xy = np.array([q.xy for q in p2], dtype=np.float64)[ok]
            verify_obs[i] = (rows.copy(), xy)
        if i % 256 == 0:
            log(f"observations: image {i}/{N}")
    obs_row = np.concatenate(obs_row_parts)
    obs_img = np.concatenate(obs_img_parts)
    n_obs = int(obs_row.size)
    log(f"observations with valid points3D: {n_obs} of {n_p2d} points2D")
    if n_obs != src_counts["num_observations_pycolmap"]:
        log(f"WARNING: valid observations {n_obs} != pycolmap num_observations "
            f"{src_counts['num_observations_pycolmap']}")

    # ------------------------------------------------------------------ depth range and interval per view
    dmin = np.zeros(N)
    dmax = np.zeros(N)
    dint = np.zeros(N)
    n_pos = np.zeros(N, np.int64)
    n_neg = np.zeros(N, np.int64)
    depth_fallback = np.zeros(N, bool)
    all_rows = np.arange(n_pts)
    for i in range(N):
        rows = obs_row[seg[i]:seg[i + 1]]
        if rows.size < 50:  # too few shared observations for a stable 1-99% depth range
            # registered view with no shared observations: depth range from all source points in its pose
            depth_fallback[i] = True
            rows = all_rows
        z = xyz0[rows] @ R[i, 2] + t[i, 2]
        zp = np.sort(z[z > 0])
        n_pos[i] = zp.size
        n_neg[i] = z.size - zp.size
        m = zp.size
        if m == 0:
            raise SystemExit(f"image index {i} ({names[i]}) has no positive sparse depth")
        dmin[i] = zp[int(m * 0.01)]
        dmax[i] = zp[int(m * 0.99)]
        if not dmax[i] > dmin[i]:
            raise SystemExit(f"degenerate depth range for image index {i} ({names[i]})")
        dint[i] = (dmax[i] - dmin[i]) / (a.num_depth - 1) / a.converter_interval_scale
    log(f"depth ranges: dmin median {np.median(dmin):.4f}, dmax median {np.median(dmax):.4f}, "
        f"fallback views (no observations) {int(depth_fallback.sum())}")

    # ------------------------------------------------------------------ co-visibility scoring from tracks
    t_score = time.time()
    ordr = np.argsort(obs_row, kind="stable")
    rs = obs_row[ordr]
    ims = obs_img[ordr]
    uniq_rows, starts, counts = np.unique(rs, return_index=True, return_counts=True)
    n_pairs_total = int(np.sum(counts.astype(np.int64) * (counts.astype(np.int64) - 1) // 2))
    log(f"tracks: {uniq_rows.size} points with >=1 obs; candidate pairs {n_pairs_total}")

    S = np.zeros(N * N, np.float64)
    CH = 2_000_000
    theta0, sig1, sig2 = a.theta0, a.sigma1, a.sigma2
    for L in np.unique(counts):
        if L < 2:
            continue
        sel = np.flatnonzero(counts == L)
        M = ims[starts[sel][:, None] + np.arange(L)[None, :]]  # (n, L) image index per observation
        Pxyz = xyz0[uniq_rows[sel]]  # (n, 3)
        iu, ju = np.triu_indices(L, 1)
        rows_per = max(1, CH // iu.size)
        for s0 in range(0, sel.size, rows_per):
            Mc = M[s0:s0 + rows_per]
            Pc = Pxyz[s0:s0 + rows_per][:, None, :]
            a_ = Mc[:, iu]
            b_ = Mc[:, ju]
            va = C[a_] - Pc
            vb = C[b_] - Pc
            na = np.linalg.norm(va, axis=2)
            nb = np.linalg.norm(vb, axis=2)
            ok = (a_ != b_) & (na > 0) & (nb > 0)
            cos = np.einsum("mkj,mkj->mk", va, vb) / np.where(ok, na * nb, 1.0)
            theta = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
            sig = np.where(theta <= theta0, sig1, sig2)
            g = np.exp(-(theta - theta0) ** 2 / (2.0 * sig ** 2))
            lo = np.minimum(a_, b_).astype(np.int64)
            hi = np.maximum(a_, b_).astype(np.int64)
            key = (lo * N + hi)[ok]
            S += np.bincount(key, weights=g[ok], minlength=N * N)
    S = S.reshape(N, N)
    S_sym = S + S.T  # official: score[i,j] = score[j,i] = s, i<j
    log(f"co-visibility scoring done in {time.time() - t_score:.1f}s")

    D = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
    extent = float(np.linalg.norm(C.max(0) - C.min(0)))
    zero_tol = a.zero_baseline_rel * extent
    zero_base = D <= zero_tol
    n_zero_offdiag = int(zero_base.sum() - N)

    view_sel = []
    n_zero_score_sel = 0
    min_kept_baseline = np.inf
    n_nonpos_candidates = np.zeros(N, np.int64)
    for i in range(N):
        row = S_sym[i].copy()
        row[zero_base[i]] = -np.inf  # excludes self and zero-baseline views
        n_nonpos_candidates[i] = int(np.sum(row[np.isfinite(row)] > 0))
        cand = np.lexsort((D[i], -row))[: a.num_src]  # score desc, then nearest camera center on ties
        if not np.all(np.isfinite(row[cand])):
            raise SystemExit(f"image index {i} has fewer than {a.num_src} valid sources")
        n_zero_score_sel += int(np.sum(row[cand] == 0))
        min_kept_baseline = min(min_kept_baseline, float(D[i, cand].min()))
        view_sel.append(cand)
    log(f"pair selection: zero-score selected entries {n_zero_score_sel}; "
        f"views with <{a.num_src} positive-score candidates: {int(np.sum(n_nonpos_candidates < a.num_src))}")

    # ------------------------------------------------------------------ write scene
    log("writing cams, pair.txt, images, scenes.txt")
    for i in range(N):
        write_cam(cams_dir / f"{i:08d}_cam.txt", E[i], K[i], dmin[i], dint[i], a.num_depth, dmax[i])

    with open(scene_dir / "pair.txt", "w") as f:
        f.write(f"{N}\n")
        for i, cand in enumerate(view_sel):
            f.write(f"{i}\n{len(cand)} ")
            f.write("".join(f"{int(j)} {float(S_sym[i, j]):f} " for j in cand))
            f.write("\n")

    src_root = Path(a.image_root)
    for i in range(N):
        src = (src_root / names[i]).absolute()
        if not src.is_file():
            raise SystemExit(f"missing source image {src}")
        dst = imgs_dir / f"{i:08d}.jpg"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        os.symlink(src, dst)

    with open(work / "scenes.txt", "w") as f:
        f.write(a.scene + "\n")

    with open(work / "mapping_manifest.tsv", "w") as f:
        f.write("mvs_index\tsource_image_id\tsource_name\tcamera_id\tframe_id\twidth\theight\t"
                "n_obs\tn_pos_depth\tdepth_min\tdepth_interval\tdepth_max\tsource_path\tcam_file\timage_link\tsrc_views\n")
        for i in range(N):
            f.write(f"{i}\t{img_ids[i]}\t{names[i]}\t{cam_ids[i]}\t{frame_ids[i]}\t{W[i]}\t{H[i]}\t"
                    f"{int(seg[i + 1] - seg[i])}\t{int(n_pos[i])}\t{float(dmin[i])!r}\t{float(dint[i])!r}\t{float(dmax[i])!r}\t"
                    f"{(src_root / names[i]).absolute()}\tcams/{i:08d}_cam.txt\timages/{i:08d}.jpg\t"
                    f"{','.join(str(int(j)) for j in view_sel[i])}\n")

    # ------------------------------------------------------------------ PLY (complete points3D, file order)
    t_ply = time.time()
    dt = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    arr = np.empty(n_pts, dt)
    arr["x"], arr["y"], arr["z"] = xyz0[:, 0], xyz0[:, 1], xyz0[:, 2]
    arr["red"], arr["green"], arr["blue"] = rgb0[:, 0], rgb0[:, 1], rgb0[:, 2]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        "comment source COLMAP points3D, file order, exact values (xyz float64, rgb uint8)\n"
        f"element vertex {n_pts}\n"
        "property double x\nproperty double y\nproperty double z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    ply_path = prod_dir / "sparse_edm_all_points.ply"
    with open(ply_path, "wb") as f:
        f.write(header)
        arr.tofile(f)
    del arr
    log(f"PLY written {ply_path} in {time.time() - t_ply:.1f}s")

    # ------------------------------------------------------------------ verification
    log("verifying outputs")
    max_pose_diff = 0.0
    max_k_diff = 0.0
    depth_field_mismatch = 0
    for i in range(N):
        Ep, Kp, dmn, dit, dnm, dmx = parse_cam(cams_dir / f"{i:08d}_cam.txt")
        max_pose_diff = max(max_pose_diff, float(np.abs(Ep - E[i]).max()))
        max_k_diff = max(max_k_diff, float(np.abs(Kp - K[i]).max()))
        if (dmn, dit, dnm, dmx) != (dmin[i], dint[i], a.num_depth, dmax[i]):
            depth_field_mismatch += 1

    reproj = []
    for i, (rows, xy) in verify_obs.items():
        Ep, Kp, *_ = parse_cam(cams_dir / f"{i:08d}_cam.txt")
        X = xyz0[rows]
        Xc = X @ Ep[:3, :3].T + Ep[:3, 3]
        u = Kp[0, 0] * Xc[:, 0] / Xc[:, 2] + Kp[0, 2]
        v = Kp[1, 1] * Xc[:, 1] / Xc[:, 2] + Kp[1, 2]
        r = np.hypot(u - xy[:, 0], v - xy[:, 1])
        reproj.append(r)
        if np.any(Xc[:, 2] <= 0):
            raise SystemExit(f"non-positive depth for observed point in sample image {i}")
    reproj = np.concatenate(reproj)
    reproj_stats = dict(sampled_images=len(verify_obs), observations=int(reproj.size),
                        mean_px=float(reproj.mean()), median_px=float(np.median(reproj)),
                        p99_px=float(np.percentile(reproj, 99)), max_px=float(reproj.max()))

    in_range = []
    for i in range(N):
        rows = all_rows if depth_fallback[i] else obs_row[seg[i]:seg[i + 1]]
        z = xyz0[rows] @ R[i, 2] + t[i, 2]
        zp = z[z > 0]
        in_range.append(float(np.mean((zp >= dmin[i]) & (zp <= dmax[i]))) if zp.size else 0.0)
    in_range = np.array(in_range)

    size_mismatch = []
    link_ok = 0
    for i in range(N):
        lk = imgs_dir / f"{i:08d}.jpg"
        if lk.is_symlink() and Path(os.readlink(lk)).is_file():
            link_ok += 1
        with Image.open(lk) as im_pil:
            if im_pil.size != (int(W[i]), int(H[i])):
                size_mismatch.append(i)

    cam_files = len(list(cams_dir.glob("*_cam.txt")))
    pair_lines = (scene_dir / "pair.txt").read_text().splitlines()
    pair_ok = int(pair_lines[0]) == N and len(pair_lines) == 1 + 2 * N
    pair_counts_ok = all(int(pair_lines[2 + 2 * i].split()[0]) == a.num_src for i in range(N))

    with open(ply_path, "rb") as f:
        hdr = f.read(len(header))
    ply_size = os.path.getsize(ply_path)
    ply_ok = hdr == header and ply_size == len(header) + n_pts * dt.itemsize
    rng_idx = np.concatenate([[0, n_pts - 1], rng.integers(0, n_pts, 1000)])
    with open(ply_path, "rb") as f:
        f.seek(len(header))
        raw = np.frombuffer(f.read(), dtype=dt)
    ply_vertex_exact = bool(np.array_equal(raw["x"][rng_idx], xyz0[rng_idx, 0]) and
                            np.array_equal(raw["z"][rng_idx], xyz0[rng_idx, 2]) and
                            np.array_equal(raw["red"][rng_idx], rgb0[rng_idx, 0]))
    ply_n = int(raw.size)
    del raw

    receipt = dict(
        created=time.strftime("%Y-%m-%dT%H:%M:%S"),
        pycolmap_version=pycolmap.__version__,
        source=dict(model_dir=str(model_dir), image_root=str(src_root), counts=src_counts,
                    file_sha256={n: sha256_file(model_dir / n) for n in
                                 ("cameras.bin", "images.bin", "points3D.bin", "frames.bin", "rigs.bin")}),
        outputs=dict(work_root=str(work), scene_dir=str(scene_dir), scenes_txt=str(work / "scenes.txt"),
                     mapping_manifest=str(work / "mapping_manifest.tsv"), ply=str(ply_path),
                     ply_sha256=sha256_file(ply_path)),
        conversion=dict(registered_views=N, cam_files=cam_files, image_links_ok=link_ok,
                        pair_lines=len(pair_lines), num_src=a.num_src, expected_views=a.expect_views,
                        num_depth=a.num_depth, converter_interval_scale=a.converter_interval_scale,
                        intrinsic_scale=a.intrinsic_scale, theta0=a.theta0, sigma1=a.sigma1, sigma2=a.sigma2,
                        zero_baseline_tol_m=zero_tol, zero_baseline_offdiag_pairs=n_zero_offdiag,
                        min_kept_source_baseline_m=float(min_kept_baseline),
                        scene_extent_m=extent, candidate_track_pairs=n_pairs_total,
                        zero_score_selected_entries=n_zero_score_sel,
                        views_with_fewer_than_num_src_positive=int(np.sum(n_nonpos_candidates < a.num_src)),
                        observations_valid=n_obs, points2d_total=int(n_p2d),
                        points3d_written=n_pts, points3d_in_ply=ply_n),
        camera_models={str(k): dict(model=v["model"], width=v["width"], height=v["height"],
                                    params_fx_fy_cx_cy=v["params"]) for k, v in cam_info.items()},
        depth=dict(dmin_min=float(dmin.min()), dmin_median=float(np.median(dmin)), dmin_max=float(dmin.max()),
                   dmax_min=float(dmax.min()), dmax_median=float(np.median(dmax)), dmax_max=float(dmax.max()),
                   interval_min=float(dint.min()), interval_median=float(np.median(dint)),
                   interval_max=float(dint.max()), positive_obs_total=int(n_pos.sum()),
                   negative_obs_total=int(n_neg.sum()),
                   inrange_fraction_min=float(in_range.min()), inrange_fraction_median=float(np.median(in_range)),
                   depth_fallback_views=int(depth_fallback.sum()),
                   depth_fallback_indices=np.flatnonzero(depth_fallback).tolist()),
        invariants=dict(
            registered_views_equals_expected=bool(a.expect_views is None or N == a.expect_views),
            cam_files_equal_views=bool(cam_files == N),
            image_links_all_resolve=bool(link_ok == N),
            image_sizes_match_source_cameras=bool(len(size_mismatch) == 0),
            image_size_mismatch_indices=size_mismatch[:20],
            pair_header_and_rows_ok=bool(pair_ok),
            pair_each_row_has_num_src=bool(pair_counts_ok),
            max_abs_extrinsic_diff_cam_file_vs_source=max_pose_diff,
            max_abs_intrinsic_diff_cam_file_vs_source=max_k_diff,
            depth_fields_mismatch_count=depth_field_mismatch,
            max_abs_RRt_minus_I=ortho_err, min_det_R=det_min,
            reprojection_residual_px=reproj_stats,
            ply_header_and_size_ok=bool(ply_ok), ply_vertex_count=ply_n,
            ply_sampled_vertices_exact=ply_vertex_exact,
            no_self_source=bool(all(i not in set(view_sel[i].tolist()) for i in range(N))),
            no_zero_baseline_source=bool(all(D[i, j] > zero_tol for i in range(N) for j in view_sel[i])),
        ),
        notes=[
            "Poses from pycolmap Image.cam_from_world() (world->camera), copied bit-exact via repr(float); no similarity transform.",
            "Intrinsics are source PINHOLE fx,fy,cx,cy at full resolution (intrinsic_scale=1.0).",
            "MVSDataset.read_cam_file divides intrinsics[:2] by 4 unconditionally and scale_mvs_input then rescales by "
            "the loaded image size; with full-resolution cam files this yields focal/principal values 4x too small "
            "relative to the image. Parent must reconcile at integration (see final report).",
            "Depth interval written as (dmax-dmin)/(num_depth-1)/converter_interval_scale; MVSDataset multiplies by "
            "--interval_scale at run time, so use --interval_scale 1.06 once, with converter_interval_scale=1.0.",
            "PLY stores complete points3D in source file order; xyz as float64 (exact), rgb uchar.",
        ],
        elapsed_s=time.time() - t_start,
    )
    with open(work / "conversion_receipt.json", "w") as f:
        json.dump(receipt, f, indent=2, default=str)
    log(f"receipt written: {work / 'conversion_receipt.json'}")
    log(f"done in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
