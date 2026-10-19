from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .runner import atomic_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prune_observation_free_registered_images(
    input_model: Path, output_model: Path
) -> dict[str, Any]:
    """Deregister pose-only images that carry no 3D observations.

    The operation fails closed if a zero-observation image shares a rig frame
    with an observed image. Point geometry and observations must remain byte
    identical after serialization.
    """

    import pycolmap

    source = input_model.resolve(strict=True)
    output = output_model.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    reconstruction = pycolmap.Reconstruction(str(source))
    registered = [image for image in reconstruction.images.values() if image.has_pose]
    zero_images = [image for image in registered if int(image.num_points3D) == 0]
    if not zero_images:
        raise RuntimeError("no observation-free registered images to prune")
    by_frame: dict[int, list[Any]] = {}
    for image in registered:
        by_frame.setdefault(int(image.frame_id), []).append(image)
    unsafe_frames = sorted(
        {
            int(image.frame_id)
            for image in zero_images
            if any(int(peer.num_points3D) > 0 for peer in by_frame[int(image.frame_id)])
        }
    )
    if unsafe_frames:
        raise RuntimeError(
            f"observation-free images share observed rig frames: {unsafe_frames}"
        )

    points_before = reconstruction.num_points3D()
    observations_before = int(reconstruction.compute_num_observations())
    registered_before = reconstruction.num_reg_images()
    removed_images = sorted(str(image.name) for image in zero_images)
    for frame_id in sorted({int(image.frame_id) for image in zero_images}):
        reconstruction.deregister_frame(frame_id)

    output.mkdir(parents=True)
    reconstruction.write(output)
    cleaned = pycolmap.Reconstruction(str(output))
    points_unchanged = cleaned.num_points3D() == points_before
    observations_unchanged = int(cleaned.compute_num_observations()) == observations_before
    binary_geometry_unchanged = all(
        _sha256(source / name) == _sha256(output / name)
        for name in ("cameras.bin", "points3D.bin", "rigs.bin")
    )
    geometry_unchanged = points_unchanged and observations_unchanged and binary_geometry_unchanged
    if not geometry_unchanged:
        raise RuntimeError("zero-observation cleanup changed camera or point geometry")

    receipt = {
        "schema_version": 1,
        "artifact_type": "OBSERVATION_FREE_IMAGE_CLEANUP",
        "input_model": str(source),
        "output_model": str(output),
        "registered_images_before": registered_before,
        "registered_images_after": cleaned.num_reg_images(),
        "removed_images": removed_images,
        "points3D": points_before,
        "observations": observations_before,
        "geometry_unchanged": geometry_unchanged,
        "binary_sha256": {
            name: _sha256(output / name)
            for name in ("cameras.bin", "points3D.bin", "rigs.bin")
        },
    }
    atomic_json(output.parent / "observation_free_cleanup.json", receipt)
    return receipt


__all__ = ["prune_observation_free_registered_images"]
