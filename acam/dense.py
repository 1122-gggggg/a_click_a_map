"""MVSFormer++ densification for the original entry (runs after EDM retriangulation).

Reuses moge3's portable `dense_stage.py` (COLMAP model + keyframe JPEGs in, dense
RGB PLY out, map frame unchanged). The MVSFormer++ checkpoint/config are
user-supplied, like the EDM weights: `ACAM_MVS_CHECKPOINT` / `ACAM_MVS_CONFIG`,
or `third_party/mvsformerpp/weights/{model_best.pth,config.json}` (see `scripts/setup.sh`).
"""
from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path

from .config import REPO

DENSE_STAGE = REPO / "moge3/portable_pipeline/dense_stage.py"
SOURCE_DIR = REPO / "third_party/mvsformerpp"
WEIGHTS_DIR = SOURCE_DIR / "weights"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def weights() -> tuple[Path, Path] | None:
    """(checkpoint, config) if the user supplied both MVSFormer++ files."""
    import os
    ckpt = Path(os.environ.get("ACAM_MVS_CHECKPOINT", WEIGHTS_DIR / "model_best.pth")).expanduser()
    cfg = Path(os.environ.get("ACAM_MVS_CONFIG", WEIGHTS_DIR / "config.json")).expanduser()
    if ckpt.is_file() and cfg.is_file():
        return ckpt, cfg
def source_dir() -> Path | None:
    """The cloned MVSFormer++ tree (must contain the SDPA patch already applied)."""
    if (SOURCE_DIR / "datasets").is_dir() and (SOURCE_DIR / "utils.py").is_file():
        return SOURCE_DIR
    return None


def run_dense(paths, run: Path, *, python: str, mvs_python: str | None = None) -> dict:
    """Run MVSFormer++ depth + fusion on the finished EDM model.

    Writes `products/map/dense/{dense.ply,dense_receipt.json,work/}` and returns
    the stage receipt. Skips (quickly) when the outputs are already fresh.
    """
    model = run / "artifacts/mapping/edm/model"
    images = run / "artifacts/keyframes/images"
    out = run / "products/map/dense"
    out.mkdir(parents=True, exist_ok=True)
    dense_ply = out / "dense.ply"
    receipt_path = out / "dense_receipt.json"
    found = weights()
    if found is None:
        raise ValueError("稠密化需要 MVSFormer++ 權重：把官方 TNT 的 model_best.pth 與 config.json 放到 "
                         "third_party/mvsformerpp/weights/（或設 ACAM_MVS_CHECKPOINT / ACAM_MVS_CONFIG）")
    checkpoint, config = found
    src = source_dir()
    if src is None:
        raise ValueError("找不到 MVSFormer++ 原始碼：先跑 ./scripts/setup.sh（會 clone 並套 SDPA patch）")
    if dense_ply.is_file() and receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            fresh = {name: sha256_file(model / name) for name in ("cameras.bin", "images.bin", "points3D.bin")}
            if receipt.get("model") == fresh:
                return {"skipped": True, **{k: receipt[k] for k in ("dense_ply", "points", "views", "seconds") if k in receipt}}
            print("map model changed since last densification; redoing it")
        except (OSError, ValueError):
            pass
    interpreter = mvs_python or python
    cmd = [interpreter, str(DENSE_STAGE), "--model", str(model), "--images", str(images),
           "--output", str(out), "--checkpoint", str(checkpoint), "--config", str(config),
           "--mvsformerpp-root", str(src), "--mvs-python", interpreter,
           "--scene-python", interpreter, "--resume"]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        tail = ""
        try:
            tail = (out / "work" / "logs" / "dense.log").read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
            tail = "\n".join(tail)
        except OSError:
            pass
        raise RuntimeError(f"MVSFormer++ densification failed (rc={proc.returncode}):\n{tail}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    return {"skipped": False, **{k: receipt[k] for k in ("dense_ply", "points", "views", "seconds")}}
