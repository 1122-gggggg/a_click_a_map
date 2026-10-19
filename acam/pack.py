"""Pack a finished run into one relocatable archive for another machine.

Text files holding absolute paths ship as `*.in` templates (`@PKG@`, `@EDM_CHECKPOINT@`);
`install.py` renders them. EDM weights are never packed (the target supplies its own).
"""
from __future__ import annotations

import hashlib
import json
import shutil
import string
import subprocess
from pathlib import Path

from .config import REPO

TEMPLATES = Path(__file__).parent / "templates"


def _template_text(text: str, mapping: list[tuple[str, str]]) -> str:
    for old, new in mapping:
        text = text.replace(old, new)
    return text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _render(src: Path, dst: Path, mapping: list[tuple[str, str]], forbidden: list[str]) -> None:
    text = _template_text(src.read_text(encoding="utf-8"), mapping)
    leftover = [line for line in text.splitlines() if any(token in line for token in forbidden)]
    if leftover:
        raise RuntimeError(f"unmapped absolute path in {src.name}: {leftover[0][:200]}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")


def pack(run: Path, out_dir: Path, name: str, *, with_megaloc_weights: bool = True, mapping_summary: str = "",
         freeze: str = "") -> Path:
    run = run.resolve(strict=True)
    config = json.loads((run / "products/localization/localizer_config.json").read_text(encoding="utf-8"))
    receipt = json.loads((run / "products/FINAL_RECEIPT.json").read_text(encoding="utf-8"))
    mega_src, mega_ckpt = Path(config["megaloc_source"]), Path(config["megaloc_checkpoint"])
    edm_root, edm_ckpt = Path(config["edm_root"]), Path(config["edm_checkpoint"])
    mapping = [
        (f"{run}/artifacts/keyframes/images", "@PKG@/localization/keyframes/images"),
        (str(config["map_model"]), "@PKG@/localization/edm_model"),
        (str(config["reference_manifest"]), "@PKG@/localization/bank/reference_manifest.jsonl"),
        (str(mega_src), "@PKG@/megaloc/source"),
        (str(mega_ckpt), "@PKG@/megaloc/model.safetensors"),
        (str(edm_ckpt), "@EDM_CHECKPOINT@"),
        (str(edm_root), "@PKG@/code/EDM"),
    ]
    forbidden = [str(run), str(mega_src), str(edm_root)]
    stage = out_dir / ".staging" / name
    if stage.exists():
        shutil.rmtree(stage)
    loc = stage / "localization"
    ignore_py = shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", ".git")
    shutil.copytree((run / "artifacts/mapping/edm/model").resolve(), loc / "edm_model")
    shutil.copytree((run / "artifacts/keyframes/images").resolve(), loc / "keyframes/images")
    _render(run / "artifacts/keyframes/keyframes.jsonl", loc / "keyframes/keyframes.jsonl.in", mapping, forbidden)
    bank = run / "products/localization"
    (loc / "bank").mkdir(parents=True)
    for file in ("megaloc_references.npy", "megaloc_references.names.json"):
        shutil.copy2(bank / file, loc / "bank" / file)
    _render(bank / "reference_manifest.jsonl", loc / "bank/reference_manifest.jsonl.in", mapping, forbidden)
    _render(bank / "localizer_config.json", loc / "bank/localizer_config.json.in", mapping, forbidden)
    shutil.copy2(run / "inputs/intrinsics.json", loc / "intrinsics.json")

    shutil.copytree(mega_src, stage / "megaloc/source", ignore=ignore_py)
    if with_megaloc_weights:
        shutil.copy2(mega_ckpt, stage / "megaloc/model.safetensors")
    shutil.copytree(REPO / "pipeline/src", stage / "code/normal_map_pipeline_src", ignore=ignore_py)
    shutil.copytree(REPO / "pipeline/compat_src", stage / "code/river_map_quality_src", ignore=ignore_py)
    shutil.copytree(edm_root, stage / "code/EDM", ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "weights"))

    dense = stage / "dense"
    shutil.copytree((run / "artifacts/mapping/direct/model").resolve(), dense / "glue_pose_model")
    for file in ("original_rgb.ply", "original_rgb_receipt.json"):
        if (run / "products/map" / file).is_file():
            shutil.copy2(run / "products/map" / file, dense / file)
    rec = stage / "receipts"
    rec.mkdir()
    for rel in ("products/FINAL_RECEIPT.json", "receipts/direct_preprocessing.json", "inputs/corpus_manifest.json",
                "inputs/gluemap_config.json"):
        if (run / rel).is_file():
            shutil.copy2(run / rel, rec / Path(rel).name)
    for rel in ("MAP_REPORT.md", "map_report.json", "mapping_summary.json"):
        if (run.parent / rel).is_file():
            shutil.copy2(run.parent / rel, rec / rel)
    for tool in ("install.py", "run_localize.py"):
        shutil.copy2(TEMPLATES / tool, stage / tool)
    (stage / "requirements_freeze.txt").write_text(freeze, encoding="utf-8")
    readme = string.Template((TEMPLATES / "README.tmpl.md").read_text(encoding="utf-8")).safe_substitute(
        name=name,
        archive=f"{name}.tar.zst",
        n_videos=len(receipt["gluemap"]["preprocessing"]["per_source"]),
        status=receipt["status"],
        n_points=_count_points(run),
        n_images=receipt["gluemap"]["preprocessing"]["keyframes"],
        n_refs=receipt["localization"]["references"],
        megaloc_weights=" and weights `model.safetensors`" if with_megaloc_weights else " (weights NOT included: supply `megaloc/model.safetensors`)",
        mapping_summary=mapping_summary or "see `receipts/`",
    )
    (stage / "README.md").write_text(readme, encoding="utf-8")
    lines = [f"{_sha256(path)}  {path.relative_to(stage).as_posix()}" for path in sorted(stage.rglob("*")) if path.is_file()]
    (stage / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return _archive(stage, out_dir, name)


def _count_points(run: Path) -> str:
    try:
        data = json.loads((run / "artifacts/mapping/edm/retriangulation_receipt.json").read_text(encoding="utf-8"))
        return f"{int(data['points3D']):,}"
    except (OSError, ValueError, KeyError, TypeError):
        return "see receipt"


def _archive(stage: Path, out_dir: Path, name: str) -> Path:
    if shutil.which("zstd"):
        archive = out_dir / f"{name}.tar.zst"
        archive.unlink(missing_ok=True)
        subprocess.run(f"tar -C '{stage.parent}' -cf - '{name}' | zstd -T0 -3 -q -o '{archive}'", shell=True, check=True)
    else:
        archive = out_dir / f"{name}.tar.gz"
        archive.unlink(missing_ok=True)
        subprocess.run(["tar", "-C", str(stage.parent), "-czf", str(archive), name], check=True)
    shutil.rmtree(stage.parent)
    return archive
