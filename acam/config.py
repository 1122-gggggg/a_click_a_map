"""Where everything lives. Environment variables beat `acam.toml` beat `third_party/` defaults."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

try:  # Python >= 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore[assignment]

REPO = Path(__file__).resolve().parents[1]
THIRD_PARTY = REPO / "third_party"

#: GlueMap config key -> file name inside the checkpoint directory.
GLUEMAP_CHECKPOINTS = {
    "path_dg": "checkpoint-dg+visym.pth",
    "path_feedforward": "pi3.safetensors",
    "path_retrieval": "dino_salad.ckpt",
    "path_tracker": "vggsfm_v2_0_0_track_predictor.bin",
}


@dataclass(frozen=True)
class Paths:
    python: Path            # interpreter of the engine environment (torch, pycolmap, gluemap)
    gluemap: Path           # patched colmap/gluemap checkout
    gluemap_checkpoints: Path
    edm: Path               # EDM repo (code)
    edm_checkpoint: Path    # edm_outdoor.ckpt, supplied by the user
    megaloc_source: Path
    megaloc_checkpoint: Path

    def engine_pythonpath(self) -> str:
        return os.pathsep.join(
            [str(REPO / "pipeline/src"), str(REPO / "pipeline/compat_src"), str(self.gluemap), str(REPO)]
        )

    def required_files(self) -> dict[str, Path]:
        files = {name: self.gluemap_checkpoints / file for name, file in GLUEMAP_CHECKPOINTS.items()}
        files.update(
            edm_checkpoint=self.edm_checkpoint,
            megaloc_checkpoint=self.megaloc_checkpoint,
            megaloc_source=self.megaloc_source / "megaloc_model.py",
            edm_repo=self.edm / "src",
            gluemap_package=self.gluemap / "gluemap",
            python=self.python,
        )
        return files


def _toml() -> dict:
    path = Path(os.environ.get("ACAM_CONFIG", REPO / "acam.toml"))
    if tomllib is None or not path.is_file():
        return {}
    return tomllib.loads(path.read_text(encoding="utf-8"))


def load_paths() -> Paths:
    cfg = _toml().get("paths", {})

    def pick(env: str, key: str, default: Path) -> Path:
        return Path(os.environ.get(env) or cfg.get(key) or default).expanduser()

    gluemap = pick("ACAM_GLUEMAP", "gluemap", THIRD_PARTY / "gluemap")
    return Paths(
        python=pick("ACAM_PYTHON", "python", Path(sys.executable)),
        gluemap=gluemap,
        gluemap_checkpoints=pick("ACAM_GLUEMAP_CHECKPOINTS", "gluemap_checkpoints", gluemap / "checkpoints"),
        edm=pick("ACAM_EDM", "edm", THIRD_PARTY / "EDM"),
        edm_checkpoint=pick("ACAM_EDM_CHECKPOINT", "edm_checkpoint", THIRD_PARTY / "EDM/weights/edm_outdoor.ckpt"),
        megaloc_source=pick("ACAM_MEGALOC_SOURCE", "megaloc_source", THIRD_PARTY / "MegaLoc"),
        megaloc_checkpoint=pick(
            "ACAM_MEGALOC_CHECKPOINT", "megaloc_checkpoint", THIRD_PARTY / "checkpoints/megaloc/model.safetensors"
        ),
    )
