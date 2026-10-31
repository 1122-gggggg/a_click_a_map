#!/usr/bin/env bash
# Build the MoGe-3 toolkit under third_party/moge3 (idempotent).
#   moge3/scripts/setup.sh [--no-envs] [--no-weights]
# Layout it creates (read by run_moge3.sh / moge3/portable_pipeline/run_pipeline.py):
#   third_party/moge3/repo        colmap/gluemap @ pinned commit + patches/gluemap-moge3-megaloc.patch + MoGe-3 + MegaLoc
#   third_party/moge3/resources   edm (patched), megaloc, mvsformerpp (SDPA patch), normal_map_pipeline, river_map_quality
#   third_party/moge3/weights     pi3, tracker, dg, megaloc, moge3 (downloaded, sha256-verified); edm + mvsformerpp (supplied)
#   third_party/moge3/envs        gluemap (mapping + localization), moge (depth worker), mvs (MVSFormer++)
# Needs: git, micromamba or conda, uv, wget, ffmpeg; internet; Linux x86_64 + NVIDIA driver (CUDA 12.8+).
# Weights you must supply: ACAM_EDM_CHECKPOINT=/path/edm_outdoor.ckpt, ACAM_MVS_CHECKPOINT=/path/model_best.pth,
#   ACAM_MVS_CONFIG=/path/config.json (official MVSFormer++ tnt_ckpt). They are copied and sha256-verified.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
M3="$ROOT/moge3"
TK="${MOGE3_TOOLKIT:-$ROOT/third_party/moge3}"
CFG="$M3/configs/versions.json"
ENVS=1; WEIGHTS=1
for arg in "$@"; do
  case "$arg" in --no-envs) ENVS=0 ;; --no-weights) WEIGHTS=0 ;; *) echo "unknown option $arg" >&2; exit 2 ;; esac
done
cfg() { python3 - "$CFG" "$@" <<'E'
import json, sys
d = json.load(open(sys.argv[1]))
for key in sys.argv[2].split("."):
    d = d[key]
print(d)
E
}
clone_pinned() { # url dir commit
  [ -d "$2/.git" ] || git clone "$1" "$2"
  git -C "$2" fetch --quiet origin "$3" 2>/dev/null || true
  git -C "$2" checkout --quiet "$3"
}
apply_patch() { # dir patch
  if git -C "$1" apply --reverse --check "$2" >/dev/null 2>&1; then echo "  patch already applied: $(basename "$2")"
  else git -C "$1" apply "$2" && echo "  applied $(basename "$2")"; fi
}
mkdir -p "$TK/resources" "$TK/weights" "$TK/envs"

echo "[1/4] GlueMap (colmap/gluemap @ $(cfg sources.gluemap.commit | cut -c1-9)) + MoGe-3 / MegaLoc patch"
clone_pinned "$(cfg sources.gluemap.url)" "$TK/repo" "$(cfg sources.gluemap.commit)"
git -C "$TK/repo" submodule update --init --recursive -- $(python3 -c "import json;print(' '.join(json.load(open('$CFG'))['sources']['gluemap']['submodules']))")
apply_patch "$TK/repo" "$ROOT/$(cfg sources.gluemap.patch)"
clone_pinned "$(cfg sources.moge.url)" "$TK/repo/$(cfg sources.moge.dest)" "$(cfg sources.moge.commit)"
clone_pinned "$(cfg sources.megaloc.url)" "$TK/repo/$(cfg sources.megaloc.dest)" "$(cfg sources.megaloc.commit)"

echo "[2/4] resources: EDM, MegaLoc, MVSFormer++, pipeline code"
clone_pinned "$(cfg sources.edm.url)" "$TK/resources/edm" "$(cfg sources.edm.commit)"
apply_patch "$TK/resources/edm" "$ROOT/$(cfg sources.edm.patch)"
ln -sfn "$TK/repo/$(cfg sources.megaloc.dest)" "$TK/resources/megaloc"
if [ ! -d "$TK/resources/mvsformerpp/.git" ]; then git clone --recursive "$(cfg sources.mvsformerpp.url)" "$TK/resources/mvsformerpp"; fi
git -C "$TK/resources/mvsformerpp" checkout --quiet "$(cfg sources.mvsformerpp.commit)"
apply_patch "$TK/resources/mvsformerpp" "$ROOT/$(cfg sources.mvsformerpp.patch)"
ln -sfn "$ROOT/pipeline/src" "$TK/resources/normal_map_pipeline"
ln -sfn "$ROOT/pipeline/compat_src" "$TK/resources/river_map_quality"

if [ "$ENVS" = 1 ]; then
  echo "[3/4] environments"
  CONDA="${CONDA_EXE:-$(command -v micromamba || command -v conda || true)}"
  [ -n "$CONDA" ] || { echo "install micromamba or conda first" >&2; exit 1; }
  command -v uv >/dev/null || { echo "install uv first: https://docs.astral.sh/uv/" >&2; exit 1; }
  E="$TK/envs"; SPEC="$M3/envs"
  if [ ! -x "$E/gluemap/bin/python" ]; then
    "$CONDA" create -y -p "$E/gluemap" -f "$SPEC/gluemap.conda.yml"
    uv pip install --python "$E/gluemap/bin/python" --extra-index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1
    uv pip install --python "$E/gluemap/bin/python" -r "$SPEC/gluemap.pip.txt"
    CMAKE_PREFIX_PATH="$E/gluemap" uv pip install --python "$E/gluemap/bin/python" --no-deps --no-build-isolation -e "$TK/repo"
  fi
  "$E/gluemap/bin/python" -c "import torch,pycolmap,pygluemap; print('gluemap env ok', torch.__version__, pycolmap.__version__)"
  if [ ! -x "$E/moge/bin/python" ]; then
    uv venv --python 3.11 "$E/moge"
    uv pip install --python "$E/moge/bin/python" --extra-index-url https://download.pytorch.org/whl/cu130 -r "$SPEC/moge.requirements.txt"
  fi
  "$E/moge/bin/python" -c "import torch,flex_gemm,utils3d; print('moge env ok', torch.__version__)"
  if [ ! -x "$E/mvs/bin/python" ]; then
    uv venv --python 3.12 "$E/mvs"
    uv pip install --python "$E/mvs/bin/python" --extra-index-url https://download.pytorch.org/whl/cu128 -r "$SPEC/mvs.requirements.txt"
  fi
  "$E/mvs/bin/python" -c "import torch,cv2,timm; print('mvs env ok', torch.__version__)"
fi

if [ "$WEIGHTS" = 1 ]; then
  echo "[4/4] weights (sha256-verified)"
  W="$TK/weights"
  PY="$TK/envs/gluemap/bin/python"; [ -x "$PY" ] || PY=python3
  want() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['weights'][sys.argv[2]]['sha256'])" "$CFG" "$1"; }
  verify() { local got; got="$(sha256sum "$W/$1" | cut -d' ' -f1)"; [ "$got" = "$(want "$1")" ] || { echo "SHA256 mismatch for $1 (got $got)" >&2; return 1; }; }
  hf() { "$PY" - "$@" <<'E'
import shutil, sys
from huggingface_hub import hf_hub_download
repo, filename, dest, *rev = sys.argv[1:]
shutil.copyfile(hf_hub_download(repo, filename, revision=(rev[0] if rev else None)), dest)
E
  }
  supply() { # name env_var
    local src="${!2:-}"
    if [ -f "$W/$1" ]; then return; fi
    if [ -n "$src" ] && [ -f "$src" ]; then cp "$src" "$W/$1"; else echo "NOTE: $1 missing; set $2=/path/to/file and re-run" >&2; return; fi
  }
  [ -f "$W/pi3.safetensors" ] || hf yyfz233/Pi3 model.safetensors "$W/pi3.safetensors"
  [ -f "$W/tracker.bin" ] || wget -q --show-progress -O "$W/tracker.bin" "https://huggingface.co/facebook/VGGSfM/resolve/main/vggsfm_v2_tracker.pt"
  [ -f "$W/dg.pth" ] || hf doppelgangers25/doppelgangers_plusplus "checkpoint-dg+visym.pth" "$W/dg.pth"
  [ -f "$W/megaloc.safetensors" ] || hf gberton/MegaLoc model.safetensors "$W/megaloc.safetensors" 37bb43d65dd6388d1578052de5eb0bcdceb497e7
  [ -f "$W/moge3.pt" ] || hf Ruicheng/moge-3-vitl model.pt "$W/moge3.pt" 184008f877d7ad1ad4c2cd2182a9bd1f63d0e5be
  [ -f "$W/edm.ckpt" ] || { src="${ACAM_EDM_CHECKPOINT:-$ROOT/third_party/EDM/weights/edm_outdoor.ckpt}"; [ -f "$src" ] && cp "$src" "$W/edm.ckpt" || echo "NOTE: edm.ckpt missing; set ACAM_EDM_CHECKPOINT=/path/edm_outdoor.ckpt" >&2; }
  supply mvsformerpp.pth ACAM_MVS_CHECKPOINT
  supply mvsformerpp_config.json ACAM_MVS_CONFIG
  missing=0
  for f in pi3.safetensors tracker.bin dg.pth megaloc.safetensors moge3.pt edm.ckpt mvsformerpp.pth mvsformerpp_config.json; do
    if [ -f "$W/$f" ]; then verify "$f" && echo "  ok $f"; else echo "  MISSING $f"; missing=1; fi
  done
  [ "$missing" = 0 ] || echo "Some weights are missing: mapping needs the first five + edm.ckpt for localization; densification also needs the two mvsformerpp files."
fi
echo "done. Run: ./run_moge3.sh VIDEO... --output OUT"
