#!/usr/bin/env bash
# Fetch and pin the third-party engine pieces. Idempotent.
#   scripts/setup.sh [--build] [--no-checkpoints]
#   --build            also `pip install -e third_party/gluemap` (needs Ceres/Eigen/METIS/Boost, see README)
#   --no-checkpoints   skip the ~8 GB of downloads
# EDM weights are NOT downloaded: place edm_outdoor.ckpt at third_party/EDM/weights/ (or set ACAM_EDM_CHECKPOINT).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ACAM_PYTHON:-python3}"
TP="$HERE/third_party"
CK="$HERE/configs/checkpoints.validated.json"
BUILD=0; DOWNLOAD=1
for arg in "$@"; do
  case "$arg" in --build) BUILD=1 ;; --no-checkpoints) DOWNLOAD=0 ;; *) echo "unknown option $arg" >&2; exit 2 ;; esac
done
pin() { "$PY" - "$CK" "$1" <<'E'
import json,sys
d=json.load(open(sys.argv[1]))["sources"]; k=sys.argv[2]
print({"gluemap":d["gluemap"]["commit"],"edm":d["edm"]["commit"],"megaloc":"5fe0dd697c4a70ba3e23607f6716ab3c606b16db"}[k])
E
}
sha() { "$PY" - "$CK" "$1" <<'E'
import json,sys
d=json.load(open(sys.argv[1]))["sources"]; k=sys.argv[2]
print(d["gluemap"]["checkpoints"].get(k) or (d["edm"]["sha256"] if k=="edm" else d["megaloc"]["sha256"]))
E
}
verify() { # file key
  local got; got="$(sha256sum "$1" | cut -d' ' -f1)"
  [ "$got" = "$(sha "$2")" ] || { echo "SHA256 mismatch for $1 (got $got)" >&2; return 1; }
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

mkdir -p "$TP"
echo "[1/4] GlueMap (colmap/gluemap @ $(pin gluemap | cut -c1-9)) + local patches"
clone_pinned https://github.com/colmap/gluemap "$TP/gluemap" "$(pin gluemap)"
git -C "$TP/gluemap" submodule update --init --recursive
apply_patch "$TP/gluemap" "$HERE/patches/gluemap.patch"
echo "[2/4] EDM (chicleee/EDM @ $(pin edm | cut -c1-9))"
clone_pinned https://github.com/chicleee/EDM "$TP/EDM" "$(pin edm)"
apply_patch "$TP/EDM" "$HERE/patches/edm.patch"
echo "[3/4] MegaLoc (gmberton/MegaLoc @ $(pin megaloc | cut -c1-9))"
clone_pinned https://github.com/gmberton/MegaLoc "$TP/MegaLoc" "$(pin megaloc)"

if [ "$BUILD" = 1 ]; then
  echo "building pygluemap (editable)"; (cd "$TP/gluemap" && CMAKE_PREFIX_PATH="${CONDA_PREFIX:-}" "$PY" -m pip install -e .)
fi

if [ "$DOWNLOAD" = 1 ]; then
  echo "[4/4] checkpoints (sha256-verified)"
  GC="$TP/gluemap/checkpoints"; mkdir -p "$GC" "$TP/checkpoints/megaloc"
  [ -f "$GC/dino_salad.ckpt" ] || wget -q --show-progress -O "$GC/dino_salad.ckpt" https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt
  [ -f "$GC/vggsfm_v2_0_0_track_predictor.bin" ] || wget -q --show-progress -O "$GC/vggsfm_v2_0_0_track_predictor.bin" https://huggingface.co/facebook/VGGSfM/resolve/main/vggsfm_v2_tracker.pt
  hf() { "$PY" - "$@" <<'E'
import shutil,sys
from huggingface_hub import hf_hub_download
repo,filename,dest,*rev=sys.argv[1:]
path=hf_hub_download(repo,filename,revision=(rev[0] if rev else None))
shutil.copyfile(path,dest)
E
  }
  [ -f "$GC/pi3.safetensors" ] || hf yyfz233/Pi3 model.safetensors "$GC/pi3.safetensors"
  [ -f "$GC/checkpoint-dg+visym.pth" ] || hf doppelgangers25/doppelgangers_plusplus "checkpoint-dg+visym.pth" "$GC/checkpoint-dg+visym.pth"
  [ -f "$TP/checkpoints/megaloc/model.safetensors" ] || hf gberton/MegaLoc model.safetensors "$TP/checkpoints/megaloc/model.safetensors" 37bb43d65dd6388d1578052de5eb0bcdceb497e7
  for f in checkpoint-dg+visym.pth pi3.safetensors dino_salad.ckpt vggsfm_v2_0_0_track_predictor.bin; do verify "$GC/$f" "$f"; done
  verify "$TP/checkpoints/megaloc/model.safetensors" megaloc
fi

if [ -f "$TP/EDM/weights/edm_outdoor.ckpt" ]; then verify "$TP/EDM/weights/edm_outdoor.ckpt" edm && echo "EDM weights present and verified"
else echo "NOTE: EDM weights missing. Put edm_outdoor.ckpt in $TP/EDM/weights/ or export ACAM_EDM_CHECKPOINT=/path/to/edm_outdoor.ckpt"; fi
echo "next: ./run.sh doctor"
