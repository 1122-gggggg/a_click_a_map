#!/usr/bin/env bash
# MoGe-3 entry point: videos -> sparse RGB PLY + MVSFormer++ dense PLY + EDM/MegaLoc localization bundle.
#   ./run_moge3.sh VIDEO [VIDEO ...] --output OUT [--intrinsics K.json | --auto_calibrate] [--skip-dense]
# One-time setup: moge3/scripts/setup.sh   (see moge3/README.md). The original entry point is ./run.sh.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TK="${MOGE3_TOOLKIT:-$HERE/third_party/moge3}"
PY="$TK/envs/gluemap/bin/python"
[ -x "$PY" ] || { echo "MoGe-3 toolkit not set up: run moge3/scripts/setup.sh (expected $PY)" >&2; exit 3; }
export TOOLKIT_ROOT="$TK"
exec "$PY" "$HERE/moge3/portable_pipeline/run_pipeline.py" --toolkit "$TK" "$@"
