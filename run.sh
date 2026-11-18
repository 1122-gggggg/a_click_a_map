#!/usr/bin/env bash
# Original mapper: ./run.sh /path/to/videos --intrinsics camera.json (see README.md)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${ACAM_PYTHON:-python3}"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -m acam "$@"
