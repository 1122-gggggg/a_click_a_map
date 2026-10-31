#!/usr/bin/env bash
# 無人機自動建圖巡航系統 - local web UI (standard-library Python only). ./gui.sh [--port 8765]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${ACAM_GUI_PYTHON:-python3}" "$HERE/gui/server.py" "$@"
