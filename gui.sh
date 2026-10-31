#!/usr/bin/env bash
# 無人機自動建圖巡航系統 - desktop application (GTK 3). Double-click the desktop shortcut (安裝桌面捷徑.sh) or run ./gui.sh
# Needs python3 with PyGObject + GTK 3 (Ubuntu desktop: python3-gi, gir1.2-gtk-3.0 are preinstalled) and a graphical session.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${ACAM_GUI_PYTHON:-/usr/bin/python3}"

fail() { # message; shown in a dialog when started from the desktop (no terminal)
  echo "$1" >&2
  if [ ! -t 2 ] && command -v zenity >/dev/null 2>&1; then zenity --error --no-markup --title="無人機自動建圖巡航系統" --text="$1" >/dev/null 2>&1 || true; fi
  exit 3
}

"$PY" -c "import gi; gi.require_version('Gtk','3.0'); from gi.repository import Gtk" >/dev/null 2>&1 \
  || fail "找不到 GTK 3 的 Python 綁定。請聯絡工程人員安裝：sudo apt install python3-gi gir1.2-gtk-3.0"
[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] || fail "沒有圖形桌面（DISPLAY 未設定），請在桌面環境中開啟。"

cd "$HERE"
"$PY" "$HERE/gui/app.py" "$@"
code=$?
[ "$code" -eq 0 ] || fail "程式異常結束（代碼 $code）。請聯絡工程人員，並在終端機執行 ./gui.sh 查看錯誤訊息。"
