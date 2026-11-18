#!/usr/bin/env bash
# 無人機自動建圖巡航系統 - desktop application (GTK 3). Double-click the desktop shortcut (安裝桌面捷徑.sh) or run ./gui.sh
# Needs python3 with PyGObject + GTK 3 (Ubuntu desktop: python3-gi, gir1.2-gtk-3.0 are preinstalled) and a graphical session.
#
# 雙擊啟動時看不到終端機，所以所有輸出都寫進 acam_gui/gui.log（超過 1 MB 自動輪轉），
# 錯誤彈窗會直接告訴你這個路徑；要回報問題時把這個檔案給工程人員即可。
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${ACAM_GUI_PYTHON:-/usr/bin/python3}"
LOG="$HERE/acam_gui/gui.log"
mkdir -p "$HERE/acam_gui"
if [ -f "$LOG" ] && [ "$(stat -c%s "$LOG")" -gt 1048576 ]; then
  mv -f "$LOG" "$LOG.1"
fi
if [ -t 2 ]; then
  exec > >(tee -a "$LOG") 2>&1   # 終端機執行：畫面和日誌同時有
else
  exec >>"$LOG" 2>&1             # 雙擊啟動：只寫日誌
fi

fail() { # message; shown in a dialog when started from the desktop (no terminal)
  echo "FATAL: $1" >&2
  if [ ! -t 2 ] && command -v zenity >/dev/null 2>&1; then
    zenity --error --no-markup --title="無人機自動建圖巡航系統" \
      --text="$1

詳細錯誤請看：
$LOG" >/dev/null 2>&1 || true
  fi
  exit 3
}

"$PY" -c "import gi; gi.require_version('Gtk','3.0'); from gi.repository import Gtk" >/dev/null 2>&1 \
  || fail "找不到 GTK 3 的 Python 綁定。請聯絡工程人員安裝：sudo apt install python3-gi gir1.2-gtk-3.0"
[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ] || fail "沒有圖形桌面（DISPLAY 未設定），請在桌面環境中開啟（不要用純文字的遠端連線直接執行）。"

cd "$HERE"
"$PY" "$HERE/gui/app.py" "$@"
code=$?
[ "$code" -eq 0 ] || fail "程式異常結束（代碼 $code）。請聯絡工程人員，並提供上面的日誌檔；或在終端機執行 ./gui.sh 查看即時錯誤訊息。"
