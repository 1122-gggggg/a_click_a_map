#!/usr/bin/env bash
# 在桌面建立「無人機自動建圖巡航系統」捷徑，之後雙擊即可開啟程式。工程人員安裝完成後執行一次。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || true)"
# 有些機器的桌面指向家目錄、或根本沒有 Desktop 資料夾，圖示會埋沒或不顯示：
# 統一改用 ~/Desktop（只動桌面指向，不動下載、文件等其他目錄）。
if [ "$DESKTOP_DIR" != "$HOME/Desktop" ]; then
  mkdir -p "$HOME/Desktop"
  if [ -f "$HOME/.config/user-dirs.dirs" ]; then
    sed -i 's|^XDG_DESKTOP_DIR=.*|XDG_DESKTOP_DIR="$HOME/Desktop"|' "$HOME/.config/user-dirs.dirs"
  else
    echo 'XDG_DESKTOP_DIR="$HOME/Desktop"' >> "$HOME/.config/user-dirs.dirs"
  fi
  DESKTOP_DIR="$HOME/Desktop"
fi
mkdir -p "$DESKTOP_DIR" "$HOME/.local/share/applications"
FILE="$DESKTOP_DIR/無人機自動建圖巡航系統.desktop"
cat > "$FILE" <<E
[Desktop Entry]
Type=Application
Name=無人機自動建圖巡航系統
Comment=輸入影片自動建圖，並用地圖做巡航定位
Exec="$HERE/gui.sh"
Path=$HERE
Terminal=false
Categories=Utility;
E
chmod +x "$FILE"
cp "$FILE" "$HOME/.local/share/applications/"
gio set "$FILE" metadata::trusted true 2>/dev/null || true
echo "已建立桌面捷徑：$FILE"
if command -v desktop-file-validate >/dev/null 2>&1; then
  desktop-file-validate "$FILE" && echo "捷徑格式檢查通過"
fi
if [ ! -x "$HERE/gui.sh" ]; then
  echo "警告：gui.sh 沒有執行權限，執行 chmod +x \"$HERE/gui.sh\" 修復" >&2
fi
echo "若圖示顯示為文字檔：在桌面上對它按右鍵，選「允許啟動」（Allow Launching）。"
echo "若桌面沒出現圖示：登出再登入一次（讓系統重新讀取桌面位置）；或按 Super 鍵搜尋「無人機」從應用程式啟動。"
echo "雙擊沒反應時，在終端機執行 \"$HERE/gui.sh\" 會顯示真正的錯誤；啟動日誌在 $HERE/acam_gui/gui.log。"
