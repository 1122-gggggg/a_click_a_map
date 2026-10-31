#!/usr/bin/env bash
# 在桌面建立「無人機自動建圖巡航系統」捷徑，之後雙擊即可開啟程式。工程人員安裝完成後執行一次。
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
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
echo "若圖示顯示為文字檔：在桌面上對它按右鍵，選「允許啟動」（Allow Launching）。"
