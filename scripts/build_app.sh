#!/usr/bin/env bash
# Builds "LungVision AI.app" (launcher bundle). Usage: scripts/build_app.sh [dest_dir]
set -euo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"; DEST="${1:-$HOME/Desktop}"; APP="$DEST/LungVision AI.app"
rm -rf "$APP"; mkdir -p "$APP/Contents/MacOS"
cat > "$APP/Contents/Info.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict><key>CFBundleName</key><string>LungVision AI</string><key>CFBundleDisplayName</key><string>LungVision AI</string>
<key>CFBundleIdentifier</key><string>local.lungvision.ai</string><key>CFBundleExecutable</key><string>launcher</string>
<key>CFBundlePackageType</key><string>APPL</string><key>CFBundleVersion</key><string>0.1</string><key>LSMinimumSystemVersion</key><string>13.0</string></dict></plist>
PL
cat > "$APP/Contents/MacOS/launcher" <<LA
#!/bin/bash
PROJ="$PROJ"; PORT=8000; URL="http://127.0.0.1:\$PORT"; LOG="\$PROJ/outputs/app_server.log"; mkdir -p "\$PROJ/outputs"
CKPT="\${LV_CKPT:-\$PROJ/experiments/pilot_unet16_v2/best.pt}"
if [ ! -f "\$CKPT" ]; then osascript -e 'display alert "LungVision AI" message "No trained checkpoint found."'; exit 1; fi
STARTED=0
if ! curl -s -m 2 "\$URL/health" >/dev/null; then
  cd "\$PROJ" && LV_CKPT="\$CKPT" nohup "\$PROJ/.venv/bin/uvicorn" app.api:app --host 127.0.0.1 --port \$PORT >"\$LOG" 2>&1 &
  SPID=\$!; STARTED=1
  for i in \$(seq 1 40); do curl -s -m 1 "\$URL/health" >/dev/null && break; sleep 0.5; done
fi
open "\$URL/ui"
if [ "\$STARTED" = 1 ]; then
  osascript -e 'display dialog "LungVision AI is running locally at http://127.0.0.1:8000/ui\n\nResearch use only. Not for clinical use.\nScans stay on this Mac." with title "LungVision AI" buttons {"Stop server"} default button 1' >/dev/null 2>&1
  kill \$SPID 2>/dev/null; pkill -f "uvicorn app.api:app" 2>/dev/null
fi
LA
chmod +x "$APP/Contents/MacOS/launcher"; echo "built $APP"
