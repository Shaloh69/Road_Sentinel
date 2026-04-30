#!/usr/bin/env bash
# Road Sentinel — Raspberry Pi 4 Setup
# Installs: Camera A (CAM-A-001) + HUB75 LED matrix display
#
# Usage:
#   bash setup_pi4.sh [NODE_URL] [CAM_A_RTSP] [AI_URL]
#
# Defaults:
#   NODE_URL   = http://192.168.8.50:3001
#   CAM_A_RTSP = rtsp://192.168.8.104:554/cam/realmonitor?channel=1&subtype=1
#   AI_URL     = http://192.168.8.50:8000

set -euo pipefail

NODE_URL="${1:-http://192.168.8.50:3001}"
CAM_A_RTSP="${2:-rtsp://192.168.8.104:554/cam/realmonitor?channel=1&subtype=1}"
AI_URL="${3:-http://192.168.8.50:8000}"
CAMERA_ID="CAM-A-001"

VENV="$HOME/venvs/cam_venv"
SCRIPTS_DIR="$HOME/roadsentinel"
LOG_DIR="$HOME/roadsentinel/logs"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================"
echo " Road Sentinel — Pi 4 Setup (Camera A + LED)"
echo "================================================"
echo " Node service : $NODE_URL"
echo " AI service   : $AI_URL"
echo " Camera A     : $CAM_A_RTSP"
echo " Camera ID    : $CAMERA_ID"
echo "================================================"
echo

# ── [1] System packages ────────────────────────────────────────────────────────
echo "[1/6] Installing system packages..."
sudo apt update -q
sudo apt install -y \
    python3-dev python3-pip python3-venv \
    ffmpeg libopencv-dev python3-opencv \
    python3-pil python3-pillow \
    git build-essential
echo "      OK"
echo

# ── [2] Build ledcat (hzeller rpi-rgb-led-matrix) ────────────────────────────
echo "[2/6] Building ledcat (hzeller rpi-rgb-led-matrix)..."
if [ ! -d "$HOME/rpi-rgb-led-matrix" ]; then
    git clone https://github.com/hzeller/rpi-rgb-led-matrix.git "$HOME/rpi-rgb-led-matrix"
fi
make -C "$HOME/rpi-rgb-led-matrix/examples-api-use" ledcat -j2
echo "      ledcat built at $HOME/rpi-rgb-led-matrix/examples-api-use/ledcat"
echo

# ── [3] Python venv ────────────────────────────────────────────────────────────
echo "[3/6] Creating Python venv..."
mkdir -p "$(dirname "$VENV")"
python3 -m venv "$VENV" --system-site-packages
source "$VENV/bin/activate"
pip install --upgrade pip -q
pip install aiohttp requests pillow numpy
python3 -c "import cv2, aiohttp, requests, PIL; print('  deps: OK')"
deactivate
echo "      Venv OK: $VENV"
echo

# ── [4] Copy scripts ───────────────────────────────────────────────────────────
echo "[4/6] Installing scripts..."
mkdir -p "$SCRIPTS_DIR" "$LOG_DIR"
cp "$SRC_DIR/camera/camera_sender.py"       "$SCRIPTS_DIR/camera_sender.py"
cp "$SRC_DIR/lcd_pi4/display_manager.py"    "$SCRIPTS_DIR/display_manager.py"
chmod +x "$SCRIPTS_DIR/camera_sender.py"
chmod +x "$SCRIPTS_DIR/display_manager.py"
echo "      Scripts installed to $SCRIPTS_DIR/"
echo

# ── [5] Systemd services ───────────────────────────────────────────────────────
echo "[5/6] Installing systemd services..."

# Camera sender service
sudo tee /etc/systemd/system/roadsentinel-camera.service > /dev/null <<EOF
[Unit]
Description=Road Sentinel Camera A Sender
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
Type=simple
User=${USER}
WorkingDirectory=${SCRIPTS_DIR}
ExecStart=${VENV}/bin/python3 ${SCRIPTS_DIR}/camera_sender.py \\
    --camera-id ${CAMERA_ID} \\
    --rtsp "${CAM_A_RTSP}" \\
    --ai   ${AI_URL} \\
    --node ${NODE_URL}
Restart=always
RestartSec=5
StandardOutput=append:${LOG_DIR}/camera.log
StandardError=append:${LOG_DIR}/camera.log

[Install]
WantedBy=multi-user.target
EOF

# LED display service
sudo tee /etc/systemd/system/roadsentinel-display.service > /dev/null <<EOF
[Unit]
Description=Road Sentinel LED Matrix Display
After=network-online.target roadsentinel-camera.service
Wants=network-online.target
StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
Type=simple
User=root
WorkingDirectory=${SCRIPTS_DIR}
ExecStart=${VENV}/bin/python3 ${SCRIPTS_DIR}/display_manager.py \\
    --api ${NODE_URL} \\
    --slowdown 4
Restart=always
RestartSec=5
StandardOutput=append:${LOG_DIR}/display.log
StandardError=append:${LOG_DIR}/display.log

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable roadsentinel-camera roadsentinel-display
echo "      Services installed"
echo

# ── [6] Helper scripts ─────────────────────────────────────────────────────────
echo "[6/6] Creating helper scripts..."

cat > "$SCRIPTS_DIR/start.sh" <<'HELPER'
#!/usr/bin/env bash
sudo systemctl start roadsentinel-camera roadsentinel-display
echo "Started. Logs:"
echo "  tail -f ~/roadsentinel/logs/camera.log"
echo "  tail -f ~/roadsentinel/logs/display.log"
HELPER

cat > "$SCRIPTS_DIR/stop.sh" <<'HELPER'
#!/usr/bin/env bash
sudo systemctl stop roadsentinel-camera roadsentinel-display
echo "Stopped."
HELPER

cat > "$SCRIPTS_DIR/status.sh" <<'HELPER'
#!/usr/bin/env bash
echo "=== Camera Sender ==="
sudo systemctl status roadsentinel-camera --no-pager -l | tail -15
echo
echo "=== LED Display ==="
sudo systemctl status roadsentinel-display --no-pager -l | tail -15
HELPER

cat > "$SCRIPTS_DIR/test_display.sh" <<HELPER
#!/usr/bin/env bash
# Run display in TEST mode (cycles fake alerts, no network needed)
sudo ${VENV}/bin/python3 ${SCRIPTS_DIR}/display_manager.py --test
HELPER

chmod +x "$SCRIPTS_DIR/start.sh" "$SCRIPTS_DIR/stop.sh" \
         "$SCRIPTS_DIR/status.sh" "$SCRIPTS_DIR/test_display.sh"

echo
echo "================================================"
echo " Pi 4 Setup Complete!"
echo "================================================"
echo
echo " Services (start on every boot):"
echo "   roadsentinel-camera  — Camera A → AI → Node"
echo "   roadsentinel-display — HUB75 LED matrix"
echo
echo " Quick commands:"
echo "   $SCRIPTS_DIR/start.sh        — start both"
echo "   $SCRIPTS_DIR/stop.sh         — stop both"
echo "   $SCRIPTS_DIR/status.sh       — check status"
echo "   $SCRIPTS_DIR/test_display.sh — test LED with fake alerts"
echo
echo " Live logs:"
echo "   tail -f $LOG_DIR/camera.log"
echo "   tail -f $LOG_DIR/display.log"
echo
echo " Starting services now..."
sudo systemctl start roadsentinel-camera roadsentinel-display
echo " Done!"
echo "================================================"
