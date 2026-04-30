#!/usr/bin/env bash
# Road Sentinel — Raspberry Pi 5 Setup
# Installs: Camera B (CAM-B-002) only — no LED display
#
# Usage:
#   bash setup_pi5.sh [NODE_URL] [CAM_B_RTSP] [AI_URL]
#
# Defaults:
#   NODE_URL   = http://192.168.8.50:3001
#   CAM_B_RTSP = rtsp://192.168.8.108:554/cam/realmonitor?channel=1&subtype=1
#   AI_URL     = http://192.168.8.50:8000

set -euo pipefail

NODE_URL="${1:-http://192.168.8.50:3001}"
CAM_B_RTSP="${2:-rtsp://192.168.8.108:554/cam/realmonitor?channel=1&subtype=1}"
AI_URL="${3:-http://192.168.8.50:8000}"
CAMERA_ID="CAM-B-002"

VENV="$HOME/venvs/cam_venv"
SCRIPTS_DIR="$HOME/roadsentinel"
LOG_DIR="$HOME/roadsentinel/logs"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================"
echo " Road Sentinel — Pi 5 Setup (Camera B)"
echo "================================================"
echo " Node service : $NODE_URL"
echo " AI service   : $AI_URL"
echo " Camera B     : $CAM_B_RTSP"
echo " Camera ID    : $CAMERA_ID"
echo "================================================"
echo

# ── [1] System packages ────────────────────────────────────────────────────────
echo "[1/4] Installing system packages..."
sudo apt update -q
sudo apt install -y \
    python3-dev python3-pip python3-venv \
    ffmpeg libopencv-dev python3-opencv
echo "      OK"
echo

# ── [2] Python venv ────────────────────────────────────────────────────────────
echo "[2/4] Creating Python venv..."
mkdir -p "$(dirname "$VENV")"
python3 -m venv "$VENV" --system-site-packages
source "$VENV/bin/activate"
pip install --upgrade pip -q
pip install aiohttp numpy
python3 -c "import cv2, aiohttp; print('  deps: OK')"
deactivate
echo "      Venv OK: $VENV"
echo

# ── [3] Copy scripts ───────────────────────────────────────────────────────────
echo "[3/4] Installing scripts..."
mkdir -p "$SCRIPTS_DIR" "$LOG_DIR"
cp "$SRC_DIR/camera/camera_sender.py" "$SCRIPTS_DIR/camera_sender.py"
chmod +x "$SCRIPTS_DIR/camera_sender.py"
echo "      Scripts installed to $SCRIPTS_DIR/"
echo

# ── [4] Systemd service ────────────────────────────────────────────────────────
echo "[4/4] Installing systemd service..."

sudo tee /etc/systemd/system/roadsentinel-camera.service > /dev/null <<EOF
[Unit]
Description=Road Sentinel Camera B Sender
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
    --rtsp "${CAM_B_RTSP}" \\
    --ai   ${AI_URL} \\
    --node ${NODE_URL}
Restart=always
RestartSec=5
StandardOutput=append:${LOG_DIR}/camera.log
StandardError=append:${LOG_DIR}/camera.log

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable roadsentinel-camera

cat > "$SCRIPTS_DIR/start.sh" <<'HELPER'
#!/usr/bin/env bash
sudo systemctl start roadsentinel-camera
echo "Started. Log: tail -f ~/roadsentinel/logs/camera.log"
HELPER

cat > "$SCRIPTS_DIR/stop.sh" <<'HELPER'
#!/usr/bin/env bash
sudo systemctl stop roadsentinel-camera
echo "Stopped."
HELPER

cat > "$SCRIPTS_DIR/status.sh" <<'HELPER'
#!/usr/bin/env bash
sudo systemctl status roadsentinel-camera --no-pager -l | tail -20
HELPER

chmod +x "$SCRIPTS_DIR/start.sh" "$SCRIPTS_DIR/stop.sh" "$SCRIPTS_DIR/status.sh"

echo
echo "================================================"
echo " Pi 5 Setup Complete!"
echo "================================================"
echo
echo " Service (starts on every boot):"
echo "   roadsentinel-camera  — Camera B → AI → Node"
echo
echo " Quick commands:"
echo "   $SCRIPTS_DIR/start.sh    — start"
echo "   $SCRIPTS_DIR/stop.sh     — stop"
echo "   $SCRIPTS_DIR/status.sh   — check status"
echo
echo " Live log:"
echo "   tail -f $LOG_DIR/camera.log"
echo
echo " Starting service now..."
sudo systemctl start roadsentinel-camera
echo " Done!"
echo "================================================"
