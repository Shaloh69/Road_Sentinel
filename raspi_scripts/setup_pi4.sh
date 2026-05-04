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
REPO_DIR="$HOME/roadsentinel-repo"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="https://github.com/vandrepaul01/RoadSentinel.git"

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
echo "[1/7] Installing system packages..."
sudo apt update -q
sudo apt install -y \
    python3-dev python3-pip python3-venv \
    ffmpeg libopencv-dev python3-opencv \
    python3-pil python3-pillow \
    git build-essential curl
echo "      OK"
echo

# ── [1b] Clone / update repo ────────────────────────────────────────────────
echo "[1b/7] Syncing RoadSentinel repo..."
if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" pull origin main
else
    git clone "$REPO_URL" "$REPO_DIR"
fi
# Point SRC_DIR at the cloned raspi_scripts so installs always use latest
SRC_DIR="$REPO_DIR/raspi_scripts"
echo "      Repo at $REPO_DIR"
echo

# ── [2] Build ledcat (hzeller rpi-rgb-led-matrix) ────────────────────────────
echo "[2/7] Building ledcat (hzeller rpi-rgb-led-matrix)..."
if [ ! -d "$HOME/rpi-rgb-led-matrix" ]; then
    git clone https://github.com/hzeller/rpi-rgb-led-matrix.git "$HOME/rpi-rgb-led-matrix"
fi
make -C "$HOME/rpi-rgb-led-matrix/examples-api-use" ledcat -j2
echo "      ledcat built at $HOME/rpi-rgb-led-matrix/examples-api-use/ledcat"
echo

# ── [3] Python venv ────────────────────────────────────────────────────────────
echo "[3/7] Creating Python venv..."
mkdir -p "$(dirname "$VENV")"
python3 -m venv "$VENV" --system-site-packages
source "$VENV/bin/activate"
pip install --upgrade pip -q
pip install aiohttp requests pillow numpy "python-socketio[client]"
python3 -c "import cv2, aiohttp, requests, PIL, socketio; print('  deps: OK')"
deactivate
echo "      Venv OK: $VENV"
echo

# ── [4] Copy scripts ───────────────────────────────────────────────────────────
echo "[4/7] Installing scripts..."
mkdir -p "$SCRIPTS_DIR" "$LOG_DIR"
cp "$SRC_DIR/camera/camera_sender.py"       "$SCRIPTS_DIR/camera_sender.py"
cp "$SRC_DIR/lcd_pi4/display_manager.py"    "$SCRIPTS_DIR/display_manager.py"
cp "$SRC_DIR/pi_agent.py"                   "$SCRIPTS_DIR/pi_agent.py"
chmod +x "$SCRIPTS_DIR/camera_sender.py"
chmod +x "$SCRIPTS_DIR/display_manager.py"
chmod +x "$SCRIPTS_DIR/pi_agent.py"
echo "      Scripts installed to $SCRIPTS_DIR/"
echo

# ── [5] Systemd services ───────────────────────────────────────────────────────
echo "[5/7] Installing systemd services..."

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

# Pi agent service — connects back to Node service for Admin Terminal remote control
sudo tee /etc/systemd/system/roadsentinel-agent.service > /dev/null <<EOF
[Unit]
Description=Road Sentinel Pi Agent (Admin Terminal relay)
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=60
StartLimitBurst=10

[Service]
Type=simple
User=${USER}
WorkingDirectory=${SCRIPTS_DIR}
ExecStart=${VENV}/bin/python3 ${SCRIPTS_DIR}/pi_agent.py \\
    --node ${NODE_URL} \\
    --id   pi4
Restart=always
RestartSec=5
StandardOutput=append:${LOG_DIR}/agent.log
StandardError=append:${LOG_DIR}/agent.log

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable roadsentinel-camera roadsentinel-display roadsentinel-agent
echo "      Services installed"
echo

# ── [6] Helper scripts ─────────────────────────────────────────────────────────
echo "[6/7] Creating helper scripts..."

cat > "$SCRIPTS_DIR/start.sh" <<'HELPER'
#!/usr/bin/env bash
sudo systemctl start roadsentinel-camera roadsentinel-display roadsentinel-agent
echo "Started. Logs:"
echo "  tail -f ~/roadsentinel/logs/camera.log"
echo "  tail -f ~/roadsentinel/logs/display.log"
echo "  tail -f ~/roadsentinel/logs/agent.log"
HELPER

cat > "$SCRIPTS_DIR/stop.sh" <<'HELPER'
#!/usr/bin/env bash
sudo systemctl stop roadsentinel-camera roadsentinel-display roadsentinel-agent
echo "Stopped."
HELPER

cat > "$SCRIPTS_DIR/status.sh" <<'HELPER'
#!/usr/bin/env bash
echo "=== Camera Sender ==="
sudo systemctl status roadsentinel-camera --no-pager -l | tail -12
echo
echo "=== LED Display ==="
sudo systemctl status roadsentinel-display --no-pager -l | tail -12
echo
echo "=== Pi Agent (Admin Terminal) ==="
sudo systemctl status roadsentinel-agent --no-pager -l | tail -12
HELPER

cat > "$SCRIPTS_DIR/test_display.sh" <<HELPER
#!/usr/bin/env bash
# Run display in TEST mode (cycles fake alerts, no network needed)
sudo ${VENV}/bin/python3 ${SCRIPTS_DIR}/display_manager.py --test
HELPER

cat > "$SCRIPTS_DIR/update.sh" <<HELPER
#!/usr/bin/env bash
# Pull latest raspi_scripts from GitHub and restart services
# This script can also be triggered from the Admin Terminal in the web UI.
set -euo pipefail
echo "Pulling latest from GitHub..."
git -C ${REPO_DIR} pull origin main
echo "Copying updated scripts..."
cp "${REPO_DIR}/raspi_scripts/camera/camera_sender.py" "${SCRIPTS_DIR}/camera_sender.py"
cp "${REPO_DIR}/raspi_scripts/lcd_pi4/display_manager.py" "${SCRIPTS_DIR}/display_manager.py"
cp "${REPO_DIR}/raspi_scripts/pi_agent.py" "${SCRIPTS_DIR}/pi_agent.py"
chmod +x "${SCRIPTS_DIR}/camera_sender.py" "${SCRIPTS_DIR}/display_manager.py" "${SCRIPTS_DIR}/pi_agent.py"
echo "Restarting services..."
sudo systemctl restart roadsentinel-camera roadsentinel-display roadsentinel-agent
echo "Done! All services restarted with latest code."
HELPER

chmod +x "$SCRIPTS_DIR/start.sh" "$SCRIPTS_DIR/stop.sh" \
         "$SCRIPTS_DIR/status.sh" "$SCRIPTS_DIR/test_display.sh" \
         "$SCRIPTS_DIR/update.sh"

# ── [7] Git remote config ──────────────────────────────────────────────────────
echo "[7/7] Verifying git remote..."
git -C "$REPO_DIR" remote -v
echo "      Run '$SCRIPTS_DIR/update.sh' anytime to pull latest and restart."

echo
echo "================================================"
echo " Pi 4 Setup Complete!"
echo "================================================"
echo
echo " Services (start on every boot):"
echo "   roadsentinel-camera  — Camera A → AI → Node"
echo "   roadsentinel-display — HUB75 LED matrix"
echo "   roadsentinel-agent   — Admin Terminal relay (connects to $NODE_URL)"
echo
echo " Quick commands:"
echo "   $SCRIPTS_DIR/start.sh        — start all"
echo "   $SCRIPTS_DIR/stop.sh         — stop all"
echo "   $SCRIPTS_DIR/status.sh       — check status"
echo "   $SCRIPTS_DIR/update.sh       — git pull + restart (or use Admin Terminal)"
echo "   $SCRIPTS_DIR/test_display.sh — test LED with fake alerts"
echo
echo " Live logs:"
echo "   tail -f $LOG_DIR/camera.log"
echo "   tail -f $LOG_DIR/display.log"
echo "   tail -f $LOG_DIR/agent.log"
echo
echo " Admin Terminal: open the web dashboard → Admin Terminal → select 'Pi 4'"
echo " (the agent must be running and the Pi must reach $NODE_URL)"
echo
echo " Starting services now..."
sudo systemctl start roadsentinel-camera roadsentinel-display roadsentinel-agent
echo " Done!"
echo "================================================"
