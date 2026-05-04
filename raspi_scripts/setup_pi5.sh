#!/usr/bin/env bash
# Road Sentinel — Raspberry Pi 5 Setup
# Installs: Camera B (CAM-B-002) + HUB75 LED matrix display (rp1-rio backend)
#
# Usage:
#   bash setup_pi5.sh [NODE_URL] [CAM_B_RTSP] [AI_URL]
#
# Defaults:
#   NODE_URL   = http://192.168.8.50:3001
#   CAM_B_RTSP = rtsp://192.168.8.108:554/cam/realmonitor?channel=1&subtype=1
#   AI_URL     = http://192.168.8.50:8000
#
# Pi 5 note: uses --led-rp1-rio=1 (RP1 GPIO chip) for the LED matrix.
#            If display is corrupt, try --led-rp1-rio=0 in display.service.

set -euo pipefail

NODE_URL="${1:-http://192.168.8.50:3001}"
CAM_B_RTSP="${2:-rtsp://192.168.8.108:554/cam/realmonitor?channel=1&subtype=1}"
AI_URL="${3:-http://192.168.8.50:8000}"
CAMERA_ID="CAM-B-002"

VENV="$HOME/venvs/cam_venv"
SCRIPTS_DIR="$HOME/roadsentinel"
LOG_DIR="$HOME/roadsentinel/logs"
REPO_DIR="$HOME/roadsentinel-repo"
REPO_URL="https://github.com/vandrepaul01/RoadSentinel.git"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pi 5 LED matrix flags (RP1 GPIO chip)
# rp1-rio=1 — uses new RP1 hardware GPIO; slowdown 3 is typical for Pi 5
LED_RPI_RIO=1
LED_SLOWDOWN=3

echo "================================================"
echo " Road Sentinel — Pi 5 Setup (Camera B + LED)"
echo "================================================"
echo " Node service : $NODE_URL"
echo " AI service   : $AI_URL"
echo " Camera B     : $CAM_B_RTSP"
echo " Camera ID    : $CAMERA_ID"
echo " LED backend  : --led-rp1-rio=$LED_RPI_RIO  --led-slowdown=$LED_SLOWDOWN"
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
SRC_DIR="$REPO_DIR/raspi_scripts"
echo "      Repo at $REPO_DIR"
echo

# ── [2] Build ledcat (rp1-rio backend for Pi 5) ────────────────────────────
echo "[2/7] Building ledcat (hzeller rpi-rgb-led-matrix, Pi 5 rp1-rio)..."
if [ ! -d "$HOME/rpi-rgb-led-matrix" ]; then
    git clone https://github.com/hzeller/rpi-rgb-led-matrix.git "$HOME/rpi-rgb-led-matrix"
else
    git -C "$HOME/rpi-rgb-led-matrix" pull
fi
# Build with HARDWARE_DESC=adafruit-hat if using Adafruit bonnet, else leave default
make -C "$HOME/rpi-rgb-led-matrix/examples-api-use" ledcat -j2
echo "      ledcat built at $HOME/rpi-rgb-led-matrix/examples-api-use/ledcat"
echo

# ── [3] Python venv ────────────────────────────────────────────────────────────
echo "[3/7] Creating Python venv..."
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
echo "[4/7] Installing scripts..."
mkdir -p "$SCRIPTS_DIR" "$LOG_DIR"
# Use the unified display_manager.py (auto-detects Pi 4 vs Pi 5 via /dev/pio0)
cp "$SRC_DIR/camera/camera_sender.py"    "$SCRIPTS_DIR/camera_sender.py"
cp "$SRC_DIR/display_manager.py"         "$SCRIPTS_DIR/display_manager.py"
chmod +x "$SCRIPTS_DIR/camera_sender.py"
chmod +x "$SCRIPTS_DIR/display_manager.py"
echo "      Scripts installed to $SCRIPTS_DIR/"
echo

# ── [5] Systemd services ───────────────────────────────────────────────────────
echo "[5/7] Installing systemd services..."

# Camera sender service
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

# LED display service — Pi 5 uses --led-rp1-rio flag
# Note: slowdown inverts when rp1-rio=1; start at 3 and tune if flickering
sudo tee /etc/systemd/system/roadsentinel-display.service > /dev/null <<EOF
[Unit]
Description=Road Sentinel LED Matrix Display (Pi 5 rp1-rio)
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
    --slowdown ${LED_SLOWDOWN} \\
    --rp1-rio ${LED_RPI_RIO}
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
echo "[6/7] Creating helper scripts..."

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
# Pi 5: pass --rp1-rio flag to display_manager
sudo ${VENV}/bin/python3 ${SCRIPTS_DIR}/display_manager.py --test \\
    --slowdown ${LED_SLOWDOWN} --rp1-rio ${LED_RPI_RIO}
HELPER

cat > "$SCRIPTS_DIR/update.sh" <<HELPER
#!/usr/bin/env bash
# Pull latest raspi_scripts from GitHub and restart services
set -euo pipefail
echo "Pulling latest from GitHub..."
git -C ${REPO_DIR} pull origin main
echo "Copying updated scripts..."
cp "${REPO_DIR}/raspi_scripts/camera/camera_sender.py" "${SCRIPTS_DIR}/camera_sender.py"
cp "${REPO_DIR}/raspi_scripts/display_manager.py"      "${SCRIPTS_DIR}/display_manager.py"
chmod +x "${SCRIPTS_DIR}/camera_sender.py" "${SCRIPTS_DIR}/display_manager.py"
echo "Restarting services..."
sudo systemctl restart roadsentinel-camera roadsentinel-display
echo "Done! Services restarted with latest code."
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
echo " Pi 5 Setup Complete!"
echo "================================================"
echo
echo " Services (start on every boot):"
echo "   roadsentinel-camera  — Camera B → AI → Node"
echo "   roadsentinel-display — HUB75 LED matrix (rp1-rio=$LED_RPI_RIO)"
echo
echo " Quick commands:"
echo "   $SCRIPTS_DIR/start.sh        — start both"
echo "   $SCRIPTS_DIR/stop.sh         — stop both"
echo "   $SCRIPTS_DIR/status.sh       — check status"
echo "   $SCRIPTS_DIR/update.sh       — git pull + restart (or use Admin Terminal)"
echo "   $SCRIPTS_DIR/test_display.sh — test LED with fake alerts"
echo
echo " If LED display flickers/corrupts, edit display.service and try:"
echo "   --led-rp1-rio=0  or  --led-slowdown=2 (or 4)"
echo
echo " Live logs:"
echo "   tail -f $LOG_DIR/camera.log"
echo "   tail -f $LOG_DIR/display.log"
echo
echo " Starting services now..."
sudo systemctl start roadsentinel-camera roadsentinel-display
echo " Done!"
echo "================================================"
