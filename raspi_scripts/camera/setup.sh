#!/usr/bin/env bash
# Road Sentinel — Camera Setup Script
# Raspberry Pi 4 Model B  AND  Raspberry Pi 5
#
# Sets up RTSP → AI-service frame pipeline. LED display not included.
#
# Usage:
#   bash setup.sh [AI_SERVER_IP] [CAM_A_RTSP] [CAM_B_RTSP]
#
# Defaults:
#   AI_SERVER_IP = 192.168.8.50
#   CAM_A_RTSP   = rtsp://192.168.8.104:554/cam/realmonitor
#   CAM_B_RTSP   = rtsp://192.168.8.108:554/cam/realmonitor
#
# What this script does:
#   1. Detects Pi model (4 vs 5)
#   2. Installs system dependencies (libopencv, ffmpeg)
#   3. Creates Python venv at ~/venvs/cam_venv
#   4. Installs Python packages (opencv-python-headless, aiohttp)
#   5. Copies camera_sender.py to ~/camera_scripts/
#   6. Generates ~/camera_scripts/launch_cameras.sh
#   7. Installs systemd services (one per camera)
#   8. Starts both services immediately

set -euo pipefail

# ── Arguments / defaults ──────────────────────────────────────────────────────
AI_SERVER_IP="${1:-192.168.8.50}"
AI_URL="http://${AI_SERVER_IP}:8000"
CAM_A_RTSP="${2:-rtsp://192.168.8.104:554/cam/realmonitor}"
CAM_B_RTSP="${3:-rtsp://192.168.8.108:554/cam/realmonitor}"
TARGET_FPS="${4:-30}"

VENV="$HOME/venvs/cam_venv"
SCRIPTS_DIR="$HOME/camera_scripts"
LOG_DIR="$HOME/camera_logs"
SCRIPT_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Pi model detection ────────────────────────────────────────────────────────
PI_MODEL=$(grep -i "Model" /proc/cpuinfo | head -1 || echo "Unknown")
if echo "$PI_MODEL" | grep -qi "Pi 5"; then
    PI_GEN=5
elif echo "$PI_MODEL" | grep -qi "Pi 4"; then
    PI_GEN=4
else
    PI_GEN=4   # default assumption
fi

echo "================================================"
echo " Road Sentinel — Camera Setup"
echo "================================================"
echo " Pi model    : $PI_MODEL"
echo " Pi gen      : $PI_GEN"
echo " AI server   : $AI_URL"
echo " Camera A    : $CAM_A_RTSP"
echo " Camera B    : $CAM_B_RTSP"
echo " Target FPS  : $TARGET_FPS"
echo " Venv        : $VENV"
echo " Scripts dir : $SCRIPTS_DIR"
echo "================================================"
echo

# ── Step 1 — System packages ──────────────────────────────────────────────────
echo "[1/5] Installing system packages..."
sudo apt update -q
sudo apt install -y \
    python3-dev python3-pip python3-venv \
    ffmpeg \
    libopencv-dev python3-opencv \
    v4l-utils

if [ "$PI_GEN" -ge 5 ]; then
    echo "       Pi 5: installing libcamera tools..."
    sudo apt install -y python3-libcamera python3-picamera2 2>/dev/null || true
fi
echo "       System packages OK"
echo

# ── Step 2 — Python venv ──────────────────────────────────────────────────────
echo "[2/5] Creating Python venv at $VENV..."
mkdir -p "$VENV"
python3 -m venv "$VENV" --system-site-packages
# --system-site-packages inherits python3-opencv from apt (saves recompiling)

source "$VENV/bin/activate"
pip install --upgrade pip -q
pip install \
    aiohttp \
    numpy

# Try headless opencv via pip as a fallback if system package missing
python3 -c "import cv2" 2>/dev/null || pip install opencv-python-headless

python3 -c "import cv2, aiohttp; print('  cv2:', cv2.__version__, '  aiohttp: OK')"
echo "       Venv OK"
deactivate
echo

# ── Step 3 — Copy scripts ─────────────────────────────────────────────────────
echo "[3/5] Installing camera_sender.py..."
mkdir -p "$SCRIPTS_DIR"
mkdir -p "$LOG_DIR"
cp "$SCRIPT_SRC/camera_sender.py" "$SCRIPTS_DIR/camera_sender.py"
chmod +x "$SCRIPTS_DIR/camera_sender.py"
echo "       Copied to $SCRIPTS_DIR/camera_sender.py"
echo

# ── Step 4 — Systemd services ─────────────────────────────────────────────────
echo "[4/5] Installing systemd services..."

install_service() {
    local CAM_ID="$1"
    local RTSP_URL="$2"
    local SVC_NAME="roadsentinel-camera-${CAM_ID}"

    sudo tee "/etc/systemd/system/${SVC_NAME}.service" > /dev/null <<EOF
[Unit]
Description=Road Sentinel Camera Sender — ${CAM_ID}
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
Type=simple
User=${USER}
WorkingDirectory=${SCRIPTS_DIR}
ExecStart=${VENV}/bin/python3 ${SCRIPTS_DIR}/camera_sender.py \\
    --camera-id ${CAM_ID} \\
    --rtsp ${RTSP_URL} \\
    --ai ${AI_URL} \\
    --fps ${TARGET_FPS}
Restart=always
RestartSec=5
StandardOutput=append:${LOG_DIR}/camera_${CAM_ID}.log
StandardError=append:${LOG_DIR}/camera_${CAM_ID}.log

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable "${SVC_NAME}"
    echo "       Installed: ${SVC_NAME}"
}

install_service "cam_a" "$CAM_A_RTSP"
install_service "cam_b" "$CAM_B_RTSP"
echo

# ── Step 5 — Launch helper script ─────────────────────────────────────────────
echo "[5/5] Generating helper scripts..."

cat > "$SCRIPTS_DIR/start_cameras.sh" <<EOF
#!/usr/bin/env bash
# Start both Road Sentinel camera services
sudo systemctl start roadsentinel-camera-cam_a
sudo systemctl start roadsentinel-camera-cam_b
echo "Started. Follow logs:"
echo "  tail -f $LOG_DIR/camera_cam_a.log"
echo "  tail -f $LOG_DIR/camera_cam_b.log"
EOF
chmod +x "$SCRIPTS_DIR/start_cameras.sh"

cat > "$SCRIPTS_DIR/stop_cameras.sh" <<EOF
#!/usr/bin/env bash
# Stop both Road Sentinel camera services
sudo systemctl stop roadsentinel-camera-cam_a roadsentinel-camera-cam_b
echo "Stopped."
EOF
chmod +x "$SCRIPTS_DIR/stop_cameras.sh"

cat > "$SCRIPTS_DIR/status_cameras.sh" <<EOF
#!/usr/bin/env bash
echo "=== Camera A ==="
sudo systemctl status roadsentinel-camera-cam_a --no-pager -l | tail -20
echo
echo "=== Camera B ==="
sudo systemctl status roadsentinel-camera-cam_b --no-pager -l | tail -20
EOF
chmod +x "$SCRIPTS_DIR/status_cameras.sh"

cat > "$SCRIPTS_DIR/test_ai_connection.sh" <<EOF
#!/usr/bin/env bash
# Quick test: can the Pi reach the AI server?
echo "Testing connection to ${AI_URL}..."
if curl -sf "${AI_URL}/health" | python3 -m json.tool; then
    echo "AI server is reachable."
else
    echo "ERROR: Cannot reach ${AI_URL}/health"
    echo "Check that the AI service is running on the server and the Pi can ping ${AI_SERVER_IP}"
fi
EOF
chmod +x "$SCRIPTS_DIR/test_ai_connection.sh"

echo "       Helper scripts created in $SCRIPTS_DIR/"
echo

# ── Start services ────────────────────────────────────────────────────────────
echo "Starting services..."
sudo systemctl start roadsentinel-camera-cam_a roadsentinel-camera-cam_b

echo
echo "================================================"
echo " Setup complete!"
echo "================================================"
echo
echo " Services installed (start on every boot):"
echo "   roadsentinel-camera-cam_a"
echo "   roadsentinel-camera-cam_b"
echo
echo " Useful commands:"
echo "   $SCRIPTS_DIR/status_cameras.sh    — check status"
echo "   $SCRIPTS_DIR/stop_cameras.sh      — stop both"
echo "   $SCRIPTS_DIR/start_cameras.sh     — start both"
echo "   $SCRIPTS_DIR/test_ai_connection.sh — ping AI server"
echo
echo " Live logs:"
echo "   tail -f $LOG_DIR/camera_cam_a.log"
echo "   tail -f $LOG_DIR/camera_cam_b.log"
echo
echo " To reconfigure (different IP / RTSP URL):"
echo "   bash setup.sh <AI_SERVER_IP> <CAM_A_RTSP> <CAM_B_RTSP>"
echo
echo " Pi model: Pi $PI_GEN — no Pi-specific changes needed for RTSP cameras."
echo " Both Pi 4 and Pi 5 use OpenCV + aiohttp for RTSP capture and HTTP POST."
echo "================================================"
