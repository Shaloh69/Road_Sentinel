#!/usr/bin/env bash
# Road Sentinel — Raspberry Pi 4 Setup
# Installs: Camera A (CAM-A-001) only.
# Pi 4 has NO LED matrix — the LED is on Pi 5.
# Camera A detections are forwarded to the Node API, which Pi 5 polls to
# update the LED display in real time.
#
# Usage:
#   bash setup_pi4.sh [NODE_URL] [CAM_A_RTSP] [AI_URL]
#
# Defaults:
#   NODE_URL   = http://192.168.8.50:3001
#   CAM_A_RTSP = rtsp://192.168.8.104:554/cam/realmonitor?channel=1&subtype=1
#   AI_URL     = http://192.168.8.50:8000
#
# After setup, SSH via:  ssh pi@pi4-sentinel.local  (no IP needed, ever)

set -euo pipefail

NODE_URL="${1:-http://192.168.8.50:3001}"
CAM_A_RTSP="${2:-rtsp://192.168.8.104:554/cam/realmonitor?channel=1&subtype=1}"
AI_URL="${3:-http://192.168.8.50:8000}"
CAMERA_ID="CAM-A-001"
HOSTNAME="pi4-sentinel"

VENV="$HOME/venvs/cam_venv"
SCRIPTS_DIR="$HOME/roadsentinel"
LOG_DIR="$HOME/roadsentinel/logs"
REPO_DIR="$HOME/roadsentinel-repo"
REPO_URL="https://github.com/Shaloh69/Road_Sentinel.git"

echo "================================================"
echo " Road Sentinel — Pi 4 Setup (Camera A only)"
echo "================================================"
echo " Node service : $NODE_URL"
echo " AI service   : $AI_URL"
echo " Camera A     : $CAM_A_RTSP"
echo " Camera ID    : $CAMERA_ID"
echo " Hostname     : $HOSTNAME"
echo " LED display  : on Pi 5 (this Pi has none)"
echo "================================================"
echo

# ── [0] Set hostname ───────────────────────────────────────────────────────────
echo "[0/6] Setting hostname to '$HOSTNAME'..."
CURRENT_HOSTNAME="$(hostname)"
if [ "$CURRENT_HOSTNAME" != "$HOSTNAME" ]; then
    sudo hostnamectl set-hostname "$HOSTNAME"
    sudo sed -i "s/127\.0\.1\.1.*/127.0.1.1\t$HOSTNAME/" /etc/hosts
    echo "      Hostname changed: $CURRENT_HOSTNAME → $HOSTNAME"
    echo "      SSH after reboot: ssh pi@${HOSTNAME}.local"
else
    echo "      Already set to '$HOSTNAME' — skipping"
fi
echo

# ── [1] System packages ────────────────────────────────────────────────────────
echo "[1/6] Installing system packages..."
sudo apt update -q
sudo apt install -y \
    python3-dev python3-pip python3-venv \
    ffmpeg libopencv-dev python3-opencv \
    python3-pil python3-pillow \
    git build-essential curl
echo "      OK"
echo

# ── [1b] Clone / update repo ───────────────────────────────────────────────────
echo "[1b/6] Syncing RoadSentinel repo..."
if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" pull origin main
else
    git clone "$REPO_URL" "$REPO_DIR"
fi
echo "      Repo at $REPO_DIR"
echo

# ── [2] Python venv ────────────────────────────────────────────────────────────
echo "[2/6] Creating Python venv..."
mkdir -p "$(dirname "$VENV")"
python3 -m venv "$VENV" --system-site-packages
source "$VENV/bin/activate"
pip install --upgrade pip -q
pip install aiohttp requests "pillow>=10.0" numpy "python-socketio[client]"
python3 -c "import cv2, aiohttp, requests, PIL, socketio; print('  deps: OK')"
deactivate
echo "      Venv OK: $VENV"
echo

# ── [3] Copy scripts ───────────────────────────────────────────────────────────
echo "[3/6] Installing scripts..."
mkdir -p "$SCRIPTS_DIR" "$LOG_DIR"
cp "$REPO_DIR/raspi_scripts/camera/camera_sender.py" "$SCRIPTS_DIR/camera_sender.py"
cp "$REPO_DIR/raspi_scripts/pi_agent.py"             "$SCRIPTS_DIR/pi_agent.py"
chmod +x "$SCRIPTS_DIR/camera_sender.py"
chmod +x "$SCRIPTS_DIR/pi_agent.py"
echo "      Scripts installed to $SCRIPTS_DIR/"
echo

# ── [4] Systemd services ───────────────────────────────────────────────────────
echo "[4/6] Installing systemd services..."

# Camera sender service — forwards frames to AI, incidents/detections to Node
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
ExecStart=${VENV}/bin/python3 ${SCRIPTS_DIR}/camera_sender.py \
    --camera-id ${CAMERA_ID} \
    --rtsp "${CAM_A_RTSP}" \
    --ai   ${AI_URL} \
    --node ${NODE_URL}
Restart=always
RestartSec=5
StandardOutput=append:${LOG_DIR}/camera.log
StandardError=append:${LOG_DIR}/camera.log

[Install]
WantedBy=multi-user.target
EOF

# Pi agent — Admin Terminal relay (lets you run commands from the web dashboard)
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
ExecStart=${VENV}/bin/python3 ${SCRIPTS_DIR}/pi_agent.py \
    --node ${NODE_URL} \
    --id   pi4
Restart=always
RestartSec=5
StandardOutput=append:${LOG_DIR}/agent.log
StandardError=append:${LOG_DIR}/agent.log

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable roadsentinel-camera roadsentinel-agent
echo "      Services installed"
echo

# ── [5] Helper scripts ─────────────────────────────────────────────────────────
echo "[5/6] Creating helper scripts..."

cat > "$SCRIPTS_DIR/start.sh" <<'HELPER'
#!/usr/bin/env bash
sudo systemctl start roadsentinel-camera roadsentinel-agent
echo "Started. Logs:"
echo "  tail -f ~/roadsentinel/logs/camera.log"
echo "  tail -f ~/roadsentinel/logs/agent.log"
HELPER

cat > "$SCRIPTS_DIR/stop.sh" <<'HELPER'
#!/usr/bin/env bash
sudo systemctl stop roadsentinel-camera roadsentinel-agent
echo "Stopped."
HELPER

cat > "$SCRIPTS_DIR/status.sh" <<'HELPER'
#!/usr/bin/env bash
echo "=== Camera Sender ==="
sudo systemctl status roadsentinel-camera --no-pager -l | tail -12
echo
echo "=== Pi Agent (Admin Terminal) ==="
sudo systemctl status roadsentinel-agent --no-pager -l | tail -12
HELPER

cat > "$SCRIPTS_DIR/update.sh" <<HELPER
#!/usr/bin/env bash
# Pull latest raspi_scripts from GitHub and restart services.
set -euo pipefail
REPO_DIR="$REPO_DIR"
SCRIPTS_DIR="$SCRIPTS_DIR"
echo "Pulling latest from GitHub..."
git -C "\$REPO_DIR" pull origin main
echo "Copying updated scripts..."
cp "\$REPO_DIR/raspi_scripts/camera/camera_sender.py" "\$SCRIPTS_DIR/camera_sender.py"
cp "\$REPO_DIR/raspi_scripts/pi_agent.py"             "\$SCRIPTS_DIR/pi_agent.py"
chmod +x "\$SCRIPTS_DIR/camera_sender.py" "\$SCRIPTS_DIR/pi_agent.py"
echo "Restarting services..."
sudo systemctl restart roadsentinel-camera roadsentinel-agent
echo "Done! All services restarted with latest code."
HELPER

chmod +x "$SCRIPTS_DIR/start.sh" "$SCRIPTS_DIR/stop.sh" \
         "$SCRIPTS_DIR/status.sh" "$SCRIPTS_DIR/update.sh"

# ── [6] Git remote config ──────────────────────────────────────────────────────
echo "[6/6] Verifying git remote..."
git -C "$REPO_DIR" remote -v
echo "      Run '$SCRIPTS_DIR/update.sh' anytime to pull latest and restart."

echo
echo "================================================"
echo " Pi 4 Setup Complete!"
echo "================================================"
echo
echo " Services (start on every boot):"
echo "   roadsentinel-camera — Camera A → AI → Node API"
echo "   roadsentinel-agent  — Admin Terminal relay"
echo
echo " NOTE: This Pi has NO LED matrix."
echo "       Detections go to Node API → Pi 5 polls them → LED updates."
echo
echo " Quick commands:"
echo "   $SCRIPTS_DIR/start.sh   — start all"
echo "   $SCRIPTS_DIR/stop.sh    — stop all"
echo "   $SCRIPTS_DIR/status.sh  — check status"
echo "   $SCRIPTS_DIR/update.sh  — git pull + restart"
echo
echo " Live logs:"
echo "   tail -f $LOG_DIR/camera.log"
echo "   tail -f $LOG_DIR/agent.log"
echo
echo " Admin Terminal: web dashboard → Admin Terminal → select 'Pi 4'"
echo
echo " SSH (no IP needed):"
echo "   ssh pi@${HOSTNAME}.local"
echo
echo " Starting services now..."
sudo systemctl start roadsentinel-camera roadsentinel-agent
echo " Done!"
echo "================================================"
