#!/usr/bin/env bash
# Road Sentinel — Raspberry Pi 5 Setup
# Installs: Camera B (CAM-B-002) + HUB75 128×32 LED matrix display
#
# Usage:
#   PI_AGENT_TOKEN=<token> bash setup_pi5.sh [NODE_URL] [CAM_B_RTSP] [AI_URL]
#
# Defaults:
#   NODE_URL      = http://100.120.27.110:3001   (server PC over Tailscale)
#   CAM_B_RTSP    = rtsp://192.168.8.108:554/cam/realmonitor?channel=1&subtype=1
#                   (provisional — Camera B's IP is DHCP-assigned, not static;
#                    camera_sender.py's auto-discovery will persist the real
#                    IP back to Node once it finds it, see camera_sender.py)
#   AI_URL        = http://100.120.27.110:8000   (server PC over Tailscale)
#   PI_AGENT_TOKEN = REQUIRED, no default. Must match server/node-service/.env's
#                    PI_AGENT_TOKEN exactly — the /admin namespace rejects the
#                    Pi agent's connection otherwise.
#
# Server addressing: the Node/AI services run on the `irm-pc` PC,
# reached over Tailscale (100.120.27.110) rather than a LAN IP — the Pis and
# the server PC aren't guaranteed to share a subnet, and Tailscale addresses
# stay stable across network changes where a DHCP LAN IP wouldn't. The camera
# RTSP URL is still a LAN address, since the cameras are on the Pi's own
# local network and aren't Tailscale nodes.
#
# Pi 5 LED note: uses led-image-viewer (coprocessor mode, no --led-rp1-rio).
#   RIO mode (--led-rp1-rio=1) causes rapid GPIO de-sync — do NOT use it.
#
# After setup, SSH via:  ssh pi@pi5-sentinel.local  (no IP needed, ever)

set -euo pipefail

NODE_URL="${1:-http://100.120.27.110:3001}"
CAM_B_RTSP="${2:-rtsp://192.168.8.108:554/cam/realmonitor?channel=1&subtype=1}"
AI_URL="${3:-http://100.120.27.110:8000}"
CAMERA_ID="CAM-B-002"
HOSTNAME="pi5-sentinel"

if [ -z "${PI_AGENT_TOKEN:-}" ]; then
    echo "ERROR: PI_AGENT_TOKEN is not set."
    echo "  Copy the PI_AGENT_TOKEN value from server/node-service/.env, then run:"
    echo "  PI_AGENT_TOKEN=<that-value> bash setup_pi5.sh"
    exit 1
fi

VENV="$HOME/venvs/cam_venv"
SCRIPTS_DIR="$HOME/roadsentinel"
LOG_DIR="$HOME/roadsentinel/logs"
REPO_DIR="$HOME/roadsentinel-repo"
REPO_URL="https://github.com/Shaloh69/Road_Sentinel.git"
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


# Absolute path to the LED binary, passed explicitly to the display service.
# The service runs as root (User=root) but the library is built under this
# /root and find nothing, which silently crash-looped this service before.
# Passing it explicitly removes the dependency on HOME resolution entirely.
VIEWER_BIN="$HOME/rpi-rgb-led-matrix/utils/led-image-viewer"

echo "================================================"
echo " Road Sentinel — Pi 5 Setup (Camera B + LED)"
echo "================================================"
echo " Node service : $NODE_URL"
echo " AI service   : $AI_URL"
echo " Camera B     : $CAM_B_RTSP"
echo " Camera ID    : $CAMERA_ID"
echo " Hostname     : $HOSTNAME"
echo "================================================"
echo

# ── [0] Set hostname ───────────────────────────────────────────────────────────
echo "[0/7] Setting hostname to '$HOSTNAME'..."
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

# ── [2] LED sign — nothing to build ───────────────────────────────────────────
# The panel is no longer driven from this Pi's GPIO, so hzeller's
# rpi-rgb-led-matrix is not built or installed any more. That whole approach
# was abandoned after ~80 configurations failed to render legible text on
# these 1/8-scan FM6124 panels; see LEDMatrixDrivers/esp32/DEBUG_LOG.md.
#
# The sign is driven by a microcontroller over USB serial instead, which also
# removes the /dev/mem root requirement and the SPI/audio GPIO conflicts.
echo "[2/7] LED sign: driven over USB serial, nothing to build."
echo


# ── [3] Python venv ────────────────────────────────────────────────────────────
echo "[3/7] Creating Python venv..."
mkdir -p "$(dirname "$VENV")"
python3 -m venv "$VENV" --system-site-packages
source "$VENV/bin/activate"
pip install --upgrade pip -q
pip install aiohttp requests "pillow>=10.0" numpy "python-socketio[client]" adafruit-blinka-raspberry-pi5-piomatter
python3 -c "import cv2, aiohttp, requests, PIL, socketio; print('  deps: OK')"
deactivate
echo "      Venv OK: $VENV"
echo

# ── [4] Copy scripts ───────────────────────────────────────────────────────────
echo "[4/7] Installing scripts..."
mkdir -p "$SCRIPTS_DIR" "$LOG_DIR"
cp "$SRC_DIR/camera/camera_sender.py"    "$SCRIPTS_DIR/camera_sender.py"
cp "$SRC_DIR/pi_agent.py"               "$SCRIPTS_DIR/pi_agent.py"
chmod +x "$SCRIPTS_DIR/camera_sender.py"
chmod +x "$SCRIPTS_DIR/led_sign_bridge.py"
chmod +x "$SCRIPTS_DIR/pi_agent.py"
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

# LED sign service — the panel is driven by a STM32 Black Pill over USB serial,
# not by this Pi's GPIO. The Pi keeps everything needing a network or a
# decision (polling Node, deciding road state, reconnecting); the board only
# draws. If the sign shows the WRONG THING the bug is here; if it shows it
# WRONGLY the bug is in the firmware.
#
# The Black Pill enumerates as native USB CDC, so it appears on /dev/ttyACM*.
#
# No sudo: unlike the old GPIO driver this needs no /dev/mem access, only
# membership of the dialout group (added above).
sudo tee /etc/systemd/system/roadsentinel-display.service > /dev/null <<EOF
[Unit]
Description=Road Sentinel LED Sign Bridge (STM32 Black Pill)
After=network-online.target roadsentinel-camera.service
Wants=network-online.target
StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
Type=simple
User=${USER}
WorkingDirectory=${SCRIPTS_DIR}
ExecStart=${VENV}/bin/python3 ${SCRIPTS_DIR}/led_sign_bridge.py --api ${NODE_URL}
Restart=always
RestartSec=10
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
Environment=PI_AGENT_TOKEN=${PI_AGENT_TOKEN}
ExecStart=${VENV}/bin/python3 ${SCRIPTS_DIR}/pi_agent.py \\
    --node ${NODE_URL} \\
    --id   pi5
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
${VENV}/bin/python3 ${SCRIPTS_DIR}/led_sign_bridge.py --test
HELPER

cat > "$SCRIPTS_DIR/update.sh" <<'HELPER'
#!/usr/bin/env bash
# Pull latest raspi_scripts from GitHub and restart services.
# Self-contained — defines its own paths so it works from any shell.
set -euo pipefail
REPO_DIR="$HOME/roadsentinel-repo"
SCRIPTS_DIR="$HOME/roadsentinel"
echo "Pulling latest from GitHub..."
git -C "$REPO_DIR" pull origin main
echo "Copying updated scripts..."
cp "$REPO_DIR/raspi_scripts/camera/camera_sender.py" "$SCRIPTS_DIR/camera_sender.py"
cp "$REPO_DIR/raspi_scripts/led_sign_bridge.py" "$SCRIPTS_DIR/led_sign_bridge.py"
cp "$REPO_DIR/raspi_scripts/pi_agent.py"             "$SCRIPTS_DIR/pi_agent.py"
cp "$REPO_DIR/raspi_scripts/color_test.py"           "$SCRIPTS_DIR/color_test.py"
chmod +x "$SCRIPTS_DIR/camera_sender.py" "$SCRIPTS_DIR/led_sign_bridge.py" \
         "$SCRIPTS_DIR/pi_agent.py" "$SCRIPTS_DIR/color_test.py"
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
echo " Pi 5 Setup Complete!"
echo "================================================"
echo
echo " Services (start on every boot):"
echo "   roadsentinel-camera  — Camera B → AI → Node"
echo "   roadsentinel-display — LED sign bridge (STM32 Black Pill over USB serial)"
echo "   roadsentinel-agent   — Admin Terminal relay (connects to $NODE_URL)"
echo
echo " Quick commands:"
echo "   $SCRIPTS_DIR/start.sh        — start all"
echo "   $SCRIPTS_DIR/stop.sh         — stop all"
echo "   $SCRIPTS_DIR/status.sh       — check status"
echo "   $SCRIPTS_DIR/update.sh       — git pull + restart (or use Admin Terminal)"
echo "   $SCRIPTS_DIR/test_display.sh — test LED with fake alerts"
echo
echo " If LED display shows garbage, check display.log and ensure"
echo "   led-image-viewer was built: ls ~/rpi-rgb-led-matrix/utils/led-image-viewer"
echo
echo " Live logs:"
echo "   tail -f $LOG_DIR/camera.log"
echo "   tail -f $LOG_DIR/display.log"
echo
echo " Live logs:"
echo "   tail -f $LOG_DIR/agent.log"
echo
echo " Admin Terminal: open the web dashboard → Admin Terminal → select 'Pi 5'"
echo " (the agent must be running and the Pi must reach $NODE_URL)"
echo
echo " SSH (no IP needed — works even after router restarts):"
echo "   ssh pi@${HOSTNAME}.local"
echo
echo " Starting services now..."
sudo systemctl start roadsentinel-camera roadsentinel-display roadsentinel-agent
echo " Done!"
echo "================================================"
