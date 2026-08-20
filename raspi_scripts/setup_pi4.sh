#!/usr/bin/env bash
# Road Sentinel — Raspberry Pi 4 Setup
# Installs: Camera A (CAM-A-001) + HUB75 128x32 LED matrix display.
# Phase 2: Pi 4 now gets the same LED matrix as Pi 5 (symmetric hardware) —
# same unified display_manager.py driver, auto-detected Pi 4 backend (ledcat).
#
# Usage:
#   PI_AGENT_TOKEN=<token> bash setup_pi4.sh [NODE_URL] [CAM_A_RTSP] [AI_URL]
#
# Defaults:
#   NODE_URL       = http://100.120.27.110:3001   (server PC over Tailscale)
#   CAM_A_RTSP     = rtsp://192.168.8.104:554/cam/realmonitor?channel=1&subtype=1
#   AI_URL         = http://100.120.27.110:8000   (server PC over Tailscale)
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
# Pi 4 LED note: uses ledcat (direct /dev/mem GPIO, needs sudo). If it shows
# intermittent garbled output, see fix_gpio_timing.sh and
# raspi_scripts/README.md's "LED Matrix Status Display" section (Phase 0 fix
# raised --led-slowdown-gpio 4->6 as a starting point — this is code-level,
# not yet hardware-verified as of this script).
#
# After setup, SSH via:  ssh pi@pi4-sentinel.local  (no IP needed, ever)

set -euo pipefail

NODE_URL="${1:-http://100.120.27.110:3001}"
CAM_A_RTSP="${2:-rtsp://192.168.8.104:554/cam/realmonitor?channel=1&subtype=1}"
AI_URL="${3:-http://100.120.27.110:8000}"
CAMERA_ID="CAM-A-001"
HOSTNAME="pi4-sentinel"

if [ -z "${PI_AGENT_TOKEN:-}" ]; then
    echo "ERROR: PI_AGENT_TOKEN is not set."
    echo "  Copy the PI_AGENT_TOKEN value from server/node-service/.env, then run:"
    echo "  PI_AGENT_TOKEN=<that-value> bash setup_pi4.sh"
    exit 1
fi

# Absolute path to the LED binary, passed explicitly to the display service.
# The service runs as root (User=root) but the library is built under this
# login user's home — a bare "~" inside display_manager.py would expand to
# /root and find nothing, which silently crash-looped the Pi 5 service before.
# Passing it explicitly removes the dependency on HOME resolution entirely.
LEDCAT_BIN="$HOME/rpi-rgb-led-matrix/examples-api-use/ledcat"

VENV="$HOME/venvs/cam_venv"
SCRIPTS_DIR="$HOME/roadsentinel"
LOG_DIR="$HOME/roadsentinel/logs"
REPO_DIR="$HOME/roadsentinel-repo"
REPO_URL="https://github.com/Shaloh69/Road_Sentinel.git"
SRC_DIR="$REPO_DIR/raspi_scripts"

echo "================================================"
echo " Road Sentinel — Pi 4 Setup (Camera A + LED)"
echo "================================================"
echo " Node service : $NODE_URL"
echo " AI service   : $AI_URL"
echo " Camera A     : $CAM_A_RTSP"
echo " Camera ID    : $CAMERA_ID"
echo " Hostname     : $HOSTNAME"
echo " LED backend  : ledcat (direct GPIO, sudo required)"
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

# ── [1b] Clone / update repo ───────────────────────────────────────────────────
echo "[1b/7] Syncing RoadSentinel repo..."
if [ -d "$REPO_DIR/.git" ]; then
    git -C "$REPO_DIR" pull origin main
else
    git clone "$REPO_URL" "$REPO_DIR"
fi
echo "      Repo at $REPO_DIR"
echo

# ── [2] Build hzeller ledcat (Pi 4 LED backend) ───────────────────────────────
echo "[2/7] Building hzeller rpi-rgb-led-matrix (ledcat)..."
if [ ! -d "$HOME/rpi-rgb-led-matrix" ]; then
    git clone https://github.com/hzeller/rpi-rgb-led-matrix.git "$HOME/rpi-rgb-led-matrix"
else
    git -C "$HOME/rpi-rgb-led-matrix" pull
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
# onvif-zeep is optional (only needed for --ir-auto) — best-effort install,
# don't fail the whole setup if it's unavailable for this Python version.
pip install aiohttp requests "pillow>=10.0" numpy "python-socketio[client]"
pip install onvif-zeep || echo "      (onvif-zeep install failed — --ir-auto will be unavailable, everything else still works)"
python3 -c "import cv2, aiohttp, requests, PIL, socketio; print('  deps: OK')"
deactivate
echo "      Venv OK: $VENV"
echo

# ── [4] Copy scripts ───────────────────────────────────────────────────────────
echo "[4/7] Installing scripts..."
mkdir -p "$SCRIPTS_DIR" "$LOG_DIR"
cp "$SRC_DIR/camera/camera_sender.py" "$SCRIPTS_DIR/camera_sender.py"
cp "$SRC_DIR/display_manager.py"      "$SCRIPTS_DIR/display_manager.py"
cp "$SRC_DIR/pi_agent.py"             "$SCRIPTS_DIR/pi_agent.py"
chmod +x "$SCRIPTS_DIR/camera_sender.py" "$SCRIPTS_DIR/display_manager.py" "$SCRIPTS_DIR/pi_agent.py"
echo "      Scripts installed to $SCRIPTS_DIR/"
echo

# ── [5] Systemd services ───────────────────────────────────────────────────────
echo "[5/7] Installing systemd services..."

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

# LED display service — Pi 4 uses --pi 4 flag (ledcat backend, needs sudo)
sudo tee /etc/systemd/system/roadsentinel-display.service > /dev/null <<EOF
[Unit]
Description=Road Sentinel LED Matrix Display (Pi 4)
After=network-online.target roadsentinel-camera.service
Wants=network-online.target
StartLimitIntervalSec=60
StartLimitBurst=5

[Service]
Type=simple
User=root
WorkingDirectory=${SCRIPTS_DIR}
ExecStart=${VENV}/bin/python3 ${SCRIPTS_DIR}/display_manager.py \
    --api ${NODE_URL} \
    --pi 4 \
    --ledcat ${LEDCAT_BIN}
Restart=always
RestartSec=5
StandardOutput=append:${LOG_DIR}/display.log
StandardError=append:${LOG_DIR}/display.log

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
Environment=PI_AGENT_TOKEN=${PI_AGENT_TOKEN}
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
sudo ${VENV}/bin/python3 ${SCRIPTS_DIR}/display_manager.py --test --pi 4
HELPER

cat > "$SCRIPTS_DIR/update.sh" <<'HELPER'
#!/usr/bin/env bash
# Pull latest raspi_scripts from GitHub and restart services.
set -euo pipefail
REPO_DIR="$HOME/roadsentinel-repo"
SCRIPTS_DIR="$HOME/roadsentinel"
echo "Pulling latest from GitHub..."
git -C "$REPO_DIR" pull origin main
echo "Copying updated scripts..."
cp "$REPO_DIR/raspi_scripts/camera/camera_sender.py" "$SCRIPTS_DIR/camera_sender.py"
cp "$REPO_DIR/raspi_scripts/display_manager.py"      "$SCRIPTS_DIR/display_manager.py"
cp "$REPO_DIR/raspi_scripts/pi_agent.py"             "$SCRIPTS_DIR/pi_agent.py"
cp "$REPO_DIR/raspi_scripts/color_test.py"           "$SCRIPTS_DIR/color_test.py"
chmod +x "$SCRIPTS_DIR/camera_sender.py" "$SCRIPTS_DIR/display_manager.py" \
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
echo " Pi 4 Setup Complete!"
echo "================================================"
echo
echo " Services (start on every boot):"
echo "   roadsentinel-camera  — Camera A → AI → Node"
echo "   roadsentinel-display — HUB75 128×32 LED matrix (ledcat)"
echo "   roadsentinel-agent   — Admin Terminal relay (connects to $NODE_URL)"
echo
echo " Quick commands:"
echo "   $SCRIPTS_DIR/start.sh        — start all"
echo "   $SCRIPTS_DIR/stop.sh         — stop all"
echo "   $SCRIPTS_DIR/status.sh       — check status"
echo "   $SCRIPTS_DIR/update.sh       — git pull + restart (or use Admin Terminal)"
echo "   $SCRIPTS_DIR/test_display.sh — test LED with fake alerts"
echo
echo " If LED display shows garbled/intermittent output, this is a known,"
echo "   not-yet-hardware-verified Phase 0 issue — see"
echo "   raspi_scripts/fix_gpio_timing.sh and raspi_scripts/README.md."
echo
echo " Live logs:"
echo "   tail -f $LOG_DIR/camera.log"
echo "   tail -f $LOG_DIR/display.log"
echo "   tail -f $LOG_DIR/agent.log"
echo
echo " Admin Terminal: web dashboard → Admin Terminal → select 'Pi 4'"
echo " (the agent must be running and the Pi must reach $NODE_URL)"
echo
echo " SSH (no IP needed — works even after router restarts):"
echo "   ssh pi@${HOSTNAME}.local"
echo
echo " Starting services now..."
sudo systemctl start roadsentinel-camera roadsentinel-display roadsentinel-agent
echo " Done!"
echo "================================================"
