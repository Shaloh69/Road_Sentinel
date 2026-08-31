#!/usr/bin/env bash
# Road Sentinel — install the WiFi provisioning portal as a systemd service.
#
# Run on each Pi:
#   bash setup_wifi_portal.sh
#
# After this the Pi raises a "RoadSentinel-Setup" access point whenever its
# WiFi fails 5 checks in a row, so a phone can re-provision it with no monitor
# or keyboard. See WIFI_PORTAL.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="/opt/roadsentinel"
SERVICE=/etc/systemd/system/roadsentinel-wifi-portal.service

echo "================================================"
echo " Road Sentinel — WiFi Portal Setup"
echo "================================================"

# ── Preconditions ─────────────────────────────────────────────────────────────
if ! command -v nmcli >/dev/null 2>&1; then
    echo "ERROR: nmcli not found. This needs NetworkManager (standard on"
    echo "Raspberry Pi OS Bookworm). On older releases:"
    echo "    sudo apt install network-manager"
    exit 1
fi

if ! systemctl is-active --quiet NetworkManager; then
    echo "ERROR: NetworkManager is installed but not running."
    echo "    sudo systemctl enable --now NetworkManager"
    exit 1
fi
echo "[1/4] NetworkManager OK"

# ── Install the script ────────────────────────────────────────────────────────
sudo mkdir -p "$INSTALL_DIR"
sudo cp "$SCRIPT_DIR/wifi_portal.py" "$INSTALL_DIR/wifi_portal.py"
sudo chmod +x "$INSTALL_DIR/wifi_portal.py"
echo "[2/4] Installed to $INSTALL_DIR/wifi_portal.py"

# ── systemd unit ──────────────────────────────────────────────────────────────
# Runs as root: nmcli needs it to create the AP, and the LED sysfs nodes are
# root-writable. Restart=always so a crash cannot leave the Pi unreachable —
# this service is the recovery path, so it must outlive its own bugs.
sudo tee "$SERVICE" > /dev/null <<EOF
[Unit]
Description=Road Sentinel WiFi Provisioning Portal
After=NetworkManager.service
Wants=NetworkManager.service

[Service]
Type=simple
ExecStart=/usr/bin/python3 $INSTALL_DIR/wifi_portal.py
Restart=always
RestartSec=15
User=root

# Tunables — change here, then: sudo systemctl daemon-reload && sudo systemctl restart roadsentinel-wifi-portal
Environment=PORTAL_FAIL_THRESHOLD=5
Environment=PORTAL_CHECK_INTERVAL=20
Environment=PORTAL_TIMEOUT=900
Environment=PORTAL_AP_SSID=RoadSentinel-Setup
Environment=PORTAL_AP_PASSWORD=roadsentinel

[Install]
WantedBy=multi-user.target
EOF
echo "[3/4] Wrote $SERVICE"

sudo systemctl daemon-reload
sudo systemctl enable --now roadsentinel-wifi-portal
sleep 2
echo "[4/4] Service enabled"
echo

systemctl status roadsentinel-wifi-portal --no-pager -l 2>&1 | head -12 || true

cat <<'EOS'

================================================
 Done.
================================================

If WiFi fails 5 checks in a row, the Pi raises:

    SSID      RoadSentinel-Setup
    Password  roadsentinel
    Portal    http://10.42.0.1   (should open by itself)

Onboard LED:
    brief blink every 5s ... connected
    even 1 Hz blink ........ connecting / retrying
    fast 4 Hz blink ........ portal is up, waiting for you
    double-blink ........... failed

Useful commands:
    sudo systemctl status roadsentinel-wifi-portal
    sudo journalctl -u roadsentinel-wifi-portal -f
    sudo python3 /opt/roadsentinel/wifi_portal.py --status
    sudo python3 /opt/roadsentinel/wifi_portal.py --portal-now   # test it now

Worth doing while you still have a connection — add your phone hotspot as a
saved fallback network, so you have a second way back in:

    sudo nmcli device wifi connect "YourHotspot" password "yourpassword"

EOS
