#!/usr/bin/env bash
set -euo pipefail

CAM_DIR="${1:-$HOME/camera_scripts}"
AUTOSTART_DIR="$HOME/.config/autostart"
AUTOSTART_FILE="$AUTOSTART_DIR/roadsentinel-cameras.desktop"
USER_NAME="$(id -un)"
USER_HOME="$HOME"

mkdir -p "$AUTOSTART_DIR"

if [ ! -f "$CAM_DIR/live_cam.sh" ] || [ ! -f "$CAM_DIR/camera_env.sh" ]; then
  echo "ERROR: $CAM_DIR/live_cam.sh or $CAM_DIR/camera_env.sh is missing."
  echo "Create ~/camera_scripts first, then run this setup again."
  exit 1
fi

cat > "$CAM_DIR/launch_both_cameras.sh" <<EOS
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="\$HOME/camera_logs"
mkdir -p "\$LOG_DIR"

export DISPLAY="\${DISPLAY:-:0}"
export XAUTHORITY="\${XAUTHORITY:-\$HOME/.Xauthority}"
export LIBVA_DRIVER_NAME="\${LIBVA_DRIVER_NAME:-v4l2_request}"

sleep "\${CAMERA_BOOT_WAIT:-12}"

pkill -f 'ffplay.*192.168.8.104:554/cam/realmonitor' >/dev/null 2>&1 || true
pkill -f 'ffplay.*192.168.8.108:554/cam/realmonitor' >/dev/null 2>&1 || true
sleep 1

if [ -x "\$HOME/venvs/onvif/bin/python" ] && [ -f "\$SCRIPT_DIR/set_ir_auto_all.py" ]; then
  "\$HOME/venvs/onvif/bin/python" "\$SCRIPT_DIR/set_ir_auto_all.py" --wsdl /home/${USER_NAME}/python-onvif-zeep/wsdl/ >>"\$LOG_DIR/ir_auto.log" 2>&1 || true
fi

nohup "\$SCRIPT_DIR/live_cam.sh" 1 sub >>"\$LOG_DIR/camera1.log" 2>&1 &
sleep 1
nohup "\$SCRIPT_DIR/live_cam.sh" 2 sub >>"\$LOG_DIR/camera2.log" 2>&1 &
EOS
chmod +x "$CAM_DIR/launch_both_cameras.sh"

cat > "$AUTOSTART_FILE" <<EOS
[Desktop Entry]
Type=Application
Name=RoadSentinel Cameras
Comment=Start both RoadSentinel low-latency camera windows at login
Exec=${CAM_DIR}/launch_both_cameras.sh
Terminal=false
X-GNOME-Autostart-enabled=true
Hidden=false
EOS

chmod 644 "$AUTOSTART_FILE"

cat > "$CAM_DIR/disable_camera_autostart.sh" <<EOS
#!/usr/bin/env bash
set -euo pipefail
rm -f "$AUTOSTART_FILE"
echo "Removed $AUTOSTART_FILE"
EOS
chmod +x "$CAM_DIR/disable_camera_autostart.sh"

cat > "$CAM_DIR/test_launch_both_now.sh" <<'EOS'
#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMERA_BOOT_WAIT=0 "$SCRIPT_DIR/launch_both_cameras.sh"
EOS
chmod +x "$CAM_DIR/test_launch_both_now.sh"

echo "Autostart installed."
echo "Desktop autostart file: $AUTOSTART_FILE"
echo
echo "Test now without reboot:"
echo "  $CAM_DIR/test_launch_both_now.sh"
echo
echo "Disable later:"
echo "  $CAM_DIR/disable_camera_autostart.sh"
echo
echo "Important: for the camera windows to appear automatically after reboot,"
echo "set Raspberry Pi OS to boot to the desktop with auto-login enabled."
echo "Use raspi-config > System Options > Boot / Auto Login > Desktop Autologin."
