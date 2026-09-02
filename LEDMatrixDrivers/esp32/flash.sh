#!/usr/bin/env bash
# Road Sentinel — build & flash the ESP32 display firmware from the Pi.
#
# The Pi is the natural place to do this: the ESP32 is plugged into it anyway
# for the serial link, so there is no cable-swapping between a laptop and the
# installation site.
set -euo pipefail
PIO=~/.pio-venv/bin/pio
PROJ=~/esp32_display

if [ ! -x "$PIO" ]; then
    echo "PlatformIO not installed. Run:"
    echo "  python3 -m venv ~/.pio-venv && ~/.pio-venv/bin/pip install platformio"
    exit 1
fi

PORT="$(ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null | head -1 || true)"
if [ -z "$PORT" ]; then
    echo "No ESP32 detected on /dev/ttyUSB* or /dev/ttyACM*."
    echo "Plug it into the Pi and re-run."
    exit 1
fi
echo "Found board on $PORT"

# The bridge holds the same port; two processes cannot share it and the
# failure mode is a silent timeout rather than a clear error.
if systemctl is-active --quiet roadsentinel-esp32-bridge 2>/dev/null; then
    echo "Stopping bridge service while flashing..."
    sudo systemctl stop roadsentinel-esp32-bridge
    RESTART=1
fi

cd "$PROJ"
"$PIO" run -t upload --upload-port "$PORT"

if [ "${RESTART:-0}" = "1" ]; then
    echo "Restarting bridge service..."
    sudo systemctl start roadsentinel-esp32-bridge
fi
echo "Done. Check with: ~/.pio-venv/bin/pio device monitor -b 115200"
