#!/usr/bin/env bash
# Road Sentinel — LED Matrix install for Raspberry Pi 4 Model B.
#
# Builds the hzeller C library and ledcat binary.
# display_manager.py pipes raw RGB24 frames to ledcat stdin — no Python bindings needed.
#
# Run once: bash install.sh
# After this: sudo python3 display_manager.py --test

set -euo pipefail

VENV="${1:-$HOME/venvs/led_venv}"
BUILD_DIR="$HOME/rpi-rgb-led-matrix"
LEDCAT="$BUILD_DIR/examples-api-use/ledcat"

echo "=== Road Sentinel — LED Matrix install for Raspberry Pi 4 ==="
echo "Build dir: $BUILD_DIR"
echo "Venv:      $VENV"
echo

# ── Step 1: system deps ───────────────────────────────────────────────────────
echo "[1/4] Installing system packages..."
sudo apt update -q
sudo apt install -y git python3-dev python3-pip python3-venv python3-pil \
                    build-essential

# ── Step 2: clone/update hzeller library ─────────────────────────────────────
echo "[2/4] Cloning hzeller/rpi-rgb-led-matrix..."
if [ -d "$BUILD_DIR" ]; then
  echo "  Already exists — pulling latest..."
  git -C "$BUILD_DIR" pull --ff-only
else
  git clone https://github.com/hzeller/rpi-rgb-led-matrix.git "$BUILD_DIR"
fi

# ── Step 3: build C library and ledcat binary ────────────────────────────────
echo "[3/4] Building C library and ledcat..."
make -C "$BUILD_DIR/lib"
make -C "$BUILD_DIR/examples-api-use"

if [ ! -f "$LEDCAT" ]; then
  echo "ERROR: ledcat not found at $LEDCAT after build"
  exit 1
fi
echo "  ledcat OK: $LEDCAT"

# ── Step 4: create Python venv with Pillow ────────────────────────────────────
echo "[4/4] Creating Python venv at $VENV..."
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip -q
pip install Pillow requests

echo
echo "=== Done! ==="
echo
echo "Verify install:"
echo "  ls $LEDCAT"
echo "  source $VENV/bin/activate && python3 -c \"from PIL import Image; print('Pillow OK')\""
echo
echo "Run test mode:"
echo "  source $VENV/bin/activate"
echo "  cd ~/raspi_scripts/lcd_pi4"
echo "  sudo \$VIRTUAL_ENV/bin/python3 display_manager.py --test"
echo
echo "IMPORTANT: sudo is required for direct GPIO /dev/mem access."
echo "  Use the full venv path: sudo \$VIRTUAL_ENV/bin/python3"
