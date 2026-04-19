#!/usr/bin/env bash
# Build and install hzeller/rpi-rgb-led-matrix Python bindings for Pi 4 Model B.
# Run once: bash install.sh
# After this, display_manager.py works with: sudo python3 display_manager.py

set -euo pipefail

VENV="${1:-$HOME/venvs/led_venv}"
BUILD_DIR="$HOME/rpi-rgb-led-matrix"

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

# ── Step 3: create venv ───────────────────────────────────────────────────────
echo "[3/4] Creating venv at $VENV..."
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip -q
pip install Pillow requests

# ── Step 4: build and install rgbmatrix Python bindings ──────────────────────
echo "[4/4] Building Python bindings (this takes ~2 minutes)..."
# The hzeller Makefile generates setup.py before building — run make first,
# then install the built package into the venv with pip.
cd "$BUILD_DIR/bindings/python"
make
pip install .

echo
echo "=== Done! ==="
echo
echo "Verify install:"
echo "  source $VENV/bin/activate"
echo "  sudo python3 -c \"from rgbmatrix import RGBMatrix; print('rgbmatrix OK')\""
echo
echo "Run test mode:"
echo "  source $VENV/bin/activate"
echo "  cd ~/raspi_scripts/lcd_pi4"
echo "  sudo \$VIRTUAL_ENV/bin/python3 display_manager.py --test"
echo
echo "IMPORTANT: sudo is required for direct GPIO /dev/mem access."
echo "  You must call the venv python via \$VIRTUAL_ENV/bin/python3 when using sudo."
