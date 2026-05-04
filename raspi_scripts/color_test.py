#!/usr/bin/env python3
"""
Quick HUB75 color + pinout test for Pi 5 (PioMatter backend).
Shows solid RED → GREEN → BLUE → WHITE → BLACK 3 seconds each.
If colors look wrong (e.g. red appears blue), you have an RGB swap — change colorspace.

Usage:
    sudo ~/venvs/cam_venv/bin/python3 ~/roadsentinel/color_test.py
    sudo ~/venvs/cam_venv/bin/python3 ~/roadsentinel/color_test.py --addr-lines 3
    sudo ~/venvs/cam_venv/bin/python3 ~/roadsentinel/color_test.py --addr-lines 3 --slowdown 4
"""

import argparse
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import adafruit_blinka_raspberry_pi5_piomatter as piomatter
except ImportError:
    raise SystemExit("PioMatter not installed. Run:\n  pip install adafruit-blinka-raspberry-pi5-piomatter")

W, H = 128, 32

COLORS = [
    ("RED",     (255,   0,   0)),
    ("GREEN",   (  0, 255,   0)),
    ("BLUE",    (  0,   0, 255)),
    ("WHITE",   (255, 255, 255)),
    ("BLACK",   (  0,   0,   0)),
]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--addr-lines", type=int, default=4)
    p.add_argument("--slowdown",   type=int, default=3)
    p.add_argument("--delay",      type=float, default=3.0, help="Seconds per color")
    args = p.parse_args()

    geo = piomatter.Geometry(
        width=W, height=H,
        n_addr_lines=args.addr_lines,
        rotation=piomatter.Orientation.Normal,
    )
    fb = np.zeros((H, W, 3), dtype=np.uint8)
    matrix = piomatter.PioMatter(
        colorspace=piomatter.Colorspace.BGR888Packed,
        pinout=piomatter.Pinout.Active3,
        framebuffer=fb,
        geometry=geo,
    )

    print(f"Color test — {W}x{H}  addr_lines={args.addr_lines}  (Ctrl-C to stop)")
    try:
        while True:
            for name, rgb in COLORS:
                print(f"  → {name}  {rgb}")
                img = Image.new("RGB", (W, H), rgb)
                # Print label in contrasting color
                label_color = (0, 0, 0) if rgb != (0, 0, 0) else (80, 80, 80)
                try:
                    font = ImageFont.load_default(size=10)
                except TypeError:
                    font = ImageFont.load_default()
                ImageDraw.Draw(img).text((2, 10), name, fill=label_color, font=font)
                fb[:] = np.asarray(img.convert("RGB"))
                matrix.show()
                time.sleep(args.delay)
    except KeyboardInterrupt:
        # Blank the panel — send black a few times so PIO flushes fully
        fb[:] = 0
        for _ in range(8):
            matrix.show()
            time.sleep(0.05)
        print("Done.")

if __name__ == "__main__":
    main()
