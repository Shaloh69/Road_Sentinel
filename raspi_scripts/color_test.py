#!/usr/bin/env python3
"""
Quick HUB75 color + pinout test for Pi 5 (PioMatter backend).
Shows solid RED → GREEN → BLUE → WHITE → BLACK then repeats.

Active3 GPIO wiring (BCM numbers → HUB75 connector):
  R1  = GPIO 11  (Pi physical pin 23)   → HUB75 pin 1
  G1  = GPIO 27  (Pi physical pin 13)   → HUB75 pin 2
  B1  = GPIO  7  (Pi physical pin 26)   → HUB75 pin 3
  R2  = GPIO  8  (Pi physical pin 24)   → HUB75 pin 5
  G2  = GPIO  9  (Pi physical pin 21)   → HUB75 pin 6
  B2  = GPIO 10  (Pi physical pin 19)   → HUB75 pin 7
  A   = GPIO 22  (Pi physical pin 15)   → HUB75 pin 9
  B   = GPIO 23  (Pi physical pin 16)   → HUB75 pin 10
  C   = GPIO 24  (Pi physical pin 18)   → HUB75 pin 11
  D   = GPIO 25  (Pi physical pin 22)   → HUB75 pin 12
  CLK = GPIO 17  (Pi physical pin 11)   → HUB75 pin 13
  LAT = GPIO  4  (Pi physical pin  7)   → HUB75 pin 14
  OE  = GPIO 18  (Pi physical pin 12)   → HUB75 pin 15
  GND = any GND  (Pi physical pin 6/9…) → HUB75 pin 4/8/16

Active3BGR = same wiring above BUT piomatter swaps R↔B in software.
Use BGR only if Red appears Blue with plain Active3.

Usage:
    sudo ~/venvs/cam_venv/bin/python3 ~/roadsentinel/color_test.py
    sudo ~/venvs/cam_venv/bin/python3 ~/roadsentinel/color_test.py --pinout active3
    sudo ~/venvs/cam_venv/bin/python3 ~/roadsentinel/color_test.py --addr-lines 3 --slowdown 4
"""

import argparse
import itertools
import signal
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import adafruit_blinka_raspberry_pi5_piomatter as piomatter
except ImportError:
    raise SystemExit("PioMatter not installed. Run:\n  pip install adafruit-blinka-raspberry-pi5-piomatter")

W, H = 128, 32

COLORS = [
    ("RED",   (255,   0,   0)),
    ("GREEN", (  0, 255,   0)),
    ("BLUE",  (  0,   0, 255)),
    ("WHITE", (255, 255, 255)),
    ("BLACK", (  0,   0,   0)),
]


def build_frame(name: str, rgb: tuple) -> np.ndarray:
    img = Image.new("RGB", (W, H), rgb)
    label_color = (0, 0, 0) if rgb != (0, 0, 0) else (80, 80, 80)
    try:
        font = ImageFont.load_default(size=10)
    except TypeError:
        font = ImageFont.load_default()
    ImageDraw.Draw(img).text((2, 10), name, fill=label_color, font=font)
    return np.asarray(img.convert("RGB"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--addr-lines", type=int, default=4)
    p.add_argument("--slowdown",   type=int, default=3)
    p.add_argument("--delay",      type=float, default=3.0, help="Seconds per color")
    p.add_argument("--pinout",     default="active3bgr",
                   choices=["active3", "active3bgr"],
                   help="active3bgr swaps R↔B in software (use if Red appears Blue)")
    args = p.parse_args()

    # Resolve pinout
    if args.pinout == "active3bgr":
        pinout = getattr(piomatter.Pinout, "Active3BGR", piomatter.Pinout.Active3)
    else:
        pinout = piomatter.Pinout.Active3

    geo = piomatter.Geometry(
        width=W, height=H,
        n_addr_lines=args.addr_lines,
        rotation=piomatter.Orientation.Normal,
    )
    fb = np.zeros((H, W, 3), dtype=np.uint8)
    matrix = piomatter.PioMatter(
        colorspace=piomatter.Colorspace.RGB888Packed,
        pinout=pinout,
        framebuffer=fb,
        geometry=geo,
    )

    # Pre-build all frames so color switching is instant (no gap in show() calls)
    frames = [(name, rgb, build_frame(name, rgb)) for name, rgb in COLORS]

    print(f"Color test — {W}x{H}  addr_lines={args.addr_lines}  "
          f"pinout={args.pinout}  slowdown={args.slowdown}  (Ctrl-C to stop)")
    print("GPIO → HUB75:  R1=GPIO11  G1=GPIO27  B1=GPIO7  CLK=GPIO17  LAT=GPIO4  OE=GPIO18")

    color_iter = itertools.cycle(frames)
    name, rgb, arr = next(color_iter)
    deadline = time.monotonic() + args.delay
    print(f"  → {name}  {rgb}")

    try:
        while True:
            now = time.monotonic()
            if now >= deadline:
                name, rgb, arr = next(color_iter)
                deadline = now + args.delay
                print(f"  → {name}  {rgb}")
            # Always keep the panel refreshed — never let show() gap between colors
            fb[:] = arr
            matrix.show()

    except KeyboardInterrupt:
        # Block second Ctrl-C so cleanup isn't interrupted mid-flush
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        fb[:] = 0
        for _ in range(16):
            matrix.show()
            time.sleep(0.02)
        print("\nDone.")


if __name__ == "__main__":
    main()
