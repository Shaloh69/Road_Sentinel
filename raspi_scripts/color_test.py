#!/usr/bin/env python3
"""
Quick HUB75 color + pinout test for Pi 5 (PioMatter backend).
Shows solid RED → GREEN → BLUE → WHITE → BLACK then repeats.

WIRING (Active3, single panel, BCM GPIO → HUB75 pin):
  R1=GPIO11(pin23)→HUB75-1   G1=GPIO27(pin13)→HUB75-2   B1=GPIO7(pin26)→HUB75-3
  R2=GPIO8(pin24)→HUB75-5    G2=GPIO9(pin21)→HUB75-6    B2=GPIO10(pin19)→HUB75-7
  A=GPIO22(pin15)→HUB75-9    B=GPIO23(pin16)→HUB75-10   C=GPIO24(pin18)→HUB75-11
  D=GPIO25(pin22)→HUB75-12   CLK=GPIO17(pin11)→HUB75-13 LAT=GPIO4(pin7)→HUB75-14
  OE=GPIO18(pin12)→HUB75-15  GND(pin6/9/14)→HUB75-4/8/16 + LED PSU negative

CRITICAL: Pi GND must be connected to the LED panel power supply GND.
Without a common ground, all signals are undefined → flickering/noise.

Usage:
    sudo ~/venvs/cam_venv/bin/python3 ~/roadsentinel/color_test.py
    sudo ~/venvs/cam_venv/bin/python3 ~/roadsentinel/color_test.py --pinout active3
    sudo ~/venvs/cam_venv/bin/python3 ~/roadsentinel/color_test.py --addr-lines 3
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
    raise SystemExit("PioMatter not installed.\n  pip install adafruit-blinka-raspberry-pi5-piomatter")

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
    return np.asarray(img.convert("RGB")) + 0  # +0 = mutable copy (Adafruit pattern)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--addr-lines", type=int, default=4,
                   help="4 for 32px tall (1/16 scan), 3 for 16px tall (1/8 scan)")
    p.add_argument("--delay",  type=float, default=3.0, help="Seconds per color")
    p.add_argument("--fps",    type=int,   default=30,  help="Refresh rate for show() calls")
    p.add_argument("--pinout", default="active3bgr",
                   choices=["active3", "active3bgr"],
                   help="active3bgr swaps R↔B in software; use active3 for standard wiring")
    args = p.parse_args()

    if args.pinout == "active3bgr":
        pinout = getattr(piomatter.Pinout, "Active3BGR", piomatter.Pinout.Active3)
    else:
        pinout = piomatter.Pinout.Active3

    geo = piomatter.Geometry(
        width=W, height=H,
        n_addr_lines=args.addr_lines,
        rotation=piomatter.Orientation.Normal,
    )
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    fb = np.asarray(canvas) + 0  # mutable writable framebuffer (Adafruit pattern)

    matrix = piomatter.PioMatter(
        colorspace=piomatter.Colorspace.RGB888Packed,
        pinout=pinout,
        framebuffer=fb,
        geometry=geo,
    )

    frames = [(name, rgb, build_frame(name, rgb)) for name, rgb in COLORS]
    frame_interval = 1.0 / args.fps  # sleep between show() calls

    print(f"Color test — {W}x{H}  addr_lines={args.addr_lines}  "
          f"pinout={args.pinout}  refresh={args.fps}fps  (Ctrl-C to stop)")
    print("Checklist before running:")
    print("  [ ] Pi GND wire connected to LED panel power supply GND")
    print("  [ ] OE = GPIO18 (Pi pin 12) → HUB75 pin 15")
    print("  [ ] Panel powered by separate 5V supply (not from Pi)")

    color_iter = itertools.cycle(frames)
    name, rgb, arr = next(color_iter)
    deadline = time.monotonic() + args.delay
    print(f"\n  → {name}  {rgb}")

    try:
        while True:
            now = time.monotonic()
            if now >= deadline:
                name, rgb, arr = next(color_iter)
                deadline = now + args.delay
                print(f"  → {name}  {rgb}")

            fb[:] = arr
            matrix.show()
            # Sleep between show() calls — piomatter PIO refreshes the panel
            # independently between calls. Hammering show() restarts the PIO
            # scan mid-frame causing micro-blackouts (flickering).
            time.sleep(frame_interval)

    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        fb[:] = 0
        for _ in range(16):
            matrix.show()
            time.sleep(0.02)
        print("\nDone.")


if __name__ == "__main__":
    main()
