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

Usage:
    sudo ~/venvs/cam_venv/bin/python3 ~/roadsentinel/color_test.py
    sudo ~/venvs/cam_venv/bin/python3 ~/roadsentinel/color_test.py --pinout active3
    sudo ~/venvs/cam_venv/bin/python3 ~/roadsentinel/color_test.py --brightness 0.3
"""

import argparse
import itertools
import signal
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

VERSION = "v2026-05-10-r3"   # bump this to confirm latest code is running

try:
    import adafruit_blinka_raspberry_pi5_piomatter as piomatter
    _PM_VERSION = getattr(piomatter, "__version__", "unknown")
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


def build_frame(name: str, rgb: tuple, brightness: float) -> np.ndarray:
    dimmed = tuple(int(c * brightness) for c in rgb)
    img = Image.new("RGB", (W, H), dimmed)
    label_color = (0, 0, 0) if any(c > 40 for c in dimmed) else (60, 60, 60)
    try:
        font = ImageFont.load_default(size=10)
    except TypeError:
        font = ImageFont.load_default()
    ImageDraw.Draw(img).text((2, 10), name, fill=label_color, font=font)
    return np.asarray(img.convert("RGB")) + 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--addr-lines",  type=int,   default=4,
                   help="4 for 32px tall (1/16 scan), 3 for 16px tall (1/8 scan)")
    p.add_argument("--delay",       type=float, default=3.0, help="Seconds per color")
    p.add_argument("--fps",         type=int,   default=30,  help="show() calls per second")
    p.add_argument("--brightness",  type=float, default=1.0,
                   help="0.0–1.0 — reduce to debug power issues (try 0.3)")
    p.add_argument("--pinout",      default="active3bgr",
                   choices=["active3", "active3bgr"],
                   help="active3bgr swaps R↔B in software")
    args = p.parse_args()

    if not 0.0 < args.brightness <= 1.0:
        raise SystemExit("--brightness must be between 0.01 and 1.0")

    if args.pinout == "active3bgr":
        pinout = getattr(piomatter.Pinout, "Active3BGR", piomatter.Pinout.Active3)
        pinout_name = "Active3BGR"
    else:
        pinout = piomatter.Pinout.Active3
        pinout_name = "Active3"

    geo = piomatter.Geometry(
        width=W, height=H,
        n_addr_lines=args.addr_lines,
        rotation=piomatter.Orientation.Normal,
    )
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    fb = np.asarray(canvas) + 0

    matrix = piomatter.PioMatter(
        colorspace=piomatter.Colorspace.RGB888Packed,
        pinout=pinout,
        framebuffer=fb,
        geometry=geo,
    )

    frames = [(name, rgb, build_frame(name, rgb, args.brightness)) for name, rgb in COLORS]
    frame_interval = 1.0 / args.fps

    # ── Startup banner ────────────────────────────────────────────────────────
    print("=" * 52)
    print(f"  color_test.py  {VERSION}")
    print(f"  piomatter      {_PM_VERSION}")
    print(f"  panel          {W}x{H}  addr_lines={args.addr_lines}")
    print(f"  pinout         {pinout_name}")
    print(f"  brightness     {int(args.brightness * 100)}%")
    print(f"  refresh        {args.fps} fps   delay={args.delay}s/color")
    print("=" * 52)
    print("  Hardware checklist:")
    print("  [ ] Pi GND (pin 6/9/14) → LED PSU negative terminal")
    print("  [ ] HUB75 pin 4, 8, 16 (GND) all connected")
    print("  [ ] OE  GPIO18 (pin 12) → HUB75 pin 15")
    print("  [ ] CLK GPIO17 (pin 11) → HUB75 pin 13")
    print("  [ ] LAT GPIO4  (pin  7) → HUB75 pin 14")
    print("  [ ] PSU: separate 5V ≥2A for panel")
    print("  [ ] Audio disabled: dtparam=audio=off in config.txt")
    print("=" * 52)
    print("  Colors (actual RGB values sent to panel):")
    for name, rgb, _ in frames:
        dimmed = tuple(int(c * args.brightness) for c in rgb)
        print(f"    {name:<6} raw={rgb}  sent={dimmed}")
    print("=" * 52)
    print("  Ctrl-C once to stop cleanly\n")

    color_iter = itertools.cycle(frames)
    name, rgb, arr = next(color_iter)
    deadline = time.monotonic() + args.delay
    dimmed = tuple(int(c * args.brightness) for c in rgb)
    print(f"  → {name:<6}  sent={dimmed}")

    try:
        while True:
            now = time.monotonic()
            if now >= deadline:
                name, rgb, arr = next(color_iter)
                deadline = now + args.delay
                dimmed = tuple(int(c * args.brightness) for c in rgb)
                print(f"  → {name:<6}  sent={dimmed}")

            fb[:] = arr
            matrix.show()
            time.sleep(frame_interval)

    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print("\n  Blanking panel...")
        fb[:] = 0
        for _ in range(16):
            matrix.show()
            time.sleep(0.02)
        print("  Done.")


if __name__ == "__main__":
    main()
