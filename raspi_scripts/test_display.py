#!/usr/bin/env python3
"""
Road Sentinel — interactive display test.

Type a number + Enter to switch screens:
  1  SLOW DOWN
  2  VEHICLE INCOMING
  3  INCIDENT AHEAD
  4  COLOR BARS
  q  Quit

Usage:
  sudo python3 raspi_scripts/test_display.py
  sudo python3 raspi_scripts/test_display.py --start vehicle
  sudo python3 raspi_scripts/test_display.py --pi 4
"""

import argparse
import logging
import os
import select
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from display_manager import (
    create_backend, _detect_pi, TICK,
    SystemState,
    render_slow_down, render_vehicle_incoming,
    render_incident_ahead, render_color_bars,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TEST] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SCREENS = {
    "1": "slow_down",
    "2": "vehicle",
    "3": "incident",
    "4": "color_bars",
}

LABELS = {
    "slow_down":  "SLOW DOWN",
    "vehicle":    "VEHICLE INCOMING",
    "incident":   "INCIDENT AHEAD",
    "color_bars": "COLOR BARS",
}


def _startup_blank(backend):
    log.info("Startup blank burst...")
    for _ in range(16):
        backend.clear()
        time.sleep(0.02)
    end = time.monotonic() + 0.5
    while time.monotonic() < end:
        backend.clear()
        time.sleep(TICK)
    log.info("Panel ready")


def _get_frame(name, state, color_phase):
    if name == "slow_down":
        return render_slow_down(state, flash_phase=0)
    if name == "vehicle":
        return render_vehicle_incoming(state, flash_phase=0)
    if name == "incident":
        return render_incident_ahead(state, flash_phase=0)
    return render_color_bars(color_phase)


class _DefaultArgs:
    cols           = 64
    chain          = 2
    slowdown       = 4
    mapping        = "regular"
    hardware_pulse = False
    multiplexing   = 0
    scan_mode      = 0
    ledcat         = None
    viewer         = None
    pinout         = "active3"
    addr_lines     = 4
    emulator       = False


def main():
    parser = argparse.ArgumentParser(description="Road Sentinel interactive display test")
    parser.add_argument("--start", default="slow_down", choices=list(LABELS.keys()),
                        help="Starting screen (default: slow_down)")
    parser.add_argument("--pi", choices=["4", "5"], default=None)
    args = parser.parse_args()

    pi_model = f"pi{args.pi}" if args.pi else _detect_pi()
    backend  = create_backend(_DefaultArgs(), pi_model)
    state    = SystemState()
    state.update_summary({
        "vehicles_today": 42, "average_speed": 35,
        "incidents_today": 0, "cameras_online": 2, "cameras_total": 2,
    })

    try:
        _startup_blank(backend)

        current     = args.start
        color_phase = 0
        phase_end   = time.monotonic() + 0.5

        log.info("Now showing: %s", LABELS[current])
        log.info("Type + Enter:  1=SLOW DOWN  2=VEHICLE  3=INCIDENT  4=COLOR BARS  q=quit")

        while True:
            # Non-blocking stdin check — works in web terminals without raw mode
            r, _, _ = select.select([sys.stdin], [], [], 0)
            if r:
                line = sys.stdin.readline()
                if not line:          # EOF — stdin closed
                    break
                key = line.strip()
                if key == "q":
                    break
                if key in SCREENS:
                    current     = SCREENS[key]
                    color_phase = 0
                    phase_end   = time.monotonic() + 0.5
                    log.info("Now showing: %s", LABELS[current])
                elif key:
                    log.info("Unknown: %s  — use 1 / 2 / 3 / 4 / q", key)

            if current == "color_bars" and time.monotonic() >= phase_end:
                color_phase = (color_phase + 1) % 6
                phase_end   = time.monotonic() + 0.5

            backend.show(_get_frame(current, state, color_phase))
            time.sleep(TICK)

    except KeyboardInterrupt:
        pass
    finally:
        backend.close()
        log.info("Stopped")


if __name__ == "__main__":
    main()
