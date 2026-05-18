#!/usr/bin/env python3
"""
Road Sentinel — interactive display test.

Keys during run:
  1  SLOW DOWN
  2  VEHICLE INCOMING
  3  INCIDENT AHEAD
  4  COLOR BARS
  q  Quit

Usage:
  sudo python3 raspi_scripts/test_display.py [--start slow_down|vehicle|incident|color_bars]
  sudo python3 raspi_scripts/test_display.py --pi 4
"""

import argparse
import logging
import os
import select
import sys
import termios
import time
import tty

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


def _read_key():
    """Return a keypress if available, else None (non-blocking)."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


def _get_frame(name, state, color_phase):
    if name == "slow_down":
        return render_slow_down(state, flash_phase=0)
    if name == "vehicle":
        return render_vehicle_incoming(state, flash_phase=0)
    if name == "incident":
        return render_incident_ahead(state, flash_phase=0)
    if name == "color_bars":
        return render_color_bars(color_phase)
    return render_slow_down(state, flash_phase=0)


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
    parser.add_argument("--start", default="slow_down",
                        choices=list(LABELS.keys()),
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

    old_tty = termios.tcgetattr(sys.stdin.fileno())
    try:
        _startup_blank(backend)

        log.info("Keys: 1=SLOW DOWN  2=VEHICLE INCOMING  3=INCIDENT AHEAD  4=COLOR BARS  q=quit")
        tty.setraw(sys.stdin.fileno())

        current      = args.start
        color_phase  = 0
        phase_end    = time.monotonic() + 0.5
        prev         = None

        while True:
            key = _read_key()
            if key == "q":
                break
            if key in SCREENS:
                current = SCREENS[key]
                color_phase = 0
                phase_end   = time.monotonic() + 0.5

            if current != prev:
                # Log outside raw mode so the line isn't garbled
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_tty)
                log.info("Now showing: %s", LABELS[current])
                tty.setraw(sys.stdin.fileno())
                prev = current

            # Cycle color bar phases automatically
            if current == "color_bars" and time.monotonic() >= phase_end:
                color_phase = (color_phase + 1) % 6
                phase_end   = time.monotonic() + 0.5

            backend.show(_get_frame(current, state, color_phase))
            time.sleep(TICK)

    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_tty)
        backend.close()
        log.info("Stopped")


if __name__ == "__main__":
    main()
