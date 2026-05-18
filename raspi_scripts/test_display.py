#!/usr/bin/env python3
"""
Road Sentinel — interactive display test.

Start the script in one terminal, then switch screens by running echo
commands from any other terminal (or the same one after backgrounding):

  sudo python3 raspi_scripts/test_display.py        # starts, shows SLOW DOWN

  echo 1 > /tmp/led_cmd    # switch to SLOW DOWN
  echo 2 > /tmp/led_cmd    # switch to VEHICLE INCOMING
  echo 3 > /tmp/led_cmd    # switch to INCIDENT AHEAD
  echo 4 > /tmp/led_cmd    # switch to COLOR BARS
  echo q > /tmp/led_cmd    # quit

Or run in background first:
  sudo python3 raspi_scripts/test_display.py &
  echo 2 > /tmp/led_cmd
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from display_manager import (
    create_backend, _detect_pi, _snap_frame, TICK, _BLACK_FRAME,
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

CMD_FILE = "/tmp/led_cmd"

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
        backend._write(_BLACK_FRAME)
        time.sleep(0.02)
    end = time.monotonic() + 0.5
    while time.monotonic() < end:
        backend._write(_BLACK_FRAME)
        time.sleep(TICK)
    log.info("Panel ready")


def _read_cmd():
    """Read and clear the command file. Returns stripped string or None."""
    try:
        if not os.path.exists(CMD_FILE):
            return None
        with open(CMD_FILE) as f:
            cmd = f.read().strip()
        open(CMD_FILE, "w").close()   # clear after reading
        return cmd or None
    except OSError:
        return None


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

    # Clear any stale command file from a previous run
    try:
        open(CMD_FILE, "w").close()
    except OSError:
        pass

    pi_model = f"pi{args.pi}" if args.pi else _detect_pi()
    backend  = create_backend(_DefaultArgs(), pi_model)

    try:
        _startup_blank(backend)

        # Pre-compute all frames as raw bytes once — zero PIL/numpy overhead in loop
        state_obj = SystemState()
        state_obj.update_summary({
            "vehicles_today": 42, "average_speed": 35,
            "incidents_today": 0, "cameras_online": 2, "cameras_total": 2,
        })
        frames = {
            "slow_down": _snap_frame(render_slow_down(state_obj, flash_phase=0)),
            "vehicle":   _snap_frame(render_vehicle_incoming(state_obj, flash_phase=0)),
            "incident":  _snap_frame(render_incident_ahead(state_obj, flash_phase=0)),
        }
        color_bar_frames = [_snap_frame(render_color_bars(p)) for p in range(6)]

        current     = args.start
        color_phase = 0
        phase_end   = time.monotonic() + 0.5

        log.info("Now showing: %s", LABELS[current])
        log.info("To switch — in another terminal:  echo 2 > /tmp/led_cmd")
        log.info("Screens: 1=SLOW DOWN  2=VEHICLE  3=INCIDENT  4=COLOR BARS  q=quit")

        while True:
            cmd = _read_cmd()
            if cmd:
                if cmd == "q":
                    break
                if cmd in SCREENS:
                    current     = SCREENS[cmd]
                    color_phase = 0
                    phase_end   = time.monotonic() + 0.5
                    log.info("Now showing: %s", LABELS[current])
                else:
                    log.info("Unknown command: %s", cmd)

            if current == "color_bars" and time.monotonic() >= phase_end:
                color_phase = (color_phase + 1) % 6
                phase_end   = time.monotonic() + 0.5

            frame = color_bar_frames[color_phase] if current == "color_bars" else frames[current]
            backend._write(frame)
            time.sleep(TICK)

    except KeyboardInterrupt:
        pass
    finally:
        backend.close()
        log.info("Stopped")


if __name__ == "__main__":
    main()
