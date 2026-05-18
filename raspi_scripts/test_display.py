#!/usr/bin/env python3
"""
Road Sentinel — auto-cycling display test.

Cycles through SLOW DOWN → VEHICLE INCOMING → VACANT ROAD every 5 seconds,
restarting led-image-viewer on each transition. This keeps the RP1 refresh
thread fresh and prevents PWM timing drift that causes white-line corruption.

Usage:
  sudo python3 raspi_scripts/test_display.py
  sudo python3 raspi_scripts/test_display.py --pi 5
  sudo python3 raspi_scripts/test_display.py --interval 10
"""

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from display_manager import (
    create_backend, _detect_pi, TICK,
    SystemState,
    render_slow_down, render_vehicle_incoming, render_vacant_road,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TEST] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CYCLE = ["slow_down", "vehicle", "vacant"]

LABELS = {
    "slow_down": "SLOW DOWN",
    "vehicle":   "VEHICLE INCOMING",
    "vacant":    "VACANT ROAD",
}


def _startup_blank(backend):
    log.info("Blanking panel...")
    for _ in range(16):
        backend.clear()
        time.sleep(0.02)
    end = time.monotonic() + 0.5
    while time.monotonic() < end:
        backend.clear()
        time.sleep(TICK)
    log.info("Panel ready")


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
    parser = argparse.ArgumentParser(description="Road Sentinel auto-cycling display test")
    parser.add_argument("--pi",       choices=["4", "5"], default=None,
                        help="Force Pi model (default: auto-detect)")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Seconds per screen (default: 5)")
    args = parser.parse_args()

    pi_model = f"pi{args.pi}" if args.pi else _detect_pi()
    log.info("Pi model: %s  |  cycle interval: %.1fs", pi_model, args.interval)

    backend = create_backend(_DefaultArgs(), pi_model)

    try:
        _startup_blank(backend)

        state_obj = SystemState()
        state_obj.update_summary({
            "vehicles_today": 42, "average_speed": 35,
            "incidents_today": 0, "cameras_online": 2, "cameras_total": 2,
        })

        frames = {
            "slow_down": render_slow_down(state_obj,        flash_phase=0),
            "vehicle":   render_vehicle_incoming(state_obj, flash_phase=0),
            "vacant":    render_vacant_road(state_obj),
        }

        cycle_idx  = 0
        next_cycle = time.monotonic() + args.interval

        current = CYCLE[cycle_idx]
        log.info("► Showing: %s", LABELS[current])

        while True:
            if time.monotonic() >= next_cycle:
                cycle_idx  = (cycle_idx + 1) % len(CYCLE)
                current    = CYCLE[cycle_idx]
                next_cycle = time.monotonic() + args.interval
                log.info("► Showing: %s", LABELS[current])

            backend.show(frames[current])
            time.sleep(TICK)

    except KeyboardInterrupt:
        log.info("Interrupted")
    finally:
        backend.close()
        log.info("Stopped")


if __name__ == "__main__":
    main()
