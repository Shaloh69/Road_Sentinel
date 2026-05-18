#!/usr/bin/env python3
"""
Road Sentinel — isolated display test.
Tests each screen individually, with and without the blank transition.

Usage:
  sudo python3 raspi_scripts/test_display.py <test> [--raw] [--hold N]

Tests:
  slow_down    SLOW DOWN screen (red)
  vehicle      VEHICLE INCOMING screen (orange)
  incident     INCIDENT AHEAD screen (magenta)
  color_bars   Color bar sweep
  all          Run all four tests in sequence

Flags:
  --raw        Skip blank transition — write content directly to panel
  --hold N     Hold each screen for N seconds (default: 4)
  --pi 4|5     Force Pi model (default: auto-detect)

Examples:
  sudo python3 raspi_scripts/test_display.py slow_down
  sudo python3 raspi_scripts/test_display.py slow_down --raw
  sudo python3 raspi_scripts/test_display.py all
  sudo python3 raspi_scripts/test_display.py all --raw
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
    render_slow_down, render_vehicle_incoming,
    render_incident_ahead, render_color_bars,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TEST] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _startup_blank(backend):
    """Rapid burst at 50fps, then hold at TICK rate for 500ms — never lets RP1 idle."""
    log.info("Startup blank burst...")
    for _ in range(16):
        backend.clear()
        time.sleep(0.02)
    end = time.monotonic() + 0.5
    while time.monotonic() < end:
        backend.clear()
        time.sleep(TICK)
    log.info("Panel ready")


def _show_forever(backend, img):
    """Push the same frame at TICK rate until Ctrl+C."""
    while True:
        backend.show(img)
        time.sleep(TICK)


# ── Individual test functions ─────────────────────────────────────────────────

def test_slow_down(backend, state):
    log.info("Now showing: SLOW DOWN")
    _show_forever(backend, render_slow_down(state, flash_phase=0))


def test_vehicle(backend, state):
    log.info("Now showing: VEHICLE INCOMING")
    _show_forever(backend, render_vehicle_incoming(state, flash_phase=0))


def test_incident(backend, state):
    log.info("Now showing: INCIDENT AHEAD")
    _show_forever(backend, render_incident_ahead(state, flash_phase=0))


def test_color_bars(backend, state):
    phase = 0
    while True:
        log.info("Now showing: COLOR BARS phase=%d", phase)
        img = render_color_bars(phase)
        end = time.monotonic() + 0.5
        while time.monotonic() < end:
            backend.show(img)
            time.sleep(TICK)
        phase = (phase + 1) % 6


TESTS = {
    "slow_down":  test_slow_down,
    "vehicle":    test_vehicle,
    "incident":   test_incident,
    "color_bars": test_color_bars,
}



# ── Entry point ───────────────────────────────────────────────────────────────

class _DefaultArgs:
    """Minimal args object that satisfies create_backend for both Pi 4 and Pi 5."""
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
    parser = argparse.ArgumentParser(
        description="Road Sentinel isolated display test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("test", choices=list(TESTS.keys()),
                        help="Screen to test")
    parser.add_argument("--pi",   choices=["4", "5"], default=None,
                        help="Force Pi model (default: auto-detect)")
    args = parser.parse_args()

    pi_model = f"pi{args.pi}" if args.pi else _detect_pi()
    log.info("Pi: %s | test: %s", pi_model, args.test)

    backend = create_backend(_DefaultArgs(), pi_model)
    state   = SystemState()
    state.update_summary({
        "vehicles_today": 42, "average_speed": 35,
        "incidents_today": 0, "cameras_online": 2, "cameras_total": 2,
    })

    try:
        _startup_blank(backend)

        log.info("--- %s --- (Ctrl+C to stop)", args.test.upper())
        TESTS[args.test](backend, state)

        log.info("Test complete")

    except KeyboardInterrupt:
        log.info("Stopped by user")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
