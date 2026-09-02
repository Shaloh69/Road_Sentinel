#!/usr/bin/env python3
"""
Road Sentinel — Pi → ESP32 display bridge.

Replaces display_manager.py's role on installations where the LED panel is
driven by an ESP32 over USB serial instead of the Pi's own GPIO.

The split: the Pi keeps everything that needs a network or a brain — polling
the Node API, deciding what state the road is in, reconnecting, logging. The
ESP32 knows only how to draw four screens. If the sign shows the wrong thing,
the bug is here; if it shows it wrongly, the bug is in the firmware. That
separation is most of the value of this design.

The same state logic as display_manager.py:
    INCIDENT AHEAD    active incident from either camera   (12s hold)
    VEHICLE INCOMING  recent detection from either camera  ( 8s hold)
    ROAD CLEAR        no recent activity
    -- NO DATA --     Node unreachable (handled by the ESP32's own timeout)

Usage:
    python3 esp32_display_bridge.py --api http://100.120.27.110:3001
    python3 esp32_display_bridge.py --port /dev/ttyUSB0 --test

Requires: pyserial  (pip install pyserial)
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
import time

try:
    import serial  # type: ignore
except ImportError:
    print("pyserial not installed.  pip install pyserial", file=sys.stderr)
    raise SystemExit(1)

try:
    import requests
except ImportError:
    print("requests not installed.  pip install requests", file=sys.stderr)
    raise SystemExit(1)

log = logging.getLogger("esp32-bridge")

# How long a detection/incident keeps the sign lit after the last event.
# Matches display_manager.py so both display paths behave identically.
VEHICLE_HOLD_SECS = 8
INCIDENT_HOLD_SECS = 12

POLL_INTERVAL = 2.0
BAUD = 115200


def find_port() -> str | None:
    """First plausible USB serial device. ESP32 boards show up as either."""
    for pattern in ("/dev/ttyUSB*", "/dev/ttyACM*"):
        found = sorted(glob.glob(pattern))
        if found:
            return found[0]
    return None


class EspLink:
    """Serial link to the display board, reconnecting on its own."""

    def __init__(self, port: str | None):
        self._explicit_port = port
        self._ser: serial.Serial | None = None
        self._last_sent: str | None = None

    def _open(self) -> bool:
        port = self._explicit_port or find_port()
        if not port:
            return False
        try:
            self._ser = serial.Serial(port, BAUD, timeout=1)
            # Opening the port toggles DTR, which resets most ESP32 boards.
            # Give the firmware time to boot before the first command, or it
            # lands in the bootloader's lap and is silently lost.
            time.sleep(2.0)
            self._ser.reset_input_buffer()
            log.info("Connected to display board on %s", port)
            self._last_sent = None      # force a resend after any reconnect
            return True
        except (serial.SerialException, OSError) as exc:
            log.warning("Could not open %s: %s", port, exc)
            self._ser = None
            return False

    def send(self, cmd: str, force: bool = False) -> bool:
        """Send a command; skip if unchanged, unless forced."""
        if cmd == self._last_sent and not force:
            return True

        if self._ser is None and not self._open():
            return False

        try:
            self._ser.write((cmd + "\n").encode())   # type: ignore[union-attr]
            self._ser.flush()                        # type: ignore[union-attr]
            self._last_sent = cmd
            log.debug("sent %s", cmd)
            return True
        except (serial.SerialException, OSError) as exc:
            log.warning("Write failed (%s) — will reconnect", exc)
            self.close()
            return False

    def close(self) -> None:
        if self._ser:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None
        self._last_sent = None


def road_state(api: str, session: requests.Session) -> str:
    """
    Ask Node what the road looks like right now.

    Uses /api/public/status, which already computes this server-side for the
    public status page — so the sign and the web page cannot disagree, which
    they would if this recomputed the rule itself.
    """
    r = session.get(f"{api}/api/public/status", timeout=5)
    r.raise_for_status()
    data = r.json()
    if not data.get("success"):
        raise RuntimeError("status endpoint returned success=false")

    state = data["data"]["state"]
    return {
        "incident": "incident",
        "vehicle_incoming": "vehicle",
        "clear": "clear",
    }.get(state, "clear")


def run_test(link: EspLink) -> int:
    """Cycle every screen so the panel can be checked without a server."""
    screens = [
        ("STATE:clear", "ROAD CLEAR"),
        ("STATE:vehicle", "VEHICLE INCOMING (flashing)"),
        ("STATE:incident", "INCIDENT AHEAD (flashing)"),
        ("STATE:offline", "NO DATA"),
        ("TEXT:ROAD|SENTINEL", "custom two-line text"),
    ]
    for cmd, desc in screens:
        print(f"  {desc:34s} -> {cmd}")
        if not link.send(cmd, force=True):
            print("  FAILED to send — is the board connected?")
            return 1
        time.sleep(4)
    link.send("STATE:clear", force=True)
    print("Test complete.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Pi -> ESP32 LED display bridge")
    ap.add_argument("--api", default="http://100.120.27.110:3001",
                    help="Node service base URL")
    ap.add_argument("--port", default=None,
                    help="Serial device (default: first /dev/ttyUSB* or ttyACM*)")
    ap.add_argument("--brightness", type=int, default=None,
                    help="Panel brightness 0-255, set once at startup")
    ap.add_argument("--test", action="store_true",
                    help="Cycle all screens and exit — no server needed")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [esp32-bridge] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    link = EspLink(args.port)

    if args.brightness is not None:
        link.send(f"BRIGHT:{args.brightness}", force=True)

    if args.test:
        return run_test(link)

    session = requests.Session()
    log.info("Bridging %s -> display board", args.api)

    last_state = None
    last_change = 0.0
    consecutive_errors = 0

    while True:
        try:
            state = road_state(args.api, session)
            consecutive_errors = 0

            now = time.monotonic()
            # Hold urgent states briefly so a single frame's detection does not
            # flicker the sign off again immediately. Drivers need time to read
            # it, and a sign that blinks between two messages reads as broken.
            if last_state in ("incident", "vehicle") and state == "clear":
                hold = (INCIDENT_HOLD_SECS if last_state == "incident"
                        else VEHICLE_HOLD_SECS)
                if now - last_change < hold:
                    state = last_state

            if state != last_state:
                log.info("state -> %s", state)
                last_change = now
                last_state = state

            link.send(f"STATE:{state}")

        except Exception as exc:
            consecutive_errors += 1
            if consecutive_errors in (1, 5) or consecutive_errors % 30 == 0:
                log.warning("Poll failed (%d): %s", consecutive_errors, exc)
            # Deliberately do NOT send STATE:offline here. The board has its own
            # 15s timeout and will show "NO DATA" by itself. Letting it decide
            # means a dead serial cable produces the same honest result as a
            # dead API, instead of the sign confidently holding a stale screen.

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        sys.exit(0)
