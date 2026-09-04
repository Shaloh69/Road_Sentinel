#!/usr/bin/env python3
"""
Road Sentinel — Pi → LED sign bridge.

Drives the roadside sign over USB serial. Both installations use an ESP32:

    Pi 4  ->  ESP32  (/dev/ttyUSB*, CP2102/CH340 bridge)
    Pi 5  ->  ESP32  (same board, same firmware)

The device probing still accepts ttyACM* as well, so a board with native USB
CDC can be swapped in without touching this script.

This replaced the Pi-GPIO LED driver entirely. That approach never worked on
these 1/8-scan FM6124 panels (~80 configurations tried) and has been removed;
see LEDMatrixDrivers/esp32/DEBUG_LOG.md for the full account.

The split: the Pi keeps everything that needs a network or a brain — polling
the Node API, deciding what state the road is in, reconnecting, logging. The
ESP32 knows only how to draw four screens. If the sign shows the wrong thing,
the bug is here; if it shows it wrongly, the bug is in the firmware. That
separation is most of the value of this design.

Road state, and what the sign shows for each:
    STOP              active incident from either camera   (12s hold)
    VEHICLE INCOMING  recent detection from either camera  ( 8s hold)
    SAFE              no recent activity
    NO DATA           Node unreachable (handled by the board's own timeout)

Usage:
    python3 led_sign_bridge.py --api http://100.120.27.110:3001
    python3 led_sign_bridge.py --port /dev/ttyUSB0 --test

Requires: pyserial  (pip install pyserial)
"""

from __future__ import annotations

import argparse
import glob
import os
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

log = logging.getLogger("led-sign")

# How long a detection/incident keeps the sign lit after the last event.
# Held here rather than in firmware: a single frame's detection would otherwise
# flicker the sign off again immediately, and a sign that blinks between two
# messages reads as broken rather than as informative.
VEHICLE_HOLD_SECS = 8
INCIDENT_HOLD_SECS = 12

POLL_INTERVAL = 2.0
BAUD = 115200


# A udev rule (installed by setup_pi4.sh / setup_pi5.sh) points this symlink at
# whichever sign board is attached. Preferring it means enumeration order no
# longer matters — plugging a second USB serial device in cannot silently steal
# /dev/ttyUSB0 and leave the sign talking to a camera.
STABLE_LINK = "/dev/roadsentinel-sign"


def find_port() -> str | None:
    """Locate the sign board.

    Order matters. The stable symlink is authoritative when present; the
    globs are the fallback for a Pi whose udev rule has not been installed.

    ESP32 boards appear as ttyUSB* (external CP2102/CH340 bridge). ttyACM* is
    checked too, which costs nothing and covers a board with native USB CDC if
    one is ever swapped in.
    """
    if os.path.exists(STABLE_LINK):
        return STABLE_LINK
    for pattern in ("/dev/ttyUSB*", "/dev/ttyACM*"):
        found = sorted(glob.glob(pattern))
        if found:
            return found[0]
    return None


class EspLink:
    """Serial link to the sign board, reconnecting on its own."""

    def __init__(self, port: str | None, brightness: int | None = None):
        self._explicit_port = port
        self._ser: serial.Serial | None = None
        self._last_sent: str | None = None
        # Reapplied on every (re)connect. Sending it only at startup meant a
        # board that reset overnight came back at its firmware default, and
        # nobody would notice until the sign looked wrong in daylight.
        self._brightness = brightness

    def _open(self) -> bool:
        port = self._explicit_port or find_port()
        if not port:
            return False
        try:
            self._ser = serial.Serial(port, BAUD, timeout=1)

            # Opening the port toggles DTR, which resets most ESP32 boards.
            # Give the firmware time to boot before the first command, or it
            # lands in the bootloader's lap and is silently lost. A board with
            # native USB CDC would not reset on open, making the wait
            # unnecessary there — but it is harmless, so no board-specific
            # branch.
            time.sleep(2.0)
            self._ser.reset_input_buffer()

            # Prove the board is actually running firmware, not just that the
            # device node exists. A wedged or half-flashed board still
            # enumerates, and without this the bridge would happily "send"
            # state into a void for hours while the sign showed nothing.
            if not self._handshake():
                log.warning("No response from board on %s — will retry", port)
                self.close()
                return False

            log.info("Connected to sign board on %s", port)
            self._last_sent = None      # force a resend after any reconnect
            if self._brightness is not None:
                self._ser.write(f"BRIGHT:{self._brightness}\n".encode())
                self._ser.flush()
            return True
        except (serial.SerialException, OSError) as exc:
            log.warning("Could not open %s: %s", port, exc)
            self._ser = None
            return False

    def _handshake(self) -> bool:
        """PING for liveness, then log INFO.

        INFO goes into the service log deliberately: when someone reports the
        sign misbehaving weeks from now, the log already says which firmware
        build, which mapping and what refresh rate were live at the time.
        Reconstructing that after the fact is otherwise guesswork.
        """
        try:
            self._ser.write(b"PING\n")
            self._ser.flush()
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                line = self._ser.readline().decode(errors="replace").strip()
                if line == "PONG":
                    break
                if line == "READY":
                    continue        # board just booted; keep waiting for PONG
            else:
                return False

            self._ser.write(b"INFO\n")
            self._ser.flush()
            info = self._ser.readline().decode(errors="replace").strip()
            if info:
                log.info("Board: %s", info)
            return True
        except (serial.SerialException, OSError):
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

    def alive(self) -> bool:
        """Round-trip check. Returns False if the board stopped answering.

        Writing to a wedged board succeeds — the OS buffers it — so a write
        that does not raise proves nothing. Without this the bridge could sit
        for hours reporting healthy while the sign showed a frozen screen.
        """
        if self._ser is None:
            return False
        try:
            self._ser.reset_input_buffer()
            self._ser.write(b"PING\n")
            self._ser.flush()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if self._ser.readline().decode(errors="replace").strip() == "PONG":
                    return True
            return False
        except (serial.SerialException, OSError):
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

    payload = data["data"]

    # Transient display-mode override, set server-side and self-expiring. When
    # present it replaces the road state entirely; when it lapses the sign
    # returns to normal status duty on the next poll with no further action.
    mode = payload.get("sign_mode")
    if mode:
        return f"@{mode}"

    state = payload["state"]
    return {
        "incident": "incident",
        "vehicle_incoming": "vehicle",
        "clear": "clear",
    }.get(state, "clear")


def run_test(link: EspLink) -> int:
    """Cycle every screen so the panel can be checked without a server."""
    screens = [
        ("STATE:clear", "SAFE (green)"),
        ("STATE:vehicle", "VEHICLE INCOMING / SLOW DOWN (flashing yellow)"),
        ("STATE:incident", "STOP (flashing red)"),
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
    ap = argparse.ArgumentParser(description="Pi -> LED sign bridge (ESP32)")
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
        format="%(asctime)s [led-sign] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    link = EspLink(args.port, args.brightness)

    if args.test:
        return run_test(link)

    session = requests.Session()
    log.info("Bridging %s -> LED sign", args.api)

    last_state = None
    last_change = 0.0
    consecutive_errors = 0
    last_health = 0.0

    # How often to prove the board is still answering. 30s is a compromise:
    # frequent enough that a wedged sign is caught within a minute, rare
    # enough that the PING traffic is negligible.
    HEALTH_INTERVAL = 30.0

    while True:
        try:
            state = road_state(args.api, session)
            consecutive_errors = 0

            now = time.monotonic()

            # An override is passed straight through as a mode switch. It also
            # bypasses the alert-hold logic below, which exists to stop urgent
            # road states flickering and has no meaning here.
            if state.startswith("@"):
                if state != last_state:
                    # Logged like any other state change. An unlogged command
                    # to a roadside sign is a gap: if someone reports the sign
                    # showing something unexpected, the service log has to be
                    # able to say what it was told and when.
                    log.info("mode -> %s", state[1:])
                    last_change = now
                    last_state = state
                link.send(f"MODE:{state[1:]}")
                time.sleep(POLL_INTERVAL)
                continue
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

            # Liveness. A board can wedge while still enumerating, in which
            # case every send() below succeeds and the sign quietly freezes.
            if now - last_health >= HEALTH_INTERVAL:
                last_health = now
                if not link.alive():
                    log.warning("Board stopped answering PING — reconnecting")
                    link.close()

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
