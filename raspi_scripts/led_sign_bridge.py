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
import datetime
import glob
import os
import logging
import subprocess
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

# Re-send the current state at least this often, even when nothing changed.
#
# The firmware falls back to "NO DATA" if no command arrives for 15s, which is
# deliberate: a dead cable or a dead Pi should produce an honest blank rather
# than a stale SAFE. But send() suppresses unchanged commands, so on a quiet
# road the bridge would fall silent after the first STATE and the board would
# time out on a perfectly healthy link — the sign showing NO DATA while
# everything worked.
#
# Well under the firmware's timeout so a single dropped write cannot trip it.
RESEND_INTERVAL = 5.0
BAUD = 115200

# ── Time-of-day brightness ──────────────────────────────────────────────────
#
# The panel is sized for daylight legibility, which after dark is glare in the
# face of a driver entering a blind curve. So the sign runs full brightness by
# day and drops to a low level at night.
#
# The schedule lives here and not in firmware because the board has no clock —
# no RTC, no WiFi, no notion of the date (see sign_app.cpp). The Pi is the only
# device in the sign that knows what time it is.
#
# Fixed clock times rather than a solar almanac: Busay is at ~10.3 degrees N,
# where sunrise and sunset move by only about 20 minutes across the whole year.
# A sunrise library would add a dependency and a failure mode to buy accuracy
# the panel cannot even display.
DAWN_DEFAULT = "05:30"
DUSK_DEFAULT = "17:45"

# Night is dim, NOT off. 0 would blank a safety sign outright, so the floor
# stays above it. 5 was chosen on site by eye — the panel is bright enough that
# a low number still reads clearly in the dark, and anything higher was glare
# for a driver coming into the curve.
NIGHT_BRIGHT_DEFAULT = 5
DAY_BRIGHT_DEFAULT = 255

# Ramp, in minutes, so the sign fades rather than snapping between levels.
# Brightening begins at dawn and completes RAMP minutes later; dimming begins
# at dusk and completes RAMP minutes after that.
RAMP_MINUTES_DEFAULT = 30

# Below this change, do not bother the board. The ramp would otherwise emit a
# BRIGHT: every poll for half an hour, and a difference of one step is not
# visible on an 8-colour panel anyway.
BRIGHT_EPSILON = 4


def _clock_is_trustworthy() -> bool:
    """Has the system clock actually been synchronised?

    This matters more than it looks. Both Pis boot with a stale RTC and think
    it is still the date they were last shut down on, until NTP corrects them
    seconds-to-minutes later. A brightness schedule that trusts that clock
    would happily run the sign at 20/255 through the middle of the morning.

    systemd-timesyncd drops this file once it has a real time; the timedatectl
    call is the fallback for images using a different sync daemon.
    """
    if os.path.exists("/run/systemd/timesync/synchronized"):
        return True
    try:
        out = subprocess.run(
            ["timedatectl", "show", "-p", "NTPSynchronized", "--value"],
            capture_output=True, text=True, timeout=3)
        return out.stdout.strip() == "yes"
    except (OSError, subprocess.SubprocessError):
        return False


def parse_hhmm(s: str) -> int:
    """'17:45' -> minutes since midnight."""
    h, _, m = s.partition(":")
    mins = int(h) * 60 + int(m or 0)
    if not 0 <= mins < 1440:
        raise ValueError(f"time out of range: {s}")
    return mins


def scheduled_brightness(now_min: int, dawn: int, dusk: int,
                         night: int, day: int, ramp: int) -> int:
    """Brightness for a given minute-of-day, with linear ramps."""
    if ramp <= 0:
        return day if dawn <= now_min < dusk else night

    def lerp(a: int, b: int, f: float) -> int:
        return int(round(a + (b - a) * max(0.0, min(1.0, f))))

    if dawn <= now_min < dawn + ramp:            # brightening
        return lerp(night, day, (now_min - dawn) / ramp)
    if dawn + ramp <= now_min < dusk:            # full day
        return day
    if dusk <= now_min < dusk + ramp:            # dimming
        return lerp(day, night, (now_min - dusk) / ramp)
    return night


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
        self._last_sent_at = 0.0
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

            # Drain first: responses queued from the PING exchange (or from
            # the board's own boot chatter) otherwise concatenate into one
            # unreadable line when readline stitches partial chunks together.
            self._ser.reset_input_buffer()
            self._ser.write(b"INFO\n")
            self._ser.flush()

            # Read until the config line appears rather than taking whatever
            # comes back first. A leftover PONG can still be in the buffer, and
            # logging that instead defeats the point of this line, which is to
            # record which firmware and mapping were live at connect time.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                line = self._ser.readline().decode(errors="replace").strip()
                if not line:
                    continue
                if "canvas=" in line:
                    # Truncated: a garbled or doubled response should not put
                    # a thousand-character line in the service log.
                    log.info("Board: %s", line[:140])
                    break
            return True
        except (serial.SerialException, OSError):
            return False

    def send(self, cmd: str, force: bool = False) -> bool:
        """Send a command; skip if unchanged, unless forced or gone stale."""
        now = time.monotonic()
        stale = (now - self._last_sent_at) >= RESEND_INTERVAL

        if cmd == self._last_sent and not force and not stale:
            return True

        if self._ser is None and not self._open():
            return False

        try:
            self._ser.write((cmd + "\n").encode())   # type: ignore[union-attr]
            self._ser.flush()                        # type: ignore[union-attr]
            self._last_sent = cmd
            self._last_sent_at = now
            log.debug("sent %s", cmd)
            return True
        except (serial.SerialException, OSError) as exc:
            log.warning("Write failed (%s) — will reconnect", exc)
            self.close()
            return False

    def set_brightness(self, value: int) -> bool:
        """Apply a brightness level and remember it across reconnects.

        Storing it rather than only writing it is the point: a board that
        browns out and resets at 02:00 comes back at the firmware default,
        which is full brightness. _open() reapplies whatever was last set, so
        the sign returns to its night level instead of blazing until dawn.

        BRIGHT: bypasses send()'s dedup deliberately — that cache tracks the
        STATE line, and letting a brightness write reset it would suppress the
        next state resend and trip the board's 15s NO DATA timeout.
        """
        value = max(0, min(255, int(value)))
        self._brightness = value
        if self._ser is None:
            return False
        try:
            self._ser.write(f"BRIGHT:{value}\n".encode())
            self._ser.flush()
            return True
        except (serial.SerialException, OSError) as exc:
            log.warning("Brightness write failed (%s) — will reconnect", exc)
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
        self._last_sent_at = 0.0
        if self._ser:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None
        self._last_sent = None


def road_state(api: str, session: requests.Session, camera_id: str | None) -> str:
    """Ask Node what THIS approach should be showing.

    Each camera has its own sign directly beneath it, both facing outward from
    the blind curve, so state is per-approach rather than global:

      * An incident on this approach shows here and only here — the driver in
        front of this sign is the one it concerns.
      * A vehicle on BOTH approaches shows on both signs. That is the case
        neither driver can see around the curve, and the reason for the system.

    The server does the deciding and returns a ready `signs` map; this reads
    its own entry. Recomputing the rule here would let the sign and the public
    web page drift apart, which is exactly what putting the logic server-side
    avoids.
    """
    r = session.get(f"{api}/api/public/status", timeout=5)
    r.raise_for_status()
    data = r.json()
    if not data.get("success"):
        raise RuntimeError("status endpoint returned success=false")

    payload = data["data"]

    # Transient display-mode override, set server-side and self-expiring.
    mode = payload.get("sign_mode")
    if mode:
        return f"@{mode}"

    # Per-approach state when this bridge knows which camera it sits under.
    signs = payload.get("signs") or {}
    if camera_id and camera_id in signs:
        return signs[camera_id].get("state", "clear")

    # Fall back to the overall state — an unconfigured bridge still warns
    # rather than sitting silent, which is the safe direction to fail.
    return {
        "incident": "incident",
        "vehicle_incoming": "vehicle",
        "clear": "clear",
    }.get(payload.get("state"), "clear")


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
                    help="Pin a FIXED brightness 0-255 and disable the "
                         "day/night schedule entirely")
    ap.add_argument("--day-brightness", type=int, default=DAY_BRIGHT_DEFAULT,
                    help=f"Daytime level (default {DAY_BRIGHT_DEFAULT})")
    ap.add_argument("--night-brightness", type=int,
                    default=NIGHT_BRIGHT_DEFAULT,
                    help=f"Night level (default {NIGHT_BRIGHT_DEFAULT}). Kept "
                         "above zero on purpose: this is a safety sign.")
    ap.add_argument("--dawn", default=DAWN_DEFAULT,
                    help=f"HH:MM brightening starts (default {DAWN_DEFAULT})")
    ap.add_argument("--dusk", default=DUSK_DEFAULT,
                    help=f"HH:MM dimming starts (default {DUSK_DEFAULT})")
    ap.add_argument("--ramp-minutes", type=int, default=RAMP_MINUTES_DEFAULT,
                    help=f"Fade length in minutes, 0 to switch instantly "
                         f"(default {RAMP_MINUTES_DEFAULT})")
    ap.add_argument("--test", action="store_true",
                    help="Cycle all screens and exit — no server needed")
    ap.add_argument("--camera-id", default=None,
                    help="This sign's camera id (e.g. CAM-A-001). Selects the "
                         "per-approach state; without it the bridge follows "
                         "the overall road state instead.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [led-sign] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        dawn = parse_hhmm(args.dawn)
        dusk = parse_hhmm(args.dusk)
    except ValueError as exc:
        log.error("Bad --dawn/--dusk: %s", exc)
        return 2
    if not dawn < dusk:
        log.error("--dawn (%s) must be earlier in the day than --dusk (%s)",
                  args.dawn, args.dusk)
        return 2

    scheduled = args.brightness is None
    link = EspLink(args.port, args.brightness)

    if args.test:
        return run_test(link)

    session = requests.Session()
    log.info("Bridging %s -> LED sign (approach: %s)", args.api,
             args.camera_id or "overall state")

    last_state = None
    last_change = 0.0
    consecutive_errors = 0
    last_health = 0.0

    # How often to prove the board is still answering. 30s is a compromise:
    # frequent enough that a wedged sign is caught within a minute, rare
    # enough that the PING traffic is negligible.
    HEALTH_INTERVAL = 30.0

    applied_bright: int | None = None
    clock_ok = False

    if scheduled:
        log.info("Brightness schedule: %s day=%d -> %s night=%d (ramp %dm)",
                 args.dawn, args.day_brightness, args.dusk,
                 args.night_brightness, args.ramp_minutes)
    else:
        log.info("Brightness pinned at %d (schedule disabled)", args.brightness)

    while True:
        try:
            # Brightness first, and deliberately OUTSIDE the API call below:
            # nightfall is not conditional on the server being reachable. A Pi
            # that has lost the network should still dim at dusk rather than
            # sit at full glare until someone notices.
            if scheduled:
                if not clock_ok:
                    clock_ok = _clock_is_trustworthy()
                    if not clock_ok:
                        # Fail toward DAY. An unreadable sign is a worse
                        # failure than a bright one: legibility is the whole
                        # function. This self-corrects within one poll of NTP
                        # landing, which is seconds after boot.
                        if applied_bright != args.day_brightness:
                            log.warning("Clock not yet synchronised — holding "
                                        "day brightness until it is")
                            link.set_brightness(args.day_brightness)
                            applied_bright = args.day_brightness

                if clock_ok:
                    t = datetime.datetime.now()
                    want = scheduled_brightness(
                        t.hour * 60 + t.minute, dawn, dusk,
                        args.night_brightness, args.day_brightness,
                        args.ramp_minutes)

                    # Apply on a meaningful move, or whenever the value has
                    # settled exactly on a plateau — otherwise the last ramp
                    # step stops up to EPSILON short and the sign sits a
                    # fraction off its true day or night level all day.
                    settled = want in (args.day_brightness,
                                       args.night_brightness)
                    moved = (applied_bright is None
                             or abs(want - applied_bright) >= BRIGHT_EPSILON)

                    if (moved or (settled and want != applied_bright)) \
                            and link.set_brightness(want):
                        log.info("brightness -> %d (%s)", want,
                                 t.strftime("%H:%M"))
                        applied_bright = want

            state = road_state(args.api, session, args.camera_id)
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
