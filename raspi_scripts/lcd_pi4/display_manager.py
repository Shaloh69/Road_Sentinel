#!/usr/bin/env python3
"""
Road Sentinel — HUB75 128×32 RGB LED Matrix Display Manager
Raspberry Pi 4 Model B version — drives hzeller/rpi-rgb-led-matrix via ledcat subprocess

Three road-warning screens (full-panel, large text):
  SLOW DOWN     — red,    default/idle (drivers are always prompted while AI processes)
  VEHICLE INCOMING — orange, flashing when a vehicle detection arrives
  INCIDENT AHEAD   — magenta, flashing when an active incident is detected

How frames travel:
  Python PIL → raw RGB24 bytes → ledcat stdin → hzeller C lib → HUB75 matrix
  This bypasses the Python bindings chain_length bug that caused panel mirroring.

Run:
  sudo python3 display_manager.py           # real mode (polls Node API)
  sudo python3 display_manager.py --test    # test mode (fake data, no API)
  (sudo required for direct GPIO /dev/mem access)
"""

import argparse
import logging
import os
import socket
import subprocess
import time
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

# ── ledcat binary ─────────────────────────────────────────────────────────────
LEDCAT_DEFAULT = os.path.expanduser("~/rpi-rgb-led-matrix/examples-api-use/ledcat")

def _find_ledcat(override: Optional[str] = None) -> str:
    path = os.path.expanduser(override or LEDCAT_DEFAULT)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"ledcat not found at {path}\n"
            "Build with:\n"
            "  cd ~/rpi-rgb-led-matrix/examples-api-use && make\n"
            "Or run install.sh"
        )
    return path

try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [LED] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Display dimensions ────────────────────────────────────────────────────────
WIDTH  = 128
HEIGHT = 32

# ── Color palette ─────────────────────────────────────────────────────────────
WHITE   = (255, 255, 255)
BLACK   = (0,   0,   0  )
RED     = (220, 0,   0  )
GREEN   = (0,   200, 0  )
YELLOW  = (220, 200, 0  )
ORANGE  = (255, 110, 0  )
MAGENTA = (200, 0,   180)
CYAN    = (0,   200, 200)
GRAY    = (90,  90,  90 )
AMBER   = (255, 160, 0  )

# ── Fonts (Pillow 10+ bitmap fonts) ──────────────────────────────────────────
FONT_SM    = ImageFont.load_default(size=4)   # status rows
FONT_XS6   = ImageFont.load_default(size=6)   # subtitle strip
FONT_MED   = ImageFont.load_default(size=10)  # secondary line
FONT_LARGE = ImageFont.load_default(size=16)  # main message

# ── Frame helpers ─────────────────────────────────────────────────────────────

def _new_frame() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img  = Image.new("RGB", (WIDTH, HEIGHT), BLACK)
    draw = ImageDraw.Draw(img)
    return img, draw

def _fill_row_px(draw: ImageDraw.ImageDraw, y0: int, y1: int, color):
    draw.rectangle([0, y0, WIDTH - 1, y1], fill=color)

def _draw_centered_in(draw: ImageDraw.ImageDraw,
                      x0: int, y0: int, x1: int, y1: int,
                      text: str, font, color):
    """Draw text centered horizontally and vertically within the given rectangle."""
    bbox = draw.textbbox((0, 0), text, font=font)
    w    = bbox[2] - bbox[0]
    h    = bbox[3] - bbox[1]
    x    = x0 + max(0, ((x1 - x0 + 1) - w) // 2) - bbox[0]
    y    = y0 + max(0, ((y1 - y0 + 1) - h) // 2) - bbox[1]
    draw.text((x, y), text, font=font, fill=color)

def _draw_two_lines_centered(draw: ImageDraw.ImageDraw,
                              y0: int, y1: int,
                              line1: str, font1,
                              line2: str, font2,
                              color):
    """Draw two lines of text centered as a group vertically within y0–y1."""
    bb1 = draw.textbbox((0, 0), line1, font=font1)
    bb2 = draw.textbbox((0, 0), line2, font=font2)
    h1, h2 = bb1[3] - bb1[1], bb2[3] - bb2[1]
    gap   = 2
    total = h1 + gap + h2
    sy    = y0 + max(0, ((y1 - y0 + 1) - total) // 2)
    x1p   = max(0, (WIDTH - (bb1[2] - bb1[0])) // 2) - bb1[0]
    x2p   = max(0, (WIDTH - (bb2[2] - bb2[0])) // 2) - bb2[0]
    draw.text((x1p, sy - bb1[1]),           line1, font=font1, fill=color)
    draw.text((x2p, sy + h1 + gap - bb2[1]), line2, font=font2, fill=color)


# ── Road warning renderers ─────────────────────────────────────────────────────

def render_slow_down(_state, flash_phase: int = 0) -> Image.Image:
    """Default screen — red background, 'SLOW DOWN' centered."""
    img, draw = _new_frame()
    bg = RED if flash_phase == 0 else (140, 0, 0)
    _fill_row_px(draw, 0, HEIGHT - 1, bg)
    _draw_centered_in(draw, 0, 0, WIDTH - 1, HEIGHT - 1, "SLOW DOWN", FONT_LARGE, WHITE)
    return img


def render_vehicle_incoming(_state, flash_phase: int = 0) -> Image.Image:
    """Vehicle detected — orange flashing, two-line 'VEHICLE / INCOMING'."""
    img, draw = _new_frame()
    bg = ORANGE if flash_phase == 0 else (160, 65, 0)
    _fill_row_px(draw, 0, HEIGHT - 1, bg)
    # Main two-line message above the subtitle strip
    _draw_two_lines_centered(draw, 0, HEIGHT - 8,
                             "VEHICLE",  FONT_LARGE,
                             "INCOMING", FONT_MED,
                             WHITE)
    # Subtitle strip at the bottom
    _draw_centered_in(draw, 0, HEIGHT - 7, WIDTH - 1, HEIGHT - 1,
                      "SLOW DOWN", FONT_XS6, YELLOW)
    return img


def render_incident_ahead(_state, flash_phase: int = 0) -> Image.Image:
    """Incident detected — magenta flashing, two-line 'INCIDENT / AHEAD'."""
    img, draw = _new_frame()
    bg = MAGENTA if flash_phase == 0 else (120, 0, 110)
    _fill_row_px(draw, 0, HEIGHT - 1, bg)
    _draw_two_lines_centered(draw, 0, HEIGHT - 8,
                             "INCIDENT", FONT_LARGE,
                             "AHEAD",    FONT_MED,
                             WHITE)
    _draw_centered_in(draw, 0, HEIGHT - 7, WIDTH - 1, HEIGHT - 1,
                      "SLOW DOWN", FONT_XS6, YELLOW)
    return img


def render_color_bar_test(phase: int) -> Image.Image:
    """Startup color-bar test — cycles through brightness levels."""
    img, draw = _new_frame()
    bars  = [RED, ORANGE, YELLOW, GREEN, CYAN, (30, 80, 255), WHITE]
    bar_w = WIDTH // len(bars)
    intensity = [255, 180, 80][min(phase // 2, 2)]
    for i, color in enumerate(bars):
        x0 = i * bar_w
        x1 = x0 + bar_w - 1
        scaled = tuple(int(c * intensity // 255) for c in color)
        draw.rectangle([x0, 0, x1, HEIGHT - 1], fill=scaled)
    return img


# ── System state ──────────────────────────────────────────────────────────────

class SystemState:
    def __init__(self):
        self._lock                    = threading.Lock()
        self.vehicles_today: int      = 0
        self.avg_speed: Optional[int] = None
        self.incidents_today: int     = 0
        self.cameras_online: int      = 0
        self.cameras_total: int       = 2
        self.cameras: list            = []
        self.last_poll_ok: bool       = False
        self.is_test_mode: bool       = False
        self.start_time: datetime     = datetime.now()
        # Alert expiry times
        self._vehicle_alert_exp: float  = 0.0
        self._incident_alert_exp: float = 0.0
        self._incident: Optional[dict]  = None

    def update_summary(self, data: dict):
        with self._lock:
            self.vehicles_today  = data.get("vehicles_today",  self.vehicles_today)
            self.avg_speed       = data.get("average_speed",   self.avg_speed)
            self.incidents_today = data.get("incidents_today", self.incidents_today)
            self.cameras_online  = data.get("cameras_online",  self.cameras_online)
            self.cameras_total   = data.get("cameras_total",   self.cameras_total)
            self.last_poll_ok    = True

    def update_cameras(self, cameras: list):
        with self._lock:
            self.cameras = cameras

    def push_vehicle_alert(self, hold_secs: float = 8.0):
        with self._lock:
            self._vehicle_alert_exp = time.monotonic() + hold_secs

    def has_vehicle_alert(self) -> bool:
        with self._lock:
            return time.monotonic() < self._vehicle_alert_exp

    def push_incident_alert(self, incident: dict, hold_secs: float = 12.0):
        with self._lock:
            self._incident          = incident
            self._incident_alert_exp = time.monotonic() + hold_secs

    def pop_incident_alert(self) -> Optional[dict]:
        with self._lock:
            if self._incident and time.monotonic() < self._incident_alert_exp:
                return self._incident
            self._incident = None
            return None

    def uptime(self) -> str:
        delta = datetime.now() - self.start_time
        h, rem = divmod(int(delta.total_seconds()), 3600)
        m = rem // 60
        return f"{h}h{m:02d}m"


# ── Data providers ────────────────────────────────────────────────────────────

class DataProvider(ABC):
    def __init__(self, state: SystemState):
        self.state = state

    @abstractmethod
    def start(self): ...

    def trigger_test_incident(self):
        pass

    def trigger_test_vehicle(self):
        pass


class ApiDataProvider(DataProvider):
    """
    Two polling threads:
      - Summary thread (every 30s): vehicles, speed, camera status.
        Triggers a vehicle alert if vehicles_today increased.
      - Incident thread (every 2s): latest active incident.
        Triggers an incident alert immediately on new incident.
    """

    POLL_INTERVAL          = 30   # summary + cameras
    INCIDENT_POLL_INTERVAL = 2    # incidents

    def __init__(self, state: SystemState, base_url: str):
        super().__init__(state)
        self._base              = base_url.rstrip("/")
        self._last_incident_id: Optional[int] = None
        self._first_poll        = True

    def start(self):
        threading.Thread(target=self._summary_loop,  daemon=True, name="led-summary").start()
        threading.Thread(target=self._incident_loop, daemon=True, name="led-incidents").start()
        log.info("API data provider started — %s", self._base)

    def _summary_loop(self):
        while True:
            try:
                self._poll_summary()
            except Exception as exc:
                log.warning("Summary poll error: %s", exc)
                self.state.last_poll_ok = False
            time.sleep(self.POLL_INTERVAL)

    def _incident_loop(self):
        while True:
            try:
                self._poll_incidents()
            except Exception as exc:
                log.debug("Incident poll error: %s", exc)
            time.sleep(self.INCIDENT_POLL_INTERVAL)

    def _poll_summary(self):
        if not _REQUESTS_OK:
            raise RuntimeError("requests library not installed")

        r = _requests.get(f"{self._base}/api/analytics/summary", timeout=5)
        r.raise_for_status()
        data = r.json().get("data", {})
        prev = self.state.vehicles_today
        self.state.update_summary(data)

        # Trigger vehicle alert when new vehicles detected (skip first poll to avoid false burst)
        if not self._first_poll and data.get("vehicles_today", 0) > prev:
            self.state.push_vehicle_alert(hold_secs=8.0)
        self._first_poll = False

        r = _requests.get(f"{self._base}/api/cameras", timeout=5)
        r.raise_for_status()
        self.state.update_cameras(r.json().get("data", []))

    def _poll_incidents(self):
        if not _REQUESTS_OK:
            return
        r = _requests.get(
            f"{self._base}/api/incidents",
            params={"status": "active", "limit": 1},
            timeout=3,
        )
        items = r.json().get("data", [])
        if items:
            inc    = items[0]
            inc_id = inc.get("id")
            if inc_id != self._last_incident_id:
                self._last_incident_id = inc_id
                self.state.push_incident_alert(inc, hold_secs=12.0)
                log.warning("INCIDENT: %s (%s)", inc.get("incident_type"), inc.get("severity"))


class TestDataProvider(DataProvider):
    """Fake data — no network. Cycles simulated alerts for local testing."""

    _FAKE_INCIDENTS = [
        {"incident_type": "speeding",   "severity": "high",
         "title": "Test Speeding",    "camera_name": "Camera A"},
        {"incident_type": "crash",      "severity": "critical",
         "title": "Test Crash",       "camera_name": "Camera B"},
        {"incident_type": "congestion", "severity": "medium",
         "title": "Test Congestion",  "camera_name": "Camera A"},
    ]

    def __init__(self, state: SystemState):
        super().__init__(state)
        self._idx = 0

    def start(self):
        self.state.update_summary({
            "vehicles_today": 0, "average_speed": None,
            "incidents_today": 0, "cameras_online": 2, "cameras_total": 2,
        })
        self.state.update_cameras([
            {"id": "CAM-A-001", "name": "Camera A", "status": "online", "fps": 30, "resolution": "1920x1080"},
            {"id": "CAM-B-002", "name": "Camera B", "status": "online", "fps": 30, "resolution": "1920x1080"},
        ])
        self.state.last_poll_ok = True
        self.state.is_test_mode = True
        threading.Thread(target=self._cycle, daemon=True).start()
        log.info("Test data provider started (no API calls)")

    def trigger_test_incident(self):
        inc = dict(self._FAKE_INCIDENTS[self._idx % len(self._FAKE_INCIDENTS)])
        self.state.push_incident_alert(inc, hold_secs=10.0)
        self._idx += 1
        log.info("Test incident triggered: %s", inc["incident_type"])

    def trigger_test_vehicle(self):
        self.state.push_vehicle_alert(hold_secs=8.0)
        log.info("Test vehicle alert triggered")

    def _cycle(self):
        """Auto-cycle: vehicle alert every 15s, incident alert every 30s."""
        count = 0
        while True:
            time.sleep(15)
            count += 1
            if count % 2 == 0:
                self.trigger_test_incident()
            else:
                self.trigger_test_vehicle()


# ── ledcat subprocess backend ─────────────────────────────────────────────────
FRAME_BYTES  = WIDTH * HEIGHT * 3   # 12,288 for 128×32
_BLACK_FRAME = bytes(FRAME_BYTES)


def start_ledcat(
    ledcat_path: str,
    gpio_slowdown: int     = 4,
    hardware_mapping: str  = "regular",
    no_hardware_pulse: bool = True,
    cols_per_panel: int    = 64,
    chain_length: int      = 0,
    multiplexing: int      = 0,
    scan_mode: int         = 0,
) -> subprocess.Popen:
    chain = chain_length if chain_length > 0 else (WIDTH // cols_per_panel)
    cmd = [
        ledcat_path,
        f"--led-rows={HEIGHT}",
        f"--led-cols={cols_per_panel}",
        f"--led-chain={chain}",
        "--led-parallel=1",
        f"--led-gpio-mapping={hardware_mapping}",
        f"--led-slowdown-gpio={gpio_slowdown}",
        "--led-no-drop-privs",
    ]
    if no_hardware_pulse:
        cmd.append("--led-no-hardware-pulse")
    if multiplexing > 0:
        cmd.append(f"--led-multiplexing={multiplexing}")
    if scan_mode != 0:
        cmd.append(f"--led-scan-mode={scan_mode}")

    log.info("Starting ledcat: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    log.info("ledcat PID=%d  frame=%d bytes (%dx%d RGB24)", proc.pid, FRAME_BYTES, WIDTH, HEIGHT)
    return proc


def show_frame(proc: subprocess.Popen, img: Image.Image) -> None:
    raw = img.convert("RGB").tobytes()
    proc.stdin.write(raw)
    proc.stdin.flush()


def clear_display(proc: subprocess.Popen) -> None:
    proc.stdin.write(_BLACK_FRAME)
    proc.stdin.flush()


# ── Main display loop ─────────────────────────────────────────────────────────

TICK = 0.25   # 4 fps — enough for smooth flashing


def run(proc: subprocess.Popen, state: SystemState):
    flash_tick = 0
    log.info("Display loop started (%dx%d) via ledcat PID=%d", WIDTH, HEIGHT, proc.pid)

    # Startup color-bar test (3s — confirms all RGB channels and both panels)
    for phase in range(6):
        show_frame(proc, render_color_bar_test(phase))
        time.sleep(0.5)

    while True:
        flash_tick = (flash_tick + 1) % 4
        fp = flash_tick % 2   # 0 or 1, alternates at 2 fps

        incident = state.pop_incident_alert()
        if incident:
            img = render_incident_ahead(state, flash_phase=fp)
        elif state.has_vehicle_alert():
            img = render_vehicle_incoming(state, flash_phase=fp)
        else:
            img = render_slow_down(state, flash_phase=fp)

        show_frame(proc, img)
        time.sleep(TICK)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Road Sentinel HUB75 128×32 LED Matrix — Raspberry Pi 4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Display states:
  SLOW DOWN      (red)     — default, always shown while AI processes frames
  VEHICLE INCOMING (orange) — shown for 8s when a new vehicle detection arrives
  INCIDENT AHEAD  (magenta) — shown for 12s when an active incident is detected

Examples:
  sudo python3 display_manager.py                          # real mode
  sudo python3 display_manager.py --test                   # test mode (cycles alerts)
  sudo python3 display_manager.py --api http://192.168.8.50:3001
  sudo python3 display_manager.py --slowdown 3             # if display is garbled
        """,
    )
    parser.add_argument("--test",           action="store_true",
                        help="Test mode: cycles fake alerts, no API calls")
    parser.add_argument("--api",            default="http://localhost:3001",
                        help="Node Service base URL (default: http://localhost:3001)")
    parser.add_argument("--slowdown",       type=int, default=4,
                        help="GPIO slowdown for Pi 4 (default: 4, try 3–5 if garbled)")
    parser.add_argument("--cols",           type=int, default=64,
                        help="Physical columns per panel (default: 64)")
    parser.add_argument("--chain",          type=int, default=0,
                        help="Number of chained panels (default: auto = 128 / cols)")
    parser.add_argument("--multiplexing",   type=int, default=0,
                        help="Panel multiplexing type (default: 0 = standard)")
    parser.add_argument("--scan-mode",      type=int, default=0, choices=[0, 1],
                        help="Row scan mode: 0=progressive (default), 1=interlaced")
    parser.add_argument("--mapping",        default="regular",
                        choices=["regular", "adafruit-hat", "adafruit-hat-pwm"],
                        help="GPIO mapping (default: regular)")
    parser.add_argument("--hardware-pulse", action="store_true", default=False,
                        help="Enable hardware PWM pulse (only after disabling snd_bcm2835)")
    parser.add_argument("--ledcat",         default=None,
                        help=f"Path to ledcat binary (default: {LEDCAT_DEFAULT})")
    parser.add_argument("--trigger-incident", action="store_true",
                        help="Fire one incident alert immediately, then continue")
    parser.add_argument("--trigger-vehicle",  action="store_true",
                        help="Fire one vehicle alert immediately, then continue")

    args = parser.parse_args()

    try:
        ledcat_path = _find_ledcat(args.ledcat)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    log.info("Road Sentinel LED Matrix — Pi 4 ledcat mode")
    log.info("  ledcat  : %s", ledcat_path)
    log.info("  mapping : %s  slowdown: %d", args.mapping, args.slowdown)
    log.info("  mode    : %s", "TEST" if args.test else "REAL")

    proc = start_ledcat(
        ledcat_path       = ledcat_path,
        gpio_slowdown     = args.slowdown,
        hardware_mapping  = args.mapping,
        no_hardware_pulse = not args.hardware_pulse,
        cols_per_panel    = args.cols,
        chain_length      = args.chain,
        multiplexing      = args.multiplexing,
        scan_mode         = args.scan_mode,
    )

    state    = SystemState()
    provider: DataProvider = (
        TestDataProvider(state) if args.test
        else ApiDataProvider(state, base_url=args.api)
    )
    provider.start()

    if args.trigger_incident:
        time.sleep(0.5)
        provider.trigger_test_incident()
    if args.trigger_vehicle:
        time.sleep(0.5)
        provider.trigger_test_vehicle()

    try:
        run(proc, state)
    except KeyboardInterrupt:
        log.info("Stopped by user")
    finally:
        try:
            clear_display(proc)
            proc.stdin.close()
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
        log.info("Display cleared")


if __name__ == "__main__":
    main()
