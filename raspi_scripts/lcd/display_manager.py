#!/usr/bin/env python3
"""
Road Sentinel — HUB75 128×32 RGB LED Matrix Display Manager
Raspberry Pi 5 version — uses Adafruit PioMatter (RP1 PIO-based driver)

NOTE: The old hzeller/rpi-rgb-led-matrix library does NOT work on Pi 5.
      Pi 5 uses the RP1 peripheral chip for GPIO — direct Broadcom GPIO is gone.
      PioMatter uses the PIO blocks inside RP1 (same concept as RP2040) instead.

Hardware: HUB75 RGB LED matrix — 128×32 pixels (full color)
  Default config: two 64×32 panels chained  → total 128×32
  Single panel:   one 128×32 panel

Wiring: Default pinout is 'active3' — matches the cheap ₱149 Chinese HUB75
  adapter board (hzeller "regular" GPIO mapping).
  For Adafruit RGB Matrix Bonnet use --pinout bonnet.
  See lcd/README.md for full wiring details.

Install:
  pip install Adafruit-Blinka-Raspberry-Pi5-Piomatter pillow numpy requests

Run:
  python3 display_manager.py           # real mode (live API)
  python3 display_manager.py --test    # test mode (fake data, no API)
"""

import argparse
import logging
import socket
import time
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

# ── Try to import PioMatter (only on Raspberry Pi 5) ─────────────────────────
try:
    import adafruit_blinka_raspberry_pi5_piomatter as piomatter
    HW_AVAILABLE = True
except ImportError:
    HW_AVAILABLE = False
    print("WARNING: PioMatter not found.")
    print("Install: pip install Adafruit-Blinka-Raspberry-Pi5-Piomatter")
    print("Or see lcd/README.md for full setup steps.")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [LED] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Display dimensions ────────────────────────────────────────────────────────
WIDTH  = 128
HEIGHT = 32

# ── Color palette (RGB tuples) ────────────────────────────────────────────────
WHITE   = (255, 255, 255)
BLACK   = (0,   0,   0  )
RED     = (220, 0,   0  )
GREEN   = (0,   200, 0  )
BLUE    = (30,  80,  255)
YELLOW  = (220, 200, 0  )
ORANGE  = (255, 110, 0  )
CYAN    = (0,   200, 200)
GRAY    = (90,  90,  90 )
AMBER   = (255, 160, 0  )   # TEST alert color
DARK_GREEN = (0, 80, 0  )

SEVERITY_COLORS = {
    "critical": RED,
    "high":     ORANGE,
    "medium":   YELLOW,
    "low":      CYAN,
}

# ── Font & layout ─────────────────────────────────────────────────────────────
# 4px font  →  6 rows on a 32px panel  (4px text + 1px gap = 5px per row)
ROW_H = 5
ROWS  = [0, 5, 10, 15, 20, 25]

FONT_SM = ImageFont.load_default(size=4)

def _trunc(text: str, max_chars: int = 42) -> str:
    return text if len(text) <= max_chars else text[:max_chars - 1] + "."

# ── Frame helpers ─────────────────────────────────────────────────────────────

def _new_frame() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img  = Image.new("RGB", (WIDTH, HEIGHT), BLACK)
    draw = ImageDraw.Draw(img)
    return img, draw

def _draw_row(draw: ImageDraw.ImageDraw, row: int, text: str,
              color=WHITE, x: int = 1):
    draw.text((x, ROWS[row]), text, font=FONT_SM, fill=color)

def _fill_row(draw: ImageDraw.ImageDraw, row: int, color):
    draw.rectangle([0, ROWS[row], WIDTH - 1, ROWS[row] + ROW_H - 1], fill=color)


# ── System state ──────────────────────────────────────────────────────────────

class SystemState:
    def __init__(self):
        self._lock              = threading.Lock()
        self.vehicles_today: int       = 0
        self.avg_speed: Optional[int]  = None
        self.incidents_today: int      = 0
        self.cameras_online: int       = 0
        self.cameras_total: int        = 2
        self.cameras: list             = []
        self._alert: Optional[dict]    = None
        self._alert_expires: float     = 0.0
        self.local_ip: str             = _get_local_ip()
        self.start_time: datetime      = datetime.now()
        self.last_poll_ok: bool        = False
        self.is_test_mode: bool        = False

    def update_summary(self, data: dict):
        with self._lock:
            self.vehicles_today  = data.get("vehicles_today", 0)
            self.avg_speed       = data.get("average_speed")
            self.incidents_today = data.get("incidents_today", 0)
            self.cameras_online  = data.get("cameras_online", 0)
            self.cameras_total   = data.get("cameras_total", 2)
            self.last_poll_ok    = True

    def update_cameras(self, cameras: list):
        with self._lock:
            self.cameras = cameras

    def push_alert(self, incident: dict, hold_secs: float = 12.0):
        with self._lock:
            self._alert         = incident
            self._alert_expires = time.monotonic() + hold_secs

    def pop_alert(self) -> Optional[dict]:
        with self._lock:
            if self._alert and time.monotonic() < self._alert_expires:
                return self._alert
            self._alert = None
            return None

    def uptime(self) -> str:
        delta = datetime.now() - self.start_time
        h, rem = divmod(int(delta.total_seconds()), 3600)
        m = rem // 60
        return f"{h}h{m:02d}m"

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "vehicles_today":  self.vehicles_today,
                "avg_speed":       self.avg_speed,
                "incidents_today": self.incidents_today,
                "cameras_online":  self.cameras_online,
                "cameras_total":   self.cameras_total,
                "cameras":         list(self.cameras),
                "last_poll_ok":    self.last_poll_ok,
                "is_test_mode":    self.is_test_mode,
                "local_ip":        self.local_ip,
            }


def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "?.?.?.?"


# ── Data providers ────────────────────────────────────────────────────────────

class DataProvider(ABC):
    def __init__(self, state: SystemState):
        self.state = state

    @abstractmethod
    def start(self): ...

    @abstractmethod
    def trigger_test_alert(self): ...


class ApiDataProvider(DataProvider):
    """Polls the Node Service REST API. New incidents → REAL alerts."""

    POLL_INTERVAL = 5

    def __init__(self, state: SystemState, base_url: str):
        super().__init__(state)
        self.base_url                   = base_url.rstrip("/")
        self._seen_incident_ids: set    = set()

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()
        log.info("API poller started → %s (every %ds)", self.base_url, self.POLL_INTERVAL)

    def trigger_test_alert(self):
        self.state.push_alert({
            "is_test": True, "incident_type": "speeding", "severity": "high",
            "title": "Test Alert", "description": "Manual test from CLI",
            "camera_name": "Camera A", "timestamp": datetime.now().isoformat(),
        })

    def _loop(self):
        while True:
            try:
                self._poll()
            except Exception as e:
                log.warning("Poll error: %s", e)
                self.state.last_poll_ok = False
            time.sleep(self.POLL_INTERVAL)

    def _poll(self):
        r = requests.get(f"{self.base_url}/api/analytics/summary", timeout=4)
        if r.ok:
            self.state.update_summary(r.json().get("data", {}))

        r = requests.get(f"{self.base_url}/api/cameras", timeout=4)
        if r.ok:
            self.state.update_cameras(r.json().get("data", []))

        r = requests.get(f"{self.base_url}/api/incidents",
                         params={"status": "active", "limit": "5"}, timeout=4)
        if r.ok:
            for inc in r.json().get("data", []):
                inc_id = inc.get("id")
                if inc_id not in self._seen_incident_ids:
                    self._seen_incident_ids.add(inc_id)
                    inc["is_test"] = False
                    self.state.push_alert(inc, hold_secs=12.0)
                    log.info("REAL alert: %s (%s)", inc.get("incident_type"), inc.get("severity"))


class TestDataProvider(DataProvider):
    """Fake data — no network. Cycles simulated TEST alerts (amber)."""

    _FAKE_INCIDENTS = [
        {"is_test": True, "incident_type": "speeding",  "severity": "high",
         "title": "Test Speeding",   "description": "85 km/h on Camera A",
         "camera_name": "Camera A",  "timestamp": ""},
        {"is_test": True, "incident_type": "crash",     "severity": "critical",
         "title": "Test Crash",      "description": "Simulated collision",
         "camera_name": "Camera B",  "timestamp": ""},
        {"is_test": True, "incident_type": "congestion","severity": "low",
         "title": "Test Congestion", "description": "Slow traffic detected",
         "camera_name": "Camera A",  "timestamp": ""},
    ]

    def __init__(self, state: SystemState):
        super().__init__(state)
        self._alert_idx = 0

    def start(self):
        self.state.update_summary({"vehicles_today": 999, "average_speed": 45,
                                   "incidents_today": 3,  "cameras_online": 2,
                                   "cameras_total":   2})
        self.state.update_cameras([
            {"id": "CAM-A-001", "name": "Camera A", "status": "online",
             "rtsp_url": "rtsp://192.168.8.104:554/cam/realmonitor",
             "fps": 30, "resolution": "1920x1080"},
            {"id": "CAM-B-002", "name": "Camera B", "status": "online",
             "rtsp_url": "rtsp://192.168.8.108:554/cam/realmonitor",
             "fps": 30, "resolution": "1920x1080"},
        ])
        self.state.last_poll_ok = True
        self.state.is_test_mode = True
        threading.Thread(target=self._cycle_alerts, daemon=True).start()
        log.info("Test data provider started — no API calls")

    def trigger_test_alert(self):
        inc = dict(self._FAKE_INCIDENTS[self._alert_idx % len(self._FAKE_INCIDENTS)])
        inc["timestamp"] = datetime.now().isoformat()
        self.state.push_alert(inc, hold_secs=10.0)
        self._alert_idx += 1
        log.info("Manual test alert triggered")

    def _cycle_alerts(self):
        time.sleep(10)
        while True:
            inc = dict(self._FAKE_INCIDENTS[self._alert_idx % len(self._FAKE_INCIDENTS)])
            inc["timestamp"] = datetime.now().isoformat()
            self.state.push_alert(inc, hold_secs=8.0)
            self._alert_idx += 1
            time.sleep(20)


# ── Screen renderers — all return a PIL Image ─────────────────────────────────

def render_main_status(state: SystemState) -> Image.Image:
    img, draw = _new_frame()
    snap = state.snapshot()
    now  = datetime.now().strftime("%H:%M")
    _draw_row(draw, 0, _trunc(f"ROAD SENTINEL  {now}"), WHITE)

    cams = snap["cameras"]
    if cams:
        cam_a = next((c for c in cams if "A" in c.get("name","") or "001" in c.get("id","")), None)
        cam_b = next((c for c in cams if "B" in c.get("name","") or "002" in c.get("id","")), None)
        a_col = GREEN if cam_a and cam_a.get("status") == "online" else RED
        b_col = GREEN if cam_b and cam_b.get("status") == "online" else RED
        a_st  = "ONLINE" if (cam_a and cam_a.get("status") == "online") else "OFFLINE"
        b_st  = "ONLINE" if (cam_b and cam_b.get("status") == "online") else "OFFLINE"
        draw.text((1,  ROWS[1]), "A:", font=FONT_SM, fill=GRAY)
        draw.text((13, ROWS[1]), a_st, font=FONT_SM, fill=a_col)
        draw.text((65, ROWS[1]), "B:", font=FONT_SM, fill=GRAY)
        draw.text((77, ROWS[1]), b_st, font=FONT_SM, fill=b_col)
    else:
        online = snap["cameras_online"]
        col    = GREEN if online == snap["cameras_total"] else (ORANGE if online > 0 else RED)
        _draw_row(draw, 1, f"Cameras: {online}/{snap['cameras_total']}", col)

    speed_str = f"{snap['avg_speed']}km/h" if snap["avg_speed"] else "N/A"
    draw.text((1,  ROWS[2]), _trunc(f"Veh:{snap['vehicles_today']:,}"), font=FONT_SM, fill=CYAN)
    draw.text((70, ROWS[2]), speed_str, font=FONT_SM, fill=YELLOW)
    _draw_row(draw, 3, _trunc(f"{snap['local_ip']}  {state.uptime()}"), GRAY)

    if snap["is_test_mode"]:
        draw.text((98, 0), "[TST]", font=FONT_SM, fill=AMBER)
    return img


def render_camera_detail(state: SystemState) -> Image.Image:
    img, draw = _new_frame()
    _draw_row(draw, 0, "--- Camera Status ---", WHITE)
    cams = state.snapshot()["cameras"]
    if len(cams) >= 1:
        c   = cams[0]
        ip  = c.get("rtsp_url", "").split("//")[-1].split(":")[0] or "?"
        col = GREEN if c.get("status") == "online" else RED
        _draw_row(draw, 1, _trunc(f"{c.get('name','Cam A')} {ip}"), col)
        _draw_row(draw, 2, _trunc(f"{c.get('fps','?')}fps {c.get('resolution','?')}"), GRAY)
    else:
        _draw_row(draw, 1, "No camera data", ORANGE)
        _draw_row(draw, 2, "Check Node Service", GRAY)
    if len(cams) >= 2:
        c   = cams[1]
        col = GREEN if c.get("status") == "online" else RED
        _draw_row(draw, 3, _trunc(f"{c.get('name','Cam B')} {c.get('status','?').upper()[:2]}"), col)
    return img


def render_daily_stats(state: SystemState) -> Image.Image:
    img, draw = _new_frame()
    snap = state.snapshot()
    _draw_row(draw, 0, _trunc(f"Today: {snap['vehicles_today']:,} vehicles"), WHITE)
    _draw_row(draw, 1, _trunc(f"Incidents: {snap['incidents_today']}"), ORANGE)
    speed_str = f"{snap['avg_speed']} km/h" if snap["avg_speed"] else "N/A"
    _draw_row(draw, 2, _trunc(f"Avg speed: {speed_str}"), CYAN)
    _draw_row(draw, 3, _trunc(f"Up: {state.uptime()}  {snap['local_ip']}"), GRAY)
    return img


def render_alert(state: SystemState, alert: dict, flash_phase: int) -> Image.Image:
    """
    TEST alerts  → amber header, [TEST] label, "SIMULATED" subtitle
    REAL alerts  → severity-colored header (red/orange/yellow/cyan)
    """
    img, draw = _new_frame()
    is_test   = alert.get("is_test", False)
    inc_type  = alert.get("incident_type", "INCIDENT").upper()
    severity  = alert.get("severity", "high")
    cam_name  = alert.get("camera_name", alert.get("camera_id", ""))
    desc      = alert.get("description", "")
    ts_raw    = alert.get("timestamp", "")
    try:
        ts = datetime.fromisoformat(str(ts_raw).replace("Z", "")).strftime("%H:%M:%S")
    except Exception:
        ts = datetime.now().strftime("%H:%M:%S")

    if is_test:
        bg        = AMBER if flash_phase == 0 else (180, 110, 0)
        fg        = BLACK
        label_col = AMBER
        header    = _trunc(f"[TEST] {inc_type}  {cam_name}")
        label     = "SIMULATED"
    else:
        sev_color = SEVERITY_COLORS.get(severity, ORANGE)
        bg        = sev_color if flash_phase == 0 else tuple(max(0, c - 80) for c in sev_color)
        fg        = BLACK if severity in ("medium", "low") else WHITE
        label_col = sev_color
        header    = _trunc(f"!! {inc_type}  {cam_name}  !!")
        label     = severity.upper()

    _fill_row(draw, 0, bg)
    draw.text((1, ROWS[0]), header, font=FONT_SM, fill=fg)
    _draw_row(draw, 1, label,                label_col)
    _draw_row(draw, 2, _trunc(desc or "See dashboard"), GRAY)
    _draw_row(draw, 3, ts, GRAY)
    return img


def render_offline(state: SystemState) -> Image.Image:
    img, draw = _new_frame()
    _draw_row(draw, 0, "ROAD SENTINEL", WHITE)
    _draw_row(draw, 1, "API: OFFLINE", RED)
    _draw_row(draw, 2, _trunc(state.local_ip), GRAY)
    _draw_row(draw, 3, _trunc(f"Up: {state.uptime()}"), GRAY)
    return img


def render_test_static(state: SystemState) -> Image.Image:
    """Fixed test-mode status screen — no rotation, no sliding content."""
    img, draw = _new_frame()
    snap = state.snapshot()

    # Row 0: header with [TEST] badge
    draw.text((1, ROWS[0]), "ROAD SENTINEL", font=FONT_SM, fill=WHITE)
    draw.text((103, ROWS[0]), "[TST]", font=FONT_SM, fill=AMBER)

    # Row 1: camera status — static coloured labels
    cams = snap["cameras"]
    if cams:
        cam_a = next((c for c in cams if "A" in c.get("name", "")), cams[0] if cams else None)
        cam_b = next((c for c in cams if "B" in c.get("name", "")), cams[1] if len(cams) > 1 else None)
        a_col = GREEN if cam_a and cam_a.get("status") == "online" else RED
        b_col = GREEN if cam_b and cam_b.get("status") == "online" else RED
        a_st  = "ON" if cam_a and cam_a.get("status") == "online" else "OFF"
        b_st  = "ON" if cam_b and cam_b.get("status") == "online" else "OFF"
        draw.text((1,  ROWS[1]), "A:", font=FONT_SM, fill=GRAY)
        draw.text((13, ROWS[1]), a_st, font=FONT_SM, fill=a_col)
        draw.text((47, ROWS[1]), "B:", font=FONT_SM, fill=GRAY)
        draw.text((59, ROWS[1]), b_st, font=FONT_SM, fill=b_col)
        draw.text((88, ROWS[1]), "SIMULATED", font=FONT_SM, fill=AMBER)
    else:
        _draw_row(draw, 1, "CAMERAS: SIMULATED", AMBER)

    # Row 2: static vehicle/speed counts
    speed_str = f"{snap['avg_speed']}km/h" if snap["avg_speed"] else "N/A"
    draw.text((1,  ROWS[2]), f"Veh:{snap['vehicles_today']:,}", font=FONT_SM, fill=CYAN)
    draw.text((70, ROWS[2]), speed_str, font=FONT_SM, fill=YELLOW)

    # Row 3: IP + uptime
    _draw_row(draw, 3, _trunc(f"{snap['local_ip']}  {state.uptime()}"), GRAY)
    return img


def render_color_bar_test(phase: int) -> Image.Image:
    """Startup color-bar test — cycles through brightness levels to verify all RGB channels."""
    img, draw = _new_frame()
    bars = [RED, ORANGE, YELLOW, GREEN, CYAN, BLUE, WHITE]
    bar_w = WIDTH // len(bars)
    intensity = [255, 180, 80][min(phase // 2, 2)]
    for i, color in enumerate(bars):
        x0 = i * bar_w
        x1 = x0 + bar_w - 1
        scaled = tuple(int(c * intensity // 255) for c in color)
        draw.rectangle([x0, 0, x1, HEIGHT - 1], fill=scaled)
    return img


# ── PioMatter matrix setup (Pi 5 specific) ────────────────────────────────────

def build_matrix(n_addr_lines: int, pinout_name: str):
    """
    Create a PioMatter matrix driver.

    n_addr_lines:
      4 = 32-row panels (2^4 = 16 half-rows, 1:16 mux) ← use this for 128×32
      5 = 64-row panels (1:32 mux)

    pinout_name:
      'bonnet'  → piomatter.Pinout.AdafruitMatrixBonnet  (Adafruit RGB Matrix Bonnet)
      'active3' → piomatter.Pinout.Active3
    """
    pinout_map = {
        "bonnet":  piomatter.Pinout.AdafruitMatrixBonnet,
        "active3": piomatter.Pinout.Active3,
    }
    if pinout_name not in pinout_map:
        raise ValueError(f"Unknown pinout '{pinout_name}'. Choose: {list(pinout_map)}")

    geometry = piomatter.Geometry(
        width        = WIDTH,
        height       = HEIGHT,
        n_addr_lines = n_addr_lines,
        rotation     = piomatter.Orientation.Normal,
    )

    # Create the initial blank canvas and a writable numpy framebuffer
    canvas     = Image.new("RGB", (WIDTH, HEIGHT), BLACK)
    # +0 forces a writeable copy (np.asarray alone gives read-only)
    framebuffer = np.asarray(canvas) + 0

    matrix = piomatter.PioMatter(
        colorspace  = piomatter.Colorspace.RGB888Packed,
        pinout      = pinout_map[pinout_name],
        framebuffer = framebuffer,
        geometry    = geometry,
    )

    return matrix, framebuffer


def show_frame(framebuffer: np.ndarray, matrix, img: Image.Image):
    """Push a PIL image to the LED matrix via the numpy framebuffer."""
    framebuffer[:] = np.asarray(img)
    matrix.show()


def clear_matrix(framebuffer: np.ndarray, matrix):
    framebuffer[:] = 0
    matrix.show()


# ── Main display loop ─────────────────────────────────────────────────────────

NORMAL_SCREENS     = [render_main_status, render_camera_detail, render_daily_stats]
SCREEN_ROTATE_SECS = 5
TICK               = 0.25   # 4 fps — plenty for a status board


def run(matrix, framebuffer: np.ndarray, state: SystemState):
    screen_idx   = 0
    last_rotate  = time.monotonic()
    flash_tick   = 0

    log.info("Display loop started (%d×%d)", WIDTH, HEIGHT)

    # Startup color-bar test (3 seconds)
    for phase in range(6):
        show_frame(framebuffer, matrix, render_color_bar_test(phase))
        time.sleep(0.5)

    while True:
        now = time.monotonic()
        flash_tick = (flash_tick + 1) % 4
        alert = state.pop_alert()

        if state.is_test_mode:
            # Test mode: static screen only — no rotation, no sliding content.
            # Only interrupts for alert flashes, then returns to static view.
            img = render_alert(state, alert, flash_phase=flash_tick % 2) if alert \
                  else render_test_static(state)
        elif alert:
            img = render_alert(state, alert, flash_phase=flash_tick % 2)
        elif not state.last_poll_ok:
            img = render_offline(state)
        else:
            if now - last_rotate >= SCREEN_ROTATE_SECS:
                screen_idx  = (screen_idx + 1) % len(NORMAL_SCREENS)
                last_rotate = now
            img = NORMAL_SCREENS[screen_idx](state)

        show_frame(framebuffer, matrix, img)
        time.sleep(TICK)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Road Sentinel HUB75 128×32 RGB LED Matrix — Raspberry Pi 5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  Real mode (default) — polls Node Service API
    REAL alerts: severity-colored (red=crash, orange=high, yellow=medium, cyan=low)
  Test mode (--test)  — fake data, no API, cycles TEST alerts
    TEST alerts: amber header with [TEST] label

Examples:
  python3 display_manager.py                         # real mode
  python3 display_manager.py --test                  # test mode
  python3 display_manager.py --trigger-alert         # fire one test alert then go live
  python3 display_manager.py --api http://192.168.8.50:3001
  python3 display_manager.py --pinout bonnet         # if using Adafruit RGB Matrix Bonnet
        """,
    )
    parser.add_argument("--test",          action="store_true",
                        help="Test mode: fake data, amber alerts, no API")
    parser.add_argument("--api",           default="http://localhost:3001",
                        help="Node Service base URL (default: http://localhost:3001)")
    parser.add_argument("--pinout",        default="active3",
                        choices=["bonnet", "active3"],
                        help="GPIO pinout (default: active3 = ₱149 Chinese adapter / hzeller regular)")
    parser.add_argument("--addr-lines",    type=int, default=4,
                        help="Address lines (default: 4 for 32-row panels)")
    parser.add_argument("--trigger-alert", action="store_true",
                        help="Fire one test alert immediately, then continue normally")

    args = parser.parse_args()

    if not HW_AVAILABLE:
        parser.error(
            "PioMatter not installed.\n"
            "Run: pip install Adafruit-Blinka-Raspberry-Pi5-Piomatter\n"
            "See lcd/README.md for full setup steps."
        )

    log.info("Initialising matrix %d×%d  pinout=%s  addr_lines=%d",
             WIDTH, HEIGHT, args.pinout, args.addr_lines)
    log.info("Mode: %s", "TEST" if args.test else "REAL")

    matrix, framebuffer = build_matrix(args.addr_lines, args.pinout)
    state = SystemState()

    provider: DataProvider = (
        TestDataProvider(state) if args.test
        else ApiDataProvider(state, base_url=args.api)
    )
    provider.start()

    if args.trigger_alert:
        time.sleep(1)
        provider.trigger_test_alert()

    try:
        run(matrix, framebuffer, state)
    except KeyboardInterrupt:
        log.info("Stopped by user")
    finally:
        clear_matrix(framebuffer, matrix)
        log.info("Display cleared")


if __name__ == "__main__":
    main()
