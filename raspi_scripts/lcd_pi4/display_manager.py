#!/usr/bin/env python3
"""
Road Sentinel — HUB75 128×32 RGB LED Matrix Display Manager
Raspberry Pi 4 Model B version — uses hzeller/rpi-rgb-led-matrix

Pi 4 retains Broadcom direct GPIO, so the hzeller library works here.
(Pi 5 users: see ../lcd/display_manager.py which uses Adafruit PioMatter instead.)

Hardware: HUB75 RGB LED matrix — 128×32 pixels (full color)
  Default config: two 64×32 panels chained  → total 128×32
  Single panel:   one 128×32 panel

Wiring: Plug the ₱149 Chinese HUB75 adapter board onto the 40-pin GPIO header.
  Uses hzeller "regular" GPIO mapping — same physical pins as the adapter board.
  See lcd_pi4/README.md for full wiring/install details.

Install (build from source — no pip):
  See lcd_pi4/README.md Step 3, or run install.sh

Run:
  sudo python3 display_manager.py           # real mode (live API)
  sudo python3 display_manager.py --test    # test mode (fake data, no API)
  (sudo required for direct GPIO /dev/mem access)
"""

import argparse
import logging
import socket
import time
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

# ── Try to import rgbmatrix (only on Raspberry Pi with hzeller library built) ──
try:
    from rgbmatrix import RGBMatrix, RGBMatrixOptions
    HW_AVAILABLE = True
except ImportError:
    HW_AVAILABLE = False
    print("WARNING: rgbmatrix not found.")
    print("Build it from source — see lcd_pi4/README.md or run install.sh")

try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

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
AMBER   = (255, 160, 0  )
DARK_GREEN = (0, 80,  0 )

SEVERITY_COLORS = {
    "critical": RED,
    "high":     ORANGE,
    "medium":   YELLOW,
    "low":      CYAN,
}

# ── Font & layout ─────────────────────────────────────────────────────────────
ROW_H = 8
ROWS  = [0, 8, 16, 24]

def _load_font(size: int = 8) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()

FONT_SM = _load_font(8)

def _trunc(text: str, max_chars: int = 21) -> str:
    return text if len(text) <= max_chars else text[:max_chars - 1] + "…"

def _get_local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "?.?.?.?"

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
                "uptime":          self.uptime(),
            }


# ── Data providers ────────────────────────────────────────────────────────────

class DataProvider(ABC):
    def __init__(self, state: SystemState):
        self.state = state

    @abstractmethod
    def start(self): ...

    def trigger_test_alert(self):
        pass


class ApiDataProvider(DataProvider):
    """Polls Node Service REST API every 10 seconds."""

    POLL_INTERVAL = 10

    def __init__(self, state: SystemState, base_url: str):
        super().__init__(state)
        self._base = base_url.rstrip("/")
        self._last_incident_id: Optional[int] = None

    def start(self):
        threading.Thread(target=self._poll_loop, daemon=True).start()
        log.info("API data provider started — polling %s", self._base)

    def _poll_loop(self):
        while True:
            try:
                self._poll()
            except Exception as exc:
                log.warning("Poll error: %s", exc)
                self.state.last_poll_ok = False
            time.sleep(self.POLL_INTERVAL)

    def _poll(self):
        if not _REQUESTS_OK:
            raise RuntimeError("requests library not installed")

        r = _requests.get(f"{self._base}/api/analytics/summary", timeout=5)
        r.raise_for_status()
        self.state.update_summary(r.json())

        r = _requests.get(f"{self._base}/api/cameras", timeout=5)
        r.raise_for_status()
        self.state.update_cameras(r.json())

        r = _requests.get(
            f"{self._base}/api/incidents",
            params={"status": "active", "limit": 1},
            timeout=5,
        )
        r.raise_for_status()
        items = r.json()
        if items:
            inc = items[0]
            inc_id = inc.get("id")
            if inc_id != self._last_incident_id:
                self._last_incident_id = inc_id
                inc["is_test"] = False
                self.state.push_alert(inc, hold_secs=12.0)
                log.info("REAL alert: %s (%s)", inc.get("incident_type"), inc.get("severity"))


class TestDataProvider(DataProvider):
    """Fake data — no network. Cycles simulated TEST alerts (amber)."""

    _FAKE_INCIDENTS = [
        {"is_test": True, "incident_type": "speeding",   "severity": "high",
         "title": "Test Speeding",    "description": "85 km/h on Camera A",
         "camera_name": "Camera A",   "timestamp": ""},
        {"is_test": True, "incident_type": "crash",      "severity": "critical",
         "title": "Test Crash",       "description": "Simulated collision",
         "camera_name": "Camera B",   "timestamp": ""},
        {"is_test": True, "incident_type": "congestion", "severity": "low",
         "title": "Test Congestion",  "description": "Slow traffic detected",
         "camera_name": "Camera A",   "timestamp": ""},
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
    return img


def render_camera_detail(state: SystemState) -> Image.Image:
    img, draw = _new_frame()
    snap = state.snapshot()
    cams = snap["cameras"]

    _draw_row(draw, 0, "CAMERA STATUS", WHITE)

    if not cams:
        _draw_row(draw, 1, "No camera data", GRAY)
        return img

    for i, cam in enumerate(cams[:2]):
        row    = i + 1
        name   = cam.get("name", f"Cam {i+1}")[:8]
        status = cam.get("status", "unknown")
        fps    = cam.get("fps")
        col    = GREEN if status == "online" else (ORANGE if status == "error" else RED)
        fps_s  = f" {fps}fps" if fps else ""
        _draw_row(draw, row, _trunc(f"{name}: {status.upper()}{fps_s}"), col)

    _draw_row(draw, 3, _trunc(f"Inc: {snap['incidents_today']}  Up:{state.uptime()}"), GRAY)
    return img


def render_daily_stats(state: SystemState) -> Image.Image:
    img, draw = _new_frame()
    snap = state.snapshot()
    _draw_row(draw, 0, _trunc(f"Today: {snap['vehicles_today']:,} vehicles"), WHITE)
    _draw_row(draw, 1, _trunc(f"Incidents: {snap['incidents_today']}"), ORANGE)
    speed_str = f"{snap['avg_speed']} km/h" if snap["avg_speed"] else "N/A"
    _draw_row(draw, 2, _trunc(f"Avg speed: {speed_str}"), CYAN)
    _draw_row(draw, 3, _trunc(f"Up: {state.uptime()}"), GRAY)
    return img


def render_alert(state: SystemState, incident: dict, flash_phase: int = 0) -> Image.Image:
    img, draw = _new_frame()

    is_test   = incident.get("is_test", False)
    inc_type  = str(incident.get("incident_type", "ALERT")).upper()
    severity  = str(incident.get("severity", "high")).lower()
    cam_name  = str(incident.get("camera_name", "Camera ?"))
    desc      = str(incident.get("description", ""))
    ts_raw    = incident.get("timestamp", "")
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

    draw.text((1, ROWS[0]), "ROAD SENTINEL", font=FONT_SM, fill=WHITE)
    draw.text((103, ROWS[0]), "[TST]", font=FONT_SM, fill=AMBER)

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

    speed_str = f"{snap['avg_speed']}km/h" if snap["avg_speed"] else "N/A"
    draw.text((1,  ROWS[2]), f"Veh:{snap['vehicles_today']:,}", font=FONT_SM, fill=CYAN)
    draw.text((70, ROWS[2]), speed_str, font=FONT_SM, fill=YELLOW)
    _draw_row(draw, 3, _trunc(f"{snap['local_ip']}  {state.uptime()}"), GRAY)
    return img


def render_color_bar_test(phase: int) -> Image.Image:
    """Startup color-bar test — cycles through brightness levels."""
    img, draw = _new_frame()
    bars  = [RED, ORANGE, YELLOW, GREEN, CYAN, BLUE, WHITE]
    bar_w = WIDTH // len(bars)
    intensity = [255, 180, 80][min(phase // 2, 2)]
    for i, color in enumerate(bars):
        x0 = i * bar_w
        x1 = x0 + bar_w - 1
        scaled = tuple(int(c * intensity // 255) for c in color)
        draw.rectangle([x0, 0, x1, HEIGHT - 1], fill=scaled)
    return img


# ── hzeller RGBMatrix setup (Pi 4 specific) ───────────────────────────────────

def build_matrix(gpio_slowdown: int, hardware_mapping: str) -> "RGBMatrix":
    """
    Create an RGBMatrix driver for Raspberry Pi 4.

    gpio_slowdown:
      Pi 4 typically needs 4. If display is garbled/flickering try 3 or 5.

    hardware_mapping:
      'regular' — hzeller "regular" pinout, matches the ₱149 Chinese adapter board.
    """
    options = RGBMatrixOptions()
    options.rows            = HEIGHT          # 32
    options.cols            = 64              # each panel is 64 wide
    options.chain_length    = WIDTH // 64     # 2 panels → total 128 wide
    options.parallel        = 1
    options.hardware_mapping = hardware_mapping
    options.gpio_slowdown   = gpio_slowdown
    options.drop_privileges = False           # keep root so we can control GPIO
    options.disable_hardware_pulsing = False
    return RGBMatrix(options=options)


def show_frame(matrix: "RGBMatrix", canvas, img: Image.Image):
    """Push a PIL image to the LED matrix and swap to vsync."""
    canvas.SetImage(img.convert("RGB"))
    return matrix.SwapOnVSync(canvas)


# ── Main display loop ─────────────────────────────────────────────────────────

NORMAL_SCREENS     = [render_main_status, render_camera_detail, render_daily_stats]
SCREEN_ROTATE_SECS = 5
TICK               = 0.25   # 4 fps


def run(matrix: "RGBMatrix", state: SystemState):
    canvas     = matrix.CreateFrameCanvas()
    screen_idx = 0
    last_rotate = time.monotonic()
    flash_tick  = 0

    log.info("Display loop started (%d×%d)", WIDTH, HEIGHT)

    # Startup color-bar test (3 seconds)
    for phase in range(6):
        canvas = show_frame(matrix, canvas, render_color_bar_test(phase))
        time.sleep(0.5)

    while True:
        now = time.monotonic()
        flash_tick = (flash_tick + 1) % 4
        alert = state.pop_alert()

        if state.is_test_mode:
            # Test mode: static screen — no rotation, no sliding content.
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

        canvas = show_frame(matrix, canvas, img)
        time.sleep(TICK)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Road Sentinel HUB75 128×32 RGB LED Matrix — Raspberry Pi 4 Model B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  Real mode (default) — polls Node Service API
    REAL alerts: severity-colored (red=crash, orange=high, yellow=medium, cyan=low)
  Test mode (--test)  — fake data, no API, cycles TEST alerts (amber)
    Static test screen — no sliding/rotating content

Examples:
  sudo python3 display_manager.py                         # real mode
  sudo python3 display_manager.py --test                  # test mode
  sudo python3 display_manager.py --trigger-alert         # fire one test alert then go live
  sudo python3 display_manager.py --api http://192.168.8.50:3001
  sudo python3 display_manager.py --slowdown 3            # if display is garbled (try 3–5)
        """,
    )
    parser.add_argument("--test",          action="store_true",
                        help="Test mode: fake data, amber alerts, no API")
    parser.add_argument("--api",           default="http://localhost:3001",
                        help="Node Service base URL (default: http://localhost:3001)")
    parser.add_argument("--slowdown",      type=int, default=4,
                        help="GPIO slowdown for Pi 4 (default: 4, try 3-5 if garbled)")
    parser.add_argument("--mapping",       default="regular",
                        choices=["regular", "adafruit-hat", "adafruit-hat-pwm"],
                        help="GPIO mapping (default: regular = ₱149 Chinese adapter board)")
    parser.add_argument("--trigger-alert", action="store_true",
                        help="Fire one test alert immediately, then continue normally")

    args = parser.parse_args()

    if not HW_AVAILABLE:
        parser.error(
            "rgbmatrix not installed.\n"
            "Build from source — see lcd_pi4/README.md or run install.sh\n"
        )

    log.info("Initialising matrix %d×%d  mapping=%s  slowdown=%d",
             WIDTH, HEIGHT, args.mapping, args.slowdown)
    log.info("Mode: %s", "TEST" if args.test else "REAL")

    matrix = build_matrix(gpio_slowdown=args.slowdown, hardware_mapping=args.mapping)
    state  = SystemState()

    provider: DataProvider = (
        TestDataProvider(state) if args.test
        else ApiDataProvider(state, base_url=args.api)
    )
    provider.start()

    if args.trigger_alert:
        time.sleep(1)
        provider.trigger_test_alert()

    try:
        run(matrix, state)
    except KeyboardInterrupt:
        log.info("Stopped by user")
    finally:
        matrix.Clear()
        log.info("Display cleared")


if __name__ == "__main__":
    main()
