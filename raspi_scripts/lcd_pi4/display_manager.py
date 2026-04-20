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
ROW_H = 5
ROWS  = [0, 5, 10, 15, 20, 25]

FONT_SM = ImageFont.load_default(size=4)

def _trunc(text: str, max_chars: int = 42) -> str:
    return text if len(text) <= max_chars else text[:max_chars - 1] + "."

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

def _safe_or_danger(snap: dict) -> tuple:
    """Return (text, color) SAFE/DANGER based on active incidents."""
    if snap["incidents_today"] > 0:
        return "!! DANGER !!", RED
    return "-- SAFE --", GREEN


def render_main_status(state: SystemState) -> Image.Image:
    img, draw = _new_frame()
    snap = state.snapshot()
    now  = datetime.now().strftime("%H:%M")

    # Row 0: header + time
    _draw_row(draw, 0, f"ROAD SENTINEL  {now}", WHITE)

    # Row 1: SAFE (green) / DANGER (red)
    status_text, status_col = _safe_or_danger(snap)
    _fill_row(draw, 1, (0, 40, 0) if status_col == GREEN else (60, 0, 0))
    _draw_row(draw, 1, status_text, status_col)

    # Row 2: Camera A / B status — explicit GREEN online, RED offline
    cams = snap["cameras"]
    if cams:
        cam_a = next((c for c in cams if "A" in c.get("name", "") or "001" in c.get("id", "")), None)
        cam_b = next((c for c in cams if "B" in c.get("name", "") or "002" in c.get("id", "")), None)
        a_on  = cam_a and cam_a.get("status") == "online"
        b_on  = cam_b and cam_b.get("status") == "online"
        draw.text((1,  ROWS[2]), "A:", font=FONT_SM, fill=GRAY)
        draw.text((9,  ROWS[2]), "ON" if a_on else "OFF", font=FONT_SM, fill=GREEN if a_on else RED)
        draw.text((33, ROWS[2]), "B:", font=FONT_SM, fill=GRAY)
        draw.text((41, ROWS[2]), "ON" if b_on else "OFF", font=FONT_SM, fill=GREEN if b_on else RED)
    else:
        online = snap["cameras_online"]
        col = GREEN if online == snap["cameras_total"] else RED
        _draw_row(draw, 2, f"Cams: {online}/{snap['cameras_total']}", col)

    # Row 3: Vehicle count + incidents
    draw.text((1,  ROWS[3]), f"Veh: {snap['vehicles_today']:,}", font=FONT_SM, fill=CYAN)
    draw.text((70, ROWS[3]), f"Inc: {snap['incidents_today']}", font=FONT_SM,
              fill=RED if snap["incidents_today"] > 0 else GRAY)

    # Row 4: Avg speed
    speed_str = f"Avg: {snap['avg_speed']} km/h" if snap["avg_speed"] else "Avg speed: N/A"
    _draw_row(draw, 4, speed_str, YELLOW)

    # Row 5: IP + uptime
    _draw_row(draw, 5, _trunc(f"{snap['local_ip']}  {state.uptime()}"), GRAY)
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
        base  = i * 2 + 1          # rows 1-2 for cam A, 3-4 for cam B
        name  = cam.get("name", f"Cam {i+1}")
        st    = cam.get("status", "unknown")
        fps   = cam.get("fps")
        res   = cam.get("resolution", "")
        col   = GREEN if st == "online" else (ORANGE if st == "error" else RED)
        _draw_row(draw, base,     f"{name}: {st.upper()}", col)
        detail = f"  {fps}fps  {res}" if fps else f"  {res}"
        _draw_row(draw, base + 1, _trunc(detail), GRAY)

    _draw_row(draw, 5, _trunc(f"Inc: {snap['incidents_today']}  Up: {state.uptime()}"), GRAY)
    return img


def render_daily_stats(state: SystemState) -> Image.Image:
    img, draw = _new_frame()
    snap = state.snapshot()
    speed_str = f"{snap['avg_speed']} km/h" if snap["avg_speed"] else "N/A"

    _draw_row(draw, 0, "TODAY'S STATS", WHITE)
    _draw_row(draw, 1, f"Vehicles:  {snap['vehicles_today']:,}", CYAN)
    _draw_row(draw, 2, f"Avg speed: {speed_str}", YELLOW)
    _draw_row(draw, 3, f"Incidents: {snap['incidents_today']}",
              RED if snap["incidents_today"] > 0 else GREEN)
    online = snap["cameras_online"]
    cam_col = GREEN if online == snap["cameras_total"] else RED
    _draw_row(draw, 4, f"Cameras:   {online}/{snap['cameras_total']} online", cam_col)
    _draw_row(draw, 5, f"Uptime:    {state.uptime()}", GRAY)
    return img


def render_alert(state: SystemState, incident: dict, flash_phase: int = 0) -> Image.Image:
    img, draw = _new_frame()

    is_test  = incident.get("is_test", False)
    inc_type = str(incident.get("incident_type", "ALERT")).upper()
    severity = str(incident.get("severity", "high")).lower()
    cam_name = str(incident.get("camera_name", "Camera ?"))
    desc     = str(incident.get("description", ""))
    ts_raw   = incident.get("timestamp", "")
    try:
        ts = datetime.fromisoformat(str(ts_raw).replace("Z", "")).strftime("%H:%M:%S")
    except Exception:
        ts = datetime.now().strftime("%H:%M:%S")

    if is_test:
        bg        = AMBER if flash_phase == 0 else (160, 100, 0)
        fg        = BLACK
        sev_label = "SIMULATED"
        sev_col   = AMBER
    else:
        sev_color = SEVERITY_COLORS.get(severity, ORANGE)
        bg        = sev_color if flash_phase == 0 else tuple(max(0, c - 70) for c in sev_color)
        fg        = WHITE if severity in ("critical", "high") else BLACK
        sev_label = severity.upper()
        sev_col   = sev_color

    # Row 0: type + camera (colored header bar)
    _fill_row(draw, 0, bg)
    prefix = "[TEST]" if is_test else "!!"
    draw.text((1, ROWS[0]), _trunc(f"{prefix} {inc_type}"), font=FONT_SM, fill=fg)

    # Row 1: severity label — explicit RED for critical, keep severity color otherwise
    label_bg = (50, 0, 0) if severity == "critical" and not is_test else BLACK
    _fill_row(draw, 1, label_bg)
    _draw_row(draw, 1, sev_label, RED if (severity == "critical" and not is_test) else sev_col)

    # Row 2: camera name
    _draw_row(draw, 2, cam_name, WHITE)

    # Row 3: description
    _draw_row(draw, 3, _trunc(desc or "See dashboard"), GRAY)

    # Row 4: timestamp
    _draw_row(draw, 4, ts, GRAY)

    # Row 5: DANGER explicit red label
    _fill_row(draw, 5, (60, 0, 0) if not is_test else (80, 50, 0))
    _draw_row(draw, 5, "!! DANGER !!" if not is_test else "-- TEST MODE --",
              RED if not is_test else AMBER)
    return img


def render_offline(state: SystemState) -> Image.Image:
    img, draw = _new_frame()
    _draw_row(draw, 0, "ROAD SENTINEL", WHITE)
    _fill_row(draw, 1, (60, 0, 0))
    _draw_row(draw, 1, "!! OFFLINE !!", RED)
    _draw_row(draw, 2, "API not reachable", GRAY)
    _draw_row(draw, 3, "", GRAY)
    _draw_row(draw, 4, _trunc(state.local_ip), GRAY)
    _draw_row(draw, 5, f"Up: {state.uptime()}", GRAY)
    return img


def render_test_static(state: SystemState) -> Image.Image:
    """Fixed test-mode status screen — no rotation, no sliding content."""
    img, draw = _new_frame()
    snap = state.snapshot()

    # Row 0: header
    _draw_row(draw, 0, "ROAD SENTINEL", WHITE)
    draw.text((90, ROWS[0]), "[TEST]", font=FONT_SM, fill=AMBER)

    # Row 1: SAFE indicator (green — no real incidents in test mode)
    _fill_row(draw, 1, (0, 40, 0))
    _draw_row(draw, 1, "-- SAFE -- (simulated)", GREEN)

    # Row 2: Camera status — explicit green
    cams = snap["cameras"]
    if cams:
        cam_a = next((c for c in cams if "A" in c.get("name", "")), cams[0] if cams else None)
        cam_b = next((c for c in cams if "B" in c.get("name", "")), cams[1] if len(cams) > 1 else None)
        a_on = cam_a and cam_a.get("status") == "online"
        b_on = cam_b and cam_b.get("status") == "online"
        draw.text((1,  ROWS[2]), "A:", font=FONT_SM, fill=GRAY)
        draw.text((9,  ROWS[2]), "ON" if a_on else "OFF", font=FONT_SM, fill=GREEN if a_on else RED)
        draw.text((33, ROWS[2]), "B:", font=FONT_SM, fill=GRAY)
        draw.text((41, ROWS[2]), "ON" if b_on else "OFF", font=FONT_SM, fill=GREEN if b_on else RED)
        draw.text((65, ROWS[2]), "SIMULATED", font=FONT_SM, fill=AMBER)
    else:
        _draw_row(draw, 2, "SIMULATED", AMBER)

    # Row 3: vehicle count (static — no animation)
    draw.text((1,  ROWS[3]), f"Veh: {snap['vehicles_today']:,}", font=FONT_SM, fill=CYAN)
    speed_str = f"{snap['avg_speed']} km/h" if snap["avg_speed"] else "N/A"
    draw.text((70, ROWS[3]), speed_str, font=FONT_SM, fill=YELLOW)

    # Row 4: IP
    _draw_row(draw, 4, snap["local_ip"], GRAY)

    # Row 5: uptime
    _draw_row(draw, 5, f"Up: {state.uptime()}", GRAY)
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

def build_matrix(gpio_slowdown: int, hardware_mapping: str,
                 no_hardware_pulse: bool = True,
                 cols_per_panel: int = 64,
                 chain_length: int = 0,
                 multiplexing: int = 0,
                 scan_mode: int = 0) -> "RGBMatrix":
    """
    Chaining / panel layout notes
    ──────────────────────────────
    Most common 64×32 P3/P4/P5 panels  →  cols=64, chain=2, multiplexing=0, scan_mode=0
    1:8 multiplexed panels (shows same content on both halves)
                                        →  cols=32, chain=4, multiplexing=1, scan_mode=0
    If everything looks correct but one panel is upside-down or reversed:
      try scan_mode=1 (interlaced row scan)

    chain_length=0 (default) → auto-calculated as WIDTH // cols_per_panel
    """
    options = RGBMatrixOptions()
    options.rows                     = HEIGHT
    options.cols                     = cols_per_panel
    options.chain_length             = chain_length if chain_length > 0 else (WIDTH // cols_per_panel)
    options.parallel                 = 1
    options.hardware_mapping         = hardware_mapping
    options.gpio_slowdown            = gpio_slowdown
    options.multiplexing             = multiplexing
    options.scan_mode                = scan_mode
    options.drop_privileges          = False
    options.disable_hardware_pulsing = no_hardware_pulse
    log.info(
        "Matrix options: rows=%d  cols=%d  chain=%d  multiplex=%d  scan=%d  slowdown=%d  mapping=%s",
        options.rows, options.cols, options.chain_length,
        options.multiplexing, options.scan_mode,
        options.gpio_slowdown, options.hardware_mapping,
    )
    matrix = RGBMatrix(options=options)
    log.info("Matrix created: width=%d  height=%d", matrix.width, matrix.height)
    return matrix


def show_frame(matrix: "RGBMatrix", canvas, img: Image.Image):
    """Push PIL image to the chained matrix.
    Uses SetPixel loop across the full 128px width — works even when
    matrix.width reports only 64 (Python bindings chain_length bug).
    """
    rgb = img.convert("RGB")
    px  = rgb.load()
    for y in range(HEIGHT):
        for x in range(WIDTH):
            r, g, b = px[x, y]
            canvas.SetPixel(x, y, r, g, b)
    return matrix.SwapOnVSync(canvas)


# ── Main display loop ─────────────────────────────────────────────────────────

NORMAL_SCREENS     = [render_main_status, render_camera_detail, render_daily_stats]
SCREEN_ROTATE_SECS = 12     # slow rotation — enough time to read all 6 rows
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
    parser.add_argument("--cols",          type=int, default=64,
                        help="Physical columns per panel (default: 64). "
                             "Try 32 if both panels mirror each other.")
    parser.add_argument("--chain",         type=int, default=0,
                        help="Number of chained panels (default: auto = 128 / cols). "
                             "Set to 4 when using --cols 32.")
    parser.add_argument("--multiplexing",  type=int, default=0,
                        help="Panel multiplexing type (default: 0=standard). "
                             "Try 1 if both panels show identical content (1:8 scan panels).")
    parser.add_argument("--scan-mode",     type=int, default=0, choices=[0, 1],
                        help="Row scan mode: 0=progressive (default), 1=interlaced.")
    parser.add_argument("--mapping",       default="regular",
                        choices=["regular", "adafruit-hat", "adafruit-hat-pwm"],
                        help="GPIO mapping (default: regular = ₱149 Chinese adapter board)")
    parser.add_argument("--hardware-pulse", action="store_true", default=False,
                        help="Enable hardware PWM pulse (only after disabling snd_bcm2835 in raspi-config)")
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

    matrix = build_matrix(
        gpio_slowdown    = args.slowdown,
        hardware_mapping = args.mapping,
        no_hardware_pulse= not args.hardware_pulse,
        cols_per_panel   = args.cols,
        chain_length     = args.chain,
        multiplexing     = args.multiplexing,
        scan_mode        = args.scan_mode,
    )
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
