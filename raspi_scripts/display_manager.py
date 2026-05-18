#!/usr/bin/env python3
"""
Road Sentinel — HUB75 128×32 RGB LED Matrix Display Manager
Unified version — works on Raspberry Pi 4 AND Raspberry Pi 5 automatically.

Pi detection at startup:
  /dev/pio0 exists  → Pi 5 → led-image-viewer (SwapOnVSync, coprocessor mode)
  otherwise         → Pi 4 → hzeller ledcat (/dev/mem GPIO)

Frame pipeline:
  PIL Image (128×32) → numpy palette-snap → raw RGB24 bytes → PPM file → led-image-viewer

Pi 4 backend:
  ledcat → hzeller C lib → /dev/mem GPIO
  Requires: ~/rpi-rgb-led-matrix/examples-api-use/ledcat  (built by install.sh)
  Requires: sudo

Pi 5 backend:
  led-image-viewer → hzeller C lib → RP1 coprocessor mode (no --led-rp1-rio=1)
  Uses SwapOnVSync for atomic frame updates — no mid-scan RP1 corruption.
  RIO mode (--led-rp1-rio=1) causes state machine de-sync → do NOT use it.
  Requires: ~/rpi-rgb-led-matrix/utils/led-image-viewer  (built by install.sh)
  Requires: sudo

Install:
  Pi 4:  bash raspi_scripts/lcd_pi4/install.sh
  Pi 5:  bash raspi_scripts/lcd/install.sh

Run:
  sudo python3 display_manager.py           # Pi 4 or Pi 5 (sudo needed)
       python3 display_manager.py --test    # test mode (cycles states automatically)
"""

import argparse
import logging
import os
import signal
import socket
import subprocess
import time
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

# ── Colors ────────────────────────────────────────────────────────────────────
WHITE     = (255, 255, 255)
BLACK     = (0,   0,   0  )
RED       = (220, 0,   0  )
GREEN     = (0,   200, 0  )
BLUE      = (30,  80,  255)
YELLOW    = (220, 200, 0  )
ORANGE    = (255, 110, 0  )
CYAN      = (0,   200, 200)
GRAY      = (90,  90,  90 )
AMBER     = (255, 160, 0  )
DARK_GREEN= (0,   80,  0  )
MAGENTA   = (200, 0,   180)

SEVERITY_COLORS = {
    "critical": RED,
    "high":     ORANGE,
    "medium":   YELLOW,
    "low":      CYAN,
}

# Palette of every color the renderers can produce.
# Used to snap anti-aliased PIL pixels to the nearest pure color
# before sending frames to the LED panel via ledcat.
_LED_PALETTE = np.array([
    [0,   0,   0  ], [255, 255, 255], [220, 0,   0  ], [0,   200, 0  ],
    [30,  80,  255], [220, 200, 0  ], [255, 110, 0  ], [0,   200, 200],
    [90,  90,  90 ], [255, 160, 0  ], [0,   80,  0  ], [200, 0,   180],
    # Background variants used in render functions
    [0,   40,  0  ], [60,  0,   0  ], [50,  0,   0  ], [110, 0,   0  ],
    [120, 0,   110], [80,  50,  0  ], [0,   55,  0  ], [150, 65,  0  ],
    [160, 100, 0  ],
], dtype=np.float32)


def _snap_frame(img: Image.Image) -> bytes:
    """Snap each pixel to the nearest palette color — eliminates PIL anti-aliasing."""
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    flat = arr.reshape(-1, 3)
    diff = flat[:, np.newaxis, :] - _LED_PALETTE[np.newaxis, :, :]
    nearest = np.argmin(np.sum(diff ** 2, axis=2), axis=1)
    return _LED_PALETTE[nearest].reshape(arr.shape).astype(np.uint8).tobytes()

# ── Font & layout ─────────────────────────────────────────────────────────────
ROW_H      = 5
ROWS       = [0, 5, 10, 15, 20, 25]

def _load_font(size: int):
    """Load default font — supports both Pillow 10+ (size kwarg) and older versions."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()

FONT_SM    = _load_font(4)
FONT_XS6   = _load_font(6)
FONT_MED   = _load_font(10)
FONT_LARGE = _load_font(16)

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

def _draw_row(draw, row: int, text: str, color=WHITE, x: int = 1):
    draw.text((x, ROWS[row]), text, font=FONT_SM, fill=color)

def _fill_row(draw, row: int, color):
    draw.rectangle([0, ROWS[row], WIDTH - 1, ROWS[row] + ROW_H - 1], fill=color)

def _fill_row_px(draw, y0: int, y1: int, color):
    draw.rectangle([0, y0, WIDTH - 1, y1], fill=color)

def _draw_centered(draw, y: int, text: str, font, color):
    """Horizontally center text at a fixed y."""
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    x = max(0, (WIDTH - w) // 2) - bbox[0]
    draw.text((x, y), text, font=font, fill=color)

def _draw_centered_in(draw, x0: int, y0: int, x1: int, y1: int, text: str, font, color):
    """Draw text centered (both axes) within the given pixel rectangle."""
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = x0 + max(0, ((x1 - x0 + 1) - w) // 2) - bbox[0]
    y = y0 + max(0, ((y1 - y0 + 1) - h) // 2) - bbox[1]
    draw.text((x, y), text, font=font, fill=color)

def _draw_two_lines_centered(draw, y0: int, y1: int,
                              line1: str, font1,
                              line2: str, font2, color):
    """Draw two lines centered as a group (H+V) within the given y range."""
    bb1 = draw.textbbox((0, 0), line1, font=font1)
    bb2 = draw.textbbox((0, 0), line2, font=font2)
    h1, h2 = bb1[3] - bb1[1], bb2[3] - bb2[1]
    gap    = 2
    total  = h1 + gap + h2
    sy     = y0 + max(0, ((y1 - y0 + 1) - total) // 2)
    x1_pos = max(0, (WIDTH - (bb1[2] - bb1[0])) // 2) - bb1[0]
    x2_pos = max(0, (WIDTH - (bb2[2] - bb2[0])) // 2) - bb2[0]
    draw.text((x1_pos, sy - bb1[1]),           line1, font=font1, fill=color)
    draw.text((x2_pos, sy + h1 + gap - bb2[1]), line2, font=font2, fill=color)


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY BACKENDS
# ══════════════════════════════════════════════════════════════════════════════

def _detect_pi() -> str:
    """Return 'pi5' if /dev/pio0 exists (RP1 chip), else 'pi4'."""
    if os.path.exists("/dev/pio0"):
        return "pi5"
    try:
        with open("/proc/cpuinfo") as f:
            if "Raspberry Pi 5" in f.read():
                return "pi5"
    except OSError:
        pass
    return "pi4"


class DisplayBackend(ABC):
    """Abstract display backend — same interface on Pi 4 and Pi 5."""
    @abstractmethod
    def show(self, img: Image.Image) -> None: ...
    @abstractmethod
    def clear(self) -> None: ...
    @abstractmethod
    def close(self) -> None: ...


# ── Pi 4 backend: ledcat subprocess ───────────────────────────────────────────

def _find_ledcat() -> str:
    for p in [
        os.path.expanduser("~/rpi-rgb-led-matrix/examples-api-use/ledcat"),
        "/home/roadsentinel/rpi-rgb-led-matrix/examples-api-use/ledcat",
    ]:
        if os.path.isfile(p):
            return p
    return os.path.expanduser("~/rpi-rgb-led-matrix/examples-api-use/ledcat")

LEDCAT_DEFAULT = _find_ledcat()
_BLACK_FRAME   = bytes(WIDTH * HEIGHT * 3)


def _find_led_image_viewer() -> str:
    for p in [
        os.path.expanduser("~/rpi-rgb-led-matrix/utils/led-image-viewer"),
        "/home/roadsentinel/rpi-rgb-led-matrix/utils/led-image-viewer",
    ]:
        if os.path.isfile(p):
            return p
    return os.path.expanduser("~/rpi-rgb-led-matrix/utils/led-image-viewer")

LED_IMAGE_VIEWER_DEFAULT = _find_led_image_viewer()


class LedcatBackend(DisplayBackend):
    """
    Pi 4/5 — direct synchronous writes to hzeller ledcat via stdin at TICK rate.
    No background thread. The caller drives the write rate via show()/clear().
    Pi 5: must write at ≥25 fps to keep the RP1 RIO state machine in steady state
    (see module docstring). Frame size: 128 × 32 × 3 = 12,288 bytes, no header.
    """

    def __init__(self, ledcat_path: str, slowdown: int, mapping: str,
                 no_hw_pulse: bool, cols: int, chain: int,
                 multiplexing: int, scan_mode: int,
                 rp1_rio: bool = False, pwm_bits: int = 0):
        if not os.path.isfile(ledcat_path):
            raise FileNotFoundError(
                f"ledcat not found: {ledcat_path}\n"
                "Build with: cd ~/rpi-rgb-led-matrix/examples-api-use && make"
            )
        _chain = chain if chain > 0 else (WIDTH // cols)
        cmd = [
            ledcat_path,
            f"--led-rows={HEIGHT}",
            f"--led-cols={cols}",
            f"--led-chain={_chain}",
            "--led-parallel=1",
            f"--led-gpio-mapping={mapping}",
            f"--led-slowdown-gpio={slowdown}",
            "--led-no-drop-privs",
        ]
        if no_hw_pulse:
            cmd.append("--led-no-hardware-pulse")
        if multiplexing > 0:
            cmd.append(f"--led-multiplexing={multiplexing}")
        if scan_mode != 0:
            cmd.append(f"--led-scan-mode={scan_mode}")
        if rp1_rio:
            cmd.append("--led-rp1-rio=1")
        if pwm_bits > 0:
            cmd.append(f"--led-pwm-bits={pwm_bits}")

        self._cmd         = cmd
        self._last_img_id = -1
        self._last_snap   = _BLACK_FRAME
        log.info("ledcat backend: %s", " ".join(cmd))
        self._proc = self._launch()

    def _launch(self) -> subprocess.Popen:
        proc = subprocess.Popen(self._cmd, stdin=subprocess.PIPE)
        log.info("ledcat PID=%d", proc.pid)
        return proc

    def _restart(self) -> None:
        """Kill ledcat, relaunch, burst 16 black frames to re-sync RP1."""
        log.warning("Restarting ledcat + RP1 re-init burst...")
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.kill()
            self._proc.wait(timeout=2)
        except Exception:
            pass
        self._proc = self._launch()
        for _ in range(16):
            try:
                self._proc.stdin.write(_BLACK_FRAME)
                self._proc.stdin.flush()
            except Exception:
                pass
            time.sleep(0.02)
        log.warning("ledcat restarted — RP1 re-synced")

    def _write(self, data: bytes) -> None:
        if self._proc.poll() is not None:
            log.error("ledcat PID=%d died — restarting", self._proc.pid)
            self._restart()
            return
        t0 = time.monotonic()
        try:
            self._proc.stdin.write(data)
            self._proc.stdin.flush()
        except OSError as e:
            log.error("ledcat write failed: %s — restarting", e)
            self._restart()
            return
        if time.monotonic() - t0 > 0.08:
            log.warning("Write blocked %.0fms — pipe backup, restarting",
                        (time.monotonic() - t0) * 1000)
            self._restart()

    def show(self, img: Image.Image) -> None:
        img_id = id(img)
        if img_id != self._last_img_id:
            self._last_snap   = _snap_frame(img)
            self._last_img_id = img_id
        self._write(self._last_snap)

    def clear(self) -> None:
        self._last_img_id = -1
        self._write(_BLACK_FRAME)

    def close(self) -> None:
        try:
            for _ in range(16):
                self._write(_BLACK_FRAME)
                time.sleep(0.02)
            self._proc.stdin.close()
            self._proc.wait(timeout=3)
        except Exception:
            pass
        try:
            self._proc.kill()
        except Exception:
            pass


# ── Pi 5 backend: led-image-viewer subprocess ────────────────────────────────

class LedImageViewerBackend(DisplayBackend):
    """
    Pi 5 — renders PIL frames to a temp PPM file, displays with led-image-viewer.
    led-image-viewer uses hzeller FrameCanvas + SwapOnVSync internally so frame
    updates are always atomic — no mid-scan corruption possible.
    Viewer process is restarted only when content actually changes.
    """

    def __init__(self, viewer_path: str, cols: int, chain: int):
        if not os.path.isfile(viewer_path):
            raise FileNotFoundError(
                f"led-image-viewer not found: {viewer_path}\n"
                "Build with: cd ~/rpi-rgb-led-matrix/utils && make"
            )
        _chain = chain if chain > 0 else (WIDTH // cols)
        self._viewer = viewer_path
        self._flags  = [
            f"--led-rows={HEIGHT}",
            f"--led-cols={cols}",
            f"--led-chain={_chain}",
            "--led-parallel=1",
            "--led-gpio-mapping=regular",
            "--led-slowdown-gpio=2",
            "--led-no-drop-privs",
            "--led-no-hardware-pulse",
            "--led-multiplexing=1",
            "--led-pwm-bits=7",
        ]
        self._ppm         = "/tmp/roadsentinel_frame.ppm"
        self._proc: Optional[subprocess.Popen] = None
        self._last_frame  = b""
        self._last_img_id = -1
        log.info("LedImageViewerBackend — %s  cols=%d  chain=%d", viewer_path, cols, _chain)

    def _write_ppm(self, data: bytes) -> None:
        with open(self._ppm, "wb") as f:
            f.write(f"P6\n{WIDTH} {HEIGHT}\n255\n".encode())
            f.write(data)

    def _restart(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            time.sleep(0.2)  # let panel fully blank after SIGTERM before next init
        cmd = [self._viewer] + self._flags + ["-l", "-1", "-w", "99999", self._ppm]
        log.info("led-image-viewer PID starting...")
        self._proc = subprocess.Popen(cmd)

    def _proc_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def show(self, img: Image.Image) -> None:
        if id(img) == self._last_img_id and self._proc_alive():
            return  # same image object, viewer still running — skip
        if not self._proc_alive():
            log.warning("led-image-viewer died — restarting")
        data = _snap_frame(img)
        self._last_img_id = id(img)
        self._last_frame  = data
        self._write_ppm(data)
        self._restart()

    def clear(self) -> None:
        if self._last_frame == _BLACK_FRAME and self._proc_alive():
            return
        self._last_frame = _BLACK_FRAME
        self._write_ppm(_BLACK_FRAME)
        self._restart()

    def close(self) -> None:
        try:
            self.clear()
            time.sleep(0.3)
        except Exception:
            pass
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()


# ── Pi 5 backend: Adafruit PioMatter ──────────────────────────────────────────

class PioMatterBackend(DisplayBackend):
    """
    Pi 5 — uses Adafruit PioMatter library via /dev/pio0 (RP1 PIO blocks).
    Install: pip install adafruit-blinka-raspberry-pi5-piomatter
    Does NOT require sudo.
    """

    def __init__(self, pinout_name: str, n_addr_lines: int = 4):
        try:
            import numpy as np
            import adafruit_blinka_raspberry_pi5_piomatter as piomatter
        except ImportError:
            raise ImportError(
                "PioMatter not installed.\n"
                "Run: pip install adafruit-blinka-raspberry-pi5-piomatter"
            )

        self._np = np

        # Active3BGR = Active3 GPIO pins but with R and B swapped at hardware level.
        # Use this for cheap ₱149 HUB75 adapter boards where R and B are crossed.
        PINOUT_MAP = {
            "active3":    getattr(piomatter.Pinout, "Active3BGR", piomatter.Pinout.Active3),
            "bonnet":     piomatter.Pinout.AdafruitMatrixBonnet,
            "bonnet-pwm": getattr(piomatter.Pinout, "AdafruitMatrixBonnetPWM",
                                  piomatter.Pinout.AdafruitMatrixBonnet),
        }
        pinout = PINOUT_MAP.get(pinout_name, getattr(piomatter.Pinout, "Active3BGR",
                                                      piomatter.Pinout.Active3))

        geometry = piomatter.Geometry(
            width=WIDTH,
            height=HEIGHT,
            n_addr_lines=n_addr_lines,
            rotation=piomatter.Orientation.Normal,
        )
        self._canvas  = Image.new("RGB", (WIDTH, HEIGHT), BLACK)
        self._fb      = np.asarray(self._canvas).copy()
        self._matrix  = piomatter.PioMatter(
            colorspace=piomatter.Colorspace.RGB888Packed,
            pinout=pinout,
            framebuffer=self._fb,
            geometry=geometry,
        )
        log.info("Pi 5 backend — PioMatter  pinout=%s  %dx%d", pinout_name, WIDTH, HEIGHT)

    def show(self, img: Image.Image) -> None:
        self._fb[:] = self._np.asarray(img.convert("RGB"))
        self._matrix.show()

    def clear(self) -> None:
        self._fb[:] = 0
        self._matrix.show()

    def close(self) -> None:
        self._fb[:] = 0
        for _ in range(16):
            self._matrix.show()
            time.sleep(0.02)


# ── Emulator backend: RGBMatrixEmulator (Windows / Mac / Linux, no Pi needed) ─

class EmulatorBackend(DisplayBackend):
    """
    Runs on any OS — opens a browser window simulating the 128×32 matrix.
    Install: pip install RGBMatrixEmulator
    Run:     python display_manager.py --emulator --test
    No sudo, no Pi hardware needed.
    """

    def __init__(self, cols: int = 64, chain: int = 2):
        try:
            from RGBMatrixEmulator import RGBMatrix, RGBMatrixOptions
        except ImportError:
            raise ImportError(
                "RGBMatrixEmulator not installed.\n"
                "Run: pip install RGBMatrixEmulator"
            )

        options = RGBMatrixOptions()
        options.rows         = HEIGHT
        options.cols         = cols
        options.chain_length = chain
        options.brightness   = 80

        self._matrix = RGBMatrix(options=options)
        self._canvas = self._matrix.CreateFrameCanvas()
        log.info("Emulator backend — %dx%d (cols=%d chain=%d) — browser window opening…",
                 WIDTH, HEIGHT, cols, chain)

    def show(self, img: Image.Image) -> None:
        self._canvas.SetImage(img.convert("RGB"))
        self._canvas = self._matrix.SwapOnVSync(self._canvas)

    def clear(self) -> None:
        self._canvas.Clear()
        self._canvas = self._matrix.SwapOnVSync(self._canvas)

    def close(self) -> None:
        self.clear()


# ── Backend factory ───────────────────────────────────────────────────────────

def create_backend(args, pi_model: str) -> DisplayBackend:
    if getattr(args, "emulator", False):
        log.info("Emulator mode → RGBMatrixEmulator backend")
        return EmulatorBackend(cols=args.cols, chain=args.chain or 2)
    elif pi_model == "pi5":
        log.info("Detected Raspberry Pi 5 → led-image-viewer backend (SwapOnVSync)")
        return LedImageViewerBackend(
            viewer_path = os.path.expanduser(
                getattr(args, "viewer", None) or LED_IMAGE_VIEWER_DEFAULT
            ),
            cols  = args.cols,
            chain = args.chain if args.chain > 0 else 2,
        )
    else:
        log.info("Detected Raspberry Pi 4 → ledcat backend")
        return LedcatBackend(
            ledcat_path  = os.path.expanduser(args.ledcat or LEDCAT_DEFAULT),
            slowdown     = args.slowdown,
            mapping      = args.mapping,
            no_hw_pulse  = not args.hardware_pulse,
            cols         = args.cols,
            chain        = args.chain,
            multiplexing = args.multiplexing,
            scan_mode    = args.scan_mode,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  SYSTEM STATE
# ══════════════════════════════════════════════════════════════════════════════

class SystemState:
    def __init__(self):
        self._lock               = threading.Lock()
        self.vehicles_today      = 0
        self.avg_speed           = None
        self.incidents_today     = 0
        self.cameras_online      = 0
        self.cameras_total       = 2
        self.cameras             = []
        self.local_ip            = _get_local_ip()
        self.start_time          = datetime.now()
        self.last_poll_ok        = False
        self.is_test_mode        = False
        # Two separate alert tracks
        self._vehicle_alert_exp  = 0.0          # expires after 8s
        self._incident           = None          # latest incident dict
        self._incident_exp       = 0.0           # expires after 12s

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
            self._incident    = incident
            self._incident_exp = time.monotonic() + hold_secs

    def pop_incident_alert(self) -> Optional[dict]:
        with self._lock:
            if self._incident and time.monotonic() < self._incident_exp:
                return self._incident
            self._incident = None
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


# ══════════════════════════════════════════════════════════════════════════════
#  DATA PROVIDERS
# ══════════════════════════════════════════════════════════════════════════════

try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False


class DataProvider(ABC):
    def __init__(self, state: SystemState):
        self.state = state

    @abstractmethod
    def start(self): ...

    def trigger_test_alert(self):
        pass


class ApiDataProvider(DataProvider):
    POLL_INTERVAL          = 30   # summary + cameras
    INCIDENT_POLL_INTERVAL = 2    # incidents — drives LED reaction time

    def __init__(self, state: SystemState, base_url: str):
        super().__init__(state)
        self._base             = base_url.rstrip("/")
        self._last_incident_id = None
        self._first_poll       = True

    def start(self):
        threading.Thread(target=self._summary_loop,  daemon=True, name="led-summary").start()
        threading.Thread(target=self._incident_loop, daemon=True, name="led-incidents").start()
        log.info("API provider started — %s", self._base)

    def trigger_test_incident(self):
        pass  # no-op for live provider

    def trigger_test_vehicle(self):
        pass

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
            raise RuntimeError("requests not installed")
        r = _requests.get(f"{self._base}/api/analytics/summary", timeout=5)
        r.raise_for_status()
        data     = r.json().get("data", {})
        prev_veh = self.state.vehicles_today
        self.state.update_summary(data)
        # Trigger vehicle alert when new vehicles arrive (skip first poll)
        if not self._first_poll and data.get("vehicles_today", 0) > prev_veh:
            self.state.push_vehicle_alert(hold_secs=8.0)
        self._first_poll = False
        r = _requests.get(f"{self._base}/api/cameras", timeout=5)
        r.raise_for_status()
        self.state.update_cameras(r.json().get("data", []))

    def _poll_incidents(self):
        if not _REQUESTS_OK:
            return
        r = _requests.get(f"{self._base}/api/incidents",
                          params={"status": "active", "limit": 1}, timeout=3)
        r.raise_for_status()
        items = r.json().get("data", [])
        if items:
            inc    = items[0]
            inc_id = inc.get("id")
            if inc_id != self._last_incident_id:
                self._last_incident_id = inc_id
                self.state.push_incident_alert(inc, hold_secs=12.0)
                log.warning("INCIDENT: %s (%s)", inc.get("incident_type"), inc.get("severity"))


class TestDataProvider(DataProvider):
    _FAKE_INCIDENTS = [
        {"incident_type": "speeding",   "severity": "high",
         "title": "Test Speeding",   "camera_name": "Camera A"},
        {"incident_type": "crash",      "severity": "critical",
         "title": "Test Crash",      "camera_name": "Camera B"},
        {"incident_type": "congestion", "severity": "medium",
         "title": "Test Congestion", "camera_name": "Camera A"},
    ]

    def __init__(self, state: SystemState):
        super().__init__(state)
        self._idx = 0

    def start(self):
        self.state.update_summary({"vehicles_today": 0, "average_speed": None,
                                   "incidents_today": 0, "cameras_online": 2,
                                   "cameras_total":   2})
        self.state.update_cameras([
            {"id": "CAM-A-001", "name": "Camera A", "status": "online",
             "fps": 30, "resolution": "1920x1080"},
            {"id": "CAM-B-002", "name": "Camera B", "status": "online",
             "fps": 30, "resolution": "1920x1080"},
        ])
        self.state.last_poll_ok = True
        self.state.is_test_mode = True
        threading.Thread(target=self._cycle, daemon=True).start()
        log.info("Test data provider started — no API calls")

    def trigger_test_incident(self):
        inc = dict(self._FAKE_INCIDENTS[self._idx % len(self._FAKE_INCIDENTS)])
        self.state.push_incident_alert(inc, hold_secs=10.0)
        self._idx += 1
        log.info("Test incident triggered")

    def trigger_test_vehicle(self):
        self.state.push_vehicle_alert(hold_secs=8.0)
        log.info("Test vehicle alert triggered")

    # Keep legacy name so --trigger-alert still works
    def trigger_test_alert(self):
        self.trigger_test_incident()

    def _cycle(self):
        """Auto-cycle: vehicle every 15s, incident every 30s."""
        count = 0
        while True:
            time.sleep(15)
            count += 1
            if count % 2 == 0:
                self.trigger_test_incident()
            else:
                self.trigger_test_vehicle()


# ══════════════════════════════════════════════════════════════════════════════
#  SCREEN RENDERERS  (identical on Pi 4 and Pi 5 — pure PIL)
# ══════════════════════════════════════════════════════════════════════════════

def _safe_or_danger(snap: dict) -> tuple:
    if snap["incidents_today"] > 0:
        return "!! DANGER !!", RED
    return "-- SAFE --", GREEN


def render_main_status(state: SystemState) -> Image.Image:
    img, draw = _new_frame()
    snap = state.snapshot()
    _draw_row(draw, 0, f"ROAD SENTINEL  {datetime.now().strftime('%H:%M')}", WHITE)
    status_text, status_col = _safe_or_danger(snap)
    _fill_row(draw, 1, (0, 40, 0) if status_col == GREEN else (60, 0, 0))
    _draw_row(draw, 1, status_text, status_col)
    cams = snap["cameras"]
    if cams:
        cam_a = next((c for c in cams if "A" in c.get("name", "")), None)
        cam_b = next((c for c in cams if "B" in c.get("name", "")), None)
        a_on  = cam_a and cam_a.get("status") == "online"
        b_on  = cam_b and cam_b.get("status") == "online"
        draw.text((1,  ROWS[2]), "A:", font=FONT_SM, fill=GRAY)
        draw.text((9,  ROWS[2]), "ON" if a_on else "OFF", font=FONT_SM, fill=GREEN if a_on else RED)
        draw.text((33, ROWS[2]), "B:", font=FONT_SM, fill=GRAY)
        draw.text((41, ROWS[2]), "ON" if b_on else "OFF", font=FONT_SM, fill=GREEN if b_on else RED)
    else:
        online = snap["cameras_online"]
        _draw_row(draw, 2, f"Cams: {online}/{snap['cameras_total']}",
                  GREEN if online == snap["cameras_total"] else RED)
    draw.text((1,  ROWS[3]), f"Veh: {snap['vehicles_today']:,}", font=FONT_SM, fill=CYAN)
    draw.text((70, ROWS[3]), f"Inc: {snap['incidents_today']}", font=FONT_SM,
              fill=RED if snap["incidents_today"] > 0 else GRAY)
    speed_str = f"Avg: {snap['avg_speed']} km/h" if snap["avg_speed"] else "Avg speed: N/A"
    _draw_row(draw, 4, speed_str, YELLOW)
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
        base  = i * 2 + 1
        st    = cam.get("status", "unknown")
        col   = GREEN if st == "online" else (ORANGE if st == "error" else RED)
        _draw_row(draw, base,     f"{cam.get('name', f'Cam {i+1}')}: {st.upper()}", col)
        fps   = cam.get("fps")
        res   = cam.get("resolution", "")
        _draw_row(draw, base + 1, _trunc(f"  {fps}fps  {res}" if fps else f"  {res}"), GRAY)
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
    _draw_row(draw, 4, f"Cameras:   {online}/{snap['cameras_total']} online",
              GREEN if online == snap["cameras_total"] else RED)
    _draw_row(draw, 5, f"Uptime:    {state.uptime()}", GRAY)
    return img


def render_alert(state: SystemState, incident: dict, flash_phase: int = 0) -> Image.Image:
    img, draw = _new_frame()
    is_test  = incident.get("is_test", False)
    inc_type = str(incident.get("incident_type", "ALERT")).upper()
    severity = str(incident.get("severity", "high")).lower()
    cam_name = str(incident.get("camera_name", "Camera ?"))
    desc     = str(incident.get("description", ""))
    try:
        ts = datetime.fromisoformat(
            str(incident.get("timestamp", "")).replace("Z", "")
        ).strftime("%H:%M:%S")
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

    _fill_row(draw, 0, bg)
    draw.text((1, ROWS[0]), _trunc(f"{'[TEST]' if is_test else '!!'} {inc_type}"),
              font=FONT_SM, fill=fg)
    label_bg = (50, 0, 0) if severity == "critical" and not is_test else BLACK
    _fill_row(draw, 1, label_bg)
    _draw_row(draw, 1, sev_label, RED if (severity == "critical" and not is_test) else sev_col)
    _draw_row(draw, 2, cam_name, WHITE)
    _draw_row(draw, 3, _trunc(desc or "See dashboard"), GRAY)
    _draw_row(draw, 4, ts, GRAY)
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
    _draw_row(draw, 4, _trunc(state.local_ip), GRAY)
    _draw_row(draw, 5, f"Up: {state.uptime()}", GRAY)
    return img


def render_test_static(state: SystemState) -> Image.Image:
    img, draw = _new_frame()
    snap = state.snapshot()
    _draw_row(draw, 0, "ROAD SENTINEL", WHITE)
    draw.text((90, ROWS[0]), "[TEST]", font=FONT_SM, fill=AMBER)
    _fill_row(draw, 1, (0, 40, 0))
    _draw_row(draw, 1, "-- SAFE -- (simulated)", GREEN)
    cams = snap["cameras"]
    if cams:
        cam_a = next((c for c in cams if "A" in c.get("name", "")), cams[0] if cams else None)
        cam_b = next((c for c in cams if "B" in c.get("name", "")), cams[1] if len(cams) > 1 else None)
        a_on  = cam_a and cam_a.get("status") == "online"
        b_on  = cam_b and cam_b.get("status") == "online"
        draw.text((1,  ROWS[2]), "A:", font=FONT_SM, fill=GRAY)
        draw.text((9,  ROWS[2]), "ON" if a_on else "OFF", font=FONT_SM, fill=GREEN if a_on else RED)
        draw.text((33, ROWS[2]), "B:", font=FONT_SM, fill=GRAY)
        draw.text((41, ROWS[2]), "ON" if b_on else "OFF", font=FONT_SM, fill=GREEN if b_on else RED)
        draw.text((65, ROWS[2]), "SIMULATED", font=FONT_SM, fill=AMBER)
    draw.text((1,  ROWS[3]), f"Veh: {snap['vehicles_today']:,}", font=FONT_SM, fill=CYAN)
    draw.text((70, ROWS[3]), f"{snap['avg_speed']} km/h" if snap["avg_speed"] else "N/A",
              font=FONT_SM, fill=YELLOW)
    _draw_row(draw, 4, snap["local_ip"], GRAY)
    _draw_row(draw, 5, f"Up: {state.uptime()}", GRAY)
    return img


def render_color_bars(phase: int) -> Image.Image:
    img, draw = _new_frame()
    bars  = [RED, ORANGE, YELLOW, GREEN, CYAN, BLUE, WHITE]
    bar_w = WIDTH // len(bars)
    intensity = [255, 180, 80][min(phase // 2, 2)]
    for i, color in enumerate(bars):
        x0 = i * bar_w
        x1 = x0 + bar_w - 1
        draw.rectangle([x0, 0, x1, HEIGHT - 1],
                       fill=tuple(int(c * intensity // 255) for c in color))
    return img


# ── Road warning state screens ────────────────────────────────────────────────

def render_vacant_road(state: SystemState) -> Image.Image:
    img, draw = _new_frame()
    _fill_row_px(draw, 0, HEIGHT - 1, (0, 55, 0))
    _draw_centered_in(draw, 0, 0, WIDTH - 1, HEIGHT - 1,
                      "VACANT ROAD", FONT_LARGE, GREEN)
    return img


def render_vehicle_incoming(state: SystemState, flash_phase: int = 0) -> Image.Image:
    img, draw = _new_frame()
    bg = ORANGE if flash_phase == 0 else (150, 65, 0)
    _fill_row_px(draw, 0, HEIGHT - 1, bg)
    # VEHICLE / INCOMING fills upper portion; bottom 7px reserved for instruction
    _draw_two_lines_centered(draw, 0, HEIGHT - 8,
                             "VEHICLE", FONT_LARGE,
                             "INCOMING", FONT_MED, WHITE)
    _draw_centered_in(draw, 0, HEIGHT - 7, WIDTH - 1, HEIGHT - 1,
                      "SLOW DOWN", FONT_XS6, YELLOW)
    return img


def render_slow_down(state: SystemState, flash_phase: int = 0) -> Image.Image:
    img, draw = _new_frame()
    bg = RED if flash_phase == 0 else (110, 0, 0)
    _fill_row_px(draw, 0, HEIGHT - 1, bg)
    _draw_centered_in(draw, 0, 0, WIDTH - 1, HEIGHT - 1,
                      "SLOW DOWN", FONT_LARGE, WHITE)
    return img


def render_incident_ahead(state: SystemState, flash_phase: int = 0) -> Image.Image:
    """Incident confirmed — magenta flashing, 'INCIDENT / AHEAD'."""
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


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

TICK = 0.04   # 25 fps — ledcat reads at ~28 fps so pipe stays near-empty;
              # 40 ms intervals stay below Pi 5 RP1 cold-start threshold


def _blank_transition(backend: LedcatBackend, hold: float = 0.3) -> None:
    """Write BLACK at TICK rate for hold seconds, then log ready."""
    log.info("Clearing panel...")
    end = time.monotonic() + hold
    while time.monotonic() < end:
        backend._write(_BLACK_FRAME)
        time.sleep(TICK)
    log.info("Panel cleared")


def run(backend: DisplayBackend, state: SystemState):
    prev_state_key = None
    log.info("Display loop started")

    log.info("Clearing panel...")
    for _ in range(16):
        backend.clear()
        time.sleep(0.02)
    end = time.monotonic() + 0.5
    while time.monotonic() < end:
        backend.clear()
        time.sleep(TICK)
    log.info("Panel cleared — starting display")

    # Pre-render frames once — passed by identity so show() skips re-snap on repeat
    _img_slow_down = render_slow_down(state, flash_phase=0)
    _img_vehicle   = render_vehicle_incoming(state, flash_phase=0)
    _img_incident  = render_incident_ahead(state, flash_phase=0)

    while True:
        # Priority: INCIDENT AHEAD > VEHICLE INCOMING > SLOW DOWN (default)
        incident = state.pop_incident_alert()
        if incident:
            state_key = "incident"
            frame_img = _img_incident
        elif state.has_vehicle_alert():
            state_key = "vehicle"
            frame_img = _img_vehicle
        else:
            state_key = "slow_down"
            frame_img = _img_slow_down

        if state_key != prev_state_key:
            prev_state_key = state_key
            log.info("Now showing: %s", state_key.replace("_", " ").upper())

        backend.show(frame_img)
        time.sleep(TICK)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Road Sentinel HUB75 128×32 — Pi 4 and Pi 5 unified",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Auto-detects Pi model at startup:
  Pi 5 (/dev/pio0 exists) → ledcat --led-rp1-rio=1 (writes at 25 fps)
  Pi 4 (default)          → ledcat /dev/mem GPIO

Examples:
  sudo python3 display_manager.py --test           # Pi 4 test mode
       python3 display_manager.py --test           # Pi 5 test mode
       python3 display_manager.py --api http://192.168.8.50:3001
       python3 display_manager.py --pi 4           # force Pi 4 backend
       python3 display_manager.py --pi 5           # force Pi 5 backend
        """
    )
    # ── Common ──
    parser.add_argument("--test",           action="store_true",
                        help="Test mode: fake data, no API")
    parser.add_argument("--api",            default="http://localhost:3001",
                        help="Node Service URL (default: http://localhost:3001)")
    parser.add_argument("--trigger-alert",    action="store_true",
                        help="Fire one test incident alert immediately (alias for --trigger-incident)")
    parser.add_argument("--trigger-incident", action="store_true",
                        help="Fire one test INCIDENT AHEAD alert immediately")
    parser.add_argument("--trigger-vehicle",  action="store_true",
                        help="Fire one test VEHICLE INCOMING alert immediately")
    parser.add_argument("--pi",             choices=["4", "5"], default=None,
                        help="Force Pi model (default: auto-detect)")
    parser.add_argument("--emulator",       action="store_true",
                        help="Run in emulator mode (Windows/Mac/Linux, no Pi needed)")

    # ── Pi 4 / ledcat ──
    parser.add_argument("--ledcat",         default=None,
                        help=f"ledcat binary path (Pi 4, default: {LEDCAT_DEFAULT})")
    parser.add_argument("--viewer",         default=None,
                        help=f"led-image-viewer binary path (Pi 5, default: {LED_IMAGE_VIEWER_DEFAULT})")
    parser.add_argument("--slowdown",       type=int, default=4,
                        help="GPIO slowdown (Pi 4, default: 4)")
    parser.add_argument("--cols",           type=int, default=64,
                        help="Columns per panel (Pi 4, default: 64)")
    parser.add_argument("--chain",          type=int, default=0,
                        help="Chained panels (Pi 4, default: auto)")
    parser.add_argument("--multiplexing",   type=int, default=0,
                        help="Panel multiplexing type (Pi 4, default: 0)")
    parser.add_argument("--scan-mode",      type=int, default=0, choices=[0, 1],
                        help="Scan mode (Pi 4, default: 0)")
    parser.add_argument("--mapping",        default="regular",
                        choices=["regular", "adafruit-hat", "adafruit-hat-pwm"],
                        help="GPIO mapping (Pi 4, default: regular)")
    parser.add_argument("--hardware-pulse", action="store_true", default=False,
                        help="Enable HW pulse (Pi 4, only after disabling snd_bcm2835)")

    # ── Pi 5 / PioMatter ──
    parser.add_argument("--pinout",         default="active3",
                        choices=["active3", "bonnet", "bonnet-pwm"],
                        help="PioMatter pinout (Pi 5, default: active3 = ₱149 adapter)")
    parser.add_argument("--addr-lines",     type=int, default=4,
                        help="Address lines (Pi 5, default: 4 for 32-row panels)")

    args = parser.parse_args()

    if args.emulator:
        pi_model = "emulator"
        log.info("Mode: EMULATOR (browser window)")
    else:
        pi_model = args.pi if args.pi else _detect_pi()
        pi_model = f"pi{pi_model}" if not pi_model.startswith("pi") else pi_model
        log.info("Pi model: %s (%s)", pi_model, "forced" if args.pi else "auto-detected")

    try:
        backend = create_backend(args, pi_model)
    except (FileNotFoundError, ImportError) as exc:
        parser.error(str(exc))

    state    = SystemState()
    provider = TestDataProvider(state) if args.test else ApiDataProvider(state, args.api)
    provider.start()

    if args.trigger_alert or args.trigger_incident:
        time.sleep(0.5)
        provider.trigger_test_incident()
    if args.trigger_vehicle:
        time.sleep(0.5)
        provider.trigger_test_vehicle()

    # SIGTERM (sent by systemctl stop) must be converted to KeyboardInterrupt
    # so the finally block runs and the panel gets blanked.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))

    try:
        run(backend, state)
    except KeyboardInterrupt:
        log.info("Stopped by user or SIGTERM")
    finally:
        backend.close()
        log.info("Display cleared")


if __name__ == "__main__":
    main()
