#!/usr/bin/env python3
"""
Road Sentinel — Camera Frame Sender
Captures RTSP stream and POSTs JPEG frames to the AI service at 30fps.

Works on both Raspberry Pi 4 and Pi 5 (pure OpenCV + aiohttp, no Pi-specific libs).

Usage:
    python3 camera_sender.py --camera-id cam_a --rtsp rtsp://192.168.8.104:554/... --ai http://192.168.8.50:8000
    python3 camera_sender.py --camera-id cam_b --rtsp rtsp://192.168.8.108:554/... --ai http://192.168.8.50:8000
"""

import argparse
import asyncio
import logging
import signal
import time
from collections import deque

import aiohttp
import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cam_sender")

# ── Config ────────────────────────────────────────────────────────────────────
JPEG_QUALITY   = 75    # 75% = ~30–50 KB/frame — good balance for LAN
TARGET_FPS     = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS   # 33.3 ms
RECONNECT_WAIT = 3.0                 # seconds before reconnecting camera
POST_TIMEOUT   = 4.0                 # seconds before giving up on one POST

# ── Stats ─────────────────────────────────────────────────────────────────────
class Stats:
    def __init__(self, camera_id: str):
        self.camera_id   = camera_id
        self.sent        = 0
        self.errors      = 0
        self.dropped     = 0
        self._fps_window = deque(maxlen=90)
        self._last_log   = time.monotonic()

    def record_send(self, ok: bool):
        if ok:
            self.sent += 1
            self._fps_window.append(time.monotonic())
        else:
            self.errors += 1

    def record_drop(self):
        self.dropped += 1

    def log_if_due(self):
        now = time.monotonic()
        if now - self._last_log < 10:
            return
        self._last_log = now
        window = list(self._fps_window)
        if len(window) >= 2:
            fps = (len(window) - 1) / (window[-1] - window[0])
        else:
            fps = 0.0
        log.info(
            "[%s] sent=%d  errors=%d  dropped=%d  fps=%.1f",
            self.camera_id, self.sent, self.errors, self.dropped, fps,
        )


# ── Frame capture (synchronous — runs in thread) ─────────────────────────────
def open_capture(rtsp_url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)    # keep buffer minimal — avoid stale frames
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open RTSP stream: {rtsp_url}")
    log.info("Opened stream: %s", rtsp_url)
    return cap


def encode_jpeg(frame) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


# ── AI service POST ───────────────────────────────────────────────────────────
async def post_frame(
    session: aiohttp.ClientSession,
    ai_url: str,
    camera_id: str,
    jpeg_bytes: bytes,
) -> dict:
    """POST one JPEG frame to /api/detect. Returns parsed JSON."""
    data = aiohttp.FormData()
    data.add_field("camera_id", camera_id)
    data.add_field(
        "image",
        jpeg_bytes,
        filename="frame.jpg",
        content_type="image/jpeg",
    )
    async with session.post(
        f"{ai_url}/api/detect",
        data=data,
        timeout=aiohttp.ClientTimeout(total=POST_TIMEOUT),
    ) as resp:
        resp.raise_for_status()
        return await resp.json()


# ── Main loop ─────────────────────────────────────────────────────────────────
async def run(camera_id: str, rtsp_url: str, ai_url: str):
    stats    = Stats(camera_id)
    shutdown = asyncio.Event()

    def _handle_signal(*_):
        log.info("[%s] Shutdown signal received", camera_id)
        shutdown.set()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, _handle_signal)
    loop.add_signal_handler(signal.SIGINT,  _handle_signal)

    connector = aiohttp.TCPConnector(limit=2, keepalive_timeout=30)
    async with aiohttp.ClientSession(connector=connector) as session:

        while not shutdown.is_set():
            # ── Open / re-open camera ─────────────────────────────────────
            try:
                cap = await asyncio.get_event_loop().run_in_executor(
                    None, open_capture, rtsp_url
                )
            except Exception as exc:
                log.error("[%s] Camera open failed: %s — retrying in %.0fs",
                          camera_id, exc, RECONNECT_WAIT)
                await asyncio.sleep(RECONNECT_WAIT)
                continue

            log.info("[%s] Streaming started → %s", camera_id, ai_url)

            try:
                while not shutdown.is_set():
                    frame_start = time.monotonic()

                    # Grab frame in thread (blocking call)
                    ret, frame = await asyncio.get_event_loop().run_in_executor(
                        None, cap.read
                    )

                    if not ret or frame is None:
                        log.warning("[%s] Frame read failed — reconnecting", camera_id)
                        break  # reconnect

                    # Encode JPEG in thread
                    try:
                        jpeg = await asyncio.get_event_loop().run_in_executor(
                            None, encode_jpeg, frame
                        )
                    except Exception as exc:
                        log.warning("[%s] Encode error: %s", camera_id, exc)
                        stats.record_drop()
                        continue

                    # POST to AI service (async — doesn't block frame capture)
                    try:
                        result = await post_frame(session, ai_url, camera_id, jpeg)
                        stats.record_send(True)
                        _handle_detections(camera_id, result)
                    except Exception as exc:
                        log.debug("[%s] POST error: %s", camera_id, exc)
                        stats.record_send(False)

                    stats.log_if_due()

                    # Rate-limit to TARGET_FPS
                    elapsed = time.monotonic() - frame_start
                    sleep_t = FRAME_INTERVAL - elapsed
                    if sleep_t > 0:
                        await asyncio.sleep(sleep_t)
                    else:
                        # Frame took too long — don't sleep, just continue
                        stats.record_drop()

            finally:
                cap.release()
                log.info("[%s] Camera released", camera_id)

            if not shutdown.is_set():
                log.info("[%s] Reconnecting in %.0fs…", camera_id, RECONNECT_WAIT)
                await asyncio.sleep(RECONNECT_WAIT)

    log.info("[%s] Stopped. Total sent=%d errors=%d dropped=%d",
             camera_id, stats.sent, stats.errors, stats.dropped)


def _handle_detections(camera_id: str, result: dict):
    """Log significant detections. Extend this to push to Node service."""
    incidents = result.get("incidents", [])
    if incidents:
        log.warning("[%s] INCIDENT: %s", camera_id, incidents)
    vehicles = result.get("detections", [])
    if vehicles:
        log.debug("[%s] %d vehicle(s) detected", camera_id, len(vehicles))


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Road Sentinel camera frame sender — Pi 4 & Pi 5"
    )
    parser.add_argument("--camera-id", required=True,
                        help="Camera identifier, e.g. cam_a or cam_b")
    parser.add_argument("--rtsp",      required=True,
                        help="RTSP URL of the IP camera")
    parser.add_argument("--ai",        default="http://localhost:8000",
                        help="AI service base URL (default: http://localhost:8000)")
    parser.add_argument("--fps",       type=int, default=TARGET_FPS,
                        help=f"Target FPS (default: {TARGET_FPS})")
    parser.add_argument("--quality",   type=int, default=JPEG_QUALITY,
                        help=f"JPEG quality 1-100 (default: {JPEG_QUALITY})")
    args = parser.parse_args()

    global TARGET_FPS, FRAME_INTERVAL, JPEG_QUALITY
    TARGET_FPS     = args.fps
    FRAME_INTERVAL = 1.0 / TARGET_FPS
    JPEG_QUALITY   = args.quality

    log.info("Road Sentinel Camera Sender")
    log.info("  camera_id : %s", args.camera_id)
    log.info("  rtsp      : %s", args.rtsp)
    log.info("  ai_url    : %s", args.ai)
    log.info("  target_fps: %d", TARGET_FPS)
    log.info("  jpeg_q    : %d%%", JPEG_QUALITY)

    asyncio.run(run(args.camera_id, args.rtsp, args.ai))


if __name__ == "__main__":
    main()
