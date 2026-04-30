#!/usr/bin/env python3
"""
Road Sentinel — Camera Frame Sender
Captures RTSP stream, POSTs JPEG frames to the AI service,
then forwards detections and incidents to the Node service for DB storage.
Also pushes one JPEG/second to Node for the MJPEG web stream.

Works on both Raspberry Pi 4 and Pi 5 (pure OpenCV + aiohttp, no Pi-specific libs).

Usage:
    python3 camera_sender.py \
        --camera-id CAM-A-001 \
        --rtsp rtsp://192.168.8.104:554/cam/realmonitor?channel=1&subtype=1 \
        --ai  http://192.168.8.50:8000 \
        --node http://192.168.8.50:3001
"""

import argparse
import asyncio
import logging
import signal
import time
from collections import deque
from typing import Optional

import aiohttp
import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cam_sender")

# ── Config ────────────────────────────────────────────────────────────────────
JPEG_QUALITY             = 75
TARGET_FPS               = 30
FRAME_INTERVAL           = 1.0 / TARGET_FPS
RECONNECT_WAIT           = 3.0
AI_TIMEOUT               = 4.0
NODE_TIMEOUT             = 5.0
DETECTION_WRITE_INTERVAL = 1.0   # write at most 1 detection per second to Node
INCIDENT_DEDUP_WINDOW    = 30.0  # suppress repeat incident type within this window (s)
FRAME_PUSH_INTERVAL      = 1.0   # push MJPEG frame to Node at most 1/second

INCIDENT_TITLES = {
    "speeding":        "Speeding Detected",
    "crash":           "Crash / Collision Detected",
    "wrong_way":       "Wrong-Way Vehicle Detected",
    "stopped_vehicle": "Stopped Vehicle Detected",
    "congestion":      "Traffic Congestion",
    "illegal_parking": "Illegal Parking",
}


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
        fps = (len(window) - 1) / (window[-1] - window[0]) if len(window) >= 2 else 0.0
        log.info("[%s] sent=%d  errors=%d  dropped=%d  fps=%.1f",
                 self.camera_id, self.sent, self.errors, self.dropped, fps)


# ── Camera config fetch ────────────────────────────────────────────────────────
async def fetch_camera_config(
    session: aiohttp.ClientSession, node_url: str, camera_id: str
) -> dict:
    """Fetch per-camera settings (pixels_per_meter, speed_limit, detection_confidence) from Node."""
    try:
        async with session.get(
            f"{node_url.rstrip('/')}/api/cameras/{camera_id}",
            timeout=aiohttp.ClientTimeout(total=5.0),
        ) as resp:
            j = await resp.json()
            return j.get("data", {})
    except Exception as exc:
        log.warning("[%s] Could not fetch camera config from Node: %s", camera_id, exc)
        return {}


# ── Node service forwarder ────────────────────────────────────────────────────
class NodeForwarder:
    """
    Receives AI detection results and forwards them to the Node service.
    - Detections: throttled to 1 write per second.
    - Incidents:  forwarded immediately with a 30s dedup window per incident type.
    - Frames:     pushes latest JPEG to Node for MJPEG web stream (1/second).
    """

    def __init__(
        self,
        node_url: str,
        camera_id: str,
        session: aiohttp.ClientSession,
        confidence_threshold: float = 0.6,
    ):
        self._base        = node_url.rstrip("/")
        self._camera_id   = camera_id
        self._session     = session
        self._conf_thresh = confidence_threshold
        self._last_det    = 0.0
        self._last_frame  = 0.0
        self._inc_last: dict[str, float] = {}

    async def handle(self, result: dict) -> None:
        detections = result.get("detections", [])
        incidents  = result.get("incidents",  [])
        now        = time.monotonic()

        # Throttled detection write — one per second, highest confidence first
        if detections and (now - self._last_det) >= DETECTION_WRITE_INTERVAL:
            self._last_det = now
            best = max(detections, key=lambda d: d.get("confidence", 0))
            await self._post_detection(best)

        # Incident forwarding — filter by confidence, deduplicated per type
        for inc in incidents:
            inc_type = str(inc.get("type", "other"))
            conf     = float(inc.get("confidence", 1.0))
            if conf < self._conf_thresh:
                log.debug("[%s] Skipping %s (conf %.2f < threshold %.2f)",
                          self._camera_id, inc_type, conf, self._conf_thresh)
                continue
            if (now - self._inc_last.get(inc_type, 0.0)) >= INCIDENT_DEDUP_WINDOW:
                self._inc_last[inc_type] = now
                await self._post_incident(inc)

    async def push_frame(self, jpeg_bytes: bytes) -> None:
        """Push latest JPEG to Node so it can serve the MJPEG web stream."""
        now = time.monotonic()
        if now - self._last_frame < FRAME_PUSH_INTERVAL:
            return
        self._last_frame = now
        try:
            async with self._session.put(
                f"{self._base}/api/cameras/{self._camera_id}/frame",
                data=jpeg_bytes,
                headers={"Content-Type": "application/octet-stream"},
                timeout=aiohttp.ClientTimeout(total=2.0),
            ) as resp:
                pass  # 204 No Content expected
        except Exception:
            pass  # best-effort — don't spam logs

    async def _post_detection(self, det: dict) -> None:
        bbox = det.get("bbox", {})
        body = {
            "camera_id":    self._camera_id,
            "vehicle_type": det.get("class", "unknown"),
            "speed":        det.get("speed"),
            "confidence":   det.get("confidence", 0),
            "bbox_x":       int(bbox.get("x", 0)),
            "bbox_y":       int(bbox.get("y", 0)),
            "bbox_width":   int(bbox.get("width", 0)),
            "bbox_height":  int(bbox.get("height", 0)),
        }
        try:
            async with self._session.post(
                f"{self._base}/api/detections",
                json=body,
                timeout=aiohttp.ClientTimeout(total=NODE_TIMEOUT),
            ) as resp:
                if resp.status not in (200, 201):
                    log.debug("[%s] Detection POST %d", self._camera_id, resp.status)
        except Exception as exc:
            log.debug("[%s] Detection forward error: %s", self._camera_id, exc)

    async def _post_incident(self, inc: dict) -> None:
        inc_type = str(inc.get("type", "other"))
        title    = INCIDENT_TITLES.get(inc_type, inc_type.replace("_", " ").title())
        body = {
            "camera_id":     self._camera_id,
            "incident_type": inc_type,
            "severity":      inc.get("severity", "medium"),
            "title":         title,
            "description":   inc.get("description", ""),
            "confidence":    inc.get("confidence"),
            "status":        "active",
        }
        try:
            async with self._session.post(
                f"{self._base}/api/incidents",
                json=body,
                timeout=aiohttp.ClientTimeout(total=NODE_TIMEOUT),
            ) as resp:
                if resp.status in (200, 201):
                    log.warning("[%s] INCIDENT → Node: %s (%s)",
                                self._camera_id, inc_type, inc.get("severity"))
                else:
                    log.debug("[%s] Incident POST %d", self._camera_id, resp.status)
        except Exception as exc:
            log.debug("[%s] Incident forward error: %s", self._camera_id, exc)


# ── Frame capture (synchronous — runs in thread) ─────────────────────────────
def open_capture(rtsp_url: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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
    pixels_per_meter: float = 0.0,
    speed_limit: float = 0.0,
    confidence_threshold: Optional[float] = None,
) -> dict:
    data = aiohttp.FormData()
    data.add_field("camera_id", camera_id)
    data.add_field("image", jpeg_bytes, filename="frame.jpg", content_type="image/jpeg")
    if pixels_per_meter > 0:
        data.add_field("pixels_per_meter", str(pixels_per_meter))
    if speed_limit > 0:
        data.add_field("speed_limit", str(speed_limit))
    if confidence_threshold is not None:
        data.add_field("confidence_threshold", str(confidence_threshold))
    async with session.post(
        f"{ai_url}/api/detect",
        data=data,
        timeout=aiohttp.ClientTimeout(total=AI_TIMEOUT),
    ) as resp:
        resp.raise_for_status()
        return await resp.json()


def _log_detections(camera_id: str, result: dict):
    incidents = result.get("incidents", [])
    if incidents:
        log.warning("[%s] INCIDENT (not forwarded — no --node): %s", camera_id, incidents)
    vehicles = result.get("detections", [])
    if vehicles:
        log.debug("[%s] %d vehicle(s) detected", camera_id, len(vehicles))


# ── Main loop ─────────────────────────────────────────────────────────────────
async def run(
    camera_id: str,
    rtsp_url: str,
    ai_url: str,
    node_url: Optional[str],
    pixels_per_meter: float,
    speed_limit: float,
    confidence: float,
):
    stats    = Stats(camera_id)
    shutdown = asyncio.Event()

    def _handle_signal(*_):
        log.info("[%s] Shutdown signal received", camera_id)
        shutdown.set()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, _handle_signal)
    loop.add_signal_handler(signal.SIGINT,  _handle_signal)

    ai_connector   = aiohttp.TCPConnector(limit=2, keepalive_timeout=30)
    node_connector = aiohttp.TCPConnector(limit=4, keepalive_timeout=30)

    async with aiohttp.ClientSession(connector=ai_connector) as ai_session:
        async with aiohttp.ClientSession(connector=node_connector) as node_session:
            forwarder: Optional[NodeForwarder] = None
            if node_url:
                # Fetch per-camera config from Node DB (overrides CLI defaults)
                cam_cfg = await fetch_camera_config(node_session, node_url, camera_id)
                if cam_cfg.get("pixels_per_meter"):
                    pixels_per_meter = float(cam_cfg["pixels_per_meter"])
                if cam_cfg.get("speed_limit"):
                    speed_limit = float(cam_cfg["speed_limit"])
                if cam_cfg.get("detection_confidence"):
                    confidence = float(cam_cfg["detection_confidence"])

                forwarder = NodeForwarder(node_url, camera_id, node_session,
                                          confidence_threshold=confidence)
                log.info("[%s] Node forwarding → %s  ppm=%.1f  limit=%.0f  conf=%.2f",
                         camera_id, node_url, pixels_per_meter, speed_limit, confidence)
            else:
                log.warning("[%s] --node not set — detections NOT saved to DB", camera_id)

            while not shutdown.is_set():
                # ── Open / re-open camera ────────────────────────────────────
                try:
                    cap = await asyncio.get_event_loop().run_in_executor(
                        None, open_capture, rtsp_url
                    )
                except Exception as exc:
                    log.error("[%s] Camera open failed: %s — retrying in %.0fs",
                              camera_id, exc, RECONNECT_WAIT)
                    await asyncio.sleep(RECONNECT_WAIT)
                    continue

                log.info("[%s] Streaming → AI:%s", camera_id, ai_url)

                try:
                    while not shutdown.is_set():
                        frame_start = time.monotonic()

                        ret, frame = await asyncio.get_event_loop().run_in_executor(
                            None, cap.read
                        )
                        if not ret or frame is None:
                            log.warning("[%s] Frame read failed — reconnecting", camera_id)
                            break

                        try:
                            jpeg = await asyncio.get_event_loop().run_in_executor(
                                None, encode_jpeg, frame
                            )
                        except Exception as exc:
                            log.warning("[%s] Encode error: %s", camera_id, exc)
                            stats.record_drop()
                            continue

                        # Push MJPEG frame to Node (best-effort, throttled to 1/s)
                        if forwarder:
                            asyncio.create_task(forwarder.push_frame(jpeg))

                        try:
                            result = await post_frame(
                                ai_session, ai_url, camera_id, jpeg,
                                pixels_per_meter=pixels_per_meter,
                                speed_limit=speed_limit,
                                confidence_threshold=confidence,
                            )
                            stats.record_send(True)
                            if forwarder:
                                await forwarder.handle(result)
                            else:
                                _log_detections(camera_id, result)
                        except Exception as exc:
                            log.debug("[%s] AI POST error: %s", camera_id, exc)
                            stats.record_send(False)

                        stats.log_if_due()

                        elapsed = time.monotonic() - frame_start
                        sleep_t = FRAME_INTERVAL - elapsed
                        if sleep_t > 0:
                            await asyncio.sleep(sleep_t)
                        else:
                            stats.record_drop()

                finally:
                    cap.release()
                    log.info("[%s] Camera released", camera_id)

                if not shutdown.is_set():
                    log.info("[%s] Reconnecting in %.0fs…", camera_id, RECONNECT_WAIT)
                    await asyncio.sleep(RECONNECT_WAIT)

    log.info("[%s] Stopped. sent=%d errors=%d dropped=%d",
             camera_id, stats.sent, stats.errors, stats.dropped)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Road Sentinel camera frame sender — Pi 4 & Pi 5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Camera A (Pi 4) — full pipeline
  python3 camera_sender.py \\
      --camera-id CAM-A-001 \\
      --rtsp rtsp://192.168.8.104:554/cam/realmonitor?channel=1&subtype=1 \\
      --ai   http://192.168.8.50:8000 \\
      --node http://192.168.8.50:3001

  # Camera B (Pi 5) — full pipeline
  python3 camera_sender.py \\
      --camera-id CAM-B-002 \\
      --rtsp rtsp://192.168.8.108:554/cam/realmonitor?channel=1&subtype=1 \\
      --ai   http://192.168.8.50:8000 \\
      --node http://192.168.8.50:3001

IMPORTANT: --camera-id must match cameras.id in the MySQL table.
Camera config (pixels_per_meter, speed_limit, detection_confidence) is fetched
from Node at startup and overrides the CLI defaults.
        """
    )
    parser.add_argument("--camera-id", required=True,
                        help="Camera ID — must match cameras.id in MySQL")
    parser.add_argument("--rtsp",      required=True,
                        help="RTSP URL of the IP camera")
    parser.add_argument("--ai",        default="http://localhost:8000",
                        help="AI service base URL (default: http://localhost:8000)")
    parser.add_argument("--node",      default=None,
                        help="Node service URL for DB forwarding (e.g. http://localhost:3001)")
    parser.add_argument("--fps",       type=int,   default=TARGET_FPS,
                        help=f"Target capture FPS (default: {TARGET_FPS})")
    parser.add_argument("--quality",   type=int,   default=JPEG_QUALITY,
                        help=f"JPEG quality 1-100 (default: {JPEG_QUALITY})")
    parser.add_argument("--ppm",       type=float, default=25.5,
                        help="Pixels per meter for speed calc (default: 25.5, overridden by DB)")
    parser.add_argument("--speed-limit", type=float, default=60.0,
                        help="Speed limit km/h for speeding detection (default: 60, overridden by DB)")
    parser.add_argument("--confidence", type=float, default=0.6,
                        help="Min incident confidence to forward (default: 0.6, overridden by DB)")
    args = parser.parse_args()

    global TARGET_FPS, FRAME_INTERVAL, JPEG_QUALITY
    TARGET_FPS     = args.fps
    FRAME_INTERVAL = 1.0 / TARGET_FPS
    JPEG_QUALITY   = args.quality

    log.info("Road Sentinel Camera Sender")
    log.info("  camera_id : %s", args.camera_id)
    log.info("  rtsp      : %s", args.rtsp)
    log.info("  ai_url    : %s", args.ai)
    log.info("  node_url  : %s", args.node or "(disabled)")
    log.info("  target_fps: %d", TARGET_FPS)
    log.info("  jpeg_q    : %d%%", JPEG_QUALITY)
    log.info("  ppm       : %.1f (may be overridden by Node DB)", args.ppm)
    log.info("  speed_lim : %.0f km/h (may be overridden by Node DB)", args.speed_limit)
    log.info("  confidence: %.2f (may be overridden by Node DB)", args.confidence)

    asyncio.run(run(
        camera_id        = args.camera_id,
        rtsp_url         = args.rtsp,
        ai_url           = args.ai,
        node_url         = args.node,
        pixels_per_meter = args.ppm,
        speed_limit      = args.speed_limit,
        confidence       = args.confidence,
    ))


if __name__ == "__main__":
    main()
