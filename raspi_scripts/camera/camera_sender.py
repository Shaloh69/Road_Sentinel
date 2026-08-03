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
import json
import logging
import os
import signal
import socket
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urlunparse

import aiohttp
import cv2
import socketio as sio_lib

try:
    from onvif import ONVIFCamera  # onvif-zeep — optional, only needed for --ir-auto
    _ONVIF_OK = True
except ImportError:
    _ONVIF_OK = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("cam_sender")

# ── Config ────────────────────────────────────────────────────────────────────
JPEG_QUALITY             = 50
TARGET_FPS               = 30
FRAME_INTERVAL           = 1.0 / TARGET_FPS
RECONNECT_WAIT           = 3.0
AI_TIMEOUT               = 4.0
NODE_TIMEOUT             = 5.0
DETECTION_WRITE_INTERVAL = 1.0   # write at most 1 detection per second to Node
INCIDENT_DEDUP_WINDOW    = 30.0  # suppress repeat incident type within this window (s)
FRAME_PUSH_INTERVAL      = 1.0 / 30  # push MJPEG frame to Node at up to 30 FPS

# Auto-discovery
DISCOVERY_AFTER_FAILURES = 3     # consecutive open failures before triggering discovery
ONVIF_MULTICAST          = "239.255.255.250"
ONVIF_PROBE_PORT         = 3702
ONVIF_TIMEOUT            = 3.0   # seconds to wait for WS-Discovery responses
RTSP_PORT_SCAN_TIMEOUT   = 0.5   # seconds per port-554 probe
RTSP_CAP_TEST_TIMEOUT    = 6.0   # seconds to test an RTSP URL with cv2

INCIDENT_TITLES = {
    "speeding":        "Speeding Detected",
    "crash":           "Crash / Collision Detected",
    "wrong_way":       "Wrong-Way Vehicle Detected",
    "stopped_vehicle": "Stopped Vehicle Detected",
    "congestion":      "Traffic Congestion",
    "illegal_parking": "Illegal Parking",
}


# ── Adaptive detection sampling (Phase 2) ─────────────────────────────────────
# Throttles AI-service calls based on recent scene activity instead of a fixed
# rate, to cut AI-service/network load while the road is empty — the common
# case for most of the day at a low-traffic blind curve. Never affects the
# live-view frame push (that stays at full rate regardless — see ai_task/
# frame_push_loop in run()), and always samples every eligible frame while a
# vehicle has been seen recently, so speed-tracking continuity is unaffected.
class AdaptiveSampler:
    ACTIVE_WINDOW_SECS = 5.0     # sample every frame if a vehicle was seen this recently
    IDLE_TIERS = [               # (idle_secs_threshold, sample_every_n_frames)
        (30.0, 3),
        (120.0, 6),
        (float("inf"), 10),
    ]

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._last_vehicle_at = time.monotonic()
        self._frames_since_ai = 0

    def note_result(self, result: dict) -> None:
        if result.get("detections"):
            self._last_vehicle_at = time.monotonic()

    def should_sample(self) -> bool:
        """Call once per eligible (not-ai_busy) frame; returns whether this frame should go to AI."""
        self._frames_since_ai += 1
        if not self.enabled:
            return True

        idle = time.monotonic() - self._last_vehicle_at
        if idle < self.ACTIVE_WINDOW_SECS:
            sample_every_n = 1
        else:
            sample_every_n = next(n for threshold, n in self.IDLE_TIERS if idle < threshold)

        if self._frames_since_ai >= sample_every_n:
            self._frames_since_ai = 0
            return True
        return False

# WS-Discovery Probe XML
_ONVIF_PROBE = (
    b'<?xml version="1.0" encoding="utf-8"?>'
    b'<s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"'
    b' xmlns:a="http://schemas.xmlsoap.org/ws/2004/08/addressing"'
    b' xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"'
    b' xmlns:dn="http://www.onvif.org/ver10/network/wsdl">'
    b"<s:Header>"
    b"<a:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</a:Action>"
    b"<a:MessageID>uuid:roadsentinel-probe-0001</a:MessageID>"
    b"<a:ReplyTo><a:Address>"
    b"http://schemas.xmlsoap.org/ws/2004/08/addressing/role/anonymous"
    b"</a:Address></a:ReplyTo>"
    b"<a:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</a:To>"
    b"</s:Header>"
    b"<s:Body>"
    b"<d:Probe><d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe>"
    b"</s:Body>"
    b"</s:Envelope>"
)


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
        self._base           = node_url.rstrip("/")
        self._camera_id      = camera_id
        self._session        = session
        self._conf_thresh    = confidence_threshold
        self._last_det       = 0.0
        self._last_frame     = 0.0
        self._inc_last: dict[str, float] = {}
        self._push_in_flight = 0

        # Socket.IO client — persistent WebSocket for low-latency frame streaming
        self._sio = sio_lib.AsyncClient(
            reconnection=True,
            reconnection_attempts=0,
            reconnection_delay=1,
            reconnection_delay_max=5,
            logger=False,
            engineio_logger=False,
        )

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

    async def connect_socketio(self) -> None:
        """Connect persistent Socket.IO WebSocket for zero-RTT frame streaming."""
        try:
            await self._sio.connect(
                self._base,
                transports=["websocket"],
                wait_timeout=8,
            )
            log.info("[%s] Socket.IO stream connected → %s", self._camera_id, self._base)
        except Exception as exc:
            log.warning("[%s] Socket.IO connect failed (%s) — HTTP fallback active",
                        self._camera_id, exc)

    async def disconnect_socketio(self) -> None:
        try:
            if self._sio.connected:
                await self._sio.disconnect()
        except Exception:
            pass

    async def _push_raw(self, jpeg_bytes: bytes) -> None:
        """Push frame via Socket.IO (zero RTT) or HTTP PUT fallback."""
        if self._sio.connected:
            try:
                await self._sio.emit("pi_frame", {
                    "camera_id": self._camera_id,
                    "data":      jpeg_bytes,
                })
                return
            except Exception:
                pass  # fall through to HTTP

        # HTTP fallback — fire-and-forget, max 2 in-flight
        if self._push_in_flight >= 2:
            return
        asyncio.create_task(self._push_http(jpeg_bytes))

    async def _push_http(self, jpeg_bytes: bytes) -> None:
        self._push_in_flight += 1
        try:
            async with self._session.put(
                f"{self._base}/api/cameras/{self._camera_id}/frame",
                data=jpeg_bytes,
                headers={"Content-Type": "image/jpeg", "Cache-Control": "no-store"},
                timeout=aiohttp.ClientTimeout(total=2.0),
            ) as resp:
                pass
        except Exception:
            pass
        finally:
            self._push_in_flight -= 1

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
            # is_heuristic: True means the incident model isn't trained yet and
            # this came from IncidentDetector's brightness-variance placeholder,
            # not a real detection — surfaced so the client can label it as such.
            "metadata":      {"is_heuristic": bool(inc.get("is_heuristic", False))},
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


# ── Recording (Phase 2 — opt-in via --record) ─────────────────────────────────
# Segments the RTSP stream into fixed-length local video files, uploads each
# finished segment to the AI service's local media storage, and registers it
# with Node's `recordings` table. Rotation/upload/POST all happen off the
# capture loop (fire-and-forget tasks, same pattern as ai_task/frame push)
# so recording never throttles the live capture rate. No-op when disabled.

class Recorder:
    def __init__(
        self,
        camera_id: str,
        ai_url: str,
        node_url: Optional[str],
        ai_session: aiohttp.ClientSession,
        node_session: Optional[aiohttp.ClientSession],
        record_dir: str,
        segment_secs: float,
        fps: int,
    ):
        self._camera_id    = camera_id
        self._ai_url       = ai_url.rstrip("/")
        self._node_url     = node_url.rstrip("/") if node_url else None
        self._ai_session   = ai_session
        self._node_session = node_session
        self._segment_secs = segment_secs
        self._fps          = fps
        self._dir           = Path(record_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        self._writer: Optional[cv2.VideoWriter] = None
        self._path: Optional[Path] = None
        self._start_time: Optional[datetime] = None
        self._frame_size: Optional[tuple] = None
        self._vehicle_count = 0
        self._incident_count = 0
        self._frames_with_vehicle: set = set()  # dedupe within a segment, approx

    def note_result(self, result: dict) -> None:
        """Called from ai_task with each AI response — accumulates counts for the segment."""
        if result.get("detections"):
            self._vehicle_count += 1  # frames-with-a-vehicle, not unique vehicles (no cross-frame ID here)
        if result.get("incidents"):
            self._incident_count += len(result["incidents"])

    def add_frame(self, frame) -> None:
        if self._writer is None:
            self._start_segment(frame)
        if self._writer is not None:
            try:
                self._writer.write(frame)
            except Exception as exc:
                log.warning("[%s] Recording write failed: %s", self._camera_id, exc)

    def _start_segment(self, frame) -> None:
        h, w = frame.shape[:2]
        self._frame_size = (w, h)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._path = self._dir / f"{self._camera_id}_{ts}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self._path), fourcc, self._fps, (w, h))
        self._start_time = datetime.now(timezone.utc)
        self._vehicle_count = 0
        self._incident_count = 0
        log.info("[%s] Recording segment started: %s", self._camera_id, self._path.name)

    async def maybe_rotate(self) -> None:
        if self._writer is None or self._start_time is None:
            return
        elapsed = (datetime.now(timezone.utc) - self._start_time).total_seconds()
        if elapsed >= self._segment_secs:
            await self._finish_segment()

    async def close(self) -> None:
        if self._writer is not None:
            await self._finish_segment()

    async def _finish_segment(self) -> None:
        writer, path, start_time = self._writer, self._path, self._start_time
        vehicle_count, incident_count = self._vehicle_count, self._incident_count
        w, h = self._frame_size or (0, 0)
        self._writer = None
        self._path = None
        self._start_time = None

        if writer is None or path is None or start_time is None:
            return
        writer.release()
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        # Upload + register happen as a background task — never blocks the caller
        # (rotation is checked once per captured frame in the main loop).
        asyncio.create_task(self._upload_and_register(
            path, start_time, end_time, duration, w, h, vehicle_count, incident_count
        ))

    async def _upload_and_register(
        self, path: Path, start_time: datetime, end_time: datetime,
        duration: float, w: int, h: int, vehicle_count: int, incident_count: int,
    ) -> None:
        recording_id = str(uuid.uuid4())
        try:
            file_size_mb = path.stat().st_size / (1024 * 1024)
        except OSError:
            file_size_mb = None

        video_url = None
        try:
            with open(path, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("file", f, filename=path.name, content_type="video/mp4")
                data.add_field(
                    "path",
                    f"recordings/{self._camera_id}/{recording_id}.mp4",
                )
                async with self._ai_session.post(
                    f"{self._ai_url}/api/storage/upload",
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=60.0),
                ) as resp:
                    if resp.status == 200:
                        payload = await resp.json()
                        video_url = payload.get("url")
                    else:
                        log.warning("[%s] Recording upload failed (status %d)",
                                    self._camera_id, resp.status)
        except Exception as exc:
            log.warning("[%s] Recording upload error: %s", self._camera_id, exc)

        if self._node_url and self._node_session:
            try:
                body = {
                    "id": recording_id,
                    "camera_id": self._camera_id,
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "duration_seconds": round(duration),
                    "video_url": video_url,
                    "file_size_mb": round(file_size_mb, 2) if file_size_mb else None,
                    "format": "mp4",
                    "resolution": f"{w}x{h}" if w and h else None,
                    "fps": self._fps,
                    "status": "completed" if video_url else "failed",
                    "vehicle_count": vehicle_count,
                    "incident_count": incident_count,
                }
                async with self._node_session.post(
                    f"{self._node_url}/api/recordings",
                    json=body,
                    timeout=aiohttp.ClientTimeout(total=NODE_TIMEOUT),
                ) as resp:
                    if resp.status not in (200, 201):
                        log.warning("[%s] Recording metadata POST failed (status %d)",
                                    self._camera_id, resp.status)
                    else:
                        log.info("[%s] Recording registered: %s (%.0fs, %d vehicle frames, %d incidents)",
                                  self._camera_id, path.name, duration, vehicle_count, incident_count)
            except Exception as exc:
                log.warning("[%s] Recording metadata POST error: %s", self._camera_id, exc)

        # Local file is only a staging area — remove it once uploaded (or if
        # upload failed, still remove it rather than filling up the Pi's SD card;
        # the segment is lost in that case, same as a dropped detection would be).
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


# ── Camera IP Auto-Discovery ──────────────────────────────────────────────────

def _get_local_subnet() -> str:
    """Best-guess local /24 subnet, e.g. '192.168.8'."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ".".join(ip.split(".")[:3])
    except Exception:
        return "192.168.1"


def _onvif_discover(timeout: float = ONVIF_TIMEOUT) -> list:
    """
    Send ONVIF WS-Discovery Probe via UDP multicast.
    Returns list of IP addresses that responded.
    """
    discovered: list = []
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 4)
        sock.settimeout(timeout)
        sock.sendto(_ONVIF_PROBE, (ONVIF_MULTICAST, ONVIF_PROBE_PORT))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                _, addr = sock.recvfrom(65536)
                ip = addr[0]
                if ip not in discovered:
                    discovered.append(ip)
                    log.debug("ONVIF: discovered device at %s", ip)
            except socket.timeout:
                break
            except OSError:
                break
    except Exception as exc:
        log.debug("ONVIF probe error: %s", exc)
    finally:
        try:
            sock.close()
        except Exception:
            pass
    return discovered


def _port_open(host: str, port: int, timeout: float = RTSP_PORT_SCAN_TIMEOUT) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _scan_subnet_for_rtsp(subnet: str, exclude_ip: str) -> list:
    """Threaded scan of subnet/24 for hosts with port 554 open."""
    found: list = []
    lock = threading.Lock()

    def probe(i: int):
        ip = f"{subnet}.{i}"
        if ip == exclude_ip:
            return
        if _port_open(ip, 554):
            with lock:
                found.append(ip)
                log.debug("Port scan: port 554 open at %s", ip)

    threads = [threading.Thread(target=probe, args=(i,), daemon=True) for i in range(1, 255)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=RTSP_PORT_SCAN_TIMEOUT + 0.5)
    return found


def _test_rtsp_url(rtsp_url: str, timeout: float = RTSP_CAP_TEST_TIMEOUT) -> bool:
    """Return True if cv2 can open and read a frame from rtsp_url within timeout."""
    result = [False]

    def _try():
        try:
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ret, _ = cap.read()
            cap.release()
            result[0] = bool(ret)
        except Exception:
            result[0] = False

    t = threading.Thread(target=_try, daemon=True)
    t.start()
    t.join(timeout=timeout)
    return result[0]


def _replace_rtsp_host(rtsp_url: str, new_host: str) -> str:
    """Swap the hostname in an RTSP URL, preserving port, path, and query."""
    parsed   = urlparse(rtsp_url)
    port_str = f":{parsed.port}" if parsed.port and parsed.port != 554 else (
        ":554" if parsed.port == 554 else ""
    )
    userinfo = f"{parsed.username}:{parsed.password}@" if parsed.username else ""
    netloc   = f"{userinfo}{new_host}{port_str}"
    return urlunparse(parsed._replace(netloc=netloc))


def discover_camera_ip(original_rtsp: str, camera_id: str) -> Optional[str]:
    """
    Attempt to find the camera at a new IP when original_rtsp fails.
    Strategy:
      1. ONVIF WS-Discovery multicast
      2. Port-554 subnet scan fallback
      3. Test each candidate with the same RTSP path
    Returns new full RTSP URL on success, None otherwise.
    """
    parsed = urlparse(original_rtsp)
    old_ip = parsed.hostname or ""
    subnet = _get_local_subnet()

    log.info("[%s] 🔍 Auto-discovering camera (last known IP: %s) …", camera_id, old_ip)

    # Step 1: ONVIF
    candidates = _onvif_discover()
    log.info("[%s] ONVIF: %d device(s) responded%s",
             camera_id, len(candidates),
             f": {candidates}" if candidates else " (none or multicast blocked)")

    # Step 2: Port scan — merge new hits
    log.info("[%s] Scanning %s.0/24 for port 554 …", camera_id, subnet)
    scan_hits = _scan_subnet_for_rtsp(subnet, old_ip)
    log.info("[%s] Port scan: %d host(s) with port 554 open%s",
             camera_id, len(scan_hits),
             f": {scan_hits}" if scan_hits else "")
    for ip in scan_hits:
        if ip not in candidates:
            candidates.append(ip)

    # Drop the old (dead) IP — we know it doesn't work
    candidates = [ip for ip in candidates if ip != old_ip]

    if not candidates:
        log.warning("[%s] ❌ Auto-discovery: no candidates found on %s.0/24", camera_id, subnet)
        return None

    # Step 3: Test each candidate
    for ip in candidates:
        new_url = _replace_rtsp_host(original_rtsp, ip)
        log.info("[%s] Testing %s …", camera_id, new_url)
        if _test_rtsp_url(new_url):
            log.info("[%s] ✅ Camera found at new IP: %s → %s", camera_id, ip, new_url)
            return new_url
        log.debug("[%s] %s: no RTSP response", camera_id, ip)

    log.warning("[%s] ❌ Auto-discovery: none of the %d candidate(s) responded to RTSP",
                camera_id, len(candidates))
    return None


async def persist_discovered_rtsp_url(
    session: aiohttp.ClientSession, node_url: str, camera_id: str, new_url: str
) -> None:
    """
    Save a newly auto-discovered RTSP URL back to Node so seed.ts's hardcoded
    default stops being the thing that matters — the DB becomes the source of
    truth for "where is this camera right now" instead of a static config
    value that drifts whenever DHCP reassigns the camera's IP.
    Best-effort: failure here just means the next restart re-discovers.
    """
    try:
        async with session.put(
            f"{node_url.rstrip('/')}/api/cameras/{camera_id}",
            json={"rtsp_url": new_url},
            timeout=aiohttp.ClientTimeout(total=NODE_TIMEOUT),
        ) as resp:
            if resp.status == 200:
                log.info("[%s] Persisted discovered RTSP URL to Node: %s", camera_id, new_url)
            else:
                log.warning("[%s] Failed to persist discovered RTSP URL (status %d)",
                            camera_id, resp.status)
    except Exception as exc:
        log.warning("[%s] Failed to persist discovered RTSP URL: %s", camera_id, exc)


# ── Night-vision / IR auto-switching (ONVIF) ──────────────────────────────────
# Brings the legacy camera_reboot_autostart_setup.sh's set_ir_auto_all.py
# behavior (which never made it into this repo — only referenced from that
# script) into the production camera_sender.py path. Opt-in via --ir-auto:
# untested against real camera hardware (no ONVIF access in this environment),
# so it's strictly best-effort — any failure just logs a warning and the main
# capture/detect/forward pipeline continues unaffected.

def set_ir_auto(host: str, port: int, user: str, password: str, camera_id: str) -> bool:
    """
    Set the camera's IR-cut filter to AUTO (day/night auto-switching) via the
    ONVIF Imaging service. Synchronous/blocking (zeep SOAP calls) — call this
    from a thread executor, not directly on the event loop.
    Returns True on success.
    """
    if not _ONVIF_OK:
        log.warning("[%s] --ir-auto requested but onvif-zeep is not installed "
                    "(pip install onvif-zeep) — skipping IR auto-switch", camera_id)
        return False
    try:
        cam = ONVIFCamera(host, port, user, password)
        media = cam.create_media_service()
        profiles = media.GetProfiles()
        if not profiles:
            log.warning("[%s] ONVIF: no media profiles returned — skipping IR auto-switch", camera_id)
            return False
        video_source_token = profiles[0].VideoSourceConfiguration.SourceToken

        imaging = cam.create_imaging_service()
        settings = imaging.GetImagingSettings({"VideoSourceToken": video_source_token})
        if hasattr(settings, "IrCutFilter"):
            settings.IrCutFilter = "AUTO"
            imaging.SetImagingSettings({
                "VideoSourceToken": video_source_token,
                "ImagingSettings": settings,
                "ForcePersistence": True,
            })
            log.info("[%s] ONVIF: IR-cut filter set to AUTO", camera_id)
            return True
        log.warning("[%s] ONVIF: camera does not expose IrCutFilter setting", camera_id)
        return False
    except Exception as exc:
        log.warning("[%s] ONVIF IR auto-switch failed (non-fatal, continuing): %s",
                    camera_id, exc)
        return False


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
    homography_points: Optional[dict] = None,
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
    if homography_points:
        # Perspective-corrected speed — takes priority over pixels_per_meter
        # in the AI service when present (see traffic_detector.py).
        data.add_field("homography_points", json.dumps(homography_points))
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
    ir_auto: bool = False,
    onvif_port: int = 80,
    onvif_user: str = "",
    onvif_pass: str = "",
    record: bool = False,
    record_dir: str = "~/roadsentinel/recordings",
    record_segment_secs: float = 300.0,
    adaptive_sampling: bool = True,
):
    stats    = Stats(camera_id)
    shutdown = asyncio.Event()

    if ir_auto:
        onvif_host = urlparse(rtsp_url).hostname or ""
        if onvif_host:
            await asyncio.get_event_loop().run_in_executor(
                None, set_ir_auto, onvif_host, onvif_port, onvif_user, onvif_pass, camera_id
            )

    def _handle_signal(*_):
        log.info("[%s] Shutdown signal received", camera_id)
        shutdown.set()

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, _handle_signal)
    loop.add_signal_handler(signal.SIGINT,  _handle_signal)

    ai_connector   = aiohttp.TCPConnector(limit=2,  keepalive_timeout=30)
    node_connector = aiohttp.TCPConnector(limit=10, keepalive_timeout=30)

    async with aiohttp.ClientSession(connector=ai_connector) as ai_session:
        async with aiohttp.ClientSession(connector=node_connector) as node_session:
            forwarder: Optional[NodeForwarder] = None
            homography_points: Optional[dict] = None
            if node_url:
                cam_cfg = await fetch_camera_config(node_session, node_url, camera_id)
                if cam_cfg.get("pixels_per_meter"):
                    pixels_per_meter = float(cam_cfg["pixels_per_meter"])
                if cam_cfg.get("speed_limit"):
                    speed_limit = float(cam_cfg["speed_limit"])
                if cam_cfg.get("detection_confidence"):
                    confidence = float(cam_cfg["detection_confidence"])
                if cam_cfg.get("homography_points"):
                    homography_points = cam_cfg["homography_points"]
                    log.info("[%s] Perspective calibration found — using homography-corrected speed",
                              camera_id)

                forwarder = NodeForwarder(node_url, camera_id, node_session,
                                          confidence_threshold=confidence)
                log.info("[%s] Node forwarding → %s  ppm=%.1f  limit=%.0f  conf=%.2f",
                         camera_id, node_url, pixels_per_meter, speed_limit, confidence)
                await forwarder.connect_socketio()
            else:
                log.warning("[%s] --node not set — detections NOT saved to DB", camera_id)

            recorder: Optional[Recorder] = None
            if record:
                recorder = Recorder(
                    camera_id=camera_id,
                    ai_url=ai_url,
                    node_url=node_url,
                    ai_session=ai_session,
                    node_session=node_session if node_url else None,
                    record_dir=os.path.expanduser(record_dir),
                    segment_secs=record_segment_secs,
                    fps=TARGET_FPS,
                )
                log.info("[%s] Recording enabled → %s (%.0fs segments)",
                         camera_id, os.path.expanduser(record_dir), record_segment_secs)

            # Sequential frame push loop — one PUT at a time so frames always
            # arrive at Node in the order they were captured (no revert glitch).
            frame_q: asyncio.Queue = asyncio.Queue(maxsize=1)
            last_frame_push = 0.0

            async def frame_push_loop():
                while not shutdown.is_set():
                    try:
                        jpeg = await asyncio.wait_for(frame_q.get(), timeout=1.0)
                        if forwarder:
                            await forwarder._push_raw(jpeg)
                    except asyncio.TimeoutError:
                        continue
                    except Exception as exc:
                        log.debug("[%s] Frame push error: %s", camera_id, exc)

            asyncio.create_task(frame_push_loop())

            consecutive_failures = 0

            while not shutdown.is_set():
                # ── Open / re-open camera ────────────────────────────────────
                try:
                    cap = await asyncio.get_event_loop().run_in_executor(
                        None, open_capture, rtsp_url
                    )
                    consecutive_failures = 0  # reset on success
                except Exception as exc:
                    consecutive_failures += 1
                    log.error("[%s] Camera open failed (%d): %s",
                              camera_id, consecutive_failures, exc)

                    # Trigger auto-discovery after N consecutive failures
                    if consecutive_failures >= DISCOVERY_AFTER_FAILURES:
                        log.info("[%s] %d consecutive failures — starting auto-discovery …",
                                 camera_id, consecutive_failures)
                        new_url = await asyncio.get_event_loop().run_in_executor(
                            None, discover_camera_ip, rtsp_url, camera_id
                        )
                        if new_url and new_url != rtsp_url:
                            log.info("[%s] Switching RTSP URL: %s → %s",
                                     camera_id, rtsp_url, new_url)
                            rtsp_url = new_url
                            consecutive_failures = 0
                            if node_url:
                                await persist_discovered_rtsp_url(
                                    node_session, node_url, camera_id, new_url
                                )
                        else:
                            log.warning("[%s] Discovery found nothing — retrying original URL",
                                        camera_id)
                            # Back off a bit longer after failed discovery
                            await asyncio.sleep(RECONNECT_WAIT * 3)
                            continue

                    await asyncio.sleep(RECONNECT_WAIT)
                    continue

                log.info("[%s] Streaming → AI:%s", camera_id, ai_url)

                # AI runs as a fire-and-forget background task so it never
                # blocks the frame-read loop. Only one AI request at a time.
                ai_busy = False
                sampler = AdaptiveSampler(enabled=adaptive_sampling)

                async def ai_task(jpeg_bytes: bytes):
                    nonlocal ai_busy
                    try:
                        result = await post_frame(
                            ai_session, ai_url, camera_id, jpeg_bytes,
                            pixels_per_meter=pixels_per_meter,
                            speed_limit=speed_limit,
                            confidence_threshold=confidence,
                            homography_points=homography_points,
                        )
                        stats.record_send(True)
                        sampler.note_result(result)
                        if recorder:
                            recorder.note_result(result)
                        if forwarder:
                            await forwarder.handle(result)
                        else:
                            _log_detections(camera_id, result)
                    except Exception as exc:
                        log.debug("[%s] AI POST error: %s", camera_id, exc)
                        stats.record_send(False)
                    finally:
                        ai_busy = False

                try:
                    while not shutdown.is_set():
                        frame_start = time.monotonic()

                        ret, frame = await asyncio.get_event_loop().run_in_executor(
                            None, cap.read
                        )
                        if not ret or frame is None:
                            log.warning("[%s] Frame read failed — reconnecting", camera_id)
                            break

                        if recorder:
                            recorder.add_frame(frame)
                            await recorder.maybe_rotate()

                        try:
                            jpeg = await asyncio.get_event_loop().run_in_executor(
                                None, encode_jpeg, frame
                            )
                        except Exception as exc:
                            log.warning("[%s] Encode error: %s", camera_id, exc)
                            stats.record_drop()
                            continue

                        # Enqueue latest frame for sequential push (drops stale if busy)
                        if forwarder:
                            _now = time.monotonic()
                            if _now - last_frame_push >= FRAME_PUSH_INTERVAL:
                                last_frame_push = _now
                                if frame_q.full():
                                    try:
                                        frame_q.get_nowait()
                                    except asyncio.QueueEmpty:
                                        pass
                                try:
                                    frame_q.put_nowait(jpeg)
                                except asyncio.QueueFull:
                                    pass

                        # Send to AI only if the previous request finished AND the
                        # adaptive sampler says this frame is due (always true while
                        # a vehicle was seen recently; throttled during idle stretches).
                        if not ai_busy and sampler.should_sample():
                            ai_busy = True
                            asyncio.create_task(ai_task(jpeg))
                        else:
                            stats.record_drop()

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
                    if recorder:
                        # Finalize whatever was captured this session rather than
                        # losing it or leaving a writer open across a reconnect gap.
                        await recorder.close()

                if not shutdown.is_set():
                    log.info("[%s] Reconnecting in %.0fs…", camera_id, RECONNECT_WAIT)
                    await asyncio.sleep(RECONNECT_WAIT)

    if recorder:
        await recorder.close()
    if forwarder:
        await forwarder.disconnect_socketio()
    log.info("[%s] Stopped. sent=%d errors=%d dropped=%d",
             camera_id, stats.sent, stats.errors, stats.dropped)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    global TARGET_FPS, FRAME_INTERVAL, JPEG_QUALITY

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

Auto-discovery: if the camera is unreachable for 3 consecutive attempts, the
script automatically runs ONVIF WS-Discovery (multicast) and a port-554 subnet
scan to locate the camera's new DHCP IP, then switches the RTSP URL silently.
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
    parser.add_argument("--ir-auto", action="store_true",
                        help="Set the camera's IR-cut filter to AUTO (day/night switching) via "
                             "ONVIF once at startup. Requires `pip install onvif-zeep`. Best-effort — "
                             "untested against real camera hardware; failures are logged and non-fatal.")
    parser.add_argument("--onvif-port", type=int, default=80,
                        help="ONVIF service port (default: 80; host is taken from --rtsp)")
    parser.add_argument("--onvif-user", default="",
                        help="ONVIF username, if the camera requires auth")
    parser.add_argument("--onvif-pass", default="",
                        help="ONVIF password, if the camera requires auth")
    parser.add_argument("--record", action="store_true",
                        help="Record local video segments and upload each finished segment to "
                             "the AI service's storage, registering it with Node's recordings "
                             "table. Off by default — untested against real camera hardware.")
    parser.add_argument("--record-dir", default="~/roadsentinel/recordings",
                        help="Local staging directory for in-progress segments (default: "
                             "~/roadsentinel/recordings) — deleted after each segment uploads")
    parser.add_argument("--record-segment-secs", type=float, default=300.0,
                        help="Recording segment length in seconds (default: 300 = 5 min)")
    parser.add_argument("--no-adaptive-sampling", action="store_true",
                        help="Disable adaptive AI sampling — always sample every eligible frame "
                             "regardless of recent activity (default: adaptive sampling is ON, "
                             "throttling AI calls during idle/no-vehicle stretches to cut load; "
                             "never affects the live-view frame rate, and always samples at full "
                             "rate while a vehicle has been seen recently)")
    args = parser.parse_args()

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
    log.info("  discovery : after %d consecutive failures", DISCOVERY_AFTER_FAILURES)

    asyncio.run(run(
        camera_id        = args.camera_id,
        rtsp_url         = args.rtsp,
        ai_url           = args.ai,
        node_url         = args.node,
        pixels_per_meter = args.ppm,
        speed_limit      = args.speed_limit,
        confidence       = args.confidence,
        ir_auto          = args.ir_auto,
        onvif_port       = args.onvif_port,
        onvif_user       = args.onvif_user,
        onvif_pass       = args.onvif_pass,
        record               = args.record,
        record_dir           = args.record_dir,
        record_segment_secs  = args.record_segment_secs,
        adaptive_sampling    = not args.no_adaptive_sampling,
    ))


if __name__ == "__main__":
    main()
