from ultralytics import YOLO
import cv2
import math
import time
import numpy as np
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TrafficDetector:
    """
    YOLOv8-based traffic detector for vehicle detection and classification
    """

    # Vehicle class mappings (COCO dataset) - used for fallback model
    COCO_VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck',
        1: 'bicycle'
    }

    # Per-camera IoU tracker: {camera_id: {track_id: {bbox, time, class}}}
    # Tracks are pruned after TRACK_TTL seconds without a match.
    TRACK_TTL  = 2.0    # seconds before a track is dropped
    IOU_THRESH = 0.25   # minimum IoU to count as the same vehicle

    def __init__(self, model_path: str, device: str = 'cuda', confidence: float = 0.75):
        self.device = device
        self.confidence = confidence
        self.is_custom_model = False
        self._trackers: Dict[str, Dict[str, dict]] = {}  # per-camera tracking state
        self._next_id: Dict[str, int] = {}

        # Per-camera homography cache: camera_id -> (raw_points_dict, 3x3 matrix).
        # Recomputed only when the stored calibration for that camera changes,
        # so a normal detect() call is just a cache lookup, not a re-solve.
        self._homography_cache: Dict[str, tuple] = {}

        try:
            # Load YOLOv8 model
            self.model = YOLO(model_path)
            self.model.to(device)

            # Check if custom model (not COCO-based)
            model_classes = self.model.names
            if model_classes and 0 in model_classes:
                # Check if class 0 is a vehicle type (custom model)
                first_class = model_classes[0].lower()
                if first_class in ['car', 'vehicle', 'truck', 'motorcycle', 'bus', 'bike']:
                    self.is_custom_model = True
                    logger.info(f"Custom vehicle model detected with classes: {model_classes}")

            logger.info(f"Traffic model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Failed to load traffic model: {e}")
            # Fallback to pretrained YOLO model
            logger.info("Loading pretrained YOLOv8n model as fallback...")
            self.model = YOLO('yolov8n.pt')
            self.model.to(device)
            self.is_custom_model = False

    # ── IoU tracker helpers ───────────────────────────────────────────────────

    @staticmethod
    def _iou(a: dict, b: dict) -> float:
        ax1, ay1 = a['x'], a['y']
        ax2, ay2 = ax1 + a['width'], ay1 + a['height']
        bx1, by1 = b['x'], b['y']
        bx2, by2 = bx1 + b['width'], by1 + b['height']
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = a['width'] * a['height'] + b['width'] * b['height'] - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _center_speed(prev_bbox: dict, curr_bbox: dict,
                      dt: float, ppm: float) -> Optional[float]:
        if dt <= 0 or ppm <= 0:
            return None
        pcx = prev_bbox['x'] + prev_bbox['width']  / 2
        pcy = prev_bbox['y'] + prev_bbox['height'] / 2
        ccx = curr_bbox['x'] + curr_bbox['width']  / 2
        ccy = curr_bbox['y'] + curr_bbox['height'] / 2
        px_dist = math.sqrt((ccx - pcx) ** 2 + (ccy - pcy) ** 2)
        return round((px_dist / ppm / dt) * 3.6, 1)  # km/h

    def _update_tracks(self, camera_id: str, detections: list, ppm: float) -> None:
        """Match detections to existing tracks via IoU; attach speed; prune stale tracks."""
        now    = time.time()
        tracks = self._trackers.setdefault(camera_id, {})

        # Prune tracks older than TTL
        stale = [tid for tid, t in tracks.items() if now - t['time'] > self.TRACK_TTL]
        for tid in stale:
            del tracks[tid]

        matched: set[str] = set()
        for det in detections:
            best_id, best_iou = None, self.IOU_THRESH
            for tid, track in tracks.items():
                if tid in matched:
                    continue
                iou = self._iou(det['bbox'], track['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_id  = tid

            if best_id:
                track = tracks[best_id]
                dt    = now - track['time']
                spd   = self._center_speed(track['bbox'], det['bbox'], dt, ppm)
                if spd is not None:
                    det['speed'] = spd
                track['bbox'] = det['bbox']
                track['time'] = now
                matched.add(best_id)
            else:
                new_id = str(self._next_id.get(camera_id, 0))
                self._next_id[camera_id] = int(new_id) + 1
                tracks[new_id] = {'bbox': det['bbox'], 'time': now, 'class': det['class']}

    # ── Homography (perspective-corrected) speed ──────────────────────────────
    # Raw pixel-distance speed (_update_tracks above) is systematically wrong
    # depending on where in frame a vehicle is tracked: perspective distortion
    # means a vehicle far from the camera covers fewer pixels per real meter
    # than one close to the camera. When a camera has been calibrated (4 image
    # points mapped to their real-world road-plane coordinates via the
    # Cameras page Calibration Tool — same convention as
    # inference/camera_calibration.py's calibrate_perspective()), transform
    # tracked points into real-world meters first, then measure distance there.

    def _get_homography_matrix(self, camera_id: str, homography_points: dict):
        """Return a cached (or freshly solved) 3x3 perspective transform, or
        None if homography_points is missing/malformed."""
        image_points = homography_points.get('image_points')
        real_points  = homography_points.get('real_points')
        if not image_points or not real_points or len(image_points) != 4 or len(real_points) != 4:
            logger.warning(f"[{camera_id}] homography_points must have exactly 4 "
                            f"image_points and 4 real_points — ignoring")
            return None

        cached = self._homography_cache.get(camera_id)
        if cached and cached[0] == homography_points:
            return cached[1]

        try:
            src = np.array(image_points, dtype=np.float32)
            dst = np.array(real_points, dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(src, dst)
        except Exception as exc:
            logger.error(f"[{camera_id}] Failed to solve homography: {exc}")
            return None

        self._homography_cache[camera_id] = (homography_points, matrix)
        logger.info(f"[{camera_id}] Homography (re)computed from calibration points")
        return matrix

    @staticmethod
    def _bbox_center(bbox: dict) -> tuple:
        return (bbox['x'] + bbox['width'] / 2, bbox['y'] + bbox['height'] / 2)

    @staticmethod
    def _transform_point(matrix: np.ndarray, point: tuple) -> tuple:
        """Map an (x, y) image-pixel point to real-world (x, y) meters."""
        pt = np.array([[[point[0], point[1]]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, matrix)
        return float(out[0][0][0]), float(out[0][0][1])

    def _homography_speed(self, matrix: np.ndarray, prev_bbox: dict,
                           curr_bbox: dict, dt: float) -> Optional[float]:
        if dt <= 0:
            return None
        prev_real = self._transform_point(matrix, self._bbox_center(prev_bbox))
        curr_real = self._transform_point(matrix, self._bbox_center(curr_bbox))
        distance_m = math.sqrt(
            (curr_real[0] - prev_real[0]) ** 2 + (curr_real[1] - prev_real[1]) ** 2
        )
        return round((distance_m / dt) * 3.6, 1)  # km/h

    def _update_tracks_homography(self, camera_id: str, detections: list,
                                   matrix: np.ndarray) -> None:
        """Same IoU-matching/track-lifecycle logic as _update_tracks, but speed
        is computed from homography-transformed (real-world meter) coordinates
        instead of raw pixel distance / pixels_per_meter."""
        now    = time.time()
        tracks = self._trackers.setdefault(camera_id, {})

        stale = [tid for tid, t in tracks.items() if now - t['time'] > self.TRACK_TTL]
        for tid in stale:
            del tracks[tid]

        matched: set[str] = set()
        for det in detections:
            best_id, best_iou = None, self.IOU_THRESH
            for tid, track in tracks.items():
                if tid in matched:
                    continue
                iou = self._iou(det['bbox'], track['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_id  = tid

            if best_id:
                track = tracks[best_id]
                dt    = now - track['time']
                spd   = self._homography_speed(matrix, track['bbox'], det['bbox'], dt)
                if spd is not None:
                    det['speed'] = spd
                    det['speed_source'] = 'homography'
                track['bbox'] = det['bbox']
                track['time'] = now
                matched.add(best_id)
            else:
                new_id = str(self._next_id.get(camera_id, 0))
                self._next_id[camera_id] = int(new_id) + 1
                tracks[new_id] = {'bbox': det['bbox'], 'time': now, 'class': det['class']}

    def detect(self, image_bytes: bytes, confidence: float = None,
               camera_id: str = "default", pixels_per_meter: float = 0.0,
               homography_points: Optional[dict] = None) -> List[Dict[str, Any]]:
        """
        Detect vehicles in image. Speed is attached to each detection as
        'speed' (km/h) when either calibration is available:
          - homography_points (preferred, perspective-corrected): a dict with
            'image_points' and 'real_points', each 4 [x, y] pairs — see
            _get_homography_matrix() above.
          - pixels_per_meter > 0 (fallback, simpler): raw pixel-distance / ppm.
        """
        try:
            # Use provided confidence or default
            conf_threshold = confidence if confidence is not None else self.confidence

            # Decode image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                logger.error("Failed to decode image")
                return []

            # Run inference - filter by COCO vehicle classes only for fallback model
            if self.is_custom_model:
                # Custom model: detect all classes (model is trained for vehicles)
                results = self.model.predict(
                    source=image,
                    conf=conf_threshold,
                    iou=0.45,
                    verbose=False
                )
            else:
                # COCO model: filter to vehicle classes only
                results = self.model.predict(
                    source=image,
                    conf=conf_threshold,
                    iou=0.45,
                    classes=list(self.COCO_VEHICLE_CLASSES.keys()),
                    verbose=False
                )

            detections = []

            # Process results
            if len(results) > 0:
                result = results[0]
                boxes = result.boxes

                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    # Get class and confidence
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())

                    # Map class to vehicle type
                    if self.is_custom_model:
                        # Use model's class names directly
                        vehicle_type = self.model.names.get(cls, 'unknown')
                    else:
                        # Use COCO mapping
                        vehicle_type = self.COCO_VEHICLE_CLASSES.get(cls, 'unknown')

                    detection = {
                        'class': vehicle_type,
                        'confidence': round(conf, 3),
                        'bbox': {
                            'x': int(x1),
                            'y': int(y1),
                            'width': int(x2 - x1),
                            'height': int(y2 - y1)
                        }
                    }

                    detections.append(detection)

            logger.debug(f"Detected {len(detections)} vehicles")

            # Speed estimation via per-camera IoU tracker — homography (if
            # calibrated) takes priority over the simpler pixels_per_meter estimate.
            if detections:
                matrix = (
                    self._get_homography_matrix(camera_id, homography_points)
                    if homography_points else None
                )
                if matrix is not None:
                    self._update_tracks_homography(camera_id, detections, matrix)
                elif pixels_per_meter > 0:
                    self._update_tracks(camera_id, detections, pixels_per_meter)

            return detections

        except Exception as e:
            logger.error(f"Detection error: {e}", exc_info=True)
            return []

    def estimate_speed(
        self,
        prev_detection: Dict[str, Any],
        curr_detection: Dict[str, Any],
        time_delta: float,
        pixels_per_meter: float
    ) -> float:
        """
        Estimate vehicle speed based on movement between frames

        Args:
            prev_detection: Previous frame detection
            curr_detection: Current frame detection
            time_delta: Time between frames (seconds)
            pixels_per_meter: Camera calibration parameter

        Returns:
            Speed in km/h
        """
        try:
            # Get center points of bounding boxes
            prev_bbox = prev_detection['bbox']
            curr_bbox = curr_detection['bbox']

            prev_center_x = prev_bbox['x'] + prev_bbox['width'] / 2
            prev_center_y = prev_bbox['y'] + prev_bbox['height'] / 2

            curr_center_x = curr_bbox['x'] + curr_bbox['width'] / 2
            curr_center_y = curr_bbox['y'] + curr_bbox['height'] / 2

            # Calculate pixel distance
            pixel_distance = np.sqrt(
                (curr_center_x - prev_center_x) ** 2 +
                (curr_center_y - prev_center_y) ** 2
            )

            # Convert to meters
            distance_meters = pixel_distance / pixels_per_meter

            # Calculate speed (m/s)
            speed_mps = distance_meters / time_delta

            # Convert to km/h
            speed_kmh = speed_mps * 3.6

            return round(speed_kmh, 2)

        except Exception as e:
            logger.error(f"Speed estimation error: {e}")
            return 0.0
