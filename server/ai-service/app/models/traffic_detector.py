from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class TrafficDetector:
    """
    YOLOv8-based traffic detector for vehicle detection and classification
    """

    # Vehicle class mappings (COCO dataset)
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck',
        1: 'bicycle'
    }

    def __init__(self, model_path: str, device: str = 'cuda', confidence: float = 0.75):
        """
        Initialize traffic detector

        Args:
            model_path: Path to YOLOv8 model weights
            device: Device to run inference on ('cuda', 'cpu', 'mps')
            confidence: Default confidence threshold
        """
        self.device = device
        self.confidence = confidence

        try:
            # Load YOLOv8 model
            self.model = YOLO(model_path)
            self.model.to(device)
            logger.info(f"Traffic model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Failed to load traffic model: {e}")
            # Fallback to pretrained YOLO model
            logger.info("Loading pretrained YOLOv8n model as fallback...")
            self.model = YOLO('yolov8n.pt')
            self.model.to(device)

    def detect(self, image_bytes: bytes, confidence: float = None) -> List[Dict[str, Any]]:
        """
        Detect vehicles in image

        Args:
            image_bytes: Image as bytes
            confidence: Confidence threshold (uses default if None)

        Returns:
            List of detection dictionaries
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

            # Run inference
            results = self.model.predict(
                source=image,
                conf=conf_threshold,
                iou=0.45,
                classes=list(self.VEHICLE_CLASSES.keys()),  # Filter only vehicle classes
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
                    vehicle_type = self.VEHICLE_CLASSES.get(cls, 'unknown')

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
