from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class IncidentDetector:
    """
    YOLOv8-based incident detector for crash and traffic violation detection
    """

    # Incident severity mapping
    SEVERITY_MAP = {
        'crash': 'critical',
        'accident': 'critical',
        'collision': 'critical',
        'stopped_vehicle': 'medium',
        'congestion': 'low',
        'wrong_way': 'high',
        'illegal_parking': 'low'
    }

    def __init__(self, model_path: str, device: str = 'cuda', confidence: float = 0.75):
        """
        Initialize incident detector

        Args:
            model_path: Path to YOLOv8 model weights
            device: Device to run inference on ('cuda', 'cpu', 'mps')
            confidence: Default confidence threshold
        """
        self.device = device
        self.confidence = confidence

        try:
            # Load custom YOLOv8 model trained for incident detection
            self.model = YOLO(model_path)
            self.model.to(device)
            logger.info(f"Incident model loaded successfully on {device}")
        except Exception as e:
            logger.error(f"Failed to load incident model: {e}")
            # Fallback to basic object detection
            logger.warning("Incident model not available - using fallback detection")
            self.model = None

    def detect(self, image_bytes: bytes, confidence: float = None) -> List[Dict[str, Any]]:
        """
        Detect incidents in image

        Args:
            image_bytes: Image as bytes
            confidence: Confidence threshold (uses default if None)

        Returns:
            List of incident dictionaries
        """
        try:
            if self.model is None:
                # Model not loaded - use heuristic detection
                return self._heuristic_detection(image_bytes)

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
                verbose=False
            )

            incidents = []

            # Process results
            if len(results) > 0:
                result = results[0]
                boxes = result.boxes

                for box in boxes:
                    # Get class and confidence
                    cls_id = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())

                    # Get class name from model
                    class_name = self.model.names[cls_id]

                    # Determine severity
                    severity = self.SEVERITY_MAP.get(class_name.lower(), 'medium')

                    incident = {
                        'type': class_name,
                        'severity': severity,
                        'confidence': round(conf, 3),
                        'description': self._generate_description(class_name, conf)
                    }

                    incidents.append(incident)

            logger.debug(f"Detected {len(incidents)} incidents")
            return incidents

        except Exception as e:
            logger.error(f"Incident detection error: {e}", exc_info=True)
            return []

    def _heuristic_detection(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Fallback heuristic-based incident detection
        Uses motion detection, color analysis, etc.

        Args:
            image_bytes: Image as bytes

        Returns:
            List of detected incidents
        """
        incidents = []

        try:
            # Decode image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                return incidents

            # Example heuristic: Detect stopped vehicles by analyzing static regions
            # This is a simplified example - real implementation would be more sophisticated

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect potential congestion by analyzing brightness variance
            brightness_var = np.var(gray)

            if brightness_var < 500:  # Low variance might indicate congestion
                incidents.append({
                    'type': 'congestion',
                    'severity': 'low',
                    'confidence': 0.6,
                    'description': 'Potential traffic congestion detected (heuristic)'
                })

            logger.debug(f"Heuristic detection found {len(incidents)} potential incidents")

        except Exception as e:
            logger.error(f"Heuristic detection error: {e}")

        return incidents

    def _generate_description(self, incident_type: str, confidence: float) -> str:
        """
        Generate human-readable incident description

        Args:
            incident_type: Type of incident
            confidence: Detection confidence

        Returns:
            Description string
        """
        descriptions = {
            'crash': 'Vehicle collision detected',
            'accident': 'Traffic accident detected',
            'stopped_vehicle': 'Vehicle stopped on roadway',
            'congestion': 'Traffic congestion detected',
            'wrong_way': 'Wrong-way vehicle detected',
            'illegal_parking': 'Illegal parking detected',
            'speeding': 'Speeding violation detected'
        }

        base_desc = descriptions.get(incident_type.lower(), f'{incident_type} detected')
        return f"{base_desc} (confidence: {confidence:.1%})"

    def analyze_crash_severity(
        self,
        detections: List[Dict[str, Any]],
        vehicle_speeds: List[float]
    ) -> str:
        """
        Analyze crash severity based on number of vehicles and speeds

        Args:
            detections: List of vehicle detections
            vehicle_speeds: List of vehicle speeds

        Returns:
            Severity level ('low', 'medium', 'high', 'critical')
        """
        try:
            num_vehicles = len(detections)
            avg_speed = np.mean(vehicle_speeds) if vehicle_speeds else 0

            # Multi-vehicle high-speed crash
            if num_vehicles >= 3 and avg_speed > 80:
                return 'critical'

            # High-speed crash
            if avg_speed > 100:
                return 'critical'

            # Multi-vehicle crash
            if num_vehicles >= 2 and avg_speed > 50:
                return 'high'

            # Single vehicle moderate speed
            if avg_speed > 60:
                return 'medium'

            # Low-speed incident
            return 'low'

        except Exception as e:
            logger.error(f"Crash severity analysis error: {e}")
            return 'medium'
