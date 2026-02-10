#!/usr/bin/env python3
"""
Calibration for Angled/Overhead Cameras
Handles perspective distortion for accurate speed measurement
"""

import cv2
import numpy as np
from ultralytics import YOLO

class AngledCameraSpeedDetector:
    """
    Speed detector for high-mounted, angled cameras
    Uses homography transformation for perspective correction
    """

    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        self.homography_matrix = None
        self.tracks = {}

        # Vehicle classes
        self.vehicle_classes = {
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }

    def calibrate_perspective(self, frame, real_world_points):
        """
        Calibrate camera perspective using known ground points

        Args:
            frame: First frame from camera
            real_world_points: List of (x_meters, y_meters) for calibration points

        Example:
            # Mark 4 corners of a known rectangle on the road
            # e.g., a parking space or road marking
            real_points = [
                (0, 0),      # Top-left (origin)
                (3.5, 0),    # Top-right (3.5m wide lane)
                (3.5, 10),   # Bottom-right (10m long)
                (0, 10)      # Bottom-left
            ]
        """
        print("="*70)
        print("📐 PERSPECTIVE CALIBRATION FOR ANGLED CAMERA")
        print("="*70)
        print()
        print("Instructions:")
        print("1. Click 4 points on the road that form a rectangle")
        print("2. Click in order: top-left, top-right, bottom-right, bottom-left")
        print("3. These should correspond to known distances")
        print()

        # Store clicked points
        image_points = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(image_points) < 4:
                image_points.append([x, y])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"P{len(image_points)}", (x+10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Calibration', frame)

                if len(image_points) == 4:
                    # Draw the quadrilateral
                    pts = np.array(image_points, np.int32)
                    cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                    cv2.imshow('Calibration', frame)

        cv2.imshow('Calibration', frame)
        cv2.setMouseCallback('Calibration', mouse_callback)

        print("Waiting for 4 points to be clicked...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if len(image_points) != 4:
            print("❌ Calibration cancelled - need 4 points")
            return False

        # Convert to numpy arrays
        src_points = np.float32(image_points)

        # Real-world coordinates (in meters)
        # Convert to pixel coordinates in "bird's eye view"
        # Scale: 100 pixels = 1 meter (adjustable)
        scale = 100  # pixels per meter

        dst_points = np.float32([
            [real_world_points[0][0] * scale, real_world_points[0][1] * scale],
            [real_world_points[1][0] * scale, real_world_points[1][1] * scale],
            [real_world_points[2][0] * scale, real_world_points[2][1] * scale],
            [real_world_points[3][0] * scale, real_world_points[3][1] * scale]
        ])

        # Calculate homography matrix
        self.homography_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self.inv_homography = cv2.getPerspectiveTransform(dst_points, src_points)
        self.scale = scale

        print()
        print("✅ Calibration complete!")
        print(f"   Homography matrix calculated")
        print(f"   Scale: {scale} pixels per meter")
        print()

        # Show bird's eye view
        h, w = frame.shape[:2]
        bird_view = cv2.warpPerspective(frame, self.homography_matrix, (w, h))
        cv2.imshow('Bird\'s Eye View', bird_view)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

        return True

    def transform_point(self, point):
        """
        Transform image point to real-world coordinates

        Args:
            point: (x, y) in image coordinates

        Returns:
            (x_meters, y_meters) in real-world coordinates
        """
        if self.homography_matrix is None:
            return point

        # Convert to homogeneous coordinates
        pt = np.array([[[point[0], point[1]]]], dtype=np.float32)

        # Apply homography
        transformed = cv2.perspectiveTransform(pt, self.homography_matrix)

        # Convert from pixels to meters
        x_meters = transformed[0][0][0] / self.scale
        y_meters = transformed[0][0][1] / self.scale

        return (x_meters, y_meters)

    def calculate_speed(self, prev_point, curr_point, time_interval):
        """
        Calculate speed using perspective-corrected coordinates

        Args:
            prev_point: Previous (x, y) in image coordinates
            curr_point: Current (x, y) in image coordinates
            time_interval: Time between points in seconds

        Returns:
            speed_kph: Speed in kilometers per hour
        """
        # Transform both points to real-world coordinates
        prev_real = self.transform_point(prev_point)
        curr_real = self.transform_point(curr_point)

        # Calculate Euclidean distance in meters
        dx = curr_real[0] - prev_real[0]
        dy = curr_real[1] - prev_real[1]
        distance_m = np.sqrt(dx**2 + dy**2)

        # Speed in m/s
        speed_ms = distance_m / time_interval

        # Convert to KPH
        speed_kph = speed_ms * 3.6

        return speed_kph

    def get_center(self, bbox):
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def process_video(self, video_path, output_path=None):
        """
        Process video with angled camera
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calibrate on first frame
        if self.homography_matrix is None:
            ret, first_frame = cap.read()
            if not ret:
                print("❌ Cannot read video")
                return

            print("\n⚠️  IMPORTANT: Camera Calibration Required")
            print()
            print("You need to mark a known rectangle on the road.")
            print("Example: A parking space or lane marking with known dimensions")
            print()

            # Example: Standard lane width = 3.5m, length = 10m
            real_world_points = [
                (0, 0),      # Top-left
                (3.5, 0),    # Top-right (3.5m wide)
                (3.5, 10),   # Bottom-right (10m long)
                (0, 10)      # Bottom-left
            ]

            if not self.calibrate_perspective(first_frame, real_world_points):
                return

            # Reset video to beginning
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Optional: Save output
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        print("\n🚀 Processing video...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Detect and track vehicles
            results = self.model.track(
                frame,
                persist=True,
                classes=[1, 2, 3, 5, 7],
                verbose=False
            )

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)

                for box, track_id, cls in zip(boxes, track_ids, classes):
                    vehicle_type = self.vehicle_classes.get(cls, 'vehicle')

                    # Get center in image coordinates
                    curr_center = self.get_center(box)
                    curr_time = frame_count / fps

                    # Calculate speed
                    speed_kph = 0
                    if track_id in self.tracks:
                        prev_center = self.tracks[track_id]['center']
                        prev_time = self.tracks[track_id]['time']

                        time_diff = curr_time - prev_time
                        if time_diff > 0:
                            speed_kph = self.calculate_speed(
                                prev_center, curr_center, time_diff
                            )

                    # Update tracking
                    self.tracks[track_id] = {
                        'center': curr_center,
                        'time': curr_time,
                        'class': vehicle_type
                    }

                    # Draw results
                    x1, y1, x2, y2 = box.astype(int)

                    colors = {
                        'car': (0, 255, 0),
                        'motorcycle': (255, 0, 0),
                        'bicycle': (0, 255, 255),
                        'bus': (255, 255, 0),
                        'truck': (255, 0, 255)
                    }
                    color = colors.get(vehicle_type, (128, 128, 128))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    label = f"ID:{track_id} {vehicle_type.upper()}"
                    if speed_kph > 0:
                        label += f" {speed_kph:.1f} KPH"

                    cv2.putText(frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            cv2.imshow('Angled Camera Speed Detection', frame)

            if output_path:
                out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()

        print(f"\n✅ Processed {frame_count} frames")
        print(f"   Tracked {len(self.tracks)} unique vehicles")


# Quick calibration guide
def calibration_guide():
    """
    Show calibration guide for angled cameras
    """
    print("="*70)
    print("📐 CALIBRATION GUIDE FOR ANGLED CAMERAS")
    print("="*70)
    print()
    print("STEP 1: Find a known rectangle on the road")
    print("   - Lane marking (usually 3.5m wide in Philippines)")
    print("   - Parking space (2.5m x 5m typical)")
    print("   - Pedestrian crossing (2m x 4m stripes)")
    print()
    print("STEP 2: Measure real dimensions")
    print("   - Use measuring tape or laser distance meter")
    print("   - Record width and length in meters")
    print()
    print("STEP 3: Mark 4 corners in video")
    print("   - Click top-left, top-right, bottom-right, bottom-left")
    print("   - Order matters!")
    print()
    print("STEP 4: Enter real dimensions")
    print("   - Program will calculate homography matrix")
    print("   - Speed measurements will be accurate")
    print()
    print("💡 TIP: For blind curve in Busay:")
    print("   - Mark a 3.5m x 10m section of road")
    print("   - This is about 1 lane width × 2 car lengths")
    print("   - Use road paint or temporary markers")
    print()


if __name__ == "__main__":
    calibration_guide()

    print("\n🚀 Ready to calibrate and detect?")
    print()
    print("Usage:")
    print("   detector = AngledCameraSpeedDetector()")
    print("   detector.process_video('busay_blind_curve.mp4', 'output.mp4')")
    print()
