# complete_speed_detector.py
from ultralytics import YOLO
import cv2
import numpy as np

class VehicleSpeedDetector:
    def __init__(self, ppm=60):
        """
        ppm: Pixels per meter (camera calibration)
        """
        # Load pretrained COCO model - NO TRAINING NEEDED!
        self.model = YOLO('yolov8n.pt')
        
        self.ppm = ppm  # Calibration parameter
        self.tracks = {}
        
        # COCO class names we care about
        self.vehicle_classes = {
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        
    def calculate_speed(self, prev_center, curr_center, time_interval):
        """
        Calculate speed from two position points
        """
        # Pixel distance
        dx = curr_center[0] - prev_center[0]
        dy = curr_center[1] - prev_center[1]
        pixel_distance = np.sqrt(dx**2 + dy**2)
        
        # Convert to meters
        distance_m = pixel_distance / self.ppm
        
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
        Process video and detect vehicle speeds
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Optional: Save output video
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect and track vehicles
            # persist=True enables tracking across frames
            results = self.model.track(
                frame, 
                persist=True,
                classes=[1, 2, 3, 5, 7],  # bicycle, car, motorcycle, bus, truck
                verbose=False
            )
            
            if results[0].boxes.id is not None:
                # Get detection info
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()
                
                for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
                    # Get vehicle type
                    vehicle_type = self.vehicle_classes.get(cls, 'vehicle')
                    
                    # Get current center position
                    curr_center = self.get_center(box)
                    curr_time = frame_count / fps
                    
                    # Calculate speed if we have previous position
                    speed_kph = 0
                    if track_id in self.tracks:
                        prev_center = self.tracks[track_id]['center']
                        prev_time = self.tracks[track_id]['time']
                        
                        time_diff = curr_time - prev_time
                        
                        if time_diff > 0:
                            speed_kph = self.calculate_speed(
                                prev_center, 
                                curr_center, 
                                time_diff
                            )
                    
                    # Update tracking info
                    self.tracks[track_id] = {
                        'center': curr_center,
                        'time': curr_time,
                        'class': vehicle_type
                    }
                    
                    # Draw bounding box
                    x1, y1, x2, y2 = box.astype(int)
                    
                    # Color based on vehicle type
                    colors = {
                        'car': (0, 255, 0),      # Green
                        'motorcycle': (255, 0, 0), # Blue
                        'bicycle': (0, 255, 255),  # Yellow
                        'bus': (255, 255, 0),      # Cyan
                        'truck': (255, 0, 255)     # Magenta
                    }
                    color = colors.get(vehicle_type, (128, 128, 128))
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with speed
                    label = f"ID:{track_id} {vehicle_type.upper()}"
                    if speed_kph > 0:
                        label += f" {speed_kph:.1f} KPH"
                    
                    # Background for text
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        frame, 
                        (x1, y1 - text_height - 10),
                        (x1 + text_width, y1),
                        color, -1
                    )
                    
                    # Text
                    cv2.putText(
                        frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
                    )
            
            # Show frame
            cv2.imshow('Vehicle Speed Detection', frame)
            
            # Save frame if output specified
            if output_path:
                out.write(frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"Processed {frame_count} frames")
        print(f"Tracked {len(self.tracks)} unique vehicles")


# Calibration helper
def calibrate_camera(video_path):
    """
    Interactive calibration tool
    Click two points with known distance
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    points = []
    
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow('Calibration', frame)
            
            if len(points) == 2:
                # Draw line
                cv2.line(frame, points[0], points[1], (0, 255, 0), 2)
                cv2.imshow('Calibration', frame)
    
    cv2.imshow('Calibration', frame)
    cv2.setMouseCallback('Calibration', click_event)
    
    print("Click two points with known distance...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    if len(points) == 2:
        pixel_distance = np.sqrt(
            (points[1][0] - points[0][0])**2 + 
            (points[1][1] - points[0][1])**2
        )
        
        real_distance = float(input("Enter real distance in meters: "))
        ppm = pixel_distance / real_distance
        
        print(f"Pixels per meter: {ppm:.2f}")
        return ppm
    
    return None


# Main execution
if __name__ == "__main__":
    # Step 1: Calibrate camera (do this once)
    print("=== CAMERA CALIBRATION ===")
    ppm = calibrate_camera('test_video.mp4')
    
    if ppm:
        # Step 2: Run speed detection
        print("\n=== VEHICLE SPEED DETECTION ===")
        detector = VehicleSpeedDetector(ppm=ppm)
        detector.process_video(
            video_path='traffic_video.mp4',
            output_path='output_with_speeds.mp4'  # Optional: save output
        )