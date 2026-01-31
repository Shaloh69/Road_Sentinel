-- Road Sentinel MySQL Database Schema
-- Database: road_sentinel
-- Platform: Aiven MySQL

-- Drop tables if exists (for clean setup)
DROP TABLE IF EXISTS analytics_hourly;
DROP TABLE IF EXISTS recordings;
DROP TABLE IF EXISTS incidents;
DROP TABLE IF EXISTS detections;
DROP TABLE IF EXISTS cameras;

-- ============================================
-- CAMERAS TABLE
-- ============================================
CREATE TABLE cameras (
  id VARCHAR(36) PRIMARY KEY,
  name VARCHAR(100) NOT NULL,
  location VARCHAR(255) NOT NULL,
  rtsp_url VARCHAR(500) NOT NULL,
  status ENUM('online', 'offline', 'error') DEFAULT 'offline',
  fps INT DEFAULT 30,
  resolution VARCHAR(20) DEFAULT '1920x1080',
  pixels_per_meter FLOAT DEFAULT 25.5 COMMENT 'For speed calculation calibration',
  speed_limit INT DEFAULT 60 COMMENT 'Speed limit in km/h',
  detection_confidence FLOAT DEFAULT 0.75 COMMENT 'YOLOv8 confidence threshold',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================
-- DETECTIONS TABLE (Vehicle Detections)
-- ============================================
CREATE TABLE detections (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  camera_id VARCHAR(36) NOT NULL,
  timestamp TIMESTAMP(3) DEFAULT CURRENT_TIMESTAMP(3),
  vehicle_type ENUM('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'unknown') NOT NULL,
  speed FLOAT COMMENT 'Speed in km/h',
  confidence FLOAT NOT NULL COMMENT 'Detection confidence 0-1',
  bbox_x INT NOT NULL COMMENT 'Bounding box X coordinate',
  bbox_y INT NOT NULL COMMENT 'Bounding box Y coordinate',
  bbox_width INT NOT NULL COMMENT 'Bounding box width',
  bbox_height INT NOT NULL COMMENT 'Bounding box height',
  direction VARCHAR(20) COMMENT 'north, south, east, west',
  lane_number INT COMMENT 'Detected lane number',
  INDEX idx_camera_timestamp (camera_id, timestamp),
  INDEX idx_vehicle_type (vehicle_type),
  INDEX idx_timestamp (timestamp),
  FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================
-- INCIDENTS TABLE
-- ============================================
CREATE TABLE incidents (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  camera_id VARCHAR(36) NOT NULL,
  timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  incident_type ENUM('crash', 'speeding', 'wrong_way', 'stopped_vehicle', 'congestion', 'illegal_parking', 'other') NOT NULL,
  severity ENUM('low', 'medium', 'high', 'critical') NOT NULL,
  title VARCHAR(255) NOT NULL,
  description TEXT,
  image_url VARCHAR(500) COMMENT 'Supabase Storage URL',
  video_url VARCHAR(500) COMMENT 'Supabase Storage URL',
  confidence FLOAT COMMENT 'Incident detection confidence 0-1',
  status ENUM('active', 'resolved', 'false_alarm', 'investigating') DEFAULT 'active',
  resolved_at TIMESTAMP NULL,
  resolved_by VARCHAR(100),
  notes TEXT,
  metadata JSON COMMENT 'Additional incident data',
  INDEX idx_camera_timestamp (camera_id, timestamp),
  INDEX idx_status (status),
  INDEX idx_incident_type (incident_type),
  INDEX idx_severity (severity),
  INDEX idx_timestamp (timestamp),
  FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================
-- ANALYTICS_HOURLY TABLE (Aggregated Statistics)
-- ============================================
CREATE TABLE analytics_hourly (
  id BIGINT AUTO_INCREMENT PRIMARY KEY,
  camera_id VARCHAR(36) NOT NULL,
  hour_timestamp TIMESTAMP NOT NULL COMMENT 'Start of the hour',
  total_vehicles INT DEFAULT 0,
  avg_speed FLOAT,
  max_speed FLOAT,
  min_speed FLOAT,
  car_count INT DEFAULT 0,
  truck_count INT DEFAULT 0,
  bus_count INT DEFAULT 0,
  motorcycle_count INT DEFAULT 0,
  bicycle_count INT DEFAULT 0,
  incident_count INT DEFAULT 0,
  speeding_violations INT DEFAULT 0,
  peak_flow_minute INT COMMENT 'Minute with highest traffic',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY unique_camera_hour (camera_id, hour_timestamp),
  INDEX idx_hour_timestamp (hour_timestamp),
  FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================
-- RECORDINGS TABLE (Video Recording Metadata)
-- ============================================
CREATE TABLE recordings (
  id VARCHAR(36) PRIMARY KEY,
  camera_id VARCHAR(36) NOT NULL,
  start_time TIMESTAMP NOT NULL,
  end_time TIMESTAMP,
  duration_seconds INT,
  video_url VARCHAR(500) COMMENT 'Supabase Storage URL',
  thumbnail_url VARCHAR(500) COMMENT 'Supabase Storage URL',
  file_size_mb FLOAT,
  format VARCHAR(20) DEFAULT 'mp4',
  resolution VARCHAR(20),
  fps INT,
  status ENUM('recording', 'completed', 'failed', 'deleted') DEFAULT 'recording',
  error_message TEXT,
  vehicle_count INT DEFAULT 0 COMMENT 'Total vehicles detected in recording',
  incident_count INT DEFAULT 0 COMMENT 'Total incidents in recording',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  INDEX idx_camera_time (camera_id, start_time),
  INDEX idx_status (status),
  FOREIGN KEY (camera_id) REFERENCES cameras(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================
-- SAMPLE DATA (Optional - for testing)
-- ============================================

-- Insert sample cameras
INSERT INTO cameras (id, name, location, rtsp_url, status, fps, resolution, pixels_per_meter, speed_limit)
VALUES
  ('CAM-A-001', 'Camera A', 'North Approach - Blind Curve', 'rtsp://192.168.1.100:8554/camera1', 'offline', 30, '1920x1080', 25.5, 60),
  ('CAM-B-002', 'Camera B', 'South Approach - Blind Curve', 'rtsp://192.168.1.101:8554/camera2', 'offline', 30, '1920x1080', 25.5, 60);

-- ============================================
-- USEFUL QUERIES
-- ============================================

-- Get live vehicle count per camera (last 5 minutes)
-- SELECT camera_id, COUNT(*) as vehicle_count, AVG(speed) as avg_speed
-- FROM detections
-- WHERE timestamp >= NOW() - INTERVAL 5 MINUTE
-- GROUP BY camera_id;

-- Get active incidents
-- SELECT i.*, c.name as camera_name, c.location
-- FROM incidents i
-- JOIN cameras c ON i.camera_id = c.id
-- WHERE i.status = 'active'
-- ORDER BY i.timestamp DESC;

-- Get hourly traffic statistics for today
-- SELECT camera_id, hour_timestamp, total_vehicles, avg_speed, incident_count
-- FROM analytics_hourly
-- WHERE DATE(hour_timestamp) = CURDATE()
-- ORDER BY hour_timestamp DESC;

-- Get speeding violations (vehicles exceeding camera speed limit)
-- SELECT d.*, c.name as camera_name, c.speed_limit
-- FROM detections d
-- JOIN cameras c ON d.camera_id = c.id
-- WHERE d.speed > c.speed_limit
-- ORDER BY d.timestamp DESC
-- LIMIT 100;
